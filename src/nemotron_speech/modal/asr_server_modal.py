"""Modal deployment for Nemotron Streaming ASR server.

Deploy to Modal with GPU support for real-time speech recognition.

Usage:
    # Deploy to Modal
    modal deploy -m src.nemotron_speech.modal.asr_server_modal

    # Test locally
    python -m src.nemotron_speech.modal.asr_server_modal

"""

import asyncio
import copy
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Any

import modal
import numpy as np

# Modal app definition
app = modal.App("nemotron-asr-server")

# Model cache volume
model_cache = modal.Volume.from_name("nemotron-speech", create_if_missing=True)
CACHE_PATH = "/cache"

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"

# Define the container image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .env({
        "DEBIAN_FRONTEND": "noninteractive",
    })
    .apt_install("git", "libsndfile1", "ffmpeg")
    .uv_pip_install(
         "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "numpy<2.0.0",
        "torch",
        "aiohttp",
        "loguru",
        "omegaconf",
        "Cython",
        "webdataset",
        "hydra-core",
        "fastapi[standard]",
        "websockets",
    ).uv_pip_install(
        # Stable release from the post-split speech repo (NVIDIA-NeMo/Speech).
        # Previously pinned to a git commit on NVIDIA/NeMo@main, which is now
        # stale for speech. Fallback if a regression appears:
        # "nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo.git@644201898480ec8c8d0a637f0c773825509ac4dc"
        "nemo-toolkit[asr]==2.7.3",
    ).env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": CACHE_PATH,
        "TORCH_HOME": CACHE_PATH,
    })
)

# Enable debug logging with DEBUG_ASR=1
DEBUG_ASR = os.environ.get("DEBUG_ASR", "0") == "1"

# Right context options for att_context_size=[70, X]
RIGHT_CONTEXT_OPTIONS = {
    0: "~80ms ultra-low latency",
    1: "~160ms low latency (recommended)",
    6: "~560ms balanced",
    13: "~1.12s highest accuracy",
}


def _hash_audio(audio: np.ndarray) -> str:
    """Get short hash of audio array for debugging."""
    if audio is None or len(audio) == 0:
        return "empty"
    return hashlib.md5(audio.tobytes()).hexdigest()[:8]


@dataclass
class ASRSession:
    """Per-connection session state with caches for true incremental streaming."""
    
    id: str
    websocket: Any
    
    # Accumulated audio buffer (all audio received so far)
    accumulated_audio: Optional[np.ndarray] = None
    
    # Number of mel frames already emitted to encoder
    emitted_frames: int = 0
    
    # Encoder cache state
    cache_last_channel: Optional[Any] = None
    cache_last_time: Optional[Any] = None
    cache_last_channel_len: Optional[Any] = None
    
    # Decoder state
    previous_hypotheses: Any = None
    pred_out_stream: Any = None
    
    # Current transcription (model's cumulative output)
    current_text: str = ""

    # Mel frames dropped from the front of accumulated_audio by buffer trimming.
    # Global frame index = local index into the current buffer + frame_offset.
    frame_offset: int = 0

    # Trailing odd byte from a binary message (an int16 sample can be split
    # across two WebSocket messages; the leftover byte joins the next message)
    pending_bytes: bytes = b""

    # Consecutive chunk-processing failures; connection closes past a threshold
    error_count: int = 0

# Encoder lookahead: 1 (160ms, default) balances latency/WER; 0 (80ms) is the
# lowest-latency mode (~+0.8 avg WER per the model card) and also halves the
# hard-reset padding, cutting finalization compute. See RIGHT_CONTEXT_OPTIONS.
RIGHT_CONTEXT = int(os.environ.get("ASR_RIGHT_CONTEXT", "1"))

# Inference precision: "bf16" (default) mirrors NVIDIA's cache-aware streaming
# reference (autocast around the streaming loop) and roughly halves encoder
# time on Ada/Hopper GPUs. Set ASR_PRECISION=fp32 to opt out.
ASR_PRECISION = os.environ.get("ASR_PRECISION", "bf16")

# Audio buffer trimming: keep at most this much audio per session. Turns shorter
# than this are processed identically to the untrimmed behavior; longer turns get
# a sliding window so per-chunk and finalization cost stay flat instead of
# growing with utterance length (which would blow the voice-to-voice budget).
TRIM_KEEP_SECONDS = 5.0
# Extra mel frames kept ahead of the pre-encode cache so STFT windows at the
# buffer edge see real audio rather than padding.
CONTEXT_MARGIN_FRAMES = 8
# Close the connection after this many consecutive chunk-processing failures
MAX_CONSECUTIVE_ERRORS = 3

with image.imports():
    import time
    import torch
    import traceback
    from loguru import logger
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import uuid


# Modal class for ASR inference
@app.cls(
    region="ap-south",
    routing_region="ap-south",
    image=image,
    volumes={
        CACHE_PATH: model_cache,
    },
    gpu="T4",
    # The Modal input timeout applies to the WebSocket's whole lifetime; the old
    # 3600 hard-killed any call at exactly 1 hour.
    timeout=24 * 60 * 60,
    min_containers=1,  # Keep warm for low latency
    max_containers=10,  # Bound autoscale cost
    # Deploy with ASR_AUTH_TOKEN set locally to require ?token=... on connect;
    # deploy without it to keep the endpoint open (previous behavior).
    secrets=[modal.Secret.from_dict({"ASR_AUTH_TOKEN": os.environ.get("ASR_AUTH_TOKEN", "")})],
)
# Without this, a WebSocket connection occupies the container's only input slot
# for its entire life, so every additional caller hits a multi-minute cold start.
# Kept low because sessions share one GPU and one inference lock: another
# session's turn-finalization sits directly in your voice-to-voice latency.
@modal.concurrent(max_inputs=4)
class NemotronASRModel:
    """Modal class for Nemotron ASR inference."""
    
    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        
        
        logger.info(f"Loading ASR model from {MODEL_NAME}...")

        # TF32 for any matmuls that stay fp32 (free speedup on Ampere+)
        torch.set_float32_matmul_precision("high")

        # bf16 autocast around the streaming loop, matching NVIDIA's
        # cache-aware streaming reference script. The CUDA-graph greedy decoder
        # stays OFF: it has known crashes with streaming partial hypotheses.
        self.amp_dtype = torch.bfloat16 if ASR_PRECISION == "bf16" else None
        logger.info(f"Inference precision: {'bf16 autocast' if self.amp_dtype else 'fp32'}")

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            MODEL_NAME
        )
        self.model = self.model.cuda()
        
        # Configure attention context for streaming
        logger.info(f"Setting att_context_size=[70, {RIGHT_CONTEXT}] ({RIGHT_CONTEXT_OPTIONS.get(RIGHT_CONTEXT, 'custom')})")
        self.model.encoder.set_default_att_context_size([70, RIGHT_CONTEXT])
        
        # Configure greedy decoding
        logger.info("Configuring greedy decoding...")
        self.model.change_decoding_strategy(
            decoding_cfg=OmegaConf.create({
                'strategy': 'greedy',
                'greedy': {
                    'max_symbols': 10,
                    'loop_labels': False,
                    'use_cuda_graph_decoder': False,
                }
            })
        )
        self.model.eval()
        
        # Disable dither for deterministic preprocessing
        self.model.preprocessor.featurizer.dither = 0.0
        
        # Get streaming config
        scfg = self.model.encoder.streaming_cfg
        logger.info(f"Streaming config: chunk_size={scfg.chunk_size}, shift_size={scfg.shift_size}")
        
        # Calculate parameters
        preprocessor_cfg = self.model.cfg.preprocessor
        hop_length_sec = preprocessor_cfg.get('window_stride', 0.01)
        self.sample_rate = 16000
        self.hop_samples = int(hop_length_sec * self.sample_rate)
        
        # shift_size[1] = 16 frames for 160ms chunks
        self.shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, list) else scfg.shift_size
        
        # pre_encode_cache_size[1] = 9 frames
        pre_cache = scfg.pre_encode_cache_size
        self.pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, list) else pre_cache
        
        # drop_extra_pre_encoded for non-first chunks
        self.drop_extra = scfg.drop_extra_pre_encoded

        # The first chunk must emit at least pre_encode_cache_size frames
        # (rounded up to whole chunks), or the second chunk's start index
        # (emitted - pre_encode_cache_size) goes negative. With right_context=0
        # the shift is 8 mel frames < 9 cache frames, which crashed the encoder
        # with a 0-length attention query. For right_context>=1 this equals
        # shift_frames, preserving the original behavior exactly.
        self.first_chunk_frames = (
            (self.pre_encode_cache_size + self.shift_frames - 1) // self.shift_frames
        ) * self.shift_frames
        
        # Calculate silence padding for final chunk:
        # - right_context chunks for encoder lookahead
        # - 1 additional chunk for decoder finalization
        # With right_context=1, this is (1+1)*160ms = 320ms
        self.final_padding_frames = (RIGHT_CONTEXT + 1) * self.shift_frames
        padding_ms = self.final_padding_frames * hop_length_sec * 1000

        # Sliding-window cap on the per-session audio buffer. Must always retain
        # enough context for the next chunk's pre-encode cache plus margin.
        self.keep_samples = max(
            int(TRIM_KEEP_SECONDS * self.sample_rate),
            (self.pre_encode_cache_size + CONTEXT_MARGIN_FRAMES + 2 * self.shift_frames + 2)
            * self.hop_samples,
        )

        shift_ms = self.shift_frames * hop_length_sec * 1000
        logger.info(f"Model loaded: {type(self.model).__name__}")
        logger.info(f"Shift size: {shift_ms:.0f}ms ({self.shift_frames} frames)")
        logger.info(f"Pre-encode cache: {self.pre_encode_cache_size} frames")
        logger.info(f"Final chunk padding: {padding_ms:.0f}ms ({self.final_padding_frames} frames)")
        logger.info(f"Audio buffer window: {self.keep_samples * 1000 // self.sample_rate}ms")
        
        # Warmup inference
        self._warmup()
        
        # Inference lock for thread safety
        self.inference_lock = asyncio.Lock()
        
        # Active sessions
        self.sessions = {}
    
    def _autocast(self):
        """Autocast context matching NVIDIA's cache-aware streaming reference."""
        return torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None)

    def _warmup(self):
        """Warm the exact production code paths and tensor shapes.

        Runs the real interim-chunk step (first-chunk and cached-chunk shapes)
        and the real finalization step twice, so kernel compilation and
        autotuning happen at container startup instead of on a live caller's
        first turn (freshly autoscaled containers would otherwise stutter).
        """

        logger.info("Warmup: exercising streaming chunk + finalization paths...")
        start = time.perf_counter()

        chunk_samples = (self.shift_frames + 1) * self.hop_samples
        for _ in range(2):
            session = ASRSession(id="warmup", websocket=None)
            self._init_session(session)

            # ~1s of silence fed through the interim chunk path
            for _ in range(6):
                session.accumulated_audio = np.concatenate(
                    [session.accumulated_audio, np.zeros(chunk_samples, dtype=np.float32)]
                )
                if self._process_chunk(session) is None:
                    raise RuntimeError("Warmup chunk step failed - see error above")

            # Hard-reset finalization path (padding + keep_all_outputs=True)
            padding = np.zeros(self.final_padding_frames * self.hop_samples, dtype=np.float32)
            session.accumulated_audio = np.concatenate([session.accumulated_audio, padding])
            if self._process_final_chunk(session) is None:
                raise RuntimeError("Warmup finalization step failed - see error above")

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Warmup complete in {elapsed:.0f}ms - GPU memory claimed")
    
    def _init_session(self, session: ASRSession):
        """Initialize a fresh session (also used to reset state after a hard reset)."""

        # Initialize encoder cache
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        session.cache_last_channel = cache[0]
        session.cache_last_time = cache[1]
        session.cache_last_channel_len = cache[2]

        # Reset audio buffer and frame counters
        session.accumulated_audio = np.array([], dtype=np.float32)
        session.emitted_frames = 0
        session.frame_offset = 0
        session.pending_bytes = b""

        # Reset decoder state
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.current_text = ""
    
    def _min_audio_for_chunk(self, session: ASRSession) -> int:
        """Samples needed in the (possibly trimmed) buffer for one more chunk."""
        local_emitted = session.emitted_frames - session.frame_offset
        needed = self.first_chunk_frames if session.emitted_frames == 0 else self.shift_frames
        return (local_emitted + needed + 1) * self.hop_samples

    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames available."""

        # int16 samples can be split across WebSocket messages; carry the odd
        # trailing byte over instead of letting np.frombuffer raise and kill
        # the whole connection.
        raw = session.pending_bytes + audio_bytes
        if len(raw) % 2:
            session.pending_bytes = raw[-1:]
            raw = raw[:-1]
        else:
            session.pending_bytes = b""
        if not raw:
            return

        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        if DEBUG_ASR:
            chunk_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
            logger.debug(f"Session {session.id}: recv chunk {len(audio_bytes)}B hash={chunk_hash}")

        session.accumulated_audio = np.concatenate([session.accumulated_audio, audio_np])

        # Process if we have enough audio for new frames
        # We need shift_frames worth of new mel frames (after skipping edge frame)
        while len(session.accumulated_audio) >= self._min_audio_for_chunk(session):
            frames_before = session.emitted_frames

            async with self.inference_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_chunk, session
                )

            if text is None:
                # Chunk processing raised. Never retry in place: emitted_frames
                # did not advance, so looping again would re-run the same
                # failing inference forever under the shared lock.
                session.error_count += 1
                if session.error_count >= MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError(
                        f"ASR chunk processing failed {session.error_count} times in a row"
                    )
                break
            session.error_count = 0

            if text != session.current_text:
                session.current_text = text
                logger.debug(f"Session {session.id} interim: {text[-50:] if len(text) > 50 else text}")
                await session.websocket.send_json({
                    "type": "transcript",
                    "text": text,
                    "is_final": False
                })

            if session.emitted_frames == frames_before:
                # No forward progress (e.g. mel frame count came up short of the
                # sample-based estimate) - wait for more audio instead of spinning.
                break
    
    def _process_chunk(self, session: ASRSession) -> Optional[str]:
        """Process accumulated audio, extract new mel frames, run streaming inference."""
        
        try:
            # Preprocess ALL accumulated audio
            audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(session.accumulated_audio)], device='cuda')
            
            if DEBUG_ASR:
                audio_hash = _hash_audio(session.accumulated_audio)
                logger.debug(f"Session {session.id}: process audio={len(session.accumulated_audio)} hash={audio_hash}")
            
            # Stage timing (DEBUG_ASR only): the cuda syncs needed for accurate
            # per-stage numbers add latency, so this must stay opt-in.
            if DEBUG_ASR:
                torch.cuda.synchronize()
                t_start = time.perf_counter()

            with torch.inference_mode(), self._autocast():
                mel, mel_len = self.model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_len
                )

                if DEBUG_ASR:
                    torch.cuda.synchronize()
                    t_preprocess = time.perf_counter()

                # Frame indices below are local to the trimmed buffer;
                # session.emitted_frames is global (since turn start).
                local_emitted = session.emitted_frames - session.frame_offset

                # Available frames (excluding last edge frame)
                available_frames = mel.shape[-1] - 1
                new_frame_count = available_frames - local_emitted

                emit_frames = (
                    self.first_chunk_frames if session.emitted_frames == 0
                    else self.shift_frames
                )
                if new_frame_count < emit_frames:
                    return session.current_text  # Not enough new frames

                # Extract chunk with pre-encode cache
                if session.emitted_frames == 0:
                    # First chunk: no cache; sized so later chunk starts stay >= 0
                    chunk_start = 0
                    chunk_end = emit_frames
                    drop_extra = 0
                else:
                    # Subsequent chunks: include pre_encode_cache frames before
                    chunk_start = local_emitted - self.pre_encode_cache_size
                    chunk_end = local_emitted + emit_frames
                    drop_extra = self.drop_extra

                chunk_mel = mel[:, :, chunk_start:chunk_end]
                chunk_len = torch.tensor([chunk_mel.shape[-1]], device='cuda')

                # Run streaming inference
                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )

                if DEBUG_ASR:
                    torch.cuda.synchronize()
                    t_step = time.perf_counter()
                    logger.debug(
                        f"Session {session.id} chunk timing: "
                        f"preprocess={(t_preprocess - t_start) * 1000:.2f}ms "
                        f"encoder+decoder={(t_step - t_preprocess) * 1000:.2f}ms "
                        f"(window={len(session.accumulated_audio)} samples)"
                    )

                # Update emitted frame count
                session.emitted_frames += emit_frames

                # Extract text
                new_text = session.current_text
                if transcribed_texts and transcribed_texts[0]:
                    hyp = transcribed_texts[0]
                    if hasattr(hyp, 'text'):
                        new_text = hyp.text
                    elif isinstance(hyp, str):
                        new_text = hyp
                    else:
                        new_text = str(hyp)

            # Keep the buffer bounded so per-chunk preprocessing and hard-reset
            # finalization cost stay flat instead of growing with the utterance
            self._trim_audio_buffer(session)
            return new_text

        except Exception as e:
            logger.error(f"Session {session.id} chunk processing error: {e}")
            logger.error(traceback.format_exc())
            return None

    def _trim_audio_buffer(self, session: ASRSession) -> None:
        """Drop already-emitted audio beyond the keep window, in hop-aligned units.

        Never drops the pre-encode cache + margin frames needed as left context
        for the next chunk, so trimming does not change what the encoder sees.
        """
        excess_samples = len(session.accumulated_audio) - self.keep_samples
        if excess_samples <= 0:
            return

        desired_drop = excess_samples // self.hop_samples
        local_emitted = session.emitted_frames - session.frame_offset
        max_drop = local_emitted - self.pre_encode_cache_size - CONTEXT_MARGIN_FRAMES
        drop = min(desired_drop, max_drop)
        if drop <= 0:
            return

        session.accumulated_audio = session.accumulated_audio[drop * self.hop_samples:]
        session.frame_offset += drop
    
    async def _reset_session(self, session: ASRSession, finalize: bool = True):
        """Handle reset with soft or hard finalization.
        
        Args:
            finalize: If True (hard reset), add padding and use keep_all_outputs=True
                      to capture trailing words, then reset decoder state.
                      If False (soft reset), just return current cumulative text
                      without forcing decoder output.
        
        Soft reset (finalize=False):
        - Speculatively finalizes on COPIES of the caches: returns the same
          text a hard reset would return right now (one extra inference,
          ~10-30ms), while real decoder/encoder state stays untouched
        - Used at silence onset so clients can start speculative LLM
          generation on text that will match the hard-reset final
        
        Hard reset (finalize=True):
        - Adds padding and processes with keep_all_outputs=True
        - Captures trailing words at segment boundaries
        - Resets decoder state to prevent corruption from multiple hard resets
        - Preserves encoder cache for acoustic context
        - Used on UserStoppedSpeakingFrame for complete transcription
        """        
        
        logger.info(f"Session {session.id} _reset_session START: finalize={finalize}")
        
        # Log audio state at reset for diagnostics
        audio_samples = len(session.accumulated_audio) if session.accumulated_audio is not None else 0
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        logger.debug(
            f"Session {session.id} {'hard' if finalize else 'soft'} reset: "
            f"accumulated={audio_samples} samples ({audio_duration_ms}ms), "
            f"emitted={session.emitted_frames} frames"
        )
        
        if not finalize:
            # SOFT RESET: speculatively finalize on *copies* of the session
            # state, so the returned text matches the upcoming hard reset
            # exactly while leaving encoder/decoder state untouched (the turn
            # may continue). Raw current_text lags the audio by the lookahead
            # and can end mid-word, which breaks clients that compare the soft
            # text against the hard final (e.g. LiveKit preemptive generation
            # only uses the speculative LLM reply on an exact text match).
            text = session.current_text
            if session.accumulated_audio is not None and len(session.accumulated_audio) > 0:
                spec_start = time.perf_counter()
                async with self.inference_lock:
                    spec_text = await asyncio.get_event_loop().run_in_executor(
                        None, self._speculative_final, session
                    )
                if spec_text is not None:
                    text = spec_text
                spec_ms = (time.perf_counter() - spec_start) * 1000
                logger.debug(f"Session {session.id} speculative finalization in {spec_ms:.1f}ms")
            
            logger.info(f"Session {session.id} soft reset: sending response")
            await session.websocket.send_json({
                "type": "transcript",
                "text": text,
                "is_final": True,
                "finalize": False  # Tell client this was soft reset
            })
            logger.info(f"Session {session.id} soft reset: response sent")
            
            logger.debug(f"Session {session.id} soft reset: '{text[-50:] if len(text) > 50 else text}'")
            logger.info(f"Session {session.id} _reset_session END: soft reset complete")
            # Keep all state intact - decoder, encoder, audio buffer
            return
        
        # HARD RESET: Full finalization with padding
        # Save original audio length before adding padding
        original_audio_length = len(session.accumulated_audio) if session.accumulated_audio is not None else 0
        
        # Pad with silence to ensure the model has enough trailing context
        # to finalize the last word. Padding = (right_context + 1) * shift_frames.
        if original_audio_length > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            session.accumulated_audio = np.concatenate([session.accumulated_audio, silence_padding])
        
        # Process all remaining audio with keep_all_outputs=True
        final_text = session.current_text
        if session.accumulated_audio is not None and len(session.accumulated_audio) > 0:
            start_time = time.perf_counter()
            async with self.inference_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_final_chunk, session
                )
                if text is not None:
                    final_text = text
                    session.current_text = text  # Update current_text for next soft reset
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Session {session.id} final chunk processed in {elapsed_ms:.1f}ms: '{final_text[-50:] if len(final_text) > 50 else final_text}'")
        
        # Since all state resets after every hard reset, final_text is exactly
        # this turn's transcript - the client emits it as-is.
        logger.info(f"Session {session.id} hard reset: sending final='{final_text[-50:] if len(final_text) > 50 else final_text}'")
        await session.websocket.send_json({
            "type": "transcript",
            "text": final_text,
            "is_final": True,
            "finalize": True  # Tell client this was hard reset
        })
        logger.info(f"Session {session.id} hard reset: response sent")

        # Clear all state after hard reset: audio buffer, decoder state, and
        # encoder cache all reset fresh so nothing carries over between turns.
        self._init_session(session)

        logger.debug(
            f"Session {session.id} hard reset complete, state fully reset for next turn"
        )

        logger.info(f"Session {session.id} _reset_session END: finalize={finalize}")
    
    def _speculative_final(self, session: ASRSession) -> Optional[str]:
        """Run hard-reset finalization on copies of the session state.

        Produces the exact text a hard reset would produce right now, without
        mutating the real session's caches, hypotheses, or counters. Caches are
        cloned (defends against any in-place updates inside the encoder step)
        and decoder hypotheses are deep-copied.
        """
        try:
            scratch = ASRSession(id=f"{session.id}-spec", websocket=None)
            scratch.emitted_frames = session.emitted_frames
            scratch.frame_offset = session.frame_offset
            scratch.current_text = session.current_text
            scratch.cache_last_channel = (
                session.cache_last_channel.clone()
                if session.cache_last_channel is not None else None
            )
            scratch.cache_last_time = (
                session.cache_last_time.clone()
                if session.cache_last_time is not None else None
            )
            scratch.cache_last_channel_len = (
                session.cache_last_channel_len.clone()
                if session.cache_last_channel_len is not None else None
            )
            scratch.previous_hypotheses = copy.deepcopy(session.previous_hypotheses)
            scratch.pred_out_stream = copy.deepcopy(session.pred_out_stream)

            padding = np.zeros(
                self.final_padding_frames * self.hop_samples, dtype=np.float32
            )
            scratch.accumulated_audio = np.concatenate(
                [session.accumulated_audio, padding]
            )
            return self._process_final_chunk(scratch)
        except Exception as e:
            logger.error(f"Session {session.id} speculative finalization error: {e}")
            logger.error(traceback.format_exc())
            return None

    def _process_final_chunk(self, session: ASRSession) -> Optional[str]:
        """Process all remaining audio with keep_all_outputs=True."""
        
        try:
            if len(session.accumulated_audio) == 0:
                return session.current_text
            
            # Preprocess ALL accumulated audio
            audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).cuda()
            audio_len = torch.tensor([len(session.accumulated_audio)], device='cuda')
            
            with torch.inference_mode(), self._autocast():
                mel, mel_len = self.model.preprocessor(
                    input_signal=audio_tensor,
                    length=audio_len
                )

                # For final chunk, use ALL remaining frames (including edge).
                # Indices are local to the trimmed buffer.
                local_emitted = session.emitted_frames - session.frame_offset
                total_mel_frames = mel.shape[-1]
                remaining_frames = total_mel_frames - local_emitted

                logger.debug(
                    f"Session {session.id} final chunk: "
                    f"total_mel={total_mel_frames}, emitted={session.emitted_frames}, "
                    f"offset={session.frame_offset}, remaining={remaining_frames}"
                )

                if remaining_frames <= 0:
                    logger.warning(f"Session {session.id}: No remaining frames to process!")
                    return session.current_text

                # Extract final chunk with pre-encode cache
                if session.emitted_frames == 0:
                    chunk_start = 0
                    drop_extra = 0
                else:
                    chunk_start = local_emitted - self.pre_encode_cache_size
                    drop_extra = self.drop_extra
                
                chunk_mel = mel[:, :, chunk_start:]
                chunk_len = torch.tensor([chunk_mel.shape[-1]], device='cuda')
                
                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self.model.conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=True,  # Final chunk - output all remaining
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )
                
                if transcribed_texts and transcribed_texts[0]:
                    hyp = transcribed_texts[0]
                    if hasattr(hyp, 'text'):
                        final_text = hyp.text
                    elif isinstance(hyp, str):
                        final_text = hyp
                    else:
                        final_text = str(hyp)
                    logger.debug(
                        f"Session {session.id} final chunk output: '{final_text[-50:] if len(final_text) > 50 else final_text}' "
                        f"(was: '{session.current_text[-30:] if len(session.current_text) > 30 else session.current_text}')"
                    )
                    return final_text
                
                logger.debug(f"Session {session.id} final chunk: no new text from model")
                return session.current_text
        
        except Exception as e:
            logger.error(f"Session {session.id} final chunk error: {e}")
            logger.error(traceback.format_exc())
            return None
    
    @modal.asgi_app()
    def api(self):
        """FastAPI app with ASR WebSocket endpoint."""
        
        web_app = FastAPI(
            title="Nemotron ASR Server (Modal)",
            description="Modal-deployed Nemotron streaming ASR server",
            version="1.0.0",
        )
        
        @web_app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "sample_rate": self.sample_rate,
                "right_context": RIGHT_CONTEXT,
                "precision": "bf16" if self.amp_dtype else "fp32",
            }
        
        @web_app.websocket("/")
        async def websocket_handler(websocket: WebSocket):
            """Handle WebSocket ASR streaming connection."""
            # Auth: enforced only when the deployment has ASR_AUTH_TOKEN set.
            # Client passes it as ?token=... on the WebSocket URL.
            expected_token = os.environ.get("ASR_AUTH_TOKEN", "")
            if expected_token:
                provided = websocket.query_params.get("token", "")
                if not hmac.compare_digest(provided, expected_token):
                    # Reject before accepting the handshake
                    await websocket.close(code=1008)
                    logger.warning("Rejected WebSocket connection with bad/missing token")
                    return

            await websocket.accept()
            
            session_id = str(uuid.uuid4())[:8]
            session = ASRSession(id=session_id, websocket=websocket)
            self.sessions[session_id] = session
            
            logger.info(f"Client {session_id} connected")
            
            try:
                # Initialize session
                async with self.inference_lock:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._init_session, session
                    )
                
                await websocket.send_json({"type": "ready"})
                logger.debug(f"Client {session_id}: sent ready")
                
                while True:
                    # Receive message
                    message = await websocket.receive()
                    
                    # FastAPI WebSocket messages have a 'type' field indicating the message type
                    # 'websocket.receive' = binary data, 'websocket.disconnect' = connection closed
                    msg_type = message.get("type", "")
                    
                    if msg_type == "websocket.disconnect":
                        break
                    
                    # Check if there's text data (JSON control messages)
                    if "text" in message and message["text"]:
                        # JSON message
                        try:
                            data = json.loads(message["text"])
                            data_type = data.get("type")
                            logger.info(f"Client {session_id}: received text message type={data_type}, data={data}")
                            
                            if data_type == "reset" or data_type == "end":
                                # finalize=True (default): hard reset with padding + keep_all_outputs
                                # finalize=False: soft reset, just return current text
                                finalize = data.get("finalize", True)
                                logger.info(f"Client {session_id}: calling _reset_session(finalize={finalize})")
                                try:
                                    await self._reset_session(session, finalize=finalize)
                                    logger.info(f"Client {session_id}: _reset_session completed successfully")
                                except Exception as e:
                                    logger.error(f"Client {session_id}: _reset_session failed: {e}")
                                    logger.error(traceback.format_exc())
                                    raise
                            else:
                                logger.warning(f"Client {session_id}: unknown message type: {data_type}")
                        
                        except json.JSONDecodeError:
                            logger.warning(f"Client {session_id}: invalid JSON")
                    
                    # Check if there's binary data (audio)
                    if "bytes" in message and message["bytes"]:
                        audio_bytes = message["bytes"]
                        await self._handle_audio(session, audio_bytes)
            
            except WebSocketDisconnect:
                logger.info(f"Client {session_id} disconnected")
            
            except Exception as e:
                logger.error(f"Client {session_id} error: {e}")
                logger.error(traceback.format_exc())
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except:
                    pass
            
            finally:
                if session_id in self.sessions:
                    del self.sessions[session_id]
        
        return web_app


# Local development entrypoint
if __name__ == "__main__":
    """Local entrypoint for testing the deployed ASR service."""
    import json
    from pathlib import Path
    
    print("Nemotron ASR Server - Modal Deployment Test")
    print("============================================")
    print()
    
    # Get the deployed API URL
    print("Getting deployed API URL...")
    ASRClass = modal.Cls.from_name("nemotron-asr-server", "NemotronASRModel")
    api_url = ASRClass().api.web_url
    print(f"API URL: {api_url}")
    print()
    
    # Test WebSocket streaming endpoint
    print("Testing WebSocket streaming endpoint...")
    
    async def test_websocket():
        try:
            import websockets
            import wave
        except ImportError:
            print("⚠️  websockets library not installed.")
            print("   Install with: pip install websockets")
            return
        
        # Load test audio file
        test_audio_file = Path("tests/fixtures/harvard_16k.wav")
        if not test_audio_file.exists():
            print(f"⚠️  Test audio file not found: {test_audio_file}")
            print("   Please provide a 16kHz WAV file for testing")
            return
        
        print(f"Loading test audio: {test_audio_file}")
        with wave.open(str(test_audio_file), 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
        
        duration_sec = n_frames / sample_rate
        print(f"Audio: {n_frames} frames, {sample_rate}Hz, {n_channels}ch, {duration_sec:.1f}s")
        print()
        
        # Convert to WebSocket URL
        ws_url = api_url.replace("https://", "wss://").replace("http://", "ws://")
        
        try:
            async with websockets.connect(ws_url, max_size=10_000_000) as websocket:
                print("📡 Connected to WebSocket")
                
                # Wait for ready message
                msg = await websocket.recv()
                data = json.loads(msg)
                print(f"✓  Received: {data}")
                print()
                
                # Send audio in chunks (simulate streaming)
                chunk_size = 3200  # 100ms at 16kHz (16000 samples/sec * 0.1s * 2 bytes)
                chunk_count = 0
                start_time = time.time()
                
                print("📤 Sending audio chunks...")
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    await websocket.send(chunk)
                    chunk_count += 1
                    
                    # Receive interim transcripts
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(msg)
                        if data.get("type") == "transcript" and not data.get("is_final"):
                            print(f"   Interim #{chunk_count}: {data['text']}")
                    except asyncio.TimeoutError:
                        pass
                    
                    # Small delay to simulate real-time streaming
                    await asyncio.sleep(0.01)
                
                # Send end signal
                await websocket.send(json.dumps({"type": "end"}))
                print()
                print("📥 Sent end signal, waiting for final transcript...")
                
                # Keep receiving until we get the final transcript
                # (there may be interim transcripts in flight before the final one)
                final_text = None
                timeout_count = 0
                max_timeout = 50  # 5 seconds total (50 * 0.1s)
                
                while timeout_count < max_timeout:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(msg)
                        
                        if data.get("type") == "transcript":
                            if data.get("is_final"):
                                # Got the final transcript
                                final_text = data["text"]
                                print(f"   ✓ Received final transcript (is_final={data.get('is_final')}, finalize={data.get('finalize')})")
                                break
                            else:
                                # Interim transcript still being processed
                                print(f"   (Draining interim: {data['text'][-50:] if len(data['text']) > 50 else data['text']})")
                        else:
                            print(f"⚠️  Unexpected message type: {data.get('type')}")
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        if timeout_count % 10 == 0:
                            print(f"   (Still waiting... {timeout_count * 0.1:.1f}s)")
                        continue
                
                if timeout_count >= max_timeout:
                    print("⚠️  Timeout waiting for final transcript")
                
                elapsed = time.time() - start_time
                
                if final_text is not None:
                    print()
                    print("✅ Final transcript received!")
                    print(f"   Text: '{final_text}'")
                    print(f"   Chunks sent: {chunk_count}")
                    print(f"   Total time: {elapsed:.2f}s")
                    print(f"   RTF: {elapsed/duration_sec:.2f}x")
                else:
                    print("⚠️  No final transcript received")
        
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
            import traceback
            traceback.print_exc()
    
    asyncio.run(test_websocket())
    
    print()
    print("=" * 60)
    print()
    print("Deployment Info:")
    print("================")
    print(f"API Base URL: {api_url}")
    print()
    print("Available Endpoints:")
    print("  GET /health  - Health check")
    print("  WS  /        - WebSocket streaming ASR")
