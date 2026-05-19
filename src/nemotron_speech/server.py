"""WebSocket ASR server for Nemotron-Speech with true incremental streaming."""

import asyncio
import argparse
import copy
import contextlib
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from aiohttp import web, WSMsgType
from loguru import logger

# Enable debug logging with DEBUG_ASR=1
DEBUG_ASR = os.environ.get("DEBUG_ASR", "0") == "1"

_DEFAULT_FINALIZE_SILENCE_MS = 2500
_MAX_FINALIZE_SILENCE_MS = 10_000

STREAMING = "STREAMING"
PENDING_FINALIZE = "PENDING_FINALIZE"
FINALIZED = "FINALIZED"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e


def _hash_audio(audio: np.ndarray) -> str:
    """Get short hash of audio array for debugging."""
    if audio is None or len(audio) == 0:
        return "empty"
    return hashlib.md5(audio.tobytes()).hexdigest()[:8]


def tensor_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def clone_tree(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Tensor-aware deepcopy for disposable ASR fork state."""
    if memo is None:
        memo = {}

    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if torch.is_tensor(obj):
        return tensor_clone(obj)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        cloned_list: list[Any] = []
        memo[oid] = cloned_list
        cloned_list.extend(clone_tree(item, memo) for item in obj)
        return cloned_list
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        cloned_tuple = tuple(clone_tree(item, memo) for item in obj)
        memo[oid] = cloned_tuple
        return cloned_tuple
    if isinstance(obj, dict):
        cloned_dict: dict[Any, Any] = {}
        memo[oid] = cloned_dict
        for key, value in obj.items():
            cloned_dict[clone_tree(key, memo)] = clone_tree(value, memo)
        return cloned_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for field in dataclasses.fields(obj):
            setattr(cloned_obj, field.name, clone_tree(getattr(obj, field.name), memo))
        return cloned_obj

    # Some NeMo helper objects may live inside optional fields. Avoid copying
    # modules or other heavy objects; copy plain object state only when present.
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for key, value in vars(obj).items():
            setattr(cloned_obj, key, clone_tree(value, memo))
        return cloned_obj

    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def clone_hypotheses_deep(previous_hypotheses: Any) -> Any:
    """Deep-copy each Hypothesis and recursively clone all tensor fields."""
    if previous_hypotheses is None:
        return None
    return [clone_tree(hyp) for hyp in previous_hypotheses]


def _tensor_assert_hash(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.md5()
    digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    digest.update(str(cpu_tensor.dtype).encode("utf-8"))
    digest.update(cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:12]


def _array_assert_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.md5()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:12]


def _assert_tree_equal(label: str, before: Any, after: Any) -> None:
    if torch.is_tensor(before) or torch.is_tensor(after):
        if not (torch.is_tensor(before) and torch.is_tensor(after)):
            raise AssertionError(f"{label}: tensor/non-tensor mismatch")
        same_meta = (
            tuple(before.shape) == tuple(after.shape)
            and before.dtype == after.dtype
            and before.device == after.device
        )
        same_bytes = same_meta and torch.equal(before, after)
        if not same_bytes:
            raise AssertionError(
                f"{label}: tensor changed "
                f"before(shape={tuple(before.shape)}, dtype={before.dtype}, "
                f"device={before.device}, hash={_tensor_assert_hash(before)}) "
                f"after(shape={tuple(after.shape)}, dtype={after.dtype}, "
                f"device={after.device}, hash={_tensor_assert_hash(after)})"
            )
        return

    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        if not (isinstance(before, np.ndarray) and isinstance(after, np.ndarray)):
            raise AssertionError(f"{label}: ndarray/non-ndarray mismatch")
        if (
            before.shape != after.shape
            or before.dtype != after.dtype
            or not np.array_equal(before, after)
        ):
            raise AssertionError(
                f"{label}: ndarray changed "
                f"before(shape={before.shape}, dtype={before.dtype}, "
                f"hash={_array_assert_hash(before)}) "
                f"after(shape={after.shape}, dtype={after.dtype}, "
                f"hash={_array_assert_hash(after)})"
            )
        return

    if before is None or after is None or isinstance(before, (str, bytes, int, float, bool)):
        if before != after:
            raise AssertionError(f"{label}: value changed from {before!r} to {after!r}")
        return

    if isinstance(before, list) or isinstance(after, list):
        if not (isinstance(before, list) and isinstance(after, list)):
            raise AssertionError(f"{label}: list/non-list mismatch")
        if len(before) != len(after):
            raise AssertionError(f"{label}: list length changed {len(before)} -> {len(after)}")
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            _assert_tree_equal(f"{label}[{index}]", before_item, after_item)
        return

    if isinstance(before, tuple) or isinstance(after, tuple):
        if not (isinstance(before, tuple) and isinstance(after, tuple)):
            raise AssertionError(f"{label}: tuple/non-tuple mismatch")
        if len(before) != len(after):
            raise AssertionError(f"{label}: tuple length changed {len(before)} -> {len(after)}")
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            _assert_tree_equal(f"{label}[{index}]", before_item, after_item)
        return

    if isinstance(before, dict) or isinstance(after, dict):
        if not (isinstance(before, dict) and isinstance(after, dict)):
            raise AssertionError(f"{label}: dict/non-dict mismatch")
        if before.keys() != after.keys():
            raise AssertionError(f"{label}: dict keys changed")
        for key in before:
            _assert_tree_equal(f"{label}[{key!r}]", before[key], after[key])
        return

    before_is_dataclass = dataclasses.is_dataclass(before) and not isinstance(before, type)
    after_is_dataclass = dataclasses.is_dataclass(after) and not isinstance(after, type)
    if before_is_dataclass or after_is_dataclass:
        if not (before_is_dataclass and after_is_dataclass and before.__class__ is after.__class__):
            raise AssertionError(f"{label}: dataclass type changed")
        for field in dataclasses.fields(before):
            _assert_tree_equal(
                f"{label}.{field.name}",
                getattr(before, field.name),
                getattr(after, field.name),
            )
        return

    before_is_nemo_obj = hasattr(before, "__dict__") and before.__class__.__module__.startswith("nemo.")
    after_is_nemo_obj = hasattr(after, "__dict__") and after.__class__.__module__.startswith("nemo.")
    if before_is_nemo_obj or after_is_nemo_obj:
        if not (before_is_nemo_obj and after_is_nemo_obj and before.__class__ is after.__class__):
            raise AssertionError(f"{label}: NeMo object type changed")
        if vars(before).keys() != vars(after).keys():
            raise AssertionError(f"{label}: NeMo object fields changed")
        for key in vars(before):
            _assert_tree_equal(f"{label}.{key}", getattr(before, key), getattr(after, key))
        return

    try:
        equal = before == after
    except Exception:
        equal = repr(before) == repr(after)
    if not equal:
        raise AssertionError(f"{label}: object changed from {before!r} to {after!r}")

# Default model - HuggingFace model name (auto-downloads) or local .nemo path
DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"

# Right context options for att_context_size=[70, X]
RIGHT_CONTEXT_OPTIONS = {
    0: "~80ms ultra-low latency",
    1: "~160ms low latency (recommended)",
    6: "~560ms balanced",
    13: "~1.12s highest accuracy",
}


@dataclass
class ASRSession:
    """Per-connection session state with caches for true incremental streaming."""

    id: str
    websocket: Any

    # Legacy/debug audio buffer name. Step 6b keeps this bounded to pending
    # audio only; the preprocessor must never see a growing full-stream buffer.
    accumulated_audio: Optional[np.ndarray] = None

    # Raw audio not yet advanced past `emitted_frames * hop_samples`.
    pending_audio: Optional[np.ndarray] = None

    # Total real audio samples received in this session, excluding synthetic
    # finalization padding.
    total_audio_samples: int = 0

    # STFT boundary state: trailing window_size - hop samples before pending.
    raw_audio_ring: Optional[np.ndarray] = None

    # Cache-aware chunker state: trailing pre_encode_cache_size mel frames.
    mel_frame_ring: Optional[torch.Tensor] = None

    # Number of mel frames already emitted to encoder
    emitted_frames: int = 0

    # Encoder cache state
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None

    # Decoder state
    previous_hypotheses: Any = None
    pred_out_stream: Any = None

    # Current transcription (model's cumulative output)
    current_text: str = ""

    # Last text emitted to client on hard reset (for server-side deduplication)
    # We only send the delta (new portion) to avoid downstream duplication
    last_emitted_text: str = ""

    # Continuous-context mode state. These fields are only used when
    # NEMOTRON_CONTINUOUS=1; the default hard-reset path ignores them.
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    continuous_state: str = STREAMING
    committed_text: str = ""
    continuous_event_queue: Optional[asyncio.Queue] = None
    continuous_worker_task: Optional[asyncio.Task] = None
    continuous_debounce_task: Optional[asyncio.Task] = None
    continuous_stop_seq: int = 0
    continuous_reset_seen: bool = False
    continuous_post_stop_audio: bytearray = field(default_factory=bytearray)

    # Audio overlap buffer for mid-utterance reset continuity
    # This preserves the last N ms of audio to provide encoder left-context
    # when a new segment starts after a reset
    overlap_buffer: Optional[np.ndarray] = None


class ASRServer:
    """WebSocket server for streaming ASR with true incremental processing."""

    def __init__(
        self,
        model: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        right_context: int = 1,
    ):
        self.model_name_or_path = model
        self.host = host
        self.port = port
        self.right_context = right_context
        self.model = None
        self.sample_rate = 16000
        # ASR benchmark server REQUIRES CUDA — fail fast, never silently fall
        # back to CPU (a CPU/wrong-device run yields invalid benchmark numbers
        # while looking 'fine'). Baseline hardening 2026-05-18 (dual review).
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the Nemotron ASR benchmark server "
                "(torch.cuda.is_available() is False); refusing to run on CPU."
            )

        # Inference lock
        self.inference_lock = asyncio.Lock()

        self.continuous_context = os.environ.get("NEMOTRON_CONTINUOUS", "") == "1"
        self.fork_assert_enabled = os.environ.get("NEMOTRON_FORK_ASSERT", "") == "1"
        self.finalize_silence_ms = _DEFAULT_FINALIZE_SILENCE_MS
        if self.continuous_context:
            self.finalize_silence_ms = _env_int(
                "NEMOTRON_FINALIZE_SILENCE_MS", _DEFAULT_FINALIZE_SILENCE_MS
            )
            if not (0 <= self.finalize_silence_ms < _MAX_FINALIZE_SILENCE_MS):
                raise ValueError(
                    "NEMOTRON_FINALIZE_SILENCE_MS must be >= 0 and < "
                    f"{_MAX_FINALIZE_SILENCE_MS}"
                )
        self.finalize_silence_seconds = self.finalize_silence_ms / 1000

        # Active sessions
        self.sessions: dict[str, ASRSession] = {}

        # Model loaded flag for health check
        self.model_loaded = False

        # Streaming parameters (calculated from model config)
        self.shift_frames = None
        self.pre_encode_cache_size = None
        self.hop_samples = None
        self.window_size_samples = None
        self.raw_audio_ring_samples = None
        self.preprocess_align_pad_samples = None
        self.preprocess_new_audio_samples = None
        self.constant_preprocess_frames = None
        self.constant_preprocess_samples = None
        self.stream_preprocess_valid_samples = None
        self.first_preprocess_mel_frame = None

        # Audio overlap for mid-utterance reset continuity (calculated in load_model)
        self.overlap_samples = None

    def load_model(self):
        """Load the NeMo ASR model with streaming configuration."""
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf

        # Detect if model is a local .nemo file or HuggingFace model name
        is_local_file = (
            self.model_name_or_path.endswith('.nemo') or
            os.path.exists(self.model_name_or_path)
        )

        if is_local_file:
            logger.info(f"Loading model from local file: {self.model_name_or_path}")
            self.model = nemo_asr.models.ASRModel.restore_from(
                self.model_name_or_path, map_location='cpu'
            )
        else:
            logger.info(f"Loading model from HuggingFace: {self.model_name_or_path}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_name_or_path, map_location='cpu'
            )
        self.model = self.model.cuda()
        logger.info("ASR model loaded on CUDA")

        # Configure attention context for streaming
        logger.info(f"Setting att_context_size=[70, {self.right_context}] ({RIGHT_CONTEXT_OPTIONS.get(self.right_context, 'custom')})")
        self.model.encoder.set_default_att_context_size([70, self.right_context])

        # Decoding strategy: greedy (default, Blackwell-safe) or beam via
        # NEMOTRON_DECODING=beam (experimental; may be slower / unsupported on
        # some GPUs).
        _decoding = os.environ.get("NEMOTRON_DECODING", "greedy").strip().lower()
        if _decoding == "beam":
            logger.info("Configuring BEAM (maes) decoding...")
            _decoding_cfg = OmegaConf.create({
                'strategy': 'beam',
                'beam': {
                    'beam_size': 4,
                    'search_type': 'maes',
                    'maes_num_steps': 2,
                    'return_best_hypothesis': True,
                }
            })
        else:
            logger.info("Configuring greedy decoding for Blackwell compatibility...")
            _decoding_cfg = OmegaConf.create({
                'strategy': 'greedy',
                'greedy': {
                    'max_symbols': 10,
                    'loop_labels': False,
                    'use_cuda_graph_decoder': False,
                }
            })
        self.model.change_decoding_strategy(decoding_cfg=_decoding_cfg)
        self.model.eval()

        # Disable dither for deterministic preprocessing
        self.model.preprocessor.featurizer.dither = 0.0

        # Get streaming config
        scfg = self.model.encoder.streaming_cfg
        logger.info(f"Streaming config: chunk_size={scfg.chunk_size}, shift_size={scfg.shift_size}")

        # Calculate parameters
        preprocessor_cfg = self.model.cfg.preprocessor
        hop_length_sec = preprocessor_cfg.get('window_stride', 0.01)
        window_size_sec = preprocessor_cfg.get('window_size', 0.025)
        featurizer = self.model.preprocessor.featurizer
        self.hop_samples = int(getattr(featurizer, "hop_length", int(hop_length_sec * self.sample_rate)))
        self.window_size_samples = int(
            getattr(featurizer, "win_length", int(window_size_sec * self.sample_rate))
        )

        # shift_size[1] = 16 frames for 160ms chunks
        self.shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, list) else scfg.shift_size

        # pre_encode_cache_size[1] = 9 frames
        pre_cache = scfg.pre_encode_cache_size
        self.pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, list) else pre_cache

        # drop_extra_pre_encoded for non-first chunks
        self.drop_extra = scfg.drop_extra_pre_encoded

        # Calculate silence padding for final chunk:
        # - right_context chunks for encoder lookahead
        # - 1 additional chunk for decoder finalization
        # With right_context=1, this is (1+1)*160ms = 320ms
        self.final_padding_frames = (self.right_context + 1) * self.shift_frames
        padding_ms = self.final_padding_frames * hop_length_sec * 1000

        # Constant-plan incremental preprocessor:
        # - raw ring is only STFT boundary context (window - hop)
        # - mel ring below is the cache-aware pre-encode context
        # - one hop of right-edge guard matches the current streaming gate
        self.raw_audio_ring_samples = self.window_size_samples - self.hop_samples
        self.preprocess_align_pad_samples = (
            self.hop_samples - (self.raw_audio_ring_samples % self.hop_samples)
        ) % self.hop_samples
        self.preprocess_new_audio_samples = (self.shift_frames + 1) * self.hop_samples
        self.stream_preprocess_valid_samples = (
            self.preprocess_align_pad_samples
            + self.raw_audio_ring_samples
            + self.preprocess_new_audio_samples
        )
        prefix_samples = self.preprocess_align_pad_samples + self.raw_audio_ring_samples
        if prefix_samples % self.hop_samples != 0:
            raise RuntimeError(
                "Constant preprocessor prefix must align to mel frame hops: "
                f"prefix={prefix_samples}, hop={self.hop_samples}"
            )
        self.first_preprocess_mel_frame = prefix_samples // self.hop_samples
        min_plan_frames = (
            self.first_preprocess_mel_frame
            + self.pre_encode_cache_size
            + self.shift_frames
            + self.final_padding_frames
            + 1
        )
        self.constant_preprocess_frames = 1 << (min_plan_frames - 1).bit_length()
        self.constant_preprocess_samples = (self.constant_preprocess_frames - 1) * self.hop_samples

        # Calculate audio overlap for mid-utterance reset continuity
        # Use pre_encode_cache_size frames = 90ms of left-context
        # This allows the encoder to have proper context when starting a new segment
        self.overlap_samples = self.pre_encode_cache_size * self.hop_samples
        overlap_ms = self.overlap_samples * 1000 / self.sample_rate

        shift_ms = self.shift_frames * hop_length_sec * 1000
        logger.info(f"Model loaded: {type(self.model).__name__}")
        logger.info(f"Shift size: {shift_ms:.0f}ms ({self.shift_frames} frames)")
        logger.info(f"Pre-encode cache: {self.pre_encode_cache_size} frames")
        logger.info(
            "Constant preprocessor plan: "
            f"K={self.constant_preprocess_samples} samples "
            f"({self.constant_preprocess_frames} STFT frames, min={min_plan_frames}) "
            f"(align={self.preprocess_align_pad_samples}, "
            f"raw_ring={self.raw_audio_ring_samples}, "
            f"new_audio={self.preprocess_new_audio_samples})"
        )
        logger.info(f"Final chunk padding: {padding_ms:.0f}ms ({self.final_padding_frames} frames)")
        logger.info(f"Audio overlap for resets: {overlap_ms:.0f}ms ({self.overlap_samples} samples)")

        # Warmup inference to ensure model is fully loaded on GPU
        # This prevents GPU memory issues when LLM starts later
        self._warmup()

    def _warmup(self):
        """Run warmup inference using streaming API to claim GPU memory.

        IMPORTANT: We use the streaming API (conformer_stream_step) for warmup,
        NOT the batch API (model.transcribe). The batch API corrupts internal
        model state and causes subsequent streaming inference to become
        non-deterministic. See docs/asr-determinism-investigation.md.
        """
        import time

        logger.info("Running warmup inference (streaming API) to claim GPU memory...")
        start = time.perf_counter()

        # Keep warmup on the same fixed preprocessor plan as live streaming.
        warmup_audio = np.zeros(self.constant_preprocess_samples, dtype=np.float32)

        # Run streaming inference to force all CUDA kernels to compile
        with torch.inference_mode():
            mel, mel_len = self._preprocess_fixed_audio(warmup_audio, len(warmup_audio))

            # Get initial cache
            cache = self.model.encoder.get_initial_cache_state(batch_size=1)

            # Run streaming step (processes entire mel as one chunk)
            _ = self.model.conformer_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache[0],
                cache_last_time=cache[1],
                cache_last_channel_len=cache[2],
                keep_all_outputs=True,
                previous_hypotheses=None,
                previous_pred_out=None,
                drop_extra_pre_encoded=0,
                return_transcription=True,
            )

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Warmup complete in {elapsed:.0f}ms - GPU memory claimed")

    def _init_session(self, session: ASRSession):
        """Initialize a fresh session.

        If an overlap_buffer is present from a previous segment, it will be
        prepended to the accumulated audio to provide encoder left-context.
        This enables seamless transcription across mid-utterance resets.
        """
        # Initialize encoder cache
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        session.cache_last_channel = cache[0]
        session.cache_last_time = cache[1]
        session.cache_last_channel_len = cache[2]

        # Reset audio buffer and frame counter
        # If overlap buffer exists, use it as the starting audio
        if session.overlap_buffer is not None and len(session.overlap_buffer) > 0:
            session.pending_audio = session.overlap_buffer.copy()
            session.accumulated_audio = session.pending_audio
            session.total_audio_samples = len(session.pending_audio)
            overlap_ms = len(session.overlap_buffer) * 1000 / self.sample_rate
            logger.debug(
                f"Session {session.id}: prepending {len(session.overlap_buffer)} samples "
                f"({overlap_ms:.0f}ms) of overlap audio"
            )
            session.overlap_buffer = None  # Clear after use
        else:
            session.pending_audio = np.array([], dtype=np.float32)
            session.accumulated_audio = session.pending_audio
            session.total_audio_samples = 0

        session.raw_audio_ring = np.zeros(self.raw_audio_ring_samples, dtype=np.float32)
        session.mel_frame_ring = None

        # (Removed 2026-05-18 baseline hardening: the NEMOTRON_ONSET_WARMUP_MS
        # buffer-prepend was an ineffective + buggy onset warm-up. PLAN Step 8
        # implements the correct conformer_stream_step warm-up from scratch.)
        session.emitted_frames = 0

        # Reset decoder state
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.current_text = ""

    def _preprocess_fixed_audio(
        self,
        audio: np.ndarray,
        valid_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the preprocessor with an invariant input shape.

        `valid_samples` may be shorter for final partial chunks; the tensor
        shape stays fixed so CUDA uses the same STFT/cuFFT plan every call.
        """
        if len(audio) != self.constant_preprocess_samples:
            raise ValueError(
                f"Expected fixed preprocessor input of {self.constant_preprocess_samples} samples, "
                f"got {len(audio)}"
            )
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).cuda()
        audio_len = torch.tensor([valid_samples], device='cuda', dtype=torch.long)
        return self.model.preprocessor(input_signal=audio_tensor, length=audio_len)

    def _build_fixed_preprocess_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Assemble align-pad + raw ring + new audio and zero-pad to K."""
        if len(raw_audio_ring) != self.raw_audio_ring_samples:
            raise ValueError(
                f"Expected raw ring of {self.raw_audio_ring_samples} samples, "
                f"got {len(raw_audio_ring)}"
            )
        prefix_len = self.preprocess_align_pad_samples + self.raw_audio_ring_samples
        valid_samples = prefix_len + len(new_audio)
        if valid_samples > self.constant_preprocess_samples:
            raise ValueError(
                f"Fixed preprocessor valid span {valid_samples} exceeds K={self.constant_preprocess_samples}"
            )

        audio = np.zeros(self.constant_preprocess_samples, dtype=np.float32)
        cursor = self.preprocess_align_pad_samples
        audio[cursor : cursor + self.raw_audio_ring_samples] = raw_audio_ring
        cursor += self.raw_audio_ring_samples
        audio[cursor : cursor + len(new_audio)] = new_audio
        return audio, valid_samples

    def _update_mel_frame_ring(self, session: ASRSession, new_mel: torch.Tensor) -> None:
        """Retain the mel pre-encode cache separately from raw STFT context."""
        if session.mel_frame_ring is None:
            combined = new_mel.detach()
        else:
            combined = torch.cat((session.mel_frame_ring, new_mel.detach()), dim=-1)
        session.mel_frame_ring = combined[:, :, -self.pre_encode_cache_size :].detach()

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket client connection."""
        import uuid

        ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
        await ws.prepare(request)

        session_id = str(uuid.uuid4())[:8]
        session = ASRSession(id=session_id, websocket=ws)
        self.sessions[session_id] = session

        logger.info(f"Client {session_id} connected")

        try:
            async with self.inference_lock:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._init_session, session
                )

            if self.continuous_context:
                self._start_continuous_session(session)

            await ws.send_str(json.dumps({"type": "ready"}))
            logger.debug(f"Client {session_id}: sent ready")

            async for msg in ws:
                if self.continuous_context:
                    await self._queue_continuous_ws_message(session, msg)
                    continue

                if msg.type == WSMsgType.BINARY:
                    await self._handle_audio(session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "reset" or msg_type == "end":
                            # finalize=True (default): hard reset with padding + keep_all_outputs
                            # finalize=False: soft reset, just return current text
                            finalize = data.get("finalize", True)
                            await self._reset_session(session, finalize=finalize)
                        elif msg_type == "vad_start" or msg_type == "vad_stop":
                            logger.debug(
                                f"Client {session_id}: received {msg_type} (no-op)"
                            )
                        else:
                            logger.warning(f"Client {session_id}: unknown message type: {msg_type}")

                    except json.JSONDecodeError:
                        logger.warning(f"Client {session_id}: invalid JSON")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"Client {session_id} WebSocket error: {ws.exception()}")
                    break

            logger.info(f"Client {session_id} disconnected")

        except Exception as e:
            logger.error(f"Client {session_id} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await ws.send_str(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except:
                pass
        finally:
            if self.continuous_context:
                await self._close_continuous_session(session)
            if session_id in self.sessions:
                del self.sessions[session_id]

        return ws

    def _start_continuous_session(self, session: ASRSession) -> None:
        """Start the ordered per-session event worker for continuous mode."""
        session.continuous_event_queue = asyncio.Queue()
        session.continuous_worker_task = asyncio.create_task(
            self._continuous_session_worker(session),
            name=f"nemotron-continuous-session-{session.id}",
        )
        logger.info(
            f"Session {session.id}: continuous context enabled "
            f"(debounce={self.finalize_silence_ms}ms)"
        )

    async def _queue_continuous_ws_message(self, session: ASRSession, msg) -> None:
        """Queue raw WS events so continuous-mode control/audio is ordered."""
        queue = session.continuous_event_queue
        if queue is None:
            logger.warning(f"Session {session.id}: continuous event queue missing")
            return

        if msg.type == WSMsgType.BINARY:
            await queue.put(("audio", msg.data))
        elif msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning(f"Client {session.id}: invalid JSON")
                return

            msg_type = data.get("type")
            if msg_type == "reset" or msg_type == "end":
                finalize = data.get("finalize", True)
                await queue.put(("reset", finalize, msg_type))
            elif msg_type == "vad_start" or msg_type == "vad_stop":
                await queue.put((msg_type,))
            else:
                logger.warning(f"Client {session.id}: unknown message type: {msg_type}")
        elif msg.type == WSMsgType.ERROR:
            logger.error(f"Client {session.id} WebSocket error: {session.websocket.exception()}")

    async def _continuous_session_worker(self, session: ASRSession) -> None:
        """Process continuous-mode events in arrival order."""
        queue = session.continuous_event_queue
        if queue is None:
            return

        while True:
            event = await queue.get()
            should_stop = event[0] == "close"
            try:
                event_type = event[0]

                async with session.state_lock:
                    if event_type == "close":
                        await self._continuous_handle_close_locked(session)
                    elif event_type == "audio":
                        await self._continuous_handle_audio_locked(session, event[1])
                    elif event_type == "vad_start":
                        await self._continuous_handle_vad_start_locked(session)
                    elif event_type == "vad_stop":
                        await self._continuous_handle_vad_stop_locked(session)
                    elif event_type == "reset":
                        await self._continuous_handle_reset_locked(
                            session,
                            finalize=event[1],
                            msg_type=event[2],
                        )
                    elif event_type == "debounce_expired":
                        await self._continuous_handle_debounce_expired_locked(
                            session,
                            stop_seq=event[1],
                        )
                    else:
                        logger.warning(
                            f"Session {session.id}: unknown continuous event {event_type}"
                        )
            except Exception as e:
                logger.error(f"Session {session.id} continuous worker error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    await session.websocket.send_str(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                except Exception:
                    pass
            finally:
                queue.task_done()
                if should_stop:
                    return

    async def _close_continuous_session(self, session: ASRSession) -> None:
        """Drain pending finalization through the worker and stop continuous mode."""
        queue = session.continuous_event_queue
        worker = session.continuous_worker_task
        if queue is not None and worker is not None and not worker.done():
            await queue.put(("close",))
            with contextlib.suppress(asyncio.CancelledError):
                await worker

        task = session.continuous_debounce_task
        session.continuous_debounce_task = None
        session.continuous_stop_seq += 1
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        session.continuous_event_queue = None
        session.continuous_worker_task = None
        session.continuous_post_stop_audio.clear()

    async def _continuous_debounce_timer(self, session_id: str, stop_seq: int) -> None:
        """Wake after server-side silence and enqueue a finalize decision."""
        try:
            await asyncio.sleep(self.finalize_silence_seconds)
            session = self.sessions.get(session_id)
            if session is None or session.continuous_event_queue is None:
                return
            await session.continuous_event_queue.put(("debounce_expired", stop_seq))
        except asyncio.CancelledError:
            pass

    async def _continuous_cancel_debounce_locked(
        self,
        session: ASRSession,
        *,
        invalidate: bool,
    ) -> None:
        task = session.continuous_debounce_task
        session.continuous_debounce_task = None
        if invalidate:
            session.continuous_stop_seq += 1

        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _continuous_handle_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
    ) -> None:
        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_post_stop_audio.extend(audio_bytes)
            if DEBUG_ASR:
                samples = len(session.continuous_post_stop_audio) // 2
                logger.debug(
                    f"Session {session.id}: held {len(audio_bytes)}B post-vad_stop "
                    f"audio ({samples} total samples) while pending finalize"
                )
            return

        await self._handle_audio_locked(session, audio_bytes)

    async def _continuous_flush_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        audio_bytes = bytes(session.continuous_post_stop_audio)
        session.continuous_post_stop_audio.clear()
        samples = len(audio_bytes) // 2
        logger.debug(
            f"Session {session.id}: flushing {samples} post-vad_stop samples "
            f"for {reason}"
        )
        await self._handle_audio_locked(session, audio_bytes)

    def _continuous_discard_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        samples = len(session.continuous_post_stop_audio) // 2
        session.continuous_post_stop_audio.clear()
        logger.debug(
            f"Session {session.id}: discarded {samples} post-vad_stop samples "
            f"at true boundary ({reason})"
        )

    async def _continuous_force_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        include_post_stop_audio: bool,
    ) -> None:
        await self._continuous_cancel_debounce_locked(session, invalidate=True)
        if include_post_stop_audio:
            await self._continuous_flush_post_stop_audio_locked(session, reason=reason)
        else:
            self._continuous_discard_post_stop_audio_locked(session, reason=reason)

        if not self._continuous_has_audio_or_text(session):
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            logger.debug(
                f"Session {session.id}: ignored empty forced continuous finalize "
                f"for {reason}"
            )
            return

        session.continuous_state = FINALIZED
        logger.debug(f"Session {session.id}: forced continuous finalize for {reason}")
        await self._continuous_finalize_and_reset_locked(session, reason=reason)

    async def _continuous_handle_close_locked(self, session: ASRSession) -> None:
        if (
            session.continuous_state == PENDING_FINALIZE
            or self._continuous_has_audio_or_text(session)
            or session.continuous_post_stop_audio
        ):
            await self._continuous_force_finalize_locked(
                session,
                reason="close",
                include_post_stop_audio=True,
            )
            return

        logger.debug(
            f"Session {session.id}: continuous close with no pending final"
        )

    async def _continuous_handle_vad_start_locked(self, session: ASRSession) -> None:
        if session.continuous_state == PENDING_FINALIZE:
            await self._continuous_cancel_debounce_locked(session, invalidate=True)
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            logger.debug(
                f"Session {session.id}: vad_start canceled pending finalize; "
                "continuing same ASR context"
            )
            await self._continuous_flush_post_stop_audio_locked(
                session,
                reason="vad_start",
            )
        else:
            logger.debug(
                f"Session {session.id}: vad_start in state={session.continuous_state}"
            )

    async def _continuous_handle_vad_stop_locked(self, session: ASRSession) -> None:
        await self._continuous_cancel_debounce_locked(session, invalidate=False)
        session.continuous_stop_seq += 1
        stop_seq = session.continuous_stop_seq
        session.continuous_state = PENDING_FINALIZE
        session.continuous_reset_seen = False
        session.continuous_debounce_task = asyncio.create_task(
            self._continuous_debounce_timer(session.id, stop_seq),
            name=f"nemotron-continuous-debounce-{session.id}-{stop_seq}",
        )
        logger.debug(
            f"Session {session.id}: vad_stop armed pending finalize seq={stop_seq} "
            f"({self.finalize_silence_ms}ms)"
        )

    async def _continuous_handle_reset_locked(
        self,
        session: ASRSession,
        *,
        finalize: bool,
        msg_type: str,
    ) -> None:
        if not finalize:
            text = session.current_text
            await session.websocket.send_str(json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": True,
                "finalize": False
            }))
            logger.debug(
                f"Session {session.id}: continuous soft reset: "
                f"'{text[-50:] if len(text) > 50 else text}'"
            )
            return

        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_reset_seen = True
            if msg_type == "end":
                await self._continuous_force_finalize_locked(
                    session,
                    reason=msg_type,
                    include_post_stop_audio=True,
                )
                return

            logger.debug(
                f"Session {session.id}: delayed client {msg_type} while "
                "server debounce is pending"
            )
            return

        if self._continuous_has_audio_or_text(session):
            logger.debug(
                f"Session {session.id}: immediate continuous {msg_type} without "
                "pending VAD stop"
            )
            await self._continuous_force_finalize_locked(
                session,
                reason=msg_type,
                include_post_stop_audio=True,
            )
            return

        logger.debug(
            f"Session {session.id}: ignored empty continuous {msg_type} in "
            f"state={session.continuous_state}"
        )

    async def _continuous_handle_debounce_expired_locked(
        self,
        session: ASRSession,
        *,
        stop_seq: int,
    ) -> None:
        if (
            session.continuous_state != PENDING_FINALIZE
            or stop_seq != session.continuous_stop_seq
        ):
            logger.debug(
                f"Session {session.id}: ignored stale debounce expiry "
                f"seq={stop_seq} current={session.continuous_stop_seq} "
                f"state={session.continuous_state}"
            )
            return

        session.continuous_debounce_task = None
        reset_seen = session.continuous_reset_seen
        session.continuous_reset_seen = False
        session.continuous_state = FINALIZED
        logger.debug(
            f"Session {session.id}: debounce expired seq={stop_seq}; "
            f"finalizing (reset_seen={reset_seen})"
        )
        await self._continuous_finalize_and_reset_locked(
            session,
            reason="reset_then_debounce" if reset_seen else "debounce_expired",
        )

    def _continuous_has_audio_or_text(self, session: ASRSession) -> bool:
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        return bool(session.current_text) or session.total_audio_samples > 0 or pending_len > 0

    def _build_continuous_finalize_fork(self, session: ASRSession) -> ASRSession:
        """Create a disposable fork for final padding without touching parent state."""
        pending_audio = (
            session.pending_audio.copy()
            if session.pending_audio is not None
            else np.array([], dtype=np.float32)
        )
        padding_samples = 0
        if session.total_audio_samples > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            pending_audio = np.concatenate([pending_audio, silence_padding])

        fork = ASRSession(id=f"{session.id}:fork", websocket=None)
        fork.pending_audio = pending_audio
        fork.accumulated_audio = fork.pending_audio
        fork.total_audio_samples = session.total_audio_samples + padding_samples
        fork.raw_audio_ring = (
            session.raw_audio_ring.copy()
            if session.raw_audio_ring is not None
            else np.zeros(self.raw_audio_ring_samples, dtype=np.float32)
        )
        fork.mel_frame_ring = clone_tree(session.mel_frame_ring)
        fork.emitted_frames = session.emitted_frames
        fork.cache_last_channel = (
            tensor_clone(session.cache_last_channel)
            if session.cache_last_channel is not None
            else None
        )
        fork.cache_last_time = (
            tensor_clone(session.cache_last_time)
            if session.cache_last_time is not None
            else None
        )
        fork.cache_last_channel_len = (
            tensor_clone(session.cache_last_channel_len)
            if session.cache_last_channel_len is not None
            else None
        )
        fork.previous_hypotheses = clone_hypotheses_deep(session.previous_hypotheses)
        fork.pred_out_stream = clone_tree(session.pred_out_stream)
        fork.current_text = session.current_text
        fork.last_emitted_text = session.last_emitted_text
        fork.committed_text = session.committed_text
        return fork

    def _snapshot_fork_assert_parent(self, session: ASRSession) -> dict[str, Any]:
        return {
            "cache_last_channel": (
                tensor_clone(session.cache_last_channel)
                if session.cache_last_channel is not None
                else None
            ),
            "cache_last_time": (
                tensor_clone(session.cache_last_time)
                if session.cache_last_time is not None
                else None
            ),
            "cache_last_channel_len": (
                tensor_clone(session.cache_last_channel_len)
                if session.cache_last_channel_len is not None
                else None
            ),
            "previous_hypotheses": clone_hypotheses_deep(session.previous_hypotheses),
        }

    def _assert_fork_flush_parent_unchanged(
        self,
        session: ASRSession,
        snapshot: dict[str, Any],
    ) -> None:
        try:
            _assert_tree_equal(
                "cache_last_channel",
                snapshot["cache_last_channel"],
                session.cache_last_channel,
            )
            _assert_tree_equal(
                "cache_last_time",
                snapshot["cache_last_time"],
                session.cache_last_time,
            )
            _assert_tree_equal(
                "cache_last_channel_len",
                snapshot["cache_last_channel_len"],
                session.cache_last_channel_len,
            )
            _assert_tree_equal(
                "previous_hypotheses",
                snapshot["previous_hypotheses"],
                session.previous_hypotheses,
            )
        except AssertionError as e:
            logger.error(
                f"Session {session.id}: fork alias assertion FAILED: {e}"
            )
            raise

        logger.info(
            f"Session {session.id}: fork alias assertion PASSED "
            "(parent cache tensors + previous_hypotheses byte-identical)"
        )

    async def _continuous_finalize_and_reset_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        """Finalize once on a disposable fork, then reset the parent at boundary."""
        import time

        audio_samples = session.total_audio_samples
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        held_len = len(session.continuous_post_stop_audio) // 2
        logger.debug(
            f"Session {session.id} continuous finalize ({reason}): "
            f"audio={audio_samples} samples ({audio_duration_ms}ms), "
            f"pending={pending_len} samples, held_post_stop={held_len} samples, "
            f"emitted={session.emitted_frames} frames"
        )

        final_text = session.current_text
        should_flush = (
            session.total_audio_samples > 0
            or (session.pending_audio is not None and len(session.pending_audio) > 0)
        )
        if should_flush:
            parent_snapshot = (
                self._snapshot_fork_assert_parent(session)
                if self.fork_assert_enabled
                else None
            )
            fork = self._build_continuous_finalize_fork(session)
            start_time = time.perf_counter()
            if fork.pending_audio is not None and len(fork.pending_audio) > 0:
                async with self.inference_lock:
                    text = await asyncio.get_event_loop().run_in_executor(
                        None, self._process_final_chunk, fork
                    )
                    if text is not None:
                        final_text = text
            if parent_snapshot is not None:
                self._assert_fork_flush_parent_unchanged(session, parent_snapshot)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Session {session.id} continuous fork final chunk processed in "
                f"{elapsed_ms:.1f}ms: "
                f"'{final_text[-50:] if len(final_text) > 50 else final_text}'"
            )

        if final_text.startswith(session.committed_text):
            delta_text = final_text[len(session.committed_text):].lstrip()
        else:
            delta_text = final_text
            logger.debug(
                f"Session {session.id}: continuous ASR correction detected, "
                f"committed='{session.committed_text[-30:]}', "
                f"new='{final_text[-30:]}'"
            )

        session.committed_text = final_text
        session.last_emitted_text = final_text

        if delta_text:
            try:
                await session.websocket.send_str(json.dumps({
                    "type": "transcript",
                    "text": delta_text,
                    "is_final": True,
                    "finalize": True
                }))
                logger.debug(
                    f"Session {session.id} continuous final: delta='{delta_text}' "
                    f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}')"
                )
            except Exception as e:
                logger.warning(
                    f"Session {session.id}: failed to send continuous final "
                    f"for {reason}: {e}"
                )
        else:
            logger.debug(
                f"Session {session.id}: suppressed empty/duplicate continuous final "
                f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}')"
            )

        # True utterance boundary: now it is safe to cold-reset the ASR state.
        session.committed_text = ""
        session.last_emitted_text = ""
        session.overlap_buffer = None
        session.continuous_post_stop_audio.clear()
        session.continuous_reset_seen = False
        session.continuous_stop_seq += 1
        self._init_session(session)
        session.continuous_state = STREAMING

        logger.debug(
            f"Session {session.id}: continuous true-boundary reset complete"
        )

    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames available."""
        await self._handle_audio_locked(session, audio_bytes)

    async def _handle_audio_locked(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames available."""
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if DEBUG_ASR:
            chunk_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
            logger.debug(f"Session {session.id}: recv chunk {len(audio_bytes)}B hash={chunk_hash}")

        session.pending_audio = np.concatenate([session.pending_audio, audio_np])
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples += len(audio_np)

        # Process if we have enough audio for new frames
        # We need shift_frames worth of new mel frames (after skipping edge frame)
        min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples

        while session.total_audio_samples >= min_audio_for_chunk:
            async with self.inference_lock:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_chunk, session
                )

            if text is not None and text != session.current_text:
                session.current_text = text
                logger.debug(f"Session {session.id} interim: {text[-50:] if len(text) > 50 else text}")
                await session.websocket.send_str(json.dumps({
                    "type": "transcript",
                    "text": text,
                    "is_final": False
                }))

            # Update minimum for next iteration
            min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples

    def _process_chunk(self, session: ASRSession) -> Optional[str]:
        """Process one fixed-plan audio window and run streaming inference."""
        try:
            if len(session.pending_audio) < self.preprocess_new_audio_samples:
                return session.current_text

            if DEBUG_ASR:
                audio_hash = _hash_audio(session.pending_audio)
                logger.debug(
                    f"Session {session.id}: process pending={len(session.pending_audio)} "
                    f"total={session.total_audio_samples} hash={audio_hash}"
                )

            new_audio = session.pending_audio[: self.preprocess_new_audio_samples]
            fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
                session.raw_audio_ring,
                new_audio,
            )

            with torch.inference_mode():
                mel, mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)

                if DEBUG_ASR:
                    mel_hash = hashlib.md5(mel.cpu().numpy().tobytes()).hexdigest()[:8]
                    logger.debug(f"Session {session.id}: mel shape={mel.shape[-1]} hash={mel_hash}")

                valid_new_mel = mel[
                    :,
                    :,
                    self.first_preprocess_mel_frame : self.first_preprocess_mel_frame + self.shift_frames,
                ]

                # Extract chunk with pre-encode cache
                if session.emitted_frames == 0:
                    # First chunk: just shift_frames, no mel cache
                    chunk_mel = valid_new_mel
                    drop_extra = 0
                else:
                    # Subsequent chunks: prepend retained mel pre-encode cache
                    chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
                    drop_extra = self.drop_extra

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

                # Update emitted frame count
                consumed_audio = session.pending_audio[: self.shift_frames * self.hop_samples]
                if len(consumed_audio) >= self.raw_audio_ring_samples:
                    session.raw_audio_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
                else:
                    keep = self.raw_audio_ring_samples - len(consumed_audio)
                    session.raw_audio_ring = np.concatenate(
                        [session.raw_audio_ring[-keep:], consumed_audio]
                    ).astype(np.float32, copy=False)
                session.pending_audio = session.pending_audio[self.shift_frames * self.hop_samples :]
                session.accumulated_audio = session.pending_audio
                self._update_mel_frame_ring(session, valid_new_mel)
                session.emitted_frames += self.shift_frames

                # Extract text
                if transcribed_texts and transcribed_texts[0]:
                    hyp = transcribed_texts[0]
                    if hasattr(hyp, 'text'):
                        return hyp.text
                    elif isinstance(hyp, str):
                        return hyp
                    else:
                        return str(hyp)

                return session.current_text

        except Exception as e:
            logger.error(f"Session {session.id} chunk processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def _reset_session(self, session: ASRSession, finalize: bool = True):
        """Handle reset with soft or hard finalization.

        Args:
            finalize: If True (hard reset), add padding and use keep_all_outputs=True
                      to capture trailing words, then reset decoder state.
                      If False (soft reset), just return current cumulative text
                      without forcing decoder output.

        Soft reset (finalize=False):
        - Returns current_text as is_final (model's streaming output)
        - No audio processing, no decoder finalization
        - Decoder state preserved (no corruption)
        - Used on VADUserStoppedSpeakingFrame for fast response

        Hard reset (finalize=True):
        - Adds padding and processes with keep_all_outputs=True
        - Captures trailing words at segment boundaries
        - Resets decoder state to prevent corruption from multiple hard resets
        - Preserves encoder cache for acoustic context
        - Used on UserStoppedSpeakingFrame for complete transcription
        """
        import time

        # Log audio state at reset for diagnostics
        audio_samples = session.total_audio_samples
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        logger.debug(
            f"Session {session.id} {'hard' if finalize else 'soft'} reset: "
            f"audio={audio_samples} samples ({audio_duration_ms}ms), "
            f"pending={len(session.pending_audio)} samples, "
            f"emitted={session.emitted_frames} frames"
        )

        if not finalize:
            # SOFT RESET: Return current text without processing
            # This is fast (~0ms) and doesn't corrupt decoder state.
            # The model's current_text is already cumulative (contains all text
            # from session start), so we just return it directly.
            # We don't concatenate with cumulative_text to avoid duplication.
            text = session.current_text

            await session.websocket.send_str(json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": True,
                "finalize": False  # Tell client this was soft reset
            }))

            logger.debug(f"Session {session.id} soft reset: '{text[-50:] if len(text) > 50 else text}'")
            # Keep all state intact - decoder, encoder, audio buffer
            return

        # HARD RESET: Full finalization with padding
        # Save original audio length before adding padding
        original_audio_length = session.total_audio_samples

        # Pad with silence to ensure the model has enough trailing context
        # to finalize the last word. Padding = (right_context + 1) * shift_frames.
        if original_audio_length > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            session.pending_audio = np.concatenate([session.pending_audio, silence_padding])
            session.accumulated_audio = session.pending_audio

        # Process all remaining audio with keep_all_outputs=True
        final_text = session.current_text
        if session.pending_audio is not None and len(session.pending_audio) > 0:
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

        # Server-side deduplication: only send the delta (new portion)
        # This avoids downstream duplication when aggregators concatenate transcripts
        if final_text.startswith(session.last_emitted_text):
            delta_text = final_text[len(session.last_emitted_text):].lstrip()
        else:
            # ASR corrected earlier text - send full text
            # (This is rare but can happen with model corrections)
            delta_text = final_text
            logger.debug(
                f"Session {session.id}: ASR correction detected, "
                f"last='{session.last_emitted_text[-30:]}', new='{final_text[-30:]}'"
            )

        # Update tracking state before sending
        session.last_emitted_text = final_text

        # Send only the delta to client
        await session.websocket.send_str(json.dumps({
            "type": "transcript",
            "text": delta_text,
            "is_final": True,
            "finalize": True  # Tell client this was hard reset
        }))

        logger.debug(
            f"Session {session.id} hard reset: delta='{delta_text}' "
            f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}')"
        )

        # MEMORY BOUNDING: Clear all state after hard reset
        # This prevents unbounded memory growth by resetting completely each turn:
        # - Audio buffer: cleared (no carryover between turns)
        # - Decoder state: reset fresh (no hypothesis accumulation)
        # - Encoder cache: re-initialized
        #
        # We considered keeping audio overlap for encoder context continuity,
        # but since we reset the encoder cache, overlap audio would just be
        # re-transcribed, causing duplicates. Clean reset avoids this.

        session.last_emitted_text = ""
        session.overlap_buffer = None
        self._init_session(session)

        logger.debug(
            f"Session {session.id} hard reset complete, state fully reset for next turn"
        )

    def _process_final_chunk(self, session: ASRSession) -> Optional[str]:
        """Process remaining pending audio with fixed-plan preprocessing."""
        try:
            if len(session.pending_audio) == 0:
                return session.current_text

            with torch.inference_mode():
                # For final chunk, use ALL remaining frames (including edge)
                padded_total_samples = (
                    session.emitted_frames * self.hop_samples + len(session.pending_audio)
                )
                total_mel_frames = (padded_total_samples // self.hop_samples) + 1
                remaining_frames = total_mel_frames - session.emitted_frames

                logger.debug(
                    f"Session {session.id} final chunk: "
                    f"total_mel={total_mel_frames}, emitted={session.emitted_frames}, "
                    f"remaining={remaining_frames}"
                )

                if remaining_frames <= 0:
                    logger.warning(f"Session {session.id}: No remaining frames to process!")
                    return session.current_text

                pending = session.pending_audio
                raw_ring = session.raw_audio_ring
                new_mels: list[torch.Tensor] = []
                frames_collected = 0
                while frames_collected < remaining_frames:
                    frames_this_call = min(self.shift_frames, remaining_frames - frames_collected)
                    needed_new_samples = min(
                        len(pending),
                        self.preprocess_new_audio_samples,
                    )
                    new_audio = pending[:needed_new_samples]
                    fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
                        raw_ring,
                        new_audio,
                    )
                    mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
                    start = self.first_preprocess_mel_frame
                    new_mels.append(mel[:, :, start : start + frames_this_call])

                    if frames_this_call == self.shift_frames:
                        consumed_samples = min(self.shift_frames * self.hop_samples, len(pending))
                        consumed_audio = pending[:consumed_samples]
                        if len(consumed_audio) >= self.raw_audio_ring_samples:
                            raw_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
                        elif len(consumed_audio) > 0:
                            keep = self.raw_audio_ring_samples - len(consumed_audio)
                            raw_ring = np.concatenate([raw_ring[-keep:], consumed_audio]).astype(
                                np.float32,
                                copy=False,
                            )
                        pending = pending[consumed_samples:]
                    frames_collected += frames_this_call

                new_mel = torch.cat(new_mels, dim=-1)

                # Extract final chunk with pre-encode cache
                if session.emitted_frames == 0:
                    chunk_mel = new_mel
                    drop_extra = 0
                else:
                    chunk_mel = torch.cat((session.mel_frame_ring, new_mel), dim=-1)
                    drop_extra = self.drop_extra

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
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy" if self.model_loaded else "loading",
            "model_loaded": self.model_loaded,
        })

    async def start(self):
        """Start the HTTP + WebSocket server."""
        self.load_model()
        self.model_loaded = True

        logger.info(f"Starting streaming ASR server on ws://{self.host}:{self.port}")

        app = web.Application()
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/", self.websocket_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"ASR server listening on ws://{self.host}:{self.port}")
        logger.info(f"Health check available at http://{self.host}:{self.port}/health")
        await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description="Nemotron Streaming ASR WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model name or path to local .nemo file"
    )
    parser.add_argument(
        "--right-context",
        type=int,
        default=1,
        choices=[0, 1, 6, 13],
        help="Right context frames: 0=80ms, 1=160ms, 6=560ms, 13=1.12s latency"
    )
    args = parser.parse_args()

    server = ASRServer(
        model=args.model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
    )

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
