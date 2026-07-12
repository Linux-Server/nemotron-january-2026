"""Modal deployment for Chatterbox TTS (Resemble AI) with WebSocket streaming.

Based on the official Modal example (https://modal.com/docs/examples/chatterbox_tts),
extended with the same WebSocket protocol as the Magpie TTS server so it drops into
the Pipecat voice pipeline: the LLM streams sentence segments over the socket and
audio for each segment streams back while the LLM generates the next one.

ChatterboxTurboTTS has no token-level streaming API (generate() returns the full
waveform), so streaming is segment-level: each sentence is synthesized as soon as
it arrives and sent back as chunked PCM. Sentence-level pipelining is what the bot
does anyway (see pipecat_bots/sentence_buffer.py).

Client compatibility: pipecat_bots/magpie_websocket_tts.py works unchanged — point
it at this server's URL with sample_rate=24000 (Chatterbox outputs 24kHz, not
Magpie's 22kHz). "mode"/"preset" fields are accepted and ignored (single path).

WebSocket protocol (/ws/tts/stream):
    -> {"type": "init", "voice": ..., "language": ...}   voice = prompt wav name
    -> {"type": "text", "text": ...}                     one segment to synthesize
    -> {"type": "close"}                                 finish pending, then done
    -> {"type": "cancel"}                                drop queue, stop now
    -> {"type": "ping"}
    <- binary PCM s16le mono 24kHz
    <- {"type": "stream_created", "stream_id": ...}
    <- {"type": "segment_complete", "segment": n, "audio_ms": ...}
    <- {"type": "done", "total_audio_ms": ..., "segments_generated": ...}
    <- {"type": "error", "message": ..., "fatal": ...}

Usage:
    # (one-time) HF token secret — Chatterbox Turbo weights need it
    modal secret create hf-token HF_TOKEN=<your-huggingface-token>

    # (optional) upload voice prompts for cloning; init "voice" picks <name>.wav
    modal volume create chatterbox-tts-voices
    modal volume put chatterbox-tts-voices <PATH-TO-VOICE-PROMPTS-DIR>

    # Deploy to Modal
    modal deploy -m src.nemotron_speech.modal.chatter_tts

    # Smoke-test the deployed service (batch + websocket streaming)
    modal run -m src.nemotron_speech.modal.chatter_tts
    python -m src.nemotron_speech.modal.chatter_tts
"""

import asyncio
import io
import json
import os
import pathlib
import time
import uuid
from collections.abc import Callable, Iterator

import modal
import numpy as np

app = modal.App("chatterbox-tts")

# Persist HF model weights across container restarts
model_cache = modal.Volume.from_name("chatterbox-tts-model-cache", create_if_missing=True)
CACHE_PATH = "/model-cache"

# Optional voice prompts (reference .wav files) for voice cloning
voices_vol = modal.Volume.from_name("chatterbox-tts-voices", create_if_missing=True)
VOICE_PROMPTS_DIR = "/chatterbox-tts/prompts"

# Chatterbox (S3Gen vocoder) output sample rate
CHATTERBOX_SAMPLE_RATE = 24000

# PCM chunk size for websocket sends (~85ms at 24kHz s16le)
WS_CHUNK_BYTES = 4096

# ---------------------------------------------------------------------------
# Streaming synthesis tuning
# ---------------------------------------------------------------------------
# ChatterboxTurboTTS.generate() runs the full T3 autoregressive token loop and
# vocodes the whole segment before returning, so TTFB equals the entire
# segment's synthesis time (~500ms even for two words). The streaming path in
# `_synthesize_pcm_chunks` instead renders audio at growing token thresholds
# while the AR loop is still running: first audio leaves after
# FIRST_CHUNK_TOKENS tokens instead of after the whole segment.
#
# S3 speech tokens are 25Hz (40ms of audio each); mels are 50Hz (2 per token,
# 480 samples per mel frame at 24kHz).
FIRST_CHUNK_TOKENS = int(os.getenv("CHATTERBOX_FIRST_CHUNK_TOKENS", "16"))
# Later chunks double from this size up to the cap; each flow render re-runs
# the causal decoder over the whole token prefix, so fewer/bigger later chunks
# keep total decoder work bounded while the audio cushion grows.
CHUNK_TOKENS_MIN = 32
CHUNK_TOKENS_MAX = 256
# HiFT vocoder chunk continuity (CosyVoice2 scheme): re-vocode the last
# MEL_CACHE_FRAMES mel frames with the cached excitation source, hold back the
# tail, and crossfade it into the next chunk.
MEL_CACHE_FRAMES = 8
SAMPLES_PER_MEL_FRAME = 480  # 24kHz / 50Hz mel rate
SOURCE_CACHE_SAMPLES = MEL_CACHE_FRAMES * SAMPLES_PER_MEL_FRAME

# Optional bf16 autocast for the T3 AR loop (roughly halves per-token time on
# Ada/Hopper GPUs; sampling changes slightly, so listen before enabling).
T3_BF16 = os.getenv("CHATTERBOX_T3_BF16", "0") == "1"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_PATH,
            "TORCH_HOME": CACHE_PATH,
        }
    )
    .uv_pip_install(
        "chatterbox-tts==0.1.6",
        "fastapi[standard]==0.124.4",
        "peft==0.18.0",
        "hf_transfer",
    )
)

with image.imports():
    import torch
    import torch.nn.functional as F
    import torchaudio as ta
    from chatterbox.models.s3gen.const import S3GEN_SIL
    from chatterbox.models.s3gen.s3gen import S3Token2Mel
    from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals, punc_norm
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    from transformers.generation.logits_process import (
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )


def _find_voice_prompt(name: str | None) -> str | None:
    """Resolve a voice prompt .wav from the voices volume, or None for the default voice."""
    if not name:
        return None
    root = pathlib.Path(VOICE_PROMPTS_DIR)
    for candidate in root.rglob(f"{name}.wav"):
        return str(candidate)
    return None


@app.cls(
    region="ap", routing_region="ap-south",
    gpu="L40S",
    image=image,
    min_containers=1,
    scaledown_window=60 * 5,
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={
        CACHE_PATH: model_cache,
        VOICE_PROMPTS_DIR: voices_vol,
    },
)
@modal.concurrent(max_inputs=10)
class Chatterbox:
    """Modal class for Chatterbox Turbo TTS inference."""

    @modal.enter()
    def load(self) -> None:
        print("Loading Chatterbox Turbo TTS model...")
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        model_cache.commit()

        # The built-in voice; prepare_conditionals() mutates model.conds, so
        # keep a stable reference and cache cloned voices per prompt path.
        self._default_conds = self.model.conds
        self._voice_conds: dict[str, Conditionals] = {}

        # Warm up: first generate() JIT-compiles CUDA kernels; without this the
        # first websocket segment pays multi-second TTFB. Run both the batch
        # path and the streaming path (finalize=False flow + cached HiFT).
        print("Warming up...")
        with torch.inference_mode():
            self.model.generate("Warm up the decoder and vocoder paths.")
        for _ in self._synthesize_pcm_chunks(
            "Warm up the streaming decoder and chunked vocoder paths as well.",
            self._default_conds,
        ):
            pass
        print(f"Model loaded and warm (sample rate: {self.model.sr}Hz)")

    def _get_conds(self, audio_prompt_path: str | None) -> "Conditionals":
        """Conditionals for a voice prompt, cached per path (VE + tokenizer +
        mel extraction cost ~100s of ms; segments reuse the same voice)."""
        if not audio_prompt_path:
            return self._default_conds
        conds = self._voice_conds.get(audio_prompt_path)
        if conds is None:
            self.model.prepare_conditionals(audio_prompt_path, exaggeration=0.0)
            conds = self.model.conds
            self._voice_conds[audio_prompt_path] = conds
        return conds

    def _synthesize_pcm_chunks(
        self,
        text: str,
        conds: "Conditionals",
        should_abort: Callable[[], bool] | None = None,
    ) -> Iterator[bytes]:
        """Streaming synthesis: yield PCM s16le chunks while generation runs.

        generate() waits for the full T3 token sequence and vocodes it all at
        once, so nothing leaves the GPU until the whole segment is done. This
        method uses the streaming seams already present in chatterbox-tts:

        - the T3 AR loop is replicated here (same sampling as generate()) so
          speech tokens are observable as they are produced;
        - the flow decoder is causal: called with finalize=False it renders
          mels for the current token prefix, trimming the 3-token lookahead
          whose mels would still change (CosyVoice2 design);
        - each render re-runs the flow over the full prefix but only mels past
          `mel_off` are new; those are vocoded with HiFT using the cached
          excitation source + an 8-frame mel overlap, and the held-back tail is
          crossfaded into the next chunk so boundaries are click-free.

        Noise for the flow is drawn once per segment and sliced per render so
        already-rendered mel frames stay consistent between calls.

        The batch endpoints keep generate()'s watermarking; chunk-wise
        watermarking would glitch at boundaries, so this path skips it.
        """
        model = self.model
        t3 = model.t3
        s3 = model.s3gen
        device = model.device
        lookahead = s3.flow.pre_lookahead_len  # 3 tokens
        mel_ratio = s3.flow.token_mel_ratio  # 2 mel frames per token

        text = punc_norm(text)
        text_tokens = model.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(device)

        # Same sampling setup as ChatterboxTurboTTS.generate() defaults.
        logits_processors = LogitsProcessorList(
            [
                TemperatureLogitsWarper(0.8),
                TopKLogitsWarper(1000),
                TopPLogitsWarper(0.95),
                RepetitionPenaltyLogitsProcessor(1.2),
            ]
        )

        with torch.inference_mode():
            prompt_feat = conds.gen["prompt_feat"]
            prompt_mels = prompt_feat.shape[1]
            # One noise draw per segment, sliced per render, covering the
            # prompt region too (noised_mels of full mu length => prompt_len=0
            # in the CFM, making every render's noise deterministic).
            max_gen_len = 1000
            noise = torch.randn(
                1,
                80,
                prompt_mels + (max_gen_len + lookahead + 3) * mel_ratio,
                device=device,
                dtype=s3.dtype,
            )

            gen_tokens: list[int] = []  # valid (< 6561) tokens for the vocoder
            mel_off = 0  # gen-region mel frames already vocoded
            mel_cache: torch.Tensor | None = None
            source_cache = torch.zeros(1, 1, 0, device=device, dtype=s3.dtype)
            speech_tail: torch.Tensor | None = None
            fade_window = torch.hamming_window(
                2 * SOURCE_CACHE_SAMPLES, periodic=False, device=device
            )

            def render(finalize: bool) -> bytes | None:
                nonlocal mel_off, mel_cache, source_cache, speech_tail
                toks = list(gen_tokens)
                if finalize:
                    toks += [S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]
                gen_mels = len(toks) * mel_ratio
                # NOTE: flow.inference(finalize=False) in chatterbox-tts 0.1.6
                # trims the lookahead from `h` but builds the decoder mask from
                # the untrimmed length and crashes on the size mismatch. Since
                # `finalize` controls nothing but that trim, always take the
                # finalize=True path and drop the unreliable lookahead frames
                # (no right-context yet, they'd change on the next render) here.
                reliable_mels = gen_mels
                if not finalize:
                    reliable_mels -= lookahead * mel_ratio
                if reliable_mels <= mel_off:
                    return None

                token_t = torch.tensor([toks], dtype=torch.long, device=device)
                mels = S3Token2Mel.forward(
                    s3,
                    token_t,
                    ref_wav=None,
                    ref_sr=None,
                    ref_dict=conds.gen,
                    finalize=True,
                    n_cfm_timesteps=2,
                    noised_mels=noise[:, :, : prompt_mels + gen_mels],
                ).to(dtype=s3.dtype)

                new = mels[:, :, mel_off:reliable_mels]
                mel_off = reliable_mels
                mel_in = new if mel_cache is None else torch.cat([mel_cache, new], dim=2)
                speech, source = s3.hift_inference(mel_in, source_cache)

                if speech_tail is None:
                    # Same ref-spillover fade-in generate() applies to the start.
                    speech[:, : len(s3.trim_fade)] *= s3.trim_fade
                else:
                    # Crossfade the held-back previous tail into this chunk.
                    n = SOURCE_CACHE_SAMPLES
                    speech[:, :n] = (
                        speech[:, :n] * fade_window[:n] + speech_tail * fade_window[n:]
                    )

                if finalize:
                    pcm = speech
                else:
                    mel_cache = mel_in[:, :, -MEL_CACHE_FRAMES:]
                    source_cache = source[:, :, -SOURCE_CACHE_SAMPLES:]
                    speech_tail = speech[:, -SOURCE_CACHE_SAMPLES:]
                    pcm = speech[:, :-SOURCE_CACHE_SAMPLES]

                audio_np = pcm.squeeze(0).detach().cpu().float().numpy()
                audio_np = np.clip(audio_np, -1.0, 1.0)
                return (audio_np * 32767).astype(np.int16).tobytes()

            # ---- T3 autoregressive loop (mirrors t3.inference_turbo) ----
            start_token = t3.hp.start_speech_token * torch.ones_like(
                text_tokens[:, :1]
            )
            embeds, _ = t3.prepare_input_embeds(
                t3_cond=conds.t3,
                text_tokens=text_tokens,
                speech_tokens=start_token,
                cfg_weight=0.0,
            )

            autocast = torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=T3_BF16
            )

            with autocast:
                out = t3.tfmr(inputs_embeds=embeds, use_cache=True)
            past = out.past_key_values
            logits = t3.speech_head(out[0][:, -1:].float())
            processed = logits_processors(start_token, logits[:, -1, :])
            next_token = torch.multinomial(F.softmax(processed, dim=-1), 1)

            generated = [next_token]
            if next_token.item() < t3.hp.start_speech_token:
                gen_tokens.append(next_token.item())

            next_emit = FIRST_CHUNK_TOKENS
            chunk_tokens = CHUNK_TOKENS_MIN

            for _ in range(max_gen_len):
                if should_abort is not None and should_abort():
                    return
                with autocast:
                    out = t3.tfmr(
                        inputs_embeds=t3.speech_emb(next_token),
                        past_key_values=past,
                        use_cache=True,
                    )
                past = out.past_key_values
                logits = t3.speech_head(out[0].float())
                input_ids = torch.cat(generated, dim=1)
                processed = logits_processors(input_ids, logits[:, -1, :])
                if torch.all(processed == -float("inf")):
                    print("Warning: All logits are -inf")
                    break
                next_token = torch.multinomial(F.softmax(processed, dim=-1), 1)
                generated.append(next_token)
                tok = next_token.item()
                if tok == t3.hp.stop_speech_token:
                    break
                if tok < t3.hp.start_speech_token:  # drop OOV/special tokens
                    gen_tokens.append(tok)

                if len(gen_tokens) >= next_emit:
                    chunk = render(finalize=False)
                    if chunk:
                        yield chunk
                    next_emit = len(gen_tokens) + chunk_tokens
                    chunk_tokens = min(chunk_tokens * 2, CHUNK_TOKENS_MAX)

            if gen_tokens and not (should_abort is not None and should_abort()):
                chunk = render(finalize=True)
                if chunk:
                    yield chunk

    def _synthesize_pcm(self, text: str, audio_prompt_path: str | None = None) -> bytes:
        """Synthesize text to raw PCM s16le bytes at model sample rate."""
        start = time.time()
        with torch.inference_mode():
            if audio_prompt_path:
                wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
            else:
                wav = self.model.generate(text)
        elapsed = time.time() - start

        audio_np = wav.squeeze().cpu().float().numpy()
        if np.abs(audio_np).max() > 1.0:
            audio_np = audio_np / np.abs(audio_np).max()
        audio_bytes: bytes = (audio_np * 32767).astype(np.int16).tobytes()

        duration_s = len(audio_bytes) / (self.model.sr * 2)
        print(
            f"TTS: {duration_s:.1f}s audio in {elapsed:.1f}s "
            f"(RTF={elapsed / max(duration_s, 1e-6):.2f}x) [{text[:50]}...]"
        )
        return audio_bytes

    @modal.method()
    def generate(self, prompt: str, voice_prompt: str | None = None) -> bytes:
        """Synthesize `prompt` to WAV bytes, optionally cloning a voice from the volume."""
        audio_prompt_path = _find_voice_prompt(voice_prompt)
        if voice_prompt and audio_prompt_path is None:
            raise ValueError(
                f"Voice prompt '{voice_prompt}.wav' not found in the chatterbox-tts-voices volume"
            )

        pcm = self._synthesize_pcm(prompt, audio_prompt_path)
        wav = torch.from_numpy(
            np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        ).unsqueeze(0)

        buffer = io.BytesIO()
        ta.save(buffer, wav, self.model.sr, format="wav")
        buffer.seek(0)
        return buffer.read()

    @modal.asgi_app()
    def api(self) -> "FastAPI":
        """FastAPI app: health check, batch synthesis, and websocket streaming."""
        from pydantic import BaseModel

        class SpeechRequest(BaseModel):
            input: str
            voice_prompt: str | None = None

        web_app = FastAPI(
            title="Chatterbox TTS Server (Modal)",
            description="Modal-deployed Resemble AI Chatterbox Turbo TTS",
            version="1.0.0",
        )

        @web_app.get("/health")
        async def health() -> dict:
            return {"status": "healthy", "model_loaded": self.model is not None}

        @web_app.get("/v1/audio/config")
        async def config() -> dict:
            return {
                "sample_rate": self.model.sr,
                "channels": 1,
                "encoding": "pcm_s16le",
            }

        @web_app.post("/v1/audio/speech")
        async def speech(request: SpeechRequest) -> "StreamingResponse":
            if not request.input.strip():
                raise HTTPException(status_code=400, detail="Empty input text")

            audio_prompt_path = _find_voice_prompt(request.voice_prompt)
            if request.voice_prompt and audio_prompt_path is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Voice prompt '{request.voice_prompt}.wav' not found",
                )

            pcm = await asyncio.to_thread(self._synthesize_pcm, request.input, audio_prompt_path)
            return StreamingResponse(
                io.BytesIO(pcm),
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(self.model.sr),
                    "X-Channels": "1",
                    "X-Encoding": "pcm_s16le",
                },
            )

        @web_app.websocket("/ws/tts/stream")
        async def websocket_tts_stream(websocket: WebSocket) -> None:
            """Segment-streaming TTS: same protocol as the Magpie server.

            Segments queue up as the LLM produces them; a background task
            synthesizes each one and streams its PCM while later segments
            keep arriving — sentence-level pipelining with the LLM.
            """
            await websocket.accept()

            stream_id: str = "--------"
            audio_prompt_path: str | None = None
            audio_task: asyncio.Task | None = None
            segment_queue: asyncio.Queue = asyncio.Queue()
            cancelled = asyncio.Event()

            async def send_audio() -> None:
                """Drain the segment queue: synthesize and stream each segment.

                Synthesis is chunk-streaming: `_synthesize_pcm_chunks` yields
                PCM while the token loop is still running, so the first bytes
                leave after ~FIRST_CHUNK_TOKENS tokens rather than after the
                whole segment. The blocking GPU generator runs in a worker
                thread and hands chunks over via an asyncio queue; `cancelled`
                aborts it between token steps (generate() couldn't be aborted
                mid-segment at all).
                """
                segments_generated = 0
                total_audio_ms = 0.0
                stream_start = time.time()
                first_audio = True
                loop = asyncio.get_running_loop()
                conds = self._get_conds(audio_prompt_path)

                try:
                    while True:
                        text = await segment_queue.get()
                        if text is None:  # close sentinel
                            break

                        chunk_q: asyncio.Queue[bytes | Exception | None] = (
                            asyncio.Queue()
                        )

                        def produce(text: str = text, q=chunk_q) -> None:
                            try:
                                for chunk in self._synthesize_pcm_chunks(
                                    text, conds, should_abort=cancelled.is_set
                                ):
                                    loop.call_soon_threadsafe(q.put_nowait, chunk)
                                loop.call_soon_threadsafe(q.put_nowait, None)
                            except Exception as e:  # noqa: BLE001 - forwarded
                                loop.call_soon_threadsafe(q.put_nowait, e)

                        producer = loop.run_in_executor(None, produce)
                        segment_bytes = 0
                        segment_start = time.time()
                        try:
                            while True:
                                item = await chunk_q.get()
                                if item is None:
                                    break
                                if isinstance(item, Exception):
                                    raise item
                                if cancelled.is_set():
                                    continue  # drain; producer aborts itself
                                segment_bytes += len(item)
                                if first_audio:
                                    first_audio = False
                                    ttfb = (time.time() - stream_start) * 1000
                                    print(
                                        f"[{stream_id[:8]}] First audio, "
                                        f"TTFB: {ttfb:.0f}ms"
                                    )
                                for i in range(0, len(item), WS_CHUNK_BYTES):
                                    await websocket.send_bytes(
                                        item[i : i + WS_CHUNK_BYTES]
                                    )
                        finally:
                            await producer

                        if cancelled.is_set():
                            break

                        segments_generated += 1
                        audio_ms = segment_bytes / (self.model.sr * 2) * 1000
                        total_audio_ms += audio_ms
                        elapsed = time.time() - segment_start
                        print(
                            f"[{stream_id[:8]}] Segment {segments_generated}: "
                            f"{audio_ms / 1000:.1f}s audio in {elapsed:.1f}s "
                            f"(RTF={elapsed / max(audio_ms / 1000, 1e-6):.2f}x) "
                            f"[{text[:50]}...]"
                        )
                        await websocket.send_json(
                            {
                                "type": "segment_complete",
                                "segment": segments_generated,
                                "audio_ms": audio_ms,
                            }
                        )

                    if not cancelled.is_set():
                        await websocket.send_json(
                            {
                                "type": "done",
                                "total_audio_ms": total_audio_ms,
                                "segments_generated": segments_generated,
                            }
                        )
                        print(
                            f"[{stream_id[:8]}] Stream complete: {total_audio_ms:.0f}ms "
                            f"audio, {segments_generated} segments"
                        )

                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - report to client before dying
                    print(f"[{stream_id[:8]}] Audio task error: {e}")
                    try:
                        await websocket.send_json(
                            {"type": "error", "message": str(e), "fatal": True}
                        )
                    except Exception:
                        pass

            async def stop_audio_task() -> None:
                nonlocal audio_task
                if audio_task is not None and not audio_task.done():
                    audio_task.cancel()
                    try:
                        await asyncio.wait_for(audio_task, timeout=0.5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                audio_task = None

            def new_stream() -> None:
                nonlocal segment_queue, stream_id
                segment_queue = asyncio.Queue()
                cancelled.clear()
                stream_id = str(uuid.uuid4())

            try:
                while True:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "init":
                        # "voice" names a prompt wav in the voices volume; unknown
                        # names (e.g. Magpie's "aria") fall back to the default voice.
                        voice = data.get("voice", "")
                        audio_prompt_path = _find_voice_prompt(voice)
                        await stop_audio_task()
                        new_stream()
                        await websocket.send_json(
                            {
                                "type": "stream_created",
                                "stream_id": stream_id,
                                "sample_rate": self.model.sr,
                            }
                        )
                        audio_task = asyncio.create_task(send_audio())

                    elif msg_type == "text":
                        text = data.get("text", "")
                        if not text.strip():
                            continue
                        # After close/cancel (or before init) start a fresh stream
                        if audio_task is None or audio_task.done():
                            new_stream()
                            await websocket.send_json(
                                {
                                    "type": "stream_created",
                                    "stream_id": stream_id,
                                    "sample_rate": self.model.sr,
                                }
                            )
                            audio_task = asyncio.create_task(send_audio())
                        await segment_queue.put(text)

                    elif msg_type == "close":
                        # Let queued segments finish, then the task sends "done"
                        await segment_queue.put(None)

                    elif msg_type == "cancel":
                        cancelled.set()
                        await stop_audio_task()

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

            except WebSocketDisconnect:
                print(f"WebSocket client disconnected [{stream_id[:8]}]")
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                cancelled.set()
                await stop_audio_task()

        return web_app


@app.local_entrypoint()
def test(
    prompt: str = "Chatterbox running on Modal [chuckle].",
    voice_prompt: str = "",
    output_path: str = "/tmp/chatterbox-tts/output.wav",
) -> None:
    """Generate a sample utterance and save it locally."""
    chatterbox = Chatterbox()
    audio_bytes = chatterbox.generate.remote(prompt, voice_prompt or None)

    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(audio_bytes)
    print(f"Audio saved to {out}")
    print(f"Play with: ffplay {out}")


if __name__ == "__main__":
    # Test the deployed websocket streaming endpoint:
    #   python -m src.nemotron_speech.modal.chatter_tts
    import wave

    async def _test_websocket() -> None:
        import websockets

        ChatterboxCls = modal.Cls.from_name("chatterbox-tts", "Chatterbox")
        api_url = ChatterboxCls().api.get_web_url()
        ws_url = api_url.replace("https://", "wss://").replace("http://", "ws://")
        print(f"Connecting to {ws_url}/ws/tts/stream")

        segments = [
            "Hello there! ",
            "This is Chatterbox streaming from Modal, segment by segment. ",
            "Each sentence is synthesized while the next one is still queued. ",
        ]
        chunks: list[bytes] = []
        sample_rate = CHATTERBOX_SAMPLE_RATE

        async with websockets.connect(f"{ws_url}/ws/tts/stream", max_size=None) as ws:
            await ws.send(json.dumps({"type": "init", "voice": ""}))
            msg = json.loads(await ws.recv())
            sample_rate = msg.get("sample_rate", sample_rate)
            print(f"stream_created: {msg['stream_id'][:8]} @ {sample_rate}Hz")

            start = time.time()
            for seg in segments:
                await ws.send(json.dumps({"type": "text", "text": seg}))
            await ws.send(json.dumps({"type": "close"}))

            first_chunk_at = None
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    if first_chunk_at is None:
                        first_chunk_at = time.time()
                        print(f"TTFB: {(first_chunk_at - start) * 1000:.0f}ms")
                    chunks.append(msg)
                else:
                    data = json.loads(msg)
                    print(f"{data['type']}: {data}")
                    if data["type"] in ("done", "error"):
                        break

        if chunks:
            out = pathlib.Path("/tmp/chatterbox-tts/streaming_output.wav")
            out.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sample_rate)
                f.writeframes(b"".join(chunks))
            print(f"Saved {sum(len(c) for c in chunks)} bytes to {out}")
            print(f"Play with: ffplay {out}")

    asyncio.run(_test_websocket())
