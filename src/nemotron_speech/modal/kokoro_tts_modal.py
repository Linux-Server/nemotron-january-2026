"""Modal deployment of the Kokoro streaming TTS engine.

A thin serving layer over `nemotron_speech.kokoro_streaming_tts`: the chunking,
buffering and synthesis logic lives in that module (framework-agnostic, runs
anywhere), and this file only adds the Modal container, the GPU-warm class, and
a generic HTTP + WebSocket API. Nothing here is specific to any client
framework.

Endpoints:
    GET  /health                      liveness + load state
    GET  /config                      sample rate, encoding, default voice
    GET  /voices                      available voice names + language
    POST /v1/audio/speech             synthesize; streams chunks as rendered
                                      {"input", "voice", "speed",
                                       "response_format": "pcm" | "wav"}
    WS   /ws/tts                      streaming text in, streaming audio out

Audio is always 24kHz mono. "pcm" is raw s16le (streams incrementally, first
bytes leave as soon as the opening chunk renders); "wav" buffers the whole
utterance because a WAV header needs the final length.

WebSocket protocol (deliberately generic):
    -> a plain (non-JSON) text frame            append text to the buffer
    -> {"type": "config", "voice": ..., "speed": ...}
    -> {"type": "text", "text": ...}            append text to the buffer
    -> {"type": "flush"}                        synthesize what's buffered now
    -> {"type": "close"}                        flush, finish, then "done"
    -> {"type": "cancel"}                       drop everything, stop now
    -> {"type": "ping"}
    <- {"type": "ready", "sample_rate": 24000, "encoding": "pcm_s16le", ...}
    <- binary frames: PCM s16le mono 24kHz
    <- {"type": "chunk", "index", "text", "audio_ms", "latency_ms"}
    <- {"type": "done", "total_audio_ms", "chunks"}
    <- {"type": "error", "message", "fatal"}

Text may be pushed incrementally (LLM tokens, one word at a time); the engine's
SentenceBuffer decides when enough has accumulated to synthesize, flushing the
first sentence eagerly so audio starts early.

Usage:
    modal deploy -m src.nemotron_speech.modal.kokoro_tts_modal

    modal run -m src.nemotron_speech.modal.kokoro_tts_modal        # batch test
    python -m src.nemotron_speech.modal.kokoro_tts_modal           # ws test
"""

import asyncio
import io
import json
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

app = modal.App("kokoro-tts")

# Persist HF weights + voice packs across container restarts
model_cache = modal.Volume.from_name("kokoro-tts-model-cache", create_if_missing=True)
CACHE_PATH = "/model-cache"

GPU = "L4"  # Kokoro is 82M params; anything with a GPU is far faster than realtime
SAMPLE_RATE = 24000  # fixed by the model
DEFAULT_VOICE = "af_heart"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("espeak-ng")  # misaki's G2P fallback shells out to this binary
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_PATH,
            "TORCH_HOME": CACHE_PATH,
        }
    )
    .uv_pip_install(
        "kokoro==0.9.4",
        "misaki[en]==0.9.4",
        "fastapi[standard]==0.124.4",
        "hf_transfer",
    )
    # Ship the engine itself; everything below just calls into it.
    .add_local_python_source("nemotron_speech")
)

with image.imports():
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import Response, StreamingResponse
    from pydantic import BaseModel

    from nemotron_speech.kokoro_streaming_tts import (
        LANG_NAMES,
        ChunkConfig,
        KokoroStreamingTTS,
        SentenceBuffer,
        write_wav,
    )


@app.cls(
    # "ap" is the whole AP continent, not ap-south: paired with an ap-south
    # ingress the worker lands in a different AP region and EVERY message pays
    # an inter-region hop. Measured here as a 132ms WebSocket round trip, which
    # alone blew the TTS latency budget. Same fix as the vLLM app on 2026-08-02.
    # NOTE: changing region needs `modal app stop kokoro-tts --yes` before
    # deploying — with min_containers=1 a plain redeploy leaves the old
    # container serving and the latency looks unchanged.
    region="ap-south",
    routing_region="ap-south",
    gpu=GPU,
    image=image,
    min_containers=1,
    scaledown_window=60 * 5,
    timeout=3600,
    volumes={CACHE_PATH: model_cache},
)
@modal.concurrent(max_inputs=8)
class Kokoro:
    """GPU-warm Kokoro engine behind a generic API."""

    @modal.enter()
    def load(self) -> None:
        # One place to tune latency vs. prosody for the whole deployment.
        # first_chunk_chars is the dominant term in time-to-first-audio:
        # render time scales with the length of the first chunk.
        self.chunk_config = ChunkConfig()
        self.tts = KokoroStreamingTTS(voice=DEFAULT_VOICE, config=self.chunk_config)
        model_cache.commit()

        try:
            self.voices = self.tts.list_voices()
        except Exception as e:  # noqa: BLE001 - listing is informational
            print(f"Could not list voices ({e}); default only")
            self.voices = [DEFAULT_VOICE]

        # First forward pass JIT-compiles CUDA kernels and the first G2P call
        # loads the lexicon; without this the first real request pays seconds.
        self.tts.warmup()
        print(f"Kokoro warm on {self.tts.device}, {len(self.voices)} voices")

    def _resolve_voice(self, name: str | None) -> str:
        """Map a requested voice onto a real one; unknown names use the default
        rather than failing a request mid-stream."""
        if name and name in self.voices:
            return name
        if name:
            print(f"Unknown voice {name!r}, using {DEFAULT_VOICE}")
        return DEFAULT_VOICE

    # -- remote python API (no HTTP) ---------------------------------------

    @modal.method()
    def generate(self, text: str, voice: str | None = None, speed: float = 1.0) -> bytes:
        """Synthesize `text` and return a complete 24kHz mono WAV."""
        audio = self.tts.synthesize(text, self._resolve_voice(voice), speed)
        buffer = io.BytesIO()
        write_wav(buffer, audio)
        return buffer.getvalue()

    @modal.method()
    def stream_pcm(self, text: str, voice: str | None = None, speed: float = 1.0):
        """Generator method: yields PCM s16le chunks as they render.

        Call with `.remote_gen(...)` to consume from a local Python client
        without going through HTTP.
        """
        for chunk in self.tts.stream(text, self._resolve_voice(voice), speed):
            yield chunk.pcm_bytes

    # -- HTTP + WebSocket ---------------------------------------------------

    @modal.asgi_app()
    def api(self) -> "FastAPI":
        class SpeechRequest(BaseModel):
            input: str
            voice: str | None = None
            speed: float = 1.0
            response_format: str = "pcm"  # "pcm" (streams) | "wav" (buffered)

        web_app = FastAPI(
            title="Kokoro TTS (Modal)",
            description="Streaming Kokoro-82M text-to-speech",
            version="1.0.0",
        )

        @web_app.get("/health")
        async def health() -> dict:
            return {"status": "healthy", "device": self.tts.device, "voices": len(self.voices)}

        @web_app.get("/config")
        async def config() -> dict:
            return {
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "encoding": "pcm_s16le",
                "default_voice": DEFAULT_VOICE,
            }

        @web_app.get("/voices")
        async def voices() -> dict:
            return {
                "voices": [
                    {"name": v, "language": LANG_NAMES.get(v[0], "unknown")}
                    for v in self.voices
                ]
            }

        @web_app.post("/v1/audio/speech")
        async def speech(request: SpeechRequest) -> Any:
            if not request.input.strip():
                raise HTTPException(status_code=400, detail="Empty input text")
            voice = self._resolve_voice(request.voice)

            if request.response_format == "wav":
                # A WAV header needs the total length, so this one can't stream.
                wav = await asyncio.to_thread(
                    self._wav_bytes, request.input, voice, request.speed
                )
                return Response(content=wav, media_type="audio/wav")

            if request.response_format != "pcm":
                raise HTTPException(
                    status_code=400, detail="response_format must be 'pcm' or 'wav'"
                )

            async def pcm_stream():
                async for chunk in self.tts.astream(request.input, voice, request.speed):
                    yield chunk.pcm_bytes

            return StreamingResponse(
                pcm_stream(),
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Channels": "1",
                    "X-Encoding": "pcm_s16le",
                },
            )

        @web_app.websocket("/ws/tts")
        async def websocket_tts(websocket: WebSocket) -> None:
            """Streaming text in, streaming audio out.

            Received text goes into a SentenceBuffer; a worker task synthesizes
            each released piece while the client keeps sending, so synthesis of
            the opening sentence overlaps with the producer generating the rest.
            """
            await websocket.accept()

            session = uuid.uuid4().hex[:8]
            voice = DEFAULT_VOICE
            speed = 1.0
            buffer = SentenceBuffer(self.chunk_config)
            work: asyncio.Queue[str | None] = asyncio.Queue()
            cancelled = asyncio.Event()
            worker: asyncio.Task | None = None

            async def synthesize_loop() -> None:
                total_audio_ms = 0.0
                render_ms = 0.0  # GPU time only, excludes waiting for text
                emitted = 0
                started = time.perf_counter()
                first = True
                try:
                    while True:
                        piece = await work.get()
                        if piece is None:  # end sentinel
                            break
                        piece_start = time.perf_counter()
                        async for chunk in self.tts.astream(
                            piece, voice, speed, cancel=cancelled.is_set
                        ):
                            if cancelled.is_set():
                                break
                            if first:
                                first = False
                                # Two different numbers, and mixing them up
                                # hides where latency actually went: the wall
                                # clock includes waiting for the producer to
                                # send enough text, while latency_ms is the
                                # render itself. chunk audio_ms is the lever —
                                # render time scales with it.
                                print(
                                    f"[{session}] first audio "
                                    f"{(time.perf_counter() - started) * 1000:.0f}ms "
                                    f"(render {chunk.latency_ms:.0f}ms for "
                                    f"{chunk.duration_s * 1000:.0f}ms of audio)"
                                )
                            await websocket.send_bytes(chunk.pcm_bytes)
                            total_audio_ms += chunk.duration_s * 1000
                            await websocket.send_json(
                                {
                                    "type": "chunk",
                                    "index": emitted,
                                    "text": chunk.text,
                                    "audio_ms": chunk.duration_s * 1000,
                                    "latency_ms": chunk.latency_ms,
                                }
                            )
                            emitted += 1
                        render_ms += (time.perf_counter() - piece_start) * 1000
                        if cancelled.is_set():
                            break

                    if not cancelled.is_set():
                        await websocket.send_json(
                            {
                                "type": "done",
                                "total_audio_ms": total_audio_ms,
                                "chunks": emitted,
                            }
                        )
                        wall_ms = (time.perf_counter() - started) * 1000
                        print(
                            f"[{session}] done: {total_audio_ms:.0f}ms audio, "
                            f"{emitted} chunks in {wall_ms:.0f}ms "
                            f"(RTF {render_ms / max(total_audio_ms, 1e-6):.3f})"
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001 - tell the client before dying
                    print(f"[{session}] synthesis error: {e}")
                    try:
                        await websocket.send_json(
                            {"type": "error", "message": str(e), "fatal": True}
                        )
                    except Exception:
                        pass

            async def stop_worker() -> None:
                nonlocal worker
                if worker is not None and not worker.done():
                    worker.cancel()
                    try:
                        await asyncio.wait_for(worker, timeout=0.5)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        pass
                worker = None

            async def ensure_worker() -> None:
                nonlocal worker, buffer
                if worker is None or worker.done():
                    # Fresh session after a close/cancel.
                    buffer = SentenceBuffer(self.chunk_config)
                    cancelled.clear()
                    worker = asyncio.create_task(synthesize_loop())

            async def push(text: str) -> None:
                await ensure_worker()
                for ready in buffer.push(text):
                    await work.put(ready)

            await websocket.send_json(
                {
                    "type": "ready",
                    "session": session,
                    "sample_rate": SAMPLE_RATE,
                    "channels": 1,
                    "encoding": "pcm_s16le",
                    "voice": voice,
                }
            )

            try:
                while True:
                    message = await websocket.receive_text()
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        # A plain text frame is just text to speak.
                        await push(message)
                        continue
                    if not isinstance(data, dict):
                        await push(message)
                        continue

                    msg_type = data.get("type", "text")

                    if msg_type == "config":
                        voice = self._resolve_voice(data.get("voice"))
                        speed = float(data.get("speed", speed))
                        await websocket.send_json(
                            {"type": "config", "voice": voice, "speed": speed}
                        )

                    elif msg_type == "text":
                        text = data.get("text", "")
                        if text.strip():
                            await push(text)

                    elif msg_type == "flush":
                        await ensure_worker()
                        rest = buffer.flush()
                        if rest:
                            await work.put(rest)

                    elif msg_type == "close":
                        await ensure_worker()
                        rest = buffer.flush()
                        if rest:
                            await work.put(rest)
                        await work.put(None)  # worker emits "done" and exits

                    elif msg_type == "cancel":
                        cancelled.set()
                        await stop_worker()
                        while not work.empty():
                            work.get_nowait()
                        buffer.flush()

                    elif msg_type == "ping":
                        await websocket.send_json({"type": "pong"})

            except WebSocketDisconnect:
                print(f"[{session}] client disconnected")
            except Exception as e:  # noqa: BLE001
                print(f"[{session}] websocket error: {e}")
            finally:
                cancelled.set()
                await stop_worker()

        return web_app

    def _wav_bytes(self, text: str, voice: str, speed: float) -> bytes:
        audio = self.tts.synthesize(text, voice, speed)
        buffer = io.BytesIO()
        write_wav(buffer, audio)
        return buffer.getvalue()


@app.local_entrypoint()
def main(
    text: str = "Kokoro running on Modal. Eighty two million parameters, streaming chunk by chunk.",
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
    output: str = "/tmp/kokoro-modal/output.wav",
) -> None:
    """Batch smoke test: synthesize one utterance and save it locally."""
    wav = Kokoro().generate.remote(text, voice, speed)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav)
    print(f"Saved {len(wav)} bytes to {out}")
    print(f"Play with: ffplay -autoexit {out}")


if __name__ == "__main__":
    # Streaming smoke test against the deployed websocket endpoint:
    #   python -m src.nemotron_speech.modal.kokoro_tts_modal
    import wave

    async def _test_ws() -> None:
        import websockets

        api_url = modal.Cls.from_name("kokoro-tts", "Kokoro")().api.get_web_url()
        ws_url = api_url.replace("https://", "wss://").replace("http://", "ws://")
        print(f"Connecting to {ws_url}/ws/tts")

        # Pushed word by word, the way an LLM would emit tokens.
        script = (
            "Hello there. This is Kokoro streaming from Modal, pushed one word "
            "at a time. The first sentence is flushed eagerly so audio starts "
            "early, and later sentences are buffered for better prosody."
        )

        chunks: list[bytes] = []
        async with websockets.connect(f"{ws_url}/ws/tts", max_size=None) as ws:
            ready = json.loads(await ws.recv())
            print(f"ready: {ready}")

            start = time.perf_counter()

            async def send_words() -> None:
                for word in script.split(" "):
                    await ws.send(json.dumps({"type": "text", "text": word + " "}))
                    await asyncio.sleep(0.02)  # imitate token pacing
                await ws.send(json.dumps({"type": "close"}))

            sender = asyncio.create_task(send_words())
            first_at = None
            while True:
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    if first_at is None:
                        first_at = time.perf_counter()
                        print(f"time to first audio: {(first_at - start) * 1000:.0f}ms")
                    chunks.append(msg)
                else:
                    data = json.loads(msg)
                    print(f"{data['type']}: {data}")
                    if data["type"] in ("done", "error"):
                        break
            await sender

        if chunks:
            out = Path("/tmp/kokoro-modal/streaming.wav")
            out.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(out), "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(SAMPLE_RATE)
                f.writeframes(b"".join(chunks))
            print(f"Saved {sum(len(c) for c in chunks)} bytes to {out}")
            print(f"Play with: ffplay -autoexit {out}")

    asyncio.run(_test_ws())
