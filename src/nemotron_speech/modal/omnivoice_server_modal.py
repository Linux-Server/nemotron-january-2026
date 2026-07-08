"""Modal deployment for OmniVoice TTS server.

OmniVoice (k2-fsa/OmniVoice) is a zero-shot TTS built on a diffusion
language-model architecture over Qwen3-0.6B. It supports three modes:

  - Voice cloning:  text + ref_audio + ref_text  (clone a reference speaker)
  - Voice design:   text + instruct              (e.g. "female, low pitch, british accent")
  - Auto voice:     text only                    (model picks a voice)

Output is a list of float np.ndarray at 24 kHz.

Deploy to Modal with GPU support for fast TTS inference.

Usage:
    # Deploy to Modal
    modal deploy -m src.nemotron_speech.modal.omnivoice_server_modal

    # Smoke-test the deployed service (HTTP endpoint)
    python -m src.nemotron_speech.modal.omnivoice_server_modal
"""

import asyncio
import base64
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from loguru import logger

import modal
import numpy as np

# Modal app definition
app = modal.App("omnivoice-tts-server")

model_cache = modal.Volume.from_name("omnivoice-model-cache", create_if_missing=True)
CACHE_PATH = "/omnivoice-model"

# Container image. OmniVoice pins torch 2.8.0 + CUDA 12.8 wheels, so we base on a
# matching CUDA 12.8 image.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_PATH,
            "TORCH_HOME": CACHE_PATH,
        }
    )
    .apt_install("git", "libsndfile1", "ffmpeg")
    .uv_pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]",
        "omnivoice",
        "soundfile",
        "fastapi[standard]",
        "pydantic",
        "loguru",
        "numpy<2.0.0",
    )
)

# Constants
OMNIVOICE_SAMPLE_RATE = 24000
MODEL_ID = "k2-fsa/OmniVoice"

# Emoji pattern for text normalization
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Misc symbols and pictographs
    "\U0001F680-\U0001F6FF"  # Transport and map symbols
    "\U0001F700-\U0001F77F"  # Alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric shapes extended
    "\U0001F800-\U0001F8FF"  # Supplemental arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # Chess symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and pictographs extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "]+",
    flags=re.UNICODE,
)


def normalize_text(text: str) -> str:
    """Normalize unicode characters in text.

    Note: OmniVoice uses bracketed tokens for non-verbal cues (e.g. "[laughter]")
    and pronunciation overrides (e.g. "[B EY1 S]"), so brackets are preserved.
    """
    text = text.replace("‘", "'")  # LEFT SINGLE QUOTATION MARK
    text = text.replace("’", "'")  # RIGHT SINGLE QUOTATION MARK
    text = text.replace("“", '"')  # LEFT DOUBLE QUOTATION MARK
    text = text.replace("”", '"')  # RIGHT DOUBLE QUOTATION MARK
    text = text.replace("—", "-")  # EM DASH
    text = text.replace("–", "-")  # EN DASH
    text = _EMOJI_PATTERN.sub("", text)
    return text


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float waveform in [-1, 1] to 16-bit PCM bytes."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return b""
    peak = float(np.abs(audio).max())
    if peak > 1.0:
        audio = audio / peak
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


with image.imports():
    import torch
    from omnivoice import OmniVoice

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response
    from pydantic import BaseModel


# Modal class for TTS inference
@app.cls(
    region="ap",
    routing_region="ap-south",
    image=image,
    volumes={
        CACHE_PATH: model_cache,
    },
    gpu="A100",  # Use L40S GPU for fast inference
    timeout=3600,  # 1 hour timeout for long-running requests
    min_containers=1,
)
class OmniVoiceServer:
    """Modal class for OmniVoice TTS inference."""

    @modal.enter()
    def load_model(self):
        """Load model on container startup."""
        logger.info(f"Loading OmniVoice model ({MODEL_ID})...")
        self.model = OmniVoice.from_pretrained(
            MODEL_ID,
            device_map="cuda:0",
            dtype=torch.float16,
        )
        # The audio tokenizer's rate (usually 24 kHz); fall back to the constant.
        self.sample_rate = getattr(self.model, "sampling_rate", None) or OMNIVOICE_SAMPLE_RATE
        logger.info(f"Model loaded successfully (sample_rate={self.sample_rate})")

        # Warm up to JIT-compile CUDA kernels and pre-allocate GPU memory.
        logger.info("Warming up OmniVoice (auto-voice path)...")
        _ = self.model.generate(
            text="This is a warm up sentence for the text to speech model.",
            num_step=16,
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info("Warm-up complete")

    def _synthesize(
        self,
        text: str,
        instruct: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: Optional[str] = None,
        num_step: int = 32,
        speed: float = 1.0,
        duration: Optional[float] = None,
    ) -> bytes:
        """Internal: synthesize speech and return 16-bit PCM bytes.

        Mode is selected by which arguments are provided:
          - ref_audio_path + ref_text -> voice cloning
          - instruct                  -> voice design
          - neither                   -> auto voice
        """
        text = normalize_text(text)

        # `language` and `speed` are named generate() params; `num_step` is a
        # generation-config field consumed via **kwargs. `duration` overrides
        # `speed` when set, so only pass one of them.
        kwargs: dict = {"text": text, "num_step": num_step, "language": language}
        if duration is not None:
            kwargs["duration"] = duration
        else:
            kwargs["speed"] = speed

        if ref_audio_path and ref_text:
            kwargs["ref_audio"] = ref_audio_path
            kwargs["ref_text"] = ref_text
        elif instruct:
            kwargs["instruct"] = instruct

        with torch.no_grad():
            audio = self.model.generate(**kwargs)

        # generate() returns a list of np.ndarray (one per input); we send one text.
        waveform = audio[0] if isinstance(audio, (list, tuple)) else audio
        return _float_to_pcm16(waveform)

    @modal.asgi_app()
    def api(self):
        """FastAPI app with OmniVoice TTS endpoints."""

        class SpeechRequest(BaseModel):
            input: str
            # Voice design: natural-language description, e.g. "female, low pitch, british accent"
            instruct: Optional[str] = None
            # Voice cloning: base64-encoded reference audio (wav) + its transcription
            ref_audio_b64: Optional[str] = None
            ref_text: Optional[str] = None
            # Optional language name/code (e.g. "English"/"en"); improves quality.
            language: Optional[str] = None
            # Generation controls
            num_step: int = 32
            speed: float = 1.0
            duration: Optional[float] = None
            response_format: str = "pcm"

        web_app = FastAPI(
            title="OmniVoice TTS Server (Modal)",
            description="Modal-deployed k2-fsa/OmniVoice zero-shot TTS inference server",
            version="1.0.0",
        )

        @web_app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "model_loaded": getattr(self, "model", None) is not None,
            }

        @web_app.get("/v1/audio/config")
        async def config():
            """Get TTS configuration."""
            return {
                "sample_rate": self.sample_rate,
                "channels": 1,
                "encoding": "pcm_s16le",
                "model": MODEL_ID,
                "modes": ["voice_cloning", "voice_design", "auto"],
            }

        @web_app.post("/v1/audio/speech")
        async def speech(request: SpeechRequest):
            """OpenAI-compatible speech synthesis endpoint."""
            text = normalize_text(request.input)
            if not text.strip():
                raise HTTPException(status_code=400, detail="Empty input text")

            # Voice cloning requires both a reference clip and its transcription.
            ref_audio_path: Optional[str] = None
            if request.ref_audio_b64 or request.ref_text:
                if not (request.ref_audio_b64 and request.ref_text):
                    raise HTTPException(
                        status_code=400,
                        detail="Voice cloning requires both ref_audio_b64 and ref_text",
                    )
                try:
                    ref_bytes = base64.b64decode(request.ref_audio_b64)
                except Exception:
                    raise HTTPException(status_code=400, detail="ref_audio_b64 is not valid base64")
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(ref_bytes)
                tmp.close()
                ref_audio_path = tmp.name

            mode = (
                "voice_cloning"
                if ref_audio_path
                else "voice_design"
                if request.instruct
                else "auto"
            )
            logger.info(f"TTS request: mode={mode}, text=[{text[:50]}...]")

            start = time.time()
            try:
                audio_bytes = await asyncio.to_thread(
                    self._synthesize,
                    text,
                    request.instruct,
                    ref_audio_path,
                    request.ref_text,
                    request.language,
                    request.num_step,
                    request.speed,
                    request.duration,
                )
            finally:
                if ref_audio_path:
                    Path(ref_audio_path).unlink(missing_ok=True)
            elapsed = time.time() - start

            duration_ms = len(audio_bytes) / (self.sample_rate * 2) * 1000
            rtf = elapsed / (duration_ms / 1000) if duration_ms > 0 else 0.0
            logger.info(
                f"TTS: {len(audio_bytes)} bytes, {duration_ms:.0f}ms audio, "
                f"latency={elapsed*1000:.0f}ms, RTF={rtf:.2f}x"
            )

            media_type = "audio/pcm" if request.response_format == "pcm" else "audio/wav"
            return Response(
                content=audio_bytes,
                media_type=media_type,
                headers={
                    "X-Sample-Rate": str(self.sample_rate),
                    "X-Channels": "1",
                    "X-Encoding": "pcm_s16le",
                    "X-Duration-Ms": str(int(duration_ms)),
                },
            )

        return web_app


# Local development entrypoint
if __name__ == "__main__":
    """Smoke-test the deployed OmniVoice service (HTTP endpoint)."""
    import wave

    import requests

    print("OmniVoice TTS Server - Modal Deployment Test")
    print("============================================")
    print()

    print("Getting deployed API URL...")
    OmniVoiceClass = modal.Cls.from_name("omnivoice-tts-server", "OmniVoiceServer")
    api_url = OmniVoiceClass().api.web_url
    print(f"API URL: {api_url}")
    print()

    print("Testing /v1/audio/speech endpoint (voice design)...")
    test_text = "Hello from Modal! This is a test of the OmniVoice TTS server."
    payload = {
        "input": test_text,
        "instruct": "female, low pitch, british accent",
        "response_format": "pcm",
    }
    print(f"Text: '{test_text}'")
    print(f"Instruct: {payload['instruct']}")
    print()

    response = requests.post(f"{api_url}/v1/audio/speech", json=payload, timeout=120)

    if response.status_code == 200:
        sample_rate = int(response.headers.get("X-Sample-Rate", str(OMNIVOICE_SAMPLE_RATE)))
        duration_ms = response.headers.get("X-Duration-Ms", "unknown")
        channels = int(response.headers.get("X-Channels", "1"))

        output_file = Path("omnivoice_test_output.wav")
        with wave.open(str(output_file), "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(response.content)

        print("✅ Success!")
        print(f"   Generated: {len(response.content)} bytes")
        print(f"   Duration: {duration_ms}ms")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"   Saved to: {output_file.absolute()}")
        print(f"   Play with: ffplay {output_file}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

    print()
    print("=" * 60)
    print("Deployment Info:")
    print(f"API Base URL: {api_url}")
    print()
    print("Available Endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /v1/audio/config  - Get TTS configuration")
    print("  POST /v1/audio/speech  - Synthesize speech (OpenAI-compatible)")
    print()
    print("Modes (POST /v1/audio/speech body):")
    print("  auto          -> {\"input\": \"...\"}")
    print("  voice design  -> {\"input\": \"...\", \"instruct\": \"female, british accent\"}")
    print("  voice cloning -> {\"input\": \"...\", \"ref_audio_b64\": \"<b64 wav>\", \"ref_text\": \"...\"}")
