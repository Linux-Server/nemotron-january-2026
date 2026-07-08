# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sample code for building low-latency voice agents (~500-700ms voice-to-voice) with three NVIDIA open models: **Nemotron Speech ASR**, **Nemotron-3-Nano LLM**, and **Magpie TTS**. It runs two ways:

- **Local**: everything in one CUDA 13.1 / Blackwell container (`Dockerfile.unified`), managed by `./scripts/nemotron.sh`, driven by a Pipecat bot.
- **Cloud**: ASR/TTS/LLM deployed as separate Modal services; the bot runs locally or on Pipecat Cloud.

The core value is the **pipelining** across STT → LLM → TTS so each stage starts before the previous finishes. Read `docs/streaming-pipeline-architecture.md` before touching latency-sensitive code — the frame ordering, buffering, and context-size choices below are load-bearing, not incidental.

## Commands

Dependencies use `uv`. Optional extras gate different workflows:

```bash
uv sync --extra dev                    # tests + lint + typecheck
uv sync --extra bot --extra modal      # Pipecat Cloud CLI + Modal deploy
uv sync --extra nemo                   # NeMo ASR/TTS (heavy CUDA deps, install separately)

uv run pytest                          # run tests (asyncio_mode=auto, testpaths=tests)
uv run pytest tests/test_streaming_tts.py::test_name   # single test
uv run ruff check .                     # lint (line-length 100, py310)
uv run mypy src                         # typecheck (disallow_untyped_defs=true)
```

Note: several files under `tests/` are `.sh` scripts or standalone benchmark/harness scripts (`benchmark_inference.py`, `measure_streaming_ttfb.py`, `run_20_turn_test.py`), not pytest tests. Only pytest-style `test_*.py` are collected.

### Running locally

```bash
docker build -f Dockerfile.unified -t nemotron-unified:cuda13 .   # 2-3 hrs, builds PyTorch/NeMo/vLLM/llama.cpp from source for sm_121
./scripts/nemotron.sh start [--mode llamacpp-q8|llamacpp-q4|vllm] [--model PATH] [--no-asr|--no-tts|--no-llm]
./scripts/nemotron.sh status | logs [asr|tts|llm] | shell | stop | restart
uv run pipecat_bots/bot_interleaved_streaming.py    # then open http://localhost:7860/client
```

Service ports: ASR `8080` (WebSocket), TTS `8001` (HTTP+WS), LLM `8000` (HTTP). Each has `/health`.

### Deploying to Modal

```bash
modal deploy -m src.nemotron_speech.modal.asr_server_modal
modal deploy -m src.nemotron_speech.modal.tts_server_modal
modal deploy -m src.nemotron_speech.modal.vllm_modal
uv run -m pipecat_bots.modal.bot_modal
```

## Architecture

Two halves that talk over WebSocket/HTTP: **inference servers** (`src/nemotron_speech/`) and **Pipecat bot clients** (`pipecat_bots/`). The same client services point at either local container ports or Modal URLs (via `NVIDIA_ASR_URL` / `NVIDIA_LLM_URL` / `NVIDIA_TTS_URL` env vars).

### Inference servers — `src/nemotron_speech/`
- `server.py` — WebSocket ASR server. Runs Nemotron/Parakeet with **true incremental streaming**: 160ms chunks, encoder/decoder cache carried across chunks per `ASRSession`. `att_context_size = [70, right_context]`; right context is the latency/accuracy knob (`--right-context 0|1|6|13`, default 1 ≈ 160ms).
- `tts_server.py` + `streaming_tts.py` + `adaptive_stream.py` — FastAPI/WebSocket Magpie TTS with **adaptive mode**: first segment streams (fast TTFB), later segments batch (higher quality).
- `modal/*_modal.py` — Modal wrappers of the above (deploy targets). `vllm_modal.py` serves the BF16 LLM via vLLM (OpenAI-compatible).

### Bot clients — `pipecat_bots/`
- `bot_interleaved_streaming.py` — main local bot; assembles the full pipeline (llama.cpp LLM path).
- `bot_vllm.py` — variant pointing at a vLLM OpenAI endpoint. `bot_simple_vad.py` — simpler VAD path. `modal/bot_modal.py` — cloud services variant.
- `nvidia_stt.py` — streaming ASR WebSocket client with soft/hard reset.
- `llama_cpp_buffered_llm.py` + `sentence_buffer.py` — **single-slot** LLM client (`--parallel 1`) that emits at sentence boundaries for **100% KV cache reuse** across turns. First segment capped ~24 tokens for fast TTFC, later ~96.
- `magpie_websocket_tts.py` — adaptive streaming TTS client.
- `v2v_metrics.py` — measures voice-to-voice time (VADUserStoppedSpeaking → BotStartedSpeaking).
- `frames.py` — shared frame types (e.g. `ChunkedLLMContinueGenerationFrame`) kept separate to avoid circular imports.

### Latency invariants (do not break casually)
- **Frame ordering**: `TranscriptionFrame` must reach the aggregator *before* `UserStoppedSpeakingFrame`, or a ~500ms aggregation timeout is incurred.
- **VAD `stop_secs = 0.2`** is aligned to ASR trailing-context needs; the encoder requires `(right_context+1)*shift_frames*hop_samples` samples of trailing silence to finalize the last word.
- **LLM single-slot** operation is what gives 100% KV cache reuse; multi-slot breaks it.

## Patches

`patches/` holds source patches applied during the container build (llama.cpp cache/slot fixes, vLLM PR31607 sm_121 support). `patches/apply-vllm-pr31607.py` applies them. `vllm_plugins/nano_v3_reasoning_parser.py` is a vLLM plugin for the Nano-3 reasoning format. If a build fails on vLLM/llama.cpp, check here first.

## Models

Auto-downloaded on first run: `nvidia/nemotron-speech-streaming-en-0.6b` (ASR), `nvidia/magpie_tts_multilingual_357m` (TTS). LLM must be downloaded manually: `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` (Q8/Q4 for llama.cpp) or `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (vLLM). Q8≈32GB, Q4≈16GB (RTX 5090), BF16≈72GB (multi-GPU/cloud).
