"""Modal deployment of the CANONICAL nemotron_speech.server for the per-GPU
concurrency-knee cost study.

Apples-to-apples with the LOCAL benchmarks: this deploys the *real* server.py
(silence0_warm200 = NEMOTRON_CONTINUOUS=1 + FINALIZE_SILENCE_MS=0 + WARMUP_MS=200,
fork-flush finalize, continuous context) — NOT the older embedded reimplementation
in asr_server_modal.py — and is driven by the same harness
(proj-2026-05-19-eou-endpointing/run_full1000_conc12.py), which already speaks
server.py's exact WS protocol (vad_start/vad_stop/reset -> transcript{is_final,finalize}).

NeMo is pinned to the SAME commit as the local omni .venv-asr runtime
(NVIDIA-NeMo/NeMo @ 056d937..., 2.8.0rc0) so the streaming behavior matches.

One container, one GPU, concurrent WS inputs allowed -> measures the single-GPU
realtime keep-up knee (the single inference_lock + batch_size=1 serialization is
identical to local). GPU type is chosen per-deploy via ASR_GPU.

Deploy (per GPU):
    ASR_GPU=L4 ASR_BENCH_APP=nemotron-asr-bench \
      .venv/bin/modal deploy -m src.nemotron_speech.modal.asr_bench_modal
Then drive run_full1000_conc12.py at the printed wss URL, sweep concurrency,
find the knee, and `modal app stop nemotron-asr-bench` before the next GPU.
"""

import os

import modal

# --- per-deploy knobs (read at deploy time) ---
APP_NAME = os.environ.get("ASR_BENCH_APP", "nemotron-asr-bench")
GPU = os.environ.get("ASR_GPU", "L40S")
RIGHT_CONTEXT = os.environ.get("ASR_RC", "1")          # English low-latency default
MAX_INPUTS = int(os.environ.get("ASR_CONCURRENT", "64"))  # concurrent WS streams / container
REGION = os.environ.get("ASR_REGION") or None          # pin region (e.g. us-east-1 for co-loc tests)
PROFILE = os.environ.get("ASR_PROFILE") or ""          # baked at deploy time -> NEMOTRON_PROFILE_CHUNK
ENABLE_BATCHING = os.environ.get("ASR_ENABLE_BATCHING", "1") == "1"
ENCODER_COMPILE = os.environ.get("ASR_ENCODER_COMPILE", "") == "1"

app = modal.App(APP_NAME)

CACHE_PATH = "/cache"
model_cache = modal.Volume.from_name("nemotron-speech", create_if_missing=True)
MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
PORT = 8080

# Pin NeMo to MATCH the local omni .venv-asr (apples-to-apples streaming behavior).
NEMO_COMMIT = "056d937544064df164b1751e9c8a1c3b597389fd"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .env({"DEBIAN_FRONTEND": "noninteractive"})
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
        "websockets",
    )
    .uv_pip_install(
        f"nemo_toolkit[asr]@git+https://github.com/NVIDIA-NeMo/NeMo.git@{NEMO_COMMIT}",
        extra_options="--no-cache",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_PATH,
            "TORCH_HOME": CACHE_PATH,
            "ASR_ENABLE_BATCHING": "1" if ENABLE_BATCHING else "0",
            "ASR_ENCODER_COMPILE": "1" if ENCODER_COMPILE else "0",
        }
    )
)

# Bake the profiling flag (deploy-time read of PROFILE) into the container env. This
# survives the container's module re-import (where ASR_PROFILE is unset); server.py
# inherits it via os.environ.copy(). MUST come before add_local_file — Modal forbids
# build steps after a non-copy add_local_file mount.
if PROFILE == "1":
    image = image.env({"NEMOTRON_PROFILE_CHUNK": "1"})

# the canonical server plus its batching helper — add LAST (mount-style final image step)
image = image.add_local_file(
    "src/nemotron_speech/batch_primitives.py", "/app/batch_primitives.py"
).add_local_file("src/nemotron_speech/server.py", "/app/server.py")


@app.function(
    image=image,
    gpu=GPU,
    volumes={CACHE_PATH: model_cache},
    secrets=[modal.Secret.from_name("hf-kwindla")],  # gated EN checkpoint download (Kwindla token)
    max_containers=1,        # pin ONE container/GPU -> single-GPU knee
    region=REGION,
    timeout=3600,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=MAX_INPUTS)  # allow N concurrent WS streams into the one container
@modal.web_server(port=PORT, startup_timeout=900)  # cover model download + load + warmup
def asr():
    """Launch the canonical server.py (silence0_warm200) listening on PORT.

    Modal proxies wss traffic to localhost:PORT once it is live.
    """
    import subprocess

    env = os.environ.copy()
    env.update(
        {
            "NEMOTRON_CONTINUOUS": "1",
            "NEMOTRON_FINALIZE_SILENCE_MS": "0",
            "NEMOTRON_WARMUP_MS": "200",
        }
    )
    if ENABLE_BATCHING:
        env.update(
            {
                "NEMOTRON_SCHEDULER_B1": "1",
                "NEMOTRON_BATCH_SCHED": "1",
                "NEMOTRON_BATCH_MAX_SIZE": "32",
                "NEMOTRON_BATCH_MAX_WAIT_MS": "8",
            }
        )
    if ENCODER_COMPILE:
        env["NEMOTRON_ENCODER_COMPILE"] = "1"
    # NEMOTRON_PROFILE_CHUNK is baked into the image env at deploy (see below) so it
    # survives the container's module re-import; server.py inherits it via os.environ.copy().
    # non-blocking: start the server, return so Modal can poll the port
    subprocess.Popen(
        [
            "python",
            "/app/server.py",
            "--model",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--right-context",
            RIGHT_CONTEXT,
        ],
        env=env,
    )
