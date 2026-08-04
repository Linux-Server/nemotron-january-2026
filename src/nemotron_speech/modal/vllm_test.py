# Modal deployment for a vLLM server tuned for the voice pipeline.
#
# Voice turns are short (a few hundred tokens of context, ~1-2 sentences out), so
# everything here is biased toward time-to-first-token rather than throughput:
# small max-model-len, prefix caching on, CUDA graphs on, thinking off.

import json
import time
from typing import Any

import aiohttp
import modal

MODEL_NAME = "Qwen/Qwen3.5-9B"
# Alias the served model so bot clients can point at a stable name
# (set NVIDIA_LLM_MODEL=llm) regardless of which checkpoint is deployed.
SERVED_MODEL_NAME = "llm"
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.26.0",
        "flashinfer-python"
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = False

APP_NAME = "qwen3p5-9b-vllm-voice"
app = modal.App(APP_NAME)

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000

# Voice turns are short: system prompt + a handful of dialogue turns.
# Keeping this small leaves the whole GPU free for KV cache / CUDA graphs.
MAX_MODEL_LEN = 32000
# Concurrent voice sessions per replica. Kept in sync with @modal.concurrent
# so vLLM never queues behind Modal's input concurrency.
MAX_NUM_SEQS = 8

with vllm_image.imports():
    import subprocess
    import torch

@app.function(
    # Both must be ap-south. `region="ap"` is the whole AP continent, so the
    # worker lands in a different AP region from the ap-south ingress and every
    # request pays an extra inter-region hop. Measured from Kerala with a
    # CPU-only echo app: GET 188->59ms, POST 340->68ms, WS round-trip 176->45ms
    # (45ms is the raw client RTT, i.e. zero proxy overhead). That hop was ~75%
    # of the voice pipeline's LLM TTFT.
    region=["ap-south"],
    routing_region="ap-south",
    image=vllm_image,
    gpu=["A100-80GB", "H100!", "B200"],
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    min_containers = 1,
    secrets=[
        modal.Secret.from_name("hf-token"),
    ],
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=MAX_NUM_SEQS
)
@modal.web_server(port=VLLM_PORT, startup_timeout=60 * MINUTES)
def serve():

    torch.set_float32_matmul_precision('high')

    cmd = [
        "vllm",
        "serve",
        # The model must be the first positional argument; vLLM's CLI rejects
        # anything (even a flag) placed between `serve` and the model name.
        MODEL_NAME,
        "--uvicorn-log-level=warning",
        "--served-model-name",
        SERVED_MODEL_NAME,
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        # Turn N of a conversation re-sends turns 1..N-1; prefix caching makes
        # that prefill nearly free, which is most of the TTFT win per turn.
        "--enable-prefix-caching",
        "--gpu-memory-utilization",
        "0.92",
        # Chunked prefill: cap prefill chunks so a long prefill can't stall the
        # decode of an in-flight turn (audible as a gap mid-sentence).
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        "8192",
        "--trust-remote-code",
        # Qwen3.5 is a VLM; the voice pipeline is text-only, so refuse
        # multimodal inputs and reclaim the vision profiling memory.
        "--limit-mm-per-prompt",
        '\'{"image": 0, "video": 0}\'',
        # Thinking mode is ON by default for Qwen3.5. Clients disable it per
        # request (chat_template_kwargs.enable_thinking=false); the parser is a
        # backstop so any reasoning that slips through lands in reasoning_content
        # instead of being spoken aloud by TTS.
        "--reasoning-parser",
        "qwen3",
        # Voice defaults. Qwen3.5's shipped generation_config uses thinking-mode
        # sampling (temp 1.0, presence_penalty 1.5), which rambles when spoken.
        "--override-generation-config",
        (
            '\'{"temperature": 0.6, "top_p": 0.95, "top_k": 20, '
            '"presence_penalty": 0.0, "max_tokens": 256}\''
        ),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        # Quoted like the other JSON args above: the cmd is joined and run through
        # a shell, which would otherwise eat the double quotes and hand vLLM
        # `{method:...}` instead of valid JSON.
        # "--speculative-config",
        # '\'{"method":"qwen3_next_mtp","num_speculative_tokens":2}\'',
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


if __name__ == "__main__":

    import asyncio

    serve_fn = modal.Function.from_name(APP_NAME, "serve")
    url = serve_fn.get_web_url()

    system_prompt = {
        "role": "system",
        "content": (
            "You are a voice assistant. Your replies are spoken aloud, so answer in "
            "one or two short sentences of plain conversational English. Never use "
            "markdown, lists, emoji, or code blocks."
        ),
    }
    content = "Where are the least rainy places in the United States?"

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async def _run_request(messages: list) -> None:

        async def _send_request(
            session: aiohttp.ClientSession, model: str, messages: list
        ) -> None:
            # `stream=True` tells an OpenAI-compatible backend to stream chunks.
            # The bot streams too, so TTFT here is what the TTS stage waits on.
            payload: dict[str, Any] = {
                "messages": messages,
                "model": model,
                "stream": True,
                "max_tokens": 256,
                "temperature": 0.6,
                "top_p": 0.95,
                "presence_penalty": 0.0,
                # Voice replies must not wait on a reasoning block.
                "chat_template_kwargs": {"enable_thinking": False},
            }

            headers = {"Content-Type": "application/json"}

            start = time.perf_counter()
            ttft: float | None = None

            async with session.post(
                "/v1/chat/completions", json=payload, headers=headers, timeout=5 * MINUTES
            ) as resp:
                resp.raise_for_status()
                async for raw in resp.content:
                    # extract new content and stream it
                    line = raw.decode().strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):  # SSE prefix
                        line = line[len("data: ") :]

                    chunk = json.loads(line)
                    assert (
                        chunk["object"] == "chat.completion.chunk"
                    )  # or something went horribly wrong
                    delta = chunk["choices"][0]["delta"]
                    # Guard against reasoning leaking into the spoken text.
                    if delta.get("reasoning_content"):
                        continue
                    text = delta.get("content")
                    if not text:
                        continue
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    print(text, end="", flush=True)
            print()
            if ttft is not None:
                print(f"\nTTFT: {ttft * 1000:.0f} ms")

        async with aiohttp.ClientSession(base_url=url) as session:
            print(f"Running health check for server at {url}")
            async with session.get("/health", timeout=10*60 - 1 * MINUTES) as resp:
                up = resp.status == 200
            assert up, f"Failed health check for server at {url}"
            print(f"Successful health check for server at {url}")

            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            # Two passes: the second reuses the cached system-prompt prefix,
            # which is the TTFT a warm voice session actually sees.
            await _send_request(session, SERVED_MODEL_NAME, messages)
            await _send_request(session, SERVED_MODEL_NAME, messages)

    asyncio.run(_run_request(messages))




