# Modal deployment for Nemotron Nano vLLM server.

import json
from typing import Any
import aiohttp

import modal

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.21.0",
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


app = modal.App("qwen-4B-awq-vllm")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000

with vllm_image.imports():
    import subprocess
    import torch

@app.function(
    region="ap", routing_region="ap-south",
    image=vllm_image,
    gpu=f"L40S:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    min_containers = 1,
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=60 * MINUTES)
def serve():

    torch.set_float32_matmul_precision('high')

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=debug",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        "10",
        "--max-model-len",
        "10000",
        "--enable-prefix-caching",
        "--trust-remote-code"
        
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

    serve_fn = modal.Function.from_name("qwen-4B-awq-vllm", "serve")
    url = serve_fn.get_web_url()

    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant. Always answer concisely and to the point in just two to three sentences.",
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
            # `stream=True` tells an OpenAI-compatible backend to stream chunks
            payload: dict[str, Any] = {
                "messages": messages,
                "model": model,
                "max_tokens": 512,
                "chat_template_kwargs": {"enable_thinking": False}
            }

            headers = {"Content-Type": "application/json"}

            async with session.post(
                "/v1/chat/completions", json=payload, headers=headers, timeout=5 * MINUTES
            ) as resp:
                async for raw in resp.content:
                    resp.raise_for_status()
                    # extract new content and stream it
                    line = raw.decode().strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):  # SSE prefix
                        line = line[len("data: ") :]

                    chunk = json.loads(line)
                    assert (
                        chunk["object"] == "chat.completion"
                    )  # or something went horribly wrong
                    print(chunk["choices"][0]['message']['content'], end="")
            print()

        async with aiohttp.ClientSession(base_url=url) as session:
            print(f"Running health check for server at {url}")
            async with session.get("/health", timeout=10*60 - 1 * MINUTES) as resp:
                up = resp.status == 200
            assert up, f"Failed health check for server at {url}"
            print(f"Successful health check for server at {url}")

            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)

    asyncio.run(_run_request(messages))




