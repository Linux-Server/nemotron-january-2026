# CUDA devel container — the glibc-compatible build env (unblocks AOTInductor + Wave-2 kernels)

## Why
The local host is Ubuntu 25.04 (**glibc 2.41**), which breaks `nvcc` for every installed CUDA (12.8 + 13.x) via the
`math_functions.h` `noexcept` conflict — so we can't compile CUDA *kernels* on the host (manual-link of prebuilt
libtorch works for graph-replay/decode, but NOT AOTInductor or custom kernels). The fix: build in an **Ubuntu-24.04
CUDA devel container (glibc 2.39)**, where `nvcc` works.

## Smoke test — PASSED (2026-05-25)
`docker run --rm --gpus all nvidia/cuda:13.0.0-devel-ubuntu24.04 ...`:
- glibc **2.39** (vs host 2.41) → no `noexcept` conflict.
- **nvcc compiles the trivial `t.cu` that FAILS on the host** → `NVCC_COMPILE_OK`.
- GPU visible: **RTX 5090, compute_cap 12.0 (sm_120)** — Blackwell works in the container.

Host has: docker 29, nvidia-container-toolkit 1.19.1, nvidia runtime + CDI (`nvidia.com/gpu=all`), 1.5 TB free.

## Image
**`nvidia/cuda:12.8.1-devel-ubuntu24.04`** — matches torch's cu128 (and sm_120 for the 5090). (The 13.x devel images are
also present and work for the nvcc smoke test, but 12.8 matches the torch wheel for AOTInductor.)

## Enter the container (project mounted)
```bash
./enter.sh        # see enter.sh — runs --gpus all, mounts the repo at /work, HF cache for the model
```
Inside, install the pinned torch once (per container or into a mounted venv):
```bash
pip install torch==2.8.0   # cu128 wheel; matches the export producer (T2a)
```

## What it's for (next steps)
1. **1.2b-wire — byte-exact C++ encoder via AOTInductor**: AOTI-compile the T2a `torch.export` steady encoder
   (`runtime/export_t2a.py` → `enc_steady_t2a.pt2`) to a `.so` (`torch._inductor.aot_compile`), load via the AOTI C++
   runtime in the streaming runtime. (`.pt2` can't be `torch::jit::load`ed; AOTI is the C++ path.)
2. **Wave-2 custom CUDA kernels** (fused encoder/decode, the 6–10 ms fusion path) — any hand-written kernel build.

## Note
GPU-byte-exact validation still happens here on the 5090 (the container sees the same GPU). The container only changes
the *build* toolchain (nvcc/glibc), not the GPU or the model.
