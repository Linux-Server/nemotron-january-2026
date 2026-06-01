#!/usr/bin/env bash
# Build the native C++ runtime on GB10/aarch64 without changing the legacy
# CUDA 12.8 x86_64 container flow.
#
# Usage:
#   runtime/container/build-gb10-aarch64.sh
#   BUILD_DIR=cpp/build_gb10 TARGETS="ws_server density_main" runtime/container/build-gb10-aarch64.sh
#   runtime/container/build-gb10-aarch64.sh bash
set -euo pipefail

IMG="${NEMOTRON_GB10_CUDA_IMG:-nemotron-aoti:gb10-aarch64}"
docker image inspect "$IMG" >/dev/null 2>&1 || IMG="nvcr.io/nvidia/pytorch:25.12-py3"

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
BUILD_DIR="${BUILD_DIR:-cpp/build_gb10}"
TARGETS="${TARGETS:-ws_server density_main}"
HF_HUB_OFFLINE_VALUE="${HF_HUB_OFFLINE:-0}"

TI=""
[ -t 0 ] && [ -t 1 ] && TI="-it"

if [ "$#" -gt 0 ]; then
  exec docker run --rm $TI --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO":/work -w /work/proj-2026-05-24-from-scratch-runtime/runtime \
    -v "$HF":/root/.cache/huggingface \
    -e HF_HUB_OFFLINE="$HF_HUB_OFFLINE_VALUE" \
    "$IMG" "$@"
fi

exec docker run --rm $TI --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO":/work -w /work/proj-2026-05-24-from-scratch-runtime/runtime \
  -v "$HF":/root/.cache/huggingface \
  -e HF_HUB_OFFLINE="$HF_HUB_OFFLINE_VALUE" \
  -e BUILD_DIR="$BUILD_DIR" \
  -e TARGETS="$TARGETS" \
  "$IMG" bash -lc '
    set -euo pipefail
    TORCH_ROOT="$(python3 - <<'"'"'PY'"'"'
import pathlib
import torch
print(pathlib.Path(torch.__file__).resolve().parent)
PY
)"
    cmake -S cpp -B "$BUILD_DIR" \
      -DTORCH_ROOT="$TORCH_ROOT" \
      -DCUDA_ROOT=/usr/local/cuda \
      -DCUDA_TARGET_TRIPLE=sbsa-linux
    cmake --build "$BUILD_DIR" --target $TARGETS -j "$(nproc)"
  '
