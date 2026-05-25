#!/usr/bin/env bash
# Drop into the CUDA 12.8 devel container (glibc 2.39 -> nvcc works) with the repo + HF model cache mounted, GPU visible.
# Usage: ./enter.sh [command...]   (no args = interactive bash)
set -euo pipefail
IMG="${NEMOTRON_CUDA_IMG:-nvidia/cuda:12.8.1-devel-ubuntu24.04}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"   # nemotron-january-2026 repo root
HF="${HF_HOME:-$HOME/.cache/huggingface}"
exec docker run --rm -it --gpus all \
  -v "$REPO":/work -w /work/proj-2026-05-24-from-scratch-runtime/runtime \
  -v "$HF":/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  "$IMG" "${@:-bash}"
