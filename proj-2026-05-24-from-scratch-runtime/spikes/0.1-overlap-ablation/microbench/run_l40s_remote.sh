#!/usr/bin/env bash
# Runs ON the L40S box (Ubuntu 22.04 DL-base AMI). Sets up libtorch (via pip torch 2.8.0, matching the .ts export),
# builds the microbench (manual-link, no nvcc), and runs the lanes=1 vs lanes=N calibrated sweep.
# Prereq: this dir (microbench/) + artifacts/encoder_steady_b1.ts + shapes.json already rsync'd to ~/microbench.
set -euo pipefail
cd "$(dirname "$0")"
log(){ echo "[l40s $(date +%H:%M:%S)] $*"; }

log "gpu"; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv | sed -n '2p'
log "apt: cmake g++"; sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq && sudo apt-get install -y -qq cmake g++ >/dev/null

log "uv + torch 2.8.0 (matches the .ts export; standard wheel supports Ada sm_89)"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
VENV=$HOME/lt-venv
uv venv --python 3.11 "$VENV"
uv pip install --python "$VENV" "torch==2.8.0"
TORCH_ROOT=$("$VENV/bin/python" -c 'import torch,os;print(os.path.dirname(torch.__file__))')
log "torch_root=$TORCH_ROOT"
"$VENV/bin/python" -c "import torch;print('arch_list',torch.cuda.get_arch_list());print('cc',torch.cuda.get_device_capability())"

# CUDA root: the DL AMI ships CUDA under /usr/local/cuda
CUDA_ROOT=$(ls -d /usr/local/cuda-12* 2>/dev/null | sort | tail -1); CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
log "cuda_root=$CUDA_ROOT"

log "build (manual-link, no nvcc)"
rm -rf build
cmake -B build -DTORCH_ROOT="$TORCH_ROOT" -DCUDA_ROOT="$CUDA_ROOT" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" 2>&1 | tail -3

log "smoke"
./build/microbench --module artifacts/encoder_steady_b1.ts --lanes 1 --streams 2 --duration-s 5 --decode-host-us 10000 2>&1 | grep -E "===|chunk_latency|gpu_util"

log "=== SWEEP: lanes=1 (single-thread baseline) ==="
for n in 12 16 20 24; do
  ./build/microbench --module artifacts/encoder_steady_b1.ts --lanes 1 --streams $n --duration-s 10 --decode-host-us 10000 2>&1 | grep -E "===|chunk_latency|gpu_util"
done
log "=== SWEEP: lanes=N (multi-thread thesis) ==="
for n in 16 24 32 48 64; do
  ./build/microbench --module artifacts/encoder_steady_b1.ts --lanes "$(nproc)" --streams $n --duration-s 10 --decode-host-us 10000 2>&1 | grep -E "===|chunk_latency|gpu_util"
done
log "DONE (nproc=$(nproc))"
