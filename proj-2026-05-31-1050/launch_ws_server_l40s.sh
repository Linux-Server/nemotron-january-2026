#!/usr/bin/env bash
# Step-0 clean-knee: launch the CURRENT scheduler ws_server on the L40S box.
# scheduler ON, background warmup, TS enc_first (default, no AOTI-enc_first flag), CAP=128/LANES=128.
set -uo pipefail

RT="$HOME/density"
BIN="$RT/cpp/build_l40s_density/ws_server"
[ -x "$BIN" ] || { echo "ws_server not found at $BIN"; exit 1; }

TORCH_ROOT="$HOME/torch280-sm89-venv/lib/python3.10/site-packages/torch"
TORCH_LIB="$TORCH_ROOT/lib"
CUDART12_DIR="$(dirname "$(ldd "$BIN" 2>/dev/null | grep -oE '/[^ ]*libcudart\.so\.12' | head -1)")"
[ -n "$CUDART12_DIR" ] || CUDART12_DIR="/usr/local/cuda-12.9/lib64"
CUDA13_LIB="/usr/local/cuda-13.0/lib64"

PORT="${PORT:-8080}"
CAP="${CAP:-128}"
LANES="${LANES:-128}"
LOG="${LOG:-$RT/step0_ws_server.log}"

# Clean stale /tmp AOTI/inductor dirs (AOTI packages unpack here on load).
rm -rf /tmp/torchinductor_ubuntu /tmp/aoti* /tmp/.aoti* 2>/dev/null || true

echo "[launch] bin=$BIN port=$PORT cap=$CAP lanes=$LANES"
echo "[launch] TORCH_LIB=$TORCH_LIB CUDART12_DIR=$CUDART12_DIR"
echo "[launch] log=$LOG  (ready when BOTH 'ws_server listening' AND 'background_warm_complete' printed)"

ENV=(
  HF_HUB_OFFLINE=1
  "LD_LIBRARY_PATH=$TORCH_LIB:$CUDART12_DIR:$CUDA13_LIB"
  NEMOTRON_CONTINUOUS=1
  NEMOTRON_FINALIZE_SILENCE_MS=0
  "NEMOTRON_ARTIFACT_DIR=$RT/artifacts_sm89"
  NEMOTRON_WS_SCHEDULER=1
  NEMOTRON_WS_BACKGROUND_WARMUP=1
  NEMOTRON_DENSITY_BATCH_STEADY=1
  NEMOTRON_DENSITY_BATCH_MAX=4
  NEMOTRON_DENSITY_BATCH_WINDOW_MS=10
  NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS=0
  "NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP=$CAP"
  "NEMOTRON_WS_LANES=$LANES"
)

cd "$RT"
nohup env "${ENV[@]}" "$BIN" \
  --port "$PORT" \
  --admission-active-cap "$CAP" \
  --steady-batch-dir "$RT/steady_b_artifacts" \
  > "$LOG" 2>&1 &
echo "[launch] pid=$! (server detached, writing to $LOG)"
