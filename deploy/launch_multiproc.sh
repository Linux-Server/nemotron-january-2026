#!/usr/bin/env bash
# Production multi-process launcher  —  DESIGN ARTIFACT for Step 2 of proj-2026-05-21-inference-opt/PLAN.md.
#
# Starts CUDA MPS + K Nemotron ASR server processes (each NEMOTRON_MODEL_LANES=2) on one box, with
# crash-restart supervision. K + config come from the per-GPU matrix (see deploy/DEPLOYMENT.md), which the
# benchmarks established: L4 -> K~2 (~32/box), L40S -> K~4 (~64/box, GPU-bound), lanes=2/process is the unit.
#
# This is the REFERENCE launcher to adapt to your substrate (systemd template / container entrypoint / ECS
# task). Substrate-dependent + production-hardening items are marked TODO and discussed in DEPLOYMENT.md.
set -uo pipefail

APP_DIR="${NEMOTRON_APP_DIR:-$HOME/nemotron}"          # holds server.py + batch_primitives.py
VENV="${NEMOTRON_VENV:-$HOME/nemo-venv}"
MODEL="${NEMOTRON_MODEL:-nvidia/nemotron-speech-streaming-en-0.6b}"
K="${NEMOTRON_PROCS:-3}"                                # processes/box (per-GPU matrix: L4=2, L40S=4)
LANES="${NEMOTRON_MODEL_LANES:-2}"                      # within-process sweet spot (>2 regresses; GIL)
BASE_PORT="${NEMOTRON_BASE_PORT:-8080}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"

# silence0_warm200 + batching + the committed TTFS levers (barrier-drain, finalize-storm).
SRV_ENV=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200
         NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8
         "NEMOTRON_MODEL_LANES=$LANES" NEMOTRON_BATCH_BARRIER_DRAIN=1 NEMOTRON_BATCH_FINALIZE=1)

cd "$APP_DIR"

start_mps(){ mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"; nvidia-cuda-mps-control -d && echo "[mps] daemon up"; }
stop_mps(){ echo quit | nvidia-cuda-mps-control 2>/dev/null || true; }
trap 'echo "[launcher] SIGTERM — stopping"; pkill -f "server.py --model" 2>/dev/null; stop_mps; exit 0' INT TERM

declare -A PIDS
launch(){ local k=$1; local port=$((BASE_PORT+k))
  env -u LD_LIBRARY_PATH "${SRV_ENV[@]}" "$VENV/bin/python" server.py --model "$MODEL" \
      --host 0.0.0.0 --port "$port" --right-context 1 > "server_$k.log" 2>&1 &
  PIDS[$k]=$!; echo "[launch] proc $k -> pid ${PIDS[$k]} port $port"; }

start_mps                                              # NOTE: unset LD_LIBRARY_PATH for torch's bundled cuDNN
for k in $(seq 0 $((K-1))); do launch "$k"; done

# Supervisor: restart a crashed process. CRITICAL with MPS — a CUDA fault in one client can corrupt the shared
# context and take down the others (blast-radius; see DEPLOYMENT.md "MPS hardening"). For correctness, the LB
# should DRAIN the dead backend before kill and re-add it only after /health passes.
while true; do
  sleep 10
  for k in $(seq 0 $((K-1))); do
    if ! kill -0 "${PIDS[$k]}" 2>/dev/null; then
      echo "[supervisor] proc $k (pid ${PIDS[$k]}) DIED — restarting"   # TODO: drain LB backend + alert/metric
      # TODO: if MPS context is corrupt (multiple procs died together), restart MPS + all procs.
      launch "$k"
    fi
  done
done
