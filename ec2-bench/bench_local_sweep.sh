#!/usr/bin/env bash
# LOCAL full-harness keep-up sweep — the on-box twin of bench_prod_sweep.sh (no EC2/SSH).
# Brings up CUDA MPS + K server procs (lanes=2, production levers default-ON, profiling OFF) behind a local LB
# (haproxy if installed, else the bundled leastconn ec2-bench/local_lb.py), then runs the SAME full-1000 client at
# CONC_LIST against the LB. Reports per-conc client TTFB p50/p95/p99 + per-proc /health + gpu_mem.
#
# Apples-to-apples with the cloud sweep EXCEPT no WAN: client is on localhost, so TTFB is ~the WAN RTT (~23 ms for
# SF<->us-west-2) LOWER than the cloud numbers. The keep-up KNEE is server-side, so it is directly comparable; just
# add ~WAN back to the absolute p50/p95 when comparing to the cloud table.
#
# Env: K (default 2), LANES (2), CONC_LIST ("8 12 16 20 24"), LIMIT (400), MAXCONN (0=unlimited, set to mirror cloud
# only for overload tests), BASE_PORT (8081), FRONT (8080), SRV (server venv python), HV (client venv python),
# MODEL, OUT_DIR. Levers default-on (override e.g. SYNC_COMPRESS=0 for A/B; FINALIZE_PROFILE=1 to capture records).
# ALWAYS cleans up (trap). Run as a FILE from the repo root.
set -uo pipefail
cd "$(dirname "$0")/.."                                  # repo root
REPO=$(pwd)

K="${K:-2}"; LANES="${LANES:-2}"
CONC_LIST="${CONC_LIST:-8 12 16 20 24}"; LIMIT="${LIMIT:-400}"
MAXCONN="${MAXCONN:-0}"; BASE_PORT="${BASE_PORT:-8081}"; FRONT="${FRONT:-8080}"
SRV="${SRV:-/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python}"   # server venv (torch+nemo)
HV="${HV:-$REPO/stt-benchmark/.venv/bin/python}"                           # client venv (websockets)
MODEL="${MODEL:-nvidia/nemotron-speech-streaming-en-0.6b}"
OUT="${OUT_DIR:-$REPO/ec2-bench/local_sweep_$(date +%H%M)}"; mkdir -p "$OUT"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "[local] GPU=$GPU  K=$K lanes=$LANES  conc=[$CONC_LIST] limit=$LIMIT maxconn=$MAXCONN  out=$OUT"
[ -x "$SRV" ] || { echo "server venv not found: $SRV (set SRV=)"; exit 1; }
[ -x "$HV" ]  || { echo "client venv not found: $HV (set HV=)"; exit 1; }

# Production server env — mirrors deploy/launch_multiproc.sh SRV_ENV (levers default-ON, env-overridable), profiling OFF.
SRV_ENV=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200
  NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8
  "NEMOTRON_MODEL_LANES=$LANES" NEMOTRON_BATCH_BARRIER_DRAIN=1 NEMOTRON_BATCH_FINALIZE=1
  NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1
  "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=${FINALIZE_PADDED:-1}"
  "NEMOTRON_SYNC_COMPRESS=${SYNC_COMPRESS:-1}" "NEMOTRON_FINALIZE_PRIORITY=${FINALIZE_PRIORITY:-1}"
  PYTHONPATH=src)
[ "${FINALIZE_PROFILE:-0}" = 1 ] && SRV_ENV+=(NEMOTRON_FINALIZE_PROFILE=1)
[ -n "${ADMISSION_MAX_BACKLOG:-}" ] && SRV_ENV+=("NEMOTRON_ADMISSION_MAX_BACKLOG=$ADMISSION_MAX_BACKLOG")

declare -a SRV_PIDS; LB_PID=""; MPS_UP=0
cleanup() {
  echo "[local] cleanup"
  [ -n "$LB_PID" ] && kill "$LB_PID" 2>/dev/null
  for p in "${SRV_PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
  pkill -f "src/nemotron_speech/server.py --model" 2>/dev/null
  [ "$MPS_UP" = 1 ] && echo quit | nvidia-cuda-mps-control 2>/dev/null
}
trap cleanup EXIT INT TERM

if [ "$K" -gt 1 ]; then
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  nvidia-cuda-mps-control -d && MPS_UP=1 && echo "[local] MPS daemon up"
fi

for k in $(seq 0 $((K-1))); do
  port=$((BASE_PORT+k))
  env -u LD_LIBRARY_PATH "${SRV_ENV[@]}" "$SRV" src/nemotron_speech/server.py \
    --model "$MODEL" --host 127.0.0.1 --port "$port" --right-context 1 > "$OUT/server_$k.log" 2>&1 &
  SRV_PIDS+=($!); echo "[local] proc $k -> pid ${SRV_PIDS[$k]} port $port"
done

echo "[local] waiting for $K procs to load + capture graphs ..."
sok=0; for _ in $(seq 1 120); do
  sleep 5; n=$(grep -l "ASR server listening" "$OUT"/server_*.log 2>/dev/null | wc -l)
  [ "${n:-0}" -ge "$K" ] && { sok=1; echo "[local] all $K procs ready"; break; }
  for p in "${SRV_PIDS[@]}"; do kill -0 "$p" 2>/dev/null || { echo "[local] a proc DIED:"; tail -20 "$OUT"/server_*.log; exit 1; }; done
done
[ $sok = 1 ] || { echo "[local] procs FAILED to start ($n/$K)"; tail -25 "$OUT/server_0.log"; exit 1; }

BACKENDS=$(seq -s, "$BASE_PORT" $((BASE_PORT+K-1)))
if command -v haproxy >/dev/null; then
  { echo "global"; echo "  maxconn 100000"; echo "defaults"; echo "  mode tcp"; echo "  timeout connect 5s";
    echo "  timeout client 1h"; echo "  timeout server 1h";
    echo "frontend f"; echo "  bind 127.0.0.1:$FRONT"; echo "  default_backend p";
    echo "backend p"; echo "  balance leastconn";
    for k in $(seq 0 $((K-1))); do echo "  server p$k 127.0.0.1:$((BASE_PORT+k)) check maxconn ${MAXCONN:-100000}"; done
  } > "$OUT/haproxy.cfg"
  haproxy -f "$OUT/haproxy.cfg" & LB_PID=$!; echo "[local] haproxy front=$FRONT -> $BACKENDS"
else
  "$HV" ec2-bench/local_lb.py --front "$FRONT" --backends "$BACKENDS" --maxconn "$MAXCONN" > "$OUT/lb.log" 2>&1 &
  LB_PID=$!; sleep 2; echo "[local] local_lb (no haproxy) front=$FRONT -> $BACKENDS"
fi
sleep 2

for C in $CONC_LIST; do
  tag="local_$(echo "$GPU"|tr -d ' ')_K${K}_c${C}"
  echo ""; echo "########## LOCAL sweep conc=$C (limit $LIMIT; ~$(awk "BEGIN{printf \"%.1f\",$C/$K}")/proc) ##########"
  "$HV" proj-2026-05-19-eou-endpointing/run_full1000_conc12.py --url "ws://127.0.0.1:$FRONT" \
    --model-tag "$tag" --concurrency "$C" --limit "$LIMIT" 2>&1 | tee "$OUT/${tag}.clientlog" \
    | grep -E "Completed in|TTFB \(speech|server finalize"
  grep -qE "ok=0 errors=" "$OUT/${tag}.clientlog" && { echo "  !!! conc=$C ok=0 -> abort"; break; }
  sleep 5
done

echo ""; echo "=== gpu_mem + per-proc /health ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee "$OUT/gpu_mem.txt"
for k in $(seq 0 $((K-1))); do
  echo "proc $k: $("$HV" -c "import urllib.request,sys;print(urllib.request.urlopen('http://127.0.0.1:$((BASE_PORT+k))/health',timeout=5).read().decode())" 2>/dev/null)" | tee -a "$OUT/health.txt"
done
echo "=== local sweep done -> $OUT ==="
