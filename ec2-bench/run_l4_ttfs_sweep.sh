#!/usr/bin/env bash
# On-box: L4 SUB-KNEE TTFS sweep — size a TIGHT latency budget (not just keep-up).
# K=2 server processes (lanes=2) + CUDA MPS started ONCE; then sweep the per-process load N and
# report finalize-TTFS p50/p95 per process at each level. Finds the max streams that hold
# p50 < P50_MAX AND p95 < P95_MAX on the WORST of the K processes. Per-box streams = K * N_PER.
# Mirrors run_multiproc.sh's server env, so N_PER=16 reproduces the 32-knee TTFS point (p95 ~344ms).
set -uo pipefail
VENV=$HOME/nemo-venv; cd "$HOME/nemotron"; export HF_HOME=$HOME/hf
N_LIST="${N_LIST:-6,8,10,12,14,16}"; K=2
ROUNDS="${ROUNDS:-5}"   # REPEATS of each level (re-run the N-burst R times, pool N*R samples) -> stable p95 (1-shot N too noisy)
P50_MAX="${P50_MAX:-250}"; P95_MAX="${P95_MAX:-300}"
MODEL=nvidia/nemotron-speech-streaming-en-0.6b
SRV=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 "HF_HOME=$HOME/hf"
     NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8
     NEMOTRON_MODEL_LANES=2)

wait_port(){ local p=$1; for _ in $(seq 1 180); do (exec 3<>/dev/tcp/127.0.0.1/"$p") 2>/dev/null && { exec 3>&-; return 0; }; sleep 2; done; return 1; }

# CUDA MPS — concurrent kernels across the K processes (essential for multi-process GPU sharing).
if ! pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1; then
  nvidia-cuda-mps-control -d && echo "MPS started" || echo "MPS start FAILED (continuing without)"
else
  echo "MPS already running"
fi

# Start K=2 servers ONCE; they stay up across the whole N sweep (only the load varies).
pkill -f "server.py --model" 2>/dev/null; sleep 5
for k in $(seq 0 $((K-1))); do
  env -u LD_LIBRARY_PATH "${SRV[@]}" "$VENV/bin/python" server.py --model "$MODEL" \
      --host 127.0.0.1 --port $((8080+k)) --right-context 1 > "srv_l4ttfs_$k.log" 2>&1 &
done
for k in $(seq 0 $((K-1))); do wait_port $((8080+k)) || { echo "server $k FAILED"; tail -25 "srv_l4ttfs_$k.log"; exit 1; }; done
sleep 3

echo "=== L4 sub-knee TTFS sweep | K=2 procs (lanes=2) + MPS | staggered, rounds=${ROUNDS} | target p50<${P50_MAX} p95<${P95_MAX} ==="
declare -a SUMMARY
for N in ${N_LIST//,/ }; do
  LG=()
  for k in $(seq 0 $((K-1))); do
    "$VENV/bin/python" ec2_loadgen.py --url ws://127.0.0.1:$((8080+k)) --sweep "$N" --rounds "$ROUNDS" \
        --audio-dir "$HOME/nemotron/loadgen_audio" > "lg_l4ttfs_${N}_$k.log" 2>&1 &
    LG+=($!)
  done
  wait "${LG[@]}"   # wait ONLY the loadgens — servers stay up for the next N level
  echo "--- N_PER=$N (per-box=$((K*N))) ---"
  wp50=0; wp95=0; errs=0; miss=0
  for k in $(seq 0 $((K-1))); do
    line=$(grep -E "^ +$N " "lg_l4ttfs_${N}_$k.log" | tail -1)
    if [ -z "$line" ]; then echo "  proc$k: (NO RESULT — server may have died)"; miss=1; continue; fi
    echo "  proc$k: $line"
    read -r _n _ok e p50 p95 _rest <<<"$line"
    [ "$p50" != "nan" ] && [ "$p50" -gt "$wp50" ] 2>/dev/null && wp50=$p50
    [ "$p95" != "nan" ] && [ "$p95" -gt "$wp95" ] 2>/dev/null && wp95=$p95
    errs=$((errs + ${e:-0}))
  done
  verdict="PASS"
  { [ "$miss" = 1 ] || [ "$wp50" -ge "$P50_MAX" ] || [ "$wp95" -ge "$P95_MAX" ] || [ "$errs" -gt 0 ]; } 2>/dev/null && verdict="FAIL"
  SUMMARY+=("$(printf '  %-6s %-8s worst_p50=%-5s worst_p95=%-5s errs=%-3s %s' "$N" "$((K*N))" "$wp50" "$wp95" "$errs" "$verdict")")
done
pkill -f "server.py --model"; sleep 2
echo ""
echo "=== VERDICT (worst of K=2 procs; target p50<${P50_MAX} AND p95<${P95_MAX}, 0 errs) ==="
echo "  N_PER  per-box  TTFS(ms, max across the 2 procs)        verdict"
for s in "${SUMMARY[@]}"; do echo "$s"; done
echo "=== L4 TTFS SWEEP DONE ==="
