#!/usr/bin/env bash
# Cheap confirm (Track A) of the eager-decode-syncs stall hypothesis. Reproduce the catastrophic stall
# (conc-24 + 400ms stream-jitter, the WAN-jitter-24 config that froze) and A/B the AMPLIFIERS:
#   lanes2_warm200 (baseline, expect stall) | lanes1_warm200 (cross-lane contention?) | lanes2_warm0 (warmup trigger?)
# On a stall: SIGUSR1 (Python stack) + gdb native bt of the wedged lane thread (cudaStreamSynchronize=backlog vs
# cudaLaunchKernel=launch-queue -- the decisive root-mechanism bit faulthandler can't show).
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
ROOT=$(pwd)
SRV_PY=/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python
CLI_VENV=$ROOT/stt-benchmark/.venv/bin/python
AUDIO=$ROOT/proj-2026-05-20-modal-cost/loadgen_audio
MODEL=nvidia/nemotron-speech-streaming-en-0.6b
STALL_S="${STALL_S:-25}"
trap 'pkill -f "server.py --model" 2>/dev/null; kill $(jobs -p) 2>/dev/null' EXIT

run_cfg() {  # $1=label $2=LANES $3=WARMUP_MS
  local label=$1 lanes=$2 warmup=$3
  local LOG=$ROOT/proj-2026-05-22-1353/stall_confirm_${label}.log
  echo ""; echo "########## $label (lanes=$lanes warmup=$warmup) | conc-24 + stream-jitter 400 ##########"
  pkill -f "server.py --model" 2>/dev/null; sleep 2; : > "$LOG"
  ( cd src/nemotron_speech && env -u LD_LIBRARY_PATH \
      NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 "NEMOTRON_WARMUP_MS=$warmup" \
      NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8 \
      "NEMOTRON_MODEL_LANES=$lanes" NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 \
      NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_BATCH_FINALIZE_PREPROC=1 NEMOTRON_BATCH_BARRIER_DRAIN=1 \
      NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1 NEMOTRON_FINALIZE_PROFILE=1 NEMOTRON_FAULTHANDLER=1 \
      "$SRV_PY" server.py --model "$MODEL" --host 127.0.0.1 --port 8080 --right-context 1 ) >> "$LOG" 2>&1 &
  local ok=0; for _ in $(seq 1 100); do sleep 3; (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null && { exec 3>&-; ok=1; break; }; done
  [ $ok != 1 ] && { echo "  SERVER FAILED"; tail -15 "$LOG"; return 1; }
  local SRVPID; SRVPID=$(pgrep -f "server.py --model" | head -1); sleep 2
  LOADGEN_JITTER_MS=400 LOADGEN_STREAM_JITTER_MS=400 "$CLI_VENV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 \
      --sweep 24 --rounds 8 --audio-dir "$AUDIO" > "$ROOT/proj-2026-05-22-1353/stall_confirm_${label}_lg.log" 2>&1 &
  local LGPID=$!
  local HUNG=0; sleep 12
  while kill -0 $LGPID 2>/dev/null; do
    local age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
    if [ "$age" -gt "$STALL_S" ]; then
      echo "  !!! STALL: server log silent ${age}s !!!"
      kill -USR1 "$SRVPID" 2>/dev/null; sleep 2
      echo "  --- gdb native bt (wedged-thread CUDA frames: sync=backlog vs launch=launch-queue) ---"
      gdb -p "$SRVPID" -batch -ex "set pagination off" -ex "thread apply all bt" 2>&1 \
        | grep -iE "cudaStreamSynchronize|cudaLaunchKernel|cudaDeviceSynchronize|cuStreamSynchronize|scatter_cache_row|cache_aware_stream_step|rnnt|label_looping" | head -20
      HUNG=1; kill $LGPID 2>/dev/null; break
    fi
    sleep 5
  done
  pkill -f "server.py --model" 2>/dev/null; sleep 2
  if [ "$HUNG" = 1 ]; then echo "  RESULT $label: *** STALLED *** (finalizes before=$(grep -c finalize_profile_record "$LOG"))"
  else echo "  RESULT $label: no catastrophic stall | worst model_batch_ms=$(grep -oE 'model_batch_ms=[0-9.]+' "$LOG" | grep -oE '[0-9.]+' | sort -n | tail -1) | $(grep -E '^ +24 ' "$ROOT/proj-2026-05-22-1353/stall_confirm_${label}_lg.log" | tail -1)"; fi
}

run_cfg lanes2_warm200 2 200
run_cfg lanes1_warm200 1 200
run_cfg lanes2_warm0   2 0
echo ""; echo "=== CONFIRM SUMMARY ==="
grep -hE "RESULT " "$ROOT"/proj-2026-05-22-1353/stall_confirm_*.log 2>/dev/null || true
echo "(baseline stalls; if lanes1 or warm0 does NOT stall -> that amplifier confirmed)"