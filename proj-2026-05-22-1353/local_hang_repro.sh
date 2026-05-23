#!/usr/bin/env bash
# Reproduce the cloud CUDA hang LOCALLY (RTX5090): full prod config (cudagraph + lanes=2 + barrier-drain +
# batch-finalize) under sustained conc-10. Watchdog: if the server log goes silent > STALL_S while the loadgen is
# still running, declare a HANG, dump all server thread stacks via SIGUSR1 (faulthandler) twice, then tear down.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
ROOT=$(pwd)
SRV_VENV=/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python
CLI_VENV=$ROOT/stt-benchmark/.venv/bin/python
AUDIO=$ROOT/proj-2026-05-20-modal-cost/loadgen_audio
LOG=$ROOT/proj-2026-05-22-1353/local_hang_srv.log
ROUNDS="${ROUNDS:-30}"; CONC="${CONC:-10}"; STALL_S="${STALL_S:-25}"
MODEL=nvidia/nemotron-speech-streaming-en-0.6b

pkill -f "server.py --model" 2>/dev/null; sleep 2; : > "$LOG"
echo "=== start server: FULL PROD CONFIG + faulthandler (cudagraph+lanes2+barrier-drain+batch-finalize) ==="
( cd src/nemotron_speech && env -u LD_LIBRARY_PATH \
    NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 \
    NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8 \
    NEMOTRON_MODEL_LANES=2 NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 \
    NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_BATCH_FINALIZE_PREPROC=1 NEMOTRON_BATCH_BARRIER_DRAIN=1 \
    NEMOTRON_FINALIZE_PROFILE=1 NEMOTRON_FAULTHANDLER=1 \
    "$SRV_VENV" server.py --model "$MODEL" --host 127.0.0.1 --port 8080 --right-context 1 ) >> "$LOG" 2>&1 &
trap 'pkill -f "server.py --model" 2>/dev/null; kill $(jobs -p) 2>/dev/null' EXIT
ok=0; for _ in $(seq 1 80); do sleep 3; (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null && { exec 3>&-; ok=1; break; }; done
[ $ok != 1 ] && { echo "SERVER FAILED"; tail -30 "$LOG"; exit 1; }
SRVPID=$(pgrep -f "server.py --model" | head -1); echo "server ready (pid=$SRVPID)"; sleep 2

echo "=== sustained load: conc=$CONC rounds=$ROUNDS (~$((CONC*ROUNDS)) finalizes; cloud hung at ~68) ==="
"$CLI_VENV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep "$CONC" --rounds "$ROUNDS" \
    --audio-dir "$AUDIO" > "$ROOT/proj-2026-05-22-1353/local_hang_loadgen.log" 2>&1 &
LGPID=$!

echo "=== watchdog (stall > ${STALL_S}s of server-log silence = HANG) ==="
HUNG=0; sleep 12
while kill -0 $LGPID 2>/dev/null; do
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ "$age" -gt "$STALL_S" ]; then
    echo "!!! HANG DETECTED: server log silent ${age}s (finalizes so far=$(grep -c finalize_profile_record "$LOG")) !!!"
    echo "--- SIGUSR1 stack dump #1 ---"; kill -USR1 "$SRVPID" 2>/dev/null; sleep 3
    echo "--- SIGUSR1 stack dump #2 (2s later — did anything move?) ---"; kill -USR1 "$SRVPID" 2>/dev/null; sleep 3
    HUNG=1; kill $LGPID 2>/dev/null; break
  fi
  sleep 5
done

pkill -f "server.py --model" 2>/dev/null; sleep 2
echo ""
if [ "$HUNG" = 1 ]; then
  echo "########## HUNG. finalizes before hang=$(grep -c finalize_profile_record "$LOG") ##########"
  echo "=== model_batch_ms tail (look for the spike before the freeze) ==="
  grep -oE "model_batch_ms=[0-9.]+" "$LOG" | tail -8
  echo "=== faulthandler thread dumps (WHERE it is stuck) ==="
  awk '/Thread 0x|Current thread|File "/{print}' "$LOG" | tail -60
else
  echo "########## NO HANG. loadgen result: ##########"; tail -3 "$ROOT/proj-2026-05-22-1353/local_hang_loadgen.log"
  echo "total finalizes=$(grep -c finalize_profile_record "$LOG")"
fi
echo "=== logs: $LOG + local_hang_loadgen.log ==="