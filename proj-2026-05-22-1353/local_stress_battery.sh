#!/usr/bin/env bash
# Large adversarial stress battery for the scheduler livelock fix (commit d8a98f3). Runs the full prod config
# (cudagraph+lanes2+barrier-drain+batch-finalize) through a dozen hard configs — overload past the knee, in-phase
# coincident finals (barrier-drain stress), WAN-mimic stream jitter (uneven backlog — the suspected trigger), and
# worst-case combos — with a continuous server-silence watchdog that USR1-dumps stacks on any freeze.
# SRV_PY selects the server venv (default 3.12 .venv-asr; set to a 3.11 venv to test the REPRODUCING env).
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
ROOT=$(pwd)
SRV_PY="${SRV_PY:-/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python}"
CLI_VENV=$ROOT/stt-benchmark/.venv/bin/python
AUDIO=$ROOT/proj-2026-05-20-modal-cost/loadgen_audio
LOG="${LOG:-$ROOT/proj-2026-05-22-1353/stress_srv.log}"
HUNG=/tmp/stress_hung.flag; STALL_S="${STALL_S:-25}"
MODEL=nvidia/nemotron-speech-streaming-en-0.6b
rm -f "$HUNG"

pkill -f "server.py --model" 2>/dev/null; sleep 2; : > "$LOG"
echo "=== server venv: $($SRV_PY --version 2>&1) | log=$LOG ==="
( cd src/nemotron_speech && env -u LD_LIBRARY_PATH \
    NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 \
    NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8 \
    NEMOTRON_MODEL_LANES=2 NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 \
    NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_BATCH_FINALIZE_PREPROC=1 NEMOTRON_BATCH_BARRIER_DRAIN=1 \
    NEMOTRON_FINALIZE_PROFILE=1 NEMOTRON_FAULTHANDLER=1 "NEMOTRON_SCHED_NO_YIELD=${NEMOTRON_SCHED_NO_YIELD:-0}" \
    "$SRV_PY" server.py --model "$MODEL" --host 127.0.0.1 --port 8080 --right-context 1 ) >> "$LOG" 2>&1 &
trap 'pkill -f "server.py --model" 2>/dev/null; kill $(jobs -p) 2>/dev/null; rm -f "$HUNG"' EXIT
ok=0; for _ in $(seq 1 100); do sleep 3; (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null && { exec 3>&-; ok=1; break; }; done
[ $ok != 1 ] && { echo "SERVER FAILED"; tail -30 "$LOG"; exit 1; }
SRVPID=$(pgrep -f "server.py --model" | head -1); echo "server ready (pid=$SRVPID)"; sleep 2

( while true; do   # watchdog: server-log silence > STALL_S WHILE a loadgen runs = HANG
    sleep 5
    pgrep -f ec2_loadgen >/dev/null 2>&1 || continue
    age=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || echo 0) ))
    if [ "$age" -gt "$STALL_S" ]; then
      echo "!!! HANG: server silent ${age}s !!!" | tee -a "$LOG"
      kill -USR1 "$SRVPID" 2>/dev/null; sleep 3; kill -USR1 "$SRVPID" 2>/dev/null; sleep 2
      touch "$HUNG"; pkill -f ec2_loadgen 2>/dev/null; break
    fi
  done ) &
WD=$!

run() {  # $1 conc  $2 start_jitter_ms  $3 stream_jitter_ms  $4 rounds  $5 allclips(1|"")  $6 label
  [ -f "$HUNG" ] && return
  echo ""; echo "### $6 | conc=$1 start_jitter=$2 stream_jitter=$3 rounds=$4 allclips=${5:-0} ###"
  env LOADGEN_JITTER_MS="$2" LOADGEN_STREAM_JITTER_MS="$3" ${5:+LOADGEN_ALL_CLIPS=1} \
    "$CLI_VENV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep "$1" --rounds "$4" \
    --audio-dir "$AUDIO" 2>&1 | grep -E "TTFSp50|^ +$1 " | tail -1
}
run 10 400 0    5  ""  "baseline staggered"
run 16 400 0    6  ""  "overload-16 (past knee)"
run 24 400 0    6  ""  "overload-24"
run 32 400 0    6  ""  "overload-32"
run 16 0   0    8  ""  "in-phase-16 (coincident finals)"
run 24 0   0    8  ""  "in-phase-24"
run 32 0   0    6  ""  "in-phase-32"
run 16 400 300  8  ""  "WAN-jitter-16"
run 24 400 400  8  ""  "WAN-jitter-24"
run 24 0   300  8  ""  "WORST: in-phase + WAN-jitter-24"
run 32 200 500  8  1   "heavy mixed-32 allclips"
run 16 400 0    25 ""  "sustained-16 (long duration)"

kill $WD 2>/dev/null; pkill -f "server.py --model" 2>/dev/null; sleep 2
echo ""
if [ -f "$HUNG" ]; then
  echo "########## !!! HANG REPRODUCED (finalizes before hang=$(grep -c finalize_profile_record "$LOG")) !!! ##########"
  echo "=== model_batch_ms tail ==="; grep -oE "model_batch_ms=[0-9.]+" "$LOG" | tail -6
  echo "=== USR1 thread dump (where stuck) ==="; awk '/^Thread 0x|^Current thread|^  File /{print}' "$LOG" | tail -45
  rm -f "$HUNG"
else
  echo "########## NO HANG across the full battery — total finalizes=$(grep -c finalize_profile_record "$LOG") ##########"
fi
echo "=== log: $LOG ==="