#!/usr/bin/env bash
# LOCAL finalize last-stage probe (RTX5090). Runs the REAL server (prod config + the new fine-grained
# last-stage instrumentation: final_gather_ms / clone_hyp_flush_ms) and drives it with the loadgen at
# conc-1 (isolates per-utterance length variation, clean B=1) + conc-10 (steady-state leaderboard load).
# Goal: localize the conc-10 P50->P95 spread to a last-stage step (gather/clone/decode) and test whether
# it scales with hypothesis/token length. Fast local CPU hides absolute ms but the DISTRIBUTION shows.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
ROOT=$(pwd)
SRV_VENV=/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python   # has nemo+aiohttp+torch
CLI_VENV=$ROOT/stt-benchmark/.venv/bin/python                         # has websockets
AUDIO=$ROOT/proj-2026-05-20-modal-cost/loadgen_audio
LOG=/tmp/local_probe_srv.log
REC=$ROOT/proj-2026-05-22-1353/local_probe_records.txt
MODEL=nvidia/nemotron-speech-streaming-en-0.6b

pkill -f "server.py --model" 2>/dev/null; sleep 2
echo "=== start local server (prod config + last-stage instrumentation) ==="
( cd src/nemotron_speech && env -u LD_LIBRARY_PATH \
    NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 \
    NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8 \
    NEMOTRON_MODEL_LANES=2 NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 \
    NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_BATCH_FINALIZE_PREPROC=1 NEMOTRON_FINALIZE_PROFILE=1 \
    "$SRV_VENV" server.py --model "$MODEL" --host 127.0.0.1 --port 8080 --right-context 1 ) > "$LOG" 2>&1 &
trap 'echo "[trap] stop server"; pkill -f "server.py --model" 2>/dev/null' EXIT

echo "waiting for server ready (model load + cudagraph capture)..."
ok=0; for _ in $(seq 1 80); do sleep 3; (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null && { exec 3>&-; ok=1; break; }; done
[ $ok != 1 ] && { echo "SERVER FAILED TO START"; tail -30 "$LOG"; exit 1; }
echo "server ready"; sleep 2
grep -E "encoder_cuda_graph_enabled=|model loaded|MODEL_LANES" "$LOG" | head -5

run_level () { # $1=conc $2=rounds $3=label
  echo ""; echo "=== loadgen $3: conc=$1 rounds=$2 (audio=$(ls "$AUDIO"|wc -l) clips) ==="
  "$CLI_VENV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep "$1" --rounds "$2" \
      --audio-dir "$AUDIO" 2>&1 | grep -E "TTFSp50|^ +$1 " | tail -3
}
run_level 1 40 "CONC-1 (isolate per-utterance last-stage variation)"
grep "finalize_profile_record" "$LOG" > "$REC.conc1"
N1=$(wc -l < "$REC.conc1")
run_level 10 12 "CONC-10 (steady-state leaderboard load)"
grep "finalize_profile_record" "$LOG" | tail -n +$((N1 + 1)) > "$REC.conc10"

echo ""; echo "=== collected finalize_profile records ==="
echo "conc-1 : $(wc -l < "$REC.conc1") -> $REC.conc1"
echo "conc-10: $(wc -l < "$REC.conc10") -> $REC.conc10"
echo "=== done; trap stops server ==="