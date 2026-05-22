#!/usr/bin/env bash
# Step 5: local 5090 keep-up knee, graph-OFF vs graph-ON (lanes=1, continuous batching on).
# Isolates the cudagraph effect on top of the ~56 batching knee. Reports each config's knee
# (max N with proc-lag p95 < 500ms) + the graph-on B-engagement mix. Run as a FILE (not inline)
# so `pkill -f "server.py --model"` does not match the driver's own command line.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
SRV=/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python
HV=stt-benchmark/.venv/bin/python
MODEL=$(cat /tmp/en-nemo-path)
AUDIO=proj-2026-05-20-modal-cost/loadgen_audio
SWEEP="${SWEEP:-24,32,40,48,56,64,72,80,88,96}"
COMMON=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200
        NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 PYTHONPATH=src)

run(){
  local tag=$1 log=$2; shift 2; local extra=("$@")
  pkill -f "server.py --model" 2>/dev/null; sleep 4
  env "${COMMON[@]}" "${extra[@]}" "$SRV" src/nemotron_speech/server.py --model "$MODEL" \
      --host 127.0.0.1 --port 8080 --right-context 1 > "$log" 2>&1 &
  local ok=0
  for _ in $(seq 1 150); do grep -q "ASR server listening" "$log" 2>/dev/null && { ok=1; break; }; sleep 2; done
  [ $ok != 1 ] && { echo "[$tag] FAILED start"; tail -20 "$log"; pkill -f "server.py --model"; return 1; }
  echo "=== [$tag] sweep $SWEEP ==="
  "$HV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep "$SWEEP" --audio-dir "$AUDIO"
  pkill -f "server.py --model"; sleep 3
}

echo "########## GRAPH OFF (lanes1, batching) ##########"; run koff /tmp/k_off.log
echo "########## GRAPH ON  (lanes1, batching) ##########"; run kon  /tmp/k_on.log NEMOTRON_ENCODER_CUDAGRAPH=1
echo "########## B ENGAGEMENT (graph on) ##########"
echo "-- enabled + captured --"; grep -E "encoder_cuda_graph_enabled=|manager_captured" /tmp/k_on.log | head
echo "-- sampled B distribution (from status logs, count x B) --"; grep -oE "B=[0-9]+" /tmp/k_on.log | sort -t= -k2 -n | uniq -c
echo "-- final replay/fallback tally --"; grep -oE "replays=[0-9]+ fallbacks=[0-9]+" /tmp/k_on.log | tail -1
echo "########## STEP5 DONE ##########"
