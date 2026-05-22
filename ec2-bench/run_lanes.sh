#!/usr/bin/env bash
# On-box: B=1 baseline + parallel-lane knee sweeps vs the localhost server.
#   LANES (comma list, default "2,4")  : NEMOTRON_MODEL_LANES values to test (each with scheduler+batching on)
#   SWEEP (comma list)                 : concurrency levels
#   SKIP_BASELINE=1                    : skip the B=1 (no-batch) baseline
# Lanes load one model replica each; lanes require scheduler + batching on.
set -uo pipefail
VENV=$HOME/nemo-venv
cd "$HOME/nemotron"
export HF_HOME=$HOME/hf
SWEEP="${SWEEP:-1,4,8,12,16,24,32,40,48,56,64}"
LANES="${LANES:-2,4}"
MODEL=nvidia/nemotron-speech-streaming-en-0.6b
COMMON=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 "HF_HOME=$HOME/hf")
BATCH=(NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8)

wait_port(){ for _ in $(seq 1 180); do (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null && { exec 3>&-; return 0; }; sleep 2; done; return 1; }

run_one(){
  local name="$1"; shift
  echo "============ $name :: $* ============"
  pkill -f "server.py --model" 2>/dev/null; sleep 3
  env -u LD_LIBRARY_PATH "${COMMON[@]}" "$@" "$VENV/bin/python" server.py --model "$MODEL" \
      --host 127.0.0.1 --port 8080 --right-context 1 > "server_$name.log" 2>&1 &
  echo "waiting for :8080 ($* — replicas load serially) ..."
  if ! wait_port; then echo "SERVER FAILED ($name); tail:"; tail -40 "server_$name.log"; pkill -f "server.py --model"; return 1; fi
  grep -iE "model_lanes_enabled|model_lanes_disabled" "server_$name.log" | tail -1
  sleep 3
  "$VENV/bin/python" ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep "$SWEEP" \
      --audio-dir "$HOME/nemotron/loadgen_audio" --output "$name.json"
  pkill -f "server.py --model"; sleep 3
}

[ "${SKIP_BASELINE:-0}" = "1" ] || run_one baseline
for L in ${LANES//,/ }; do
  run_one "lanes$L" "${BATCH[@]}" "NEMOTRON_MODEL_LANES=$L"
done
echo "============ LANES DONE ============"
