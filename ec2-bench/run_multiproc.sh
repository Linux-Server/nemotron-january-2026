#!/usr/bin/env bash
# On-box: multi-PROCESS scaling test. K independent server processes (each lanes=2, own port) +
# K concurrent load-gens (each N_PER streams). Tests whether GIL-independent processes use the idle
# GPU to push per-box throughput past the single-process lane cap (~16). Per-box target = K * N_PER.
set -uo pipefail
VENV=$HOME/nemo-venv; cd "$HOME/nemotron"; export HF_HOME=$HOME/hf
K_LIST="${K_LIST:-1,2,3,4}"; N_PER="${N_PER:-16}"
MODEL=nvidia/nemotron-speech-streaming-en-0.6b
SRV=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 "HF_HOME=$HOME/hf"
     NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8
     NEMOTRON_MODEL_LANES=2)

wait_port(){ local p=$1; for _ in $(seq 1 180); do (exec 3<>/dev/tcp/127.0.0.1/"$p") 2>/dev/null && { exec 3>&-; return 0; }; sleep 2; done; return 1; }

for K in ${K_LIST//,/ }; do
  echo "############ K=$K processes x lanes2 ; N_PER=$N_PER ; per-box target=$((K*N_PER)) ############"
  pkill -f "server.py --model" 2>/dev/null; sleep 5
  for k in $(seq 0 $((K-1))); do
    env -u LD_LIBRARY_PATH "${SRV[@]}" "$VENV/bin/python" server.py --model "$MODEL" \
        --host 127.0.0.1 --port $((8080+k)) --right-context 1 > "server_mp${K}_$k.log" 2>&1 &
  done
  ok=1; for k in $(seq 0 $((K-1))); do wait_port $((8080+k)) || { echo "server $k FAILED"; tail -25 "server_mp${K}_$k.log"; ok=0; }; done
  if [ $ok != 1 ]; then pkill -f "server.py --model"; sleep 3; continue; fi
  sleep 3
  LG=()
  for k in $(seq 0 $((K-1))); do
    "$VENV/bin/python" ec2_loadgen.py --url ws://127.0.0.1:$((8080+k)) --sweep "$N_PER" \
        --audio-dir "$HOME/nemotron/loadgen_audio" --output "mp${K}_$k.json" > "loadgen_mp${K}_$k.log" 2>&1 &
    LG+=($!)
  done
  for _ in $(seq 1 6); do sleep 3; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed "s/^/  [K=$K gpu] /"; done
  wait "${LG[@]}"   # wait ONLY the loadgens — bare `wait` would block on the never-exiting servers
  echo "--- K=$K per-process result at N_PER=$N_PER (all must keep up for per-box=$((K*N_PER))) ---"
  upcount=0
  for k in $(seq 0 $((K-1))); do
    line=$(grep -E "^ +$N_PER " "loadgen_mp${K}_$k.log" | tail -1)
    echo "  proc$k: $line"
    echo "$line" | grep -q "YES" && upcount=$((upcount+1))
  done
  echo "  >>> K=$K: $upcount/$K processes kept up  => per-box keep-up ~= $((upcount*N_PER))"
  pkill -f "server.py --model"; sleep 3
done
echo "############ MULTIPROC DONE ############"
