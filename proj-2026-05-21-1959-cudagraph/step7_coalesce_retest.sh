#!/usr/bin/env bash
# Step 7: 2x2 tight-budget sweep MAX_WAIT in {8,0} x CUDAGRAPH in {0,1} on ONE box.
# Tests whether dropping the coalescing tick (MAX_WAIT=0 = work-conserving) costs throughput once cudagraph
# collapses per-launch cost. Predict: MAX_WAIT=0 hurts the knee with graphs OFF, ~neutral with graphs ON, lower
# p95. Boots NEMOTRON_EC2_ITYPE, bootstraps, pushes committed code, runs the 4 sweeps, ALWAYS terminates.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026/ec2-bench
PY=/home/khkramer/src/nemotron-january-2026/stt-benchmark/.venv/bin/python
KEY=/home/khkramer/src/nemotron-january-2026/ec2-bench/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15"
ITYPE="${NEMOTRON_EC2_ITYPE:?set NEMOTRON_EC2_ITYPE}"
K="${K:-2}"; LANES="${LANES:-2}"; N_LIST="${N_LIST:-8,10,12,14,16}"; ROUNDS="${ROUNDS:-5}"
CUDAGRAPH_MAX_B="${CUDAGRAPH_MAX_B:-8}"

echo "=== launch $ITYPE for coalesce 2x2 (K=$K lanes=$LANES N_LIST=$N_LIST) ==="
$PY ec2_up.py || { echo "up FAILED"; exit 1; }
IP=$($PY -c "import json;print(json.load(open('.instance.json'))['ip'])")
echo "IP=$IP"
trap "echo '=== [trap] terminate ==='; $PY /home/khkramer/src/nemotron-january-2026/ec2-bench/ec2_down.py" EXIT

echo "=== push (committed) ==="; bash ec2_push.sh || { echo "push FAILED"; exit 1; }

echo "=== bootstrap ==="
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'cd ~/nemotron && nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started'
boot_ok=0
for _ in $(seq 1 80); do sleep 15
  ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi "DONE" ~/nemotron/bootstrap.log' 2>/dev/null && { boot_ok=1; echo "bootstrap DONE"; break; }
  ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -1 ~/nemotron/bootstrap.log' 2>/dev/null | sed 's/^/  boot: /'
done
[ $boot_ok != 1 ] && { echo "bootstrap TIMEOUT"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -30 ~/nemotron/bootstrap.log'; exit 1; }

for MW in 8 0; do
  for CG in 0 1; do
    echo ""; echo "############################ MAX_WAIT=${MW}ms  CUDAGRAPH=${CG} ############################"
    ssh -i "$KEY" $SSHO ubuntu@"$IP" \
      "cd ~/nemotron && K=$K LANES=$LANES N_LIST=$N_LIST ROUNDS=$ROUNDS MAX_WAIT=$MW CUDAGRAPH=$CG CUDAGRAPH_MAX_B=$CUDAGRAPH_MAX_B P50_MAX=250 P95_MAX=300 bash run_l4_ttfs_sweep.sh" 2>&1
  done
done

echo ""; echo "=== coalesce 2x2 done; trap terminates ==="
