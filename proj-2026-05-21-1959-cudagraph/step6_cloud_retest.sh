#!/usr/bin/env bash
# Step 6: cloud GPU-bound cudagraph retest at the TIGHT budget (p50<250/p95<300).
# Boots an EC2 box (NEMOTRON_EC2_ITYPE), bootstraps, pushes COMMITTED server.py + cudagraph_encoder.py,
# runs the tight-budget TTFS sweep GRAPH-OFF then GRAPH-ON (multi-process + MPS), and ALWAYS terminates.
# Env: NEMOTRON_EC2_ITYPE (g6.4xlarge|g6e.8xlarge), K, LANES, N_LIST, ROUNDS, CUDAGRAPH_MAX_B, P50_MAX, P95_MAX.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026/ec2-bench
PY=/home/khkramer/src/nemotron-january-2026/stt-benchmark/.venv/bin/python
KEY=/home/khkramer/src/nemotron-january-2026/ec2-bench/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15"
ITYPE="${NEMOTRON_EC2_ITYPE:?set NEMOTRON_EC2_ITYPE}"
K="${K:-2}"; LANES="${LANES:-2}"; N_LIST="${N_LIST:-8,10,12,14,16,18,20}"; ROUNDS="${ROUNDS:-5}"
CUDAGRAPH_MAX_B="${CUDAGRAPH_MAX_B:-}"; P50_MAX="${P50_MAX:-250}"; P95_MAX="${P95_MAX:-300}"

echo "=== launch $ITYPE for cudagraph retest (K=$K lanes=$LANES N_LIST=$N_LIST maxB=${CUDAGRAPH_MAX_B:-16}) ==="
$PY ec2_up.py || { echo "up FAILED"; exit 1; }
IP=$($PY -c "import json;print(json.load(open('.instance.json'))['ip'])")
echo "IP=$IP"
trap "echo '=== [trap] terminate ==='; $PY /home/khkramer/src/nemotron-january-2026/ec2-bench/ec2_down.py" EXIT

echo "=== push (committed server.py + cudagraph_encoder.py) ==="
bash ec2_push.sh || { echo "push FAILED"; exit 1; }

echo "=== bootstrap ==="
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'cd ~/nemotron && nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started'
boot_ok=0
for _ in $(seq 1 80); do sleep 15
  ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi "DONE" ~/nemotron/bootstrap.log' 2>/dev/null && { boot_ok=1; echo "bootstrap DONE"; break; }
  ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -1 ~/nemotron/bootstrap.log' 2>/dev/null | sed 's/^/  boot: /'
done
[ $boot_ok != 1 ] && { echo "bootstrap TIMEOUT"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -30 ~/nemotron/bootstrap.log'; exit 1; }

SWEEP_ENV="K=$K LANES=$LANES N_LIST=$N_LIST ROUNDS=$ROUNDS P50_MAX=$P50_MAX P95_MAX=$P95_MAX"
echo ""; echo "########################## GRAPH-OFF (baseline) ##########################"
ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && $SWEEP_ENV CUDAGRAPH=0 bash run_l4_ttfs_sweep.sh" 2>&1
echo ""; echo "########################## GRAPH-ON (cudagraph, maxB=${CUDAGRAPH_MAX_B:-16}) ##########################"
ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && $SWEEP_ENV CUDAGRAPH=1 CUDAGRAPH_MAX_B=$CUDAGRAPH_MAX_B bash run_l4_ttfs_sweep.sh" 2>&1

echo ""; echo "=== retest done; trap terminates the box ==="
