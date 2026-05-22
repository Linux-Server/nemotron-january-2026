#!/usr/bin/env bash
# Does NEMOTRON_BATCH_FINALIZE (pinned-lane finalize) collapse the host-side finalize-serialization tail that the
# K=4+MPS gate exposed (lock_wait p95 95 + queue_wait p95 100, with BATCH_FINALIZE OFF = global-exclusive path)?
# Run BOTH configs on ONE box (clean), K=4 + MPS, FINALIZE_PROFILE=1, conc-10 (in-keep-up regime, so the signal is
# finalize serialization, not backlog). Compare the per-final lock_wait / queue_wait / model_wall p50/p95.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
E=ec2-bench
PY=stt-benchmark/.venv/bin/python
KEY=$E/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15"
ITYPE="${NEMOTRON_EC2_ITYPE:-g6e.8xlarge}"; K="${K:-4}"; N_LIST="${N_LIST:-10}"; ROUNDS="${ROUNDS:-10}"

echo "=== launch $ITYPE for BATCH_FINALIZE on/off comparison (K=$K + MPS) ==="
NEMOTRON_EC2_ITYPE=$ITYPE "$PY" $E/ec2_up.py || { echo "up FAILED"; exit 1; }
IP=$("$PY" -c "import json;print(json.load(open('$E/.instance.json'))['ip'])")
echo "IP=$IP"
trap "echo '[trap] terminate'; $PY /home/khkramer/src/nemotron-january-2026/$E/ec2_down.py" EXIT
bash $E/ec2_push.sh || { echo "push FAILED"; exit 1; }
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'cd ~/nemotron && nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started'
ok=0; for _ in $(seq 1 80); do sleep 15; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi DONE ~/nemotron/bootstrap.log' 2>/dev/null && { ok=1; echo bootstrap DONE; break; }; done
[ $ok != 1 ] && { echo "bootstrap TIMEOUT"; exit 1; }

for BF in 0 1; do
  echo ""; echo "########## BATCH_FINALIZE=$BF (K=$K + MPS, conc $N_LIST) ##########"
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && K=$K LANES=2 N_LIST=$N_LIST ROUNDS=$ROUNDS FINALIZE_PROFILE=1 BATCH_FINALIZE=$BF P50_MAX=250 P95_MAX=300 bash run_l4_ttfs_sweep.sh" 2>&1 | grep -E "VERDICT|proc[0-9]:|per-box|^  [0-9]+ "
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "cat ~/nemotron/srv_l4ttfs_*.log | grep finalize_profile_record" > /tmp/bf${BF}_records.txt 2>/dev/null
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -f 'server.py --model'" 2>/dev/null || true; sleep 3
done

echo ""; echo "########## COMPARE: per-final finalize split, BATCH_FINALIZE 0 vs 1 ##########"
"$PY" - <<'PY'
import json
def load(p):
    rs=[]
    for line in open(p):
        j=line.find('{', line.find('finalize_profile_record'))
        if j<0: continue
        try: rs.append(json.loads(line[j:]))
        except Exception: pass
    return rs
def pct(vals,q):
    v=sorted(x for x in vals if x is not None)
    return v[min(len(v)-1,int(round(q*(len(v)-1))))] if v else float('nan')
for f in ('lock_wait_ms','queue_wait_ms','model_wall_ms','encoder_wall_ms','fork_clone_ms','preproc_wall_ms'):
    row=f'{f:16}'
    for bf in (0,1):
        rs=load(f'/tmp/bf{bf}_records.txt'); vals=[r.get(f) for r in rs if r.get(f) is not None]
        row += f'  BF{bf}: p50={pct(vals,.5):6.1f} p95={pct(vals,.95):6.1f} max={ (max(vals) if vals else float("nan")):6.1f} (n={len(vals)})'
    print(row)
print("\nVERDICT: if BF1 lock_wait/queue_wait p95 << BF0, pinned-lane finalize collapses the host-side")
print("serialization tail -> enabling NEMOTRON_BATCH_FINALIZE in prod is the (near-zero-code) fix.")
PY
echo "=== compare done; trap terminates ==="