#!/usr/bin/env bash
# Step 1 gate, REDONE in the PRODUCTION regime: multi-process + MPS (the single-process gate was the wrong regime).
# K=4 L40S servers (lanes=2) + MPS + NEMOTRON_FINALIZE_PROFILE=1, all loaded (conc 10 AND 16/proc), to answer:
#   (a) does the eager finalize bifurcate/spike under MPS like the steady path did (-> GPU graph lever)?
#   (b) is the tail the Python/host gap (total server-side finalize span MINUS the sum of GPU/lock/clone components,
#       i.e. asyncio/GIL/GC) under parallel load (-> Python-path lever)?
# Captures BOTH the on-box loadgen TTFS (total server-side finalize span under MPS) AND the per-server finalize
# telemetry (component split). ALWAYS terminates.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
E=ec2-bench
PY=stt-benchmark/.venv/bin/python
KEY=$E/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15"
ITYPE="${NEMOTRON_EC2_ITYPE:-g6e.8xlarge}"; K="${K:-4}"; N_LIST="${N_LIST:-10,16}"; ROUNDS="${ROUNDS:-10}"

echo "=== launch $ITYPE for the MULTI-PROCESS+MPS finalize gate (K=$K) ==="
NEMOTRON_EC2_ITYPE=$ITYPE "$PY" $E/ec2_up.py || { echo "up FAILED"; exit 1; }
IP=$("$PY" -c "import json;print(json.load(open('$E/.instance.json'))['ip'])")
echo "IP=$IP"
trap "echo '[trap] terminate'; $PY /home/khkramer/src/nemotron-january-2026/$E/ec2_down.py" EXIT

bash $E/ec2_push.sh || { echo "push FAILED"; exit 1; }
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'cd ~/nemotron && nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started'
ok=0; for _ in $(seq 1 80); do sleep 15; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi DONE ~/nemotron/bootstrap.log' 2>/dev/null && { ok=1; echo bootstrap DONE; break; }; done
[ $ok != 1 ] && { echo "bootstrap TIMEOUT"; exit 1; }

echo "=== run K=$K + MPS sweep with FINALIZE_PROFILE=1 (loadgen TTFS = total server-side finalize span under MPS) ==="
ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && K=$K LANES=2 N_LIST=$N_LIST ROUNDS=$ROUNDS FINALIZE_PROFILE=1 P50_MAX=250 P95_MAX=300 bash run_l4_ttfs_sweep.sh" 2>&1 | grep -vE "MPS|^\s*$"

echo "=== collect + parse per-server finalize telemetry (component split, under K=$K MPS) ==="
ssh -i "$KEY" $SSHO ubuntu@"$IP" "cat ~/nemotron/srv_l4ttfs_*.log | grep finalize_profile_record" > /tmp/mp_finalize_records.txt 2>/dev/null
ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -f 'server.py --model'" 2>/dev/null || true
"$PY" - <<'PY'
import json
recs=[]
for line in open('/tmp/mp_finalize_records.txt'):
    j=line.find('{', line.find('finalize_profile_record'))
    if j<0: continue
    try: recs.append(json.loads(line[j:]))
    except Exception: pass
print('finalize records parsed (across K servers under MPS):', len(recs))
if not recs: raise SystemExit('NO RECORDS')
def pct(vals,q):
    v=sorted(x for x in vals if x is not None)
    return v[min(len(v)-1,int(round(q*(len(v)-1))))] if v else float('nan')
comps=('model_wall_ms','encoder_wall_ms','decode_wall_ms','preproc_wall_ms','lock_wait_ms','queue_wait_ms','cuda_sync_ms','fork_clone_ms')
print('--- per-final finalize component split (ms), p50 / p95 / max (UNDER K-proc MPS) ---')
for f in comps:
    vals=[r.get(f) for r in recs if r.get(f) is not None]
    if vals: print(f'  {f:18} p50={pct(vals,.5):7.2f}  p95={pct(vals,.95):7.2f}  max={max(vals):7.2f}')
# component-sum per final -> compare to the loadgen TTFS (total span) printed by the sweep above = the PYTHON/HOST gap
def s(r): return sum((r.get(c) or 0.0) for c in ('model_wall_ms','preproc_wall_ms','lock_wait_ms','queue_wait_ms','fork_clone_ms'))
sums=[s(r) for r in recs]
print(f'--- component-SUM (model+preproc+lock+queue+fork) p50={pct(sums,.5):.1f} p95={pct(sums,.95):.1f} max={max(sums):.1f} ---')
print('  (Compare to the loadgen TTFS p95 above = total server-side vad_stop->final span. total - this sum = the')
print('   un-instrumented Python/host overhead [asyncio/GIL/GC]. If that gap balloons under MPS -> the tail is Python.)')
from collections import Counter
print('--- (B,T,first) hist top8 ---'); [print('  ',k,v) for k,v in Counter((r.get('B'),r.get('T'),r.get('first')) for r in recs).most_common(8)]
PY
echo "=== multi-proc gate done; trap terminates ==="