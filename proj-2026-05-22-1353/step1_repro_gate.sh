#!/usr/bin/env bash
# Step 1 REPRODUCIBILITY GATE: does the ~178ms finalize-P95 reproduce as real SERVER COMPUTE (not network/noise)?
# Measured ON-BOX (WAN only affects the client number; server-side finalize compute is local + net-excluded), in
# the production config (cudagraph on, lanes=2) with NEMOTRON_FINALIZE_PROFILE=1 + the on-box loadgen at conc 10.
# Parse the per-final telemetry: model_wall / encoder / decode / preproc / lock_wait / queue_wait / sync (p50/p95)
# + the (B,T,first) histogram + first/second-half reproducibility. GATE READOUT: is the tail model_wall/encoder
# (-> GO build the graph) or lock/queue/sync contention (-> PIVOT)? And does it reproduce?
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
E=ec2-bench
PY=stt-benchmark/.venv/bin/python
KEY=$E/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15"
ITYPE="${NEMOTRON_EC2_ITYPE:-g6e.8xlarge}"

echo "=== launch $ITYPE for the finalize reproducibility gate ==="
NEMOTRON_EC2_ITYPE=$ITYPE "$PY" $E/ec2_up.py || { echo "up FAILED"; exit 1; }
IP=$("$PY" -c "import json;print(json.load(open('$E/.instance.json'))['ip'])")
echo "IP=$IP"
trap "echo '[trap] terminate'; $PY /home/khkramer/src/nemotron-january-2026/$E/ec2_down.py" EXIT

bash $E/ec2_push.sh || { echo "push FAILED"; exit 1; }
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'cd ~/nemotron && nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started'
ok=0; for _ in $(seq 1 80); do sleep 15; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi DONE ~/nemotron/bootstrap.log' 2>/dev/null && { ok=1; echo bootstrap DONE; break; }; done
[ $ok != 1 ] && { echo "bootstrap TIMEOUT"; exit 1; }

echo "=== start server FINALIZE_PROFILE=1 (lanes=2, cudagraph) ==="
ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && FINALIZE_PROFILE=1 LANES=2 CUDAGRAPH_MAX_B=8 bash start_prod_server.sh > srv_prof.log 2>&1" &
SSHSRV=$!
sok=0; for _ in $(seq 1 90); do sleep 4; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -q "ASR server listening" ~/nemotron/srv_prof.log' 2>/dev/null && { sok=1; break; }; done
[ $sok != 1 ] && { echo "server FAILED"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -30 ~/nemotron/srv_prof.log'; kill $SSHSRV 2>/dev/null; exit 1; }
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -E "FINALIZE_PROFILE|encoder_cuda_graph_enabled=" ~/nemotron/srv_prof.log | head'

for burst in A B; do
  echo "=== loadgen burst $burst (conc 10, rounds 20 ~= 200 finals) ==="
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && \$HOME/nemo-venv/bin/python ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep 10 --rounds 20 --audio-dir \$HOME/nemotron/loadgen_audio 2>&1 | tail -2"
done

echo "=== pull + parse server-side finalize telemetry ==="
scp -i "$KEY" $SSHO ubuntu@"$IP":/home/ubuntu/nemotron/srv_prof.log /tmp/srv_prof_gate.log 2>/dev/null
ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -f 'server.py --model'" 2>/dev/null || true
kill $SSHSRV 2>/dev/null || true
"$PY" - <<'PY'
import json
recs=[]
for line in open('/tmp/srv_prof_gate.log'):
    i=line.find('finalize_profile_record')
    if i<0: continue
    j=line.find('{', i)
    if j<0: continue
    try: recs.append(json.loads(line[j:]))
    except Exception: pass
print('finalize_profile records parsed:', len(recs))
if not recs: raise SystemExit('NO RECORDS — telemetry did not emit (investigate)')
def pct(vals,q):
    v=sorted(x for x in vals if x is not None)
    return v[min(len(v)-1,int(round(q*(len(v)-1))))] if v else float('nan')
print('--- server-side finalize split (ms), p50 / p95 / max ---')
for f in ('model_wall_ms','encoder_wall_ms','encoder_cuda_event_ms','decode_wall_ms','preproc_wall_ms','lock_wait_ms','queue_wait_ms','cuda_sync_ms','fork_clone_ms'):
    vals=[r.get(f) for r in recs if r.get(f) is not None]
    if vals: print(f'  {f:22} p50={pct(vals,.5):7.2f}  p95={pct(vals,.95):7.2f}  max={max(vals):7.2f}  n={len(vals)}')
mw=[r.get('model_wall_ms') for r in recs if r.get('model_wall_ms') is not None]
h=len(mw)//2
print(f'--- reproducibility: model_wall p95 firstHalf={pct(mw[:h],.95):.2f} secondHalf={pct(mw[h:],.95):.2f} (n={len(mw)}) ---')
from collections import Counter
bt=Counter((r.get('B'),r.get('T'),r.get('first')) for r in recs)
print('--- (B,T,first) histogram (top 10) ---'); [print('  ',k,v) for k,v in bt.most_common(10)]
# GATE hint
enc=pct([r.get('encoder_wall_ms') for r in recs if r.get('encoder_wall_ms') is not None],.95)
mdl=pct(mw,.95); lk=pct([r.get('lock_wait_ms') for r in recs if r.get('lock_wait_ms') is not None],.95)
print(f'\nGATE HINT: model_wall p95={mdl:.1f}ms (encoder {enc:.1f}) vs lock_wait p95={lk:.1f}ms.')
print('  Compare model_wall p95 to the ~178ms client-attributed server-finalize: if model_wall is the bulk AND')
print('  encoder dominates it -> GO (graph the encoder). If model_wall is small but the 178ms client tail is real,')
print('  the tail is network/control-path/contention -> PIVOT (graph will not move client P95).')
PY
echo "=== gate measurement done; trap terminates ==="