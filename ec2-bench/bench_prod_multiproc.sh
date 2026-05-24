#!/usr/bin/env bash
# PRODUCTION re-measurement: is the conc-10 finalize lock_wait tail (single-proc bench: p50 0.36 -> p95 21ms)
# a single-process artifact, or real on the REAL prod config (multi-proc + CUDA MPS + LB)?
# Brings up ONE g6e.8xlarge (L40S), runs deploy/launch_multiproc.sh (MPS + K=3 procs lanes=2, FINALIZE_PROFILE on)
# behind HAProxy (leastconn, maxconn 12/proc), then runs the SAME full-1000 @ conc CONC client over WAN to the LB.
# K=3 (not 4): with the finalize graph on, each proc is ~11GB -> K=4 OOMs the 44GB L40S (measured). leastconn
# spreads ~10 sessions across 3 procs (~3.3/proc -> ~1.65/lane) => contention SHOULD drop if it was a
# single-proc artifact. Pulls per-proc finalize records for the lock_wait decomposition. ALWAYS terminates. Run as a FILE.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
E=ec2-bench
PY=stt-benchmark/.venv/bin/python
KEY=$E/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15"
PROFILE=AWSAdministratorAccess-419599258555
REGION=us-west-2
MYIP="${MYIP:-$(curl -s https://checkip.amazonaws.com)}"
CONC="${CONC:-10}"
ITYPE="${ITYPE:-g6e.8xlarge}"          # L40S 44GB
# K=3 (not the matrix's K=4): each proc is ~11GB (model + 2-lane STEADY + FINALIZE graph pools); 4x11~=44GB
# exhausts the 44GB L40S (OOM cascade observed at K=4). 3x11~=33GB leaves ~11GB streaming headroom. K=3 is a
# documented prod config (g6e.4xlarge row). FINDING: the launcher auto_pick_K=4 for g6e.8xlarge is memory-infeasible
# with the finalize graph on -> auto_pick_K should account for graph-pool memory (see DEPLOYMENT.md note).
K="${K:-3}"                            # procs; backends 8081..808(K)
FRONT=8080                             # HAProxy front (client hits this); procs on 8081..
TAG="prod_multiproc_l40s_c${CONC}_K${K}"
OUT=$E/prodmp_$(date +%H%M); mkdir -p "$OUT"

SG=$("$PY" - <<PY
import boto3
ec2=boto3.Session(profile_name="$PROFILE").client("ec2","$REGION")
print(ec2.describe_security_groups(GroupNames=["nemotron-bench-sg"])["SecurityGroups"][0]["GroupId"])
PY
)
echo "[sg] $SG ; opening tcp/$FRONT <- ${MYIP}/32"
"$PY" - "$SG" "$MYIP" "$FRONT" <<'PY'
import sys, boto3
sg, myip, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
ec2=boto3.Session(profile_name="AWSAdministratorAccess-419599258555").client("ec2","us-west-2")
try:
    ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{"IpProtocol":"tcp","FromPort":port,"ToPort":port,
        "IpRanges":[{"CidrIp":f"{myip}/32","Description":"bench client ws (LB front)"}]}]); print("  opened")
except Exception as e:
    print("  ", str(e)[:90])
PY

trap 'echo "[trap] terminate box"; '"$PY"' '"$E"'/ec2_down.py 2>/dev/null || true' EXIT

NEMOTRON_EC2_ITYPE=$ITYPE "$PY" $E/ec2_up.py || { echo "up FAILED"; exit 1; }
IP=$("$PY" -c "import json;print(json.load(open('$E/.instance.json'))['ip'])"); echo "IP=$IP"
bash $E/ec2_push.sh || { echo "push FAILED"; exit 1; }
# push the multi-proc launcher (ec2_push only sends server.py/batch_primitives/cudagraph_encoder + audio)
scp -i "$KEY" $SSHO deploy/launch_multiproc.sh ubuntu@"$IP":~/nemotron/launch_multiproc.sh

ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && PYVER=${PYVER:-3.11} nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started"
ok=0; for _ in $(seq 1 80); do sleep 15; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi DONE ~/nemotron/bootstrap.log' 2>/dev/null && { ok=1; echo "bootstrap DONE"; break; }; done
[ $ok != 1 ] && { echo "bootstrap TIMEOUT"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -20 ~/nemotron/bootstrap.log'; exit 1; }
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'sudo apt-get install -y -qq haproxy >/dev/null 2>&1 && haproxy -v | head -1'

# HAProxy: tcp leastconn, front :FRONT -> 127.0.0.1:8081..808K, maxconn 12/proc (the prod operating point).
"$PY" - "$K" "$FRONT" > "$OUT/haproxy_asr.cfg" <<'PY'
import sys
K, front = int(sys.argv[1]), int(sys.argv[2])
print("global\n    maxconn 100000")
print("defaults\n    mode tcp\n    timeout connect 5s\n    timeout client 1h\n    timeout server 1h")
print(f"frontend asr_ws\n    bind *:{front}\n    default_backend asr_pool")
print("backend asr_pool\n    balance leastconn")
for k in range(K):
    print(f"    server p{k} 127.0.0.1:{8081+k} check maxconn 12")
PY
scp -i "$KEY" $SSHO "$OUT/haproxy_asr.cfg" ubuntu@"$IP":~/nemotron/haproxy_asr.cfg

# Launch MPS + K procs (procs on 8081..; FINALIZE_PROFILE on) in the FOREGROUND of a backgrounded ssh.
ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && NEMOTRON_PROCS=$K NEMOTRON_BASE_PORT=8081 FINALIZE_PROFILE=1 \
    HF_HOME=\$HOME/hf bash launch_multiproc.sh > launcher.log 2>&1" &
SSHSRV=$!
echo "=== waiting for $K procs to report 'ASR server listening' ==="
sok=0; for _ in $(seq 1 120); do sleep 5
  n=$(ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -l "ASR server listening" ~/nemotron/server_*.log 2>/dev/null | wc -l' 2>/dev/null)
  [ "${n:-0}" -ge "$K" ] 2>/dev/null && { sok=1; echo "  all $K procs ready"; break; }
done
if [ $sok != 1 ]; then echo "procs FAILED ($n/$K ready)"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -15 ~/nemotron/launcher.log ~/nemotron/server_0.log' 2>/dev/null; kill $SSHSRV 2>/dev/null; exit 1; fi
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -hE "encoder_cuda_graph_enabled=|encoder_finalize_cuda_graph_enabled=" ~/nemotron/server_0.log | head -2'

# Start HAProxy (user-mode, front >1024 so no root bind needed); verify it listens.
ssh -i "$KEY" $SSHO ubuntu@"$IP" "haproxy -f ~/nemotron/haproxy_asr.cfg -D && sleep 2 && ss -ltn | grep -q ':$FRONT ' && echo 'haproxy up on :$FRONT'" || { echo "haproxy FAILED"; kill $SSHSRV 2>/dev/null; exit 1; }

echo "=== client full-1000 @ conc $CONC -> ws://$IP:$FRONT (LB -> $K procs + MPS; TTFB incl WAN) ==="
( sleep 60; while pgrep -f run_full1000_conc12 >/dev/null 2>&1; do
    age=$(ssh -i "$KEY" $SSHO ubuntu@"$IP" 'echo $(( $(date +%s) - $(stat -c %Y ~/nemotron/server_0.log 2>/dev/null || echo 0) ))' 2>/dev/null)
    [ -n "$age" ] && [ "$age" -gt 240 ] 2>/dev/null && { echo "[watchdog] server_0 silent ${age}s -> kill client"; pkill -f run_full1000_conc12 2>/dev/null; break; }
    sleep 30; done ) & WD=$!
timeout 1700 "$PY" proj-2026-05-19-eou-endpointing/run_full1000_conc12.py --url "ws://$IP:$FRONT" --model-tag "$TAG" --concurrency "$CONC" 2>&1 | tee "$OUT/${TAG}.clientlog" | tail -8
kill $WD 2>/dev/null || true

# Pull per-proc finalize records + logs, merge.
for k in $(seq 0 $((K-1))); do
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "grep finalize_profile_record ~/nemotron/server_$k.log" >> "$OUT/${TAG}.records" 2>/dev/null
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "cat ~/nemotron/server_$k.log" > "$OUT/server_$k.log" 2>/dev/null
done
ssh -i "$KEY" $SSHO ubuntu@"$IP" "cat ~/nemotron/launcher.log" > "$OUT/launcher.log" 2>/dev/null
ssh -i "$KEY" $SSHO ubuntu@"$IP" "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader" > "$OUT/gpu_mem.txt" 2>/dev/null || true
echo "  pulled $(wc -l < "$OUT/${TAG}.records" 2>/dev/null || echo 0) finalize records (merged across $K procs) -> $OUT/"

ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -f haproxy; pkill -f 'server.py --model'; echo quit | nvidia-cuda-mps-control" 2>/dev/null || true
kill $SSHSRV 2>/dev/null || true
"$PY" $E/ec2_down.py
echo "=== done + terminated. records: $OUT/${TAG}.records  client: $OUT/${TAG}.clientlog ==="
