#!/usr/bin/env bash
# Production per-proc LOAD SWEEP: how do client TTFB + finalize lock_wait grow as each proc fills toward its
# maxconn=12 operating point? ONE g6e.8xlarge (L40S) running K=3 procs lanes=2 + MPS + HAProxy(leastconn,maxconn12),
# then the full-1000 client (--limit LIMIT) at CONC_LIST = "10 20 30 36" (=> ~3.3/8.3/10/12 streams per proc).
# conc-36 saturates the box (3 procs x maxconn 12). Records are binned to each level by timestamp (post-step).
# ALWAYS terminates. Run as a FILE.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
E=ec2-bench; PY=stt-benchmark/.venv/bin/python; KEY=$E/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15"
PROFILE=AWSAdministratorAccess-419599258555; REGION=us-west-2
MYIP="${MYIP:-$(curl -s https://checkip.amazonaws.com)}"
ITYPE="${ITYPE:-g6e.8xlarge}"; K="${K:-3}"; FRONT=8080
CONC_LIST="${CONC_LIST:-10 20 30 36}"; LIMIT="${LIMIT:-500}"
OUT=$E/prodsweep_$(date +%H%M); mkdir -p "$OUT"; : > "$OUT/level_windows.txt"

SG=$("$PY" - <<PY
import boto3; print(boto3.Session(profile_name="$PROFILE").client("ec2","$REGION").describe_security_groups(GroupNames=["nemotron-bench-sg"])["SecurityGroups"][0]["GroupId"])
PY
)
echo "[sg] $SG ; opening tcp/$FRONT <- ${MYIP}/32"
"$PY" - "$SG" "$MYIP" "$FRONT" <<'PY'
import sys, boto3
sg, myip, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
ec2=boto3.Session(profile_name="AWSAdministratorAccess-419599258555").client("ec2","us-west-2")
try:
    ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{"IpProtocol":"tcp","FromPort":port,"ToPort":port,
        "IpRanges":[{"CidrIp":f"{myip}/32","Description":"bench sweep"}]}]); print("  opened")
except Exception as e: print("  ", str(e)[:90])
PY

trap 'echo "[trap] terminate box"; '"$PY"' '"$E"'/ec2_down.py 2>/dev/null || true' EXIT

NEMOTRON_EC2_ITYPE=$ITYPE "$PY" $E/ec2_up.py || { echo "up FAILED"; exit 1; }
IP=$("$PY" -c "import json;print(json.load(open('$E/.instance.json'))['ip'])"); echo "IP=$IP"
bash $E/ec2_push.sh || { echo "push FAILED"; exit 1; }
scp -i "$KEY" $SSHO deploy/launch_multiproc.sh ubuntu@"$IP":~/nemotron/launch_multiproc.sh
ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && PYVER=${PYVER:-3.11} nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started"
ok=0; for _ in $(seq 1 80); do sleep 15; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi DONE ~/nemotron/bootstrap.log' 2>/dev/null && { ok=1; echo "bootstrap DONE"; break; }; done
[ $ok != 1 ] && { echo "bootstrap TIMEOUT"; exit 1; }
ssh -i "$KEY" $SSHO ubuntu@"$IP" 'sudo apt-get install -y -qq haproxy >/dev/null 2>&1 && haproxy -v | head -1'

"$PY" - "$K" "$FRONT" > "$OUT/haproxy_asr.cfg" <<'PY'
import sys; K, front = int(sys.argv[1]), int(sys.argv[2])
print("global\n    maxconn 100000\ndefaults\n    mode tcp\n    timeout connect 5s\n    timeout client 1h\n    timeout server 1h")
print(f"frontend asr_ws\n    bind *:{front}\n    default_backend asr_pool\nbackend asr_pool\n    balance leastconn")
[print(f"    server p{k} 127.0.0.1:{8081+k} check maxconn 12") for k in range(K)]
PY
scp -i "$KEY" $SSHO "$OUT/haproxy_asr.cfg" ubuntu@"$IP":~/nemotron/haproxy_asr.cfg

ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && NEMOTRON_PROCS=$K NEMOTRON_BASE_PORT=8081 FINALIZE_PROFILE=1 HF_HOME=\$HOME/hf bash launch_multiproc.sh > launcher.log 2>&1" &
SSHSRV=$!
sok=0; for _ in $(seq 1 120); do sleep 5; n=$(ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -l "ASR server listening" ~/nemotron/server_*.log 2>/dev/null | wc -l' 2>/dev/null); [ "${n:-0}" -ge "$K" ] 2>/dev/null && { sok=1; echo "all $K procs ready"; break; }; done
[ $sok != 1 ] && { echo "procs FAILED ($n/$K)"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -15 ~/nemotron/server_0.log' 2>/dev/null; kill $SSHSRV 2>/dev/null; exit 1; }
ssh -i "$KEY" $SSHO ubuntu@"$IP" "haproxy -f ~/nemotron/haproxy_asr.cfg -D && sleep 2 && ss -ltn | grep -q ':$FRONT ' && echo 'haproxy up'" || { echo "haproxy FAILED"; kill $SSHSRV 2>/dev/null; exit 1; }

for C in $CONC_LIST; do
  tag="prod_mp_l40s_c${C}_K${K}"
  echo ""; echo "########## SWEEP conc=$C (limit $LIMIT; ~$(awk "BEGIN{printf \"%.1f\", $C/$K}")/proc) -> $tag ##########"
  echo "LEVEL conc=$C start=$(date +%s)" >> "$OUT/level_windows.txt"
  timeout 1200 "$PY" proj-2026-05-19-eou-endpointing/run_full1000_conc12.py --url "ws://$IP:$FRONT" --model-tag "$tag" --concurrency "$C" --limit "$LIMIT" 2>&1 | tee "$OUT/${tag}.clientlog" | grep -E "Completed in|TTFB \(speech|server finalize"
  echo "LEVEL conc=$C end=$(date +%s)" >> "$OUT/level_windows.txt"
  sleep 12
done

for k in $(seq 0 $((K-1))); do ssh -i "$KEY" $SSHO ubuntu@"$IP" "grep finalize_profile_record ~/nemotron/server_$k.log" >> "$OUT/all_procs.records" 2>/dev/null; done
echo "  pulled $(grep -c finalize_profile_record "$OUT/all_procs.records" 2>/dev/null || echo 0) records (all levels, all procs) -> $OUT/"
ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -f haproxy; pkill -f 'server.py --model'; echo quit | nvidia-cuda-mps-control" 2>/dev/null || true
kill $SSHSRV 2>/dev/null || true
"$PY" $E/ec2_down.py
echo "=== sweep done + terminated. records: $OUT/all_procs.records  windows: $OUT/level_windows.txt ==="
