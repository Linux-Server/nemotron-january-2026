#!/usr/bin/env bash
# SAME-BOX lanes=2 vs lanes=1 A/B for the conc-10 leaderboard (apples-to-apples, controls run-to-run network
# noise — P95 is the whole question). On ONE fresh prod-configured box per ITYPE we run the client full-1000 @
# conc CONC TWICE: lanes=2 (the current leaderboard config, baseline 246/279 on L40S) then lanes=1 (the Track-A
# stall-eliminating config). EXACT prod finalize config both runs (cudagraph ON + CUDAGRAPH_FINALIZE + BATCH_FINALIZE
# + BARRIER_DRAIN + silence0_warm200 + rc1); ONLY NEMOTRON_MODEL_LANES differs. FINALIZE_PROFILE on -> server-side
# decomposition (expect lock_wait to drop with lanes=1: single lane = no cross-lane contention). ALWAYS terminates.
# Run as a FILE.
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
ITYPES="${ITYPES:-g6e.8xlarge g6.4xlarge}"
CUDAGRAPH_MAX_B="${CUDAGRAPH_MAX_B:-8}"
LANES_SET="${LANES_SET:-2 1}"           # baseline first, then the experiment
OUT=$E/lanes_ab_$(date +%H%M)
mkdir -p "$OUT"

base_for(){ case "$1" in g6.4xlarge|g6.*) echo l4;; g6e.*) echo l40s;; *) echo "$(echo "$1"|tr . _)";; esac; }

# Open the bench SG to MY_IP:8080 (box binds 0.0.0.0; server has no auth -> restrict to MY_IP).
SG=$("$PY" - <<PY
import boto3
ec2=boto3.Session(profile_name="$PROFILE").client("ec2","$REGION")
print(ec2.describe_security_groups(GroupNames=["nemotron-bench-sg"])["SecurityGroups"][0]["GroupId"])
PY
)
echo "[sg] $SG ; opening tcp/8080 <- ${MYIP}/32"
"$PY" - "$SG" "$MYIP" <<'PY'
import sys, boto3
sg, myip = sys.argv[1], sys.argv[2]
ec2=boto3.Session(profile_name="AWSAdministratorAccess-419599258555").client("ec2","us-west-2")
try:
    ec2.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{"IpProtocol":"tcp","FromPort":8080,"ToPort":8080,
        "IpRanges":[{"CidrIp":f"{myip}/32","Description":"bench client ws"}]}]); print("  opened")
except Exception as e:
    print("  ", str(e)[:90])
PY

trap 'echo "[trap] terminate current box"; '"$PY"' '"$E"'/ec2_down.py 2>/dev/null || true' EXIT

run_client(){  # $1=IP $2=tag $3=logfile
  local IP=$1 tag=$2 log=$3
  ( SILENCE_S=180; sleep 60
    while pgrep -f run_full1000_conc12 >/dev/null 2>&1; do
      age=$(ssh -i "$KEY" $SSHO ubuntu@"$IP" 'echo $(( $(date +%s) - $(stat -c %Y ~/nemotron/srv_prod.log 2>/dev/null || echo 0) ))' 2>/dev/null)
      if [ -n "$age" ] && [ "$age" -gt "$SILENCE_S" ] 2>/dev/null; then
        echo "[watchdog] server log silent ${age}s -> HANG; USR1 + killing client"
        ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -USR1 -f 'server.py --model'" 2>/dev/null || true
        pkill -f run_full1000_conc12 2>/dev/null || true; break
      fi; sleep 30
    done ) & local WD=$!
  timeout 1500 "$PY" proj-2026-05-19-eou-endpointing/run_full1000_conc12.py \
      --url "ws://$IP:8080" --model-tag "$tag" --concurrency "$CONC" 2>&1 | tee "$log" | tail -8
  kill $WD 2>/dev/null || true
}

run_box(){
  local ITYPE=$1 base; base=$(base_for "$ITYPE")
  echo ""; echo "############################## $ITYPE ($base) — lanes A/B @ conc $CONC ##############################"
  NEMOTRON_EC2_ITYPE=$ITYPE "$PY" $E/ec2_up.py || { echo "up FAILED"; return 1; }
  local IP; IP=$("$PY" -c "import json;print(json.load(open('$E/.instance.json'))['ip'])"); echo "IP=$IP"
  bash $E/ec2_push.sh || { echo "push FAILED"; "$PY" $E/ec2_down.py; return 1; }
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && PYVER=${PYVER:-3.11} nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started"
  local ok=0; for _ in $(seq 1 80); do sleep 15; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi DONE ~/nemotron/bootstrap.log' 2>/dev/null && { ok=1; echo "bootstrap DONE"; break; }; done
  [ $ok != 1 ] && { echo "bootstrap TIMEOUT"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -20 ~/nemotron/bootstrap.log'; "$PY" $E/ec2_down.py; return 1; }

  for L in $LANES_SET; do
    local tag="prod_${base}_c${CONC}_lanes${L}"
    echo ""; echo "===== $base LANES=$L -> $tag ====="
    ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && CUDAGRAPH_MAX_B=$CUDAGRAPH_MAX_B LANES=$L \
        FINALIZE_PROFILE=1 BATCH_FINALIZE=1 BARRIER_DRAIN=1 CUDAGRAPH_FINALIZE=1 FAULTHANDLER=1 \
        bash start_prod_server.sh > srv_prod.log 2>&1" & local SSHSRV=$!
    local sok=0; for _ in $(seq 1 100); do sleep 4; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -q "ASR server listening" ~/nemotron/srv_prod.log' 2>/dev/null && { sok=1; break; }; done
    if [ $sok != 1 ]; then echo "server FAILED (LANES=$L)"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -30 ~/nemotron/srv_prod.log' 2>/dev/null; kill $SSHSRV 2>/dev/null; continue; fi
    ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -E "MODEL_LANES|encoder_cuda_graph_enabled=|manager_captured|finalize.*captured" ~/nemotron/srv_prod.log | head' 2>/dev/null
    echo "--- client full-1000 @ conc $CONC (TTFS INCLUDES WAN to $REGION) ---"
    run_client "$IP" "$tag" "$OUT/${tag}.clientlog"
    ssh -i "$KEY" $SSHO ubuntu@"$IP" "grep finalize_profile_record ~/nemotron/srv_prod.log" > "$OUT/${tag}.records" 2>/dev/null
    ssh -i "$KEY" $SSHO ubuntu@"$IP" "cat ~/nemotron/srv_prod.log" > "$OUT/${tag}.srvlog" 2>/dev/null
    echo "  pulled $(wc -l < "$OUT/${tag}.records" 2>/dev/null || echo 0) finalize records"
    ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -f 'server.py --model'" 2>/dev/null || true
    kill $SSHSRV 2>/dev/null || true; sleep 4
  done
  "$PY" $E/ec2_down.py; echo "=== $ITYPE done + terminated ==="
}

for ITYPE in $ITYPES; do run_box "$ITYPE" || echo "[$ITYPE] FAILED — continuing"; done

echo ""; echo "================= LANES A/B SUMMARY (client TTFS = speech-end->final, incl WAN) ================="
printf "%-28s %8s %8s %8s | %s\n" "tag" "p50" "p95" "p99" "server-finalize p50/p95"
for f in "$OUT"/*.clientlog; do
  [ -f "$f" ] || continue
  t=$(basename "$f" .clientlog)
  ttfb=$(grep -E "^TTFB \(speech-end" "$f" | tail -1)
  p50=$(echo "$ttfb" | grep -oE 'p50=[0-9.]+' | cut -d= -f2)
  p95=$(echo "$ttfb" | grep -oE 'p95=[0-9.]+' | cut -d= -f2)
  p99=$(echo "$ttfb" | grep -oE 'p99=[0-9.]+' | cut -d= -f2)
  srv=$(grep -E "^server finalize" "$f" | tail -1 | sed -E 's/.*ms\): //')
  printf "%-28s %8s %8s %8s | %s\n" "$t" "${p50:-?}" "${p95:-?}" "${p99:-?}" "${srv:-?}"
done
echo "(leaderboard refs: Deepgram 247/298  Soniox 249/281 ; current L40S lanes2 baseline 246/279)"
echo "logs: $OUT/"
