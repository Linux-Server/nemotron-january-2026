#!/usr/bin/env bash
# Client-side WAN latency benchmark: the LOCAL machine is the client; it streams the full 1000-sample
# stt-benchmark over the internet to a fresh, production-configured EC2 box in us-west-2 (cudagraph ON,
# lanes=2, silence0_warm200, rc1, scheduler+batching). This is the apples-to-apples setup vs the first-party
# streaming APIs in stt-benchmark/README.md (all of which include client->endpoint network latency).
# Runs each ITYPE sequentially (clean, no client contention), full 1000 @ conc CONC, ALWAYS terminates.
# Run as a FILE. All pkills are remote (over ssh); the local harness is run_full1000_conc12.py (not server.py).
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
E=ec2-bench
PY=stt-benchmark/.venv/bin/python                 # local: harness + boto3 (has websockets)
KEY=$E/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=15"
PROFILE=AWSAdministratorAccess-419599258555
REGION=us-west-2
MYIP="${MYIP:-$(curl -s https://checkip.amazonaws.com)}"
CONC="${CONC:-10}"
ITYPES="${ITYPES:-g6.4xlarge g6e.8xlarge}"
MODEL=nvidia/nemotron-speech-streaming-en-0.6b
CUDAGRAPH_MAX_B="${CUDAGRAPH_MAX_B:-8}"

tag_for(){ case "$1" in g6.4xlarge|g6.*) echo "prod_l4_full1000_c${CONC}";; g6e.*) echo "prod_l40s_full1000_c${CONC}";; *) echo "prod_$(echo "$1"|tr . _)_c${CONC}";; esac; }

# Open the bench SG to MY_IP:8080 once (the box binds 0.0.0.0; restrict to MY_IP — server has no auth).
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

run_one(){
  local ITYPE=$1 tag; tag=$(tag_for "$ITYPE")
  echo ""; echo "############################## $ITYPE -> $tag ##############################"
  NEMOTRON_EC2_ITYPE=$ITYPE "$PY" $E/ec2_up.py || { echo "up FAILED"; return 1; }
  local IP; IP=$("$PY" -c "import json;print(json.load(open('$E/.instance.json'))['ip'])")
  echo "IP=$IP"
  bash $E/ec2_push.sh || { echo "push FAILED"; "$PY" $E/ec2_down.py; return 1; }
  ssh -i "$KEY" $SSHO ubuntu@"$IP" 'cd ~/nemotron && nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started'
  local ok=0; for _ in $(seq 1 80); do sleep 15; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -qi DONE ~/nemotron/bootstrap.log' 2>/dev/null && { ok=1; echo "bootstrap DONE"; break; }; done
  [ $ok != 1 ] && { echo "bootstrap TIMEOUT"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -20 ~/nemotron/bootstrap.log'; "$PY" $E/ec2_down.py; return 1; }
  # ONE production server, bound 0.0.0.0 (external client), cudagraph ON + lanes=2. Run in the FOREGROUND of a
  # BACKGROUNDED ssh so the server lives with the connection (one-shot `ssh "... &"` gets torn down at close).
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "cd ~/nemotron && CUDAGRAPH_MAX_B=$CUDAGRAPH_MAX_B FINALIZE_PROFILE=${FINALIZE_PROFILE:-0} BATCH_FINALIZE=${BATCH_FINALIZE:-0} BARRIER_DRAIN=${BARRIER_DRAIN:-0} FAULTHANDLER=${FAULTHANDLER:-0} bash start_prod_server.sh > srv_prod.log 2>&1" &
  local SSHSRV=$!
  local sok=0; for _ in $(seq 1 90); do sleep 4; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -q "ASR server listening" ~/nemotron/srv_prod.log' 2>/dev/null && { sok=1; break; }; done
  if [ $sok != 1 ]; then echo "server FAILED to start"; ssh -i "$KEY" $SSHO ubuntu@"$IP" 'tail -30 ~/nemotron/srv_prod.log' 2>/dev/null; kill $SSHSRV 2>/dev/null; "$PY" $E/ec2_down.py; return 1; fi
  ssh -i "$KEY" $SSHO ubuntu@"$IP" 'grep -E "encoder_cuda_graph_enabled=|manager_captured" ~/nemotron/srv_prod.log | head'
  echo "=== LOCAL harness (client) -> ws://$IP:8080 | full 1000 @ conc $CONC (latency INCLUDES WAN to $REGION) ==="
  # WATCHDOG: if the server log goes silent > SILENCE_S (default 180s) the server hung -> kill the client so the
  # run tears down + terminates instead of idling for hours (the b4wp7pcn0 stall burned ~1.5h with no watchdog).
  ( SILENCE_S="${SILENCE_S:-180}"; sleep 60
    while pgrep -f run_full1000_conc12 >/dev/null 2>&1; do
      age=$(ssh -i "$KEY" $SSHO ubuntu@"$IP" 'echo $(( $(date +%s) - $(stat -c %Y ~/nemotron/srv_prod.log 2>/dev/null || echo 0) ))' 2>/dev/null)
      if [ -n "$age" ] && [ "$age" -gt "$SILENCE_S" ] 2>/dev/null; then
        echo "[watchdog] server log silent ${age}s -> HANG; dumping stacks (USR1) + killing client"
        ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -USR1 -f 'server.py --model'" 2>/dev/null || true
        pkill -f run_full1000_conc12 2>/dev/null || true; break
      fi
      sleep 30
    done ) &
  local WD=$!
  timeout 1500 "$PY" proj-2026-05-19-eou-endpointing/run_full1000_conc12.py --url "ws://$IP:8080" --model-tag "$tag" --concurrency "$CONC" 2>&1 | tail -6
  kill $WD 2>/dev/null || true
  if [ "${FINALIZE_PROFILE:-0}" = 1 ]; then   # pull server-side finalize decomposition for this run
    ssh -i "$KEY" $SSHO ubuntu@"$IP" "grep finalize_profile_record ~/nemotron/srv_prod.log" > "$E/leaderboard_decomp_${tag}.records" 2>/dev/null
    echo "  pulled $(wc -l < "$E/leaderboard_decomp_${tag}.records" 2>/dev/null || echo 0) finalize_profile records -> $E/leaderboard_decomp_${tag}.records"
  fi
  ssh -i "$KEY" $SSHO ubuntu@"$IP" "pkill -f 'server.py --model'" 2>/dev/null || true
  kill $SSHSRV 2>/dev/null || true
  "$PY" $E/ec2_down.py
  echo "=== $ITYPE done + terminated (tag=$tag) ==="
}

for ITYPE in $ITYPES; do run_one "$ITYPE" || echo "[$ITYPE] FAILED — continuing"; done
echo ""; echo "=== ALL DONE — tags in results.db: $(for i in $ITYPES; do tag_for "$i"; done | tr '\n' ' ') ==="
