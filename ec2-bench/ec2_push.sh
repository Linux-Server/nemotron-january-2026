#!/usr/bin/env bash
# Push the benchmark bundle FROM THIS DIRECTORY to the EC2 box (no GitHub).
# Default: COMMITTED (HEAD) server.py + batch_primitives.py (the baseline).
#   --working : push the working-tree versions instead (for the lanes/graphs runs).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
IP=$(python3 -c "import json;print(json.load(open('ec2-bench/.instance.json'))['ip'])")
KEY=ec2-bench/nemotron-bench-key.pem
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

if [ "${1:-}" = "--working" ]; then
  cp src/nemotron_speech/server.py src/nemotron_speech/batch_primitives.py \
     src/nemotron_speech/cudagraph_encoder.py "$STAGE/"
  echo "[push] WORKING-TREE server.py + batch_primitives.py + cudagraph_encoder.py"
else
  git show HEAD:src/nemotron_speech/server.py > "$STAGE/server.py"
  git show HEAD:src/nemotron_speech/batch_primitives.py > "$STAGE/batch_primitives.py"
  git show HEAD:src/nemotron_speech/cudagraph_encoder.py > "$STAGE/cudagraph_encoder.py"
  echo "[push] COMMITTED (HEAD) server.py + batch_primitives.py + cudagraph_encoder.py"
fi
cp ec2-bench/ec2_loadgen.py ec2-bench/run_bench.sh ec2-bench/run_lanes.sh ec2-bench/run_multiproc.sh \
   ec2-bench/run_l4_ttfs_sweep.sh ec2-bench/bootstrap.sh "$STAGE/"
cp -r proj-2026-05-20-modal-cost/loadgen_audio "$STAGE/loadgen_audio"
rsync -az -e "ssh -i $KEY $SSHO" "$STAGE"/ ubuntu@"$IP":/home/ubuntu/nemotron/
echo "[push] done -> ubuntu@$IP:~/nemotron/  ($(ls "$STAGE"/loadgen_audio | wc -l) audio clips)"
