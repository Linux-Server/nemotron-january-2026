#!/usr/bin/env bash
# A/B: does NEMOTRON_BATCH_BARRIER_DRAIN (which PROD/launch_multiproc.sh sets, but my probes + the WAN bench +
# the K=4 gate did NOT) batch coincident finalizes and collapse the queue_wait serialization? Same box, in-phase
# conc-12 same-clip burst (coincident finals), BARRIER_DRAIN=0 then =1, prod config + FINALIZE_PROFILE.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
ROOT=$(pwd)
SRV_VENV=/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python
CLI_VENV=$ROOT/stt-benchmark/.venv/bin/python
AUDIO=$ROOT/proj-2026-05-20-modal-cost/loadgen_audio
LOG=/tmp/local_probe3_srv.log
REC=$ROOT/proj-2026-05-22-1353/local_probe3_records.txt
MODEL=nvidia/nemotron-speech-streaming-en-0.6b
BURSTDIR=$(mktemp -d); cp "$(ls "$AUDIO"/*.pcm | head -1)" "$BURSTDIR/"
trap 'pkill -f "server.py --model" 2>/dev/null; rm -rf "$BURSTDIR"' EXIT

for BD in 0 1; do
  echo ""; echo "########## BARRIER_DRAIN=$BD (in-phase conc-12 same-clip burst) ##########"
  pkill -f "server.py --model" 2>/dev/null; sleep 2; : > "$LOG"
  ( cd src/nemotron_speech && env -u LD_LIBRARY_PATH \
      NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 \
      NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8 \
      NEMOTRON_MODEL_LANES=2 NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 \
      NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_BATCH_FINALIZE_PREPROC=1 "NEMOTRON_BATCH_BARRIER_DRAIN=$BD" \
      NEMOTRON_FINALIZE_PROFILE=1 \
      "$SRV_VENV" server.py --model "$MODEL" --host 127.0.0.1 --port 8080 --right-context 1 ) > "$LOG" 2>&1 &
  ok=0; for _ in $(seq 1 80); do sleep 3; (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null && { exec 3>&-; ok=1; break; }; done
  [ $ok != 1 ] && { echo "SERVER FAILED (BD=$BD)"; tail -20 "$LOG"; exit 1; }
  sleep 2
  LOADGEN_JITTER_MS=0 "$CLI_VENV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep 12 --rounds 12 \
      --audio-dir "$BURSTDIR" 2>&1 | grep -E "TTFSp50|^ +12 " | tail -2
  grep "finalize_profile_record" "$LOG" > "$REC.bd$BD"
done
pkill -f "server.py --model" 2>/dev/null; sleep 2

echo ""; echo "########## COMPARE: B distribution + queue_wait, BARRIER_DRAIN 0 vs 1 ##########"
"$CLI_VENV" - <<'PY'
import json
from collections import Counter
def load(p):
    rs=[]
    for line in open(p):
        j=line.find('{', line.find('finalize_profile_record'))
        if j>=0:
            try: rs.append(json.loads(line[j:]))
            except: pass
    return rs
def pct(v,q):
    v=sorted(x for x in v if x is not None); return v[min(len(v)-1,int(round(q*(len(v)-1))))] if v else float('nan')
for bd in (0,1):
    rs=load(f'proj-2026-05-22-1353/local_probe3_records.txt.bd{bd}')
    for f in ('queue_wait_ms','vad_stop_to_sent_ms','fork_flush_wall_ms','model_wall_ms'):
        v=[r.get(f) for r in rs if r.get(f) is not None]
        if f=='queue_wait_ms':
            print(f"  BD={bd} (n={len(rs)}) B={dict(sorted(Counter(r.get('B') for r in rs).items()))}")
        print(f"      {f:20} p50={pct(v,.5):7.2f} p95={pct(v,.95):7.2f} max={(max(v) if v else float('nan')):7.2f}")
PY
echo "=== done ==="