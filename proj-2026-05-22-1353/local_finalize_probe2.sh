#!/usr/bin/env bash
# Follow-up local probe closing the two gaps from probe1: LENGTH (conc-1 used 1 clip) + B>1 BATCH (B=1 always).
#  TEST A (LENGTH): LOADGEN_ALL_CLIPS=1, conc-1, rotate all 25 clips across rounds -> isolated, varied length.
#  TEST B (BATCH):  LOADGEN_JITTER_MS=0 + a 1-clip dir, conc-12 -> in-phase same-clip -> coincident finals -> B>1.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
ROOT=$(pwd)
SRV_VENV=/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python
CLI_VENV=$ROOT/stt-benchmark/.venv/bin/python
AUDIO=$ROOT/proj-2026-05-20-modal-cost/loadgen_audio
LOG=/tmp/local_probe2_srv.log
REC=$ROOT/proj-2026-05-22-1353/local_probe2_records.txt
MODEL=nvidia/nemotron-speech-streaming-en-0.6b

echo "=== clip duration range (s) ==="
"$CLI_VENV" - "$AUDIO" <<'PY'
import sys, glob, os
ds=sorted(os.path.getsize(f)/2/16000 for f in glob.glob(sys.argv[1]+"/*.pcm"))
print(f"  n={len(ds)} min={ds[0]:.1f}s med={ds[len(ds)//2]:.1f}s max={ds[-1]:.1f}s")
PY

pkill -f "server.py --model" 2>/dev/null; sleep 2
echo "=== start local server (prod config + last-stage instrumentation) ==="
( cd src/nemotron_speech && env -u LD_LIBRARY_PATH \
    NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 \
    NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8 \
    NEMOTRON_MODEL_LANES=2 NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 \
    NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_BATCH_FINALIZE_PREPROC=1 NEMOTRON_FINALIZE_PROFILE=1 \
    "$SRV_VENV" server.py --model "$MODEL" --host 127.0.0.1 --port 8080 --right-context 1 ) > "$LOG" 2>&1 &
trap 'echo "[trap] stop server"; pkill -f "server.py --model" 2>/dev/null; rm -rf "${BURSTDIR:-/nonexistent}"' EXIT
ok=0; for _ in $(seq 1 80); do sleep 3; (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null && { exec 3>&-; ok=1; break; }; done
[ $ok != 1 ] && { echo "SERVER FAILED"; tail -30 "$LOG"; exit 1; }
echo "server ready"; sleep 2

echo ""; echo "=== TEST A (LENGTH): conc-1, all clips rotated, isolated ==="
LOADGEN_ALL_CLIPS=1 "$CLI_VENV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep 1 --rounds 50 \
    --audio-dir "$AUDIO" 2>&1 | grep -E "TTFSp50|^ +1 |distinct" | tail -3
grep "finalize_profile_record" "$LOG" > "$REC.len"; NA=$(wc -l < "$REC.len")

echo ""; echo "=== TEST B (BATCH): conc-12, in-phase (JITTER=0), SAME clip -> B>1 ==="
BURSTDIR=$(mktemp -d); cp "$(ls "$AUDIO"/*.pcm | head -1)" "$BURSTDIR/"
LOADGEN_JITTER_MS=0 "$CLI_VENV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep 12 --rounds 12 \
    --audio-dir "$BURSTDIR" 2>&1 | grep -E "TTFSp50|^ +12 " | tail -3
grep "finalize_profile_record" "$LOG" | tail -n +$((NA + 1)) > "$REC.burst"

echo ""; echo "=== collected ==="
echo "LENGTH (conc-1): $(wc -l < "$REC.len") -> $REC.len"
echo "BATCH (in-phase): $(wc -l < "$REC.burst") -> $REC.burst"
echo "=== done; trap stops server ==="