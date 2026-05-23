#!/usr/bin/env bash
# Kernel-level finalize profile (RTX5090): is the per-finalize compute LAUNCH-BOUND (many small kernel launches,
# CPU-launch >> CUDA time -> a CUDA graph / fused kernel would collapse it) or COMPUTE-bound? And are there
# H2D/D2H transfers or syncs that stall the pipeline? NEMOTRON_FINALIZE_TORCH_PROFILE=N wraps the first N finalize
# model calls in torch.profiler (observation-only, byte-exact). conc-1 -> each finalize is isolated (no concurrent
# steady to pollute the trace). The finalize encoder is eager (keep_all_outputs bypasses the steady cudagraph) +
# decoder is eager (use_cuda_graph_decoder=False) -> the profile shows whether that is launch-bound.
set -uo pipefail
cd /home/khkramer/src/nemotron-january-2026
ROOT=$(pwd)
SRV_VENV=/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python
CLI_VENV=$ROOT/stt-benchmark/.venv/bin/python
AUDIO=$ROOT/proj-2026-05-20-modal-cost/loadgen_audio
LOG=$ROOT/proj-2026-05-22-1353/local_kernel_profile_srv.log
MODEL=nvidia/nemotron-speech-streaming-en-0.6b

pkill -f "server.py --model" 2>/dev/null; sleep 2
echo "=== start local server (prod config + NEMOTRON_FINALIZE_TORCH_PROFILE=5) ==="
( cd src/nemotron_speech && env -u LD_LIBRARY_PATH \
    NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 \
    NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8 \
    NEMOTRON_MODEL_LANES=2 NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8 \
    NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_BATCH_FINALIZE_PREPROC=1 NEMOTRON_BATCH_BARRIER_DRAIN=1 \
    NEMOTRON_FINALIZE_PROFILE=1 NEMOTRON_FINALIZE_TORCH_PROFILE=5 \
    "$SRV_VENV" server.py --model "$MODEL" --host 127.0.0.1 --port 8080 --right-context 1 ) > "$LOG" 2>&1 &
trap 'pkill -f "server.py --model" 2>/dev/null' EXIT
ok=0; for _ in $(seq 1 80); do sleep 3; (exec 3<>/dev/tcp/127.0.0.1/8080) 2>/dev/null && { exec 3>&-; ok=1; break; }; done
[ $ok != 1 ] && { echo "SERVER FAILED"; tail -30 "$LOG"; exit 1; }
echo "server ready"; sleep 2

echo "=== drive conc-1 (isolated finalizes; profiling first 5) ==="
LOADGEN_ALL_CLIPS=1 "$CLI_VENV" ec2-bench/ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep 1 --rounds 8 \
    --audio-dir "$AUDIO" 2>&1 | grep -E "TTFSp50|^ +1 " | tail -2
sleep 1; pkill -f "server.py --model" 2>/dev/null; sleep 2

echo ""; echo "########## KERNEL PROFILE SUMMARIES (all profiled finalizes) ##########"
grep "finalize_torch_profile #" "$LOG"
echo ""; echo "########## REPRESENTATIVE PROFILE (#3 — past warmup): summary + tables ##########"
awk '/finalize_torch_profile #3 /{p=1} p{print} /finalize_torch_profile #4 /{exit}' "$LOG" | head -70
echo ""; echo "=== full log: $LOG ==="