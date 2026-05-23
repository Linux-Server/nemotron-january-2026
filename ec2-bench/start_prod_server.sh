#!/usr/bin/env bash
# On-box: launch ONE production-config ASR server bound 0.0.0.0 for the client-side WAN benchmark.
# cudagraph ON, lanes=2, silence0_warm200, rc1, scheduler+batching. CUDAGRAPH_MAX_B from env (default 8).
# Run in the FOREGROUND of a (backgrounded) ssh so the server lives with the connection; `exec` so the
# python process IS this script (clean pkill + signal handling). cuDNN: drop the DLAMI LD_LIBRARY_PATH.
set -uo pipefail
cd "$HOME/nemotron"
pkill -f "server.py --model" 2>/dev/null || true
sleep 3
ENV=(NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200
     NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32 NEMOTRON_BATCH_MAX_WAIT_MS=8
     "NEMOTRON_MODEL_LANES=${LANES:-2}" NEMOTRON_ENCODER_CUDAGRAPH=1
     "NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=${CUDAGRAPH_MAX_B:-8}" "HF_HOME=$HOME/hf")
# Step-1 probe + Step-7 config passthroughs (default off):
[ "${FINALIZE_PROFILE:-0}" = 1 ] && ENV+=(NEMOTRON_FINALIZE_PROFILE=1)
[ "${BATCH_FINALIZE:-0}" = 1 ] && ENV+=(NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_BATCH_FINALIZE_PREPROC=1)
[ "${BARRIER_DRAIN:-0}" = 1 ] && ENV+=(NEMOTRON_BATCH_BARRIER_DRAIN=1)   # match prod (launch_multiproc) for the steady path
exec env -u LD_LIBRARY_PATH "${ENV[@]}" \
  "$HOME/nemo-venv/bin/python" server.py --model nvidia/nemotron-speech-streaming-en-0.6b \
  --host 0.0.0.0 --port 8080 --right-context 1
