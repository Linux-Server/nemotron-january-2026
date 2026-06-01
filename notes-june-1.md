# June 1 Notes

## GB10 C++ STT Benchmark

Configuration:
- Server: C++ `ws_server`, Docker image `nemotron-aoti:gb10-aarch64`, port `8091`.
- Artifacts: full 32-bucket local GB10 artifact set under `runtime/artifacts_gb10` plus `steady_b_artifacts_gb10`.
- Server env: `NEMOTRON_FINALIZE_SILENCE_MS=5000`, `NEMOTRON_WS_LANES=8`, `NEMOTRON_WS_FINALIZE_RUNNERS=2`, `NEMOTRON_DENSITY_STEADY_RUNNERS=1`, `NEMOTRON_WS_SCHEDULER=1`, admission cap `1000`.
- Benchmark: standard `stt-benchmark`, `--services nemotron_local`, `--limit 1000`, `--concurrency 8`, default realtime pacing and VAD stop behavior.
- Benchmark tag: `gb10_cpp_32bucket_n8_finalize5000_20260601`.

Results:
- Cold start confirmed-to-health: about `487.7s` (`2026-06-01T05:00:29.692080191Z` container start to `2026-06-01T05:08:37.419862199+00:00` health confirmation).
- Samples/errors: `1000 / 0`.
- Benchmark wall time: `26:04`.
- TTFS/TTFB mean: `234.0 ms`.
- TTFS/TTFB P50/P90/P95/P99: `232.3 / 244.7 / 252.0 / 271.5 ms`.
- TTFS/TTFB min/max: `219.5 / 327.4 ms`.
- Semantic WER mean: `2.11%`.
- Semantic WER pooled: `2.06%`.
- Semantic WER median/min/max: `0.00% / 0.00% / 40.54%`.
- Perfect semantic WER: `736/1000`.

Notes and issues:
- `NEMOTRON_FINALIZE_SILENCE_MS=5000` is intentionally set on the server process. With `0`, C++ can emit final on `vad_stop` before Pipecat arms finalization for `reset`, which causes dropped or unarmed final transcripts in this benchmark harness.
- With `5000 ms`, final events are reset-driven and match the Python server behavior.
- The cold-start number is a confirmed-to-health measurement. Exact readiness may be slightly earlier because the first waiter script failed after health due to `python: command not found`.

## Python STT Benchmark: English Checkpoint

Configuration:
- Server: current checkout's `src/nemotron_speech/server.py`, run with `/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python` and `PYTHONPATH=/home/khkramer/src/nemotron-january-2026/src`.
- Model: `nvidia/nemotron-speech-streaming-en-0.6b`, loaded from local Hugging Face cache with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- Server args/env: `--host 0.0.0.0 --port 8091 --right-context 1`, `NEMOTRON_MODEL_NAME=english`, `NEMOTRON_FINALIZE_SILENCE_MS=5000`.
- Benchmark: standard `stt-benchmark`, `--services nemotron_local`, `--limit 1000`, `--concurrency 8`, default realtime pacing and VAD stop behavior.
- Benchmark tag: `python_en_n8_finalize5000_20260601`.

In-progress notes:
- Confirmed cold start: about `10.0s` from process launch (`2026-06-01T14:29:14.668235957+00:00` health wait start) to healthy (`2026-06-01T14:29:24.682729027+00:00`). Server log shows it was listening at `2026-06-01 14:29:19.873`, so exact server readiness was closer to `10.9s` from process launch log at `14:29:08.999`.
- The repo-local `.venv` cannot run the Python server on this machine because it lacks `torch` and `nemo`; the ASR venv from the sibling checkout has `torch 2.11.0+cu130` and NeMo.
- Benchmark emits repeated Pipecat metadata warnings: `STTSettings` has `model, language` as `NOT_GIVEN`, and `ttfs_p99_latency` defaults to `1.0s`. These appear to be harness/service metadata issues, not server failures.
- Harness caveat: `stt-benchmark run --model ...` stores the result tag, but `BenchmarkRunner` currently calls `create_stt_service(...)` without passing that model argument. I am using `NEMOTRON_LOCAL_MODEL_NAME` for server route validation instead.
- First attempt was stopped at `485/1000` after `0:12:45` because the server was launched without the optimized continuous/batch scheduler. Server startup flags confirmed `scheduler_enabled=False batch_requested=False batch_enabled=False`. No partial rows were written to SQLite before interrupt.
- Second attempt restarted with the optimized continuous/batch scheduler:
  - Env: `NEMOTRON_CONTINUOUS=1`, `NEMOTRON_SCHEDULER_B1=1`, `NEMOTRON_BATCH_SCHED=1`, `NEMOTRON_BATCH_MAX_SIZE=32`, `NEMOTRON_BATCH_MAX_WAIT_MS=8`, `NEMOTRON_WARMUP_MS=200`, `NEMOTRON_DECODING=greedy`, `NEMOTRON_FINALIZE_SILENCE_MS=5000`.
  - Server startup flags confirmed `scheduler_enabled=True`, `batch_enabled=True`, `decoder_strategy=greedy_batch`, `batch_max_size=32`.
  - Issue found: in this Python continuous/batch path, `NEMOTRON_FINALIZE_SILENCE_MS=5000` is honored as a real 5 second finalization delay. `/stats` during the run showed `vad_stop_to_finalize_start_ms` P50 about `5022 ms` and `vad_stop_to_sent_ms` P50 about `5046 ms`. That is not the reset-driven behavior seen in the C++ benchmark with the same env.
  - This run was stopped at `320/1000` after `0:11:33`; no final benchmark rows were written before interrupt.

## C++ Standard Benchmark After `vad_stop` No-Op

Configuration:
- Server: C++ `ws_server`, Docker image `nemotron-aoti:gb10-aarch64`, port `8091`.
- Artifacts: local GB10 artifacts under `runtime/artifacts_gb10` and `runtime/steady_b_artifacts_gb10`.
- Server env: `NEMOTRON_FINALIZE_SILENCE_MS=0`, `NEMOTRON_WS_LANES=8`, `NEMOTRON_WS_FINALIZE_RUNNERS=2`, `NEMOTRON_DENSITY_STEADY_RUNNERS=1`, `NEMOTRON_WS_SCHEDULER=1`, admission cap `1000`.
- Benchmark: standard `stt-benchmark`, `--services nemotron_local`, `--limit 1000`, `--concurrency 8`, default realtime pacing and VAD behavior.
- Benchmark tag: `cpp_std_n8_20260601`.

Code/protocol change:
- C++ `ws_server` now treats text control `{"type":"vad_stop"}` as side-effect free.
- Current clients finalize turns with `{"type":"reset","finalize":true}`. This matches the Python server and the YC starter-kit Pipecat service.
- The C++ lower-level `vad_stop` runtime hook remains available for a future server-side endpointing optimization, but no current client depends on it.

Results:
- Samples/errors: `1000 / 0`.
- Benchmark wall time: `26:04`.
- TTFS/TTFB mean: `234.0 ms`.
- TTFS/TTFB P50/P90/P95/P99: `232.8 / 244.4 / 251.6 / 268.7 ms`.
- TTFS/TTFB min/max: `219.4 / 293.3 ms`.
- Server-side reset/final path remained much faster than benchmark TTFB: observed `vad_stop_to_sent_ms` was about P50 `30 ms`, P95 `40 ms`, P99 `46 ms` during the full run.
- Semantic WER mean: `2.11%`.
- Semantic WER pooled: `2.05%`.
- Semantic WER max: `45.95%`.

Startup issue:
- Scheduler-enabled C++ startup spends a long time in batched steady manifest verification and scheduler package loading on GB10.
- Root cause: startup validates multi-GB batched steady package/EP/shared-weight files and then loads each scheduler bucket.
- Current behavior: manifest validation keeps optional byte-size checks as a fast precheck, but still verifies the recorded SHA-256 digests for integrity before accepting the artifacts. The server reaches health and completes scheduler warmup, but cold start can take several minutes.

YC starter-kit smoke:
- Verified against `../yc-voice-agents-hackathon/server/nvidia_stt.py` using the actual `NVidiaWebSocketSTTService`.
- The service sent audio, then `{"type":"reset","finalize":true}`, and received a finalized `TranscriptionFrame`.
- Smoke transcript: `This is Samantha Lee with a knight to remember.`
- This confirms the C++ `vad_stop` no-op does not break the YC client path, because that service is reset-driven.
- Current-code C++ `ws_server --selftest-and-exit` in the GB10 Docker image also passed all websocket/server smoke cases on `2026-06-01`.
  - Summary: `SELFTEST_SUMMARY pass=true passed=12 total=12`.
  - Relevant lifecycle case: `SELFTEST 8 PASS - Bound port health + stats + WS lifecycle PCM+vad_stop+reset`, with `final=true`, `timing=true`, and clean websocket close.

## Python Standard Benchmark Attempt

Configuration:
- Server: `src/nemotron_speech/server.py`, English checkpoint `nvidia/nemotron-speech-streaming-en-0.6b`.
- Server env for the fresh attempts: `NEMOTRON_FINALIZE_SILENCE_MS=0`, `NEMOTRON_CONTINUOUS=1`, `NEMOTRON_SCHEDULER_B1=1`, `NEMOTRON_BATCH_SCHED=1`, `NEMOTRON_BATCH_MAX_SIZE=32`, `NEMOTRON_BATCH_MAX_WAIT_MS=8`, `NEMOTRON_MODEL_LANES=2`, `NEMOTRON_BATCH_BARRIER_DRAIN=1`, `NEMOTRON_BATCH_FINALIZE=1`, `NEMOTRON_ENCODER_CUDAGRAPH=1`, `NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8`, `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1`, `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1`, `NEMOTRON_SYNC_COMPRESS=1`, `NEMOTRON_FINALIZE_PRIORITY=1`.
- Intended benchmark: standard `stt-benchmark`, `--services nemotron_local`, `--limit 1000`, `--concurrency 8`, default realtime pacing and VAD behavior.

Attempts and issues:
- Reusing the old Python server on port `8081` was not valid: it was a six-day-old process and N=8 quickly showed websocket keepalive timeouts, missing transcripts, 10+ second TTFB, and no final SQLite rows.
- A fresh Python server on port `8092` while the C++ container was still resident had only about `4 GB` free at startup and also failed the N=8 smoke.
- After stopping the C++ container and restarting Python alone, startup had about `47.6 GB` free, but the N=8 smoke still did not pass cleanly.
- Python `/stats` during the failed smoke showed emitted finals were fast when they happened (`vad_stop_to_sent_ms` P50 about `33 ms`, P95 about `109 ms`), but the server signal/admission queue backed up badly (`queued_events` around `1900`, `oldest_ready_age_ms` several seconds). The benchmark then saw silence timeouts and very large TTFB values such as `11s`, `18s`, and `23s`.
- No full Python N=8 result was recorded because the smoke did not establish that the Python server was healthy under this exact N=8 benchmark configuration.
- Sanity check at `--concurrency 1 --limit 16` passed cleanly with tag `python_std_n1_l16_20260601`:
  - Samples/errors: `16 / 0`.
  - TTFS/TTFB mean: `225.0 ms`.
  - TTFS/TTFB min/max: `223.5 / 227.0 ms`.
  - Console P50/P90/P95/P99: `225 / 226 / 226 / 227 ms`.
  - Python `/stats` after the run had no backlog (`queued_events=0`, `ready_count=0`) and server-side `vad_stop_to_sent_ms` P50 about `23.7 ms`.
  - Conclusion: the Python server and benchmark protocol work at low concurrency; the open issue is Python queueing/backpressure under higher benchmark concurrency.
- Concurrency sanity sweep:
  - `python_std_n2_l16_20260601`: `16 / 0` rows, mean TTFS/TTFB `225.5 ms`, min/max `223.1 / 235.1 ms`.
  - `python_std_n4_l16_20260601`: `16 / 0` rows, mean TTFS/TTFB `228.9 ms`, min/max `224.7 / 237.9 ms`, but the run logs already showed silence timeouts and websocket keepalive failures near the end of the run and the server queue remained backed up afterward.
  - `python_std_n8_l16_20260601`: no valid completed rows; the run was stopped after the server queue was already stale from the N=4 failure.
- Platform bottleneck check on DGX Spark / GB10:
  - Restarted the Python server cleanly with the same optimized English config (`NEMOTRON_CONTINUOUS=1`, scheduler/batching enabled, `NEMOTRON_MODEL_LANES=2`, CUDA graphs enabled).
  - Startup memory was not the immediate limiter: server logged `free_bytes=47813566464`, `effective_max=32`.
  - Ran a monitored `--limit 32 --concurrency 4` probe (`python_memcheck_n4_l32_20260601`) with `nvidia-smi dmon -s pucm`.
  - Failure pattern: the first few finals returned normally at about `225-229 ms`, then later samples hit benchmark silence/TTFB timeouts and websocket keepalive failures.
  - GPU telemetry during the failing window did not show GPU saturation: `dmon` summary over 209 samples was avg power `10.6 W`, max power `20.0 W`, avg SM `1.9%`, max SM `61%`, avg/max memory util `0.0% / 0.0%`.
  - Server `/stats` after stopping the probe showed only `4` emitted finalize samples, server-side `vad_stop_to_sent_ms` P50/P95 `27.3 / 34.1 ms`, but signal backlog `queued_events=1005`, `ready_count=1`, `backlog_count=1006`, `oldest_ready_age_ms` about `25.7s`.
  - After the benchmark was stopped, the Python server kept burning about `125%` CPU while the queue stayed backed up; `pidstat` showed roughly `98% user + 27% system` CPU and no page faults.
  - `perf` sampling during the stuck drain showed heavy Python/GIL/PyTorch-thread activity, not GPU kernels.
  - Interpretation: this evidence does not support a GPU memory-bandwidth bottleneck. It points to a Python host-side bottleneck before GPU submission, likely event-loop/GIL/CPU memory-copy work in the websocket/audio intake or scheduler path. Host memory bandwidth may still be involved, but the limiter is not visible as GPU memory-controller saturation.

## Python DGX Spark One-Lane Baseline

Configuration:
- Server: `src/nemotron_speech/server.py`, English checkpoint `nvidia/nemotron-speech-streaming-en-0.6b`, port `8092`.
- Server env: same optimized continuous/batch settings as the failing Python attempts, but with `NEMOTRON_MODEL_LANES=1` instead of `2`.
- Key env: `NEMOTRON_FINALIZE_SILENCE_MS=0`, `NEMOTRON_CONTINUOUS=1`, `NEMOTRON_SCHEDULER_B1=1`, `NEMOTRON_BATCH_SCHED=1`, `NEMOTRON_BATCH_MAX_SIZE=32`, `NEMOTRON_BATCH_MAX_WAIT_MS=8`, `NEMOTRON_MODEL_LANES=1`, `NEMOTRON_BATCH_BARRIER_DRAIN=1`, `NEMOTRON_BATCH_FINALIZE=1`, `NEMOTRON_ENCODER_CUDAGRAPH=1`, `NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8`, `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1`, `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1`, `NEMOTRON_SYNC_COMPRESS=1`, `NEMOTRON_FINALIZE_PRIORITY=1`.

Short probes:
- `python_lanes1_n4_l32_20260601`: `32 / 0` rows, mean TTFS/TTFB `239.7 ms`, median `226 ms`, min/max `223.0 / 294.8 ms`, P90/P95/P99 `268 / 281 / 294 ms`. Server queue drained cleanly.
- `python_lanes1_n8_l32_20260601`: `32 / 0` rows, mean TTFS/TTFB `258.9 ms`, median `241 ms`, min/max `223.5 / 363.9 ms`, P90/P95/P99 `335 / 342 / 359 ms`. Server queue drained cleanly.
- GPU telemetry for the one-lane probes showed real GPU use instead of the lane=2 idle/backlog pattern: N=4 avg/max SM `36.1% / 82%`; N=8 avg/max SM `52.4% / 86%`.

Full N=8 run:
- Benchmark tag: `python_lanes1_n8_20260601`.
- Benchmark: standard `stt-benchmark`, `--services nemotron_local`, `--limit 1000`, `--concurrency 8`, default realtime pacing and VAD behavior.
- Samples/errors: `1000 / 0`.
- Benchmark wall time: `26:06`.
- TTFS/TTFB mean: `278.4 ms`.
- TTFS/TTFB P50/P90/P95/P99: `271 / 333 / 363 / 443 ms`.
- TTFS/TTFB min/max: `222 / 608 ms`.
- Server-side `/stats` stayed healthy during the run: no aged ready backlog; final post-run signal was `queued_events=0`, `ready_count=0`, `backlog_count=0`.
- Server emitted `2529` final segments and suppressed `1117` during the run; multiple final segments per benchmark sample are expected for this service path.

Semantic WER:
- Completed after Anthropic credits were restored.
- Samples: `1000 / 1000`.
- Semantic WER mean: `2.31%`.
- Semantic WER pooled: `2.29%`.
- Semantic WER median/min/max: `0.00% / 0.00% / 66.67%`.

Interpretation:
- `NEMOTRON_MODEL_LANES=1` is the current DGX Spark Python-server baseline. It completes the standard N=8 benchmark cleanly, but N=2 is the better default sanity-test concurrency on this platform.
- `NEMOTRON_MODEL_LANES=2` is not a good DGX Spark default: it creates host-side backlog before GPU saturation and can fail the benchmark even at moderate concurrency.
- The top-level README now documents the DGX Spark Python ASR config so future clean-checkout bring-up uses the one-lane baseline.
