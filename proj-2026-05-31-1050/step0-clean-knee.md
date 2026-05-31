# Step 0 — CLEAN-KNEE measurement (RAW DATA ONLY — no GO/STOP verdict)

**Box:** ubuntu@34.214.169.199 — NVIDIA L40S sm_89, g6e.4xlarge (16 vCPU, 124 GiB, 45 GiB free VRAM at idle), driver 580.159.04, CUDA-13 system / cudart-12.9 runtime, torch 2.8.0+cu128.
**Binary:** `~/density/cpp/build_l40s_density/ws_server` (current scheduler runtime, the Step-11 cold-start binary; links libcudart.so.12 only).
**Date:** 2026-05-31.

> This document is RAW DATA for the orchestrator + paired review. It does NOT compute the GO/STOP verdict.

## Server launch (the measured config)
Launched via `~/density/launch_ws_server_l40s.sh`. Env:
`HF_HUB_OFFLINE=1`, `LD_LIBRARY_PATH=<torch/lib>:<cudart-12.9>:<cuda-13/lib64>`,
`NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_ARTIFACT_DIR=~/density/artifacts_sm89`,
`NEMOTRON_WS_SCHEDULER=1 NEMOTRON_WS_BACKGROUND_WARMUP=1`,
`NEMOTRON_DENSITY_BATCH_STEADY=1 NEMOTRON_DENSITY_BATCH_MAX=4 NEMOTRON_DENSITY_BATCH_WINDOW_MS=10 NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS=0`,
`NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP=128 NEMOTRON_WS_LANES=128`.
CLI: `--port 8080 --admission-active-cap 128 --steady-batch-dir ~/density/steady_b_artifacts`.
TS enc_first (default `NEMOTRON_WS_ENC_FIRST_TS=1`; NO AOTI-enc_first flag set). Stale `/tmp/torchinductor_ubuntu` removed before launch.
Readiness confirmed: stdout printed BOTH `ws_server listening on 127.0.0.1:8080` AND
`COLD_START_PHASE phase=background_warm_complete ... warmed_lanes=128 lanes=128` (background warm ~226s; all 128 lanes warmed; GPU returned to 0% util before measurement).

## Telemetry-source mechanism
**HTTP curl to the ws_server port (8080)** — both endpoints are served by the same ws_server HTTP router:
- `GET /scheduler_telemetry` → `counts.{dispatch_cycles,B1,B2,B4,backlog_gt_bmax,dispatcher_exceptions}`, `dispatcher_cpu_pct`, `dispatcher_stream_util_pct`, `dispatcher_cpu_us`, `dispatcher_wall_us`, `dispatcher_stream_run_us`, `queue_depth.{p50,p95,p99}`.
- `GET /stats?last=4096` → `metrics.scheduler_future_wait_ms` / `metrics.scheduler_enqueue_wait_ms` (quantile summaries p50/p95/p99), `samples`.
- `nvidia-smi dmon -s u -d 1` → per-second GPU `sm %` (col 2), reduced to mean/max over the window.

**Counter semantics (important for interpretation):**
- `counts.*` are CUMULATIVE since dispatcher start → reported per-N as the **delta** between a before-window and after-window snapshot.
- `dispatcher_cpu_pct` / `dispatcher_stream_util_pct` (the JSON fields) are LIFETIME-cumulative ratios (×100), so a single short window dilutes them with the server's idle history. The table therefore ALSO reports **`_win` variants** computed from the per-window **deltas** of `dispatcher_cpu_us` / `dispatcher_stream_run_us` over `dispatcher_wall_us` — the faithful per-window dispatcher utilization. (`dispatcher_stream_util_pct` excludes pack/unpack = GPU-work-fraction-on-dispatcher-timeline, NOT dispatcher CPU saturation, per the plan's R1 caveat.)
- `queue_depth` p50/p95/p99 and `scheduler_future_wait_ms`/`scheduler_enqueue_wait_ms` are windowed summaries over the retained ring (queue_depth = dispatcher ring; stats ring = last 2048 finalize samples).

## Loadgen
`~/density/rt_loadgen.py` — realtime MULTIPROCESS pacer (written for this measurement; the single-process
`ec2_loadgen.py` would itself become the bottleneck pacing 64-128 WS conns). It shards N sessions across
`--procs` worker processes, each running the byte-identical proven `ec2_loadgen.py` session logic
(vad_start → stream PCM @ 1x realtime → vad_stop → reset/finalize, recording finalize-TTFS + proc-lag),
looping back-to-back utterances for `--window-s` so N stays continuously active (sustained steady window).
Audio: `~/density/loadgen_audio_smoke/` (24 distinct 16 kHz int16 PCM clips, mean ~9.5 s, cycled to N).
Metrics: ok(completed)/err, finalize-TTFS p50/p95/p99/max, proc-lag (keepup) p50/p95/p99, maxjit (worst per-chunk realtime-pacing lateness; loadgen-health).
**keepup_lag = proc-lag p95** (KEEPUP_LAG_MS=500). Each N: `--procs 32`, `--window-s 45`.

## RESULTS TABLE

All windows: `--procs 32 --window-s 45`. TTFS/lag/maxjit in ms. `disp_cpu%win` / `disp_util%win` =
per-window dispatcher CPU-fraction and GPU-work-fraction-on-dispatcher-timeline (from us-deltas; the
faithful windowed signal). queue_depth ring capacity = 16 (so 16 = ring full). counts are per-window deltas.
NVML = `nvidia-smi dmon -s u` sm% mean/max @1Hz over the window.

### Adaptive bracketing sweep (one row per N; first run at each N)

| N | ok | err | TTFS p50 | TTFS p95 | TTFS p99 | TTFS max | keepup_lag (lag p95) | maxjit | p95<400 | disp_cpu%win | disp_util%win | qdepth p50/p95/p99 | backlog_gt_bmax Δ | disp_exc Δ | dispatch_cycles Δ | B1/B4 Δ | future_wait p50/p95/p99 | enqueue_wait p50/p95/p99 | NVML sm% mean/max |
|---|----|-----|----------|----------|----------|----------|----------------------|--------|---------|--------------|---------------|--------------------|-------------------|-----------|-------------------|---------|--------------------------|--------------------------|-------------------|
| 48 | 245 | 0 | 14 | **19** | 28 | 33 | 39 | 0 | PASS | 54.6 | 53.1 | 16/16/16 | 893 | 0 | n/a | n/a | —/4988/— | —/14235/— | 77.8 / 93 |
| 56 | 280 | 0 | 15 | **23** | 33 | 186 | 44 | 0 | PASS | 73.6 | 71.7 | 16/16/16 | 2144 | 0 | n/a | n/a | —/4949/— | —/13491/— | 83.0 / 94 |
| 64 | 314 | 0 | 147 | **304** | 855 | 1051 | 325 | 0 | PASS¹ | 51.6 | 50.6 | 16/16/16 | 4447 | 0 | 5184 | 517/4667 | 3220/4926/5233 | 6203/9632/10635 | 87.2 / 94 |
| 72 | 328 | 0 | 1195 | 2540 | 3286 | 3822 | 2561 | 0 | FAIL | 57.5 | 56.4 | 16/16/16 | 4578 | 0 | 5114 | 350/4764 | 3395/4977/5294 | 7674/15185/18056 | 88.0 / 94 |
| 80 | 336 | 0 | 2494 | 4390 | 4867 | 5959 | 4409 | 0 | FAIL | 83.0 | 81.4 | 16/16/16 | 4816 | 0 | 5285 | 358/4927 | 3366/4935/5280 | 7095/12808/14578 | 89.9 / 94 |
| 96 | 345 | 0 | 4840 | 8247 | 9656 | 10641 | 8267 | 0 | FAIL | 82.9 | 81.3 | 16/16/16 | 4990 | 0 | 5314 | 223/5091 | 3402/4961/5282 | 7754/15563/18234 | 89.8 / 94 |

¹ N=64 PASSES on this first run (304 ms) but is the cliff edge and FAILS the p95<400 gate on 3 of 4 repeats (482–531 ms) — see noise-band section.

- **Sharp cliff.** TTFS p95 is flat-and-tiny up to N=56 (19→23 ms) then jumps a cliff: 56 → 64 → 72 = ~23 ms → ~300–530 ms → 2540 ms. N=48 and N=56 PASS with enormous margin; N=72/80/96 fail by 6–20×. So the highest N that PASSES the 0-error ∧ p95<400 gate ROBUSTLY (every run) = **N=56** (23 ms p95, huge headroom). N=64 is the boundary; 72 is over.
- The plan's adaptive bracketing was followed: started {64,80,96}; added 72 (knee on a boundary, 88 unneeded since 80 fails); then bracketed DOWN to {48,56,60} after the N=64 repeats showed 64 straddles the gate.
- Across ALL N the dispatcher `queue_depth` is pinned at the ring capacity (16 = ring full) and `future_wait` p95 sits ~4.9–5.0 s — essentially FLAT vs N. What climbs with load is client-observed TTFS, `enqueue_wait` p95 (13.5 → 14.2 → 9.6 → 15.2 → 12.8 → 15.6 s) and the windowed dispatcher CPU fraction (48→56→64→72→80: 54.6 → 73.6 → 51.6/65.8/77.4/77.9 → 57.5 → 83 %). `backlog_gt_bmax` Δ grows monotonically (893 → 2144 → ~4440 → 4578 → 4816 → 4990). NVML sm% creeps 77.8 → 83 → 87 → 88 → 90 % — never near 100%. 0 dispatcher exceptions, 0 loadgen errors, maxjit=0 (loadgen never fell behind realtime → the load driver is NOT the limiter) at every N.

> CAVEAT on `future_wait`/`enqueue_wait`: these stats-ring finalize metrics read in the multi-second range even at N=48/56 where client TTFS is ~20 ms and the system is clearly healthy — i.e. their absolute scale does NOT track client-observed TTFS (consistent with the PLAN's note that `scheduler_future_wait_ms` semantics shifted and these are not "waited-for-GPU-completion" wall times). They are reported RAW as instructed; treat their ABSOLUTE values with caution, their TREND vs N is the usable signal. `dispatch_cycles`/`B1`/`B4` deltas were added to the reducer after the N=48/56 run, hence "n/a" for those two rows (the other N have them).

### Noise-band repeats at N=64 (the boundary)

Four independent N=64 runs (sweep1 + 3 dedicated repeats), identical config, fresh sessions each:

| run | ok | err | TTFS p50 | TTFS p95 | TTFS p99 | TTFS max | keepup_lag (lag p95) | p95<400 gate | disp_cpu%win | disp_util%win | qdepth p95 | backlog Δ | disp_exc | future_wait p95 | enqueue_wait p95 | NVML sm% mean/max |
|-----|----|-----|----------|----------|----------|----------|----------------------|--------------|--------------|---------------|-----------|-----------|----------|-----------------|------------------|-------------------|
| sweep1 | 314 | 0 | 147 | **304** | 855 | 1051 | 325 | **PASS** | 51.6 | 50.6 | 16 | 4447 | 0 | 4926 | 9632 | 87.2 / 94 |
| rep1   | 312 | 0 | 241 | **531** | 914 | 987  | 551 | FAIL | 65.8 | 64.4 | 16 | 4416 | 0 | 4993 | 14850 | 87.2 / 94 |
| rep2   | 313 | 0 | 226 | **482** | 971 | 1220 | 502 | FAIL | 77.4 | 75.9 | 16 | 4428 | 0 | 4999 | 14374 | 87.4 / 94 |
| rep3   | 313 | 0 | 231 | **523** | 942 | 1027 | 541 | FAIL | 77.9 | 76.3 | 16 | 4428 | 0 | 5021 | 14289 | 88.1 / 94 |

**Noise band at N=64:** TTFS p95 ranges **304–531 ms** (median of the four ≈ 502 ms; one run 304, three runs 482–531) and lag p95 325–551 ms. **N=64 straddles the strict P95-TTFS<400 ms gate run-to-run: 1 of 4 runs passes, 3 of 4 fail.** All four are 0-error with maxjit=0 (loadgen kept realtime) and 0 dispatcher exceptions. The windowed dispatcher CPU fraction at N=64 itself varies 51.6%→77.9% across runs (the passing 304 ms run had the lowest, 51.6%). So the robust (consistently-passing) knee is BELOW 64; N=64 is the boundary/marginal point, not a stable knee.

### Noise-band repeats at the ROBUST knee (N=56) + edge probe (N=60)

| run | ok | err | TTFS p50 | TTFS p95 | TTFS p99 | TTFS max | keepup_lag (lag p95) | maxjit | p95<400 | disp_cpu%win | qdepth p95 | backlog Δ | disp_exc | NVML sm% mean/max |
|-----|----|-----|----------|----------|----------|----------|----------------------|--------|---------|--------------|-----------|-----------|----------|-------------------|
| 56 (run1) | 280 | 0 | 15 | **23** | 33 | 186 | 44 | 0 | PASS | 73.6 | 16 | 2144 | 0 | 83.0 / 94 |
| 56 (rep2) | 280 | 0 | 15 | **23** | 32 | 395 | 43 | 0 | PASS | 57.8 | 16 | 2051 | 0 | 82.1 / 93 |
| 56 (rep3) | 280 | 0 | 15 | **24** | 37 | 421 | 45 | 0 | PASS | 73.4 | 16 | 2072 | 0 | 82.1 / 93 |
| 60 (probe)| 294 | 0 | 16 | **32** | 110 | 262 | 51 | 0 | PASS | 75.7 | 16 | 3002 | 0 | 85.5 / 95 |

**Noise band at N=56:** TTFS p95 = **23 / 23 / 24 ms** across 3 runs — essentially zero run-to-run variance (±1 ms), ~17× under the 400 ms gate, 0-error, maxjit=0, 0 exceptions. N=60 also PASSES (p95 32 ms, p99 110 ms). So:
- **Robust clean knee = N=56–60** (every run passes 0-error ∧ p95<400 ms with large margin; N=56 p95 band = 23±1 ms, N=60 p95 = 32 ms).
- **N=64 = the boundary** (1/4 runs pass at 304 ms; 3/4 fail at 482–531 ms).
- **N≥72 = over the cliff** (2540 ms → 8247 ms).
- **Knee noise band overall:** the SLO-robust zero-error knee sits at **N ≈ 56–60 (cliff edge at 64)**. The dominant run-to-run noise is NOT at the robust knee (N=56 is ±1 ms) but in the steep transition zone (N=64 p95 swings 304↔531 ms, a ~±115 ms band straddling the gate). For the plan's "+4 streams moves-the-knee" rule, the run-to-run knee uncertainty is ≈ ±4 streams (56 stable, 60 passing, 64 marginal).

## RAW SUMMARY (data only — NO GO/STOP verdict)

- **Clean knee (highest N, 0-error ∧ P95-TTFS<400 ms, robust across runs): N = 56–60.** N=56 p95 = 23±1 ms (3 runs), N=60 p95 = 32 ms (1 run). N=64 is the boundary (1/4 runs pass @304 ms, 3/4 fail @482–531 ms). N≥72 fails hard (2540 ms+).
- **Noise band:** N=56 TTFS p95 = 23/23/24 ms (negligible, ±1 ms). N=64 TTFS p95 = 304/482/523/531 ms (straddles the gate). Knee uncertainty ≈ ±4 streams.
- **Telemetry-source mechanism:** HTTP `curl` to the ws_server port 8080 — `GET /scheduler_telemetry` (dispatcher_cpu_pct + the windowed us-delta variant, dispatcher_stream_util_pct, queue_depth p50/p95/p99, counts.backlog_gt_bmax, counts.dispatcher_exceptions, B1/B4, dispatch_cycles) and `GET /stats?last=4096` (metrics.scheduler_future_wait_ms / scheduler_enqueue_wait_ms quantile summaries). GPU util from `nvidia-smi dmon -s u -d 1` (sm% mean/max). All on-box, no AWS creds. counts reported as before/after per-window deltas; pct as both the lifetime field AND the windowed us-delta `_win` value.
- **Salient raw signals for the arbiter (NOT interpreted here):** at the robust knee→cliff (N56→64→72→96): NVML sm% = 83 → 87 → 88 → 90 % (never saturates 100%); windowed dispatcher_cpu_pct = ~58–74 → 52–78 → 57 → 83 %; queue_depth pinned at ring-capacity 16 at ALL N; backlog_gt_bmax Δ = 2.1k → 4.4k → 4.6k → 5.0k (monotone); 0 dispatcher exceptions and 0 loadgen errors at every N; maxjit=0 everywhere (loadgen not the limiter). future_wait/enqueue_wait are flat-and-multi-second even where TTFS is ~20 ms — absolute scale not client-meaningful (semantics-shift caveat above); use trend only.

## Artifacts (on box, ubuntu@34.214.169.199)
- Server stdout log: `~/density/step0_ws_server.log`
- Per-N raw captures: `~/density/step0_out/N<NN>_<tag>_<HHMMSS>.{telem_before.json,telem_after.json,stats_after.json,dmon.txt,loadgen.json,loadgen.txt,row.json}`
- Launch script: `~/density/launch_ws_server_l40s.sh`; loadgen: `~/density/rt_loadgen.py`; per-N capture driver: `~/density/measure_one_n.sh`
- Local copies of the scripts: `proj-2026-05-31-1050/{rt_loadgen.py,launch_ws_server_l40s.sh,measure_one_n.sh}`
- **Server + box left RUNNING (not torn down), per instructions.** Stale `/tmp` AOTI/inductor dirs cleaned at start and end.


