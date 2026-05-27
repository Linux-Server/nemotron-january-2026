# L40S W3 density profile review

## Verdict

Your core read is right: the observed knee is in `(32, 48)`, and the first failing resource is keep-up / GPU compute contention, not memory.

Telemetry recap:

| N | SLO | wall_ms | rt streams | lag p95 | TTFS p95 / p99 | steady GPU p50 / p95 | item wait p95 | enc_first lock p95 | peak mem |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | yes | 2,863,880 | 0.999 | -149.9 ms | 10.6 / 11.4 ms | 7.13 / 7.62 ms | 1.74 ms | ~0 ms | 11.0 GiB |
| 8 | yes | 396,857 | 7.21 | -142.7 ms | 16.8 / 21.3 ms | 7.50 / 11.87 ms | 4.92 ms | ~0 ms p95, 106 ms p99 | 12.3 GiB |
| 32 | yes | 97,492 | 25.7 | -66.9 ms | 72.6 / 112.8 ms | 13.55 / 40.20 ms | 26.77 ms | 570 ms | 16.5 GiB |
| 48 | no | 118,315 | 31.2 | +4461 ms | 499.6 / 946.5 ms | 30.73 / 39.33 ms | 30.89 ms | 893 ms | 19.3 GiB |

Sources: row summaries in `runtime/artifacts/l40s_w3_logs/l40s_density_N1_20260527T021950Z.stdout.log:71`, `runtime/artifacts/l40s_w3_logs/l40s_density_N8_20260527T021950Z.stdout.log:71`, `runtime/artifacts/l40s_w3_logs/l40s_density_N32_20260527T040756Z.stdout.log:71`, and `runtime/artifacts/l40s_w3_logs/l40s_density_N48_20260527T040756Z.stdout.log:71`; full per-phase JSON is on line 70 of each same log. N=48 is at 19.3 / 44.4 GiB peak/total, CPU is 7.46 / 32 cores, and GPU util is 83.6% mean / 96% p50 / 98% p95, so memory and CPU are not the limiter.

## 1. Wall-Time

`wall_ms` is only the measured concurrent replay, not the process wall. The harness starts resource sampling and the timed gate only after worker contexts and warmup complete: `resources.start()` and `gate.start_now()` happen at `runtime/cpp/density_main.cpp:3286-3287`, threads join at `runtime/cpp/density_main.cpp:3288`, and `result.wall_ms` is computed from `gate.start_time` at `runtime/cpp/density_main.cpp:3292`. Everything before that is setup.

The two setup costs are:

1. Serial oracle build. `run_density_sweep()` computes the max rows needed for the largest N, prints `SERIAL ORACLE BUILD`, builds the reference, then runs the N loop: `runtime/cpp/density_main.cpp:3600-3624`. The build itself loads one worker context, preloads all finalize buckets, and serially replays every reference row: `runtime/cpp/density_main.cpp:1463-1499`.
2. Per-N warmup. After loaders and contexts are created, every worker runs a two-chunk steady warmup, then warms every worker-local representative finalize bucket: `runtime/cpp/density_main.cpp:3135-3194`. The expensive part is that each bucket warmup first calls `prepare_finalize_parent()`, which replays the whole steady utterance from scratch: `runtime/cpp/density_main.cpp:2581-2612`.

The high-N setup blow-up is dominated by the warmup loop, with the serial oracle still a large repeated fixed cost. Evidence: N=1 and N=8 both built a 300-row oracle (`runtime/artifacts/l40s_w3_logs/l40s_density_N1_20260527T021950Z.stdout.log:3`, `runtime/artifacts/l40s_w3_logs/l40s_density_N8_20260527T021950Z.stdout.log:3`), but warmup grew from 16 to 113 finalize bucket worker-runs (`runtime/artifacts/l40s_w3_logs/l40s_density_N1_20260527T021950Z.stdout.log:69`, `runtime/artifacts/l40s_w3_logs/l40s_density_N8_20260527T021950Z.stdout.log:69`). N=32 used fewer oracle rows than N=8, 256 vs 300, but did 196 finalize warmups and took the much longer high-N slot (`runtime/artifacts/l40s_w3_logs/l40s_density_N32_20260527T040756Z.stdout.log:3`, `runtime/artifacts/l40s_w3_logs/l40s_density_N32_20260527T040756Z.stdout.log:69`). N=48 then scales both rows and warmups to 384 rows / 318 warmups (`runtime/artifacts/l40s_w3_logs/l40s_density_N48_20260527T040756Z.stdout.log:3`, `runtime/artifacts/l40s_w3_logs/l40s_density_N48_20260527T040756Z.stdout.log:69`). The current logs do not timestamp `SERIAL ORACLE PASS -> WARMUP COMPLETE`, so add explicit phase timers before the next long run.

Make the sweep fast without weakening the knee:

- Build the serial oracle once for a bracket. The binary already supports this if you pass multiple N values in one process; the reference is built once before the N loop (`runtime/cpp/density_main.cpp:3613-3624`). The wrapper defeats that by launching one fresh process per N (`runtime/run_l40s_density.sh:737-756`, `runtime/run_l40s_density.sh:969-975`). Use one direct binary call for search, then fresh-process only for final confirmation.
- Do not preload all 32 finalize buckets for the serial oracle. `build_serial_reference()` calls `finalize_loaders.preload_all()` at `runtime/cpp/density_main.cpp:1471-1473`; the measured N=32/N=48 runs only need 16 loaded buckets. Preload only buckets reachable by the reference rows.
- Replace per-worker-per-bucket warmup with per-loaded-bucket/per-runner warmup. Current N=48 warms 318 full-parent replays for 16 loaded buckets and 2 finalize runners. The cold-start target is bucket/runner kernels, not every worker/bucket pair. Warm `needed_buckets * finalize_num_runners`, plus one small per-worker joint/predict touch if needed.
- Keep `--density-sessions-per-worker 8` for headline rows, but use a cheaper first bracket with 4 if needed. Then rerun the boundary at 8. The correctness oracle only needs to cover assigned utterances (`runtime/cpp/density_main.cpp:3057-3060`), and the finalize p95 floor is already enforced (`runtime/cpp/density_main.cpp:3046-3055`).

## 2. Pinning the Knee

Efficient sequence:

1. Search in one process: `density_main --mode density-sweep --n-values 36,40,44 --density-sessions-per-worker 8 --density-chunk-period-ms 160 artifacts_sm89`. This builds one oracle for max 352 sessions, then measures N=36/40/44. Expect the process to return `1` if N=1 is not included because `pass_to_1b` is a whole-sweep summary gate; parse the emitted row telemetry, not the process code, for this bracket search.
2. Confirm the boundary fresh-process. If 40 passes and 44 fails, rerun N=40 and N=44 fresh-process. If 44 passes, rerun 48 and optionally 46. If 36 fails, rerun 34/36.
3. Treat same-process search as a bracket finder only. The plan correctly called for fresh-process-per-N because same-process memory/warm state can bias the headline (`PHASE2-PLAN.md:249-252`), but that cost is unnecessary for every exploratory point.

Keep the pass criteria unchanged: completed sessions, 0 mismatches/errors, stream uniqueness, lag p95 < 500 ms, TTFS p95 <= 175 ms and p99 <= 250 ms (`runtime/cpp/density_main.cpp:3318-3327`).

## 3. Optimization Levers

Ranked by expected density gain:

1. **Steady encoder contention / scheduling.** This is the real ceiling. N=48 fails with steady GPU p50 30.7 ms vs 13.6 ms at N=32 and 7.1 ms at N=1, while runner wait is ~0 and GPU util is high. The steady path is the AOTI loader call at `runtime/cpp/density_main.cpp:794-799`, invoked for continuation chunks at `runtime/cpp/density_main.cpp:927-933`. Pushing materially beyond 40 needs either a T1-safe faster encoder package, better kernel scheduling, or deliberate phase smoothing of chunk arrivals. Expected gain: enough for 36/40/44 if you get 10-25% effective GPU-time reduction; N=48 likely needs more than sync/lock cleanup.
2. **D2H syncs and decode item waits.** The sync sites are explicit: `scalar_i64_timed()` copies to CPU and `.item()`s at `runtime/cpp/density_main.cpp:802-805`; greedy argmax does `.item()` at `runtime/cpp/density_main.cpp:809-812`; steady consumes both at `runtime/cpp/density_main.cpp:847-858`; finalize consumes `enc_len` and decode at `runtime/cpp/density_main.cpp:1264-1284`. At N=32, item wait p95 is 26.8 ms and finalize `enc_len_sync` p95 is 23.4 ms; at N=48, item wait p95 is 30.9 ms and finalize decode wall p95 is 454 ms. First target: make encoder output length host-deterministic for fixed chunk/finalize geometry. Second target: move greedy decisions on device or batch/specialize the decode loop. Expected gain: mainly TTFS/tail and a few streams of knee; not enough alone to rescue N=48.
3. **`enc_first` lock.** It is real but not the main N=48 binding. The shared module is wrapped with one mutex (`runtime/cpp/density_main.cpp:607-641`), acquired for every first chunk (`runtime/cpp/density_main.cpp:861-868`, called at `runtime/cpp/density_main.cpp:911-920`). N=32 lock p95 is 570 ms, N=48 is 893 ms. Safe near-term path is a small TorchScript `enc_first` pool, not removing the mutex from one shared module. K=4 costs about 3 extra copies * 2.31 GiB = 6.9 GiB, still well within the N=48 memory headroom, and should cut first-chunk lock wait roughly K-fold while preserving TorchScript semantics. K=8 may still fit but needs fragmentation margin. The AOTI-fold path should stay blocked unless it passes the event gate; the plan notes that AOTI first-chunk flipped an interim event (`PHASE2-PLAN.md:305`). Expected gain: startup/tail cleanup and possibly 32 -> 36/40, but not 48 without steady-compute work.
4. **Setup-only cleanup.** Oracle reuse and lean warmup will turn a multi-hour bracket into minutes, but they do not change the measured knee. They are still urgent because they make N=36/40/44 cheap enough to measure repeatedly.

Practical target: with current kernels, expect the honest knee to land around 36-40, maybe 44 if the N=48 collapse is partly synchronized-burst amplification. To push above that, prioritize steady encoder contention plus sync removal; lock cleanup is worth doing, but it is a tail/starter fix, not the compute ceiling fix.
