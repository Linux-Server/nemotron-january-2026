# Scaling above the L40S N=36 knee

## Verdict

Do not rank the current candidate levers in the proposed order. The observed knee is not set by memory or a fully
saturated GPU, but it is also not primarily an `enc_first` problem. The best reading is:

1. **Missing first-class lever: steady cross-stream batching / phase-aware coalescing**, if real traces can produce
   B>1 without adding wait and a batched AOTI path is T1-clean. This is the only lever that can plausibly unlock most
   of the 17-27% idle GPU reported at N=36-40 and move the knee to the mid/high 40s.
2. **Steady encoder launch/dispatch reduction**, including a steady CUDA-graph/AOTI-graph wrapper if Nsight shows launch
   gaps. This is plausible but unproven; NVML util alone is insufficient.
3. **Steady/finalize scalar-sync + decode cleanup**, especially deterministic `enc_len` and then a device/batched greedy
   decode path. This can shrink TTFS and some keep-up tail, but the easy scalar-length fix alone will not make N=40 pass.
4. **`enc_first` TorchScript pool / AOTI fold**. The lock is real in the synchronized harness, but stagger nearly removes
   the lock and N=40 still fails. Treat this as a churn/tail hygiene lever, not the density ceiling lever.

Upper-bound math: if the only issue were idle-gaps at the same per-stream work, N=36 at 73.0% mean GPU util has a perfect
100% bound of about 49 streams and a more realistic 90% bound of about 44 streams. N=40 at 75.4% has a 90% bound of about
48. These are ceilings, not forecasts; any claim above roughly N=48 needs reduced per-stream compute, not just smoother
dispatch (`runtime/artifacts/l40s_w3_logs/w3_run13.log:317`, `runtime/artifacts/l40s_w3_logs/w3_run13.log:411`).

## A. Paradox

The paradox is real: N=36 passes with GPU util mean 73.0%, N=40 fails with util mean 75.4%, and N=44 fails with util mean
83.3% (`runtime/artifacts/l40s_w3_logs/w3_run13.log:317`, `runtime/artifacts/l40s_w3_logs/w3_run13.log:411`,
`runtime/artifacts/l40s_w3_logs/w3_run13.log:505`). A saturated-throughput story would predict
util near 95-100% before collapse. Instead, the code alternates short GPU work with host-side serialization points:

- First chunks run through one shared TorchScript `enc_first.ts` behind a single mutex. The shared module is loaded once
  (`runtime/cpp/density_main.cpp:624-658`), the lock is taken for each first chunk
  (`runtime/cpp/density_main.cpp:887-895`), and the first branch records `enc_first_lock_wait` and `enc_first_total`
  (`runtime/cpp/density_main.cpp:937-946`).
- Steady chunks do use the intended shared AOTI loader with `num_runners=N` and explicit worker streams
  (`runtime/cpp/density_main.cpp:3115`, `runtime/cpp/density_main.cpp:3142-3148`), but every continuation immediately
  does scalar extraction and greedy decode after the AOTI call (`runtime/cpp/density_main.cpp:947-995`).
- The scalar sites are blocking D2H / host decisions: `scalar_i64_timed()` copies to CPU and `.item()`s, and
  `argmax_item_timed()` does `argmax().item<int64_t>()` (`runtime/cpp/density_main.cpp:828-839`). The greedy decode loop
  runs `joint.forward()` then one `.item()` per symbol decision (`runtime/cpp/density_main.cpp:842-871`).
- Finalize repeats the same pattern: AOTI bucket run, blocking `enc_len`, greedy decode, then event sync and telemetry
  (`runtime/cpp/density_main.cpp:1220-1368`).

That explains why the GPU is idle while the service falls behind. At N=40, steady_gpu p50 rises to 28.5 ms, but p95 stays
near 38 ms; the wall tail instead explodes in `steady_latency` and `decode_wall`
(`runtime/artifacts/l40s_w3_logs/w3_run13.log:411`). The GPU is not fed as one efficient batch; it is fed by 40
independent B=1 streams with scalar barriers and many small decode launches.

This also explains why final TTFS follows `finalize_total`, not finalize-runner wait. The finalize pool wait is p95 0 at
N=36/40/44, but `finalize_total` p95 jumps 90.9 -> 361.1 -> 483.7 ms, driven mostly by `decode_wall` at N=40/44
(`runtime/artifacts/l40s_w3_logs/w3_run13.log:317`, `runtime/artifacts/l40s_w3_logs/w3_run13.log:411`,
`runtime/artifacts/l40s_w3_logs/w3_run13.log:505`). The code records runner wait separately from GPU and total time
(`runtime/cpp/density_main.cpp:1354-1368`).

## B. Lever Ranking

| Rank | Lever | Expected knee gain | Why |
|---:|---|---:|---|
| 1 | Cross-stream steady/decode batching | **N=36 -> 44-48 if GO; 36 -> 38-40 if weak B; 0 if B~1/T1 fails** | It attacks the main waste: 36-44 independent B=1 launches and decode chains. It can unlock the observed 17-27% idle GPU and reduce per-item work. |
| 2 | Steady CUDA graph / launch-gap reduction | **N=36 -> 40-44 if launch-bound; 0-2 if kernel-bound** | The graph win transfers only if Nsight shows launch gaps. Current telemetry lacks launch counts/SM occupancy. |
| 3 | Scalar-sync and decode cleanup | **N=36 -> 38-42 combined; enc_len-only likely +0-2** | `item_wait` is large, but removing local sync can also reduce overlap fill. Finalize TTFS improves more than keep-up. |
| 4 | `enc_first` K-pool or AOTI first-chunk | **N=36 -> 37-40 in no-stagger harness; production density near 0 unless high churn** | Stagger collapses the lock wait but N=40 still fails. This is not the steady 160 ms ceiling. |

### 1. Cross-stream batching

This is missing from the current native density path. Today every worker calls the B=1 steady package independently
(`runtime/cpp/density_main.cpp:947-958`), so the topology is "36 x B=1", not "B=N". A batch scheduler that forms B=2-4
steady encoder groups could reduce launch count, improve kernel occupancy, and make the GPU do useful work during the
idle gaps.

Skeptical constraints:

- The prior batching simulator already warns that "fill B" is not free: at the deployed 8 ms window, synthetic realistic
  arrivals give mean B about 1.5-2.1 and B=1 at 36-63%; only unrealistic phase alignment fills B (`spikes/0.5-batching-sim/FINDINGS.md:8-28`).
- The simulator's provisional verdict is that the old 3-5x steady-throughput batching claim is effectively dead
  (`spikes/0.5-batching-sim/FINDINGS.md:30-35`). So batching is a candidate for a 20-35% knee push, not a 3-5x rewrite.
- Native batching is not a wrapper around the current package. It needs B>1 encoder export/AOTI, cache gather/scatter,
  per-row decode state, and T1 proof across batched-vs-alone rows. The current C++ state path is per-session, not packed.

Expected gain: if production-like traces at N=36-44 show median B >= 2.5, p95 B >= 4, B=1 <= 30-35%, and B=4 per-row
steady event time is at least 25% lower than B=1, N=44 becomes plausible and N=48 becomes worth measuring. If median
B is around 1.5-2.0, expect N=38-42 at best.

### 2. Steady CUDA graph / launch-gap reduction

The Python finalize graph win does not automatically transfer. That win collapsed many launches in a known graph-managed
path; native steady currently uses AOTI `run(inputs, stream)` with no graph wrapper (`runtime/cpp/density_main.cpp:747-762`,
`runtime/cpp/density_main.cpp:804-825`). We do not yet have Nsight/CUPTI launch counts for the L40S N=36/40 failure,
even though the plan requires
launch gaps, SM occupancy, DRAM throughput, and runner wait in the telemetry schema (`PHASE2-PLAN.md:225-232`).

This is plausible because GPU util is not saturated, but it is not established. The steady p50 inflation from N=36 to N=40
could be launch/stream scheduling gaps, SM contention, memory-system contention, or AOTI-internal queueing. The current
`steady_runner_wait` p95 is 0, which argues against an obvious host runner queue
(`runtime/artifacts/l40s_w3_logs/w3_run13.log:317`, `runtime/artifacts/l40s_w3_logs/w3_run13.log:411`), but it does not
prove the device timeline is launch-gap free.

Expected gain: N=40-44 if Nsight shows launch gaps >= 15% of the measured gate and graph replay cuts steady p50/p95 by
at least 15%. Otherwise this should be deprioritized.

### 3. Scalar-sync and decode cleanup

The easy target is deterministic `enc_len`: both steady and finalize read encoder output length with a D2H scalar sync
(`runtime/cpp/density_main.cpp:873-884`, `runtime/cpp/density_main.cpp:1289-1307`). For fixed geometry, this should be
host-computable after a shadow check over all `(drop,T)` buckets and steady chunks.

But do not over-credit it. At N=40, finalize `enc_len_sync` p95 is only 22.1 ms while `finalize_total` p95 is 361.1 ms;
dropping `enc_len_sync` alone still leaves TTFS far over budget
(`runtime/artifacts/l40s_w3_logs/w3_run13.log:411`). The broader `item_wait` p95 is 32.3 ms at N=40 and 33.0 ms at N=44,
and `item_wait_pct_of_steady_gpu` p95 exceeds 100%, so it is a real local-stream stall
(`runtime/artifacts/l40s_w3_logs/w3_run13.log:411`, `runtime/artifacts/l40s_w3_logs/w3_run13.log:505`). However, the plan
is right that `.item()` wait is also part of the multi-thread fill window, so net density must be measured, not inferred
(`PHASE2-PLAN.md:275-280`).

Expected gain: deterministic `enc_len` is worth doing for TTFS and variance, but likely only +0-2 streams by itself.
Moving greedy decode decisions on-device or batching decode is larger; it could make N=40 TTFS pass, but keep-up lag still
needs steady-path relief.

### 4. `enc_first` lock

The lock is measurable: N=36 no-stagger has `enc_first_lock_wait` p95 639.9 ms and p99 1101.2 ms, and N=40 has p95
708.0 ms (`runtime/artifacts/l40s_w3_logs/w3_run13.log:317`, `runtime/artifacts/l40s_w3_logs/w3_run13.log:411`). But
two facts demote it:

- It fires once per session start, while the steady encoder fires every 160 ms. The harness uses 8 short sessions per
  worker and synchronizes starts, so it over-represents first-chunk churn relative to long-lived multi-turn production.
- The stagger run nearly removes this burst: N=36 lock p95 drops to 10.3 ms, N=40 to 36.5 ms, yet N=40 still fails with
  lag p95 1149.7 ms and TTFS p95 301.7 ms (`runtime/artifacts/l40s_w3_logs/w3_run14.log:225`,
  `runtime/artifacts/l40s_w3_logs/w3_run14.log:321`).

A TorchScript K=4 pool is a sensible hygiene patch if short-session churn is a real product load. It costs roughly
3 extra copies * 2.31 GiB = 6.9 GiB; even N=48 peak 19.3 GiB plus that pool remains below the 44.4 GiB L40S total
(`runtime/artifacts/l40s_w3_logs/l40s_density_N48_20260527T040756Z.stdout.log:70`). K=8 may fit but should be tested for
fragmentation. The cleaner AOTI first-chunk fold is still blocked by T1/event behavior; the plan records that
`Fix-1 AOTI-first-chunk` remains blocked and TorchScript stays (`PHASE2-PLAN.md:305`).

Expected gain: it may certify 37-39 in the synchronized harness and reduce first-turn tail, but I would not count it as
the lever that moves production density above 36.

## C. Missing / Under-Specified Items

### Cross-stream batching

This is the biggest omission in the post-knee plan. PHASE2 mentions "cross-stream transfer/compute batching" only as a
lower Tier-3/Step-2/3 item (`PHASE2-PLAN.md:275-299`), but the L40S evidence says it should be a first-class Step-1c
candidate before scheduler design is frozen.

Transfer from Python is partial:

- Yes, the idea transfers mechanically: gather same-geometry ready sessions, run one B>1 encoder/decode job, scatter
  caches and hypotheses back.
- No, the current Python "continuous batching" evidence does not justify assuming high B. Existing findings argue mean
  B is usually around 1.5-2.1 at an 8 ms wait window, and the old 3-5x claim should be dropped
  (`spikes/0.5-batching-sim/FINDINGS.md:18-35`).
- The native runtime may have more incentive to batch than Python because GPU util is only 73-83% around the knee, but
  that makes it a measured opportunity, not a plan assumption.

### Steady CUDA graph

PHASE2 names CUDA-graph-of-AOTI as a fallback if AOTI dispatch serializes (`PHASE2-PLAN.md:24-32`), and older graph
ownership analysis warns that graph buffers are not shared just because weights are shared (`spikes/0.11-graph-ownership.md:21-46`).
There is no current Step-1b follow-up that measures the steady graph resident memory, graph replay lost-overlap, or
launch-gap reduction on L40S.

Graphing steady may be large if the N=36 idle is launch gaps. It may be near zero if the AOTI package is already fused
enough and the p50 inflation is true kernel contention. This needs Nsight, not debate.

### C++ dispatch / exclusive-gate serialization

The Python server's exclusive gate and `inference_lock` are not present in this native density loop. The native code has
only an optional global `g_aoti_run_mutex`, used when `mutex_serialize_run` is set (`runtime/cpp/density_main.cpp:745-762`);
the L40S run is the explicit-stream, mutex-false topology (`runtime/artifacts/l40s_w3_logs/w3_run14.log:150`). The loop
uses per-worker streams and passes them into AOTI for steady/finalize (`runtime/cpp/density_main.cpp:804-825`,
`runtime/cpp/density_main.cpp:1284-1287`).

What remains possible is lower-level dispatch serialization inside AOTI/libtorch/CUDA launch, not a project-code
exclusive gate. The evidence against a simple hidden mutex is that GPU event p50 rises with N instead of staying flat,
and runner wait p95 is 0 at the boundary (`runtime/artifacts/l40s_w3_logs/w3_run13.log:317`,
`runtime/artifacts/l40s_w3_logs/w3_run13.log:411`). The evidence for more work is the unsaturated NVML util and missing
launch/SM/BW counters. A L40S negative control with `--mutex-serialize-run` and default-stream should still be part of
the next attribution run.

## D. Plan Update

PHASE2-PLAN already contains pieces of these levers:

- Step 1b is done and correctly records N=36, N=40 first fail, not memory, and stagger-robust collapse
  (`PHASE2-PLAN.md:163-177`, `PHASE2-PLAN.md:306`).
- Step 2/3/4 cover scheduler, real WS, and realized apples-to-apples density (`PHASE2-PLAN.md:179-208`).
- W2 names `enc_len`; W4 names the autotune ladder; W5 names glue/pinned copies/cross-stream batching
  (`PHASE2-PLAN.md:292-299`).

But the plan does not yet have a pre-registered "push above 36" triage step. I recommend **extending PHASE2-PLAN.md with
an optional Step 1c**, not creating a new Phase 3 yet. These are still core-runtime density-ceiling questions; they decide
what Step 2 scheduler should be designed to schedule. A new Phase 3 should start only after Step 1c selects a larger
architectural change, such as B>1 native batching or steady graph pools.

Suggested Step 1c structure in the existing style:

- **Step 1c-0 - L40S attribution trace.** N=36 and N=40, no-stagger and 10s stagger, same artifacts. Add Nsight/CUPTI:
  launch gaps, kernel count, SM occupancy, DRAM throughput, stream overlap, AOTI host launch time, negative controls
  `mutex_serialize_run` and default stream. GO to batching/graph work if idle/launch gaps are >=15% of measured gate while
  SM/DRAM are below 85-90%. STOP that branch if counters show true SM/DRAM saturation and controls behave.
- **Step 1c-A - Batching kill gate.** Trace or replay production-like readiness at N=36-44 with an 8 ms max wait and the
  real batch key. In parallel, compile B=2/B=4 steady AOTI fixtures and compare batched-vs-alone tokens/caches. GO if
  median B >=2.5, p95 B >=4, B=1 <=35%, added wait p95 <=8 ms, B=4 per-row GPU time <=0.75x B=1, and 0 T1 mismatch.
  STOP if median B <2 or per-row gain <15% or any T1 mismatch.
- **Step 1c-B - Steady graph/launch kill gate.** Only if Step 1c-0 shows launch gaps. Capture/replay steady B=1 on a
  known stream with graph-pool memory measured. GO if steady p50 or p95 drops >=15%, graph pool has >=8 GiB L40S headroom
  at the target lane count, and N=40 or N=44 full replay passes. STOP if improvement <10%, graph memory is large enough
  to reintroduce the old lane cap, or output lifetime/correctness is not clean.
- **Step 1c-C - Scalar/decode cleanup.** Deterministic `enc_len` first, then async/pinned tiny copies, then device/batched
  decode only if still needed. GO for plan inclusion if item_wait p95 drops >=50% and N=40 lag p95 improves >=30% without
  widening TTFS. STOP density-credit if it only improves TTFS and not lag.
- **Step 1c-D - `enc_first` pool.** Run only if product traffic is short-session/high-churn or if N=37-39 is needed for
  a certification margin. GO if K=4 cuts lock p95 below 75 ms and certifies a higher no-stagger knee with <8 GiB extra
  memory. STOP as density lever if N=40 still fails, which is the likely outcome.

## E. Cheapest Decisive Step-0 For The #1 Lever

For cross-stream batching, the cheapest decisive kill gate is a two-part **BATCH-0** that avoids building the full server:

1. **Opportunity trace, no model changes.** Reuse the density harness or recorded production/loadgen arrivals to emit
   ready timestamps and batch keys for steady/finalize. Simulate an 8 ms and 12 ms window. GO only if median B >=2.5,
   p95 B >=4, B=1 <=35%, and added wait p95 <=8 ms at N=36-44. STOP if median B <2 or B=1 >50%.
2. **Batched steady fixture microbench.** Export/compile B=2 and B=4 steady AOTI packages for the exact current geometry,
   pack independent session caches from the bundle, run batched-vs-alone shadow comparisons, and record CUDA event time
   per row plus launch counts. GO only if B=4 per-row steady time <=0.75x B=1, B=2 <=0.85x B=1, 0 token/cache/event
   mismatches over the full fixture set, and predicted N=44 has both lag p95 <500 ms and TTFS p95/p99 within 175/250.
   STOP if correctness drifts, per-row gain is <15%, or B opportunity fails.

This is cheaper and more decisive than implementing a native production scheduler first. It directly tests whether the
only lever large enough to unlock the idle GPU is real for this workload.
