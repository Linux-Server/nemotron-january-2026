# Red-Team Review - Going-Forward Plan After L40S N=36

**Verdict: GO-WITH-CHANGES.** Do not ship the going-forward plan as written. The plan is directionally right to refuse a rubber-stamp of N=36, but it overstates certainty in three places that are now load-bearing: the Python baseline, the decode-sync diagnosis, and the N=40/44 target. Those are fixable with tighter pre-registered gates; without them this becomes an optimization treadmill pointed at a moving bar.

The single biggest issue: **Step 1c is being made load-bearing before the bar it must clear is locked.** `PHASE2-PLAN.md:42-46` admits the Python baseline is preliminary, single-utterance, and not the full Step-4 manifest, but `PHASE2-PLAN.md:181-183` treats `S_py ~= 20` as pinned and makes "push >36" the path to a robust Step-4 GO. That is not a safe decision structure.

## Must-Fix 1 - Lock `S_py` Before Spending Optimization Work

**Finding.** The plan uses a noisy same-box Python remeasure as if it were a hard bar. `PHASE2-PLAN.md:42-46` says `S_py_L40S` must be remeasured under the Step-4 apples-to-apples manifest, then immediately records only a preliminary single-utterance burst. `CHECKPOINT-scaling-above-36.md:87-88` correctly caveats that the exact knee is unpinned and `S_py ~= 20` is a 2-repeat sweep with no p99, but `PHASE2-PLAN.md:181-183` still turns that into a load-bearing Step 1c.

Telemetry is not stable enough to carry that:

- `runtime/artifacts/l40s_w3_logs/spy_high.json:11` has only `repeats=2`.
- There is no p99 field in `runtime/artifacts/l40s_w3_logs/spy_high.json`; the plan's own Step 4 says the loadgen still needs p99 and P99-P50 output (`PHASE2-PLAN.md:231-235`).
- The Python p95 curve is non-monotone: level 16 has `ttfs_p95=147.438ms` (`runtime/artifacts/l40s_w3_logs/spy_high.json:25-35`), level 18 improves to `66.966ms` (`runtime/artifacts/l40s_w3_logs/spy_high.json:37-47`), level 20 is `158.488ms` (`runtime/artifacts/l40s_w3_logs/spy_high.json:49-59`), and level 24 is `248.569ms` (`runtime/artifacts/l40s_w3_logs/spy_high.json:61-71`). The JSON's own `"knee": 24` (`runtime/artifacts/l40s_w3_logs/spy_high.json:86`) is not the exact native gate because level 24 violates the 175ms p95 budget in `PHASE2-PLAN.md:49-50`.

**Which way the uncertainty cuts.** It is two-sided, and both sides undermine the current plan:

- If the full Step-4 manifest plus p99 makes Python worse, `S_py` could be lower. At `S_py=18`, the Step-1b bar is `max(34, 1.80*18)=34`, and Step-4 after the 0.83 realization haircut gives `0.83*36=29.9` versus `1.5*18=27`. The "problem" largely dissolves.
- If the full manifest, Python continuous batching, long-lived multi-turn sessions, or repeat noise makes Python better, `S_py` could be higher. At `S_py=24`, the Step-1b bar is `43.2`, and the Step-4 haircut requires roughly `1.5*24/0.83=43.4` native streams. A Step 1c that only proves N=40 is not enough.
- At `S_py=20`, the plan is exactly at bar: `1.80*20=36`, and Step 4 is `0.83*36=29.9` versus `1.5*20=30`. This is not a margin; it is a rounding dependency.

**Correction to pre-register.**

Add a blocking baseline-lock step before Step 1c is allowed to decide scope:

```markdown
- [ ] **Step 1b.5 - S_py_LOCK (BEFORE Step 1c scope/funding decisions).**
  Same L40S box, same Step-4 apples-to-apples manifest, same loadgen/server flags, p50/p95/p99 and P95-P50/P99-P50 emitted, multi-turn included or explicitly scoped out. Sweep levels 16/18/20/22/24 plus one bracket point above the first fail. Run >=3 repeats, or until repeat CV <=10% and pass/fail is monotone; if pass/fail is non-monotone, keep measuring or use the largest contiguous pass from the bottom, not an isolated higher pass.
  `S_py_lock` = highest level that passes lag_p95 <500ms, ttfs_p95 <=175ms, ttfs_p99 <=250ms, non-intentional error <=1% in all qualifying repeats.
  Set the native Step-1c target to:
  `S_native_req = ceil(max(34, 1.80*S_py_lock, (1.50*S_py_lock)/0.83))`.
  A robust Step-4 plan needs `S_native_candidate >= S_native_req`; `>36` is not a sufficient target unless the formula says so.
```

## Must-Fix 2 - Add an Explicit Funding Re-Justification Checkpoint

**Finding.** The plan acknowledges that Step 4 technical GO is not the funding GO (`PHASE2-PLAN.md:236-237`), but it does not add an explicit decision point after the multiplier collapsed from the hoped 2.0-2.25x range to a zero-margin 1.8x. That changes the 40-60 eng-week second-stack calculus. The old thesis was "meaningful L40S density win"; the current measured low case is "maybe 1.5x after scheduler/WS haircut, with no margin."

**Correction to pre-register.**

```markdown
- [ ] **Funding Recheck F1 - after S_py_LOCK and Step 1c, before freezing Step 2/3 build scope.**
  Report:
  `nominal_realized_multiplier = 0.83*S_native_candidate/S_py_lock`
  `pessimistic_realized_multiplier = 0.75*S_native_candidate/S_py_lock`
  `engineering_scope = remaining eng-weeks + permanent dual-stack carry`.
  GO-to-build without escalation only if nominal >=1.70x and pessimistic >=1.50x, with p99 guardrail met.
  Otherwise mark TECHNICAL-CONDITIONAL and require explicit human re-justification of the second-stack bet.
```

This does not change the technical density gate. It prevents a technically valid but strategically weak 1.50x result from silently inheriting the earlier 2.0x funding rationale.

## Must-Fix 3 - Step 1c-0 Is Not Yet a Clean Arbiter

**Finding.** `PHASE2-PLAN.md:188-193` makes the 5090 `--decode-no-host-sync` ablation "THE ARBITER": GO if staggered N=40 has `decode_wall p95 <50ms` and TTFS passes; pivot if `decode_wall` stays >150ms. That is too strong.

First, 5090 is a development proxy, not a density arbiter for L40S. The plan itself says 5090 and L40S differ: Step 1a's 5090 true knee is N=40 with N=48 first non-SLO (`PHASE2-PLAN.md:342`), while Step 1b's L40S knee is N=36 with N=40 first non-SLO (`PHASE2-PLAN.md:343`). A 5090-local result can validate T1 and expose gross regressions; it cannot justify skipping batching on L40S.

Second, the proposed thresholds are only loosely calibrated. The staggered L40S N=36 baseline has `decode_wall p95=8.945ms`, `ttfs_p95=42.003ms`, and `lag_p95=-119.106ms` (`runtime/artifacts/l40s_w3_logs/w3_run14.log:225-226`). Staggered N=40 still fails with `decode_wall p95=252.971ms`, `ttfs_p95=301.747ms`, and `lag_p95=1149.671ms` (`runtime/artifacts/l40s_w3_logs/w3_run14.log:321-322`). A `decode_wall <50ms` target is plausible, but not sufficient: if steady latency/lag stays bad, the runtime still fails.

Third, `decode_wall` is not the same as `.item()` wait. The code does per-token `argmax().item()` at `runtime/cpp/density_main.cpp:835-861`, and it then conditionally mutates predictor state at `runtime/cpp/density_main.cpp:862-868`. But in the failing N=40 row, `decode_wall p95=308.629ms` while `decode_item_wait p95=17.051ms` (`runtime/artifacts/l40s_w3_logs/w3_run13.log:411-412`). In the staggered N=40 row, `decode_wall p95=252.971ms` while `decode_item_wait p95=16.883ms` (`runtime/artifacts/l40s_w3_logs/w3_run14.log:321-322`). Removing host argmax sync may help, but the telemetry does not prove it removes the whole `decode_wall` explosion.

Fourth, the proposed implementation may not be T1-clean. The greedy loop is host-controlled and data-dependent (`runtime/cpp/density_main.cpp:842-871`), and finalization repeats the same pattern after AOTI bucket output (`runtime/cpp/density_main.cpp:1289-1307`). A "device-side argmax + fused blank flag" still needs an exact decision about blank/nonblank, predictor updates, token append order, and event timing. If that cannot be made token/event-exact, the arbiter cannot run cleanly.

Fifth, the plan knows the double-edge risk but does not gate on it. The lever inventory says `.item()` is also the multi-thread fill window (`PHASE2-PLAN.md:303-306`). Yet Step 1c-0 GO only requires `decode_wall`, TTFS, and mismatch (`PHASE2-PLAN.md:191-193`). It can miss a density regression where sync removal improves local TTFS but reduces aggregate overlap/fill.

**Correction to pre-register.**

Replace Step 1c-0 with a two-stage gate:

```markdown
- **1c-0a 5090 dev/T1 smoke (not a density arbiter).**
  Build `--decode-no-host-sync`; T1 binds to the exact build over 1000/1000 finals + strict events. If token/event mismatch: STOP this lever and run enc_len-only plus diagnostic no-decode/Nsight; do not infer density.

- **1c-0b L40S causal confirm (the arbiter).**
  Same g6e/L40S, same sm_89 package, same manifest, flag-on/flag-off paired controls, N=36/40/44 plus `S_native_req` from Step 1b.5. Run enough repeats for CV <=10%.
  GO: `N >= S_native_req` is SLO-robust (lag_p95 <500ms, ttfs_p95 <=175ms, ttfs_p99 <=250ms), 0 token/event mismatch, non-intentional errors <=1%, and no net regression at N=36 (`throughput_rt >=0.98x control`, `lag_p95 <= control+50ms`, `TTFS_spread <= control+25ms`).
  PIVOT: N=40 remains non-SLO OR `decode_wall p95 >=150ms` OR lag worsens >=30% versus control.
  AMBIGUOUS: `decode_wall p95` lands 50-150ms or local TTFS improves without admitted-successful density. Run 1c-A/1c-B before deciding.
```

Also make `1c-A(1)` opportunity tracing unconditional and parallel. It is model-free and cheaper than a device-side RNNT decoder build. Waiting for 1c-0 to fail biases the plan toward the convenient sync lever and delays the only evidence that can justify or kill batching.

## Must-Fix 4 - The Diagnosis Is Overstated

**Finding.** The checkpoint says the ceiling is host-side serialization, not the GPU (`CHECKPOINT-scaling-above-36.md:17-20`), and the plan repeats that as an attributed fact (`PHASE2-PLAN.md:184-187`, `PHASE2-PLAN.md:296-302`). That is too certain for the telemetry in hand.

The stagger result proves that synchronized start bursts are not the whole story; it does **not** prove the remaining N=40 collapse is specifically decode host sync. Staggered N=40 has `steady_latency p95=378.2ms`, `lag_p95=1149.67ms`, `ttfs_p95=301.747ms`, `decode_wall p95=252.971ms`, and `gpu_util_mean=72.6%` (`runtime/artifacts/l40s_w3_logs/w3_run14.log:321-322`). That could be decode host sync, but it could also be steady stream contention, repeated finalize-pool contention, AOTI/CUDA launch scheduling, or a workload phase issue.

The 2-runner finalize pool remains a confound. The code caps finalize runners at `min(workers_or_runners, 2)` (`runtime/cpp/density_main.cpp:1211-1213`). Finalize then obtains the bucket loader and calls AOTI inside the measured region (`runtime/cpp/density_main.cpp:1272-1287`), while `finalize_runner_wait_ms` is computed as `runner_host_ms - gpu_ms` (`runtime/cpp/density_main.cpp:1354-1359`). So `finalize_wait=0` in the logs is not proof that no pool wait exists; it can be hidden inside the timed AOTI region, exactly as the checkpoint notes at `CHECKPOINT-scaling-above-36.md:21`.

The util-bound extrapolation is especially weak. `CHECKPOINT-scaling-above-36.md:13` and `PHASE2-PLAN.md:300-302` imply N=36 at 73% util gives a reclaimable ceiling around 44-48 streams. But the detailed row shows N=36 has `gpu_util_mean=73.0%`, **p50=91%**, and **p95=97%** (`runtime/artifacts/l40s_w3_logs/w3_run13.log:317-318`), while N=40 has `gpu_util_mean=75.4%`, **p50=96%**, and **p95=98%** (`runtime/artifacts/l40s_w3_logs/w3_run13.log:411-412`). Mean NVML utilization is not linear headroom. Some of the "idle" may be sampling, phase lulls, stream dependency bubbles, or irreducible latency between scalar decisions. The 44-48 number is an optimistic bound, not a forecast.

**Correction to pre-register.**

Downgrade the plan language from "attribute the ceiling to host-side serialization" to "working hypothesis pending 1c-B counters." Make 1c-B mandatory before selecting any Tier-3 or before declaring batching unnecessary:

```markdown
- **1c-B attribution is mandatory, not optional narrative support.**
  Capture L40S N=36/40 with no-stagger and 10s stagger, flag-on/flag-off if 1c-0 exists, plus mutex/default-stream controls.
  Required counters: kernel timeline, launch gaps, stream overlap, SM occupancy, DRAM throughput, AOTI host launch time, finalize pool wait/reclaim time separately from AOTI run.
  Do not use NVML mean util to forecast streams. Report 44-48 only as an upper bound until Nsight/CUPTI shows reclaimable launch/sync gaps.
```

## Must-Fix 5 - Re-Tiering Is Too Confident About `enc_first`

**Finding.** Demoting `enc_first` is probably right for long-lived production sessions, but the plan calls it a harness artifact too broadly. `PHASE2-PLAN.md:315-317` says `enc_first` is lag-not-TTFS, stagger-erased, and overrepresented by short sessions. The lock is real in code (`runtime/cpp/density_main.cpp:887-895`) and large in synchronized telemetry: N=36 `enc_first_lock_p95=639.909ms` (`runtime/artifacts/l40s_w3_logs/w3_run13.log:318`), N=40 `708.014ms` (`runtime/artifacts/l40s_w3_logs/w3_run13.log:412`), N=44 `788.320ms` (`runtime/artifacts/l40s_w3_logs/w3_run13.log:506`). Stagger reduces it sharply to 10.287ms at N=36 and 36.519ms at N=40 (`runtime/artifacts/l40s_w3_logs/w3_run14.log:226`, `runtime/artifacts/l40s_w3_logs/w3_run14.log:322`), but that only proves one harness pattern is bad.

Real traffic can resurrect it: short sessions, reconnects, barge-in, multiple streams per call, call-center greeting snippets, or load balancer churn all increase first-chunk pressure. If the product is not actually "long-lived multi-turn with low start churn," treating `enc_first` as last-tier is unsafe.

**Correction to pre-register.**

```markdown
- **enc_first demotion is traffic-conditional.**
  Before declaring `enc_first` a harness artifact, collect or simulate product start-churn:
  session_start_rate_per_box, turns_per_connection, reconnect rate, barge-in/cancel rate, and simultaneous first-chunk bursts.
  Keep `enc_first` de-prioritized only if target traffic has P95 session lifetime >=60s OR first-chunk starts are <=5% of steady chunks at the admitted cap, and staggered first-chunk p95 remains <75ms at `S_native_req`.
  Otherwise keep a K-pool/AOTI-first-chunk hygiene gate in Step 2, but do not count it as steady density unless N=40+ actually becomes SLO-robust.
```

## Must-Fix 6 - Step 2 Should Not Be Fully Blocked, But Step 1c Needs a Stop Line

**Finding.** `PHASE2-PLAN.md:208-214` blocks scheduler design on Step-1 telemetry. That is partly right: if native B>1 batching is selected, the scheduler architecture changes. But several Step-2/3 items are invariant and can move in parallel: admission close semantics, admitted-vs-offered accounting, stale-generation tests, WS-tail microbench scaffolding, and telemetry fields. Blocking all of Step 2 behind 1c serializes work unnecessarily.

The opposite risk also exists: Step 1c can become an open-ended optimization treadmill. The current plan lists sync removal, batching, Nsight, scalar cleanup, and `enc_first` without a hard "good enough" stop or a hard "rework/funding" stop (`PHASE2-PLAN.md:181-206`).

**Correction to pre-register.**

```markdown
- **Step 2a - invariant scheduler/server work may proceed after S_py_LOCK starts.**
  Admission accounting, close-shed behavior, stale-generation harnesses, WS-tail microbench, and telemetry schema are allowed in parallel.

- **Step 2b - lever-specific scheduling waits for Step 1c.**
  Final scheduler topology (B=1 threaded, B>1 batched, priority finalize partitioning) waits for 1c selection.

- **Step 1c STOP line.**
  STOP optimizing and proceed to Step 2/3 once `S_native_candidate >= S_native_req` with p99 guardrail and >=2-stream measurement margin.
  If 1c-0, BATCH-0, and 1c-B do not forecast or measure `S_native_req`, pause for REWORK/Funding Recheck F1. Do not continue into Tier-3/autotune/enc_first work without a written expected-streams delta that closes the formula.
```

## What The Plan Got Right

- It correctly marks Step 4 as at risk instead of pretending N=36 is a comfortable pass (`PHASE2-PLAN.md:343-347`).
- It keeps correctness/T1 binding before performance claims, including exact-build T1 for Step 1c (`PHASE2-PLAN.md:188-191`).
- It separates technical GO from funding GO in Step 4 (`PHASE2-PLAN.md:236-237`), even though it needs a real funding checkpoint.
- It correctly recognizes that `finalize_wait=0` is not by itself proof of no finalize contention (`CHECKPOINT-scaling-above-36.md:21`).
- It correctly treats native B>1 batching as Phase-3-scale if selected (`PHASE2-PLAN.md:194-198`).

## Single Biggest Risk And One Change

**Biggest risk:** the plan optimizes against an unstable Python denominator and an over-specific mechanism diagnosis. That can waste the cheap Step 1c work if `S_py` falls, underbuild if `S_py` rises, or skip batching because a 5090 proxy says sync cleanup helped while L40S Step 4 still fails.

**One change:** insert **Step 1b.5 S_py_LOCK + Funding Recheck F1** before Step 1c is allowed to set scope, and redefine Step 1c's target using `S_native_req = ceil(max(34, 1.80*S_py_lock, 1.50*S_py_lock/0.83))`. Then make the 5090 ablation a dev/T1 smoke and require an L40S net-density confirmation before skipping batching.

## Caveats

- I did not rerun measurements; this review uses committed plan text, source, and logs only.
- The native exact knee remains bracketed, not pinned; N=37-39 are still untested.
- The apples-to-apples p50/p95/p99 chart is still in flight. Its result should supersede every `S_py ~= 20` statement in this review and in the plan.
- I co-authored one of the source analyses folded into the plan, so I deliberately weighted this review against my prior conclusions where the telemetry is not causal.
