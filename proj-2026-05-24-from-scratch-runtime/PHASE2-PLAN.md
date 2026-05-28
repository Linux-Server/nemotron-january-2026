# Phase 2 — multi-stream density (the streams/box win)

Project directory: /home/khkramer/src/nemotron-january-2026/proj-2026-05-24-from-scratch-runtime

> **v2 (2026-05-26):** rewritten after a 5-round paired Codex+Opus adversarial review (`reviews/{codex,opus}-
> phase2-round{1..5}.md` + `phase2-round{1..5}-FOLDED.md`). The "meaningfully" gate is replaced by a pre-registered
> numeric GO/STOP tree; the cheap kill-gates are front-loaded; the loader/ownership/topology contract and a
> correctness-before-performance gate are pinned. **Reviewer sign-off: GO-to-build Step 1 (Step-0 kill-gates
> first), conditional on the edits below.**

> **v3 (2026-05-27) — STEADY-BATCH-0 PASSED → the batched-steady encoder is the GREEN-LIT density build.** The L40S
> profiling (nsys+ncu, paired Codex+Opus) inverted the framing: the N=36 knee is **DRAM-bandwidth-bound
> weight-streaming in the STEADY encoder** (88% GEMM, 72% DRAM, 15% occupancy = a BW-wall, NOT idle SMs — ~1.4
> concurrent encoder GEMMs already saturate HBM), and the **only** lever that moves the byte floor is cross-stream
> batching (load the weight once, reuse across B streams). Both kill-gate conjuncts then PASSED:
> **(1) OPPORTUNITY** — steady B-fill is real (mean B 2.7/3.5 at N=36, 3.6/4.3 at N=44, 3.8/4.4 at N=56, for 8/12ms
> windows; refutes the prior "batching dead" sim, which modeled *finalize* bursts not the *steady* 160ms cadence);
> **(2) SPEEDUP** — per-row steady GPU time B=1 5.11ms → B=2 3.18ms (**0.62×**) → B=4 1.92ms (**0.38×**), total grows
> sub-linearly (the weight-load amortizes), correctness byte-exact per-row (enc_out max 6e-6, within tol). ⟹ Tier-2
> (batched steady) is the **validated #1 lever** (no longer "B-fill-gated" — the fill is MEASURED); projected knee
> **37 → ~47-64** (Amdahl-bounded by the 88% steady GEMM share × the measured fill); the funding multiplier moves
> from at-bar **1.8× → ~2-2.5×**, **provisionally clearing F1** pending the build's own L40S sweep (Step B3).
> **User call (2026-05-27): enough evidence to proceed → build the batched-steady runtime now.** The L40S confirm of
> the SPEEDUP *ratio* is a formality (the per-row amortization is a roofline property that transfers sm_120→sm_89);
> the build's B3 sweep measures the realized *absolute* knee. Build = Steps **B1-B3** (after Step 1c). 1c-0
> sync-ablation is moot (profiling routed straight to batching); 1b.5 S_py_LOCK is de-blocked from the build start
> but still feeds the Step-4 apples-to-apples. Gate data: `reviews/profiling-paired-verdict.md`,
> `reviews/{opus,codex}-l40s-profiling-analysis.md`; SPEEDUP bench `runtime/steady_b_artifacts/bench_out.log`.

> **v4 (2026-05-28) — L40S knee sweep DECISIVELY confirmed the lift; 10 insights folded into ongoing
> Step 2/3 + Tier 3 + B3-follow-up work.** B_max=4, W=0, L=0 reached the top of the registered N axis
> (**N=64 SLO-robust, 0 token divergences**) on the L40S — the projected 47-64 ceiling was hit at the
> upper bound; the **true knee is >64**. Realized ratio vs `S_python_high_end≈20`: 64/20 = **3.2×**; after
> the 0.83× Step-4 haircut, **2.65× ≫ F1 thresholds (1.7× nominal / 1.5× pessimistic) — funding case
> clears decisively pending the final B3 paired-verdict consolidation**. The 10 actionable insights from
> the sweep:
>
> **Defaults that are now wrong (1-line code changes — bundle with B3 verdict commit):**
> - **(1)** Admission `active_cap` default of 40 is stale (it was the OFF-baseline knee). Change to either
>   a deploy-required-no-default OR 64+ (Step 2a `density_admission.h:42`).
> - **(2)** Scheduler default `window_ms=10` should be **`0`** (Step 2a `batched_steady_scheduler.h:25`).
>   W=0 L=0 won on the L40S — at the knee the queue is structurally full, the gather wait adds nothing.
>   W/L stay configurable for diagnostic sweeps but their production setpoint is "off."
>
> **Lift shape (informs production setpoints):**
> - **(3)** B_max=2 barely lifts (L40S knee=40, only +4 over OFF=36); B_max=4 lifts heavily (≥+28). The
>   crossover is sharp (per the microbench's 0.62× vs 0.38× per-row ratios). B_max=4 is the right
>   production default (already is); B_max=2 is debug-only — don't burn sweep cycles on it going forward.
> - **(4)** **Known ceiling at ~N=80-100** = the dispatcher single-thread saturates. Not urgent (≥64
>   clears F1 by huge margin); the multi-dispatcher / per-stream-pool architecture is the lever when this
>   becomes binding — **deferred to Tier-4 future-work** (post-Phase-2).
>
> **Step 3 sizing rules (when Step 3b builds the real WS server):**
> - **(5)** Worker thread pool must support **64+ concurrent streams** (size for ~80-100, not 40). On
>   g6e.8xlarge (32 vCPU) we're at 2:1 oversubscribe — fine for I/O-bound workers.
> - **(6)** **WS-tail microbench (Step 3a) should be exercised up to N=128** to characterize WS overhead
>   at and above the actual knee. The realized operating point ≥64; testing 128 covers the headroom +
>   the projected future-lift regime (post-multi-dispatcher).
>
> **Measurement debt — B3 follow-ups (bracket the true knee + close §II.13):**
> - **(7)** True knee is >64 (sweep capped). A follow-up L40S sweep at N ∈ **{72, 80, 88, 96, 112, 128}**
>   should bracket the actual ceiling. NOT BLOCKING the B3 verdict (funding case is settled regardless);
>   informs production sizing.
> - **(8)** Burst-injection at N=64 (synchronized starts) per §II.13 hasn't completed yet — pathological
>   arrival pattern surfaces worst-case HOL the staggered-start sweep hides. Either Codex's in-flight L40S
>   sweep covers it, or it's a small B3 follow-up.
>
> **Things that the lift CONFIRMS (no action needed):**
> - **(9)** Tier 3 memory shrink (+11.8 GiB → +~5 GiB) is comfortably under L40S budget: at N=64 ON ≈ 22
>   GiB peak; 26 GiB headroom. The stretch <4 GiB miss doesn't matter for L40S deployment (mattered more
>   for the 5090 dev cycle, still tight there).
> - **(10)** W=0 L=0 + B_max=4 = the simplest possible config is the production winner. The complexity
>   (window/lone tuning, configurable buckets) is justified for diagnostic + low-load preserve purposes
>   but the production setpoint is the boring one. Validates the design didn't over-tune.

## Why / the bet's open conjunct
Phase 1 proved the single native stream is correct (token/event-exact to the server's final transcript;
preproc→AOTI steady→decode→finalize-bucket→state-machine→emit). It did NOT measure DENSITY, and that compute
correctness does NOT carry the concurrency machinery — the validated core runs ONE AOTI loader synchronously on
the default stream (no per-thread streams). The project rests on the BET conjunct that the residual is
**dispatch/host-bound (reclaimable by true multi-thread overlap), not memory-bandwidth-bound** → one native
multi-thread process lifts streams/box. ("No GIL" is necessary, not sufficient — see the decisive question.)

The only density evidence so far is the **0.1b microbench (PRELIMINARY)**: 5090 ≥3×, L40S ~2–2.5× — but it used a
**MOCK decode (host sleep + dummy GEMM)** dispatched via `graph.replay()`, NOT the real `AOTIModelPackageLoader::
run()`. **Phase 2 measures it for real** with the validated single-stream native core + the proven shared-weights
mechanism, and reports the SLO-robust knee (not the keep-up knee, which overstated ~2–3×).

## THE decisive first question (the hard gate)
The architecture is right and natively supported: **one C++ process, true OS threads, ONE shared weight set** —
libtorch 2.8 `AOTIModelPackageLoader(num_runners=N)` builds N runners that **share one constants set** (one 2.5GB
copy, solving the earlier `lanes=32` OOM), and `run(inputs, stream_handle)` takes an explicit per-thread stream.
Per-stream `SessionState` is cleanly isolated. **So the single decisive unknown is: does the container's shared
execution lock (`model_container.h:92-103`) let concurrent `run()` actually OVERLAP on the GPU, or does it
serialize the dispatch (an AOTI-internal "GIL")?** That is conjunct 2, decisively, with REAL decode (not the 0.1b
mock). **De-risked fallback if it serializes:** per-thread CUDA-graph-of-AOTI-steady (the primitive already ships
in the Python finalize-graph win) → a lock-serialization finding is a topology PIVOT, not a STOP.

## Goal contract (user-ratified 2026-05-26)
- **G1 = system utilization / streams-box density.** This is the **GO/STOP gate** (the frozen 0.0 threshold:
  ≥1.5× L40S density, strategic capability bet, no COGS break-even).
- **G2 = tightened P50↔P95 spread, measured as server-side `TTFS_spread = ttfs_p95 − ttfs_p50` (vad_stop→final).
  REPORTED at every gate, NOT a STOP criterion.** It bounds the user-visible finalization tail; VAD/WAN are out
  of scope (only the server-side slice is movable — roofline). A density pass is NOT described as a "two-goal"
  success; G2 is reported alongside so density is never bought by widening the tail unnoticed.

## Definitions used by the gates
- `S_py_L40S` — fresh Python baseline, SLO-robust **successful** streams/box, re-measured back-to-back under the
  Step-4 apples-to-apples manifest (NOT the stale ~16–20). **Preliminary same-box single-proc re-measure 2026-05-27
  = ~20 @ ttfs p50 ~42ms (single-utterance burst, `spy_*.json`); the full multi-turn Step-4 manifest is still pending
  — but ~20 already pins the 1.8× multiplier at-bar.**
- `G1_floor = max(28, 1.50·S_py_L40S)`. SLO-robust = keep-up `lag_p95 < 500ms` AND `ttfs` within the ttfs SLO
  budget AND non-intentional admitted error rate ≤1%; count **admitted-successful** streams, never offered.
- **ttfs SLO budget** (server-side vad_stop→final, co-located → NO real WAN): **`ttfs_p95 ≤ 175ms` (gate),
  `ttfs_p99 ≤ 250ms` (guardrail)** — PINNED 2026-05-26. *Derivation:* the **<400ms END-TO-END** hard target counts
  endpoint-wait + WAN, and `ttfs` is measured from vad_stop→final (endpoint-wait already elapsed; loadgen is
  co-located so WAN≈0) → server-side budget = 400 − endpoint_wait(~200) − WAN_delivery(~25) ≈ **175ms** p95;
  p99=250 is a tail guardrail, not a hard gate. *Empirical cross-check (binding if tighter):* native `ttfs_p95` ≤
  the Step-4 **re-measured** Python server-side `ttfs_p95` — the shipped server meets <400 end-to-end (finalize-
  graph win 246/279 p50/p95 WITH WAN; `vad_stop_to_sent`~43ms p50), so its co-located server-side number is the
  empirical reference the native runtime must not regress. *If the deployment endpoint-wait differs* (e.g. the
  silence0/warm200 config), recompute `BUDGET = 400 − endpoint_wait − WAN`. DISTINCT axis from keep-up
  `lag_p95 < 500ms`; always name `ttfs_*` vs `lag_*`.
- `WER_bound` — native semantic-WER ≤ max(WER_py + 0.5pp, 1.10·WER_py), same corpus/tool/model/prompt/version.
- `Reject_bound` — intentional WS-1013 rejects ≤10% of offered; non-intentional errors ≤1% of admitted; report the
  no-shed curve too.

## Compile & artifact policy (2026-05-26 — autotune is a Phase-2 perf lever)
Phase 1 compiled AOTI with **autotune OFF** (default) because the goal was token-EXACTNESS, where autotune only
adds numeric variation. **Phase 2 is a perf/density measurement, so autotune is ON** — it's the AOT-time cost we
pay for faster runtime kernels, and it lands hardest on the **steady encoder, which Step 0 identified as the
GPU-contention bottleneck that caps density**. No OOM concern (AOT). Rules:
- **Two NATIVE artifact sets per target: `<arch>-autotune-ON` (HEADLINE) + `<arch>-autotune-OFF` (FLOOR), BOTH
  compiled natively on the target GPU.** `max-autotune` benchmarks configs on the PRESENT GPU → cross-compile-
  autotune is invalid. 5090 → sm_120 on+off (local); L40S/g6e → sm_89 on (+ a sm_89 off floor, or an explicit
  waiver per Step 1b). Report the **off→on density win as the absolute SLO-robust streams/box Δ and %** (headline
  vs floor, same everything) — NOT each artifact's multiplier over its own N=1.
- **One artifact dir per (arch × autotune) variant; NEVER clobber a validated baseline.** `enc_steady_aoti.pt2`
  (autotune-off sm_120) is used by `session_main` + the Step-0 harness → autotune-on writes a SEPARATE path
  (`artifacts_at_sm120/`, `artifacts_sm89/`), SHAs preserved.
- **Autotune-on is CONTINGENT — and a T1-gated LADDER, not a binary shelve.** T1 token/event + WER-neutral
  re-validation binds to the **EXACT package SHA** benchmarked; **a T1 FAIL retreats one rung down the ladder, it
  does NOT silently ship**. *Mechanism (MEASURED 2026-05-26, verified+CORRECTED by
  `reviews/codex-autotune-drift-verify.md`):* autotune diverged the **precision/accumulation policy + kernel
  choices** away from eager's TF32-reduced path — **NOT purely reduction-order (first framing overstated).** Smoking
  gun: autotune-on `cache_t` max_abs **10.27 == the earlier knob-matrix result for forcing `fp32_highest`/
  `emulate_precision_casts`** (`knob_matrix.log`, `0.2b-aoti-findings.md:109-128`) → eager uses TF32-reduced
  accumulation, autotune-OFF matches it (1.66e-2), autotune-ON diverged (matmul/bmm kept ALLOW_TF32=False but
  **convs went ALLOW_TF32=True**). Amplified by `cache_t` dynamic range (abs.mean 0.39 / max 54, confirmed for the
  fixture; corpus-representativeness unproven) + recurrence — but MEASURED autotune-OFF long-stream drift
  **PLATEAUED not compounded** (drift-probe 0/830 flips, shadow 1/1000), so "flips tokens over a stream" was
  overstated; autotune-ON 10.27 is unmeasured (T1 check settles it). **Ladder
  (cheapest→surgical, each rung T1-gated):** (1) **match eager's TF32/precision policy** in the autotune compile
  (the 10.27==fp32-precise clue says precision-policy divergence is the prime suspect — pin TF32 on/off + matmul
  precision to eager's, incl. the conv ALLOW_TF32) + `max_autotune` WITHOUT `coordinate_descent_tuning`; (2)
  **exclude the `cache_t`-producing recurrent GEMMs** from autotune (keep them eager-default → minimal recurrent
  drift, tune the rest); (3) numeric-aware: autotune → T1-filter candidates → keep the fastest T1-passing config. Drop to autotune-OFF (the validated floor) only if no rung passes T1. The in-flight T1 check
  (rung 0 = max+coord) quantifies the drift→token-flip gap to target the ladder; the warm floor re-sweep shows how
  much density autotune even needs to add (if off-warmed ≈ the gate, the marginal GEMM-autotune speed may not be
  worth the drift battle). If the autotune **compile** fails (Triton/driver/toolchain/timeout) → **compile-blocked**;
  the off artifact is a **diagnostic floor only, NEVER silently substituted as the headline**.
- **Reproducible off→on claim:** repeated-run stability (CV ≤10%) for BOTH headline + floor before reporting the
  win; **pin/cache the autotune configs**; log warm/cold Inductor cache. Autotune compile is heavier than the
  default (billable g6e hours for 32 buckets) → **smoke the autotune path on the DL AMI with one small bucket
  first**; capture compile acceptance criteria (timeout/retry, log, torch/CUDA/driver/Triton/Inductor config +
  cache state, package SHA256).
- **Exported programs (EPs) are the reusable, arch-agnostic intermediate — PRESERVE durably (Q1 fix).** They were
  gitignored throwaways (`runtime/.gitignore: artifacts/`) and got cleaned, forcing a full re-export. Back up to
  **S3 with a MANIFEST** (object path, byte size, SHA256, generating commit/command, model+fixture hashes, the
  32-bucket contract keys); the **one-command regen FAILS CLOSED** if the regenerated set differs from the
  contract. The S3 copy doubles as the g6e compile source (75 GB upload = the L40S long pole — start it first).
  **Decided (user + both reviewers): back up the 75 GB AS-IS now; do NOT externalize EP weights pre-gate** — it's
  a build-time storage saving that doesn't affect runtime; logged as tech-debt (export-with-shared-weights would
  shrink EPs ~30×).

## Ownership / topology contract (pin before building)
One shared steady AOTI loader **`num_runners=N`** + explicit **per-worker CUDA streams**; **per-worker**
`SessionState`, `AudioFrontend`, `enc_first`, `joint`, `predict`, `preproc` (concurrent `forward()` on one shared
TorchScript handle is unproven — the mock used per-lane handles). `user_managed`/codisk shared constants for the
**finalize buckets** only (steady is covered by the runner pool). Negative controls: mutex-serialized,
default-stream, `num_runners=1`. Log `num_runners`/stream-mode/topology in every artifact name.

## Steps (the pre-registered GO/STOP tree)

- [x] **Step 0 — cheap native kill-gates (5090, ~hours, BEFORE any EC2 spend).** DONE — conjunct-2 binary = YES
  (paired-reviewed, commits d77bede + this). 0a/0b/0c all PASS; overlap real + correctness-safe + ONE weight copy;
  the ~1.7× encoder-only plateau is **GPU CONTENTION (encoder saturates the 5090 ~N=4), NOT the execution lock**
  (kernel p50 inflates 5.28→13.63→28.54ms with N=1/4/16 while throughput flattens — a lock would leave duration
  flat). Each is a potential early STOP (corroborated — see Rules). PAIRED REVIEW.
  - **0a steady AOTI pool overlap:** one loader `num_runners=N`, per-worker streams, real `run(inputs,stream)`,
    **per-worker distinct input/cache tensors** (not a shared tensor set). PASS = ≥1.15× throughput @N=2 and
    ≥1.30× @N=4 vs N=1 same-harness, **steady-output concurrent==serial vs a serial oracle (0 mismatch — 0a needs
    its own correctness check, not just `out.size()`)**, an **Nsight/CUPTI trace showing kernels issued on ≥2
    non-default streams** (the `sum_gpu/wall` "overlap estimate" is contention-confounded → diagnostic only, NOT
    proof), and the **one-weight-copy memory gate measured fresh-process-per-N via the loader-delta**
    (`used_after_loader−used_before_loader` flat across N; total peak is allocator-polluted → not the gate) at the
    target N=16–40. STOP-candidate = <1.10×@N=2 / <1.20×@N=4, any mismatch, or per-runner weight duplication.
  - **0b decode + per-thread ownership:** per-worker SessionState + per-thread handles. TWO sub-gates: **identity**
    = concurrent==serial token+event 0 mismatch; **scalar-locality** = the per-thread `.item()` sync is LOCAL not
    global — default-stream negative control ≥15% worse AND a sentinel/Nsight probe confirms an explicit-stream
    `.item()` does NOT drain unrelated streams. Report `item_wait_pct_of_gpu` as telemetry only — **a HIGH
    per-thread `.item()` fraction is the overlap OPPORTUNITY threads fill, NOT a failure; do NOT gate on it.**
    STOP-candidate = identity mismatch OR `.item()` causes global/default-stream serialization.
  - **0c real finalize + hot-bucket:** real fork/clone + bucket route, same- and mixed-bucket. PASS = 0
    token/event/WER regression (incl. collector fields + stale-gen-sensitive events), finalize p95 runner-wait
    ≤25% of vad_stop→final, no aliasing/stale leak. STOP-candidate = mismatch, stale final, or wait >50%.
  - All PASS → Step 1a. (Density ceiling = min(GPU-BW, CPU-cores, launch/lock) — a CPU-core-bound knee on the
    32-vCPU box is a "more-cores/fewer-syncs" finding, not a STOP.)

- [x] **Step 1a — 5090 mini-sweep (spend-control proxy, NOT a project GO).** Real decode + real finalize +
  per-thread streams/handles + the full telemetry schema, with a **BOUNDED workload** (`--density-sessions-per-worker`,
  not the full corpus at real-time — the unbounded default ran ~30min/N; pre-register a MINIMUM sessions-per-worker
  + repeat count, SAME bound for headline + floor). **WARMUP IS MANDATORY (paired investigation finding, FOLDED):**
  the native AOTI loaders pay a one-time ~225ms CUDA-12 lazy-module-load/first-launch per loader (universal — the
  steady encoder shows the same max spike, amortized over ~292 chunks; finalize buckets run ≈once so it dominated a
  6-sample p95 → the "234ms@N=1 / finalize-NO_PASS" was a COLD-START + tiny-sample artifact, NOT a 10× bug — warm
  native finalize is ~11ms p95, Python-class). → **the RUNTIME must warm EVERY finalize bucket at startup** (one
  throwaway forward per `(drop,T)`, like Python's `NEMOTRON_WARMUP_MS=200`; optionally `CUDA_MODULE_LOADING=EAGER`),
  and **the harness must warm ALL buckets + take ≥20–100 finalize samples for a valid p95 + split telemetry**
  (`fork_clone/aoti_run/enc_len_sync/decode_wall/decode_item_wait/decode_tokens/glue`). Autotune is ORTHOGONAL to
  cold-start (won't fix it). Re-sweep after de-contaminating before trusting any finalize/ttfs verdict. **Headline = sm_120-AUTOTUNE-ON** (recompiled locally to a
  SEPARATE dir, not clobbering `enc_steady_aoti.pt2`) + a **bounded autotune-OFF floor** run → report the
  **autotune win** (absolute SLO-robust streams/box Δ + %, headline vs floor). Knee **attributed to a binding resource (BW / launch / lock / CPU-core)** via Nsight/CUPTI
  counters — "BW-bound" is a hypothesis, not NVML util. PASS-to-1b = autotune-on real-decode multiplier **≥2.00×**
  vs N=1 AND 0 mismatch AND WER in bound AND `ttfs` within the ttfs SLO budget (Definitions). Report `TTFS_spread`.
  TUNE/RETEST = 1.50–2.00× with attribution to a fixable harness/topology issue (no EC2 until ≥2.00× or a written
  non-predictive exception). STOP-candidate = <1.50× or SLO/WER/correctness fail. The 5090→L40S transfer is noisy
  → a marginal 5090 PASS does NOT substitute for the L40S measurement.

- [x] **Step 1b — L40S CEILING hard gate (EC2). DONE → PASS. L40S native knee = N=36 SLO-robust (true ∈[36,39]); N=40 first non-SLO; keep-up/GPU-compute-bound (not memory); ~1.8-2.25× Python; stagger-robust; paired-reviewed GO (Codex+Opus). See the progress table 1b row + reviews/codex-l40s-knee-verdict.md.** Same harness, **sm_89-AUTOTUNE-OFF compiled NATIVELY on the
  g6e** as the PRIMARY T1-valid ceiling — and what the gate is read against. (Autotune-ON is currently **T1-BROKEN**:
  cache_t drift 10.27, 995/1000, precision-policy divergence from eager's TF32-reduced path — see Levers — so the
  autotune-OFF measurement is the only token-safe ceiling.) EPs shipped via S3 per the Compile & artifact policy;
  artifact package SHA recorded. **Autotune-ON is a CONTINGENT Tier-2 follow-up, NOT this gate:** attempted only IF
  the L40S returns compute-CONTENTION-bound (autotune speeds compute → moot if memory/CPU-bound) AND a
  precision-matched ladder (rung-1 = match eager TF32) passes T1 first; if attempted, reported as the off→on
  streams/box Δ + % with identical workload/N/bound/cadence/topology + distinct artifact hashes. Run config: g6e.8xlarge
  (1× L40S 48GB, 32 vCPU — matches the 5090 box's thread count, where the knee used only 3.2/32 cores, so CPU is not
  an artificial cap), fresh-process-per-N N=1..80. PASS-to-build = `S_native_step1b ≥ max(34, 1.80·S_py_L40S)` (the 1.80× leaves margin for the ~17% Step-4 scheduler/WS haircut;
  34 = 28/0.83) + 0 mismatch + WER in bound + `ttfs` within the ttfs SLO budget (Definitions). `TTFS_spread`
  reported (GREEN ≤1.10× Python; otherwise a build-risk signal, not a STOP — the binding tail is reported at
  Step 4). CONDITIONAL = ≥G1_floor but <max(34, 1.80·S_py_L40S) → proceed only on explicit human risk-acceptance +
  a narrow de-risking prototype. STOP-candidate = <G1_floor or SLO/WER/correctness fail. PAIRED REVIEW (decisive
  measurement).

- [x] **Step 1b.5 — S_py_LOCK — DEFERRED, non-blocking (user call 2026-05-27: build on the STEADY-BATCH-0 projection).** *(Was BLOCKING; now feeds the Step-4 apples-to-apples, NOT the build start. The apples p50/p95/p99 chart = the first half, done. Original spec retained below.)* *(Paired-review must-fix MF-1, 2026-05-27 — `reviews/goforward-paired-verdict.md`.)* The same-box re-measure that yielded S_py≈20 is a **noisy `repeats=2` single-utterance burst, no p99, non-monotonic** (conc 12/14/16/18/20 ttfs_p95 = 131/58/147/67/158; first FAIL=24) **AND not apples-to-apples** — Python ttfs is client-over-WS (`ec2_loadgen.py:77`), native N=36 is server-side pre-WS (Step 3 unbuilt), so the 1.8× is inflated by the WS slice native hasn't paid. Before 1c sets scope: re-measure Python on the **same L40S, the Step-4 apples-to-apples manifest, WS-matched (or WS-subtracted)**, `repeats≥10` (or until repeat CV≤10% AND pass/fail monotone), levels {16,18,20,22,24}+a bracket point, emit **p50/p95/p99 + P95−P50/P99−P50**, multi-turn included or explicitly scoped. `S_py_lock` = highest level passing lag_p95<500 ∧ ttfs_p95≤175 ∧ ttfs_p99≤250 ∧ err≤1% in ALL qualifying repeats (non-monotone → use the largest contiguous-from-bottom pass, not an isolated higher one). Set **`S_native_req = ceil(max(34, 1.80·S_py_lock, 1.50·S_py_lock/0.83))`**. **Branches:** `S_py_lock≤18` → 36 already clears a robust GO → **Step 1c DEMOTED to optional**; `[19,21]` → proceed to 1c; `≥22` → trigger Funding-recheck F1 before any g6e/Phase-3 spend. *(The in-flight apples p50/p95/p99 chart is the first half of this.)*

- [x] **Step 1c — RESOLVED → batching (the L40S profiling + STEADY-BATCH-0 answered the triage; see v3 + Steps B1-B3).** *(Original triage spec retained below for the record.)* **Step 1c — push the knee >36 (sync/batch triage; target = `S_native_req` from 1b.5, NOT a bare ">36").** Motivation: the
  same-box Python re-measure (2026-05-27) pins **S_py≈20**, making the 1.8× multiplier **at-bar with ZERO margin**,
  and the Step-4 ~17% haircut keeps it at-bar (0.83·36≈30 vs 1.5·20=30) → a robust Step-4 GO needs knee **>36**.
  Two independent analyses (`reviews/{opus,codex}-scaling-above-36-levers.md` → `CHECKPOINT-scaling-above-36.md`)
  attribute the ceiling to **host-side serialization** (36× B=1 forwards + per-token `.item()` D2H syncs leaving
  17–27% GPU idle — NOT memory, NOT the AOTI exec lock = a `std::shared_lock`, NOT `enc_first`); util-bound ceiling
  **≈44–48** (>48 needs reduced per-stream compute). Pre-registered sub-gates, **cheapest arbiter first**:
  - **1c-0a decode-sync ablation — 5090 DEV/T1 SMOKE (NOT a density arbiter).** *(MF-3.)* `--decode-no-host-sync`:
    host-compute `enc_len` from `(drop,T)` + device-side greedy argmax / fused blank flag. **PRECONDITION:** prove
    device-argmax **T1 bit-exact** vs the CPU `.argmax().item()` on a fixture FIRST (device-vs-CPU tie-breaking is
    not guaranteed identical); if it fails, 1c-0 collapses to enc_len-only (~no-op, +0–2 streams) → skip to the
    batching decision. T1 binds to the exact build (1000/1000 finals + strict events; any flip → STOP this lever).
    **⚠️ EXPECTED OUTCOME = PIVOT (FACT-2, verified):** `decode_item_wait` p95 is only **~17ms (flat across N)** while
    `decode_wall` explodes to **309ms@40** — the `.item()` sync is NOT the explosion (that's joint/predict GEMMs
    queued under cross-stream contention), so sync-removal is **bounded ~17ms** and almost certainly does NOT reach
    `S_native_req`. 5090 is a dev/T1 proxy, NOT an L40S density verdict.
  - **1c-0b L40S net-density confirm (THE arbiter).** Same g6e/L40S, sm_89 pkg, manifest; flag-on/off paired
    controls; N=36/40/44 + `S_native_req`; CV≤10%. **GO:** `N≥S_native_req` SLO-robust (lag<500, ttfs p95≤175,
    p99≤250), 0 mismatch, err≤1%, **AND no net regression at N=36** (throughput_rt ≥0.98× control, lag_p95 ≤
    control+50ms, TTFS_spread ≤ control+25ms — catches the fill-window double-edge). **PIVOT→1c-A:** N=40 non-SLO OR
    `decode_wall` p95 ≥150ms OR lag worsens ≥30% vs control. **AMBIGUOUS (50–150ms or TTFS-only gain):** run 1c-A/1c-B first.
  - **1c-A batching kill-gate (only if 1c-0 PIVOTs).** (1) opportunity trace, no model change: replay arrivals at
    8/12ms windows — GO if median B≥2.5, p95 B≥4, B=1≤35%, added-wait p95≤8ms; STOP-batching if median B<2 or
    B=1>50% (the workload won't fill batches; the `spikes/0.5-batching-sim` prior says realistic B≈1.5–2). (2)
    B=2/B=4 steady AOTI fixture shadow-vs-alone — GO if B=4 per-row ≤0.75× B=1, 0 token/cache/event mismatch,
    predicted N=44 holds SLO; STOP on drift or per-row gain <15%. (Native B>1 ⇒ Phase-3.)
  - **1c-B Nsight/CUPTI attribution (MANDATORY — gates the batching decision, not optional narrative).** *(MF-4.)*
    N=36/40, ±stagger, ±`--mutex-serialize-run`/default-stream controls. Counters: kernel timeline, launch gaps,
    stream overlap, SM occupancy, DRAM throughput, AOTI host-launch time, **finalize pool wait/reclaim split OUT of
    AOTI-run** (`finalize_wait=0` is mis-attributed). **3-way routing** (resolves sync-vs-contention): (a) decode_wall
    <50 after 1c-0 → sync-bound; (b) decode_wall>150 AND SM/DRAM<85% with launch gaps≥15% → **cross-stream
    kernel-queue contention → steady-graph/coalescing (Tier-3), NOT batching**; (c) decode_wall>150 AND SM/DRAM≥90%
    → compute-bound → batching. **Do NOT use mean-util to forecast streams** (N=36 mean 73% but p50/p95=91/96;
    staggered N=40 FAILS at 72.6%) — "44–48" is an upper bound, not a forecast.
  - **1c-C scalar/decode cleanup** (deterministic `enc_len` first): GO for density-credit if item_wait p95 −≥50%
    AND N=40 lag p95 −≥30% without widening TTFS; STOP-density-credit if it only improves TTFS (still report it).
  - **1c-D `enc_first` pool** — run ONLY if product traffic is short-session/high-churn or N=37–39 certification
    margin is needed; STOP as a density lever (N=40 still fails — the likely outcome; the 6.9 GiB buys ~0 prod density).
  **STOP line (MF-6):** 1c STOPS (success) once a knee clearing `S_native_req` is **stagger-robust at the knee N and
  N+4**; OR, after 1c-0+1c-B, if no ≤Tier-1b lever closes the formula → **do NOT auto-continue into Tier-3/batching** —
  escalate to Funding-recheck F1 with the realized ceiling (Phase-3 batching needs its own explicit GO). PAIRED REVIEW
  (1c-0 changes the gate math). The outcome selects what Step 2b schedules; a B>1 selection spins up Phase-3.

### Batched-steady build (B1-B3) — GREEN-LIT by STEADY-BATCH-0 (the Tier-2 lever, paired-validated 2026-05-27)
The kill-gate passed (v3 block) → build the cross-stream batched steady encoder, the #1 BW-floor lever. **Byte-exact
when the flag is OFF** (the B=1 path is untouched); the batched path is **T1-gated per-row**. Each step paired-reviewed
before `[x]` (the implement-loop contract). The build is the concrete realization of the Step-2b topology decision
(now made: **B>1-batched steady**), and B3 feeds Steps 3/4.

- [x] **Step B1 — batched-steady forward mechanism + T1 (5090 dev, topology-agnostic core).** — PASS-by-policy 2026-05-27 (paired-reviewed Opus+Codex: `reviews/B1-paired-verdict.md`); 0/2014 token divergences across K=3/B=4 + K=2/B=2 coverage; 6 interim event-timing drifts < 5/1000 prior bar per-pass (DENSITY_GOLD_EVENTS_TOLERANT class); 0 enc_len/cache_len mismatches; flag-OFF byte-exact preserved. Audit follow-ups A1-A8 deferred to B2 (see below). Load B∈{1,2,4} steady
  AOTI buckets (the microbench artifacts + `runtime/export_steady_batched.py`; integrate as a bucket set like the
  finalize buckets — one dir, SHAs recorded, the same shared-constants discipline). Ragged **pack → run → unpack**:
  K ready `(stream, mel-chunk, cache_ch, cache_t)` → stack into the nearest bucket B≥K (pad rows K..B-1) → one batched
  `run(inputs, stream)` → scatter `enc_out`/new-caches/`enc_len` back per stream, discard pad rows. Behind
  `NEMOTRON_DENSITY_BATCH_STEADY` (default OFF → B=1 byte-exact preserved). **T1 GATE:** the K real rows ==
  K individual B=1 forwards, AND the full decode+event path emits byte-exact tokens/events per stream over real
  streams (the microbench showed kernel-level 6e-6 / within-tol — B1 confirms end-to-end through decode). STOP-this-
  lever on any token/event flip not attributable to a documented tolerance. PAIRED REVIEW (correctness-critical).
- [x] **Step B2 — batching scheduler + density integration (5090).** — PASS-with-followup 2026-05-27 (paired-reviewed Opus+Codex: `reviews/B2-build-paired-verdict.md`). Central-dispatcher topology held under both passes. Bidirectional CUDA sync per §II.2 implemented; b2-t1 0/0 token+event divergences across 6 cases (incl. forced-concurrency formed 20 actual B=4 batches + Bmax=1 control); A1 outcome B (SHAs differ, tensors bit-identical → OFF stays on PRODUCTION B=1); OFF-path preserved by structural non-construction. Follow-ups F1-F6 carry to B3 (F1 full-corpus b2-t1 + F2-T telemetry hardening + F2-M memory headroom telemetry are B3 pre-conditions; F3 test hardening; F4-F6 cleanup). 5090 knee re-measure (§II.13) is the next perf-validation task; the F2-M memory delta is part of that sweep. The cross-stream dispatcher: collect ready
  steady chunks in an 8-12ms window, **low-occupancy short-circuit** (run B=1 immediately if <2 ready after a short
  timeout → single-stream/best-case latency preserved), bucket-select K→{1,2,4}. **Topology fork (paired-reviewed):**
  central steady-batching dispatcher vs borrow-and-batch on the arriving worker — pick the lighter that leaves
  per-stream decode/finalize ownership unchanged (the decode runs on the unpacked `enc_out`). Wire into the density
  worker loop. Re-measure the 5090 knee (sanity: does it lift from N=40?). **GO** if knee lifts ∧ 0 mismatch ∧
  ttfs/lag within SLO ∧ added batch-wait p95 ≤ window (absorbed by keep-up slack; does NOT touch finalize ttfs).
  PAIRED REVIEW (the topology is a design fork). **Carry-over audit follow-ups from B1's paired verdict (`reviews/
  B1-paired-verdict.md` — fold into the B2 scope):** **A1** verify NEW `enc_steady_aoti_b1.pt2` is bit-identical to
  PRODUCTION `enc_steady_aoti.pt2` (or use NEW B=1 as the alone reference) — cleans drift attribution. **A2** add
  per-stream **all-chunks-batched T1** (the structural gap B1 couldn't cover — B1 measured one batched chunk inserted
  into a B=1 stream; B2's scheduler batches every chunk, drift compounds). **A3** debug-flagged top-2 joint-score
  margin probe at near-tie blanks. **A4** reentrancy: enforce `preload_all()`-at-startup contract OR add a mutex
  around `BatchedSteadyLoaderSet::get()` (currently non-atomic find-or-create — safe in B1, unsafe under B2's
  concurrent scheduler). **A5** pre-allocated pack/unpack scratch tensor + `index_copy_` (avoid per-call `torch::cat`
  alloc churn on the steady hot path). **A6** tolerant-mode wrapper or explicit CI policy gate around the b1-t1 exit
  code (currently exits nonzero under counted-not-gated policy — don't wire strict exit as the policy gate). **A7**
  steady-batch manifest / memory record (B-bucket equivalent of finalize's manifest/loader-delta discipline). **A8**
  header self-containment if the primitive moves out of `runtime/cpp/`.
- [~] **Step B3 — L40S batched-density sweep (the realized knee = F1 confirmation + Step-4 feed).** **Carry-over follow-ups from B2's paired build verdict (`reviews/B2-build-paired-verdict.md` — B3 pre-conditions):** **F1** full-corpus b2-t1 equivalent (split-case / fresh-process / streamed-reference; the B2 run was 4 ref rows due to OOM); **F2-T** telemetry hardening — emit p50/p95/p99 for the 5 timer buckets + dispatcher CPU% + stream util + queue depth + per-stream fairness spread; fix `service_wait_us` semantics to include scratch pack; clarify or fix `output_sync_us` (currently CPU-cost of enqueueing wait, not device-side wait); **F2-M** scheduler-ON vs OFF peak-memory delta per N in the knee sweep (Codex hit OOM on full-corpus b2-t1 → existing 5090 N=40 may shift down); **F3** test hardening — deterministic forced-concurrency under `lone_timeout_ms=0` (tiny test-only lone OR stronger enqueue gate); **F4** cleanup — drop unused `ep` in `set_pending_exception_locked`, nonblocking timing plumbing; **F5** EP SHA verification in C++ loader (provenance strengthening); **F6** abandoned-future event cleanup (error-path hygiene). Compile sm_89
  B∈{1,2,4} steady buckets natively on the g6e; fresh-process-per-N sweep N=36..72; report the SLO-robust knee
  (lag<500, ttfs p95≤175 / p99≤250, err≤1%, 0 mismatch), **stagger-robust at the knee N and N+4**. **Confirms the
  projected ~47-64** and sets the realized `S_native_batched` that F1 re-checks (nominal `0.83·S_native_batched/
  S_py_lock ≥1.70×`, pessimistic `0.75·… ≥1.50×`, p99 guardrail). PAIRED REVIEW (decisive density measurement). The
  batched topology then becomes the Step-3/4 scheduler/WS baseline.

- [x] **Funding-recheck F1 — PROVISIONALLY CLEARED by the STEADY-BATCH-0 projection (~2-2.5×); re-fires at Step B3 if the realized knee lands <47.** *(MF-2; user authorized building on the projection 2026-05-27.)* after Step 1b.5 + Step 1c, before freezing the Step-2/3 build scope. The
  multiplier softened from the hoped 2.0–2.25× to a zero-margin 1.8× (lower once native pays its WS tax). Report
  `nominal_realized = 0.83·S_native_candidate/S_py_lock`, `pessimistic = 0.75·S_native_candidate/S_py_lock`, and the
  remaining eng-weeks + permanent dual-stack carry. **GO-to-build without escalation only if nominal ≥1.70× AND
  pessimistic ≥1.50× with the p99 guardrail met;** otherwise mark **TECHNICAL-CONDITIONAL** → explicit human
  re-justification of the 2nd-stack bet. (Does not change the technical density gate; prevents a strategically weak
  1.5× from silently inheriting the 2.0× rationale.) **PROVISIONALLY CLEARED by the v3 STEADY-BATCH-0 projection
  (~2-2.5× → both thresholds met IF the knee realizes ~47-64); Step B3's measured `S_native_batched` is the binding
  re-check — if B3 lands <47 the multiplier re-tightens and F1 re-fires.**

- [~] **Step 2 — scheduler design + admission.** **(MF-6 split: Step 2a — invariant work was implemented as
  commit `7035d01` 2026-05-28 [`density_admission.{h,cpp}`, stale-gen generation tokens, telemetry blocks,
  `--mode admission-smoke` + `--mode stalegen-smoke`]. Step 2b — topology RESOLVED to B>1-batched central
  dispatcher per B2 verdict.)** Original spec: From the Step-1 **telemetry schema** (not a scalar knee): one
  **box-global active/admitted cap** + one **box-global backlog-COUNT cap** (ready-age is dead; sweep
  ~8/10/12), shed = **close** (count admitted, not offered; two curves). A declared numeric
  **priority-finalize-lane policy over the `num_runners=N` pool** — either partitioned
  (`N_finalize_reserved≥1`, steady-starvation p95 ≤2× no-finalize) or weighted (finalize runner-wait ≤25%
  TTFS, steady queue-wait ≤2× no-finalize). Faithful to the Python shed/priority behavior re-derived for one
  process. PAIRED REVIEW of the design before building.
  **v4 follow-ups from L40S ≥64 lift (insights 1+2 — bundle with B3 verdict commit, 2 one-line code
  changes):** **(i)** `density_admission.h:42` `active_cap` default 40 is stale; change to either
  deploy-required-no-default OR 64+. **(ii)** `batched_steady_scheduler.h:25-26` default `window_ms=10`
  → `0` (matches L40S production winner W=0 L=0; W/L stay configurable for diagnostic sweeps but the
  production setpoint is "off"). **Priority-finalize-lane** still pending: the F2-T telemetry separates
  finalize_wait from steady_wait per the Step 2a-invariant-design.md §V, so the decision can be made
  empirically post-Step-3b production load testing — not blocking pre-B3.

- [~] **Step 3 — multi-session runtime + real WS server.** **(SPLIT: Step 3a — WS-tail microbench standalone
  is IN FLIGHT [Codex job `bcmhesbv0`, fills the `ws_tail_microbench.cpp` stub Step 2a created]; Step 3c —
  stale-gen validation via injection is COVERED by Step 2a's `--mode stalegen-smoke` [all 4 scenarios PASS,
  0 stale events]; Step 3b — the real WS server is DEFERRED per `reviews/Step3-scoping.md` — 2-3 days of
  focused work, not bounded local smoke; revisit post-B3 verdict.)** Original spec: Wrap the session core
  in the scheduler + a real WS server (also closes the 1.4b interim-cadence residual). **Required before
  Step 4:** a **WS-tail microbench** (accept→ready, send→recv, recv→queue, queue→scheduler, serialize/send,
  client-recv, event-loop lag under N idle + N streaming sockets; WS overhead p95 ≤10% of TTFS or decompose,
  don't claim as runtime tail) — production needed a cooperative-yield to avoid socket starvation, so this
  is not a rounding error. **Stale-generation suppression is a Step-3 gate** (per-session generation tokens;
  close-while-inflight, reset-while-queued, reset-while-finalizer-owns-runner, final-after-shed; 0
  stale/mismatch) — so a Step-4 tail "win" can't be a dropped-final artifact.
  **v4 follow-ups from L40S ≥64 lift (insights 5+6):** **Step 3b sizing rule** — worker thread pool MUST
  support 64+ concurrent streams (size for ~80-100, not 40); on g6e.8xlarge (32 vCPU) we're at 2:1
  oversubscribe — fine for I/O-bound workers. **Step 3a smoke-and-sweep** — the microbench (once Step 3a
  lands) should be exercised up to **N=128** to characterize WS overhead at + above the actual knee
  (realized ≥64; testing 128 covers headroom + the projected post-multi-dispatcher regime). Add a
  `n_idle ∈ {0, 64, 96, 128}` × `m_streaming ∈ {1, 8, 32, 64}` follow-up matrix as a Step 3a-RUN task once
  Step 3a-BUILD commits.

- [ ] **Step 4 — realized density vs the Python stack (apples-to-apples) — the binding TECHNICAL GO.** The
  stt-benchmark / `ec2_loadgen.py` network harness driving the native WS server. **TECHNICAL GO (density-only):**
  `S_native_step4 ≥ G1_floor` admitted-successful, ceiling realization ≥0.83·S_native_step1b (below 0.75× not
  clean), `ttfs` within the ttfs SLO budget (Definitions), 0 oracle mismatch + no stale final, WER in bound, meets
  `Reject_bound`.
  **REPORT G2:** native `TTFS_spread` vs Python at matched admitted load (the user-visible finalization tail —
  reported, not gated). **Apples-to-apples MANIFEST:** re-measure Python back-to-back same L40S; pin
  corpus/commits/artifacts/loadgen-env (`LOADGEN_JITTER_MS`, `LOADGEN_STREAM_JITTER_MS`, rounds, sweep)/driver/
  server-flags (incl. `NEMOTRON_SYNC_COMPRESS`, `NEMOTRON_FINALIZE_PRIORITY`)/admission-caps/WER-config. **Extend
  the loadgen** to emit p99 + `P95−P50`/`P99−P50`. **Multi-turn:** the loadgen is one-utterance-per-connection →
  either add a multi-turn subcurve OR explicitly scope the gate to single-utterance sessions. Optional 5090
  rehearsal first; **L40S is the binding Step-4 technical GO.** (Technical GO ≠ funding GO — the ~40–60 eng-wk +
  permanent 2nd-stack strategic bet is a separate human call.)

- [ ] **Step 5 — per-target CONFIRMATION sweep (L40S, L4, DGX Spark).** Not part of the initial L40S GO. Step-1
  counter attribution PRE-PREDICTS each target (BW-bound L40S → L4 more so → "no lift" pre-confirmed; lock/launch-
  bound → L4 might lift). Write a per-target prediction table (expected limiting resource / density / tail +
  falsifying counters); confirm vs measure. L4 is OUT as a fleet target — its run is confirmation, not a
  re-decision. **Spark aarch64 preflight is mandatory** (the AOTI container has an aarch64-specific
  runner-reclamation branch, `model_container.h:718-731`): build/load + Step-0-equivalent micro-gates before any
  density sweep; budget for build-from-source. EC2 for L40S/L4.

### B3 measurement follow-ups (v4 insights 7+8 — bracket the true knee + close §II.13)

These do NOT block the B3 verdict (the realized N≥64 clears F1 decisively per v4 banner), but inform
production sizing + close the §II.13 measurement spec:

- [ ] **B3-FU-1 (insight 7): bracket the true L40S knee.** Follow-up sweep at N ∈ **{72, 80, 88, 96, 112,
  128}** with `B_max=4, W=0, L=0` (the L40S-validated winner). The realized ceiling is somewhere in this
  range (dispatcher single-thread saturates at ~80-100 per insight 4 + 5090 CPU%-extrapolation). Inform
  production sizing (active_cap default per insight 1) + the multi-dispatcher decision (Tier-4 lever per
  insight 4). EC2 ~$2.50-4/hr × ~1-2hr = $5-10. Run after the in-flight L40S sweep terminates.
- [ ] **B3-FU-2 (insight 8): burst-injection at N=64.** §II.13 spec'd a synchronized-start variant to
  surface worst-case HOL the staggered-start sweep hides. Either Codex's in-flight L40S sweep covers it
  (verify in the B3-L40S-result.md), or queue as a small follow-up. Bounded.
- [ ] **B3-FU-3 (v4 insight 9): L40S Tier-3-impact baseline measurement.** Tier 3 memory shrink was
  measured on 5090 (+11.8 → +5 GiB overhead). L40S has plenty of headroom (insight 9 estimate ~22 GiB at
  N=64) but the absolute number isn't measured. Capture during B3-FU-1 (the higher-N sweep). Informs
  multi-process MPS density planning (Tier-5).
- [ ] **B3-FU-4 (v4 insights 1+2 follow-through): apply the defaults updates.** 2 one-line code changes
  bundled with the B3 verdict commit (see Step 2 v4 follow-ups (i) and (ii)). Single Opus review post-impl.

## Rules
See PLAN_RULES.md. Step 0, 1b, 2, 3, 5 are decision-critical → PAIRED adversarial review (Codex + Opus), folded to
`reviews/`, before marking [x] (Step 2 = a paired DESIGN review before building). Step 4 → Opus review +
independent re-run.
- **Correctness before performance, non-negotiable:** no throughput number is trusted until concurrent==serial
  token/event equality (0 mismatch) holds over real decode + real finalize + same/mixed/hot-bucket + collector
  fields + stale-gen-sensitive events; semantic-WER within `WER_bound`.
- **Telemetry schema (every Step-0/1 run):** throughput, p50/p95/p99, **P95−P50 / P99−P50**, enqueue→first,
  enqueue→final, queue/runner/`.item()`/finalize waits, CUDA-event durations, Nsight/CUPTI counters (kernel
  overlap, SM occupancy, DRAM throughput, launch gaps, runner wait), GPU+CPU-core util, loadgen/server health,
  `num_runners`/stream-mode/corpus-SHA/artifacts/topology in artifact names.
- **A STOP must be CORROBORATED** (protects a viable project from a harness-bug false STOP): 3 runs (CV ≤10%) +
  the negative control detects serialization + the topology sweep (`num_runners=1`/=N/fallback loaders/MPS) all
  fail + counter attribution to a HARDWARE limit (not lock/default-stream/un-wired-stream) + harness health
  logged + the CUDA-graph-of-AOTI fallback tried. A passing fallback topology is a PIVOT, not a STOP.
- **Honesty (Phase-1 lesson):** real decode (no mock), SLO-robust not keep-up, attribute the knee to a resource,
  report the spread; don't overclaim. If a step's bar isn't met, mark the residual; correct any prior over-claim.

## Live findings & lever inventory (2026-05-26 — Step-1a in progress)
- **Warm 5090 floor (commit e5f2753):** SLO-robust knee = **N=4** (TTFS p95 14ms ≪ 175ms budget, lag_p95
  negative, 0 mismatch), finalize **warm ~8ms** (Python-class). The earlier **234ms@N=1 was a per-bucket
  COLD-START artifact** (CUDA-12 lazy module load, ~225ms first-launch, universal to every AOTI loader; the steady
  encoder shows the same max spike amortized over chunks) + a 6-sample p95=max — **FIXED by per-bucket warmup +
  ≥20 samples** (`density_main` `CUDA_MODULE_LOADING=EAGER` + per-worker bucket warm). **BINDING RESOURCE = MEMORY,
  and W1 (paired, `reviews/phase2-W1-finalize-memory-FOLDED.md`) PINNED it to `enc_first.ts` — a 2.48 GiB
  full-fp32 encoder loaded ONCE PER WORKER** (`make_worker_context`, `density_main.cpp:629`; ≈ the whole
  ~2.51 GiB/stream) for one first-chunk forward. **NOT the finalize buckets** (they share ONE 2.30 GiB constants
  set, `loader_delta=0/bucket`; proven by the 0c control 8-vs-16-finalize-runners→same ~30.8 GiB + activation
  scaling as N-workers). Cause: the per-thread-handles concurrency choice applied to a 2.48 GiB module; Python has
  NO enc_first (one shared encoder via drop_extra). ⟹ N=4 is a memory-capped FLOOR from **enc_first duplication**,
  not the density verdict. **FIX = dedup enc_first** (Fix-2 shared+locked ref → fast confirm; Fix-1 fold into the
  shared steady AOTI loader → clean + closes the first-chunk-TorchScript residual). **Est. new knee: 5090 N=4→~40-45,
  L40S ~13→~60-69** (per-stream 2.51→~0.4 GiB; ±20%, confirmable ~5min). **Run sweeps FRESH-PROCESS-PER-N** (same-proc
  used_before grows 4.98→9.70 GiB, inflates OOM). **enc_first dedup is a PREREQUISITE for the L40S sweep** (else it
  hits the same ~N=13 enc_first wall + mis-measures the binding resource).
- **Autotune-ON (max_autotune+coordinate_descent) T1-FAILED CATASTROPHICALLY (MEASURED):** steady-only full-1000
  shadow vs eager = **995/1000 token-divergent, WER 3.68%→82.77% (+79pp)**, first divergence at chunk 2 (`cache_t`
  diff already 22.8; drift plateaus at a catastrophic mean ~30). Confirms the diagnosis: cache_t 10.27 =
  **precision-policy divergence from eager's TF32-reduced path** (==the earlier `fp32_highest` knob-matrix drift;
  convs went ALLOW_TF32=True), NOT pure reduction-order. ⟹ **aggressive autotune SHELVED.** The **precision-matched
  rung R1a** (`reviews/codex-autotune-params-strategy.md`, match eager's TF32 policy) is UNTESTED and only worth
  trying IF the L40S is contention-bound (W4) AND it passes T1 first — but autotune speeds COMPUTE, so it is **moot
  while the binding resource is MEMORY**. **Net: autotune is OFF the critical path** (contingent Tier-2 at best).
- **Finalize is ~half sync+glue, not compute** (phase-split telemetry): `enc_len_sync` **3.67–6.47ms** (a blocking
  D2H `.item()` on the encoder output length — likely **host-eliminable**, the length is geometry-deterministic)
  + `glue` **3.77–10.94ms** (event/text/FORK_ASSERT) vs `aoti_run_cuda` 6.59–12.49ms. No `pin_memory`/`non_blocking`
  in the copies — but volumes are tiny (mel ~8KB, scalars) so it's the **sync stall, not bandwidth.**

### Lever inventory — TIERED by the binding resource
**L40S post-W3 re-tiering (2026-05-27 — SUPERSEDES the 5090-era `enc_first`-memory tiers; that 5090 memory wall was
RESOLVED by W1 Fix-2, see Live findings + the 1a row).** Two independent analyses (`reviews/{opus,codex}-scaling-
above-36-levers.md` → `CHECKPOINT-scaling-above-36.md`) attribute the N=36 ceiling to **host-side serialization**
(36× B=1 forwards + blocking per-token `.item()` D2H syncs → 17–27% GPU idle) — NOT memory (0.035 GiB/stream), NOT a
saturated GPU, NOT the AOTI exec lock (it's a `std::shared_lock`, `model_container.h:82`), NOT `enc_first`. **Ceiling:
the util-mean extrapolation is INVALID** (N=36 mean 73% but p50/p95=**91/96**; staggered N=40 FAILS at 72.6%) — "≈44–48"
is an upper bound pending the 1c-B Nsight attribution, NOT a forecast. Multiplier **1.8× at-bar/zero-margin — and
INFLATED** (native ttfs is server-side pre-WS vs Python's client-over-WS; the honest WS-paid number is lower — lock it
in Step 1b.5/MF-1). >36 is load-bearing for Step-4 *if* S_py_lock ≥19.
- **Tier 1a — decode/`enc_len` scalar-sync removal** (device argmax + fused blank flag; host-computed `enc_len`):
  **CORRECTED (FACT-2, verified): BOUNDED ~17ms.** `decode_item_wait` p95 is flat ~17ms across N while `decode_wall`
  explodes to 309ms@40 — the explosion is joint/predict GEMMs **queued under cross-stream contention**, NOT the
  `.item()` sync. So this is a **TTFS/tail cleanup (~17ms + enc_len ~22ms), NOT the decode-contention fix**, and
  **1c-0 is expected to PIVOT.** Cheap + T1-gated (do it for TTFS), but it does NOT reach `S_native_req` alone.
- **Tier 1b — `finalize_num_runners` > 2 + priority-finalize-lane:** the `min(N,2)` pool serializes synchronized
  finalize bursts; the wait is mis-attributed into aoti-time (so "finalize_wait=0" hides it). **~free** (buckets
  share one constants set). Fold into Step 2.
- **Tier 2 — cross-stream batched STEADY encoder — VALIDATED #1 LEVER (STEADY-BATCH-0 PASSED 2026-05-27):** the
  binding is BW-bound steady-encoder weight-streaming (profiling), and batching is the **only** lever that lowers the
  aggregate byte floor (load weight once, reuse across B). **OPPORTUNITY** (fill mean B 2.7-4.4 @ 8-12ms over
  N=36-56) AND **SPEEDUP** (per-row B=2 **0.62×** / B=4 **0.38×**, byte-exact per-row) both PASS — the prior
  "B≈1.5-2, batching dead" sim was *finalize* bursts, not the *steady* 160ms cadence. Projected **37 → ~47-64**.
  **Build = Steps B1-B3** (the green-lit Phase-3). Decode-batching folds in (the `decode_wall` queue clears once the
  steady GEMM stops monopolizing the BW).
- **Tier 3 — steady-encoder CUDA-graph** (the shipped finalize-graph primitive) + the autotune-ON ladder
  (T1-blocked): help launch/dispatch, NOT GEMM time (steady is BW-bound). **Nsight-gated** (only if launch gaps ≥15%).
- **De-prioritized — `enc_first` K-pool / AOTI fold — but TRAFFIC-CONDITIONAL (MF-5):** `lag`-not-`ttfs`,
  stagger-erased (640→10ms), a harness artifact for the ~12s sessions. Keep de-prioritized **ONLY IF** target
  traffic has p95 session lifetime ≥60s OR first-chunk starts ≤5% of steady chunks at the admitted cap
  (barge-in / reconnects / multi-stream-per-call / greeting snippets resurrect it); else a Step-2 hygiene gate.
  ~0 prod *density* regardless (don't count it as steady density unless N=40+ actually becomes SLO-robust).
- **NOT levers (ruled out):** a finalize CUDA graph for the "234ms" (cold-start; warm=8ms); autotune for memory;
  aggressive `max_autotune`+`coordinate_descent` (breaks T1).

### Scoped next work (priority / dependency)
- **W0 DONE:** autotune-on T1-FAILED (995/1000); finalize-234ms = cold-start (warmup-fixed); W1 root-caused the
  memory wall to `enc_first` dup (all paired).
- **W1 = `enc_first` DEDUP — Fix-2 DONE ✅ (commit 99fbba3): 5090 knee N=4 → ≥N=32 SLO-robust, memory wall gone,
  thesis validated.** **W1b (next, CRITICAL PATH + clean-L40S prerequisite):** **Fix-1** (fold first-chunk into the
  shared steady AOTI runner pool = **LOCK-FREE** via num_runners — removes the new enc_first-lock bottleneck (p95
  245ms@N=32) + closes the first-chunk-TorchScript residual) **+ the unique-streams(>32) harness fix** (getStreamFromPool
  caps at 32) → re-sweep → the TRUE 5090 knee (est ~40-45+). Both are prerequisites for a clean L40S sweep too.
- **W2 (Tier-3, UNBLOCKED, cheap):** eliminate the `enc_len` D2H `.item()` (verify geometry-deterministic →
  host-compute). Finalize+steady latency/tail win.
- **W3 (THE GATE, GATED ON W1 + the S3 upload→g6e):** the L40S apples-to-apples density sweep. **Must run AFTER the
  enc_first dedup** (else it hits the same ~N=13 enc_first wall + mis-measures). Fresh-process-per-N → the real
  streams/box number + the binding-resource attribution → decides Tier-2.
- **W4 (Tier-2, contingent on W3 = contention-bound):** autotune ladder R1a (precision-matched, T1-gated) per the
  strategy doc.
- **W5 (Tier-3, lower):** glue trim; pinned/`non_blocking` copies; cross-stream transfer/compute batching (Step-2/3).

## Progress
| Step | Status | Commit | Notes |
|---|---|---|---|
| 0 cheap kill-gates (5090) | done | d77bede, 92b8a9f | density_main.cpp, paired-reviewed + gate-soundness fixes. **conjunct-2 binary = YES.** 0a PASS (~1.7× encoder-only; serial-oracle 0 mismatch; loader-delta 2.309GiB flat N=1→16 = ONE weight copy; controls: single-runner≡N=1, mutex −22%, default −30%). 0b PASS (identity 0/200; scalar-locality sentinel: `.item()` doesn't drain unrelated streams). 0c PASS (8-worker same+mixed 0 mismatch after finalize-pool fix: cap min(workers,2)/bucket + preload-needed + hot=workers; 10-worker OOM → finalize memory-tight ~30.8/31.3GiB on 5090, L40S has headroom). **ATTRIBUTION: the plateau is GPU CONTENTION (encoder saturates 5090 ~N=4), NOT the execution lock** (kernel p50 5.28→13.63→28.54ms with N). nsys absent → kernel-duration-vs-N substitutes. Density magnitude (full session w/ host-bound decode) = Step 1a. |
| 1a 5090 spend-control | DONE (PASS) | e5f2753, 99fbba3, +W1b' | **TRUE 5090 knee = N=40 SLO-robust (TTFS p95 82.5ms ≪175; lag_p95 −60ms; 0 mismatch at EVERY N=1..64); N=48 first non-SLO. BINDING = GPU CONTENTION** (not memory 19.8/32GiB, not streams unique-to-64, not CPU 3.2/32, not the enc_first lock — lock p95 300ms@N=40 but TTFS 82.5/lag −60, so NO K-pool needed). Full arc: N=4 (enc_first-dup memory-capped) → N=40 (compute-bound) after the Fix-2 dedup + unique-streams. ~2.5× per-process vs Python 5090 (~14-16/proc). Spend-control PASS → L40S. enc_len_sync(25ms)+glue(50ms) grow → W2 could nudge higher. Earlier: 234ms=cold-start(fixed); autotune T1-FAIL(995/1000, shelved); Fix-1 AOTI-first-chunk T1-blocked(stays TorchScript). |
| 1b L40S ceiling gate | DONE (PASS) | run#13/#14 logs in runtime/artifacts/l40s_w3_logs/ | **L40S native knee = N=36 SLO-robust (true ∈[36,39]); N=40 first non-SLO → PASS** (≥34 floor; ~1.8-2.25× Python S_py~16-20; at-bar vs S_py=20). g6e.8xlarge (32 vCPU/L40S 48GB), sm_89 autotune-OFF, 8 sessions/worker, fresh-process-per-N. **BINDING = keep-up/GPU-compute**: lag p95 −35ms@36 → +1337@40; ttfs p99 147@36 → 720@40 (budget 175/250); finalize_wait 0; per-stream mem 0.035 GiB (NOT memory — could hold 1000s); CPU 5.9/32. steady_gpu+finalize_gpu p50 ~18→~28ms + decode_wall p95 17→309ms at 40 = GPU-scheduling + decode tail. **N=32 control == run#9** (finalize_gpu 14ms, ttfs p99 110) validates the rig. **STAGGER-ROBUST (run#14)**: a 10s per-worker start-stagger improves margins (N=36 ttfs p99 147→50, lag −35→−119) but N=40 STILL collapses (lag +1149) → genuine compute saturation, NOT a synchronized-burst artifact (closes the reviewers' false-fail objection). **PAIRED REVIEW: Codex + Opus both GO** (reviews/codex-l40s-w3-profile.md, codex-l40s-knee-verdict.md). Caveats: exact knee bracketed [36,39] (37-39 untested); at-bar vs high-end S_py=20. **SAME-BOX PYTHON RE-MEASURE DONE 2026-05-27 (spy_*.json): S_py≈20 single-proc @ ttfs p50 ~42ms → 1.8× CONFIRMED AT-BAR / ZERO MARGIN** (the prod 245ms was an MPS-multiproc artifact, not inherent — one proc ≈ same density at ~6× lower ttfs). ⟹ pushing knee >36 is **load-bearing for Step-4** (see Step 1c + CHECKPOINT-scaling-above-36.md). aggregator summary prints ttfs/lag 0.0 + binding=not_observed (cosmetic parser bug; per-N rows authoritative). **RIG (cross-arch sm_89, cu128 wheel + CUDA-13):** share-ONE-bundle context (0.8s — concurrent 668MB jit::load LIVELOCKS on torch's global registry; was the ~60min/N hog, not warmup); cudart-12 unify (CUDA-13 vs torch cudart-12 deadlocked the multi-thread path); full per-worker finalize warmup (the lean per-bucket-runner variant under-warmed 34/36 worker streams → false-fail); SKIP_EPS_VERIFY (90GB SHA ~12min); arch_list/venv/cmake-source fixes. autotune-ON shelved (T1-broken, moot while compute-bound). T1: 1000/1000 finals byte-exact vs gold (run#9); 5/1000 interim-event-timing drift (WER-neutral, counted not gated). |
| 1b.5 S_py_LOCK | DEFERRED (non-blocking) | | **De-blocked from the build (user call 2026-05-27); now feeds Step-4 apples-to-apples, not the build start.** **MF-1 — (orig) gates 1c scope+funding.** Re-measure Python same-box, Step-4 manifest, **WS-matched**, repeats≥10, p50/p95/p99, levels {16,18,20,22,24}. S_py noisy/non-monotonic/**not-apples-to-apples** (147ms@conc16 PASSES; first fail conc24@249). Set **S_native_req=ceil(max(34, 1.80·S_py, 1.50·S_py/0.83))**. ≤18→1c optional; 19-21→proceed; ≥22→F1. In-flight apples chart = first half. |
| 1c push knee >36 (sync/batch triage) | RESOLVED → batching | | **Triage answered by L40S profiling + STEADY-BATCH-0, NOT the 1c-0 sequence.** nsys+ncu (paired) → binding = **BW-bound steady-encoder weight-streaming** (88% GEMM / 72% DRAM / 15% occ = BW-wall, not idle SMs); the 1c-B Nsight attribution routed straight to **batching** (the only byte-floor lever). 1c-0 sync-ablation = moot (sync bounded ~17ms, FACT-2). **STEADY-BATCH-0 PASSED both conjuncts:** OPPORTUNITY fill mean B 2.7-4.4 @ 8-12ms (N=36-56); SPEEDUP per-row B=2 0.62×/B=4 0.38× byte-exact (`steady_b_artifacts/bench_out.log`). ⟹ knee proj 37→~47-64; build = B1-B3. reviews/profiling-paired-verdict.md + {opus,codex}-l40s-profiling-analysis.md (paired). |
| B1 batched-steady mechanism+T1 (5090) | done (PASS-by-policy) | 3887cb3 | PASSED both runs: K=3/B=4 grouping 1007 rows/336 cases (2 interim event drifts, 0 token); K=2/B=2 coverage closure 1007 rows/502 cases (4 interim drifts, 0 token); 0 enc_len/cache_len mismatches; finals byte-exact; flag-OFF preserved. Combined 0/2014 token divergences across all 3 buckets. PAIRED REVIEW (opus+codex) → reviews/B1-paired-verdict.md PASS-by-policy. Audit follow-ups A1-A8 folded into B2 step body. Codex job: codex-jobs/step-B1-b4ml9h322.log. |
| B2 batching scheduler+integration (5090) | done (PASS-with-followup) | 0925fa6 | Central dispatcher built per binding spec §II.1-II.14. Spec faithful: §II.2 bidirectional CUDA sync ✓; §II.4 explicit nullable integration (no globals) ✓; §II.10 sealed loader fail-closed ✓; §II.11 scratch + index_copy_ ✓; §II.12 manifest fail-closed (with built-in SHA256+JSON parser) ✓; §II.8 fault tolerance + process exit ✓. b2-t1: 6/6 cases PASS, 0 token + 0 event divergences (1007 rows from 4 ref over forced K2/K3-padded/B4/staggered/Bmax1-control); A1 outcome B (SHAs differ but tensors bit-identical → OFF stays on PRODUCTION B=1 per §II.9). OFF-path smoke 20 sessions N=4 mismatches=0. Scope reductions (4 ref rows, 20 OFF sessions) flagged as B3 pre-conditions (F1, full-corpus b2-t1). Codex F2-T telemetry hardening + Opus F2-M memory headroom both pre-knee-remeasure. PAIRED REVIEW (design fork + build) → reviews/B2-design-paired-verdict.md + reviews/B2-build-paired-verdict.md. |
| B3 L40S batched-density sweep | todo | | sm_89 B∈{1,2,4} buckets; fresh-proc-per-N N=36..72; SLO-robust knee + stagger-robust; confirms ~47-64 → sets S_native_batched for the F1 re-check. PAIRED (decisive). |
| F1 funding recheck | provisional CLEAR | | **MF-2** — GO-to-build only if nominal 0.83·S_native/S_py≥**1.70×** AND pessimistic 0.75·…≥**1.50×** + p99 guardrail. **STEADY-BATCH-0 projection (~47-64 → ~2-2.5×) provisionally clears both; Step B3's realized knee is the binding re-check (re-fires if B3 <47).** User authorized building on the projection (2026-05-27). |
| 2 scheduler+admission design | partial (2a done, 2b resolved) | 7035d01 | **Step 2a invariant work committed `7035d01`** — `density_admission.{h,cpp}`, stale-gen generation tokens, telemetry blocks (`admission`+`stale_gen`+`ws_tail` schema), `--mode admission-smoke` + `--mode stalegen-smoke` (all 4 scenarios PASS). **Step 2b topology RESOLVED** = B>1-batched central dispatcher per B2 verdict. **v4 follow-ups**: 2 one-line default updates (active_cap=40 stale → 64+; window_ms=10 → 0). Priority-finalize-lane: telemetry separates finalize_wait from steady_wait per design §V; empirical decision post-Step-3b. |
| 3 multi-session + real WS | partial (3a in flight, 3c done, 3b deferred) | 7035d01 (3c via Step 2a) | **Step 3c stale-gen validation COVERED by Step 2a's `--mode stalegen-smoke`** (4/4 scenarios PASS, 0 stale events, drops counted exactly: 3 encode + 3 finalize_output = 6 total). **Step 3a WS-tail microbench IN FLIGHT** (codex job `bcmhesbv0` — fills the stub Step 2a created; boost::beast standalone echo server + loadgen + per-stage timing). **Step 3b real WS server DEFERRED** per `reviews/Step3-scoping.md` (2-3 days focused work; revisit post-B3 verdict). **v4 sizing rule**: worker thread pool sized for ~80-100 concurrent (not 40). WS-tail microbench RUN should sweep N up to 128. |
| 4 realized density (apples) | todo | | TECHNICAL GO ≥G1_floor; G2 TTFS_spread reported; manifest + re-measured baseline. **AT-RISK (2026-05-27): S_py≈20 → 1.8× at-bar; after the 0.83 haircut 0.83·36≈30 vs 1.5·20=30 = at-bar → push knee >36 (Step 1c) for a robust GO.** |
| 5 per-target confirmation | todo | | confirm Step-1 attribution; Spark aarch64 preflight; EC2 |
