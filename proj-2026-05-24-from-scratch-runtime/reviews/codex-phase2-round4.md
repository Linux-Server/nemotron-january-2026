# Codex Phase 2 Review - Round 4 Decision Structure

Scope: Round-4 decision tree review of `PHASE2-PLAN.md`, building on the three folded paired-review rounds. Focus: G1 system utilization / density and G2 P50-P95 tail spread.

## BLOCKER

### B1. The plan still needs one pre-registered GO/STOP tree; `meaningfully` is not a gate.

`PHASE2-PLAN.md:25-32` still gates Step 1 on "meaningfully" while the project-level decision is an L40S density decision, and Step 4 is where the like-for-like native-vs-Python result is measured (`PHASE2-PLAN.md:39-41`). Round 1 already identified the numeric L40S bar as `>=1.5x / >=~28` streams per box (`phase2-round1-FOLDED.md:8-12`), and Round 4 asks for a pre-registered tree over Step 0, Step 1a, Step 1b, and Step 4 (`phase2-review-brief-round4.md:9-12`).

Recommended plan edit: replace the current Step 1/Step 4 prose with this decision tree.

#### Definitions used by every gate

- `S_py_L40S`: fresh Python baseline, SLO-robust successful streams per L40S box, re-measured back-to-back under the Step 4 apples-to-apples manifest.
- `G1_floor = max(28 streams/box, 1.50 * S_py_L40S)`.
- `S_native_step1b`: Step 1b native L40S compute-ceiling streams/box, using real decode/finalize but without the production WS/server overhead.
- `S_native_step4`: Step 4 native L40S realized admitted successful streams/box, with production scheduler, WS server, stale-generation handling, admission, and shedding.
- `TTFS_spread = ttfs_p95 - ttfs_p50`, server-side `vad_stop -> final` unless literal first-token TTFT instrumentation is added.
- `TTFT_spread = ttft_p95 - ttft_p50`, first interim/token latency. If Phase 2 keeps using `TTFS` as proxy, the plan must explicitly rename G2 to the server-side finalization tail.
- `WER_bound`: native semantic-WER at the claimed knee must be `<= max(WER_python + 0.5 percentage points, 1.10 * WER_python)`, with the same corpus, tool, model, prompt, retry policy, and scoring manifest.
- `Reject_bound`: at the claimed shed-protected knee, intentional WS 1013 rejects must be `<=10%` of offered sessions, and non-intentional errors must be `<=1%` of admitted sessions. The no-shed curve must also be reported.
- A STOP is valid only after the false-STOP checks in B5 pass.

#### Step 0 kill-gates: cheap native feasibility before any EC2 spend

Step 0a - steady AOTI runner pool:

- Metric: one shared AOTI loader with `num_runners=N`, explicit per-worker CUDA streams, `run(inputs, stream_handle)` path, logged topology. Round 2 showed this is the evidence-backed topology (`phase2-round2-FOLDED.md:11-31`).
- PASS: at `N=4`, throughput is `>=1.30x` the `N=1` same-harness baseline; at `N=2`, throughput is `>=1.15x`; concurrent output equals serial output with `0` token/event mismatches; profiler trace shows kernels submitted on `>=2` streams; peak GPU memory at `N=4` is `<1.35x` the `N=1` peak and does not include a second full constants/weights copy.
- STOP candidate: throughput is `<1.10x` at `N=2` or `<1.20x` at `N=4`, or correctness has any mismatch, or memory shows per-runner weight duplication.
- Action: PASS continues to Step 0b. STOP candidate triggers the corroboration protocol in B5 before stopping. Values between STOP and PASS are TUNE/RETEST, not GO.

Step 0b - decode and per-thread object ownership:

- Metric: per-worker `SessionState`, per-thread TorchScript/preproc handles, explicit telemetry for `.item()` / default-stream waits, and no shared mutable decode state. Round 2 narrowed the concurrency risk to shared model objects and `.item()` synchronization (`phase2-round2-FOLDED.md:61-82`).
- PASS: concurrent decode/finalize-visible event stream equals the serial oracle with `0` token/event mismatches; p95 `.item()` / scalar-sync wait is `<=5%` of per-chunk GPU elapsed time at the Step 0 test knee; default-stream negative control is at least `15%` worse in throughput or shows global synchronization in trace.
- STOP candidate: any correctness mismatch, scalar-sync p95 `>10%` of per-chunk GPU elapsed time, or trace shows default-stream serialization in the supposedly per-stream path.
- Action: PASS continues to Step 0c. STOP candidate requires B5 corroboration.

Step 0c - finalize fork/clone and hot-bucket path:

- Metric: same-bucket and mixed-bucket finalize tests, real fork/clone finalize path, finalizer wait telemetry, and equality against the serial oracle. Round 1 flagged that the 0.1b mock omitted real finalize (`phase2-round1-FOLDED.md:33-36`), and Round 2 kept finalize as a secondary but real hazard (`phase2-round2-FOLDED.md:35-48`).
- PASS: same-bucket and mixed-bucket tests have `0` token/event/WER regressions; finalize p95 runner-wait at the Step 0 knee is `<=25%` of total `vad_stop -> final` time; no per-bucket aliasing or stale-generation leak is observed.
- STOP candidate: any final text/event mismatch, any stale final after close/reset/shed, or finalize p95 runner-wait `>50%` of total `vad_stop -> final` time before scheduler priority is even introduced.
- Action: PASS permits Step 1a. STOP candidate requires B5 corroboration.

#### Step 1a 5090 gate: pass-to-L40S, not a project GO

`PHASE2-PLAN.md:25-32` starts on 5090, but Round 1 and Round 2 both concluded the actual decision gate is L40S, not 5090 (`phase2-round1-FOLDED.md:8-12`, `phase2-round2-FOLDED.md:35-41`). The 5090 therefore needs a numeric spend-control bar, not the final project threshold.

- PASS to L40S Step 1b: real-decode 5090 ceiling multiplier is `>=2.00x` vs its own `N=1` same-harness baseline, with Step 0 correctness still at `0` mismatches, semantic WER within `WER_bound`, p95 server-side `vad_stop -> final <=300ms`, p99 `<=500ms`, and `TTFS_spread <=1.25x` the `N=1` spread at matched admitted load.
- TUNE/RETEST: multiplier is `>=1.50x` but `<2.00x`, with profiler attribution pointing to a fixable harness/topology issue rather than GPU saturation. Do not start L40S EC2 until either the multiplier reaches `2.00x` or a written exception explains why 5090 is non-predictive.
- STOP candidate: multiplier `<1.50x`, p95 `>300ms`, p99 `>500ms`, WER outside `WER_bound`, or any correctness mismatch.
- Action: PASS buys the L40S Step 1b experiment. STOP candidate requires B5 corroboration. This is not a technical GO for Phase 2.

DISAGREE with any Round-1 interpretation that `>=1.5x` on the 5090 is enough to proceed. If the headroom-rich local GPU cannot clear `2.00x` with real decode, the L40S has too little margin for the Step 4 scheduler/WS haircut. This is stricter than Round 1's "5090 smoke" wording (`phase2-round1-FOLDED.md:10-12`) and makes the spend-control role explicit.

#### Step 1b L40S gate: ceiling must exceed the final realized gate

Round 3 correctly separated Step 1b as the ceiling and Step 4 as the realized result (`phase2-round3-FOLDED.md:69-70`). Therefore Step 1b cannot merely equal the final `1.5x` bar.

- PASS to Steps 2-3 implementation: `S_native_step1b >= max(34 streams/box, 1.80 * S_py_L40S)`, with real decode, real finalize, `0` correctness mismatches, semantic WER within `WER_bound`, p95 server-side `vad_stop -> final <=300ms`, p99 `<=500ms`, and `TTFS_spread <=1.10x` Python's Step 4 baseline spread at matched admitted load. The `34` floor is the rounded `28 / 0.83` ceiling needed to preserve the original `>=28` realized floor after roughly a 17% scheduler/WS haircut.
- CONDITIONAL / funding-risk hold: `S_native_step1b >= G1_floor` but `< max(34, 1.80 * S_py_L40S)`. This proves the primitive may clear density in a best-case harness, but it does not justify a full scheduler/server build as a technical gate. Only proceed if a human explicitly accepts the haircut risk and the next step is a narrow de-risking prototype.
- STOP candidate: `S_native_step1b < G1_floor`, WER outside `WER_bound`, p95 `>300ms`, p99 `>500ms`, or any correctness mismatch.
- Action: PASS enters Step 2/3. CONDITIONAL requires explicit risk acceptance. STOP candidate requires B5 corroboration.

DISAGREE with the prior folded shorthand that an L40S Step 1b result of `>=1.5x / >=~28` alone is a hard pass (`phase2-round2-FOLDED.md:94-98`). That is the final realized gate, not the ceiling gate. Keeping it as the Step 1b pass threshold would create a predictable false GO once scheduler, WS, stale-generation, and shedding overhead appear in Step 4.

#### Step 4 realized gate: the actual technical GO/STOP

Step 4 is the only binding like-for-like project result because it uses the native WS server and the same harness/WER/hardware against the Python baseline (`PHASE2-PLAN.md:39-41`). Round 3 also requires two curves: no-shed SLO knee and admitted-through-shed capacity (`phase2-round3-FOLDED.md:59-63`).

- TECHNICAL GO: all of the following are true on L40S:
  - Density: `S_native_step4 >= G1_floor`.
  - Ceiling realization: `S_native_step4 >=0.83 * S_native_step1b` or, if below that, it still clears `G1_floor` and the lost margin is attributed to a named scheduler/WS cost with a bounded remediation plan. Below `0.75 * S_native_step1b` is not a clean GO even if the raw density floor barely clears.
  - Tail: at the claimed knee, native `TTFS_spread` is improved by at least `max(10ms, 10%)` versus Python at matched admitted load. If Python spread is already `<50ms`, native may be considered pass if it is within `+5ms` and p95 is no worse.
  - Absolute SLO: native p95 server-side `vad_stop -> final <=300ms` and p99 `<=500ms`.
  - Correctness: `0` token/event equality mismatches on the oracle corpus, no stale final/interim after close/reset/shed, and semantic WER within `WER_bound`.
  - Shedding: admitted-through-shed result meets `Reject_bound`; no-shed curve is also reported and not used as the only capacity number.
- TECHNICAL STOP: density `<G1_floor`; or tail spread fails the G2 criterion; or p95/p99 SLO fails; or WER/correctness fails; or intentional rejects are `>10%` at the claimed knee; or non-intentional errors are `>1%` of admitted sessions.
- Action: TECHNICAL GO means the runtime passed the Phase 2 technical gate. It is not an automatic funding GO; see B6.

DISAGREE with the original density-only framing as sufficient for the current round. Round 4's brief correctly says the original pre-registration was density-only, but G2 was added as a second goal (`phase2-review-brief-round4.md:16-20`). The updated GO must be conjunctive: density and tail. Because that changes the original pre-registered threshold, the plan should call out that the user must ratify the added tail conjunct before Step 4 runs.

### B2. G1 and G2 traceability is not yet end-to-end; literal TTFT is currently orphaned.

The plan says Phase 2 must prove density and spread, but the current line-level plan names density more concretely than tail (`PHASE2-PLAN.md:46-49`). Round 1 warned that literal TTFT includes VAD/WAN components outside the native runtime's control (`phase2-round1-FOLDED.md:68-71`), while Round 3 says the existing EC2 loadgen measures `TTFS` / `vad_stop -> final`, not literal first-token TTFT (`phase2-round3-FOLDED.md:53-58`).

Recommended plan edit: add this traceability table to the plan and use the exact metric names in output schemas.

| Goal | Step | Metric | Threshold | Status |
| --- | --- | --- | --- | --- |
| G1 density primitive exists | Step 0a | AOTI runner throughput `N=2`, `N=4`; memory-flat; CUDA streams | `>=1.15x` at `N=2`, `>=1.30x` at `N=4`, peak memory `<1.35x N=1`, no extra full weight copy | Measures primitive feasibility |
| G1 decode/finalize do not invalidate primitive | Step 0b/0c | oracle equality, scalar sync %, finalize runner wait | `0` mismatches; `.item()` p95 `<=5%` chunk GPU elapsed; finalize wait p95 `<=25%` TTFS | Measures concurrency safety |
| G1 spend-control on local GPU | Step 1a 5090 | real-decode ceiling multiplier vs `N=1` | PASS `>=2.00x`; STOP `<1.50x` | Non-binding proxy |
| G1 L40S ceiling | Step 1b L40S | compute-ceiling streams/box | PASS `>=max(34, 1.80*S_py_L40S)`; STOP `<G1_floor` | Binding build gate |
| G1 realized | Step 4 L40S | admitted successful native streams/box | TECH GO `>=G1_floor = max(28, 1.50*S_py_L40S)` with `Reject_bound` | Binding technical gate |
| G2 server-side tail reference | Step 1a/1b | `TTFS_p50/p95/p99`, `TTFS_spread` | 1a `<=1.25x N=1 spread`; 1b `<=1.10x Python spread`; p95 `<=300ms`, p99 `<=500ms` | Early warning only |
| G2 realized server-side tail | Step 4 | native-vs-Python `TTFS_spread` at matched admitted load | improve by `>=max(10ms,10%)`; if Python spread `<50ms`, no worse than `+5ms`; p95 `<=300ms`, p99 `<=500ms` | Binding technical gate |
| Correctness validity | Step 0-4 | token/event equality, stale generation, semantic WER | `0` oracle mismatches; WER within `WER_bound` | Validates metric meaning |
| Admission validity | Step 4 | rejected/offered and errors/admitted | 1013 rejects `<=10%`; non-intentional errors `<=1%`; no-shed curve reported | Prevents shed-cheating |

Orphaned goal: if the user really means client-visible first-token TTFT, not `vad_stop -> final` TTFS, it remains orphaned until Step 3/4 add first-interim/token timestamps and a `TTFT_spread` gate. The corrected plan can either (a) rename G2 to server-side finalization tail (`TTFS`) and say VAD/WAN are out of scope, or (b) add literal first-token TTFT instrumentation and require both `TTFT_spread` and `TTFS_spread` to pass. I recommend option (b) for honesty: bind Step 4 on `TTFT_spread` and keep `TTFS_spread` as the attribution metric.

### B3. Scheduler/admission and WS confounds must be preconditions for the Step 4 gate, not implementation details.

Round 3 found that the architecture changes from external LB plus K Python processes to one native process with threads, so admission and priority-lane behavior must be re-derived (`phase2-round3-FOLDED.md:8-18`). It also found Step 3 can create a false G2 win or loss through WS tail and stale-generation handling (`phase2-round3-FOLDED.md:40-50`).

Recommended plan edit: Step 4 is invalid unless the Step 2/3 preconditions below are green.

- Admission precondition: one box-global active/admitted cap and one box-global backlog-count cap are enabled and logged. The backlog-count cap must be swept across at least `8, 10, 12` or centered around the Step 1b knee if that knee makes those values obsolete. Report offered, admitted, rejected, reject code, queue depth, active streams, and successful streams; rejected sessions are never counted as density.
- Priority-lane precondition: declare one of two numeric policies before running Step 4:
  - partitioned pool: `N_finalize_reserved >=1`, `N_steady >= N_total - N_finalize_reserved`, and steady starvation p95 queue wait `<=2x` the no-finalize p95; or
  - weighted priority: finalize p95 runner-wait `<=25%` TTFS and steady p95 queue wait `<=2x` the no-finalize p95.
- WS-tail precondition: accept-to-ready, send-to-recv, recv-to-queue, queue-to-scheduler, final-serialize/send, client-recv, and event-loop lag are reported. WS/server overhead p95 must be `<=10%` of total server-side TTFS p95 or the G2 result must be decomposed and not claimed as runtime tail.
- Stale-generation precondition: close-while-inflight, reset-while-queued, reset-while-finalizer-owns-runner, and final-after-shed tests all pass with `0` stale finals/interims and `0` event/token mismatches.

Without these preconditions, Step 4 can falsely STOP G1 by counting intentional shed as errors, falsely GO G2 by dropping slow finals, or falsely STOP G2 by measuring event-loop tail instead of runtime tail.

### B4. The tail criterion is a real change to the project gate and must be ratified before the run.

The Round 4 brief is explicit: original pre-registration was density-only, but G2 is a second goal added later (`phase2-review-brief-round4.md:16-20`). The plan should not silently move the goalpost after results arrive.

Recommended plan edit:

- State: "Phase 2 technical GO now requires both G1 density and G2 tail. This is stricter than the 0.0 density-only gate and is intentionally adopted before running Step 4."
- Use the Step 4 G2 gate from B1: native `TTFT_spread` and/or `TTFS_spread` improves by `>=max(10ms,10%)` versus Python at matched admitted load; if Python spread is already `<50ms`, native must be within `+5ms` and p95 no worse.
- State the fallback: if user declines the added G2 conjunct, report G2 as an independent pass/fail but do not let a density-only pass be described as satisfying the updated two-goal Phase 2.

### B5. STOP decisions need a corroboration protocol to avoid harness-bug false negatives.

Round 4 asks that a STOP be corroborated, not just a harness bug (`phase2-review-brief-round4.md:27-31`). This is especially important because Round 2 found the key residual risk is a shared execution lock or dispatch serialization inside the correct AOTI topology (`phase2-round2-FOLDED.md:27-31`), and Round 3 found the Step 4 harness can misclassify intentional shedding (`phase2-round3-FOLDED.md:59-63`).

Recommended plan edit: no Step 0/1/4 STOP is accepted until all applicable checks below are attached to the review artifact.

- Repeatability: three independent runs; coefficient of variation for throughput and p95 metrics `<=10%`, or explain why not.
- Negative control: serial/mutex/default-stream control is measured and is worse in the expected direction by at least `15%` or shows the expected trace serialization.
- Topology sweep: at minimum test `num_runners=1`, `num_runners=N` with explicit streams, and a fallback topology with separate loader instances or process isolation if memory permits. If fallback topology passes, the result is a topology pivot, not a Phase 2 STOP.
- Profiler attribution: include CUDA/Nsight or CUPTI evidence for the limiting resource: execution lock, launch-bound, memory bandwidth, SM occupancy, CPU/loadgen, scalar sync, WS event loop, or admission. A utilization number alone is insufficient.
- Harness health: loadgen CPU, server CPU, network loopback/co-location, audio corpus, semantic-WER config, server flags, `num_runners`, stream IDs, admission caps, and shed flags are logged in the artifact name or manifest.
- Counterfactual: if a metric failed because correctness/WER failed, density/tail results are advisory only and do not become a STOP on the utilization thesis.

### B6. Technical gate and funding decision must be separated.

The Round 4 brief calls this out explicitly (`phase2-review-brief-round4.md:32-33`). `PHASE2-PLAN.md:5-7` frames Phase 2 as a strategic capability bet, not a COGS break-even guarantee.

Recommended plan edit:

- Technical GO: Step 4 clears G1 density, G2 tail, WER/correctness, and shed bounds.
- Technical STOP: Step 4 fails after B5 corroboration, or Step 1b fails the L40S ceiling gate after B5 corroboration.
- Funding GO: separate human decision after Technical GO, considering engineering weeks, operational risk, fleet composition, and whether the capability bet is still worth it without COGS break-even.
- Funding STOP despite Technical GO is allowed and must not be rewritten as a failed technical experiment.

## MAJOR

### M1. The 5090-to-L40S bar should be asymmetric and stricter than the final L40S bar.

The plan currently starts with 5090 (`PHASE2-PLAN.md:25`), while the preliminary mock showed 5090 `>=3x` and L40S only `~2-2.5x` (`PHASE2-PLAN.md:9-10`). Round 4 asks what 5090 multiplier should pass to L40S (`phase2-review-brief-round4.md:22-24`).

Recommended threshold: `>=2.00x` on 5090 passes to L40S; `<1.50x` stops after corroboration; `1.50x-2.00x` is tune/retest or explicit risk acceptance. This keeps 5090 as a spend-control proxy and prevents a weak 5090 result from consuming EC2 budget on a less forgiving target.

### M2. Step 1b must be a ceiling-margin gate, not the original realized-density gate.

Round 4 specifically asks what margin above `1.5x` is needed so Step 4 does not falsify a Step 1b GO (`phase2-review-brief-round4.md:25-26`). Round 3 also made Step 1b ceiling vs Step 4 realized a decision-structure item (`phase2-round3-FOLDED.md:69-70`).

Recommended threshold: Step 1b PASS `>=1.80x` and `>=34 streams/box`; Step 4 PASS `>=1.50x` and `>=28 streams/box`. The `1.80x` ceiling allows about a 17% realized haircut while preserving the final `1.50x` bar. If the actual measured scheduler/WS tax after Step 3 is known before Step 1b is re-run, replace `1.80x` with `1.50 / (1 - measured_tax_p95)` rounded up, but never below `1.70x`.

### M3. The semantic-WER and correctness gates need to be numeric because they define whether perf numbers mean anything.

The plan promises the same semantic-WER tool in Step 4 (`PHASE2-PLAN.md:39-41`), but Round 3 says the pass/fail bound is not pinned (`phase2-round3-FOLDED.md:67-68`). Without this, a faster system could pass by silently changing output quality.

Recommended threshold: `WER_native <= max(WER_python + 0.5 percentage points, 1.10 * WER_python)`, `0` token/event oracle mismatches on the concurrency corpus, and `0` stale-generation violations in Step 3 tests. If semantic WER is noisy, use a fixed corpus with at least three Step 4 repetitions and require the bound on the pooled result; still fail any deterministic oracle mismatch.

### M4. Step 5 should confirm attribution, not reopen the GO decision.

Step 5 currently says it will measure L40S/L4/Spark and test the per-target hypothesis (`PHASE2-PLAN.md:42-44`). Round 3 correctly says Step 1 counter attribution should pre-determine what Step 5 is expected to show, and Spark has concrete aarch64 `num_runners` risk (`phase2-round3-FOLDED.md:78-84`).

Recommended plan edit:

- Step 5 is not part of the initial L40S technical GO. It is confirmation and portability.
- Before Step 5, write a per-target prediction table: target, expected limiting resource, expected density multiplier, expected tail effect, counters that would falsify the prediction.
- Spark preflight is mandatory before full benchmarking: build/load, `num_runners=N` overlap, concurrent==serial, memory-flat, and stream trace on aarch64. Thresholds mirror Step 0: `>=1.15x` at `N=2`, `>=1.30x` at `N=4`, `0` mismatches, peak memory `<1.35x N=1`.

### M5. Adversarial stress of the proposed thresholds

The proposed thresholds are intentionally strict enough to prevent a false GO, but each can fail in a misleading way. The plan should state these stress cases next to the gates.

- `G1_floor = max(28, 1.50*S_py_L40S)`: robust against stale low Python baselines and against the historical `~28` floor becoming too low. Stress case: if the fresh Python baseline is abnormally bad due to a harness regression, the `28` floor prevents an easy pass, but B5 still needs harness-health evidence before accepting a native win.
- Step 1a `5090 >=2.00x`: robust against spending on L40S after a weak local result. Stress case: it can false-STOP if 5090 has a target-specific pathology that L40S does not share. That is why `1.50x-2.00x` is TUNE/RETEST plus written exception, not automatic STOP.
- Step 1b `L40S >=1.80x / >=34`: robust against a Step 4 scheduler/WS haircut. Stress case: it can reject a primitive that would barely clear `1.50x` realized after excellent scheduler work. The conditional band preserves that option, but it moves the choice out of the technical gate and into explicit risk acceptance.
- Step 4 `S_native_step4 >=0.83*S_native_step1b`: robust against hiding a large scheduler/server tax behind a barely passing density number. Stress case: a system can still be worth funding if it clears `G1_floor` but realizes only `0.75-0.83` of the ceiling and has a clear optimization path. That is why `<0.83` is not an automatic STOP if G1 clears, while `<0.75` is not a clean GO.
- G2 improvement `>=max(10ms,10%)`: robust against buying density by widening the tail and against noise-only "wins." Stress case: if Python's spread is already very small, a 10ms improvement may be impossible or meaningless. The `<50ms` baseline exception changes the gate to no-worse within `+5ms` plus p95 no-worse.
- p95 `<=300ms`, p99 `<=500ms`: robust against optimizing P50/P95 spread while leaving absolute tail unacceptable. Stress case: if the actual product SLO is the older p95 `<500ms` keep-up proxy, this is stricter than the existing harness. The plan must pin one SLO before running Step 4; do not choose after seeing results.
- Reject cap `<=10%`: robust against cheating by shedding most offered sessions and counting only admitted successes. Stress case: it may be too loose for product semantics. The technical gate should keep it as an anti-cheat bound; a stricter product/funding gate can set `<=1-2%` without changing the utilization experiment.
- WER bound `<= max(+0.5pp, 1.10x)`: robust against output-quality regression. Stress case: semantic-WER can be noisy or mask rare catastrophic failures. The deterministic oracle and stale-generation tests remain hard `0`-mismatch gates; pooled semantic-WER only covers semantic equivalence on the benchmark corpus.
- Number of gates: this is not too many gates if each has a distinct decision role. Step 0 is a cheap kill-gate, Step 1a is a 5090 spend-control gate, Step 1b is the L40S ceiling/build gate, and Step 4 is the realized technical GO/STOP. Step 2/3 are preconditions, not new business gates.

## MINOR

### m1. Use dynamic baseline plus absolute floor everywhere.

The `>=~28` number only makes sense if the Python L40S baseline is still roughly `18-19` streams/box. The plan says Python is `~16-20/L40S` (`PHASE2-PLAN.md:40-41`), and Round 3 says it must be re-measured back-to-back (`phase2-round3-FOLDED.md:63-68`). Use `max(28, 1.50*S_py_L40S)` so a weak stale baseline cannot lower the bar and a stronger fresh baseline cannot be ignored.

### m2. Treat p99 as a guardrail, not the headline G2 metric.

Round 3 requires p99 and spread fields (`phase2-round3-FOLDED.md:57-58`). The headline G2 gate should remain P95-P50 because that is the user-stated goal, but p99 should guard pathological outliers: Step 1/4 p99 `<=500ms` or no GO.

### m3. Artifact names should encode gate-critical topology.

Round 2 asked to log `num_runners` in output filenames (`phase2-round2-FOLDED.md:107-108`). Add `target`, `num_runners`, `stream_mode`, `admission_cap`, `backlog_cap`, `finalize_policy`, `shed_mode`, `corpus_sha`, `server_commit`, and `loadgen_commit` to artifact manifests. This is cheap and prevents comparing unlike runs.

## QUESTIONS

1. Does the user ratify the stricter two-goal gate? The original 0.0 threshold was density-only (`phase2-review-brief-round4.md:16-20`). My recommendation is conjunctive G1+G2, but that is a real scope change.
2. Is G2 literal first-token TTFT or server-side `vad_stop -> final` TTFS? If literal TTFT, current loadgen coverage is insufficient and Step 3/4 must add first-interim/token timestamps before the gate is valid.
3. What is the exact production SLO for this Phase 2 gate: p95 `<=300ms` and p99 `<=500ms`, or the older harness keep-up proxy p95 `<500ms`? I used `300/500` because Round 1/3 push toward user-visible tail and p99 guardrails, but the plan should pin it once.
4. Should the bounded reject rate be `<=10%` or stricter? I recommend `10%` as an anti-cheating cap for admitted-through-shed capacity, while still requiring the no-shed curve. If product semantics require near-zero rejects, the final funding gate should use a lower number.
5. If Step 1b lands in the conditional band (`>=G1_floor` but `<1.80x`), who can authorize proceeding? The plan should name that as a human funding-risk decision, not let it happen by inertia.

## Bottom Line

With the edits above, the corrected plan measures G1 end-to-end: Step 0 proves the native primitive, Step 1a controls 5090 spend risk, Step 1b proves L40S ceiling with margin, and Step 4 proves realized L40S density against a fresh Python baseline.

G2 is measured end-to-end only if the plan either renames the goal to server-side TTFS spread or adds literal first-token TTFT instrumentation. As written, literal TTFT remains orphaned; the existing evidence chain measures server-side `vad_stop -> final` tail, not the full client-visible first-token path.
