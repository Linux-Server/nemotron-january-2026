# Phase 2 — multi-stream density (the streams/box win)

Project directory: /home/khkramer/src/nemotron-january-2026/proj-2026-05-24-from-scratch-runtime

> **v2 (2026-05-26):** rewritten after a 5-round paired Codex+Opus adversarial review (`reviews/{codex,opus}-
> phase2-round{1..5}.md` + `phase2-round{1..5}-FOLDED.md`). The "meaningfully" gate is replaced by a pre-registered
> numeric GO/STOP tree; the cheap kill-gates are front-loaded; the loader/ownership/topology contract and a
> correctness-before-performance gate are pinned. **Reviewer sign-off: GO-to-build Step 1 (Step-0 kill-gates
> first), conditional on the edits below.**

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
  Step-4 apples-to-apples manifest (NOT the stale ~16–20).
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

- [ ] **Step 1a — 5090 mini-sweep (spend-control proxy, NOT a project GO).** Real decode + real finalize +
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

- [ ] **Step 1b — L40S CEILING hard gate (EC2).** Same harness, **sm_89-AUTOTUNE-ON compiled NATIVELY on the
  g6e** (EPs shipped via S3 per the Compile & artifact policy; autotune-on re-validated token-exact on the exact
  package SHA) **+ a bounded sm_89 autotune-OFF floor** (or an explicit waiver — win inferred from the 5090 off→on),
  reported as the absolute streams/box Δ + % with identical workload/N/bound/cadence/topology + distinct artifact
  hashes. PASS-to-build = `S_native_step1b ≥ max(34, 1.80·S_py_L40S)` (the 1.80× leaves margin for the ~17% Step-4 scheduler/WS haircut;
  34 = 28/0.83) + 0 mismatch + WER in bound + `ttfs` within the ttfs SLO budget (Definitions). `TTFS_spread`
  reported (GREEN ≤1.10× Python; otherwise a build-risk signal, not a STOP — the binding tail is reported at
  Step 4). CONDITIONAL = ≥G1_floor but <max(34, 1.80·S_py_L40S) → proceed only on explicit human risk-acceptance +
  a narrow de-risking prototype. STOP-candidate = <G1_floor or SLO/WER/correctness fail. PAIRED REVIEW (decisive
  measurement).

- [ ] **Step 2 — scheduler design + admission (blocked on Step-1 telemetry).** From the Step-1 **telemetry
  schema** (not a scalar knee): one **box-global active/admitted cap** + one **box-global backlog-COUNT cap**
  (ready-age is dead; sweep ~8/10/12), shed = **close** (count admitted, not offered; two curves). A declared
  numeric **priority-finalize-lane policy over the `num_runners=N` pool** — either partitioned
  (`N_finalize_reserved≥1`, steady-starvation p95 ≤2× no-finalize) or weighted (finalize runner-wait ≤25% TTFS,
  steady queue-wait ≤2× no-finalize). Faithful to the Python shed/priority behavior re-derived for one process.
  PAIRED REVIEW of the design before building.

- [ ] **Step 3 — multi-session runtime + real WS server.** Wrap the session core in the scheduler + a real WS
  server (also closes the 1.4b interim-cadence residual). **Required before Step 4:** a **WS-tail microbench**
  (accept→ready, send→recv, recv→queue, queue→scheduler, serialize/send, client-recv, event-loop lag under N idle
  + N streaming sockets; WS overhead p95 ≤10% of TTFS or decompose, don't claim as runtime tail) — production
  needed a cooperative-yield to avoid socket starvation, so this is not a rounding error. **Stale-generation
  suppression is a Step-3 gate** (per-session generation tokens; close-while-inflight, reset-while-queued,
  reset-while-finalizer-owns-runner, final-after-shed; 0 stale/mismatch) — so a Step-4 tail "win" can't be a
  dropped-final artifact.

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
- **Tier 1 — `enc_first` DEDUP** (the current 5090 binding resource — W1 corrected this from "finalize-memory"):
  the per-worker 2.48 GiB `enc_first` copy is the memory wall. **Fix-2** (share one enc_first ref + a lock for the
  rare first-chunk forward, ~10 lines, fast confirm) or **Fix-1** (fold first-chunk into the shared steady AOTI
  loader → ~0 GiB/worker + closes the first-chunk-TorchScript residual). Est. 5090 N=4→~40-45. (Finalize-bucket
  hygiene — load+warm only the needed subset — is ALREADY implemented + correct; padded-bucket consolidation is
  REJECTED: ~0 saving + not token-safe.)
- **Tier 2 — GATE-DEPENDENT** (pending the L40S sweep's binding-resource attribution): L40S memory-bound → Tier 1;
  **contention-bound → the autotune ladder** (R1a precision-matched, T1-gated); host-bound → Tier 3.
- **Tier 3 — sync / host-ceiling / tail levers** (first-order ONLY once memory is relaxed + the binding shifts to
  host): eliminate the `enc_len` D2H `.item()` (compute host-side; ~4–6ms/chunk + a sync point); trim the finalize
  glue (async/cheaper FORK_ASSERT+text); pinned+`non_blocking` copies (minor — tiny volumes); **cross-stream
  transfer/compute batching** (a Step-2/3 *scheduler* change). *Double-edge:* the `.item()` idle is the multi-thread
  fill window → eliminating syncs cuts per-thread latency but also fill-opportunity; net density effect must be
  MEASURED (the phase-split telemetry does).
- **NOT levers (ruled out this session):** a finalize CUDA graph for the "234ms" (it was cold-start; warm=8ms);
  autotune for the memory wall; aggressive `max_autotune`+`coordinate_descent` (breaks T1).

### Scoped next work (priority / dependency)
- **W0 DONE:** autotune-on T1-FAILED (995/1000); finalize-234ms = cold-start (warmup-fixed); W1 root-caused the
  memory wall to `enc_first` dup (all paired).
- **W1 = `enc_first` DEDUP (Tier-1, CRITICAL PATH, the L40S prerequisite).** Implement **Fix-2** (share one
  enc_first ref + lock, ~10 lines) → re-run 1a **fresh-process-per-N** → confirm the knee jumps from N=4 (est.
  ~40-45) + correctness holds (concurrent==serial with the shared+locked enc_first). Then **Fix-1** (fold into the
  shared steady AOTI loader) for production. ~10× est. density upside; confirmable ~5min.
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
| 1a 5090 spend-control | in-progress | e5f2753 | **Warm autotune-OFF floor: knee N=4 SLO-robust (TTFS p95 14ms; 0 mismatch), finalize warm ~8ms; MEMORY-bound (N=8 OOM)** — 234ms was cold-start (fixed by warmup). autotune-ON breaks T1 (precision-policy drift; ladder ready, contingent on L40S contention-bound). N=4 is a memory-capped 5090 FLOOR, not the verdict — L40S is the gate. See Live findings + lever inventory. PASS bar ≥2.00× still pending the real density multiplier. |
| 1b L40S ceiling gate | todo | | PASS ≥max(34, 1.80·S_py); paired; the density ceiling |
| 2 scheduler+admission design | todo | | blocked on Step-1 telemetry; paired (design) |
| 3 multi-session + real WS | todo | | WS-tail microbench + stale-gen gate; closes 1.4b interim-cadence |
| 4 realized density (apples) | todo | | TECHNICAL GO ≥G1_floor; G2 TTFS_spread reported; manifest + re-measured baseline |
| 5 per-target confirmation | todo | | confirm Step-1 attribution; Spark aarch64 preflight; EC2 |
