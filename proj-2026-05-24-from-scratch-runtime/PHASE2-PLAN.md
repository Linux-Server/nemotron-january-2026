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

## Ownership / topology contract (pin before building)
One shared steady AOTI loader **`num_runners=N`** + explicit **per-worker CUDA streams**; **per-worker**
`SessionState`, `AudioFrontend`, `enc_first`, `joint`, `predict`, `preproc` (concurrent `forward()` on one shared
TorchScript handle is unproven — the mock used per-lane handles). `user_managed`/codisk shared constants for the
**finalize buckets** only (steady is covered by the runner pool). Negative controls: mutex-serialized,
default-stream, `num_runners=1`. Log `num_runners`/stream-mode/topology in every artifact name.

## Steps (the pre-registered GO/STOP tree)

- [~] **Step 0 — cheap native kill-gates (5090, ~hours, BEFORE any EC2 spend).** Each is a potential early STOP
  (corroborated — see Rules). PAIRED REVIEW.
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
  per-thread streams/handles + the full telemetry schema. Knee **attributed to a binding resource (BW / launch /
  lock / CPU-core)** via Nsight/CUPTI counters — "BW-bound" is a hypothesis, not NVML util. PASS-to-1b =
  real-decode multiplier **≥2.00×** vs N=1 AND 0 mismatch AND WER in bound AND `ttfs` within the ttfs SLO budget
  (Definitions). Report `TTFS_spread`. TUNE/RETEST = 1.50–2.00× with attribution to a fixable harness/topology issue (no
  EC2 until ≥2.00× or a written non-predictive exception). STOP-candidate = <1.50× or SLO/WER/correctness fail.
  The 5090→L40S transfer is noisy → a marginal 5090 PASS does NOT substitute for the L40S measurement.

- [ ] **Step 1b — L40S CEILING hard gate (EC2).** Same harness. PASS-to-build = `S_native_step1b ≥ max(34,
  1.80·S_py_L40S)` (the 1.80× leaves margin for the ~17% Step-4 scheduler/WS haircut; 34 = 28/0.83) + 0 mismatch +
  WER in bound + `ttfs` within the ttfs SLO budget (Definitions). `TTFS_spread` reported (GREEN ≤1.10× Python;
  otherwise a build-risk signal, not a STOP — the binding tail is reported at Step 4). CONDITIONAL = ≥G1_floor but
  <max(34, 1.80·S_py_L40S) → proceed only on explicit human risk-acceptance + a narrow de-risking prototype.
  STOP-candidate = <G1_floor or SLO/WER/
  correctness fail. PAIRED REVIEW (decisive measurement).

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

## Progress
| Step | Status | Commit | Notes |
|---|---|---|---|
| 0 cheap kill-gates (5090) | in-progress (harness sound; gates partial) | | density_main.cpp built + paired-reviewed + gate-soundness fixes. **conjunct-2 binary = YES:** overlap real (sentinel: stream B runs while A mid-encoder; not the GIL), correctness-safe (0a serial-oracle 0 mismatch; 0b identity 0/200), ONE weight copy (loader-delta 2.309GiB flat across N=1→16). 0a PASS ~1.7× (encoder-ONLY, lock-bounded; controls: single-runner≡N=1, mutex −22%, default −30%), 0b PASS. **0c PARTIAL** (4-worker same/mixed 0 mismatch; 8-worker mixed OOM ~33GB on 32GB 5090). Nsight pending (sentinel substitutes). Density magnitude = Step 1a. |
| 1a 5090 spend-control | todo | | PASS ≥2.00×; not a project GO |
| 1b L40S ceiling gate | todo | | PASS ≥max(34, 1.80·S_py); paired; the density ceiling |
| 2 scheduler+admission design | todo | | blocked on Step-1 telemetry; paired (design) |
| 3 multi-session + real WS | todo | | WS-tail microbench + stale-gen gate; closes 1.4b interim-cadence |
| 4 realized density (apples) | todo | | TECHNICAL GO ≥G1_floor; G2 TTFS_spread reported; manifest + re-measured baseline |
| 5 per-target confirmation | todo | | confirm Step-1 attribution; Spark aarch64 preflight; EC2 |
