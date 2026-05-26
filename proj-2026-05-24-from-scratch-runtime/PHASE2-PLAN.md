# Phase 2 — multi-stream density (the streams/box win)

Project directory: /home/khkramer/src/nemotron-january-2026/proj-2026-05-24-from-scratch-runtime

## Why / the bet's open conjunct
Phase 1 proved the single native stream is correct (token/event-exact to the server's final transcript; preproc→AOTI
steady→decode→finalize-bucket→state-machine→emit). It did NOT measure DENSITY. The whole project rests on the BET conjunct
that the residual is **GIL/scheduler-bound, not MPS/bandwidth-bound** → a native multi-thread runtime lifts streams/box.
The only density evidence so far is the **0.1b microbench (PRELIMINARY, mock-decode = zero-GPU-load host sleep)**: 5090
≥3×, L40S ~2-2.5× — flagged optimistic. Roofline: GPU 46-65% idle at the knee; limit = single-thread intake +
inference_lock + GIL decode. Per-target hypothesis (UNMEASURED natively): 5090/L40S launch-bound → native helps;
L4/Spark BW-bound → native does NOT help density.

**Phase 2 measures it for real**, with the validated single-stream native core + the proven shared-weights mechanism.

## THE decisive first question (Step 1, hard gate)
Can N OS threads CONCURRENTLY run the real native session compute (AOTI steady encoder + label-looping decode) sharing ONE
weight set on ONE GPU, and actually OVERLAP to lift throughput — or does the single CUDA context / launch dispatch
serialize them (→ MPS or per-context required)? This is conjunct 2, decisively, with REAL decode (not the 0.1b mock).
Unknowns to resolve: is `AOTIModelPackageLoader::run()` + the shared `load_constants(user_managed=true)` weights thread-safe
for concurrent `run()` from N threads? Do per-thread CUDA streams overlap, or does the context serialize launches? Where is
the knee, and is GPU util > the Python stack's 46-65%?

## Steps
- [~] **Step 1 — Native concurrent-dispatch density spike (5090).** Build a minimal multi-thread harness: N worker threads,
  each replaying the real per-session compute (steady AOTI + decode, shared weights via load_constants user_managed, a
  CUDA stream per worker) over the session bundle, all on ONE GPU/context. Measure throughput (streams sustained at the
  SLO) + latency tail + GPU util as N grows → the KNEE. Compare to the single-thread baseline (N=1). HARD GATE: does
  concurrent native dispatch lift the knee meaningfully above single-thread (the conjunct-2 question)? Report the real knee
  + GPU util + whether the CUDA context serializes (and whether MPS/green-contexts are needed). PAIRED REVIEW (this is the
  decisive measurement — the same scrutiny the 0.1b mock-decode result lacked). If concurrent dispatch does NOT overlap
  (context-serialized) → that's a STOP/reassess finding for the density thesis.
- [ ] **Step 2 — Scheduler design + admission.** From Step 1's knee: design the multi-stream scheduler (thread/stream
  ownership, the finalize priority lane, admission/backlog-cap shedding) faithful to the Python stack's shed behavior.
  PAIRED REVIEW of the design before building.
- [ ] **Step 3 — Multi-session runtime + real WS server.** Wrap the session core in the scheduler + a real WS server
  (also closes the 1.4b interim-cadence residual: the session behind real WS). N concurrent real streams, correct
  per-stream events/finals.
- [ ] **Step 4 — Density measurement vs the Python stack (apples-to-apples).** The stt-benchmark network harness driving
  the native WS server: measure SLO-robust streams/box vs the Python stack's ~16-20/L40S, ~6/L4 — SAME harness, SAME
  semantic-WER tool, SAME hardware (the true like-for-like the user asked about). 5090 first.
- [ ] **Step 5 — Per-target sweep (L40S, L4, DGX Spark).** Measure native density on each target; test the hypothesis
  (5090/L40S launch-bound → native lifts; L4/Spark BW-bound → no lift). EC2 for L40S/L4; Spark aarch64 (libtorch maturity
  risk). Cost/density verdict per target.

## Rules
See PLAN_RULES.md. Step 1 + 2 are decision-critical → PAIRED adversarial review (Codex + Opus). Honesty bar (Phase-1
lesson): measure with the REAL decode (no mock), report GPU util + the knee with SLO definition + don't overclaim the knee
(the 0.1b "keep-up knee" overstated ~2-3×; report SLO-robust).

## Progress
| Step | Status | Commit | Notes |
|---|---|---|---|
| 1 concurrent-dispatch spike | in-progress | | decisive conjunct-2 measurement, paired |
| 2 scheduler+admission design | todo | | paired (design) |
| 3 multi-session + real WS | todo | | closes 1.4b interim-cadence residual |
| 4 density vs Python (apples) | todo | | the like-for-like number |
| 5 per-target sweep | todo | | 5090/L40S/L4/Spark; EC2 |
