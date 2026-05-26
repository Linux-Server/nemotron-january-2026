# Phase-2 plan — adversarial review brief (charge for the reviewer)

You are one of two **independent** adversarial reviewers (the other is an Opus agent reviewing in parallel; do
not coordinate — independence is the point). Your job: a rigorous, skeptical, **line-by-line** review of
`proj-2026-05-24-from-scratch-runtime/PHASE2-PLAN.md` **and all the code/evidence it depends on**. This is a
static review — read, grep, reason. No GPU/container build needed.

## The two goals to foreground (the user's stated objectives for this plan)
1. **Improved system utilization** — more concurrent streams per GPU (streams/box density); reclaim the
   40–65% idle GPU the Python stack leaves at its knee.
2. **Tightened spread between P50 and P95 TTFT** — i.e. the latency *tail* under load, not just the median.

For EVERY finding, where relevant, tie it back to whether the plan, as written, will actually measure/achieve
these two goals — or whether it could produce a misleading PASS/STOP on either.

## Context (facts, not conclusions — verify them yourself in the code)
- This is "Phase 2" of a from-scratch native (C++/libtorch) ASR serving runtime. Phase 1 proved a single native
  stream is token/event-exact vs the Python reference (`runtime/cpp/session_main.cpp`).
- THE BET conjunct Phase 2 must settle: the density residual is **GIL/scheduler-bound, not MPS/bandwidth-bound**
  → a native multi-thread runtime lifts streams/box. The decisive measurement is Step 1.
- The ONLY density evidence so far is the **0.1b microbench**, which used a **MOCK decode** (a host `sleep` +
  a dummy GEMM — `spikes/0.1-overlap-ablation/microbench/microbench.cpp`). Results:
  `spikes/0.1-overlap-ablation/microbench/RESULTS-5090.md` and `RESULTS-L40S.md`. Spec + pre-registered
  go/no-go: `spikes/0.1-overlap-ablation/0.1b-microbench-spec.md`. The plan's premise is that Step 1 replaces
  the mock with the REAL native session compute.
- The pre-registered gate (from the spec/memory): **≥1.5× L40S density (≥~28/box)** is GO; fleet target is
  L40S/Ada (L4 is OUT, BW-bound). The deployable Python baseline is ~16–20 streams/box on L40S, ~6 on L4.
- Honesty bar (Phase-1 lesson, stated in the plan): measure with REAL decode (no mock), report GPU util + the
  knee with an explicit SLO definition, and do NOT overclaim the knee (the 0.1b "keep-up knee" reportedly
  overstated the SLO-robust number by ~2–3×).

## Files to read line-by-line (the relevant code/evidence)
- `proj-2026-05-24-from-scratch-runtime/PHASE2-PLAN.md` — the plan under review (Steps 1–5).
- `proj-2026-05-24-from-scratch-runtime/PLAN_RULES.md` — environment, oracle, test protocol, review intensity.
- `spikes/0.1-overlap-ablation/microbench/microbench.cpp` — the MOCK-decode harness Step 1 evolves from.
- `spikes/0.1-overlap-ablation/0.1b-microbench-spec.md` + `RESULTS-5090.md` + `RESULTS-L40S.md` — the
  preliminary evidence and its own caveats.
- `runtime/cpp/session_main.cpp` — the validated single-stream native core Step 1 says it will replay. Focus on:
  how the AOTI steady encoder is loaded and run (`AOTIModelPackageLoader`, `.run(`, `load_constants`,
  `run_steady_encoder`); whether any CUDA streams / stream guards / device guards are used; how decode runs;
  how finalize buckets are loaded/selected/run; any `static`/global/`thread_local`/shared mutable state.
- `runtime/cpp/finalize_main.cpp` — the finalize-bucket substrate (shared-weights `load_constants`,
  `constants_for_bucket`, per-bucket loaders, fork/clone, FORK_ASSERT).
- `runtime/export_shared_weights.py`, `runtime/validate_shared_weights.py`, `runtime/strip_bucket_weights.py`
  — the "proven shared-weights mechanism" the plan leans on. Verify *what* was actually proven (single-thread
  load-time sharing? concurrent run-time sharing?).
- `runtime/cpp/CMakeLists.txt` — build/runtime assumptions.
- Skim the folded Phase-1 reviews in `reviews/*-FOLDED.md` to avoid re-raising settled points and to inherit
  the project's honesty conventions.

## What I want from you (be specific and adversarial)
Produce a structured review. For each finding give a severity, a precise `file:line` citation, the concrete
risk, and a recommended plan edit. Categories:
- **BLOCKER** — the plan, as written, could produce a false GO or a false STOP, or omits something decision-
  critical to the two goals.
- **MAJOR** — a real gap/risk that should change the plan before Step 1 is built.
- **MINOR** — tighten-ups.
- **QUESTIONS** — genuine unknowns the plan must resolve.

Push hard on at least these (and anything else you find):
1. **Mock→real fidelity.** Exactly what does replacing the mock decode with the real native compute change vs
   the 0.1b harness? Does the 0.1b *mechanism* (per-lane TorchScript CUDA-graph `graph.replay()` on a per-lane
   stream) even match how `session_main.cpp` actually dispatches the encoder? If the dispatch primitive differs,
   does the 0.1b overlap result transfer at all?
2. **Concurrency of the AOTI path.** Is `AOTIModelPackageLoader::run()` safe to call concurrently from N
   threads? One shared loader vs N loaders vs the loader's `num_runners` pool — which does the plan assume, and
   is it stated? Does `run()` honor a per-thread CUDA stream (so kernels actually overlap), or does it
   serialize? What does `load_constants(..., user_managed=true)` actually guarantee under concurrency, and was
   that ever tested (vs only single-thread load-time sharing)?
3. **The gate/SLO definition.** Is Step 1's HARD GATE a number or a vibe ("meaningfully")? Is the SLO the same
   one the Python baseline used? Is "streams/box" reported SLO-robust or keep-up? Does Step 1 (on the 5090)
   even produce a number comparable to the L40S ≥1.5× gate, or is it only a binary overlap check?
4. **The P50/P95 spread goal.** Does the plan make the tail (P95−P50) a first-class measured quantity with a
   target? Is the spike's load model capable of *producing* a realistic tail (decode is data-dependent /
   variable-iteration; finalizes arrive asynchronously per-stream)? Or does a fixed-trace replay understate the
   tail and thus the very thing goal 2 cares about?
5. **Finalize + correctness under concurrency.** Does Step 1 include the REAL finalize path (fork +
   load_constants/bucket swap + heavier encoder + continuation decode), or repeat the 0.1b "no finalize" gap?
   Is there any **concurrency-correctness gate** (N-thread output == serial output, token/event-exact) before
   any throughput number is trusted — or does the plan measure speed without proving the concurrent compute is
   still correct?
6. **5090→L40S transfer + STOP semantics.** Step 1 is on the 5090 but the gate is L40S. What exactly can a 5090
   PASS establish? If single-context dispatch does NOT overlap, the plan says "STOP/reassess" — is the MPS /
   per-context fallback actually viable given the prior cross-process result (K=3≡K=4, MPS/BW contention
   cancels added procs)?
7. **Sequencing / decision-criticality.** Are Steps 2–5 correctly gated on Step 1? Anything that should be
   measured earlier/cheaper? Any dependency or honesty risk in Steps 3–5 (real WS server, apples-to-apples
   harness, per-target sweep)?

Write your review to `proj-2026-05-24-from-scratch-runtime/reviews/codex-phase2-round1.md`. Be concrete, cite
lines, and rank findings. Do not soften: a wrong GO here funds ~weeks of byte-exact native ports.
