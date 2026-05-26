# Step-0 build — paired adversarial review charge (Codex)

Codex built `runtime/cpp/density_main.cpp` (Step-0 density kill-gate harness) and ran limited smoke on the 5090:
- 0a steady overlap: N=1 113.6/s, N=2 1.69×, N=4 1.71×, N=16 1.92×; default-stream N=4 22.6% worse; unique
  streams up to 16.
- 0b: 0 token/event mismatch (N=2 smoke); scalar `.item()` wait p95 4.07ms = 63.8% of steady GPU p95 → Codex
  flagged this as a FAIL vs the plan's ≤5% threshold.
- 0c: same-bucket 0 mismatch, mixed-bucket 0 mismatch; finalize runner-wait ~0.
- Build clean in `nemotron-aoti:cu128`; `session_main` N=200 regression still 0/200 token + 0/200 event.

This is an INDEPENDENT adversarial review of that build (an Opus agent reviews in parallel — don't coordinate).
Read `runtime/cpp/density_main.cpp` (it `#include`s `session_main.cpp` with `main` renamed) + the
`runtime/cpp/CMakeLists.txt` diff + the telemetry logs under `runtime/artifacts_n200_mel/logs/` + the archived job
log `codex-jobs/step-0-*.log`. Cross-check against `PHASE2-PLAN.md` Step 0 + the "Ownership/topology contract".

## Scrutinize (file:line, be adversarial)
1. **Topology faithfulness.** Is it genuinely ONE shared steady loader with `num_runners=N` (not N loaders)?
   Explicit per-worker stream into `run(inputs, stream_handle)`? Per-thread `SessionState` + per-thread
   `joint`/`predict`/`enc_first`/`preproc` handles? Does it prove ONE weight copy (memory flat) at the target N?
2. **Is the 0a overlap REAL or an artifact?** `loader.run()` appears host-synchronous (Codex's concern) — so the
   1.69× at N=2 is throughput from N threads each blocking on their own `run()` while the GPU runs them on
   separate streams. Is that valid evidence of GPU overlap, or could it be CPU-side parallelism / measurement
   artifact? Is the "unique streams up to 16" + "overlap estimate 11.06" meaningful or hand-wavy? Does the plan's
   requirement for Nsight/CUPTI kernel-overlap proof still stand (I believe yes)? Why does the multiplier plateau
   (N=2 1.69× ≈ N=4 1.71×)? Attribute it.
3. **Is the 0b concurrent==serial assertion SOUND?** Does it compare each worker's FULL token AND ordered event
   stream to the SERIAL (N=1) oracle, byte/token-exact — or is it weakened (subset, cumulative-only, tolerance)?
   A weak assertion = a false correctness PASS.
4. **The scalar-sync ≤5% "FAIL" — is the THRESHOLD mis-specified?** A high single-thread `.item()` wait fraction
   (63.8%) is arguably the GPU-idle window the multi-thread thesis FILLS (and 0a's 1.69× + the default-stream
   control being 22.6% worse corroborate that the overlap works). So is "≤5%" measuring the wrong thing? The real
   serialization concern (review B5) was whether `.item()` forces a GLOBAL/default-stream sync — which the
   default-stream negative control addresses. Recommend the correct threshold semantics.
5. **Harness concurrency bugs.** Any race in the harness's own telemetry/comparison/shared state? Is the
   `#include "session_main.cpp"` + main-rename clean (no ODR/double-main/static-init hazard)? Do the per-thread
   handles actually get separate instances (not shared pointers)?
6. **Honesty / smoke vs full.** 0b/0c were "limited smoke" (N=2, few rows). Is the full 0b over 200 rows + higher
   N needed before any PASS is credible? Any number Codex reported that over-claims?

Write BLOCKER/MAJOR/MINOR with file:line + recommended fixes to
`proj-2026-05-24-from-scratch-runtime/reviews/codex-step0build-review.md`. End with a verdict: is the harness
sound + the gates correctly specified, and what must change before the Step-0 PASS/STOP call.
