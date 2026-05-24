# Codex adversarial review - Round 5

Final consistency pass on `proj-2026-05-24-from-scratch-runtime/PLAN.md` v5. No new source-thesis blockers found.

1. [MAJOR] `proj-2026-05-24-from-scratch-runtime/PLAN.md:373` - `proj-2026-05-24-from-scratch-runtime/PLAN.md:382` - The 0.4 decision tree is close, but not complete: non-decode feasibility failures still fall through.

   The proceed row requires `0.6a + 0.2 + 0.8 + 0.11 pass` (`PLAN.md:378`), and the table maps `0.6a` failure (`PLAN.md:379`), `0.5` B~1 (`PLAN.md:380`), and `0.11` graph-pool failure (`PLAN.md:381`). It does not map:
   - `0.2` fails T2a / libtorch byte-exactness is unattainable, even though 0.2 says no-go means reclassify B1a as non-identical or fall back to B4 (`PLAN.md:304` - `PLAN.md:306`).
   - `0.8` fails native-preprocessor byte-exactness, which poisons the Phase-1 native-preproc -> encoder chain (`PLAN.md:321` - `PLAN.md:326`).
   - `0.9` fails the pure-parameter mutability audit, despite being the stated prerequisite for the shared-read-only-weights density lever (`PLAN.md:335` - `PLAN.md:340`). That prerequisite is real: the current code mutates `drop_extra_pre_encoded` around encoder calls (`src/nemotron_speech/cudagraph_encoder.py:56` - `src/nemotron_speech/cudagraph_encoder.py:64`), and today's lane replicas avoid shared-object races by loading separate model objects (`src/nemotron_speech/server.py:3109` - `src/nemotron_speech/server.py:3137`).
   - `0.5` produces B>1 but poor exact-B graph hit rate / high eager fallback / insufficient memory headroom. The simulator is explicitly required to report those values (`PLAN.md:362` - `PLAN.md:367`), but the decision table only maps the special case "B stays ~1."

   Concrete fix: add explicit rows: `0.2 fails -> no B1a; STOP, B4 if 0.3 already won, or explicit B2/T1-only risk sign-off`; `0.8 fails -> no native-preproc Phase 1; STOP or keep Python preproc as a named non-v1 topology`; `0.9 fails -> drop shared-weight density, use per-lane replicas, re-run 0.0, likely STOP`; `0.5 graph hit/fallback/memory fails -> drop graph throughput/density claims and re-run 0.0`. The `native-under-MPS` branch is no longer a false density escape in v5: `PLAN.md:376` correctly makes it tail-only, no density gain, and forces a 0.0 re-run.

2. [MAJOR] `proj-2026-05-24-from-scratch-runtime/PLAN.md:353` - `proj-2026-05-24-from-scratch-runtime/PLAN.md:367` - Two Wave-1 gates are still not falsifiable before measurement because their pass thresholds are unnamed.

   Spike 0.1 requires a single native process to overlap finalize+steady by "a named factor" (`PLAN.md:353` - `PLAN.md:354`), but the factor is not named in the plan. Spike 0.5 says `Go: median B >> 1 AND graph pool fits` (`PLAN.md:365` - `PLAN.md:366`), but `>> 1`, acceptable exact-B hit rate, eager-fallback percentage, added-wait tolerance, and graph-memory headroom are not numeric. These are Wave-1 kill decisions, so defining them later in 0.4 is too late; the thresholds must be registered before collecting 0.1/0.5 data.

   Concrete fix: before running 0.1/0.5, add a small "pre-registered thresholds" block: overlap factor vs Python/MPS, max queue/lane wait, max added latency, median/p95 B target, minimum graph replay hit rate, maximum eager fallback %, and required L4/L40S memory headroom. The raw data is measurable: the plan already requires metric-schema parity for `_continuous_finalize_timing`, batch telemetry, and finalize telemetry (`PLAN.md:385` - `PLAN.md:392`; `src/nemotron_speech/server.py:5388` - `src/nemotron_speech/server.py:5424`, `src/nemotron_speech/server.py:6594` - `src/nemotron_speech/server.py:6609`, `src/nemotron_speech/server.py:7254` - `src/nemotron_speech/server.py:7281`). The remaining gap is the pass/fail line.

3. [MINOR] `proj-2026-05-24-from-scratch-runtime/PLAN.md:546` - `proj-2026-05-24-from-scratch-runtime/PLAN.md:547` - The review log still contains a stale contradiction about the decode target.

   The active plan is consistent that deployed decode is `greedy_batch` label-looping and that frame-looping is the non-batched fallback that rejects streaming continuation (`PLAN.md:54` - `PLAN.md:61`, `PLAN.md:138` - `PLAN.md:150`). But the Round-1 log says the plan "made the native RNNT frame-looping decode (0.6) the real go/no-go" and "fixed label-looping -> frame-looping" (`PLAN.md:546` - `PLAN.md:547`). Round 2 immediately corrects this (`PLAN.md:553` - `PLAN.md:556`), so this is historical drift, not an active design bug.

   Concrete fix: rewrite the Round-1 log sentence to make the historical reversal explicit, e.g. "Round 1 initially changed the target to frame-looping; Round 2 corrected that to deployed `greedy_batch` label-looping." Or delete the frame-looping detail from the Round-1 summary.

4. [MINOR] `proj-2026-05-24-from-scratch-runtime/PLAN.md:75` and `proj-2026-05-24-from-scratch-runtime/PLAN.md:172` - The density numbers in the plan are internally usable, but the cited launcher still carries legacy keep-up-style 48/64-box wording.

   The plan consistently frames the Python baseline as ~20 today and ~28 in-budget after the Python plan, with 40-48 as aspirational native density (`PLAN.md:75`, `PLAN.md:99` - `PLAN.md:105`). However, the deploy source cited for the K cap says L40S `K=3 (~48/box)` and "K~4/~64" while also saying K=4 OOMs (`deploy/launch_multiproc.sh:6` - `deploy/launch_multiproc.sh:9`, `deploy/launch_multiproc.sh:18` - `deploy/launch_multiproc.sh:21`). That source comment appears to be the old over-budget/keep-up number, not the in-budget density definition used by this plan.

   Concrete fix: in the plan's deploy anchor or in the launcher comment, explicitly label those 48/64 numbers as legacy keep-up/over-budget figures and state that the worth-it gate uses in-budget streams/box from the Python plan. Otherwise readers following the citation will think Python already reaches the native aspirational target.

Verdict: YELLOW. The plan is now structurally sound and actionable, and there is no remaining blocker to the overall approach, but the 0.4 decision table and Wave-1 thresholds need the small fixes above before this should be treated as a fully closed engineering plan. The phase/wave ordering is executable: 0.10 precedes Phase 1, and 4.4 now correctly supplements rather than gates Phase-1 T1. The one thing still most likely to be wrong is the business premise: after the Python plan lands, the residual tail/density value probably will not justify a second runtime stack.

## Top 4 / fewer if converged

1. Complete 0.4 with explicit rows for `0.2`, `0.8`, `0.9`, and non-B~1 `0.5` graph-capacity failures.
2. Pre-register numeric pass/fail thresholds for 0.1 and 0.5 before collecting Wave-1 data.
3. Fix the stale Round-1 review-log frame-looping sentence.
4. Reconcile the launcher’s legacy 48/64 density comments with the plan’s in-budget ~20/~28 density definition.
