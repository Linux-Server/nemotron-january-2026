# Codex Step 3b Plan Review - Round 3

Verdict: **GO-with-1-fold**, not MINOR_ONLY/CONVERGED yet. v3 lands the intended StatsCollector
ownership/order in the main prose: Step 5 now says `SessionRuntime::finalize_now()` only produces
`WireEvent` + `SessionTiming` / `last_timing()`, and Step 9 records exactly once after the worker's
stale-gen/send-drop decision. However, one operative Step 5 "Key files" line still instructs the
implementer to wire `record()` in `runtime.cpp`, directly contradicting the v3 fold. This is a
small text fix, but it is the same ownership/order hazard as Round 2, so I would not declare
convergence until that line is corrected.

## Must Fold

1. **Remove the stale Step 5 instruction to wire `record()` in `SessionRuntime`.**

   `STEP3B-WS-PLAN.md:234-240` is correct: `StatsCollector` is built/tested in isolation in Step 5;
   `finalize_now()` does not call `record()`; the WS worker records after the emit/drop/timeout
   decision. `STEP3B-WS-PLAN.md:243-245` still says:

   > `runtime/cpp/lib/session/runtime.cpp` (wire `record` call)

   That line reintroduces the exact Round 2 bug. An implementer following the key-file checklist
   could put `StatsCollector::record()` back into `SessionRuntime`, before stale-gen suppression
   and send failure are known, while Step 9 later records again after the emit decision.

   Fold text: replace the Step 5 key-file parenthetical with something like:
   `runtime/cpp/lib/session/runtime.cpp` (populate `SessionTiming` / `last_timing()` only; no
   `StatsCollector::record()` call).

## Fold Check

- **StatsCollector ownership/order:** mostly landed. Step 5's ownership paragraph and Step 9's
  lifecycle sequence now match v4 §V: `finalize_now()` produces output/timing, stale-gen and final
  emit/drop happen in the worker, then the worker stamps `was_suppressed` + `emitted` and calls
  `record()` exactly once. The single stale key-file bullet above is the only blocker.
- **No new lifecycle contradiction:** Step 9's lifecycle prose and bar agree on stale final drops:
  final dropped silently, `drops_at_finalize_output++`, `record()` sees `was_suppressed=true` and
  `emitted=false`.
- **Bars-additive cleanup:** adequate. The v3 summary explicitly exempts Step 1 as markdown-only.
- **Odd-length PCM cleanup:** functionally adequate in Step 9 (`STEP3B-WS-PLAN.md:340-341`) because
  that is where binary PCM is converted to `PCMFrame`. The v3 summary claims Step 6 also says this,
  but Step 6 does not. That is minor; either add a Step 6 ws-lib-smoke row for odd-length payloads
  or narrow the summary to Step 9.
- **Step 11 port cleanup:** only partially folded. The v3 summary says ports are allocated free /
  configurable, but Step 11 still says fixed ports 8080/8081 (`STEP3B-WS-PLAN.md:383`). Minor fold:
  say default 8080/8081 but allow CLI/env override or free-port allocation in `run_compat.py`.

## Remaining Underspecification

No additional must-ask implementation question beyond the stale Step 5 key-file line. Once that
line is corrected, the remaining Step 6 summary mismatch and Step 11 port wording are minor
implementation clarifications, not plan blockers.

## Net

**GO-with-1-fold.** Fix the Step 5 key-file bullet so no instruction anywhere says to wire
`StatsCollector::record()` inside `SessionRuntime`. After that, this should reduce to
**MINOR_ONLY = CONVERGED**.
