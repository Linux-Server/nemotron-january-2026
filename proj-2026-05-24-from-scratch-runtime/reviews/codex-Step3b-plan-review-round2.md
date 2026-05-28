# Codex Step 3b Plan Review - Round 2

Verdict: **GO-with-1-must-fold-to-v3**. v2 correctly folds the Round 1 blockers in shape: Step 11
commit semantics are no longer circular, the Part A v1 salvage audit is a real Step 3 gate, the Step
4/9/11 bars are executable, and the v4 contract restorations called out in Round 1 are mostly present.
I would not redesign the plan. One remaining lifecycle/stats ownership contradiction is still
must-fold before `/implement`, because it can make `/stats` and stale-generation suppression wrong even
if the Step 9 smoke superficially passes.

## Must Fold

1. **Pin exactly where `StatsCollector::record()` happens: after the WS final emit decision, exactly once.**

   Step 5 says to wire `StatsCollector::record(timing, emitted)` into
   `SessionRuntime::finalize_now()` (`STEP3B-WS-PLAN.md:216-217`). That is too early: `finalize_now()`
   cannot know whether the final event was stale-dropped, sent successfully, send-failed, or timed out.
   Step 9 then gives two different orderings: the lifecycle prose records before emitting the final
   event (`:309-311`), while the bar correctly expects `record()` to receive `was_suppressed=true` and
   an emitted flag matching the stale/drop/send decision (`:321-324`). v4 §V says the worker records
   after the emit decision.

   Fold text:
   - `SessionRuntime::finalize_now()` produces `WireEvent` plus `SessionTiming` / `last_timing()`;
     it does **not** own or call `StatsCollector::record()`.
   - The WS worker performs stale-gen check, final send/drop/timeout decision, stamps
     `was_suppressed` and `emitted`, then calls `StatsCollector::record(timing, emitted)` exactly once.
   - Step 5 can build/test `StatsCollector` in isolation; Step 9 owns the production recording
     integration.

## Fold Check

- **Step 11 commit semantics:** clear now. Steps 8-10 can mark `[x]` on their local bars; Step 11 is the
  integration gate and stays blocked if combined behavior fails.
- **Part A v1 salvage audit:** clear enough. Step 3 now says audit first, classify every hunk, write
  `reviews/part-a-v1-salvage-audit.md`, then carve.
- **Step 4 wrapper-equivalence harness:** implementable. Fixture-driven `SessionRuntime` equivalence
  against `finalize_ref` is the right bar.
- **Step 9 lifecycle oracle:** nearly complete, but blocked by the `record()` ordering contradiction
  above.
- **Step 11 correctness/perf gate:** concrete. The N=8 WS-overhead check is bounded and executable; it
  is a low-load overhead guard, not a full production N=64 load test, but that is acceptable for this
  step plan if documented as such.
- **Step 5 admission Python-shape:** matches v4 §IV fields.
- **Step 7 v4 §XI/§XIII/§XVI additions:** landed: admin pool, pid/process_label, no default port,
  MPS smoke, grouped config table.
- **Step 8 Silero N=64 probe:** implementable. Minor wording issue: replace "`e.g., > 50% of one core`"
  with a hard failure threshold so the implementer cannot treat it as illustrative.

## Minor Cleanup

- The bars-additive header says every Step's bar is additive, but Step 1 is explicitly markdown-only
  and Step 2 has no local bar. Clarify that the global build/harness/N=200 smoke protocol applies to
  code steps and setup steps once they affect a build target; Step 1 is exempt.
- Add the v4 odd-length PCM payload rejection explicitly to Step 6 or Step 9: odd byte-length binary
  payloads close WS-1003 before constructing `PCMFrame`.
- Consider making Step 11's port selection configurable/free-port allocated instead of fixed 8080/8081
  to avoid local test flake.

## Net

v2 is very close, but not quite converged. Fold the single StatsCollector ownership/order fix, then I
expect the next pass to be **MINOR_ONLY = CONVERGED**.
