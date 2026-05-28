<task>
**Round 3 adversarial review of `STEP3B-WS-PLAN.md` v3** (just committed). Rounds 1-2 folded; v3
addresses Codex Round 2's 1 must-fold (StatsCollector ownership/order) + 3 minor cleanups.

Stop condition: if both Round 3 reviews come back with MINOR_ONLY → CONVERGED → ready to
`/implement`. Codex Round 2 predicted "next pass MINOR_ONLY = CONVERGED" — verify or refute.

Write your fold-ready review to `reviews/codex-Step3b-plan-review-round3.md`. Adversarial,
specific. Don't re-flag Rounds 1-2 folds.
</task>

<context>
**The plan under review**: `STEP3B-WS-PLAN.md` v3 (committed in latest commit). v3 fold:
- Step 5's `record()` wiring removed from SessionRuntime; WS worker (Step 9) owns it.
- Step 9 lifecycle prose updated to: finalize_now() → produce WireEvent + last_timing() → stale-gen
  check → emit final (or drop) → stamp was_suppressed + emitted → record() once → close WS-1000.
- Bars-additive header notes Step 1 exemption.
- Odd-length PCM WS-1003 in Step 9 prose.
- Step 11 ports configurable.

**ASK / structure your Round 3 attack** (narrow given v3 is a tight fold):
1. Did v3's must-fold land correctly + completely? Cross-check Step 5 + Step 9 wording.
2. Did v3 introduce any new ambiguity or contradiction?
3. Are the 3 minor cleanups adequate?
4. Anything still under-specified that the implementer would have to ask about?
5. Net verdict: MINOR_ONLY = CONVERGED / GO-with-1-fold / substantive.
</context>

<verification_loop>
Doc/plan review only — NO BUILD, NO RUN. Read v3 + the diff against v2.
</verification_loop>

<action_safety>
Write only the review doc.
</action_safety>

<compact_output_contract>
Report path + one-paragraph verdict + top 1-3 must-folds (or "minor only — converged").
</compact_output_contract>
