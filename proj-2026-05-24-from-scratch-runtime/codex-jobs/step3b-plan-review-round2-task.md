<task>
**Round 2 adversarial review of `STEP3B-WS-PLAN.md` v2** (just committed). Rounds 1 folded;
Round 2 attacks v2.

Stop condition: if both Round 2 reviews come back with MINOR_ONLY → CONVERGED → ready to
`/implement`.

Write your fold-ready review to `reviews/codex-Step3b-plan-review-round2.md`. Adversarial,
specific. Don't re-flag Round 1 folds.
</task>

<context>
**The plan under review**: `STEP3B-WS-PLAN.md` v2 (committed in latest commit). v2 folds Round 1's
3 convergent must-folds + Codex's v4-contract restorations + Opus's bars-additive + N=200 gate.

**Related context**: same as Round 1 task (PHASE2-PLAN.md, PLAN_RULES.md, Step3b-WS-architecture.md
v4, the Round 1 reviews codex-Step3b-plan-review-round1.md + opus-Step3b-plan-review-round1.md).

**ASK / structure your Round 2 attack**:

1. **Did v2's folds land correctly?**
   - Step 11 commit semantics: clear + executable now?
   - Part A v1 salvage audit gate in Step 3: clear what produces the audit + when?
   - Step 4 wrapper-equivalence harness: implementable as written?
   - Step 9 lifecycle oracle: complete + executable?
   - Step 11 bars (correctness + perf gate): concrete enough?
   - Step 5 admission Python-shape: matches v4 §IV exactly?
   - Step 7 v4 §XI/§XIII/§XVI additions: clear + scoped?
   - Step 8 Silero N=64 probe: implementable + fail criterion concrete?

2. **Did v2 introduce new ambiguities or contradictions?** Cross-check the bars-additive header
   vs each step's bar — any step that's incompatible?

3. **What remains genuinely under-specified** that the implementer would have to ask about?

4. **Net verdict**:
   - **MINOR_ONLY = CONVERGED**: ready to /implement.
   - **GO-with-1-2-must-folds-to-v3**: small specific changes.
   - **GO-with-substantive-revisions**: would surprise me.
</context>

<verification_loop>
Doc/plan review only — NO BUILD, NO RUN.
</verification_loop>

<action_safety>
Write only the review doc.
</action_safety>

<compact_output_contract>
Report path + one-paragraph verdict + top 1-3 must-folds (or "minor only — converged").
</compact_output_contract>
