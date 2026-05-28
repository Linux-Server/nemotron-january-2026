<task>
**Round 4 adversarial review of `STEP3B-WS-PLAN.md` v4** (just committed). Rounds 1-3 folded;
v4 = one-line fix to Step 5's "Key files" parenthetical (per Codex Round 3 must-fold).

Stop condition: if both Round 4 reviews come back with MINOR_ONLY → CONVERGED → ready to
`/implement`. Codex Round 3 predicted this would converge after the one-line fix.

Write your fold-ready review to `reviews/codex-Step3b-plan-review-round4.md`.
</task>

<context>
**The plan under review**: `STEP3B-WS-PLAN.md` v4. v4 is a tiny tight fold of v3 (one-line edit to
Step 5's "Key files" parenthetical removing the contradictory "wire record call" instruction).

**ASK**: did the v4 fix land + are there any other Step-5 / Step-9 / cross-step contradictions?
Anything else under-specified that the implementer would have to ask about?

**Net verdict**: MINOR_ONLY = CONVERGED OR GO-with-1-fold.
</context>

<verification_loop>
Doc/plan review only. Bounded — v4 is a one-line fold of v3.
</verification_loop>

<action_safety>
Write only the review doc.
</action_safety>

<compact_output_contract>
Report path + one-paragraph verdict + top 1-2 must-folds (or "minor only — converged").
</compact_output_contract>
