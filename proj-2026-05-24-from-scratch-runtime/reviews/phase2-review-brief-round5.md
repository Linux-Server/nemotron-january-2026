# Phase-2 review — Round 5 charge (final convergence + sign-off)

Rounds 1–4 folded (`phase2-round{1,2,3,4}-FOLDED.md` — read all four). This is the final round. The goal is
CONVERGENCE and a clean sign-off, not new scope. Goals: **G1 density**, **G2 P50↔P95 tail**.

## What to produce
1. **Residual-disagreement scan.** Across all four rounds, is there ANY finding where Codex and Opus still
   disagree, or any threshold in the Round-4 tree that is internally inconsistent or unsupported? List them; if
   none, say so explicitly.
2. **Completeness check.** Is there any decision-critical gap STILL not covered by Rounds 1–4 — anything that
   could make a GO/STOP wrong that we haven't gated? Re-scan the two goals end-to-end one last time.
3. **The minimum-viable edit set.** Of all the recommended edits across 4 rounds, which are the MUST-HAVE edits to
   PHASE2-PLAN.md before Step 1 is built (vs nice-to-have)? Rank them. The plan author needs a crisp "change these
   N things" list, not 4 rounds of prose.
4. **Sign-off.** Your bottom line: is the Phase-2 plan — WITH the must-have edits applied — sound enough to build
   Step 1 (the Step-0 kill-gates first)? Or are there remaining blockers? State it plainly.

## Constraints on the sign-off
- Distinguish what the reviews can settle (the measurement design, the gates, the thresholds) from what only the
  USER can settle (ratifying the two-goal gate; G2 = first-token TTFT vs vad_stop→final TTFS).
- Do not relitigate settled points. Build on the folds.
- Be honest: if the plan with edits is sound, say GO-to-build; if a blocker remains, name it.

Write to `proj-2026-05-24-from-scratch-runtime/reviews/codex-phase2-round5.md`. Round 5 of 5 (final).
