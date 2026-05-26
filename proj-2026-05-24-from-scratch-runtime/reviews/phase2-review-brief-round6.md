# Phase-2 review — Round 6 charge (VERIFICATION of the v2 rewrite)

`PHASE2-PLAN.md` was REWRITTEN (v2) to apply the 8 must-have edits from Rounds 1–5 (folds:
`phase2-round{1..5}-FOLDED.md`) and the user's ratified goal contract. This round VERIFIES the rewrite. It is a
fresh adversarial read of the NEW plan — do not assume the edits were applied correctly.

## The user's ratified goal contract (must be encoded correctly)
- **GO/STOP gate = G1 density ONLY** (the frozen 0.0 threshold: ≥1.5× L40S realized). NOT conjunctive with G2.
- **G2 = server-side `TTFS_spread = ttfs_p95 − ttfs_p50` (vad_stop→final), REPORTED at every gate, NOT gated.**
  VAD/WAN out of scope. A density pass must NOT be described as a "two-goal" success.

## Verify (read the NEW PHASE2-PLAN.md line-by-line)
1. **Faithful application** of all 8 edits: (1) the staged numeric tree replacing "meaningfully"; (2) the goal
   contract; (3) ownership/topology contract; (4) correctness-before-perf first; (5) telemetry schema; (6)
   corroborated-STOP protocol; (7) Step-2/3 preconditions; (8) Step-4 apples-to-apples manifest + multi-turn
   decision. Flag any edit that's missing, half-applied, or distorted.
2. **Internal consistency.** Any contradictory thresholds across sections (the definitions block vs the Step
   text vs the progress table)? Is the Step-1b tail correctly a REPORTED/build-risk signal and not an auto-STOP
   (given G2 is reported, not gated)? Are `G1_floor`, the 1.80×/34 ceiling, the ≥2.00× 5090 bar, `WER_bound`,
   `Reject_bound` used consistently everywhere they appear?
3. **Goal-contract correctness.** Is the GO genuinely density-only everywhere (no leftover place where G2 gates)?
   Is G2 consistently "reported, server-side TTFS, VAD/WAN out of scope"? Is the multi-turn coverage caveat
   present?
4. **New problems.** Did the rewrite introduce any new error, ambiguity, dangling reference, or over-claim that
   the 5 rounds did not have? Is anything now UNDER-specified that was clear before?
5. **Buildability.** Is Step 0 now crisp enough for an engineer to start the harness without re-deriving the
   gates? Anything still blocking the first build action?

Be concise. Confirm what is correctly applied, then list any DEFECT (with the plan line) + a fix. End with a
plain verdict: is the v2 plan ready to build Step 0, or are there residual defects? Write to
`proj-2026-05-24-from-scratch-runtime/reviews/codex-phase2-round6.md`.
