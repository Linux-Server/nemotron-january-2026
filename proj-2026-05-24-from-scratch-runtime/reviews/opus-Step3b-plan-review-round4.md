# STEP3B-WS-PLAN v4 — Opus Round 4 adversarial review (2026-05-28)

Reviewing v4 (committed `89d8ced`) from-scratch. v4 is a tight one-line fold of v3 (Step 5
"Key files" parenthetical fix per Codex Round 3 must-fold).

## Verdict

**MINOR_ONLY → CONVERGED.** v4's one-line fix lands cleanly. Step 5's "Key files" parenthetical
now says "(populate SessionTiming / last_timing() ONLY; no StatsCollector::record() call — v4 fold
per Codex Round 3: WS worker owns record() in Step 9)" — matches the Step 5 ownership prose AND
the Step 9 lifecycle ordering. The implementer following the key-files checklist now sees the
no-record() instruction. The Round 2 hazard cannot be re-introduced via the key-files line.

## Cross-check

- Step 5 ownership prose ↔ key-files parenthetical: consistent. ✓
- Step 9 lifecycle prose: unchanged (still has correct AFTER-emit-decision record() ordering). ✓
- No other v4 edits → no regressions possible.

## Remaining minor items (acceptable; fold during implementation)

Same as Opus Round 3:
1. Step 11 ports configurable: claimed in fold-notes; Step 11 prose still defaults 8080/8081.
   Implementer adds `--python-port` / `--cpp-port` flags (or env) naturally.
2. Step 6 odd-length PCM note: Step 9 handles it correctly; Step 6 bar could mention "framing
   layer passes binary through; PCM validation is Step 9" for clarity. Not blocking.
3. Bars-additive Step 1 exemption: in fold-notes only, not in the in-body header. Implementer
   reads both naturally.

None of these blocks `/implement`.

## Net

**v4 = CONVERGED at Round 4 for both reviewers** (predicted by Codex Round 3). Total: 3 substantive
rounds + 1 verification round = converged 1 round under the 5-round budget. PLAN ready for
`/implement STEP3B-WS-PLAN.md` after the in-flight Part A v1 + L40S B3-FU sweeps land + are
audited (per the existing prerequisite chain).
