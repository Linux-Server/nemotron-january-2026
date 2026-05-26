# Phase-2 plan — Round 6 FOLDED (verification of the v2 rewrite)

Inputs: `codex-phase2-round6.md`, `opus-phase2-round6.md`. A fresh paired adversarial read of the rewritten
`PHASE2-PLAN.md` (v2) against the 8 must-have edits + the user's ratified goal contract.

## Both reviewers confirm: edits applied faithfully; rewrite buildable
- All 8 must-have edits present and correct; goal contract is **density-only GO + G2 (server-side TTFS spread)
  reported-not-gated, VAD/WAN out of scope**; Step-1b tail is correctly a reported/build-risk signal (not an
  auto-STOP); multi-turn caveat present; Step 0 is crisp enough to build without re-deriving the gates.
- **VERDICT (both): ready to build Step 0 — YES.** The 6 defects below are SLO/threshold-text precision issues
  that bite at the Step-1a/1b/4 gates, NOT at Step 0 (overlap/correctness/memory), so Step 0 can start now.

## 6 DEFECTS FOUND → ALL FIXED in v2 (complementary; little overlap between reviewers)
| # | Reviewer | Defect | Fix applied |
|---|---|---|---|
| 1 | Opus D1 | `ttfs` SLO 300/500ms unpinned + likely too LOOSE (server-side must satisfy server+VAD~200+WAN~23 ≤ 400 end-to-end → ≈≤175–200ms p95; 300ms busts it) | Added a **ttfs SLO budget** definition: PROVISIONAL 300/500, marked **PIN before Step-1a** by back-solving from the <400ms end-to-end budget |
| 2 | Codex 3 | bare `p95/p99` ambiguous vs the keep-up `lag_p95` | Renamed to a single named **ttfs SLO budget**; Steps 1a/1b/4 now reference it; "always name `ttfs_*` vs `lag_*`" |
| 3 | Codex 1 | `SLO-robust` said "zero non-intentional errors" but `Reject_bound` allows ≤1% — contradiction | Aligned SLO-robust to "non-intentional admitted error rate ≤1%" |
| 4 | Codex 2 | Step-1b CONDITIONAL band ambiguous when `1.80·S_py < 34` | CONDITIONAL = `≥G1_floor but <max(34, 1.80·S_py_L40S)` |
| 5 | Codex 4 | Step-4 "5090 first, then L40S" misreadable as a binding 5090 gate | "**Optional 5090 rehearsal**; L40S is the binding Step-4 technical GO" |
| 6 | Opus D2 | Step-2 text said "PAIRED REVIEW" but Rules listed Steps 2,4 as Opus-review — contradiction | Moved **Step 2 into the paired set** (it's a decision-critical DESIGN review); Step 4 stays Opus-review |

(Cosmetic, left as-is: Step 0 marked `[~]` though the harness isn't started — accepted as "Phase 2 is the active
focus.")

## Net
The 6-round paired adversarial review is complete. The v2 plan with these 6 fixes is internally consistent,
faithfully encodes the user's ratified contract, and is signed off by both reviewers to build Step 0 (the cheap
kill-gates) first. The one item that still needs a real number before it bites: **pin the ttfs SLO budget**
(BUDGET_P95/P99) from the production <400ms end-to-end budget — required before Step-1a's SLO gate is used, not
before Step 0.
