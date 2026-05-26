# Codex Phase 2 Review - Round 6 Verification

Scope: verification of rewritten `PHASE2-PLAN.md` v2 against `phase2-review-brief-round6.md` and `phase2-round5-FOLDED.md`.

## Correctly Applied

- All 8 must-have edits are present: staged numeric tree, goal contract, ownership/topology contract, correctness-before-perf, telemetry schema, corroborated STOP protocol, Step 2/3 preconditions, and Step 4 apples-to-apples manifest plus multi-turn decision (`PHASE2-PLAN.md:34-49`, `PHASE2-PLAN.md:51-56`, `PHASE2-PLAN.md:58-128`, `PHASE2-PLAN.md:130-145`).
- The goal contract is correctly density-only: G1 is the GO/STOP gate; G2 is server-side `TTFS_spread = ttfs_p95 - ttfs_p50`, reported at every gate, not a STOP criterion; VAD/WAN are out of scope; a density pass is not called a two-goal success (`PHASE2-PLAN.md:34-40`).
- Step 1b tail is correctly non-gating: `TTFS_spread` is GREEN at `<=1.10x` Python but otherwise a build-risk signal, not an auto-STOP; binding tail is only reported at Step 4 (`PHASE2-PLAN.md:85-90`, matching `phase2-round5-FOLDED.md:21-27`).
- Step 4 correctly remains a density-only technical GO and reports G2 separately (`PHASE2-PLAN.md:109-118`).
- The multi-turn caveat is present: add a multi-turn subcurve or explicitly scope the gate to single-utterance sessions (`PHASE2-PLAN.md:117-118`).
- Step 0 is buildable: 0a/0b/0c have concrete topology, pass thresholds, STOP-candidates, correctness requirements, memory checks, and negative controls (`PHASE2-PLAN.md:60-75`, `PHASE2-PLAN.md:140-143`).

## Defects To Fix

1. **Inconsistent non-intentional error threshold.**
   - Plan lines: `PHASE2-PLAN.md:45-49`, `PHASE2-PLAN.md:109-113`.
   - Problem: `SLO-robust` says "zero non-intentional errors" at line 45, but `Reject_bound` permits non-intentional errors `<=1%` of admitted at line 48, and Step 4 uses `Reject_bound` at line 112.
   - Fix: make line 45 consistent with `Reject_bound`: `SLO-robust = lag_p95 < 500ms AND ttfs_p95/ttfs_p99 in budget AND non-intentional admitted error rate <=1%; count admitted-successful streams, never offered.`

2. **Step 1b conditional band is under-specified when the absolute 34-stream floor dominates.**
   - Plan lines: `PHASE2-PLAN.md:85-90`, `PHASE2-PLAN.md:151-152`.
   - Problem: PASS is `S_native_step1b >= max(34, 1.80*S_py_L40S)`, but CONDITIONAL is written as `>=G1_floor but <1.80x`. If `1.80*S_py_L40S <34`, the band between `1.80*S_py_L40S` and `34` is ambiguous.
   - Fix: change line 88 to: `CONDITIONAL = >=G1_floor but <max(34, 1.80*S_py_L40S) -> proceed only on explicit human risk-acceptance + a narrow de-risking prototype.`

3. **Bare `p95/p99` references are ambiguous because the plan also uses `lag_p95`.**
   - Plan lines: `PHASE2-PLAN.md:45`, `PHASE2-PLAN.md:80-82`, `PHASE2-PLAN.md:85-90`, `PHASE2-PLAN.md:109-113`.
   - Problem: some gates say `p95 <=300ms / p99 <=500ms` without naming the metric. Since line 45 also defines `lag_p95 <500ms`, an implementer could confuse processing lag with `ttfs` SLO.
   - Fix: spell these as `ttfs_p95 <=300ms / ttfs_p99 <=500ms` everywhere, and keep `lag_p95 <500ms` as the separate keep-up requirement.

4. **Step 4's "5090 first, then L40S" wording can be misread as a binding 5090 Step 4 gate.**
   - Plan lines: `PHASE2-PLAN.md:109-119`.
   - Problem: Step 4 is the binding L40S realized-density decision, but line 118 says "5090 first, then L40S" without saying 5090 is only a rehearsal.
   - Fix: change to: `Optional 5090 rehearsal first; L40S is the binding Step-4 technical GO.`

## Verdict

Ready to build Step 0: **yes**. The v2 plan faithfully applies the Round 5 requirements and is crisp enough for an engineer to start the Step 0 kill-gate harness without re-deriving the gates.

Residual defects: the four text/threshold fixes above should be applied before freezing v2 or using it to judge Step 1b/Step 4, but none blocks the first Step 0 build action.
