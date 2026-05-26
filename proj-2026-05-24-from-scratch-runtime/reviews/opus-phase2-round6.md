# Opus — Phase-2 plan review, Round 6 (VERIFICATION of the v2 rewrite)

Fresh adversarial read of the rewritten `PHASE2-PLAN.md` against the 8 must-have edits + the user's ratified goal
contract (density-only GO; G2 = server-side TTFS spread, reported not gated; VAD/WAN out of scope).

## Correctly applied (confirmed)
- **Goal contract:** G1 is the GO/STOP gate; G2 = `ttfs_p95−ttfs_p50`, "REPORTED at every gate, NOT a STOP
  criterion," VAD/WAN out of scope, "a density pass is NOT described as a two-goal success" (`:35-40`). ✓ matches
  the ratification exactly.
- **Edit #1 staged numeric tree:** Step 0 → 1a (≥2.00×) → 1b (≥max(34, 1.80·S_py)) → 4 (≥G1_floor). ✓
- **Edit #3 ownership/topology contract** (`:51-56`), **#4 correctness-first** (Rules), **#5 telemetry schema**,
  **#6 corroborated-STOP**, **#7 Step-2/3 preconditions**, **#8 apples-to-apples manifest + multi-turn decision**
  (`:109-120`). ✓ all present.
- **Consistency win:** Step-1b tail is correctly a REPORTED/build-risk signal, not an auto-STOP (`:88-90`); Step-1b
  and Step-4 STOP/GO lists exclude tail — consistent with "G2 reported, not gated." ✓ (resolves Codex R5's
  consistency note cleanly.)
- `G1_floor`, the 1.80×/34 ceiling, ≥2.00×, `WER_bound`, `Reject_bound` are defined once and used consistently. ✓
- Completeness reinforcements landed: target-N memory re-check (`:65`), min(BW,CPU,lock) ceiling (`:74-75`),
  de-risked graph fallback (`:31-33` + Rules), multi-turn caveat (`:118-120`). ✓

## DEFECTS (introduced/surviving in the rewrite)

### D1 (MAJOR) — the absolute SLO numbers (p95 ≤300ms / p99 ≤500ms) are UNPINNED placeholders and may be too loose
The plan hardcodes server-side `ttfs` p95 ≤300ms / p99 ≤500ms at Steps 1a/1b/4 (`:80, :87, :111`). These are
Codex's Round-4 *provisional* numbers (it explicitly flagged "I used 300/500" as a QUESTION). Two problems:
1. **Not reconciled with the real budget.** The production budget is **<400ms END-TO-END** (= VAD + WAN +
   server-side). `ttfs` here is **server-side, co-located (no WAN)**. So the server-side bound must satisfy
   `server-side + VAD(~200) + WAN(~23) ≤ 400` → server-side ttfs p95 ≈ **≤175ms**, not 300ms. A 300ms server-side
   p95 likely **busts the end-to-end budget**. (Sanity: the shipped Python server already hits ~279ms p95
   *end-to-end-ish* on L40S after the finalize-graph win — so server-side alone must be well under that.)
2. The loadgen keep-up SLO is `lag_p95 < 500ms` (a different axis from ttfs). The plan correctly separates them,
   but the ttfs budget itself is unpinned.
→ **Fix:** mark 300/500 as PROVISIONAL and pin the server-side `ttfs` budget by back-solving from the <400ms
end-to-end budget (≈≤175–200ms p95 server-side) OR explicitly state 300/500 is a placeholder to be replaced by the
Step-4 production-SLO reconciliation. As written, a stream could "pass" the SLO gate at 300ms server-side yet bust
the real budget — a latent false GO on the SLO axis.

### D2 (MINOR-consistency) — Step-2 review intensity is self-contradictory
Step 2's text says "**PAIRED REVIEW** of the design before building" (`:98`), but the Rules line lists only
"Step 0, 1b, 3, 5" as paired and puts "Steps 2, 4 → Opus review + independent re-run" (`:130-131`). Contradiction.
→ **Fix:** scheduler design (Step 2) is decision-critical (Round 3 leaned on it heavily) and has a "before
building" design gate → **add Step 2 to the paired set** in Rules. (Or, if keeping Step 2 as Opus-review, drop
"PAIRED" from the Step-2 text. Recommend the former.)

### D3 (COSMETIC) — Step 0 marked `[~]` in-progress though the harness isn't started
`:60` marks Step 0 `[~]` and the progress table says "in-progress," but no Step-0 harness code exists yet (carried
over from the old Step-1 `[~]`). Acceptable as "Phase 2 is the active focus," but strictly it's the next-to-build,
not in-progress. Leave or relabel "next" — author's call; not a blocker.

## New problems / under-specification
- None beyond D1–D3. Nothing the 5 rounds had is lost; the harness "shape" is implicit in the ownership contract +
  Step 0a/b/c but an engineer can start from it.

## Verdict
**The v2 rewrite faithfully applies the 8 edits and the ratified goal contract, and is internally consistent
except for D1 (unpinned/too-loose SLO numbers — MAJOR, fix before Step 1a relies on them) and D2 (Step-2 paired/
Opus contradiction — quick fix).** With D1 pinned and D2 aligned, the plan is ready to build Step 0. D1 doesn't
block the Step-0 kill-gates (they're overlap/correctness/memory, not SLO-latency), so Step 0 can start now while
D1 is resolved before Step 1a's SLO gate bites.
