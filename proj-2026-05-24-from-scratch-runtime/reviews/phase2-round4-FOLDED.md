# Phase-2 plan — Round 4 FOLDED (Codex + Opus, decision structure)

Inputs: `codex-phase2-round4.md`, `opus-phase2-round4.md`. Both built the SAME numeric GO/STOP tree independently
and converged tightly; Codex contributed two genuine improvements over Opus's draft (the 5090 ≥2.00× bar; the
TTFT-vs-TTFS orphan), and the full goal-traceability table. Goals: **G1 density**, **G2 P50↔P95 tail**.

## THE PRE-REGISTERED GO/STOP TREE (unified — freeze before building Step 1)

**Definitions:** `S_py_L40S` = fresh Python baseline, SLO-robust successful streams/box, re-measured back-to-back.
`G1_floor = max(28, 1.50·S_py_L40S)`. `TTFS_spread = ttfs_p95 − ttfs_p50` (server-side vad_stop→final).
`WER_bound`: native ≤ max(WER_py + 0.5pp, 1.10·WER_py), same corpus/tool/model/prompt. `Reject_bound`: ≤10%
intentional 1013 rejects of offered, ≤1% non-intentional errors of admitted; no-shed curve also reported.

**Step 0 — cheap kill-gates (5090, ~hours, BEFORE any EC2 spend):**
- 0a steady AOTI pool: one loader `num_runners=N` + per-worker streams. PASS = ≥1.15× at N=2, ≥1.30× at N=4,
  concurrent==serial (0 token/event mismatch), profiler shows kernels on ≥2 streams, peak mem <1.35×N=1 (no 2nd
  weight copy). STOP-candidate = <1.10×@N=2 / <1.20×@N=4, any mismatch, or weight duplication.
- 0b decode/ownership: per-worker SessionState + per-thread handles. PASS = 0 mismatches, `.item()` p95 ≤5% of
  per-chunk GPU elapsed, default-stream negative control ≥15% worse. STOP-candidate = mismatch or scalar-sync p95
  >10% or default-stream serialization in the per-stream path.
- 0c finalize fork + hot-bucket: same/mixed-bucket real finalize. PASS = 0 token/event/WER regression, finalize
  p95 runner-wait ≤25% of vad_stop→final, no stale-gen leak. STOP-candidate = mismatch / stale final / wait >50%.

**Step 1a — 5090 mini-sweep (spend-control proxy, NOT a project GO):**
- PASS-to-1b = real-decode multiplier **≥2.00×** vs N=1 same-harness + Step-0 still 0 mismatches + WER in bound +
  p95 vad_stop→final ≤300ms, p99 ≤500ms + `TTFS_spread` ≤1.25× the N=1 spread.
- TUNE/RETEST = 1.50×–2.00× with profiler attribution to a fixable harness/topology issue (no EC2 until ≥2.00× or
  a written non-predictive exception).
- STOP-candidate = <1.50×, or p95/p99/WER/correctness fail.
- **Both reviewers DISAGREE with Round-1's "≥1.5× on 5090 is enough"** — the headroom-rich GPU needs ≥2.00× so
  the tighter L40S has margin for the Step-4 haircut. (Codex's bar; Opus adopts.)

**Step 1b — L40S HARD GATE (ceiling, must EXCEED the realized bar):**
- PASS-to-build = `S_native_step1b ≥ max(34, 1.80·S_py_L40S)` (the 34 = 28/0.83, preserving the ≥28 realized
  floor after a ~17% scheduler/WS haircut) + 0 mismatches + WER in bound + p95 ≤300ms / p99 ≤500ms + `TTFS_spread`
  ≤1.10× Python's Step-4 baseline spread.
- CONDITIONAL (funding-risk hold) = ≥G1_floor but <1.80× → only proceed on explicit human risk-acceptance + a
  narrow de-risking prototype next.
- STOP-candidate = <G1_floor or SLO/WER/correctness fail.
- **Both DISAGREE with the R2 shorthand that L40S ≥1.5× is the Step-1b pass** — that's the *realized* gate, not the
  *ceiling*; using it at 1b is a predictable false GO.

**Step 4 — realized end-to-end (the binding TECHNICAL GO):**
- Density: `S_native_step4 ≥ G1_floor`; ceiling realization ≥0.83·S_native_step1b (below 0.75× not a clean GO).
- Tail (G2): native `TTFS_spread` improves ≥max(10ms,10%) vs Python at matched admitted load; **if Python spread
  already <50ms (it is — finalize-graph win ≈33ms), native within +5ms and p95 no worse** ⟸ this clause = the
  NON-REGRESSION bar Opus argued for; the two drafts converge here.
- SLO: p95 ≤300ms, p99 ≤500ms. Correctness: 0 oracle mismatch, no stale final after close/reset/shed, WER in
  bound. Shedding: meets Reject_bound; report both curves.

## GOAL TRACEABILITY (Codex's table — adopt into the plan)
| Goal | Step | Metric | Threshold |
|---|---|---|---|
| G1 primitive exists | 0a | AOTI pool throughput N=2/N=4, mem-flat, streams | ≥1.15×@N=2, ≥1.30×@N=4, mem<1.35×, no 2nd copy |
| G1 decode/finalize safe | 0b/0c | oracle equality, scalar-sync %, finalize wait | 0 mismatch; `.item()` p95 ≤5%; finalize wait ≤25% TTFS |
| G1 5090 spend-control | 1a | real-decode multiplier vs N=1 | PASS ≥2.00×; STOP <1.50× |
| G1 L40S ceiling | 1b | compute-ceiling streams/box | PASS ≥max(34,1.80·S_py); STOP <G1_floor |
| G1 realized | 4 | admitted-successful streams/box | GO ≥G1_floor with Reject_bound |
| G2 tail reference | 1a/1b | TTFS p50/p95/p99, spread | 1a ≤1.25×N=1; 1b ≤1.10×Py; p95≤300/p99≤500 |
| G2 realized | 4 | native-vs-Py TTFS_spread @matched load | ≥max(10ms,10%) better; if Py<50ms within +5ms, p95 no worse |
| Correctness | 0–4 | token/event equality, stale-gen, WER | 0 oracle mismatch; WER in bound |
| Admission validity | 4 | rejected/offered, errors/admitted | 1013 ≤10%; errors ≤1%; no-shed curve reported |

## TWO OPEN QUESTIONS FOR THE USER (require ratification before Step 4)
1. **Does the user ratify the two-goal gate?** The original 0.0 threshold was **density-ONLY**. Adding G2 (tail)
   as a co-equal Step-4 conjunct is a real scope change. Recommended: adopt it as a **NON-REGRESSION guardrail**
   (don't widen the tail while adding density), with strict-improvement as upside — NOT a strict-improvement STOP
   bar (Python's tail is already ~33ms; an over-tight bar would be a wrong project-killer). If declined, report G2
   independently but don't call a density-only pass a two-goal Phase-2 success.
2. **Is G2 "first-token TTFT" or "vad_stop→final TTFS"?** (Codex, sharp.) The user said "TTFT"; the loadgen
   measures **TTFS** (vad_stop→final), NOT first-token TTFT. If literal first-token TTFT is meant, it's currently
   **ORPHANED** (un-instrumented). Recommended (Codex option b): add first-token TTFT timestamps in Step 3/4 and
   bind on `TTFT_spread`, keeping `TTFS_spread` as the attribution metric. (Opus note: per the roofline, first-token
   TTFT P50 is VAD+WAN-dominated and only ~12–19ms is server-side-movable — so a first-token TTFT *spread* gate
   only makes sense on the server-side component; state that scope either way.)

## PRECONDITIONS, CORROBORATION, SEPARATION (both converge)
- **Step 4 invalid unless Step 2/3 preconditions are green (Codex B3):** box-global active cap + backlog-count cap
  (swept ~8/10/12) with offered/admitted/rejected/queue logged; a declared numeric **priority-lane policy**
  (partitioned `N_finalize_reserved≥1` with steady-starvation p95 ≤2× no-finalize, OR weighted with finalize
  runner-wait ≤25% TTFS); **WS-tail** decomposed (overhead p95 ≤10% of TTFS or G2 not claimed as runtime tail);
  **stale-gen** tests all pass.
- **A STOP must be CORROBORATED (both, B5):** 3 runs (CV ≤10%) + negative control detects serialization +
  topology sweep (num_runners=1, =N, fallback loaders/MPS) all fail + profiler/counter attribution to a HARDWARE
  limit (not lock/default-stream/un-wired-stream) + harness-health logged. A fallback-topology PASS is a *pivot*,
  not a Phase-2 STOP. (Protects a viable project from a harness-bug false STOP.)
- **Technical gate ≠ funding decision (both, B6):** Technical GO = Step 4 clears density+tail+WER+shed. Funding GO
  = separate human call (≥40–60 eng-wk + a permanent 2nd stack, strategic bet, no COGS break-even). A Funding STOP
  despite a Technical GO is legitimate and must not be rewritten as a failed experiment.
- **Opus distinctive (preserve):** consider a **"Step 3-lite"** (minimal WS echo + admission/priority scheduler,
  no full feature set) to get an early *realized* number before sinking the full Step-3 build — cheaper mitigation
  than relying only on the 1.8× ceiling margin.

## NET FOR ROUND 4
The decision is now a fully numeric, adversarially-stressed tree. Two improvements banked (5090 ≥2.00×; the
TTFT-vs-TTFS orphan). Everything else converged. Two items genuinely need the USER: ratify the two-goal gate
(as a non-regression guardrail) and disambiguate G2 = TTFT vs TTFS. Round 5: final convergence, confirm no
residual disagreement, and produce the consolidated PHASE2-PLAN.md edit set.
