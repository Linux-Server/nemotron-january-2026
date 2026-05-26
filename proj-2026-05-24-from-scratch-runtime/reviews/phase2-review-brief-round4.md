# Phase-2 review — Round 4 charge (decision structure: numeric GO/STOP tree + goals traceability + stress the bars)

Rounds 1–3 folded (`phase2-round{1,2,3}-FOLDED.md` — read them). Round 3 enumerated the false-GO/false-STOP modes.
Round 4 makes the decision RIGOROUS: a pre-registered numeric GO/STOP tree, end-to-end traceability of both goals,
and an adversarial stress of the thresholds themselves. Goals: **G1 density**, **G2 P50↔P95 tail**.

## What to produce
1. **A pre-registered GO/STOP tree with NUMBERS for every gate**, spanning: Step 0 kill-gates (binary), Step 1a
   (5090), Step 1b (L40S), Step 4 (realized). For each gate state the metric, the threshold, and the action on
   pass/fail. Where a threshold isn't yet derivable, say what measurement sets it.
2. **End-to-end goal traceability:** map G1 and G2 each to the exact step + metric + threshold that measures it.
   Flag any orphaned goal (named but never gated) or any gate not tied to a goal.
3. **Stress the bars** (the most important part): are the thresholds the RIGHT bars?

## Specific tensions to resolve (be adversarial)
- **The original gate was DENSITY-ONLY.** The project's pre-registered 0.0 threshold is "≥1.5× L40S density
  (≥~28/box), strategic capability bet, no COGS break-even." **G2 (tail) is a SECOND goal the user added.** So a
  GO should now require BOTH density ≥1.5× AND a tail criterion — you must not buy density by widening the tail.
  Is that the right framing? What exactly is the G2 bar: native `ttfs_p95−p50` ≤ Python's at matched load (no
  worse spread while denser)? strictly tighter? an absolute ms bound? Pre-register it and note that adding a tail
  conjunct CHANGES the original density-only threshold (the user should ratify).
- **5090 vs L40S numeric bar.** Step 1a is on the 5090; the gate is L40S. The mock got ≥3× (5090) vs ~2–2.5×
  (L40S). What 5090 real-decode multiplier should PASS to 1b vs STOP? (e.g., STOP if 5090 < 1.5× — if even the
  headroom-rich GPU can't clear the bar, L40S won't; require a margin ≥X above 1.5× to pass to the L40S gate.)
- **Ceiling vs realized margin.** Step 1b is a best-case ceiling; Step 4 is realized (scheduler+WS+shed+stale-gen).
  What Step-1b margin above 1.5× is needed so a GO isn't falsified at Step 4?
- **False-STOP robustness (protect the project).** The whole project hinges on Step 1. A STOP from a badly-built
  spike (wrong loader topology, per-thread streams not wired, keep-up-vs-SLO-robust confusion, the AOTI execution
  lock mis-measured) would kill a viable project. So a STOP finding must be CORROBORATED — what evidence must agree
  (negative control + topology sweep + profiler trace + counter attribution) before a negative is accepted as a
  real STOP rather than a harness bug?
- **Technical gate vs funding decision.** Even a clean ≥1.5× + tighter-tail PASS still must clear conjunct-1
  ("worth ~40–60 eng-wk + a 2nd stack," resolved as a strategic bet). Keep the technical gate and the funding
  decision SEPARATE so a technical PASS isn't auto-read as "build it." Where in the tree does the human funding
  decision sit?
- **Are there enough/too many gates?** Could the plan STOP-early cheaper (a Step-0 kill that obviates the rest)?
  Could it falsely continue past a gate that should have stopped it?

## Method
Read `PHASE2-PLAN.md` once more against the three folds; reason about the decision tree end-to-end. You do not
need to re-cite every line already cited in Rounds 1–3 — build ON them. Where you DISAGREE with a prior round's
proposed threshold, say so.

Write to `proj-2026-05-24-from-scratch-runtime/reviews/codex-phase2-round4.md`. Round 4 of 5.
