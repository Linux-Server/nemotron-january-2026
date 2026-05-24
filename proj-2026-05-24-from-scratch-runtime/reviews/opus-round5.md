# Opus adversarial review — Round 5 (final consistency pass + verdict)

By v5 the substantive findings (rounds 1–4) are folded and convergent. This pass hunts residual contradictions and
gives a verdict. Only a few items remain.

## [MAJOR] 1. The "native-under-MPS, reduced density" escape in the 0.4 decision tree is weaker than it reads — say so
If 0.1 shows a single native process can't overlap finalize+steady without MPS/multi-proc, the decision tree offers
"proceed native-under-MPS with reduced density target." But in that world the native runtime runs the *same* MPS +
multi-proc topology as Python, so the **density story is gone** — the only remaining native win is the **per-process
tail** (off-event-loop dispatch + on-GPU decode + finalize/steady overlap *within* a process). That is a real but much
smaller prize, and it overlaps heavily with what `proj-2026-05-24-0859` Step 5 (off-event-loop dispatch) already
attempts in Python.
- **Recommendation:** Annotate that tree row: "native-under-MPS = **tail-only**, no density gain; re-run 0.0 because the
  value is now much smaller and may dip below the worth-it threshold — and check it isn't already captured by the Python
  off-thread-dispatch lever." This keeps the escape honest rather than a soft landing that hides a likely STOP.

## [MINOR] 2. Capacity-number consistency (fixed this pass — verify it reads cleanly)
The §1 table now reads "~20/box today; ~28/box after the Python plan; 40–48 aspirational, delta to beat is vs ~28."
Confirm risk #1's "single-process 40–48/box may be false" is consistent (it is — 40–48 is explicitly aspirational now).
No other stale 20-vs-28-vs-64 numbers should remain outside the Review log (which is historical and correct to leave).

## [MINOR] 3. No executive TL;DR — this plan is long and the decision logic is the point
A reader (or the person deciding whether to fund Wave 2) must reconstruct THE BET + the decision tree + the worth-it
gate from three separate sections. For a plan whose entire value is "should we even do this, and if so on what
evidence," a 5-line TL;DR at the very top (the bet, the three early-exits, the two budgets, "p50 can't move") would make
it actionable at a glance.
- **Recommendation:** Add a short TL;DR block under the title.

## [MINOR] 4. Ordering sanity (no circular dependency — confirmed)
- 0.10 (runtime contract / metrics) is Wave 2 and "pre-1"; Phase 1.2 consumes its metrics. Since Phase 1 starts only
  after the worth-it gate (post-Wave-1) and 0.10 is also post-Wave-1, there's no cycle. OK.
- 4.4 shadow-traffic is labeled "strongest T1 check" but lives in Phase 4 — Phase 1–3 T1 uses **offline golden
  fixtures**, so T1 does not *depend* on 4.4 (it's a supplementary live check). Confirm the wording doesn't imply Phase
  1 waits for 4.4. Consider noting "4.4 supplements, does not gate, the Phase-1 fixture-based T1."

## Verdict
**YELLOW → GREEN for Wave 1 only.** The plan is now sound, honest, and unusually self-aware: it correctly identifies
that p50 cannot move, that the real prize is tail+density, that the bottleneck is the asyncio/lock/decode architecture
(not a mythical GIL-launch ceiling), that the decode is `greedy_batch` label-looping (the hard, un-exportable,
byte-exact-risky part), and that **B4 (free-threaded Python) may make the whole C++ rewrite unnecessary**. The decision
tree + value-based worth-it gate + the explicitly-stated triple-conjunction BET make this fundable as a *staged* bet:
**green-light Wave 1 (the cheap kill-shots: finish the Python plan → 0.0/0.1/0.3/0.5 + the paper audits 0.9/0.11/0.7),
which can only cost a few engineer-weeks and will most likely STOP the project honestly.** Do NOT green-light Wave 2
(the ~12–20 eng-wk byte-exact ports) until Wave 1 clears all three conjuncts. **The ONE thing still most likely to be
wrong:** that a residual gap worth ~40–60 eng-weeks even exists after the Python plan lands — i.e., 0.0 is the most
probable exit, and that is the correct, intended outcome.

## Top items to fix (converged — 3 minor)
1. Annotate the "native-under-MPS" decision-tree row as **tail-only / re-run 0.0** (likely a STOP, not a soft landing).
2. Add a 5-line **TL;DR** under the title (the bet, three early-exits, two budgets, p50-can't-move).
3. Clarify 4.4 shadow-traffic **supplements** (does not gate) the Phase-1 fixture-based T1.
