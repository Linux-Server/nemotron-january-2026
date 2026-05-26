# Phase-2 review — Round 3 charge (downstream Steps 2–5 + does the corrected plan measure G1/G2?)

Rounds 1–2 are folded (`phase2-round1-FOLDED.md`, `phase2-round2-FOLDED.md`) — read both. Rounds 1–2 hammered
Step 1; **Steps 2–5 have had only light review.** Round 3 = line-by-line on the downstream steps and the
infrastructure they depend on, foregrounding **G1 density** and **G2 P50↔P95 tail**.

## Context the native runtime changes
The Python deployment is **K=3 processes behind an external LB** (`ec2-bench/local_lb.py`, leastconn + maxconn).
The native runtime is **ONE process, multi-thread**. So the LB + per-process backlog admission collapses into a
**single-process internal admission**. Production shed behavior that the plan must stay faithful to (from the
project memory): **admission = backlog-COUNT cap ≈8–12 (ready-AGE signal was proven NOT to work)**, a **priority
finalize lane** (shipped, byte-exact, default-on), **sync compression**, and the global `inference_lock` as the
serializer being replaced.

## Files to read line-by-line
- `PHASE2-PLAN.md` Steps 2–5 (`:33-44`) + the Progress table.
- `ec2-bench/local_lb.py` — the LB stand-in (leastconn/maxconn, "all backends at maxconn → shed"). What does the
  single-process native runtime replace this with, and does that change the apples-to-apples comparison?
- `ec2-bench/ec2_loadgen.py` — the load generator (note `STREAM_JITTER_MS` WAN-mimic, bursty arrival). This is the
  Step-4 apples-to-apples driver: what must be held identical between Python and native runs?
- `stt-benchmark/src/stt_benchmark/evaluation/semantic_wer.py` — the SAME semantic-WER tool Step 4 must use.
- Any production admission/priority-lane/shed references you can find (grep `backlog|admission|shed|priority|
  inference_lock|scheduler_exclusive` outside the proj dir) so "faithful to the Python shed behavior" is checkable.

## What to review (Steps 2–5)
1. **Step 2 (scheduler + admission).** Is the design faithful to the Python shed behavior when the architecture
   flips from LB+K-procs to one process + threads? Specifically: (a) does the plan specify a **single global
   active-session/inflight admission cap** (backlog-COUNT, not ready-age) matching production? (b) how does the
   **priority finalize lane** map onto the Round-2 `num_runners=N` runner pool — does a finalize burst starve
   steady, or is there a dedicated finalize runner/lane? (c) is Step 2 correctly **blocked on Step 1's telemetry
   schema** (Codex R1-M5)? (d) does the plan avoid re-litigating ready-age (already proven NOT to work)?
2. **Step 3 (multi-session runtime + real WS server).** The native WS server has its OWN latency/tail
   contribution (accept loop, framing, write path). Could a naive WS server ADD tail that confounds the Step-4
   density/G2 number — the very thing the runtime is trying to fix? Does the plan require characterizing the WS
   server's own overhead (e.g., loopback echo) so Step 4 separates runtime-tail from WS-tail? Also: Step 3 is
   where stale-generation suppression (deferred from Phase 1, Codex R1-M6) must land — is it scoped there?
3. **Step 4 (apples-to-apples density).** What must be IDENTICAL for the comparison to be fair: SLO definition,
   `STREAM_JITTER_MS`/WAN model, semantic-WER tool/version, hardware, AND a **freshly re-measured** Python
   baseline (memory says the baseline is not frozen; don't compare to the stale 16–20). Is "streams/box" the
   right unit across 1-proc-native vs K-proc-Python? Is there a risk Step 1b's (placeholder-dispatch) density
   over-states Step 4's (real scheduler + WS) realized density — i.e., should Step 1b be labeled a CEILING and
   Step 4 the realized number?
4. **Step 5 (per-target sweep).** The hypothesis (5090/L40S launch-bound → native lifts; L4/Spark BW-bound → no
   lift) is **testable from Step 1's resource attribution** (Round 2 made BW a hypothesis to measure with
   counters). So is Step 5 independent, or a CONFIRMATION of the Step-1-predicted per-target behavior? Does the
   plan state Step 5's purpose (confirm the L4 negative + the L40S number + Spark exploration) vs a fleet
   re-decision? Spark aarch64: does the `num_runners` pool / AOTI behavior even hold on aarch64 (libtorch
   maturity risk)?

## Cross-cutting
Does the **corrected** plan (after Rounds 1–2 edits) now actually MEASURE G1 (density, end-to-end at Step 1b +
Step 4) and G2 (the server-side tail, reference at Step 1 → binding at Step 4, WS-confound controlled)? Name any
remaining gap where a GO/STOP could still be wrong, or where G2 in particular falls through the cracks.

Write to `proj-2026-05-24-from-scratch-runtime/reviews/codex-phase2-round3.md` (BLOCKER/MAJOR/MINOR/QUESTIONS,
file:line, recommended edits). Round 3 of 5.
