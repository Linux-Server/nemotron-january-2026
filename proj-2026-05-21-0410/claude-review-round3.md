# Claude direct review — round 3 (PLAN.md v3, final)

v3 has folded rounds 1-2 comprehensively (cache axis dim1, greedy_batch+Probe C, compile B=1-only,
exact ready predicate, drop_extra try/finally, encode/decode split + fallback gate, concrete defaults,
memory gates, fork-lane ownership, git-dirty baseline, quantified thresholds). I find **no remaining
correctness blockers**. A few final notes:

## Minor (non-blocking)
1. **DAG nuance:** Step 5 (batch primitives) and Step 6 (5a scheduler infra, B=1) are INDEPENDENT — 5a
   runs B=1 and doesn't need the stack/unstack primitives; both feed Step 7 (5b). The current order is
   fine; they could be done in either order. Real DAG: 0→{1,2,3}; 1→4; {2→5, 0→6}; {5,6,3}→7→8→9→10→11.
2. **Step 6 is the largest single step** (scheduler task + deduped ready set + generation tokens +
   single model-call lane + fork-lane ownership + bounded queues + cancel/close/reset). It may need 2
   Codex sub-delegations (6a: queue/lane/state-ownership refactor at B=1 byte-exact; 6b: fork/finalize
   migration onto the lane). Flag for the implementer to split if one pass is too big.
3. **Confirm the WS protocol/handlers are unchanged** by the scheduler (it's internal): the
   `concurrency_test.py` harness + Pipecat client keep working without changes. State this in Step 6 so
   the implementer doesn't touch the handshake.
4. **Probe ordering for the fallback decision:** Probe C's encode/decode split + the ≥1.5× fallback gate
   should be evaluated BEFORE committing to Steps 6-8 — if both greedy_batch is NO-GO AND the encoder-only
   fallback is <1.5×, the plan says STOP. Make that an explicit decision point after Step 3 (don't sink
   the scheduler effort first).

## Verdict
**Implementation-ready.** The two probes (B state-correctness, C decoder+split) are the load-bearing
gates; if both pass, the scheduler work is justified and the correctness rules are now precise enough to
implement byte-exactly. Recommend: after Probe C, make an explicit GO/STOP decision on the scheduler
based on the measured ceiling, then proceed Step 4 (compile, if Probe A GO) in parallel with Steps 5-8.
