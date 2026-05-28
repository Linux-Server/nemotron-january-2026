<task>
Adversarial paired review of the Step B2 DESIGN (NOT a build — design pass per the plan's
"PAIRED REVIEW (the topology is a design fork)"). The design takes a stance (central dispatcher) with
rationale; your job is to attack the stance, the integration model, the policy values, the T1 coverage, and
the A1-A8 fold-in. Write a fold-ready adversarial review to `proj-2026-05-24-from-scratch-runtime/reviews/
codex-B2-design-review.md`. The design + your review + a fold doc feed the BUILD task that follows.
</task>

<context>
**The design under review:** `proj-2026-05-24-from-scratch-runtime/reviews/B2-design.md` (read it in full). It
proposes a central steady-batching dispatcher (a dedicated thread that owns a shared ready-queue, gathers
ready chunks under a window+short-circuit policy, runs the batched B=1/2/4 forward via the B1 primitive on a
dedicated CUDA stream, scatters results back to per-stream futures, while decode/finalize ownership stays
per-stream). The alternative considered + rejected: borrow-and-batch on the arriving worker.

**Context that already exists** (do not re-derive — read briefly to ground):
- `PHASE2-PLAN.md` Step B2 (the carry-over audit follow-ups A1-A8 from B1).
- `reviews/B1-paired-verdict.md` (the audit follow-ups origin + the B1 design discipline that B2 inherits).
- `runtime/cpp/steady_batch_primitive.h` (the B1 primitive that B2 wires through).
- `runtime/cpp/density_main.cpp` lines 1700-2260 (the b1-t1 gate as the model for the b2-t1 gate B2 will add)
  + the worker loop integration point (where `run_steady_encoder_stream` is currently called per-worker —
  this is the swap point).
- `reviews/profiling-paired-verdict.md` + `reviews/steady-batch0-RESULT.md` (the BW-bound binding + the
  measured OPPORTUNITY fill 2.7-4.4 at 8-12ms — sets the window policy values).
- `reviews/steady-batch0-RESULT.md` (the SPEEDUP per-row B=2 0.62× / B=4 0.38× — the per-row gain to compound
  via batching).

**The user's SLO + correctness contract** (binding):
- Server-side `ttfs_p95 ≤ 175ms / p99 ≤ 250ms`; keep-up `lag_p95 < 500ms`.
- **0 token divergences** end-to-end through decode+finalize (the SLO signal).
- Events counted-not-gated per DENSITY_GOLD_EVENTS_TOLERANT (project policy, prior bar 5/1000 per pass).
- Byte-exact when `NEMOTRON_DENSITY_BATCH_STEADY=0` (the flag-OFF path is unchanged).
- Added batch-wait p95 ≤ window (10ms) absorbed by keep-up slack; does NOT touch finalize ttfs.

ASK / structure the review:
1. **Topology fork**: is central dispatcher the right choice? Steelman borrow-and-batch and Brittle-test
   central. Specifically: (a) does the central dispatcher add latency that borrow doesn't (the lone-arrival
   case)?, (b) does the dedicated thread become a CPU-single-thread bottleneck at high N (e.g., N=64)?,
   (c) does borrow have a subtle ownership/race bug central avoids?, (d) which is easier to instrument and
   profile?
2. **Policy values**: window=10ms, lone_timeout=1ms, B_max=4 — are these well-chosen given the OPPORTUNITY
   data (mean B 2.7-4.4 at 8-12ms, N=36-56)? Should the window be CONFIGURABLE in tighter increments?
   Should the lone_timeout be 0 (no wait if alone) instead of 1ms?
3. **Integration model**: the `enqueue + future.get() + cross-stream-sync + consume` pattern — does it
   correctly replace the existing `run_steady_encoder_stream` call site without breaking the per-stream
   decode/finalize ownership? Are there other call sites to `run_steady_encoder_stream` (e.g., enc_first,
   finalize buckets) that must NOT route through the scheduler? Verify.
4. **Bucket selection + K > B_max**: the design says K>B_max takes B_max items per batch, leaves the rest
   for the next dispatcher iteration (natural backpressure). Is there a head-of-line blocking risk for the
   queued-rest? At high N (N=64, B_max=4 → 16 batches/cycle), does the queue drain order matter?
5. **The new `b2-t1` gate** (all-chunks-batched per-stream): is the comparison structure right?
   single-stream batched vs single-stream B=1, AND multi-stream concurrent batched vs per-stream B=1?
   Does the multi-stream case need to drive enough concurrency that the dispatcher actually forms batches
   (otherwise it short-circuits to B=1 and tests nothing new)?
6. **A1-A8 fold-in**: is each addressed in the right way / right step?
   - A1 (NEW vs PRODUCTION B=1 parity): the recommendation is to USE NEW B=1 in the flag-OFF path too,
     making the new B=1 the single source of truth. Reasonable? Risk?
   - A4 (reentrancy): preload-discipline-only, no mutex. Is the contract "preload_all before any enqueue"
     enforceable in the construction order? What if the dispatcher thread tries to lazy-load later (e.g.,
     a B-bucket that wasn't preloaded)?
   - A5 (pre-allocated scratch + index_copy_): correctness considerations vs torch::cat (in-place index_copy
     into a persistent tensor — does this break gradient/autograd? Not relevant in inference mode but worth
     flagging).
   - A6 (tolerant-mode wrapper for the b1/b2 strict exit): right placement (wrapper, not in the gate code)?
   - A7 (steady-batch MANIFEST): emit at B2 or defer to B3? The design says B2.
7. **Risk register**: any risks missing? The five listed are dispatcher CPU, compound drift, HOL blocking,
   window-timer jitter, OOM. Anything else (e.g., interaction with the existing admission control /
   stale-gen suppression that Step 2/3 will add; interaction with `finalize_num_runners=2` constraints
   on shared resources)?
8. **5090 knee re-measure plan**: is the B_max ∈ {1, 2, 4} sweep design adequate? Should it also sweep
   `BATCH_WINDOW_MS` to validate 10ms is the right setpoint?
9. **Build sketch**: is the file layout (`batched_steady_scheduler.h`/`.cpp` + `density_main.cpp` mods + new
   `b2-t1` gate) sensible? Any naming / organization concerns?
10. **Net verdict**: GO-with-design / GO-with-changes-to-design / HOLD-and-pick-borrow-instead /
    PIVOT-the-mechanism. If GO-with-changes, list the changes concisely.

Write your review to `proj-2026-05-24-from-scratch-runtime/reviews/codex-B2-design-review.md`. Adversarial,
specific, attacks rationale where weak.
</context>

<verification_loop>
This is a design-doc review, NO BUILD, NO RUN. Read the files cited, think hard about the trade-offs, write
the review. Bounded — no need to spend hours; the design isn't huge.
</verification_loop>

<action_safety>
Write only the review doc. Do not modify the design, the primitive, density_main, or any plan files. Fold
decisions go through a separate fold doc the user will write after both reviews land.
</action_safety>

<compact_output_contract>
Report path of the review doc + a one-paragraph verdict summary (GO / GO-with-changes / HOLD / PIVOT) + the
single highest-priority change-or-attack from your review.
</compact_output_contract>
