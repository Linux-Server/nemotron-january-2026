# Step B2 design — Opus second-pass adversarial review

Re-reading `reviews/B2-design.md` (which I authored as Opus 1st-pass) with a deliberately skeptical eye —
the items I rationalized away or under-specified. Folds with `codex-B2-design-review.md` after Codex returns.

## Verdict (preview)

**GO-with-changes** to the design before delegating the build. The central-dispatcher *topology* recommendation
holds, but five specific design details need revision; one was a soft spot I genuinely missed (HOL burst
arrival math), two were under-specified (flag-OFF dispatcher lifecycle + dispatcher fault tolerance), and two
were optimistic phrasings I should sharpen (the lone-arrival timeout policy + A1's flag-OFF byte-exact risk).

## Real weaknesses I want to surface (ordered by likely impact)

### 1. The HOL burst-arrival math is concerning at high N — needs measurement before B3
The risk register said "HOL blocking" abstractly. Concretely: at steady-state N=64 with continuous arrivals
the queue oscillates and the tail is ~1 batch wait ≈ 5-6ms (fine). **But if arrivals burst-synchronize** (a
real possibility despite the existing 10s per-worker stagger — VAD edges, network jitter, or chunk-boundary
alignment can re-synchronize streams), the queue can reach depth ~N briefly. With B_max=4 and per-batch
time ~5-6ms, drain time = ⌈N/B_max⌉ × per_batch. At N=64 this is **16 batches × ~6ms = ~96ms** — the
**last enqueued item waits ~96ms for its turn**. That's a 5-10× SLO budget excursion vs the 10ms window
setpoint.

**Action:** B2 must telemetry per-stream `enqueue_to_dispatch_wait` p50/p95/p99 *separately* from
`batch_wait_to_fill`. The 5090 knee re-measure should include a deliberate burst-injection test
(synchronized N-stream start without stagger) to surface the worst-case HOL tail. If it exceeds the SLO,
B2's fix is a **priority lane or short-batch preemption** (already planned in Step 2 admission, but B2 may
need to lay the groundwork explicitly).

### 2. The lone_timeout=1ms recommendation is wrong for the truly-alone case
I wrote "lone_timeout=1ms → effectively zero added latency." Untrue. At single-stream load, *every* steady
chunk waits 1ms before firing → ~100 chunks per session = 100ms cumulative latency on the interim path.
**Fix:** when the dispatcher pops K=1 AND the ready-queue is otherwise empty, run B=1 *immediately*
(lone_timeout=0). The 1ms wait should only fire when there's a *plausible* second arrival imminent (which
the dispatcher can't know — so the only safe default is 0). This is the borrow-option's strongest argument;
adopt it inside central. Lone_timeout=1ms is the WRONG default; **default 0**.

### 3. The flag-OFF dispatcher lifecycle is under-specified
"When `NEMOTRON_DENSITY_BATCH_STEADY=0`, the worker calls the existing B=1 path directly. Untouched path."
But: **is the dispatcher thread spawned at all?** If yes, it's wasted memory + a sleeping thread (cheap but
non-zero). If no, the code must branch on the flag at startup (the scheduler is constructed conditionally).
**Action:** the scheduler is constructed iff `NEMOTRON_DENSITY_BATCH_STEADY=1`. The dispatcher thread only
exists in the ON case. Flag-OFF byte-exactness is preserved by the SAME mode-separation B1 used (the
production density-sweep path is unchanged regardless of the flag's value when the scheduler isn't
constructed). Document explicitly + add an assertion that `g_scheduler == nullptr` when the flag is OFF.

### 4. A1 fix as written ("use NEW B=1 in the flag-OFF path too") breaks the byte-exact contract
This was a sloppy A1 recommendation. The flag-OFF path's byte-exact guarantee is against the *currently
shipping* production B=1 (`enc_steady_aoti.pt2`). Swapping in a freshly-compiled `enc_steady_aoti_b1.pt2` —
even if bit-identical at one snapshot — re-introduces compile-time-noise risk on every CI rebuild and
breaks the durable byte-exact contract. **Fix A1 properly:** (a) for the **b1-t1 / b2-t1 GATES** only, use
NEW B=1 (`enc_steady_aoti_b1.pt2`) as the alone reference so the comparison is apples-to-apples; (b) the
production density-sweep path continues to use PRODUCTION B=1 (`enc_steady_aoti.pt2`) unchanged when the
flag is OFF. The clean-attribution fix lives in the gate code, not the production path.

### 5. Dispatcher fault tolerance is missing
A kernel error or exception in `batched_loader.run()` is caught and propagated to the K futures via
`set_exception` — fine. But: **if the dispatcher thread itself dies** (crash, OOM kill of the loader thread,
exception escaping the dispatcher loop), **all in-flight workers block forever on their futures**, the
process appears alive but is hung. **Action:** (a) wrap the dispatcher loop in an outer try-catch that, on
any escape, fault-injects an exception into ALL pending and future-enqueued futures (so workers fail-fast
into the existing per-worker error path), then exits the process with a clear error; (b) add a finite
timeout on the worker's `future.get()` (e.g., `wait_for(W + 200ms)` where 200ms >> any plausible single-
batch time) and treat timeout as a fault. The watchdog timeout must be much larger than the SLO so it
doesn't trigger in normal operation but bounded so it does trigger if the dispatcher dies.

## Less-impactful items (worth mentioning, not strictly blocking)

### 6. b2-t1 multi-stream specification is too thin
The design says "Concurrent multi-stream variant: run N copies of the corpus through the scheduler, compare
to per-stream B=1." Not enough: *which* N (16? 32? 40 (the current 5090 knee)?), *how many corpus passes*,
*does it spawn fresh workers or reuse the density-sweep harness?* The cleanest answer: b2-t1 is
density-sweep mode with `--mode b2-t1` that adds a per-stream comparison hook — reuse the existing harness,
choose N to actually exercise the dispatcher (N≥4 minimum), require K_avg≥2 in the dispatcher telemetry
(else the test ran B=1 paths and is not measuring batching). Specify in the build task.

### 7. A4 (reentrancy) — "preload-discipline-only" assumes the dispatcher never lazy-loads
Correct in the steady-state design (B is one of {1,2,4}, all preloaded at construction). But: if a B_max
override later allows B=8 or B=3, the lazy-load path activates. Add an assertion in `get()` that the bucket
is in the preloaded set, else throw — fail-fast instead of silently lazy-loading + potentially racing.

### 8. The `index_copy_` perf optimization (A5) has an in-place semantics consideration
Persistent scratch tensor + `index_copy_(0, indices, src)` overwrites the scratch's row data per call. If
the previous batched run's outputs were aliased into the scratch (they shouldn't be — the unpack
`.contiguous()` makes per-row copies), this would be a use-after-free. The B1 unpack already does
`.contiguous()` so this is safe; just document the dependency.

### 9. Window-timer jitter under load (risk #4) needs a concrete instrumentation plan
The design lists this risk but doesn't say how to detect it. Add: telemetry `window_wakeup_jitter_us` =
the actual wakeup time minus the condvar timeout time, per dispatch cycle.

## What HOLDS in the design (worth saying)

- **Central dispatcher (topology choice)** — Codex may attack this; my own re-attack didn't find a fatal
  weakness. The single-owner / single-stream / single-policy benefits are real, and the dispatcher CPU
  bottleneck risk is measurable + mitigable. Stand by it.
- **B_max=4 default** — matches the measured OPPORTUNITY fill (mean B 2.7-4.4 at 8-12ms).
- **A1-A8 fold-in placement** — mostly right (with #4 above fixing A1's flaw).
- **The b2-t1 gate concept** — necessary structural completion of B1's coverage gap; just needs
  specification tightening (#6).
- **Risk register completeness** — 5 risks named; my self-attack added "dispatcher fault tolerance" as a
  sixth. Risk #1 (HOL burst) needs sharper measurement plan.

## Concrete deltas to apply to the design BEFORE delegating the build

1. **lone_timeout default 0**, not 1ms (fix #2).
2. **Conditional dispatcher construction** based on the flag at startup; assert `g_scheduler == nullptr`
   when OFF (fix #3).
3. **A1 split**: NEW B=1 only in the b1/b2-t1 gates; production OFF path UNCHANGED (fix #4).
4. **Dispatcher fault-tolerance**: outer try-catch + worker future timeout + process-fail on dispatcher
   death (fix #5).
5. **HOL burst telemetry + injection test** in the 5090 knee re-measure (fix #1).
6. **Specify b2-t1 multi-stream**: density-sweep + per-stream compare hook, N≥4, K_avg≥2 telemetry-required
   (fix #6).
7. **A4 lazy-load fail-fast**: throw if `get()` is called for a non-preloaded bucket (fix #7).
8. **Document index_copy_ aliasing safety** (fix #8 — documentation only).
9. **Window-jitter telemetry** (fix #9).

## Net

Stand by the central dispatcher recommendation. Apply the 9 deltas above to the design (update
`reviews/B2-design.md` post-fold). Then delegate the build to Codex with the revised design as the spec. The
biggest substantive fix is #1 (HOL burst) — the others are sharpenings.
