# Step B2 design — batching scheduler + density integration

**Status:** draft for paired adversarial review (Codex + Opus). The plan flagged the topology as a design fork
requiring paired review BEFORE build. This doc takes a stance, with rationale; the paired review attacks the
stance + the integration.

## Goal

Wire the B1 batched-steady primitive (`BatchedSteadyLoaderSet`) into the density runtime via a cross-stream
**scheduler** that lifts the 5090 knee above the current single-process compute-bound ceiling of N=40 (Step 1a,
commit 99fbba3), validating the projected ~47-64 L40S knee mechanism end-to-end before B3's L40S sweep.
Byte-exact correctness when batching is OFF (`NEMOTRON_DENSITY_BATCH_STEADY=0` → unchanged Step-1a behavior),
PASS-by-policy when ON (the per-row B1 T1 carries over plus a new all-chunks-batched T1 — A2 from the B1 audit).

## SLO + correctness contract (must hold both)

- **Server-side ttfs**: p95 ≤ 175ms, p99 ≤ 250ms (Definitions, plan v3).
- **Keep-up lag**: p95 < 500ms.
- **Tokens**: 0 divergences vs the production B=1 reference, end-to-end through decode + finalize, on the
  *new* all-chunks-batched per-stream T1 (A2 — B1 only covered single-chunk insertion).
- **Events**: counted-not-gated per project policy (DENSITY_GOLD_EVENTS_TOLERANT); reported as raw count.
- **Added batch-wait p95 ≤ window (8-12ms)** — absorbed by keep-up slack; does NOT touch the finalize-driven
  ttfs (steady-batch is on the steady path, NOT finalize).

## The topology fork — RECOMMENDATION: **central dispatcher**

Two viable shapes for the cross-stream batching mechanism:

### Option A — central steady-batching dispatcher (RECOMMENDED)
A dedicated thread (NOT a per-stream worker) owns the steady-batch dispatch. Workers, when their steady chunk
is ready, enqueue `(BatchedSteadyInput, promise)` to a shared queue and wait on the promise's future. The
dispatcher loop:
```
loop:
  gather = []; t0 = now()
  while len(gather) < B_max:
    if len(gather) == 0:
      wait_until_ready(timeout=W_warm or infinity)   # the lone-arrival timeout
    else:
      poll_or_wait(timeout=W - (now()-t0))           # the batch-fill timeout
    if no_more_ready: break
    gather.append(pop_ready())
  K = len(gather)
  outputs = batched_loader.run(gather.inputs, dispatcher_stream)
  for i, item in enumerate(gather):
    item.promise.set_value(outputs[i])
```
Workers then sync the dispatcher's stream into their own stream (`cudaStreamWaitEvent`) and consume the
returned per-row tensors → continue per-stream decode → finalize unchanged.

### Option B — borrow-and-batch on the arriving worker
No dedicated thread. When worker W's steady chunk is ready, W checks the shared ready-queue:
- If the queue has K-1 other ready items AND the global wait-window hasn't elapsed → W *borrows* them, runs
  the batched forward on W's stream, delivers each row's result back to its owner's future.
- If empty after W's own wait W_self → W runs alone (B=1, short-circuit) on its own stream.

### Why central wins (the rationale to attack)

1. **Single owner of the batching policy**: the window, short-circuit, bucket-select, B_max cap, future
   backpressure all live in one place — easy to reason about, profile, and instrument with admission/priority
   later (Step 2 of the plan). Borrow distributes the policy across N workers, multiplying the surface area
   where a subtle race or starvation bug can hide.
2. **Single batching stream**: the dispatcher uses one dedicated CUDA stream for the batched forward.
   Workers `cudaStreamWaitEvent` on that one stream before consuming. In borrow, the batched forward runs on
   *whichever worker happened to borrow* — measurement, ncu attribution, and stream-overlap analysis become
   per-call dependent on who got the dispatch. Central is much cleaner to profile.
3. **No "worker running another worker's work"** semantic. Borrow makes worker W's CUDA stream do worker X's
   forward; failure modes (OOM, kernel error) attribution becomes confusing. Central keeps each worker's
   stream owning that worker's work.
4. **Symmetric N-1 cross-stream sync** vs borrow's N-1 + ownership ambiguity. Both designs pay one
   cross-stream sync per worker per batch; central pays it for a known dispatcher stream which is uniform.
5. **Step 2 (admission/priority) trivially attaches** to a central dispatcher (it already serializes through
   one queue + thread; priority finalize-lane + close-shed naturally compose). Borrow makes this a multi-thread
   coordination problem.

### Why central might lose (the attack vector for the review)
- **Single-thread bottleneck risk**: at very high N (e.g., N=64), the dispatcher might become CPU-bound (its
  thread is single). Measurable: at the densest run, watch the dispatcher CPU%; if pegged, the dispatch loop
  is the lever. *Mitigation*: dispatcher work is lightweight (pop K, build tensor list, call run, set N
  futures, repeat) — no real heavy compute in the dispatcher; GIL doesn't exist (C++); profile to confirm.
- **Lone-arrival latency at low load**: a single worker arriving alone waits W_self before running B=1, vs
  borrow which runs immediately. *Mitigation*: short-circuit policy — if K==1 and no other in queue after a
  short poll (W_warm = 0-2ms), the dispatcher runs B=1 immediately. Effectively zero added latency at low
  load.
- **Dedicated thread overhead**: +1 thread per process. Trivial cost (the 5090 box has 32 vCPUs, currently 6
  used at knee). Worth it.

If the review converges on borrow, the integration is ~similar (same primitive, same futures, just no
dispatcher thread). Either way, the per-stream decode/finalize ownership is unchanged.

## Scheduler policy details

### Window + short-circuit
- `NEMOTRON_DENSITY_BATCH_WINDOW_MS` — default **10ms** (midpoint of the 8-12ms band; absorbed by keep-up
  slack per the OPPORTUNITY gate measurement).
- `NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS` — default **1ms** (lone-arrival short-circuit: if only K=1 ready
  after this, run B=1 immediately; preserves single-stream/best-case latency).
- `NEMOTRON_DENSITY_BATCH_MAX` — default **4** (largest bucket); allow override to 2 or 1 for diagnostic
  comparison sweeps (5090 knee re-measure).

### Bucket selection
- K → `bucket_for_k(min(K, B_max))` → {1, 2, 4}. K=3 padded to B=4 (1 pad), K=2 → B=2, K=1 → B=1.
- K > B_max: the dispatcher pops only B_max items per batch (leaves the rest in the queue for the next
  iteration — natural backpressure).

### Backpressure / overflow
- The ready-queue is bounded (e.g., 4× N_workers). Workers' enqueue blocks if full (back-pressures the worker
  loop — same effect as the existing per-worker keep-up slack absorbing the wait).
- The dispatcher processes FIFO. Future enhancement (Step 2 hook): priority lane for finalize-adjacent
  streams.

### Failure handling
- If `batched_loader.run()` throws, the dispatcher catches, sets each pending promise's `set_exception`, and
  continues. Workers raise the exception in their loop → existing per-worker error path (close/shed) handles
  it.

## Per-stream integration

The existing `density_main.cpp` worker loop calls `run_steady_chunk_density(...)` which internally calls
`run_steady_encoder_stream(enc_steady, chunk, session, ctx.stream, ...)`. **Replace** that B=1 call (when the
flag is ON) with:
```
auto fut = scheduler.enqueue({chunk, session.clc, session.clt, session.clcl, label});
auto out = fut.get();   // blocks until dispatcher fills the batch and runs it
sync_stream_into(out.event, ctx.stream);   // cross-stream wait
// consume out.tensors as if they came from B=1 run_steady_encoder_stream
```
The downstream `apply_encoder_outputs_density` consumes the same 5-tensor list shape (the B1 unpack already
shaped per-row). `decode_range_density` / `run_finalize_density` are unchanged.

**Flag-OFF byte-exact**: when `NEMOTRON_DENSITY_BATCH_STEADY=0`, the worker calls the existing B=1
`run_steady_encoder_stream` directly (no scheduler involvement). Untouched path.

## T1 strategy (folds A2 from the B1 audit)

Two T1 modes both required to pass before commit:
1. **`b1-t1` (existing)** — single-chunk-insert primitive correctness; carried forward unchanged.
2. **`b2-t1` (new)** — all-chunks-batched per-stream end-to-end. For each utt:
   - Run the full session via the batched scheduler (`NEMOTRON_DENSITY_BATCH_STEADY=1`), capture
     tokens + events.
   - Run the same utt via pure B=1 (`NEMOTRON_DENSITY_BATCH_STEADY=0`), capture tokens + events.
   - Compare: **0 token divergences** required; events counted-not-gated per policy; reference is the
     production B=1 path (the existing `build_serial_reference` is the gold).
   - Concurrent multi-stream variant: run N copies of the corpus through the scheduler (so the dispatcher
     actually has cross-stream batches to form), compare to per-stream B=1. This is the production-shape T1.

`b2-t1` is the structural completion of B1's coverage gap.

## 5090 knee re-measure plan

Pre-/post-B2 comparison (B_max sweep):
- N=40 / N=48 / N=56 / N=64 each at B_max ∈ {1 (control = B=1 path, sanity), 2, 4}.
- Per-N: 8 sessions/worker, fresh-process-per-N, stagger-robust (10s per-worker stagger from Step 1a).
- Report: SLO-robust knee per B_max; the lift vs B_max=1 control; the added batch-wait p95; the
  dispatcher CPU%; the per-stream cross-stream-sync overhead.

**Sanity goal**: knee lifts from N=40 toward 47-64 on the 5090 (the L40S-projected band). If 5090 lifts the
mechanism is sound and B3's L40S sweep will confirm the absolute. If 5090 does NOT lift (or regresses), the
scheduler has a flaw — escalate to the paired review.

## A1-A8 audit follow-ups — explicit fold

- **A1** — verify `enc_steady_aoti.pt2` (production) and `enc_steady_aoti_b1.pt2` (new) are bit-identical
  via `sha256sum` + a tensor-diff fixture. If they differ (likely due to compile-time variance), the cleanest
  fix is to **use NEW B=1 as the production loader in the scheduler-OFF path too** (one source of truth for
  the B=1 forward). Decision point in the review.
- **A2** — `b2-t1` new gate covers all-chunks-batched per-stream (the structural gap above).
- **A3** — debug-flagged top-2 joint-score margin probe: add `NEMOTRON_DENSITY_DEBUG_MARGIN=1` that, on each
  greedy argmax decision, logs the top-1 score, top-2 score, gap, the chosen token, and the chunk index.
  Off in production; available for diagnostic runs.
- **A4** — **reentrancy**: B2 must call `batched_loader.preload_all()` at scheduler construction, BEFORE
  any worker enqueues. The dispatcher then only ever READS the loader map (no concurrent get()). Documented
  contract; no mutex needed.
- **A5** — pre-allocated pack/unpack scratch: the dispatcher owns persistent `B_max`-sized scratch tensors
  (one per bucket). `pack_inputs` uses `index_copy_` into the scratch instead of `torch::cat` allocating new
  tensors. Implement as part of B2 (avoid post-hoc retrofit).
- **A6** — tolerant-mode wrapper: rename the existing `b1-t1` gate's strict exit to a wrapper that consults
  `DENSITY_GOLD_EVENTS_TOLERANT` for the exit code. Apply same to `b2-t1`. Documentation update.
- **A7** — steady-batch manifest: emit a `steady_b_artifacts/MANIFEST.json` with package SHA256, EP SHA, the
  inductor configs, torch version, CUDA cap, byte sizes — and CI-style fail-closed regen if missing. Mirror
  finalize's discipline.
- **A8** — header self-containment: move `fs`, AOTI type aliases, `load_shared_constants`, `file_exists`,
  `directory_exists` into a shared header (`runtime/cpp/runtime_io.h`) and include from both
  `density_main.cpp` and `steady_batch_primitive.h`. Mechanical refactor.

## Risk register (for the review to interrogate)

1. **The 5090 knee might not lift** at B_max=4 if the binding becomes the dispatcher's single thread (CPU
   single-thread bound) or cross-stream sync overhead. Mitigation: profile dispatcher CPU + the sync wait;
   if pegged, reduce dispatcher work or split dispatcher across 2 threads (round-robin).
2. **`b2-t1` might surface compound drift** that B1 didn't (multi-chunk batching could accumulate drift past
   the near-tie threshold). Mitigation: the A3 margin probe surfaces near-tie cases; the project policy
   tolerates interim-event drift; only TOKEN flips would block.
3. **Backpressure on the dispatcher queue at high N** could starve some streams (head-of-line blocking).
   Mitigation: monitor per-stream enqueue→dequeue latency; if HOL forms, Step 2's priority lane addresses
   it (B2 lays the groundwork, Step 2 adds the policy).
4. **Window timer accuracy**: 8-12ms windows on Linux without RT scheduling can jitter. Mitigation: use a
   condvar with timeout (not a sleep); measure dispatcher wakeup jitter in telemetry.
5. **OOM at high N**: B-bucket activations are larger; verify peak memory at N=64 × B_max=4 stays within the
   5090's 32GB headroom.

## Build sketch (for Codex's delegated build)

1. New file `runtime/cpp/batched_steady_scheduler.h` (or `.cpp` + `.h`) — implements the central dispatcher,
   the ready-queue, futures, scratch tensors, env-flag parsing.
2. Modify `runtime/cpp/density_main.cpp` — when `NEMOTRON_DENSITY_BATCH_STEADY=1`, the worker calls
   `scheduler.enqueue(...)` + `fut.get()` + cross-stream-sync instead of `run_steady_encoder_stream`.
   The flag-OFF path is unchanged. Add `--mode b2-t1`.
3. New `b2-t1` gate (run scheduler-batched concurrently and compare to B=1 reference).
4. A1 hash-and-tensor-parity check + the steady-batch MANIFEST emit (A7).
5. The 5090 knee re-measure script (small shell wrapper around `--mode density-sweep
   --batch-scheduler-mode central --b-max {1,2,4}` for the N sweep).

## Verdict

**Recommend central dispatcher**. The paired review should attack (a) the topology choice, (b) the
short-circuit / window policy values, (c) the integration with the existing per-worker stream model, (d)
whether the `b2-t1` design covers the right cases, (e) the A1-A8 fold-in, (f) the risk register
completeness. The build delegated to Codex AFTER the review converges.
