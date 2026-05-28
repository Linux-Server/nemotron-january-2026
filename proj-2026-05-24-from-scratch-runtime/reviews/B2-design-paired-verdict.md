# Step B2 design — paired adversarial verdict + revised spec (2026-05-27)

**Folds:** `opus-B2-design-review.md` + `codex-B2-design-review.md` (two independent adversarial reviews of
`B2-design.md` v1, written without seeing each other). The two pages here:
**(I)** the verdict and the convergence / disagreement / distinctive findings;
**(II)** the **revised build-binding spec** that supersedes v1 on every changed item — the Codex build task
quotes this section.

---

## I. Verdict + paired analysis

**VERDICT: GO-with-design-changes.** Central-dispatcher topology HELD (both reviewers independently agreed
after attacking it). Build is gated on the 17 design changes below, the most critical of which is **the
input-side CUDA stream race Codex caught and Opus missed**. No PIVOT, no HOLD; the revised spec is buildable.

### Convergence (both reviewers, independently)

1. **Central dispatcher is the right choice.** Borrow-and-batch's strongest case (immediate lone-arrival,
   no extra thread, owner-stream-for-owner-row) is real but the ownership/race/exception/scatter complexity
   makes it worse for a first build. Single owner + single stream + single profiling surface wins.
2. **`lone_timeout=1ms` was wrong; default 0** (Opus: 100ms cumulative per session at single-stream; Codex:
   "0ms is the right diagnostic control" + low-load latency). Keep configurable; both reviewers say sweep
   the value and report B=1 rate vs enqueue→run-start.
3. **A1 ("use NEW B=1 in OFF path") was wrong** — breaks the durable byte-exact contract. Both: hash/tensor/
   token-parity check NEW vs PRODUCTION first; if not bit-identical, OFF path stays on PRODUCTION B=1,
   scheduler-ON owns the full B=1-vs-production T1 risk explicitly.
4. **Dispatcher fault tolerance + shutdown/cancellation** is a real missing piece. Outer try-catch +
   close()/drain + exception broadcast + worker future timeout + process fail-fast on dispatcher death.
5. **HOL burst tail is a real concern** at high N; needs telemetry separation and burst-injection test.
6. **A4 (reentrancy)**: preload + SEAL + fail-closed `get()` (don't rely on a comment that no lazy load will
   happen later).
7. **A5 (preallocated scratch + index_copy_)**: correctness OK under inference-mode + dispatcher-stream
   confinement; don't return views into scratch; record completion event after unpack `.contiguous()`.

### Disagreement / Codex distinctive (Codex caught what Opus missed)

1. **INPUT-SIDE CUDA STREAM RACE** *(Codex's highest-priority finding — my single biggest miss)*. The v1
   design only specified output-side waiting (worker waits for dispatcher event before consuming). But the
   worker prepares `chunk = torch::cat({state.ring, new_mel}, 2)` and the cache clones on the WORKER's
   stream — CUDA work is async. If the dispatcher reads/packs those tensors before the worker stream has
   completed the cat/cache-producing work, the dispatcher can read INCOMPLETE OR UNINITIALIZED data → silent
   correctness bug, would only manifest as token noise under load and look like more reduction-order drift.
   **Fix:** each enqueued item carries a *producer-ready event* recorded on the worker stream after input
   tensor preparation; dispatcher stream `cudaStreamWaitEvent`s on all producer events before
   `index_copy_`/AOTI run. This makes the wait bidirectional. *This single fix is the most important change.*
2. **Control-path contamination via `run_steady_chunk_density`**: that helper is reused by the serial
   reference (`build_serial_reference`), B1 T1 prep, finalize parent prep, warmup, and the density sweep. If
   the scheduler is wired as a hidden global inside it, ALL these paths route through it — including the b1/
   b2-t1 references that MUST be pure B=1. **Fix:** scheduler integration is an explicit parameter/policy
   object, not a global swap. Only the B2 measured path opts in.
3. **K > B_max as a queueing/service-time problem, not just "backpressure":** at N=64 with 16 queued B=4
   forwards × ~7.7ms each = ~123ms drain → last item's enqueue→output wait can far exceed the 10ms gather
   window even if gather itself is correct. Opus noted the burst risk; Codex made it quantitative + said the
   "backpressure" framing is wrong (it's service-time-bound).
4. **b2-t1 false-pass via short-circuit**: a natural multi-stream run might never form B>1 batches if
   lone_timeout fires too eagerly; the test passes vacuously without exercising batching. **Fix:** bucket-
   count assertions (B=1, B=2, K=3→B=4 padded, K=4) + a deterministic forced-concurrency case with a barrier
   + the staggered case + a scheduler-ON `B_max=1` control to isolate scheduler overhead from real batching.
5. **Warmup design decision**: route warmup through the scheduler with telemetry-excluded OR warm outside.
   Otherwise warmup might form the only B>1 batches in `b2-t1` while measured phase short-circuits.
6. **OPPORTUNITY data limitation**: the published mean B values (2.7-4.4) include B>4 collapsing to ≤4 only
   in the published table; the fill_trace must be re-replayed with `B_max=4` to get the actual realistic
   bucket distribution (B=1%, B=2%, B=3→4%, B=4%, queued-rest latency).
7. **Telemetry must split** into 5 distinct buckets (not one "batch wait"): gather wait, service wait, CUDA
   run time, output sync wait, worker blocked time.
8. **Dispatcher-stream utilization profiling**, not just dispatcher CPU. At 400 steady chunks/s × ~7.7ms B=4
   service = ~769ms/s of dispatcher-stream service → the stream itself, not the CPU thread, may be the
   serialized service station.
9. **N=44 in the sweep** (just above current 5090 knee N=40 → expected lift boundary).
10. **Manifest CONSUMPTION fail-closed**, not just emission (otherwise A7 is theater).
11. **Configurable via CLI args**, not env-only (telemetry must record effective policy).
12. **Scheduler API shape**: `enqueue(input, producer_stream, label)` → output with row tensors + bucket +
    completion event; lifecycle methods start/close/destructor-drain/exception-broadcast.

### Opus distinctive (Codex implicitly endorsed by silence)

13. **Flag-OFF dispatcher lifecycle**: scheduler constructed conditionally only when flag is ON; assert
    `g_scheduler == nullptr` when OFF. (Codex's "control-path contamination" is related but is about the
    integration point, not the construction lifecycle.)
14. **Window-wakeup-jitter telemetry**: `wakeup_us - condvar_timeout_us` per dispatch cycle, to surface
    Linux scheduler jitter under load.

### What both reviewers said HOLDS (worth recording)

- The central-dispatcher topology choice itself.
- `B_max=4` as the cap.
- The bucket structure {1, 2, 4}.
- The A1-A8 fold-in placement (with the corrections above).
- The 5090 knee re-measure approach (just needs more sweep dimensions per #9 / #11).

---

## II. Revised build-binding spec (supersedes `B2-design.md` v1 on every conflict)

### II.1 Scheduler API + lifecycle

- **Class**: `BatchedSteadyScheduler`.
- **Construction**: takes a constructed `BatchedSteadyLoaderSet` (already `preload_all()`'d), a CUDA device,
  and a policy struct (`window_ms`, `lone_timeout_ms`, `B_max`, `queue_capacity`).
- **Conditional construction**: only instantiated when `NEMOTRON_DENSITY_BATCH_STEADY=1`; integration sites
  hold `BatchedSteadyScheduler*` (nullable); when null, the production B=1 path is taken unchanged. Assert
  `scheduler == nullptr` when the flag is OFF.
- **API** (replaces the v1 "futures everywhere" hint):
  ```
  struct EnqueueRequest {
    BatchedSteadyInput input;        // chunk + cache_ch + cache_t + cache_ch_len + label
    c10::cuda::CUDAStream producer;  // the worker's stream (the dispatcher waits on its event)
  };
  struct DispatchResult {
    std::vector<at::Tensor> row_tensors;   // enc_out, enc_len, cache_ch, cache_t, cache_ch_len (per-row)
    int bucket; int row;
    cudaEvent_t completion;          // recorded on dispatcher stream after unpack contiguous() copies
  };
  std::future<DispatchResult> enqueue(EnqueueRequest&&);
  void start();   // spawn dispatcher thread
  void close();   // close queue, drain in-flight, broadcast exception, join thread
  ~BatchedSteadyScheduler();  // calls close() if not already
  ```

### II.2 Bidirectional CUDA stream synchronization (THE critical correctness fix)

- **Worker side, BEFORE enqueue:** worker records a producer-ready event on its own stream after
  `chunk = cat(ring, new_mel)` + cache clones. The event is part of the EnqueueRequest.
- **Dispatcher side, BEFORE pack/run:** for each gathered request, `cudaStreamWaitEvent(dispatcher_stream,
  request.producer_event)`. Only then is `index_copy_`/AOTI `run()` issued.
- **Dispatcher side, AFTER unpack (`.contiguous()` per-row copies queued on dispatcher_stream):** record a
  completion event on dispatcher_stream; this event is returned in DispatchResult.
- **Worker side, AFTER `future.get()`:** `cudaStreamWaitEvent(worker_stream, result.completion)` before
  consuming `result.row_tensors`. (Output-side wait — was the only one v1 specified.)

### II.3 Policy values (defaults + sweep-required)

| Param | Default | Sweep set for B2's 5090 knee re-measure |
|---|---|---|
| `NEMOTRON_DENSITY_BATCH_STEADY` | 0 (OFF) | {0, 1} (1 with B_max≥2 = real batching) |
| `B_max` | 4 | {1, 2, 4} |
| `window_ms` | 10 | {0, 4, 8, 10, 12} |
| `lone_timeout_ms` | **0** (NOT 1) | {0, 1} |
| `queue_capacity` | 4·N_workers | (not swept) |

CLI flags are the primary configuration path: `--batch-steady on/off`, `--batch-b-max`, `--batch-window-ms`,
`--batch-lone-timeout-ms`. Env vars are the fallback (so prior env-based runs still work). Telemetry always
records the EFFECTIVE policy values.

### II.4 Integration point (no global swap)

The worker's `run_steady_chunk_density` becomes:
```
if (scheduler && session.continuation_chunk) {
   // scheduler-routed: record producer event, enqueue, wait future, sync, consume
} else {
   // unchanged B=1 path
   run_steady_encoder_stream(enc_steady, chunk, session, ctx.stream, ...);
}
```
**The scheduler is passed in explicitly** (not a global, not an env-checked-inside-the-helper). The reference
paths (`build_serial_reference`, b1 T1 prep/replay, finalize parent prep, warmup) explicitly pass `nullptr`
for the scheduler → they always take the production B=1 path. Only the B2 measured density-sweep + the
`b2-t1` "scheduler-on" runs pass a real scheduler.

### II.5 Warmup design

- Batched-bucket warmup runs OUTSIDE the measured gate at scheduler construction time:
  `scheduler.warmup_buckets()` runs one synthetic forward per bucket {1, 2, 4} on a throwaway fixture. The
  warmup uses the dispatcher stream + the dispatcher's preloaded loader set; it does NOT route through the
  enqueue/future path (no contamination of the dispatcher loop's telemetry).
- The b2-t1 gate explicitly calls `scheduler.warmup_buckets()` before the measured phase + asserts
  `dispatcher_telemetry.warmup_runs == 3` (one per bucket).

### II.6 `b2-t1` gate (new)

Required cases + assertions:
1. **Single-stream scheduler-ON vs single-stream B=1**: produces tokens identical to production B=1
   reference. (Trivial; the lone_timeout=0 short-circuit means K=1 always → B=1 batched.) Validates the
   scheduler overhead doesn't itself perturb.
2. **Multi-stream FORCED-CONCURRENCY**: N parallel workers driving the corpus, with an EXPLICIT BARRIER
   around continuation chunks so the dispatcher forms B=K batches. Required N≥4 to exercise B=4. The
   barrier guarantees batching actually happens.
3. **Multi-stream STAGGERED**: N parallel workers with the standard 10s per-worker stagger (production-
   shape). Used for fairness/p95/p99 tail.
4. **Scheduler-ON `B_max=1` control**: isolates scheduler/future/stream-sync/NEW-B1 overhead from real B>1
   batching.
5. **Bucket-count assertions** required for cases 2-4: telemetry must report B=1 count, B=2 count,
   K=3→B=4-padded count, K=4 count, K>B_max backlog count. **The gate FAILS if case 2 doesn't form at least
   N_workers/B_max batches at B=B_max** (else the test ran B=1 and proved nothing).
6. **Token comparison**: 0 token divergences fatal. Events counted-not-gated per
   `DENSITY_GOLD_EVENTS_TOLERANT` (A6 wrapper).
7. Output: `B2_T1 START / B2_T1_CASE / B2_T1_BUCKET / B2_T1_RESULT` telemetry + JSON sidecar (same shape as
   b1-t1).

### II.7 Telemetry split (no "batch wait" composite)

Five distinct timers, all per-request:
- `gather_wait_us` = enqueue → dispatcher pops this item.
- `service_wait_us` = pops → batched run starts on dispatcher_stream.
- `cuda_run_us` = batched run CUDA event delta (dispatcher-stream).
- `output_sync_us` = completion event → worker's `cudaStreamWaitEvent` returns.
- `worker_blocked_us` = enqueue → `future.get()` returns AND output_sync completes.

Plus aggregates per dispatch cycle: bucket counts, dispatcher CPU%, dispatcher-stream utilization%,
`window_wakeup_jitter_us`, dispatcher-stream queue depth, and fairness spread (`max_worker_blocked -
median_worker_blocked` across the N streams in the cycle).

### II.8 Fault tolerance + shutdown

- Dispatcher loop wrapped in `try { ... } catch (...) { fault_inject_all_pending_and_close(); exit(1); }`.
- `close()` drains the queue (broadcasts a fault exception to all pending futures), joins the dispatcher
  thread, frees resources.
- Worker's `future.wait_for(W + 200ms)` (where W = window + B_max × per_batch_estimate + slack):
  timeout → fault path (matches the existing per-worker error handling).
- `~BatchedSteadyScheduler()` calls `close()` if not already.

### II.9 A1 (NEW vs PRODUCTION B=1 parity), correctly

- B2 emits a parity check at scheduler construction: `sha256(enc_steady_aoti.pt2)` vs
  `sha256(enc_steady_aoti_b1.pt2)` AND a tensor-level forward parity check on a fixture (atol/rtol within
  documented tolerance).
- Outcome A (bit-identical SHA): log it, free to use NEW B=1 in the scheduler-ON K=1 path without risk
  argument.
- Outcome B (different SHA, but tensor parity within tolerance): log + WARN; the OFF path keeps PRODUCTION
  B=1 (unchanged byte-exact contract); the scheduler-ON path explicitly owns the risk that scheduler-ON
  K=1 uses NEW B=1 (the b2-t1 case 1 catches it).
- Outcome C (tensor parity FAILS): STOP — re-export and re-validate the B=1 package before continuing.

### II.10 A4 (reentrancy), correctly

- `BatchedSteadyLoaderSet`: add a `sealed_` boolean; `preload_all()` sets `sealed_ = true`; `get()` throws
  if `!sealed_` OR if the requested bucket isn't already in the map. No mutex needed.
- Scheduler's construction explicitly calls `preload_all()` after `BatchedSteadyLoaderSet` construction,
  before `start()`.

### II.11 A5 (scratch + index_copy_), correctly

- Scratch tensors live in the dispatcher; one per bucket B (chunks, cache_ch, cache_t, cache_ch_len, length).
- All scratch ops are issued on `dispatcher_stream`.
- Unpack `.contiguous()` per-row produces fresh standalone tensors (per B1's existing pattern). The
  completion event is recorded AFTER all unpack copies are enqueued on the dispatcher stream.
- No scratch tensor or batched output tensor is returned to a worker (workers only receive the per-row
  contiguous copies + the completion event).

### II.12 A7 (manifest), correctly

- `runtime/steady_b_artifacts/MANIFEST.json` emitted by `export_steady_batched.py`: per-B
  `{package_sha256, ep_sha256, shared_weight_sha256, torch_version, cuda_arch, inductor_configs,
  byte_sizes}` + the A1 B=1 parity result.
- C++ loader CONSUMES the manifest fail-closed at `BatchedSteadyLoaderSet` construction: if missing OR if
  the computed package SHA doesn't match the manifest's, throw. Mirrors the finalize bucket manifest
  discipline.

### II.13 5090 knee re-measure plan (post-build)

A 4-axis sweep (smaller than the cartesian product — pre-registered cuts):

**Axis 1 (control)**: scheduler OFF (production B=1 baseline) AND scheduler ON `B_max=1` (scheduler
overhead control).

**Axis 2 (B_max)**: ON × {2, 4}.

**Axis 3 (window/lone)**: at the chosen B_max=4, `window_ms ∈ {0, 4, 8, 10, 12}` × `lone_timeout_ms ∈ {0,1}`
= 10 cells. Pick the SLO-robust knee maximizer at the chosen window/lone.

**Axis 4 (N)**: {40, 44, 48, 56, 64} fresh-process-per-N, stagger-robust.

**Plus**: a burst-injection variant (synchronized N-stream start, no stagger) at N=64 to surface worst-case
HOL tail. Report the enqueue→run-start p99 + the fairness spread on burst-injected runs.

### II.14 Build file layout

- New: `runtime/cpp/batched_steady_scheduler.h` + `.cpp` (the dispatcher, queue, futures, telemetry,
  scratch).
- Modified: `runtime/cpp/density_main.cpp` — scheduler construction conditional on flag, integration via
  explicit pointer (not global), `--mode b2-t1` gate, new CLI args.
- Modified: `runtime/cpp/steady_batch_primitive.h` — add `sealed_` + fail-closed `get()` (A4).
- Modified: `runtime/export_steady_batched.py` — emit MANIFEST.json (A7).
- Optional B2-scope: `runtime/cpp/runtime_io.h` for A8 (only if it doesn't bloat B2 scope).

---

## III. Net + sequencing

GO to delegate the build now, with **this verdict doc (II) as the binding spec**, NOT the v1 `B2-design.md`.
The Codex build task should explicitly quote section II and treat each numbered subsection (II.1-II.14) as
the contract.

Sequencing (no surprises for the Codex build task):
1. Codex builds per §II.1-II.12 + §II.14 (file layout).
2. Codex runs `b2-t1` (§II.6) before reporting; STOP if any token divergence on any case; OK if interim
   events drift in the documented tolerated class.
3. Codex also runs the A1 parity check (§II.9) and logs the outcome.
4. The 5090 knee re-measure (§II.13) is a SEPARATE post-build task (not part of the B2 build task);
   blocked on Codex's build PASS.
5. Paired review of the build (Codex code-review pass + Opus pass) → fold → commit → mark B2 `[x]` → B3.

If the build hits a blocker on §II.2 (input-side sync) or §II.4 (explicit integration), come back for a
fold; both are non-negotiable. The other items can be implementation-pragmatic (e.g., scratch tensor layout
detail in §II.11), but the listed contract is the build's source of truth.
