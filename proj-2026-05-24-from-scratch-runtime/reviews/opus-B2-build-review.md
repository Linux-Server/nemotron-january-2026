# Step B2 build — independent Opus review (2026-05-27)

Adversarial audit of the B2 implementation (new `BatchedSteadyScheduler` + the substantial primitive-header
refactor + density_main integration + manifest discipline) against the binding spec at
`reviews/B2-design-paired-verdict.md` §II. Folded with `codex-B2-build-review.md` after Codex returns.

## TL;DR

**ENDORSE PASS** for B2 commit, with two pre-B3 follow-ups (not blocking):
- **F1**: full-corpus b2-t1 run (1000 rows) once the memory pressure is addressed. The 4-row scope is sound
  for B2 acceptance (all 6 cases cover the bucket coverage + forced concurrency + Bmax=1 control; 0/0
  divergences), but the depth doesn't probe long-horizon compound drift.
- **F2**: the 5090 knee re-measure must report **scheduler memory overhead at N≥40** — the build hit OOM on
  the full-corpus b2-t1 and the existing knee was N=40 on the 5090 at 19.8/32GB before the scheduler. The
  knee sweep must verify the scheduler ON doesn't push N below 40 by memory before measuring the projected
  lift.

Spec faithfulness: high. The §II.2 input-side CUDA sync (the Codex-caught design bug we folded) is correctly
implemented; sealed loader, manifest fail-closed, scratch + index_copy_, explicit-pointer integration, fault
tolerance, telemetry split — all match. The b2-t1 result (0 token / 0 event divergences across 6 cases, real
B=4 batching in forced_concurrency case, max diffs *smaller* than B1's) is the SLO signal.

## 1. Scheduler implementation (`batched_steady_scheduler.{h,cpp}`)

**§II.2 bidirectional CUDA sync — verified correct.**
- Producer side: worker records `producer_event` on its stream after `chunk = cat(ring, new_mel)` + cache
  clones (the integration in density_main does this before enqueue).
- Dispatcher side: `cudaStreamWaitEvent(dispatcher_stream_.stream(), item->request.producer_event, 0)`
  for each row before pack/run (`.cpp` line 276). This records the dependency on dispatcher_stream — the
  subsequent `index_copy_`/`loader.run` will not start until each producer's work completes. ✓
- `cudaEventDestroy(producer_event)` immediately after the WaitEvent (line 280) is safe per CUDA semantics —
  the wait dependency is recorded into the stream, not held in the event object; the destroy releases the
  event handle once no further `WaitEvent` will be issued. ✓
- Completion side: `cudaEventCreateWithFlags(cudaEventDisableTiming)` + `cudaEventRecord(completion,
  dispatcher_stream_.stream())` per row (line 322-324). Worker waits via `cudaStreamWaitEvent(worker_stream,
  completion)` then consumes. ✓

**§II.7 telemetry — all five distinct buckets + wakeup jitter recorded:**
gather_wait_us / service_wait_us / cuda_run_us / output_sync_us / worker_blocked_us /
window_wakeup_jitter_us. ✓ Bucket counts (b1, b2, b4, k3_padded_to_b4, k4, backlog_gt_bmax). ✓

**§II.8 fault tolerance — outer try/catch + propagate + exit(1):**
- The dispatcher_loop's try/catch (line 172-206) on ANY exception sets `fault_`, marks `closing_=true`,
  propagates the exception to all pending QueueItems' promises, logs the error, and `std::exit(1)`. ✓
- `enqueue` checks `fault_` before adding to queue (line 79) and re-throws — so an enqueue after a fault
  immediately surfaces it. ✓
- The `cv_capacity_.wait` predicate includes `fault_` (line 76-77) — an enqueue blocked on capacity wakes
  when the dispatcher faults and re-throws. ✓
- **One race I want to flag (not a blocker):** there's a small window between the dispatcher's fault and
  `std::exit(1)` where a worker's `future.wait_for(timeout)` could fire BEFORE the dispatcher's
  set_item_exception lands. The worker would throw a timeout exception; the dispatcher would still exit(1)
  shortly. Both paths terminate the process; the timeout exception is benign. ✓ Acceptable.

**§II.10 A4 sealed loader — verified:**
- `BatchedSteadyLoaderSet::preload_all()` sets `sealed_ = true` at completion (line 408).
- `get(bucket)` (line 548-557) throws if `!sealed_` OR if `bucket` not in the loader map. No lazy load
  possible from any caller. ✓
- The scheduler's constructor calls `loader_set_.preload_all()` and asserts `loader_set_.sealed()` (`.cpp`
  line 42-43). ✓

**§II.11 A5 scratch + index_copy_:**
- Scratch tensors are stored in `Scratch scratch_` per-bucket map (`.h` line 147).
- All scratch ops are inside `dispatch_batch` under `CUDAStreamGuard(dispatcher_stream_)` (line 261).
- `index_copy_(dim, idx, src)` runs on dispatcher_stream (current stream within guard). The
  `cudaStreamWaitEvent` for each producer was recorded BEFORE the index_copy_, so the index_copy_ sees
  completed producer data. ✓ No scratch view returned to workers (the per-row unpack `.contiguous()` makes
  fresh copies). ✓

**§II.5 warmup_buckets — separate path:**
- Runs synthetic zeros forward on dispatcher_stream + dispatcher's loader directly (`.cpp` line 129-159).
- Does NOT enqueue (so no telemetry contamination of the gather/service buckets).
- Bumps `warmup_runs` counter for separate accounting. ✓

**§II.3 policy values — correct defaults, configurable:**
- Default policy: window_ms=10, lone_timeout_ms=0 (the fold-corrected default, not v1's 1ms), B_max=4,
  queue_capacity=16. ✓
- All four configurable via env vars OR CLI args (density_main parses `--batch-{steady,b-max,window-ms,
  lone-timeout-ms,queue-capacity}`). Env vars are the fallback. ✓

### Minor implementation observations (no fixes required for B2)

- **`set_pending_exception_locked` discards its `ep` parameter** (line 397 `(void)ep`). The function just
  moves items out of the queue; the caller separately calls `set_item_exception(item, ep)` after releasing
  the lock. The ep parameter is dead code. Cosmetic cleanup, not a bug.
- **Per-row completion event** (line 322-324): K events per batch when one event-per-batch would be
  functionally equivalent (all rows complete on the same stream at the same point). The per-row design is
  slightly wasteful (16 events per B=4 batch × hundreds of batches per second at N=64), but correctness-
  identical. Not a blocker.
- **`cudaEventSynchronize(ev_stop)` in dispatch_batch** (line 302) — synchronous block that holds the
  dispatcher thread until the batch's CUDA work completes. This is for the cuda_run_us measurement. Means
  the dispatcher serializes batch-N's CUDA wait with batch-N+1's gather (no overlap). For B2 acceptable;
  if dispatcher becomes the bottleneck at high N, replace with event-polling at gather time.
- **`add_dispatch_telemetry` is called with wakeup_jitter=0.0** (line 319) — the actual jitter is recorded
  separately in `gather_batch` (line 250-252) directly to telemetry. Two paths, both work; mildly
  inconsistent. Not a bug.

## 2. Primitive refactor (`steady_batch_primitive.h`, 196→631 lines)

Codex added a built-in SHA256 + JSON parser inside the header (no external deps), plus the manifest
verification, plus the prepacked API. Big growth but each piece is justified:

**SHA256 (lines 33-161)**: standard FIPS-180-2 implementation; matches the export script's `hashlib.sha256`
output (verified by Codex's A1 parity outcome B: SHAs computed match when packages are byte-equal). Edge
cases (empty file → returns empty-hash; large file → 1MB read buffer). Correctness-clean.

**JSON parser (lines 163-290)**: a hand-rolled mini-parser that handles strings, nested objects/arrays, and
the specific manifest schema. Limitation: doesn't handle escaped Unicode, comments, or unusual whitespace.
Acceptable for a controlled-emit manifest format. Defensive — throws on malformed input. The export script
emits JSON via `json.dumps` which never produces these problematic forms.

**Manifest verify (line 486-524) — fail-closed semantics:**
- Throws on: missing MANIFEST.json file; missing top-level "buckets" array; duplicate B; package name
  mismatch ("enc_steady_aoti_b{B}.pt2"); missing package file; package SHA mismatch; shared_weight SHA
  mismatch across buckets; missing bucket from kBuckets={1,2,4}. ✓ §II.12 fail-closed satisfied.
- Runs at primitive constructor, BEFORE any tensor is loaded → fails fast. ✓

**A4 sealed enforcement (line 549-557)**: the existing `get()` (renamed from B1's lazy version) now
explicitly throws if `!sealed_` or bucket not in map. The lazy `load_bucket()` is only called from
`preload_all()`. ✓

**Prepacked API (lines 424-448)**: `run_prepacked` / `run_raw_prepacked` / `unpack_prepacked_outputs` —
allows the scheduler to use its scratch tensors instead of internal torch::cat. B1's `b1-t1` gate still
uses the original `run()` path (torch::cat); B2's scheduler uses the prepacked path. Both share `get(bucket)`
+ AOTI run + `unpack_outputs`. Clean separation; no regression to B1. ✓

## 3. density_main.cpp integration

**§II.4 explicit pointer (no globals) — verified:**
- `run_steady_chunk_density` takes `BatchedSteadyScheduler* scheduler = nullptr` as an optional parameter
  (line 1734). Default nullptr → unchanged B=1 path. ✓
- Scheduler-routed branch (line 1237-1281): records producer event, enqueues, waits future with timeout,
  records output_sync, optionally compares against B=1 reference for diff stats. ✓
- Reference paths (build_serial_reference for b1-t1, finalize parent prep, warmup, density-sweep flag-OFF)
  explicitly pass nullptr → take the production B=1 path. ✓ §II.4 satisfied.

**b2-t1 case design — substantively matches §II.6:**
- `single_stream_scheduler_on` — single-stream sanity (scheduler overhead control without batching).
- `multi_stream_forced_K2_B2` — exercises B=2 bucket.
- `multi_stream_forced_K3_padded_B4` — exercises K=3 padding.
- `multi_stream_forced_concurrency_B4` — the §II.6 forced-concurrency barrier case (20 B=4 batches actually
  formed!).
- `multi_stream_staggered` — production-shape (short-circuits to B=1; tests degradation case).
- `scheduler_on_Bmax1_control` — the §II.6 Bmax=1 control to isolate scheduler overhead from real batching.

All 6 cases PASS with bucket-count assertions met. ✓ §II.6 satisfied.

**A1 parity check + outcome B handling:**
- Codex emits the SHA256 + tensor-parity check at scheduler construction. ✓
- Outcome was B (SHAs differ, tensors bit-identical max=0). Per §II.9 outcome B, the OFF path stays on
  PRODUCTION B=1 unchanged. Verified: the OFF-path code uses `args.dir + "/enc_steady_aoti.pt2"`
  (production); the scheduler ON path uses the steady_b_artifacts NEW packages. The b2-t1 case 1
  (single_stream_scheduler_on) demonstrates the scheduler-ON path on NEW B=1 produces 0 token divergences
  vs the production B=1 reference. ✓

**OFF-path preservation:**
- Conditional construction: `BatchedSteadyScheduler` is only constructed when `args.batch_steady == "on"`
  (effective) AND the b2-t1 or density-sweep mode requires it. When OFF, the scheduler pointer is nullptr
  and never instantiated → no thread, no scratch, no memory. ✓ §II.3 satisfied.
- The 20-session N=4 smoke confirms no token regression at one operating point. The full byte-exact
  guarantee remains the B1 commit's 1000/1000 (which used the production B=1 path unchanged).

## 4. Scope reductions Codex flagged honestly

**b2-t1 ran 4 reference rows, not 1000.** Reason: memory pressure on the 5090 32GB. The 6 cases × multi-
stream concurrent scheduler instances + per-case fresh loaders OOM'd the GPU on the full corpus. Codex's
fix-1 (shared loaders across cases) reduced pressure but the full corpus still OOM'd; Codex fell back to 4
rows which works.

**Is 4 rows acceptable for B2 acceptance?**
- The 6 cases cover all 3 buckets × all bucket-formation regimes (forced, padded, staggered, control).
- 0 token divergences across all 6 cases = the SLO signal is clean for the cases that ran.
- Real B=4 batching was exercised (forced_concurrency formed 20 B=4 batches).
- max diffs are *smaller* than B1's (compound multi-chunk drift is no worse than single-chunk-insert).

**My take: acceptable for B2 commit; flag the full-corpus b2-t1 as a B3 pre-condition** (follow-up F1). The
B3 L40S sweep needs full-corpus confidence anyway; running it as a B3 prerequisite makes natural sense, and
the L40S has 48GB vs the 5090's 32GB so the memory budget is easier there.

**OFF-path smoke ran 20 sessions, not 1000.** Acceptable for "no regression at this operating point" given
the B1 commit's full 1000/1000 byte-exact guarantee on the production B=1 path (which B2 leaves untouched).

## 5. Memory considerations for the 5090 knee re-measure (§II.13)

Codex hit OOM on the full-corpus b2-t1, which is a real signal that **B2 adds memory overhead the existing
5090 knee N=40 doesn't have budget for at high N**. The added overhead:
- 3 B-bucket AOTI packages preloaded (~2.5GB each — but shared constants means ≪ 7.5GB, probably ~2.5GB +
  small per-bucket activations).
- Scheduler scratch tensors (small — per-bucket B×geometry, ~MBs total).
- Dispatcher stream + thread (trivial memory).

The existing N=40 5090 knee was at 19.8/32GB ≈ 62% headroom. If the scheduler adds ~2.5GB, that's
22.3/32GB ≈ 70% — still feasible. **But the OOM on b2-t1 at full corpus suggests** there's something more
expensive happening (perhaps the per-case fresh loader instantiation Codex partly fixed, or accumulated
activations across the b2-t1 cases). The knee re-measure (§II.13) MUST report memory headroom at N=40 with
scheduler ON; if N=40 is no longer feasible, B2's projected lift evaporates before it gets a chance to
demonstrate.

**Follow-up F2: 5090 knee re-measure includes per-N peak memory + headroom telemetry; report scheduler ON
vs OFF memory delta explicitly.**

## 6. Net verdict

**PASS B2.** The implementation is faithful to §II, the b2-t1 SLO signal is clean (0 token divergences), the
A1 outcome is correctly handled, OFF-path byte-exactness is preserved by conditional construction. Commit
B2; mark B2 row `[x]` in PHASE2-PLAN.md; record the two follow-ups (F1 full-corpus b2-t1, F2 knee-sweep
memory telemetry) against the §II.13 knee re-measure task (which is a separate post-build task).

The 5090 knee re-measure is the next-real-work — that's where N=40 → ~47+ lift is either demonstrated or
not, and where the F2 memory question gets answered. Don't conflate it with B2 acceptance.
