# Finalize Python Tail Analysis

Scope: code-level Python/host-side analysis only. Runtime code is unchanged. The line references below are current for `src/nemotron_speech/server.py` in this workspace.

## Verdict

The hypothesis is code-plausible if the parallel cloud run localizes the missing TTFS tail to server-side wall time outside `model_wall_ms`: the finalize path has several serialized Python sections that can add variance when many sessions finalize together on one event loop and a shared GIL.

The single biggest Python-tail source is the serialized "finalize host envelope": scheduler/session locks plus lane/global model reservation wrap fork cloning, per-final preprocessing/stacking, decoder-state cloning, text delta/json/logging, and websocket emission. Inside that envelope, the highest-value concrete source is the fork/decode-state cloning path, especially the initial deep fork clone at `server.py:5783-5915` plus the second decoder-state clone before batched model execution at `server.py:6768-6789`.

Existing `NEMOTRON_FINALIZE_PROFILE` telemetry already measured `model_wall_ms` near ~40 ms and explicit fork/preproc/lock buckets much smaller in one config. That does not rule out Python tail, because event-loop scheduling, websocket send, logging, profile JSON, GC pauses, and some lock/lane serialization are either outside or only partly represented by those buckets.

## Critical Path Map

- Control events enter through websocket reset parsing at `server.py:3637-3644`, scheduler queueing at `server.py:3780-3788`, continuous reset handlers at `server.py:5189-5300`, non-scheduler continuous handlers at `server.py:5672-5777`, and legacy hard reset at `server.py:7943-8044`.
- Scheduler drain collects one event per session and optionally batches debounce finalizes at `server.py:3923-4046`; batched finalizes acquire all involved `state_lock`s and then prepare, flush, emit, and finish while still in that lock scope at `server.py:4089-4151`.
- Fork creation does audio/caches/decoder-state cloning at `server.py:5783-5915`.
- Finalize timing currently records vad/debounce/fork/lock/send timestamps at `server.py:6007-6021`.
- Batched finalize preprocessing and model execution run through `_prepare_final_fork_batch_row`, `_prepare_final_fork_batch_rows_batched_preprocess`, `_process_final_fork_groups`, and `_process_final_batch_rows` at `server.py:6309-6929`.
- Serial finalize runs through `_continuous_finalize_emit_locked` at `server.py:6958-7265` and `_process_final_chunk` at `server.py:8072-8236`.
- Emission/dedup uses `_continuous_append_only_delta` at `server.py:358-383`, `_send_json_locked` at `server.py:5343-5365`, continuous prepared emit at `server.py:6218-6308`, and serial emit at `server.py:7183-7265`.
- Finalize profiling allocates profile dicts and emits JSON at `server.py:2012-2072` and `server.py:2235-2340`; the encoder profiling wrapper adds synchronizes/events at `server.py:2354-2387`.

## Ranked Optimizations

| Rank | Optimization | Where | Mechanism | Expected Tail Impact | Byte-Exact Risk | Effort |
|---:|---|---|---|---|---|---|
| 1 | `NEMOTRON_FINALIZE_PARALLEL_PINNED_LANES=1`: run finalize lane groups concurrently | `server.py:6166-6185`, `server.py:6096-6148`, `server.py:2615-2624` | Current batched finalize groups by lane, then awaits each lane group serially. Dispatch one task per reserved lane and `gather` them so host wait and GPU work on independent lane executors overlap. Keep per-session lane affinity and per-stream emit ordering. | High when `model_lanes > 1`: can remove one-lane-at-a-time tail, commonly tens of ms and potentially more under conc-10 synchronized finals. | Medium. Cross-session ordering can change, but per-stream final output should remain identical if each session stays on its pinned lane. Needs byte-exact multi-stream canary. | Medium |
| 2 | `NEMOTRON_FINALIZE_PINNED_SERIAL=1`: avoid exclusive-model fallback for serial finalize | `server.py:7044-7172`, `server.py:2646-2665` | When `model_lanes > 1` but `NEMOTRON_BATCH_FINALIZE` is off, serial finalize falls back to `_scheduler_exclusive_model_path` plus `inference_lock`, waiting for all lane work and blocking all lanes. Use the pinned-lane path for one-session finalize too, still gated by session generation. | High if production/gate ever runs with batch finalize off. This is the clearest lock/lane tail amplifier. | Medium. Finalization is non-steady geometry; keep lane affinity and fail closed to exclusive path on any conflict. | Medium |
| 3 | `NEMOTRON_FINALIZE_NO_RECLONE_BATCH_STATE=1`: remove duplicate decoder-state clone before stacking | `server.py:5832-5833`, `server.py:5893-5899`, `server.py:6768-6789` | The fork already deep-clones `previous_hypotheses` and `pred_out_stream`. `_process_final_batch_rows` clones both again before `stack_hypotheses`/`stack_pred_out`. For disposable forks, pass fork-owned decoder objects directly into the batch stack. | Medium to high for long/complex decoder state. Reduces GIL-held recursive object traversal, Python allocations, tensor clones, and GC pressure. | Low-medium. Normal path should be byte-exact because the fork is disposable. Risk is fallback-after-mutation semantics if a batched call errors after the decoder mutates state; keep old path on exceptions or clone only for fallback snapshots in debug. | Low |
| 4 | `NEMOTRON_FINALIZE_FAST_FORK=1`: preallocate/reuse fork audio buffers and use a light fork object | `server.py:5789-5838`, `server.py:5842-5915`, helpers at `server.py:115-182` | Replace `pending.copy() + zeros + concatenate` with one `np.empty(total)` and deterministic slice fills using a precomputed final-padding zero view. Avoid allocating unused `ASRSession` locks/queues/bytearrays for disposable forks by using a minimal fork state object with the fields consumed by `_process_final_chunk` and `_process_final_batch_rows`. Optionally share read-only `mel_frame_ring` under the existing lock. | Medium. Removes per-final heap churn and can reduce GC-driven P95 spikes when several sessions fork together. | Low-medium. Audio bytes remain identical if slices are filled exactly. Light fork risk is missing an ASRSession field; gate and test both serial and batched finalize. Do not alias mutable cache/hyp/pred unless separately proven with `NEMOTRON_FORK_ASSERT=1`. | Medium |
| 5 | `NEMOTRON_FINALIZE_PREPROC_WORKSPACE=1` and `NEMOTRON_FINALIZE_PREPROC_VIEW_SLICES=1`: reuse fixed-audio buffers and avoid row clones | `server.py:3128-3173`, `server.py:6335-6384`, `server.py:6392-6435`, `server.py:6498-6635`, `server.py:8104-8163` | `_build_fixed_preprocess_audio` allocates a zeroed `K` array per 16-frame step. Use per-lane/per-executor NumPy workspaces, fill only the deterministic regions, and zero the invalid tail. In batched preproc, fill a reusable `[B,K]` array instead of `np.stack`. Return views from `mel[index:index+1,...]` at `server.py:6432-6434` because the later `torch.cat` copies anyway. | Medium if final tails need multiple preproc invocations or if `_PREPROC` is enabled; otherwise low to medium because measured explicit preproc was small. Main value is reducing allocation variance. | Low-medium. Exact buffer initialization is critical; stale workspace bytes would break output. Validate mel tensor byte equality before transcript canary. | Medium |
| 6 | `NEMOTRON_FINALIZE_OBSERVABILITY_SAMPLE=N` and `NEMOTRON_FINALIZE_PROFILE_LIGHT=1`: defer or sample finalize logs/profile JSON | `server.py:6077-6081`, `server.py:6680-6693`, `server.py:7039-7043`, `server.py:7281-7288`, `server.py:1068-1081`, `server.py:2235-2340`, `server.py:2354-2387` | Per-final info logs and profile records do string formatting, dict construction, `json.dumps(sort_keys=True)`, and synchronous sink writes. Profile mode also adds CUDA syncs/events and small GPU-to-CPU tensor value reads at `server.py:1974-1977`, `server.py:2144-2162`, `server.py:6882-6887`, and `server.py:8212-8219`. Sample records, aggregate counters, and avoid tensor value reads in light mode. | Medium if profiling or verbose logs are on, especially with stdout/cloud logging. Low if all verbose/profile paths are off. | Very low for transcript bytes. Telemetry changes only. | Low |
| 7 | `NEMOTRON_FINALIZE_DELTA_FASTPATH=1`: avoid token-list work on prefix finals | `server.py:358-383`, `server.py:6228-6278`, `server.py:7183-7236`, prompted strip at `server.py:1303-1321` | Fast path when `final_text` is an exact string prefix of `continuous_emitted_text` or vice versa: compute suffix using string length and normalize with `" ".join(suffix.split())`, falling back to current token-overlap algorithm. For prompted models, skip regex stripping if the text has no `<`. | Low to medium. Helps long cumulative transcripts and reduces GIL allocations during parallel emit. | Low if property-tested against the current delta function across whitespace/correction cases. | Low |
| 8 | `NEMOTRON_FINALIZE_PRESERIALIZE_JSON=1`: serialize payloads outside broad session-lock scopes | `_send_json_locked` at `server.py:5343-5365`, emit call sites at `server.py:6261-6274` and `server.py:7219-7232` | Build the exact same JSON string before awaiting websocket send, after setting `final_sent`. Keep the websocket-closed check and send in a small helper that accepts already serialized text. This removes `json.dumps` CPU time from the critical lock/send section. | Low to medium. JSON payloads are small, but this can reduce event-loop stalls when many finals emit together. | Low if using identical `json.dumps` options. Do not change separators/sort behavior for client payloads unless explicitly allowed. | Low |
| 9 | `NEMOTRON_FINALIZE_NARROW_LOCKS=1`: prepare under lock, run fork model outside, reacquire to validate and emit | `server.py:4097-4148`, serial path `server.py:6958-7265`, ready lane lock example `server.py:4603-4624` | Current batched finalize holds all session `state_lock`s across fork model execution, websocket send, logging, and finish bookkeeping. A split-lock design would clone a complete fork and expected generation under lock, release during model work, then reacquire per session to validate generation and emit. | Potentially high for event-loop responsiveness and multi-session lock contention. | High. The current generation model does not invalidate all cancel/resume cases; e.g. scheduler VAD-start cancellation at `server.py:5144-5157` changes stop state but does not clearly increment `scheduler_generation`. This must be fixed or the stale final suppression can be wrong. | High |
| 10 | `NEMOTRON_FINALIZE_SCHEDULER_TIMER=1`: scheduler-owned debounce deadlines instead of one task+queue hop per session | timer at `server.py:5479-5492`, scheduler waits at `server.py:3884-3921`, drain at `server.py:3923-4046` | Per-session debounce tasks all wake on one event loop, enqueue, and wake the scheduler. A scheduler-owned deadline heap/scanner would remove task creation/cancel churn and one queue hop, and can record timer-due lag directly. | Low to medium normally; can remove outliers when many timers expire simultaneously or the loop is busy. | Medium. Must preserve `stop_seq`, `reset_seen`, close behavior, and exact debounce timing semantics. | Medium-high |

## Host-Side Tail Inventory

### 1. Allocations, Copies, and GC Pressure

- `clone_tree` and `clone_hypotheses_deep` recursively copy tensors, arrays, lists, dicts, dataclasses, and NeMo objects at `server.py:123-182`. This is pure Python traversal for object structure plus tensor/array allocations.
- The fork builder copies `pending_audio`, allocates silence, concatenates, allocates an `ASRSession`, copies `raw_audio_ring`, clones mel/cache tensors, and deep-clones decoder state at `server.py:5789-5838` and `server.py:5842-5915`. Under conc-10, these happen per final on the main event-loop thread during prepare.
- `NEMOTRON_FORK_ASSERT` doubles down on clone cost: parent snapshots clone cache/hyp/pred at `server.py:5917-5936`, then deep equality and tensor/array hashing live at `server.py:256-355` and `server.py:5938-5978`. Keep it for canaries, not latency gates.
- Final preprocessing allocates a fixed audio buffer every call at `server.py:3168`, then creates CUDA tensors and length tensors at `server.py:3143-3145`. The serial final loops build lists of mel slices and concatenate at `server.py:6335-6384` and `server.py:8104-8163`.
- Batched final preprocessing builds Python state objects/lists/dicts at `server.py:6461-6635`, does `np.stack` at `server.py:6420`, creates CUDA length tensors at `server.py:6424-6425`, and clones every returned mel row at `server.py:6432-6434`.
- Batched finalize stacks and reclones already-forked decoder state at `server.py:6768-6789`; scatter then assigns row cache/decoder/text per row at `server.py:6841-6929`.
- Profile dicts and public-record dicts allocate on every profiled final at `server.py:2012-2072` and `server.py:2299-2307`.

Why this makes tails: Python allocation and recursive clone traversal are serialized by the GIL. They also create bursty garbage. Even if tensor clones enqueue GPU copies asynchronously, the object traversal and allocator work can delay every other final on the same event loop.

Best fixes: ranks 3, 4, and 5.

### 2. GIL-Held Pure-Python Sections

- `_continuous_append_only_delta` splits both cumulative strings into token lists, compares lists, slices, and joins at `server.py:358-383`. Emit calls use it at `server.py:6234-6237` and `server.py:7189-7192`.
- Continuous emission checks correction state, updates cumulative strings, and logs string slices at `server.py:6228-6278` and `server.py:7183-7236`.
- `_extract_hypothesis_text` may stringify hypotheses and, for prompted models, run two regex passes at `server.py:1303-1321`; final batch/serial call it at `server.py:6894` and `server.py:8222`.
- Batch grouping and fallback paths build dicts/lists and strings at `server.py:6736-6746`, `server.py:6660-6693`, and `server.py:6931-6956`.
- Debug log f-strings are evaluated even when debug output is filtered. Examples on the finalize path include `server.py:6044-6049`, `server.py:6212-6216`, `server.py:6228-6233`, `server.py:6279-6283`, `server.py:7008-7013`, and `server.py:7177-7181`.

Why this makes tails: each final's text/delta/log work is small alone, but at conc-10 it runs serially on the event loop after model completion. It can become visible exactly as a P95 "post-GPU gap."

Best fixes: ranks 6, 7, and 8.

### 3. Asyncio Scheduling and Handoffs

- Debounce expiry is a separate task per session, sleeping and then queueing a scheduler event at `server.py:5479-5492`. Simultaneous expiries wake and enqueue serially on one loop.
- Scheduler drain scans sessions and gets at most one event from each queue per pass at `server.py:3923-3962` and `server.py:3964-4046`. In the batched-barrier path, audio events are processed immediately before finalize events at `server.py:3991-3996`.
- Non-batched scheduler events drain ready backlog before dispatch at `server.py:4975-4982`; the barrier drain itself can process ready chunks at `server.py:4883-4892`.
- Batched finalize locks every involved session in sorted order at `server.py:4096-4100`, then keeps those locks through prepare, model flush, emit, and finish at `server.py:4101-4148`.
- Model execution crosses to a thread executor at `server.py:2519-2527` or lane executor at `server.py:2615-2624`. The event loop is free while awaiting, but the session locks and lane/global reservations are still held by the coroutine.
- `_send_json_locked` awaits websocket I/O at `server.py:5363-5365` while called from locked finalize emit paths at `server.py:6261-6274` and `server.py:7219-7232`.

Why this makes tails: a final can be ready but wait for the debounce task to run, the queue item to be drained, prior ready chunks/barriers, lock acquisition, executor availability, and then send/log CPU. These are not GPU model time and can be bursty under synchronized finals.

Best fixes: ranks 1, 8, 9, and 10.

### 4. Logging, Telemetry, and JSON

- `_send_json_locked` serializes every client payload with `json.dumps` at `server.py:5363-5365`.
- Scheduler finalize fork-clone timing is logged at info level for every scheduler final at `server.py:6077-6081` and `server.py:7039-7043`.
- Batch finalize telemetry logs every B>1 batch, including sorted dict formatting, at `server.py:6680-6693`.
- Speculative finalize finish logs a long retention summary at `server.py:7281-7288`; true-boundary cleanup/cold reset logs at `server.py:7326-7330` and `server.py:7370-7374`.
- Retained-cache telemetry calls CUDA memory stats and logs when `batch_requested` is true at `server.py:1068-1081`.
- `NEMOTRON_FINALIZE_PROFILE` emits one sorted JSON record per final at `server.py:2299-2307` and histogram JSON every 10 records at `server.py:2312-2340`.
- The profiling wrapper adds synchronize/event work around encoder calls at `server.py:2354-2387`, and tensor value extraction uses `.cpu().tolist()` at `server.py:1974-1977`.

Why this makes tails: loguru sinks and cloud stdout can block unpredictably. `json.dumps(sort_keys=True)` and formatted dicts hold the GIL. Profile mode is intentionally perturbing and should not be used as a production-latency mode.

Best fix: rank 6, plus keep profile runs separate from TTFS gates or use a light/sampled profile.

### 5. Lock and Lane Contention

- The global `inference_lock` is created at `server.py:560-561`. Batch finalize uses it for all flush items when pinned lanes are unavailable or inactive at `server.py:6191-6198`.
- With lanes and batch finalize, `_scheduler_pinned_model_lane_path` waits on the lane condition until no exclusive path, the lane is available, and none of the sessions are inflight at `server.py:6122-6134`.
- Lane-group finalize then processes `by_lane` groups sequentially at `server.py:6173-6185`.
- If `model_lanes > 1` but batch finalize is off, serial finalize enters `_scheduler_exclusive_model_path` plus `inference_lock` at `server.py:7097-7133`. `_scheduler_exclusive_model_path` waits for all inflight lane tasks and blocks new lane work at `server.py:2646-2665`.
- Legacy hard reset also holds `inference_lock` around `_process_final_chunk` at `server.py:8010-8017` and may reacquire it for session warmup at `server.py:8061-8065`.
- Ready lane batches hold session locks while awaiting lane model calls at `server.py:4603-4624`, which can delay finalize lock acquisition in `server.py:4097-4100`.

Why this makes tails: under parallel finalize, one session's host work and model call can become another session's wait even when GPU compute per final is only ~40 ms. This is especially true when finalizes do not coalesce into B>1 same-shape batches, or when different `T` values force sequential model calls inside one lane executor.

Best fixes: ranks 1 and 2 first; rank 9 only after generation/cancel semantics are hardened.

## Suggested Measurement Additions Before Flipping Fixes

These are instrumentation-only and should be default-off:

- `NEMOTRON_FINALIZE_HOST_PROFILE=1`: add phase timers for `prepare_item_ms`, `fork_build_ms`, `executor_submit_to_start_ms`, `executor_total_ms`, `post_model_emit_ms`, `json_dumps_ms`, `websocket_send_await_ms`, `profile_emit_ms`, and `finish_log_ms`.
- Record event-loop lag at debounce: include target due time in the event, not just queue time, so timer wake delay is visible next to `queue_wait_ms`.
- Record `gc.get_count()` deltas and optional `gc.callbacks` pause times around finalize batches. Do not disable GC until the pauses are proven material.
- Separate `lock_wait_ms` from "wait behind prior lane group" in `server.py:6166-6185`; currently one `lock_wait_start` can make later lane groups look like lock wait even when they are waiting on earlier finalize work.

## Implementation Guardrails

- All changes must be env-flag gated and default-off.
- Byte-exact means the per-stream final transcript event sequence: final count, delta text, empty/suppressed behavior, and subsequent dedup state. Telemetry bytes can change only under telemetry flags.
- Keep `NEMOTRON_FORK_ASSERT=1` and multi-final continuous-session canaries as the correctness gate for any fork/clone optimization.
- Do not remove final padding, change `keep_all_outputs=True`, change rc1 behavior, or alias mutable decoder/cache state unless the alias is proven read-only by canary.
- If lock narrowing is attempted, first ensure every cancellation/resume path increments or otherwise invalidates the finalize `expected_generation`; otherwise stale fork output can be emitted after a VAD-start cancellation.
