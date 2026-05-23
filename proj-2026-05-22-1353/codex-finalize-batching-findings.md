# Codex finalize batching findings

## Executive summary

`NEMOTRON_BATCH_FINALIZE=1` is active, but it is not a coalescing barrier. It only batches
`debounce_expired` events that are already visible at the head of per-session queues during one scheduler scan.
That scan pops at most one event per session, then immediately flushes whatever it found. There is no finalize
equivalent of `batch_max_wait_ms`.

After that small event set is prepared, `_continuous_flush_finalize_items_locked` splits it by pinned model lane
and calls `_process_final_fork_groups` once per lane. Therefore effective `B` is not "all coincident finals";
it is "number of collected final events on the same pinned lane and same finalize batch key in this one flush."
With coincident debounce timers arriving slightly apart, and with two pinned lanes, the usual result is one event
per flush or one event per lane group, so `_process_final_batch_rows` sees `B=1`.

The high-value fix is idea 1: add a flag-gated finalize coalescing drain before
`_scheduler_process_finalize_event_batch`, then keep the existing byte-exact batch path. A second stage can run
per-lane finalize groups concurrently. I would not start by forcing a mixed-lane single `B=N` model call, because
that crosses an explicit pinned-lane invariant and has a larger byte-exact risk.

Idea 2, a fused kernel for the Python pieces, is not a good first lever. The measured Python pieces are about
0.05 to 0.12 ms p95, while the model call is about 11 ms and the observed tail is about 100 ms of scheduler queue
serialization. Fusing those pieces would not materially move the tail.

## Exact code path that makes B=1

1. Debounce finalization is enqueued per session by `_continuous_debounce_timer`.
   `src/nemotron_speech/server.py:5508-5521` sleeps, then queues
   `("debounce_expired", stop_seq, time.perf_counter())` and wakes the scheduler.

2. The scheduler loop calls `_scheduler_drain_once`.
   `src/nemotron_speech/server.py:3913-3951` repeatedly drains work until no progress is made.

3. Without batched barrier drain, `_scheduler_drain_once` does a single pass over sessions.
   `src/nemotron_speech/server.py:3952-3987`:
   - creates `finalize_events = []`;
   - for each session, skips if in-flight;
   - pops exactly one event with `queue.get_nowait()`;
   - appends it only if `_scheduler_finalize_event_if_batchable` accepts it;
   - after the pass, immediately calls `_scheduler_process_finalize_event_batch(finalize_events)`.

4. Batchable finalize events are only `debounce_expired`.
   `src/nemotron_speech/server.py:4106-4116` returns a `SchedulerFinalizeEvent` only when
   `_scheduler_batch_finalize_active()` is true and `event[0] == "debounce_expired"`. `close`, `reset`, and
   immediate forced finalizes are not included in this batch path.

5. Batched barrier drain does not solve this. The probe had `batch_barrier_drain=False`, and even when active,
   `src/nemotron_speech/server.py:3993-4073` still appends batchable finalize events and immediately flushes them
   at `4069-4070`. The barrier logic defers other non-audio events behind ready backlog; it does not coalesce
   finalizes.

6. `_scheduler_process_finalize_event_batch` prepares all events it was given and flushes immediately.
   `src/nemotron_speech/server.py:4118-4180` acquires session locks, validates `PENDING_FINALIZE` and `stop_seq`,
   invalidates the scheduler generation, calls `_continuous_prepare_finalize_item_locked`, then calls
   `_continuous_flush_finalize_items_locked(items)` at `4170-4171`.

7. `queue_wait_ms` is measured before preparation, so it captures time waiting for this scheduler pickup.
   `src/nemotron_speech/server.py:2012-2027` computes `queue_wait_ms` from the event's queued perf timestamp to
   `_new_finalize_profile` start. `_continuous_prepare_finalize_item_locked` calls that at
   `src/nemotron_speech/server.py:6065-6069`.

8. The flush is lane-local, not global.
   `src/nemotron_speech/server.py:6179-6231` filters `flush_items`, then when lanes and batch finalize are active:
   - groups items by `_scheduler_assign_session_model_lane` at `6195-6200`;
   - iterates `for lane_id, lane_items in sorted(by_lane.items())` at `6202`;
   - reserves one pinned lane and calls `_process_final_fork_groups(lane_items)` at `6210-6214`.

9. `_process_final_batch_rows` only sees the lane group rows that also share the exact finalize key.
   `src/nemotron_speech/server.py:6724-6775` prepares rows, groups them by `_finalize_batch_group_key_for_row`,
   chunks by `batch_max_size`, and calls `_process_final_batch_rows(batch_rows, key)`. The actual model call is at
   `src/nemotron_speech/server.py:6791-6868`, with `B = len(rows)`.

So the measured behavior is expected from the current code: the scheduler does not wait for coincident finalizers,
and the flush splits any small collected set by lane before the batched model call.

## Measurement cross-check

From `local_probe2_records.txt.burst`, considering only the emitted `path=="batch_finalize"` records:

- 144 batch-finalize records: `B=1` for 134, `B=2` for 10.
- The paired `path=="serial_finalize"` close records add another 144 `B=1` records, which explains the overall
  `{1: 278, 2: 10}` distribution.
- Batch-finalize `queue_wait_ms`: p50 about 39.5 ms, p95 about 102.7 ms, max about 116.8 ms.
- Batch-finalize model wall: p50 about 10.5 ms, p95 about 12.4 ms.
- Batch-finalize `final_gather_ms` p95 about 0.12 ms and `clone_hyp_flush_ms` p95 about 0.08 ms.

The rare `B=2` records line up with the code path: two same-lane sessions happened to be present in the same flush.
The server log also shows those as lane-local groups, for example one `finalize_batch` on lane 0 with two sessions,
then another on lane 1 with two sessions. Most other flushes show a single session per pinned lane acquisition.

The `close` records are a separate inefficiency: because `_scheduler_finalize_event_if_batchable` accepts only
`debounce_expired`, `close` goes through the serial `_continuous_finalize_emit_locked` path at
`src/nemotron_speech/server.py:6995-7130`. In these probe records the close final is usually suppressed as
`empty_or_duplicate`, but it still runs a model call. That is not the `queue_wait` tail being asked about, but it is
worth cleaning up separately after the debounce batching issue is fixed.

## Idea 1: batch coincident finals

Recommendation: do this first, default-off, with a staged implementation.

### Stage 1: add a finalize coalescing drain

Add a new flag such as `NEMOTRON_BATCH_FINALIZE_DRAIN=1` or
`NEMOTRON_BATCH_FINALIZE_COALESCE_MS=<n>`, requiring `NEMOTRON_BATCH_FINALIZE=1`. Keep default off.

Specific scheduler change:

- In both `_scheduler_drain_once` and `_scheduler_drain_once_batched_barrier`, when the first batchable finalize
  event is found, stage it instead of flushing immediately.
- Continue collecting only head-of-queue `debounce_expired` events from other non-inflight sessions until one of:
  `len(finalize_events) >= batch_max_size`, the finalize coalescing deadline expires, or no more events arrive.
- The cleanest implementation is a small pending-finalize accumulator with a first-event deadline integrated into
  `_scheduler_wait_timeout`. A simpler probe can use a bounded `await asyncio.sleep(0)` or short max-wait loop
  before `_scheduler_process_finalize_event_batch`, as long as no session state locks are held during the wait.
- Never skip around a non-final head event in a session queue. Per-session FIFO ordering must stay intact.
- Keep `task_done()` ownership unchanged: only mark queued finalize events done after
  `_scheduler_process_finalize_event_batch` completes, as current code does at `4178-4180`.

This alone should turn coincident `conc-12` debounce events from many one-row scheduler flushes into one prepared
item set. With current lane grouping, the model calls would likely become two lane-local batches around `B=6` each,
not one `B=12` call.

### Stage 2: avoid serial lane-group flushing

The current lane-group loop at `src/nemotron_speech/server.py:6202-6219` awaits one lane group at a time. Once a
coalesced item set exists, this can still serialize lane 0 then lane 1.

Safer second-stage change:

- Keep pinned lane grouping.
- Reserve and dispatch each lane group to its own `_run_scheduler_model_lane_call` task.
- Await all lane tasks, then update `item.final_text` and emit in the original deterministic item order.

This respects the existing lane-affinity model and should produce one batched model call per lane in parallel.

Riskier optional change:

- Force all finalize items into a single mixed-lane `B=N` batch on one lane.
- This requires relaxing the explicit mixed-lane guard in `_scheduler_pinned_model_lane_path`
  (`src/nemotron_speech/server.py:6140-6148`) or bypassing that path.
- I would not do this until the per-lane coalesced version is byte-exact and measured. It may be unnecessary, and
  it violates a current invariant that keeps session work on its pinned model lane.

### Byte-exact and fail-closed requirements

This path is byte-exact-critical. The implementation should be behind new flags, default off, and fail closed to
the current behavior.

Required gates:

- Default-off identity: with the new flag unset, byte-for-byte current behavior and current B histogram.
- Full continuous-session byte-exact canary: compare flag-off vs flag-on over multi-final sessions, including
  final count, final text, deltas, ordering, empty/suppressed finals captured server-side, and downstream
  delta-suppression on later turns.
- `NEMOTRON_FORK_ASSERT=1` clean: parent cache tensors, `previous_hypotheses`, and `pred_out_stream` must remain
  unchanged across fork finalization.
- Forced coalescing test: queue N `debounce_expired` events before wakeup, assert `B>1` is actually seen in
  `_process_final_batch_rows`, and assert outputs match the serial path.
- Stale event tests: `vad_start`, `reset`, `end`, and `close` during the coalescing wait must not emit stale finals.
  The existing `stop_seq` and `scheduler_generation` checks should remain the authority.
- Lane matrix: lanes 1 and 2, `NEMOTRON_BATCH_FINALIZE_PREPROC` on and off, and K=4 + MPS.
- Bounded memory: cap coalesced finalize B with `batch_max_size` or a finalize-specific max. If stacking, model
  call, or scatter fails for a multi-row batch, log once and fall back to current per-item or per-lane serial
  behavior.

## Idea 2: fused kernel for Python pieces

Do not prioritize this.

The telemetry rejects it as a tail lever:

- `final_gather_ms` p95 is about 0.12 ms.
- `clone_hyp_flush_ms` p95 is about 0.08 ms.
- Scatter is also sub-millisecond in the records.
- The model call is about 11 ms.
- The observed tail is `queue_wait_ms` around 100 ms from scheduler serialization.

Even a perfect fused implementation for stack/cache/scatter would remove far less than 1 ms from a path whose tail
is dominated by waiting behind many separate model calls. Also, `clone_hypotheses_deep` and decoder state are Python
object structures, not simple device buffers, so "one fused on-device kernel" is not a straightforward local
optimization without redesigning decoder-state representation. That carries correctness risk in exactly the
byte-exact-sensitive fork path.

Revisit only if, after idea 1 succeeds and real `B>1` final batches are common, fresh telemetry shows gather/scatter
growing into a material share. The current data says it will not move P95.

## Ranked next steps

1. Implement a default-off finalize coalescing probe in the scheduler. Target: make coincident `debounce_expired`
   events reach one `_scheduler_process_finalize_event_batch` call instead of separate immediate flushes.

2. Add the forced coalescing and multi-final byte-exact canaries before any production rollout. Run with
   `FORK_ASSERT` and include suppressed/empty finals.

3. Re-run the local conc-12 in-phase probe and the cloud K=4 profile. Success criteria: batch-finalize B histogram
   moves from almost all `B=1` to lane-local `B>1`, and `queue_wait_ms` p95 collapses from about 100 ms toward one
   coalescing window plus one or two batched model-call spans.

4. If lane groups are still serialized materially, add concurrent per-lane finalize group dispatch while preserving
   pinned lane affinity and deterministic emit order.

5. Only consider a mixed-lane single `B=N` finalize call after the per-lane version is byte-exact and measured. It
   crosses a current lane invariant, so it should be a separate flag and canary.

6. Do not build the fused-kernel idea now. It optimizes about 0.1 ms of Python glue while the measured problem is
   about 100 ms of scheduler/model-call serialization.
