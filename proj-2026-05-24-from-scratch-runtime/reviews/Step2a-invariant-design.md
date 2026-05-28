# Step 2a — invariant work design (admission + stale-gen + telemetry)

PHASE2-PLAN.md Step 2 was split (MF-6) into **2a invariant** (parallel-OK once 1b.5 starts; doesn't depend
on the scheduler topology decision) and **2b topology** (waited on the 1c-A selection — now made: central
dispatcher + B>1-batched per B2). This doc designs **Step 2a** — the box-global admission policy, the
stale-generation harness, the telemetry schema extensions, and the WS-tail microbench scaffolding skeleton.

**Out of scope here (Step 3):** the real WS server wrapping. Step 3 consumes Step 2a's primitives + adds
the actual `websockets` server.

## I. Admission policy

The plan calls for: "one box-global active/admitted cap + one box-global backlog-COUNT cap (ready-age is
dead; sweep ~8/10/12), shed = close (count admitted, not offered; two curves)."

### Concepts
- **Offered**: a stream connection arrival. Could be from a real WS client or the loadgen.
- **Admitted**: a stream the box accepted (passed admission). The session-level processing kicks in.
- **Active**: a stream currently being processed (admitted AND not yet finalized/closed).
- **Backlog**: streams waiting for admission (the gap between offered and active).

### Policy primitives

#### Box-global active cap (`NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP`)
- Hard cap on simultaneously active streams.
- Default: dynamically determined from the realized knee N (from B3); a static fallback (e.g., 40) for dev.
- When at-cap: new offers are SHED (closed immediately with a documented WS reject code, e.g., `1013 Try Again Later`).
- Telemetry: `admission.active_count`, `admission.active_cap_hits` (count of rejects due to cap).

#### Box-global backlog count cap (`NEMOTRON_DENSITY_ADMISSION_BACKLOG_CAP`)
- Cap on streams allowed to QUEUE for admission (separate from active cap).
- Default: sweep `{8, 10, 12}` in Step-4 measurement; pick the one minimizing tail-p95 latency.
- When at-cap: new offers are SHED with the same reject code.
- Telemetry: `admission.backlog_count`, `admission.backlog_cap_hits`.

#### Shed = close (not reject)
- The shed action is `socket.close(1013)` not a 503/HTTP-style reject. WS-native.
- Rationale: the client should treat shed as "try again later" not "permanent failure"; backoff + retry is
  the right behavior.
- Telemetry: emit per-shed log line with reason (active_cap_hit / backlog_cap_hit), so the offered-vs-admitted
  curve is reconstructable.

#### Two curves: admitted-throughput + offered-throughput
- Reported as separate JSON fields in the run summary: `admitted.{count, rps, throughput}` and
  `offered.{count, rps, throughput}`.
- The DENSITY metric is `admitted_streams_per_box` — never offered (which can be inflated by retry storms).
- The shed rate is `(offered - admitted) / offered`; the project bar is `Reject_bound ≤ 10% of offered`.

### Code surface

A new module `runtime/cpp/density_admission.{h,cpp}` (or extend a small section of density_main.cpp if
small enough):
- `class DensityAdmission` with: `active_cap`, `backlog_cap`, `try_admit(stream_id) -> AdmitResult {ADMITTED |
  SHED_ACTIVE_CAP | SHED_BACKLOG_CAP}`, `on_admit_complete(stream_id)`, `on_close(stream_id)`,
  `telemetry_snapshot() -> AdmissionTelemetry`.
- Thread-safe (the WS server in Step 3 will be multi-threaded).
- Lock-free counters where possible (`std::atomic` for active_count + backlog_count).
- Env-var + CLI configuration (`--admission-active-cap`, `--admission-backlog-cap`).

### Integration point (Step 2a contract)

This module is STANDALONE — it doesn't know about the scheduler, the session core, or the WS server. It just
manages 2 atomic counters + the try_admit decision. Step 3 wires it into the WS server's connection-accept
handler. b2-t1 / density-sweep mode can integrate via a thin shim that calls `try_admit` per session-start.

## II. Stale-generation suppression

The plan: "per-session generation tokens; close-while-inflight, reset-while-queued, reset-while-finalizer-
owns-runner, final-after-shed; 0 stale/mismatch."

### The bug class
A session may be CLOSED or RESET by the client mid-processing:
- **Close-while-inflight**: client closes the WS while a finalize is being computed → the finalize completes
  + produces a final → but the session is gone → the final is "stale" (no consumer; resource leak; possible
  observer confusion if logged).
- **Reset-while-queued**: a session reset request arrives while the session is queued behind another work
  item → the queued work should NOT produce output after the reset.
- **Reset-while-finalizer-owns-runner**: similar — the runner is computing finalize for an already-reset
  session → output should be suppressed.
- **Final-after-shed**: a session was shed mid-processing → no final should be emitted.

### Mechanism: per-session generation token

Each session has a `uint64_t generation` counter; every reset / close / shed bumps it. Every queued work item
records the `generation` at enqueue time. The consumer (the worker / dispatcher / finalize-runner) verifies
the session's CURRENT generation == the work item's RECORDED generation before processing/emitting; if
mismatch, drop the work silently (and log).

```cpp
struct SessionState {
  std::atomic<uint64_t> generation{0};
  // ...
};

struct WorkItem {
  uint64_t generation;
  // ... payload
};

// Producer enqueues:
work_item.generation = session.generation.load();
queue.push(std::move(work_item));

// Consumer dequeues:
if (work_item.generation != session.generation.load()) {
  metrics.stale_drops++;
  return; // silently drop
}
// ... process work_item
```

### Test cases (the "0 stale/mismatch" gate)
1. **close-while-inflight**: open session A; run 5 chunks; close A; verify no final emitted.
2. **reset-while-queued**: open session A; enqueue 3 chunks; reset A before any are processed; verify the 3
   chunks are silently dropped with stale_drops counter += 3.
3. **reset-while-finalizer-owns-runner**: open A; trigger finalize; reset A while the finalize is running;
   verify the finalize completes (the runner finishes its work) but the FINAL EVENT is not emitted (the
   downstream EVENT_FINAL emit checks generation).
4. **final-after-shed**: shed session A mid-processing; verify no final/interim events afterwards.

### Code surface
- Extend `SessionState` with `std::atomic<uint64_t> generation` (in density_main.cpp's existing struct).
- Extend `WorkItem`/`EnqueueRequest`/`FinalizeRequest` etc. with `generation` field.
- Consumer checks at: pre-encode, pre-decode, pre-event-emit, pre-finalize-output.
- Telemetry: `stale_drops_per_stage` (encode/decode/event/finalize).

### Integration point (Step 2a contract)
Standalone primitive — the generation counter + the check helper. The WS server (Step 3) bumps generation
on close/reset. b2-t1 mode can exercise this by simulating closes (the test cases above).

## III. Telemetry schema extensions

Build on F2-T's `scheduler_telemetry`. Additional buckets:

### `admission` block
```json
"admission": {
  "active_cap": 40,
  "backlog_cap": 12,
  "offered": 320,
  "admitted": 318,
  "active_peak": 38,
  "backlog_peak": 5,
  "active_cap_hits": 0,
  "backlog_cap_hits": 2,
  "shed_close_count": 2,
  "shed_close_rate": 0.00625
}
```

### `stale_gen` block
```json
"stale_gen": {
  "drops_at_encode": 0,
  "drops_at_decode": 0,
  "drops_at_event_emit": 0,
  "drops_at_finalize_output": 0,
  "total_drops": 0
}
```

### `ws_tail` block (Step 3 WS-tail microbench)
```json
"ws_tail": {
  "accept_to_ready_p50_us": ...,
  "send_to_recv_p50_us": ...,
  "recv_to_queue_p50_us": ...,
  "queue_to_scheduler_p50_us": ...,
  "serialize_and_send_p50_us": ...,
  "client_recv_p50_us": ...,
  "event_loop_lag_p50_us": ...
}
```
All with p50/p95/p99 + sample count `n`.

### Existing F2-T `scheduler_telemetry` block — no changes needed
The F2-T telemetry already covers the dispatcher-side timings; admission + stale-gen + ws_tail are
orthogonal additions.

## IV. WS-tail microbench scaffolding (skeleton)

The plan: "a WS-tail microbench (accept→ready, send→recv, recv→queue, queue→scheduler, serialize/send,
client-recv, event-loop lag under N idle + N streaming sockets; WS overhead p95 ≤10% of TTFS or decompose,
don't claim as runtime tail)".

### Skeleton (the Step 3 expansion will fill it)
A new mode `--mode ws-tail` in density_main.cpp (or a separate small binary `ws_tail_microbench`):
- Spin up a minimal echo-WS server with the admission shim.
- Drive N idle sockets + M streaming sockets from a loadgen client.
- Instrument each WS stage timestamp via `std::chrono::steady_clock` + log per-event JSON to a sidecar.
- Compute p50/p95/p99 per stage and emit the `ws_tail` telemetry block.

### Scope of Step 2a (skeleton-only)
- Define the JSON schema (this doc).
- Stub the binary entry point (`runtime/cpp/ws_tail_microbench.cpp` — just an empty main).
- Add the `ws_tail` JSON helper in density_main.cpp's telemetry emit (writes empty/null if not measured).

Real WS implementation = Step 3.

## V. Priority-finalize-lane integration with central dispatcher

The B2 central dispatcher serves all dispatched work serially through one dispatcher thread. The plan calls
for a priority-finalize-lane:
- Partitioned (`N_finalize_reserved ≥ 1`, steady-starvation p95 ≤ 2× no-finalize), OR
- Weighted (finalize runner-wait ≤25% TTFS, steady queue-wait ≤2× no-finalize).

### B2 reality check
In B2's current architecture, the scheduler dispatches **steady** chunks only. Finalize is per-stream (each
worker owns its finalize via the `FinalizeBucketLoaderPool`). So "priority-finalize-lane" doesn't directly
apply to the steady dispatcher — finalize is on a separate per-worker path.

### Possible Step 2 extension: a separate finalize dispatcher
If finalize-on-worker becomes a bottleneck (TBD by L40S sweep + production load testing), one could:
- Add a parallel finalize dispatcher (similar to BatchedSteadyScheduler but for finalize).
- Priority-route finalize through it before steady.
- This is **Step 2b or a Step-3 extension**, not Step 2a invariant work.

### Step 2a contract on priority-finalize
- Telemetry: track `finalize_wait_us` separately from `steady_wait_us` (already in F2-T) so the priority
  decision can be made empirically.
- Don't yet implement the lane; just expose the telemetry to inform the decision.

## VI. Bounded smoke tests for Step 2a

When the Step 2a code work lands:
1. **Admission unit test**: synthesize 100 offers; verify active_cap shed kicks in at the right point;
   verify backlog_cap shed; verify counters.
2. **Stale-gen unit test**: simulate the 4 test cases above; verify 0 stale events.
3. **Telemetry emit smoke**: run b2-t1 or density-sweep N=4 with the new modules; verify JSON sidecars have
   `admission` + `stale_gen` blocks.

No new corpus runs needed — the unit tests + integration with existing density-sweep is sufficient.

## VII. Sequencing + dependency

Step 2a invariant work can be implemented now (in parallel with Tier 3 memory shrink, since the files don't
overlap):
- New files: `runtime/cpp/density_admission.{h,cpp}`, `runtime/cpp/ws_tail_microbench.cpp` (stub).
- Modified: `runtime/cpp/density_main.cpp` (extend SessionState + WorkItem with generation; integrate
  admission shim into b2-t1 + density-sweep modes; extend telemetry emit).
- No conflict with Tier 3 (which touches scheduler + primitive); no conflict with the L40S sweep (different
  box).

After Step 2a lands + commits:
- Step 2b: priority-finalize-lane decision (informed by telemetry from real production-shape runs).
- Step 3: real WS server (consumes Step 2a primitives + adds the websockets server).

## Net

Step 2a is a **bounded, parallel-friendly invariant** chunk that can land while Tier 3 + L40S sweep are in
flight. The pieces are: admission (atomic counters + 2 caps + shed-close), stale-gen (per-session generation
tokens + check helper), telemetry extensions (admission + stale_gen + ws_tail blocks). Unit tests + a
density-sweep N=4 integration smoke. No new architecture, no corpus runs, no EC2.

Codex delegation prompt + paired review same pattern as Step 2/Step 3 nodes in PHASE2-PLAN.md (decision-
critical per plan → paired review before commit).
