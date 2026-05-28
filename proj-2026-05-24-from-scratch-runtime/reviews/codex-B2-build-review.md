# Codex B2 Build Review

## Verdict

**PASS-with-followup.** I found no HOLD blocker in the B2 implementation against the binding spec in
`reviews/B2-design-paired-verdict.md` §II. The core correctness properties are present: explicit scheduler
injection, conditional construction, bidirectional CUDA stream handoff, sealed preloaded buckets, scratch
confinement, manifest fail-closed consumption, A1 outcome-B handling, and OFF-path production B=1 preservation.
The reported gate result has 0 token divergences and the OFF smoke stays byte-exact at its tested point.

The highest-priority flag is proof scope: `b2-t1` passed on 4 reference rows, not the full corpus. Treat a
full-corpus or fresh-process/split-case equivalent `b2-t1` as a **pre-B3 condition**, not a B2 commit blocker.

## Scheduler Correctness

- **Bidirectional CUDA sync is implemented.** The worker records `producer_event` on the worker stream after
  `chunk` and cache tensors are made contiguous (`runtime/cpp/density_main.cpp:1237-1254`). The dispatcher
  waits every producer event before packing (`runtime/cpp/batched_steady_scheduler.cpp:273-282`). Destroying
  the producer event immediately after `cudaStreamWaitEvent` is correctly placed: the wait dependency has
  already been enqueued on the dispatcher stream, and CUDA event destruction does not invalidate prior stream
  wait work.
- **Output-side sync is implemented.** The dispatcher unpacks into per-row `.contiguous()` tensors, records a
  completion event after those copies are queued, and returns that event (`runtime/cpp/batched_steady_scheduler.cpp:309-336`).
  The worker waits on the event before consuming row tensors and then destroys it (`runtime/cpp/density_main.cpp:1263-1271`).
- **Per-row completion events are correctness-equivalent to one event per batch.** All row events are recorded
  on the same dispatcher stream at the same point. This is slightly wasteful, not a correctness issue.
- **Fault handling is acceptable for runtime fail-fast.** The dispatcher loop is wrapped in `try/catch`, sets
  `fault_`, drains queued promises, logs, and exits (`runtime/cpp/batched_steady_scheduler.cpp:172-205`).
  Enqueue waits on `closing_ || fault_ || capacity` and rethrows fault/close before queueing
  (`runtime/cpp/batched_steady_scheduler.cpp:75-88`), so capacity-blocked producers fail fast. A worker timeout
  racing before dispatcher fault propagation is benign for correctness: the worker reports timeout and the
  dispatcher either exits or finishes/drains. The abandoned-future path can leak a raw completion event if a
  timed-out item later completes; that is an error-path cleanup issue, not a normal-path blocker.
- **No enqueue/dispatch mutex race found.** `enqueue()` and `gather_batch()` share `mutex_`; capacity wakeups
  happen after queue pops. The queue ownership model is coherent.
- **A4 sealed loader discipline is enforced.** `BatchedSteadyLoaderSet::preload_all()` seals the set, the
  scheduler constructor calls it and verifies sealed state, and `get()` throws if called before seal or for an
  unpreloaded bucket (`runtime/cpp/steady_batch_primitive.h:403-409`, `548-556`;
  `runtime/cpp/batched_steady_scheduler.cpp:28-43`).
- **A5 scratch confinement is correct.** `dispatch_batch()` and `warmup_buckets()` guard the dispatcher stream,
  scratch packing uses `index_copy_` on that stream, K=3 pad rows duplicate row 0
  (`runtime/cpp/batched_steady_scheduler.cpp:340-359`), and workers receive only per-row contiguous copies.

Telemetry caveat: the five timer buckets exist in memory (`gather_wait_us`, `service_wait_us`, `cuda_run_us`,
`output_sync_us`, `worker_blocked_us`) plus jitter, but the measurements are not fully faithful to §II.7.
`service_wait_us` stops before scratch pack, so it is not true pop-to-run-start; `output_sync_us` measures the
CPU cost of enqueueing `cudaStreamWaitEvent`, not actual device-side wait; and density JSON emits bucket
counts but not timer summaries, dispatcher CPU%, stream utilization, queue depth, or fairness spread. This is
a **pre-knee-remeasure telemetry follow-up**.

## Integration

- **No hidden global scheduler.** `run_steady_chunk_density()` takes a nullable
  `BatchedSteadyScheduler*` (`runtime/cpp/density_main.cpp:1173-1193`). Serial reference construction, B1 prep
  and replay, finalize parent prep, and warmup call it with `nullptr`
  (`runtime/cpp/density_main.cpp:1842-1870`, `2076-2095`, `2143-2163`, `4061-4080`, `4457-4476`).
- **OFF path does not instantiate the scheduler.** Density sweep constructs `BatchedSteadyLoaderSet` and
  `BatchedSteadyScheduler` only under `batch_steady_on`; the OFF branch asserts the local scheduler pointer is
  null (`runtime/cpp/density_main.cpp:4562-4590`). The production `enc_steady_aoti.pt2` loader is still the
  OFF-path steady encoder (`runtime/cpp/density_main.cpp:4558-4559`).
- **A1 outcome B is handled per spec.** The implementation compares production B=1 SHA to new B=1 SHA, runs a
  tensor parity fixture, throws on outcome C, and logs outcome A/B (`runtime/cpp/density_main.cpp:2352-2420`).
  Outcome B leaves OFF on production B=1; scheduler-ON owns the new-B1 risk through the B2 gate.
- **`b2-t1` covers the required shapes.** It includes single-stream scheduler-on, forced K=2/B=2, forced K=3
  padded to B=4, forced B=4, staggered, and scheduler-on `B_max=1` control
  (`runtime/cpp/density_main.cpp:2811-2914`). Bucket assertions prevent a false pass
  (`runtime/cpp/density_main.cpp:2716-2729`).

Test-hardening caveat: the forced barrier is before enqueue, not a stronger "all items are queued before the
dispatcher can pop" barrier (`runtime/cpp/density_main.cpp:1245-1254`). With default `lone_timeout_ms=0`, that
can theoretically race into B=1 if the dispatcher wakes on the first enqueue before peers enqueue. The bucket
assertions make that a caught failure rather than a false pass, and the reported run did form B>1/B4 batches.
For repeatability, either use a tiny forced-case lone timeout or add a stronger test-only enqueue gate.

## Primitive And Manifest

- **Manifest consumption is fail-closed.** Construction requires `MANIFEST.json`, parses buckets, verifies all
  B={1,2,4} entries exist, verifies package names, computes package SHA256, and verifies the supplied shared
  weights file SHA against every bucket entry (`runtime/cpp/steady_batch_primitive.h:486-524`). Missing files,
  malformed/missing keys, missing buckets, package mismatch, package SHA mismatch, or shared-weight SHA
  mismatch throw.
- **The C++ SHA256 and manifest parser are adequate for the emitted schema.** Empty/missing/malformed files
  fail closed. The parser is intentionally small and not a general JSON implementation, but it handles the
  generated manifest shape.
- **API separation is clean.** Existing B1 tests keep using `run()`/`torch::cat`; the scheduler uses
  `run_raw_prepacked()` plus `unpack_prepacked_outputs()` with its own scratch
  (`runtime/cpp/steady_batch_primitive.h:411-448`).
- **Exporter emits the manifest automatically after compile.** `export_steady_batched.py` writes per-B package
  SHA, ExportedProgram SHA, shared-weight SHA, torch/CUDA arch, inductor config, byte sizes, and A1 package
  parity metadata (`runtime/export_steady_batched.py:118-181`, `394-397`). `--manifest-only` is optional, not
  the only path. The Python and C++ SHA implementations both hash file content in 1 MiB chunks.

Minor A7 caveat: the C++ verifier consumes package SHA and shared-weight SHA but does not separately verify
`ep_sha256`. Since runtime executes the AOTI package, package SHA is the critical fail-closed check; explicit
EP verification would only strengthen provenance.

## OFF Path

The binding byte-exact contract is preserved structurally. When batch steady is OFF, the batched loader and
scheduler are never constructed, `run_steady_chunk_density()` receives `nullptr`, and the existing production
B=1 path runs. The reported 20-session N=4 OFF smoke (`NEMOTRON_DENSITY_BATCH_STEADY=0`) with mismatches=0 is
consistent with that trace.

## Memory And B3 Readiness

The scheduler adds a dispatcher stream, small scratch tensors, and preloaded B={1,2,4} AOTI packages. The
on-disk B packages are each about 2.4 GiB in this build, and the scheduler-on path also keeps production B=1,
shared constants, finalize loaders, and worker contexts alive. It is plausible that the 5090 knee shifts down
or that full-corpus `b2-t1` needs fresh-process/split-case execution to avoid allocator pressure. This should
be captured before the B3/knee remeasure.

## Follow-Ups

1. **Pre-B3:** run a full-corpus `b2-t1` equivalent, possibly split by case/fresh process or streamed
   reference reuse, because the current pass is 4 reference rows.
2. **Pre-knee-remeasure:** emit and fix the B2 timer telemetry summaries so §II.7 can support p95/p99
   service, sync, worker-blocked, stream-utilization, queue-depth, and fairness claims.
3. **Test hardening:** make forced-concurrency enqueue deterministic under `lone_timeout_ms=0` or document the
   bucket assertion as the intended anti-false-pass guard.
4. **Cleanup:** remove the unused `ep` parameter in `set_pending_exception_locked()` and consider replacing
   `cudaEventSynchronize(ev_stop)` with nonblocking timing plumbing if dispatcher CPU blocking shows up in
   the knee sweep.

## Net

No design-fork retreat and no correctness blocker found. B2 can fold as **PASS-with-followup** with the
full-corpus proof and telemetry hardening tracked before B3.
