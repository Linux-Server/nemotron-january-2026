# Codex B2 Design Review

## Verdict

**GO-with-changes-to-design.** Keep the central dispatcher as the default topology; do not pivot to
borrow-and-batch. But the current design is not build-ready as written. The two changes I would make before
implementation are:

1. Add an explicit **input-side CUDA dependency** to every enqueued item: record a ready event on the worker
   stream after `chunk`/cache tensors are produced, make the dispatcher stream wait on those events before
   pack/run, then record a completion event for the worker to wait on before decode. The design only states
   the output-side wait today.
2. Treat `K > B_max` as a queueing/service-time problem, not just "natural backpressure": replay and measure
   with `B_max=4` and actual B=4 service time. At N=64, the tail of a burst can wait behind many 7.7ms B=4
   forwards, so added wait can exceed the 10ms window even if the gather policy itself is correct.

## 1. Topology Fork

Central dispatcher is still the better first build. The strongest case for borrow-and-batch is real:
the arriving worker can run immediately when alone, avoids one extra CPU thread, uses the owner's stream for
at least the owner row, and spreads CPU-side pack/launch work across workers instead of forcing one dispatcher
thread to serialize it. It also makes the lone-arrival path naturally zero-wait if implemented that way.

I still would not pick borrow first. The hard part is not launching one batched forward; it is ownership of
other workers' GPU-visible tensors, futures, exceptions, cancellation, and stream dependencies. Borrow needs
multiple workers racing to claim queue entries without double-borrowing, then it must run another worker's
encoder on an arbitrary worker stream and scatter errors/results back to owners. That is exactly where
starvation and misattribution bugs hide. Central has one policy owner, one stream to profile, one place to
attach Step-2 admission/priority, and a cleaner failure model.

The central design is brittle in two places:

- **Lone-arrival latency:** a nonzero `lone_timeout` is measurable at low occupancy. This needs a scheduler
  ON / `B_max=1` control and a `lone_timeout=0` control, not just the claim that 1ms is effectively zero.
- **Single-thread and single-stream service:** CPU single-thread bottleneck is plausible but not my main fear.
  The bigger issue is the single dispatcher CUDA stream becoming the serialized service station for all steady
  chunks. With B=4 p50 total around 7.69ms, N=64 implies about 400 steady chunks/s, or about 100 B=4 batches/s
  if perfectly filled. That is already roughly 769ms of dispatcher-stream service per second before packing,
  decode bursts, jitter, and under-filled batches. Profile dispatcher CPU, but also profile dispatcher-stream
  utilization and enqueue-to-output latency.

Instrumentation is clearly easier centrally: actual batch-size distribution, pad count, enqueue->dequeue,
enqueue->run-start, run CUDA ms, enqueue->promise, per-stream p95/p99, dispatcher wake jitter, CPU%, and
stream sync wait all live in one object. Borrow makes those labels depend on whichever worker happened to
borrow.

## 2. Policy Values

`B_max=4` is the right initial cap because the B1 primitive only has buckets {1,2,4} and the measured per-row
gain is strongest at B=4. But the opportunity data being cited is not sufficient as-is for the runtime policy:
the published table has mean B values above 4 at 12ms, so it cannot be a direct `B_max=4` formed-batch
distribution. Re-run or report the replay with `B_max=4`, including B=1%, B=2%, B=3-padded-to-4%, B=4%, and
the queued-rest latency.

`window=10ms` is a reasonable starting point, not a settled default. The kill-gate used 8/12ms and the plan
originally mentioned added-wait p95 <= 8ms; the B2 doc relaxes this to p95 <= window. Make the window
configurable in 1ms increments at minimum and sweep at least `{0, 4, 8, 10, 12}` for B2. If the B=4 cap is
usually saturated by 8ms at N>=44, 10-12ms may only add lag.

`lone_timeout=1ms` is also a hypothesis. `0ms` protects low-load latency and is the right diagnostic control.
`1ms` may be better at high N because the first arrival in a cluster should not instantly force B=1. Keep it
configurable and report its actual effect on B=1 rate and p95 enqueue->run-start.

## 3. Integration Model

The replacement point is correct only for continuation steady chunks. In `density_main.cpp`,
`run_steady_chunk_density` uses `enc_first` for `expected_first` and calls `run_steady_encoder_stream` only
in the continuation branch. That is the right branch to replace. `decode_range_density` and
`run_finalize_density` should remain per-stream.

The integration design is missing a required input-side stream dependency. Today the proposed sequence is:

`enqueue + future.get() + wait dispatcher event on worker stream + consume`.

That handles output visibility, but not input visibility. In the current worker path, `chunk =
torch::cat({state.ring, new_mel}, 2)` and previous cache clones are produced on the worker CUDA stream. CUDA
work is asynchronous. If a dispatcher stream packs those tensors before the worker stream has completed the
cat/cache-producing work, the dispatcher can read incomplete data. Each queue item should carry a ready event
recorded on the producer stream after the input tensors are prepared; the dispatcher stream must wait on all
ready events before `index_copy_`/AOTI run. Borrow has the same problem for borrowed rows, but central has it
for every row.

Do not wire the scheduler as a hidden global inside all `run_steady_chunk_density` calls. That helper is used
by serial reference construction, existing correctness modes, B1 prep/replay, finalize parent prep, warmup,
and the density sweep. The build should pass an optional scheduler pointer or policy object so only the B2
batched side and the measured scheduler-on sweep route through it. `build_serial_reference`, B1 T1, and
finalize-specific gates must keep using production B=1 unless explicitly testing scheduler behavior.

Warmup also needs a design decision: either warm batched buckets by calling the primitive/scheduler directly
outside the measured gate, or route a controlled warmup through the scheduler and exclude it from telemetry.
Do not let warmup accidentally form the only B=4 batches in `b2-t1` while the measured phase short-circuits.

## 4. Bucket Selection And `K > B_max`

Leaving queued-rest for the next dispatcher iteration is acceptable only if B2 measures service wait separately
from gather wait. FIFO with `B_max=4` can create a tail even when the window is 10ms. In a synchronized or
partially synchronized N=64 cycle, 16 B=4 forwards at about 7.7ms each means the last group can wait well over
100ms behind earlier groups. The real staggered workload should be better, but this is exactly the failure mode
the design must prove away.

Queue drain order matters. Pure FIFO maximizes age fairness but can repeatedly put a stream behind many older
items when bursts happen. B2 does not need Step-2 priority lanes yet, but it should report per-stream
enqueue->run-start p95/p99 and max, plus a fairness spread. If tail queue wait exceeds the window materially,
do not defer the fix to Step 2; the batching mechanism itself is not meeting the B2 SLO contract.

## 5. `b2-t1` Gate

The comparison structure is directionally right: single-stream scheduler-on versus production B=1, and
multi-stream scheduler-on versus per-stream B=1. But the gate must assert it actually exercised batching.

Required additions:

- Record and assert bucket counts: B=1, B=2, K=3->B=4 padded, K=4, and preferably `K>B_max` backlog.
- Add a deterministic forced-concurrency case with a barrier around continuation chunks so the dispatcher
  forms B=4 batches. A natural multi-stream run can accidentally pass while mostly short-circuiting to B=1.
- Keep a production-shaped staggered case too, because the deterministic barrier overstates burstiness.
- Compare both cumulative steady tokens and final tokens, with **0 token divergences** fatal. Event drift
  should be counted under `DENSITY_GOLD_EVENTS_TOLERANT`, not used as the process exit gate when tolerant mode
  is requested.
- Include a scheduler-on `B_max=1` control to isolate scheduler/future/stream-sync overhead and NEW B=1
  package behavior from actual B>1 batching.

## 6. A1-A8 Fold-In

**A1:** I disagree with "use NEW B=1 in the flag-OFF path too" as the clean fix. The binding contract says
`NEMOTRON_DENSITY_BATCH_STEADY=0` is byte-exact and unchanged. Replacing production `enc_steady_aoti.pt2` on
the OFF path violates that unless the packages are proven bit-identical and behavior-identical first. Do the
hash/tensor/token parity check. If NEW B=1 differs, keep OFF on production B=1 and make scheduler-ON explicitly
own the risk with a full B=1-vs-production T1; do not hide it by moving the production reference.

**A2:** Correct to add all-chunks-batched T1. Strengthen it with forced batch formation and bucket-count
assertions as above.

**A3:** Margin probe is useful and should stay debug-flagged. It should log only around decode decisions where
the top-2 gap is small or where a replay mismatch is detected; full logging on every greedy step can perturb
timing and produce huge logs.

**A4:** Preload discipline is acceptable if enforced mechanically. `BatchedSteadyLoaderSet::get()` is currently
lazy and not mutex-protected. B2 should either add a mutex like `FinalizeBucketLoaderPool`, or make
construction call `preload_all()`, mark the set sealed, and make `run()` fail closed if any requested bucket
was not preloaded. Do not rely on a comment that no lazy load will happen later.

**A5:** Preallocated scratch plus `index_copy_` is the right performance move. Correctness constraints:
run under `torch::NoGradGuard`/inference, keep all scratch reuse on the dispatcher stream, do not return views
into scratch or batched outputs, and record the completion event after unpack `.contiguous()` copies are
enqueued. Autograd is not relevant here, but async reuse and aliasing are.

**A6:** Wrapper placement is right. The current B1 code computes `ok` with `event_divergences == 0`, so the
wrapper must map strict event differences into counted-not-gated policy without watering down token/cache-len
failures. Apply the same split to B2 from the start.

**A7:** Emit the steady-batch manifest in B2, but also consume it fail-closed in the C++ loader. A manifest that
is generated but not verified does not buy the finalize bucket discipline. Include package SHA256 per B,
ExportedProgram SHA, shared-weight SHA, torch/CUDA/arch, compile config, byte sizes, and the A1 B=1 parity
result.

**A8:** Moving shared filesystem/AOTI/helper declarations into `runtime_io.h` is the right organization once
`batched_steady_scheduler.cpp` exists. This should be mechanical and should not change runtime behavior.

## 7. Missing Risks

Add these to the register:

- **Input stream race:** dispatcher reads tensors produced on worker streams without waiting on producer
  events.
- **Shutdown/cancellation deadlock:** workers can block in `enqueue()` or `future.get()` if the dispatcher
  throws, exits, or is destroyed. The scheduler needs `close()`, draining, and exception propagation for both
  queued and in-flight items.
- **Service-time backlog:** enqueue->output wait can exceed the gather window when many full B=4 batches are
  queued.
- **B=1 package drift:** NEW B=1 may differ from production B=1; scheduler ON can drift even with no real
  batching.
- **Control-path contamination:** serial reference, B1 T1, finalize prep, and warmup can accidentally route
  through the scheduler if the flag is checked too low in `run_steady_chunk_density`.
- **T1 false pass by short-circuit:** multi-stream T1 can pass while forming few or no B>1 batches.
- **Finalizer/resource coupling:** batching may change decode burst timing and finalize pressure even if TTFS
  is measured from `vad_stop`. Keep reporting finalize wait/total and lag together.
- **Telemetry ambiguity:** "batch wait" must be split into gather wait, service wait, CUDA run time, output
  sync wait, and worker blocked time.

## 8. 5090 Knee Re-Measure

The B_max sweep `{1,2,4}` is necessary but not sufficient. Add a window sweep for at least B_max=4:
`window_ms in {0,4,8,10,12}` and `lone_timeout_ms in {0,1}` unless runtime is too expensive, in which case
do `{0,8,10,12}` plus lone `{0,1}` at the chosen window.

Also add a true baseline split:

- flag OFF production B=1: existing behavior and byte-exact baseline.
- scheduler ON, `B_max=1`: scheduler/future/stream-sync/NEW-B1 overhead control.
- scheduler ON, `B_max=2/4`: real batching.

N values `{40,48,56,64}` are fine for a first sanity sweep, but they are coarse for locating the knee. Add
`N=44` if possible because the expected lift boundary starts just above the current N=40 5090 knee. Every row
should report actual batch distribution, p95/p99 enqueue->run-start, p95/p99 enqueue->promise, dispatcher CPU,
dispatcher-stream CUDA ms/utilization, sync wait, and fairness spread across streams.

## 9. Build Sketch

The file layout is sensible:

- `runtime/cpp/batched_steady_scheduler.h/.cpp` for queue/thread/policy/telemetry.
- `runtime/cpp/steady_batch_primitive.h` remains the primitive.
- `density_main.cpp` gets an optional scheduler integration point and `--mode b2-t1`.

Prefer a small scheduler API over exposing futures everywhere:

- `enqueue(input, producer_stream, label)` records or accepts a producer-ready event.
- returned output includes row tensors, bucket/K metadata, and a completion event.
- scheduler owns lifecycle: `start`, `close`, destructor drain, exception broadcast.

Avoid making env parsing the only configuration path. Tests and sweeps need explicit args for B_max/window/lone
timeout so telemetry always records the effective policy.

## Net

Central dispatcher: **GO**. Borrow-and-batch: keep as fallback only if central's measured service/future/sync
overhead is the bottleneck. Required design changes before build: input-side stream waits, explicit scheduler
injection so reference/finalize prep paths stay pure B=1, actual-batch assertions in `b2-t1`, B_max=4 queue
service telemetry, configurable/swept window and lone timeout, fail-closed preload/manifest discipline, and no
NEW-B1 replacement of the flag-OFF path unless byte-exact identity is proven first.
