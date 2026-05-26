# Phase-2 paired review - Round 3 (Codex)

Scope: downstream Steps 2-5 of `PHASE2-PLAN.md`, plus the load balancer/loadgen/WER infrastructure and the
production admission/priority-lane/shed code. Foreground goals: **G1 = utilization / streams-box density** and
**G2 = P50/P95 TTFT spread**.

## BLOCKER

### B1 - Step 2 collapses two production controls into one vague phrase.

`PHASE2-PLAN.md` says Step 2 should design "the finalize priority lane, admission/backlog-cap shedding" from the
Step-1 knee (`PHASE2-PLAN.md:33-35`). That is not enough to be faithful to production when the architecture changes
from K Python processes behind an LB to one native process with threads.

Production has two distinct gates:
- External active-connection distribution/capping: local LB chooses least-loaded eligible backend and respects
  per-backend `maxconn` (`ec2-bench/local_lb.py:19-27`), then sheds if every backend is at maxconn
  (`ec2-bench/local_lb.py:44-48`). Deployment docs explicitly set HAProxy `maxconn` per process at the SLO-robust
  point (`deploy/DEPLOYMENT.md:39-42`).
- Server-local backlog admission: the server computes queued per-session events plus ready sessions
  (`src/nemotron_speech/server.py:4974-5003`), rejects if `backlog_count > NEMOTRON_ADMISSION_MAX_BACKLOG`
  (`src/nemotron_speech/server.py:5007-5015`), and closes the WebSocket with 1013 on rejection
  (`src/nemotron_speech/server.py:5043-5063`). Deployment says backlog-count 8-12 is the signal and ready-age did
  not track overload (`deploy/DEPLOYMENT.md:43-48`).

The native one-process server has no external per-backend `maxconn` unless Step 2 recreates it internally. A backlog-only
cap can still admit too many active connections before the backlog signal rises; an active-session-only cap can miss the
intake queue cliff. Either mistake can inflate G2 tail while making G1 "streams/box" ambiguous.

Recommended plan edit:
- Step 2 must specify both a **single box-global active/admitted-session cap** replacing LB `maxconn` and a **single
  box-global backlog-count cap** replacing the per-process `NEMOTRON_ADMISSION_MAX_BACKLOG`.
- It must report offered streams, admitted streams, rejected streams, close-code counts, queue depth, and SLO-robust
  admitted streams/box. Do not count rejected connections as capacity.

### B2 - Step 2 is not explicitly blocked on Step-1 telemetry, so it can design from the wrong "knee."

Round 1 required Step 1 to emit scheduler telemetry, not just a knee (`reviews/phase2-round1-FOLDED.md:48-50`).
Round 2 refined that to queue wait, AOTI/runner wait, `.item()` wait, finalize wait, and shed counters
(`reviews/phase2-round2-FOLDED.md:103-106`). The current Step 2 still says only "From Step 1's knee"
(`PHASE2-PLAN.md:33-35`).

A scalar knee does not tell the scheduler whether the limiting queue is active sessions, steady-runner wait, finalize
runner wait, WS intake, CPU core saturation, or admission shed. That is a direct G1/G2 failure mode: a scheduler tuned to
the wrong bottleneck can preserve Step-1 throughput in a synthetic harness but widen P50/P95 in the real server.

Recommended plan edit:
- Make Step 2 acceptance conditional on a Step-1 telemetry schema with at least: active sessions, offered/admitted/rejected,
  ready count, event queue depth, runner wait by phase, steady/finalize service time, `.item()` wait, CPU-core util,
  WS recv/send overhead, and P50/P95/P99/P95-P50 for server-side TTFT/TTFS.
- The scheduler design review should reject any design that cites only a knee.

### B3 - The finalize priority lane is not mapped onto the native `num_runners` pool.

Production's shipped finalize priority is lane-aware, but it is not a magic "priority lane." It stages finalize events
before ready work (`src/nemotron_speech/server.py:5334-5444`), excludes lanes with pending finalizes from steady ready
selection (`src/nemotron_speech/server.py:5340-5354`, `src/nemotron_speech/server.py:5945-5979`), and then runs finalizes
on the session's pinned lane (`src/nemotron_speech/server.py:7788-7840`, `src/nemotron_speech/server.py:7858-7877`).
For non-parallel keys, the Python scheduler reserves one lane and marks the model path exclusive
(`src/nemotron_speech/server.py:4185-4213`); tests confirm only steady same-drop continuation batches can run in parallel,
while first/exclusive work blocks other lanes (`tests/test_scheduler_model_lanes.py:52-84`).

The native plan currently names "the finalize priority lane" (`PHASE2-PLAN.md:33-35`) but does not say whether a native
finalize consumes a dedicated runner, a reserved subset of the shared `num_runners=N` pool, or the same queue as steady.
If finalizes can take every runner, steady starves and G2 spreads. If steady can occupy every runner, final P95 spreads.
Either can turn a valid Step-1 density ceiling into a bad Step-4 realized number.

Recommended plan edit:
- Step 2 must define the runner partition/policy: e.g. `N_steady + N_finalize_reserved`, or weighted priority with a
  hard bound on finalize wait and steady starvation.
- Require stress tables for coincident finals at the Step-1b knee: steady queue wait, finalize queue wait, P50/P95/P99,
  and utilization. The hot-bucket stress from Rounds 1-2 is necessary but not sufficient; this is scheduler policy.

### B4 - Step 3 can add its own WS tail, and the plan has no control experiment for it.

Step 3 says to wrap the session core in the scheduler plus a real WS server and then run N concurrent streams with correct
events/finals (`PHASE2-PLAN.md:36-38`). The production server path has nontrivial WS and event-loop work: aiohttp prepares
the WebSocket, initializes a session under the model path, sends `ready`, then parses every WS message and queues it
(`src/nemotron_speech/server.py:5028-5095`, `src/nemotron_speech/server.py:5247-5272`). Sends are JSON serialization plus
`websocket.send_str()` (`src/nemotron_speech/server.py:7033-7055`). The scheduler loop already needed an explicit
cooperative yield to avoid starving socket I/O under WAN timing (`src/nemotron_speech/server.py:5537-5547`).

That means WS overhead is not a rounding error; it can be the very G2 tail Step 4 attributes to the runtime. Existing
telemetry partially captures `vad_stop_recv` and final send/receive timestamps (`src/nemotron_speech/server.py:2868-2884`,
`src/nemotron_speech/server.py:3178-3194`, `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py:517-535`), but Step 3
does not require a loopback echo/WS-overhead characterization.

Recommended plan edit:
- Step 3 must include a WS-tail microbench before Step 4: accept-to-ready, client send-to-server receive, receive-to-queue,
  queue-to-scheduler, final JSON serialization/send, client receive, and event-loop lag under N idle and N streaming sockets.
- Step 4 must subtract/report WS-tail separately from model/scheduler tail, or the G2 conclusion is confounded.

### B5 - Stale-generation suppression must be a Step-3 gate, not an implied part of "correct events."

Round 1 noted native session correctness excludes stale-generation suppression and prints it as deferred
(`reviews/phase2-round1-FOLDED.md:51-55`). Production has concrete generation checks for scheduler chunk outputs
(`src/nemotron_speech/server.py:6297-6323`, `src/nemotron_speech/server.py:6440-6445`) and finalization before/after model
work (`src/nemotron_speech/server.py:8797-8815`, `src/nemotron_speech/server.py:8941-8959`). Step 3 only says "correct
per-stream events/finals" (`PHASE2-PLAN.md:36-38`).

Under overload, stale finals can be dropped, duplicated, or misordered. That can make G2 look better by suppressing slow
finals, or make semantic WER look better/worse by joining the wrong final text. This must be explicit before Step 4.

Recommended plan edit:
- Step 3 gate: per-session generation tokens; stale interim/final suppression; tests for close while in-flight, reset while
  queued, reset while a finalizer owns a runner, and final output after admission shed.
- Require token/event equality against the serial oracle and client-observed final ordering under concurrent load.

### B6 - Step 4 says "same harness" but not the full apples-to-apples contract.

The plan's Step 4 names same harness, same semantic-WER tool, and same hardware (`PHASE2-PLAN.md:39-41`). The files make
the contract stricter:
- The loadgen fixes 16 kHz, 20 ms chunks, 200 ms trail, start jitter, optional WAN-like stream jitter, and 500 ms p95
  proc-lag keepup (`ec2-bench/ec2_loadgen.py:20-27`, `ec2-bench/ec2_loadgen.py:86-105`).
- It reports TTFS p50/p95, lag p50/p95, overrun, keepup, and defines the knee as max N with p95 proc-lag under 500 ms
  and zero errors (`ec2-bench/ec2_loadgen.py:128-170`).
- The stt-benchmark service uses one WS connection per sample and treats the final reset transcript as the final frame
  for TTFS (`stt-benchmark/src/stt_benchmark/nemotron_local_stt.py:1-25`, `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py:423-457`).
- Semantic WER is not just "a tool"; it pins a model default, prompt, retry behavior, trace storage, and pooled WER logic
  (`stt-benchmark/src/stt_benchmark/evaluation/semantic_wer.py:356-376`,
  `stt-benchmark/src/stt_benchmark/evaluation/semantic_wer.py:799-893`,
  `stt-benchmark/src/stt_benchmark/evaluation/semantic_wer.py:895-938`).

Step 4 also compares to "~16-20/L40S, ~6/L4" (`PHASE2-PLAN.md:39-41`), but Round 3's brief correctly says the baseline
must be freshly re-measured. If Step 4 compares native to stale Python numbers or changes `STREAM_JITTER_MS`, rounds,
audio corpus, WER model, close-code handling, or SLO, the G1/G2 verdict is not apples-to-apples.

Recommended plan edit:
- Pre-register an apples-to-apples manifest: commit SHA, artifact SHA, audio corpus manifest, loadgen script/args/env
  (`LOADGEN_JITTER_MS`, `LOADGEN_STREAM_JITTER_MS`, `LOADGEN_ALL_CLIPS`, rounds, sweep), hardware/driver, server flags,
  admission caps, semantic-WER model/version, and DB/tool version.
- Step 4 must run Python baseline and native back-to-back on the same hardware/date, not compare to the old 16-20/6.

## MAJOR

### M1 - Step 1b is a ceiling; Step 4 is the realized G1/G2 number.

Round 2 already warned that Step 1 has placeholder scheduling and Step 4 has the binding G2 tail
(`reviews/phase2-round2-FOLDED.md:82-89`). The current plan still lets Step 1 read as "the hard gate" and Step 4 as
confirmation (`PHASE2-PLAN.md:16-32`, `PHASE2-PLAN.md:39-41`). After the Step 2/3 findings above, that wording is unsafe.

Recommended plan edit:
- Label Step 1b "native compute ceiling under controlled scheduler placeholder."
- Label Step 4 "realized end-to-end density/tail with production scheduler + WS + admission." A GO requires both:
  Step 1b says the compute ceiling exists, and Step 4 realizes enough of it without widening G2.

### M2 - The current loadgen does not report P99 or P95-P50, so it cannot satisfy the corrected G2 gate as-is.

Rounds 1-2 require TTFT/TTFS p50/p95/p99 and P95-P50 (`reviews/phase2-round1-FOLDED.md:38-41`,
`reviews/phase2-round2-FOLDED.md:103-106`). `ec2_loadgen.py` reports p50/p95 for TTFS and proc-lag but no p99 and no
spread field (`ec2-bench/ec2_loadgen.py:128-139`, `ec2-bench/ec2_loadgen.py:156-170`). The stt-benchmark reports can
show p95/p99 in some paths, but Step 4 does not require those exact outputs.

Recommended plan edit:
- Extend the Step-4 result schema to include TTFT/TTFS p50/p95/p99, P95-P50, P99-P50, proc-lag p50/p95/p99, offered vs
  admitted vs rejected, and semantic-WER at the same N.

### M3 - Admission shed semantics can make the "knee = zero errors" rule undercount or mislabel capacity.

`ec2_loadgen.py` sets `keepup` from lag p95 and the printed knee requires zero errors (`ec2-bench/ec2_loadgen.py:133-139`,
`ec2-bench/ec2_loadgen.py:168-170`). Production admission intentionally rejects overload with WS close 1013
(`src/nemotron_speech/server.py:5047-5063`), and deployment says the LB must drain on 1013 (`deploy/DEPLOYMENT.md:43-48`).
If the native server correctly sheds, the current zero-errors knee can mark a healthy admitted-load regime as failure
unless Step 4 distinguishes rejected from admitted sessions.

Recommended plan edit:
- Report two curves: no-shed SLO-robust knee and admitted-through-shed SLO-robust capacity.
- For GO/STOP, compare admitted successful streams/box at target SLO and a bounded rejection rate, not just offered N
  with zero client errors.

### M4 - Step 5 should be confirmation of Step-1 resource attribution, not an independent rediscovery.

Step 5 says "Measure native density on each target; test the hypothesis" (`PHASE2-PLAN.md:42-44`). Round 2 downgraded
"BW-bound" to a hypothesis requiring counters, not NVML averages (`reviews/phase2-round2-FOLDED.md:53-57`). Therefore
Step 5 should not be a blind sweep. Step 1 should predict each target's behavior from resource attribution, and Step 5
should confirm or falsify that prediction.

Recommended plan edit:
- Step 5 table must include Step-1 predicted binding resource, measured counters on target, predicted density, measured
  realized Step-4 density, and explanation for any miss.
- State purpose explicitly: confirm L40S deploy number, confirm/falsify L4 negative, and explore Spark risk. Do not reopen
  the fleet decision without a counter-based mismatch.

### M5 - Spark aarch64 risk is broader than "libtorch maturity" and must include `num_runners` behavior.

The plan mentions Spark aarch64 only as "libtorch maturity risk" (`PHASE2-PLAN.md:42-44`). The local libtorch AOTI model
container has an aarch64-specific branch in runner reclamation (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/model_container.h:718-731`).
That does not prove a bug, but it means Step 5 must not assume the x86_64 L40S `num_runners` conclusion carries over.

Recommended plan edit:
- Add a Spark preflight: build/load AOTI packages, run `num_runners=N` overlap/concurrent==serial/memory-flat microgates,
  verify explicit stream behavior, then run the full Step-4 harness only if the primitive passes.

## MINOR

### m1 - `local_lb.py` is a density stand-in, not an overload-faithful HAProxy clone.

The file says local LB closes when all backends are full, while HAProxy would queue (`ec2-bench/local_lb.py:10-13`,
`ec2-bench/local_lb.py:44-48`). That is acceptable for density sweeps but not for overload/G2 shed behavior. Step 4
should use the production LB/ALB behavior for the Python baseline or explicitly label local-LB overload results as
approximate.

### m2 - The semantic-WER evaluator is nondeterministic infrastructure unless pinned and cached.

The evaluator depends on an external Claude model default (`stt-benchmark/src/stt_benchmark/evaluation/semantic_wer.py:362-376`)
and warms an ephemeral prompt cache (`stt-benchmark/src/stt_benchmark/evaluation/semantic_wer.py:378-400`). Same "tool"
must mean same code SHA, same model name, same prompt, same retry/timeout settings, and stored traces.

### m3 - Production sync-compression is part of the apples-to-apples flag surface.

The launcher ships `NEMOTRON_SYNC_COMPRESS` and `NEMOTRON_FINALIZE_PRIORITY` on by default (`deploy/launch_multiproc.sh:45-55`),
and the server logs sync-compress mode at startup (`src/nemotron_speech/server.py:743-749`). Step 4 must declare whether
the Python baseline and native run both include equivalent sync-compression behavior.

## QUESTIONS

1. What is the native admission unit: active WS connections, active sessions that have sent `vad_start`, sessions with
   nonempty audio backlog, or AOTI runner inflight jobs?
2. Does the native scheduler reserve at least one runner for finalize, or does it rely on priority ordering over a shared
   runner pool?
3. Will Step 4 use `ec2_loadgen.py`, stt-benchmark's full pipeline, or both? The plan names stt-benchmark, while the
   Round-3 brief calls out `ec2_loadgen.py`; the result schema must reconcile them.
4. What semantic-WER threshold is required at the density knee? Step 4 names the tool but not the pass/fail WER bound.
5. Is G2 measured as TTFT, TTFS, or both? The current loadgen measures finalize TTFS and proc-lag; the project language
   often says TTFT.

## Does the corrected plan measure G1 and G2 end-to-end?

Not yet.

After Rounds 1-2, a corrected Step 1 can measure the **G1 compute ceiling** and a **reference server-side tail**. But
Steps 2-4 still need the edits above before the plan measures end-to-end G1/G2 rather than a ceiling plus a confounded
real-server run. The remaining ways a GO/STOP can be wrong are:

- G1 false GO: Step 1b shows compute overlap, but Step 2 admission/priority policy or Step 3 WS server loses enough
  density that Step 4 cannot realize it.
- G1 false STOP: Step 4 counts intentional admission rejects as errors and undercounts admitted sustainable streams.
- G2 false GO: stale finals, shed behavior, or WS-tail confounds suppress/shift slow finals, hiding P95/P99 spread.
- G2 false STOP: WS implementation overhead dominates but is attributed to scheduler/model runtime.

With explicit box-global admission, runner-pool priority policy, WS-tail characterization, fresh same-harness baselines,
and Step-5 resource-attribution confirmation, the plan would measure G1 and G2 end-to-end. As written, it does not.
