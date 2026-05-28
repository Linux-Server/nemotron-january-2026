# Step 3b — production WS server architecture (design)

Goal: build the production WebSocket server for the native runtime. The user's mental model is the
right one: **(1) core library** that everything in `runtime/cpp/` shares + **(2) a production WS+HTTP
application** that imports the library and adds protocol/lifecycle/observability.

Status: written 2026-05-28, while L40S B3-FU-1 + B3-FU-2 sweeps run in parallel. This doc precedes the
CMakeLists refactor + WS skeleton implementation (delegated separately to Codex).

## I. Current state — the starting point

`runtime/cpp/` today is **a set of monolithic-compile binaries**, not a library + clients. Crucial
finding: `density_main.cpp:1` literally does `#include "session_main.cpp"` (Phase-1 pattern: one
translation unit per binary). Every binary re-compiles the full ~5000-line `session_main.cpp` from
source. There is no `libnemotron_runtime.a`.

```
runtime/cpp/
├── session_main.cpp     4668 lines   (session core: preproc, encoder, decode, finalize, emit)
├── density_main.cpp     6213 lines   (density harness; #includes session_main.cpp)
├── batched_steady_scheduler.{h,cpp}  202+524 lines  (B2 scheduler)
├── steady_batch_primitive.h          663 lines      (B1 primitive + manifest)
├── density_admission.{h,cpp}         77+161 lines   (Step 2a admission)
├── ws_tail_microbench.cpp            851 lines      (raw RFC6455 standalone)
├── ws_tail_microbench_client.cpp     472 lines      (loadgen client)
├── decode_main.cpp / pipeline_main.cpp / steady_main.cpp / aoti_encoder_main.cpp / finalize_main.cpp /
│   steady_batch_bench.cpp            (Phase-1 piece-wise binaries; smaller)
└── CMakeLists.txt
```

The WS server needs a real library boundary. We'll do a **bounded refactor** (carve the library, migrate
existing binaries to link it, smoke that nothing breaks) BEFORE the WS server implementation.

## II. Target architecture

### Binary layout
```
runtime/cpp/
├── lib/
│   ├── session/           preproc, encoder, decode, finalize, emit
│   ├── scheduler/         batched_steady_scheduler + steady_batch_primitive
│   ├── admission/         density_admission
│   ├── telemetry/         SessionTiming + StatsCollector + scheduler telemetry
│   └── runtime_io/        manifest, AOTI loader helpers, SHA256, JSON parser (moved out of primitive.h)
│   → produces libnemotron_runtime.a (static link)
├── ws_server.cpp          NEW — production WS+HTTP server (links libnemotron_runtime)
├── density_main.cpp       MIGRATED — links libnemotron_runtime instead of #include
├── ws_tail_microbench.cpp UNCHANGED — standalone characterization tool (no library dep)
├── ws_tail_microbench_client.cpp  UNCHANGED
└── CMakeLists.txt         + add_library(nemotron_runtime STATIC ...); existing binaries → target_link
```

The Phase-1 piece-wise binaries (decode/pipeline/steady/aoti_encoder/finalize/steady_batch_bench) can
stay as-is — they're small standalone tools for diagnostic / one-piece work; not worth migrating in this
sprint.

### Library API surface — what `libnemotron_runtime` exposes

Public headers (under `lib/`, each with stable C++ API):
- `lib/session/session.h` — `SessionState`, `SessionMode`, `reset_session`, `run_steady_chunk_density`,
  `run_finalize_density`, `EmittedEvent`, `Tokenizer`, etc. (carved from `session_main.cpp`).
- `lib/scheduler/scheduler.h` — `BatchedSteadyScheduler`, `BatchedSteadyLoaderSet`,
  `BatchedSteadyInput`, `DispatchResult`, `BatchedSteadySchedulerPolicy` (already a header-mostly module
  — minimal change).
- `lib/admission/admission.h` — `DensityAdmission`, `AdmissionDecision`, `AdmissionTelemetry` (already
  a clean module — minimal change).
- `lib/telemetry/session_timing.h` — `SessionTiming` struct (NEW; the C++ analog of Python's
  per-finalize timing dict).
- `lib/telemetry/stats_collector.h` — `StatsCollector` (NEW; the /stats engine).
- `lib/runtime_io/io.h` — `file_exists`, `directory_exists`, `sha256_file`, the JSON parser (moved out
  of `steady_batch_primitive.h` so it can be reused).

Internal (implementation .cpp files):
- The current `session_main.cpp` body is carved into several `.cpp` files matching the lib/ layout, OR
  stays as one big `session.cpp` initially (less invasive — incremental cleanup later).

### Telemetry: SessionTiming + StatsCollector — the /stats engine

The Python /stats PR (279f033) records 5 per-finalize metrics. C++ equivalent:

```cpp
// lib/telemetry/session_timing.h
struct SessionTiming {
  // Required (the 5 metrics from /stats):
  std::optional<double> vad_stop_to_sent_ms;            // server-side TTFS — the SLO signal
  std::optional<double> fork_flush_wall_ms;             // encoder-finalize wall
  std::optional<double> vad_stop_recv_to_process_ms;    // intake-backlog / load-tail
  std::optional<double> lock_wait_ms;                   // inference_lock contention
  std::optional<double> vad_stop_to_finalize_start_ms;  // trigger latency

  // Extension fields (counts, identifiers; not aggregated by /stats):
  uint64_t finalize_seq = 0;
  int active_sessions_at_emit = 0;
  // ... (any additional timing the runtime collects)
};

// lib/telemetry/stats_collector.h
class StatsCollector {
 public:
  explicit StatsCollector(size_t window_size = 2048, bool enabled = true);

  // Called from the finalize emit path; thread-safe.
  void record(SessionTiming timing);

  // Called from the HTTP /stats handler; thread-safe; returns JSON.
  std::string snapshot_json(std::optional<size_t> last_n = std::nullopt) const;

  // Operator hooks:
  bool enabled() const { return enabled_; }
  size_t window_size() const { return window_size_; }

 private:
  // Implementation: bounded deque + mutex; nearest-rank quantile computed in snapshot().
  // ~80 bytes/sample × 2048 = ~160KB upper bound (matches the Python implementation).
};
```

Per-finalize cost: **one mutex-guarded deque push** (~tens of nanoseconds). No GPU sync. The 5 input
metrics ALL come from timestamps the finalize path already collects (matching Python's "free" property);
the runtime library just hands them to `StatsCollector::record(timing)`.

### WS server (the new binary) — `ws_server.cpp`

Responsibilities:
1. **HTTP+WS listener** on a single port (defaults to whatever Python server uses; likely 8080).
   Routes:
   - `GET /health` → JSON: admission counters + uptime (mirror Python).
   - `GET /stats` → JSON: `StatsCollector::snapshot_json(last_n)` + admission counters +
     active-session distribution.
   - `GET /stats?last=<N>` → narrow to recent N.
   - `WS /` (or `/v1/transcribe` matching Python convention) → the main streaming endpoint.
2. **WS lifecycle**:
   - `accept` → `DensityAdmission::try_admit(stream_id)` → if `SHED_*` then close with WS-1013.
   - `ADMITTED` → construct `SessionState`, start per-connection worker thread.
   - Recv loop: read PCM frames → `run_steady_chunk_density(...)` (via scheduler when ON) →
     emit interim events as JSON over WS.
   - Client close OR stream-end VAD → trigger finalize → `run_finalize_density(...)` →
     `SessionTiming` populated → `StatsCollector::record(timing)` → emit final event → close gracefully.
3. **Stale-gen safety** (per Step 2a `--mode stalegen-smoke` 4 scenarios): close/reset/shed bump
   `SessionState::generation`; downstream emit checks generation; never emit after a bump.
4. **Env config**:
   - `NEMOTRON_STATS_ENABLED=0` opts out of /stats (and removes the deque.push from the finalize path).
   - `NEMOTRON_STATS_WINDOW=N` overrides the 2048 default.
   - `NEMOTRON_DENSITY_BATCH_STEADY=1` enables the scheduler (off → unchanged B=1 path; on → batched).
   - All the existing scheduler/admission env vars carry through.
5. **Startup smoke** (the 1257d47 bug-lesson):
   - The integration tests MUST exercise full server `main()` startup under each env config combo, not
     just isolated helper functions.
   - Specifically: a test that runs `ws_server --port <ephemeral> --selftest-and-exit` which constructs
     the server, binds, accepts one connection, sends one frame, closes, exits 0. ANY constructor
     failure (env-var bug, missing artifact, port binding race) is caught at smoke-test time.

### WS protocol contract

The shipped Python server's WS API is the reference. C++ server matches byte-for-byte where reasonable:
- **Client → server frames**: binary PCM 16-bit @ 16kHz (or whatever the existing API expects).
- **Server → client frames**: JSON `EmittedEvent` objects (interim + final), per the
  `EmittedEvent` type in `session_main.cpp`. Keys: `kind` (interim/final), `text`, `tokens`,
  `start_ts_ms`, `end_ts_ms`, etc.
- **Close codes**: 1000 = normal; 1013 = admission shed; 1011 = server fault.
- **Reset**: client sends a control message (TBD — match Python or a documented JSON command);
  server bumps `SessionState::generation`, suppresses downstream emits, restarts the session state.

We need to AUDIT the Python `server.py` for the exact protocol details (control messages, close codes,
header semantics, etc.) before the C++ implementation. **This audit is the first sub-task** of the full
Step 3b implementation.

## III. Bounded "get started" scope (this sprint)

What to do NOW (while L40S runs):

**A) Architecture design doc** (this file) — DONE.

**B) CMakeLists refactor + WS server skeleton** (Codex delegation, ~1-2hr compute):
   - Carve `add_library(nemotron_runtime STATIC ...)` from the current `#include "session_main.cpp"`
     pattern. Move shared code into the `lib/` layout. Migrate `density_main` to `target_link_libraries`
     instead of #include. Smoke that `density_main --mode b2-t1` still PASSes (the existing gate).
   - Scaffold `runtime/cpp/ws_server.cpp` with: empty `main()`, the HTTP+WS listener boilerplate cribbed
     from `ws_tail_microbench.cpp` (the RFC6455 plumbing is the same), the route-handler stubs (`/health`
     returns `{"status":"ok"}` for now; `/stats` returns `null` for now; `WS /` returns
     `"not implemented"`), and the integration TODO list.
   - Add `lib/telemetry/session_timing.h` + `lib/telemetry/stats_collector.h` + a minimal
     `StatsCollector` implementation (deque + record + nearest-rank quantile + JSON serializer) +
     unit tests via `--mode stats-smoke` in density_main or a small `stats_smoke.cpp` binary.

**C) Full Step 3b implementation** (separate post-L40S delegation):
   - Audit Python `server.py` for exact protocol details.
   - WS lifecycle integration: accept → admit → session → recv-loop → emit → close → record.
   - HTTP /health + /stats wired to the runtime telemetry.
   - Stale-gen integration via Step 2a's generation primitive.
   - Startup-smoke test (the 1257d47 bug-lesson).
   - Bounded integration test: Python client (the existing test client?) drives one session through
     the new C++ server end-to-end; transcription matches existing server's output.
   - Paired adversarial review (decision-critical per Step 3 in PHASE2-PLAN.md).

**D) Productionization** (post-3b, separate sprint):
   - Deploy config (systemd unit, ASR.env, RUNBOOK update).
   - Multi-process MPS packing.
   - Load balancer integration.
   - SageMaker integration (per `deployment-target-sagemaker` memory).

## IV. Risks + things to think about

1. **The session_main.cpp refactor risks breaking density_main + the existing tests.** Mitigation: do
   the refactor in 2 steps — (1) extract the public API to headers + leave the implementation as
   `session.cpp` (single file, same code); (2) re-run b2-t1 + density-sweep-N=4 smoke; only after green,
   start splitting `session.cpp` into per-module .cpp files. Bounded blast radius.
2. **WS+HTTP on one listener vs separate**. Some implementations use separate ports for WS and HTTP
   admin (e.g., 8080 WS, 8081 admin). Python server has WS+HTTP on the same listener. **Match Python**
   for compatibility unless there's a reason to split.
3. **boost::beast vs raw RFC6455.** Step 3a chose raw (no boost dep complexity). Production WS server
   needs more robust framing (compression, ping/pong, large frames, etc.). **Recommendation: stay raw
   for v1**, port to boost::beast later only if real protocol gaps surface. The Phase-1 ws_tail RFC6455
   handles the basics + we have the code in front of us.
4. **The Python server's "telemetry dict" semantics** need exact matching for /stats output to be
   apples-to-apples with the Python server's /stats. Specifically the 5 metric NAMES + their UNITS
   (ms) must match exactly so operator dashboards work cross-binary.
5. **Concurrency model**: thread-per-connection (matches existing density runtime's per-worker model).
   On g6e.8xlarge (32 vCPU), supports up to ~80-100 concurrent per the v5 sizing rule. For production
   that needs N>100 per box, the multi-dispatcher/per-stream-pool work (Tier 4) is the next lever.

## V. Net

The architecture is **library + WS application**, with the library carved from the existing monolithic
code. The /stats endpoint is a **first-class library API** (`StatsCollector` + `SessionTiming`), wired
through the finalize path, served by the WS server's HTTP route handler. **The 1257d47 bug-lesson sets
a startup-smoke requirement** for any test plan.

Bounded scope this sprint = architecture doc (this file) + CMakeLists refactor + WS server skeleton
with stub routes + the `StatsCollector` library piece. Full Step 3b implementation is the next
delegation, after L40S sweeps land.

The deferred items (production deploy config, MPS packing, SageMaker integration) are post-Phase-2 work.
