# Step 3b — production WS server architecture (design v2)

**v2 2026-05-28** — supersedes v1 (committed `8b7e783`) after Round 1 paired adversarial review folded
both `reviews/codex-Step3b-design-round1.md` (verdict: HOLD Part A) and `reviews/opus-Step3b-design-
round1.md` (GO-with-changes). v1's direction was right; v2 fills in the substantive gaps: explicit
library boundary, Python-compatible WS protocol (with values extracted from `server.py`), exact
StatsCollector contract, threading model, close-code table, graceful shutdown, OpenSSL dependency fix,
expanded startup smoke matrix.

Goal: build the production WebSocket + HTTP server for the native runtime. **(1) Core library**
(`libnemotron_runtime.a`) carved from the current `#include "session_main.cpp"` monolith + **(2) a
production WS+HTTP application** (`ws_server`) that links the library and adds protocol/lifecycle/
observability/admission/stale-gen/stats.

## I. Current state — the starting point

`runtime/cpp/` today is monolithic-compile binaries (Phase-1 pattern). `density_main.cpp:1` literally
does `#include "session_main.cpp"`. No `libnemotron_runtime.a` exists. The WS server task needs us to
create that boundary first.

## II. Library boundary — EXPLICIT public/private table

**Public** (exposed via `lib/session/session.h`; production callers — WS server + density_main use
these):
- Types: `SessionState`, `SessionMode`, `FinalizeFinish`, `Tokenizer`, `WorkerContext`, `AOTIArtifacts`.
- Wire DTO (NEW — see §IV): `WireEvent` { `std::string type; std::optional<std::string> text;
  std::optional<bool> is_final; std::optional<std::map<...>> finalize_timing;` } — separate from
  internal `EmittedEvent` so production output isn't coupled to internal harness shapes.
- Session lifecycle entry points (NEW — production-shaped wrappers, not the harness-shaped
  `run_steady_chunk_density(bundle, prefix, chunk_idx, ...)`):
  - `void reset_session(SessionState&, ...)`.
  - `void append_pcm_and_drain(SessionState&, const PCMFrame&, ...)` — drives preproc → encoder
    (via scheduler when on, direct B=1 when off) → decode → interim emit.
  - `void handle_vad_start(SessionState&, ...)` / `handle_vad_stop(SessionState&, ...)`.
  - `WireEvent finalize_session(SessionState&, FinalizeFinish, SessionTiming& out)` — runs finalize,
    returns the final wire event, populates timing for StatsCollector.
- Constants: WS protocol message types (`"transcript"`, `"reset"`, `"end"`, `"vad_start"`,
  `"vad_stop"`, `"ready"`), close codes (see §V).
- `SessionTiming` (`lib/telemetry/session_timing.h`) — see §III.
- `StatsCollector` (`lib/telemetry/stats_collector.h`) — see §III.
- `DensityAdmission` + `BatchedSteadyScheduler` + `BatchedSteadyLoaderSet` (existing modules, already
  clean).

**Private** (implementation only; NOT exposed in headers):
- `EmittedEvent` (internal struct that downstream `WireEvent` is constructed from).
- `emit_event`, append-only delta helpers, state-machine internals.
- `AudioFrontend`, `FinalizeAudioInputs`, internal mel/buffer types.
- `TimingBuckets`, `MarginStats`, `CacheOwnershipStats` (debug telemetry — internal).
- `gold_events_from_bundle`, equality checks, bundle/gold fixture helpers (HARNESS — NOT production).
- All `--mode <test>` mode dispatchers (density-sweep, b2-t1, stalegen-smoke, etc.) — these stay in
  density_main.cpp.
- The current harness-shaped `run_steady_chunk_density(bundle, prefix, chunk_index, ...)` stays
  internal; production calls `append_pcm_and_drain` which wraps it without leaking harness params.

**Refactor blast-radius mitigation** (per Codex review must-fold #1 + Opus #1):
1. Step 1: extract the public API to `session.h` + leave the implementation as a single
   `lib/session/session.cpp` (a moved copy of `session_main.cpp`).
2. Step 2: migrate density_main to `target_link_libraries(density_main nemotron_runtime ...)` instead
   of `#include`. Smoke that b2-t1 + density-sweep N=4 + stalegen-smoke still PASS.
3. Step 3 (optional, later): split `session.cpp` into sub-modules (`preproc.cpp`, `encoder.cpp`,
   `decode.cpp`, `finalize.cpp`, `emit.cpp`) — bounded refactor; not in this sprint.

## III. Telemetry: SessionTiming + StatsCollector — Python-compatible contract

### SessionTiming struct (per-finalize record)
```cpp
struct SessionTiming {
  // The 5 SLO metrics (matching Python /stats):
  std::optional<double> vad_stop_to_sent_ms;            // server-side TTFS — primary SLO signal
  std::optional<double> fork_flush_wall_ms;             // encoder-finalize wall
  std::optional<double> vad_stop_recv_to_process_ms;    // intake-backlog / load-tail
  std::optional<double> lock_wait_ms;                   // inference_lock contention
  std::optional<double> vad_stop_to_finalize_start_ms;  // trigger latency

  // Lifecycle metadata (populated by caller at emit time, BEFORE session removal):
  uint64_t finalize_seq;            // monotonic per-server counter
  int active_sessions_at_emit;       // SNAPSHOT at the emitting worker, NOT live count at /stats time
  bool was_suppressed = false;       // true if stale-gen check fired — record is counted as suppressed not emitted
  double emit_unix_ts;               // server clock at the moment of final-send attempt

  // Optional context:
  std::optional<std::string> close_reason;     // "normal", "shed", "vad_stop", "client_reset", etc.
};
```

### StatsCollector class
```cpp
class StatsCollector {
 public:
  // Construction reads NEMOTRON_STATS_ENABLED (default 1) + NEMOTRON_STATS_WINDOW (default 2048).
  explicit StatsCollector(size_t window_size = 2048, bool enabled = true);

  // record() is called per finalize, from the emitting worker AFTER the final send attempt and the
  // stale-gen check, BEFORE session removal. Thread-safe; ~tens of nanoseconds (mutex + deque push).
  // If !enabled_: no-op. If timing.was_suppressed: increment suppressed_in_window + lifetime_suppressed
  // only, do NOT append to the metric window.
  void record(SessionTiming timing);

  // snapshot() returns a structured object that JSON / Prometheus serializers wrap.
  // Lock semantics: copy the deque + counters under mutex, sort/serialize OUTSIDE mutex (avoid stall
  // of finalizers during high-rate /stats polling — Codex must-fold #6 + Opus #9).
  struct Snapshot {
    bool enabled;
    size_t window_size;
    size_t samples;                       // total appended in window
    double since_unix;                    // oldest sample's emit_unix_ts
    double until_unix;                    // newest sample's emit_unix_ts
    size_t emitted_in_window;             // appended samples (success path)
    size_t suppressed_in_window;          // stale-gen suppressed (not appended)
    uint64_t lifetime_emitted;
    uint64_t lifetime_suppressed;
    std::map<std::string, MetricSummary> metrics;     // per-metric p50/p90/p95/p99/max/count
    Distribution active_sessions_at_emit;             // histogram at finalize time
  };
  Snapshot snapshot(std::optional<size_t> last_n = std::nullopt) const;
  std::string snapshot_json(std::optional<size_t> last_n = std::nullopt) const;     // wraps snapshot()
  std::string snapshot_prometheus(std::optional<size_t> last_n = std::nullopt) const;  // future-cheap

  bool enabled() const { return enabled_; }
  size_t window_size() const { return window_size_; }
};

struct MetricSummary {
  size_t count;        // n samples WHERE THIS METRIC IS POPULATED (per-metric, per Codex #4 + Opus #4)
  double p50, p90, p95, p99, max;     // quantile formula: round(p * (n-1)) clamped to [0, n-1]
                                       // matches Python exactly; "nearest-rank" was ambiguous.
};
```

### /stats response shape (matching Python)
```json
{
  "enabled": true,
  "window_size": 2048,
  "samples": 1024,
  "since_unix": 1701234567.123,
  "until_unix": 1701234567.789,
  "emitted_in_window": 1020,
  "suppressed_in_window": 4,
  "lifetime_emitted": 12345,
  "lifetime_suppressed": 12,
  "metrics": {
    "vad_stop_to_sent_ms": {"count": 1020, "p50": 13.5, "p90": 19.2, "p95": 21.1, "p99": 27.2, "max": 89.1},
    "fork_flush_wall_ms": {"count": 1020, "p50": ..., ...},
    "vad_stop_recv_to_process_ms": {...},
    "lock_wait_ms": {...},
    "vad_stop_to_finalize_start_ms": {...}
  },
  "active_sessions_at_emit": {"p50": 32, "p90": 47, "p95": 50, "p99": 56, "max": 64},
  "admission": {
    "active_cap": 64, "backlog_cap": 12,
    "offered": 12500, "admitted": 12480,
    "active_count": 38, "backlog_count": 0,
    "active_peak": 64,
    "shed_close_count": 20, "shed_close_rate": 0.0016
  }
}
```

### `?last=N` query
Snapshot narrows the metric-quantile computation to the most recent N samples (after sort by
`emit_unix_ts` if necessary). `samples`, `since_unix`, `until_unix`, `emitted_in_window`,
`suppressed_in_window` reflect the narrowed slice. `lifetime_*` counters are global.

## IV. WS protocol contract — Python-compatible (extracted from `server.py`)

### Routing on the single listener (matching Python)
```
TCP port (default 8080) → accept → read HTTP request line + headers:
  GET /health           HTTP/1.1                         → JSON 200, terminate connection
  GET /stats[?last=N]   HTTP/1.1                         → JSON 200, terminate connection
  GET /                 with `Upgrade: websocket` headers → WS handshake → frames
  GET /                 without Upgrade                  → HTTP 400 (or a docs redirect)
  anything else                                          → HTTP 404
```
The Step 3a raw RFC6455 handshake (in `ws_tail_microbench.cpp`) only checks `Sec-WebSocket-Key` and
ignores method/path/Upgrade/Connection/version. **It cannot be copied as-is** — the production
handler reads + parses the full HTTP request line + headers, dispatches by path + Upgrade-presence,
validates `Sec-WebSocket-Version: 13` + `Connection: Upgrade` + `Upgrade: websocket`, computes
`Sec-WebSocket-Accept` (SHA1(key + magic) base64).

### WS frames

**Client → server**:
- Binary frames: PCM 16-bit @ 16kHz (default; confirm via server.py audit if codec flexibility exists).
  Frame size: implementation-bounded by max-message-size (10 MiB per Python).
- Text frames: control messages as JSON `{"type": "<type>", ...}`. Recognized types:
  - `{"type": "reset"[, "finalize": true/false]}` — reset session (default finalize=true).
  - `{"type": "end"[, "finalize": true/false]}` — end session (default finalize=true).
  - `{"type": "vad_start"}` — client-side VAD hint.
  - `{"type": "vad_stop"}` — client-side VAD hint (triggers finalize if applicable).
- Unknown control message types: log + IGNORE (match Python behavior; do NOT close on unknown type
  to preserve forward-compat).

**Server → client** (all as text frames, JSON):
- On connect: `{"type": "ready"}` immediately after WS handshake completes.
- Transcripts: `{"type": "transcript", "text": "...", "is_final": false/true, ...}`. Optional fields
  (depending on event type / client config — confirm during full audit): `finalize_timing` on final.
- Errors: `{"type": "error", "message": "..."}` for recoverable errors (before close).

### Close codes (explicit table)
| Code | Meaning | When |
|---|---|---|
| 1000 | Normal | Client-initiated close received & processed; server-initiated post-final-sent. |
| 1001 | Going away | SIGTERM graceful shutdown drain. |
| 1003 | Unsupported data | Non-PCM binary frame OR text frame in an unsupported encoding. |
| 1008 | Policy violation | Per-connection rate limit / WS subprotocol mismatch (if used). |
| 1009 | Message too big | Frame > NEMOTRON_WS_MAX_MESSAGE_SIZE (default 10 MiB matching Python). |
| 1011 | Internal server error | Any unhandled C++ exception in the WS handler; scheduler fault. |
| 1013 | Try again later | Admission shed (SHED_ACTIVE_CAP or SHED_BACKLOG_CAP); message `"admission_backpressure"`. |

## V. Stale-gen integration — emit point enumeration

Per Step 2a's generation primitive, the WS lifecycle has these output paths that need a generation
check before emitting:

| Path | Check location | On stale | Telemetry |
|---|---|---|---|
| Steady chunk → interim emit (WS text frame) | Before `ws.send(transcript_interim_json)` | Drop silently | `stale_gen.drops_at_event_emit++` |
| Finalize completes → final emit (WS text frame) | Before `ws.send(transcript_final_json)` | Drop silently | `stale_gen.drops_at_finalize_output++` |
| Finalize completes → StatsCollector::record | After the emit check, only if emit went through | If suppressed: `record(timing{was_suppressed=true})` → counts in `suppressed_in_window` not `emitted_in_window` | StatsCollector lifetime counters |
| /stats response | NO check — snapshots already-vetted records | n/a | n/a |
| Bumps (close/reset/shed) | Owned by the connection's worker thread (no cross-thread bump) | Generation counter atomic-incremented before any sub-emit | — |

## VI. Threading + concurrency model

- **1 accept/router thread**: accepts TCP, reads HTTP request, dispatches.
- **1 dispatcher thread**: BatchedSteadyScheduler (existing, from B2; only constructed if
  `NEMOTRON_DENSITY_BATCH_STEADY=1`).
- **N per-connection worker threads**: one per active WS connection. Owns SessionState, generation
  counter, emit path. Handles its own VAD-stop → finalize → stats-record → close sequence.
- **HTTP admin handler**: synchronous, on the accept thread or a small dedicated pool. `/health` and
  `/stats` are read-only / cheap operations.
- **StatsCollector mutex**: at the realized N=64 with B_max=4, finalize rate is ~33-34/s; mutex
  acquires/sec ~50 including /stats polls. Negligible contention. Snapshot copies the deque under
  mutex (microseconds), sorts/serializes OUTSIDE mutex (no finalizer stall).

## VII. Graceful shutdown (SIGTERM)

```
SIGTERM → set shutting_down_=true in DensityAdmission
       → new accepts: try_admit returns SHED_SHUTDOWN → close WS-1013 immediately (or HTTP 503 for /health/stats)
       → existing connections: send WS-1001 "going away" frame; await natural VAD-stop → finalize → close
       → wait up to NEMOTRON_SHUTDOWN_DRAIN_SEC (default 30s) for in-flight sessions to complete
       → after drain timeout: force-close remaining with WS-1011, log session IDs
       → flush StatsCollector lifetime totals to stdout for post-deploy analysis
       → exit 0 if clean drain, non-zero if forced
```

Critical for cluster rolling deploys; LB / orchestrator orchestrates the rolling restart.

## VIII. Startup smoke matrix (expanded per the 1257d47 lesson)

The `--selftest-and-exit` flag must exercise these constructor-path scenarios; any failure
catches at build/test time, not in live deploy:

| Scenario | Expected |
|---|---|
| Default env, valid artifacts | clean startup, /health 200, exit 0 |
| `NEMOTRON_STATS_ENABLED=0` | startup OK, /stats returns `{"enabled":false}`, exit 0 |
| `NEMOTRON_STATS_WINDOW=abc` (invalid int) | startup error logged, non-zero exit (don't silently fall through) |
| `NEMOTRON_DENSITY_BATCH_STEADY=1` + missing `steady_b_artifacts/MANIFEST.json` | startup error logged, non-zero exit |
| `--port 0` (auto-bind) | clean startup, /health 200 on bound port, exit 0 |
| `--admission-active-cap 0` (degenerate) | startup error (cap must be positive), non-zero exit |
| Bound port + receive 1 /health request + 1 /stats request + 1 WS handshake (no audio) + clean close | all return correct JSON / 101 Switching Protocols, exit 0 |
| One admission-shed scenario (cap=1, attempt 2 connections) | second connection gets WS-1013 with `"admission_backpressure"` message |

## IX. Environment + CLI contract

Env vars (production override default at deploy):
- `NEMOTRON_DENSITY_BATCH_STEADY` (default 0; 1 enables scheduler).
- `NEMOTRON_DENSITY_BATCH_MAX` (default 4 when scheduler ON).
- `NEMOTRON_DENSITY_BATCH_WINDOW_MS` (default 0 per v5 / plan-v5 banner).
- `NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS` (default 0).
- `NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP` (NO default — deploy-required, per v5-B).
- `NEMOTRON_DENSITY_ADMISSION_BACKLOG_CAP` (default 12).
- `NEMOTRON_STATS_ENABLED` (default 1).
- `NEMOTRON_STATS_WINDOW` (default 2048).
- `NEMOTRON_WS_MAX_MESSAGE_SIZE` (default 10485760 = 10 MiB).
- `NEMOTRON_SHUTDOWN_DRAIN_SEC` (default 30).
- `NEMOTRON_GOLD_EVENTS_TOLERANT` (default 1 per Fix #8; opt-in strict for debug).

CLI args (mirror env where it makes sense; `--<name>` flags override env):
- `--port <int>` (required or default 8080).
- `--steady-batch-dir <path>` (default ./steady_b_artifacts).
- `--admission-active-cap <int>` (REQUIRED if env not set).
- `--selftest-and-exit` (run the §VIII matrix, exit 0/non-zero).

## X. Build dependencies + container fix

**OpenSSL is REQUIRED** for the SHA1 + base64 in `Sec-WebSocket-Accept`. The current
`runtime/container/Dockerfile` does NOT install `libssl-dev` (only `Dockerfile.unified` does, which
isn't the build env used by Phase-2 work). **This will fail Part A's build.**

**Two options to fix** (both work):
- **Option A**: add `libssl-dev` to `runtime/container/Dockerfile`. Then `find_package(OpenSSL
  REQUIRED)` works, `ws_server` links `OpenSSL::Crypto`.
- **Option B**: replace OpenSSL SHA1 with a repo-local SHA1 implementation (we already have a
  SHA256 impl in `steady_batch_primitive.h`; add SHA1 similarly). Avoids the dep.

**Recommendation**: Option A (small Dockerfile change, no code dup). Document in the design + the
CMakeLists.

## XI. MPS-readiness (production deploy considers it)

The deployment target is L40S + multi-process MPS per the project memory. The WS server design must
NOT preclude this:
- **Per-process port binding**: take `--port` arg; never hard-code.
- **Per-process StatsCollector window**: each MPS slot has its own 2048-sample window; cross-process
  aggregation deferred to LB/Prometheus.
- **Process identity**: include `pid` + optionally `process_label` in /health + /stats responses for
  operators to distinguish MPS slots.
- **No cross-process state assumptions**: each instance is independent.

LB-level aggregation (across MPS slots) is deployment-owned, not server-owned.

## XII. Bounded "get started" scope (this sprint)

**Part A revised** (delegated separately; **MAY NEED REDO** if the in-flight `bdajesege` Part A
made wrong choices from v1):
- Dockerfile fix: `libssl-dev` added.
- CMakeLists: `add_library(nemotron_runtime STATIC ...)`; density_main migrated to `target_link`.
- `lib/session/session.{h,cpp}` carved per §II's explicit boundary table.
- `lib/telemetry/session_timing.h` + `stats_collector.{h,cpp}` per §III's Python-compatible contract
  (including the `Snapshot` struct + json/prom serializers).
- `lib/ws/handshake.{h,cpp}` — proper HTTP request/header parser + WS handshake (replaces the
  ws_tail_microbench's minimal version per §IV routing).
- `runtime/cpp/ws_server.cpp` — minimal main with `--selftest-and-exit` exercising the §VIII matrix.
  Route stubs (`{"type":"ready"}` on WS, `{"status":"ok"}` on /health, `null` on /stats placeholder
  until Part B wires it).

**Part B** (separate post-Part-A delegation, ~half day):
- Full `server.py` audit (line-by-line) producing a final protocol-compatibility table; revise any
  Part A choices that the audit invalidates.
- WS lifecycle wiring: accept → admit → session create → recv-loop → emit interim → vad_stop →
  finalize → emit final + StatsCollector::record → close.
- /stats route wired to StatsCollector.
- Stale-gen integration at emit points (per §V table).
- Graceful shutdown (per §VII).
- Backpressure: WS frame size cap (1009 close), per-connection send timeout (1011), ping/pong
  keep-alive, /stats polling rate limit (configurable; default 10/sec).
- Bounded integration test: spin C++ server + drive one full session with a Python client; transcript
  byte-for-byte == existing Python server output for the same audio.
- Startup smoke matrix per §VIII fully implemented.

## XIII. Risk register

1. **Part A in-flight may need redo** if its v1-driven choices conflict with v2. Mitigation: when Part
   A lands, audit the diff against v2; redo if substantive.
2. **server.py audit moved forward into Part A** (was Part B). The audit might still surface details
   the v2 design missed (e.g., per-client config negotiation, language hints, exact `finalize_timing`
   shape). Mitigation: build a TEST ORACLE — a script that runs the same audio through Python and
   C++ servers and diffs the wire JSON byte-for-byte. Catches any miss.
3. **Dispatcher single-thread saturation at higher N**: same as v5 plan ceiling (~80-100). WS server
   doesn't change this; multi-dispatcher is Tier-4 future work.
4. **MPS deployment** is Phase-3 work; this design says "don't preclude" but doesn't implement
   cross-process aggregation.
5. **Auth/authz** explicitly NOT in v2 scope. Production deployment assumes trusted-network behind
   an LB / API gateway. If auth is later required, the WS server gets an `Authorization` header
   check on the HTTP handshake.

## XIV. What HOLDS from v1 (with v2 sharpening)

- Architecture: library + WS application.
- StatsCollector as a first-class library API (now with the exact Python contract).
- Default-on /stats with `NEMOTRON_STATS_ENABLED=0` opt-out.
- Startup-smoke discipline (now with the §VIII expanded matrix).
- WS_1013 admission shed + stale-gen via generation tokens (Step 2a primitive).

## XV. Net

v2 is **build-ready for Part A** (with the Dockerfile fix + the explicit boundary table). The
remaining ambiguities are bounded test-oracle items (byte-for-byte server.py compatibility) deferred
to Part B with a test mechanism that catches mismatches.

If the in-flight Part A (`bdajesege`) lands with v1 choices that conflict with v2 on any of: library
boundary, WS routing, wire format, control messages, close codes — REDO those parts. Else, fold
forward.

Round 2 paired review on v2 is the next adversarial pass (per the user's 5-round directive).
