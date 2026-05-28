# Step 3b — production WS server architecture (design v4)

**v4 2026-05-28** — supersedes v3 (committed `5e83a44`) after Round 3 paired adversarial review.
Codex Round 3: GO-with-1-2-must-folds-to-v4 (NOT converged); Opus Round 3: MINOR_ONLY → effectively
converged + 1 clarification.

**v4 folds the 2 Codex must-folds (public-contract-affecting) + the 9 minor items** woven into the
relevant sections:
- (§II) `WireEvent` gets explicit `finalize` field; `finalize_timing` becomes flexible-shape
  (JSON-object/map-of-variant) rather than `map<string,double>` until Part A's audit pins exact keys.
- (§IV) StatsCollector completion predicate pinned: `!was_suppressed && vad_stop_to_sent_ms.has_value()`
  is the deque-append gate; `fork_flush_wall_ms` and the other metrics are optional + per-metric
  `count`.
- (§VI) Silero execution budget pinned: CPU inference matching Python; warm at startup; bounded
  thread pool; selftest covers VAD-load failure.
- (§IX) Shutdown step 5 default pinned: enqueue server-side `finalize_now(close_reason="shutdown")`
  on existing sessions; alt mode `natural_wait_only` deferred.
- (§IX) Scheduler-close fence pinned: workers MUST stop enqueueing before being "joined"; close()
  drains/broadcasts if outstanding work + timeout hit.
- (§VII) /health enum unified: `loading|healthy|draining|degraded`.
- (§VII) Admin endpoint exposure: trusted-network single-listener default; production multi-tenant
  notes added.
- (§II) PCM endianness: compile-time LE assertion + odd-length payload rejection.
- (§VII) picohttpparser edge-case ownership: case-insensitive headers, comma-token Connection,
  duplicate headers, partial-request returns, body-on-GET rejection.
- (§XIV) Test oracle: port allocation explicit; volatile-strip list pinned; Part B pre-merge gate.
- (§XVI) Config sprawl: add operator-facing grouped summary table.

**v3 → v4 is a targeted edit; structure unchanged.**

---

# Step 3b — production WS server architecture (was v3, now v4)

**v3 2026-05-28** — supersedes v2 (committed `0e58e46`) after Round 2 paired adversarial review:
- `reviews/codex-Step3b-design-round2.md` (verdict: GO-with-substantive-revisions-to-v3, NOT converged).
- `reviews/opus-Step3b-design-round2.md` (verdict: GO-with-1-substantive-fold + must-fixes → v3).

Both Round 2 reviews converged on: hard-abort Part A; move full server.py audit INTO Part A
(resolve the v2 §XII vs §XIII contradiction); StatsCollector needs Python-exact-or-native-extended
decision; library boundary needs concrete signatures; graceful shutdown ordering bug.

v3 makes the substantive decisions + folds the concrete Python contract values Codex extracted from
`src/nemotron_speech/server.py`.

## I. Architecture (unchanged direction)

**(1) Core library `libnemotron_runtime.a`** carved from the current `#include "session_main.cpp"`
monolith + **(2) production WS+HTTP server `ws_server`** that links the library and adds protocol/
lifecycle/observability/admission/stale-gen/stats.

## II. Library boundary — concrete public signatures (v3 §1 fold)

### Public types + signatures (`lib/session/session.h`, `lib/session/runtime.h`)
```cpp
// PCM audio frame; v3 §13 fold concrete type.
struct PCMFrame {
  const int16_t* samples;     // signed 16-bit LE
  size_t count;               // number of samples (NOT bytes)
  // Convention: sample_rate=16000, channels=1; rejected at handshake/first-frame otherwise.
};

// Wire-shaped event sent to the WS client; matches Python server.py format.
// v4 fold (Codex Round 3 must-fold #1): add `finalize` field; finalize_timing flexible-shape
// (JSON object) since exact keys/types depend on Part A's server.py audit.
struct WireEvent {
  std::string type;                                              // "ready"|"transcript"|"error"
  std::optional<std::string> text;                               // transcript text
  std::optional<bool> is_final;                                  // transcript-only
  std::optional<bool> finalize;                                  // on transcript reset/end responses
                                                                  // (matches Python's {"finalize":...} flag)
  std::optional<nlohmann::json> finalize_timing;                 // on final only; flexible JSON object
                                                                  // (likely subset of 5 SLO metrics + emit_unix_ts;
                                                                  // exact keys confirmed by Part A audit, see §III)
  std::optional<std::string> message;                            // on "error" only
};
// Note: ws_server's WireEvent → JSON serializer uses field-by-field optional check, so omitting
// any optional field produces the same wire byte stream Python emits (compatibility-by-omission).
// nlohmann::json (or repo-local equivalent) chosen for `finalize_timing` to avoid premature
// schema-freezing before the audit.

// Per-finalize timing record (passed to StatsCollector).
struct SessionTiming {
  // 5 SLO metrics:
  std::optional<double> vad_stop_to_sent_ms;
  std::optional<double> fork_flush_wall_ms;
  std::optional<double> vad_stop_recv_to_process_ms;
  std::optional<double> lock_wait_ms;
  std::optional<double> vad_stop_to_finalize_start_ms;
  // Lifecycle metadata:
  uint64_t finalize_seq;
  int active_sessions_at_emit;
  bool was_suppressed = false;
  double emit_unix_ts;
  std::optional<std::string> close_reason;
};

// Production session runtime — owns SessionState + per-session VAD + lifecycle.
class SessionRuntime {
 public:
  SessionRuntime(const SharedRuntime& shared, SessionConfig cfg);
  ~SessionRuntime();

  // Append PCM, drive preproc+encoder+decode+interim-emit.
  // Returns emitted WireEvents (may be empty if no interim threshold hit this call).
  std::vector<WireEvent> append_pcm_and_drain(const PCMFrame& frame);

  // Server-side VAD hints (informational; vad_stop triggers finalize if applicable).
  void handle_vad_start();
  std::vector<WireEvent> handle_vad_stop();   // may include the final WireEvent

  // Explicit lifecycle.
  std::vector<WireEvent> reset(bool finalize);    // {"type":"reset","finalize":true|false}
  std::vector<WireEvent> end(bool finalize);      // {"type":"end","finalize":true|false}
  std::vector<WireEvent> finalize_now();          // server-side: VAD silence detected → finalize

  // Generation token (Step 2a stale-gen primitive).
  uint64_t generation() const noexcept;
  void bump_generation() noexcept;     // on reset/close/shed

  // Timing for /stats; populated by finalize_now() / handle_vad_stop().
  std::optional<SessionTiming> last_timing() const;
};

// "Shared" runtime resources: AOTI loaders, scheduler (when ON), tokenizer, finalize buckets, etc.
// Owned ONCE by the binary (ws_server / density_main); SessionRuntimes hold a const& to it.
class SharedRuntime {
 public:
  SharedRuntime(SharedRuntimeConfig cfg);     // loads artifacts, optionally constructs scheduler
  ~SharedRuntime();
  // Accessors return const refs; no globals.
  const Tokenizer& tokenizer() const;
  // ... internal getters used by SessionRuntime, kept package-private behind friend if practical.
};

// HTTP/WS dispatch helpers (the production-shaped public surface — NOT the harness
// run_steady_chunk_density which stays private).
```

### Private (NOT in headers)
- `EmittedEvent` (internal struct; `WireEvent` is its public projection).
- `emit_event`, append-only delta helpers, state-machine internals.
- `AudioFrontend`, `FinalizeAudioInputs`, internal mel/buffer types.
- `TimingBuckets`, `MarginStats`, `CacheOwnershipStats` (debug telemetry — internal).
- `gold_events_from_bundle`, equality checks, bundle/gold fixture helpers (HARNESS).
- `run_steady_chunk_density`, `run_finalize_density` (harness-shaped; private; `SessionRuntime`
  methods wrap them with production-shaped signatures).
- All `--mode <test>` mode dispatchers (density-sweep, b2-t1, stalegen-smoke, admission-smoke) stay
  in density_main.cpp.

### Refactor blast-radius mitigation (v3 §2 fold)
Static-library global-state risk: `session_main.cpp` likely has static globals (tokenizer caches,
finalize bucket loaders, AOTI handles). v3 §II adds an explicit step:
- **Step 1.5**: audit `session_main.cpp` for static globals; transfer their ownership to
  `SharedRuntime` (which is binary-owned, not library-owned). Library code has NO statics for
  resource state; pure functions/methods over passed-in `SharedRuntime&`.

### Layout
```
runtime/cpp/
├── lib/
│   ├── session/
│   │   ├── session.h        public types: SessionState, WireEvent, PCMFrame, SessionTiming
│   │   ├── runtime.h        SessionRuntime + SharedRuntime classes
│   │   ├── session.cpp      session lifecycle implementation
│   │   └── shared.cpp       SharedRuntime (artifact loading)
│   ├── telemetry/
│   │   ├── session_timing.h struct (also in session.h's umbrella)
│   │   ├── stats_collector.h
│   │   └── stats_collector.cpp
│   ├── ws/
│   │   ├── handshake.h      HTTP request parser + WS handshake
│   │   ├── handshake.cpp    (uses picohttpparser — see §X)
│   │   ├── framing.h        WS frame read/write
│   │   └── framing.cpp
│   └── runtime_io/
│       ├── io.h             file_exists, directory_exists, sha256_file, JSON parser, picohttpparser
│       └── io.cpp
├── ws_server.cpp            production binary (links libnemotron_runtime)
├── density_main.cpp         MIGRATED: links library, no #include
├── ws_tail_microbench.{cpp,_client.cpp}   UNCHANGED (standalone)
└── CMakeLists.txt           add_library + target_link migrations + ws_server target
```

## III. The server.py protocol audit (v3 §2 fold — MOVED INTO PART A as its first sub-task)

Per Round 2 (both reviews), the audit produces a concrete protocol-compatibility table that v3
references for all WS protocol details. This must be Part A's FIRST sub-task; protocol stubs in
ws_server.cpp depend on it. v3 includes the values Codex extracted; Part A audits the remaining
gaps + produces `reviews/server-py-protocol-audit.md` as the durable artifact.

### Protocol values extracted in Round 2

**HTTP routes**:
- `GET /health` → `{"status":"healthy"|"loading","model_loaded":<bool>[,"admission":...]}`.
- `GET /stats[?last=N]` → see §IV stats response shape (Python-exact).

**WS endpoint**: `GET /` with `Upgrade: websocket` headers; client may include query parameters
`?model=<name>&language=<code>` validated before admission.

**WS frames**:
- Client → server: binary PCM int16 LE @ 16kHz mono OR text JSON control messages.
- Control message types (text JSON, recognized):
  - `{"type":"reset"[,"finalize":true|false]}` (finalize defaults true).
  - `{"type":"end"[,"finalize":true|false]}` (finalize defaults true).
  - `{"type":"vad_start"}` (informational).
  - `{"type":"vad_stop"}` (informational; server-side VAD is authoritative — see §VI).
- Unknown control message types: log + IGNORE (forward-compat).
- Invalid JSON in a text frame: log + IGNORE (matches Python).
- Max WS message size: **10 MiB** (`NEMOTRON_WS_MAX_MESSAGE_SIZE`).

**Server → client** (all text JSON):
- On WS handshake completion: `{"type":"ready"}` exactly.
- Transcripts: `{"type":"transcript","text":"...","is_final":false|true[,"finalize_timing":{...}]}`.
- Errors (recoverable, before close): `{"type":"error","message":"..."}`.

### Open audit items (Part A's first sub-task)
- Exact `finalize_timing` map keys (likely subset of the 5 SLO metrics — confirm via line-by-line
  audit).
- Per-client config negotiation (model/language validation rules — what happens on invalid?
  HTTP 400 before upgrade vs WS error+close).
- Connection-level rate limits beyond admission.
- Any custom headers / CORS / cookies handling.

## IV. StatsCollector — Python-exact /stats contract (v3 §3 fold)

Decision per Codex Round 2 must-fold #3: **Python-exact** for the /stats output. Native scheduler
telemetry (the F2-T `scheduler_telemetry` block) is a SEPARATE optional sub-object, NOT mixed into
the Python /stats core shape.

### Per-finalize record semantics (matches Python)
- Main deque contains ONLY complete samples (records where `vad_stop_ms` and `final_sent_ms` are
  both present). Incomplete suppressed/stale finalizes increment lifetime counters only.
- Complete send-failed samples CAN enter the deque with `emitted=false`.
- Per-metric `count` = samples in window where THIS metric is populated.

### /stats response shape (Python-exact)
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
    "vad_stop_to_sent_ms": {"count": 1020, "p50": ..., "p90": ..., "p95": ..., "p99": ..., "max": ...},
    "fork_flush_wall_ms": {...},
    "vad_stop_recv_to_process_ms": {...},
    "lock_wait_ms": {...},
    "vad_stop_to_finalize_start_ms": {...}
  },
  "active_sessions_at_emit": {"p50": 32, "p90": 47, "p95": 50, "p99": 56, "max": 64},
  "admission": {
    "enabled": true,
    "attempted": 12500,
    "admitted": 12480,
    "rejected": 20,
    "max_backlog": 12,
    "max_ready_age_ms": ...,
    "signal": {...}
  }
}
```

The admission sub-object shape matches Python's `attempted/admitted/rejected/max_backlog/...` (NOT
v2's v2-native names). The C++ DensityAdmission's native counters (`active_count`, `active_peak`,
etc.) map into Python-compatible names at /stats serialization time. Native counters are also
accessible via a separate `/admission` endpoint if operators want the detailed view (optional).

The F2-T `scheduler_telemetry` (timer p50/p95/p99 + dispatcher CPU + queue depth + fairness spread)
is available via `GET /scheduler_telemetry` (separate endpoint, native-extended; never mixed into
/stats).

### Quantile formula
`round(p * (n - 1))` clamped to `[0, n-1]` (matches Python `_compute_quantile_summary`).

### StatsCollector class signature (v3)
```cpp
class StatsCollector {
 public:
  explicit StatsCollector(size_t window_size = 2048, bool enabled = true);

  // Called per finalize from emit path. Completion predicate (v4 fold per Codex Round 3 must-fold
  // #2 — fork_flush_wall_ms is OPTIONAL + per-metric, NOT a completion gate):
  //   COMPLETE = !timing.was_suppressed && timing.vad_stop_to_sent_ms.has_value()
  // (vad_stop_to_sent_ms is the TTFS signal — only metric required for record to count as a
  // "sample"; others contribute only via their per-metric `count`.)
  //
  // Behavior:
  //   if !COMPLETE: increment lifetime_suppressed only; do NOT append to deque.
  //   if COMPLETE: append sample (with `emitted` flag — true if final sent, false if send failed
  //                or stale-gen suppressed at emit time). lifetime_emitted++ if emitted,
  //                lifetime_suppressed++ if !emitted. Per-metric `count` in snapshot computed
  //                from samples where THAT metric is populated.
  void record(SessionTiming timing, bool emitted);

  struct Snapshot { /* same fields as JSON above */ };
  Snapshot snapshot(std::optional<size_t> last_n = std::nullopt) const;
  std::string snapshot_json(std::optional<size_t> last_n = std::nullopt) const;
  std::string snapshot_prometheus(std::optional<size_t> last_n = std::nullopt) const;

  bool enabled() const;
  size_t window_size() const;
};
```

### Lock semantics
`snapshot()` copies the deque + counters under mutex (microseconds), then sorts/serializes
OUTSIDE the mutex. Avoids /stats-polling starvation of finalizer record() calls.

## V. Stale-gen integration — emit-point enumeration (v3 §3 fold, was-suppressed clarified)

| Emit path | Check location | On stale | `was_suppressed` set by | StatsCollector behavior |
|---|---|---|---|---|
| Steady interim WS text | Before `ws.send(json)` | Drop silently | Worker, before send call | N/A (interim doesn't record to StatsCollector) |
| Finalize → final WS text | Before `ws.send(json)` | Drop silently | Worker, before send call | record(timing{was_suppressed=true}, emitted=false) → lifetime_suppressed++ only |
| Finalize → StatsCollector::record | Worker, after emit decision | (does nothing — recording behavior depends on was_suppressed flag) | N/A | (handled in record() per §IV) |
| /stats response | NO check (snapshots already-vetted records) | N/A | N/A | N/A |
| Generation bumps | Connection's worker thread on reset/close/shed | atomic-increment | N/A | N/A |

## VI. VAD trigger — server-side Silero (v3 §1 substantive decision)

**Decision: match Python — server-side Silero VAD is authoritative; client-side `vad_stop` is a HINT.**

Rationale: Python shipped server uses server-side Silero per the `silence0-warm200-shippable`
memory. Production reliability requires not depending on client-side VAD (which varies in quality +
latency). C++ port adds:
- Per-session Silero state (a torch::jit model loaded once into `SharedRuntime`, per-session
  processing state).
- After each `PCMFrame` ingested, Silero processes the audio → returns a probability that
  silence-of-`NEMOTRON_FINALIZE_SILENCE_MS` (default 0ms, with the 200ms VAD-cancellation hold per
  memory) has occurred → triggers `finalize_now()`.
- Client-sent `vad_stop` is a HINT (logged); server's own VAD remains authoritative.

**Part B scope expansion**: includes Silero integration. Estimated +1-2 days vs the v2 scope. The
cost is unavoidable for Python-compatible behavior.

**Configuration**:
- `NEMOTRON_FINALIZE_SILENCE_MS` (default 0; matches existing prod silence0_warm200 config).
- `NEMOTRON_VAD_WARMUP_MS` (default 200; matches existing prod).
- `NEMOTRON_VAD_MODE` enum (default `"server_authoritative"`; alternative `"client_only"` for
  future deployments that want to skip Silero — saves the model load + per-frame eval).

## VII. WS protocol routing (v3 §1 fold, includes malformed handling)

```
TCP accept → read HTTP request line + headers (with 2s timeout, max 8 KiB headers; HTTP 400 on
                                              malformed, HTTP 431 on oversize headers):
  GET /health           → JSON 200 {"status":"healthy"|"loading","model_loaded":<bool>}, close.
  GET /stats[?last=N]   → JSON 200 (Python-exact shape per §IV), close.
  GET /scheduler_telemetry → JSON 200 (native-extended F2-T block), close.
  GET / + Upgrade: websocket + Sec-WebSocket-Version: 13 + Sec-WebSocket-Key + Connection: Upgrade
                        → validate ?model + ?language query params → admission check →
                          if ADMITTED: WS handshake (Sec-WebSocket-Accept) → frames.
                          if SHED_*: HTTP 503 with body `{"error":"admission_backpressure"}` (HTTP-
                          level, not WS-level — matches what Python does on pre-upgrade reject).
  GET / without Upgrade → HTTP 400 (or static "WS endpoint" page if we want).
  anything else         → HTTP 404.
  bytes that don't parse as HTTP → close without WS handshake, log + count as malformed.
```

The Step 3a raw RFC6455 handshake only checks `Sec-WebSocket-Key` and IGNORES method/path/Upgrade/
Connection/version. Production handshake uses `lib/ws/handshake.{h,cpp}` (NEW) which uses
**picohttpparser** (single-header, no deps, ~500 LOC, permissive license — v3 §15 fold) for the
HTTP request line + header parsing.

## VIII. WS close codes (v3 unchanged — explicit table from v2)

| Code | Meaning | When |
|---|---|---|
| 1000 | Normal | Client-initiated close received & processed; server-initiated after final sent. |
| 1001 | Going away | SIGTERM drain timeout reached (sent ONLY at the final close, not at SIGTERM receipt — see §IX). |
| 1003 | Unsupported data | Non-PCM binary frame OR text frame in an unsupported encoding. |
| 1008 | Policy violation | WS subprotocol mismatch (if used). |
| 1009 | Message too big | Frame header indicates payload > `NEMOTRON_WS_MAX_MESSAGE_SIZE` (10 MiB). **Frame-header check is read BEFORE the payload** to prevent OOM (v3 §20 fold). |
| 1011 | Internal server error | Any unhandled C++ exception in WS handler; scheduler fault; pong timeout. |
| 1013 | Try again later | Admission shed (post-handshake — pre-handshake gets HTTP 503 per §VII). |

## IX. Graceful shutdown — corrected ordering (v3 §5 fold)

v2's sequence sent WS-1001 to existing connections BEFORE the drain — wrong (close frame starts the
close handshake; client stops sending). v3 corrected:

```
1. SIGTERM received → DensityAdmission.shutting_down_=true.
2. New TCP accepts on the WS path: HTTP 503 with body `{"error":"draining"}`, no WS handshake.
3. /health returns {"status":"draining","model_loaded":true}.
4. /stats continues to serve (operators want post-deploy visibility).
5. EXISTING WS connections: NO close frame yet. Server-side VAD continues; finalize triggers
   naturally; final event sent; THEN socket-level close with WS-1000 (normal).
   OR: enqueue a server-side finalize_now() on each existing session to accelerate drain (cleaner
   than waiting on client speech).
6. Wait up to `NEMOTRON_SHUTDOWN_DRAIN_SEC` (default 30s) for in-flight sessions to drain.
7. After drain timeout: for any remaining open sockets, send WS-1001 "going away" close frame +
   force-close. Mark forced; log session IDs.
8. Call `scheduler.close()` AFTER workers have all completed (else in-flight enqueues deadlock —
   v3 §4 fold). scheduler's close() drains its queue + joins dispatcher thread.
9. Flush StatsCollector lifetime totals to stdout.
10. Exit 0 (clean) or non-zero (forced).
```

## X. Build dependencies + container fix (unchanged from v2 §X)

- **`libssl-dev`** must be added to `runtime/container/Dockerfile` (Option A, recommended). Container
  rebuild invalidates the image SHA but does NOT invalidate AOTI artifacts (content-keyed via
  MANIFEST.json SHAs).
- **`picohttpparser.h`** added to `lib/runtime_io/` as a vendored single-header (MIT license, ~500
  LOC). No new package dep.

## XI. Threading + concurrency model (v3 §10 fold)

- **1 accept/router thread**: TCP accept, HTTP parse, route dispatch. Bounded work; reads with
  2s header timeout to prevent slowloris.
- **HTTP admin handler pool**: **fixed size 2**, bounded queue depth 16. Handles /health, /stats,
  /scheduler_telemetry. Each handler runs synchronously; queue bounds prevent operator from DoS'ing
  via /stats polling.
- **1 dispatcher thread** (scheduler, ONLY constructed when `NEMOTRON_DENSITY_BATCH_STEADY=1`):
  exists per B2's BatchedSteadyScheduler.
- **N per-connection worker threads**: one per active WS connection. Owns SessionRuntime, generation
  counter, emit path, Silero VAD eval. When scheduler ON: enqueues to dispatcher; when OFF: runs B=1
  directly on its own CUDA stream.
- **StatsCollector mutex**: at N=64 with B_max=4, finalize rate ~33-34/s; /stats poll rate
  operator-configurable but typically 1-10/s. Total mutex contention ~50/s on a microsecond-scale
  critical section = negligible. Snapshot copies under lock + serializes outside.

## XII. Bounded build scope (v3 §1 fold — Part A v1 superseded; relaunch on v3)

**Part A v1-driven work (in-flight `bdajesege` as of v3 commit) is SUPERSEDED.** When it lands:
- AUDIT its diff against v3.
- KEEP mechanical pieces that match v3: CMakeLists `add_library` target structure, basic file moves,
  any small fixes (e.g., dropping unused `ep` param if it shows up).
- DISCARD any v1-driven choices that conflict with v3: WS routing/handshake stubs, StatsCollector
  shape if non-Python, /health Python shape, library public API surface, OpenSSL workarounds, etc.
- Specifically: do NOT use Part A's WS skeleton/handshake/route stubs as the baseline; they predate
  the v3 server.py audit. Re-do those parts on v3.

**Part A revised (v3 — relaunch as a NEW Codex delegation after v3 commits)**:
1. **server.py audit FIRST** (the first sub-task): line-by-line read of `src/nemotron_speech/server.py`;
   produce `reviews/server-py-protocol-audit.md` with the protocol-compatibility table (route shapes,
   header validation, query params, control message handling, close codes, error-frame format,
   `finalize_timing` keys, `_compute_quantile_summary` formula confirmation, /health shape, admission
   sub-object shape). v3 §III + §IV reference this artifact for all protocol details.
2. **Dockerfile fix**: `libssl-dev` added to `runtime/container/Dockerfile`.
3. **CMakeLists library carve**: `add_library(nemotron_runtime STATIC ...)` per §II.
4. **lib/session/{session,runtime,shared}.{h,cpp}** carved per §II with the explicit public API
   (PCMFrame, WireEvent, SessionTiming, SessionRuntime, SharedRuntime).
5. **Static-global audit + transfer to SharedRuntime ownership** (the v3 §II step 1.5).
6. **lib/telemetry/{session_timing.h, stats_collector.{h,cpp}}** per §IV (Python-exact contract).
7. **lib/ws/{handshake,framing}.{h,cpp}** per §VII (picohttpparser-based).
8. **lib/runtime_io/{io.h, io.cpp, picohttpparser.h}** (vendored single-header parser).
9. **runtime/cpp/ws_server.cpp** skeleton: `main()` with `--port` + admission CLI flags +
   `--selftest-and-exit` exercising the §XV smoke matrix. Route stubs return Python-exact shapes
   (per the audit table); WS endpoint accepts handshake + sends `{"type":"ready"}` + closes 1011
   not-implemented.
10. **density_main migrated** to `target_link_libraries(density_main nemotron_runtime ...)`. Smoke
    b2-t1 + density-sweep N=4 OFF + stalegen-smoke + admission-smoke all PASS.

**Part B (separate delegation, post-Part-A)**:
- Server-side Silero VAD integration (per §VI).
- Full WS lifecycle: accept → admit → SessionRuntime construct → recv-loop → server-VAD →
  interim emit (stale-gen-checked) → finalize → final emit (stale-gen-checked) → StatsCollector
  record → close.
- /stats / /scheduler_telemetry / /admission routes wired live.
- Graceful shutdown per §IX.
- Backpressure: WS frame-size header-first check (1009), per-connection send timeout (1011),
  ping/pong keep-alive (configurable; defaults match RFC: 60s ping interval, 30s pong timeout),
  /stats polling rate via the bounded queue.
- The test oracle (§XIV).

## XIII. /port default + MPS-readiness (v3 §21 fold)

- `--port` has NO compiled default (server logs warning + exits non-zero if not set). Forces
  operators to think about MPS port assignment.
- Each process binds its own port; `--port-base + slot_id` is the conventional pattern.
- StatsCollector window is per-process; aggregation is LB-layer's job.
- /health, /stats responses include `pid` + optional `process_label` (`--process-label <s>` CLI) so
  operators distinguish MPS slots.

## XIV. Test oracle (v3 §8 fold — concrete + canonicalized)

`tests/server_compat/run_compat.py` (NEW; reuses `runtime/step6_server_oracle.py` infrastructure):
- **Audio fixture**: `runtime/artifacts/session_audio_bundle.ts` rows `utt0..utt7` (smoke; expand to
  full bundle in Part B+).
- **Wire**: 16 kHz mono signed int16 LE PCM, 640-byte/20ms chunks, `vad_start` before audio,
  `vad_stop` after audio, `NEMOTRON_CONTINUOUS=1`, `NEMOTRON_FINALIZE_SILENCE_MS=0`.
- **Setup**: Python server on port 8080, C++ server on port 8081, both with batching ON, same
  artifacts.
- **Assertions (canonicalized, NOT byte-for-byte)**:
  - Ready frame exact: `{"type":"ready"}`.
  - Transcript sequence: matches Python's `type`, `text`, `is_final`, `finalize` flag, final
    `collector_text`, and event count.
  - `finalize_timing`: required keys present + numeric/non-null; exact values NOT compared
    (volatile).
  - Invalid-query (`?model=bogus`) → expected status/error frame.
  - Invalid `?last=` query → expected /stats error response.
- **Volatile fields stripped** before diff: timestamps, finalize_timing values, sequence numbers.
- **JSON field order canonicalized** (sort keys before diff) — Python `json.dumps` order isn't
  guaranteed.

## XV. Startup smoke matrix (v3 §19 fold — includes scheduler-ON case)

`ws_server --selftest-and-exit` exercises these constructor-path scenarios:

| Scenario | Expected |
|---|---|
| Default env, valid artifacts | clean startup, /health 200, exit 0 |
| `NEMOTRON_STATS_ENABLED=0` | startup OK, /stats returns `{"enabled":false}`, exit 0 |
| `NEMOTRON_STATS_WINDOW=abc` (invalid int) | startup error logged, non-zero exit |
| `NEMOTRON_DENSITY_BATCH_STEADY=1` + missing `steady_b_artifacts/MANIFEST.json` | startup error logged, non-zero exit |
| **`NEMOTRON_DENSITY_BATCH_STEADY=1` + valid artifacts** | clean startup, scheduler constructed, /scheduler_telemetry serves, exit 0 |
| `--port 0` (auto-bind) | clean startup, /health 200 on bound port, exit 0 |
| `--admission-active-cap 0` | startup error (cap must be positive), non-zero exit |
| Bound port + 1 /health + 1 /stats + 1 WS handshake + 1 PCM frame + clean close | all return correct shapes, exit 0 |
| Cap=1, attempt 2 connections | second connection gets HTTP 503 with body `{"error":"admission_backpressure"}` (pre-upgrade reject per §VII) |
| Malformed first HTTP request line | HTTP 400 + connection closed, server continues |
| Oversize headers (>8 KiB) | HTTP 431, connection closed, server continues |

## XVI. Environment + CLI contract (v3 §9 fold — consistent naming)

Env vars (production override default at deploy):
- `NEMOTRON_DENSITY_BATCH_STEADY` (default 0; 1 enables scheduler).
- `NEMOTRON_DENSITY_BATCH_MAX` (default 4 when scheduler ON).
- `NEMOTRON_DENSITY_BATCH_WINDOW_MS` (default 0 per plan v5).
- `NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS` (default 0).
- `NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP` (NO default — deploy-required).
- `NEMOTRON_DENSITY_ADMISSION_BACKLOG_CAP` (default 12).
- `NEMOTRON_STATS_ENABLED` (default 1).
- `NEMOTRON_STATS_WINDOW` (default 2048).
- `NEMOTRON_WS_MAX_MESSAGE_SIZE` (default 10485760 = 10 MiB).
- `NEMOTRON_WS_PING_INTERVAL_SEC` (default 60; matches RFC suggestion).
- `NEMOTRON_WS_PONG_TIMEOUT_SEC` (default 30).
- `NEMOTRON_SHUTDOWN_DRAIN_SEC` (default 30).
- `NEMOTRON_FINALIZE_SILENCE_MS` (default 0 per silence0_warm200 shipped config).
- `NEMOTRON_VAD_WARMUP_MS` (default 200 per silence0_warm200).
- `NEMOTRON_VAD_MODE` (default `"server_authoritative"`; alt `"client_only"`).
- `NEMOTRON_GOLD_EVENTS_TOLERANT` (default 1 per Fix #8; opt-in strict for debug).

CLI (override env):
- `--port <int>` REQUIRED.
- `--admission-active-cap <int>` REQUIRED (env can satisfy).
- `--admission-backlog-cap <int>`.
- `--steady-batch-dir <path>` (default ./steady_b_artifacts).
- `--process-label <str>`.
- `--selftest-and-exit`.

## XVII. Risk register

1. **Server-side Silero adds Part B scope** (~1-2 days). Acceptable; production needs it.
2. **Part A v1-driven work conflicts with v3 in non-trivial ways**. Mitigation: §XII spells out
   what to keep vs discard.
3. **picohttpparser is a single-header dep** — tested in many production servers, but a new dep
   to vendor. Low risk.
4. **DensityAdmission native counters → Python /stats shape mapping** is a serialization-layer
   choice; mistakes hurt operator dashboards. Test oracle covers this.
5. **MPS-readiness deferred to deployment config**; the design doesn't preclude it.
6. **Static-global audit (§II step 1.5)** may surface unexpected complexity in session_main.cpp's
   resource ownership. Bounded; budget some time.

## XVIII. Net

v3 makes the substantive decisions Round 2 demanded:
- Library boundary concrete signatures (SessionRuntime + SharedRuntime + PCMFrame + WireEvent +
  SessionTiming).
- Server-side Silero VAD as authoritative (matches Python, +1-2 days Part B).
- StatsCollector Python-exact /stats output; native scheduler_telemetry separate endpoint.
- Admission sub-object Python-shape wrapper.
- /health Python shape.
- Test oracle canonicalized (not byte-for-byte).
- Graceful shutdown ordering corrected.
- Malformed-request handling concrete.
- Part A v1 superseded; v3 §XII relaunch scope.
- HTTP parser library: picohttpparser.
- Frame-size header-first check.
- Smoke matrix expanded (scheduler-ON case + malformed + oversize headers).

Round 3 on v3 is the next pass. If Round 3 returns "minor only" on both reviews → CONVERGED → fold
final into v4 + relaunch Part A on v3 (or v4).
