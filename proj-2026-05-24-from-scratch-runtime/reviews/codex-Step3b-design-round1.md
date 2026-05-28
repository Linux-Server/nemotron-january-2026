# Codex Step 3b Design Review - Round 1

Verdict: **HOLD Part A as scoped.** The design is directionally right, but it is not implementation-ready
for the CMake refactor + WS skeleton delegation. The unsafe part is not "write a stub server"; it is that
Part A would freeze a public C++ boundary and a protocol skeleton while both are still `etc.` / `TBD`.
Move the Python protocol audit and the library API inventory into Part A, fold the points below, then build.

## Must Fold Before Code

1. **The library boundary is undefined.**

   Section II says `lib/session/session.h` exposes `SessionState`, `SessionMode`, `reset_session`,
   `run_steady_chunk_density`, `run_finalize_density`, `EmittedEvent`, `Tokenizer`, "etc." That is not a
   boundary for a 4668-line `session_main.cpp`; it is a handoff of ad-hoc decisions to the implementer.

   Fold an explicit public/private table. My recommended public surface for v1:
   - Public: `SessionState`, `SessionMode`, `FinalizeFinish`, `EmittedEvent` or a separate wire event DTO,
     `Tokenizer`, `SessionRuntime`/`SessionCore` owning modules/loaders/audio frontend, `reset_session`,
     `append_pcm_and_drain`, `handle_vad_start`, `handle_vad_stop`, `finalize_session`.
   - Public only if the server really calls them directly: scheduler-facing `run_steady_chunk_density` and
     `run_finalize_density`, but wrapped so the server does not pass harness-only `bundle/prefix/chunk_index`
     gold-fixture parameters.
   - Private: `emit_event`, append-only delta helpers, `AudioFrontend`, `FinalizeAudioInputs`,
     `TimingBuckets`, `MarginStats`, `CacheOwnershipStats`, `gold_events_from_bundle`, equality checks,
     bundle/gold fixture helpers, SHA helpers unrelated to runtime API, and all test/probe modes.

   Better: do not expose `run_steady_chunk_density(...)` as the production API at all. Its current signature
   leaks density harness, CUDA stream, stale telemetry, scheduler diff, and gold-bundle concepts. The WS
   server should call a production-shaped session object, not a test harness function.

2. **The WS protocol section is too thin and partly wrong.**

   The doc says server events are JSON `EmittedEvent` objects with `kind`, `text`, `tokens`,
   `start_ts_ms`, `end_ts_ms`. Current Python sends `{"type":"ready"}` on connect and transcript payloads
   shaped like `{"type":"transcript","text":...,"is_final":true/false,...}`; final events may include
   `finalize_timing`. Current `EmittedEvent` has `kind/tokens/collector_tokens/text/collector_text`, not
   `start_ts_ms/end_ts_ms`.

   Fold the `server.py` audit into Part A, before the skeleton chooses routes, control messages, event JSON,
   query parameters, max message size, or close behavior. The current audit deferral to Part B creates
   predictable rework.

3. **One listener: decide it and specify routing.**

   Use one listener to match Python. The HTTP router must distinguish plain HTTP from WS by method/path plus
   `Connection: Upgrade`, `Upgrade: websocket`, `Sec-WebSocket-Key`, and version headers. `GET /health` and
   `GET /stats` are plain HTTP. `GET /` with valid Upgrade is WS. `GET /` without Upgrade should be an
   explicit HTTP response. Unknown paths get 404.

   The Step 3a raw handshake cannot be copied as-is: it only looks for `Sec-WebSocket-Key` and ignores
   method, path, `Upgrade`, `Connection`, version, and admin routes.

4. **StatsCollector needs the Python contract, not just a sketch.**

   Null handling must match Python:
   - If `vad_stop` or `final_sent` is missing, do not append a sample; increment emitted/suppressed lifetime
     counters only.
   - For appended samples, skip null values per metric. Each metric summary has its own `count`.
   - Include the Python response shape: `enabled`, `window_size`, `samples`, `since_unix`, `until_unix`,
     `emitted_in_window`, `suppressed_in_window`, `lifetime_emitted`, `lifetime_suppressed`, `metrics`,
     `active_sessions_at_emit`, `admission`.
   - Specify the exact percentile index formula. "nearest-rank" is ambiguous; Python uses
     `round(p * (n - 1))`, clamped.

   Also change the API from `snapshot_json()` only to `snapshot()` returning a structured object, with JSON
   and optional Prometheus serializers layered on top. A string-only API makes `/metrics` harder later.

5. **`active_sessions_at_emit` needs a race-tolerant definition.**

   It is a finalize-time sample, not "current when `/stats` is polled." Define it as an atomic/registry count
   captured on the emitting worker immediately after the final send attempt and before session removal /
   `on_close`. Pass that value in `SessionTiming`. Do not have `StatsCollector` read `DensityAdmission`
   after the fact.

6. **Threading and lock behavior need a concrete model.**

   Section IV says thread-per-connection, but not how HTTP handlers run. Fold this:
   - One accept/router thread.
   - WS connections get worker threads.
   - `/health` and `/stats` are handled synchronously on a small admin handler pool or the accept loop only
     after confirming they are cheap.
   - `StatsCollector::record()` does one mutex push. At N=64 and Bmax=4, finalize rate is only about
     33-34/s, so record-side mutex contention is not material.
   - `snapshot()` must copy the deque under lock and sort/serialize outside the lock. Holding the lock while
     sorting five vectors and formatting JSON lets aggressive `/stats` polling stall finalizers.

   Lock-free stats are unnecessary for v1 if snapshot copying is short and `/stats` has a sane rate limit.

7. **Close-code mapping must be exact.**

   Fold a table. Minimum:
   - `1000`: normal client/server completion.
   - `1001`: server shutdown/drain "going away".
   - `1003`: unsupported frame/content type if the protocol rejects it.
   - `1008`: policy violation or invalid query/auth if used.
   - `1009`: binary/text frame exceeds max size; Python currently uses 10 MiB max WS message size.
   - `1011`: internal server fault after admission.
   - `1013`: admission/backpressure shed; Python uses message `admission_backpressure`.

   Also decide whether invalid JSON/unknown control messages are ignored for Python compatibility or closed
   with an error code. Do not leave this to the skeleton.

8. **Reset/control semantics are not TBD.**

   Python recognizes text JSON `type in {"reset","end","vad_start","vad_stop"}`; reset/end carry
   `finalize` defaulting true. C++ must either match this exactly or version the protocol. Part A cannot
   invent a placeholder reset command and still claim compatibility.

9. **Stale-gen integration must enumerate every output path.**

   Fold the exact check pattern:
   - Capture generation when steady/finalize work is enqueued.
   - Check before encode, before decode, before interim event emit, and before final output emit.
   - `StatsCollector::record()` happens only after the final output generation check. Stale/suppressed work
     must not enter the latency sample window as a successful TTFS sample.
   - `/stats` response itself does not need a per-session generation check; it snapshots already-vetted
     records.
   - Close/reset/shed bump generation and define which thread owns the bump.

10. **Graceful shutdown is missing.**

    SIGTERM for rolling deploys needs a contract: stop accepting, reject new WS with 1013 or HTTP 503 during
    drain, optionally send 1001 to existing sockets, allow in-flight finalizes for a bounded grace period,
    then force-close and exit nonzero/zero according to policy. Also define whether final stats are flushed.

11. **Backpressure is broader than admission.**

    Add limits for WS frame size, per-connection queued bytes/events, send timeout, ping/pong timeout, and
    `/stats` polling rate. A slow client must not block a worker indefinitely in `send`, and a high-QPS
    stats poller must not starve finalizer `record()`.

12. **OpenSSL dependency is currently a build risk.**

    `ws_tail_microbench.cpp` includes `openssl/sha.h` and CMake has `find_package(OpenSSL REQUIRED)`.
    `runtime/container/Dockerfile` installs `python3 python3-pip python3-venv python3-dev cmake g++ git
    libsndfile1 ffmpeg`; it does not install `libssl-dev`. `Dockerfile.unified` has `libssl-dev`, but the
    `nemotron-aoti:cu128` container for this project does not. Fold either `libssl-dev` into the runtime
    container or replace OpenSSL SHA1 with a repo-local implementation.

13. **MPS/multi-process deployment must not be an afterthought.**

    The design can punt cross-process aggregation, but it must not preclude it. Fold: per-process port
    binding strategy, per-process `StatsCollector` window semantics, process identity in `/health` and
    `/stats`, and LB/HAProxy aggregation as deployment-owned. Do not hard-code one process/one GPU
    assumptions into the server API.

14. **Prometheus: decide now at the API layer.**

    Exposing `/metrics` can be deferred, but the collector should produce a structured snapshot that can be
    serialized to JSON and Prometheus text. If production monitoring is Prom-first, adding this now is cheap;
    bolting it onto `snapshot_json()` later is not.

15. **Startup smoke is right but too narrow.**

    Keep the 1257d47 lesson: full `main()` must be exercised. Add matrix entries for stats enabled/disabled,
    invalid env values, admin HTTP requests, WS handshake, one rejected admission path, and at least one
    constructor failure path. The prior bug was a constructor/env parsing failure, not a data-path failure.

## Secondary Missing Items

- Auth/authz or an explicit "trusted network only" production assumption.
- TLS/proxy/header assumptions; raw POSIX server probably sits behind an LB, so document the boundary.
- Correlation IDs/session IDs in logs and response metadata for distributed tracing.
- Exact audio contract: PCM encoding, sample rate, endian, channels, chunking, and whether codecs like Opus
  are rejected.
- Query/header compatibility with Python, including target language validation.
- Readiness versus liveness: `/health` should distinguish model loading, ready, draining, and degraded.
- Fragmentation, ping/pong, close-frame echoing, and protocol error handling in the raw RFC6455 code.
- Admin endpoint exposure in multi-tenant deployments.
- Test oracle for byte-for-byte protocol compatibility with Python.

## Net

**HOLD Part A as currently written.** Allow a no-code or read-only Part A0 that inventories
`session_main.cpp` symbols and audits `server.py`; do not start the CMake carve-out or WS skeleton until v2
spells out the public C++ API, the one-listener routing contract, the Python-compatible WS protocol,
StatsCollector semantics, shutdown/backpressure, and the OpenSSL/container dependency fix.
