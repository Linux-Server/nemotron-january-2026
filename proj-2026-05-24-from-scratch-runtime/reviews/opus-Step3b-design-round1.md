# Step 3b WS server architecture — Opus Round 1 adversarial review (2026-05-28)

Reviewing `reviews/Step3b-WS-architecture.md` (committed `8b7e783`) from-scratch, adversarially. Folded
with the parallel Codex Round 1 review after both land. **Verdict preview: GO-with-changes** —
substantial revisions needed before the design is build-ready; specifically the library boundary needs
enumeration, server.py audit needs to MOVE FORWARD not be deferred, and several specific semantics
(stale-gen emit enumeration, close-code mapping, threading model) need to be pinned.

## High-impact gaps (must fold to v2)

### 1. Library boundary is hand-wavy; the refactor will balloon
§II says "Public headers carved from session_main.cpp" but lists only example types (SessionState,
EmittedEvent, Tokenizer, reset_session, etc.) without an EXPLICIT enumerated boundary. session_main.cpp
is 4668 lines. Without an exact list, the Part A delegation has to make ad-hoc decisions about what's
"public" — and Part A is RUNNING NOW. Risk: Part A makes wrong calls; we revise after the rounds land.

**Fold action**: enumerate the public API surface (header symbols) explicitly. At minimum:
- `SessionState`, `SessionMode`, `EmittedEvent`, `Tokenizer`, `WorkerContext`, `AOTIArtifacts`.
- `reset_session`, `run_steady_chunk_density`, `run_finalize_density`, `apply_encoder_outputs_density`,
  `emit_event`, `EmittedEvent::to_json` (if it exists).
- The `EVENT_INTERIM`/`EVENT_FINAL` constants.
- Anything else density_main.cpp currently references — there's a finite set; just grep.

Everything else stays in the library's .cpp implementation, not exported.

### 2. server.py audit is deferred but blocks Part A correctness
§III-C lists "audit Python server.py for exact protocol details" as Part B's first task. But Part A's
WS skeleton (running NOW in `bdajesege`) makes choices about WS handshake, framing, route paths, etc.
Those choices need to be COMPATIBLE with the audit's findings or get redone.

**Fold action**: pull the server.py audit FORWARD into Part A (a 30-min task that surfaces the
protocol contract: route paths, message formats, control protocols, close codes). The Part A
skeleton's stubs then commit to that contract from day 1.

### 3. WS+HTTP listener — design wave-off
§IV.2 says "Match Python for compatibility unless there's a reason to split" but doesn't commit. The
HTTP routing (distinguish `GET /health` from a WS handshake) is non-trivial in raw RFC6455:
- Both arrive on TCP port 8080.
- The WS upgrade has `Upgrade: websocket` + `Connection: Upgrade` headers in the HTTP request.
- The server reads the HTTP request line + headers, routes based on `Upgrade` presence + path.

**Fold action**: explicitly specify the routing pattern. Read 1 HTTP request (lines until blank line),
inspect: if `GET /health` → return JSON 200; if `GET /stats[?last=N]` → return JSON 200; if any path
with `Upgrade: websocket` → run WS handshake + transition to WS frames. Reject otherwise with 404.

### 4. StatsCollector null-handling contract under-specified
§III's `SessionTiming` fields are `std::optional<double>`. Nearest-rank quantile doesn't naturally
handle missing values. What's the contract?

**Fold action**: specify "per-metric": the quantile over `vad_stop_to_sent_ms` is computed from only
the records where that field is set. A record with `vad_stop_to_sent_ms=nullopt` is counted in the
collector's overall sample count but skipped for that metric's quantile. Each metric reports its own
`n` (samples that contributed) so operators can spot missing-data anomalies. Matches Python
`_compute_quantile_summary`'s implicit semantics.

### 5. Stale-gen emit enumeration
§II.3 says "downstream emit checks generation; never emit after a bump" but doesn't enumerate the emit
points. In the WS server lifecycle there are at least:
- WS interim event emit (per steady chunk emit).
- WS final event emit (post-finalize).
- StatsCollector::record() (post-finalize timing record).
- /stats response (read snapshot — not emit-dependent, but the recorded data may include stale-gen
  drops).

**Fold action**: enumerate explicitly + specify the check pattern (current generation == work-item
generation; mismatch → drop silently + `stale_gen.drops_at_<stage>++` per Step 2a design §III).

### 6. WS close-code mapping — be specific
§II.4 lists 1000/1011/1013 generically. Make explicit:
- **1000 (normal)**: server-initiated close after final event sent + acknowledged; client-initiated
  close received and processed.
- **1009 (message-too-big)**: PCM frame exceeds configured per-frame size cap.
- **1011 (server-fault)**: any unhandled C++ exception in the WS handler; the scheduler/dispatcher
  fault path (per Step 2a §II.8); session OOM.
- **1013 (try-again-later)**: admission shed (SHED_ACTIVE_CAP or SHED_BACKLOG_CAP).

Close-code consistency matters for client retry logic + load-balancer behavior.

### 7. Threading model and StatsCollector contention
§IV.5 says "thread-per-connection." Concretely at N=64:
- 64 worker threads.
- 1 dispatcher thread (scheduler).
- 1 HTTP-handler thread? Or async via poll? — UNSPECIFIED.
- StatsCollector mutex acquired on every finalize record (~34 records/sec at the realized rate per
  B3) and on every /stats poll (frequency operator-dependent, typically 1-10/sec from monitoring).
- Mutex contention at ~50 ops/sec on a microsecond-scale critical section is negligible. But the
  design should say so explicitly + commit to a model.

**Fold action**: specify "HTTP routes (/health, /stats) served on a SINGLE handler thread (separate
from connection workers). StatsCollector uses std::mutex; contention rate ~50 acquires/sec is
negligible per Bench-verifiable; if it ever becomes hot, swap to RCU/read-write lock — not needed at
phase-2 throughput."

### 8. Graceful shutdown is unspecified
SIGTERM is the standard cluster-deploy signal. The design says nothing about:
- Should in-flight sessions drain (and how long)?
- Send graceful close (WS-1001 "going away") to all open WS connections?
- Refuse new admissions during drain?
- Hard-kill after a drain timeout (e.g., 30s)?

**Fold action**: specify the shutdown sequence. Recommend:
1. On SIGTERM: set `shutting_down_=true` in DensityAdmission → new accepts get SHED_SHUTDOWN
   (close immediately with WS-1001).
2. Wait for in-flight sessions to drain naturally (their VAD-stop triggers finalize, finalize emits,
   close).
3. After 30s drain timeout: force-close remaining sessions with WS-1011, exit with non-zero status.
4. Steady-state: clean exit 0.

This matters for cluster rolling deploys — without it, deploys lose in-flight transcriptions.

## Medium-impact gaps (recommend fold)

### 9. Backpressure under load — /stats poller starvation
A monitoring system polling /stats every 1s with 64 finalizes/sec recording = the mutex is acquired
~65 times/sec. Each acquire is microseconds. Not a bottleneck at this scale, but:
- If /stats response serialization is slow (the JSON snapshot iterates the window), a single /stats
  poll could hold the mutex for hundreds of microseconds.
- A burst of /stats polls (e.g., from a misconfigured monitor) could starve recorders.

**Fold action**: snapshot under lock + serialize OUTSIDE lock. The snapshot copy is bounded
(~160 KB) so memcpy under lock is fast; JSON serialization (~10-100 µs depending on window) happens
unlocked.

### 10. Reset semantics — pin the control message format
§II.4 says "TBD — match Python." This is a real ambiguity that the server.py audit (point 2) closes.

### 11. MPS-awareness
Production = SageMaker + (likely) multi-process MPS per the deployment-target memory. The design
mentions MPS as post-Phase-2 but should at minimum NOT preclude it. Specifically:
- Per-process port binding (don't hard-code; take a `--port` arg).
- Per-process StatsCollector window (each MPS slot has its own window; aggregation is the LB
  layer's job).
- No assumption of single-server-per-box in any naming/identity logic.

**Fold action**: brief §IV addition: "MPS-ready: server is per-process, takes --port; no
cross-process state; aggregation deferred to LB."

### 12. Dependency confirmation
§II inherits `openssl/sha.h` from `ws_tail_microbench.cpp` for the WS handshake (sec-websocket-key
SHA1). Confirm `nemotron-aoti:cu128` container has OpenSSL. If not, build breaks.

**Fold action**: explicit dependency list in the design. OpenSSL is most likely present (it's a
standard system lib) but verifying once removes the risk.

### 13. Build target naming + layout
The design says `lib/session/session.cpp` etc. but CMakeLists conventions in this repo aren't shown
explicitly. Make sure the design's directory layout matches existing patterns (the current
`runtime/cpp/` is flat). Consider whether a `lib/` subdirectory is the right pattern or whether to
keep flat with naming conventions.

**Fold action**: pick a layout convention + cite a comparable repo file as the model.

## Low-impact / future considerations

### 14. Prometheus /metrics
§II mentions future. Designing the StatsCollector to expose Prom-format too is cheap NOW (one
additional `snapshot_prometheus()` method). Or genuinely defer if Prom isn't the chosen monitoring
stack.

### 15. Authentication / authorization
Not mentioned at all in the design. For a production WS server, auth is typically required (API key
in header, JWT, mTLS, etc.). If we're integrating into an existing system with its own auth layer
(e.g., an LB or API gateway in front), then the WS server doesn't need its own — but say so.

### 16. Per-connection metadata / correlation IDs
For distributed tracing, each connection should have a correlation ID propagated through logs +
optionally back to the client. Not strictly required for v1 but a small addition that pays off later.

### 17. Audio codec assumptions
PCM 16-bit @ 16kHz is the standard but the design doesn't say. If the Python server has codec
flexibility (Opus, etc.), match. If not, document the assumption.

### 18. Health-check interval contract
Load balancers expect /health to be very fast (≤10ms p99). The design says /health returns
"admission counters + uptime" — at high N this might take longer than the LB tolerates. Explicitly
say /health is read-mostly + bounded-time (don't lock anything heavy).

## What HOLDS in the design

- **Library + WS application split** — right architecture.
- **StatsCollector as a first-class library API** — right level of abstraction; mirrors the proven
  Python design.
- **5 primary metrics + nearest-rank quantile** — direct port of the proven Python pattern.
- **Default-on with `NEMOTRON_STATS_ENABLED=0` opt-out** — sensible production default.
- **Startup-smoke discipline** (from the 1257d47 bug-lesson) — necessary; the `--selftest-and-exit`
  mechanism is the right shape.
- **WS_1013 admission shed** — correct WS-native pattern.
- **Stale-gen integration via Step 2a generation tokens** — right primitive.

## Net

**GO-with-changes-to-v2** before Part B implementation begins. The 8 high-impact gaps (items 1-8)
should fold into a v2 of the design; the 5 medium-impact (9-13) should fold for completeness; the 4
low-impact (14-17) can defer or be small additions.

**Critical immediate action**: server.py audit (item 2) should not be deferred to Part B — it should
be a pre-flight to Part A's WS skeleton, otherwise we accumulate revision debt. If Part A
(`bdajesege`) lands before the audit + v2 design, we may have to redo parts of its WS skeleton.
Practical mitigation: keep Part A's WS skeleton VERY THIN (just stubs that close 1011 not-implemented)
so the audit doesn't invalidate much. Confirm Part A's task spec did say this.

The library boundary, stale-gen emit enumeration, close-code mapping, and graceful-shutdown are the
4 specific gaps that materially shape the implementation; they should fold to v2 before delegation B.
