<task>
**Step 3a — WS-tail microbench standalone.** Per `reviews/Step3-scoping.md` §3a: a bounded standalone
binary that characterizes per-stage WebSocket latency (accept, recv, queue, serialize, send) under N idle
+ M streaming sockets. Echo-only — no transcription, no scheduler, no session. The output is the
`ws_tail` telemetry block (schema in Step2a-invariant-design.md §III) that Step 4's apples-to-apples
measurement uses to decompose "WS overhead" from "runtime tail."
</task>

<context>
**Goal**: produce a JSON sidecar with `ws_tail` block having p50/p95/p99 + sample count `n` for each of:
- `accept_to_ready_us` — time from `accept()` returning to the connection being ready for read.
- `send_to_recv_us` — round-trip time of a client→server frame echo.
- `recv_to_queue_us` — time from `recv` callback returning to the handler queueing the work.
- `queue_to_scheduler_us` — time from queue push to handler pickup.
- `serialize_and_send_us` — time to serialize the response JSON and write it.
- `client_recv_us` — client-side time from send to recv (the network round-trip).
- `event_loop_lag_us` — measured via sampling the event loop's wakeup latency vs scheduled time.

**WS library choice**: prefer `boost::beast` (already a torch-dependency-adjacent lib; typically in
torch boxes). Fallback: `websocketpp` (header-only, less common). Pick boost::beast if available.

**Architecture**:
- Server side (`runtime/cpp/ws_tail_microbench.cpp`): C++ boost::beast WS server.
  - Listens on a configurable port (`--port`, default 8765).
  - Per-connection: echo-only handler (receive a binary frame, immediately send it back).
  - Per-stage timestamping via `std::chrono::steady_clock` + a per-event log line (JSON), buffered to a
    sidecar.
  - Termination: receives `--duration-ms` flag, runs for that long, then dumps the summary JSON.
- Client side (`runtime/cpp/ws_tail_microbench_client.cpp` OR a Python script
  `runtime/ws_tail_loadgen.py`): drives N idle + M streaming sockets.
  - Idle sockets: keep open, no traffic.
  - Streaming sockets: send a frame every chunk_period_ms (configurable, default 160ms matching the
    audio cadence).
  - Loadgen flags: `--server <host:port>`, `--n-idle`, `--m-streaming`, `--duration-ms`,
    `--chunk-period-ms`.

**Output**: a JSON file with the `ws_tail` block schema (per Step2a-invariant-design.md §III):
```json
{
  "ws_tail": {
    "accept_to_ready_us": {"n": ..., "p50": ..., "p95": ..., "p99": ...},
    "send_to_recv_us":    {"n": ..., ...},
    ...
  },
  "config": {
    "n_idle": ..., "m_streaming": ..., "duration_ms": ..., "chunk_period_ms": ...
  }
}
```

**Build target**: `cmake -S cpp -B cpp/build_ws_tail` → `ws_tail_microbench` + `ws_tail_microbench_client`
(if C++ client; otherwise just the server).

**Out of scope**:
- No integration with the session core, scheduler, or admission.
- No transcription — pure echo.
- No real WS server (that's Step 3b, deferred per scoping doc).
</context>

<verification_loop>
Container build. Run a small smoke: server on :8765, client with `--n-idle 0 --m-streaming 1
--duration-ms 5000`. Verify the JSON output has p50/p95/p99 values (don't care about absolute numbers —
just schema + sanity that the timers are non-zero).
</verification_loop>

<action_safety>
Local only. The microbench server listens on localhost; don't expose to external network. Bounded scope.
</action_safety>

<compact_output_contract>
When done, report:
1. Files created + line counts; WS library used (boost::beast / websocketpp / other).
2. Build result.
3. Smoke run result (one cell with n_idle=0, m_streaming=1) + sample JSON sidecar showing the schema.
4. Loadgen + server CLI usage examples.
</compact_output_contract>
