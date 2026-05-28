# Step 3b WS server design v2 — Opus Round 2 adversarial review (2026-05-28)

Reviewing the v2 design (`Step3b-WS-architecture.md` committed in `0e58e46`) from-scratch,
adversarially. Builds on Round 1 (`opus-Step3b-design-round1.md` + `codex-Step3b-design-round1.md`) —
does NOT re-flag what's already folded. Looks for what v2 incorrectly folded + new issues introduced
+ remaining ambiguities.

## Verdict (preview)

**GO-with-1-substantive-fold-to-v3** — one BIG missed architectural question (the VAD trigger
integration, §18 below) + 2-3 must-fixes + several minor sharpenings. Not converged yet, but
trajectory is good. After Round 3 fold v3 should be near-final.

## The big miss: VAD trigger integration

**§18 — How does the WS server INITIATE FINALIZE?** v2 mentions `vad_stop` as a client control
message (§IV) but never says whether the server ALSO does server-side VAD detection. The Phase-1
shipped Python server uses Silero VAD on the server (per the project memory:
`silence0-warm200-shippable`). Two triggers possible:

| Source | Implementation |
|---|---|
| Client-side VAD via `{"type":"vad_stop"}` control message | C++ server just listens for the message, triggers finalize on receipt. Trivial. |
| Server-side Silero VAD on incoming PCM | C++ server needs to RUN Silero (torch model) on the audio buffer, detect silence, autonomously trigger finalize. Non-trivial — adds a model dependency + per-stream VAD state. |

The Python server uses server-side Silero per the existing memory. **C++ port must either**:
- Port the Silero integration (substantial; adds another AOTI compile target + per-stream model state).
- Document that client-side VAD is the only supported trigger + revise the protocol to require
  `vad_stop` for finalize.
- Provide both, with the server-side as primary + client-side as override.

**This is a real architectural decision** that v2 doesn't make. v3 must pin it — and if "port Silero
to C++" is the answer, that scope easily doubles Part B's size.

**Fold action**: v3 §IV adds a VAD-trigger subsection. Recommend: server-side Silero (match Python),
with `vad_stop` from client treated as an override / hint. Part B includes the Silero integration.

## Must-folds (substantive)

### 1. WireEvent's `finalize_timing` shape is under-specified
§II says `std::optional<std::map<...>> finalize_timing` — but what keys + value types? Python's
finalize_timing on the final event probably matches the 5 SLO metrics. Pin it concretely:
```cpp
struct WireEvent {
  std::string type;
  std::optional<std::string> text;
  std::optional<bool> is_final;
  std::optional<std::map<std::string, double>> finalize_timing;  // keys per Python audit;
                                                                  // value type double (ms).
};
```
Add to v3: the EXACT key set after server.py audit (Part A's first sub-task).

### 2. Static-library global-state risk
§II Step-1 says "extract public API + leave implementation as single `lib/session/session.cpp`". But
`session_main.cpp` has static globals (tokenizer caches, finalize bucket loaders, AOTI handles).
Behavior under `#include` vs static-link differs subtly: order-of-initialization, symbol visibility,
multiple-TU global re-init.

**Fold action**: v3 §II adds a step-1.5: audit `session_main.cpp` for static globals; mark any that
need explicit ownership transfer to the binary (e.g., density_main owns the AOTI loader pool, passes
a handle into library functions). Common pattern is "library has no globals; binary holds all
ownership; library is pure functions/methods over passed-in state."

### 3. SessionTiming `was_suppressed` semantics
§III mentions `was_suppressed` but doesn't say WHO sets it. The flow:
1. Worker hits emit point → checks generation.
2. Mismatch → "drop silently."
3. StatsCollector::record(timing) — but only if emit went through; the dropped path needs to ALSO
   call record() with `was_suppressed=true` so the lifetime_suppressed counter increments.

**Fold action**: v3 §V table — add explicit columns for "who sets was_suppressed", "who calls
record()". The emitter (or its caller in the worker thread) is responsible. Single line of code per
emit site.

### 4. Dispatcher shutdown ordering
§VII graceful shutdown says "workers drain naturally + force-close after 30s". But the dispatcher
thread (when scheduler ON) has its own shutdown sequence per B2 design (`scheduler.close()` drains
its queue, broadcasts exception, joins thread). v2 doesn't say WHEN `scheduler.close()` is called.

**Fold action**: v3 §VII explicit ordering:
1. SIGTERM → set DensityAdmission.shutting_down_=true (new accepts shed).
2. Wait for workers to drain naturally (up to 30s).
3. AFTER workers done OR after 30s timeout: call `scheduler.close()` (which drains queue + joins
   dispatcher thread; workers still enqueuing while scheduler closing would deadlock).
4. Flush StatsCollector totals to stdout.
5. Exit 0 (clean) or non-zero (forced).

The ordering matters: if scheduler closes before workers finish, in-flight enqueues fail.

### 5. Smoke matrix missing scheduler-ON case
§VIII §all-OFF startup cases. Add: scheduler-ON case with a single WS handshake + 1 PCM frame →
expect interim event JSON + clean close. Catches scheduler integration startup failures (the kind of
thing the F2-T deadlock would have surfaced).

### 6. HTTP parser choice — design, don't punt
§XII "lib/ws/handshake.{h,cpp} — proper HTTP request/header parser". Hand-rolling a robust HTTP/1.1
parser is real work (line termination, header folding, chunked encoding, malformed bytes). Library
options:
- `picohttpparser` — small (single header), permissive license, used by many production servers.
- `nodejs/llhttp` — battle-tested but larger.
- `boost::beast` — heavy dep, but full WS+HTTP if we go this route.
- Hand-rolled — risk of corner cases (continued requests on keep-alive, etc.).

**Fold action**: v3 picks one. Recommend **picohttpparser** — single .h, no deps, ~500 LOC. Add to
the build (just include the header).

### 7. WS frame-size check must be header-first
§IV §1009 close. Implementation note: check the frame header's payload length BEFORE reading the
payload. Otherwise a malicious client can stream a 1GB frame to OOM the server.

**Fold action**: v3 §IV explicit: "Frame payload length is read from the 2/8-byte length field in
the header per RFC6455; if > `NEMOTRON_WS_MAX_MESSAGE_SIZE`, close immediately with 1009 WITHOUT
reading the payload."

## Medium-impact (recommend fold)

### 8. Test oracle (§XIII.2) is too vague
"a script that runs the same audio through Python + C++ servers and diffs the wire JSON" — what
audio? what tolerance? where does it live?

**Fold action**: spec'd as a file path. `tests/server_compat/run_compat.py` runs N fixed audio
clips (from the existing gold corpus, e.g., utts 0, 100, 500, 999) against both servers (Python on
port 8080, C++ on port 8081), captures wire JSON sequences, diffs. Identical except: timing fields
(timestamps differ); event order MUST match exactly; event text+is_final MUST match. Codifies the
compatibility contract.

### 9. Env naming inconsistency
§IX lists `NEMOTRON_DENSITY_*` (admission, batch_*), `NEMOTRON_STATS_*`, `NEMOTRON_WS_*`,
`NEMOTRON_SHUTDOWN_*`. Why is `NEMOTRON_WS_MAX_MESSAGE_SIZE` not `NEMOTRON_WS_MAX_FRAME_BYTES`?
Minor but consistent naming helps operator memorability.

**Fold action**: v3 §IX standardizes (`NEMOTRON_SERVER_*`, `NEMOTRON_STATS_*`, `NEMOTRON_RUNTIME_*`?
or align with existing). Trivial.

### 10. `--port` default risks MPS collision
§IX `--port <int> (required or default 8080)`. For MPS multi-process, default port causes collision.

**Fold action**: v3 either:
- No default — always require explicit `--port`.
- Default + log a warning if running under MPS (detectable via `CUDA_MPS_PIPE_DIRECTORY` env).

### 11. Ping/pong keepalive timing spec
§XII Part B mentions ping/pong without timing. Python's default may be RFC-recommended 60s ping
interval / 30s pong timeout. Match.

**Fold action**: v3 spec: server pings every 60s if no client traffic; pong must arrive within 30s
else close 1011. Make configurable via `NEMOTRON_WS_PING_INTERVAL_SEC` / `NEMOTRON_WS_PONG_TIMEOUT_SEC`.

### 12. Container rebuild risk
§X Option A (add libssl-dev to Dockerfile) rebuilds the container → new image SHA. Any cached
artifacts pinned to the old container image will need re-validation.

**Fold action**: v3 §X notes: "Container rebuild invalidates the image SHA but does NOT invalidate
the AOTI artifacts (which are content-keyed via MANIFEST.json SHAs, not container hashes)." A small
note + suggest verifying.

## Low-impact (acceptable to defer)

### 13. Audio codec spec
§II `append_pcm_and_drain(SessionState&, const PCMFrame&, ...)` — `PCMFrame` not defined. Add type:
`{int16_t* samples; size_t count; int sample_rate; int channels;}` with the convention that
`sample_rate=16000` and `channels=1` are the supported config (others rejected at handshake or first
frame).

### 14. Language hint
Python server has a `lang` field on connect (per the broader API; not directly cited in v2). C++
port should accept it (even if no-op for now). Defer to Part B's audit.

### 15. Correlation IDs
For distributed tracing. Defer to Part B; small addition (one UUID per WS connection, logged + in
JSON responses).

### 16. /health readiness vs liveness
§IV says /health returns `{"status":"ok"}`. Production LBs may want a distinction between liveness
(server alive) vs readiness (model loaded, ready to admit). Defer to Part B with a note: `/health`
returns `{"status":"ok"|"loading"|"draining"|"degraded"}`.

## What's missing entirely (defer or fold?)

- **Auth/authz** — v2 §XIII.5 explicitly defers as "trusted-network behind LB/API gateway." OK.
- **TLS** — same: handled by LB / reverse proxy in front of the server. OK to defer.
- **Per-connection rate limits beyond admission** — defer.
- **Admin endpoint authn** (/health and /stats are world-readable?) — defer with a note that
  production should put admin endpoints on a separate port or require auth.

## Position on Part A in-flight

v2 §XIII.1 says "audit the diff against v2; redo if substantive." I now think we should **hard-abort
Part A and re-launch on v2** because:
- v2's library boundary is concretely different from v1's hand-wave (item §II public/private table).
- v2's WS routing is concretely different from v1's "match Python" wave.
- v2's StatsCollector contract is significantly more specific.
- v2 has the OpenSSL Dockerfile fix that Part A v1 doesn't know about.

The aborted compute is bounded (~30-60min of Codex work — sunk cost). Re-launching from v2 is
cleaner than auditing-and-rebuilding piecemeal.

**Fold action for v3**: §XIII.1 changes from "redo if substantive" to "abort Part A as of v2
landing; relaunch fresh on v2 (or v3 if convergence completes first)."

## What HOLDS

- The substantive v1→v2 folds (library boundary table, Python wire compat, OpenSSL fix, threading
  model, shutdown sequence) all land correctly.
- The architecture direction (library + WS app, StatsCollector first-class, startup-smoke
  discipline) is sound.
- The §XII Part A/B split is right (Part A = foundation; Part B = WS protocol + lifecycle).

## Net

**GO-with-1-substantive-fold + several minor must-fixes → v3.** The substantive fold is the
VAD-trigger architectural question (§18) — pin it to either server-side Silero (match Python) or
client-side-only (simpler) before Part B begins. The 7 must-folds (items 1-7) sharpen v2 without
changing direction. Items 8-12 are medium and 13-16 low; can fold for completeness.

**Recommend hard-aborting Part A** now that v2 is committed — Part A v1 will produce work that we'll
substantively redo against v2.

Round 3 on v3 is the next pass. If Round 3 returns "minor only," we're converged.
