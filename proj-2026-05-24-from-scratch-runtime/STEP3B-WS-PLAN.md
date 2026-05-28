# Plan: Step 3b — Production WS+HTTP Server for the Native Runtime

Project directory: `./proj-2026-05-24-from-scratch-runtime`

## Context

Phase 2 landed the batched-steady scheduler + the density runtime (B2 + Tier 3 + Step 2a committed
2026-05-28), demonstrating ≥1.60× bounded lift on L40S at N=64 staggered with deep SLO headroom
(ttfs p95 21ms vs 175ms budget; F1 funding CLEARED). Step 3b productionizes this work as a real
**WS+HTTP server** that imports a carved-out `libnemotron_runtime` library, matches the existing
Python `server.py` protocol byte-for-byte (canonicalized), and ships a `/stats` endpoint mirroring
PR `279f033` for cheap always-on rolling-latency telemetry. The binding architecture is
`reviews/Step3b-WS-architecture.md` v4 — CONVERGED after 4 rounds of paired Codex+Opus adversarial
review.

## Reference implementations

- **`src/nemotron_speech/server.py`** — the shipped Python WS server. **THE PROTOCOL REFERENCE**:
  WS frames, control messages (`{"type":"reset|end|vad_start|vad_stop"}`), wire format
  (`{"type":"transcript","text":...,"is_final":...,"finalize"?:bool,"finalize_timing"?:{...}}`),
  routes (`/health` returns `{"status":"healthy"|"loading","model_loaded":bool}`, `/stats`),
  query validation (`?model=`, `?language=`), close codes (1013 admission shed = HTTP 503
  pre-handshake; WS-1009 message > 10 MiB; WS-1011 server fault), Silero VAD integration
  (`silence0_warm200` per `silence0-warm200-shippable` memory). Step WS-A1 produces the
  audit table.
- **Shipped Python /stats endpoint** (PRs `279f033` + `1257d47`) — defines the response shape
  (`enabled / window_size / samples / since_unix / until_unix / emitted_in_window /
  suppressed_in_window / lifetime_emitted / lifetime_suppressed / metrics{5 SLO} /
  active_sessions_at_emit / admission{Python-shape}`), the 2048-finalize sliding window default,
  `?last=N` narrowing, nearest-rank quantile via `round(p * (n-1))`. C++ port mirrors exactly via
  `StatsCollector`. The 1257d47 bug-lesson sets a **startup-smoke discipline**: any test plan
  MUST exercise the full server `main()` startup with each env config combo (the original
  `_env_int` bug shipped because tests AST-extracted helpers but skipped the constructor).
- **`runtime/cpp/ws_tail_microbench.cpp`** (Step 3a, committed `14cd282`) — standalone WS echo
  server with raw RFC6455 POSIX framing + per-stage timestamping. The handshake code (lines
  ~handshake region) is the STARTING POINT for `lib/ws/handshake.cpp` but **cannot be copied
  as-is**: it only checks `Sec-WebSocket-Key` and ignores method/path/`Upgrade`/`Connection`/version
  headers; the production handler must do full HTTP request-line + header parsing + route dispatch.
- **`runtime/cpp/batched_steady_scheduler.{h,cpp}`** (B2, committed `0925fa6` +
  Tier 3 `0a4994f` + F2-T `5155a96`) — the cross-stream batched scheduler the WS server's per-
  worker threads enqueue to when `NEMOTRON_DENSITY_BATCH_STEADY=1`. Already library-shaped; just
  needs the CMake target migration.
- **`runtime/cpp/density_admission.{h,cpp}`** (Step 2a, committed `7035d01`) — admission
  policy with atomic counters + `try_admit` returning `{ADMITTED, QUEUED, SHED_ACTIVE_CAP,
  SHED_BACKLOG_CAP}`. The WS server's accept path calls this; pre-handshake shed = HTTP 503;
  post-handshake shed = WS-1013.
- **`runtime/cpp/steady_batch_primitive.h`** (B1+B2, committed `3887cb3` + later) — the steady
  batch primitive + the built-in repo-local SHA256 + JSON parser (a starting point for the JSON
  emit helpers in `lib/runtime_io`). The SHA256 implementation is reusable for the WS handshake's
  `Sec-WebSocket-Accept` if Option B (avoid OpenSSL) is taken.
- **`runtime/step6_server_oracle.py`** — the existing Python oracle harness; Codex Round 2
  pointed to it as the starting point for the Step 3b test oracle (Step WS-B6). Either reuse or
  supersede.
- **`reviews/Step3b-WS-architecture.md` v4** — the binding architecture spec (CONVERGED after
  Round 4 paired review, commit `22df3d1`). Each step below cites the relevant §.

## Current state

**The library boundary doesn't exist yet** — `runtime/cpp/density_main.cpp:46` does
`#include "session_main.cpp"` (Phase-1 pattern: monolithic-compile binaries). Step WS-A3 carves
`libnemotron_runtime.a`.

| File | Lines | Status / Role |
|---|---|---|
| `runtime/cpp/session_main.cpp` | 4668 | Phase-1 session core. To be carved into `lib/session/session.{h,cpp}` + `lib/session/runtime.{h,cpp}` exposing `SessionState`, `SessionRuntime`, `SharedRuntime`, `PCMFrame`, `WireEvent`, `SessionTiming`, `Tokenizer`. Private internals stay in `.cpp`. |
| `runtime/cpp/density_main.cpp` | ~7000 | Density harness; migrates to `target_link_libraries(density_main nemotron_runtime ...)` instead of `#include`. Smokes (b2-t1, density-sweep N=4, stalegen-smoke, admission-smoke, stats-smoke) MUST stay green. |
| `runtime/cpp/batched_steady_scheduler.{h,cpp}` | 202 + 524 | B2 scheduler. Already header-mostly; moves under `lib/scheduler/` (header-only or thin .cpp). |
| `runtime/cpp/steady_batch_primitive.h` | 663 | B1 primitive + built-in SHA256 + JSON parser + manifest verify. SHA256 lifts to `lib/runtime_io/io.h` for reuse by WS handshake (if Option B). |
| `runtime/cpp/density_admission.{h,cpp}` | 77 + 161 | Step 2a admission. Already library-shaped. |
| `runtime/cpp/ws_tail_microbench.{cpp,_client.cpp}` | 851 + 472 | Step 3a standalone echo server. Stays standalone; the framing/handshake logic in it is the starting reference for `lib/ws/handshake.{h,cpp}` (with the caveats above). |
| `runtime/cpp/ws_server.cpp` | (exists) | **v1 from in-flight Part A `bdajesege` (PRE-v4 design).** Per v4 §XII, **v1's WS skeleton / route stubs are SUPERSEDED** — discard those bytes, salvage only mechanical CMake/file-move pieces if useful. |
| `runtime/container/Dockerfile` | — | Currently installs `python3 python3-pip python3-venv python3-dev cmake g++ git libsndfile1 ffmpeg`. **Missing `libssl-dev`** — Step WS-A2 adds it (Option A) OR Step WS-A2 vendors a repo-local SHA1 to avoid the dep (Option B). |
| `runtime/cpp/CMakeLists.txt` | ~74 | Currently has one `add_executable` per binary, no `add_library`. Step WS-A3 adds `add_library(nemotron_runtime STATIC ...)` + `target_link_libraries` migrations. |

**Existing key entry points** (private until carved):
- `runtime/cpp/session_main.cpp` `run_steady_chunk_density(bundle, prefix, chunk_index, ...)` — HARNESS-shaped (takes gold-fixture params); STAYS PRIVATE per v4 §II. Production `SessionRuntime::append_pcm_and_drain` wraps it.
- `runtime/cpp/session_main.cpp` `run_finalize_density(...)` — same: PRIVATE; `SessionRuntime::finalize_now` wraps.
- `runtime/cpp/density_admission.h` `class DensityAdmission` — public, moves under `lib/admission/`.
- `runtime/cpp/batched_steady_scheduler.h` `class BatchedSteadyScheduler` — public, moves under `lib/scheduler/`.

## Rules

### From PLAN_RULES.md (project-wide)

- **Environment**: Python (export/oracle): `HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python <script>` from `runtime/` (has nemo+torch 2.8+cu128). C++ build+run: in-container `nemotron-aoti:cu128` via `runtime/container/enter.sh`. Strip-validation + nemo-dependent steps: HOST only.
- **Oracle + bars**: Per-step oracle = `finalize_ref.py` (extended); AOTI is NOT byte-exact (~1e-2 drift, F'); bar is TOKEN-exact + EVENT/DELTA-exact vs finalize_ref. Do NOT loosen token/event checks to WER except where Step 4 explicitly measures corpus WER.
- **Test protocol (per step)**: (1) Build affected C++ target in-container (must compile clean); (2) run the step's harness — relevant equivalence assertion (token / event-delta / mel-hash / WER) MUST PASS vs finalize_ref + report real numbers (investigate divergence, don't loosen); (3) re-run existing N=200 session gate + the B2+Tier3 smoke set (b2-t1 4-row, density-sweep N=4 OFF, stalegen-smoke, admission-smoke, stats-smoke) to confirm no regression; (4) artifacts (.ts/.pt2/bundles) are gitignored, commit code + docs + logs (force-add logs under `runtime/artifacts/logs/`).
- **Review intensity**: Steps WS-A1, A3, A6, B1, B2, B6 = decision-critical → PAIRED adversarial review (Codex `/cx-delegate` + independent Opus pass), folded to `reviews/`, before marking `[x]`. Steps WS-A2, A4, A5, B3, B4, B5 = Opus review + independent re-run.
- **Honesty**: if a step's full bar isn't met, mark the residual explicitly (no over-claim); correct any prior over-claim.

### Step 3b-specific rules

- **Architecture v4 is the binding spec.** Each step below cites the v4 §; if implementation surfaces ambiguity, escalate (re-open the review, don't paper over).
- **Python compatibility canonicalized.** The test oracle (WS-B6) diffs wire JSON after stripping volatile fields (timestamps, sequence IDs, native scheduler counters) + sorting JSON keys. Byte-for-byte JSON is NOT the contract because Python's `json.dumps` order isn't stable.
- **No silent loosening of close codes / control-message handling.** v4 §VIII pins the WS close-code table + §IV pins control messages. Audit (WS-A1) refines; deviations from the audit's findings need an explicit `reviews/` justification.
- **Library has NO statics for resource state.** Per v4 §II step 1.5, `SharedRuntime` (binary-owned) holds all model loaders / tokenizer / scheduler / etc. Library functions/methods take `SharedRuntime&` references.
- **Startup-smoke discipline** (the 1257d47 bug-lesson). Every step that touches the ws_server constructor / env parsing path adds a row to the `--selftest-and-exit` matrix.
- **OFF-path byte-exactness preserved.** When `NEMOTRON_DENSITY_BATCH_STEADY=0`, the production code path is unchanged. Each step that touches `SessionRuntime` re-runs `density-sweep N=4 OFF smoke` to confirm.
- **No commit before its step's PAIRED REVIEW** (for the marked decision-critical steps). Codex via `/cx-delegate` + independent Opus pass, folded to `reviews/{step}-paired-verdict.md` before marking `[x]`.

## Steps

- [ ] **1. server.py protocol audit → `reviews/server-py-protocol-audit.md`** (PAIRED REVIEW)
  Line-by-line read of `src/nemotron_speech/server.py`. Produce a canonical protocol-compatibility
  table covering: HTTP routes (`/health` exact JSON shape; `/stats` exact shape; query params),
  WS handshake header validation (`?model`, `?language` validation + invalid-value behavior — HTTP
  400 pre-upgrade vs WS error+close), WS control messages (`{"type":"reset|end|vad_start|vad_stop"}`
  with `finalize` flag default), WS frame conventions (text JSON + binary PCM int16 LE 16kHz mono,
  10 MiB max), WS close codes (1000/1001/1003/1008/1009/1011/1013) per scenario, server-side
  Silero VAD integration (when does the server trigger finalize vs the `vad_stop` control message),
  `finalize_timing` exact key set + value types, error frame format. Output drives all subsequent
  steps' contract decisions. This step is markdown-only (no code).
  Key files: `src/nemotron_speech/server.py` (read), `reviews/server-py-protocol-audit.md` (write).

- [ ] **2. Dockerfile libssl-dev + vendor nlohmann::json + picohttpparser** (Opus review)
  v4 §X: add `libssl-dev` to `runtime/container/Dockerfile` (Option A; needed for OpenSSL SHA1 +
  base64 in WS handshake) OR (Option B) extend `lib/runtime_io/io.h` to expose a repo-local SHA1
  (using the SHA256 pattern already there). Recommend Option A (one-line Dockerfile edit; container
  rebuild doesn't invalidate AOTI artifacts which are content-keyed via MANIFEST.json SHAs). Vendor
  `nlohmann/json.hpp` single-header (MIT) at `runtime/cpp/lib/runtime_io/json.hpp` for
  `WireEvent::finalize_timing` flexible-shape serialization (per v4 §II + Codex Round 4 fold).
  Vendor `picohttpparser.h` (MIT, single-header, ~500 LOC by H2O) at
  `runtime/cpp/lib/runtime_io/picohttpparser.h` for the HTTP request-line + header parsing in
  `lib/ws/handshake.cpp` (Step WS-A5).
  Key files: `runtime/container/Dockerfile`, `runtime/cpp/lib/runtime_io/json.hpp` (NEW vendored),
  `runtime/cpp/lib/runtime_io/picohttpparser.h` (NEW vendored).

- [ ] **3. CMakeLists library carve + lib/{session,runtime_io,admission,scheduler,telemetry}
       skeleton + density_main migration** (PAIRED REVIEW)
  Per v4 §II + §XII (Part A revised scope). Carve `add_library(nemotron_runtime STATIC ...)` in
  `runtime/cpp/CMakeLists.txt`. Initial library content (mechanical first pass):
  `lib/session/session.cpp` (a moved copy of `session_main.cpp`) + `lib/session/session.h` exposing
  the public surface (`SessionState`, `SessionMode`, `FinalizeFinish`, `Tokenizer`,
  `WorkerContext`, `AOTIArtifacts`) + `lib/admission/` (move `density_admission.{h,cpp}`) +
  `lib/scheduler/` (move `batched_steady_scheduler.{h,cpp}` + `steady_batch_primitive.h` —
  extract its SHA256+JSON parser helpers to `lib/runtime_io/`). Migrate `density_main.cpp` to
  `target_link_libraries(density_main nemotron_runtime ...)` instead of
  `#include "session_main.cpp"`. **Static-global audit (v4 §II step 1.5):** identify any static
  globals in `session_main.cpp` (tokenizer caches, finalize bucket loaders, AOTI handles); leave
  them in place as part of `session.cpp` for this step (transfer to `SharedRuntime` ownership is
  Step WS-A4). Bar: b2-t1 4-row PASS (0 token/0 event), density-sweep N=4 OFF smoke PASS,
  stalegen-smoke PASS, admission-smoke PASS, stats-smoke PASS. If ANY smoke regresses, STOP this
  step.
  Key files: `runtime/cpp/CMakeLists.txt`, `runtime/cpp/lib/session/session.{h,cpp}` (NEW; carved),
  `runtime/cpp/lib/admission/`, `runtime/cpp/lib/scheduler/`, `runtime/cpp/lib/runtime_io/io.{h,cpp}`,
  `runtime/cpp/density_main.cpp` (migrate include).

- [ ] **4. lib/session/runtime: SessionRuntime + SharedRuntime + PCMFrame + WireEvent + SessionTiming
       + static-global ownership transfer** (Opus review)
  Per v4 §II concrete signatures. Add `runtime/cpp/lib/session/runtime.{h,cpp}` exposing
  `SessionRuntime` (production session: `append_pcm_and_drain(PCMFrame)` → `vector<WireEvent>`,
  `handle_vad_start()`, `handle_vad_stop()` → may include final WireEvent, `reset(bool finalize)`,
  `end(bool finalize)`, `finalize_now()`, `generation()`, `bump_generation()`, `last_timing()`) +
  `SharedRuntime` (binary-owned; loads AOTI loaders + scheduler when ON + tokenizer + finalize
  buckets — all the statics from session.cpp transferred here). `PCMFrame` struct (int16 LE samples
  + count + endianness `static_assert`). `WireEvent` struct (per v4 §II: `type, text, is_final,
  finalize, finalize_timing as nlohmann::json, message` — Python-compatibility-by-omission via
  optional fields). `SessionTiming` struct (5 SLO metrics + lifecycle metadata: `finalize_seq,
  active_sessions_at_emit, was_suppressed, emit_unix_ts, close_reason`). The production methods
  wrap the existing `run_steady_chunk_density` / `run_finalize_density` (which stay private).
  Bar: b2-t1 4-row still PASS (the public-API wrappers don't change density behavior); a NEW
  `--mode runtime-smoke` exercises `SessionRuntime` end-to-end with 1 synthetic PCM frame +
  expects one interim emit + clean finalize.
  Key files: `runtime/cpp/lib/session/runtime.{h,cpp}` (NEW), `runtime/cpp/density_main.cpp`
  (add `--mode runtime-smoke`).

- [ ] **5. lib/telemetry/StatsCollector — Python-exact contract** (Opus review)
  Per v4 §IV. Add `runtime/cpp/lib/telemetry/{session_timing.h, stats_collector.{h,cpp}}`.
  `StatsCollector::record(SessionTiming, bool emitted)` with the v4 completion predicate
  (`!was_suppressed && vad_stop_to_sent_ms.has_value()`); `snapshot(last_n)` returns a `Snapshot`
  struct (Python-exact fields: `enabled, window_size, samples, since_unix, until_unix,
  emitted_in_window, suppressed_in_window, lifetime_emitted, lifetime_suppressed, metrics{5 SLO
  with per-metric count}, active_sessions_at_emit Distribution`); `snapshot_json` + future
  `snapshot_prometheus` serializers. Lock semantics: copy deque under mutex (~µs), sort/serialize
  outside mutex. Env: `NEMOTRON_STATS_ENABLED` (default 1), `NEMOTRON_STATS_WINDOW` (default 2048).
  Quantile formula: `round(p * (n-1))` clamped to `[0, n-1]`. Wire `StatsCollector::record(timing,
  emitted)` into `SessionRuntime::finalize_now()` (caller passes timing + emit-decision flag).
  Bar: `--mode stats-smoke` (50 synthetic finalizes; assert p50/p95/max behave; `last=N` narrowing;
  missing-field tolerance; `enabled=false` short-circuits).
  Key files: `runtime/cpp/lib/telemetry/session_timing.h` (NEW),
  `runtime/cpp/lib/telemetry/stats_collector.{h,cpp}` (NEW), `runtime/cpp/lib/session/runtime.cpp`
  (wire `record` call), `runtime/cpp/density_main.cpp` (add `--mode stats-smoke`).

- [ ] **6. lib/ws: handshake + framing + route dispatch** (PAIRED REVIEW)
  Per v4 §VII. Add `runtime/cpp/lib/ws/{handshake,framing,routes}.{h,cpp}`.
  `lib/ws/handshake.cpp` uses `lib/runtime_io/picohttpparser.h` for HTTP request-line + header
  parsing; validates `Sec-WebSocket-Version: 13`, `Connection: Upgrade`, `Upgrade: websocket`,
  `Sec-WebSocket-Key`; computes `Sec-WebSocket-Accept` (SHA1(key + magic) base64) via OpenSSL OR
  the repo-local SHA1 per WS-A2 choice. Server-owned HTTP semantics (per v4 fold from
  picohttpparser limits): case-insensitive headers, comma-token `Connection: keep-alive,
  Upgrade`, duplicate headers, partial reads, body-on-GET rejection. `lib/ws/framing.cpp`
  implements RFC6455 binary + text frame read/write with **frame-header-first size check** (v4
  §IV WS-1009 anti-OOM: reject frames > 10 MiB based on header length BEFORE reading payload).
  `lib/ws/routes.h` defines the route table (`GET /health`, `GET /stats[?last=N]`,
  `GET /scheduler_telemetry`, `GET / with Upgrade → WS`). Header timeout (2s), max header bytes
  (8 KiB, return HTTP 431 on oversize), HTTP 400 on malformed. Bar: a small `lib/ws/ws_lib_smoke`
  binary or `--mode ws-lib-smoke` in density_main exercises: HTTP 400 on malformed, HTTP 431 on
  oversize, HTTP 404 on unknown path, correct WS-handshake on valid request.
  Key files: `runtime/cpp/lib/ws/handshake.{h,cpp}` (NEW), `runtime/cpp/lib/ws/framing.{h,cpp}`
  (NEW), `runtime/cpp/lib/ws/routes.{h,cpp}` (NEW), `runtime/cpp/density_main.cpp` (add
  `--mode ws-lib-smoke`).

- [ ] **7. ws_server.cpp skeleton + --selftest-and-exit smoke matrix** (Opus review)
  Per v4 §XII + §XV. **Discard the v1 ws_server.cpp** (from in-flight Part A `bdajesege`); start
  from v4. New `runtime/cpp/ws_server.cpp` with: `main()` parsing CLI (`--port` REQUIRED;
  `--admission-active-cap` REQUIRED-or-env; `--admission-backlog-cap`; `--steady-batch-dir`;
  `--process-label`; `--selftest-and-exit`); construct `SharedRuntime` + `DensityAdmission` +
  `StatsCollector` + (when env-enabled) `BatchedSteadyScheduler`; bind HTTP+WS listener via
  `lib/ws/` routes. **Route stubs** returning Python-exact shapes per WS-A1 audit (e.g., `/health`
  returns `{"status":"healthy"|"loading","model_loaded":bool}`; `/stats` returns
  `StatsCollector::snapshot_json(?last)`; `/scheduler_telemetry` returns
  scheduler's `telemetry_snapshot()` JSON; `WS /` accepts handshake + sends `{"type":"ready"}` +
  closes with WS-1011 not-implemented). The `--selftest-and-exit` flag exercises the v4 §XV smoke
  matrix (default env / `NEMOTRON_STATS_ENABLED=0` / invalid `NEMOTRON_STATS_WINDOW=abc` /
  scheduler-ON with valid artifacts / scheduler-ON with missing MANIFEST / `--port 0` / invalid
  cap / 1 /health + 1 /stats + 1 WS handshake / cap=1 + 2 connections / malformed first line /
  oversize headers). Bar: `--selftest-and-exit` exits 0 for clean cases, non-zero with diagnostic
  for failure cases.
  Key files: `runtime/cpp/ws_server.cpp` (REWRITE; discard v1), `runtime/cpp/CMakeLists.txt`
  (ws_server target).

- [ ] **8. Server-side Silero VAD integration** (PAIRED REVIEW)
  Per v4 §VI. Add Silero VAD as a torch::jit model owned by `SharedRuntime` (loaded once at
  startup; warmed). Per-session VAD state in `SessionRuntime` (running probability window). On
  each `append_pcm_and_drain(PCMFrame)` call, run Silero per-frame on CPU (matches Python
  `silence0_warm200`; bounded thread pool via `torch::set_num_threads(2)` to prevent
  oversubscription at N=64). When silence exceeds `NEMOTRON_FINALIZE_SILENCE_MS` (default 0 per
  shipped config) with the `NEMOTRON_VAD_WARMUP_MS` (default 200) cancellation hold, trigger
  `finalize_now(close_reason="vad_stop")`. Client-sent `vad_stop` is a HINT (logged but Silero is
  authoritative). Env: `NEMOTRON_VAD_MODE` (default `"server_authoritative"`; alt `"client_only"`
  for debug). Bar: `--mode vad-smoke` runs N=4 synthetic streams of silence + speech and asserts
  Silero triggers finalize at the right boundaries (per oracle); add `vad_load_failure` row to
  the WS-A7 selftest matrix. Re-run b2-t1 4-row + density-sweep N=4 OFF; both PASS (VAD is
  only-active-on-WS-server path; doesn't perturb density harness).
  Key files: `runtime/cpp/lib/session/runtime.{h,cpp}` (add VAD wiring),
  `runtime/cpp/lib/session/shared.cpp` (load Silero), `runtime/cpp/density_main.cpp` (add
  `--mode vad-smoke`), `runtime/cpp/ws_server.cpp` (selftest matrix update).

- [ ] **9. WS lifecycle wiring + stale-gen integration** (PAIRED REVIEW)
  Per v4 §V + the v3 lifecycle table. Wire `ws_server.cpp` per-connection worker: `accept` →
  `lib/ws/handshake` → query validation (`model`/`language`) → `DensityAdmission::try_admit` (if
  `SHED_*` → HTTP 503 with `{"error":"admission_backpressure"}` pre-upgrade) → construct
  `SessionRuntime` → send `{"type":"ready"}` → recv-loop (binary PCM → `append_pcm_and_drain`
  with stale-gen check; emit interim `WireEvent`s as JSON text frames with stale-gen check) →
  control messages (`reset`/`end`/`vad_start`/`vad_stop`) → `finalize_now()` (triggered by Silero
  OR client control) → `StatsCollector::record(timing, emitted)` → emit final WireEvent with
  stale-gen check → close WS-1000. Generation bumps on reset/close/shed. **Stale-gen emit-point
  enumeration** (v4 §V table): drops_at_event_emit (interim suppressed); drops_at_finalize_output
  (final suppressed); StatsCollector record sees `was_suppressed=true`. Bar: `--mode
  ws-lifecycle-smoke` (or a small Python client driving 1 WS connection through the full lifecycle)
  asserts the correct event sequence. The Python-compat oracle (WS-B6) is the deeper check.
  Key files: `runtime/cpp/ws_server.cpp` (lifecycle worker), `runtime/cpp/lib/session/runtime.cpp`
  (stale-gen check points).

- [ ] **10. Graceful shutdown + backpressure** (Opus review)
  Per v4 §IX + §XI. Graceful shutdown: SIGTERM → `DensityAdmission.shutting_down_=true` (new
  accepts get HTTP 503 with `{"error":"draining"}`); `/health` returns
  `{"status":"draining",...}`; enqueue `finalize_now(close_reason="shutdown")` on each existing
  session (per v4 fold default — NOT natural-wait); wait up to `NEMOTRON_SHUTDOWN_DRAIN_SEC`
  (default 30s) for in-flight to complete; force-close remaining with WS-1001 + log session IDs;
  AFTER workers complete, call `scheduler.close()` (NOT before — else in-flight enqueues
  deadlock); flush StatsCollector lifetime totals to stdout; exit 0 (clean) or non-zero (forced).
  Backpressure: WS frame-size header-first check (1009 reject without reading payload);
  per-connection send timeout `NEMOTRON_WS_SEND_TIMEOUT_SEC` (default 5; WS-1011 on timeout);
  ping/pong keep-alive `NEMOTRON_WS_PING_INTERVAL_SEC` (default 60) +
  `NEMOTRON_WS_PONG_TIMEOUT_SEC` (default 30; WS-1011 on pong timeout). Bar: `--mode
  shutdown-smoke` simulates SIGTERM with 2 in-flight sessions and asserts the drain sequence +
  exit code; `--mode backpressure-smoke` sends a > 10 MiB frame header and asserts WS-1009 close
  WITHOUT reading payload.
  Key files: `runtime/cpp/ws_server.cpp` (signal handling, drain logic),
  `runtime/cpp/lib/ws/framing.cpp` (size check refined), `runtime/cpp/density_main.cpp` (add
  `--mode shutdown-smoke` + `--mode backpressure-smoke`).

- [ ] **11. Test oracle: run_compat.py + canonicalized diff (Part B pre-merge gate)** (PAIRED REVIEW)
  Per v4 §XIV. Create `tests/server_compat/run_compat.py` (or extend `runtime/step6_server_oracle.py`).
  Audio fixture: `runtime/artifacts/session_audio_bundle.ts` rows `utt0..utt7` for smoke (smoke
  scope is bounded; full bundle in a follow-up). Wire: 16 kHz mono signed int16 LE PCM, 640-byte/
  20ms chunks, `vad_start` before audio, `vad_stop` after, `NEMOTRON_CONTINUOUS=1`,
  `NEMOTRON_FINALIZE_SILENCE_MS=0`. Setup: Python server on port 8080, C++ ws_server on port 8081,
  both with same artifacts (sm_120 local or sm_89 on L40S). **Canonicalized assertions** (not
  byte-for-byte JSON): ready frame exact `{"type":"ready"}`; transcript sequence matches Python's
  `type`, `text`, `is_final`, `finalize` flag, final `collector_text` (if applicable), event count;
  `finalize_timing` required keys present + numeric/non-null (values NOT compared — volatile);
  invalid-query (`?model=bogus`) → expected status/error; invalid `?last=` → expected error.
  Volatile fields stripped: timestamps, finalize_timing values, sequence numbers, pid,
  process_label, native scheduler/admission counters. JSON keys sorted before diff. Bar: 8 utts
  PASS canonicalized diff against Python server. This step is the PART B PRE-MERGE GATE — Steps
  WS-B1...WS-B5 are not committed-as-`[x]` until this oracle PASSes for them in combination.
  Key files: `tests/server_compat/run_compat.py` (NEW), possibly
  `runtime/step6_server_oracle.py` (extend), `reviews/Step3b-WS-test-oracle.md` (oracle spec doc).

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | server.py protocol audit | pending | — | PAIRED. Markdown-only; no code. Drives all subsequent contract decisions. |
| 2 | Dockerfile libssl-dev + vendor nlohmann::json + picohttpparser | pending | — | Opus review. Small Dockerfile + 2 vendored single-headers. |
| 3 | CMakeLists library carve + density_main migration | pending | — | PAIRED. Major refactor; smokes MUST stay green. Discard v1 ws_server.cpp; salvage only mechanical pieces. |
| 4 | SessionRuntime + SharedRuntime + concrete public DTOs | pending | — | Opus review. Static-global ownership transfer per v4 §II 1.5. |
| 5 | StatsCollector Python-exact | pending | — | Opus review. Per-metric count + completion predicate per v4 §IV. |
| 6 | lib/ws handshake + framing + routes | pending | — | PAIRED. picohttpparser + server-owned edge cases. |
| 7 | ws_server.cpp skeleton + --selftest-and-exit | pending | — | Opus review. v4 §XV smoke matrix; route stubs return Python-exact shapes. |
| 8 | Server-side Silero VAD | pending | — | PAIRED. Adds substantive scope; CPU inference + thread cap. |
| 9 | WS lifecycle + stale-gen wiring | pending | — | PAIRED. The substantive integration. |
| 10 | Graceful shutdown + backpressure | pending | — | Opus review. SIGTERM drain + frame-size header-first + ping/pong. |
| 11 | Test oracle run_compat.py + canonicalized diff | pending | — | PAIRED. Part B pre-merge gate. |
