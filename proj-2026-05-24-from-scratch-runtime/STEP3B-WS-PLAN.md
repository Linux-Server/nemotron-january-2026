# Plan: Step 3b — Production WS+HTTP Server for the Native Runtime

**v5 2026-05-28** — supersedes v4 after user-caught design error: v3/v4 incorrectly assumed Python
`server.py` runs server-side Silero VAD. Verification (`grep -i silero src/nemotron_speech/server.py`
= zero hits) confirms the Python server has **NO server-side Silero** — the client (Pipecat) runs
Silero VAD and sends `vad_stop` control messages, which the server consumes with a debounce + a
cancellation hold (the `silence0_warm200` config is a debounce-on-client-signal, not a server-side
VAD). v5 corrections:
- **Step 8 retitled + shrunk**: "Server-side Silero VAD integration" (PAIRED REVIEW, ~1-2 days)
  → "Client vad_stop debounce + finalize trigger" (Opus review, hours). No torch::jit Silero load,
  no per-frame CPU eval, no thread-pool sizing, no `NEMOTRON_VAD_MODE` enum. Step 8 is a state
  machine + a timer (per-session) implemented in the existing recv-loop.
- **Reference implementations** (Python server.py): the "Silero VAD integration (`silence0_warm200`
  per `silence0-warm200-shippable` memory)" line is corrected to "client `vad_stop` consumed with
  debounce + cancellation hold per `silence0_warm200` config — Python has NO server-side Silero;
  Pipecat client runs the VAD."
- **Step 7 selftest matrix**: drop the `vad_load_failure` row (no model load to fail).
- **Rules — decision-critical list**: Step 8 moves from PAIRED to Opus review (the implementation is
  small + obviously-correct enough to skip paired review).
- **Architecture binding spec**: now `reviews/Step3b-WS-architecture.md` **v5** (was v4).

**Process lesson logged**: when design depends on existing shipped behavior, audit the shipped code
FIRST, not the project-memory summary. The `silence0-warm200-shippable` memory described the
*configuration tuning* (0/200) without specifying that the VAD itself is client-side. v5 risk
register adds this as risk #6.

**v4 2026-05-28** — supersedes v3 (committed `85a73c7`) after Round 3 paired adversarial review.
Codex Round 3: GO-with-1-fold (Step 5 key-files line contradicted the prose by instructing to
wire `record()` in `runtime.cpp` — the exact Round 2 ownership/order hazard). Opus Round 3:
MINOR_ONLY (missed Codex's specific). v4 = one-line fix to Step 5's "Key files" parenthetical
("populate SessionTiming / last_timing() ONLY; no record() call"). Step 6 + Step 11 minor wording
left for impl-fold.

**v3 2026-05-28** — supersedes v2 (committed `b5e4bd0`) after Round 2 paired adversarial review:
- `reviews/codex-Step3b-plan-review-round2.md` (verdict: GO-with-1-must-fold-to-v3 —
  StatsCollector ownership/order).
- `reviews/opus-Step3b-plan-review-round2.md` (verdict: MINOR_ONLY → CONVERGED).

v3 folds Codex's single must-fold + 3 minor cleanups:
1. **StatsCollector ownership/order pinned** (Codex Round 2): `SessionRuntime::finalize_now()`
   produces `WireEvent` + `SessionTiming` (accessible via `last_timing()`) but does **NOT** call
   `StatsCollector::record()`. The WS worker owns the call sequence: stale-gen check → final
   send/drop/timeout decision → stamp `was_suppressed` + `emitted` flag → call `record(timing,
   emitted)` exactly once. Step 5 builds/tests StatsCollector in isolation; Step 9 owns the
   production recording integration.
2. Bars-additive header exempts Step 1 (markdown-only, no build target affected).
3. Step 6 / Step 9 explicitly: odd-length binary PCM payloads close WS-1003 BEFORE constructing
   PCMFrame (per v4 §VIII).
4. Step 11 ports allocated free / configurable (default 8080/8081; allow override to avoid local
   test flake).

**v2 2026-05-28** — supersedes v1 (committed `c266771`) after Round 1 paired adversarial review:
- `reviews/codex-Step3b-plan-review-round1.md` (verdict: GO-with-1-2-must-folds-to-v2).
- `reviews/opus-Step3b-plan-review-round1.md` (verdict: GO-with-3-must-folds-to-v2).

v2 folds the convergent must-folds + Codex's v4-contract-restoration findings: Part B commit
semantics, in-flight Part A v1 salvage audit gate inline in Step 3, tightened bars on Steps 4/9/11,
restored v4 contract items (/stats admission Python-shape, /health enum, pid/process_label, HTTP
admin pool, /health PCM-frame matrix row deferral note), Silero N=64 CPU probe, WS-overhead perf
gate, "Bars are ADDITIVE to the global per-step protocol" header rule, N=200 + full smoke set on
every code step.

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
  pre-handshake; WS-1009 message > 10 MiB; WS-1011 server fault). **(v5 correction)** **NO
  server-side Silero VAD** — the server consumes `vad_stop` control messages from the client
  (Pipecat runs Silero) and applies the `silence0_warm200` debounce: `FINALIZE_SILENCE_MS=0` is
  the wait-window after `vad_stop` before finalize_now() (default 0 = immediate),
  `VAD_WARMUP_MS=200` is the cancellation hold during which a subsequent `vad_start` cancels the
  pending finalize. Step WS-A1 produces the audit table (and double-confirms the no-Silero finding).
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

- **Environment**: Python (export/oracle): `HF_HUB_OFFLINE=1 ./.venv/bin/python <script>` from `runtime/` (has nemo+torch 2.8+cu128). The venv is created by `bash setup-venv.sh` (~5-20 min, ~11 GiB; uv-based; Python 3.12.10 + torch 2.8.0+cu128 + NeMo 2.4.1; gitignored; lockfile = `runtime/requirements.txt`, top-level intents = `runtime/requirements.in`). C++ build+run: in-container `nemotron-aoti:cu128` via `runtime/container/enter.sh`. Strip-validation + nemo-dependent steps: HOST only.
- **Oracle + bars**: Per-step oracle = `finalize_ref.py` (extended); AOTI is NOT byte-exact (~1e-2 drift, F'); bar is TOKEN-exact + EVENT/DELTA-exact vs finalize_ref. Do NOT loosen token/event checks to WER except where Step 4 explicitly measures corpus WER.
- **Test protocol (per step)**: (1) Build affected C++ target in-container (must compile clean); (2) run the step's harness — relevant equivalence assertion (token / event-delta / mel-hash / WER) MUST PASS vs finalize_ref + report real numbers (investigate divergence, don't loosen); (3) re-run existing N=200 session gate + the B2+Tier3 smoke set (b2-t1 4-row, density-sweep N=4 OFF, stalegen-smoke, admission-smoke, stats-smoke) to confirm no regression; (4) artifacts (.ts/.pt2/bundles) are gitignored, commit code + docs + logs (force-add logs under `runtime/artifacts/logs/`).
- **Review intensity**: Steps 1, 3, 6, 9, 11 = decision-critical → PAIRED adversarial review (Codex `/cx-delegate` + independent Opus pass), folded to `reviews/`, before marking `[x]`. Steps 2, 4, 5, 7, 8, 10 = Opus review + independent re-run. (v5: Step 8 demoted from PAIRED to Opus — Silero scope removed; remaining work is a small state machine.)
- **Honesty**: if a step's full bar isn't met, mark the residual explicitly (no over-claim); correct any prior over-claim.

### Step 3b-specific rules

- **Architecture v5 is the binding spec** (v5 supersedes v4 with Silero correction; v4 §-references in step bodies remain accurate since §I-V, §VII-XVIII numbering is unchanged — only §VI rewritten, §XII Part B Silero bullet dropped, §XVI dropped NEMOTRON_VAD_MODE, §XVII risk register updated). Each step below cites the relevant §; if implementation surfaces ambiguity, escalate (re-open the review, don't paper over).
- **Python compatibility canonicalized.** The test oracle (WS-B6) diffs wire JSON after stripping volatile fields (timestamps, sequence IDs, native scheduler counters) + sorting JSON keys. Byte-for-byte JSON is NOT the contract because Python's `json.dumps` order isn't stable.
- **No silent loosening of close codes / control-message handling.** v4 §VIII pins the WS close-code table + §IV pins control messages. Audit (WS-A1) refines; deviations from the audit's findings need an explicit `reviews/` justification.
- **Library has NO statics for resource state.** Per v4 §II step 1.5, `SharedRuntime` (binary-owned) holds all model loaders / tokenizer / scheduler / etc. Library functions/methods take `SharedRuntime&` references.
- **Startup-smoke discipline** (the 1257d47 bug-lesson). Every step that touches the ws_server constructor / env parsing path adds a row to the `--selftest-and-exit` matrix.
- **OFF-path byte-exactness preserved.** When `NEMOTRON_DENSITY_BATCH_STEADY=0`, the production code path is unchanged. Each step that touches `SessionRuntime` re-runs `density-sweep N=4 OFF smoke` to confirm.
- **No commit before its step's PAIRED REVIEW** (for the marked decision-critical steps). Codex via `/cx-delegate` + independent Opus pass, folded to `reviews/{step}-paired-verdict.md` before marking `[x]`.

## Steps

**Important rule (v2 fold per Codex Round 1 #3 + bars-additive concern)**: every Step's "Bar"
below is **ADDITIVE** to the global per-step test protocol (PLAN_RULES.md test protocol: build →
harness → **N=200 session gate (`cpp/session_main`) + b2-t1 4-row + density-sweep N=4 OFF +
stalegen-smoke + admission-smoke + stats-smoke** all PASS to confirm no regression). The Bar
specifies the step's NEW assertion; the smoke set + N=200 always re-run.

**Commit semantics (v2 fold per Round 1 #1)**: each Step commits with `[x]` when its OWN Bar +
the global smoke set PASS. **Step 11 (the test oracle) is the integration test for Steps 8-10;
its failure is documented as an INTEGRATION regression to fix (most likely by amending Step 9's
lifecycle wiring), NOT as un-marking prior steps.** Steps 8-10 stay `[x]` based on their per-step
bars; Step 11 stays `[!]` until the integration passes.

- [x] **1. server.py protocol audit → `reviews/server-py-protocol-audit.md`** (PAIRED REVIEW)
  Line-by-line read of `src/nemotron_speech/server.py`. Produce a canonical protocol-compatibility
  table covering: HTTP routes (`/health` exact JSON shape; `/stats` exact shape; query params),
  WS handshake header validation (`?model`, `?language` validation + invalid-value behavior — HTTP
  400 pre-upgrade vs WS error+close), WS control messages (`{"type":"reset|end|vad_start|vad_stop"}`
  with `finalize` flag default), WS frame conventions (text JSON + binary PCM int16 LE 16kHz mono,
  10 MiB max), WS close codes (1000/1001/1003/1008/1009/1011/1013) per scenario, **(v5 corrected
  audit scope)** the `vad_stop` debounce semantics — confirm there is NO server-side Silero (the
  v5 correction predicts zero hits for `grep -i silero src/nemotron_speech/server.py`), document
  the exact debounce state machine + the `silence0_warm200` config (FINALIZE_SILENCE_MS=0,
  VAD_WARMUP_MS=200) semantics including what cancels a pending finalize,
  `finalize_timing` exact key set + value types, error frame format. Output drives all subsequent
  steps' contract decisions. This step is markdown-only (no code).
  Key files: `src/nemotron_speech/server.py` (read), `reviews/server-py-protocol-audit.md` (write).

- [x] **2. Dockerfile libssl-dev + vendor nlohmann::json + picohttpparser** (Opus review)
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

- [x] **3. Part A v1 salvage audit + CMakeLists library carve + lib/{session,runtime_io,admission,scheduler,telemetry}
       skeleton + density_main migration** (PAIRED REVIEW)
  Per v4 §II + §XII. **Salvage audit FIRST** (v2 fold per both reviewers — explicit gate, not
  hand-wavy):
  list every hunk in the in-flight Part A v1 commit (`bdajesege`'s commit, once landed) →
  classify each as `keep-mechanical` (CMake target shape, file moves, include-path fixes, no-
  behavior cleanup matching v4) / `discard-protocol` (WS routing/handshake/route stubs predating
  v4) / `discard-public-api` (WS skeleton + StatsCollector if shaped non-Python). Output to
  `reviews/part-a-v1-salvage-audit.md`. Then proceed with the carve.
  Carve: `add_library(nemotron_runtime STATIC ...)` in `runtime/cpp/CMakeLists.txt`. Initial
  library content (mechanical first pass): `lib/session/session.cpp` (a moved copy of
  `session_main.cpp`) + `lib/session/session.h` exposing the public surface (`SessionState`,
  `SessionMode`, `FinalizeFinish`, `Tokenizer`, `WorkerContext`, `AOTIArtifacts`) + `lib/admission/`
  (move `density_admission.{h,cpp}`) + `lib/scheduler/` (move `batched_steady_scheduler.{h,cpp}`
  + `steady_batch_primitive.h` — extract its SHA256+JSON parser helpers to `lib/runtime_io/`).
  Migrate `density_main.cpp` to `target_link_libraries(density_main nemotron_runtime ...)` instead
  of `#include "session_main.cpp"`. **Static-global audit (v4 §II step 1.5)**: identify any static
  globals in `session_main.cpp` (tokenizer caches, finalize bucket loaders, AOTI handles); output
  to `reviews/session-cpp-static-globals.md` listing every static + disposition (stays in session.cpp
  / transfer to SharedRuntime in Step 4 / irrelevant). Leave statics in place for this step;
  transfer is Step 4. Bar: b2-t1 4-row PASS (0 token/0 event), density-sweep N=4 OFF smoke PASS,
  stalegen-smoke PASS, admission-smoke PASS, stats-smoke PASS, **N=200 session gate** (per global
  rule) PASS. If ANY smoke regresses, STOP this step.
  Key files: `runtime/cpp/CMakeLists.txt`, `runtime/cpp/lib/session/session.{h,cpp}` (NEW; carved),
  `runtime/cpp/lib/admission/`, `runtime/cpp/lib/scheduler/`, `runtime/cpp/lib/runtime_io/io.{h,cpp}`,
  `runtime/cpp/density_main.cpp` (migrate include).

- [x] **4. lib/session/runtime: SessionRuntime + SharedRuntime + PCMFrame + WireEvent + SessionTiming
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
  Bar (v2 fold per both reviewers' tightening): **wrapper-equivalence harness** — feed real
  20ms PCM chunks from the b2-t1 4-row fixture through `SessionRuntime::append_pcm_and_drain` +
  `SessionRuntime::finalize_now`; assert token/event/delta equality vs `finalize_ref` for those
  4 rows (NOT just "1 synthetic frame" which was brittle). Specifically: `--mode runtime-smoke`
  exercises the SessionRuntime wrappers; tokens + events match finalize_ref's outputs for the
  4-row fixture (0 token / 0 event divergence). Plus the global smoke set re-run.
  **Static-global transfer audit gate**: `grep -rE "^static [^ ]+ [a-z]" lib/session/session.cpp`
  returns only `static constexpr` / `static inline` / verified-OK statics from
  `reviews/session-cpp-static-globals.md`; every other static is transferred to SharedRuntime
  ownership.
  Key files: `runtime/cpp/lib/session/runtime.{h,cpp}` (NEW), `runtime/cpp/density_main.cpp`
  (add `--mode runtime-smoke`).

- [x] **5. lib/telemetry/StatsCollector — Python-exact contract** (Opus review)
  Per v4 §IV. Add `runtime/cpp/lib/telemetry/{session_timing.h, stats_collector.{h,cpp}}`.
  `StatsCollector::record(SessionTiming, bool emitted)` with the v4 completion predicate
  (`!was_suppressed && vad_stop_to_sent_ms.has_value()`); `snapshot(last_n)` returns a `Snapshot`
  struct (Python-exact fields: `enabled, window_size, samples, since_unix, until_unix,
  emitted_in_window, suppressed_in_window, lifetime_emitted, lifetime_suppressed, metrics{5 SLO
  with per-metric count}, active_sessions_at_emit Distribution`, **`admission` sub-object with
  Python-shape fields `enabled/attempted/admitted/rejected/max_backlog/max_ready_age_ms/signal{}`
  mapped from DensityAdmission's native counters** — v2 fold per Codex Round 1 #2). `snapshot_json`
  + future `snapshot_prometheus` serializers. Lock semantics: copy deque under mutex (~µs), sort/
  serialize outside mutex. Env: `NEMOTRON_STATS_ENABLED` (default 1), `NEMOTRON_STATS_WINDOW`
  (default 2048). Quantile formula: `round(p * (n-1))` clamped to `[0, n-1]`.
  **Ownership (v3 fold per Codex Round 2 must-fold)**: `StatsCollector` is built/tested in
  ISOLATION in this step. **`SessionRuntime::finalize_now()` does NOT call `record()`** — it
  populates `SessionTiming` (accessible via `last_timing()`) and returns the `WireEvent`. The WS
  worker (Step 9) owns the call sequence after the emit decision: stale-gen check → send/drop/
  timeout decision → stamp `was_suppressed` + `emitted` flag → `StatsCollector::record(timing,
  emitted)` exactly once. Step 5's bar tests StatsCollector directly via `--mode stats-smoke`; the
  production-integration recording is Step 9's bar.
  Bar: `--mode stats-smoke` (50 synthetic finalizes; assert p50/p95/max behave; `last=N` narrowing;
  missing-field tolerance; `enabled=false` short-circuits).
  Key files: `runtime/cpp/lib/telemetry/session_timing.h` (NEW),
  `runtime/cpp/lib/telemetry/stats_collector.{h,cpp}` (NEW), `runtime/cpp/lib/session/runtime.cpp`
  (populate `SessionTiming` / `last_timing()` ONLY; **no `StatsCollector::record()` call** — v4
  fold per Codex Round 3: WS worker owns record() in Step 9), `runtime/cpp/density_main.cpp` (add
  `--mode stats-smoke`).

- [x] **6. lib/ws: handshake + framing + route dispatch** (PAIRED REVIEW)
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

- [x] **7. ws_server.cpp skeleton + --selftest-and-exit smoke matrix + HTTP admin handler pool +
       /health Python enum + pid/process_label + grouped config table** (Opus review)
  Per v4 §VII + §XI + §XII + §XIII + §XV + §XVI. **Discard the v1 ws_server.cpp** (Part A v1
  superseded). New `runtime/cpp/ws_server.cpp` with:
  - **CLI**: `--port` REQUIRED (NO compiled default per v4 §XIII MPS protection);
    `--admission-active-cap` REQUIRED-or-env; `--admission-backlog-cap`; `--steady-batch-dir`;
    `--process-label`; `--selftest-and-exit`.
  - **HTTP admin handler pool** (v4 §XI; v2 fold per Codex Round 1): fixed size 2 worker threads,
    bounded queue depth 16, serves /health + /stats + /scheduler_telemetry. Accept/router thread
    dispatches; bounded queue prevents /stats poller DoS.
  - **Construct**: `SharedRuntime` + `DensityAdmission` + `StatsCollector` + (when env-enabled)
    `BatchedSteadyScheduler`; bind HTTP+WS listener via `lib/ws/` routes.
  - **Route stubs** (Python-exact shapes per WS-A1 audit + v4 §VII):
    - `GET /health` → `{"status":"healthy"|"loading"|"draining"|"degraded","model_loaded":<bool>,
      "pid":<int>,"process_label":<string?>}` (v2 fold: unified enum per v4 §VII; pid/process_label
      per v4 §XIII).
    - `GET /stats[?last=N]` → `StatsCollector::snapshot_json(?last)` (includes admission
      Python-shape per Step 5 + pid/process_label per v4 §XIII).
    - `GET /scheduler_telemetry` → scheduler's `telemetry_snapshot()` JSON.
    - `WS /` accepts handshake + sends `{"type":"ready"}` + closes WS-1011 not-implemented (Step 9
      replaces with the real lifecycle).
  - **v4 §XVI grouped operator config table**: emit a `--print-config` flag (or include in /health)
    returning the grouped env+CLI table (runtime/admission/stats/WS/shutdown/VAD knobs).
  - **`--selftest-and-exit` matrix per v4 §XV** (v2 fold per Codex Round 1 — explicit handshake-only
    deferral): default env / `NEMOTRON_STATS_ENABLED=0` / invalid `NEMOTRON_STATS_WINDOW=abc` /
    scheduler-ON with valid artifacts / scheduler-ON with missing MANIFEST / `--port 0` / invalid
    cap / 1 /health + 1 /stats + **1 WS handshake-only** (lifecycle row "1 WS handshake + 1 PCM
    frame + clean close" deferred to Step 9 which upgrades the matrix) / cap=1 + 2 connections
    (gets HTTP 503) / malformed first line / oversize headers / **2 ws_server instances on
    different ports both /health PASS** (v4 §XIII MPS readiness smoke per v2 fold). Bar:
    `--selftest-and-exit` exits 0 for clean cases, non-zero with diagnostic for failure cases;
    plus the global smoke set re-run.
  Key files: `runtime/cpp/ws_server.cpp` (REWRITE; discard v1), `runtime/cpp/CMakeLists.txt`
  (ws_server target).

- [x] **8. Client vad_stop debounce + finalize trigger** (Opus review — v5 demoted from PAIRED)
  Per v5 §VI (correction). **NO Silero in the C++ server** — the Python server has no server-side
  Silero either; client (Pipecat) runs the VAD and sends `vad_stop` control messages, which the
  server consumes with a debounce + cancellation hold. Implement the per-session state machine:
  - **State**: `IDLE | SPEAKING | PENDING_FINALIZE(deadline_ts)`.
  - **`vad_start`** → `state=SPEAKING`; cancel any pending finalize timer.
  - **`vad_stop`** → if `NEMOTRON_FINALIZE_SILENCE_MS == 0`: invoke `finalize_now()` immediately.
    Else: `state=PENDING_FINALIZE(now + FINALIZE_SILENCE_MS)`, arm timer (cancellation-hold-aware).
  - **`vad_start` while `PENDING_FINALIZE` within `NEMOTRON_VAD_WARMUP_MS`** → cancel pending,
    `state=SPEAKING` (the bounce case).
  - **Timer fires (no cancellation in WARMUP)** → invoke `finalize_now()`, `state=IDLE`.
  - **`end` / `reset` with `finalize=true`** → invoke `finalize_now()` immediately (control-message
    lifecycle takes precedence; no debounce).
  Implementation: per-session deadline stored in `SessionRuntime`; the per-connection worker's
  recv-loop uses `poll()` / `select()` with a deadline (no separate timer thread — keeps the
  threading model unchanged from §XI). Env: `NEMOTRON_FINALIZE_SILENCE_MS` (default 0),
  `NEMOTRON_VAD_WARMUP_MS` (default 200) — names per v5 §VI now precisely describe the debounce
  semantics. `NEMOTRON_VAD_MODE` dropped (no server-side VAD to enable/disable).
  Bar (v5 — Silero CPU probe + thread-pool sizing removed):
  - **(state machine)** `--mode vad-smoke` runs synthetic event sequences exercising all 6
    transitions (vad_start, vad_stop immediate, vad_stop debounced, vad_start cancels pending,
    timer-fires-no-cancel, end/reset-bypasses-debounce); asserts `finalize_now()` invocation count
    + timing match the state machine.
  - **(default config parity)** confirm `FINALIZE_SILENCE_MS=0` invokes finalize immediately on
    `vad_stop` (no debounce) — matches Python's default + the production silence0_warm200 config.
  - **(global)** re-run b2-t1 4-row + density-sweep N=4 OFF + stalegen-smoke + admission-smoke +
    stats-smoke + N=200 session gate (per global rule).
  Key files: `runtime/cpp/lib/session/runtime.{h,cpp}` (state machine + timer hook),
  `runtime/cpp/density_main.cpp` (add `--mode vad-smoke`). (No `shared.cpp` change — no Silero
  model load. No selftest matrix change — no `vad_load_failure` row needed.)

- [x] **9. WS lifecycle wiring + stale-gen integration** (PAIRED REVIEW)
  Per v4 §V + the v3 lifecycle table. Wire `ws_server.cpp` per-connection worker: `accept` →
  `lib/ws/handshake` → query validation (`model`/`language`) → `DensityAdmission::try_admit` (if
  `SHED_*` → HTTP 503 with `{"error":"admission_backpressure"}` pre-upgrade) → construct
  `SessionRuntime` → send `{"type":"ready"}` → recv-loop (binary PCM → `append_pcm_and_drain`
  with stale-gen check; emit interim `WireEvent`s as JSON text frames with stale-gen check) →
  control messages (`reset`/`end`/`vad_start`/`vad_stop`) → `finalize_now()` (triggered by the
  Step 8 vad_stop debounce timer firing OR by `end`/`reset` control with `finalize=true`; v5: NO
  server-side Silero — client sends `vad_stop` after its own VAD; server debounces; returns
  `WireEvent` + populates `last_timing()`) → **stale-gen check** →
  emit final WireEvent (or drop silently if stale) → **stamp `was_suppressed` + `emitted` flag** →
  **`StatsCollector::record(timing, emitted)` exactly once** (v3 fold per Codex Round 2 must-fold —
  WS worker owns the record() call AFTER the emit decision; SessionRuntime never calls record()) →
  close WS-1000. Generation bumps on reset/close/shed. **Stale-gen emit-point enumeration**
  (v4 §V table): drops_at_event_emit (interim suppressed); drops_at_finalize_output (final
  suppressed); StatsCollector record sees `was_suppressed=true`.
  **Binary PCM frame validation** (v3 fold per Codex Round 2 minor): odd-length binary payloads
  close WS-1003 BEFORE constructing `PCMFrame` (per v4 §VIII).
  Bar (v2 fold per both reviewers — concrete oracle, not "asserts the correct event sequence"):
  `--mode ws-lifecycle-smoke` runs a Python client driving 1 WS connection through full lifecycle
  + asserts: (handshake) ready frame EXACTLY `{"type":"ready"}`; (binary PCM) server accepts +
  drives through SessionRuntime; (interim ordering) interim transcripts monotonically time-
  ordered, `is_final=false`, `text` non-empty; (vad_start) log-only; (vad_stop/Silero)
  finalize_now triggered + final transcript with `is_final=true`; (stale interim drop) reset bumps
  generation mid-stream → interim dropped silently + `stale_gen.drops_at_event_emit++`; (stale
  final drop) reset bumps between finalize_now() and emit → final dropped silently +
  `stale_gen.drops_at_finalize_output++` + StatsCollector::record with `was_suppressed=true`;
  (close code) clean WS-1000 on natural completion; (StatsCollector record) per-finalize record
  called with timing + emitted flag matching emit decision. Plus the global smoke set + N=200
  session gate (per global rule). The Python-compat oracle (Step 11) is the deeper check.
  **Selftest matrix lifecycle row upgrade** (Step 7 had handshake-only): "1 WS handshake + 1
  binary PCM frame + 1 control vad_stop + expect final + clean WS-1000 close" now PASSes.
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

- [ ] **11. Test oracle: run_compat.py + canonicalized diff + WS-overhead perf gate** (PAIRED REVIEW)
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
  Volatile fields stripped (v2 fold per both reviewers — exact list, NOT hand-wave):
  timestamps, finalize_timing numeric values (key set + value types ARE compared after Step 1
  pins them), sequence numbers, pid, process_label, native scheduler/admission counters. JSON
  keys sorted before diff.

  **Bar** (v2 fold per Codex Round 1 + Opus #3 + #6 — concrete + adds WS-overhead perf gate):
  - **(correctness)** for each of 8 utts (utt0..utt7): event count equal Python's; per-event
    `type` equal; per-event `text` equal; per-event `is_final` equal; per-event `finalize` flag
    equal where applicable; final `collector_text` equal (if applicable); `finalize_timing` keys
    present + value types numeric.
  - **(invalid query)** `?model=bogus` returns expected error response per Step 1 audit.
  - **(invalid `?last=`)** /stats `?last=bogus` returns expected error response per Step 1 audit.
  - **(WS-overhead perf gate — v2 fold per Codex + Opus #6, NEW)** measure `ttfs_via_ws_server`
    at N=8 (the low-load break-even from B3-FU low-load sweep). Assert `ws_overhead_p95 =
    ttfs_via_ws_p95 − ttfs_via_density_scheduler_p95 ≤ max(2ms, 10% · ttfs_via_density_scheduler_p95)`
    per v4 §IV (WS overhead must not silently regress the Phase-2 ttfs margin).
  - **(integration role)** Step 11's failure is INTEGRATION REGRESSION (most likely Step 9's
    lifecycle wiring); fix it via Step 9; do NOT un-mark Steps 8-10 (per the commit-semantics rule
    at the top of Steps).
  Key files: `tests/server_compat/run_compat.py` (NEW), possibly
  `runtime/step6_server_oracle.py` (extend), `reviews/Step3b-WS-test-oracle.md` (oracle spec doc).

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | server.py protocol audit | done | 54fbd24 | PAIRED. Codex draft (527 lines) + Opus parallel pass — no disagreements. 6 explicit v5-architecture deviations flagged: /health enum (2 vs 4 values), finalize_silence_ms default 150 vs 0, no NEMOTRON_VAD_WARMUP_MS, post-handshake 1013 vs pre-handshake 503, no Python emits 1003/1008/1011, **finalize_timing wire = 9 RAW timing keys NOT 5 derived SLO metrics**. Codex log: codex-jobs/step-01-server-py-audit-bkddcxjjd.log. |
| 2 | Dockerfile libssl-dev + vendor nlohmann::json + picohttpparser | done | e960aca | Opus review. Dockerfile +1 line (libssl-dev → OpenSSL 3.0.13). lib/runtime_io/: nlohmann/json v3.11.3 (24,765 LOC, MIT) + picohttpparser h+c (803 LOC, MIT-or-Perl). CMakeLists: project LANGUAGES C CXX + picohttpparser.c in nemotron_runtime. Smoke set 5/5 PASS (b2-t1 0/0, density-sweep N=4 OFF 0/0/0, stalegen/admission/stats), N=200 gate 0/200. Codex log: codex-jobs/step-02-vendoring-b7ikzs2hi.log. |
| 3 | CMakeLists library carve + density_main migration | done | 68eb956 | PAIRED. Codex implementation + independent Opus parallel audit — BOTH found ZERO static resource-state globals in lib/session/session.cpp (audit doc + opus-pass converged). Moved: density_admission → lib/admission/, batched_steady_scheduler + steady_batch_primitive → lib/scheduler/. Extracted SHA256+JSON helpers to lib/runtime_io/io.{h,cpp} (25+233 LOC). density_main now compiles only density_main.cpp + links nemotron_runtime. Smoke set 5/5 PASS + N=200 0/200. Codex log: codex-jobs/step-03-library-carve-b8h1l5zpx.log. |
| 4 | SessionRuntime + SharedRuntime + concrete public DTOs | done | f2f7da8 | Opus review. lib/session/runtime.{h,cpp} 110+440 LOC. PCMFrame + WireEvent + SharedRuntime + SessionRuntime + SessionConfig public API, pimpl-pattern. SessionTiming REPLACED: 5 derived metrics → 9 RAW timing keys per Step 1 audit + to_wire_json() helper; StatsCollector::record signature preserved (derives 5 SLO metrics from raw internally). Native-endian compile assert. Wrapper-equivalence harness `--mode runtime-smoke` PASS (4 rows, 95 wire events, 0/0). Smokes 7/7 PASS + N=200 0/200. Static count unchanged at 113. Codex log: codex-jobs/step-04-runtime-bz00fxoxb.log. |
| 5 | StatsCollector Python-exact | done | d8e8c29 | Opus review. New signature: record(SessionTiming, bool emitted) + set_admission(DensityAdmission*) + snapshot_json with Python-exact /stats shape per Step 1 audit. Completion predicate `!was_suppressed && vad_stop_ts && final_sent_ts`; 5 derived SLO metrics computed inside record() from raw SessionTiming. Admission mapping: direct for offered/admitted/shed_close_count/backlog_peak; stubbed for per-session ready-queue fields (Step 9 will wire). Smokes 7/7 PASS + N=200 0/200. Codex log: codex-jobs/step-05-stats-collector-bqp3ostdp.log. |
| 6 | lib/ws handshake + framing + routes | done | d43cff6 | PAIRED. Codex impl + Opus parallel review of security-sensitive paths (comma-token Connection, unmasked-frame rejection, frame-header-first anti-OOM). lib/ws/{handshake,framing,routes}.{h,cpp} 606 LOC. RFC 6455 spec vector exact match. 8 ws-lib-smoke cases PASS; smokes 7/7 + N=200 0/200. Codex log: codex-jobs/step-06-ws-lib-bj0537qdt.log. |
| 7 | ws_server.cpp skeleton + --selftest-and-exit | done | d4f4143 | Opus review. New ws_server.cpp 1603 LOC (v1 was 382 — full v5 §VII/XI/XIII/XV/XVI scope). All 12 selftest scenarios PASS. Python parity: 2-value /health enum, post-handshake WS-1013 admission shed. HTTP admin pool size=2 queue=16. Global smokes 6/6 + N=200 0/200. Codex log: codex-jobs/step-07-ws-server-bn26d6w2p.log. |
| 8 | Client vad_stop debounce + finalize trigger | done | 74f3fc8 | Opus review (v5 demoted from PAIRED). SessionRuntime VadState enum + poll_timer; +248/-8 LOC. vad-smoke 6/6 transitions PASS, 4 finalize invocations. ZERO Silero hits (v5 §VI compliance). Smokes 6/6 + N=200 0/200. Codex log: codex-jobs/step-08-vad-debounce-b2m6t118l.log. |
| 9 | WS lifecycle + stale-gen wiring | in-progress | — | PAIRED. The substantive integration: wires SessionRuntime + StatsCollector::record AFTER emit decision + stale-gen drops at interim/finalize emit-points + Step 8 debounce timer in recv-loop. |
| 10 | Graceful shutdown + backpressure | pending | — | Opus review. SIGTERM drain + frame-size header-first + ping/pong. |
| 11 | Test oracle run_compat.py + canonicalized diff | pending | — | PAIRED. Part B pre-merge gate. |
