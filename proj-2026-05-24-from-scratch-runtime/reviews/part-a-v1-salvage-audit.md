# Part A v1 Salvage Audit (per STEP3B-WS-PLAN v5 Step 3 + architecture v5 §XII)

Date: 2026-05-28
Auditor: Claude (Opus 4.7)
Architecture binding: `Step3b-WS-architecture.md` v5 (commit f624008).

## Scope

The in-flight Part A v1 work (Codex job `bdajesege`) landed on disk before the architecture
v3→v4→v5 review cycle converged. Per v5 §XII, audit each v1 hunk against v5 and classify as
**keep-mechanical** (matches v5; mechanical extraction / CMake / minor cleanup), **keep-with-note**
(matches v5 in spirit but needs revision in a later PLAN step), **replace-in-later-step** (starting
point only; PLAN step rewrites the contract), or **discard-protocol/public-api** (predates v5 audit;
delete and re-do).

## v1 Inventory

| File | LOC | Disposition |
|---|---|---|
| `runtime/cpp/session_main.cpp` (modified, 9 lines) | wrapper shim | **KEEP** (as-is) |
| `runtime/cpp/lib/session/session.h` (new, 287 lines) | carved public surface | **KEEP-WITH-NOTE** (as-is for now; Step 4 adds `runtime.h` alongside) |
| `runtime/cpp/lib/session/session.cpp` (new, 4668 lines) | mechanical move of session_main.cpp contents | **KEEP** (as-is) |
| `runtime/cpp/lib/telemetry/session_timing.h` (new, 15 lines) | v1 SessionTiming struct | **REPLACE-IN-STEP-4** (starting point; missing v5 §II fields) |
| `runtime/cpp/lib/telemetry/stats_collector.h` (new, 32 lines) | v1 StatsCollector header | **REPLACE-IN-STEP-5** (starting point; not Python-exact) |
| `runtime/cpp/lib/telemetry/stats_collector.cpp` (new, 160 lines) | v1 StatsCollector implementation | **REPLACE-IN-STEP-5** (starting point; not Python-exact) |
| `runtime/cpp/CMakeLists.txt` (modified, 103 lines) | library target + ws_server target | **KEEP-WITH-FIX** (remove ws_server target until Step 7 rebuilds it) |
| `runtime/cpp/density_main.cpp` (modified, +82/-9) | include-path migration + stats-smoke wiring | **KEEP-WITH-NOTE** (include-path migration ✓; stats-smoke revisit in Step 5) |
| `runtime/cpp/ws_server.cpp` (new, 382 lines) | v1 WS server skeleton | **DISCARD-PROTOCOL** (predates v5; delete) |
| `.gitignore` (modified) | un-ignore `runtime/cpp/lib/` paths | **KEEP** (necessary for lib/ tree to be tracked) |

## File-by-file classification

### KEEP (v5-compliant; commit alongside the discard)

**1. `runtime/cpp/session_main.cpp` (9 lines)** — KEEP as-is

Thin 9-line wrapper: `#include "lib/session/session.h"` + `main()` calls
`session_main_entrypoint(argc, argv)`. This is exactly v5 §II's library boundary contract:
density_main and ws_server share the runtime library; session_main.cpp stays as a compat shim
for the existing test gates (1.4 session composition gate + downstream). No changes needed.

**2. `runtime/cpp/lib/session/session.cpp` (4668 lines)** — KEEP as-is

Mechanical move of `session_main.cpp` content into `lib/session/`. Implementation only; no public
API redefinition. Step 4+ will incrementally evolve (static-global transfer to `SharedRuntime`).

**3. `runtime/cpp/CMakeLists.txt` (103 lines)** — KEEP the library target; **REMOVE the
`ws_server` target** for now

- `add_library(nemotron_runtime STATIC lib/session/session.cpp lib/telemetry/stats_collector.cpp)`
  ✓ matches v5 §II structure.
- `density_main`, `session_main` migrated to `target_link_libraries(... nemotron_runtime ...)` ✓
  matches v5 §XII contract.
- **PROBLEM**: lines 93-95 register `ws_server` target using the v1 `ws_server.cpp`. After we
  discard v1 ws_server.cpp (item 9 below), the build target must be removed too. Step 7 in the
  PLAN re-adds `ws_server` with the new (Python-shape, picohttpparser-based) ws_server.cpp.
- **Action**: Remove lines 93-95 (the `add_executable(ws_server ...)` + `target_link_libraries`
  + `set_target_properties` for ws_server). Step 7's build re-introduces it with new file.

**4. `runtime/cpp/density_main.cpp` diff (+82/-9)** — KEEP

- Include-path migration: `#include "session_main.cpp"` → `#include "lib/session/session.h"` + 
  `#include "lib/telemetry/stats_collector.h"` ✓ exactly v5 §XII requirement.
- Adds `--mode stats-smoke` to the harness ✓ this is PLAN Step 5's bar (verifies the
  StatsCollector in isolation).
- Adds optional `StatsCollector* stats_collector` parameter to `run_finalize_density` →
  density-harness-side recording for the smoke test only. Not the production integration pattern
  (per v5 §V the WS worker owns `record()` AFTER the emit decision). Acceptable as a smoke harness;
  Step 5's StatsCollector revision will preserve this wiring with the v5 contract.
- **NOTE**: when Step 5 replaces `StatsCollector::record(timing)` with `record(timing, emitted)`,
  this density_main caller will need a small fix to pass `emitted=true` (smoke records are
  always "emitted" by construction).

**5. `runtime/cpp/lib/session/session.h` (287 lines)** — KEEP-WITH-NOTE as-is for now

This is the carved public-surface header. Contains: `SessionState`, `SessionMode`,
`FinalizeFinish`, `Tokenizer`, `AsrSnapshot`, `AudioGeometry`, `ManifestContract`,
`ManifestBucket`, `BucketManifest`, `BucketConstants`, `FinalizeOutcome`, `EmittedEvent`,
`TokenMargin`, plus ~30 free function declarations.

**v5 alignment note**:
- v5 §II distinguishes PUBLIC (the small surface for production: `SessionRuntime`,
  `SharedRuntime`, `PCMFrame`, `WireEvent`, `SessionTiming`) from PRIVATE (the broad internal
  types like `EmittedEvent`, `TokenMargin`, `AsrSnapshot`, `BucketConstants`, plus the harness
  helpers like `gold_events_from_bundle`, `tensor_close`).
- v1 session.h dumps EVERYTHING (mechanically reflecting session_main.cpp's organization).
  This is fine as a starting point — density_main + the existing 1.4 gate need every symbol.
- **Step 4 ADDS `lib/session/runtime.{h,cpp}`** alongside this file, which defines the v5
  public surface. session.h stays as the "internal umbrella" used by density_main and the
  harnesses.
- **Eventual cleanup** (future work; not in Step 3): once `runtime.h` is the canonical
  production surface, session.h can be moved to `lib/session/internal/session.h` or split into
  smaller headers (the harness-only `gold_events_from_bundle` etc. lift out). NOT required for
  Step 3b ship; tracked as a TODO in the file.
- **No code changes** needed in this step; the carve itself is correct.

### REPLACE in later PLAN steps (v1 contract not v5-compliant; starting point only)

**6. `runtime/cpp/lib/telemetry/session_timing.h` (15 lines)** — REPLACE in Step 4

```cpp
// v1 (current)                         // v5 §II (target)
struct SessionTiming {                  struct SessionTiming {
  optional<double> vad_stop_to_sent_ms;   optional<double> vad_stop_to_sent_ms;
  optional<double> fork_flush_wall_ms;    optional<double> fork_flush_wall_ms;
  optional<double> vad_stop_recv_to_process_ms;  optional<double> vad_stop_recv_to_process_ms;
  optional<double> lock_wait_ms;          optional<double> lock_wait_ms;
  optional<double> vad_stop_to_finalize_start_ms;  optional<double> vad_stop_to_finalize_start_ms;
  uint64_t finalize_seq = 0;              uint64_t finalize_seq = 0;
  int active_sessions_at_emit = 0;        int active_sessions_at_emit = 0;
};                                        // v5 adds:
                                          bool was_suppressed = false;
                                          double emit_unix_ts = 0.0;
                                          optional<string> close_reason;
                                        };
```

Step 4 (in PLAN: SessionRuntime + SharedRuntime + concrete public DTOs) adds the three missing
fields when it pulls `SessionTiming` into `lib/session/runtime.h` per v5 §II concrete signatures.

**7. `runtime/cpp/lib/telemetry/stats_collector.{h,cpp}` (32 + 160 lines)** — REPLACE in Step 5

v1 is a reasonable starting point but **not Python-exact** per v5 §IV. Concrete deltas:

- `record()` signature: v1 takes `SessionTiming` only; v5 §IV requires
  `record(SessionTiming, bool emitted)`. The `emitted` flag is needed for the
  `emitted_in_window` vs `suppressed_in_window` tallies + lifetime counters.
- Completion predicate: v1 records ALL inputs unconditionally; v5 §IV requires
  `COMPLETE = !timing.was_suppressed && timing.vad_stop_to_sent_ms.has_value()` — incomplete
  records increment `lifetime_suppressed` only, do not enter the deque.
- `emitted_in_window` / `suppressed_in_window`: v1 hardcodes
  `emitted_in_window = samples.size()` and `suppressed_in_window = 0` — wrong. v5 §IV: counts
  derived from per-sample `emitted` flag within window.
- `lifetime_emitted` / `lifetime_suppressed`: v1 increments only `lifetime_records`; v5 §IV
  splits these.
- **Missing `admission` sub-object** (Python-shape; mapped from `DensityAdmission`'s native
  counters): v5 §IV requires `{enabled, attempted, admitted, rejected, max_backlog,
  max_ready_age_ms, signal{}}`. v1 has no admission block at all.
- **Missing per-metric `count`**: v1 quantile_summary includes `count` per metric ✓ matches v5
  (good — this part is right).
- Quantile formula: v1 uses `llround(p * (n-1))` ✓ matches v5 (good).
- Env vars: v1 reads `NEMOTRON_STATS_ENABLED` + `NEMOTRON_STATS_WINDOW` ✓ matches v5 §XVI.
- Lock semantics: v1 already does "copy under lock, sort outside" ✓ matches v5 §IV.

Step 5 keeps the v1 structure + lock pattern + env-var loading, but rewrites `record()` +
`snapshot_json()` to the v5 §IV contract.

### DISCARD (predates v5 audit)

**8. `runtime/cpp/ws_server.cpp` (382 lines)** — DISCARD; PLAN Step 7 rewrites

Per v5 §XII: "do NOT use Part A's WS skeleton/handshake/route stubs as the baseline; they
predate the v3 server.py audit. Re-do those parts on v3" (v3 = v4 = v5 for §XII purposes).

Concrete violations of v5 contracts:

| v1 ws_server.cpp | v5 contract violation |
|---|---|
| `/health` returns `{"status":"ok","admission":{...}}` | v5 §VII: `{"status":"healthy|loading|draining|degraded","model_loaded":<bool>,"pid":<int>,"process_label":<string?>}` — different enum + missing pid/process_label + admission block doesn't belong in /health (it's in /stats per v5 §IV) |
| `/stats` returns `{"status":"todo"}` | v5 §IV: full StatsCollector Python-exact response |
| No `/scheduler_telemetry` endpoint | v5 §VII requires it |
| WS handshake validates ONLY `Sec-WebSocket-Key` | v5 §VII requires full HTTP request-line parsing via picohttpparser: method/path/Upgrade/Connection/version, case-insensitive headers, body-on-GET rejection, etc. |
| Hand-rolled WS-1011 close frame | v5 §VI/§VIII: use `lib/ws/framing.cpp` (Step 6) with proper RFC6455 framing |
| `--port` defaults to 8080 | v5 §XIII: NO compiled default — server logs warning + exits non-zero (MPS protection) |
| No CLI for `--process-label`, `--steady-batch-dir`, `--admission-active-cap REQUIRED` enforcement | v5 §XVI grouped config table |
| `--selftest-and-exit` calls accept_once + `selftest_http_get` from a worker thread | OK for a basic smoke; v5 §XV requires the full smoke matrix (env-config combos, scheduler-ON case, malformed/oversize headers, port=0 auto-bind, etc.) |
| `RuntimeServer` constructor takes admission caps + stats; binds + accepts forever | v5 §XI requires HTTP admin handler pool (fixed size 2, bounded queue 16); v5 §XII Part A skeleton sends `{"type":"ready"}` + closes WS-1011 not-implemented (Step 9 replaces with real lifecycle) |
| `is_ws && (path == "/" || path == "/v1/transcribe")` | Audit (Step 1) confirms Python uses `GET /` only; "/v1/transcribe" is a v1 invention |
| No admission check before WS handshake | v5 §VII: pre-handshake admission → HTTP 503 with `{"error":"admission_backpressure"}` |
| No frame-header-first size check | v5 §VIII WS-1009: must reject > 10 MiB before reading payload |

**Action**: delete the file. Step 7 rewrites from scratch using:
- picohttpparser (Step 2/3 vendored).
- `lib/ws/handshake.cpp` (Step 6 — paired-reviewed RFC6455 + HTTP-semantics conformant).
- `lib/ws/routes.h` (Step 6).
- v5 §XV smoke matrix (Step 7's bar).
- v5 §XVI grouped config table.
- v5 §XIII no-default `--port`.

## Required pre-/implement action items (FOR ME, not for the PLAN steps)

1. **Delete `runtime/cpp/ws_server.cpp`** — discard the v1 WS skeleton.
2. **Edit `runtime/cpp/CMakeLists.txt`**: remove lines 93-95 (the `add_executable(ws_server ...)`
   + `target_link_libraries` + `set_target_properties` for ws_server). Step 7's build re-adds it
   with the new file.
3. **Verify build still works** (density_main + session_main): `cmake --build build --target
   density_main session_main` in-container — must compile clean against the v1 KEEP/KEEP-WITH-NOTE
   pieces.
4. **Commit the salvage** with a clear message: "part-a-v1 salvage: discard v1 ws_server.cpp +
   keep mechanical carve (lib/{session,telemetry} + CMake library target + density_main include
   migration); Step 5 + Step 7 will rewrite telemetry contract + WS server".

After these 4 steps, the v1 work is salvage-audited and `/implement STEP3B-WS-PLAN.md` can
start cleanly. Steps 4-5 will revise the telemetry contract; Steps 6-7 will rebuild the WS
server from scratch per v5.

## What this audit is NOT

This audit does not:
- Re-validate the v1 mechanical move (it compiled; trust the v1 process).
- Re-design the public API (v5 §II is the binding spec; this audit only checks v1 vs that spec).
- Block the PLAN's later steps' paired reviews — Step 4 + Step 5 + Step 6 + Step 7 each get
  their own review per the PLAN's decision-criticality matrix.

## Net

| Category | Count |
|---|---|
| KEEP (compilable, v5-compliant, no changes needed in this audit) | 4 files |
| KEEP-WITH-NOTE (compilable; revisit in a later PLAN step) | 2 files |
| KEEP-WITH-FIX (small edit needed in this audit pass) | 1 file (CMakeLists.txt) |
| REPLACE-IN-LATER-STEP (starting point; contract rewrites in Step 4/5) | 3 files |
| DISCARD (delete; Step 7 rewrites) | 1 file (ws_server.cpp) |
| TOTAL | 11 files |

**Net judgment**: Part A v1's mechanical carve (library boundary, CMake target, density_main
migration) is **right and reusable**. The protocol skeleton (ws_server.cpp) and the
telemetry contract (StatsCollector + SessionTiming) are **starting points that will be
rewritten in PLAN Steps 4/5/7** per the v5 §II/§IV/§VII/§XII contracts. The PLAN's per-step
test protocol (build → harness → smoke set → N=200) catches any regression during the
rewrites; nothing here needs more upfront work.
