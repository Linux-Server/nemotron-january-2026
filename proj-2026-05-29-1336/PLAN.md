# Plan: Wire density-parity scheduled inference into ws_server

Project directory: `./proj-2026-05-29-1336`

_Revision history: **v5 — CONVERGED** after 4 paired adversarial review rounds (Codex + Opus). Issues tagged `[R1:*]`/`[R2:*]`/`[R3:*]`/`[R4:*]`. v4→v5 (round 4): the lane model is now **session-bound** = density's `WorkerContext` precisely (a session is bound to one lane for its lifetime — per-session lane affinity — not a per-call free-lane pool; resolves the cross-lane stream-ownership hazard); K = admission `active_cap`; named the dead `*_long_check` duplicate handles; disable legacy batch-steady env for the clean Milestone-A checkpoint. Round-4 verdict: all R1-R3 fixes substantively present; remaining items were the session-binding (now fixed) + 2 minor nits (now applied) → converged._

## Context
ws_server runs all inference on a single warm `InferenceExecutor` thread (fixed cold-thread finalize 230→8 ms) that **serializes** concurrent sessions → Step-11 perf-gate N=8 tail (ws_p95 42.7 vs density 13.9 ms; per-session ttfs `[…19.7,20.3,42.7]`). This plan replaces that single thread with density's substrate: a pool of **warm, isolated InferenceLanes** (each owns a stream + its own `joint`/`predict`/`preproc`), the **BatchedSteadyScheduler** batching steady-encode across lanes, and a **shared finalize loader pool** — so concurrent inference runs in parallel, *provably* matching the inline/Python output via a shadow parity gate. Cross-refs `../proj-2026-05-24-from-scratch-runtime/STEP3B-WS-PLAN.md` (deferred Step-9); gate = `tests/server_compat/run_compat.py`.

## Reference implementations
- **density worker substrate** `[R2:C1]`: N warm worker threads, **each with its own `joint`/`predict`/`preproc`** (`density_main.cpp:1116,1149`) + a non-blocking stream (`stream_for_worker` → `cudaStreamCreateWithFlags(cudaStreamNonBlocking)`), all warmed before measurement (`density_main.cpp:5211,5251`); shared `enc_first` is **locked** (`density_main.cpp:1486`, `run_first_encoder` exported `session.h:287`, density call `:1493`).
- **Steady scheduler usage**: `density_main.cpp:1580-1615` (build `BatchedSteadyInput`, producer event on worker stream, `enqueue`, `wait_for(future_timeout_ms())`, `cudaStreamWaitEvent(stream, completion)`, consume `row_tensors`); diagnostic compare syncs the stream after (`density_main.cpp:1623`).
- **Finalize loader sharing** `[R2:M6]`: density preloads only needed buckets + tracks loader-delta MB (`density_main.cpp:1793,1846,5135`); finalize concurrency comes from `num_runners` on a **shared** loader, not duplicated loaders.
- **Parity reality** `[R1:M6]`: batched b1 is SHA-different from production `enc_steady_aoti.pt2` (`steady_b_artifacts/MANIFEST.json:85`), `B2_A1_PARITY` tolerance ≤5e-2 (`density_main.cpp:2836`), divergences counted-not-gated (`:2156`). → parity must be *proven per live rows* (Step 5), never assumed.
- **Scheduler hazards** `[R2:C2]`: `enqueue` can block on capacity (`batched_steady_scheduler.cpp:93`), `future_timeout_ms()`≥1000 (`:141`), dispatcher exceptions `std::exit(1)` (`:225,257`).

## Current state
- `runtime/cpp/lib/session/runtime.cpp`: single `InferenceExecutor` (warm thread, blocks on `future.get()` ~49,524); shared `preproc/joint/predict/enc_first` (~365,375); `warm_inference_executor()` warms only that one thread (~414); gated scheduler block (~389, false); eager finalize-loader load with `num_runners` (~378); session dtor only bumps generation (~668).
- `runtime/cpp/lib/session/session.h:300` runtime APIs take **no stream**; `:287` `run_first_encoder` exported (density uses it). `session.cpp`: preproc on thread's current stream (~2045), steady AOTI `loader.run(inputs)` (~1829), finalize AOTI `loader_it->second->run(inputs)` no stream (~2866); steady apply mutates state + emits (~2304,2314,2330); `run_finalize_runtime` mutates `parent.mode=FINALIZED` **before** clone (~2848), collector commit pre-gen-check (~2883,2902); `session_runtime_finalize` mutates true-boundary audio before input prep (~2924); `clone_session` clones tensors (~1015); decode scalar `.item()` syncs (~1721).
- `runtime/cpp/ws_server.cpp`: `scheduler_enabled=false` (~1744); **telemetry-only `state_->scheduler` already exists** (~829,988); `finalize_num_runners=read_env_int("NEMOTRON_DENSITY_FINALIZE_RUNNERS",1)` (~604,612); per-connection thread ~1917; stale-drop only at output flush (~1256).
- Validation: `run_compat.py` correctness-vs-Python + WS gate `ws_p95-density_p95≤max(2ms,10%)` (~933); env `NEMOTRON_DENSITY_BATCH_STEADY=1` (~670). `density_main --mode b2-t1 / runtime-smoke`.

## Rules

### Correctness / parity `[R1:M6][R2:M5,m8]`
- NEVER say "byte-exact/byte-identical" for the scheduler steady path — it's "tensor diff within tolerance (≤5e-2 documented) AND token/event identical." Prove it per live rows via the Step-5 shadow gate (clone pre-decode state, decode/apply BOTH inline+scheduler into separate clones, compare tensors+tokens+events, commit exactly one path; shadow-mode timing is invalid for perf). End gate: oracle correctness vs Python stays 8/8 event-for-event + shadow report recorded.
- `b2-t1` + `runtime-smoke` stay 0 divergences every step.

### Execution substrate — track density's proven `WorkerContext` model precisely `[R2:C1][R1:C1/C3][R4:Opus]`
- The `InferenceLane` IS density's `WorkerContext` (`density_main.cpp:1116`): a thread + a non-blocking CUDA stream + its OWN `joint`/`predict`/`preproc` instances. Shared `enc_first` stays locked. A lane never shares mutable module state with another lane.
- **SESSION-BOUND, not free-pooled** (this is how density works — sessions are assigned to a worker, `density_main.cpp:3097`, and that worker runs ALL of the session's inference — steady + finalize + decode — on its own stream/modules). Each admitted session is **bound to one lane for its entire lifetime**; do NOT bounce a session's successive chunks across lanes (its GPU state — mel ring, caches `clc/clt/clcl`, decoder `g/h/c` — is produced on the lane's stream; cross-lane hopping forces cross-stream sync hazards density never has). enc_first (first chunk) is the only shared+locked exception.
- **K = concurrent-session capacity = admission `active_cap`**: admit a session only when a lane is free (this unifies the lane pool with existing admission control — density's N workers = N concurrent sessions). K is the SLO-robust concurrency (memory notes: ~16-20/L40S), capped; `NEMOTRON_WS_LANES=K`.
- Submit-to-the-session's-lane is the only way inference runs; no model op runs on a cold connection/WS thread.

### Streams `[R1:C2/C4][R2:M4]`
- Thread an explicit stream/`ExecutionContext` through the runtime inference path as **overloads** — preserve the exported `session.h` signatures density uses (`run_first_encoder`), or update every call site; `density_main` MUST build whenever `session.h` changes. Producer/completion events recorded on the stream that owns the tensors. AOTI runs stream-aware (`loader.run(inputs, stream)`).

### Memory `[R2:M6][R3:M4]` — affordable; hard rule
Measured (artifacts) — **per-lane (duplicated) modules are small; the big modules are SHARED**:
| object | size | per-lane or shared |
|---|---|---|
| joint_step.ts | 6.6 MB | per-lane |
| predict_step.ts | 27.6 MB | per-lane |
| preproc.ts | 0.1 MB | per-lane |
| **per-lane total** | **~34 MB/lane** | (K=8 → ~272 MB) |
| enc_first.ts | 2.36 GB | **shared** (locked) |
| enc_steady_aoti.pt2 | 2.36 GB | **shared** (scheduler) |
| stripped_finalize_buckets | 125 MB | **shared** (loader pool) |
(density N=8 measured `worker_context_delta_per_worker=37.7 MB`, peak 13.5 GiB — consistent.)
- **Hard rule**: per-lane owns ONLY a stream + `joint`/`predict`/`preproc`. `enc_first`, `enc_steady`/scheduler, and the finalize bucket loaders are SHARED — never duplicated per lane. Finalize AOTI loaders = ONE shared `FinalizeBucketLoaderPool` with `num_runners=F`. Log actual MB; K is affordable, but assert no big-module duplication.
- `[R4:m1]` While here, the existing dead `enc_first_long_check` (2.36 GB) + `enc_steady_long_check` (2.36 GB) handles in `SharedRuntime::Impl` (runtime.cpp:368,372 — loaded, unused; flagged in STEP3B-WS-PLAN cold-load follow-up) should be removed/reused, and MUST NOT be duplicated per lane. Confirm they're gone or shared, not multiplied.

### Config `[R1:m11][R2:m9][R3:M3]` — K and F are SEPARATE
- `NEMOTRON_WS_LANES=K` (lane-pool size) is **independent** from `NEMOTRON_WS_FINALIZE_RUNNERS=F` (shared finalize loader `num_runners`, default `capped_general_finalize_runners(active_cap)`, density caps at 2) and `NEMOTRON_WS_SCHEDULER` (default on). Do NOT tie F to K. Test a small K×F matrix. Each env added MUST update the config-print AND `run_compat.py` env in the SAME step.

### enc_first lock `[R3:M5]`
- The shared `enc_first` (2.36 GB) stays locked → the FIRST chunk of each utterance serializes. This is costly under burst (density N=8 `enc_first_lock_wait` p95 ≈ 42 ms, N=16 ≈ 97 ms). It does NOT affect the measured finalize ttfs gate (vad_stop→final), but report `enc_first_lock_wait` separately and do NOT claim "tail closed" if first-chunk lock ever dominates a measured TTFS. Out of scope to fix here (density-parity); flag if it becomes the binding cost.

### Warmth `[R2:M7]`
- Warm EVERY lane (its stream + its joint/predict/preproc + representative finalize buckets) and the scheduler dispatcher BEFORE any session receives `ready`. Log lane-count + bucket coverage.

### Async lifecycle / safety `[R2:C2/C3][R1:M8/M9]`
- **Finalize is synchronous on a lane** (WS thread submits, lane runs, returns after commit) — so no finalize future outlives the request; the async snapshot/future machinery below is required ONLY for the steady scheduler.
- Split finalize into `FinalizeInputSnapshot` → pure lane compute (on clone) → `FinalizeCommit` (mutate parent only here, under a generation check). Move the pre-clone `parent.mode=FINALIZED` (`:2848`) and true-boundary audio mutation (`:2924`) into the snapshot/commit boundary so a stale generation can't corrupt parent state.
- Steady scheduler path owns its async safety AT THE SAME STEP it's introduced. Note `enqueue` currently blocks **indefinitely on capacity before a future exists** (`batched_steady_scheduler.cpp:93`) — the `future_timeout_ms()` only bounds the wait AFTER enqueue returns. So a concrete **`try_enqueue_until(deadline)`** (bounded admission) is required, plus future timeout, plus replacing the dispatcher's `std::exit(1)` (`:225,257`) with fault propagation to a server error/controlled drain. Producer/completion events RAII; per-lane in-flight futures drained before lane/stream teardown. Lanes are pool-owned (outlive sessions), so session destruction just bumps generation and the lane drops the stale result.

### Milestones / de-risking `[R3:M6]`
- **Milestone A = the finalize tail (Steps 1-3):** lane substrate + synchronous finalize on it. This alone is expected to close the measured N=8 perf-gate tail — validate the perf gate at the end of Step 3 BEFORE touching the scheduler. If A passes the gate, the steady scheduler is a throughput enhancement, not gate-critical.
- **Milestone B = steady throughput (Steps 4-6):** single scheduler owner + shadow gate + steady→scheduler routing (the async-heavy part). Kept after A so the gate is de-risked and the async machinery is isolated.
- Step 2 stands up the lane abstraction at **K=1 (behavior-equivalent to today, inert)**; K>1 and any concurrent-finalize validation are FORBIDDEN until Step 3's snapshot/commit split lands (else stale-gen corrupts parent state).

### Build / validation / config
- Container build `cd runtime && ./container/enter.sh cmake --build cpp/build_step10 --target ws_server -j 8` (+ `density_main` on any `session.h`/shared-helper change). Cold-load ~5.5 min — budget it.
- Phase-decomposition telemetry `[R1:M10]` (lane queue wait, preproc, AOTI, enc_len sync, decode, scheduler wait, WS send/recv) BEFORE claiming the perf gate; gate-pass is a hypothesis to measure.
- Config `[R1:m11][R2:m9]`: introduce `NEMOTRON_WS_SCHEDULER` (default on) + `NEMOTRON_WS_FINALIZE_RUNNERS` (default `capped_general_finalize_runners(active_cap)`) + `NEMOTRON_WS_LANES`; update the config-print AND `run_compat.py` env in the SAME step that adds each, so ws/density stay intentionally comparable.
- PAIRED per step (Codex impl + Opus review), built + validated in container.

## Steps

- [x] **1. `ExecutionContext` + stream-aware runtime APIs (overloads; density-safe; inert)** `[R1:C2/C4][R2:M4]`
  Introduce an `ExecutionContext` (CUDA stream + references to the per-lane `joint`/`predict`/`preproc`) and thread it through the runtime-only inference entry points + `session.cpp` helpers as **overloads** that default to today's behavior, preserving the exported `run_first_encoder`/`session.h:300` signatures density depends on. Switch steady + finalize AOTI to stream-aware `loader.run(inputs, stream)`. No concurrency, no pool yet — single executor still; output identical. **`density_main` must build.** Validate: build ws_server + density_main, `b2-t1`+`runtime-smoke` 0 divergences, oracle 8/8 (stream plumbing numerically inert).
  Key files: `runtime/cpp/lib/session/session.h`, `runtime/cpp/lib/session/session.cpp`, `runtime/cpp/lib/session/runtime.cpp`

- [ ] **2. Warm `InferenceLane` pool (the substrate); replace the single executor** `[R2:C1/M7][R1:C3]`
  Build a `SharedRuntime`-owned pool of K warm lanes = density `WorkerContext`s (env `NEMOTRON_WS_LANES`, K = admission `active_cap`): each lane = thread + non-blocking stream + its OWN `joint`/`predict`/`preproc` instances + `ExecutionContext`; shared `enc_first` stays locked; assert NO big-module (enc_first/enc_steady/finalize-loader) duplication per lane. **Bind each admitted session to a lane for its lifetime** (admission already caps concurrency; tie lane acquisition to admission so admit ⇒ a free lane exists). Replace the single `InferenceExecutor` with the bound-lane execution (the session's chunks all run on its lane). Warm every lane (stream + its modules + representative buckets) at startup before any `ready`; log coverage + per-lane MB. **Run at K=1 only here — behavior-equivalent/inert** `[R3:C1]`; K>1 and concurrent-finalize validation are deferred to Step 3 (snapshot/commit must land first, else a stale generation corrupts parent state). Validate: build, smokes 0 divergences, oracle 8/8 at K=1, warmup log shows the lane warmed, memory logged, no big-module duplication.
  Key files: `runtime/cpp/lib/session/runtime.cpp`, `runtime/cpp/ws_server.cpp`

- [ ] **3. Finalize on the lane pool (synchronous) + shared finalize loader pool + snapshot/commit split** `[R2:C3/M6][R1:C3/C4/M8]`
  Run finalize synchronously on **the session's own bound lane** (its own modules + stream, stream-aware AOTI) — same lane that ran the session's steady chunks, matching density (the worker that owns the session also finalizes it). Finalize AOTI loaders = ONE shared `FinalizeBucketLoaderPool` with `num_runners=F` (`NEMOTRON_WS_FINALIZE_RUNNERS`, default `capped_general_finalize_runners(active_cap)`, **F is independent of lane count K** `[R3:M3]`) — NOT per-lane duplicated; log MB. Refactor `run_finalize_runtime` into `FinalizeInputSnapshot` → lane compute (on `clone_session` fork) → `FinalizeCommit` under generation check; move the pre-clone `parent.mode=FINALIZED` (:2848) + true-boundary mutation (:2924) into the snapshot/commit boundary so a stale generation can't corrupt parent state. enc_first locked (report `enc_first_lock_wait` separately `[R3:M5]`). Update config-print + `run_compat.py` env for the new vars in THIS step `[R3:m8]`. This step **enables K>1 and closes the measured N=8 finalize tail** (synchronous, no async future). **Milestone-A perf checkpoint** `[R3:M6][R4:m2]`: for a clean *pre-scheduler* measurement, disable/ignore the legacy `NEMOTRON_DENSITY_BATCH_STEADY` env (which `run_compat.py:670` forces and `ws_server.cpp:604` reads to build the telemetry scheduler) until Step 4 — Milestone A must measure the finalize pool alone. Then run the compat oracle here — correctness 8/8 (event-for-event; not "byte-exact" for any scheduler path — there's none yet, finalize is the inline AOTI) + check whether the WS-overhead gate already passes on the finalize pool alone. Validate: oracle correctness 8/8 + `b2-t1` 0 div; concurrent finalizes parallel (ttfs tail collapses) across a K×F mini-matrix; memory within budget.
  Key files: `runtime/cpp/lib/session/runtime.cpp`, `runtime/cpp/lib/session/session.cpp`, `runtime/cpp/ws_server.cpp`

- [ ] **4. Single scheduler owner: un-gate + de-duplicate** `[R1:C5]`
  Make `SharedRuntime` the sole `BatchedSteadyScheduler` owner (`NEMOTRON_WS_SCHEDULER`, default on); remove/redirect ws_server's telemetry-only `state_->scheduler` so the multi-GB `BatchedSteadyLoaderSet` is built once; route `/scheduler_telemetry` to the single scheduler; assert exactly one scheduler/loader set. Construct + `warmup_buckets()` + `start()`. Steady still inline on lanes (no enqueue yet). Update config-print + `run_compat.py` env in this step. Validate: build, smokes 0 div, one-scheduler assertion, oracle 8/8.
  Key files: `runtime/cpp/lib/session/runtime.cpp`, `runtime/cpp/ws_server.cpp`, `tests/server_compat/run_compat.py`

- [ ] **5. Shadow parity gate (isolated): prove scheduler == inline** `[R1:M6][R2:M5]`
  Env-gated diagnostic (`NEMOTRON_WS_STEADY_SHADOW=1`): for each steady continuation, clone the pre-decode state, run BOTH inline `enc_steady_aoti.pt2` and the scheduler `b{1,2,4}` path into SEPARATE clones, decode/apply both, compare exact tensor max-diff (enc_out, cache_ch/t/len) + token/event equality with metadata, then commit exactly ONE path (the inline one in shadow mode). Mark shadow-mode timing invalid for perf. Run over utt0..utt7 under concurrency (force B>1). GO/NO-GO for Step 6: proceed only if diffs ≤ tolerance AND tokens/events identical to inline (and Python). Record numbers. Validate: shadow report 0 token divergences + max-diff within tolerance across B=1 and B>1.
  Key files: `runtime/cpp/lib/session/session.cpp`, `runtime/cpp/lib/session/runtime.cpp`

- [ ] **6. Steady → scheduler `enqueue` from a lane, with async future lifecycle in-step** `[R1:C1][R2:C2]`
  On a lane, for steady continuations: build `BatchedSteadyInput` on the lane stream, record `cudaEventDisableTiming` producer event, admit via a new **`try_enqueue_until(deadline)`** bounded API (NOT the current unbounded capacity-block at `batched_steady_scheduler.cpp:93`), wait on the future (timeout-handled) WITHOUT blocking other lanes (each lane is its own thread → concurrent enqueues → scheduler batches B>1), `cudaStreamWaitEvent(lane_stream, completion)`, decode on the lane, commit under gen-check. Include AT THIS STEP: the bounded `try_enqueue_until` + future timeout, **replace the dispatcher `std::exit(1)` (`:225,257`) with fault propagation** → server error/controlled drain, event RAII, per-lane in-flight future drained before teardown; lanes pool-owned so session destruction just bumps gen. enc_first stays locked. Validate: Step-5 shadow + oracle 8/8 (tokens identical), scheduler telemetry B>1 under load, no re-serialization, re-run ws-lifecycle/stale-gen/shutdown/backpressure smokes + N=200 0/200.
  Key files: `runtime/cpp/lib/session/session.cpp`, `runtime/cpp/lib/session/runtime.cpp`, `runtime/cpp/ws_server.cpp`

- [ ] **7. Phase telemetry + end-to-end oracle validation + perf gate (hypothesis) + docs** `[R1:M10]`
  Add per-phase timing (lane queue wait, preproc, AOTI, enc_len sync, decode, scheduler wait, WS send/recv). Run the compat oracle: correctness 8/8 + decode per-session ttfs (serialization staircase gone?) + WS-overhead gate. Treat the gate as a hypothesis — if it doesn't pass, telemetry localizes the residual (decode `.item()` syncs at session.cpp:1721, WS round-trip, scheduler window) and we tune (B_max/window/K) or document. Update `STEP3B-WS-PLAN.md` + this Progress.
  Key files: `tests/server_compat/run_compat.py`, `proj-2026-05-24-from-scratch-runtime/STEP3B-WS-PLAN.md`, `proj-2026-05-29-1336/PLAN.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | ExecutionContext + stream-aware APIs (overloads, density-safe) | done | ab5c8fb | ExecutionContext{stream,joint,predict,preproc} + density-safe overloads (originals preserved, density_main builds); per-session non-blocking stream (RAII); stream-aware AOTI; inert (post-task sync). Validation: b2-t1 0/0, runtime-smoke 0/0, compat oracle 8/8 PASS. Opus review: clean. |
| 2 | Warm InferenceLane pool; replace single executor | pending | — | the substrate [R2:C1] |
| 3 | Finalize on lane pool (sync) + shared loader pool + snapshot/commit | pending | — | closes N=8 tail [R2:C3/M6] |
| 4 | Single scheduler owner: un-gate + de-dup | pending | — | [R1:C5] |
| 5 | Shadow parity gate (isolated) | pending | — | GO/NO-GO for Step 6 [R2:M5] |
| 6 | Steady → scheduler enqueue from lane + async lifecycle in-step | pending | — | [R1:C1][R2:C2] |
| 7 | Telemetry + oracle validation + perf gate (hypothesis) + docs | pending | — | [R1:M10] |
