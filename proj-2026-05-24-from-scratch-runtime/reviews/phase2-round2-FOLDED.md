# Phase-2 plan — Round 2 FOLDED (Codex + Opus, adversarial-on-Round-1 + deeper code)

Inputs: `codex-phase2-round2.md`, `opus-phase2-round2.md`. Round 2 was charged to attack Round 1 and go deeper
into the concurrency code. Result: **Round 1's 6 blockers all survive, but several are re-ranked or made
evidence-based**, and Codex's read of the installed **libtorch 2.8 headers** resolves the central topology
question and re-targets the residual risk to one precise place. Goals: **G1 density**, **G2 P50↔P95 tail**.

## DECISIVE NEW EVIDENCE — the libtorch 2.8 AOTI headers (Codex, primary source)
Codex read `…/torch/include/torch/csrc/inductor/aoti_package/model_package_loader.h` and
`…/aoti_runtime/model_container.h`:
- `AOTIModelPackageLoader(num_runners=…)`; `run(inputs, stream_handle)` accepts an explicit stream
  (`model_package_loader.h:11-29`).
- The container builds **`num_models` runners that SHARE one constants map/array** (`model_container.h:42-55,
  84-86`). ⇒ **One loader with `num_runners=N` gives concurrent `run()` AND one shared weight copy natively.**
- Normal `run()` takes a **shared execution lock**, returns models to a pending queue, and **waits/reclaims when
  no runner is free** (`model_container.h:92-103,130-142,651-759`). The explicitly unsafe path is the separate
  **`run_single_threaded()`** variant (`model_container.h:145-176`, `model.h:250-265`) — which the session does
  NOT use.

**Three consequences:**
1. **Topology answer is now evidence-based (resolves R1-B2/B3):** the gate path is **one loader, `num_runners=N`,
   one shared constants set**, with explicit per-worker stream handles. Per-worker *separate* loaders become a
   fallback only if the pool fails. The mutex case is the negative control.
2. **The steady memory problem is solved by the pool, NOT by codisk/user_managed.** The 0.1b OOM (N separate
   2.5GB copies) disappears with `num_runners=N` (constants shared across runners). The `user_managed`/codisk
   sharing is needed only for the **finalize buckets** (many distinct packages sharing ONE weight set) — that is
   where R1-B3's "steady-codisk artifact" worry mostly dissolves; re-scope it to finalize.
3. **The residual risk is now ONE precise thing: the container's shared execution lock.** Does that lock serialize
   the actual GPU *dispatch* (an AOTI-internal "GIL" → no overlap, num_runners=N just queues), or only guard
   runner-acquisition bookkeeping (→ real overlap)? **This is exactly what Step 1's overlap micro-gate must
   measure.** The whole density thesis now reduces to: *does num_runners=N `run()` overlap on the GPU, or does the
   execution lock serialize it?*

## ROUND-1 BLOCKERS — status after Round 2 (both reviewers)
- **R1#1 (gate is a vibe; 5090≠L40S):** CONFIRMED. Refined: the Step 1a/1b split is directionally right but "1a
  5090 smoke" is NOT cheap once it carries correctness+topology+finalize+tail. → restructure as **Step 0 cheap
  kill-gates → Step 1a full 5090 mini-sweep → Step 1b L40S numeric gate** (both reviewers, independently).
- **R1#2 (AOTI primitive unspecified):** CONFIRMED but DOWNGRADED in cost — the pool primitive EXISTS (headers);
  it's "configure + prove the runner pool," not "invent concurrency."
- **R1#3 (shared weights only proven serial):** CONFIRMED for finalize; **RE-SCOPED** — steady doesn't need codisk
  if it uses `num_runners=N`. Still must prove memory-flat + concurrent==serial.
- **R1#4 (correctness before perf):** CONFIRMED, both reinforce. Narrowed (see state isolation below) to the
  shared model objects + the execution-lock path.
- **R1#5 (real finalize omitted):** CONFIRMED.
- **R1#6 (G2 not first-class):** CONFIRMED, both expand into a concrete server-side telemetry schema.

## RE-RANKINGS & DOWNGRADES (the honest output of an adversarial round)
- **Hazard re-rank (both):** the **shared STEADY loader is the PRIMARY concurrent-`run()` hazard** (~200 steady
  `run()`/sec at 32 streams, every continuation chunk — `session_main.cpp:1753-1766,3868`), NOT the finalize
  hot-bucket (once per utterance, bucketed — Codex R1-M4 **downgraded to a secondary stress case**). Order:
  (1) shared steady loader / execution-lock overlap, (2) decode `.item()`/default-stream behavior,
  (3) finalize hot-bucket.
- **BW-bound (both DOWNGRADE):** "encoder is BW-bound" is a **hypothesis**, not established by the 0.1b data
  (which only shows GPU saturated at high L40S load; util can't distinguish BW vs SM-occupancy vs launch vs
  runner-queue). Keep the "attribute the knee to a resource" requirement; require Nsight/CUPTI counters (kernel
  overlap, SM occupancy, DRAM throughput, launch gaps, runner wait, `.item()` wait). Don't pre-cap ambition at
  ~2.5× nor assume more.
- **Per-stream state isolation (both CONFIRM good → DOWNGRADE a worry):** `SessionState` is fully self-contained
  (`session_main.cpp:82-102`; only `static const` strings at `:565-570`); audio ring/pending/post-stop are
  per-state (`:94-101,1954-2018`); finalize fork is a fresh `.clone()` per call (`:953-974,2577-2603`). ⇒ N
  threads can each own one `SessionState` safely. The correctness gate is therefore **narrowly about the shared
  model objects**, not a SessionState refactor.

## DEEPER-CODE FINDINGS (Round 2 new)
- **Per-thread handles required (both):** `enc_first`, `joint`, `predict`, and `preproc` are single shared
  modules today (`session_main.cpp:3862-3915`); concurrent `torch::jit::Module::forward` on a shared handle is
  not proven safe (JIT executor mutates per-graph state). → each worker owns its own handles (the mock did —
  `microbench.cpp:103`); shared-handle is an ablation only.
- **`AudioFrontend` has shared mutable stats (Codex, new):** it carries mutable stats/margin accumulators and
  currently one instance points at one `preproc` (`session_main.cpp:1942-1953,3900-3915`). Sharing it across
  threads races the stats. → `AudioFrontend` per worker, or split immutable geometry from per-thread stats.
- **`.item()` / default-stream serialization trap (both):** decode does `argmax().item()` (and `topk().item()`)
  per label (`session_main.cpp:1648-1682`). With per-thread streams it syncs only that thread; WITHOUT them (or
  if any op lands on the default stream) it forces a device-wide sync that serializes ALL threads. → install
  per-worker stream guards around joint/predict/enc_first/preproc AND pass the same stream into AOTI `run()`;
  prove with CUDA events, not NVML.
- **FORK_ASSERT is not a concurrency oracle (Codex, precise):** `AsrSnapshot` **excludes** the collector fields
  (`last_interim_*`, `continuous_emitted_*`, `post_stop_audio` — `session_main.cpp:104-119`) and finalize updates
  the collector before the assert (`:2651-2674`). FORK_ASSERT proves serial parent-state isolation only. → add a
  separate N-thread finalize correctness check vs the serial oracle.

## DISTINCTIVE — Opus (preserve into Round 3/4)
- **G2 tail is scheduler-dependent (Opus C2):** the P95−P50 under load is a QUEUEING phenomenon shaped by the
  dispatch discipline (per-stream thread vs M-worker pool + shared queue vs priority finalize lane). Step 1 has
  no scheduler yet (Step 2). So **Step 1's tail is the tail of a PLACEHOLDER discipline** — label it a
  reference/floor, and make the *binding* G2 number come from Step 4 (real scheduler + real WS). You can't fully
  measure the tail before designing the thing that shapes it → sequence G2 as **reference (Step 1) → binding
  (Step 4)**, and name which scheduler knobs (finalize priority lane, admission cap) are expected to move it.
  (Codex Q1 — "what is the worker ownership model?" — is the upstream of this; resolve it explicitly.)
- **Quantify finalize arrival rate at the knee** (~every 100–200ms aggregate at ~28 streams) so the harness's
  finalize model is calibrated to reality, not the 0.1b's arbitrary `--finalize-every 15`.

## CONSOLIDATED EDITS AFTER ROUND 2 (supersedes R1 list)
1. **Stage the gates:** Step 0a steady `num_runners=N` overlap (concurrent==serial + memory-flat + CUDA-event
   overlap + execution-lock-serialization check); 0b decode per-thread-handle token equality + `.item()` wait
   telemetry; 0c finalize same/mixed-bucket equality + per-bucket wait. → Step 1a full 5090 mini-sweep (not a GO)
   → Step 1b L40S numeric gate (≥1.5× / ≥~28 SLO-robust streams/box + G2 spread target, baseline re-measured).
2. **Object-ownership spec:** per-thread `SessionState` + `AudioFrontend` + CUDA stream + `enc_first`/`joint`/
   `predict`/`preproc`; **shared AOTI loader with `num_runners=N`** (proven pool) for steady; `user_managed`
   shared constants for the finalize buckets.
3. **Correctness-before-perf**, scoped to the smallest corpus that catches cross-runner aliasing, incl. real
   finalize + hot-bucket; collector fields explicitly checked (FORK_ASSERT doesn't).
4. **BW-bound = hypothesis;** require counter-based resource attribution of the knee.
5. **G2 server-side schema in every Step-1 table:** TTFT p50/p95/p99 + **P95−P50** + enqueue→first-token,
   enqueue→final, queue wait, AOTI/runner wait, `.item()` wait, finalize wait, shed counters; reference-tail in
   Step 1 → binding-tail in Step 4; don't imply Phase 2 moves VAD/WAN.
6. Name the Phase-2 harness build target + log `num_runners` in output filenames (don't benchmark the serial
   baseline as the candidate).

## OPEN FOR ROUND 3 (downstream Steps 2–5)
Steps 2–5 have had only light review. Round 3: scheduler/admission design faithfulness to the Python shed
behavior (`backlog-count cap`, priority finalize lane); the Step-4 apples-to-apples harness pitfalls (same SLO /
same WER tool / re-measured baseline / WS-server's own tail contribution); Step-5 per-target hypothesis (L40S/L4/
Spark) and how the binding-resource finding from Step 1 pre-informs it; whether the real WS server (Step 3)
introduces its own tail that confounds the density number.
