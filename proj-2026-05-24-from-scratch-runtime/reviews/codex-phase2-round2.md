# Phase-2 paired review - Round 2 (Codex)

Round-2 charge executed against `PHASE2-PLAN.md`, `reviews/phase2-round1-FOLDED.md`,
`reviews/phase2-review-brief-round2.md`, the 0.1b spec/results, and the named C++ runtime files.
Foreground goals: **G1 = utilization / streams-box density** and **G2 = P50/P95 TTFT spread**.

## BLOCKER

### B1 - DISAGREE / DOWNGRADE Round-1 "AOTI primitive unknown": `num_runners>1` is real, but the current session still serializes the hot steady path.

Round 1 was right that the plan cannot trust current `session_main`, but overstated the work as if a concurrent runner primitive might not exist. In the installed libtorch 2.8 headers, `AOTIModelPackageLoader` takes `num_runners` and `run()` takes an optional `stream_handle` (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_package/model_package_loader.h:11-24`). The underlying container constructs `num_models` runners sharing one constants map/array (`.../aoti_runtime/model_container.h:42-55`), uses a shared execution lock for normal `run()` (`.../aoti_runtime/model_container.h:92-103`), returns running models to a pending queue (`.../aoti_runtime/model_container.h:130-142`), and waits/reclaims when no runner is available (`.../aoti_runtime/model_container.h:651-759`). The explicitly unsafe path is the separate `run_single_threaded()` variant (`.../aoti_runtime/model_container.h:145-176`; `.../aoti_runtime/model.h:250-265`).

That downgrades the required edit from "invent or redesign AOTI concurrency" to "configure and prove the existing runner pool." It does **not** downgrade the plan risk: current `session_main` constructs the steady loader with `num_runners=1` (`runtime/cpp/session_main.cpp:3868-3869`), calls `run_steady_encoder()` with no stream handle (`runtime/cpp/session_main.cpp:1753-1766`), and hits that path on every continuation chunk (`runtime/cpp/session_main.cpp:2085-2087`, `runtime/cpp/session_main.cpp:2231-2234`). With one runner, concurrent callers must queue on the one available model, which directly harms G1 density and creates G2 tail spread.

Recommended plan edit:
- Replace the broad topology matrix with a front-loaded AOTI runner proof: shared `enc_steady` loader with `num_runners=N`, explicit per-worker stream handles, concurrent==serial output equality, CUDA-event overlap, and GPU memory proof of one shared weight set. Keep the mutex-serialized case as the negative control.
- Only add per-worker loader variants if the `num_runners=N` shared-loader proof fails or shows aliasing/queueing.

### B2 - Re-rank Round-1 hot-bucket finalize: the shared STEADY loader is the primary concurrent `run()` hazard.

Round 1's hot-bucket finalize collision is real, but it is not the primary hazard. The 0.1b harness models steady chunks every 160 ms (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:55`, `spikes/0.1-overlap-ablation/microbench/microbench.cpp:204-208`), while its finalize model is about once every 15 chunks, approximately 2.4 s (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:57-60`, `spikes/0.1-overlap-ablation/microbench/microbench.cpp:182-184`). In the real session, every continuation chunk funnels through the single shared `enc_steady` loader (`runtime/cpp/session_main.cpp:1753-1766`, `runtime/cpp/session_main.cpp:3868-3869`). At 32 active streams, that is roughly 200 steady `run()` calls/sec before considering first/final work. Finalize is per utterance and bucketed (`runtime/cpp/session_main.cpp:1545-1580`, `runtime/cpp/session_main.cpp:2622-2634`, `runtime/cpp/session_main.cpp:2702-2715`).

So Codex R1-M4 should be **DOWNGRADED from primary blocker to stress case / MAJOR**. If Step 1 first proves finalize hot buckets but leaves steady as one runner on default stream, it can still falsely fail G1/G2 or understate overlap. If it proves a trace with sparse finalizes but does not saturate steady concurrency, it can falsely pass.

Recommended plan edit:
- Step 1's first workload must be a steady-loader concurrency microgate at the 160 ms cadence, not a finalize collision test.
- Keep a separate hot-bucket finalize stress after the steady proof, with same-bucket final_T distribution and per-bucket wait telemetry.

### B3 - Step 1a/1b split from Round 1 is directionally right but mis-sequenced; "5090 smoke" is not cheap if it includes the full harness.

Round 1's L40S hard gate is correct: the pre-registered number is L40S >=1.5x / >=~28 streams-box at the SLO (`spikes/0.1-overlap-ablation/0.1b-microbench-spec.md:42-45`), and the plan's current "meaningfully" gate is vague (`PHASE2-PLAN.md:24-32`). But I **DISAGREE** with treating "1a 5090 overlap+correctness smoke" as a full correctness+topology+finalize+tail harness. Once 1a includes real decode, concurrent token/event equality, all loader topologies, real finalize, audio, multiturn, and G2 tail accounting, it is no longer a cheap smoke; it is most of Step 1.

The corrected sequencing should mirror the original 0.1b cheap-kill pattern: 0.1b explicitly existed before full byte-exact work because it was the cheapest density gate (`spikes/0.1-overlap-ablation/0.1b-microbench-spec.md:53-59`). Phase 2 needs the same structure, but with the real primitives.

Recommended plan edit:
- Insert **Step 0 / pre-Step-1 kill gates**:
  1. AOTI steady `run()` overlap: shared `num_runners=N`, explicit streams, concurrent==serial tensors, memory flat, CUDA-event overlap.
  2. Decode-loop overlap: per-thread `joint`/`predict` handles, per-thread CUDA stream, serial-equivalent tokens for a small corpus.
  3. Finalize bucket overlap: same-bucket and mixed-bucket cases, concurrent==serial final tokens.
- Then run the full 5090 mini-sweep.
- Only then spend EC2 money on the L40S hard gate.

### B4 - Shared TorchScript module ownership is not specified; the mock avoided this by using per-lane handles.

The plan says "N worker threads ... real per-session compute" (`PHASE2-PLAN.md:24-30`) but does not define object ownership. Current `session_main` loads single shared `enc_first`, `joint`, `predict`, and optional `preproc` modules (`runtime/cpp/session_main.cpp:3862-3875`, `runtime/cpp/session_main.cpp:3908-3915`). The decode loop calls `joint.forward()` and `predict.forward()` repeatedly (`runtime/cpp/session_main.cpp:1629-1685`), first chunks call `enc_first.forward()` (`runtime/cpp/session_main.cpp:1715-1726`), and audio preprocessing calls `preproc->forward()` (`runtime/cpp/session_main.cpp:1976-1982`).

The only existing concurrency bench deliberately did **not** share a TorchScript module: each lane owns its own module handle and stream (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:97-105`, `spikes/0.1-overlap-ablation/microbench/microbench.cpp:159-167`). There is no repo evidence proving concurrent `torch::jit::Module::forward()` on one shared handle is safe, non-serializing, and scratch-isolated for these modules. Sharing one handle could corrupt correctness, silently serialize, or confound the G1/G2 measurement.

Recommended plan edit:
- Specify the Step-1 ownership model: one `SessionState`, one `AudioFrontend`, one CUDA stream, and separate `enc_first`/`joint`/`predict`/`preproc` handles per worker thread unless a smaller official/module-level proof is added first.
- If shared TorchScript handles are tested, make them an explicit ablation, not the gate path.

### B5 - The per-label `.item()` sync is the decode idle-window thesis, but default-stream use can make it a global serialization point.

Real decode performs `joint.forward()`, optional `topk().item<double>()`, `argmax().item<int64_t>()`, and then `predict.forward()` for every emitted label (`runtime/cpp/session_main.cpp:1648-1682`; finalize-only harness has the same argmax sync at `runtime/cpp/finalize_main.cpp:620-647`). That host readback is exactly the gap the overlap thesis wants other streams to fill. But the current steady AOTI call uses no explicit stream (`runtime/cpp/session_main.cpp:1753-1766`), and the already-validated explicit-stream check is only single-threaded (`runtime/cpp/aoti_encoder_main.cpp:63-68`). The AOTI API can accept a stream handle (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_package/model_package_loader.h:22-29`), but `session_main` does not use it.

If Step 1 does not install a per-worker stream guard for TorchScript ops and pass the same stream into AOTI, each `.item()` may wait on the default stream and serialize unrelated work. That directly attacks G1 (lost overlap) and G2 (P95 inflation from cross-stream stalls). Conversely, if per-thread streams are used, `.item()` should only block that thread's queued stream work; Step 1 must prove that with CUDA events, not infer it from NVML averages.

Recommended plan edit:
- Require per-worker stream guards around `joint`, `predict`, `enc_first`, `preproc`, and AOTI calls.
- Report CUDA-event service time, host `.item()` wait time, queue wait, and default-stream negative-control results.

## MAJOR

### M1 - DISAGREE with treating BW-bound ceiling as established; current 0.1b data only establishes "GPU saturated" at high L40S load.

Opus R1's resource-attribution requirement should be kept, but the "BW-bound encoder ceiling" conclusion is not established by the cited results. L40S lanes=8 sustains 32 streams at p95 127 ms / 80% GPU, then runs away at 48 streams / 98% GPU (`spikes/0.1-overlap-ablation/microbench/RESULTS-L40S.md:17-23`). The result text concludes the ceiling is "GPU encoder compute" and not lane-bound (`spikes/0.1-overlap-ablation/microbench/RESULTS-L40S.md:25-32`). That is not the same as proving memory bandwidth is binding. The 5090 still had 60% GPU util at 48 streams and did not reach the knee (`spikes/0.1-overlap-ablation/microbench/RESULTS-5090.md:15-24`).

The plan currently asks for GPU util and knee (`PHASE2-PLAN.md:27-32`), but NVML util cannot distinguish memory bandwidth, launch dispatch, host sync, SM occupancy, or runner queueing. Treating BW-bound as pre-confirmed could cap ambition too early and hide SM headroom; treating it as disproven would be equally wrong.

Recommended plan edit:
- Keep "attribute the knee" as a requirement, but phrase BW-bound as a hypothesis.
- Require Nsight/CUPTI or equivalent counters: kernel overlap, SM occupancy, memory throughput, launch gaps, AOTI runner wait, and host `.item()` wait.

### M2 - SessionState isolation is mostly good; downgrade any implied global-state blocker, but do not let one `SessionState` cross threads.

I found no mutable file-scope or function-scope static state in the inspected session path. `SessionState` owns caches, decoder tensors, token vectors, audio buffers, collector text, mode, and counters (`runtime/cpp/session_main.cpp:82-102`). Reset clones initial tensors or clears vectors (`runtime/cpp/session_main.cpp:1035-1055`), and audio reset initializes per-state raw ring and counters (`runtime/cpp/session_main.cpp:1057-1063`). The only function-scope statics I found in the core text path are const tokenizer strings (`runtime/cpp/session_main.cpp:565-570`).

So I **DOWNGRADE** a broad "SessionState is unsafe" concern: N threads can each own one `SessionState`. The unsafe case is sharing one `SessionState` across threads or mixing per-stream data structures with shared mutable stats (see M4). Step 1's correctness gate should assert object ownership explicitly instead of refactoring SessionState.

Recommended plan edit:
- State "one session object is single-thread-owned; no concurrent methods on the same SessionState."
- The concurrent correctness test should create N independent SessionStates and compare each stream's final tokens and ordered events to the serial oracle before collecting performance.

### M3 - Finalize fork/clone is fresh per call; `FORK_ASSERT` is useful serially but not a concurrency oracle.

The real finalize path snapshots parent ASR state, marks the parent finalized, then clones into a fork (`runtime/cpp/session_main.cpp:2577-2603`). The clone allocates fresh tensors for encoder cache, decoder state, optional mel ring, and copies vectors/text (`runtime/cpp/session_main.cpp:953-974`). Finalize then runs AOTI on fork cache inputs and decodes into fork state (`runtime/cpp/session_main.cpp:2628-2643`; runtime variant `runtime/cpp/session_main.cpp:2709-2724`). Parent ASR invariants are checked after decode (`runtime/cpp/session_main.cpp:996-1033`, `runtime/cpp/session_main.cpp:2674`, `runtime/cpp/session_main.cpp:2754`). The standalone finalize harness has the same clone/assert pattern for tensors (`runtime/cpp/finalize_main.cpp:123-142`) and a serial Phase-B loop (`runtime/cpp/finalize_main.cpp:793-830`).

This means the fork path itself does not appear to reuse a shared parent buffer. I **DOWNGRADE** any Round-1 concern that the fork clone is inherently unsafe. But `FORK_ASSERT` only checks that a serial fork decode did not mutate its parent. It does not prove shared AOTI loaders, shared TorchScript modules, shared CUDA streams, or shared constants are concurrent-safe. It also does not cover collector fields by design: `AsrSnapshot` excludes `last_interim_*`, `continuous_emitted_*`, and `post_stop_audio` (`runtime/cpp/session_main.cpp:104-119`), while finalize updates the continuous collector before the assert (`runtime/cpp/session_main.cpp:2651-2674`).

Recommended plan edit:
- Keep `FORK_ASSERT` as a per-call state-isolation check.
- Add a separate N-thread finalize correctness check that compares final tokens/events to serial outputs and stresses same-bucket loader concurrency.

### M4 - Audio ring/remainder recompute is per-state, but `AudioFrontend` and `preproc` ownership must become per-thread or reduced under lock.

The raw audio ring, pending audio, and post-stop audio live in `SessionState` (`runtime/cpp/session_main.cpp:94-101`). Steady audio builds a local CPU tensor from state buffers, calls preproc, then advances only that state's raw ring and pending audio (`runtime/cpp/session_main.cpp:1954-1982`, `runtime/cpp/session_main.cpp:1985-2018`, `runtime/cpp/session_main.cpp:2167-2173`). Runtime VAD stop appends to per-state `post_stop_audio`, and VAD start swaps it back into per-state `pending_audio` (`runtime/cpp/session_main.cpp:2301-2321`, `runtime/cpp/session_main.cpp:2335-2363`). Finalize recompute copies `parent.pending_audio` and `parent.raw_audio_ring` into local vectors before deriving final mel chunks (`runtime/cpp/session_main.cpp:2457-2531`).

So I **DOWNGRADE** an audio-ring shared-scratch blocker: the ring/remainder algorithm is per-session if each thread owns its state. The remaining risk is `AudioFrontend`: it contains mutable stats and margin accumulators (`runtime/cpp/session_main.cpp:1942-1953`), and the current program creates one `AudioFrontend` pointing at one `preproc` module (`runtime/cpp/session_main.cpp:3900-3915`). Sharing that across worker threads would race stats and reintroduce the shared TorchScript-handle issue.

Recommended plan edit:
- Make `AudioFrontend` per worker/session for Step 1 audio-fed tests, or split immutable geometry from per-thread stats and reduce stats after the run.
- Treat shared `preproc` as an explicit ablation only after per-thread preproc handles pass.

### M5 - Shared weights proof remains incomplete for steady, but libtorch runner-pool sharing changes the required experiment.

Round 1 correctly flagged that the steady encoder is not wired through the shared-weight constants-on-disk path. `session_main` uses `enc_steady_aoti.pt2` directly (`runtime/cpp/session_main.cpp:3868-3869`), while `validate_shared_weights.py` says the shared-weight proof needs `artifacts/enc_steady_codisk.pt2` and `finalize_shared_weights.pt` (`runtime/validate_shared_weights.py:1-7`, `runtime/validate_shared_weights.py:13-25`). The artifact listing in this workspace has `enc_steady_aoti.pt2` and `enc_steady_t2a.pt2`, not an `enc_steady_codisk.pt2`.

But libtorch's container shares constants across its `num_models` runners (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/model_container.h:46-55`, `.../aoti_runtime/model_container.h:84-86`). That means the plan edit should not require N independent per-worker loaders as the default. The sharper proof is one loader with N runners, one user-managed shared constant set, flat memory, and concurrent==serial.

Recommended plan edit:
- Build/prove steady codisk if shared user-managed constants are required for the memory budget.
- For the gate path, prefer "one loader, N runners, one shared constants set" before per-worker loaders.

### M6 - G2 still needs server-side tail decomposition and queue telemetry; "latency tail" is not enough.

`PHASE2-PLAN.md` mentions latency tail only generically in Step 1 (`PHASE2-PLAN.md:24-30`) and same-harness density in Step 4 (`PHASE2-PLAN.md:39-41`). The 0.1b results were explicit that the metric was a chunk intake-to-done keep-up proxy, not client TTFT (`spikes/0.1-overlap-ablation/microbench/RESULTS-5090.md:3-6`, `spikes/0.1-overlap-ablation/microbench/RESULTS-5090.md:39-39`). Round 1's G2 concern remains valid: without TTFT p50/p95/p99, P95-P50, and phase waits, Step 1 can lift density while making the tail worse.

Recommended plan edit:
- Define G2 as server-side TTFT/tail for Phase 2: enqueue-to-first-token, enqueue-to-final, queue wait, AOTI wait, decode `.item()` wait, finalize wait, and admission/shed counters.
- Preserve end-to-end TTFT as a Step 4 apples-to-apples metric, but do not imply Phase 2 can move VAD/WAN components.

## MINOR

### m1 - First-chunk dispatch remains mixed TorchScript/AOTI and should be labeled in any density number.

The steady continuation path is AOTI (`runtime/cpp/session_main.cpp:1753-1766`), but first chunk still uses `enc_first.forward()` (`runtime/cpp/session_main.cpp:1715-1726`), and the main program loads `enc_first.ts` separately (`runtime/cpp/session_main.cpp:3862-3867`). This is not the primary G1 risk because first chunks are less frequent than steady chunks, but it can affect TTFT P50/P95 and should be labeled or AOTI-converted before Step 4.

### m2 - Phase-2 harness still lacks a build target.

`runtime/cpp/CMakeLists.txt` builds the existing single-purpose executables through `session_main` (`runtime/cpp/CMakeLists.txt:11-56`) but has no Phase-2 concurrent harness target. This is minor only because the plan review is about design, but the next plan edit should name the target and output artifacts so Round 3 can review actual commands/logs.

### m3 - Existing finalize and session loaders use `num_runners=1`; do not accidentally benchmark the serial baseline as the candidate.

Finalize bucket loaders are also created with one runner (`runtime/cpp/session_main.cpp:1568-1579`; `runtime/cpp/finalize_main.cpp:793-803`). That is fine for the current serial correctness harness. The Phase-2 harness must make runner count explicit in logs and output filenames, or a `num_runners=1` run can be mistaken for the candidate topology.

## QUESTIONS

1. What is the target worker ownership model for Step 1: one worker per active stream, a fixed runner pool serving many streams, or a smaller lane pool like 0.1b? This affects whether `num_runners` equals active streams, CPU cores, or scheduler lanes.
2. What is the empirical `(drop,T)` distribution for real finalizes under the target workload? Hot-bucket finalize remains a required stress, but its priority depends on how concentrated production final_T is.
3. What exact Python baseline should the L40S hard gate compare against: 16, 20, or a re-measured same-date baseline with the same SLO and TTFT definition?
4. Will the Step-1 harness run audio-fed paths, mel-fed paths, or both? For G2, audio-fed first-token behavior matters; for the cheapest G1 kill-gate, mel-fed is enough.

## Recommended consolidated plan edits for Round 2

1. Replace the vague Step-1 gate with staged gates:
   - Step 0a: steady AOTI `num_runners=N` + explicit streams + concurrent==serial + memory-flat + overlap evidence.
   - Step 0b: real decode per-thread handles + per-thread stream + token equality + `.item()` wait telemetry.
   - Step 0c: real finalize same-bucket and mixed-bucket concurrency equality + per-bucket wait telemetry.
   - Step 1a: 5090 full mini-sweep, not a GO.
   - Step 1b: L40S hard gate, numeric >=1.5x / >=~28 SLO-robust streams-box, with G2 spread target.
2. Make object ownership explicit: per-thread `SessionState`, `AudioFrontend`, CUDA stream, `enc_first`, `joint`, `predict`, and `preproc`; shared AOTI loader only if using proven `num_runners=N` runner pool.
3. Re-rank hazards: shared steady loader first, decode `.item()` stream behavior second, finalize hot-bucket third.
4. Treat BW-bound as a hypothesis, not a conclusion. Require resource attribution with counters beyond NVML GPU util.
5. Keep correctness-before-perf, but scope the earliest check to the smallest corpus that can catch cross-runner aliasing before building the full server/scheduler harness.
6. Define G2 as server-side TTFT/tail for Phase 2 and require P50/P95/P99/P95-P50 plus queue/AOTI/decode/finalize waits in every Step-1 table.
