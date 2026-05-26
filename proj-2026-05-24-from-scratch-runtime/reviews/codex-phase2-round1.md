# Phase-2 plan review - Codex round 1

Scope: adversarial static review of `PHASE2-PLAN.md` and the listed code/evidence. I foregrounded the two stated objectives: (1) reclaim idle GPU / increase SLO-robust streams per box, and (2) tighten the P50/P95 TTFT spread under load.

## BLOCKER

### B1 - Step 1's hard gate is not the pre-registered density gate, and a 5090-only pass can false-GO the L40S decision.

Evidence: `PHASE2-PLAN.md:25-32` makes Step 1 a 5090 spike and gates on whether the knee lifts "meaningfully" above N=1. The frozen 0.1b spec's gate is numeric and deploy-target-specific: L40S sustainable density `>=1.5x` / `>=~28/box` at the same SLO, with GPU util rising (`spikes/0.1-overlap-ablation/0.1b-microbench-spec.md:42-45`). The 5090 result itself says the 5090 is not the gate and "GO/NO-GO must come from the L40S run" (`spikes/0.1-overlap-ablation/microbench/RESULTS-5090.md:36-38`). The L40S result is still mock/steady-only and says real finalizes can lower the multiplier (`spikes/0.1-overlap-ablation/microbench/RESULTS-L40S.md:34-48`).

Risk to goals: a 5090 "meaningful lift" can approve weeks of native work without proving the deploy-box density objective. It also cannot quantify the P50/P95 TTFT spread on the target hardware.

Recommended plan edit: split Step 1 into:

1. `Step 1a (5090 overlap/correctness smoke)`: local binary answer only. PASS here permits L40S testing; it is not a GO for the density thesis.
2. `Step 1b (L40S hard gate)`: real native compute, real finalize path, same SLO as the Python baseline, numeric gate `>=1.5x` and `>=~28 SLO-robust streams/box` versus a same-hardware baseline. Replace "meaningfully" with the numeric threshold and require P50, P95, and `P95-P50` reporting.

### B2 - The plan does not pin the AOTI concurrency primitive, so Step 1 can test the wrong thing.

Evidence: 0.1b's overlap mechanism is per-lane TorchScript module + captured CUDA graph + per-lane stream guard (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:97-124`) with one module per lane (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:159-167`). The real session path is not that primitive: steady encoder dispatch is `AOTIModelPackageLoader::run(inputs)` (`runtime/cpp/session_main.cpp:1753-1766`), with a single steady loader built as `num_runners=1` (`runtime/cpp/session_main.cpp:3868-3869`). Finalize also builds one loader per bucket with `num_runners=1` (`runtime/cpp/session_main.cpp:1568-1575`) and calls `loader_it->second->run(inputs)` (`runtime/cpp/session_main.cpp:2622-2635`, `runtime/cpp/session_main.cpp:2702-2716`). There is no CUDA stream guard around these session-main AOTI calls. The separate AOTI seam harness shows an explicit stream API exists (`loader.run(inputs, (void*)stream.stream())`) but only proves stream-invariant numerics, not concurrent overlap (`runtime/cpp/aoti_encoder_main.cpp:63-68`).

Risk to goals: if Step 1 uses session-main's default `run(inputs)` path, a "no overlap" result may just be a default-stream/one-runner artifact, not proof that a native runtime cannot overlap. Conversely, if it uses per-loader/per-stream graph replay like 0.1b, a PASS may not transfer to the actual AOTI session runtime.

Recommended plan edit: Step 1 must pre-register and separately test the loader topology:

- one shared loader with `num_runners=1`;
- one shared loader with `num_runners=N` if supported;
- per-worker loaders sharing one weight tensor map;
- optional mutex-serialized loader as the negative control.

For each topology, call AOTI with an explicit per-worker CUDA stream pointer, set a device/stream guard per worker, record CUDA events, and report whether kernels overlap on the intended streams. The gate must name which topology is eligible for the production scheduler.

### B3 - "Proven shared weights" is only proven for serial load/run, not concurrent N-thread AOTI execution, and the steady encoder does not currently use the shared-weight package.

Evidence: the plan leans on "the proven shared-weights mechanism" (`PHASE2-PLAN.md:14`) and Step 1 says shared weights via `load_constants(user_managed)` (`PHASE2-PLAN.md:25-27`). The actual proof script loads constants once and runs one package once (`runtime/validate_shared_weights.py:13-25`). The stripper/validator reuses CUDA tensors from a map (`runtime/strip_bucket_weights.py:296-324`) but validates buckets serially (`runtime/strip_bucket_weights.py:579-594`). `finalize_main.cpp` also wires shared constants into one loader per bucket and then iterates rows serially (`runtime/cpp/finalize_main.cpp:793-808`, `runtime/cpp/finalize_main.cpp:810-830`). Session finalize loaders load constants serially into per-bucket loaders (`runtime/cpp/session_main.cpp:1565-1579`). No listed proof calls `run()` concurrently from N threads while sharing the same user-managed constants. Also, the current session steady path loads `enc_steady_aoti.pt2` directly (`runtime/cpp/session_main.cpp:3868-3869`), while the shared-weight validation needs a constants-on-disk `enc_steady_codisk.pt2` (`runtime/validate_shared_weights.py:6`).

Risk to goals: Step 1 can false-GO if it accidentally duplicates weights per lane and fits on 5090 but not L40S, or false-STOP/OOM if it fails to convert the steady path to constants-on-disk sharing. More importantly, shared user-managed constants may be safe for serial load-time sharing but unsafe or internally serialized under concurrent runtime access.

Recommended plan edit: before any throughput number is trusted, add a Step 1 correctness/memory subgate:

- build/use a constants-on-disk steady AOTI package for the real steady encoder;
- run N workers with one shared CUDA weight set and the chosen loader topology;
- force hot-bucket finalize collisions where multiple threads hit the same `(drop,T)` bucket;
- assert peak GPU memory proves one weight copy, not N copies;
- assert concurrent outputs equal serial outputs.

### B4 - Step 1 measures speed before proving concurrent correctness.

Evidence: Step 1 lists throughput, latency tail, GPU util, and the knee (`PHASE2-PLAN.md:25-30`), while correctness is deferred to Step 3's "correct per-stream events/finals" (`PHASE2-PLAN.md:36-38`). The project rules say Step 1+ must be token-exact and event/delta-exact, not WER-only (`PLAN_RULES.md:8-16`). The single-stream harness has detailed token/event/fork gates (`runtime/cpp/session_main.cpp:4516-4529`) and same-process replay determinism (`runtime/cpp/session_main.cpp:4480-4502`), but those are serial gates.

Risk to goals: a concurrent race in AOTI runner state, shared constants, TorchScript `joint/predict`, cache ownership, or finalizer bucket reuse can produce fast but wrong transcripts/events. That is a direct false-GO on density.

Recommended plan edit: Step 1 must require a "correctness before perf" phase: run the N-thread harness over the same serial corpus/fingerprints, including audio mode and multiturn/finalize, and compare each stream's final tokens and ordered event text to the serial `session_main` result. Only then collect throughput and TTFT statistics.

### B5 - Step 1 does not explicitly include the real finalize path, which is load-bearing for both density and tail.

Evidence: Step 1 says each worker replays "steady AOTI + decode" (`PHASE2-PLAN.md:25-27`), with no explicit real finalize/fork/bucket replay. The 0.1b L40S evidence is explicitly no-finalize (`spikes/0.1-overlap-ablation/microbench/RESULTS-L40S.md:38-39`). The 5090 "finalize" check was only extra graph replays plus host sleep, not a real `keep_all_outputs` bucket (`spikes/0.1-overlap-ablation/microbench/RESULTS-5090.md:41-57`). The real C++ finalize path clones the parent, selects a bucket, runs AOTI, decodes continuation, emits final/suppressed events, and verifies fork preservation (`runtime/cpp/session_main.cpp:2577-2680`); audio mode recomputes final remainder geometry from live audio (`runtime/cpp/session_main.cpp:2457-2531`).

Risk to goals: omitting real finalizes overstates streams/box and hides the main TTFT tail generator: asynchronous final events, fork/clone overhead, shared bucket contention, and decode-continuation variability.

Recommended plan edit: Step 1 must replay a realistic session trace with real `vad_stop`/finalize events, including `prepare_finalize_inputs_from_audio`, fork/clone, bucket lookup, `load_constants`-backed finalizer, and continuation decode. Report steady-only as an ablation, not the gate.

### B6 - The P50/P95 spread objective is not a first-class gate.

Evidence: the plan says "latency tail" once (`PHASE2-PLAN.md:27-28`) but does not define TTFT, P50, P95, spread, or a target. 0.1b used a chunk intake-to-done proxy rather than full client TTFT (`spikes/0.1-overlap-ablation/microbench/RESULTS-5090.md:3-6`, `spikes/0.1-overlap-ablation/microbench/RESULTS-5090.md:39`). The spec's SLO definition is the Python sweep criterion: `vad_stop->final p95` in budget and `vad_stop_recv_to_process` bounded (`spikes/0.1-overlap-ablation/0.1b-microbench-spec.md:36-40`).

Risk to goals: the plan can improve median/throughput while widening the P95 tail, or can pass a keep-up proxy that says nothing about user-visible TTFT spread under load.

Recommended plan edit: make Step 1 and Step 4 report `TTFT_p50`, `TTFT_p95`, `TTFT_p95_minus_p50`, `p99`, queue wait, lane wait, and finalize wait. Add an explicit non-regression/improvement target versus the Python baseline on the same hardware and load. Use chunk latency only as diagnostic telemetry.

## MAJOR

### M1 - The real decode loop is data-dependent and synchronization-heavy; the plan does not specify decode module/thread topology.

Evidence: C++ decode runs per frame and up to `MAX_SYMBOLS` per frame (`runtime/cpp/session_main.cpp:1645-1683`), dispatches TorchScript `joint.forward` and `predict.forward` (`runtime/cpp/session_main.cpp:1648`, `runtime/cpp/session_main.cpp:1678-1682`), and synchronizes through `argmax().item()` per emitted/blank symbol (`runtime/cpp/session_main.cpp:1657`). 0.1b used fixed host sleep and optional dummy GEMM (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:52-61`, `spikes/0.1-overlap-ablation/microbench/microbench.cpp:180-186`). Prior folded status explicitly warns that real decode's joint/predict GPU calls plus host argmax syncs may not transfer from the pure-sleep mock (`reviews/worksofar-FOLDED.md:20-25`) and that the "no per-frame .item()" density premise is not implemented (`reviews/worksofar-FOLDED.md:41-43`).

Risk to goals: fixed mock decode underestimates `P95-P50` spread and can misidentify the next utilization bottleneck. Shared `joint/predict` modules may also serialize or race unless their per-thread topology is stated.

Recommended plan edit: Step 1 must state whether `joint` and `predict` are shared, per-thread, or pooled; test each chosen topology for correctness and throughput; and collect per-utterance decode iteration counts so latency spread can be attributed to data-dependent decode.

### M2 - The load model can understate the very tail it is meant to measure.

Evidence: 0.1b stream generators emit perfectly periodic chunks every `chunk_ms` (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:204-208`), and synthetic finalizes are random heavier bursts (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:182-184`). Step 1 says "replaying" the session bundle (`PHASE2-PLAN.md:25-27`) but does not define arrivals, burstiness, utterance-length mix, or concurrent finalizer clustering.

Risk to goals: deterministic replay can make P95 look tight because finalizes and high-token decode loops are not adversarially clustered. Production tail is a queueing problem, not just per-chunk service time.

Recommended plan edit: add at least three Step 1 load modes: fixed periodic trace, recorded Python benchmark arrivals, and adversarial/bursty finalize clustering. The gate should be based on the recorded or adversarial mode, with fixed periodic as a lower-bound diagnostic.

### M3 - STOP/reassess semantics do not actually evaluate the stated fallback.

Evidence: the plan asks whether serialization means "MPS or per-context required" (`PHASE2-PLAN.md:17-19`) and says no overlap is STOP/reassess (`PHASE2-PLAN.md:30-32`). The 0.1b spec records that adding Python processes did not help because `K=3 == K=4` and MPS/BW contention cancels added processes (`spikes/0.1-overlap-ablation/0.1b-microbench-spec.md:3-8`), while also listing topology as an A/B axis (`spikes/0.1-overlap-ablation/0.1b-microbench-spec.md:29-34`).

Risk to goals: if single-context AOTI does not overlap, the plan has no immediate numeric fallback test to decide whether MPS/per-context native is density-relevant or only tail-relevant. That can create either a premature STOP or a vague reassessment.

Recommended plan edit: define a same-day fallback subtest: real compute under single-context, MPS/multi-process, and any green-context option. Require the same L40S SLO-robust density and TTFT spread metrics, plus memory overhead. If fallback cannot meet the L40S numeric gate, STOP; if it can only improve tail, reclassify the project as tail-only before funding.

### M4 - Current finalizer loader sharing creates a hot-bucket concurrency hazard that Step 1 must exercise.

Evidence: the runtime stores one `AOTIModelPackageLoader` per `(drop,T)` in a map (`runtime/cpp/session_main.cpp:1545-1580`). `run_finalize` and `run_finalize_runtime` look up that one loader and call `run()` on it (`runtime/cpp/session_main.cpp:2622-2635`, `runtime/cpp/session_main.cpp:2702-2716`). Under load, many sessions can finalize into the same `(drop,T)` bucket.

Risk to goals: a Step 1 trace with mostly distinct finalizer buckets can pass while production hot buckets serialize or race. That directly affects GPU utilization and TTFT P95.

Recommended plan edit: add a hot-bucket stress case where N threads finalize the same `(drop,T)` bucket concurrently, plus a per-bucket runner-pool design if one-loader concurrency is unsafe.

### M5 - Step 2's scheduler cannot be designed from a knee alone.

Evidence: Step 2 says "From Step 1's knee" design thread/stream ownership, finalize priority, admission, and backlog shedding (`PHASE2-PLAN.md:33-35`). But the metrics Step 1 names are only throughput, latency tail, GPU util, and knee (`PHASE2-PLAN.md:27-30`).

Risk to goals: scheduler design needs queue wait, lane wait, stream occupancy, finalize interference, and admission drop behavior. A single knee number is insufficient to tune streams/box or P95 tail.

Recommended plan edit: Step 1 must emit a scheduler telemetry schema: per-worker queue depth, service time by phase, CUDA event durations, finalizer wait, admission/backlog state, CPU core utilization, and shed/drop counters. Step 2 should be blocked on this telemetry, not just the knee.

### M6 - The plan's "validated single native stream" wording still hides known scope limits relevant to Phase 2.

Evidence: the plan states Phase 1 proved the single native stream "token/event-exact to the server's final transcript" (`PHASE2-PLAN.md:5-14`). The folded reviews narrowed earlier claims: stale-generation suppression is deferred to Phase 2 (`reviews/step1-event-FOLDED.md:24-28`), and current session output itself prints `stale_generation=DEFERRED_PHASE2_SERVER_ORACLE` in the coverage manifest (`runtime/cpp/session_main.cpp:3533-3537`).

Risk to goals: stale-generation correctness is a scheduler/tail issue. Under overload, wrong stale-final suppression can make TTFT tails look better by dropping or misordering finals.

Recommended plan edit: change the Phase 2 premise to "single-stream compute/event path validated against finalize_ref/session gates; scheduler generation suppression remains a Phase 2 server-oracle requirement." Add stale-final/generation suppression to Step 3, before Step 4 density benchmarking.

## MINOR

### m1 - First chunk remains TorchScript, not pure AOTI.

Evidence: session startup loads `enc_first.ts` (`runtime/cpp/session_main.cpp:3862-3864`), and first chunks run through `run_first_encoder` TorchScript (`runtime/cpp/session_main.cpp:1715-1726`). The harness summary explicitly warns that "pure-native runtime still requires an AOTI first chunk" (`runtime/cpp/session_main.cpp:3034-3036`).

Recommended plan edit: either exclude first-chunk work from Step 1's density claim and label it, or include an AOTI first-chunk package/topology in the Step 1 harness.

### m2 - GPU utilization needs timelines, not just NVML averages.

Evidence: 0.1b sampled average NVML GPU utilization (`spikes/0.1-overlap-ablation/microbench/microbench.cpp:192-223`). Step 1 asks for GPU util but not kernel overlap evidence (`PHASE2-PLAN.md:27-30`).

Recommended plan edit: require CUDA event timelines per worker and at least one Nsight/CUPTI trace at the knee. NVML average can stay as summary telemetry.

### m3 - The CMake/build surface does not yet include a Phase 2 harness target.

Evidence: `runtime/cpp/CMakeLists.txt` defines decode, pipeline, steady, aoti_encoder, finalize, and session targets (`runtime/cpp/CMakeLists.txt:11-56`) but no concurrent-dispatch density harness.

Recommended plan edit: add the Step 1 target name, inputs, and required runtime flags to the plan so review/build reproducibility is explicit.

## QUESTIONS

1. Which AOTI topology is Step 1 actually betting on: one shared loader, `num_runners=N`, per-thread loaders with shared constants, or something else?
2. Where is the constants-on-disk steady encoder package for shared weights? Is `enc_steady_aoti.pt2` being replaced by `enc_steady_codisk.pt2` or another artifact?
3. What exact TTFT/SLO definition is the gate using: `vad_stop->final`, client-observed final event, `vad_stop_recv_to_process`, or chunk intake-to-done?
4. What numeric target defines success for `P95-P50` spread versus the Python baseline?
5. Are `joint_step.ts` and `predict_step.ts` safe and intended to be called concurrently from N C++ threads, or does each worker own its own modules?
6. Does `AOTIModelPackageLoader::run(inputs, stream)` honor the supplied stream under concurrent calls, and does the loader require `num_runners > 1` to avoid internal serialization?
7. If single-context overlap fails, what precise MPS/per-context result is good enough to continue rather than STOP?
8. Will Step 1 use the audio/multiturn bundles and real finalizer buckets, or only steady chunks from the session bundle?

## Required plan edits before Step 1 build

1. Replace the Step 1 gate with a two-stage 5090 smoke + L40S numeric hard gate.
2. Pre-register loader topology, explicit stream usage, shared-weight artifact strategy, and concurrent correctness gates.
3. Include real finalizes and hot-bucket contention in the gated workload.
4. Define TTFT, SLO budget, P50/P95/P95-P50 targets, and load model.
5. Require serial-vs-concurrent token/event equality before any throughput number is accepted.
6. Add fallback topology semantics for MPS/per-context and a STOP threshold tied to L40S density plus TTFT spread.
