# Round 1 inference optimization analysis

Date: 2026-05-21. Scope: analysis only. No `server.py`, NeMo, cloud, or commit changes were made.

Target: improve (A) TTFS / finalize latency and (B) per-instance realtime keep-up knee for
`src/nemotron_speech/server.py` on AWS EC2 g6/L4 and g6e/L40S.

## Ground truth

The current bottleneck is not "raw GPU too slow" at one lane. The prior forced-batch sweep found a local
forced-batch ceiling of 184 stream-equivalents at B=32 and 220 at B=46, with T(B) still sublinear
(`proj-2026-05-21-0410/max-parallelism-sweep.md:23-40`). The practical out-of-phase realtime knee was only 56
because independent 160 ms streams reached the scheduler in small groups, avg B about 4.16 at the knee
(`max-parallelism-sweep.md:112-129`).

The EC2 g6 measurements supplied for this round tighten the conclusion for production hardware:

- g6/L4 + EPYC Milan, 16 vCPU: B=1 knee about 4-8, `server.py` one core about 100%, GPU about 46%.
- Batching alone did not raise the low-N keep-up knee on Milan because batches did not form before the CPU lane
  saturated.
- `NEMOTRON_MODEL_LANES=2` raised the g6 knee to about 16 and GPU utilization to about 92%.
- `NEMOTRON_MODEL_LANES=4` regressed to about 4, consistent with GPU SM/stream oversubscription.
- Therefore g6/L4 is now effectively GPU-bound at lanes=2 for normal steady B=1-ish traffic. Further low-N
  launch optimizations will stack only partially on g6 after lanes=2; they should stack better on g6e/L40S if
  the larger GPU has headroom.

Relevant code shape:

- One global `inference_lock` still guards non-lane calls and is also acquired inside several lane-exclusive
  paths (`server.py:514`, `server.py:3269-3284`, `server.py:3583-3608`, `server.py:4785-4824`).
- Lanes are available only with scheduler + batching enabled (`server.py:516-565`). They use one restored model
  replica, one worker thread, and one CUDA stream per lane (`server.py:1646-1729`).
- Only steady normal chunks with same key can run concurrently across lanes; first chunks, barrier drains,
  finalize, and other non-steady geometry are exclusive (`server.py:1779-1818`, `server.py:1905-1936`).
- The normal batched ready path already stacks mels, caches, hypotheses, and pred state, then scatters owned
  clones back (`server.py:5197-5391`, `batch_primitives.py:59-97`).
- The `vad_stop` barrier path drains a single session one ready chunk at a time through `_process_chunk`, B=1
  and exclusive under lanes (`server.py:3732-3759`, `server.py:3568-3608`).
- Finalize clones a fork, appends silence padding, then runs one B=1 `_process_final_chunk` with
  `keep_all_outputs=True`; with lanes it takes the exclusive model path (`server.py:4581-4631`,
  `server.py:4723-4832`, `server.py:5657-5757`).
- `NEMOTRON_WARMUP_MS=200` runs a real per-session warmup model call in `_init_session`; connection init and
  true-boundary cold reset can serialize this work (`server.py:671-684`, `server.py:2156-2202`,
  `server.py:4939-4952`).
- rc1 final padding is `(right_context + 1) * shift`, so rc1 pads 32 frames / 320 ms of synthetic audio into
  the final fork; rc0 would pad 16 frames / 160 ms (`server.py:383-389`, `server.py:1357-1362`).
- Decoder config uses greedy/greedy_batch `max_symbols=10` and decoder CUDA graphs disabled
  (`server.py:1284-1314`).

## Axis A: TTFS / finalize candidates

These are ordered by value/risk for the next implementation phase. "TTFS" here includes first ready/first
transcript latency and end-of-utterance final emission latency.

| Rank | Candidate | Mechanism and exact code location | Expected gain | Effort | Risk | Interaction with lanes / batching / graphs | Test needed |
|---:|---|---|---|---|---|---|---|
| A1 | Batch the `vad_stop` barrier drain | Replace the per-session while loop in `_scheduler_drain_ready_barrier_locked` (`server.py:3732-3759`) that calls `_scheduler_process_one_ready_chunk_locked` (`server.py:3568-3608`) with a scheduler-visible barrier-drain queue that can group same-key ready barrier chunks through `_process_ready_batch` (`server.py:5197-5391`). Preserve event ordering by resolving each control event only after that session's backlog reaches not-ready. | High-N TTFS/finalize cliff fix. In-phase N=120 currently fails at TTFS p95 2567 ms with 452 B=1 barrier chunks; N=150 has 2236 B=1 barrier chunks and TTFS p95 21.5 s (`inphase-confirmation.md:58-88`). Expected N=120 to return under 400 ms and in-phase knee 115 -> 150-180 if no new limiter appears. Single-stream gain: none. | M | Medium: ordering, generation invalidation, state locks, close/reset semantics. | Batching is the core mechanism. Lanes should allow steady barrier batches as normal steady work instead of lane-exclusive B=1. CUDA graphs can engage for steady T=25 barrier chunks if wired for batched steady. | Local first: in-phase N=115/120/150 with `FORK_ASSERT=1`, compare finals. Cloud g6/g6e after correctness, because EC2 Milan event storms are the target. |
| A2 | Remove global lane exclusivity for final/fork when model replicas make it safe | `_continuous_finalize_emit_locked` currently enters `_scheduler_exclusive_model_path` for lanes (`server.py:4785-4808`), blocking all steady lane work. Because lanes now use per-lane model replicas (`server.py:1646-1700`), route final work to the session's pinned lane and block only that session/lane, not every lane. Keep same-session in-flight exclusion. | Mixed streaming + final workload: saves one or more in-flight batch waits per final. On g6 lanes=2, expected 10-50 ms p95 under moderate load and larger wins under final storms; no single-stream compute gain. | M | Medium-high: the original exclusivity protects mixed `drop_extra` and model-global mutation. Need prove replica-local mutation and prompt state are isolated. | Only valuable with lanes>1. Complements A1/A3. Graphs probably fall back for final variable T. | Local lanes=2 mixed test: N steady streams while M streams finalize; strict final diff. Cloud g6 because g6 lanes=2 is the production sweet spot. |
| A3 | Batch concurrent final fork-flush calls | Add a finalization batch path for forks built by `_build_continuous_finalize_fork` (`server.py:4581-4631`) before `_process_final_chunk` (`server.py:5657-5757`). Group by target_lang, `keep_all_outputs=True`, `drop_extra`, final chunk_T, decoder mode, fresh/established decoder state, then use the same stack/scatter invariants as `_process_ready_batch`. | For simultaneous `vad_stop`/close/end storms, convert O(N) final model calls to O(N/B). If 32 finals share T, final model dispatch can drop roughly 5-15x versus serial B=1. Single-stream gain: near zero except maybe preprocessor batching. | L | High: `keep_all_outputs=True`, variable final T, fork state isolation, final text deltas, and deep-cloned RNNT state all need gates. | Batching lever. Lanes can run one final batch on a lane; do not expect CUDA graphs unless final T buckets are captured. | Local only first: synthetic aligned finalize storm with fixed pending lengths, then real clips. Cloud only after byte-exact final deltas. |
| A4 | Eliminate or batch warm200 per-session ready contention | `_init_session` calls `_run_session_warmup` when `NEMOTRON_WARMUP_MS>0` (`server.py:2156-2202`), and websocket init holds exclusive/locked model access (`server.py:2705-2721`). Options: precompute a per-lane/per-language silent-warm template and clone it into new sessions, or batch connection warmups with the existing B path. | Concurrent connect ready latency becomes O(clone) or O(N/B) instead of O(N B=1 warm calls). On g6, if one warm call is 10-20 ms, 16 concurrent connects can save about 150-300 ms p95 ready latency; on cloud this may be the difference between staying below and above 400 ms. | M | Medium: must prove silent warm template is byte/state-equivalent, no shared tensor aliasing, prompted model state isolated. | Batching can implement batched warmup; lanes require per-lane template or lane-specific cloned tensors. Graphs can capture warmup bucket if static. | Local connect-storm test N=1/4/16/64; fork/assert alias checks. Cloud g6/g6e for ready latency and memory. |
| A5 | rc0 / shorter final padding experiment | Right context options include rc0 and rc1 (`server.py:383-389`). Final padding is `(rc+1)*shift` (`server.py:1357-1362`), so rc1 appends 320 ms synthetic audio and rc0 appends 160 ms. Test rc0 globally or a final-only shorter padding variant. | Compute: final chunk T drops by 16 frames for rc1->rc0, likely 10-30% faster final call depending on pending audio. User-visible: if downstream waits on acoustic lookahead, possible 160 ms earlier final eligibility. | S for config probe, M for final-only variant | Medium-high product risk: last-word stability and WER may regress. Final-only rc is not guaranteed valid if encoder context is model-global. | Helps all modes. If g6 lanes=2 is GPU-bound, shorter T also raises final throughput slightly. Graphs unaffected unless final T captured. | Local rc0 A/B: final text/WER/delta stability on endpointing set, TTFS/final p95. Cloud only if local quality acceptable. |
| A6 | Manual CUDA graphs for B=1/small-B steady, plus warmup bucket | Existing `torch.compile` path is in `_conformer_stream_step`/`_configure_encoder_compile` (`server.py:1437-1632`) but cloud compile failed. Manual probe showed B=1 steady encoder graph is byte-exact, 251 ms capture, 1.36x synced speedup (`manual-cudagraph-probe.md:20-40`). Wire per-B graph replay at `_process_ready_batch` dispatch (`server.py:5290-5311`) and possibly `_run_session_warmup` bucket (`server.py:2173-2202`). | Low-N TTFS and knee: B=1 step 8.845 -> 6.510 ms locally. On g6 lanes=1, expect B=1 knee 4-8 -> 6-12; on g6 lanes=2 already GPU 92%, expect smaller 1.0-1.25x. On g6e, likely 1.3-1.8x until GPU-bound. | L, but already planned | Low-medium for steady; medium for warmup. Must remain fail-closed and byte-exact per B. | Partial substitute for lanes because both attack launch overhead. Stacks best before GPU saturation and at small B. Does not fix B=1 barrier/final unless those paths are routed/captured. | Local per-B B=1..K byte/state exact, then cloud g6/g6e. Cloud is mandatory because the measured bottleneck is Milan launch dispatch. |
| A7 | Profile and tune finalize decoder work (`max_symbols`) | Decoder uses `max_symbols=10` for greedy/greedy_batch (`server.py:1289-1314`). `_process_final_chunk` uses `keep_all_outputs=True` (`server.py:5735-5745`), which may spend extra RNNT decode work on trailing silence. Add profiling around decode/encoder if NeMo exposes it; test lower final-only `max_symbols` or early blank stop. | Unknown until profiled. If final decode is 20-40% of final step, a safe final-only cap could save 5-20 ms per final and 5-10% high-N final throughput. | S to profile, M to implement | High correctness risk: `max_symbols` can truncate fast speech or alter transcripts. Not byte-exact by design unless unchanged. | Helps after lanes make GPU/decoder active time the ceiling. Graphs do not cover decoder per current plan. | Local profiler first, then transcript/WER gate. Cloud only if profile shows meaningful decode share. |
| A8 | Background speculative finalization during debounce | Current speculative final paths exist for reset/debounce (`server.py:4075-4084`, `server.py:4528-4533`, `server.py:4894-4914`). If nonzero debounce is used, start fork-flush immediately on `vad_stop`, hold result, and commit or discard on debounce expiry / `vad_start`. | Hides 10-50 ms final compute behind debounce. If production uses `NEMOTRON_FINALIZE_SILENCE_MS=0` as in the benchmark, gain is near zero. If default 150 ms is used, can make final emission essentially timer-bound. | M | Medium: stale forks, duplicate final deltas, barge-in semantics. | Adds final work earlier; needs A1/A2 to avoid worsening high-N final storms. | Local barge-in and duplicate-delta tests. Cloud only if production debounce >0. |
| A9 | Reduce fork clone/copy overhead | `_build_continuous_finalize_fork` deep-clones audio, cache tensors, hypotheses, and pred state (`server.py:4581-4631`), and optional fork assert snapshots clone again (`server.py:4633-4694`). Replace with clone-on-write or pool fixed silence/audio buffers; keep assert off in prod. | Probably small: existing logs already record `fork_clone_ms`; expected 1-5 ms per final, maybe 10 ms if hypotheses grow. Worth only if telemetry shows clone is nontrivial. | S/M | Medium if clone elision aliases parent state. | Orthogonal. Does not move keep-up knee unless final storms are clone-CPU-bound, which current evidence does not show. | Local telemetry grep first; implement only if p95 clone cost is visible. |

## Axis B: realtime scaling / keep-up knee candidates

The scaling question has two regimes:

- g6/L4 with lanes=2 is already near GPU saturation for current traffic (GPU about 92%). After that point,
  launch-only optimizations are partly redundant unless they also reduce GPU work or form real batches.
- g6e/L40S should have more GPU headroom. Lanes, graphs, and batching should stack until the larger GPU becomes
  full, but the correct lane count has to be measured. The g6 lanes=4 regression proves "more lanes" is not
  monotonic.

| Rank | Candidate | Mechanism and exact code location | Expected gain | Effort | Risk | Interaction with lanes / batching / graphs | Test needed |
|---:|---|---|---|---|---|---|---|
| B1 | Per-GPU lane-count sweep and default matrix | Use landed `NEMOTRON_MODEL_LANES` path (`server.py:522-565`, `server.py:1646-1729`, `server.py:1779-1818`) but treat lane count as per-GPU. g6 evidence says lanes=2 is sweet spot and lanes=4 regresses. Sweep g6e lanes 1/2/3/4/6, then stop at first clear regression. | Already measured on g6: B=1 baseline 4-8 -> lanes=2 about 16, GPU 46% -> 92%; lanes=4 -> about 4. For g6e, expected lanes=3-6 may reach about 32-64 if L40S compute headroom absorbs extra streams. | S | Low if config-only; medium if adding auto-tune. | Lanes require scheduler+batch enabled by current checks (`server.py:553-559`). Batching may not form at low N but must be on for lanes. Graphs stack mostly before GPU-bound. | Cloud g6/g6e mandatory. Local 5090 lane data is not predictive enough for Milan/L4/L40S. |
| B2 | Batch the barrier drain | Same as A1, but viewed as scaling. The B=1 loop at `server.py:3732-3759` collapses effective batch size during control storms. | In-phase local knee 115 -> toward 150-180. At N=150, avg B ready-pass was 31.35 but effective avg B including 2236 barrier chunks fell to 4.67 (`inphase-confirmation.md:40-88`). Fix removes the known high-N cap before the forced-batch ceiling. | M | Medium | Strongly stacks with batching and phase alignment. With lanes, barrier chunks should not force global exclusive B=1. Graphs can help steady barrier chunks. | Local in-phase first; cloud g6e high-N once g6e is measured. |
| B3 | Manual per-B CUDA graphs | Implement the existing plan: per-B graph capture B=1..K, no padding, fallback eager (`proj-2026-05-21-1959-cudagraph/PLAN.md:6-17`, `:77-84`). Wire at `_process_ready_batch` model dispatch (`server.py:5290-5311`), static buffers mirroring `batch_primitives.py:59-97`. | B=1 steady probe: 1.36x local. Expected g6 lanes=1 1.3-1.8x, g6 lanes=2 only 1.0-1.25x because GPU is about 92%. Expected g6e 1.3-2.0x with enough lanes until GPU-bound. | L | Low-medium steady; medium multi-lane static buffer safety. | Partial substitute for lanes. Complements batching at avg B about 1-8; diminishing at large B. Per-lane graph buffers are required when lanes run concurrently. | Local byte/state exact per B first, then cloud g6/g6e. |
| B4 | Coarse phase alignment / global dispatch tick | The scheduler currently dispatches when solo/max/timer conditions fire (`server.py:3138-3214`) with `NEMOTRON_BATCH_MAX_WAIT_MS=8` (`server.py:580-581`). Introduce a bounded global tick or client/server phase bucket (for example 16-40 ms) so independent streams form larger ready groups without exceeding 400 ms TTFS. | Local proof: out-of-phase MAX_SIZE=32 knee 56, in-phase knee 115, about 2.1x (`SUMMARY.md:47-50`, `inphase-confirmation.md:71-82`). On g6 lanes=2, expect less than 2.1x if GPU is already full, but real batches can still improve work/stream. On g6e, likely a major lever. | M | Medium: added latency, fairness, jitter, burstiness. | Needs batching. Stacks with lanes only while GPU has headroom or batching reduces GPU work/stream. Graphs help small-B tick batches. | Local tick sweep 8/16/24/32/40 ms; then cloud g6/g6e with TTFS p95 <400. |
| B5 | Per-GPU starting configs and guarded auto-selection | Encode measured profiles into deployment config, not one universal default. Current code has `batch_max_size`, `batch_max_wait_ms`, memory cap, and lane count knobs (`server.py:580-605`, `server.py:995-1028`). | Prevents known regressions. g6 recommended starting point: lanes=2, MAX_SIZE=16 or 32, MAX_WAIT=8, graphs off until tested. g6e starting sweep: lanes=2/3/4/6, MAX_SIZE=32 then 64 if memory, MAX_WAIT=8/16/24, graphs K=8-16. | S | Low | Makes lanes/batching/graphs conditional on measured knee, GPU util, avg B, and memory. | Cloud g6/g6e. Local only for harness sanity. |
| B6 | Remove unconditional synchronizations and heavy telemetry from hot batched path | `_process_ready_batch` synchronizes at entry, after preprocessor, after model, and after scatter (`server.py:5203-5225`, `server.py:5290-5311`, `server.py:5380-5390`). Keep correctness syncs only where CPU consumes GPU results; gate memory telemetry and use CUDA events for timing. | Expected 5-15% throughput gain in batched/lane mode, especially with lanes because extra syncs reduce stream overlap. Could be worth 1-3 streams on g6 lanes=2 and more on g6e. | M | Medium: hidden async ordering bugs, telemetry changes. | Directly improves lanes and batching overlap. Graph replay benefits from fewer forced sync points. | Local correctness with CUDA_LAUNCH_BLOCKING off/on, then g6/g6e perf. |
| B7 | Relax mixed-key lane concurrency with per-lane model replicas | Current global active key permits concurrent lanes only for one steady key; non-steady keys become exclusive (`server.py:1779-1818`, `server.py:1905-1936`). Since lane models are replicas, allow different keys on different lanes when sessions are disjoint and prompt state/drop_extra mutation is lane-local. | Mixed workloads: 10-30% throughput/latency gain when first chunks, warmups, finals, and steady chunks overlap. Low impact on homogeneous steady traffic. | M/L | High: prompt switching, `drop_extra_pre_encoded`, and RNNT mutable state must be proven lane-local. | Extends lanes beyond steady-same-drop-extra. Batching still groups same key within each lane. Graphs can replay per lane/key if bucket captured. | Local mixed workload strict canary; cloud g6 after. |
| B8 | Precision / dtype path: bf16/fp16 inference after lanes make GPU the ceiling | Current server does no `autocast`, half, or bf16 path; batching disables TF32 for byte compatibility (`server.py:566-572`). Test model/encoder under bf16/fp16 autocast or converted weights, probably English-only first. | Once g6 lanes=2 is GPU-bound, precision is one of the few ways to push the GPU-compute ceiling. Expected 1.2-1.8x if tensor-core eligible and numerically stable; possibly zero if NeMo/RNNT remains launch or decoder bound. | M/L | High: not byte-exact, possible WER/regression, cache drift, RNNT instability. | Most valuable after lanes. Batching/graphs reduce launch; precision reduces GPU active time. | Local WER/text stability and perf first. Cloud g6/g6e if quality acceptable. |
| B9 | Optimize Python scatter/clone/preprocess orchestration | Hot path deep-clones hypotheses/pred state and scatters cache clones (`server.py:5268-5277`, `server.py:5313-5378`; `batch_primitives.py:90-97`). Preprocessor batching is present (`server.py:5033-5063`) but audio staging uses NumPy stacks/copies. Reduce clone frequency, pool tensors, and measure CPU time outside CUDA. | Expected 5-15% at high B, little at low-N B=1. Helps high-N robustness and g6e larger batches. | M | Medium: aliasing and state corruption are the exact hazards earlier gates caught. | Batching-only lever. Less relevant once g6 low-N is GPU-bound; more relevant after phase alignment/barrier batching creates large B. | Local microbench forced B=8/16/32 with strict state checks. Cloud only if local CPU share visible. |
| B10 | Memory model: lane replicas, per-stream cache, and MAX_SIZE on 24GB vs 48GB | Current memory cap estimates extra rows and clamps `batch_max_size` (`server.py:939-1028`), and telemetry logs retained session cache bytes (`server.py:5160-5195`). Lanes duplicate the 0.6B model per lane (`server.py:1646-1691`). Build a memory budget model for active sessions x lanes x max batch. | Not a direct latency win unless memory is capping lanes/batch. It prevents OOM and may enable larger g6e configs. L4 24GB should fit lanes=2; L40S 48GB likely enables more lanes and larger MAX_SIZE. | S/M | Low for measurement; medium if changing cap formula. | Required for lanes>2 and MAX_SIZE>32. Graph static buffers add per-B/per-lane memory. | Cloud g6/g6e memory telemetry under connect churn and high N. |
| B11 | Chunk/right-context geometry experiments | rc0 reduces right context and final padding (`server.py:383-389`, `server.py:1357-1362`). Smaller shift chunks would lower per-call T but double call rate; larger chunks would reduce call rate but raise first-token latency. | Smaller chunks are likely negative in launch-bound regimes. rc0 may save 5-10% normal/final compute and 160 ms synthetic final padding. Larger 320 ms chunks could improve scaling up to about 2x but risks the 400 ms TTFS target. | M/L | High: model config/training assumptions, WER, UX latency. | Geometry changes all lanes/batches/graphs; graphs need recapture per shape. | Local quality/perf experiment only. Cloud if a product latency/accuracy trade is accepted. |
| B12 | Adaptive coalescing rather than fixed MAX_WAIT | Current batching waits until solo, max_size, or a fixed deadline (`server.py:3138-3163`). Adapt wait based on utilization, avg B, queue lag, and TTFS budget: near knee increase wait to form batches; below load dispatch immediately. | Could recover part of the phase-alignment win without always paying a fixed delay. Expected 1.1-1.5x in mid/high N where streams are almost alignable; no low-N gain. | M | Medium: control-loop instability, p95 latency spikes. | Batching lever. Complements lanes/g6e; less useful when g6 lanes=2 is already GPU-bound unless it increases batch efficiency. | Local sweep with synthetic phase jitter; cloud g6/g6e once stable. |

## Lanes, batching, and graphs: how they stack

On g6/L4, the new EC2 lane result changes the priority order. One lane leaves the L4 half idle because a Milan
single core cannot dispatch enough tiny kernels. Two lanes fill the GPU to about 92% and move the knee to about
16. At that point:

- More lanes are not automatically better; lanes=4 regressed to about 4.
- CUDA graphs are still worth testing, but mostly for lanes=1, g6e, TTFS, and reducing CPU headroom. On g6
  lanes=2 they are likely a small incremental win unless they also reduce GPU-active time.
- Batching only stacks if real batches form. Independent low-N g6 traffic did not form them. Phase alignment,
  adaptive coalescing, and barrier batching are what make batching relevant again.
- Once GPU-bound, precision and chunk geometry are the levers that can raise the compute ceiling; they carry
  higher transcript-quality risk than scheduler work.

For g6e/L40S, assume nothing until measured. The likely path is lanes until GPU is near full, graphs to reduce
per-lane launch cost, then batching/phase alignment to improve work per stream. The 48GB memory budget should
make more lane replicas and graph buckets feasible, but stream/SM contention can still create a lanes=4-style
regression.

## Ranked shortlist for next phase

### Top TTFS wins

| Rank | Win | Expected gain | Test flag |
|---:|---|---|---|
| 1 | Batch `vad_stop` barrier drain (A1) | High-N final TTFS from seconds back toward <400 ms; in-phase knee 115 -> 150-180 if no new limiter. | Local first, then cloud g6/g6e |
| 2 | Remove global lane exclusivity for final on replica lanes (A2) | Mixed steady/final p95 saves 10-50 ms on g6 lanes=2, larger under final storms. | Local first, then cloud g6 |
| 3 | Fix warm200 connect contention via template or batched warmup (A4) | Concurrent connect ready p95 saves about 150-300 ms at N=16 if warm calls are 10-20 ms each. | Local first, then cloud g6/g6e |
| 4 | rc0 / shorter final padding experiment (A5) | 10-30% faster final call and possible 160 ms less lookahead, subject to quality. | Local quality gate first |
| 5 | Manual graphs for small-B/warm buckets (A6) | B=1 step 1.36x local; cloud low-N TTFS/knee could improve 1.3-2.0x before GPU-bound. | Local correctness, then cloud g6/g6e |

### Top scaling wins

| Rank | Win | Expected gain | Test flag |
|---:|---|---|---|
| 1 | Per-GPU lane-count sweep and config matrix (B1/B5) | g6 already 4-8 -> 16 with lanes=2; g6e may reach 32-64 if 3-6 lanes fit before oversubscription. | Cloud g6/g6e |
| 2 | Batch `vad_stop` barrier drain (B2) | In-phase high-N cap 115 -> 150-180; removes thousands of B=1 barrier calls. | Local first, then cloud g6e |
| 3 | Manual per-B CUDA graphs (B3) | g6 lane1 1.3-1.8x; g6 lanes=2 1.0-1.25x; g6e 1.3-2.0x until GPU-bound. | Local correctness, then cloud g6/g6e |
| 4 | Coarse phase alignment / adaptive coalescing (B4/B12) | Proven 56 -> 115 local in fully aligned case; practical bounded-tick target 1.1-1.5x without exceeding 400 ms. | Local tick sweep, then cloud g6/g6e |
| 5 | Precision path once GPU-bound (B8) | Potential 1.2-1.8x compute-ceiling lift after lanes fill GPU, but not byte-exact. | Local quality/perf first, then cloud |

## Single highest-value next experiment

Run a cloud lane-count/config sweep on g6 and g6e before implementing more code:

- g6: confirm `lanes=1/2/3/4`, with batching enabled because lanes require it, MAX_SIZE 32, MAX_WAIT 8.
- g6e: sweep `lanes=1/2/3/4/6`, MAX_SIZE 32, MAX_WAIT 8, record GPU util, CPU core util, avg B, lane wait,
  TTFS/final p95, and memory.
- Stop each sweep at the first obvious regression, as g6 lanes=4 already regressed.

This is the highest-value experiment because it sets the actual production ceiling and tells whether the next
implementation should prioritize L40S lane scaling, CUDA graphs, or GPU-compute reductions. In parallel, the
highest-value code prototype is A1/B2: batch the `vad_stop` barrier drain, because it has a known code location,
a measured failure signature, and a quantified upside.
