# Roofline and In-Code Lever Investigation

Status: complete.

## Q1. Roofline

### Inputs and assumptions

I use the supplied measurements as fixed facts: L40S conc-10 TTFT p50/p95 is 246/279 ms, dominated by the fixed
~200 ms VAD window plus ~23 ms WAN plus ~22 ms server finalize. The measured L40S finalize model slice is
~13.4 ms p50: ~9.7 ms CUDA-graphed finalize encoder plus ~3.6 ms eager RNNT decode. The measured steady RNNT
decode-only slice is ~0.829 ms at the real conc-10 small-B distribution.

The production path is not a tensor-core FP16 roofline. The model is loaded onto CUDA without a half/bfloat16
conversion, batching disables TF32 (`src/nemotron_speech/server.py:655`), and the measured fp16 variant was 0.79x
the speed of baseline. Therefore the realistic roofline below uses dense FP32/SIMT peak, not FP16/FP8/FP4 marketing
peaks. This is conservative for the current code and matches the observed launch-dispatch/CPU-bound behavior.

Model-side estimates:

- Parameters: ~0.6B. FP32 weight stream lower bound: `0.6B * 4 B = 2.4 GB` per encoder pass if weights are not
  resident in cache. L40S L2 is far smaller than this, so DRAM streaming is the right lower-bound model.
- Finalize model slice: encoder `T ~= 50` with 128 mel bins plus RNNT final decode. Central estimate:
  ~60 GFLOP and ~2.6 GB moved, including ~0.2 GB / a few GFLOP for eager decode and small activation/cache traffic.
- Steady decode-only slice: central estimate ~2.5 GFLOP and ~0.25 GB moved for the eager RNNT label-looping decode
  that was measured at ~0.829 ms.
- Full steady chunk, for context: non-first steady encoder input is `pre_encode_cache_size + shift_frames`, i.e.
  9 + 16 = 25 frames (`src/nemotron_speech/server.py:1526`, `src/nemotron_speech/server.py:1530`,
  `src/nemotron_speech/server.py:2072`). Counting the full encoder+decode chunk gives ~32.5 GFLOP and ~2.65 GB
  moved, but that is not the supplied ~0.829 ms decode-only measurement.
- The shipped `.nemo` config uses 128 mel bins. Feature tensor bytes are tiny compared with weights:
  `50 * 128 * 4 ~= 25 KB` for finalize and `25 * 128 * 4 ~= 13 KB` for a full steady chunk. Using 80 bins would
  understate the input feature shape, but it would not change the roofline conclusion because weight traffic
  dominates.

Hardware source links: RTX 5090 specs from NVIDIA's
[RTX Blackwell architecture PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf);
L40S and L4 specs from NVIDIA product pages
([L40S](https://www.nvidia.com/en-in/data-center/l40s/),
[L4](https://www.nvidia.com/en-us/data-center/l4/)); DGX Spark specs from the
[NVIDIA DGX Spark user guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html). DGX Spark publishes FP4 sparse
peak and CUDA core count, not an FP32 peak; I use ~30.9 TFLOP/s FP32 by 6,144-core Blackwell-class equivalence and
treat it as approximate.

| Platform | Memory bandwidth | Realistic peak used | Ridge point | Why this peak |
|---|---:|---:|---:|---|
| RTX 5090 | 1,792 GB/s | 104.8 TFLOP/s FP32 | 58.5 FLOP/B | Current code is FP32/eager; tensor peaks require a different precision/kernel path. |
| DGX Spark GB10 | 273 GB/s LPDDR5x | ~30.9 TFLOP/s FP32 | ~113 FLOP/B | Official 1 PFLOP is FP4 sparse; current code cannot use that. Low bandwidth is the key limit. |
| L4 | 300 GB/s | 30.3 TFLOP/s FP32 | 101 FLOP/B | FP16/BF16/FP8 tensor peaks are not representative for measured production path. |
| L40S | 864 GB/s | 91.6 TFLOP/s FP32 | 106 FLOP/B | This is the measured platform; current code also disables TF32 under batching. |

### Finalize encoder + decode roofline

Arithmetic intensity: `60 GFLOP / 2.6 GB ~= 23 FLOP/B`. This is below every platform's ridge point, so the model
slice is memory-bound even before considering launch overhead, host syncs, Python label-looping, or scheduler
contention.

| Platform | Compute floor | Memory floor | Roofline floor | 22 ms server-finalize distance | 13.4 ms model-slice distance |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 0.57 ms | 1.45 ms | 1.45 ms | ~15x | ~9x |
| DGX Spark GB10 | 1.94 ms | 9.52 ms | 9.52 ms | ~2.3x | ~1.4x |
| L4 | 1.98 ms | 8.67 ms | 8.67 ms | ~2.5x | ~1.5x |
| L40S | 0.66 ms | 3.01 ms | 3.01 ms | ~7.3x | ~4.5x |

Interpretation: the measured L40S finalize is nowhere near the RTX 5090 or L40S theoretical memory roof, but the
gap is not a compute gap. The measured decomposition already says GPU utilization is only ~46-65% and the bottleneck
is launch-dispatch/single-thread CPU behavior. The L40S encoder graph replay at ~9.7 ms is ~3.2x above the L40S
memory floor for the whole finalize model estimate, and the full server-finalize wall is ~7x above that floor once
eager decode, preproc, clone/scatter, and lane waits are included.

DGX Spark is the outlier to call out explicitly: 273 GB/s LPDDR5x gives a ~9.5 ms memory floor for the same finalize
model slice, before Python and launch overhead. It is capacity-rich but bandwidth-poor for this workload, so it is
likely memory-bound and unlikely to improve TTFT versus L40S for this FP32 eager server.

### Steady decode-only roofline

Arithmetic intensity for the measured decode-only slice: `2.5 GFLOP / 0.25 GB ~= 10 FLOP/B`. That is also
memory-bound on all four platforms. This comparison is intentionally for the supplied ~0.829 ms eager RNNT decode
measurement, not the full steady encoder+decode chunk.

| Platform | Compute floor | Memory floor | Roofline floor | 0.829 ms measured decode distance |
|---|---:|---:|---:|---:|
| RTX 5090 | 0.02 ms | 0.14 ms | 0.14 ms | ~5.9x |
| DGX Spark GB10 | 0.08 ms | 0.92 ms | 0.92 ms | not comparable; GB10 floor is already ~0.8-0.9 ms |
| L4 | 0.08 ms | 0.83 ms | 0.83 ms | ~1.0x, but this uses an L40S measurement on an L4 floor |
| L40S | 0.03 ms | 0.29 ms | 0.29 ms | ~2.9x |

The steady decode result reinforces the same conclusion: the actionable gap is launch/dispatch and Python-side
RNNT label-looping, not raw FLOPs. The fact that 86% of decodes are B=1 makes tensor-core throughput mostly
irrelevant unless the implementation changes enough to batch or graph the label loop safely.

For the full steady chunk context, the non-first encoder+decode estimate is ~32.5 GFLOP / ~2.65 GB, or ~12 FLOP/B.
That is still memory-bound, with rough floors of 1.48 ms on RTX 5090, 9.71 ms on DGX Spark GB10, 8.83 ms on L4,
and 3.07 ms on L40S. Since the supplied measured steady number is decode-only, I do not use those full-chunk floors
as the distance denominator for the 0.829 ms datapoint.

## Q2. In-Code Levers

The ranking below assumes the measured bottleneck is correct: launch-dispatch and single-thread host orchestration,
not raw GPU compute. I am ranking by expected TTFT impact first, then by effect on useful in-budget parallelism.

| Rank | Lever | TTFT impact | Parallelism impact | Code anchors |
|---:|---|---|---|---|
| 1 | Enforce an in-budget admission/backpressure cap from observed scheduler backlog. | Highest production impact. It prevents the measured `vad_stop_recv_to_process` blow-up from turning a 200 ms VAD window into ~1 s server backlog at overload. | Reduces nominal max streams, but raises useful in-budget streams. The measured production limit is ~20/box L40S K=3, not 48. | Sessions are admitted into `self.sessions` without a capacity gate at `src/nemotron_speech/server.py:4163`; scheduler queues are per-session and deep at `src/nemotron_speech/server.py:4326`; enqueue backpressure is only `await queue.put` at `src/nemotron_speech/server.py:4341`; the exact backlog metric is recorded at `src/nemotron_speech/server.py:2790`. |
| 2 | Make finalize lane scheduling priority-aware. Do not let finalizes sit behind steady work once VAD stop/debounce is due. | Directly attacks p95/p99 TTFT spread. The measured p95 spread is mostly lane/lock wait, not model time. | Slightly reduces steady throughput during finalize bursts, but improves visible latency under mixed steady/final workloads. | Finalize events are collected before ready batches at `src/nemotron_speech/server.py:4514`, but finalization still waits for the pinned model lane at `src/nemotron_speech/server.py:6755`; steady batches reserve lanes and create in-flight lane tasks at `src/nemotron_speech/server.py:5077` and `src/nemotron_speech/server.py:5140`; lane release is at `src/nemotron_speech/server.py:5281`. |
| 3 | Compress host synchronizations and per-batch CPU launch work in steady and finalize paths. | Medium to high for keep-up, modest for p50 TTFT. This is the most direct code-level attack on the launch-dispatch ceiling. | Improves per-process steady keep-up by reducing CPU occupancy per chunk. Risk is correctness around tensor state visibility and scatter. | Steady path synchronizes before preprocess, after preprocess, after model, and after scatter at `src/nemotron_speech/server.py:8223`, `src/nemotron_speech/server.py:8243`, `src/nemotron_speech/server.py:8329`, `src/nemotron_speech/server.py:8399`; finalize has syncs around group processing/model/scatter at `src/nemotron_speech/server.py:7300`, `src/nemotron_speech/server.py:7341`, `src/nemotron_speech/server.py:7430`, `src/nemotron_speech/server.py:7513`. |
| 4 | One-shot finalize preprocessor, byte-exact gated. | Modest but real p50 win: measured preproc is ~2.4 ms across 3 invocations, so likely ~1-2 ms headroom. It also shortens lane hold. | Small positive effect on finalize bursts; does not fix steady keep-up alone. | Finalize walks remaining frames in a loop at `src/nemotron_speech/server.py:6927` and calls `_preprocess_fixed_audio` at `src/nemotron_speech/server.py:6941`; the batched-preproc path still groups repeated invocations and falls back to per-row calls at `src/nemotron_speech/server.py:7121` and `src/nemotron_speech/server.py:7173`. |
| 5 | Increase effective B for steady decode without adding user-visible wait. | Low to medium. 86% of decodes are B=1, so each chunk pays B=1 Python/launch overhead. | Could improve GPU efficiency, but raising `batch_max_wait_ms` directly trades latency for batching and may hurt TTFT. | Batch wait/size defaults are `src/nemotron_speech/server.py:669` and `src/nemotron_speech/server.py:670`; candidate selection favors larger ready batches then deadline at `src/nemotron_speech/server.py:4918`; batch grouping includes decoder-state freshness in the key at `src/nemotron_speech/server.py:4808`. |
| 6 | More in-process model lanes. | Usually not a good TTFT lever now. More lanes add model copies and more launch threads, but the measured limit is already CPU/dispatch plus memory footprint. | May improve per-process overlap until CPU/GPU dispatch saturates; hurts density. Per-proc memory is already ~11-12 GB. | Lane 0 reuses the main model, additional lanes load separate models at `src/nemotron_speech/server.py:3150`; each lane is a single-worker threadpool plus one CUDA stream at `src/nemotron_speech/server.py:3155` and `src/nemotron_speech/server.py:3161`; every lane call synchronizes the stream at `src/nemotron_speech/server.py:3175`. |

Low-ROI or non-levers from the measured facts:

- RNNT decoder CUDA graphing is not the current answer. The server deliberately configures greedy_batch with
  `use_cuda_graph_decoder=False` at `src/nemotron_speech/server.py:1463`; the prior probe found small-B graph
  replay unsafe and only ~0.83 ms steady decode upside.
- fp16 is not a lever for this code path because it measured 0.79x slower.
- Clone/scatter cleanup is below the noise floor for TTFT: finalize clone/hyp stack is at
  `src/nemotron_speech/server.py:7371`, finalize scatter is at `src/nemotron_speech/server.py:7443`, and the
  measured combined clone/scatter contribution is <1 ms.

## Q3. Brief From-Scratch Verdict

A from-scratch runtime could get materially closer to the roofline, but the win is on the server-finalize slice, not
the whole TTFT. On L40S, the model roofline floor for finalize is ~3 ms, while the measured model slice is ~13.4 ms
and server finalize is ~22 ms. A well-engineered C++/CUDA or TensorRT-style runtime with static buffers, full encoder
graphing, fused/graphable RNNT label-loop decode, CUDA-event dependency tracking, and no per-chunk Python scatter/sync
could plausibly move server finalize from ~22 ms into the ~6-10 ms range. End-to-end TTFT would still be bounded by
the fixed ~200 ms VAD window plus ~23 ms WAN, so the p50 user-visible gain is likely tens of milliseconds, not 10x.

The real architectural fix is removing the GIL/single-thread launch ceiling: one persistent GPU worker/runtime should
own admission, batching, lane priority, graph replay, decode state, and output scatter without bouncing through Python
for every chunk. That is also the path to higher in-budget parallelism. Without that, buying more FLOPs or enabling
fp16/tensor peaks does not address the measured keep-up failure.
