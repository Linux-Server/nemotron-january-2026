# L40S profiling analysis: higher-concurrency levers

Date: 2026-05-27

## Verdict

The profiling does not support a single-cause answer. It shows both:

1. **Launch pressure is real:** Nsys reports 2,158,786 launch API calls in the capture, with `cudaLaunchKernel` plus
   `cuLaunchKernel` taking 94.3% of CUDA API time and a weighted average launch API cost of about 3.69 us
   (`runtime/artifacts/l40s_w3_logs/profiling/nsys_stats.txt:204-212`).
2. **The dominant GEMMs are poor GPU work units:** the sampled SGEMMs are small-grid, low-occupancy kernels. One
   `ampere_sgemm_64x32` sample runs 128 blocks on 142 SMs, 0.45 waves/SM, 16.68% achieved occupancy, 72.36% DRAM
   throughput, and 34.39% SM throughput (`runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm.csv:96-103`,
   `runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm.csv:143-165`). The 32x32 sample is similar: 71.23% DRAM,
   39.05% SM, 15.03% occupancy (`runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm.csv:5-12`,
   `runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm.csv:50-72`).

Ranked by expected concurrency gain, I would route work as:

| Rank | Lever | Expected knee effect | Confidence |
|---:|---|---:|---|
| 1 | **Cross-stream batching, starting with batched decode/joint and then steady encoder if B-fill is real** | **N=37 -> 40-44 likely if weak/good B-fill; 44-48 plausible only if opportunity + B=4 per-row gain pass** | Medium on mechanism, low until B-fill trace |
| 2 | **CUDA graph / launch coalescing for steady/finalize/decode sequences** | **0-2 streams if kernel-queue/BW-bound; 2-6 streams if Nsight shows launch gaps >=15% of gate** | Medium-low from current data |
| 3 | **Sync/transfer removal (`enc_len`, argmax/readback cleanup, pinned tiny copies)** | **TTFS variance cleanup; probably 0-1 streams, maybe +2 at N=38, not enough for N=40** | High that it is bounded |

The decisive next move for the #1 lever is a **BATCH-0 kill gate**, not a full scheduler rewrite: capture ready timestamps
and batch keys at N=36-44, then microbench B=2/B=4 batched decode and steady fixtures with T1 equality. GO only if the
trace can form real batches and per-row GPU time drops enough; STOP batching if median B is still <2 or B=1 remains >50%.

## What The Data Shows

The current L40S knee is effectively **N=37** under the pinned rerun: N=37 passes with ttfs p95/p99 91.5/153.2 ms and
lag p95 67.8 ms, while N=38 fails the TTFS budget at 223.9/467.7 ms even though lag p95 is still 388.9 ms
(`runtime/artifacts/l40s_w3_logs/knee_pin.log:223`, `runtime/artifacts/l40s_w3_logs/knee_pin.log:317`,
`runtime/artifacts/l40s_w3_logs/knee_pin.log:421`). N=39 and N=40 are not near misses: N=39 has ttfs p95 341.2 ms
and lag p95 915.7 ms (`runtime/artifacts/l40s_w3_logs/knee_pin.log:411`); N=40 has ttfs p95 361.3 ms and lag p95
1337.3 ms (`runtime/artifacts/l40s_w3_logs/w3_run13.log:411-412`).

The SLO tail tracks `decode_wall`, not explicit sync or transfer volume. At N=40, `finalize_total` p95 is 361.1 ms,
`decode_wall` p95 is 308.6 ms, `enc_len_sync` p95 is 22.1 ms, and `decode_item_wait` p95 is 17.1 ms
(`runtime/artifacts/l40s_w3_logs/w3_run13.log:411-412`). At N=44, `decode_wall` p95 is 455.4 ms while
`decode_item_wait` p95 is still only 16.9 ms (`runtime/artifacts/l40s_w3_logs/w3_run13.log:505-506`). The paired plan
review made the same correction: sync removal is bounded at about 17 ms for `decode_item_wait`; the wall explosion is
joint/predict GEMM queueing under cross-stream contention (`reviews/goforward-paired-verdict.md:11-13`).

The runtime code matches that reading. Steady chunks run a B=1 AOTI encoder, immediately compute encoder length, then run
the greedy decode loop (`runtime/cpp/density_main.cpp:840-858`, `runtime/cpp/density_main.cpp:982-1001`). The decode loop
is per encoder frame and per emitted symbol: `joint.forward`, flatten, `argmax`, optional `predict.forward`
(`runtime/cpp/density_main.cpp:951-980`). Finalize repeats the same structure after the finalize AOTI bucket
(`runtime/cpp/density_main.cpp:1408-1435`).

## 1. Launch-Bound Or GEMM-Bound?

**Both, but batching should be tested first because it attacks both effects.**

The launch evidence is strong but not sufficient by itself. Nsys says the two launch APIs account for 4.416 s +
3.554 s = **7.97 s of aggregate host launch API time** in the capture, with 2.16M launch calls
(`runtime/artifacts/l40s_w3_logs/profiling/nsys_stats.txt:204-208`). `cudaStreamSynchronize` is only 0.7% of API time
and 10,190 calls (`runtime/artifacts/l40s_w3_logs/profiling/nsys_stats.txt:210-212`), so "remove stream syncs" is not
the main launch-side lever. GPU memcpy operations are also small in absolute GPU time, about 50.5 ms total
(`runtime/artifacts/l40s_w3_logs/profiling/nsys_stats.txt:234-243`).

The GEMM evidence is also strong. The dominant kernel is `ampere_sgemm_64x32_sliced1x4_tn`: 334,442 instances, 63.0% of
GPU kernel time, 24.3 us average (`runtime/artifacts/l40s_w3_logs/profiling/nsys_stats.txt:4-12`). NCU shows the sampled
SGEMM is not compute-saturated; it is either DRAM-heavy or tiny-grid limited, with low occupancy and fewer blocks than SMs
in some samples (`runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm.csv:96-103`,
`runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm.csv:143-165`).

CUDA graphs can cut launch count and launch gaps, but they do not change the arithmetic intensity of the 64x32/32x32
SGEMMs. Cross-stream batching can reduce launch count **and** make each GEMM a better work unit. That makes batching the
first lever to kill-gate. Graphs become first only if the next Nsight trace shows the GPU is waiting in launch gaps while
SM and DRAM utilization are below saturation.

## 2. Batching Headroom

### Encoder vs decode attribution

I would not claim the 334K `ampere_sgemm_64x32` instances are proven to be decode GEMMs. The data suggests the opposite
more than it proves the prompt hypothesis.

The count pattern is encoder-like. In the same Nsys kernel table, `conv_depthwise2d_forward` appears 47,759 times
(`runtime/artifacts/l40s_w3_logs/profiling/nsys_stats.txt:13-14`). The dominant 64x32 SGEMM count is 334,442, almost
exactly 7.0x that count (`runtime/artifacts/l40s_w3_logs/profiling/nsys_stats.txt:8`). Decode has no depthwise-conv
kernel in its loop; it only calls joint/predict and argmax (`runtime/cpp/density_main.cpp:951-980`). That ratio is a
strong clue that most of the 334K 64x32 SGEMM instances are from the conformer encoder AOTI stack, not from decode alone.

The NCU file also cannot settle source attribution. The README explicitly warns that NCU serializes and replays kernels
and should be run single-stream (`runtime/run_l40s_density.README.md:74-76`). The `ncu_sgemm_run.log` shows the SGEMM
profile attached during the N=1 serial-oracle build, before the measured N=1 gate, and profiled the first 12 matching
SGEMM launches (`runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm_run.log:1-5`,
`runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm_run.log:26-37`). So NCU proves the sampled kernels are low-occupancy
and often memory-heavy; it does **not** map the N=38 334K instances to steady encoder, finalize encoder, or decode.

What is still true: the SLO binding is `decode_wall`, and decode is where B=1 hurts most. At N=38, `decode_wall` p95 is
166.3 ms while `decode_item_wait` p95 is 11.1 ms (`runtime/artifacts/l40s_w3_logs/knee_pin.log:317-319`). At N=40,
`decode_wall` p95 is 308.6 ms and `decode_item_wait` p95 is 17.1 ms (`runtime/artifacts/l40s_w3_logs/w3_run13.log:411-414`).
That means decode is not merely waiting on its own scalar readback; it is waiting behind cross-stream GPU work and launch
queues. Batching decode/joint across streams directly attacks that queue.

### Expected B=N gain

The best case is not "B=N means N times faster." The current traffic may not fill B, and the encoder already has time
extent inside each chunk.

For **decode joint/predict**, B=1 is the most favorable target: a batched active set can amortize weight loads and
collapse many small launches and host decisions into one step. If active-set B is 4+, I would expect per-active-row decode
GEMM time to drop materially, plausibly 35-50% for the joint/predict subpath. This is the "near-linear-ish" part of the
argument, but only for the decode subpath and only if active masks are nontrivial.

For the **steady/finalize encoder**, gain should be sublinear. Each per-stream chunk already has T extent; batching adds
cross-stream B and may raise grid size/occupancy, but it does not erase all per-row work. Given the NCU shape
(128 blocks vs 142 SMs, 16.7% occupancy for a 64x32 sample), B=2/B=4 could still help, but the realistic bar should be
per-row time <=0.85x at B=2 and <=0.75x at B=4, not a 2-4x claim.

The prior batching sim is a serious constraint: with an 8 ms deployed window, synthetic independent/bursty arrivals showed
mean B about 1.5-2.1 and B=1 at 36-63%; the doc explicitly says the old 3-5x batching claim is effectively dead without
adding latency (`spikes/0.5-batching-sim/FINDINGS.md:8-20`, `spikes/0.5-batching-sim/FINDINGS.md:30-35`).

My quantified knee estimate:

| Batching outcome | Requirement | Expected knee |
|---|---|---:|
| Weak but real B | median B 1.5-2.0, B=1 still 35-60%, B=2 per-row <=0.85x | **37 -> 40-42** |
| Good production B | median B >=2.5, p95 B >=4, B=1 <=35%, B=4 per-row <=0.75x | **37 -> 44-48** |
| Engineered high B | B>=4 most of the hot path, decode per-row <=0.6x, steady <=0.75x, no added SLO wait | **upper bound low/mid 50s**, not established by current data |

The clean Amdahl bound explains the scale: if the dominant 63% SGEMM time were sped up 1.5x, total service speedup is
about 1/(0.37 + 0.63/1.5) = 1.27x, or N=37 -> 47. If it sped up 2x, the upper bound is 1.46x, or N=37 -> 54. Those are
upper bounds before B-fill, scheduler wait, non-SGEMM kernels, and correctness constraints.

## 3. CUDA Graph Headroom

Graphs have real but bounded upside.

The launch count is large: 2.16M launch API calls, about 35.98K launches/s if the curated 60 s window is used
(`reviews/l40s-profiling-data.md:7-10`), with about 3.69 us average host API time per launch from the raw Nsys table
(`runtime/artifacts/l40s_w3_logs/profiling/nsys_stats.txt:204-208`). The aggregate launch API time is about 7.97 s. If a
steady/finalize/decode graph replay removes 80-95% of those calls on the hot path, it could remove 6.4-7.6 s of aggregate
host launch API time plus some device-side launch gaps.

But two caveats matter:

1. Nsys API time is not the same as critical-path wall time. The calls come from many host threads, and the summary does
   not show whether the GPU timeline has launch gaps. The README says the missing GPU-idle/launch-gap answer requires a
   CUDA GPU trace or GUI inspection, not just the summary (`runtime/run_l40s_density.README.md:60-72`).
2. Graphs do not raise occupancy or weight reuse. A graph-replayed `ampere_sgemm_64x32` with 128 blocks, 0.45 waves/SM,
   and 16.7% achieved occupancy is still that same GEMM (`runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm.csv:143-165`).

So graph headroom is conditional:

| Nsight result | Graph expectation |
|---|---:|
| Launch gaps >=15% of measured gate, SM/DRAM below 85-90% in active windows | **N=37 -> 40-43**, maybe 44 if decode graphing also works |
| GPU queue is continuously occupied by tiny SGEMMs, with DRAM/SM active pressure high | **0-2 streams**, mostly host CPU/variance cleanup |
| Graph memory or graph-pool ownership forces fewer concurrent contexts | Negative or no density gain |

The Python finalize-graph precedent is relevant only to launch/gap removal. It does not imply the BW-bound SGEMMs become
faster unless graphing also enables fusion, batching, or better reuse. Current native Nsys says launch reduction is worth
measuring; NCU says launch reduction alone is not enough to fix the GPU work shape.

## 4. Occupancy Paradox

The "73% util but 15% occupancy" result is not contradictory. NVML utilization mostly answers "was any work running on the
GPU during the sampling interval?" Occupancy answers "how much warp capacity did this kernel keep resident while it ran?"
N=36 had mean GPU util 73.0% but p50/p95 util 91/97%, and N=40 failed at 75.4% mean util with p50/p95 96/98%
(`runtime/artifacts/l40s_w3_logs/w3_run13.log:317-318`, `runtime/artifacts/l40s_w3_logs/w3_run13.log:411-412`).
The sampled SGEMM then shows why "busy" is not "full": 128 blocks on 142 SMs, 0.45 waves/SM, achieved occupancy 16.68%
(`runtime/artifacts/l40s_w3_logs/profiling/ncu_sgemm.csv:143-165`).

If batching lifted the hot GEMMs from ~16% occupancy toward 60-80%, the hardware has real headroom. But the streams/box
headroom is governed by per-row service time, not occupancy alone. A plausible production gain is:

| Assumption | Effective service gain | Streams estimate |
|---|---:|---:|
| Batching improves only decode tail and a small part of encoder | 1.08-1.15x | **40-42** |
| Batching improves most SGEMM rows 20-25% | 1.18-1.30x | **44-48** |
| Batching doubles dominant SGEMM throughput and B-fill is high | 1.4-1.5x upper bound | **52-55**, not forecast |

This is why mean-util extrapolation is unsafe. The data supports real GPU-work-shape headroom, but not a 2x box-density
claim.

## 5. Ranked Lever Verdict

### 1. Cross-stream batching: first to kill-gate

This is the only lever that can plausibly move the knee from 37 into the mid/high 40s because it reduces per-row GPU work
and launch count together. It should be scoped as **batched greedy decode first**, with steady encoder batching measured
in parallel. The SLO symptom is decode-wall explosion; the largest structural waste is many independent B=1 joint/predict
and encoder work units.

Expected gain: **1.08-1.30x**, or **N=40-48**, depending on B-fill and microbench. Treat anything beyond 48 as speculative
until real traces show high B without SLO wait.

### 2. CUDA graph / launch coalescing: second, conditional on timeline gaps

Graphs can remove millions of launches and likely reduce host scheduling variance, but they cannot fix 16% occupancy or
DRAM-heavy B=1 GEMMs. They should run as a targeted kill-gate after, or alongside, the BATCH-0 trace:
if launch gaps are large, graphing may be the cheaper near-term patch; if not, it is a hygiene optimization.

Expected gain: **0-6 streams**. The current data is enough to justify measuring it, not enough to rank it above batching.

### 3. Sync/transfer removal: do it for hygiene, not density credit

At N=40, even deleting the p95 `enc_len_sync` and `decode_item_wait` terms would remove only about 39 ms from a 361 ms
`finalize_total` p95 and a 309 ms `decode_wall` p95 (`runtime/artifacts/l40s_w3_logs/w3_run13.log:411-414`). At N=38,
the same arithmetic removes about 33 ms from a 224 ms finalize p95 (`runtime/artifacts/l40s_w3_logs/knee_pin.log:317-319`).
That may turn some borderline N=38 TTFS samples into passes, but it does not explain N=39/N=40 collapse.

Expected gain: **0-1 streams**, maybe **+2** if queue nonlinearities are favorable. It is still worth doing because it
reduces variance and simplifies batching/graphing, but it is not the higher-concurrency lever.

## Cheapest Decisive Next Build/Measure

Run **BATCH-0** before building a production scheduler.

1. **Opportunity trace, no model changes.** Instrument or replay the density harness at N=36-44 to emit ready timestamps
   and batch keys for steady encoder, finalize encoder, and decode active-set steps. Simulate 8 ms and 12 ms max wait.
   GO only if median B >=2.5, p95 B >=4, B=1 <=35%, and added wait p95 <=8 ms. STOP batching if median B <2 or B=1 >50%.
2. **Batched fixture microbench.** Export/compile B=2 and B=4 fixtures for the current steady geometry and for the
   joint/predict decode active-set path. Pack independent caches, run batched-vs-alone shadow comparisons, and record
   CUDA event time per row plus launch counts. GO only if B=4 steady per-row <=0.75x B=1, B=2 <=0.85x, decode per-active-row
   materially improves, and there are 0 token/cache/event mismatches. STOP on any T1 drift or <15% per-row gain.
3. **Attribution trace in parallel.** Re-run Nsys with a GPU trace/NVTX ranges for steady AOTI, finalize AOTI, joint, and
   predict. The current summary cannot map the 334K SGEMMs to source modules; the next trace should.

If BATCH-0 passes, build the smallest batched greedy decode path first and measure N=38/40/44. If BATCH-0 fails, route to
graph/coalescing only if the attribution trace shows launch gaps; otherwise the single-process L40S knee is probably near
pinned without reducing model work.

## What The Data Cannot Determine

- It cannot prove the 334K `ampere_sgemm_64x32` instances are decode. Count ratios point toward encoder dominance, but
  source attribution requires NVTX/CUPTI or a kernel trace with module ranges.
- It cannot quantify graph benefit. The API summary shows launch volume, not device launch gaps or critical-path wall time.
- It cannot prove batching benefit. There is no native B>1 AOTI/decode microbench and no production B-fill trace for this
  exact runtime.
- It cannot extrapolate from NVML mean util to streams/box. N=38 and N=40 fail with mean util in the 70s because active
  windows can be busy while individual kernels still underfill SMs.
- It cannot rank correctness risk. Batched greedy decode and graph replay both need T1 equality before any throughput
  number is meaningful.
