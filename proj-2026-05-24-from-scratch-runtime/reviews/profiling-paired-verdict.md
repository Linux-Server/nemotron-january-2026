# Paired verdict — L40S profiling → higher-concurrency routing

**Date:** 2026-05-27 | **Folds:** `opus-l40s-profiling-analysis.md` + `codex-l40s-profiling-analysis.md` (two independent analyses of the same nsys+ncu data). Both **independently inverted** the framing in `l40s-profiling-data.md`.

## Consensus (both, and I verified the load-bearing facts)
1. **The dominant 334K SGEMM (63% GPU) is the conformer ENCODER, not the B=1 decode.** VERIFIED: `ampere_sgemm_64x32/conv_depthwise2d = 334442/47759 = 7.003`, `32x32/conv = 95556/47759 = 2.001` (exact integers → 7+2 FP32 GEMMs per encoder layer; decode has no conv). 24μs avg = weight-streaming (16.8MB FF weight ÷ 623 GB/s = 26.9μs), not a 1-3μs matvec. The actual decode matvecs are the `gemv*` kernels = **only ~3.7% of GPU time** — they *queue behind* the encoder GEMMs (FACT-2: decode_item_wait flat ~17ms while decode_wall explodes).
2. **The binding is DRAM-BW-bound weight-streaming in the STEADY encoder** (88% GEMM, 72% DRAM = 623/864 GB/s). The knee = aggregate GPU **bytes/second**, not SM occupancy.
3. **The occupancy paradox is a BW-wall trap.** 15% occupancy *looks* like 6× idle-SM headroom — FALSE: ~1.4 concurrent encoder GEMMs already saturate HBM. Lifting occupancy stalls harder on DRAM. **The only way up is fewer bytes/stream-second = amortize the weight load = batching.** (The checkpoint's util-bound ~44-48 was coincidentally-similar but mechanistically wrong.)
4. **Ranked levers:** **#1 cross-stream batching** (the only lever that lowers the BW byte floor) ≫ #2 CUDA-graphs (cut launches, NOT the byte floor → tail/host hygiene, +0-4 streams) ≈ #3 sync-removal (bounded ~17ms, +0-2). Launches are 94% of API time but *parallel across 38 threads* → not the GPU binding (caveat: cross-thread driver launch-serialization is unmeasured — the one place a steady graph could surprise).
5. **Cheapest decisive next move: a model-free batching kill-gate** before any big build. Both proposed it (Codex "BATCH-0", Opus "STEADY-BATCH-0").
6. **Missing measurement:** ncu was **single-stream (N=1)** → occupancy/DRAM are intrinsic-kernel-character, the *contention* at the knee is inferred. A **multi-stream ncu/CUPTI at N=36/40** is the gap (= the checkpoint's 1c-B).

## The one productive disagreement — the batching MAGNITUDE (hinges on STEADY B-fill)
- **Codex (conservative): 1.08-1.30× → N=40-48**, keeping the prior sim's B-fill skepticism (mean B≈1.5-2, "batching dead").
- **Opus (bullish): 1.5-2.5× → N=47-64**, with a sharp correction: **the prior "batching dead" sim modeled FINALIZE (rare/bursty); STEADY chunks arrive at a continuous 160ms cadence** → at N=37 an 8ms window gives B≈2.8, 12ms → B≈3.7 (and the knee is stagger-robust = well-distributed arrivals). 8-12ms wait fits keep-up (lag p50 −55ms slack) and doesn't touch the finalize-driven ttfs budget.
- **Resolution: the difference is ENTIRELY the steady B-fill, and it's cheaply testable.** The fill-trace (below) settles whether steady B≈2.8-3.7 (Opus → 47-64) or stays ~1.5-2 (Codex → 40-48). Amdahl bound either way: 1.27× (SGEMM 1.5×) → ~47; 1.46× (2×) → ~54; only the BW-reuse via batching gets there.

## ROUTING VERDICT
**Build cross-stream batching of the STEADY encoder** (B=2-4, shared weight load) — the #1 lever, the only one that moves the BW-bound ceiling; projected **N=37 → ~44-64** (range = the B-fill uncertainty). CUDA-graphs + sync-removal are **tail/latency hygiene** (0-4 / 0-2 streams), worth doing for p99 but not the knee. Decode-batching folds into the steady-batching build (clears the decode_wall queue).

### Next step — STEADY-BATCH-0 (model-free, ~hours, no g6e build) — DO BEFORE the build
1. **Steady fill-trace** (zero model changes): log per-stream steady-chunk ready-timestamps at N=36/44/56 (stagger-robust mode), replay through 8ms & 12ms windows. **GO** if median steady B≥2.5 ∧ B=1≤35% at N≥44 (settles the Codex↔Opus magnitude). **STOP-batching** if median B<2.
2. **Batched-steady microbench** (one geometry, no T1 yet): compile B=2/B=4 steady AOTI, pack B independent caches, per-row GPU time vs B=1. **GO** if B=4 per-row ≤0.6× ∧ B=2 ≤0.8×; **STOP** if <15% gain (roofline wrong → re-profile multi-stream).
3. **(Parallel) multi-stream ncu/CUPTI at N=36/40** — the missing contention measurement + the NVTX module attribution (confirms encoder-vs-decode on the *real* multi-stream timeline).

Pass 1+2 → build the smallest batched-steady path + measure N=44/48/56. Fail (1) → batching dead → graph+sync for tail only, single-process L40S knee ~pinned near 37-40. Then this feeds the Funding-recheck F1 (a clean ~1.5-2.5× changes the funding calculus vs the current at-bar ~1.5×).
