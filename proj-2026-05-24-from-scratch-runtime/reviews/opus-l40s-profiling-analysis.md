# L40S native-runtime profiling — independent Opus analysis (2026-05-27)

Re-derived from raw `nsys_stats.txt`/`ncu_sgemm.csv`/run-logs + the model config (24-layer conformer, d_model=1024,
RNN-T pred/joint=640, vocab=1025). Adversarial-paired with `codex-l40s-profiling-analysis.md`. **One premise in
`l40s-profiling-data.md` is wrong and it flips the verdict.**

## 0. The load-bearing correction
`l40s-profiling-data.md` framed the 334K-instance `ampere_sgemm` (63%) as possibly the **B=1 decode** GEMMs. **It is
the T-batched ENCODER, ~98% the per-chunk STEADY encoder.** Proof (instance arithmetic):
- `ampere_sgemm_64x32 / conv_depthwise2d = 334442/47759 = 7.003` (exact); `ampere_sgemm_32x32/conv = 95556/47759 = 2.001`.
- `conv_depthwise` exists ONLY in the conformer encoder (24 layers) → the dominant SGEMMs are **7+2 FP32 GEMMs/encoder-layer**, not decode.
- avg `ampere_sgemm` = **24μs** = weight-streaming (1024×4096 FF weight = 16.8MB ÷ 623 GB/s = 26.9μs); a true B=1 matvec is 1-3μs. The actual decode matvecs ARE the 1-3μs `gemvSN_*` kernels = only **~3.7% of GPU time**.
- ⇒ **right batching target = the STEADY ENCODER, not the decode loop.** Decode matvecs *queue behind* the BW-saturated encoder GEMMs (consistent with FACT-2: decode_item_wait flat ~17ms while decode_wall explodes).

## 1. Launch vs BW/compute — which binds
- **Per-stream (N=1):** launch/dispatch+sync-bound (~6.5ms GPU compute/forward but ~90% host-gap; ~924 launches/forward).
- **At the knee (N≈37):** **GPU-compute-time oversubscribed (4.1%/stream × 37 ≈ 152% of the 160ms window), and that compute is DRAM-BW-bound** (88% GEMM, 623 GB/s ≈ 72% of the L40S 864 GB/s).
- **BW/compute binds first, not launches.** Launch API time (7.97s) is *parallel across 38 threads* (~0.3% of a core); CUDA-graphs can't lower the byte floor → can't raise the compute-saturation ceiling. **CAN'T determine:** cross-thread driver launch-serialization (nsys is per-thread) — the one place a steady graph could surprise.

## 2. Batching headroom (quantified)
Steady chunk has **M≈2-3 encoder frames** → each GEMM ≈ matvec against a full weight → time ≈ weight_bytes/BW, ~independent of M → **textbook cross-stream batching regime** (load weight once for B streams). Roofline: steady GEMM at SM 34% / ~21% of FP32 peak → **~3× compute headroom** before SM-bound, so B≈3-4 amortizes the weight on the same kernel time.

| Assumption | Knee mult | Knee (from 37) |
|---|---|---|
| amp64 cut 1.5× (B≈2, conservative) | 1.27× | ~47 |
| amp64 cut 2.0× | 1.46× | ~54 |
| all-enc-GEMM cut 2.0× (B≈3-4) | 1.73× | ~64 |

**Crucial B-fill correction:** the prior "B≈1.5-2, batching dead" sim modeled **finalize** (rare/bursty). **STEADY chunks arrive at a continuous 160ms cadence** → at N=37 an 8ms window → B≈2.8, 12ms → B≈3.7 (higher at N=48-64); knee-collapse is stagger-robust (arrivals well-distributed) → steady batching **fills structurally**. 8-12ms wait fits keep-up (lag p50 −55ms slack) and doesn't touch the finalize-driven ttfs budget. **Realistic banked gain ~1.5-2.5× → knee ~47-64, B-fill-gated.**

## 3. CUDA-graphs
Collapses ~924 launches/forward + host gaps → cuts host launch rate + shrinks per-stream wall span (tail/p99 + host-thread relief). **Does NOT raise occupancy or lower the DRAM byte floor** → since the knee = aggregate GPU bytes, a graph can't turn 152%-of-timeline into <100%. **Steady density gain ~0-4 streams.** (Caveat: if cross-thread driver launch-serialization is real — unmeasured — a graph could relieve it. Second-order maybe.)

## 4. Occupancy paradox — NOT 6× headroom (the trap)
15% occupancy is single-kernel-in-isolation (grid 128-256 blocks on 142 SMs, 0.45-0.60 waves/SM). The trap: "85% SMs idle → 6× headroom" is **FALSE** — a single steady GEMM already pulls **623 GB/s = 72% of HBM**, so **~1.4 concurrent encoder GEMMs saturate the bus**. Overlapping streams hit the **DRAM wall, not idle SMs**; lifting occupancy stalls harder on DRAM. The only way up is **fewer bytes/stream-second = batch the weight load** → §2 (~47-64), not the 1/0.15 fantasy. (The checkpoint's util-bound ~44-48 is a coincidentally-similar but mechanistically-wrong number.)

## 5. Ranked verdict
| Rank | Lever | Attacks | Est. knee | Confidence |
|---|---|---|---|---|
| **1** | **Batch the STEADY encoder** (B=2-4, shared weight load) | the 88%/BW-bound weight-stream floor — the real binding | **37 → ~47-64** | Med-High (mechanism proven; magnitude B-fill-gated) |
| 2 | Batched/queue-cleared decode (rides on #1) | the decode_wall queue | folds into #1 | Med |
| 3 | Steady+decode CUDA-graph | launches + host gaps + tail | +0-4 (hygiene) | Med |
| 4 | scalar-sync removal | ~17ms host-sync | +0-2 | High it's bounded |
| last | enc_first K-pool | first-chunk churn (harness artifact) | ~0 prod | High |

## Cheapest decisive kill-gate — STEADY-BATCH-0 (model-free, ~hours, no g6e build)
1. **Steady fill-trace (zero model changes):** log per-stream steady-chunk ready-timestamps at N=36/44/56 (stagger-robust), replay through 8ms/12ms windows. **GO** if median steady B≥2.5, B=1≤35% at N≥44; **STOP** if median B<2. (Model predicts B≈2.8@8ms / 3.7@12ms — settled in an afternoon.)
2. **Batched-steady microbench (one geometry, no T1 yet):** compile B=2/B=4 steady AOTI, pack B independent caches, measure per-row GPU time vs B=1. **GO** if B=4 per-row ≤0.6×, B=2 ≤0.8×; **STOP** if <15% gain (roofline wrong → re-profile under contention).
Pass both → build batching (knee ~47-64). Fail (1) → batching dead, ship graph+sync for tail. Fail (2) pass (1) → re-profile multi-stream (N=38 ncu).

## Key uncertainties
- **ncu was single-stream (N=1)** → occupancy/DRAM/SM are intrinsic-kernel-character, NOT measured contention at the knee. Need a **multi-stream ncu/CUPTI at N=36/40** (= the checkpoint's 1c-B).
- **Cross-thread driver launch-serialization unmeasured** (nsys is per-thread) — the sole scenario a steady graph beats §3.
- **B-fill magnitude** is the make-or-break for the #1 lever's *size*; STEADY-BATCH-0 settles it.
- Knee ∈[36,39] unpinned; "47-64" are roofline projections, not measurements.
