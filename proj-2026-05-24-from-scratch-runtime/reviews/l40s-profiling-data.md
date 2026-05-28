# L40S native-runtime profiling data (nsys + ncu, 2026-05-27)

Raw: `runtime/artifacts/l40s_w3_logs/profiling/` (nsys_stats.txt, nsys_n38.nsys-rep, ncu_sgemm.csv, ncu_sgemm.ncu-rep).
Profiled the **sm_89 L40S** native density runtime (the gate binary), torch 2.8.0+cu128. Context: SLO-knee N=37; A (host_enc_len)
relieves keep-up to ~40-41; the **ttfs p99 tail (`decode_wall`, 455ms@N=40) is the hard binding**; nvidia-smi util ~73% at the knee.

## nsys — N=38 multi-stream timeline (60s capture window in the measured gate)
**CUDA API (host) — LAUNCH-HEAVY:**
- `cudaLaunchKernel` **52.3%** (1,138,228 calls, avg 3880ns) + `cuLaunchKernel` **42.0%** (1,020,558 calls) = **~94% of API time is kernel launches; ~2.16M launches** over 60s.
- `cudaMemcpyAsync` 2.4%; `cudaStreamSynchronize` **0.7%** (10,190 calls); `cudaEventRecord` 1.2%. → **syncs are negligible** (confirms FACT-2: A helped keep-up but sync-removal is tapped).

**GPU kernels — SGEMM-DOMINATED (~80% GEMM):**
| Time% | kernel | inst | avg ns |
|---|---|---|---|
| **63.0** | `ampere_sgemm_64x32_sliced1x4_tn` | 334,442 | 24,279 |
| 7.0 | `ampere_sgemm_32x32_sliced1x4_tn` | 95,556 | 9,409 |
| 6.2 | cutlass `s1688gemm_64x64_32x6_nn` | 47,759 | 16,835 |
| 4.6 | cublasLt `splitKreduce_kernel` | 434,027 | 1,356 |
| 3.3 | cutlass `s1688gemm_64x64_16x6_nn` | 47,836 | 9,014 |
| 1.2 | `conv_depthwise2d_forward` (conformer) | 47,759 | 3,103 |
| ~1.6 | triton `layer_norm` + `silu` | | |

**GPU MemOps (small absolute, ~50ms total):** H2D 46.9% / D2D 35.0% / D2H 16.6%.

## ncu — `ampere_sgemm` (the 63% kernel), single-stream roofline
- **DRAM/Memory throughput 71-72%**, **Compute (SM) throughput 34-39%** → **MEMORY-BANDWIDTH-BOUND**.
- **Achieved occupancy 15-17%** (LOW) → small, matvec-like (B=1) GEMMs; SMs far from saturated.
- Duration ~34-36μs.

## The picture
- **FP32 GEMMs are the work** (`ampere_sgemm` = CUDA-core FP32, not tensor-core; model is fp32). ~80% GEMM.
- **Launch-bound** (94% API time, 2.16M launches) AND **occupancy-starved** (15% on the dominant kernel) — the GPU is mostly *idle within its busy windows*. nvidia-smi 73% "util" hides this.
- The dominant GEMM is **mem-BW-bound at B=1** → batching B=1→B=N amortizes the weight load (intensity ↑, off the BW wall) and raises occupancy.

## Questions for the paired analysis (rank by realistic L40S concurrency headroom)
1. **Launch-bound vs compute/BW-bound:** 94% launches + 15% occupancy + mem-BW-bound GEMM. Is the binding the launch overhead (→ CUDA-graphs) or the per-GEMM BW/occupancy (→ batching)? Both — and which first?
2. **Batching headroom:** distinguish the *encoder* GEMMs (T-batched per chunk → less to gain) vs the *decode* joint/predict GEMMs (B=1 per-token, matvec-like → big to gain). Which dominates the 334K-instance `ampere_sgemm`? Expected gain from cross-stream B=N.
3. **CUDA-graph headroom:** 2.16M launches at ~3.7μs API each. Collapsing (Python finalize-graph precedent) → how much launch overhead + gap removed? Does it raise occupancy (no) or just cut launches (yes)?
4. **Occupancy paradox:** 73% util vs 15% occupancy — what's the true streams/box headroom if occupancy is lifted?
5. **Ranked verdict + the cheapest decisive next build/measure** for higher concurrency.
