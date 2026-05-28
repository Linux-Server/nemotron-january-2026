# STEADY-BATCH-0 — kill-gate RESULT (both conjuncts PASS, 2026-05-27)

The model-free kill-gate that decides whether to build the cross-stream batched steady encoder (the #1 BW-floor
lever from `profiling-paired-verdict.md`). **Two GO conditions, both required. Both PASSED.**

## Conjunct 1 — OPPORTUNITY (does steady arrival fill batches?)
Fill-trace: log per-stream steady-chunk ready timestamps from the L40S density runtime
(`NEMOTRON_DENSITY_FILL_TRACE=1` → `FILL_TRACE worker=… t_ready_ns=… kind=steady` lines), replay through a greedy
windowed batcher (`runtime/fill_replay.py`, `simulate_batches(times_ns, window_ms, bmax)`).

| N streams | mean B @ 8ms | mean B @ 12ms |
|---|---|---|
| 36 | 2.71 | 3.47 |
| 44 | 3.63 | 4.30 |
| 56 | 3.75 | 4.40 |

**GO** (gate: median steady B≥2.5, B=1≤35% at N≥44). The steady chunks arrive at a continuous **160ms cadence** per
stream, so batches fill structurally — refuting the prior `spikes/0.5-batching-sim` "B≈1.5-2, batching dead" result,
which modeled *finalize* bursts, not the steady cadence. The 8-12ms window fits keep-up slack (lag p50 had ~55ms) and
does not touch the finalize-driven ttfs budget.

## Conjunct 2 — SPEEDUP (does batching drop per-row GPU time?)
Microbench `runtime/cpp/steady_batch_bench.cpp`: load B∈{1,2,4} steady AOTI, pack B independent caches, measure
per-row GPU time vs B=1 (warmup 10, iters 100). **5090 / sm_120** (the ratio is a roofline property → transfers to
sm_89; the L40S confirm of the absolute is a formality the user waived).

```
CORRECTNESS B=2 vs B=1: ok=1  enc_out_max=6.050e-06  cache_ch_max=8.153e-05  cache_t_max=6.309e-03  (within atol 5e-2)
CORRECTNESS B=4 vs B=1: ok=1  enc_out_max=6.169e-06  cache_ch_max=8.048e-05  cache_t_max=5.486e-03
TIMING B=1  per_row_ms p50=5.1096
TIMING B=2  total p50=6.3502  per_row p50=3.1751   ratio_vs_B1=0.621   (threshold ≤0.80)  PASS
TIMING B=4  total p50=7.6901  per_row p50=1.9225   ratio_vs_B1=0.376   (threshold ≤0.60)  PASS
=== RESULT correctness=PASS perf_signal=GO B2_ratio=0.621 B4_ratio=0.376 ===
```

**GO** (gate: B=4 per-row ≤0.6× ∧ B=2 ≤0.8×). Total time grows **sub-linearly** (5.11 → 6.35 → 7.69ms; the marginal
row is ~1.3ms vs the 5.11ms first) = the weight-load is paid once and reused across the batch, exactly the
DRAM-BW-amortization the roofline predicted. Correctness is byte-exact per-row (the cache_t ~5e-3 is fp32
reduction-order, within tolerance — not an approximation).

## Verdict
Both conjuncts PASS → **build the batched-steady encoder** (PHASE2-PLAN.md Steps B1-B3). Projected knee **37 →
~47-64** (Amdahl-bounded by the 88% steady GEMM share × the measured fill); funding multiplier **1.8× → ~2-2.5×**
(provisionally clears F1; Step B3's L40S sweep is the binding re-check). Artifacts (gitignored, regen via
`runtime/export_steady_batched.py` + per-B `--compile-only`): `runtime/steady_b_artifacts/enc_steady_aoti_b{1,2,4}.pt2`.
