# Roofline + optimization — COMBINED verdict (Codex `roofline-codex.md` + Opus subagent `roofline-subagent.md`)

Two independent investigations (Codex code-archaeology + Opus 4.7 first-principles) converged on every material
conclusion. Where they agree it's high-confidence; divergences + unique findings noted.

## Q1 — Roofline: the encoder is MEMORY(weight-stream)-BOUND, and we're ~3.4× above it (not a compute gap)
The model is ~0.6B params, **fp32** (no half/bf16 conversion; TF32 disabled under batching), **128 mel bins** (both
agents independently corrected the brief's "80"). The finalize encoder must **stream ~2.44 GB of fp32 weights** but
does little math per pass → **arithmetic intensity well below every platform's ridge** (Codex: ~23 FLOP/B for the
full finalize slice; subagent: ~5 FLOP/B encoder-weights-only; ridge points 58-113). **Memory(weight)-bound on all
four platforms** — the binding floor is `2.44 GB / mem-BW`, and compute-floor ≪ memory-floor everywhere.

| platform | mem BW | finalize roofline floor (reconciled) | measured | distance |
|---|---:|---:|---:|---:|
| RTX 5090 | 1792 GB/s | ~1.4 ms | — | ~9× (model slice) / ~15× (full wall) |
| **L40S** (measured) | 864 GB/s | **~2.9-3.0 ms** | 9.7 ms enc / 13.4 ms model / 22 ms wall | **~3.4× enc, ~4.5× model, ~7.3× wall** |
| L4 | 300 GB/s | ~8.2-8.7 ms | — | ~1.5× (model) / ~2.5× (wall) |
| DGX Spark GB10 | **273 GB/s** | ~9.0-9.5 ms | — | ~1.4× (model) / ~2.3× (wall) |

**Why the 3.4× encoder gap is NOT compute (both agree):** it's a CUDA-graph *replay* (not dispatch), but it's **24
sequential conformer layers of tiny (M≈7 query-frame) kernels that individually can't saturate DRAM bandwidth** → ~29%
of peak BW. **fp16 is 0.79× SLOWER** because at M≈7 the GEMMs are latency/occupancy-bound, so byte-savings are swamped
by per-op cast overhead — **precision only helps once you fuse.**
**DGX Spark call-out (both):** its 273 GB/s LPDDR5x gives a ~9 ms memory floor for this workload → **capacity-rich but
bandwidth-poor; unlikely to beat L40S for TTFT** despite 128 GB.
*(Divergence: FLOP estimate 12 GFLOP (subagent, encoder/7-query-frames) vs 60 GFLOP (Codex, 2·N·T full slice) — ~5×
spread, but MOOT: both are memory-bound since compute-floor ≪ memory-floor on every platform.)*

## The honest ceiling (both agree — sets expectations)
TTFT p50 = 246 ms ≈ **~200 ms fixed VAD trailing-silence window + ~23 ms WAN + ~22 ms server finalize.** Even a
*perfect* engine (finalize → roofline ~3 ms) moves **p50 by only ~12-19 ms.** The from-scratch prize is the **p95/p99
tail at load + streams/box** (both scheduler-bound), NOT the encoder math. The largest remaining p50 lever — shrinking
the VAD window — lives **outside this inference stack.**

## The REAL limit (both agree): steady keep-up is scheduler/GIL-bound, not GPU
At the maxconn=12/proc operating point the GPU is **46-65% idle** yet `vad_stop_recv_to_process` (end-of-speech
received → finalize *starts*) blows **8 ms → ~930 ms** → client TTFB p95 → 1230 ms. Pure head-of-line blocking on the
**single asyncio scheduler thread + GIL**. Reliable in-budget capacity ≈ **20/box, not 48**.

## Q2 — In-code levers (ranked; merged, file:line)
1. **Admission control / backpressure** from the observed scheduler backlog (`server.py:4163` no capacity gate,
   `:4326`/`:4341` deep per-session queues, backlog metric `:2790`). Kills the overload cliff; makes ~20/box reliable.
   *Highest production impact, byte-exact, low risk.*
2. **Priority finalize-lane scheduling + move GPU dispatch/eager-decode off the event-loop thread** (finalize waits
   for the pinned lane `:6755`; lanes `:3150/3155/3175`; steady reserves lanes `:5077/5140`). Directly attacks the
   p95/p99 spread + the keep-up tail. *Biggest tail + parallelism win.*
3. **Compress host syncs + per-batch CPU launch** (steady syncs `:8223/8243/8329/8399`; finalize `:7300/7341/7430/7513`).
   The direct code-level attack on the launch-dispatch ceiling → higher per-proc keep-up. *Byte-exact-gated.*
4. **One-shot finalize preprocessor** (`:6927/6941`, batched fallback `:7121/7173`): ~1-2 ms p50, byte-exact-gated.
5. **Fill B for steady decode** (86% are B=1 → weights never amortized) without adding user wait (`:669/670/4918/4808`).
6. **[Unique — subagent] Single padded-T_max finalize-graph bucket instead of 16 per-T graphs** — this per-T bucketing
   is what caps L40S at K=3 (memory); collapsing it could cut the finalize graph-pool **~10-16×** and **recover K=4
   (~64/box)** — possibly the **cheapest density win**, and it directly fixes the K=4 OOM we hit.
**Non-levers (both):** RNNT decode-graph (NO-GO), fp16 (0.79× slower), clone/scatter (<1 ms).

## Q3 — From-scratch (both agree on direction + magnitude)
A persistent **C++/CUDA (or TensorRT) runtime** that owns admission, batching, lane priority, graph replay, decode
state, and output scatter **without bouncing through Python per chunk** could move **server finalize 22 → ~6-10 ms**
(fused encoder, fp16→fp8 once fused, fixed-trip/graphed decode, CUDA-event dependency tracking, no per-chunk
sync/scatter). **Real continuous batching that fills B → steady throughput 3-5×.** The durable scaling fix is
**breaking the GIL/single-thread-launch ceiling** (per-lane dispatcher threads → no-GIL Python 3.13t or a C++/Rust
serving core): removes the MPS compute tax (+15-40%) and the K×11 GB graph-pool duplication (today's L40S K=3 *memory*
cap) → **reliable capacity ~20 → ~40-48/box.** BUT p50 still bounded by VAD+WAN → **the from-scratch prize is the tail
+ density, not p50.**

## Recommended order (both converge)
**L1 admission + L2 priority-finalize-lane / off-event-loop dispatch** (no model risk, biggest tail/knee win) →
**single padded-T finalize bucket** (cheap density, recovers K=4) → host-sync compression → fused fp16 encoder →
fixed-trip/fused decode → fp8 + B-fill → **no-GIL/C++ serving core as the endgame** for parallelism.

## Pending
The empirical **L4 K=2 keep-up sweep** (in flight) will add the L4 in-budget-capacity datapoint to complement this
analysis (the roofline already predicts L4 is bandwidth-floor-limited at ~8.2 ms, ~2.5× the L40S floor).
