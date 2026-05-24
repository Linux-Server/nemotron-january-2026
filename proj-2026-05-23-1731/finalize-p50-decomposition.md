# Finalize p50/spread decomposition (conc-10 L40S lanes=2, the 246/279 baseline)

Source: `ec2-bench/lanes_ab_1624/prod_l40s_c10_lanes2.records` (2134 finalize_profile_records, finalize-graph ON,
all B=1, encoder finalize cudagraph replay 2134/2134). No new run — mined existing data.

| component | p50 | p95 | p99 | p95−p50 | note |
|---|---|---|---|---|---|
| **finalize_wall_ms (total)** | **22.1** | **41.3** | **54.4** | **19.1** | the server-finalize span |
| lock_wait_ms | 0.36 | 21.1 | 35.7 | **20.7** | **≈ the ENTIRE spread** — lane/inference-lock contention |
| model_wall_ms | 13.4 | 17.6 | 20.0 | 4.2 | encoder + decode |
|  ├ encoder_wall_ms | 9.7 | 12.0 | 12.9 | 2.4 | **already CUDA-graphed (replay)** — little headroom |
|  └ decode_wall_ms | 3.6 | 6.4 | 8.4 | 2.8 | eager; **graphing = NO-GO (step 2)** |
| preproc_wall_ms | 2.4 | 5.5 | 6.8 | 3.1 | 3 invocations → one-shot could cut ~1-2ms |
| fork_clone_ms | 0.44 | 2.1 | 3.0 | 1.7 | **Codex's top pivot candidate — but only ~0.4ms; NOT a lever** |
| final_gather/scatter/clone_hyp/queue/debounce | ≤0.23 each | — | — | — | negligible |

## The two goals map to two very different places

**"Reduce P50" → near the floor.** The 22ms p50 is mostly the **already-graphed finalize encoder (9.7ms)** + eager
decode (3.6ms, graph NO-GO) + preproc (2.4ms). The big rocks are already optimized or proven NO-GO. Realistic p50
headroom ≈ **1-2ms** from a one-shot finalize preprocessor (preproc 2.4→~1ms, 3 invocations→1). The encoder is the
single biggest p50 chunk and it is already a graph replay. **P50 is close to its compute floor.**

**"Narrow the p95−p50 spread" → it's almost entirely `lock_wait` (lane contention), NOT compute.** lock_wait is
**0.36ms at p50 but 21ms at p95 / 36ms at p99** — i.e. ~the whole 19ms finalize spread. ~5-10% of finalizes wait
20-36ms to acquire a model lane because finalizes queue behind steady batches + each other on the 2 lanes (each
finalize holds a lane ~13ms for model_wall). **This is the real, large lever for the user's "narrow the spread" goal —
and it is a scheduling/contention problem, not a kernel/compute one.**

## Correction to the earlier pivot doc
`conc10-pivot-findings.md` led with finalize fork/clone dedup (server.py 6370/6480/7371). The data says **fork_clone is
only ~0.44ms p50 / 2.1ms p95** — not worth it. The one-shot preprocessor (~1-2ms p50) is the only modest compute win;
the **dominant lever is lock_wait/lane-contention** (the spread).

## Candidate levers, ranked by the data
1. **lock_wait / lane contention (≈19ms of spread, the headline)** — shorten the lane-hold and/or schedule finalizes
   to avoid queueing: e.g. admission control on concurrent in-flight finalizes; a dedicated/priority finalize path so
   a finalize isn't stuck behind steady batches; or reducing the ~13ms lane-hold. Targets p95/p99 directly. Needs a
   design probe (what exactly serializes on the lane during finalize).
2. **one-shot finalize preprocessor (~1-2ms p50)** — the only modest P50 compute win; byte-exact-gated (a prior
   batched-final-preproc attempt dropped terminal punctuation).
3. NOT levers: decode-graph (NO-GO), fork/clone dedup (~0.4ms), gather/scatter/clone (<0.25ms).
