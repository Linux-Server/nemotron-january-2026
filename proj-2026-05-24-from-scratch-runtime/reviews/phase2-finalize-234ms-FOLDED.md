# Finalize 234ms-@N=1 anomaly — paired investigation FOLDED (Codex + Opus-4.7 max-thinking)

Inputs: `codex` investigation (task byq740rvf) + `opus-finalize-234ms-investigation.md` (independent). Strong
convergence. **VERDICT: the 234 ms is NOT a finalize bug or a 10× regression — it's a per-AOTI-bucket COLD-START
(CUDA-12 lazy module load + first-kernel-launch) amplified by a tiny-sample p95. Warm native finalize is
order-of-magnitude lower (≈11 ms p95) — Python-class. The fix is WARMUP, not the graph / decode / clone / autotune.**

## What both found (convergent)
- **It's on the GPU stream, one-time-per-bucket, contention-free.** `finalize_runner_wait` (host−GPU) ≈ 0 → the
  ~225 ms is genuinely GPU first-launch, not host glue/lock/queue. `finalize_gpu` p50 is FLAT at ~225/228/230 ms
  across N=1/2/4 with ~0 wait → cold-start that never amortizes (rules out contention). (Opus)
- **The clincher (Opus):** the STEADY encoder shows the IDENTICAL spike — `steady_gpu` p50 5.5 ms but **max
  230/269/311 ms at N=1/2/4** (~one spike per worker's FIRST forward), amortized away over ~292 chunks. The same
  ~225 ms first-launch cost is **universal to every AOTI loader**; finalize runs each bucket ≈once, so it can't
  hide it.
- **Tiny-sample p95=max (Codex):** the bounded floor sweep had **6 finalize samples**, and the harness percentile
  makes **p95 == max at n=6** (`density_main.cpp:215`) → "p95" = a cold outlier. The **200-session N=1 run
  (`20260526T170254Z`) shows the warm path: ttfs p50 7.9 / p95 11.4 ms; p99 241 / max 250** — the ~240 ms lives at
  p99/max as cold outliers, NOT the steady floor.
- **Ruled out (both):** the missing finalize CUDA graph as the 200 ms cause (a graph collapses launch *dispatch*
  ~tens of ms — Python encoder 39→9.7 ms — but here dispatch≈0; the 225 ms is GPU first-launch latency a graph
  does NOT remove); decode `.item()` (≤4 ms); fork/clone (~1 ms); manifest/SHA (loader-time, once-per-pool).
- **Autotune is ORTHOGONAL (Opus):** autotune speeds steady-state kernels, NOT first-launch latency — so
  autotune-ON will NOT fix the 234 ms without warmup. Chase the residual warm cost only after de-contaminating.

## The mechanism (Opus, sharper) + the Python lesson (the doc the user pointed to)
- **Root cause (conf ~0.9): CUDA-12 default LAZY module loading.** The harness sets no `CUDA_MODULE_LOADING`, so
  each bucket's embedded cubins are lazily loaded + cold-launched on FIRST use (~225 ms one-time per loader).
- **Python AVOIDS this by folding warmup into startup** (`NEMOTRON_WARMUP_MS=200`) — its finalize encoder is the
  same warmed module, not a separate cold per-T package. So there's **no Python ~200 ms precedent**; the native
  path hit a *different* mistake Python already solved: **no per-bucket warmup.** Python's actual finalize lever
  was launch-collapse (~30 ms), a separate, smaller thing.
- **Decomposition (Opus):** of the 234 ms ≈ **~160 ms cold-start (artifact) + ~60 ms warm bucket forward.** The
  warm residual is a *modest* real gap (~3-4× the Python *graphed* finalize, partly because these are autotune-OFF
  artifacts over T≈45-58). (Minor divergence: Codex's 200-session warm p95 ≈ 11 ms vs Opus's ~60 ms "fast" ones —
  difference is how warm the bucket was; the clean re-run pins it. Either way ≫ below 234.)
- **Honest residual (Opus):** couldn't confirm the cubins' SM target on the host (no `cuobjdump`); lazy-load
  explains the magnitude without an arch mismatch (these are sm_120 artifacts on the 5090, so no mismatch expected).

## FIX (both agree) — warmup, not the graph
1. **Runtime requirement (real, not just measurement):** warm EVERY finalize bucket at startup (one throwaway
   forward per `(drop,T)`), matching Python's `NEMOTRON_WARMUP_MS=200` — else the first real finalize per bucket
   pays ~225 ms in production. Optionally `CUDA_MODULE_LOADING=EAGER`.
2. **Harness/measurement fix:** the density sweep currently warms only the first utterance/bucket per worker
   (`density_main.cpp:2797`; buckets preloaded-not-run-warmed `:2775`) → warm ALL loaded buckets before timing;
   require ≥20–100 finalize samples for a valid p95 (flag n<20 invalid); **split telemetry** into
   `fork_clone / aoti_run_cuda / enc_len_sync / decode_wall / decode_item_wait / decode_tokens / glue`.
3. **Re-sweep after de-contaminating.** The floor sweep's **finalize/TTFS NO_PASS verdict is cold-start-driven and
   UNTRUSTWORTHY** until warmup covers all buckets. Only then chase the residual warm gap (autotune-ON, then maybe
   a finalize graph for the ~tens-of-ms launch-dispatch).

## Implications
- **Track A (autotune-on sweep) will be cold-contaminated too** with the current harness → apply the warmup +
  sampling fix before trusting its finalize p95 (its steady numbers + the autotune-on *artifacts* are still valid).
- **The user's instinct to chase the 10× was right:** it surfaced a real **startup-warmup gap** (a genuine
  runtime requirement the Python stack already meets) that would otherwise have silently corrupted the density/SLO
  numbers — even though the headline 234 ms turned out to be ~70% measurement artifact.
