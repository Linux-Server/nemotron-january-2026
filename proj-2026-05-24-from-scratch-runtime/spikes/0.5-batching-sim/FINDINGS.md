# Spike 0.5 — findings (synthetic phase-sensitivity; pre-Python, no GPU spend)

**Status:** synthetic + existing-data cross-check COMPLETE. Real post-Python traces still pending (BLOCKED on server
instrumentation), but the **kill direction is already well-supported** — run before any GPU spend, as intended.

## Runs (`simulate.py --synthetic`, 160 ms steady cadence, batch key honored incl. fresh/established split)

| arrival | sessions | window | mean B | %B=1 |
|---|---:|---:|---:|---:|
| poisson | 24 | 8 ms (deployed `BATCH_MAX_WAIT_MS`) | **2.08** | 36% |
| bursty | 24 | 8 ms | **1.89** | 41% |
| phase_aligned (unrealistic) | 24 | 8 ms | 24.0 | 0% |
| poisson | 12 (in-budget per-proc) | 8 ms | **1.52** | **63%** |
| poisson | 48 | 8 ms | 3.28 | 11% |
| poisson | 24 | 20 ms | 3.78 | 6% |
| poisson | 24 | 40 ms | 6.49 | 3% |

## Reading
- **At the deployed window (8 ms) with realistic independent/bursty arrivals, mean B ≈ 1.5–2.1 and B=1 is 36–63%.** At
  the *in-budget per-process* concurrency (~12 sessions/proc) it lands at **mean B 1.5 / 63% B=1**, which **converges
  with the measured "86% B=1"** (memory `streaming-batching-outcome`) — synthetic + first-principles + the real datapoint
  agree.
- **First-principles why:** independent arrivals at 160 ms cadence into an 8 ms window → expected coincidence ≈ 1.
  Voice traffic is independent humans, not phase-aligned, so B≈1 is the *structural default*. Only `phase_aligned`
  (unrealistic) fills B.
- **The only knobs that raise B both have a catch:** (a) widen the window (8→20→40 ms lifts mean B 2.1→3.8→6.5) but that
  **adds user-facing latency** — forbidden for steady by the plan's non-goals; (b) raise per-process concurrency (48
  sessions → mean B 3.3) but that's **capped by the in-budget operating point**.

## Verdict (provisional, confirm with real traces)
- **The 3–5× steady-throughput claim is effectively dead** without adding latency: realistic mean B ≈ 1.5–2.1 ≪ the
  pre-registered `median B ≥ 2, p95 B ≥ 4` bar. → **drop 3–5× from the value case** (re-run 0.0-pre).
- **The steady-graph *throughput/density* rationale weakens with it:** batching can't amortize weights across B if B≈2;
  exact-B graphs for B=1–3 would have a *high hit-rate* but a *small* throughput payoff. Native density must therefore
  come from **shared read-only weights + finalize/steady overlap** (spikes 0.9/0.1/0.11), **NOT** from batch-fill.

## Caveats
Synthetic arrival models are idealized; `gen_synthetic` is coarse. The number to *act on* must come from
post-Python instrumented traces (the trace schema in `README.md`). But the kill **direction** (B≈1, not 3–5×) is robust:
synthetic + existing 86%-B=1 measurement + the cadence/window arithmetic all agree. No GPU/cloud was used.
