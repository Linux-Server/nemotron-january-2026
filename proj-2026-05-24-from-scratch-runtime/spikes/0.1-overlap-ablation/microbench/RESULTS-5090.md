# 0.1b microbench — 5090 calibrated sweep (2026-05-24)

**Box:** local RTX 5090 (sm_120). **Build:** libtorch 2.8.0+cu128 (manual link, no nvcc). **Decode model:** mock =
host sleep `--decode-host-us 10000`, calibrated to the measured per-chunk thread-busy ~10.4 ms (decode 8.2 ms + glue;
`proj-2026-05-24-0859/gil-attribution.md`). Steady bucket B=1 only; real encoder graph replayed; per-lane stream +
event completion. SLO proxy = chunk intake→done p95 (keep-up = doesn't run away).

## Data
| lanes | streams | p50 (ms) | p95 (ms) | p99 (ms) | gpu_util | keep-up? |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 70 | 121 | 121 | 38% | yes |
| 1 | 16 | 130 | **218** | 240 | 47% | marginal |
| 1 | 20 | 1410 | **2601** | 2727 | 47% | **NO (runaway)** |
| 1 | 24 | 2682 | 5024 | 5247 | 48% | NO |
| 8 | 16 | 27 | 32 | 33 | 24% | yes |
| 8 | 24 | 31 | 47 | 48 | 32% | yes |
| 8 | 32 | 42 | 63 | 64 | 44% | yes |
| 8 | 48 | 56 | **94** | 96 | 60% | yes (knee not reached) |

## Reading
- **lanes=1 knee ≈ 16 streams** (p95 218 ms at 16; runaway at 20). **This matches the real server** (gil-attribution
  kept up at 16 streams single-proc) → the calibration is validated against reality.
- **lanes=8 sustains ≥48 streams** at p95 94 ms / GPU 60% — knee not reached; GPU util climbs ~linearly
  (24→32→44→60% at 16→24→32→48) → GPU-bound ceiling extrapolates to **~75–80 streams**.
- **Native multi-thread intake gain on the 5090: ~16 → ≥48 (≥3×), ceiling ~4–5×.** The mechanism matches the
  production diagnosis: the GIL-held decode is **host-bound** (GPU idle during it), so spreading intake across threads
  fills the idle GPU. **Conjunct 2 empirically supported.**

## Caveats (why this is a strong signal, NOT yet the gate)
1. **Mock decode = pure host sleep, ZERO GPU load.** Defensible — gil-attribution shows the GPU is 17–26% busy-bound /
   mostly idle *during* the decode, i.e. decode is host/CPU+sync-bound, so parallelizing it across lanes genuinely fills
   idle GPU. BUT it's a model: the real decode has small GPU ops; if they're non-trivial the GPU saturates sooner and
   the ceiling drops. **TODO: sensitivity rerun with `--decode-gpu-iters > 0`.**
2. **Steady chunks only — NO finalize path.** Finalize is heavier GPU + the fork; interspersed finalizes add GPU load →
   real ceiling lower than the steady-only number here. **TODO: add the finalize path to the harness.**
3. **5090 ≠ L40S (the gate).** The 5090 is a faster GPU with (per this bench) lots of headroom. The **≥1.5× gate is on
   L40S**, where production showed 40–65% idle → at least as much headroom. **The GO/NO-GO must come from the L40S run**
   (blocked on AWS SSO re-auth as of this run).
4. Single steady bucket B=1; SLO proxy is the per-chunk keep-up latency, not the full client TTFB.

## Verdict
**Preliminary GO signal: ≥3× on the 5090, conjunct 2 confirmed empirically.** Easily clears the ≥1.5× bar *on this box*
even before the finalize path. Remaining to convert to the actual gate: (a) the **L40S run** (the deploy target), (b)
the **finalize path + GPU-load sensitivity** (which can only lower the number). If L40S clears ≥1.5× with those, GO to
fund the Wave-2 byte-exact ports.
