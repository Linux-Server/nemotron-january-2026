# 0.1b microbench — L40S sweep (THE GATE) — 2026-05-24

**Box:** AWS g6e.8xlarge, NVIDIA **L40S** (sm_89, 46 GB), Ubuntu 22.04, 32 vCPU. `i-0101ff8d60e09658f` —
**terminated + leak-checked clean.** Build: pip torch 2.8.0 (manual-link, no nvcc — Ubuntu 22.04/glibc 2.35, no
conflict). Decode model: mock host sleep `--decode-host-us 10000` (calibrated to the measured ~10.4 ms per-chunk
thread-busy, `proj-2026-05-24-0859/gil-attribution.md`). Steady bucket B=1; real encoder graph replayed; per-lane stream
+ event completion. **The exported `.ts` (traced on the 5090) loaded + graph-captured cleanly on the L40S** — TorchScript
portability across machines confirmed.

## Data
| lanes | streams | p50 | p95 (ms) | gpu_util | keep-up? |
|---:|---:|---:|---:|---:|---|
| 1 | 12 | 70 | 121 | 51% | yes |
| 1 | 16 | 172 | **285** | 67% | marginal (**knee ≈16**) |
| 1 | 20 | 1457 | **2688** | 66% | NO (runaway) |
| 1 | 24 | 2738 | 5121 | 67% | NO |
| 8 | 16 | 62 | 64 | 40% | yes |
| 8 | 24 | 64 | 96 | 59% | yes |
| 8 | 32 | 93 | **127** | **80%** | yes (**knee ≈32–40**) |
| 8 | 48 | 1162 | 2117 | **98%** | NO (GPU-saturated) |
| 12 | 32 | 95 | 126 | 81% | yes (≡ lanes=8) |
| 12 | 48 | 1132 | 2055 | 98% | NO |
| 12 | 64 | 3154 | 5891 | 98% | NO |

## Verdict — GATE PASS (≥1.5× cleared): native multi-thread intake ≈ **2–2.5×** on L40S
- **lanes=1 knee ≈ 16 streams** — and the GPU is only **67% there (33% idle, wasted)**: the wall is the single-thread
  intake, exactly the production diagnosis. (Matches the 5090 and the real server's 16-stream keep-up.)
- **multi-thread sustains ~32–40 streams** (p95 127 ms at 32, GPU 80%; runs away at 48 = GPU 98%). **lanes=12 ≡ lanes=8**
  → above ~32 the ceiling is the **GPU encoder compute, not the lanes** → adding lanes/threads past that won't help.
- **So native ≈ 16 → ~32–40 = ~2–2.5× density on L40S, clearing the ≥1.5× gate.** Multi-thread intake reclaims the
  idle GPU until the (real) encoder compute saturates it — precisely the roofline prediction. **Conjunct 2 confirmed on
  the deploy target.** vs production (~16–20/box across K=3 procs), native is ~2× in a SINGLE process.

## Caveats / robustness
1. **The ceiling is GPU-bound by the REAL encoder replay** (lanes=12≡8 proves it), so the ~2× is robust to the
   mock-decode model on the *ceiling* side. The mock decode being pure-host could shift the *knee* a little, not the
   GPU-saturation ceiling.
2. **NO finalize path yet.** Finalize is heavier GPU bursts + the fork; interspersed finalizes raise GPU load → lower
   the steady ceiling. With finalize the real multiplier is likely **~1.5–2×** (still clears the gate, but confirm).
3. **K× model-copy memory bug surfaced:** each lane loads its own 2.5 GB model copy → **lanes=32 OOM'd the 46 GB L40S**
   (80 GB needed). A real runtime needs **shared read-only weights** (spike 0.9) — BUT note density here is **GPU-bound
   at ~32–40, not memory- or lane-bound**, so shared-weights is a memory-efficiency requirement, not the density-ceiling
   lever. lanes=8 (20 GB) sufficed for the A/B.

## Bottom line
**0.1b GATE: PASS — ~2–2.5× on L40S (steady), ≥1.5× cleared, conjunct 2 confirmed on the deploy target.** Remaining to
fully bank it before Phase-1 funding: add the **finalize path** (+ GPU-load sensitivity) to confirm the multiplier holds
≥1.5× with finalizes interspersed. If it does → **GO** to fund the Wave-2 byte-exact ports (0.6a decode / 0.2 encoder).
