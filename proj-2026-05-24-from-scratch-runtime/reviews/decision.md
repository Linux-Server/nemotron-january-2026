# Decision memo — from-scratch runtime (LIVING; not the final 0.4 memo)

Status: **POST-PYTHON evidence now IN (2026-05-24).** The near-term Python plan (`proj-2026-05-24-0859`) landed and was
cloud-measured on L4/L40S — source: **`proj-2026-05-24-0859/validation.md`** (6 boxes, all terminated/leak-checked
clean). This is the input the 0.0 worth-it gate was waiting for. Outcome space: **native Rust/C++ (B1) or STOP** (B4/B5
rejected). The 0.4 decision + the 0.0 numeric threshold are still to be *set by the user*, but the two BET conjuncts can
now be evaluated against real data.

---

## POST-PYTHON RESULTS (2026-05-24) — the headline hypothesis was REFUTED; that's the valuable result

1. **Density hypothesis refuted.** The plan was premised on "padded bucket → recover K=4 → ~28/box." Cloud-confirmed:
   **K=4 fits memory (no OOM, 3× pools) but is NOT a density win.** L40S is **~16–20/box regardless of K** (K=3 even
   edges it on headroom + clean p99 → **keep K=3**). L4 is **~6/box (bandwidth-bound)**. The old **48/64 and 24/box
   figures were keep-up knees / a different metric that overstated deployable density ~2–3×** — discard them.
2. **Why:** the entire load tail is **`vad_stop_recv_to_process`** — the per-proc **single-thread asyncio intake
   saturating while the GPU sits 40–65% idle.** None of the four finalize-side levers (admission, padded-T, host-sync,
   finalize-priority) touch that wall — they correctly bound finalize compute + lane-HOL (**worth shipping for
   robustness / overload-cliff**, but they are NOT the density lever).
3. **The real ceiling is the single-thread intake** → exactly what Step-5's GIL probe found and what this project
   targets. The next real density lever is the **native / no-GIL runtime**, not more in-Python finalize work.

## What this does to THE BET

- **Conjunct 2 (native-capturable: GIL/single-thread-bound, not MPS/bandwidth-bound) — CONFIRMED (the decisive one).**
  Production-measured (clean, profiling-OFF: Box 3 K=4 + Box 5 K=3): the multi-second load tail is **entirely
  `vad_stop_recv_to_process`** — the per-proc single asyncio thread saturating on steady-chunk dispatch — while **GPU
  compute is fine at ALL loads** (model_wall ≤35 ms, finalize_wall ≤103 ms, lock_wait ≤72 ms), consistent with the
  roofline's 40–65% idle. Per-proc knee ≈ **4–5/proc (16/box) solid, ~6.7/proc (20/box) marginal**. This is exactly the
  single-thread wall a no-GIL multi-thread intake/dispatch can reclaim.
- **The decisive argument FOR native (new, from the K=3≡K=4 result):** you **cannot** fix the per-proc intake wall by
  *just running more Python processes* — **K=3 ≡ K=4 ≈ the same 16–20/box aggregate** (more procs → proportionally more
  MPS/BW contention, which cancels the added intake parallelism). So the only way to parallelize intake *without*
  multiplying per-proc model-replica + MPS/BW contention is **one native process with multi-thread intake** — precisely
  this project. (Also: ~half the L40S intake tax was `FINALIZE_PROFILE` logging *on* the asyncio thread — profiling-OFF
  was ~2× better at the knee; native off-thread logging removes that class of tax structurally.)
- **Caveat: L4 is BANDWIDTH-bound (~6/box, profiling-OFF didn't help) → conjunct 2 does NOT hold there.** The native
  density play is **L40S / high-BW-GPU-shaped (and the 5090)**, NOT L4 — on L4 the answer stays "add boxes."
- **Conjunct 1 (residual worth ~40–60 eng-wk + carry) — still the user's call, but now mechanism-grounded.** See below.

## 0.0-pre — Residual-CEILING arithmetic (CORRECTED; old numbers were the discredited metric)

Drop the old 20→28→48 static comparison (those used the inflated keep-up-knee metric). The honest, mechanism-grounded
bound:

| | deployable in-budget streams/box | note |
|---|---:|---|
| **post-Python, L40S (K=3)** | **~16–20** | the real baseline the native build must beat |
| post-Python, L4 | ~6 | **bandwidth-bound — native won't help here** |
| GPU utilization at that ceiling (L40S) | **40–65% idle** | the headroom a native intake could reclaim |
| **native ceiling, L40S** | **TBM by 0.1** | reclaiming 40–65% idle ⇒ a plausible **~1.5–2.5× → ~28–40/box** *before the next bottleneck — likely MPS/BW contention or finalize-compute — binds* (K=3's higher per-proc knee vs K=4 shows BW/MPS contention is the next wall). Measure, don't assume |

**Reading:** unlike the pre-Python "thin ~20-streams triple-conditional" framing, the prize is now **a real,
production-confirmed wasted-headroom mechanism** (40–65% idle GPU on L40S behind a single-thread wall). That is a
*stronger* case than before — the upside is plausibly ~1.5–2.5× L40S density — **but it is still unmeasured** (0.1) and
still costs ~40–60 eng-wk + dual-stack carry with **zero p50 upside** and **no L4 benefit**. Whether ~1.5–2.5× L40S
density (5090-class GPUs only) justifies that is conjunct 1 = the user's number.

## 0.5 — Batching / 3–5× throughput (unchanged, still dead)
Realistic mean B ≈ 1.5–2.1 (see `spikes/0.5-batching-sim/FINDINGS.md`). **3–5× steady-throughput remains dead**, and the
density lever is confirmed NOT batching and NOT shared-weights (memory wasn't the cap — K=4 fit) — it is the **no-GIL
multi-thread intake** reclaiming the idle GPU.

## 0.0 THRESHOLD — SET (2026-05-24, user)
- **Fleet:** **L40S / Ada-density** is the production posture → native applies. **L4 is OUT** (BW-bound; native won't
  help; on L4 the answer stays "add boxes").
- **0.0 PASS bar:** **0.1b must show ≥1.5× sustainable L40S density (≥~28/box) at the SLO** vs the ~16–20/box baseline,
  with GPU util rising toward saturation. Below 1.5× → STOP.
- **Cost basis:** **strategic capability bet** — no COGS break-even gate; the build is justified as a capability
  investment (future models, the fusion path, runtime control) *provided* the ≥1.5× floor clears (so we're not building
  on a refuted/negligible gain). This resolves **conjunct 1 in principle.**
- **Net:** the project go/no-go now reduces to **(0.1b ≥1.5× on L40S)** AND the downstream **B1 byte-exact feasibility
  (0.6a decode + 0.2 encoder)**. Conjunct 2 is already confirmed; conjunct 1 is satisfied pending the 0.1b number.

## Running conclusion (updated)
- **Conjunct 2 CONFIRMED on L40S** (single-thread intake wall, GPU 40–65% idle) → the native thesis's core premise holds
  in production. This materially **weakens the earlier "honest prior is STOP"** — there is a real, measured headroom the
  native runtime is uniquely positioned to capture.
- **Conjunct 1 RESOLVED in principle (2026-05-24):** fleet = L40S/Ada-density; cost basis = strategic capability bet;
  PASS bar = **0.1b ≥1.5× L40S density (≥~28/box).** No break-even gate. So the project now hinges on the **0.1b number**
  + B1 byte-exact feasibility, not on a business debate.
- **Deploy nuance:** native density is L40S/high-BW-shaped; on L4 (BW-bound) the answer stays "add boxes." Value
  concentrates on the denser Ada/Blackwell SKUs.

## Pre-registered thresholds + filled decision tree
See `../spikes/decision-template.md`. With conjunct 2 confirmed, set the 0.0 streams/box threshold against the **~16–20
L40S baseline** (not the discredited 28), and gate Wave-2 funding on the 0.1 microbench showing the idle GPU is actually
reclaimable.
