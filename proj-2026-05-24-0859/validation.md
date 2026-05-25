# Step 6 — Combined cloud validation

Date: 2026-05-24. Region: us-west-2. Account: 419599258555 (Daily Demos).

Validates the proj-2026-05-24-0859 TAIL + DENSITY levers ON, on the real multi-proc + CUDA-MPS +
HAProxy(leastconn, maxconn) production topology. Success metric is the **TAIL (p95/p99 at the operating
point) + DENSITY (in-budget streams/box) + overload-robustness + K=4 recovery — NOT p50** (p50 is VAD+WAN-bound;
see roofline-and-real-limit). All levers are byte-exact / behavior-gated; this step measures, it does not change
correctness.

## Levers under test (all flag-gated, default-off; promoted to base config only if proven here)
- **Step 1** admission/backpressure — `NEMOTRON_ADMISSION_MAX_READY_AGE_MS` (SLO-tied shed; reject = WS-close 1013
  before admit), `NEMOTRON_ADMISSION_MAX_BACKLOG`.
- **Step 2b** padded-T_max finalize bucket — `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1` (one B=1×T_max bucket
  vs 19 per-T; ~19× less finalize pool, local-measured → the K=4 / L4-K=2 enabler).
- **Step 3** host-sync compress — `NEMOTRON_SYNC_COMPRESS=1`.
- **Step 4** priority finalize-lane — `NEMOTRON_FINALIZE_PRIORITY=1`.

## Gates (what PASS means)
| Gate | Box | Config | PASS criterion |
|------|-----|--------|----------------|
| G1 K=4 no-OOM | g6e.8xlarge (L40S 44GB) | K=4, PADDED=1, all levers | All 4 procs serve; NO `OutOfMemory`; gpu_mem fits 44GB w/ headroom (was K=4 OOM cascade `ok=56/944` pre-padded) |
| G2 in-budget density + tail | same | maxconn 12, CONC 12→52 | ~28/box (7/proc) keeps up (`keep-up YES`, vad_stop_recv_to_process bounded); p95/p99 finalize tail at the operating point reported; no p50 regression vs 246/279 baseline |
| G3 overload shed (cliff gone) | same | ADMISSION age=400ms, CONC 40/52 (10–13/proc) | /health `rejected` climbs (attempted ≫ admitted); ADMITTED streams stay in-budget (no ~930ms vad_stop blow-up) |
| G4 L4 K=2-padded no-OOM | g6.4xlarge (L4 24GB) | K=2, PADDED=1 | Both procs serve; NO OOM; gpu_mem fits 24GB — confirms padded obsoletes the per-T finalize-T trim. (Capacity stays keep-up-bound ~7/box regardless — NOT a capacity gain.) |

Baselines for comparison (recorded, levers OFF): conc-10 WAN L40S **246/279** p50/p95 (finalize-graph win);
the overload cliff **vad_stop_recv_to_process 8ms@3/proc → ~930ms@12/proc** (proj-2026-05-23-1731).

## Results

### Box 1 — L40S g6e.8xlarge, K=4, all levers ON (sweep CONC 12/28/40/52, maxconn 12, admission age 400ms) — `prodsweep_1536`
Instance i-0116ee0183fd288c3, terminated + leak-checked clean.

**G1 (K=4 no-OOM): PASS ✓** — gpu_mem **40967 / 46068 MiB** (~5 GB headroom), **0 OOM lines**, all 4 procs served
`ok=500` at every level. Padded bucket active + byte-exact on cloud (4297/4301 records `encoder_finalize_cudagraph=replay`,
key `mode=padded_T_max real_T=…→graph_T=60`). The old per-T K=4 was the `ok=56/944` OOM cascade — **recovered.**

**Client TTFB (vad_stop→final, ms):** conc12 `p50=244 p95=260 p99=308` (matches 246/279 baseline, no p50 regression);
conc28 `p50=260 p95=1317 p99=6125`; conc40 `p50=282 p95=3526 p99=5287`; conc52 `p50=298 p95=2093 p99=2943`.

**Per-finalize-record breakdown (binned per level) — the tail is INTAKE, not compute:**
| conc/proc | vad_recv→proc p50/p95/p99 | lock_wait p95/p99 | finalize_wall p50/p95/p99 | model_wall p95 |
|---|---|---|---|---|
| 12/3p | 0/1/29 | 11/21 | 20/32/41 | 22 |
| 28/7p | 0/**1010/5845** | 31/67 | 33/61/92 | 30 |
| 40/10p | 0/**3238/5006** | 47/72 | 45/82/103 | 33 |
| 52/13p | 15/**1799/2631** | 42/52 | 50/78/98 | 35 |

**G2 (density ~28/box): NOT supported by this run.** The whole multi-second tail is `vad_stop_recv_to_process_ms`
(events queued before the finalize even starts) — i.e. the **per-proc single asyncio thread saturating on steady-chunk
dispatch**, exactly the roofline scheduler/GIL-HOL prediction (GPU compute is fine: finalize_wall ≤103ms, lock_wait
≤72ms, model_wall ≤35ms at ALL loads). The 4 levers correctly bound the finalize/lane side but **none address the
intake-dispatch bottleneck**. Per-proc knee here is ~3–4 streams (conc12=3/proc clean, conc28=7/proc blown).

**G3 (overload shed): UNTESTED (my misconfig).** /health `rejected=0` on all procs. I set the age gate
(`MAX_READY_AGE_MS=400`, which watches `_scheduler_ready` age — drains fast, stayed ~0) but left `MAX_BACKLOG=1e9`.
The real backlog is in `queued_events` (the intake queue), which the **backlog-count** gate would catch — so the shed
needs a re-run with `MAX_BACKLOG` set, not an admission-design failure.

**Two confounds before any deploy decision:** (1) `FINALIZE_PROFILE=1` (sweep forces it) emits a per-finalize log on
the asyncio thread → inflates the intake backlog → density numbers are **pessimistic**; (2) admission untested per above.
→ a corrected L40S re-run (profiling OFF, `MAX_BACKLOG` set, finer conc incl. 16/20/24, ideally K=3 vs K=4) would give
clean density + a real shed test.

### Box 2 — L4 g6.4xlarge, K=2, PADDED=1 (no-OOM check) — `prodsweep_1601`
Instance i-0325c1a7f59d1d1fa, terminated + leak-checked clean.

**G4 (L4 K=2-padded no-OOM): PASS ✓** — gpu_mem **19087 / 23034 MiB** (~4 GB headroom), **0 OOM**, both procs served,
1234 padded_T_max replays. **Full finalize-graph coverage fits L4 24 GB at K=2 with NO per-T trim** → the padded
bucket obsoletes the trim (confirmed). Bonus L4 knee: conc6=3/proc clean (p50/p95 257/286), conc14=7/proc blown
(p50 341 / p95 3362) → L4 in-budget ~3/proc ≈ **~7/box, keep-up-bound** (matches the deploy note; NOT a capacity gain).

### Box 3 — L40S K=4 CLEAN (profiling OFF, admission OFF) — `prodsweep_1620`, i-0863a52, terminated clean
gpu_mem 39835/46068, 0 OOM. **Profiling-off is ~2× better at the knee** (conc28 p95 **1317 prof-on → 636 prof-off**) —
the FINALIZE_PROFILE intake tax was real. But even clean the SLO knee is **~4/proc solid (16/box), ~5/proc marginal
(20/box)**, NOT 7/proc/28: 16/4p `245/271/693`, 20/5p `249/290/3204`, 24/6p `251/393/7629`, 28/7p `259/636/1255`,
36/9p `277/2262/5366`. **G2: ~28/box thesis REFUTED** — box is intake+BW-limited to ~16-20/box.

### Box 5 — L40S K=3 CLEAN (profiling OFF) — `sweep_k3clean`, i-0cf3bd2, terminated clean
gpu_mem 29602/46068 (~16 GB headroom — far more than K=4's 40 GB). 12/4p `243/262/3268`, 16/5.3p `246/292/503`,
20/6.7p `251/302/444`, 24/8p `255/397/3540`. **K=3 SLO knee = ~5.3/proc solid (16/box), ~6.7/proc marginal (20/box).**
**K=3 ≡ K=4 ≈ 16-20/box** (per-proc knee higher at K=3 — less MPS/BW contention — but ×3 vs ×4 nets the same aggregate).
→ **keep K=3** (same density, ~16 GB more headroom, fewer procs).

### Box 6 — admission shed (G3) — L40S K=4, MAX_BACKLOG=24, profiling OFF — `sweep_admshed`, i-06ddc17, terminated clean
**G3 PASS ✓ (mechanism)** — admission FIRED (the backlog-count signal is correct; Box-1's age signal was the misconfig).
/health cumulative attempted≈2000 / admitted≈1138 / **rejected≈862**. Sheds proportionally; **admitted p50 stays <300 at
every level** (246/257/278/292 at 4/7/10/11-per-proc *attempted*). conc28: shed 178/500 → p95 636→**350**. BUT
MAX_BACKLOG=24 too loose to protect the p95 *tail* at heavy overload (40-44/proc still 1.5-2s). **Fix: tighter default
cap ~8-12.** Cliff converted from "everyone degrades" → "shed excess + admitted p50 protected."

### Box 4 — L4 K=2 CLEAN (profiling OFF) — `sweep_l4clean`, i-03e048, terminated clean
gpu_mem 19109/23034, 0 OOM (padded re-confirmed). 6/3p `256/301/1113`, 8/4p `265/390/4012`, 10/5p `284/1583`, 12/6p
`307/2038`. **L4 ≈ ~6/box (3/proc, budget edge).** Profiling-off did NOT help L4 (vs ~2× on L40S) because **L4 is
memory-BW-bound** (GPU encoder is the wall, intake overhead hidden) — confirms L4 ~6/box is real, NOT a profiling
artifact and NOT a regression (matches pre-plan profiled ~7/box; the old "24/box" was a different tight-budget-capacity
metric on g6.2xlarge that overstated the sustainable knee).

### Box 7 — RTX 5090 K=2 LOCAL (full harness: MPS + 2 procs + local_lb, profiling OFF) — `local_5090_k2`
On-box twin of the cloud sweep via the new `ec2-bench/bench_local_sweep.sh` (+ `local_lb.py` leastconn TCP stand-in
for haproxy; no sudo). Client on localhost → **no WAN** (~23 ms lower p50; the keep-up knee is server-side, comparable).
gpu_mem 21/32 GB, 0 OOM. 8/4p `213/219/243`, 16/8p `215/229/268`, 20/10p `215/249/336`, 24/12p `216/256/275`, 28/14p
`219/268/672`. **5090 SLO-robust knee ≈ 24/box (K=2, 12/proc)** (28 marginal — p99 spikes); holds even +WAN.
**Higher than L40S (~16-20) — because the bottleneck is the per-proc single-thread INTAKE and this box has a fast
DESKTOP CPU** (sustains ~12/proc vs cloud server CPUs ~5-7/proc). Confirms the density ceiling is CPU-single-thread-
bound, NOT GPU-FLOPS-bound. CAVEAT: partly reflects the desktop CPU — a server chassis with a slower core would be
lower. server-finalize p95 19.5→67.7 ms (the 1.4 ms-floor GPU shows; finalize never the wall here). K=3 (~30 GB fits)
untested — likely higher ceiling but MPS/BW contention may cap it.

## VERDICT
- **G1 ✅** K=4 no-OOM (padded, byte-exact, 3× confirmed) · **G4 ✅** L4 K=2-padded no-OOM (obsoletes per-T trim)
- **G2** clean density **~16-20/box on L40S regardless of K**, **~6/box on L4** — the ~28/box & old 48/64 numbers are
  refuted/overstated; box is intake+BW-limited. **No p50 regression** (conc-low p50 244-246 = baseline).
- **G3 ✅** admission works (backlog-count signal, protects admitted p50, ship with cap ~8-12)
- **No regression** from Steps 1-5; **profiling ≈2× tail tax on L40S, ~0 on L4** (BW-bound) now quantified

## Decision / deploy update
- [x] `auto_pick_K` stays **K=3** (K=4 NOT a density win — same ~16-20/box, less headroom)
- [x] launch_multiproc: bake the proven byte-exact levers (PADDED + SYNC_COMPRESS + FINALIZE_PRIORITY); admission documented opt-in w/ cap ~8-12
- [x] DEPLOYMENT.md: honest SLO-robust numbers (L40S ~16-20/box, L4 ~6/box), padded=memory/L4 win not density, profiling/BW notes
- [x] memory update (roofline-and-real-limit / deployment-target-sagemaker)
- [x] staged-rollout checklist (below)

## Staged-rollout checklist
1. Deploy with PADDED+SYNC_COMPRESS+FINALIZE_PRIORITY on (byte-exact); verify /health + a canary stream transcript == baseline.
2. HAProxy maxconn per backend sized to the SLO knee: **L40S ~5/proc (16/box solid) or 6-7/proc if 20/box marginal acceptable; L4 ~3/proc (~6/box)**.
3. Enable admission `NEMOTRON_ADMISSION_MAX_BACKLOG≈8-12` once the LB is configured to DRAIN on 1013 (client-facing shed); watch /health attempted/admitted/rejected.
4. Alert on `vad_stop_recv_to_process` p95 (the intake-saturation early-warning) and admission.rejected rate.
5. K=3 default; K=4 only if a future memory-headroom need arises (padded makes it fit, but it won't raise density).

## EC2 lifecycle — ALL CLEAN
SSO `aws sso login --sso-session khk` (~hourly). All 6 boxes terminated + leak-checked (final: 0 nemotron boxes, us-west-2;
other regions confirmed empty). Incident: two same-ITYPE g6e runs collided via `ec2_up.py`'s tag-reuse → fixed with
`NEMOTRON_EC2_NAME` override + state-file override; shared box terminated, no leak.
