# Nemotron streaming ASR — production deployment design

DESIGN ARTIFACT covering Steps 2 (launcher), 3 (routing), and 7 (substrate) of
`proj-2026-05-21-inference-opt/PLAN.md`. Grounded in the measured per-instance scaling
(`proj-2026-05-21-inference-opt/g6-vs-g6e-results.md`). "Infra as design" per the user's scope choice — these are
the reference artifacts + the substrate decision, not a live cluster.

## Architecture
```
            Load balancer (HAProxy / AWS ALB)   leastconn + maxconn≈7/process on L40S, ≈3-4 on L4, /health, drain
           /              |               \
   box (g6/g6e)       box (g6/g6e)      box ...        ← autoscaling group / ECS service
   ├ CUDA MPS daemon                                    each box:
   ├ proc0  server.py lanes=2  :8080                    - 1 MPS daemon (concurrent GPU sharing)
   ├ proc1  server.py lanes=2  :8081                    - K server processes, each lanes=2 (knee ~16)
   └ procK  server.py lanes=2  :808K                    - launcher = deploy/launch_multiproc.sh
```
A single process's *keep-up* knee is ~16 streams, but the **SLO-robust** point (sustained p95<300ms + clean p99) is
far lower — ~5–7/proc on L40S, ~3/proc on L4 — because the per-proc asyncio **intake thread** saturates
(`vad_stop_recv_to_process` blows to seconds) well before the keep-up knee, while the GPU sits 40–65% idle.
**Provision for the SLO-robust point, not the keep-up knee.**

> **⚠️ Re-measure correction (2026-05-27, same g6e/L40S, single-utterance burst — `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/l40s_w3_logs/spy_*.json`):** a **single** server process on the *full* GPU is SLO-robust to **~20 streams** at **ttfs p50 ~42–54ms** (p95 158ms @20; server-side vad_stop→final). The multi-proc **K=3+MPS** config reaches the *same* ~16–20/box but at **ttfs p50 ~245ms** at matched load — so MPS+multi-proc here buys **≈no density over one process** yet pays a **~190ms median-latency tax** (it was sized on the since-refuted "48/box" keep-up belief). The "~5–7/proc" above is the *MPS-degraded* per-proc, **not** a single proc's ceiling. **This is a LEAD, not yet a flip:** the loadgen is one-utterance-per-connection; the genuine multi-proc rationale is per-proc asyncio **intake** parallelism under *sustained multi-turn* load, which this test does not exercise. **Action before changing prod:** run a single-proc *sustained multi-turn* load test — if a single proc holds ~16–20 under sustained load, prefer **fewer procs / no MPS** for the latency win.

## Per-GPU config matrix (measured 2026-05-24, cloud SLO-robust — `proj-2026-05-24-0859/validation.md`)
SLO-robust = sustained **p95<300ms + clean p99** (the competitive bar). Relaxing to the 400ms hard budget gives
~24/box L40S / ~8/box L4, but p99 degrades. The OLD "knee" numbers (L4 ~32 / L40S 48) were *keep-up* knees that do
**not** hold the SLO — they overstate deployable density ~2–3×.
| GPU | instance | lanes/proc | K | **SLO-robust /box** | bound by | notes |
|---|---|---:|---:|---:|---|---|
| L4 | g6.4xlarge | 2 | 2 | **~6** (3/proc) | GPU mem-BW | ~3× costlier/stream than L40S |
| L40S | g6e.4xlarge | 2 | 3 | **~16–20** (5.3–6.7/proc) | per-proc intake + GPU-BW | preferred (cheaper than 8xlarge) |
| L40S | g6e.8xlarge | 2 | 3 | **~16–20** | same (extra vCPU idle) | no density gain over 4xlarge |

K=4 fits memory with the padded bucket but is **NOT** a density win — L40S is ~16–20/box regardless of K (K=3 even
edges K=4 on clean-p99 headroom). $/stream is ~3× the old keep-up-knee estimate (recompute against current instance
pricing using these /box numbers).

- **lanes=2/process** is the unit (>2 regresses on the GIL; 1 wastes per-process overhead).
- **MPS is required for K>2** (turns time-slice contention into concurrent SM sharing).
- **Operate each process at the SLO-robust point and shed above it**: L40S ~5–7 streams/proc (HAProxy `maxconn 6`
  for the 16/box solid point, `7` for ~20/box if marginal p95 is acceptable); L4 ~3 streams/proc (`maxconn 3`). This
  enforces the latency-safe point; it does **not** increase capacity. `maxconn 12` is well above it and drives the
  scheduler/**intake** backlog cliff (`vad_stop_recv_to_process` → seconds).
- Server-side defense-in-depth: set **`NEMOTRON_ADMISSION_MAX_BACKLOG` ≈ 8–12** (the backlog-count signal — queued
  per-session scheduler events + scheduler-ready sessions). It WS-closes (1013) new connections past the cap, so the
  **LB must DRAIN on 1013**. Cloud-verified 2026-05-24: sheds proportionally under overload and keeps admitted
  **p50<300** (a tighter cap also protects the p95 tail). The age signal `NEMOTRON_ADMISSION_MAX_READY_AGE_MS` does
  **NOT** track this intake-bound overload (the ready-set drains fast; the backlog is upstream in intake) — use the
  backlog-count cap. `/health` reports `admission.attempted/admitted/rejected` + the live signal.
- **Keep K=3 on L40S** (`auto_pick_K`=3). K=4 fits memory with the padded bucket (gpu_mem 40/46 GB, 0 OOM, cloud-
  confirmed 2026-05-24) but is **not a density win** — L40S is ~16–20/box regardless of K (per-proc intake + GPU-BW
  ceiling, not proc count), and K=3 has ~16 GB more headroom (30 vs 40 GB) + edges K=4 on clean-p99. The old K=4
  `ok=56/944` OOM was the per-T finalize pool; the padded bucket fixes the *fit*, but use it for headroom/L4, not density.
  Prefer the cheaper **g6e.4xlarge** for L40S (extra 8xlarge vCPU sits idle — the limit is per-proc intake + GPU-BW).
- **Padded-T_max finalize bucket SHIPS ON** (`NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED`, default 1 in
  launch_multiproc): one `B=1 × T_max` bucket replaces the 19 per-T buckets (T=42–60) → **~19× less finalize
  graph-pool memory**, FULL T coverage, byte-exact (padded+masked → byte-identical encoder output for the real
  frames; cloud-verified `mode=padded_T_max` replay). It obsoletes the per-T `_T_MAX`/`_MAX_B` trim (now fallback only).
- **L4 (g6.4xlarge / 24 GB) ≈ ~6/box** (K=2, ~3 streams/proc; cloud-confirmed clean AND profiled 2026-05-24, gpu_mem
  19/23, 0 OOM, padded no-trim). ~3× worse than L40S because the encoder is **memory-bandwidth-bound** (L4 ~8.2ms
  floor vs L40S ~2.9ms) — the GPU *is* the wall on L4, so unlike L40S, profiling-off does NOT lift it. Genuinely
  BW-bound-low, not a tuning artifact and not a regression (the old "~24/box" was a different tight-budget-capacity
  metric on g6.2xlarge that overstated the sustainable knee).

## Substrate decision (Step 7)
For a long-lived **WebSocket** service (not request/response):
| Substrate | Fit | Notes |
|---|---|---|
| **EC2 + ASG + ALB** | **recommended start** | Full control of MPS + the launcher (systemd unit); ALB does WS + least-outstanding-requests + drain. Simplest path. |
| ECS (EC2 launch type) + ALB | good if already on ECS | Launcher = task entrypoint; ASG via capacity provider; MPS needs the daemon in the task/host. |
| EKS | only if k8s-native | K containers/pod or a K-process supervisor; MPS via device-plugin/daemonset. Heaviest. |
| SageMaker real-time endpoint | **not a fit** | Request/response, no raw WS. Only via a custom-container streaming/async pattern + glue — revisit later if mandated. |

**Recommendation:** EC2 + ASG + ALB to start (or ECS if you're already there). The launcher + MPS are
substrate-portable, so this decision is reversible.

## MPS hardening (the isolation tradeoff — important)
MPS shares one CUDA context, so a CUDA fault in one process can corrupt the context and **take down the others on
that GPU** (bigger blast radius than separate contexts). Mitigations:
- Per-process supervision + fast restart (the launcher); if multiple procs die together, restart MPS + all procs.
- LB **health-check + drain** so a restarting process doesn't receive new streams until `/health` passes.
- Optional per-client SM caps (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`) for QoS.
- **MIG** (hard isolation) is an A100/H100 alternative — **not available on L40S/Ada**.
- Fallback: run **without MPS at K=2** (full isolation, lower density) if the blast radius is unacceptable.

## Autoscaling
Scale boxes on aggregate utilization = active_streams / (boxes × per-box-knee), targeting ~75%. ALB target-group +
ASG; deregistration delay = the drain window. Cold start (model load + MPS + K-process warmup) is ~minutes → keep
warm headroom or pre-warm new boxes before adding to the LB.

## Native runtime (density roadmap — Phase-2, when funded)
The Python multi-proc+MPS design above is the *shipping* path. The L40S **density ceiling** is GIL/asyncio per-proc
intake + scheduler serialization (GPU 40–65% idle at the SLO-robust point) — **not** the GPU. A **native
C++/libtorch runtime** (one process, N true threads, ONE shared weight set via `AOTIModelPackageLoader(num_runners=N)`)
breaks that ceiling.

**Phase-2 Step-1b gate — DONE/PASS (2026-05-27, `proj-2026-05-24-from-scratch-runtime/`):**
- **SLO-robust knee = ~36 streams/box on L40S** (one process) vs Python's ~16–20/box = **~1.8–2.25×**. Binding =
  keep-up/GPU-compute, **not memory** (0.035 GiB/stream → could hold 1000s).
- **Density, not latency:** server-side ttfs is *comparable* to a single Python proc (native p99 147ms @36 vs
  single-proc Python ~42ms p50 @20 — both well under the 175/250 budget). The ~2× is purely the GIL-break letting
  the GPU saturate at 36 vs ~20. (The Python *245ms* is the MPS multi-proc tax noted above, not inherent.)
- **No MPS, no multi-proc, one weight copy** → smaller blast radius than the MPS design.

**When to invest:** a ~40–60 eng-week 2nd-stack bet (separate *funding* call; technical-GO ≠ funding-GO). Take it
when L40S density (fewer/denser boxes at ~2×) beats the Python fleet's ops simplicity. Levers to push the knee >36
(steady-encoder contention / decode D2H syncs / enc_first lock / cross-stream batching) are under analysis — see
`proj-2026-05-24-from-scratch-runtime/reviews/` + the checkpoint notes.

## $/stream summary
**L4 (g6.2xlarge) ≈ $0.031/stream is cheapest → cost-optimized + horizontal scale.** L40S (g6e) is the
density play (**~16–20/box** SLO-robust — *not* the old "48"; that was a keep-up knee that overstated ~2–3×) at
~3× $/stream, capped by per-proc intake + GPU-BW (not proc count), so use the cheaper **g6e.4xlarge** (not
g6e.8xlarge — its extra vCPU sits idle). Choose L4-multi-box unless ops strongly prefers fewer/denser boxes. Spot
pricing ~halves both (keeps the ratio); use spot for stateless capacity with drain on interruption. **The native
runtime (below) is the real L40S density play (~36/box, ~2× Python) when funded.**

## Artifacts
- `deploy/launch_multiproc.sh` — multi-process + MPS launcher (Step 2).
- `deploy/haproxy.cfg.example` — routing layer (Step 3); ALB equivalent noted inline.
- `ec2-bench/` — benchmark toolkit + runbook to measure the per-GPU matrix on the real production instance
  (the knee is CPU-bound, so confirm on the actual instance type before sizing the fleet).
