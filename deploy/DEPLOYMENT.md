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
A single process is GIL-capped at ~16 streams regardless of GPU. Per-box scale = **K processes × ~16**, where K
fills the GPU via MPS; fleet scale = N boxes behind the LB.

## Per-GPU config matrix (measured — `g6-vs-g6e-results.md`)
| GPU | instance | lanes/proc | K | per-box knee | bound by | ~$/stream-hr |
|---|---|---:|---:|---:|---|---:|
| L4 | g6.2xlarge | 2 | 2 | ~32 | GPU | **$0.031** |
| L40S | g6e.4xlarge | 2 | 3 | 48 | vCPU + GPU-mem | $0.063 |
| L40S | g6e.8xlarge | 2 | 3 | 48 | **GPU memory** | ~$0.095 |

- **lanes=2/process** is the unit (>2 regresses on the GIL; 1 wastes per-process overhead).
- **MPS is required for K>2** (turns time-slice contention into concurrent SM sharing).
- **Operate each process at the measured in-budget point and shed above it**: L40S is ~6.7 streams/proc, so set
  HAProxy `maxconn 7`; L4 is ~3.5 streams/proc, so use `maxconn 3` or `4` depending on SLO margin. This enforces the
  latency-safe operating point; it does **not** increase capacity. The old `maxconn 12` setting is above the
  in-budget point and can drive the scheduler backlog cliff.
- Server-side defense-in-depth is available via `NEMOTRON_ADMISSION_MAX_BACKLOG` and optional
  `NEMOTRON_ADMISSION_MAX_READY_AGE_MS` (both default effectively off). It WS-closes new connections when the
  always-on backlog signal exceeds the cap: queued per-session scheduler events + scheduler-ready sessions, with
  oldest-ready age included for age-based caps. When enabled, `/health` reports `admission.attempted`,
  `admission.admitted`, and `admission.rejected` plus the live signal.
- **L40S is MEMORY-bound to K=3 with the finalize encoder graph on** (the default 246/279 latency win). Each proc is
  **~11 GB** (model + 2-lane STEADY + FINALIZE graph pools), so K=4 OOMs the 44 GB L40S (4×11≈44 GB; measured
  2026-05-23 — `ok=56/944` error cascade). The 64/box "GPU ceiling" was the pre-finalize-graph *compute* knee; with
  the finalize graph the GPU runs out of **memory** before compute. **Consequence: g6e.8xlarge's extra vCPU buys
  nothing now (the 48 GB GPU, not vCPU, is the limit) → prefer the cheaper g6e.4xlarge for L40S (same K=3/48).** To
  recover K=4 on g6e.8xlarge, the **preferred** shrink is the **padded-T_max finalize bucket**
  (`NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1`): one `B=1 × T_max` bucket replaces the 19 per-T buckets
  (T=42–60) → **~19× less finalize graph-pool memory** (local-measured), with FULL T coverage and no trim
  (shorter finalize inputs are padded to T_max and masked → byte-identical encoder output for the real frames).
  The per-T `_T_MAX`/`_MAX_B` trim is the fallback. *Note the recovered K=4 is a MEMORY fit, not a capacity gain:
  the in-budget operating point stays keep-up-bound (~6–7 streams/proc) — see the maxconn note above.*
  Cloud K=4 no-OOM + density confirm is pending (proj-2026-05-24-0859 Step 6).
- **L4 (g6.* / 24 GB) is ~3× worse than L40S** (encoder is memory-bandwidth-bound; L4 ~8.2ms floor vs L40S ~2.9ms),
  in-budget at **K=2, ~3–4 streams/proc (~7/box)** — keep-up-bound, NOT a capacity choice. The padded-T_max bucket
  (above) is also the preferred L4 K=2 fit: full finalize-graph coverage projected to fit 24 GB at K=2 with headroom
  (from the ~19× local drop), replacing the per-T trim previously needed. Cloud no-OOM confirm pending (Step 6).

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

## $/stream summary
**L4 (g6.2xlarge) ≈ $0.031/stream is cheapest → cost-optimized + horizontal scale.** L40S (g6e) is the
density play (**48/box at K=3**, fewer instances) at ~3× $/stream — its density shrank because the finalize graph
caps L40S at K=3 (memory), so use the cheaper **g6e.4xlarge** (not g6e.8xlarge) when you do want L40S density.
Choose L4-multi-box unless ops strongly prefers fewer/denser boxes. Spot pricing ~halves both (keeps the ratio);
use spot for stateless capacity with drain on interruption.

## Artifacts
- `deploy/launch_multiproc.sh` — multi-process + MPS launcher (Step 2).
- `deploy/haproxy.cfg.example` — routing layer (Step 3); ALB equivalent noted inline.
- `ec2-bench/` — benchmark toolkit + runbook to measure the per-GPU matrix on the real production instance
  (the knee is CPU-bound, so confirm on the actual instance type before sizing the fleet).
