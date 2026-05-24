# Spike 0.1 — Overlap / MPS ablation matrix (SKELETON; run BLOCKED on GPU + post-Python baseline)

**Goal (PLAN §6 / 0.1):** the thesis kill-switch. On the **real post-Python finalize+steady scheduler path**, isolate
*which* serializer dominates and whether a single native process can overlap finalize+steady — vs needing MPS/multi-proc.
**Resolves the "no MPS tax" / 40–48-box claim with data.**

## Ablation matrix (hold model + decode constant; toggle one factor at a time)
| Factor | Off | On | Source anchor for the "on" behavior |
|---|---|---|---|
| batch-finalize | fallback (`inference_lock`) | pinned-lane path | `server.py:6755-6773` vs `:6779-6785` |
| exclusive-model gate | bypassed | enforced | `server.py:3213-3233` |
| `inference_lock` scope | minimal | whole-call | `server.py:4990-5005`, `:5310-5323` |
| lane-end sync | CUDA event | `stream.synchronize()` | `server.py:3175-3178` |
| finalize vs steady lane | cross-lane | same-lane | affinity `:3295-3308`, pinned wait `:6711-6720` |
| process/context | single-proc single-context | MPS | multi-proc | `deploy/launch_multiproc.sh:57-68` |
| CPU thread affinity | none | pinned | — |

**Report (not just the throughput knee):** per-lane CUDA-event timelines + queue/lane wait, from the metric schema
0.10 mandates (`_continuous_finalize_timing`, batch/finalize telemetry — `server.py:5388-5424`, `:6594-6609`,
`:7254-7281`).

## Go / No-go
- **Go:** isolates the dominant serializer AND a single native process overlaps finalize+steady ≥ the
  **pre-registered factor** (`../decision-template.md`).
- **No-go:** only MPS/multi-proc overlaps → single-process "no MPS tax"/40–48-box story is FALSE → PLAN 0.4 tree.

## Run prerequisites — **BLOCKED**
1. The near-term Python plan (`proj-2026-05-24-0859`) must have LANDED; record its exact baseline commit.
2. A GPU (5090 local for dev; L4/L40S on EC2 for the deploy-relevant numbers — `aws sso login --sso-session khk`).
3. Pre-registered thresholds frozen in `../decision-template.md`.

`run_ablation.py` is a skeleton harness (matrix runner + metric parser) — **does not run any model**; it shells out to
the existing `ec2-bench/bench_prod_sweep.sh` / `bench_lanes_ab.sh` with the toggles above once unblocked.
