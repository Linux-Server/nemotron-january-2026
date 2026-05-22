# EC2 g6 (L4) vs g6e (L40S) — per-instance scaling results

Date: 2026-05-22. Measured with `ec2-bench/` on real EC2 instances (server + load-gen co-located, us-west-2).
The knee = max concurrent realtime streams with proc-lag p95 < 500 ms and 0 errors.

## TL;DR
- **A single server process is GIL/scheduler-capped at knee ~16, regardless of GPU** (L4 and L40S identical).
- **Scale by running multiple GIL-independent processes per box, with CUDA MPS.** L40S: K=3 processes → **48/box**.
- **The cap is vCPU-bound, not GPU-bound** for this small (0.6 B) model — bigger GPUs are wasted unless paired with
  more vCPUs (more processes). **fp16/bf16 and "bigger single-process GPU" are dead ends.**

## The regime frame
Per-instance behavior moves through three regimes; the lever depends on which you're in:
1. **launch-dispatch bound** (1 lane) — one core can't feed the GPU; GPU idle; FLOPs irrelevant.
2. **filling the GPU** (more lanes / more processes).
3. **GPU-compute bound** — only reached for this model by *many* processes; FLOPs finally matter.

## Single-process lane sweep (NEMOTRON_MODEL_LANES, one process)
| lanes | g6 / L4 knee | g6e / L40S knee | note |
|---:|---:|---:|---|
| 1 (batched) | 4 | 4 | batches don't form at low N |
| **2** | **16** | **16** | **sweet spot; GPU ~46% (L4), ~30% (L40S)** |
| 4 | 4 | 4 | regressed |
| 6 / 8 | — | 4 | regressed (L40S GPU only 27% at lanes=6) |

**Lanes cap at 2 (knee 16) on BOTH GPUs, with the L40S GPU at ~27% → the cap is GIL/scheduler serialization, not
the GPU** (an L40S has 3–4× the SMs and still regresses at lanes=4). The lane lever = thread-level parallelism;
the GIL caps useful threads at ~2.

## fp16 / bf16 — REJECTED
| encoder step (L40S, steady B=1) | time | vs fp32 |
|---|---:|---:|
| fp32 | 33.5 ms* | 1.00× |
| bf16 autocast | 42.2 ms | **0.79× (slower)** |
| fp16 autocast | 42.4 ms | **0.79× (slower)** |

Precision is *slower* — autocast cast-overhead exceeds compute savings because the workload is launch-bound, not
compute-bound. Confirms the GPU is not the ceiling. (*absolute inflated by the synthetic probe; the relative is robust.*)

## Multi-process scaling (K processes × lanes=2, L40S, g6e.4xlarge / 16 vCPU)
| K | per-box target | no-MPS knee | no-MPS GPU | **MPS knee** | **MPS GPU** |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 30% | 16 | ~30% |
| 2 | 32 | **32** | 90% | **32** | **~50%** |
| 3 | 48 | ~16 (regress) | 99% | **48** ✅ | **~65%** |
| 4 | 64 | 0 | 100% | ~16 (regress) | ~75% |
| 6 | 96 | — | — | 0 | ~90% |

- **MPS is essential**: it dropped K=2 GPU util 90%→50% (so the no-MPS 90% was time-slice contention) and
  **unlocked K=3 → 48/box** (vs 32 without MPS).
- **The K=3 cap is vCPU-bound, NOT GPU**: at K=3 the L40S is only ~65% utilized. K=4 regresses at ~75% GPU because
  the 16 vCPUs are exhausted (4 servers × 2 lanes + 4 *co-located* load-gens > 16 cores).
- **→ more vCPUs (bigger instance *size*) would buy more density** toward GPU saturation (~K=4–5 → 64–80/box).
- **Caveat:** the co-located load-gens steal ~K vCPUs; in production (remote clients) the per-box cap is likely
  **higher** than 48 even on this 16-vCPU box.

### Multi-process scaling — g6e.8xlarge (L40S, 32 vCPU, MPS)
| K | per-box target | GPU util | kept up | per-box knee |
|--:|--:|--:|:--:|--:|
| 2 | 32 | ~48% | 2/2 | 32 |
| 3 | 48 | ~70% | 3/3 | 48 |
| **4** | 64 | ~78% | **4/4** | **64** |
| 5 | 80 | ~88% | 3/5 | regress |
| 6 | 96 | ~93% | 1/6 | collapse |

**>50 CONFIRMED: 64/box at K=4.** Doubling vCPUs (16→32) moved K 3→4 (48→64). The binding limit then **shifts from
vCPU to the GPU**: K=4 at ~78%, K=5 saturates (~88%) → regresses. So the **L40S per-box ceiling is ~64** — more
vCPUs (g6e.16xlarge) won't exceed it. (In production, without co-located load-gens stealing cores, g6e.4xlarge's
16 vCPUs may also reach K=4 → 64, improving its $/stream below.)

## L4 multi-process — INFERRED (to confirm; creds expired before measuring)
Single-process L4 = 16 @ 46% GPU → K=2 should ≈ fill the (smaller) L4 GPU → **~32/box** (L4 likely GPU-saturates
near K=2, sooner than the L40S). Needs a measured g6 multi-process+MPS run to confirm.

**MEASURED (g6.4xlarge, MPS) — confirms the inference:** K=1 -> 16; **K=2 -> 32** (GPU ~88%); K=3 -> 0 (GPU
saturates at ~100%, collapses). So the **L4 per-box ceiling = 32 (K=2, GPU-bound)** — the smaller L4 fills at K=2
where the L40S fills at K=4, so L40S (64) is ~2x L4 (32), matching its ~2x compute. **Full matrix now measured.**

## $/stream and recommendation (approximate EC2 on-demand)
| Instance | GPU | est. per-box knee | ~$/hr | ~$/stream-hr |
|---|---|---:|---:|---:|
| g6.2xlarge | L4 | ~32 (K=2) | $0.978 | **$0.031** |
| g6e.4xlarge | L40S | 48 (K=3, MPS) | $3.004 | $0.063 |
| g6e.8xlarge | L40S | **64 (K=4, MPS — the L40S GPU-bound ceiling)** | $4.529 | $0.071 |
| g6e.16xlarge | L40S | ~64 (GPU-capped; extra vCPU wasted) | $7.577 | worse |

- **g6 / L4 wins $/stream** (cheapest GPU, fits ~2 processes) → best for cost + horizontal scale.
- **g6e / L40S wins density-per-box** (48+, fewer instances to manage) at higher $/stream → choose if ops prefers
  fewer/denser boxes. Size it for **vCPUs** (the binding limit), and **MPS is required**.
- **Routing layer**: an LB (HAProxy / nginx / ALB) with `leastconn` + per-backend `maxconn` (≈12 = 75% of the
  ~16 process knee, for the 400 ms TTFS headroom) + health-check + drain. Fronts all processes, local + remote.
  A custom proxy only if lag-aware routing is needed. (WS streams are sticky-for-life → no mid-stream rebalance.)
- **Dead ends for this workload**: a bigger GPU for a *single* process; fp16/bf16.

## TTFS (committed, flag-gated, byte-exact)
- `NEMOTRON_BATCH_BARRIER_DRAIN` (round 2): in-phase N=120 finalize TTFS 7947→207 ms.
- `NEMOTRON_BATCH_FINALIZE` (round 3): collapses the finalize storm; in-phase knee 115→~120 (next limiter =
  per-fork finalize preprocessing + close/cold-reset).
- These are TTFS-under-burst robustness (in-phase worst case); production (out-of-phase) TTFS is already ~40–150 ms.
