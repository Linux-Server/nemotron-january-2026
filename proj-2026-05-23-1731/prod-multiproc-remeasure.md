# Production re-measurement: conc-10 on multi-proc + MPS (K=3) vs single-proc baseline

Config: g6e.8xlarge (L40S 44GB), **K=3 procs lanes=2 + CUDA MPS + HAProxy (leastconn, maxconn 12/proc)**, finalize
graph ON, conc-10 full-1000 client over WAN to the LB. ok=998/2 err, no OOM (37.7/46 GB resident). 2157 finalize
records merged across 3 procs. (K=4 OOMs — see the launcher/DEPLOYMENT updates.)

## Client TTFB (the headline)
| config | p50 | p95 | p99 | spread (p95−p50) |
|---|---|---|---|---|
| single-proc lanes=2 (old leaderboard bench) | 246 | 279 | 474 | 33 |
| **multi-proc K=3 + MPS (production)** | **244** | **262** | **339** | **18** |

At conc-10 the multi-proc box is BETTER (p95 −17, p99 −135, spread 33→18) and still beats Deepgram (247/298) +
Soniox (249/281). Server-finalize p50/p95: 46/79 → 44/62.

## Finalize decomposition: single-proc (SP) vs multi-proc K=3 (MP)
| field | SP p50 | MP p50 | SP p95 | MP p95 | SP p99 | MP p99 |
|---|---|---|---|---|---|---|
| finalize_wall | 22.1 | 19.8 | 41.3 | 31.8 | 54.4 | 42.9 |
| **lock_wait** | 0.36 | **0.06** | 21.1 | **12.4** | 35.7 | **20.8** |
| model_wall | 13.4 | **15.3** | 17.6 | **20.6** | 20.0 | **23.2** |
| encoder_wall | 9.7 | **11.4** | 12.0 | **15.8** | 12.9 | **18.0** |
| decode_wall | 3.6 | 3.7 | 6.4 | 6.2 | 8.4 | 7.7 |
| preproc_wall | 2.4 | 2.4 | 5.5 | 3.9 | 6.8 | 5.1 |

## Two effects, opposite signs
1. **lock_wait DROPS** (p95 21→12, p99 36→21): conc-10 spread across 3 procs (leastconn) ⇒ ~3.3 streams/proc
   (~1.65/lane) vs 10/proc (5/lane) single-proc ⇒ much less per-proc lane contention.
2. **MPS adds a GPU-sharing tax** (encoder p99 12.9→18, model_wall p95 17.6→20.6): 3 procs sharing the GPU via MPS
   run each kernel on fewer SMs, so the compute itself is ~15-40% slower.

Net at conc-10: the lock_wait win (−9ms p95) outweighs the MPS tax (+3ms p95) ⇒ finalize + TTFB improve.

## THE CRUCIAL CAVEAT — conc-10 UNDER-LOADED each proc
The production operating point is **maxconn=12/proc** (75% of the ~16 knee). At conc-10-to-LB each proc ran only
~3.3 streams — far below 12. **lock_wait scales with per-proc load**: ~3/proc → p95 12ms; 10/proc (single-proc) →
p95 21ms. Extrapolating to the maxconn=12/proc operating point, lock_wait (and the spread) returns to ≈ the
single-proc 279/474 level — PLUS the MPS tax. So the lock_wait spread is **NOT a single-process artifact; it is
per-proc-load-dependent**, and the conc-10-to-LB number (244/262) reflects a lightly-loaded box, not peak.

## Verdict + next step
- Production at LIGHT aggregate load (conc-10 total): excellent (244/262/339), frontier-beating.
- Production at PEAK per-proc load (each proc ~12, maxconn): the spread/tail re-appears (≈279/474 + MPS tax).
- To answer "is the spread worth a lever in production," measure at the OPERATING POINT: **conc ≈ K×12 = 36 to the
  LB** (saturates the box: all 3 procs at maxconn). That gives the real production-peak p50/spread. The conc-10
  number is reassuring but not the peak.
- MPS tax on model_wall is a density-vs-latency knob (fewer procs = more SMs/proc = faster compute, lower density).
