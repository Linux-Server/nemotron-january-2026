# idea 2 (finalize encoder CUDA graph) — the WIN: frontier-competitive + byte-exact

## Result (full-1000 @ conc-10 WAN, L40S, prod config + livelock fix + finalize graph ON)
| config | P50 | P95 | P99 |
|---|--:|--:|--:|
| graph-OFF (prior) | 274 | 401 | — |
| **graph-ON** | **246** | **279** | 702 |
| Deepgram | 247 | 298 | — |
| Soniox | 249 | 281 | — |

**P50 246 beats Deepgram; P95 279 beats Deepgram (298) and edges Soniox (281).** All 1000 finals replayed
the finalize graph (encoder_finalize_cudagraph=replay). Byte-exact (graph-on==graph-off finals, FORK_ASSERT 20/0).

## Why it over-delivered (-122ms p95, vs the ~28ms "server compute" estimate): per-call savings COMPOUND
The launch-bound finalize was the dominant tail driver, and fixing it cascaded:
- model_wall 44 -> **13.4** (encoder 39 -> 9.6: ~1376 eager launches collapsed to one graph replay)
- lock_wait 45 -> **20** (faster finalize -> less lane contention)
- vad_stop_to_sent ~79-97 -> **43**
- the ~98ms client<->server gap -> **~13ms** (the slow finalize was CAUSING the backlog/contention behind the gap)
So idea 2 (initially dismissed as the "smaller ~28ms lever") was actually the dominant lever -- the
serialization-drain insight (per-call savings compound across queued finalizes) was correct.

## The residual tail (P95 279 -> P99 702) = occasional SERVER backlog, NOT WAN
vad_stop_recv_to_process_ms: p50 0.1 / p95 18.9 (tiny) but **p99 400 / max 2456ms**. So sporadically the server
processes a received vad_stop seconds-late (end-of-stream backlog under bursts) -> the p99 tail. Server-side
fixable (the next lever if we want to tighten P99). NOT WAN/transmission.

## Shipping
The finalize graph is default-off (NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE), fail-closed, byte-exact -> enable in prod
(deploy/launch_multiproc.sh). Memory: ~2.9GB reserved/proc (B=1 x T=42-60 across 3 managers) -- fine on L40S; on
L4(24GB) trim the T range to the observed 43-58 (16 buckets). The livelock fix (cooperative yield) ships with it.
