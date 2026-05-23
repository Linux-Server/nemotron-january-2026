# Leaderboard P95 decomposition (prod config + livelock fix, conc-10 WAN, L40S) — task #60

Client TTFS (results.db, 1000 samples, prod config = cudagraph+lanes2+BATCH_FINALIZE+BARRIER_DRAIN + the
scheduler-livelock fix): **p50 274 / p90 359 / p95 401 / p99 477 ms** — IDENTICAL to the original (no
BF/barrier/fix). Expected: the fix prevents the hang but doesn't change latency; BF/barrier don't help at conc-10
(no finalize coincidence). Frontier: Deepgram 247/298, Soniox 249/281.

## Decomposition (pairing client TTFS with the 814 emitted server-side finalize records)
| layer | p50 | p95 |
|---|--:|--:|
| trailing-silence VAD window (FIXED benchmark) | 200 | 200 |
| client `vad_stop→final-received` (= TTFS − 200) | 74 | 201 |
| └ server `vad_stop→sent` (FINALIZE_PROFILE) | 49.9 | 79.4 |
| └ **client↔server gap** (network + end-of-stream audio backlog + WS transmission) | ~24 | ~122 |

Server `vad_stop→sent` (79 p95) breakdown: `model_wall` 43 (encoder 39 launch-bound + decode 5) + `lock_wait` 31
(lane contention) + trigger/queue/emit ≈ 0. **`queue_wait` p95 0.4 → idea 1 (coalescing) is NOT the leaderboard
lever** (finals don't coincide at conc-10).

## The tail (274→401 = 127 ms spread) splits:
- **~29 ms server finalize** (p50→p95 of `vad_stop→sent`, 50→79).
- **~98 ms client↔server gap** (the gap grows from ~24 at p50 to ~122 at p95) — the DOMINANT tail driver. This is
  the I/O path AFTER `final_sent`: network delivery + the server's end-of-stream audio backlog (vad_stop processed
  late under load) + the WS send/transmission. NOT the finalize compute. (Deepgram, same harness+WAN, has a tight
  51 ms spread → our extra tail is something WE do on this path, not the harness.)

## Levers for the creative optimization grind (task #61)
1. **Server finalize (idea 2 — finalize encoder CUDA graph):** `model_wall` 43 is the biggest server-side
   component and is LAUNCH-BOUND (~1376 kernel launches/finalize; kernel profile). A graph collapses launches ->
   encoder ~39 toward its GPU floor -> server finalize ~79->~50 -> client p95 401->~372. Concrete, buildable
   (cudagraph_encoder.py exists; per-(B,T) finalize buckets). ~28 ms.
2. **lock_wait (31, lane contention):** finalize lane priority / dedicated finalize lane.
3. **The ~98 ms client↔server gap (the BIGGER prize):** investigate — is it (a) the server's end-of-stream audio
   backlog (vad_stop processed late under conc-10 load -> the finalize starts late from the CLIENT's view), (b) the
   WS send path under conc-10 (Nagle/TCP_NODELAY, send queuing), or (c) WAN tail (multi-region/closer POPs)?
   This is most of the 127 ms tail and is NOT the finalize compute — needs its own probe.
- idea 1 (coalescing): NOT needed at conc-10 (queue_wait ~0); only relevant at high coincidence/scale.
