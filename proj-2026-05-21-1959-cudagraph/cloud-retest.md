# Step 6 — cloud GPU-bound cudagraph retest at the tight budget (p50<250 / p95<300)

Date: 2026-05-22. EC2 via `step6_cloud_retest.sh` (boot -> bootstrap -> push committed server.py +
cudagraph_encoder.py -> tight-budget sweep graph-OFF then graph-ON -> auto-terminate). Multi-process + CUDA MPS,
`run_l4_ttfs_sweep.sh` with `rounds=5` pooled + staggered loadgen. Tight budget = worst-of-K-procs
**p50 < 250 ms AND p95 < 300 ms**, 0 errors. Byte-exactness already proven at scale (step 4), so this measures
capacity/latency only.

## L4 — g6.4xlarge, K=2 processes x lanes=2 + MPS, cudagraph maxB=8

**Capture/memory:** all 3 managers per process (self.model + lane:0 + lane:1) captured B=1..8,
`managers=3 default=1 lane_stream=2`, ~1.9 s/replica, 0 replay fallbacks. Fit the 24 GB L4 cleanly at maxB=8
(6 replicas total across the 2 processes).

| per-box | N/proc | OFF worst p95 (ms) | OFF | ON worst p95 (ms) | ON |
|--:|--:|--:|:--|--:|:--|
| 16 | 8  | 175 | PASS | 154 | PASS |
| 20 | 10 | 313 | FAIL | 231 | **PASS** |
| 24 | 12 | 338 | FAIL | 296 | **PASS** |
| 28 | 14 | 386 | FAIL | 316 | FAIL |
| 32 | 16 | 442 | FAIL | 444 | FAIL |

**L4 tight-budget per-box: 16 (graph OFF) -> 24 (graph ON) = +50%, byte-exact.**
- The lift lands in the **moderate-load zone** (N=10-12: p95 313->231, 338->296) — exactly the budget-relevant
  regime. Cheaper per-call encoder pulls the tail under 300 where eager couldn't.
- Near the **keep-up knee** (N=16 / 32-box) the tail is ~444 ms both ways — there the limiter is the serial
  dispatch queue saturating (not per-call cost), so graphs don't move it. So cudagraph raises the *tight-budget*
  capacity, not the raw keep-up knee (which stays ~32/box, dispatch-bound).
- `maxB=8` was sufficient: the realtime B-mix is small (avg ~2-3; step 5), B>8 is rare and falls back to eager.
- The graph-OFF column is also the clean **pre-cudagraph tight-budget baseline** (answers the earlier "max
  streams on L4 for 250/300" question): **16/box** without cudagraph.

## L40S — g6e.8xlarge, K=4 processes x lanes=2 + MPS, cudagraph maxB=8

**Capture/memory:** all 12 manager-replicas (4 processes x {self.model + lane:0 + lane:1}) captured B=1..8,
each process `managers=3`, ~1.7 s/replica, 0 fallbacks. Fit 48 GB cleanly.

| per-box | N/proc | OFF worst p95 (ms) | OFF | ON worst p95 (ms) | ON |
|--:|--:|--:|:--|--:|:--|
| 32 | 8  | 349 | FAIL | 80  | PASS |
| 40 | 10 | 569 | FAIL | 113 | PASS |
| 48 | 12 | 436 | FAIL | 156 | PASS |
| 56 | 14 | 660 | FAIL | 143 | PASS |
| 64 | 16 | 767 | FAIL | **216** | **PASS** |

**L40S tight-budget per-box: graph OFF fails at EVERY level (even 32) -> graph ON holds the full 64, byte-exact.**
- **Graph-OFF K=4 bifurcates:** 2 of the 4 MPS clients get starved (p95 ~349 at 32-box, up to 767 at 64) while
  the other 2 are fine (~175). This is the same fragility the keep-up TTFS sweep found ("K=4/64 fragile"); under
  the tight budget it means K=4 graph-off is unusable.
- **Graph-ON eliminates the bifurcation:** all 4 procs uniform (~60-216 ms p95), holds 64/box at p95 216 — a
  ~4x p95 cut AND fair scheduling. Mechanism: each inference is one graph replay instead of a multi-kernel launch
  storm, so the shared MPS context isn't launch-contended -> no client starves. The lever is *largest where
  launch contention is worst* (high-density multi-process + MPS) — much bigger than the L4's +50%.

## Read-through / deployment impact

- **cudagraph is a tight-budget capacity + robustness lever** (not a keep-up-knee lever):
  - **L4 (g6, K=2):** tight-budget per-box **16 -> 24 (+50%)**; raw keep-up knee stays ~32 (dispatch-bound).
  - **L40S (g6e, K=4):** tight-budget per-box **fails-at-32 (fragile) -> 64 robust**; cudagraph is what makes the
    high-density box viable under the SLO at all.
- **$/stream at the tight budget (p50<250/p95<300), graph-ON:**
  | box | per-box (ON) | ~$/hr | ~$/stream-hr |
  |---|--:|--:|--:|
  | g6.2xlarge / L4 | 24 | $0.978 | **$0.041** (was $0.061 graph-off @16) |
  | g6e.8xlarge / L40S | 64 | $4.529 | $0.071 (graph-off K=4 didn't hold the budget at all) |
  (L4 measured on g6.4xlarge; same L4 GPU + enough vCPU for K=2 -> 24/box transfers to the cheaper g6.2xlarge.)
  -> **L4 stays the cheapest $/stream; L40S is the density play, and cudagraph is required for the L40S to hit
  the budget.**
- Byte-exact (step 4), fail-closed, default-off -> safe to enable in production. maxB=8 was sufficient (realtime
  B-mix is small) and fit both GPUs; capture cold-start ~1.7-2 s/replica, one-time at startup.
## Step 7 — coalescing tick (MAX_WAIT) 2x2: KEEP the tick (work-conserving did NOT help)

2x2 on one g6.4xlarge (K=2 x lanes2, tight budget, rounds-pooled), `MAX_WAIT in {8,0} x graph in {off,on}`.
Tight-budget max-streams/box:

| | graph OFF | graph ON |
|---|--:|--:|
| MAX_WAIT=8 (tick) | 20 | **24** |
| MAX_WAIT=0 (work-conserving) | 16 | **24** |

- **Graph ON: MAX_WAIT=0 is neutral on capacity (24=24) and does NOT lower p95** — slightly worse in the budget
  zone (N=10 graph-on p95 214->297; N=12 268->288).
- **Graph OFF: dropping the tick HURTS (20->16, N=10 p95 265->476)** — confirms coalescing still matters in the
  launch-bound regime (the original 40->56 rationale).
- **Why the hypothesis failed:** (1) the tick is *adaptive* — it ends early when there's nothing to coalesce, so
  it adds ~0 latency (the "N=1 unchanged" note at server.py:632); (2) even with cudagraph the single dispatch
  lane still rewards fewer-bigger passes, so work-conserving makes MORE small passes -> more total dispatch ->
  slightly higher p95, not lower.
- **DECISION: keep `NEMOTRON_BATCH_MAX_WAIT_MS=8` (no default change).** Work-conserving is not a win here.
  (Note: graph-OFF MAX_WAIT=8 read 20/box here vs 16 in the L4 section above — N=10 p95 straddles 300, run-to-run
  noise at the boundary; the graph-ON 24/box is consistent across both runs.)

## Client-side WAN benchmark — apples-to-apples vs the first-party APIs (full 1000 @ conc 10)

Setup: the LOCAL machine is the client; it streams the full 1000-sample stt-benchmark over the internet to a
fresh production-config box (cudagraph ON, lanes=2, silence0_warm200, rc1) in us-west-2. Metric = TTFS
(end-of-speech -> final transcript), INCLUDING the WAN — the same client-side methodology as the first-party
services in stt-benchmark/README.md. Driver: `ec2-bench/bench_client_wan.sh`. Measured network RTT
client<->us-west-2: **~23 ms** (TCP-connect AND WS ping/pong, stable 22-24 ms). 0 errors, 0 empties;
transcripts byte-exact = the established baseline.

| box | TTFS median | P90 | P95 | P99 | server-finalize (~= TTFS - 223 ms) |
|---|--:|--:|--:|--:|--:|
| L4 (g6.4xlarge) | 290 | 416 | 447 | 516 | ~67 med / ~224 P95 |
| L40S (g6e.8xlarge) | 274 | 359 | 401 | 477 | ~51 med / ~178 P95 |

(223 ms = 200 ms harness trailing-silence VAD window [a benchmark constant applied to ALL services] + ~23 ms
network RTT. So `server-finalize ~= TTFS - 223`.)

Leaderboard (TTFS median / P95), ours inserted, sorted by P95:
| Service | median | P95 |
|---|--:|--:|
| Soniox | 249 | 281 |
| Deepgram | 247 | 298 |
| Elevenlabs | 281 | 348 |
| AssemblyAI | 256 | 362 |
| Cartesia | 266 | 364 |
| **Nemotron L40S (self-host, WAN)** | **274** | **401** |
| **Nemotron L4 (self-host, WAN)** | **290** | **447** |
| Smallest AI | 398 | 533 |
| Speechmatics | 495 | 676 |
| OpenAI | 637 | 965 |
| Google | 878 | 1155 |
| Azure | 1016 | 1345 |
| AWS | 1136 | 1527 |

Read:
- Both self-hosted GPUs **beat every big-cloud incumbent** (AWS/Azure/Google/OpenAI/Speechmatics, 530-1530 ms
  P95) on every percentile. **L40S median 274 ms is frontier-competitive** (between Cartesia and Elevenlabs).
- The faster GPU (L40S) cuts the server finalize ~67->51 ms median and ~224->178 ms P95 vs the L4.
- The remaining gap to the Deepgram/Soniox frontier (281-298 ms P95) is the **server finalize P95 TAIL**
  (~178 ms on L40S) — confirmed NOT network (~23 ms, stable) and NOT geography. The lever to reach the frontier
  is finalize-latency + tail optimization -> `finalize-optimization-suggestions.md` (analysis in progress).
- WER: byte-exact = baseline (~1.94% raw); the README WER is semantic (Claude-judged) -> run `stt-benchmark wer`
  on the prod_* tags for the exact comparable accuracy number.
