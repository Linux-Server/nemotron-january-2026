# Step 1 — finalize telemetry + reproducibility gate: VERDICT = STOP (do not build the finalize graph)

Measured on-box (L40S g6e.8xlarge, conc 10, production config: cudagraph ON + lanes=2 + silence0_warm200,
`NEMOTRON_FINALIZE_PROFILE=1`), **803 finals** across 2 bursts. Driver: `step1_repro_gate.sh`.

## Server-side finalize split (ms), net of WAN — p50 / p95 / max
| component | p50 | p95 | max |
|---|--:|--:|--:|
| model_wall | 36.4 | **39.3** | 45.7 |
| encoder | 32.9 | 35.4 | 41.7 |  (~90% of model_wall) |
| decode | 3.5 | 4.6 | 5.3 |
| preproc | 2.5 | 3.0 | 3.6 |
| lock_wait | 9.5 | 22.4 | 28.9 |
| queue_wait | 0.14 | 0.34 | 91.3 (1 outlier) |
| cuda_sync | 0.08 | 0.13 | 0.35 |
| fork_clone | 0.41 | 2.21 | 4.78 |

Reproducible: model_wall p95 firstHalf 39.4 / secondHalf 39.0. (B,T): all **B=1, non-first**, T ∈ {44,46,49,51,53,55,56}
(matches the predicted T≈42–58 — would have been a clean, small bucket set).

## The finding: the "~178 ms server finalize" was a mis-attribution
- The client WAN TTFS P95 (401) was decomposed as 200 (trailing) + 23 (network RTT) + **178 "server finalize."**
- But the ACTUAL on-box server-side finalize compute is **~39 ms p95 model_wall** (encoder ~35), +~22 ms lock_wait
  +~3 preproc +~2 fork ≈ **~40–66 ms p95 total**, reproducible, encoder-dominated.
- MEDIAN decomposes cleanly: client 274 ≈ 200 + 23 + ~48 server. ✓
- P95 does NOT: client 401 vs 200 + 23 + ~66 server ≈ 289 → a **~110 ms gap that is NOT server compute** — it is
  network / control-path / client-side WAN tail (TCP/WS jitter, the client's conc-10 scheduling over the
  cross-country path). The ~178 ms conflated ~66 ms of real server finalize with ~110 ms of network/client tail.

## Gate verdict: STOP — do not build the finalize-graph subsystem (Steps 2–7)
- A finalize encoder graph would cut the eager encoder (~35 ms p95) toward the steady-graph per-call cost — a
  server-side saving of maybe ~20 ms.
- That moves the CLIENT P95 by ~20 ms (401→~381) — it does NOT close the ~100 ms gap to Deepgram (298), because
  that gap is **network/client-side, not server compute.** The server finalize is already fast.
- Plan's business-payoff gate (reproducible ≥60–80 ms P95, or a robustness win): **NOT met.** Stop at the probe.
- (The ~20 ms server saving would help the MEDIAN — already frontier-competitive at 274 — not the stated P95 goal.)

## The real lever (pivot)
- The client P95 gap to the frontier is **network-bound** (this client → a single us-west-2 region, over WAN, at
  conc 10). Deepgram/Soniox are CDN/multi-region → nearer POPs → smaller WAN tail. The lever is **multi-region
  deployment / closer POPs**, not server compute. The server side is already fast (~40–66 ms p95 finalize).
- The 200 ms VAD window is a fixed benchmark requirement. The final-padding track is accuracy-trading and only
  affects the (already-small) server compute → not worth it.
- Speculative-only: a finalize graph MIGHT reduce multi-process+MPS finalize-contention (unmeasured here —
  single-process conc-10 only), but the finalize is a small once-per-utterance fraction and the steady graph
  already removes the dominant MPS bifurcation → low value. Revisit ONLY if a multi-process finalize-contention
  problem ever surfaces.

## CORRECTION (multi-process+MPS gate, step1b) — the single-process gate measured the WRONG regime
The single-process gate above said "server fast ~40ms, client gap must be network" — but production is K-proc+MPS,
and the WAN bench was also single-process. Re-measured under **K=4 + MPS** (L40S, 2089 finals, conc 10+16/proc,
`step1b_repro_gate_multiproc.sh`, BATCH_FINALIZE OFF):

| component | p50 | p95 | max |
|---|--:|--:|--:|
| model_wall (GPU) | 36.4 | 40.4 | 61 |  (encoder 36.0 p95 — UNCHANGED vs single-proc; does NOT bifurcate under MPS) |
| **lock_wait** | 65.7 | **94.6** | **393** |
| **queue_wait** | 0.14 | **99.6** | 226 |
| fork_clone / preproc / sync | ~1 / 2.4 / 0.1 | ~4 / 3 / 0.1 | — |

Total server-side finalize span (loadgen TTFS): **226–496 ms p95 @ 40/box, 533–701 ms @ 64/box** (vs ~66 ms
single-proc) — while the GPU stayed ~40 ms. **The tail is HOST-SIDE serialization** (lock_wait + queue_wait: a
finalize waiting behind steady inference on the per-process inference lock + scheduler queue under parallel
finalize), NOT GPU, NOT network. Confirms the user's "parallel-finalize host variation" hypothesis + Codex's
"serialized finalize host envelope" (`finalize-python-tail-analysis.md`).

**REVISED VERDICT:** drop the GPU finalize graph (encoder is fine ~40ms, stable under MPS). **PIVOT to the
host/Python path.** The gate ran with `NEMOTRON_BATCH_FINALIZE` OFF → the global-exclusive serial finalize path
(the worst case). **Cheapest first test: turn `NEMOTRON_BATCH_FINALIZE` (+`_PREPROC`) ON** (existing byte-exact
flag → pinned-lane finalize, no global lock) and re-measure the lock/queue tail. Then the deeper Python fixes
(de-dup the decoder-state clone, parallelize finalize, buffer reuse — Codex's ranked list).

## BATCH_FINALIZE on/off (step1c, same box, K=4+MPS, conc-10) — partial win
Worst-proc server-side finalize span (loadgen TTFS, no WAN): **BF=0 (global-exclusive) 447 ms p95 (spread
354-447, a straggler); BF=1 (pinned-lane) 350 ms p95 (uniform 327-350).** BF=1 cuts ~100 ms + removes the
straggler + erased a pathological 318 ms fork_clone outlier (BF0 max 319 -> BF1 max 5). Byte-exact, existing flag.
BUT not the full fix: `lock_wait` p95 ~86 ms BOTH ways (pinned-lane did NOT eliminate it — the finalize still waits
for its pinned lane, which is busy with steady inference at this load). The loadgen span (~350) is much larger than
any single named component, BUT the per-COMPONENT p95s aren't additive — I can't validly subtract them to claim a
fixed "un-instrumented gap." NEEDS a follow-up: the per-FINAL total instrumented span (sum the components per
final, then p95) vs the loadgen ttfs p95 — to learn whether the 350 is dominated by `lock_wait` (lane contention)
or by an un-instrumented host edge (debounce / scheduler-pickup / emit / asyncio wakeup — the un-instrumented
edges). (Per-final model_wall/encoder rise under BF=1 because batching makes B>1 — confounded; the loadgen span is
the clean outcome signal.) Production `deploy/launch_multiproc.sh` already sets BATCH_FINALIZE (not `_PREPROC`).

**The 401/178 target was DOUBLY unrepresentative of production: single-process AND BATCH_FINALIZE-off.** Production
is K-proc + MPS + BATCH_FINALIZE-on; under that density (40-64/box) the finalize is host-bound and ~350 ms p95
server-side (much larger than the single-process bench). Next levers (host-side, byte-exact; GPU graph stays
dropped): (1) compute the per-final TOTAL span + extend the telemetry to the un-instrumented edges (debounce /
scheduler-pickup / emit) to LOCALIZE the 350 ms — is it `lock_wait`/lane-contention or a host/asyncio edge?
(2) reduce finalize-vs-steady lane contention (`lock_wait` ~86); (3) add `_PREPROC` to prod; (4) a clean
production-config (K=4 + BATCH_FINALIZE + LB) WAN bench for the true client number.

## DECISIVE (step1c free decomposition): the finalize is TWO halves, we instrumented ONE
Per-final decomposition of the step1c records (finalize_wall ≈ fork_flush; other components nested INSIDE it —
the residual was negative, confirming nesting):

| half | what | p95 (K=4+MPS) | instrumented |
|---|---|--:|:--:|
| **COMPUTE** (`finalize_wall`/`fork_flush`) | `lock_wait` ~87 + GPU model ~38, nested | **~130 ms** (terminal max 138, tight) | yes |
| **TRIGGER** (`vad_stop` → finalize-START) | loadgen vad_stop→final (~340) − finalize_wall (~130) | **~210 ms** | **NO — dark** |

1. The COMPUTE half is **`lock_wait`-dominated (~87 p95)** = lane contention (finalize waits for its inference lane,
   busy with steady), NOT GPU (model ~38). GPU/graph out for the 3rd time; compute lever = host-side lock contention.
2. **BF=1 is a tail-OUTLIER fix, not a bulk win**: `finalize_wall` p95 130 BOTH ways; BF only erased a pathological
   318 ms `fork_clone` spike + capped `fork_flush` max 436→313. Each utterance emits 2 records
   (`reset_then_debounce` emitted=False + `close` emitted=True/delta>0); terminal/emitted both p95 ~130. Keep BF
   (+`_PREPROC`) for the outlier cap, but it is NOT the lever.
3. **The bigger half (~210 ms) is the un-instrumented TRIGGER latency** (`vad_stop`→finalize-start: reset / debounce
   / scheduler-pickup). NOT a loadgen artifact — the REAL WAN bench's "178 ms server finalize" decomposes as
   ~66 ms compute + **~112 ms trigger** even single-process; it grows ~112→~210 ms under K=4. `FINALIZE_PROFILE`
   only instrumented the compute half, so the plan's "178 ms server finalize" was itself a conflation.

**NEXT PROBE (cheap, decisive):** add ONE field to `FINALIZE_PROFILE` — the `vad_stop`-received → finalize-start
delta — and re-run step1c (K=4). Localizes the dark ~210 ms (reset vs debounce vs scheduler-pickup). The dominant,
growing half of the client finalize is host-side and currently dark — fix THAT, not the (already-fast) compute.
Then a production-config (K + BATCH_FINALIZE + LB, real client) WAN bench for the true client TTFS.

## Value of the probe (why this is a success, not a failure)
The probe-first plan (gate Steps 1–2 before building Steps 3–7) + the **reproducibility gate added in review R5**
caught this for ~1 instrumentation pass + ~1 cloud run (~30 min, ~$2) — instead of building a 7-step finalize-graph
subsystem that would have moved the client P95 by ~20 ms (not the ~100 ms needed). The 5 rounds of adversarial
review — specifically R5's premise "the 178 ms may be network/control-path, not pure server compute" — directly
prevented the wasted build. The instrumentation (`NEMOTRON_FINALIZE_PROFILE`, default-off) stays as reusable
finalize profiling.
