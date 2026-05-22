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

## Value of the probe (why this is a success, not a failure)
The probe-first plan (gate Steps 1–2 before building Steps 3–7) + the **reproducibility gate added in review R5**
caught this for ~1 instrumentation pass + ~1 cloud run (~30 min, ~$2) — instead of building a 7-step finalize-graph
subsystem that would have moved the client P95 by ~20 ms (not the ~100 ms needed). The 5 rounds of adversarial
review — specifically R5's premise "the 178 ms may be network/control-path, not pure server compute" — directly
prevented the wasted build. The instrumentation (`NEMOTRON_FINALIZE_PROFILE`, default-off) stays as reusable
finalize profiling.
