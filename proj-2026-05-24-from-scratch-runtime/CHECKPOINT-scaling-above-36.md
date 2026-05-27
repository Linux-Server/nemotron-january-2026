# Checkpoint notes — scaling the L40S native runtime above the N=36 knee

**Date:** 2026-05-27 | **Folds:** `reviews/opus-scaling-above-36-levers.md` (independent Opus 4.7, max-thinking) + `reviews/codex-scaling-above-36-levers.md` (independent Codex) — two independent analyses of the same telemetry + source, adversarially paired. This doc reconciles them and pre-registers the next move.

---

## 0. The strategic frame (why this matters now)

The same-box Python re-measure (today, `runtime/artifacts/l40s_w3_logs/spy_*.json`) pins **S_py ≈ 20** under the native runtime's *exact* gate → the native multiplier is **36/20 = 1.8×, exactly at the `max(34, 1.80·S_py)=36` bar with ZERO margin** (Opus computed this independently from the json). The 1.80× was *designed* to bank the ~17% Step-4 WS/scheduler haircut — but at S_py=20 (top of the 16–20 band), 0.83·36 ≈ 30 vs the Step-4 bar 1.5·20 = 30 → **at-bar again, razor-thin**.

**⇒ Pushing the knee above 36 is not a nice-to-have; it is what protects the Step-4 GO from collapsing.** This elevates everything below from "future optimization" to "load-bearing for the funding decision."

**The ceiling is bounded, though.** Codex's util-bound math: N=36 @ 73% util → a perfect-fill ceiling of ~49 streams (~44 at 90% fill); N=40 @ 75% → ~48 at 90%. So **the realistic prize from dispatch/sync levers is ~44–48 streams**; anything above ~48 needs *reduced per-stream compute*, not just smoother dispatch (`w3_run13.log:317,411,505`).

---

## 1. CONSENSUS — what both analyses agree on (high confidence)

1. **The paradox cause is host-side serialization, not the GPU.** The knee is set by **36–40 independent B=1 forwards punctuated by blocking scalar `.item()` D2H syncs**, leaving the GPU 17–27% idle. It is **NOT** memory (0.035 GiB/stream), **NOT** a saturated GPU, **NOT** an AOTI dispatch lock (Opus confirmed `model_exec_mutex_` is a `std::shared_lock`, `model_container.h:82` — concurrent runs share it), and **NOT** primarily `enc_first`.
2. **Two regimes (don't conflate them):** N ≤ ~38 = *synchronized-burst harness artifacts* (enc_first lock, 2-runner finalize pool — **stagger erases them**: lock_wait 640→10ms, finalize_total 91→42ms at N=36); N ≥ 40 = *genuine ceiling* (`decode_wall` p95 explodes to ~253ms **even staggered**). The true knee sits between 36 and 40.
3. **`finalize_total` drives final TTFS, and `decode_wall` drives `finalize_total`** (91→361→484ms at N=36/40/44). The "finalize-runner-wait p95 = 0" is a **mis-attribution** — the pool-exhaustion `wait_for_completion` lands inside the AOTI-timed region (`density_main.cpp:1354-1368`); it does *not* mean no finalize contention.
4. **`enc_first` is DEMOTED to last** (both ranked it #4). It's `lag`-not-`ttfs`, stagger-erased, and fires *once per session start* — the harness's ~12s/8-per-worker sessions over-represent first-chunk churn ~30× vs long-lived multi-turn production. A K=4 TorchScript pool costs ~6.9 GiB for **~0 production density gain**. Keep it only as short-session-churn/tail hygiene.
5. **Plan: extend PHASE2-PLAN.md with a Step 1c, do NOT spawn Phase-3.** These are core density-ceiling questions that decide what the Step-2 scheduler must schedule. Phase-3 starts only if Step 1c selects a big architectural change (B>1 batching / steady graph pools).
6. **The telemetry can't fully prove launch-bound vs compute-bound** — both flag that NVML util is insufficient and the plan-required Nsight/CUPTI counters (launch gaps, SM occupancy, DRAM BW) are missing (`PHASE2-PLAN.md:225-232`). Need an attribution trace.
7. **`enc_len` is geometry-deterministic** and its D2H `.item()` (`density_main.cpp:873-884,1289-1307`) can be host-computed — but **both agree enc_len-alone is only +0–2 streams** (at N=40 enc_len_sync is 22ms of a 361ms finalize_total). Not sufficient by itself.

---

## 2. THE PRODUCTIVE DISAGREEMENT — what's the #1 lever?

| | **Opus** | **Codex** |
|---|---|---|
| #1 | **Decode `.item()` sync removal** (device-side argmax + fused blank flag) — cheap, attacks the term that explodes (`decode_wall`) and survives stagger | **Cross-stream batching** — attacks the root B=1 topology directly |
| batching | #2, "Phase-3-scale; do only if #1 misses" | #1, but **B-fill-skeptical** (`spikes/0.5-batching-sim`: realistic mean B≈1.5–2.1, B=1 36–63%; old 3–5× claim "effectively dead" → a **20–35% push, not a rewrite**) |
| decode-sync | bullish: removing it should unlock 40–44 | **skeptical it helps alone**: the `.item()` wait is *also* the multi-thread fill window (`PHASE2-PLAN.md:275-280`) — removing one stream's sync may just shift the stall; **net density must be measured, not inferred** |

**The crux:** at N=40, is the idle (a) *sync-bound* — correlated `.item()` stalls collectively idle the GPU, so removing them fills it [Opus], or (b) *compute-bound* — the B=1 GEMM-scheduling genuinely saturates and the syncs are already hidden by other streams, so only batching (less per-stream work) helps [Codex]?

**This is empirically decidable and cheap to settle** — and it's the same test both proposed. So the disagreement is really about *sequencing*, and it resolves in favor of running the cheap arbiter first.

---

## 3. RECONCILED LEVER RANKING

| Tier | Lever | Touches true ceiling (≥40)? | Production-real? | Est. gain | Cost | Gated by |
|---|---|---|---|---|---|---|
| **1a** | **Decode/enc_len scalar-sync removal** (device argmax + host enc_len) | tests it | YES (every turn) | 36→40–44 *if sync-bound* | low (T1 risk) | the kill-gate below |
| **1b** | **`finalize_num_runners` > 2 + priority lane** | partial (burst) | YES | synchronized-burst relief | **~free** (buckets share one constants set, `density_main.cpp:1184`) | measure SM competition |
| **2** | **Cross-stream batched greedy decode/steady** | YES (biggest) | YES | 40→44–48 *if B-fill real* | high (Phase-3, B>1 export + ragged batched decode + T1) | BATCH-0 opportunity trace |
| **3** | **Steady CUDA-graph** (the shipped finalize-graph primitive) | partial | YES | 0–4 *if launch-bound* | med | Nsight launch-gap trace |
| **3** | autotune-ON steady pkg | floor only | YES | 0–2 | high, **T1-blocked** | T1 precision ladder |
| **last** | **`enc_first` K-pool / AOTI fold** | NO | **NO (harness/lag artifact)** | ~0 prod | 6.9 GiB | only if high-churn product |

**Ceiling reality (Codex):** these dispatch/sync levers top out ~44–48; >48 requires reduced per-stream compute.
**Batching reality (Codex):** B-fill is the make-or-break — the sim says mean B≈1.5–2 at an 8ms window, so batching is a 20–35% push contingent on the opportunity trace, not an assumed 2–3×.

---

## 4. THE DECISIVE NEXT MOVE (pre-registered)

Run the **cheapest arbiter first** — it tells us whether the cheap fix (Tier-1a) suffices or whether we must commit to the expensive batching build (Tier-2). Both analysts independently landed on this test.

### Step 1c-0 (FIRST, ~2h, 5090-local, no g6e spend) — decode-sync ablation
Rebuild `density_main` with `--decode-no-host-sync`: (a) host-compute `enc_len` from `(drop,T)` (delete `:1290`/`:884` `.item()`); (b) device-side greedy argmax + a single fused device→host blank flag per step (replace `argmax_item_timed` `:835-861`). Run `--mode density-sweep --n-values 36,40,44 --density-start-stagger-ms 10000` + the flag-off paired control. **MANDATORY: T1 binds to the exact build** — 1000/1000 finals byte-exact + strict events; any token flip → STOP this lever (retreat to host-argmax-only / enc_len-only).
- **GO (cheap fix wins — wire into Step-2, skip the batching build):** staggered N=40 `decode_wall` p95 **< 50ms** AND ttfs p95 **< 175ms** AND 0 token mismatch. ⇒ it *was* sync-bound; knee → 40–44.
- **STOP/PIVOT (need batching):** `decode_wall` p95 at N=40 stays **> 150ms** with syncs removed. ⇒ it's GEMM-scheduling/B=1, not host sync → go to Step 1c-A.

### Step 1c-A (only if 1c-0 PIVOTs) — batching kill-gate (Codex's BATCH-0, model-free first)
1. **Opportunity trace, no model changes:** replay density-harness / recorded production arrivals → ready-timestamps + batch keys; simulate 8ms & 12ms windows. **GO** if median B ≥ 2.5, p95 B ≥ 4, B=1 ≤ 35%, added-wait p95 ≤ 8ms at N=36–44. **STOP batching** if median B < 2 or B=1 > 50% (the workload won't fill batches → batching is dead for this traffic).
2. **Batched steady fixture microbench:** compile B=2/B=4 steady AOTI for the exact geometry, pack independent caches, shadow-compare batched-vs-alone. **GO** if B=4 per-row ≤ 0.75× B=1, B=2 ≤ 0.85×, 0 token/cache/event mismatch, predicted N=44 holds lag p95<500 & ttfs within 175/250. **STOP** on any correctness drift or per-row gain < 15%.

### Step 1c-B (parallel, attribution) — Nsight/CUPTI trace
N=36 & 40, no-stagger & stagger: launch gaps, kernel count, SM occupancy, DRAM throughput, stream overlap, AOTI host launch time + negative controls (`--mutex-serialize-run`, default-stream). **Routes the work:** idle/launch gaps ≥15% of the gate while SM/DRAM < 85–90% → steady CUDA-graph (Tier-3) is live; true SM/DRAM saturation → graph is dead, only batching/compute-reduction remains. (Fills the `PHASE2-PLAN.md:225-232` telemetry-schema gap both analysts flagged.)

---

## 5. PLAN RECOMMENDATION

**Extend `PHASE2-PLAN.md`; do not start Phase-3 yet.** Concretely:
1. **Progress table (1b) + Step-4:** record the same-box Python re-measure DONE → **S_py≈20 → 1.8× at-bar, zero margin** → Step-4 realized-GO is at risk from the 0.83 haircut. Make **"push knee >36" a named precondition for a robust Step-4 GO.**
2. **Rewrite the stale lever inventory** (it still ranks `enc_first DEDUP` Tier-1, correct only for the *resolved* 5090 memory regime; cross-stream batching is buried as a W5 parenthetical). Replace with the §3 tiers.
3. **Insert Step 1c** (§4) with the pre-registered GO/STOP tree above; it decides what Step-2's scheduler is designed to schedule. Phase-3 spins up only if 1c-A selects native B>1 batching.

---

## 6. OPEN QUESTIONS / CAVEATS (carried)
- **Exact knee ∈(36,40) still unpinned** (37–39 untested, sync & stagger). The "→40–44/44–48" numbers are *ceilings/estimates*; Step 1c-0 is what causally proves the binding mechanism.
- **S_py≈20 is a noisy 2-repeat sweep, no p99** (gated p95∧lag), and **non-monotonic**: conc 12/14/16/18/20 ttfs_p95 = 131/58/**147**/67/158ms; first FAIL (>175) is **conc-24 (249ms)**. ⚠️ **CORRECTION (paired red-team 2026-05-27):** an earlier draft said "level-16 fails 147ms" — **wrong, 147<175 = PASS**; 1–2 samples set each p95, so S_py≈20 is a coin-flip near the line, not a wall. Also **not apples-to-apples** (native ttfs is server-side pre-WS; Python is client-over-WS) → the 1.8× is inflated. **The going-forward plan was paired-red-teamed (GO-WITH-CHANGES) → `reviews/goforward-paired-verdict.md`: lock S_py first (Step 1b.5), the decode-sync lever is bounded ~17ms (batching is the real decode-contention lever), the 44–48 ceiling is an upper bound not a forecast.** The in-flight chart is the first half of the S_py lock.
- **Batching B-fill is unproven for this workload** — the whole Tier-2 upside hinges on the §4 opportunity trace; the prior sim is discouraging (mean B≈1.5–2).
- **Nsight counters are missing** — until 1c-B, launch-bound-vs-compute-bound is inferred (decode-sync ablation is a causal proxy, not a full attribution).
- All numbers parsed from committed logs; nothing was re-run for these analyses. Cross-stream-batching upside is an architectural argument from the Python `greedy_batch` precedent, not measured in native.
