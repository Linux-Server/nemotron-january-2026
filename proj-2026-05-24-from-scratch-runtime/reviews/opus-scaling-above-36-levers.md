# L40S Native-Runtime Density Knee — Skeptical Architecture Review (N=36 → ?)

**Author:** independent Opus 4.7 subagent (max thinking) | **Date:** 2026-05-27
**Adversarial-paired with** `codex-scaling-above-36-levers.md`. Folded into `checkpoint-notes-scaling-above-36.md`.
**Evidence base:** `density_main.cpp`, `session_main.cpp`, libtorch 2.8 `model_container.h`/`model.h`, `w3_run13.log` (synchronized), `w3_run14.log` + `rerun_logs/*T154255Z` (staggered fresh-process), `spy_remeasure.json`/`spy_high.json` (same-box Python re-measure), `server.log`.

> **Headline finding that reframes the whole question:** the same-box Python re-measure (`spy_*.json`) under the native runtime's *exact* gate (ttfs_p95≤175 ∧ lag_p95<500) gives a **Python SLO-robust knee ~20** (level-20 ttfs_p95=158ms passes; level-24=248ms fails). So the native multiplier is **36/20 = 1.8× — exactly at the pre-registered `max(34, 1.80·S_py)=36` bar, ZERO margin.** Pushing the knee above 36 is therefore not a nice-to-have; **it is what protects the GO from collapsing once the Step-4 WS/scheduler haircut (~17%) is applied** (0.83·36 ≈ 30 realized vs Step-4 bar 1.5·20=30 → at-bar again, razor-thin).

---

## A. RESOLVE THE PARADOX — what serializes and leaves 17–27% GPU idle

**The GPU is idle not because dispatch is locked, but because the host worker threads keep *blocking on the GPU* (D2H `.item()` syncs) and on *each other* (the enc_first mutex + the 2-runner finalize pool), in synchronized bursts.** The idle is the GPU waiting for serialized host work to issue the next launch.

Kill the wrong hypothesis first: the "AOTI execution lock = GIL" theory is **false**, confirmed in code — `AOTInductorModelContainer::run` takes `model_exec_mutex_` as a **`std::shared_lock`** (`model_container.h:82`); concurrent `run()` share it, the exclusive lock is only for weight-swap. So the knee is *not* AOTI-internal dispatch serialization.

The real serialization is four mechanisms, each tied to a telemetry signal:

**(1) Per-token `.item()` host-syncs in the decode loop — THE dominant knee mechanism.** `decode_range_density` (`density_main.cpp:842-871`) runs, per encoder frame, up to `MAX_SYMBOLS` iterations of joint.forward → `argmax().item()` (`:861`, `argmax_item_timed` `:835-840`) → predict.forward. Each `.item()` is a blocking D2H that drains that worker's stream under its `CUDAStreamGuard` (`:1236`). Per-stream — but its wall cost scales with GPU contention (the host can't get argmax until the joint kernel finishes, now queued behind 35 other streams' steady work):

| N | decode_wall p95 (sync) | decode_wall p95 (10s-stagger) |
|---:|---:|---:|
| 32 | 11 ms | — |
| 36 | 17 ms | **9 ms** |
| 40 | **309 ms** | **253 ms** |
| 44 | 455 ms | — |

`decode_wall` explodes 17→309ms across the knee while `decode_tokens` stays flat (p50=1) and `aoti_run_cuda` p95 barely moves (37→39ms). **This is the GPU-idle generator:** each blocked `.item()` is a window where that worker issues nothing; 36–40 workers ping-ponging host↔GPU on tiny launches can't keep the SMs fed → 73–83% util.

**(2) The shared+locked enc_first mutex — synchronized-burst amplifier (hurts `lag`, not `ttfs`).** Every session's first chunk takes `enc_first.mutex` (`:892`, `run_first_encoder_locked_density`). `enc_first_lock_wait` p95 grows 568→788ms. It's in the *streaming* phase → loads `lag` (keep-up), **not TTFS** (TTFS starts at `vad_stop`, `:3295`). Stagger is decisive: at N=36 stagger drops lock_wait p95 **640→10ms** → **~entirely a synchronized-arrival artifact**.

**(3) The 2-runner finalize pool — synchronized end-of-session pile-up.** `finalize_num_runners = min(N, 2)` (`:1211-1213`). At N=36, 36 sessions hit their VAD deadline in lockstep and contend for **2** runners; pool exhaustion → `get_available_model` → `reclaim_finished_models` → `wait_for_completion()` blocks the host on a CUDA event (`model_container.h:435-517`, `model.h:441-463`). The wait is **mis-attributed** into `run_aoti_loader`'s timed region, so `finalize_runner_wait_ms` reads 0 (`:1358`) — "finalize_wait=0" does **not** mean no finalize contention. Stagger: finalize_total p95 91→42ms at N=36.

**(4) cudaEventSynchronize + enc_len `.item()` in finalize** (`:1312`, `:1290`, enc_len_sync ~23ms p95). The encoder forward inflates 7→28ms (p50) from kernel-scheduling contention but its p95 *plateaus* ~38ms — near a compute ceiling but not the thing that explodes at the knee.

**Synthesis — two regimes (conflating them is the central confusion):**
- **N ≤ ~38 (synchronized-burst):** knee set by (2)+(3)+(1)-burst — **artifacts of the harness's lockstep arrivals**. Stagger moves them away (N=36 staggered: ttfs_p95=42ms, lag=−119ms — enormous headroom).
- **N ≥ 40 (genuine saturation):** decode_wall explodes to 253ms **even with stagger**. GPU GEMM-scheduling capacity (steady + finalize encoders + per-token joint kernels) is genuinely exceeded; host syncs amplify nonlinearly. **The true ceiling sits between 36 and 40.**

The 17–27% idle is **launch/sync-bound, not bandwidth-bound** — the "dispatch/host-bound residual" the project bet on. The lever that unlocks it is **removing host syncs from the per-token critical path**, not making the encoder faster.

---

## B. RANKED LEVERS (the candidate list is mis-ranked — evidence inverts it)

| Rank | Lever | Touches true ceiling (≥40)? | Production-real? | Est. knee gain | Cost |
|---|---|---|---|---|---|
| **1** | Decode/enc_len **sync removal** (device argmax + host-computed enc_len) | **YES** (only one) | YES (every turn) | **36→40–44** | med (T1 risk) |
| 2 | Cross-stream **batched greedy decode** (Part C.i) | YES | YES | 40→? (biggest if it transfers) | high (Phase-3) |
| 3 | Steady-encoder **CUDA-graph** (C.ii) | partial | YES | 0–2 | med |
| 4 | Steady faster pkg (autotune-ON) | floor only | YES | 0–2 | high, T1-blocked |
| 5 | enc_first **K-pool** | **NO** | **NO (harness artifact)** | 0 | 6.9 GiB |

**Lever 1 (NEW, was missing) — kill per-token decode `.item()` syncs. Expected N=36→40–44; the only lever touching the true ceiling, and production-real.** decode_wall is what explodes at the knee (A.1) and survives stagger (the true wall). Fixes cheapest-first: (a) device-side greedy argmax + a single fused device→host blank flag per step (or speculative window) — cuts sync count 3–5×; (b) `enc_len` is geometry-deterministic from `(drop,T)` (`PHASE2-PLAN.md:292`) → host-compute it, delete `:1290 .item()`. Quantified: N=40 staggered finalize_total ≈ decode-dominated 253ms; cutting sync amplification 2× → ~150ms, under the 175 gate → staggered knee 36→**40, plausibly 44**. **Assumption (= the kill-gate): argmax/blank can go device-side without breaking T1 token-exactness.**

**Lever 2 — steady-encoder contention. Small for ≤40; necessary not sufficient above 44.** steady_gpu p50 7→28ms is real GPU-time, but p95 plateaus ~38ms and does not explode at the knee. Don't lead with it.

**Lever 3 (was #1!) — enc_first lock / TorchScript K-pool. ~0 streams for PRODUCTION; a harness/lag fix only.** Down-rank hardest: (1) loads `lag` not `ttfs`; (2) stagger nearly eliminates it (640→10ms); (3) **production sessions are long-lived multi-turn** — enc_first fires once per session start, but the harness uses ~12s single-utterance sessions (256/97.5s) making the lock look ~30× more stressed than reality. A K=4 pool would "fix" a number production won't see. **Do not spend 6.9 GiB on it as a density lever** (keep only as cheap lag insurance if a staggered re-measure ever shows lock_wait on the keep-up critical path).

---

## C. WHAT'S MISSING

**(i) Cross-stream BATCHING of decode/steady — the genuinely biggest theoretical win, absent from the list.** Current topology = **36 independent B=1 forwards**; the Python self-host server *already ships continuous batching* (`server.log:655,663`), the native runtime threw it away (`make_worker_context` = per-worker everything, B=1). Transfers to one-process-N-threads as the natural Step-2/3 scheduler change, with caveats: **steady** is BW-bound (38ms plateau) → batching a BW-bound GEMM gives sub-linear gain + adds batch-collection latency. **Decode** is where it's large — collapsing 36 streams' per-token joint GEMMs into one B=36 forward eliminates ~35/36 launches AND lets one `.item()` serve 36 streams (amortizes the host sync that *is* the knee mechanism). But greedy RNN-T is **ragged** (per-stream blank steps) → needs a batched-greedy decoder with active masks = exactly the deployed Python `greedy_batch` LABEL-looping. **Phase-3-scale build, not a tweak.** Biggest upside in principle, but gated on (a) proving cheap Lever-1 isn't already enough, (b) a batched-greedy correctness proof. **Do Lever 1 first.**

**(ii) Steady-encoder CUDA-graph — plausibly worth it; the finalize-graph precedent is direct.** Python finalize-graph collapsed ~1376 launches/finalize and compounded in the serialized queue. Steady runs every 160ms on every stream and is a launch-heavy conformer; per-thread CUDA-graph-of-steady-AOTI is the project's named de-risked primitive (`PHASE2-PLAN.md:32`). Helps launch/dispatch, not GEMM time (BW-bound) → medium priority, below sync-removal.

**(iii) C++ inference-dispatch serialization — checked, mostly clean, two real items.** `g_aoti_run_mutex` (`:745`) only under `--mutex-serialize-run` (off in the sweep). **The 2-runner finalize pool (A.3) IS real C++ serialization** — raising `finalize_num_runners` past 2 is **memory-free** (buckets share ONE 2.3 GiB constants set, `loader_delta=0/bucket`, `:1184`) → near-free fix for the synchronized finalize pile-up (caveat: more finalize runners compete with steady for SMs → measure net). `get_available_model`'s `models_mutex_` is held only for O(1) pop/reclaim — not a concern.

---

## D. PLAN — extend PHASE2-PLAN.md (do NOT spawn Phase-3)

These levers ARE what Step-2 (scheduler/admission/priority-finalize-lane) and Step-3 (multi-session runtime) are for; a separate Phase-3 duplicates them. But the plan's lever inventory is **stale** (still ranks enc_first DEDUP Tier-1 — correct for the resolved 5090 *memory* regime; lists cross-stream batching only as a parenthetical W5). Amendments:
- **Progress table 1b + Step-4:** record same-box Python re-measure DONE → **S_py≈20 → 1.8× at-bar, zero margin** → Step-4 GO at risk from the 0.83 haircut. Make "push knee >36" a **named precondition for a robust Step-4 GO**.
- **Rewrite lever tiers (L40S, post-W3):** Tier-1 decode host-sync removal (NEW); Tier-1b finalize runner count + priority lane (memory-free); Tier-2 cross-stream batched greedy decode (Step-3, build only if Tier-1 misses); Tier-3 steady CUDA-graph + autotune (launch not GEMM); **de-prioritize enc_first K-pool** (harness/lag artifact).
- **New Step 1c** (decode-sync ablation + staggered knee re-pin, T1-gated) before Step-2; **new Step 1d** (staggered admission re-pin, gated on 1c).

> **[ ] Step 1c — decode-sync ablation + staggered knee re-pin (5090 + short L40S confirm).** Flag two changes: (a) `enc_len` host-computed from `(drop,T)` (delete `:1290 .item()`); (b) device-side argmax + single fused blank-flag read per step (replace `:861`). **MANDATORY T1 re-validation binds to the exact build** — 1000/1000 finals byte-exact + strict events; any token flip → STOP (retreat to host-argmax-only). Measure staggered N=36/40/44 + sync-on control. **PASS-to-Step-2 = staggered knee ≥40 with decode_wall p95<50ms at N=40 AND 0 token mismatch.** STOP-candidate = knee+decode_wall unchanged (sync wasn't binding → pivot to Tier-2). PAIRED REVIEW (it changes gate math).

---

## E. CHEAPEST DECISIVE KILL-GATE (Lever #1, ~2h, 5090-local — no g6e spend)

Rebuild `density_main` with `--decode-no-host-sync` (host-computed enc_len + device argmax/blank). Run `--mode density-sweep --n-values 36,40,44 --density-start-stagger-ms 10000` on the 5090 + the flag-off paired control; capture `finalize_phases.decode_wall`, ttfs/lag, and the serial-oracle T1 check. Arch-independent (5090 shows the same decode-sync structure), reuses harness/telemetry/T1-oracle.
- **GO (wire into Step-2 + L40S confirm):** staggered N=40 `decode_wall` p95 **<50ms** AND ttfs p95 **<175ms** AND **0 token mismatch**.
- **STOP/PIVOT:** decode_wall p95 at N=40 stays **>150ms** with syncs removed → binding is GEMM-scheduling, not host sync → lever dead → pivot to Tier-2 batched decode; accept the single-process knee may be near-pinned ~38.

## Caveats
- Exact knee ∈(36,40) unpinned (37–39 untested). "→40–44" assumes the staggered N=40 collapse is decode-sync-dominated (strongly supported by decode_wall, causally proven only by Step 1c).
- Python S_py≈20 is a 2-repeat sweep (noisy: level-16 fails 147ms, level-18 passes 67ms), no p99 (gated p95∧lag). Direction (top of 16–20 band → 1.8× at-bar) is robust.
- All numbers parsed from committed logs (nothing re-run). Batched-decode upside (C.i) is an architectural argument from the Python `greedy_batch` precedent, not measured in native — correctly a Step-3 bet.
