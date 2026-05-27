# RED-TEAM REVIEW — PHASE2-PLAN.md Step 1c / scaling-above-36 plan (Opus 4.7, independent)

**Date:** 2026-05-27 | Adversarial-paired with `codex-goforward-plan-review.md`. Folded → `checkpoint-goforward-verdict` section below / plan amendments.

## OVERALL VERDICT: **GO-WITH-CHANGES**

Directionally sound and unusually honest (real-decode, SLO-robust, stagger-robust, paired, pre-registered). But it makes "push knee >36" load-bearing for the Step-4 GO on a **single noisy point estimate (S_py≈20)** that is (a) partly justified by a factual error in the fold, (b) **not apples-to-apples** with the native number, and (c) cheaper to tighten than to engineer around. It sequences the expensive arbiter (1c-0 → maybe 1c-A batching) *before* nailing the cheap measurement that decides whether any of it is needed. Two load-bearing quantitative claims — the **44-48 util ceiling** and the **"genuine saturation ≥40"** diagnosis — do not survive scrutiny of the telemetry they cite. All fixable with re-sequencing + threshold corrections before spending on 1c.

## WHAT IT GOT RIGHT
- AOTI-exec-lock kill is code-verified (`model_container.h:103` `std::shared_lock`). enc_first demotion is well-supported by the stagger data. Tier-1b "free" is verified (`density_main.cpp:1212` `min(N,2)`; `loader_delta=0`). "Extend, don't Phase-3" is right. The finalize-wait mis-attribution catch (`:1354-1368`) is real. Batching B-fill skepticism is honest.

## MUST-FIX

### MF-1 — TIGHTEN S_py BEFORE committing to 1c (cheaper than the arbiter; can dissolve the problem)
- **Factual error in the fold:** CHECKPOINT line 9 + `opus-scaling-above-36-levers.md:91` say *"level-16 fails 147ms."* 147 < 175 = **PASS**. First failing level is **24** (248ms). The series is grossly **non-monotonic** and with `repeats=2` **1-2 samples set the knee** — a coin-flip, not a capacity wall.
- **Not apples-to-apples (cuts AGAINST the project):** Python spy ttfs is **client-observed over WS** (`ec2_loadgen.py:77`, `now − vad_stop_t`); native N=36 ttfs is **server-side, pre-WS** (Step 3 not built). So `36(native,no-WS)/20(Python,with-WS)` flatters native by exactly the WS slice the 17% haircut is meant to cover. **Honest WS-paid multiplier < 1.8×**; the plan doesn't acknowledge the direction.
- **Which way it cuts → measure first.** Native knee needed to clear Step-4 (`1.5·S_py/0.83`): S_py=18→**33** (36 clears → 1c unnecessary); 20→36 (at-bar); 22→40; 24→**43**. A 2h Python re-sweep (`repeats≥10`, fine levels {18,20,22,24}, + the in-flight p99) is far cheaper than 1c-0→1c-A.
- **Proposed: Step 1c-(−1) "S_py re-pin", BLOCKING.** WS-matched (or WS-subtracted) `ec2_loadgen.py`, `repeats≥10`, levels {16,18,20,22,24}, report S_py as a CI. **≤18 → 1c DEMOTED to optional; [19,21] → proceed to 1c-0; ≥22 → trigger MF-2 funding re-confirm before any g6e/Phase-3 spend.**

### MF-2 — explicit FUNDING re-justification checkpoint at 1.8× (not 2-2.25×)
The bet was greenlit on a hoped 2-2.25×; it landed 1.8× at-bar, and MF-1 likely pushes the WS-paid multiplier below that. "Technical GO ≠ funding GO" (line 237) has **no trigger** to revisit now that the headline softened. **Proposed:** a named "Funding re-confirm" gate fired by MF-1, BEFORE 1c-A's batching build (the first big spend): if (post-WS multiplier < 1.8×) OR (S_py≥22 → need knee ≥43), escalate to the human funding call with corrected numbers before building.

### MF-3 — the 44-48 "util-bound ceiling" is a statistical fallacy; correct or strike
"73% mean util → ~49 streams" treats the 27% idle as linearly reclaimable. Telemetry: at N=36 **mean 73% but p50=91%, p95=97%** — bimodal; GPU is 91-97% busy in active windows, idle in correlated sync troughs. **Direct refutation:** staggered N=40 **fails** (decode_wall 253ms) at **72.6% util** — *more* idle than N=36, yet collapses. Util headroom ≠ stream headroom. Realistic ceiling plausibly **<44**. **Proposed:** replace with "ceiling unknown; mean-util extrapolation invalid (active-window util 91-97%); 1c-B Nsight is the only valid attribution." No GO threshold may inherit 44-48 as established.

### MF-4 — "genuine saturation ≥40" rests on ONE staggered point + is internally contradictory
Rests on a single staggered N=40 point (no staggered N=44 — empty ROW; no 37/38/39). That point is more consistent with **sync/queueing-latency-under-contention** (each tiny joint kernel's time-to-result inflates behind 40 streams; steady_gpu p95 flat ~38ms, util 72.6%) than compute-saturation — the same evidence the fold *also* labeled "host-sync-bound." **Mis-routing risk:** 1c-0's STOP (decode_wall>150 → "compute-bound → batching") conflates "sync removal didn't help" with "compute-bound"; a third cause — **cross-stream kernel-queue contention** — produces the same decode_wall signal and is fixed by *neither* cheap-sync nor batching alone. **Proposed:** 1c-0 GO/STOP must read a CONTENTION discriminator (1c-B Nsight SM/DRAM occupancy) ALONGSIDE, not optional. Re-tier routing: (a) decode_wall<50 → sync-bound, cheap win; (b) >150 AND SM/DRAM<85% with launch gaps≥15% → **kernel-queue-contention → steady-graph/coalescing (Tier-3), NOT batching**; (c) >150 AND SM/DRAM≥90% → compute-bound → batching.

### MF-5 — add an explicit STOP / "good-enough" terminus to Step 1c
1c has inter-sub-step gates but no top-level "ship it" exit → can recurse (1c-0 PIVOT→1c-A→weak-B→Tier-3→…), the treadmill, financed by the soft headline. **Proposed:** 1c STOPS (success) when a knee clearing `max(34, 1.5·S_py/0.83)` is demonstrated **stagger-robust at the knee N and N+4**, OR after 1c-0+1c-B if no cheap (≤Tier-1b) lever clears it → escalate to funding (MF-2), NOT auto-continue. Phase-3 batching needs its own explicit GO.

## ADDITIONAL CAVEATS
- **The arbiter may not run clean.** The in-flight N=16 rerun failed at setup on a serial-oracle token mismatch (utt198) — the T1 oracle is fragile. 1c-0 needs device-argmax **bit-exact** vs CPU `.argmax().item()`, but tie-breaking isn't guaranteed identical device-vs-CPU. **Pre-register: prove device-argmax T1-exactness on a fixture FIRST (hours) as a 1c-0 precondition;** if it fails, 1c-0 collapses to enc_len-only (conceded +0-2 streams ≈ no-op) → route straight to 1c-B + batching.
- **5090-as-proxy:** make the L40S confirm a REQUIRED gate-completion for 1c-0, not a footnote.
- **Step 1c BLOCKING Step-2 is over-serialized:** Step-2 admission/backlog/priority-lane DESIGN doesn't depend on knee=36-vs-44; only the batched-vs-not topology (1c-A) feeds it. Let Step-2 design proceed in parallel; gate only its final topology on 1c-A.
- **enc_first resurrection:** demotion is right for the harness's 12s sessions, but real traffic (barge-in / reconnects / short first turns / multi-stream-per-call) re-stresses the first-chunk path. The production session-length distribution is an untested assumption underpinning the demotion — validate in Step-4's multi-turn subcurve.

## BIGGEST RISK + ONE CHANGE
**Risk:** an entire workstream (1c + possible Phase-3 batching) is load-bearing on a **noisy, not-apples-to-apples single point (S_py≈20)**, and the plan spends on the arbiter/batching *before* nailing it. ≤18 → unnecessary; ≥22 → its own (over-estimated) ceiling can't save the GO.
**One change:** make a **hardened, apples-to-apples S_py re-measurement (MF-1) the blocking first gate of Step 1c** — `repeats≥10`, WS-matched, CI + p99, with the three pre-registered branches. Hours of compute (half already in-flight) that converts the plan's most expensive uncertainty from an assumption into a measurement.
