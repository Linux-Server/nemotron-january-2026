# Opus — Phase-2 plan review, Round 4 (decision structure: numeric GO/STOP tree + goals traceability + stress the bars)

Built on Rounds 1–3. This round pre-registers the decision, traces both goals, and adversarially stresses the
thresholds. Goals: **G1 density**, **G2 P50↔P95 tail**.

## 1. Pre-registered GO/STOP tree (freeze before building Step 1)

### Step 0 — cheap kill-gates (5090, ~hours, BINARY). Run BEFORE the full harness.
| Gate | Metric | PASS | FAIL action |
|---|---|---|---|
| **K1 Overlap** | 2 threads, `num_runners=2`, real `run(inputs,stream)`; wall(2 concurrent) ÷ wall(serial) | < ~1.6× **and** profiler shows kernel interleave | branch: per-thread CUDA-graph-of-AOTI, or MPS/per-context; if all fail → STOP (corroborated) |
| **K2 Concurrency-correctness** | 2 threads replay 2 bundles; each thread's token+event == its serial result | exact | fix topology (per-thread handles); if a true race in shared weights → STOP |
| **K3 Memory-one-copy** | `num_runners=N` peak GPU mem ÷ single-loader | ≤ ~1.1× | wire codisk/`user_managed`; re-test |

All three PASS → Step 1a. (These are the cheapest decisive experiments; each is a potential early STOP that saves
the whole build.)

### Step 1a — full 5090 mini-sweep (real decode + real finalize + per-thread streams/handles + telemetry)
- **Metric:** SLO-robust streams sustained (NOT keep-up), knee **attributed to a binding resource** (BW/launch/
  lock) via counters; + reference tail P95−P50.
- **PASS to 1b:** 5090 real-decode multiplier ≥ **1.5×** (sanity floor) AND overlap is real AND correctness holds
  AND the knee is attributed. **STOP if < 1.5×** on the headroom-rich 5090 (L40S, tighter, won't clear).
- **Caveat (see stress §3):** the 5090 number is a NOISY proxy for L40S — it is a mechanism + sanity gate, **not**
  the density gate. Do not let a marginal 5090 PASS authorize skipping L40S.

### Step 1b — L40S numeric HARD GATE (same harness, EC2) — the density gate
- **G1 ceiling:** SLO-robust streams/box at the dual SLO (keep-up `lag_p95<500ms` AND `ttfs` distribution) ≥
  **1.5×** the **re-measured** L40S Python baseline (≥~28/box if baseline ~16–20). **TARGET ≥1.8×** to leave
  margin for the Step-4 realized haircut (S4 < S1b always — scheduler+WS+shed).
- **G2 reference:** `ttfs_p95−p50` reported + attributed (reference only; binding at Step 4).
- PASS G1 → fund the Step-3 build. FAIL (<1.5×) → STOP (corroborated).

### Step 4 — realized end-to-end (post Step-3 build) — the SHIP-decision inputs
- **G1 realized:** **admitted-successful** SLO-robust streams/box ≥ **1.5×** re-measured Python baseline (exclude
  shed-rejects; bounded reject rate, e.g. <5% at the knee). Two curves: no-shed knee + admitted-through-shed.
- **G2 guardrail:** native `ttfs_p95−p50` ≤ Python's at **matched** streams/box (no worse spread while denser),
  **WS-tail subtracted** + **stale-gen correctness verified** (so a tail "win" isn't a dropped-final artifact).
  Stretch = strictly tighter.
- **WER:** semantic-WER non-inferior to Python within CI at the knee (pinned model/prompt/version).
- PASS all → the funding/ship decision (human-owned, see §4).

## 2. Goal traceability (no orphans)
- **G1 (density):** Step 0 K1 (overlap exists) → Step 1a (5090 ≥1.5×, mechanism) → **Step 1b (L40S ≥1.5×,
  ceiling)** → Step 4 (realized admitted ≥1.5×). **Fully gated.**
- **G2 (tail):** Step 1 (reference P95−P50, attributed) → **Step 4 (binding `ttfs_p95−p50` ≤ Python,
  WS-subtracted, stale-gen-correct).** Gated at Step 4, referenced at Step 1.
- **ORPHAN FLAG:** the *original* pre-registered 0.0 threshold gated **density ONLY** ("≥1.5× L40S, strategic
  bet"). **G2 was never gated.** Round 4 elevates G2 to a co-equal Step-4 conjunct — this is a **CHANGE to the
  frozen threshold** and the user should ratify it (see §3.2 for the right strength).

## 3. Stress the bars (the most important part — are these the RIGHT thresholds?)

### 3.1 Is ≥1.5× the right density bar? — Yes, but it's STRATEGIC, not COGS
≥1.5× (≈28/box vs 16–20) is the user-set 0.0 threshold, justified as a "strategic capability bet, no COGS
break-even." Don't move it. But the GO weighs 1.5× against the **second-stack maintenance burden** (a native C++
runtime alongside the Python one) — that's conjunct-1, human-owned (§4). The technical bar is necessary; it is not
sufficient for "build it."

### 3.2 The G2 bar should be a GUARDRAIL, not a strict-improvement requirement
**This is the key calibration.** Python's tail is already good after the finalize-CUDA-graph win (L40S p50/p95
246/279 → spread ~33ms). So "native must be **strictly tighter**" could be **unachievable** (native carries its
own WS/scheduler tail) and would be a wrong STOP criterion. The honest bar: **G2 = NON-REGRESSION** — native must
not make the tail materially WORSE while adding density; strictly-tighter is upside, not a gate. So: **G1 ≥1.5× is
the GO gate; G2 tail-non-regression is a GUARDRAIL** (STOP only if native is materially worse on tail at matched
load). This matches the user's intent ("tightened spread") without making an over-tight bar the project-killer.
The user should confirm this strength.

### 3.3 The 5090→L40S transfer is noisy → L40S is mandatory
Mock ratios: 5090 ~3× vs L40S ~2–2.5× (ratio ~0.7–0.83). So a 5090 number near 1.5× implies L40S possibly <1.5×.
The tree must **forbid a marginal 5090 PASS from substituting for the L40S measurement**. 5090 = mechanism + floor;
L40S = the gate. (This is why Step 1a's bar is a *floor* with the real gate at 1b.)

### 3.4 False-STOP robustness — protect a viable project from a bad spike
The whole project hinges on Step 1; a STOP from a mis-built spike (wrong topology, un-wired per-thread streams,
keep-up-vs-SLO-robust confusion, the AOTI execution lock mis-attributed) would kill a viable project. **A STOP
finding must be CORROBORATED** by all of: (a) the **negative control** (mutex-serialized run) proving the harness
CAN detect serialization (i.e., the harness isn't blind); (b) the **topology sweep** (`num_runners=N`, per-thread
loaders, MPS) all failing; (c) **counter attribution** showing the binding resource is a **hardware** limit (DRAM
BW saturated), not a **software** artifact (container lock / default-stream sync / un-wired streams). Only then is
a STOP "native can't help" rather than "our harness was wrong." Pre-register this corroboration requirement.

### 3.5 False-STOP guard #2: count admitted, not offered
The loadgen's "knee = max N with zero errors" (Round 3) would count intentional shed-rejects as failures →
under-count → false STOP. Pre-register **admitted-successful streams/box** as the metric, with shed as expected
overload behavior.

### 3.6 Are there too many / too few gates?
Right number. The expensive build (Step 3 WS server) is correctly gated AFTER the cheap Step-1b L40S number — the
project can STOP cheaply at Step 0 or Step 1b before sinking the build. **One residual risk:** Step-1b ceiling
PASSes, then Step-4 realized FAILs after the Step-3 build is sunk. The ≥1.8× Step-1b margin (§1) mitigates; a
cheaper mitigation is a **"Step 3-lite"** (minimal WS echo + the admission/priority scheduler, no full feature
set) to get an early realized number before the full build. Recommend adding it.

## 4. Technical gate vs funding decision (keep separate)
The tree's leaves are TECHNICAL (≥1.5× density + tail non-regression + WER). A technical PASS is **necessary, not
sufficient**: the **funding/ship decision** (conjunct-1: worth ~40–60 eng-wk + a permanent second stack) is a
human-owned strategic call that sits AFTER Step 4's technical PASS. The plan should mark this explicitly so a green
Step-4 isn't auto-read as "ship it." Conversely, the human call could STOP even on a technical PASS if the second-
stack burden outweighs the 1.5×.

## Net for Round 4
The decision is gateable with numbers: Step-0 binary kills → Step-1a 5090 floor (≥1.5×, mechanism) → **Step-1b
L40S gate (≥1.5×, target ≥1.8×)** → Step-4 realized (admitted ≥1.5× + tail non-regression + WER). Two framing
corrections: **(a) G2 is a co-equal goal the original threshold omitted → add it as a NON-REGRESSION guardrail
(not strict-improvement), user to ratify; (b) a STOP must be corroborated (negative control + topology sweep +
counter attribution) to protect against a false STOP.** The technical gate and the funding decision stay separate.
