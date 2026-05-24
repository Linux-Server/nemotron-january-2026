# Opus path-forward recommendations

Date: 2026-05-24
Author: Opus (independent advisory; complements `codex-pathforward.md`, does not restate it)

**Position.** The v6 gate is correct and should hold: do not start Phase 1 or any Wave-2 native port. But I disagree
with one framing in both the plan and the codex memo — they treat "finish the Python plan, then write the 0.0 packet" as
the next action, which puts the *most likely STOP-determining evidence last*. I think the single highest-value move is to
**front-load the two cheapest pieces of STOP evidence that do NOT require the Python plan to fully land**, because the
project is most likely dead and the fastest way to confirm that costs days, not the weeks the Python plan will take to
complete. I also think the 5 rounds under-weighted one structural risk (the residual is measured against a *moving*
baseline that the Python plan is *still actively shrinking*) and one strategic option (a scoped, in-tree decode lane that
isn't a second stack). Details below, prioritized.

---

## 1. Single most valuable next action

**Do NOT wait for the full Python plan to land before generating STOP evidence.** The plan sequences 0.0 strictly after
`proj-2026-05-24-0859` completes (all 6 steps). That is the right gate for a GO decision, but it is the *wrong* gate for
the STOP decision, which is the likely outcome. Two of the three BET conjuncts can be attacked with evidence that is
*already in flight or nearly free*:

- **Conjunct 2 (GIL/scheduler-bound, not MPS/bandwidth-bound) is partly answerable from the Python plan's own Step 5
  GIL probe** (`proj-2026-05-24-0859/PLAN.md:148-156`), which is already scoped and produces the decode-vs-glue
  attribution. That attribution is the *core input to 0.1's thesis*. If Step 5 shows the wall is the eager-decode
  `.item()` loop on lane threads (likely, given the deployed `loop_labels=True` path at `server.py:1463-1474`), the
  GIL-bound premise survives; if it shows the wall is MPS context-launch serialization, conjunct 2 is dead and the whole
  project STOPs without ever running 0.1.
- **Conjunct 1 (residual worth ~40-60 eng-wk) is bounded ABOVE before the plan even finishes.** Steps 2b (K=4 -> ~28/box)
  and 4 (finalize priority) are the density/tail levers. The plan already states the realistic ceiling is ~28/box
  in-budget, and the native *aspiration* is 40-48/box but only "contingent on shared weights AND single-context isn't
  launch-serialized" — both unproven. So the residual the native project can capture is at most `48 - 28 = ~20 streams/box`
  AND only if 0.1/0.9/0.11 all clear. That is a small, triple-conditional prize against an engineer-year. **This arithmetic
  can be written down today** and is, I think, already close to a STOP on its own.

**NOW:** (a) ensure the Python Step 5 GIL probe explicitly emits the decode-vs-glue split in a form 0.1 can consume;
(b) write the residual *ceiling* arithmetic into `reviews/decision.md` as a pre-Python upper bound, with the explicit
note that the Python plan is actively lowering the residual from below. **WAIT:** everything native.

The reframe: the question is not "what's the residual after the Python plan" (measured last) but "is there ANY plausible
residual ceiling that clears the bar" (boundable now). If the ceiling doesn't clear, STOP without spending the Wave-1
GPU/cloud time at all.

## 2. Cheapest kill-signals, in decision-value-per-effort order

The plan numbers spikes 0.0/0.1/0.3/0.5; the cheapest *kill* order is different, and some kills are cheaper than the plan
specifies. Ranked by (probability-of-killing × 1/effort):

1. **The residual-ceiling arithmetic (free, paper).** Section 1. Upper-bounds the prize before any measurement. If the
   best-case native delta over the *post-Python* baseline can't clear the threshold even assuming all native gates pass,
   STOP. This is the single cheapest kill and it is not in the spike list — it's a precondition to spending on spikes.

2. **0.5 as a one-pass histogram query, NOT the simulator (cheap, partly answerable from EXISTING bench data).** The
   codex memo already says run the histogram first; I go further — **an existing bench result may already answer it.**
   Memory `streaming-batching-outcome` records "86% B=1" today and a phasing penalty (in-phase 115 vs out-of-phase 56).
   The deployed steady cadence is 160 ms (`batch_primitives.py` gen_synthetic comment; `server.py` BATCH_MAX_WAIT_MS=8)
   with an 8 ms batch window. Independent Poisson arrivals at 160 ms cadence into an 8 ms window give an expected
   coincidence count near 1 — i.e. B≈1 is the *structural default* unless arrivals phase-align. The simulator's own
   `gen_synthetic` (lines 43-47) bakes this in: only `phase_aligned` fills B. **So the 3-5x throughput claim is probably
   already dead from existing data + first principles**, and the synthetic simulator can confirm the sensitivity to
   arrival phase in an afternoon (run the three arrival modes, show poisson/bursty -> frac_B1 high). If B≈1, drop the
   3-5x AND the steady-graph density claim, which removes most of conjunct-1's value -> re-run the ceiling arithmetic ->
   likely STOP. **This is cheaper than the plan's "instrument server -> capture post-Python traces -> replay" path and
   may not need the server instrumented at all for the kill direction.**

3. **0.3 stage-1 only: py3.13t environment feasibility (1-2 days, time-boxed).** Just `import torch; import nemo; run one
   chunk` under free-threaded 3.13t. If PyTorch/NeMo's C-extensions don't opt into `Py_mod_gil` disable, or the stack is
   unstable, B4 is killed cheaply (and that pushes toward STOP-or-B1, not B4). If it imports clean, that does NOT fund
   B4 — it only buys the right to spend on stage-2. Crucially: stage-1 is baseline-independent and can run **today** in
   parallel, because it tests ecosystem maturity, not residual size.

4. **0.1 reduced matrix, seeded by the Step 5 attribution (moderate; needs post-Python baseline + GPU).** Don't build the
   full 8-toggle ablation. The Step 5 GIL probe already tells you *which* serializer dominates; 0.1 then only needs to
   answer the single binary question the decision tree turns on: **does a single process overlap finalize+steady, or only
   MPS/multi-proc?** Test single-process-single-context vs MPS first. If only MPS overlaps, native-under-MPS is tail-only
   (no density) -> re-run 0.0 -> almost certainly STOP. The full lock/gate/sync matrix is only worth building if there IS
   a large residual that you cannot attribute — an unlikely branch.

5. **0.11 memory measurement (cheap GPU, but conditional).** Only run if 0.5 keeps steady-graphs alive. The analysis is
   already done; only the per-B resident-pool number is missing. If 0.5 kills B>1, skip 0.11 entirely — the graph-pool
   memory is moot.

6. **0.7 aarch64 (cheap, but non-gating).** Platform pre-check; runs whenever GB10 exists. Never blocks the L4/L40S
   STOP/B4/B1 decision. Lowest decision-value of the set for the *core* go/no-go.

**Cheapest path to the 0.0/0.1/0.3 kill-signals:** ceiling arithmetic (free) -> 0.5 histogram + existing-data check
(afternoon) -> 0.3 stage-1 import probe (1-2 days, parallel) -> Step 5 GIL attribution (already in the Python plan) ->
0.1 binary single-vs-MPS overlap (only if the above haven't already STOP'd). Three of the four leading kills need no new
cloud spend and two need no post-Python baseline.

## 3. Is the gating correct? What's safe to start in parallel

**The gating is correct in its conservatism but slightly wrong in its sequencing.** Two refinements:

- **0.6a is correctly gated and must NOT start.** The plan and codex agree; I reinforce it with the strongest reason:
  0.6a is baseline-*independent* (frozen fixtures) which makes it *tempting* to start early, but it is the single most
  expensive Track-A item (4-8 eng-wk, the funding-gate risk) and reproducing NeMo's eager `greedy_batch` label-looping
  state machine — `LabelLoopingStateItem`, partial-hyp in-place merge (`rnnt_greedy_decoding.py:783-804`), max_symbols=10
  forced-blank saturation (`rnnt_label_looping.py:466-484`), fork deep-clone (`server.py:6370-6425`) — is exactly the
  work that is wasted if 0.0 STOPs. Starting it early is the classic sunk-cost trap. **Safe to start: the fixture-capture
  harness + comparator schema (the cheap shell). Do NOT start: the native decode itself.**

- **What IS safe to start in parallel (and wastes nothing if 0.0 STOPs):**
  - The residual-ceiling arithmetic and threshold freeze (pure decision hygiene; needed for any outcome).
  - 0.3 stage-1 import feasibility (tests ecosystem, not residual; result is durable knowledge regardless).
  - The 0.5 synthetic arrival-phase sensitivity run (sharpens the B-fill kill; reusable).
  - 0.9 / 0.11 are already complete (paper) — no further spend until needed.
  - Making the Python Step 5 probe emit the decode-vs-glue attribution in a 0.1-consumable form (it's already scoped Python work).

- **What wastes effort if 0.0 STOPs (do NOT start):** all native ports (0.6a impl, 0.2 export, 0.8 preproc), the runtime
  contract (0.10), any Rust/C++ scaffolding, the full 0.1 ablation matrix, 0.11 GPU memory if 0.5 already killed B>1.

## 4. B4 vs B1 vs STOP: the flipping evidence and cheapest experiment for each

| Outcome | Concrete evidence that flips TO it | Cheapest experiment to get it |
|---|---|---|
| **STOP** | Residual ceiling < threshold (Sec 1); OR 0.5 shows B≈1 (kills 3-5x + steady-graph density); OR 0.1 shows only-MPS overlap (native-under-MPS = tail-only, no density); OR Step 5 says the wall is MPS/launch not GIL | **Ceiling arithmetic (free)** + 0.5 synthetic phase-sensitivity (afternoon). Most likely STOP'd here with zero cloud spend. |
| **B4** | 0.3 stage-1 imports clean AND stage-2 off-event-loop dispatcher closes the post-Python tail residual end-to-end AND the free-threaded PyTorch/NeMo stack is production-deployable | Stage-1 import probe (1-2 days) is the *gate*; only if it passes, build the stage-2 dispatcher against the post-Python baseline. B4 keeps NeMo's decode -> skips 0.6a/0.2/0.8 entirely (the whole expensive port). |
| **B1** | ALL of: residual clears threshold; 0.1 proves single-process overlap (not just MPS); 0.3 fails or is unstable; 0.5/0.11 keep B>1 + graph pool fits; 0.6a achieves exact Hypothesis/state equivalence | This is the *last* and most expensive branch. Don't spend toward it until every cheaper kill above has failed to fire. The 0.6a state-equivalence proof is the dominant cost and the dominant risk. |

**Key asymmetry the plan gets right and is worth restating:** B4 dominates B1 on cost by ~30-50 eng-wk because it keeps
NeMo's Python decode. So the *order of evidence* should be tuned to give B4 its shot before B1: confirm GIL-bound (Step 5)
-> confirm py3.13t viable (0.3 stage-1) -> measure B4 closes the residual (0.3 stage-2). Only if B4 fails should B1's
expensive ports be considered. The plan's decision tree encodes this but the spike *ordering* in the README buries 0.3
behind 0.1; I'd run 0.3 stage-1 first because it's the cheapest thing that can either kill B4 or make it the answer.

## 5. The 2-4 highest-leverage open questions to DISCUSS/decide together

1. **What is the residual, in dollars or streams/box or p99-ms, that justifies an engineer-year + a permanent second
   stack?** Without a written number this decision is unfalsifiable and will be rationalized post-hoc. This is the same
   point codex makes, and it is the #1 thing to settle. My addition: also decide **who maintains the second stack** —
   the cost is not just the build; it's the ongoing dual-stack tax (every NeMo upgrade, every model swap, every CUDA
   bump now hits two codebases). The plan counts ~40-60 eng-wk to *build*; it does not price the *carry*.

2. **Is the prize even the right problem?** (Under-weighted by all 5 rounds — see Sec 6.) p50 is immovable; the levers
   are tail + density, both of which the Python plan + horizontal scaling (more boxes) also address, just less
   cost-efficiently. The strategic question is whether ~20 extra streams/box (best case) at the cost of an engineer-year
   beats *spending those same eng-weeks elsewhere* (multilingual quality, the front-drop bug in memory
   `multilingual-front-drop-bug`, a better VAD to actually move p50, or just buying more L4 boxes which are the cheapest
   $/stream per `deployment-target-sagemaker`). Density is a COGS lever; quantify the COGS at projected scale before
   funding a COGS-reduction project.

3. **Is the baseline allowed to keep moving?** The Python plan is *actively shrinking the residual the native project is
   meant to capture* (finalize graph already landed 246/279; Step 2b adds K=4; Step 4 cuts tail). Every Python win is a
   native loss. Decide explicitly: do we freeze the Python plan at a commit and measure the native residual against that,
   or do we let Python keep improving (in which case the native residual may asymptote to zero before the native build
   even finishes)? This is a real risk the rounds noted ("baseline moves, record the commit") but did not escalate to a
   *strategic* question: a fast-improving Python baseline can make B1 obsolete mid-build.

4. **If the evidence says "decode-GIL-bound but a full runtime isn't worth it," is the in-tree native-decode extension
   (Sec 6 / codex's middle option) acceptable, or is the answer just STOP + B4?** This determines whether a 5th option
   stays on the table.

## 6. Under-weighted risks and strategic options

- **The whole tail+density prize may not be worth pursuing vs accepting the Python result.** This is the biggest thing
  the 5 rounds under-weighted — they converged on "the business premise is the most-likely-wrong thing" but framed it as
  a *gate* (0.0) rather than as a *prior*. My read of the evidence already assembled: p50 immovable; steady is 86% B=1
  (batching is a self-host lever that already showed ~0 benefit on cloud GPUs per `streaming-batching-outcome`); density
  ceiling is ~28/box post-Python with native upside triple-conditional; L4 is already the cheapest $/stream and you can
  just add boxes. **The honest prior is that this project STOPs, and the most valuable thing the team can do is confirm
  that cheaply (Sec 1-2) and redirect the engineer-year.** The plan is excellently rigorous about *how* to decide; it is
  less forceful about *how likely STOP is* — and an engineer-year of opportunity cost deserves that forcefulness.

- **A genuine 5th option beyond STOP/B4/B1: scoped in-tree native decode (pybind/C++ extension under today's server).**
  Codex raises this as a "middle option"; I want to elevate it, because it specifically targets the *most likely true*
  bottleneck (the eager-decode GIL on lane threads) without the *most expensive* parts of B1 (WS protocol, deploy
  topology, admission, metrics, scheduler rewrite, MPS topology, shared-weights). It still needs the 0.6a state-
  equivalence gate (the hard part of the decode is intrinsic), but it is NOT a second stack — it's a CUDA extension
  inside the existing Python process, so it inherits all the deploy/metrics/protocol machinery and the ongoing-carry
  cost is far lower. **If Step 5 says "decode GIL is the wall" and 0.3 kills B4 (py3.13t not production-ready), this is
  the option that avoids both STOP-with-residual-on-the-table AND the full B1 engineer-year.** It deserves an explicit
  row in the 0.4 decision tree. Caveat: it does not deliver shared-weight density or fix the exclusive-gate/MPS topology,
  so it's a tail/decode-throughput play only — which is fine if that's where the residual actually is.

- **Under-weighted: the Python Step 5 probe is a free pre-payment on conjunct 2.** The rounds treat 0.1 as the conjunct-2
  kill, but Step 5 (already in the Python plan, already scoped) produces the decode-vs-glue attribution that *seeds* 0.1
  and can pre-kill it. The two plans should be explicitly wired: Step 5's deliverable should be a required input to 0.1,
  not re-derived. This shaves real time off the most expensive Track-B spike.

- **Mild over-weighting: the byte-exactness bar may be stricter than the business needs.** The plan rightly relaxes to
  T2b named-tolerance cross-B, but the T1 streaming-aware event/delta/generation equivalence is an extremely tall bar
  that the rounds treated as immovable. Worth one discussion: is *behavioral non-inferiority within a WER-CI + a bounded
  interim-flicker rate* acceptable instead of exact event-stream equivalence? If yes, the native correctness cost drops
  materially. If the product genuinely needs byte-identical event streams, fine — but that requirement should be a
  product decision, not a default inherited from the byte-exact-shipping history.

---

## Top 3 next actions

1. **Write the residual-CEILING arithmetic into `reviews/decision.md` now (free, paper):** best-case native streams/box
   and p99-ms delta over the *post-Python* baseline, assuming all native gates pass, against the engineer-year + carry
   cost. If the ceiling doesn't clear a (to-be-written) threshold, STOP without spending on spikes. Front-load the kill.

2. **Run the two cheapest baseline-independent kills in parallel this week:** (a) the 0.5 synthetic arrival-phase
   sensitivity run + cross-check against the existing "86% B=1 / phasing 115-vs-56" data to confirm/kill the 3-5x
   batching claim; (b) the 0.3 stage-1 py3.13t import-and-one-chunk feasibility probe (time-boxed to 1-2 days). Neither
   needs the Python plan to finish or any cloud spend.

3. **Wire the Python plan's Step 5 GIL probe to 0.1:** make its decode-vs-glue attribution a required, 0.1-consumable
   deliverable, and freeze the pre-registered 0.0/0.1/0.5 numeric thresholds (Sec 5 Q1) BEFORE any Wave-1 measurement.
   Do NOT start 0.6a/0.2/0.8/0.10 or any native code.

## Topics to discuss with the user

1. **The number.** What residual (streams/box, p99-ms, or $/stream at projected scale) justifies an engineer-year PLUS
   the ongoing dual-stack carry cost? And who carries it? (No number -> STOP/defer.)
2. **Is this the right problem at all?** Density is a COGS lever; L4 is already cheapest $/stream and scales horizontally.
   Does ~20 best-case extra streams/box beat spending the engineer-year on multilingual quality, the front-drop bug, or a
   p50-moving VAD? Quantify projected-scale COGS before funding a COGS project.
3. **Does the baseline freeze?** The Python plan is actively shrinking the native residual (finalize graph already landed;
   K=4 + finalize-priority coming). Do we measure native against a frozen Python commit, or accept that a fast-improving
   Python baseline could make B1 obsolete mid-build?
4. **The 5th option:** if Step 5 says decode-GIL-bound and 0.3 kills B4, is a scoped in-tree native-decode extension
   (no second stack, inherits deploy/metrics/protocol; still needs 0.6a equivalence) the answer instead of full-B1 or
   leaving residual on the table? Should it get a row in the 0.4 decision tree?
5. **Is byte-exact event-stream equivalence (T1) a product requirement or an inherited default?** A WER-CI + bounded-
   flicker non-inferiority bar would materially cut native correctness cost if the product can accept it.
