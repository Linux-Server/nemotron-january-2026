# Adversarial review brief #3 — EOU PLAN pre-flight (post-parent-project, evidence-grounded)

Target: /home/khkramer/src/nemotron-january-2026/proj-2026-05-19-eou-endpointing/PLAN.md
Status: pre-`/implement` final-gate review. The plan was twice dual-reviewed in
the prior session (Codex `bi49ruh1q` + Codex `bll46sgud` + Claude — 9 defects
folded total) AGAINST AN IN-PROGRESS parent project. The parent project has
now CLOSED (last commit `ef1a7a7` 2026-05-19, all 9 PLAN steps `[x]`); the
canonical doc is committed. This third round is a **tight delta review** —
NOT another structural pass — to confirm nothing has drifted vs the now-frozen
baselines and to fold three cheap pre-flight probe results that resolve key
plan assumptions empirically. Do NOT edit/create/fix any file — findings only.

## Parent-project frozen baselines (the plan must cite these correctly)

- `fork` (Step-7d shippable): WER full **2.10 %**, slice-A −1.04 [−1.47, −0.64],
  slice-B −1.04 [−1.50, −0.63] paired Δ vs `''`; finalize-budget p95
  **325.8 ms**; TTFS p95 **366 ms**. Server commit **`6181310`** +
  stt-benchmark **`e6cf064`**.
- `warm200` (Step-8 recommended shippable): WER full **2.07 %**, slice-A −1.29
  [−1.84, −0.74], slice-B −1.39 [−1.96, −0.87]; budget p95 325.8 ms; TTFS p95
  366 ms. Matches/exceeds the analytical 2.5 s cc7c ceiling on both slices.
  Server commit **`0462679`**.
- Canonical doc + final table at commit **`ef1a7a7`** in
  `docs/semantic-wer-finalization-finding.md` (the Measured Outcome addendum).

## Pre-flight probe results (fold into the review)

**Probe A — NeMo decoding-config placement** (`probe_a_decoding_cfg.py` in
the proj dir). PASS, both candidate placements verified on the configured
cache-aware FastConformer-RNNT greedy path:
- `flat_confidence_cfg` (top-level `preserve_alignments` + `confidence_cfg:
  {preserve_frame_confidence: true, method_cfg: {name: entropy, ...}}`):
  config accepted, `Hypothesis.alignments` populated, `Hypothesis.frame_confidence`
  populated, transcript **byte-identical** to baseline (argmax-invariant
  to log_normalize).
- `nested_greedy` (under `greedy: {preserve_alignments, preserve_frame_confidence,
  confidence_method_cfg}`): same — accepted, both fields populated, transcript
  identical.
- `frame_confidence` in entropy mode returns a float in [0,1] (sample value
  ~0.38; **normalized confidence**, NOT raw entropy — confirms the plan's
  threshold-direction note. HIGH = confident).
- Implication: Step 1 instrumentation will work as designed; use `nested_greedy`
  for cleanliness (existing server cfg already has a `greedy:` block). The
  "preserve_* non-invasive" assertion is verified, not assumed.

**Probe C — client EOU-acceptance gating** (code-read,
`nemotron_local_stt.py:456-458`). VERIFIED. Any `finalize=true` frame received
when `_finalize_requested` is False is **dropped with a debug log**, no
`TranscriptionFrame` pushed → invisible to the benchmark collector.
`_finalize_requested` is only armed via the client's own `_send_finalize_reset()`
(called on `VADUserStoppedSpeakingFrame`). The text-dedup at line 453 already
bypasses in `continuous_context` mode (`and not self._continuous_context`) so
multi-finals/sample work. Implication: **Step 1b is required, not optional** —
the env-gated `NEMOTRON_EOU_CLIENT=1` path must bypass the unarmed-drop in
continuous_context mode (modify lines 456-458). Without it, the server's
confidence-triggered final is invisible and the plan's whole approach
unmeasurable.

**Probe B — cumulative-text transitions eyeball** (warm200 server log scan).
USEFUL NUANCE. The "NON-PREFIX" interim transitions in the server log are
mostly a **display-truncation artifact** (the interim debug log shows a
fixed-width sliding tail; leading chars chopped as cumulative grows). The
prior dual review's "~40 % of Nemotron cumulative transitions are non-prefix"
figure was likely measuring this same artifact, not genuine ASR prefix-rewrite.
Implication: the real rate of beyond-rc1 token edits on this greedy path may
be ~0 (consistent with the greedy algorithm + the user's mental model). This
does NOT invalidate the always-append-only fix (robust to whatever the true
rate is) but **Step 2b's per-token global-frame classification is now even
more decision-relevant** — it will measure the true rate cleanly. The plan's
risk-model paragraph + Rules' "the fork is NOT a free undo" should be slightly
re-tuned to reflect this: the bounded-fork-flush-suffix cost remains the WER
risk, but the "pervasive correction" framing inherited from the prior reviews
is likely overstated.

## Read for this review
- The PLAN.md as it stands (Status: DRAFT — dual-reviewed twice).
- The parent-project closing doc `docs/semantic-wer-finalization-finding.md`
  (commit `ef1a7a7`, the Measured Outcome addendum starting line 319).
- The committed Step-7d/8 code: `src/nemotron_speech/server.py`,
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`,
  `stt-benchmark/scripts/measure.py` — verify the plan's code references
  still match the now-frozen line numbers and symbols.
- The two probe scripts in the proj dir (`probe_a_decoding_cfg.py`) for
  Probe-A reproducibility.

## Attack (focused delta)
1. **Drift vs frozen parent baselines.** Does the plan correctly cite
   `warm200`/`fork` as the baselines to compare against (not stale references
   like "fork has small cost vs analytical ceiling — the EOU plan must close
   that gap" which is now WRONG because `warm200` already closed it)? The
   EOU plan's actual headline value is now "lower TTFS at WER ≥ `warm200`
   (the new ceiling)", not "match the analytical ceiling". Does the goal
   statement reflect that?
2. **Probe-A fold-in.** Given the Probe-A result, can Step-1's Gate (b)
   (probe-enabled WER-equivalent smoke) be tightened (no longer "must verify"
   — empirically verified at single-chunk scale; the smoke remains required to
   confirm it holds across realistic chunk sequences). Does the plan need
   editing to acknowledge?
3. **Probe-C fold-in.** Step 1b's necessity is now empirically established,
   not deduced. Does any other plan section still treat the client change as
   optional?
4. **Probe-B fold-in.** Step 2b's expected-(iii)≈0 hypothesis is now even
   more strongly anticipated (the "~40 %" prior was likely artifactual).
   Does the plan's risk-model wording need de-escalation? Are the GO/NO-GO
   thresholds still well-calibrated?
5. **Code reference drift.** server.py line numbers in the plan
   (`:432-454`, `:1604-1612`, `:1851`, `:571`, `:1410`, `:1479`, etc.) —
   are they still accurate against the now-committed Step-8 server.py
   (commit `0462679`)? Step 8 added ~100 lines around `_init_session` /
   `_run_session_warmup` / `_session_timeline_samples`; some downstream
   line numbers may have shifted.
6. **Anything new since the last review that the plan should fold** (e.g.,
   warm-up's `synthetic_prefix_samples` cursor — does it interact with the
   Step-1 per-token global-frame-index logging?).
7. **`/implement`-readiness checklist.** With the above folded, is the
   plan handoff-ready? Any remaining items that would force a fix round
   during /implement Step 1.

## Output contract
Per item 1-7: SOUND / NEEDS-EDIT / DEFECT. For NEEDS-EDIT/DEFECT items:
exact plan statement, suggested edit (concise), with code/PLAN refs. End
with: overall **READY** / **READY-WITH-EDITS** / **NEEDS-FIX-ROUND** for
handing to `/implement proj-2026-05-19-eou-endpointing/PLAN.md`, and the
top 1-3 must-edit items. Review only; modify no files.
