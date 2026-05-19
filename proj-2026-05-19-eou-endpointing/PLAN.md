# Plan: ASR-internal endpointing — feasibility toward a lower-latency finalize at non-inferior WER

Project directory: `./proj-2026-05-19-eou-endpointing`
Status: **DRAFT — dual-reviewed (Codex `bi49ruh1q` + Claude), revised, then risk-model corrected
(greedy append-only / rc1-stability). NOT `/implement`-ready until the parent project closes and
a final pre-flight re-review passes.**

## Context
Step-7d `fork` reaches a finalize budget ~325 ms p95 by waiting two stacked static endpoint
timers — the client Silero `vad-stop-secs` (~200 ms, framework-locked) then the 150 ms server
debounce. Encoder right-context (rc1 ≈ 160 ms) overlaps that trailing-silence window, and at
finalize the disposable fork supplies the last chunk's right-context as **synthetic
`(R+1)*shift` zeros (faster-than-wallclock)**. So **~175 ms is a *modeled-formula* floor under
the locked additive budget, not an irreducible wall-clock floor**; the binding limit is the
**endpoint-evidence window** of a reliable "speech ended" signal. Goal: **collapse the ~350 ms
of stacked static waits down to that evidence window, at WER non-inferior to the ~375 ms `fork`
baseline**, via signals the RNNT joint/decoder can expose through *supported* NeMo config.
Candidates: (1) RNNT blank-run, (2) hypothesis stability, (3) joint entropy/confidence.
Feasibility-first: cheap offline analysis must pass quantitative GO/NO-GO gates before any build.

**Risk model (corrected).** The configured decode is **greedy** RNNT (`strategy: greedy`),
which is *append-only on the committed token sequence* — it never un-emits a token, and once a
frame has its full ~160 ms right context its decoded tokens are stable. Therefore a
false-early-fire (trigger fires on a within-utterance pause and we emit a final for an
unfinished utterance) does **not** rewrite arbitrary earlier text. But the cost is **not just
one seam token**: the fork flush appends `(R+1)*shift` synthetic silence and decodes it with
`keep_all_outputs=True`, and greedy may emit up to `max_symbols` per encoder timestep — so the
frozen cost is **the entire measured fork-flush suffix** (potentially several tokens decoded
against fake silence rather than the real continuing speech), **plus** any frozen
word-prefix/render correction the always-append-only helper drops at the false boundary,
**plus** the segmentation/restart family. It is bounded (no arbitrary earlier-text rewrite on
greedy) and *local to the false boundary*, but its magnitude is the whole flushed suffix and
must be **measured** (Step 4), not assumed to be ~one word. All are irrevocable *post-emit*
(append-only collector + 7d always-append-only freeze);
a *pre-emit* discard is still free. The prior "~40% non-prefix transitions ⇒ earlier text
rewritten" is **not** evidence of beyond-rc1 rewrite on greedy — it conflates the provisional
≤rc1 tail, interim-vs-flush tail, and detok/casing/punct churn (token-prefix-stable,
semantic-WER-tolerant). Whether any *genuine* older-than-rc1 token edit ever occurs is an
explicit measurement here (Step 2b), not an assumption — and its result sets how aggressive the
trigger can safely be.

## Reference implementations
- **Server tap points (our code — changeable):**
  - `src/nemotron_speech/server.py:~446` builds the decoding cfg with `strategy: greedy`
    (NeMo instantiates `GreedyRNNTInfer`; **`loop_labels` is irrelevant here — it only selects
    the batched `greedy_batch` path**) and calls `change_decoding_strategy(decoding_cfg=...)`
    at `:~454`.
  - **Correct config placement (verified vs NeMo source):** `preserve_alignments: true`
    (top-level or under `greedy:`) **and** `confidence_cfg: {preserve_frame_confidence: true,
    method_cfg: {...}}` **or** nested `greedy: {preserve_frame_confidence: true,
    confidence_method_cfg: {...}}` — NeMo reads frame-confidence from `confidence_cfg`/`greedy`,
    **not** flat top-level keys (`nemo/.../submodules/rnnt_decoding.py:~332`,
    `nemo/.../utils/asr_confidence_utils.py:~341`).
  - `conformer_stream_step` routes through
    `rnnt_decoder_predictions_tensor(..., return_hypotheses=True, partial_hypotheses=...)`
    (`nemo/.../mixins/mixins.py:~707`), so correctly-placed flags populate
    `Hypothesis.alignments`/`.frame_confidence` per chunk (to be VERIFIED non-invasive, Step 1).
  - Parent-stream step `server.py:~1604`; fork-flush `server.py:~1374/1851`
    (`_process_final_chunk(fork)` produces the emitted `final_text`); warm-up `:~571`.
    Blank id via `model.decoding.blank_id` (or `joint.num_classes_with_blank-1`; tensor-vs-int).
  - 7d substrate: `_continuous_handle_vad_stop_locked` /
    `_continuous_handle_debounce_expired_locked` / `_continuous_finalize_emit_locked` (sends
    delta + advances `continuous_emitted_text` at `server.py:~1410`) / `_continuous_append_only_delta`.
- **Client (our code — changeable; REQUIRED for measurability):**
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py:~453` — the client **drops** any
  `finalize=true` unless `_finalize_requested` was armed by a client reset, so a server-driven
  EOU final is **not benchmark-visible** without an env-gated client change (Step 1b/5). Audio
  is available early: `VADProcessor` forwards audio before analyzing
  (`pipecat-core-code/.../audio/vad_processor.py:~112`); synthetic transport streams silence
  until transcription (`stt-benchmark/.../pipeline/synthetic_transport.py:~150`).
- **NeMo (UNCHANGEABLE — read-only):** `nemo/.../utils/rnnt_utils.py` `Hypothesis`
  (`.alignments`, `.frame_confidence`, `.score`, `non_blank_frame_confidence`);
  greedy RNNT appends confidence **before** the blank check
  (`nemo/.../submodules/rnnt_greedy_decoding.py:~486` — frame-confidence defined for blank
  frames); **`entropy` method returns a *normalized confidence* in [0,1], not raw entropy —
  threshold direction must be explicit.**
- Step-1 ground truth: `stt-benchmark/stt_benchmark_data/vad_preflight_silero_stop0.2.json`
  records **Silero state-change timestamps (post `stop_secs=0.2`), NOT acoustic offset**
  (`measure.py:~787,848`).

## Current state
7d `fork` committed/measured. Default decoding cfg sets `strategy: greedy` only — no
`preserve_*`/`confidence_cfg`, so `Hypothesis.alignments/.frame_confidence` are `None` today.
The client only surfaces finals a client reset armed.

## Rules
**Carried from the parent project (binding):** only `src/nemotron_speech/server.py`,
`stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`, `stt-benchmark/scripts/measure.py`,
and benchmark CLI flags may change; **NeMo + benchmark framework UNCHANGEABLE**; no Smart Turn;
no new deps; don't disturb the other project's vLLM; key only from `~/src/pipecat/.env`.
Measurement validity: paired same-ID Δ vs `''` + duration-stratified slice + 95% bootstrap CIs;
**any WER-SCORED config runs the full 1000.**
**Plan-specific (revised):**
- Signals via supported config only; the exact `confidence_cfg`/`greedy` placement above
  (flat keys are silently ignored). Editing any NeMo file is forbidden.
- **The fork is NOT a free undo, but the cost is bounded (corrected risk model).** Fork-discard
  protects parent ASR *state*. On greedy, committed tokens are append-only and beyond-rc1
  tokens are stable, so a false-early-fire does not mutate arbitrary earlier text — but its
  irrevocable post-emit cost is the **entire frozen fork-flush suffix** (the `(R+1)*shift`
  synthetic-silence flush, `keep_all_outputs=True`, up to `max_symbols`/timestep) **+ any
  dropped word-prefix/render correction + the segmentation family** — not "one seam token".
  This is real WER risk, bounded and local but of suffix magnitude; the design must still gate
  emission behind a hard false-final confirmation **or** emit early content as **interim** and
  finalize-for-scoring only on stronger confirmation. "Aggressive trigger, cheap discard" is
  rejected. Step 2b validates the *no-rewrite mechanism* (descriptive); the *per-false-fire WER
  cost* is measured by the Step-4 fork-flush proxy and is what sets Step-3's `F` — 2b does not
  by itself loosen `F`.
- Scratch vs source: offline analysis scripts live in the proj dir as scratch (parent
  precedent: `probe_alias.py`, `equiv_harness.py`, `ringbuf_perf.py`); reusable readers become
  `measure.py` subcommands. No new source files outside the 3 changeable files.
- Full-1000 rule: offline oracle phases are **feasibility proxies, explicitly
  non-authoritative** (subset, timing + reference-distance proxy — NOT a semantic-WER gate);
  the **only authoritative WER decision is the Step-6 full-1000 measured run.**
- Honest scope: server-side endpoint elimination is benchmark-measurable only with the Step-5
  client-acceptance change; the framework-locked Silero `stop_secs` (~200 ms) is a
  production-side caveat, modeled not measured; `~175 ms` stays modeled until measurable.
- Default `''`/phaseG/7d `fork` byte-identical when `NEMOTRON_EOU_*` unset. Gates decisive,
  quantitative, ordered cheap-before-expensive.

## Steps

- [ ] **1. Instrumentation: server config + token-level signal capture + client EOU-acceptance (env-gated, no behavior change when off)**
  (a) Server: under `NEMOTRON_EOU_PROBE=1`, inject the **correctly-placed**
  `preserve_alignments` + `confidence_cfg:{preserve_frame_confidence:true,
  method_cfg:{name:entropy,...}}` (and a max_prob variant) before `change_decoding_strategy`
  (`server.py:~446`). After the parent-stream step (`:~1604`), from
  `session.previous_hypotheses[0]` capture per chunk: **the cumulative token-id sequence
  (`y_sequence`/alignment ids), not just rendered text**; the per-frame alignment
  (blank vs non-blank via `model.decoding.blank_id`); `frame_confidence` (record the
  entropy→normalized-confidence convention + threshold direction); `score`; and — because
  `partial_hypotheses` does **not** carry historical timestamp/alignment forward (only
  `last_token`/`y_sequence`/`dec_state`) — **at the moment each token position is first
  emitted, persist its global encoder-frame index** (and chunk idx), the token string, and its
  decoded **word-boundary state** (does this token complete/extend a word). Also record the
  **model's rc1 right-context frame span `R`** (so a token can be aged vs its right-context
  window) and a **token-level** `changed_positions` (which token ids differ from the previous
  chunk), not a string bool.
  → per-session probe JSONL (reuse the 7d telemetry writer; new keys; do not perturb the
  finalize-budget schema). (b) Client: under `NEMOTRON_EOU_CLIENT=1`, env-gated path in
  `nemotron_local_stt.py` to **accept server-driven `finalize=true` without a prior client
  reset** (Pipecat finalize bookkeeping, push `TranscriptionFrame`, record receipt timing).
  **Gate (TWO smokes — env-unset alone proves nothing about `preserve_*`):** (1) env-unset ⇒
  server AND client byte-identical to 7d `fork`; (2) **`NEMOTRON_EOU_PROBE=1`
  signal-capture-only** ⇒ transcript **WER-equivalent to 7d `fork`** on a 20-sample smoke —
  this is the real non-invasiveness test, because enabling `confidence_cfg` forces
  `log_normalize=True` in the greedy path (argmax is invariant to monotonic normalization, so
  the transcript *should* be identical, but it must be **verified**, not assumed). Plus: probe
  JSONL populated incl. per-token global-frame index + token string + word-boundary state +
  `R` + `changed_positions`, and the client accepts a synthetic server-driven final; only the
  3 files; no new deps.
  Key files: `src/nemotron_speech/server.py`,
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`, `stt-benchmark/scripts/measure.py`

- [ ] **2. Offline collection over the Step-1 subset (signals + replayable fork-flush material)**
  Run instrumented (`NEMOTRON_EOU_PROBE=1`, rc1, continuous) over a documented subset (Step-1
  multi-segment ids + duration-stratified slice-B ids; persist the id list). Persist the
  per-chunk token/signal series **and** enough state to reproduce the **fork-flushed**
  `final_text` at any candidate trigger chunk (log the would-be fork-flush output per chunk
  boundary, or snapshot minimal fork inputs so `oracle_*` can call the real
  `_process_final_chunk`-equivalent). Parent-stream text alone is insufficient (online emits
  fork-flushed text). **Gate:** for ≥5 hand-checked samples the logged fork-flush-at-chunk
  reproduces the actual online `fork` emission at the true endpoint.
  Key files: `proj-2026-05-19-eou-endpointing/collect_signals.py` (proj scratch)

- [ ] **2b. rc1-stability measurement — does prior settled text get rewritten? (mechanism check; NOT the F-setter)**
  Pure offline analysis of the Step-2 per-token global-frame series. For every
  cumulative-change event classify it as: (i) **provisional tail** — all changed token
  positions map to encoder frames within the last ≤`R` (rc1) frames (not yet fully
  right-contexted); (ii-a) **semantic-benign render** — token-id prefix unchanged, change is
  pure casing/punctuation/whitespace (tolerated by the semantic-WER prompt); (ii-b)
  **WER-relevant render/word change** — token-id prefix unchanged or only a trailing subword
  changed but the *decoded word* changes (BPE/word-boundary completion) → can affect WER even
  though "render-only"; (iii) **genuine beyond-rc1 edit** — a token whose right-context window
  already closed changed id. Report rate/distribution of (i)/(ii-a)/(ii-b)/(iii) with examples.
  Expectation from the greedy algorithm: (iii) ≈ 0. **Decision input (descriptive, not the
  gate):** if (iii) ≈ 0 it *confirms the no-arbitrary-rewrite mechanism* (the false-fire cost
  is the local fork-flush suffix, not earlier-text mutation) — it does **not** by itself price
  the false-final WER cost and does **not** loosen Step-3's `F` (the locked append-only
  collector freezes the whole emitted suffix; cost is set by the Step-4 fork-flush proxy). If
  (iii) is materially > 0, that is a surprising, approach-gating finding — record it and
  re-scope before Step 3. This step answers the rewrite question with data instead of assuming
  the algorithm; it de-risks the *mechanism*, while Step 4 measures the *cost*.
  Key files: `proj-2026-05-19-eou-endpointing/rc1_stability.py` (proj scratch)

- [ ] **3. Offline ROC: signal vs endpoint — quantitative GO/NO-GO #1**
  Join per-chunk signals with Step-1 ground truth. Report detection latency **relative to the
  Silero stop-event time** AND, separately, an **estimated acoustic stop** (state the chunk/VAD
  uncertainty band). Sweep thresholds for blank-run K / hypothesis-unchanged K chunks /
  normalized-confidence τ for T ms. Per operating point: detection-latency dist (p50/p95) and
  **false-early-fire rate** — the full latency↔false-fire ROC curve, not a single point.
  **GO/NO-GO #1 (quantitative):** at least one operating point must hit detection-latency
  p95 ≤ X (X set so projected budget ≤ a stated target band) at a false-early-fire rate that
  is *plausibly affordable*. `F` is **not** set here from 2b — Step 3 carries the small set of
  latency-viable candidates forward under a **conservative provisional `F`**; the binding
  `F`/non-inferiority decision is the **combined Step 3+4** judgement, where Step 4 prices each
  candidate's per-false-fire fork-flush WER cost against the pre-registered bound. If no
  operating point can even reach the latency target → STOP, write the negative finding.
  Key files: `proj-2026-05-19-eou-endpointing/oracle_roc.py` (proj scratch)

- [ ] **4. Offline fork-flush oracle proxy — quantitative GO/NO-GO #2 (non-authoritative)**
  For the best 1–2 operating points, replay each subset sample: at the trigger chunk produce
  the **actual fork-flushed `final_text`** (Step-2 material), reconstruct the emitted transcript
  via the exact `_continuous_append_only_delta` semantics, compute a **reference-distance proxy**
  (normalized edit distance / token-F vs ground truth) paired vs the 7d `fork` proxy on the
  same ids. Explicitly **non-authoritative** (subset + proxy; full-1000 rule reserves scored
  WER for Step 6). **GO/NO-GO #2:** proxy paired Δ vs `fork` within a pre-registered
  non-inferiority margin. State the simulation≠online residue.
  Key files: `proj-2026-05-19-eou-endpointing/oracle_wer.py` (proj scratch)

- [ ] **5. Minimal env-gated online prototype (only if 2b, 3 & 4 pass) + dual pre-run review**
  Server `NEMOTRON_EOU_TRIGGER=...` fires the speculative finalize on the chosen signal from
  the parent stream, with the Rule's hard false-final confirmation (or
  interim-until-confirmed), replacing/gating the 150 ms debounce; fork-discard +
  always-append-only + FORK_ASSERT + close-drain unchanged. Client `NEMOTRON_EOU_CLIENT=1`
  acceptance path. Contained smoke (7d-style): fires on real end, holds through real
  intra-utterance pauses, no crash, JSONL shows endpoint_wait collapsed, client surfaces the
  server-driven final. **Gate:** smoke passes; default/`fork` byte-unchanged with envs unset;
  **dual adversarial review (Codex + Claude) pre-run.**
  Key files: `src/nemotron_speech/server.py`, `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`

- [ ] **6. Measured full-1000 `eou` — the only authoritative gate**
  Full-1000 measured run (server `NEMOTRON_EOU_TRIGGER=...` `NEMOTRON_FORK_ASSERT=1`, client
  `NEMOTRON_EOU_CLIENT=1`, rc1). Paired Δ vs `''` and vs `fork`, slice-A + slice-B + full,
  bootstrap CIs; finalize-budget p95; emit-once; 0 fork-alias-FAILED. **Gate:** WER
  non-inferior to `fork` (CIs overlap / Δ within noise both slices) AND finalize-budget p95
  collapses toward the target band AND emit-once early_final ≈ 0 AND 0 fork-alias-FAILED.
  Honest measured-vs-modeled: server endpoint elimination measured; Silero `stop_secs` modeled.
  Key files: `src/nemotron_speech/server.py`, `stt-benchmark/scripts/measure.py`

- [ ] **7. Consolidate: extend the canonical table + docs + recommendation**
  Add `eou` row(s) (full + slice-A + slice-B, paired Δ + CIs, in-budget tag, signal + operating
  point, the Step-2b rc1-stability result, the Step-3 latency↔false-fire curve), update
  `docs/ttfs-latency-explainer.html` + the canonical finding doc with measured reality and the
  measured-vs-modeled / framework-VAD caveats, recommendation, model revision hashes.
  Key files: `docs/ttfs-latency-explainer.html`, `docs/semantic-wer-finalization-finding.md`,
  `stt-benchmark/scripts/measure.py`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Instrumentation (cfg + token-level capture + client accept) | pending | — | correct confidence_cfg placement; token-id series + R + changed_positions; preserve_* non-invasive verified; client-accept REQUIRED |
| 2 | Offline collection (subset + fork-flush replay material) | pending | — | parent-stream text alone insufficient |
| 2b | rc1-stability measurement (rewrite question, with data) | pending | — | classify (i) ≤rc1 tail / (ii-a) benign render / (ii-b) WER-relevant render-word / (iii) genuine beyond-rc1 edit; (iii)≈0 expected; **mechanism check, NOT the F-setter** |
| 3 | Oracle ROC vs endpoint | pending | — | **GO/NO-GO #1**: ROC curve; conservative provisional F; binding F = combined Step 3+4 (Step-4 prices per-fire cost); gt = Silero stop-event + est. acoustic±band |
| 4 | Fork-flush oracle proxy | pending | — | **GO/NO-GO #2**, non-authoritative proxy |
| 5 | Online prototype (env-gated) + dual review | pending | — | hard false-final gate / interim-until-confirmed |
| 6 | Measured full-1000 `eou` | pending | — | only authoritative WER+latency gate |
| 7 | Consolidate table + docs | pending | — | measured-vs-modeled honesty; framework-VAD caveat |

## Dual-review record
- Codex `bi49ruh1q` + Claude: 5 DEFECTs folded (NeMo `confidence_cfg`/`greedy` placement +
  `strategy:greedy` selects `GreedyRNNTInfer`; fork not a free undo; ground truth = Silero
  stop-event not acoustic; entropy→normalized confidence; Step-4 must replay fork-flushed text;
  Step-3 quantitative; client EOU-acceptance REQUIRED; scratch-vs-source + full-1000 proxy).
- Claude↔user refinement: corrected the risk model — greedy is append-only, no beyond-rc1
  rewrite; "~40% non-prefix" is provisional-tail + render-only + segmentation, not rewrite.
  Added **Step 2b** (rc1-stability measurement) to answer the rewrite question empirically;
  Step-1 captures the per-token series.
- Codex `bll46sgud` (review #2) + Claude: Item 4 SOUND (5 prior defects resolved); 4 DEFECTs
  folded — (1) false-fire cost is the **whole frozen fork-flush suffix** (`(R+1)*shift`
  silence, `keep_all_outputs`, ≤`max_symbols`/step), not "one seam token"; (2) Step-1 must
  persist a per-token global encoder-frame index **at first emission** (partial_hypotheses
  drops historical frame/alignment) + word-boundary state, and Step-2b splits render into
  benign vs WER-relevant; (3) **2b is the no-rewrite *mechanism* check, NOT the F-setter** —
  Step-3 keeps a conservative provisional `F`, binding `F` = combined Step 3+4 (Step-4 prices
  per-fire cost); (5) non-invasiveness needs a **probe-enabled** smoke (enabling confidence
  forces `log_normalize=True`), not just env-unset.
- A final pre-flight re-review is required before this is handed to `/implement`.
