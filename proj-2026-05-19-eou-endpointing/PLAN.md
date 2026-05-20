# Plan: ASR-internal endpointing — feasibility toward a lower-latency finalize at non-inferior WER

Project directory: `./proj-2026-05-19-eou-endpointing`
Status: **READY for `/implement`** — parent project closed 2026-05-19 (last commit `ef1a7a7`);
pre-flight delta review (round 3: Codex `b510t8mq0` + Claude parallel) + three cheap probes
(Probe A NeMo confidence-cfg placement PASS, Probe C client-acceptance gating VERIFIED, Probe B
"~40% non-prefix" largely a server-log display-truncation artifact) folded. The recommended ship
baseline is now **`warm200`** (Step-8, server commit `0462679`, full 2.07%, slice-A −1.29
[−1.84,−0.74], slice-B −1.39 [−1.96,−0.87] paired Δ vs `''`, budget p95 325.8 ms, TTFS p95
366 ms — matches/exceeds the analytical 2.5 s cc7c ceiling); `fork` (`6181310`, full 2.10%) is
retained as an ablation control to isolate the EOU contribution from warm-up.

## Context
Step-7d `fork` reaches a finalize budget ~325 ms p95 by waiting two stacked static endpoint
timers — the client Silero `vad-stop-secs` (~200 ms, framework-locked) then the 150 ms server
debounce. Encoder right-context (rc1 ≈ 160 ms) overlaps that trailing-silence window, and at
finalize the disposable fork supplies the last chunk's right-context as **synthetic
`(R+1)*shift` zeros (faster-than-wallclock)**. So **~175 ms is a *modeled-formula* floor under
the locked additive budget, not an irreducible wall-clock floor**; the binding limit is the
**endpoint-evidence window** of a reliable "speech ended" signal. Goal: **collapse the ~350 ms
of stacked static waits down to that evidence window, at WER non-inferior to the now-frozen
`warm200` ship baseline** (Step-8 commit `0462679`, full 2.07% / slice-A −1.29 / slice-B −1.39
paired Δ vs `''`, budget p95 325.8 ms / TTFS p95 366 ms) — i.e., **lower TTFS than `warm200`
at WER ≥ `warm200`**. (`fork`, full 2.10%, is the no-warm-up ablation control that isolates
the EOU contribution from the per-session warm-up.) Signals come via *supported* NeMo config
(Probe A verified both `confidence_cfg` and nested-`greedy` placements work, non-invasive).
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
rewritten" framing from review round 2 was, per Probe B (round-3 eyeball scan of warm200 server
logs), **largely a server-log display-truncation artifact** — the per-chunk interim debug log
shows a fixed-width sliding tail (leading chars chopped as the cumulative grows), which a
naïve `startswith` check reads as "non-prefix" even when the underlying token sequence is
monotone-extending. On greedy, the genuine rate of beyond-rc1 token-rewrite is expected ≈0 by
the algorithm itself. Step 2b's per-token global-frame classification measures the real rate
cleanly; the always-append-only contract is robust to whatever that rate turns out to be.

## Reference implementations
- **Server tap points (our code — changeable; line numbers current at parent commit `0462679`):**
  - `src/nemotron_speech/server.py:~452-460` builds the decoding cfg with `strategy: greedy`
    (NeMo instantiates `GreedyRNNTInfer`; **`loop_labels` is irrelevant here — it only selects
    the batched `greedy_batch` path**) and calls `change_decoding_strategy(decoding_cfg=...)`
    at `:461`.
  - **Correct config placement (Probe A VERIFIED both work, single-fixture single-clip
    transcript-identical to baseline + Hypothesis.alignments and .frame_confidence populated):**
    `preserve_alignments: true` (top-level or under `greedy:`) **and** `confidence_cfg:
    {preserve_frame_confidence: true, method_cfg: {name: entropy, ...}}` **or** nested `greedy:
    {preserve_frame_confidence: true, confidence_method_cfg: {...}}`. Recommend the nested
    `greedy` placement (matches the existing server cfg block). NeMo reads frame-confidence
    from `confidence_cfg`/`greedy` only — flat top-level keys are silently ignored
    (`nemo/.../submodules/rnnt_decoding.py:~332`,
    `nemo/.../utils/asr_confidence_utils.py:~341`). Probe-A measured `frame_confidence` is a
    float in [0,1] for entropy mode = **normalized confidence** (HIGH = confident).
  - `conformer_stream_step` routes through
    `rnnt_decoder_predictions_tensor(..., return_hypotheses=True, partial_hypotheses=...)`
    (`nemo/.../mixins/mixins.py:~707`); flags populate `Hypothesis.alignments`/`.frame_confidence`.
  - Parent-stream `conformer_stream_step`: **`server.py:~1695`**; the speculative-fork flush
    site (`_process_final_chunk(fork)` via the fork built by `_build_continuous_finalize_fork`
    at `:~1280`, invoked from `_continuous_finalize_emit_locked` at `:~1418` which
    **dispatches `_process_final_chunk` at `:1465-1467`** (inside `async with inference_lock`
    + `run_in_executor`); `_process_final_chunk` def at `:~1872`; **the actual final
    `conformer_stream_step` call is at `:~1948`** (inside `_process_final_chunk`).
    Per-session warm-up (Step-8 addition) at **`:~641`** (`_run_session_warmup`); global startup
    warmup unchanged at `:~552`. Blank id via `model.decoding.blank_id` (or
    `joint.num_classes_with_blank-1`; tensor-vs-int).
  - 7d substrate: `_continuous_handle_vad_stop_locked` (`:~1157`),
    `_continuous_handle_debounce_expired_locked` (`:~1243`), `_continuous_finalize_emit_locked`
    (`:~1418`, advances `committed_text`/`last_emitted_text` at `:~1491`),
    `_continuous_finish_speculative_finalize_locked` (`:~1526`),
    `_continuous_cold_reset_after_finalize_locked` (`:~1548`, the only continuous `_init_session`
    site; resets `continuous_emitted_text=""` at `:~1563`), `_continuous_append_only_delta`
    pure helper (`:~233`).
  - Step-8 surface (warm-up offset interaction): `synthetic_prefix_samples: int = 0` on
    `ASRSession` (`:~293`); set at `_run_session_warmup` (`:~696`); fork copies it (`:~1297`);
    `_session_timeline_samples` helper (`:~710-711`) = `synthetic_prefix_samples + total_audio_samples`,
    used at the chunk-cadence gate (`:~1620`). `session.emitted_frames` after warm-up
    equals `warmup_frames` (`:~695`), NOT 0 — Step 1 instrumentation must use it as the
    starting offset for the per-token global-frame index (see Step 1).
- **Client (our code — changeable; REQUIRED for measurability — Probe C VERIFIED):**
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py:456-458` — `if not
  self._finalize_requested: ... return` — any `finalize=true` frame received when
  `_finalize_requested` is False is dropped with a debug log, no `TranscriptionFrame` pushed,
  invisible to the benchmark collector. `_finalize_requested` is only armed via the client's
  own `_send_finalize_reset()` (called on `VADUserStoppedSpeakingFrame`). The text-dedup at
  `:453` already bypasses in `continuous_context` mode (`and not self._continuous_context`)
  so multi-finals/sample work. So a server-driven EOU final is **not benchmark-visible**
  without an env-gated client change (Step 1b/5). Audio is available early: `VADProcessor`
  forwards audio before analyzing
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

- [x] **1. Instrumentation: server config + token-level signal capture + client EOU-acceptance (env-gated, no behavior change when off)**
  (a) Server: under `NEMOTRON_EOU_PROBE=1`, inject the **correctly-placed**
  `preserve_alignments` + nested `greedy:{preserve_frame_confidence:true,
  confidence_method_cfg:{name:entropy,...}}` (Probe-A-recommended; flat
  `confidence_cfg:{...}` also works per Probe A; both verified non-invasive at single-fixture
  scale) before `change_decoding_strategy` (`server.py:~452-461`, with the call itself at
  `:461`). After the parent-stream step (`:~1695`), from `session.previous_hypotheses[0]`
  capture per chunk: **the cumulative token-id sequence (`y_sequence`/alignment ids), not just
  rendered text**; the per-frame alignment (blank vs non-blank via `model.decoding.blank_id`);
  `frame_confidence` (Probe A confirmed entropy mode returns a float in [0,1] = normalized
  confidence, HIGH = confident); `score`; and — because `partial_hypotheses` does **not** carry
  historical timestamp/alignment forward (only `last_token`/`y_sequence`/`dec_state`) — **at
  the moment each token position is first emitted, persist its global encoder-frame index**
  (and chunk idx), the token string, and its decoded **word-boundary state** (does this token
  complete/extend a word). Also record the **model's rc1 right-context frame span `R`** (so a
  token can be aged vs its right-context window) and a **token-level** `changed_positions`
  (which token ids differ from the previous chunk), not a string bool.
  **Step-8 warm-up offset (when running on top of `warm200`):** the per-session warm-up at
  `_run_session_warmup` (`server.py:~641`) consumes `warmup_frames` of encoder frames at
  session init and sets `session.emitted_frames = warmup_frames` (`:~695`) and
  `session.synthetic_prefix_samples = warmup_samples` (`:~696`). **CRITICAL implementation
  detail (Codex round-4 catch):** `session.emitted_frames` is *incremented at server.py:~1720
  AFTER* the parent-stream `conformer_stream_step` call inside `_process_chunk` — so reading
  it *after* the step gives the post-increment value, off by one chunk for every token. The
  instrumentation MUST snapshot **before the step**: `chunk_model_frame_start =
  session.emitted_frames` and `prev_y_len = len(session.previous_hypotheses[0].y_sequence)
  if session.previous_hypotheses else 0`. After the step, for each newly-emitted token (index
  ≥ `prev_y_len` in `y_sequence`), derive its model-frame index from
  `Hypothesis.alignments`/`Hypothesis.timestamp` (whichever the model populates, both gated by
  `preserve_alignments=True` per Probe A) **plus** the pre-call `chunk_model_frame_start`.
  Persist two cursors: (i) **model-frame index** = the value above (used by Step 2b for rc1
  aging); (ii) **real-audio-time** = the chunk's monotonic wall-clock timestamp (used by
  Step 3 for endpoint-latency joins with `vad_stop`/`debounce_expiry`). The two diverge by
  exactly the warm-up's synthetic prefix when warm-up is on, by zero otherwise.
  → per-session probe JSONL (reuse the 7d telemetry writer; new keys; do not perturb the
  finalize-budget schema). (b) Client: under **`NEMOTRON_EOU_CLIENT=1` AND
  `continuous_context`** (both required — preserves default-client behavior under any
  misconfig), env-gated path in `nemotron_local_stt.py` (modify the unarmed-drop gate at
  `:456-458`) to **accept server-driven `finalize=true` without a prior client reset**
  (Pipecat finalize bookkeeping, push `TranscriptionFrame`, record receipt timing).
  **Gate (TWO smokes — env-unset alone proves nothing about `preserve_*`):** (1) env-unset ⇒
  server AND client byte-identical to 7d `fork`; (2) **`NEMOTRON_EOU_PROBE=1`
  signal-capture-only** ⇒ transcript **WER-equivalent to 7d `fork`** on a 20-sample smoke.
  Probe A (single-fixture single-clip, both placements) already verified config acceptance +
  `Hypothesis.alignments`/`.frame_confidence` populated + transcript byte-identical to baseline
  (argmax invariant to the `log_normalize=True` that enabling `confidence_cfg` forces); the
  20-sample smoke remains required only to confirm this holds across realistic chunk sequences
  and the full 7d stream path. Plus: probe
  JSONL populated incl. per-token global-frame index + token string + word-boundary state +
  `R` + `changed_positions`, and the client accepts a synthetic server-driven final; only the
  3 files; no new deps.
  Key files: `src/nemotron_speech/server.py`,
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`, `stt-benchmark/scripts/measure.py`

- [ ] **2. Offline collection over the Step-1 subset (per-chunk signals only)**
  Run instrumented (`NEMOTRON_EOU_PROBE=1`, rc1, continuous) over a documented subset (Step-1
  multi-segment ids + duration-stratified slice-B ids; persist the id list). Persist the
  per-chunk token/signal series only — **no fork-flush replay material captured here**.
  Fork-flush reconstruction is **deferred to Step 4**, which only does the 1–2 selected
  operating points (much cheaper than per-chunk-boundary flushing across the entire subset:
  Codex round-4 recommendation). For Step 4's per-operating-point reconstruction, Step 2 must
  persist enough parent state to *resume the live session at the candidate trigger chunk* and
  call the real `_process_final_chunk` on the fork built there — minimally:
  `cache_last_channel/time/len`, `previous_hypotheses` (deep-cloned per Step-3 recipe),
  `pred_out_stream`, `pending_audio`, `total_audio_samples`, `synthetic_prefix_samples`,
  `emitted_frames`, raw/mel ring snapshots, and the audio bytes up to that chunk. **Gate:**
  for ≥5 hand-checked samples, replaying via Step-4's reconstruction at the true endpoint
  chunk reproduces the actual online `fork` emission byte-for-byte.
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
  server-driven final. **Gate:** smoke passes (including: a server-driven final emitted via
  the EOU trigger, followed later by the client's own VAD-stop reset for the same utterance,
  produces **exactly one** scored `TranscriptionFrame` — no duplicate from the post-trigger
  client reset path; Codex round-4 catch); default/`fork` byte-unchanged with envs unset;
  **dual adversarial review (Codex + Claude) pre-run.**
  Key files: `src/nemotron_speech/server.py`, `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`

- [ ] **6. Measured full-1000 `eou` — the only authoritative gate (dual-baseline ablation)**
  Run **two** measured full-1000 tags so the EOU contribution is isolated from the warm-up
  contribution (otherwise we ship a bundle of unknown attribution):
  - **`eou`** — `NEMOTRON_EOU_TRIGGER=<chosen>` + `NEMOTRON_FORK_ASSERT=1`,
    `NEMOTRON_CONTINUOUS=1`, client `NEMOTRON_EOU_CLIENT=1`, rc1, **no warm-up**. Compares vs
    `fork` (the no-warm-up control) to isolate the EOU trigger's WER/latency contribution.
  - **`eou_warm200`** — same as above PLUS `NEMOTRON_WARMUP_MS=200`. Compares vs `warm200`
    (the now-frozen ship baseline) as the actual **ship gate**.
  Paired Δ vs `''` AND vs `fork` AND vs `warm200`, slice-A + slice-B + full, bootstrap CIs;
  finalize-budget p95; emit-once; 0 fork-alias-FAILED. **Ship gate (`eou_warm200`):** WER
  non-inferior to `warm200` (paired Δ-vs-`warm200` CIs include 0 or are negative both slices;
  no slice-Δ point estimate worse than the cc7c-vs-cc7b known-noise reference ±0.10 pp) AND
  **finalize-budget p95 < `warm200`'s 325.8 ms** (the headline win — lower TTFS at non-inferior
  WER) AND emit-once early_final-class behavior preserved AND 0 fork-alias-FAILED.
  **Ablation read (`eou`):** confirms the EOU contribution alone; if `eou` ≥ `fork` accuracy
  at lower latency, the trigger is doing real work even without warm-up. Honest measured-vs-
  modeled: server endpoint elimination measured; framework-locked Silero `stop_secs` modeled.
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
| 1 | Instrumentation (cfg + token-level capture + client accept) | done | — | Codex `mpdna7a0` impl + Claude diff-review ACCEPT. server.py +372 (all-additive, env-gated NEMOTRON_EOU_PROBE), nemotron_local_stt.py +4/-1 (gate at :457 `not _finalize_requested and not (_eou_client_accept and _continuous_context)`); EOU probe writes a SEPARATE `<run_tag>.eou_probe.jsonl` (finalize-budget schema untouched). Round-4 pre-call snapshot at server.py:2056 (chunk_model_frame_start + prev_y_len) BEFORE conformer_stream_step at :2063; emitted_frames increment at :2086; write at :2092 — ordering correct. Per-token derives model_frame_index from Hypothesis.alignments/timestamp + pre-call offset; model_frame_event_index = model_frame_index*1024+subindex for RNNT same-frame ties. timeline_cursor uses Step-8 helper (warm-up-aware). **Smokes:** env-unset 2/2 exact-match vs `fork`; probe-enabled smoke (EOU_PROBE+EOU_CLIENT+CONTINUOUS+SILENCE_MS=150+FORK_ASSERT) 20/20 exact-match vs `fork` + 1417 probe rows + monotone frame indices + frame_confidence ∈ [0.021,0.9999] + client bypass all 3 modes tested (default-dropped/misconfig-dropped/enabled-accepted) |
| 2 | Offline collection (subset + fork-flush replay material) | pending | — | parent-stream text alone insufficient |
| 2b | rc1-stability measurement (rewrite question, with data) | pending | — | classify (i) ≤rc1 tail / (ii-a) benign render / (ii-b) WER-relevant render-word / (iii) genuine beyond-rc1 edit; (iii)≈0 expected; **mechanism check, NOT the F-setter** |
| 3 | Oracle ROC vs endpoint | pending | — | **GO/NO-GO #1**: ROC curve; conservative provisional F; binding F = combined Step 3+4 (Step-4 prices per-fire cost); gt = Silero stop-event + est. acoustic±band |
| 4 | Fork-flush oracle proxy | pending | — | **GO/NO-GO #2**, non-authoritative proxy |
| 5 | Online prototype (env-gated) + dual review | pending | — | hard false-final gate / interim-until-confirmed |
| 6 | Measured full-1000 `eou` + `eou_warm200` (dual-baseline) | pending | — | `eou` (no warm-up, vs `fork` ablation) + `eou_warm200` (vs `warm200` ship gate); ship gate = WER non-inferior to `warm200` AND budget p95 < 325.8 ms |
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
- **Round 3 (pre-flight delta, 2026-05-19; Codex `b510t8mq0` + Claude parallel; both
  CONVERGED on the same fix set, verdict NEEDS-FIX-ROUND on mechanical edits):** parent
  project closed (`ef1a7a7`); three cheap pre-review probes run before this round to ground it
  empirically. Probe results folded:
  - **Probe A (NeMo confidence-cfg placement) — PASS**: both `flat_confidence_cfg` and
    `nested_greedy` placements accepted; `Hypothesis.alignments` and `.frame_confidence`
    populated; transcript byte-identical to baseline (argmax-invariant to `log_normalize=True`);
    `frame_confidence` is a float in [0,1] (entropy mode = normalized confidence, HIGH =
    confident). Plan now recommends nested-`greedy` placement; Step-1 Gate (b) tightened
    from "must verify" to "Probe A verified at single-fixture; 20-sample smoke confirms at
    realistic scale". Script: `probe_a_decoding_cfg.py`.
  - **Probe C (client EOU-acceptance gating) — VERIFIED** at `nemotron_local_stt.py:456-458`
    (`if not _finalize_requested: drop`). Step-1b's "REQUIRED for measurability" upgraded
    from presumed to empirically anchored.
  - **Probe B (cumulative-text transitions, warm200 server-log eyeball) — clarifying nuance**:
    the prior dual-review's "~40% non-prefix transitions" figure was largely a
    *display-truncation artifact* of the server's interim debug log (fixed-width sliding tail),
    not measured ASR prefix rewrite. Genuine beyond-rc1 token-rewrite rate on greedy expected
    ≈0. Context risk-model paragraph de-escalated accordingly; Step 2b's measurement is now
    even more decision-relevant (it cleanly settles the question).
  - **Mechanical folds (5)**: (a) baseline pivot — ship gate now `warm200` (`0462679`, full
    2.07 % / slice-A −1.29 / slice-B −1.39), `fork` retained as ablation control. (b) Step 6
    becomes dual-baseline: `eou` (no warm-up, vs `fork` ablation) AND `eou_warm200`
    (vs `warm200` ship gate) — isolates EOU contribution from warm-up. (c) Step-1 capture
    folds the Step-8 warm-up offset (`session.emitted_frames` = `warmup_frames` after warm-up;
    log dual model-frame-index + real-audio-time cursors). (d) Server line citations updated
    to the now-frozen parent commit `0462679` (parent-stream at `:~1695`, fork-flush at `:~1948`,
    per-session warm-up `_run_session_warmup` at `:~641`, etc.). (e) Status line updated to
    "READY for `/implement`" — parent project has closed and this round's edits are folded.
- **Round 4 (post-fold verification, 2026-05-19; Codex `blca94vy2` + Claude parallel; both
  CONVERGED on the same fix set):** focused verification pass on round-3's mechanical edits +
  red-team for blind spots. Both reviewers independently found the same 3 line-number drifts
  from round 3 (`:~1167` → `:1280` `_build_continuous_finalize_fork`; `:~1511` → `:1491`
  `committed_text=final_text` advance; `:~1812` → `:1620` chunk-cadence gate). **Codex
  additionally caught one substantive new defect Claude missed: `session.emitted_frames` is
  *incremented at server.py:~1720 AFTER* the parent-stream `conformer_stream_step` call**
  — so reading it *after* the step (the round-3 wording) gives the post-increment value, off
  by one chunk for every token's frame index in Step-2b classification. Step-1 instrumentation
  now requires a *pre-call snapshot* of `chunk_model_frame_start = session.emitted_frames`
  and `prev_y_len = len(previous_hypotheses[0].y_sequence)`, with per-token model-frame
  derived from `Hypothesis.alignments`/`timestamp` + the pre-call offset. Also folded:
  Step-1b's bypass made explicitly `EOU_CLIENT=1 AND continuous_context`; Step 2 captures
  per-chunk signals only (fork-flush replay deferred to Step 4, where it's per-operating-point
  not per-chunk — Codex's cost-saving recommendation); Step 5's smoke must assert
  no-duplicate-final after a post-trigger client VAD-stop reset; the round-3 ":~1466 runs
  the model step" wording corrected to "dispatches `_process_final_chunk` at `:1465-1467`;
  actual final `conformer_stream_step` at `:1948`". Step-6 dual-baseline confirmed SOUND
  (extra compute justified for attribution; reject the sequencing-optimization suggestion).
- **Plan is now READY for `/implement`** at the post-round-4 commit. The round-4 catches
  (especially the emitted_frames pre-call snapshot) prevent real implementation bugs at
  Step 1 — exactly what round 4 was for.
