# Plan: Multilingual checkpoint support + unified multi-model server

Project directory: `./proj-2026-05-20-1947`
Status: **DRAFT v2** (revised after Codex review `b4jynzb8e` + model-card findings + dual-model
design). Pending one more review pass, then `/implement`.

## Context
The English `silence0_warm200` phase shipped (full-1000 semantic WER 1.95%, TTFB p95 247 ms
@ conc-12; [[silence0-warm200-shippable]]). Next target: serve the multilingual checkpoint
`NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b` (cached `.nemo`) **through the same server
+ Pipecat client**, so the Pipecat service code is identical regardless of which model is behind
the socket. The client passes a **required `model_name` + optional `language`**; the server hosts
**either or both** checkpoints and abstracts their architecture difference.

The multilingual model is a **prompted, multilingual Cache-Aware FastConformer-RNNT**
(`EncDecRNNTBPEModelWithPrompt`): a `target_lang` prompt (lang-ID conditioning) selects the
language. Model card facts (from `private_README.md` + `model_config.yaml`):
- **Source / runtime**: NeMo EA branch `git clone https://github.com/kingformatty/NeMo.git;
  git checkout prompt_unitifed_architecture_hf_EA`. It ships the model class AND a reference
  `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py` that runs
  **prompted streaming** with `target_lang=… att_context_size="[56,0]" strip_lang_tags=true`.
- **Languages**: 36+ (en-US, en-GB, es-*, de-DE, fr-*, …). `prompt_dictionary`: `auto`=101,
  `en`=0, `en-US`=0, `en-GB`=1.
- **`auto` is supported** (language-agnostic) → it is the **default** when the client passes no
  `language`. The benchmark run pins `en-US` (=0).
- **`strip_lang_tags=true`**: the model emits language tags in the token stream that must be
  stripped from output.
- **`att_context_size`**: `[[56,0] (default), [56,3], [56,6], [56,13]]`, `subsampling_factor: 8`
  (80 ms encoder frames, matches English so the harness/concurrency assumptions carry). **No rc1.**

Goal: a unified server hosting English and/or multilingual, uniform client protocol, **with the
validated English path byte-identical**.

## Reference implementations
- **EA branch (authoritative source)**: `kingformatty/NeMo @ prompt_unitifed_architecture_hf_EA`
  — the `EncDecRNNTBPEModelWithPrompt` class + the prompted-streaming infer script. This is the
  runtime we install (in isolation); its streaming-infer script is the Step-1 proof.
- **Closest installed reference**: `…/nemotron-nano-omni/NeMo/.../hybrid_rnnt_ctc_bpe_models_prompt.py`
  — prompt applied in `forward()` (`:578-594`) and `transcribe()` (`:333-369`), but the generic
  `conformer_stream_step()` (`mixins.py:592-712`) has **no prompt arg**. So the prompted class
  must override the streaming step (or the EA branch already does — Step 1 confirms which).
- **English server paths to make model-aware**: `src/nemotron_speech/server.py` — model load
  `:517-531`, streaming setup `set_default_att_context_size` `:538` / `change_decoding_strategy`
  `:574` / CLI rc `choices=[0,1,6,13]` `:2539`, `shift_frames`/`final_padding_frames` `:594-660`,
  `_process_chunk` `:2210`, global+per-session warmup `:691,:787`, fork clone `:1805-1838`,
  `_process_final_chunk` `:2465`.
- **Reused unchanged**: `proj-2026-05-19-eou-endpointing/run_full1000_conc12.py` (the conc-12 harness).
- **Regression baseline**: English `silence0_warm200` (1.95% / 247 ms).

## Current state
- `from_pretrained("nvidia/…Multilingual…")` FAILS (no `model_config.yaml` in NeMo HF-cache
  layout); load via the direct `.nemo` path (`restore_from`). `restore_from` resolves the
  config's `target:` by **direct class import** (not a registry), so an importable class suffices.
- The omni NeMo (2.8.0rc0 @ `056d93754`) lacks `rnnt_bpe_models_prompt.py`; the omni checkout is
  shared with the nemotron-nano-omni project (must not be disturbed).
- Server is single-model: per-model facts (`shift_frames`, `final_padding_frames`, `right_context`,
  att_context `[70,R]`, decoding cfg, prompted-or-not) are `ASRServer` globals. CLI rc `choices`
  exclude rc3. No prompt/handshake/lang-tag-strip path exists.
- `aux_ctc` is present in the multilingual config — the class may NOT be a clean "hybrid minus CTC";
  Step 1 must verify state-dict load + that aux CTC is harmless for RNNT-only decoding.

## Rules
- **ENGLISH REGRESSION GUARD (hard gate, every NeMo/server change):** English `silence0_warm200`
  must stay byte-identical (short fixture) or WER-within-CI. This now *includes the NeMo runtime
  swap* — if we serve from the EA-branch NeMo, the English checkpoint must reproduce its validated
  streaming behavior on that NeMo, or unification under one runtime is off.
- **NeMo isolation (Codex must-fix #2):** install the EA-branch NeMo in a **dedicated venv /
  checkout**, never mutating the omni NeMo or omni venv. Add an omni import-smoke if anything
  shared is touched. Prefer the EA branch as the *single unified runtime* for BOTH models **only
  if** English reproduces byte-identically on it (Step 1c); otherwise fall back to per-model venvs
  or NO-GO on in-process unification.
- **Decisive streaming-prompt gate (Codex must-fix #1):** Step 1 proceeds only after the prompt is
  proven applied inside cache-aware streaming (`conformer_stream_step`), not offline `transcribe()`.
- **Opt-in / backward-compatible:** English-only is the default config and stays byte-identical.
  Multilingual + prompts activate only when the multilingual model is loaded AND the client passes
  a `language` (default `auto`).
- **No benchmark-gaming, no new pip deps beyond what the EA NeMo needs, no full-1000 except Step 7.**
- **Don't assume rc1:** multilingual rc set is {0,3,6,13}; test rc0 + rc3 (Codex must-fix #3).

## Steps

- [ ] **1. Probe (GO/NO-GO #1): EA-branch NeMo + decisive streaming-prompt proof + English non-regression**
  Three sub-proofs, all required:
  (1a) **Install EA branch in a dedicated venv** (`kingformatty/NeMo @ prompt_unitifed_architecture_hf_EA`),
  isolated from the omni venv. Confirm `restore_from(ml.nemo)` instantiates
  `EncDecRNNTBPEModelWithPrompt` (state-dict loads; verify `aux_ctc` is harmless / ignorable for
  RNNT decoding).
  (1b) **Prompted-streaming proof (the critical gate):** run the EA branch's
  `speech_to_text_cache_aware_streaming_infer.py` (or a minimal equivalent) with
  `target_lang=en-US att_context_size="[56,3]" strip_lang_tags=true` on an English fixture →
  sensible English text. This proves the prompt IS applied in the cache-aware streaming path. If
  not, STOP (a NeMo streaming-method port becomes the critical path — re-scope or NO-GO).
  (1c) **English non-regression on the EA NeMo:** load the English `nemotron-speech-streaming-en-0.6b.nemo`
  under the EA-branch NeMo and reproduce its validated streaming transcription byte-identically /
  WER-within-CI vs the omni-NeMo baseline. Decides whether one runtime can host both.
  **GATE:** 1a + 1b pass AND 1c passes (→ unified runtime) OR 1c fails (→ documented fallback:
  per-model venvs, or NO-GO on in-process dual-hosting). Record the prompt API + lang-tag format.
  Key files: `proj-2026-05-20-1947/probe_ea_nemo.py` (scratch)

- [ ] **2. Server: per-model state refactor + multi-model hosting (0/1/2, configurable)**
  Move per-model facts (`shift_frames`, `final_padding_frames`, right/att-context, decoding cfg,
  tokenizer, prompted-or-not, prompt_dictionary) out of `ASRServer` globals into a per-model
  structure; a session binds to one model. Config loads English and/or multilingual (e.g. repeated
  `--model name=path` or a small config); English-only default unchanged. Run under the Step-1
  runtime. **REGRESSION:** English `silence0_warm200` byte-identical + FORK_ASSERT clean.
  **GATE:** server hosts either/both; English unchanged.
  Key files: `src/nemotron_speech/server.py`

- [ ] **2b. English regression checkpoint (explicit)** — short English `silence0_warm200` +
  FORK_ASSERT smoke after the refactor + runtime swap. **GATE:** byte-identical / WER-within-CI.
  Key files: (scratch smoke)

- [ ] **3. Model-aware streaming config + right-context (test rc0 AND rc3)**
  Per-model att_context: English `[70,{0,1,6,13}]`, multilingual `[56,{0,3,6,13}]` (default
  `[56,0]`). Make CLI/selection model-aware (current `choices=[0,1,6,13]` + hardcoded `[70,R]` are
  English-only). Smoke **rc0 and rc3** on the multilingual; pick the low-latency valid one (rc3
  expected for the budget; if rc0 runs, record why chosen/rejected — cf. English rc0 crash).
  Recompute `final_padding_frames=(R+1)*shift` for the multilingual R. Verify
  `change_decoding_strategy` works on the prompted model (config default `greedy_batch`).
  **GATE:** a valid streaming rc runs chunked transcription without crash for each hosted model.
  Key files: `src/nemotron_speech/server.py`

- [ ] **4. Client handshake protocol + validation (uniform Pipecat interface)**
  Client (`nemotron_local_stt.py`) sends, at connect, `{model_name: <required>, language: <optional>}`.
  Server validates BEFORE streaming: unknown `model_name` → error; **English model + `language`
  present → error** ("model does not support a language argument"); multilingual + `language` →
  map via `prompt_dictionary` (unknown language → error); multilingual + no `language` → **`auto`
  (101)**. Pipecat service code becomes model-agnostic. **GATE:** all cases behave correctly,
  especially English+language=error and multilingual-default=auto.
  Key files: `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`, `src/nemotron_speech/server.py`

- [ ] **5. Prompt threading + language-tag stripping (streaming + warmup + fork)**
  Thread the per-session prompt (from the handshake `language`) through `_process_chunk`, BOTH
  warmups (global + per-session), and `_process_final_chunk`; **clone it into the fork** (session
  state, immutable per-session, not mutable model-global). Apply `strip_lang_tags` to emitted text.
  English path (no prompt) byte-identical. **GATE:** multilingual *streaming* of an English fixture
  (`language=en-US`) → correct English, **no lang-tag leakage**; English path unchanged.
  Key files: `src/nemotron_speech/server.py`

- [ ] **6. Transcription correctness (subset, no full-1000)**
  ~20-50 English benchmark samples through multilingual (`language=en-US`, `silence0_warm200`,
  chosen rc); semantic-compare vs the English checkpoint. Spot-check one non-English fixture if
  obtainable. **GATE:** sensible transcripts (no garbage/looping/lang-tags); plausible WER band.
  Key files: `proj-2026-05-20-1947/` (scratch)

- [ ] **7. Full-1000 concurrency-12 + semantic WER (apples-to-apples; benchmark = en-US)**
  Reuse `run_full1000_conc12.py` against multilingual (`language=en-US`, `silence0_warm200`,
  chosen rc), tag `ml_silence0_warm200_c12`. GPU-mem check before launch (both models resident
  ≈5-6 GB). Claude semantic-WER judge (no `--test`, from stt-benchmark dir). Compare WER + TTFS to
  English 1.95% / 247 ms. Carry the same raw-harness-≠-Pipeline caveat
  (`proj-2026-05-19-eou-endpointing/readme-row-silence0-warm200.md`). **GATE:** valid WER + TTFS +
  documented comparison; diagnose prompt/rc if WER is wildly off.
  Key files: `run_full1000_conc12.py` (reused); `proj-2026-05-20-1947/` notes

- [ ] **8. Consolidate: docs + final English re-validation + recommendation**
  Document the unified multi-model server (EA-NeMo runtime + isolation, handshake protocol +
  validation, prompt/lang-tag handling, valid rc, the WER/TTFS comparison). Final English
  `silence0_warm200` regression sign-off. Update canonical doc / explainer if warranted.
  Key files: `docs/`, `proj-2026-05-20-1947/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Probe: EA-NeMo + streaming-prompt proof + English non-regression | pending | — | **GO/NO-GO #1**; 1b (prompt-in-streaming) + 1c (English byte-identical on EA NeMo) decisive |
| 2 | Per-model state refactor + multi-model hosting (0/1/2) | pending | — | English-only default byte-identical |
| 2b | English regression checkpoint | pending | — | after refactor + runtime swap |
| 3 | Model-aware streaming config + rc0/rc3 | pending | — | [56,{0,3,6,13}]; test rc0+rc3; model-aware CLI |
| 4 | Client handshake `{model_name, language}` + validation | pending | — | English+language=error; ml-default=auto(101) |
| 5 | Prompt threading + strip_lang_tags (stream/warmup/fork) | pending | — | per-session prompt cloned into fork |
| 6 | Transcription correctness (subset) | pending | — | en-US sensible; optional 2nd-lang |
| 7 | Full-1000 conc-12 + semantic WER (en-US) | pending | — | vs English 1.95%/247ms; GPU-mem check |
| 8 | Consolidate docs + final English re-validation | pending | — | sign-off + caveats |
