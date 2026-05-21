# Plan: Multilingual checkpoint support — process-per-model + uniform client protocol

Project directory: `./proj-2026-05-20-1947`
Status: **DRAFT v3** — architecture decided (process-per-model, 2026-05-20) after two
independent reviews (Codex `b4jynzb8e`, `bcq6plc7g` + Claude) converged on NEEDS-REWORK.
Pending final sign-off, then `/implement`.

## Context
The English `silence0_warm200` shipped (full-1000 semantic WER 1.95%, TTFB p95 247 ms @ conc-12;
[[silence0-warm200-shippable]]). Next: serve the prompted multilingual checkpoint
`NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b` through the **same Pipecat client protocol**,
so the service code is identical regardless of which model is behind the socket.

**Architecture (decided): process-per-model, client routes by endpoint.**
- The **English server is UNCHANGED**: same `server.py`, omni venv (NeMo 2.8.0rc0), English `.nemo`.
  Zero runtime-swap risk. (The two reviews showed in-process dual-hosting would force English onto
  the EA NeMo — rejected.)
- A **separate multilingual server process** runs `server.py` under a **dedicated EA-NeMo venv**
  (`kingformatty/NeMo @ prompt_unitifed_architecture_hf_EA`, which ships the
  `EncDecRNNTBPEModelWithPrompt` class + a prompted-streaming reference script), loading the
  multilingual `.nemo`.
- The **Pipecat client** is configured with a `model_name → endpoint` map + an optional
  `language`; it connects to the right server and passes `language`. Each server knows its own
  model and validates.

**One `server.py`, two runtimes.** Multilingual-specific code (prompt handling, the
`EncDecRNNTBPEModelWithPrompt` class) must be **lazily/guard-imported** so `server.py` still
imports + runs clean under the omni NeMo for the English server. All multilingual additions are
flag/model-gated → the English path stays byte-identical.

Model-card facts (`private_README.md` + `model_config.yaml`): `auto`=101 (**default** when no
`language`), `en-US`=0 / `en-GB`=1 (benchmark pins `en-US`); `strip_lang_tags=true` (lang tags are
literal vocab tokens `<en-US>`, `<en-GB>`, `<bg-BG>`, … in a 13047-token vocab); `att_context_size`
`[[56,0] default, [56,3], [56,6], [56,13]]` (**no rc1**; rc3 is the low-latency choice);
`subsampling_factor: 8`, `self_attention_model: rel_pos` (same family as the English rc0 crash —
test rc0 carefully). The prompt is **per-call** in the offline path (a batch tensor, not a model
attribute — `hybrid_rnnt_ctc_bpe_models_prompt.py:521,583`); Step 1 must confirm the same for the
streaming path (per-call = concurrency-safe).

Goal: a multilingual server reachable through the uniform client protocol, benchmarked
apples-to-apples vs English — **with the English server byte-identical**.

## Reference implementations
- **EA branch (multilingual runtime + source)**: `kingformatty/NeMo @ prompt_unitifed_architecture_hf_EA`
  — `EncDecRNNTBPEModelWithPrompt` + `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`
  (the prompted-streaming proof for Step 1).
- **Prompt mechanism reference (installed)**: `…/omni/NeMo/.../hybrid_rnnt_ctc_bpe_models_prompt.py`
  — prompt is a per-call batch tensor in `forward()` (`:521,583`) / `transcribe()` (`:461`).
- **Server paths**: `src/nemotron_speech/server.py` — model load `:517-531`; streaming setup
  `:534-590` (`set_default_att_context_size :538`, `change_decoding_strategy :574`, CLI rc
  `choices=[0,1,6,13] :2539`); preprocessor/FFT-plan geometry `:611,:832`; `shift_frames` /
  `final_padding_frames=(R+1)*shift` `:594-660`; `_process_chunk` `:2210`; warmups `:691,:787`;
  fork `_build_continuous_finalize_fork :1792`, `_process_final_chunk :2465`; text/delta update
  (`current_text`/`committed_text`/`continuous_emitted_text`) `:1992-2003,:2239,:2478`; `ready`
  handshake `:1298`.
- **Client + factory**: `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py` (connect `:210-230`,
  `ready` wait `:218`); `stt-benchmark/src/stt_benchmark/services.py:263` (factory passes only URL).
- **Reused harness**: `proj-2026-05-19-eou-endpointing/run_full1000_conc12.py` (needs the handshake
  added; runs under the bench venv, talks to the multilingual server over WS — venv-independent).
- **Regression baseline**: English `silence0_warm200` (1.95% / 247 ms), unchanged.

## Rules
- **ENGLISH BYTE-IDENTITY (hard gate):** the English server (omni venv, English `.nemo`) must stay
  byte-identical / FORK_ASSERT-clean. Because English keeps its runtime, the only risk is the
  additive `server.py` edits perturbing the inactive English path — guard everything model/flag.
- **`server.py` must import + run clean under BOTH NeMo runtimes** (omni for English, EA for
  multilingual). Multilingual-only imports (the prompt class) are lazy/guarded.
- **NeMo isolation:** the EA NeMo lives in its own dedicated venv/checkout; never mutate the omni
  NeMo/venv. ("per-model venv" = a real process+venv boundary, not an in-process fallback.)
- **Opt-in:** multilingual prompts/tag-stripping activate only on the multilingual server. Default
  English path is byte-identical.
- **Prompt = immutable per-session scalar** (`prompt_id` from `prompt_dictionary`), built into a
  per-call prompt vector at each `conformer_stream_step`; copied (not tensor-cloned) into the fork;
  never set as mutable model-global state (avoids cross-session language races under concurrency).
- **`strip_lang_tags` BEFORE state:** strip complete language-tag tokens from raw hypothesis text
  **before** updating `current_text`/`last_emitted_text`/`committed_text`/`continuous_emitted_text`
  or computing append-only deltas, or tags poison multi-segment delta state.
- **Don't assume rc1:** multilingual rc set {0,3,6,13}; test rc0 AND rc3 through live chunking +
  final flush, not a one-call smoke. **rc3's larger final-pad is NOT a finalize-TTFS regression:**
  the fork-flush pads `(R+1)*shift = 640 ms` of synthetic zeros and processes them in one
  `conformer_stream_step` (faster-than-wallclock, exactly the silence_0 mechanism) — so the
  last frame's rc3 right-context is closed synthetically at finalize, not by waiting 480 ms of real
  audio. rc3 finalize TTFS ≈ rc1 + a small GPU delta (~2× the flush compute, tens of ms), NOT
  +320 ms. rc3's real cost vs a smaller rc is **mid-stream interim-transcript look-ahead lag**
  (480 ms vs 160 ms) — which is the interim/partial responsiveness, not the final TTFS the
  benchmark measures — plus a likely accuracy *upside* (more right-context). So rc3 is a reasonable
  default for this model on the finalize use case.
- **Do NOT enable `NEMOTRON_EOU_PROBE` with the multilingual model** — its token-string probe
  assumes the English SentencePiece vocab; the 13047-token multilingual vocab would misbehave.
- **No benchmark-gaming; no new pip deps beyond what the EA NeMo needs; no full-1000 except Step 6.**

## Steps

- [ ] **1. Probe (GO/NO-GO #1): EA-NeMo venv + decisive prompted-STREAMING proof**
  (1a) Install the EA branch in a dedicated venv (isolated from omni). Confirm
  `restore_from(ml.nemo)` instantiates `EncDecRNNTBPEModelWithPrompt` (state-dict loads; `aux_ctc`
  harmless for RNNT decoding). (1b) **Critical gate:** run the EA branch's
  `speech_to_text_cache_aware_streaming_infer.py` (or minimal equivalent) with `target_lang=en-US
  att_context_size="[56,3]" strip_lang_tags=true` on an English fixture → sensible English. This
  proves the prompt IS applied in the **cache-aware streaming** path (not just offline `transcribe`).
  (1c) Pin the exact prompt API (per-call arg vs model attribute — confirm per-call/concurrency-safe)
  and the lang-tag token format. **GATE:** 1a + 1b pass and the prompt is applied in streaming. If
  the prompt is offline-only (a NeMo streaming-method port would be needed) → STOP / re-scope.
  Note: NO English-on-EA test needed — English stays on the omni NeMo.
  Key files: `proj-2026-05-20-1947/probe_ea_streaming.py` (scratch)

- [ ] **2. `server.py` dual-runtime-clean + model-aware streaming config (test rc0 AND rc3)**
  Guard multilingual-only imports so `server.py` imports clean under the omni NeMo. Make streaming
  config model-aware: English `[70,{0,1,6,13}]`, multilingual `[56,{0,3,6,13}]` (default `[56,0]`);
  model-aware rc selection (the English-only `choices`/`[70,R]` are insufficient). Recompute
  per-model `shift_frames`, the **constant FFT-plan ring** (geometry depends on shift/hop —
  server.py:611,832), `drop_extra`, `final_padding_frames=(R+1)*shift`. Smoke **rc0 and rc3** on the
  multilingual server through live chunking + final flush; pick the low-latency valid rc (rc3
  expected; if rc0 runs, record why chosen/rejected — cf. English rc0 crash). Verify
  `change_decoding_strategy` on the prompted model. **REGRESSION:** English server byte-identical +
  FORK_ASSERT clean. **GATE:** English unchanged AND multilingual runs chunked transcription on a
  valid rc.
  Key files: `src/nemotron_speech/server.py`

- [ ] **2b. English byte-identity checkpoint** — short English `silence0_warm200` + FORK_ASSERT smoke
  under the omni venv after the server.py edits. **GATE:** byte-identical / WER-within-CI.
  Key files: (scratch smoke)

- [ ] **3. Prompt threading (scalar, per-call) + language-tag stripping (multilingual server)**
  Store `prompt_id` (from `prompt_dictionary`) as immutable session metadata; build the prompt
  vector per-call and apply it in `_process_chunk`, BOTH warmups, and `_process_final_chunk`; copy
  the scalar into the fork (no tensor clone; no model-global mutation). Strip complete language-tag
  tokens from raw hypothesis text **before** any `current_text`/`committed_text`/delta update.
  English path (no prompt, no stripping) byte-identical. **GATE:** multilingual *streaming* of an
  English fixture (`language=en-US`) → correct English, **no tag leakage**, multi-segment deltas
  clean; English path unchanged.
  Key files: `src/nemotron_speech/server.py`

- [ ] **4. Uniform client protocol: handshake + endpoint routing + validation**
  Client sends, immediately after connect, an explicit `{"type":"init","model_name":<required>,
  "language":<optional>}`; the server validates **before** `_init_session`/continuous-worker/`ready`,
  and the client requires `ready` / raises on `error`. Validation: English server + `language`
  present → **error**; multilingual + `language` → `prompt_dictionary` lookup (unknown → error);
  multilingual + no `language` → **`auto`(101)**. Endpoint routing: `services.py` factory maps
  `model_name → endpoint URL` (+ passes `language`); Pipecat service code becomes model-agnostic.
  **GATE:** all cases correct — especially English+language=error, multilingual-default=auto, and
  routing to the right endpoint.
  Key files: `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`,
  `stt-benchmark/src/stt_benchmark/services.py`, `src/nemotron_speech/server.py`

- [ ] **5. Transcription correctness (subset, no full-1000)**
  ~20-50 English benchmark samples through the multilingual server (`language=en-US`,
  `silence0_warm200`, chosen rc); semantic-compare vs the English checkpoint. Spot-check one
  non-English fixture if obtainable. **GATE:** sensible transcripts (no garbage/looping/tag leakage);
  plausible English WER band.
  Key files: `proj-2026-05-20-1947/` (scratch)

- [ ] **6. Full-1000 concurrency-12 + semantic WER (apples-to-apples; benchmark = en-US)**
  Add the `init` handshake (`model_name`, `language=en-US`) to `run_full1000_conc12.py` (it sends
  none today). Run against the multilingual server (`silence0_warm200`, chosen rc), tag
  `ml_silence0_warm200_c12`. GPU-mem check first. Claude semantic-WER judge (no `--test`, from the
  stt-benchmark dir). Compare WER + TTFS to English 1.95% / 247 ms. **TTFS IS comparable**: rc3's
  640 ms final-pad is faster-than-wallclock (synthetic single-call flush, like silence_0), so the
  finalize TTFS reflects only the synthetic-flush GPU cost, not a 480 ms real-audio wait — expect
  TTFS in the same ballpark as English (a small GPU delta for the ~2× flush). Carry the
  raw-harness-≠-Pipecat-pipeline caveat (`proj-2026-05-19-eou-endpointing/readme-row-silence0-warm200.md`).
  **GATE:** valid WER + TTFS + documented comparison; diagnose prompt/rc if WER is wildly off.
  Key files: `run_full1000_conc12.py`; `proj-2026-05-20-1947/` notes

- [ ] **7. Consolidate: docs + final English re-validation + recommendation**
  Document the process-per-model design (EA-NeMo venv + isolation, endpoint routing, handshake +
  validation, prompt/lang-tag handling, chosen rc + its latency, the WER/TTFS comparison). Final
  English `silence0_warm200` regression sign-off. Update canonical doc / explainer if warranted.
  Key files: `docs/`, `proj-2026-05-20-1947/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Probe: EA-NeMo venv + prompted-STREAMING proof | pending | — | **GO/NO-GO #1**; 1b (prompt-in-streaming) decisive; no English-on-EA needed |
| 2 | server.py dual-runtime-clean + model-aware config (rc0/rc3) | pending | — | hidden per-model state: FFT-plan ring, drop_extra, warmup, padding |
| 2b | English byte-identity checkpoint | pending | — | omni venv, after server.py edits |
| 3 | Prompt (scalar/per-call) + strip_lang_tags before delta | pending | — | concurrency-safe; no model-global lang state |
| 4 | Uniform protocol: init handshake + endpoint routing + validation | pending | — | English+language=error; ml-default=auto; services.py + client |
| 5 | Transcription correctness (subset) | pending | — | en-US sensible; optional 2nd-lang |
| 6 | Full-1000 conc-12 + semantic WER (en-US) | pending | — | harness needs init handshake; rc3 latency caveat |
| 7 | Consolidate docs + final English re-validation | pending | — | sign-off + caveats |
