# Multilingual checkpoint support — design + results (proj-2026-05-20-1947)

Canonical reference for how the Nemotron speech server supports the multilingual checkpoint
`nvidia/NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b` alongside the English specialist
`nvidia/nemotron-speech-streaming-en-0.6b`. Results/numbers live in
[`step6-ml-comparison.md`](step6-ml-comparison.md); this doc is the architecture + how-to.

## Architecture: process-per-model, client routes by endpoint

Two **separate server processes**, one per checkpoint, each on its own port. The Pipecat client
(or any client) picks the endpoint by `model_name` and passes an optional `language`. The shared
`server.py` code runs under both — but a given process loads exactly one checkpoint.

```
                         ┌─────────────────────────────────────────┐
  client (Pipecat /      │  English server   (omni venv)            │
  stt-benchmark /        │    NeMo 2.8.0rc0 · torch 2.x · port 8080 │
  raw-WS harness)        │    EncDecHybridRNNTCTCBPEModel (English) │
        │                │    att_context [70,1] (rc1), no prompt   │
        │  model_name ──▶ └─────────────────────────────────────────┘
        │  + language    ┌─────────────────────────────────────────┐
        └──────────────▶ │  Multilingual server  (EA venv)          │
                         │    kingformatty/NeMo @ ...EA · torch     │
                         │    2.12.0+cu130 · port 8081              │
                         │    EncDecRNNTBPEModelWithPrompt          │
                         │    att_context [56,3] (rc3) · 128-lang    │
                         │    prompt set per-session under lock      │
                         └─────────────────────────────────────────┘
```

**Why process-per-model (decided 2026-05-20, commit 2e8f863):** the two checkpoints need
**different, incompatible NeMo runtimes** (the multilingual `EncDecRNNTBPEModelWithPrompt` lives
only on the EA branch with torch 2.12+cu130; the English checkpoint is validated under the omni
venv's NeMo 2.8.0rc0). Hosting both *in one process* would force an English-runtime swap and risk
perturbing the English checkpoint's validated streaming/cuFFT behavior
([[cufft-stft-plan-size-nondeterminism]]). Two earlier reviews (Codex + Claude) converged on
NEEDS-REWORK for in-process dual-hosting; process-per-model eliminates that risk:
**English byte-identity is guaranteed because the English process and runtime are untouched.**

## Runtime isolation (the EA venv lives OUTSIDE the repo)

| | English server | Multilingual server |
|---|---|---|
| NeMo | omni ASR venv (`/home/khkramer/src/nemotron-nano-omni/.venv-asr`), torch 2.11.0+cu130 · NeMo 2.8.0rc0 — note `.venv` lacks torch | EA venv (`/home/khkramer/src/nemotron-ea-nemo/.venv-ea`), torch 2.12.0+cu130 |
| NeMo source | omni checkout | `kingformatty/NeMo @ prompt_unitifed_architecture_hf_EA` (tip `2d8fcad82`), cloned to `/home/khkramer/src/nemotron-ea-nemo` (6.3 GB, **outside** this repo) |
| Model class | `EncDecHybridRNNTCTCBPEModel` | `EncDecRNNTBPEModelWithPrompt` (prompted multilingual RNNT, `num_prompts: 128`, `prompt_field: target_lang`) |
| Checkpoint | `…/nemotron-speech-streaming-en-0.6b.nemo` | `…/nemotron-asr-streaming-multilingual-0.6b.nemo` (2.3 GB, HF cache) |
| loguru | present | installed into `.venv-ea` (Step 5; server.py logging dep) |

`server.py` imports cleanly under **both** runtimes: every multilingual-only NeMo import is lazy /
guarded, and the prompted-model code paths are gated behind a runtime feature check.

## How the shared server.py adapts (all multilingual paths guarded)

The single gate is `prompted_model = hasattr(model, "set_inference_prompt")` — **False for English**
(→ original behavior, byte-identical) and **True for multilingual**. Changes by step:

- **Step 2 (`da593a9`)** — model-aware streaming config:
  - `prompted_model` detection; `_select_att_context_size()` picks English `[70,R]` vs reads the
    multilingual cfg `[56,R]` (default rc3).
  - `final_padding_frames = (right_context + 1) * shift_frames` — auto-adapts the synthetic
    finalize pad to rc (English rc1 → 320 ms; multilingual rc3 → 640 ms).
  - `_apply_inference_prompt()` set-under-`inference_lock` at the 4 inference sites.
  - `_extract_hypothesis_text` strips language tags **prompted-only**.
  - CLI `--right-context` default `None`, `choices=[0,1,3,6,13]`.
- **Step 3 (`86bed1a`)** — per-session prompt + tag stripping:
  - `ASRSession.target_lang: Optional[str]`; `_apply_inference_prompt(session)` sets the
    **model-global** prompt from the session's language *per call, under the lock* (concurrency-safe
    because `batch_size=1` and one global `inference_lock` serializes all inference).
  - The disposable finalize **fork copies `session.target_lang`** (scalar) so the flush uses the
    right prompt.
  - `_strip_lang_tags()` removes literal `<xx-XX>` vocab tokens (complete + partial-trailing
    fragment + whitespace-collapse) **before** current/committed/delta — so tags never reach the
    append-only delta or the client.
- **Step 4 (`be40121`)** — connection validation (query-param handshake):
  - `_validate_connection_query` / `_validate_session_target_lang` / `_validate_model_query_param` /
    `_read_prompt_dictionary`; `PROMPTED_DEFAULT_TARGET_LANG = "auto"`.
  - Validated **before** `_init_session`/`ready`; on failure sends `{"type":"error",…}` + closes.
  - **English + any `language` → error** (per the user's requirement); **multilingual + no language
    → `auto`** (index 101).

### Client side (`83f01ea`)
- `nemotron_local_stt.py::_connect_url()` appends `?language=&model=` **only when set** → the
  English URL is unchanged (backward-compatible). Raises on `{"type":"error"}` before ready.
- `services.py::resolve_nemotron_local_route(model_name, language)` maps `english →
  NEMOTRON_LOCAL_URL`, `multilingual → NEMOTRON_LOCAL_ML_URL`; `create_nemotron_local(...)`.

## Language prompt & tags

- `prompt_dictionary`: `en = 0`, `en-US = 0`, `en-GB = 1`, `auto = 101` (128 languages total).
  `auto` is the **default** when no language is passed (the model card supports language ID).
- Language tags are literal `<xx-XX>` tokens in the 13047-token vocab; stripped by regex
  `\s*<[a-z]{2}-[A-Z]{2}>` plus a partial-trailing-fragment guard, before any text leaves the
  server. **Verified zero leakage** across the 1000-sample run.
- ⚠ Do **not** enable `EOU_PROBE` with the multilingual model (13047-vocab assumptions differ).

## Right-context choice: rc3, and it is NOT a latency regression

Multilingual `att_context_size = [[56,0],[56,3],[56,6],[56,13]]` (no rc1). **rc3 `[56,3]`** is the
low-latency choice. Its synthetic finalize pad is `(3+1)*shift = 640 ms`, but that pad is fed to the
decoder as **one `conformer_stream_step` call** — faster than wall-clock, the same mechanism that
makes `silence_0` safe ([[silence0-warm200-shippable]]). **Measured TTFS p95 245 ms = parity with
English's 247 ms** — confirmed, not a +480 ms wait. (rc0 `[56,0]` also loads without crashing,
unlike English `[70,0]`; diagnostics showed rc0 gives no accuracy win here.)

## Results (full detail in step6-ml-comparison.md)

Apples-to-apples (same harness, 1000 samples, conc-12, silence0+warm200), multilingual forced to
`en-US`:
- **Latency: parity** — TTFS p95 245 ms vs English 247 ms.
- **English WER: ~2.4× worse** — 4.72% mean / 4.84% pooled vs 1.94% / 1.95%. Median 0% for both;
  the gap is entirely in the error *tail* (10 catastrophic >50% clips + 3 empties vs zero for
  English) — the expected 128-language-0.6b vs English-specialist-0.6b capacity tradeoff.
- **Integration: clean** — 0 errors, 0 tag leakage, 0 hangs across 1000 concurrent-12 sessions.

## English regression sign-off (Step 7 — GO)

After all Steps 2-4 server.py edits, the English path was re-validated under the omni ASR runtime
(`.venv-asr`, rc1):
- Startup: **no** `Prompted model detected` / `Inference prompt set` lines, `att_context_size=[70,1]`,
  shift 160 ms, no traceback (`prompted_model=False`, original behavior).
- **Byte-identity: 100/100** transcripts exactly equal to the `silence0_warm200_c12` baseline; 0
  mismatches, 0 errors, 0 empties. FORK_ASSERT clean (only PASSED entries).
- Validation handshake exactly **error / error / ready**: `?language=en-US` and `?language=auto`
  both return `{"type":"error","message":"this model does not accept a language argument"}` + close;
  no query params → `{"type":"ready"}`.

**The multilingual changes are inert for English.** Safe to keep both servers from the same
`server.py`.

## Recommendation

- **English-only voice agents → English checkpoint** (same latency, much better accuracy).
- **Multi-language coverage required → multilingual checkpoint** — latency is production-ready and
  the integration is sound; expect a fatter English error tail and occasional empties on very
  quiet/short clips (mitigate with a definitive end-of-turn trigger + input gain normalization).

## How to run each server

```bash
# English (omni ASR venv = .venv-asr, NOT .venv; rc1)
NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 \
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python src/nemotron_speech/server.py \
    --model <…/nemotron-speech-streaming-en-0.6b.nemo> --host 127.0.0.1 --port 8080 --right-context 1

# Multilingual (EA venv, rc3) — different port
NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 NEMOTRON_MODEL_NAME=multilingual \
  /home/khkramer/src/nemotron-ea-nemo/.venv-ea/bin/python src/nemotron_speech/server.py \
    --model <…/nemotron-asr-streaming-multilingual-0.6b.nemo> --host 127.0.0.1 --port 8081 --right-context 3

# client handshake (multilingual): ws://host:8081?language=en-US   (omit language → auto)
# client handshake (English):     ws://host:8080                   (passing language → error)
```

## Commit trail (branch khk/20260516)
`2e8f863` architecture · `4a3d499` rc3-latency framing · `4fc3b07` Step1 GO ·
`da593a9` Step2 (+2b) · `86bed1a` Step3 · `be40121` Step4 · `cca20a4` Step5 · `4ebf011` Step6.
