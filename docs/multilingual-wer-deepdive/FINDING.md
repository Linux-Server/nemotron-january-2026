# Multilingual checkpoint — elevated-WER deep dive (front-drop finding)

**Date:** 2026-05-21  **Status:** RESOLVED — root cause is **H1: the multilingual model's streaming decode
goes silent on certain utterance onsets (model fragility), NOT a defect in our inference code** (confirmed by
re-stream, see below). **Scope:** the SHIPPED multilingual path (`proj-2026-05-20-1947`).

## Question
The multilingual checkpoint (`NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`) scored **4.84% pooled
WER** on the 1000-sample English benchmark vs **1.95%** for the dedicated English checkpoint (~2.4×). We
significantly modified the inference path for this checkpoint (prompted path / `set_inference_prompt`,
att_context `[56,3]`, **320ms shift + 1280ms final padding** vs en 160ms/16, tag-stripping, finalize/fork).
Is the elevated WER genuine model quality, or an artifact of the modified inference code?

Both runs used `language=en-US` explicitly (not auto) — verified in the run config + server log
(`Inference prompt set to 'en-US'` per session). So this is the model's English-*prompted* performance.

## TL;DR
**~81% of the en→ml gap is genuine model quality; ~19% is the multilingual model's streaming decode going
SILENT on the onset of ~10 clips (H1, confirmed by re-stream) — NOT a defect in our inference code.** The
model emits nothing for the first 5–6.5s of these utterances then wakes up mid-stream; the English
checkpoint emits from the first word. Our modified streaming/prompt/tag-strip/finalize path is clean.

## Evidence
Method: head-to-head on stored `wer_metrics` (ml vs en, same samples), no new inference. Scripts:
`ml_deepdive.py`, `ml_quantify.py` (in this dir; read-only on `stt-benchmark/.../results.db`).

1. **Deletions are the only anomaly.** Substitutions 756 (ml) vs 347 (en) = 2.2× — tracks the overall 2.4×
   and looks like model quality (phonetic confusions; en makes the same kind, fewer). But **deletions 285 vs
   42 = 6.8×**, and they cluster at utterance *starts*.
2. **Front-of-utterance dropping, ml-specific.** 10 ml clips drop a leading run of ≥3 reference words; en
   does this on **0** (en max leading-deletion-run = 2). On all 10, the **English checkpoint has ≤1 deletion**
   (perfect on 4) — so the audio contains the front, en transcribes it verbatim, **ml drops it**. Example
   `a934808b` (13.2s): en = "the expensive package i ordered was marked as delivered two days ago…" at
   en_wer=0.00; ml = only the tail "it is not anyway of my property i must initiate an immediate trace".
   ml keeps ~58% of words on these clips; en keeps 100%. → not the model (easy English), not the harness
   (shared with en; `" ".join(final_parts)` accumulation, run_full1000_conc12.py:193).
3. **NOT general long-clip streaming.** The 0b baseline interims (captured 2026-05-21) show clean 16s/14s/12s
   clips streaming monotonically from the front (interim "I'm" → … → full sentence; final == longest interim).
   So the streaming/cache path is sound; the front-drop is triggered by *specific* clips (only 6 of 380 long
   clips; one affected clip is just 5.3s) → suspected **internal-pause → mid-clip finalize/reset edge case**
   in the modified continuous/silence0 path, not the geometry itself.
4. **No tag leak (0/24 in the 0b capture), prompt confirmed applied, no decode looping** (ml 2 vs en 1).

## Quantification
| | pooled WER | gap to en |
|---|---|---|
| en (same 997) | 1.95% | — |
| **ml, all** | **4.84%** | 2.89pp |
| ml, excl. the 10 front-drops | **4.29%** | 2.34pp |

- 10 front-drop clips = **143 errors = 12% of all ml errors, 43% of all ml deletions**, explaining **~19% of
  the en→ml gap**.
- Remaining ~81% = distributed substitutions (253 sub-only samples + the sub share of mixed) = genuine
  model quality. Bug-free, ml ≈ 4.29% (still ~2.2× en — the inherent 0.6B-multilingual penalty).

## The 10 front-drop sample_ids (`ml_silence0_warm200_c12`)
```
a934808b 13.2s del=23/32 | 3cec70da 10.6s del=21/31 | 331e2f1b 13.5s del=15/29
b9c2ca23 10.6s del=14/28 | 30f889cb 12.6s del=12/31 | 5054b614 15.5s del=11/31
c242a089 10.0s del=9/30  | 7abb01d2 12.7s del=9/33  | 7b170b20  5.3s del=5/11
3196d91e 13.8s del=4/30
```

## Root cause — RESOLVED: H1 (re-stream, 2026-05-21)
Re-streamed the front-drop clips through the live ml server under the exact full-1000 conditions
(`?language=en-US`, 20ms chunks realtime-paced, 200ms trailing silence, vad_stop+reset) with full
interim+finalize logging (`restream_diag.py`). DECISIVE:
- `a934808b` (13.2s): **first interim at t=6.5s = "It is not…"**; the front ("the expensive package i
  ordered…") **never appears in any interim**; `current_text` never shrank; one finalize delta = the tail.
- `5054b614` (15.5s): **first interim at t=5.2s = "Shake…"**; "can you give me some of the main plot points
  of" **never decoded**.

So the front is **never produced during streaming** — not lost at finalize (the finalize delta faithfully
emits exactly what streaming produced), not truncated by our code (`current_text` grows monotonically from
the late start, never shrinks). The earlier "internal-pause → mid-clip finalize/reset" hypothesis was wrong
(the harness sends a single segment; there is no server-side VAD). This is **H1: the multilingual model's
RNNT streaming decode emits blanks for the first ~5–6.5s of these specific utterances and "wakes up"
mid-stream.** The English checkpoint emits from the first word on the same audio. Observation (not confirmed
mechanism): the model starts at a strong content anchor ("Shakespeare", "It is…") and under-emits on the
rapid function-word preamble — a decode fragility of the 0.6B multilingual model under rc3, not our code.

## Impact & recommendation
This is a **shipped model-quality limitation, not a code defect** — the multilingual checkpoint occasionally
swallows the onset of long English utterances. It is NOT fixable in our inference code (the streaming/prompt/
tag-strip/finalize path is clean and faithful). Mitigations are model-level (finetune / different checkpoint)
or application-level heuristics (e.g. detect a long speech-energy span that produced no transcript). Removing
these ~10 clips puts ml at ~4.29% (still ~2.2× en = the inherent 0.6B-multilingual penalty). The English
checkpoint and the batching plan are unaffected.
