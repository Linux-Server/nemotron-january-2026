# Multilingual checkpoint — elevated-WER deep dive (front-drop finding)

**Date:** 2026-05-21  **Status:** LOGGED (not yet root-caused/fixed — by decision, to keep the batching
`/implement` loop moving). **Scope:** the SHIPPED multilingual path (`proj-2026-05-20-1947`), not the
batching plan.

## Question
The multilingual checkpoint (`NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`) scored **4.84% pooled
WER** on the 1000-sample English benchmark vs **1.95%** for the dedicated English checkpoint (~2.4×). We
significantly modified the inference path for this checkpoint (prompted path / `set_inference_prompt`,
att_context `[56,3]`, **320ms shift + 1280ms final padding** vs en 160ms/16, tag-stripping, finalize/fork).
Is the elevated WER genuine model quality, or an artifact of the modified inference code?

Both runs used `language=en-US` explicitly (not auto) — verified in the run config + server log
(`Inference prompt set to 'en-US'` per session). So this is the model's English-*prompted* performance.

## TL;DR
**~81% of the en→ml gap is genuine model quality; ~19% is a real, ml-specific inference bug** (front-of-
utterance dropping on ~10 clips). The modified streaming/prompt/tag-strip path is otherwise clean.

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

## Root cause (hypothesis — NOT yet confirmed)
A pause/disfluency mid-clip likely triggers a finalize + cold-reset in the continuous/silence0 path, and the
first segment's text is lost from the captured finalize delta (the harness records only `is_final&&finalize`
deltas, not interims). To confirm: re-stream `a934808b` through the ml server (EA venv,
`/tmp/ml-nemo-path`, rc3, silence0_warm200) and capture the raw interim + finalize message stream + server
log — does the front appear as interims but not in a finalize delta (delta bug), or never at all (streaming
dropout)? Needs the GPU.

## Impact & recommendation
This is a **shipped** bug: long utterances with an internal pause get their front truncated in production —
a voice-agent dealbreaker (drops the first half of a user's sentence). Fix would take the English-set WER
4.84% → ~4.29% and, more importantly, stop truncating real utterances. Worth a dedicated fix pass on the
multilingual finalize/segment path (independent of the batching plan).
