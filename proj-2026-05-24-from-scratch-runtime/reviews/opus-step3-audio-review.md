# Opus review — 1.4b Step 3 (audio-fed C++ front)

Verdict: **audio-front equivalence MET at the correct bar.** From PCM through the native C++ preproc + raw-ring + remainder
recompute: token + event(text) + FORK_ASSERT exact (20/20), geometry-recompute exact (PASS), mel within-CI (max_abs 2.44e-4)
with downstream token-exactness. The mel non-byte-equality is the DOCUMENTED cuFFT finding, not a bug.

## 1. Mel non-byte-equality (17/1204 byte-equal, max_abs 2.44e-4) — EXPECTED + token-benign.
The project memory (`cufft-stft-plan-size-nondeterminism`) is explicit: the STREAMING preprocessor is NOT byte-equal on
CUDA — "use a constant FFT plan + validate by WER-within-CI, NEVER byte-equality." The 0.8 byte-exactness
(`validate_preproc.py`) was Python-eager vs Python-`.ts` IN ONE PROCESS (same cuFFT plan); the C++ session is libtorch in a
DIFFERENT process → cuFFT plan-selection heuristics differ → ~2.4e-4. That is the documented cross-process nondeterminism,
not a C++ preproc bug. The CORRECT bar is token/event exact (holds 20/20) + mel within-CI — both met. NOTE: the 2.44e-4
mel drift is SMALLER than the AOTI encoder drift (~1e-2) already shown token-benign over 1000 (E.2), so the argmax-margin
headroom is the same order, not worse. MINOR: the within_tol `atol` threshold should be DOCUMENTED as ≥ observed cuFFT
nondeterminism (2.4e-4) and ≪ a token-flipping margin, not an arbitrary constant — and ideally tied to validate_preproc's CI.

## 2. Geometry recompute — truly INDEPENDENT (review B6 satisfied).
The C++ recomputes padded_total/total_mel/remaining_frames/final_T from `pending_audio` + `hop_samples` + the geometry
constants (not a bundle-baked value) and asserts == Python gold (geometry=PASS). This is the real recompute B6 wanted —
an off-by-one in the sample→mel finalization would now FAIL, not hide in a pre-baked mel.

## 3. raw-ring / fixed-block / FORK_ASSERT scope — covered.
The C++ maintains `raw_audio_ring` + `pending_audio` + the fixed constant-plan block (constant_preprocess_samples,
first_preprocess_mel_frame, align-pad) mirroring finalize_ref. FORK_ASSERT now covers `raw_audio_ring` AND `pending_audio`
(float_vec_equal) — broader than before (closes part of session-FOLDED's lifecycle gap). Confirm in the paired check:
multi-turn audio (retained raw-ring across speculative finalize) — is it run/asserted, and are total_audio_samples/emitted
threaded for turn N+1?

## 4. Chunk boundaries from PCM.
The C++ must reproduce finalize_ref's steady-drain boundaries (_session_ready: preprocess_new_audio_samples + ready
predicate) so the SAME chunks are produced from PCM — the 20/20 per-chunk mel/event alignment is evidence it does (a
boundary mismatch would misalign the per-chunk mel comparison and fail). Good.

## Net
Step 3 MET: native audio-fed front, token/event/FORK/geometry exact from PCM, mel within-CI per the cuFFT finding (the
right bar). Residuals are MINOR: justify+document the within_tol threshold (tie to validate_preproc CI); confirm multi-turn
audio retained-raw-ring is exercised. No blocker. The "single native AUDIO stream" claim (not just mel stream) is now
substantiated — closing session-FOLDED B5.
