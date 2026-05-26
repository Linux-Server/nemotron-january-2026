# Opus review — 1.4 single-session e2e (Phase-1 exit gate)

Verdict: **Phase-1 exit gate MET-WITH-CAVEATS.** The native pipeline COMPOSITION (steady-AOTI + decode + finalize-bucket +
state machine + fork isolation) is token-exact to the verified Python reference `finalize_ref` over 200 corpus streams
(steady 200/200, final 200/200, FORK_ASSERT 200/200) — the hard, novel integration is solid. But "token-exact vs
finalize_ref" is narrower than the full T1 bar; the caveats below are the residual.

## 1. [MAJOR] Event-stream / delta / generation-suppression equivalence is NOT validated.
`session_main.cpp` asserts the CUMULATIVE final token sequence == finalize_ref gold. But the pinned T1 ship gate
(0.4-decision-FINAL §"Correctness bar") is "full-1000 WER within CI + per-utterance WER bounded + **exact per-session
event-stream / final-delta / generation-suppression equivalence**." The session neither emits nor checks the partial
emissions, the append-only delta (`_continuous_append_only_delta`, finalize_ref ~144-167 / server.py ~392-417), or
generation suppression. So 1.4 validates COMPUTE equivalence (the token sequence) but not EMIT-BEHAVIOR equivalence — a
runtime could emit a different partial/delta stream and still pass. FIX: extend the bundle gold + the session to check the
per-step emitted delta sequence (not just end-state cumulative tokens).

## 2. [MAJOR] Mel-fed, not audio-fed — the C++ audio-front is unexercised.
The session consumes a MEL bundle (`export_session_bundle.py`); the C++ audio->mel preprocessor AND the finalize
remainder-from-audio assembly (server.py mel/raw-ring math, finalize_ref.prepare_finalize_inputs) are NOT run in C++. So
"single native stream" is really "single native MEL-stream." Unvalidated in the session path: C++ preproc fidelity
(0.8 was validated standalone via .ts, not in the session), sample->mel alignment, the remainder assembly. FIX: wire the
C++ preproc + remainder-from-audio (a later 1.4 increment) and re-run the gate from audio.

## 3. [MAJOR] Multi-turn CONTINUOUS context-retention is untested.
`session_main` cold-resets per utterance (`reset_session` re-inits init_*), i.e. each bundle utterance is INDEPENDENT.
The continuous-context SPECULATIVE path — multiple turns within ONE stream, context RETAINED across a finalize
(server.py `_continuous_finish_speculative_finalize_locked` ~9014-9035; finalize_ref `_finish_speculative`) — is the
most production-relevant continuous behavior and is NOT exercised. FIX: add a multi-turn bundle (vad_stop mid-stream,
continue, finalize again) that exercises context retention + the speculative reset, and assert vs finalize_ref.

## 4. [MINOR] First steady chunk uses enc_first.ts (TorchScript trace, ~1e-5 drift), not a native AOTI bucket.
Token-exact here, and it's once-per-utterance (off hot path), so acceptable — but it's a TorchScript dependency in the
"pure native" path. Note it; a first-chunk AOTI bucket would make the path uniformly native if desired.

## 5. [MINOR] Oracle is finalize_ref (our Python reference), not the shipping server.
The chain is C++==finalize_ref (200) ∧ finalize_ref==NeMo (4 canaries, 1.3a). The actual production SERVER's emitted
output is not in the chain — "behavioral equivalence to Python" = equivalence to OUR reference, not to the shipping
server. Acceptable given finalize_ref is the carefully-validated spec, but the server-equivalence link is the weakest.

## 6. Composition correctness — looks sound.
FORK_ASSERT (caches+decoder+ring+emitted+hyp) runs after the finalize fork+decode and the parent is re-init per utterance;
AOTI steady outputs are threaded (contiguous) without aliasing the parent. 200/200 FORK_ASSERT pass + token-exact is
strong evidence the state threading + isolation are correct.

## Net / verdict
**MET-WITH-CAVEATS.** Achieved: the native compute pipeline (mel→steady-AOTI→decode→finalize-bucket, wired by the state
machine with fork isolation) is token-exact to the reference over 200 corpus streams — discharges the C++-corpus
validation (review B3) and proves the integration composes. Residual for a COMPLETE single-stream T1 gate: (1) event-stream
/ delta / suppression equivalence; (2) audio-front (mel-fed today); (3) multi-turn continuous context retention; (4) the
.ts first chunk; (5) server-vs-reference link. These are the Phase-1-finish / Phase-2-entry items, not defects in the
composition.
