# Independent Opus review — finalize_ref.py / export_finalize_encoder.py (paired with codex-finalize-ref-review.md)

Reviewer: independent Opus agent (model=opus), reasoning from server.py directly. Verdict: the finalize remainder
arithmetic + delta logic are FAITHFUL to server.py; the divergences are all in post-finalize state-machine semantics +
the validation harness.

## 1. [BLOCKER] Speculative finish copies the fork's finalized state back into the live session — the server does NOT.
`_finish_speculative` (finalize_ref.py:817-827) → `_copy_finalized_fork_state` (799-815) overwrites the live session's
caches/decoder_state/pred_out_stream/mel_frame_ring/emitted_frames/raw_audio_ring/pending_audio/total_audio_samples/
hyp_tokens with the FORK's post-finalize values. Server does the opposite: the finalize `_process_final_chunk(fork,...)`
mutates only the fork (server.py:9981-9988 where `session`=fork); then `_assert_fork_flush_parent_unchanged` asserts the
live session byte-identical (8920-8921); then `_continuous_finish_speculative_finalize_locked` (9014-9035) updates ONLY
committed_text/continuous_emitted_text + clears debounce — never caches/decoder/mel/emitted/raw/pending. The fork (with
`final_padding_frames`=32 trailing silence + keep_all_outputs tail) is DISCARDED. Copying it back injects synthetic
padding/look-ahead into the ongoing continuous stream → wrong left-context for turn N+1. This inverts the whole point of
the fork/FORK_ASSERT isolation. FIX: `_finish_speculative` must NOT copy fork state; mirror 9014-9035. (Spec line :42
"advance from the fork's final state IF committed" is misleading and seeded this bug — correct it.)

## 2. [BLOCKER] The reset/resume test cannot catch #1, and its PASS target is wrong.
Speculative test (964-983) asserts `norm(spec_b.delta_text)==norm(full_b)` where `full_b`=standalone full decode of
turn_b — a target INDEPENDENT of the carried state, so it passes under BOTH the buggy copy-back and the correct
keep-parent behavior. Structurally incapable of distinguishing them. FIX: assert post-finalize live-session tensors ==
the PRE-finalize snapshot (reuse the FORK_ASSERT snapshot across `_finish_speculative`), or compare turn-2 against an
independent two-turn streaming reference.

## 3. [BLOCKER] The validation oracle (`full_greedy_tokens`) is the wrong target.
`full_greedy_tokens` (856-864) = `model.forward(entire audio)` (one offline encode) + ref_greedy. The canary passes on
`exact OR wer_equiv` (931) where wer_equiv is normalized-text equality → the gate validates "chunked streaming+finalize
within whisper-normalized WER of offline," NOT finalize token-correctness. The chunked-streaming encode ≠ offline encode
on CUDA. Per spec :51 the oracle must be NeMo's OWN cache-aware streaming + keep_all_outputs finalize (partial_hypotheses
continuation), compared by TOKENS. FIX: build the streaming+finalize oracle (reuse stream_decode.py for steady + a NeMo
finalize for the tail); separate the assertions and fail loudly when `exact` is false instead of masking with wer_equiv.

## 4. [MAJOR] FORK_ASSERT runs BEFORE the copy-back and omits numpy/scalar fields.
The assert (762) passes at its call point, but `_finish_speculative` (819-827) then rewrites all the parent state it
checked + the uncovered numpy fields (pending_audio/raw_audio_ring/mel_frame_ring/emitted_frames/total_audio_samples).
The snapshot (572-580) matches the server's limited snapshot (faithful) but, given this ref DOES mutate the parent after,
the assert must move to AFTER the speculative finish and cover the numpy/scalar fields too.

## 5. [MAJOR] `total_audio_samples` corrupted by copy-back → breaks `_session_ready` for turn N+1.
`_copy_finalized_fork_state` sets `total_audio_samples=fork.total_audio_samples` (813), inflated by `padding_samples`
(626) = 32×160=5120 samples never received. `_session_ready` (360-367) over-counts the timeline → turn N+1's first steady
chunk readiness fires too early. Fixed by removing the copy-back (#1).

## 6. [MAJOR] vad_start cancelling PENDING_FINALIZE is never tested — the central reason the fork exists.
`vad_start` (399-412) is implemented but no gate test exercises PENDING_FINALIZE→STREAMING cancel. ADD: append, vad_stop,
append post-stop audio, vad_start (assert STREAMING, post-stop flushed back to pending, parent ASR state byte-identical),
then continue + compare to no-interruption reference.

## 7-9. [MINOR]
- (7) `debounce_expire` (414-421) has no stale-`continuous_stop_seq` guard (server.py:6964-6967); benign single-stream but
  unmodeled — document or represent for the C++ port.
- (8) Server has TWO true-boundary finishers: cold reset (`_init_session`, for `end`, 9079-9121) AND close-cleanup (caches
  →None, NO model re-init, for `close`, 9037-9077). The ref models only cold reset — call out if `close` is out of scope.
- (9) export_finalize_encoder "first" fixture sets pending=wav, emitted_frames=0 (bypasses drain) → exercises the real
  drop_extra=0 first-chunk geometry but is NOT a reachable production state for a non-trivial utterance; add a comment.

## Net
Remainder math (frame counts, multi-chunk mel collection, drop_extra 0-vs-2, raw-ring advance, final padding,
_update_mel_frame_ring) and _continuous_append_only_delta are FAITHFUL (server.py 8009-8082 / 9900-9956 / 392-417). All
divergences are post-finalize state-machine semantics + the validation harness.
