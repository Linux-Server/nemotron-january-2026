# finalize_ref.py paired-review FOLD (authoritative fix list)

Three independent reviews: my own (Opus 4.7 inline), `opus-finalize-ref-review.md` (independent Opus agent),
`codex-finalize-ref-review.md` (Codex). High convergence. The finalize remainder ARITHMETIC + `_continuous_append_only_delta`
are confirmed FAITHFUL to server.py by all three; every defect is in post-finalize state-machine semantics + the
validation/export harness. Codex#1 caught a real BLOCKER (#3 below) the Opus passes missed.

## BLOCKERs (fix before the C++ port)
- **B1 — Wrong validation/export oracle.** `full_greedy_tokens` = `model.forward(whole audio)`+ref_greedy (offline encode)
  gated on `exact OR wer_equiv` (normalized text). Validates WER-closeness, not finalize token-correctness. FIX: add a
  NeMo CHUNKED streaming + keep_all_outputs finalize oracle — steady loop `cache_aware_stream_step(keep_all_outputs=False)`
  + finalize remainder `keep_all_outputs=True`, decode each via `model.decoding.rnnt_decoder_predictions_tensor(..., partial_hypotheses=prev)` (the stream_decode.py pattern). Require TOKEN-EXACT vs this oracle; keep offline-full only as a diagnostic. (Codex#1, Opus#3, me#1)
- **B2 — Speculative finish copies the disposable fork back into the live session.** `_finish_speculative`→`_copy_finalized_fork_state` (finalize_ref.py:799-824) overwrites live caches/decoder/pred/mel-ring/emitted/raw/pending/total_audio_samples. Server `_continuous_finish_speculative_finalize_locked` (server.py:9014-9035) only updates committed/emitted text + clears debounce; the fork (with final_padding silence + keep_all_outputs tail) is DISCARDED, parent keeps pre-finalize STREAMING state. FIX: delete the copy-back; `_finish_speculative` sets STREAMING, clears debounce/reset flags, updates committed_text/continuous_emitted_text only. Fix the misleading spec line (1.3-finalize-spec.md §5 "advance from the fork's final state"). (Codex#2, Opus#1, me#2)
- **B3 — Post-stop audio stranded after a debounce finalize.** While PENDING_FINALIZE, `append_audio` buffers into
  `continuous_post_stop_audio` (finalize_ref.py:373-377). After debounce→STREAMING, the next `append_audio` never flushes
  the held buffer (only `vad_start` does). Server flushes on stream resume (server.py:6688-6709). FIX: in `append_audio`,
  when STREAMING and held post-stop audio nonempty, move it to pending_audio (+total_audio_samples), drain, then append. (Codex#3)

## MAJORs (the tests/assertions that guard the BLOCKERs)
- **M1 — FORK_ASSERT runs BEFORE the copy-back and omits numpy/scalar fields.** Move a parent-unchanged assert to AFTER
  `_finish_speculative`; cover pending_audio/raw_audio_ring/mel_frame_ring/emitted_frames/total_audio_samples too. This is
  the assertion that catches B2. (Codex#4, Opus#4)
- **M2 — reset/resume test too weak.** Compares normalized delta to standalone offline full_b → passes under both buggy and
  correct behavior. FIX: snapshot live ASR state before speculative finalize, assert byte-identical after; then continue a
  segment and compare next steady chunk + next finalize to the B1 chunked oracle. (Codex#7, Opus#2)
- **M3 — vad_start-cancels-PENDING_FINALIZE untested** (the reason the fork exists). FIX: vad_stop → append post-stop →
  vad_start before debounce → assert no emit, held audio flushed, parent byte-identical, stream matches no-stop oracle. (Codex#8, Opus#6)
- **M4 — Exporter gold is self-referential** (gold = the code under review) and `main()` returns success unconditionally →
  the C++ port could inherit the bug. FIX: export gold from the B1 chunked oracle, FAIL if finalize_ref differs; rename
  offline tokens `offline_full_greedy_tokens` (diagnostic only). (Codex#6)
- **M5 — Exporter "first" finalize geometry unreachable** (emitted_frames==0 for a full utterance; production drains steady
  first). FIX: for first-finalize fixtures clip to < preprocess_new_audio_samples, else build via append_audio. (Codex#5, Opus#9)
- **M6 — Remainder math only spot-checked at 4 indices.** FIX: residual-frame grid (0..shift_frames, emitted==0 and
  continuation, boundary ±1); assert remaining_frames/chunk_mel.shape[-1]/drop_extra/tokens vs the oracle per case. (Codex#9)

## Deferred / note (lower priority, carry to the C++ port)
- clone_tree/assert_tree_equal lack a generic-NeMo-object branch (Codex#10) — becomes relevant once B1's partial_hypotheses
  oracle is wired; port server.py's clone_hypotheses_deep / NeMo __dict__ clone or fail-loud on unclonable state.
- TorchScript trace not a dynamic-T proof + output arity unchecked (Codex#11) — already known (we use torch.export/AOTI for
  the steady encoder); apply same to finalize, add residual-T grid + arity check, `--require-trace`.
- Geometry constants printed not asserted (Codex#12); hard-coded RIGHT_CONTEXT/FINALIZE_SILENCE_MS/att-context (Codex#13) —
  assert against the targeted server config or document the intentional pin.
- close-vs-cold-reset: server has two true-boundary finishers (cold-reset `_init_session` for `end`; close-cleanup caches→None
  for `close`); ref models only cold-reset — fine if end-only scope, document it. (Opus#8)

## Iteration plan
Fix pass (Codex): B1 (chunked oracle, shared by gate+export), B2 (delete copy-back), B3 (post-stop flush), M1-M3 (guarding
tests), M4-M5 (export gold + reachable fixtures), M6 (boundary grid). Then re-test (token-exact vs the new oracle; parent
byte-identical after speculative finish; vad_start-cancel; boundary grid) and a follow-up paired re-review.
