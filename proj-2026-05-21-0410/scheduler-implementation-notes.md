# Scheduler / `_process_batch` implementation recipe (Steps 6-7 acceleration)

Concrete implementation guidance for the in-server batching, written while the analysis is fresh
(all probes GO). Composes the **validated** pieces: `batch_primitives` (Step 5, unit+model-tested) +
the existing `_process_chunk` per-session logic (`server.py:2429`). For the `/implement` loop.

## The key decomposition
`_process_chunk(session)` today does, per session: **(a) preprocess** (build fixed audio → mel →
`valid_new_mel` → `chunk_mel = cat(mel_frame_ring, valid_new_mel)`, server.py:2442-2471) → **(b) the
inference** (`conformer_stream_step`, 2484) → **(c) post-step** (advance `raw_audio_ring`,
`pending_audio`, `mel_frame_ring`, `emitted_frames`; extract text; 2497-2517).

Only **(b)** is batched. So:

```
_process_batch(sessions):                 # sessions = a ready group, same batch_group_key
    # (a) per-session preprocess (UNCHANGED per-session logic, just collected)
    chunk_mels, drops, Ts = [], set(), set()
    for s in sessions:
        new_audio = s.pending_audio[:preprocess_new_audio_samples]
        fixed_audio, valid = _build_fixed_preprocess_audio(s.raw_audio_ring, new_audio)
        mel, _ = _preprocess_fixed_audio(fixed_audio, valid)
        valid_new_mel = mel[:, :, first_mel_frame : first_mel_frame + shift_frames]
        if s.emitted_frames == 0: chunk_mel, drop = valid_new_mel, 0
        else: chunk_mel, drop = cat(s.mel_frame_ring, valid_new_mel), self.drop_extra
        chunk_mels.append(chunk_mel); drops.add(drop); Ts.add(chunk_mel.shape[-1])
        s._valid_new_mel = valid_new_mel     # stash for (c)
    assert len(drops) == 1 and len(Ts) == 1   # grouping invariant (else split — fail-closed)

    # (b) ONE batched inference (batch_primitives + try/finally)
    if self.prompted_model: _apply_inference_prompt(sessions[0])   # group shares target_lang; set once
    processed, lengths = stack_processed(chunk_mels)
    clc, clt, clcl = stack_caches([(s.cache_last_channel, s.cache_last_time, s.cache_last_channel_len) for s in sessions])
    prev_hyps = stack_hypotheses([s.previous_hypotheses for s in sessions])
    prev_preds = stack_pred_out([s.pred_out_stream for s in sessions])
    saved_drop = self.model.encoder.streaming_cfg.drop_extra_pre_encoded
    try:
        pred_out, txts, clc, clt, clcl, best_hyp = self.model.conformer_stream_step(
            processed_signal=processed, processed_signal_length=lengths,
            cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
            keep_all_outputs=False, previous_hypotheses=prev_hyps, previous_pred_out=prev_preds,
            drop_extra_pre_encoded=drops.pop(), return_transcription=True)
    finally:
        self.model.encoder.streaming_cfg.drop_extra_pre_encoded = saved_drop   # NeMo doesn't restore on exception

    # (c) per-session scatter + post-step (UNCHANGED per-session advance logic)
    for i, s in enumerate(sessions):
        s.cache_last_channel, s.cache_last_time, s.cache_last_channel_len = scatter_cache_row(clc, clt, clcl, i)
        s.previous_hypotheses = [best_hyp[i]]
        s.pred_out_stream = [pred_out[i]]
        # ...identical to _process_chunk 2497-2511: advance raw_audio_ring, pending_audio,
        #    _update_mel_frame_ring(s, s._valid_new_mel), s.emitted_frames += shift_frames
        text = _extract_hypothesis_text(txts[i]) if (txts and txts[i]) else s.current_text
        # emit append-only delta per session (existing logic)
```

## Gotchas (validated / flagged)
- **Cache axes**: `stack_caches`/`scatter_cache_row` already do dim1 (channel/time) + dim0 (len). DON'T
  hand-roll. Probe B proved byte-identical incl. mid-stream stacking.
- **Grouping**: only batch sessions with identical `(target_lang, keep_all_outputs=False, drop, chunk_T)`.
  First-chunk (`emitted_frames==0`, drop=0) MUST be a separate group from steady (drop=`self.drop_extra`).
  `_batch_group_key` exists; the scheduler groups by it, splits/fails-closed otherwise.
- **Prompt**: model-global → one `target_lang` per group, `_apply_inference_prompt` once before the call.
- **`drop_extra_pre_encoded`**: NeMo restores it WITHOUT try/finally (streaming.py:53-74) → wrap the
  call in try/finally (above) so an exception/OOM can't poison the next call.
- **`greedy_batch`**: Probe C GO → set `strategy=greedy_batch` (loop_labels=True, cuda_graph=False) under
  the batch flag for the batched decode; keep `greedy` for flag-off + finalize/fork (B=1).
- **mel ring per-session**: `_update_mel_frame_ring` + `_valid_new_mel` are PER-SESSION — keep them in (c),
  not shared. The batch only shares the inference call.

## The scheduler (Step 6, the harder concurrency part — needs care + byte-exact testing)
- Single drain task owns ASR-state mutation; WS handlers only enqueue audio/control. Per-session
  generation token; single model-call lane (normal batch / warmup / finalize-fork serialize).
- Dispatch: on first-ready start `NEMOTRON_BATCH_MAX_WAIT_MS` (default 5) timer; fire the largest safe
  same-group batch when timer elapses / `NEMOTRON_BATCH_MAX_SIZE` (default 4) hit / all-ready gathered;
  immediate at N=1. Deduped ready set (session IDs). No frame drop (bounded awaited put). Requeue backlog.
- Finalize/fork stays B=1 on the lane; FORK_ASSERT clean; final p95 <400ms (incl. clone cost).
- **Gate every increment against `baseline/english_baseline.json`** (interim seq + final + delta), with
  `NEMOTRON_FORK_ASSERT=1`, flag-off == byte-identical to today.
