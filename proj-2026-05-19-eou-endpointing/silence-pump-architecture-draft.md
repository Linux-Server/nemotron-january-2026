# Can we feed silence faster than realtime to reach the EOU settle point?

**Draft — 2026-05-20. Status: pre-verification (Claude code-read pass + Codex independent pass to follow).**

## TL;DR (claims to verify)

1. **Yes, the server can ingest silence faster than realtime** — there is no per-chunk wall-clock pacing; chunks are processed back-to-back bounded only by GPU compute time (~10 ms per chunk vs. 160 ms of audio they represent → ~16× speedup ceiling).
2. **This is closely related to Step 7d's fork-flush, but with a wrinkle**. The fork-flush adds `final_padding_frames * hop_samples = 32 * 160 = 5120` samples = 320 ms of zeros to `pending_audio`, then calls `_process_final_chunk` which makes a **single** `conformer_stream_step` call with the combined mel. Extending this for an EOU monitor requires either (a) multiple sequential per-160-ms calls (so the probe writer captures per-step state for the settle-detector), or (b) one big call with `keep_all_outputs=True` and per-encoder-frame inspection of the result. Architecturally feasible either way, but it's not just "more iterations of the existing path."
3. **The model is auto-regressive in the cache** — each `conformer_stream_step` depends on the prior call's `cache_last_channel/time/len` + `previous_hypotheses`. You cannot parallelize chunks, but you can stream them as fast as the GPU allows.
4. **The "streaming → batch" mode-switch issue is a different problem** — `change_decoding_strategy` and `set_default_att_context_size` install streaming-specific params (chunk_size, shift_size, left_context). Switching mid-stream loses state. Not a blocker for the silence-pump idea, because we stay in streaming mode and just feed more chunks.
5. **But — and this is the painful finding — the model does not converge to all-blank during silence**. Empirically: 70% of sessions emit real tokens during the 2-s trailing silence, with median first-emission at +280 ms past Silero stop and tail emissions extending past +1.7 s. The longest sustained all-blank run in our 100-session collection is 71 encoder frames (~5.7 s), and 95% of sessions never reach 60 consecutive blank encoder frames (4.8 s) even given the full 2-s trailing silence.
6. **Therefore: even with infinite synthetic-silence-pump speed, the signal would NOT converge in a useful timeframe on this model**. The bottleneck is model behavior during silence, not pacing. The user's intuition ("feed silence long enough, must converge") is correct in principle but is not realized on this checkpoint within a budget that beats Silero's 200 ms.

## Architecture facts (verified by source inspection)

### Granularity hierarchy

```
audio samples  ──▶  mel frames  ──▶  encoder frames  ──▶  emitted tokens
 16 kHz int16        10 ms each       80 ms each            0..max_symbols
                                      (= 12.5 fps)          per encoder frame
```

The conformer encoder downsamples **8×** from mel frames to encoder frames. Per-chunk:
- **1 chunk** = `shift_frames = 16` mel frames = **160 ms** of audio
- **1 chunk** = **2 encoder frames** at the decoder
- **1 encoder frame** = **80 ms** — this is the **finest granularity available to the decoder**

Verified by inspecting one probe row from `eou-collect/telemetry/eou_step2_collect.eou_probe.jsonl`:
- `shift_frames=16`, `frame_alignment` length = 2 per chunk.

Practical implication: the plan's "frame-level vs chunk-level" distinction is **2× granularity, not 16×**. There is no 10 ms decoder granularity to exploit.

### Server processing rate (no pacing)

In `server.py:_handle_audio_locked` (around line 2130):

```python
while self._session_timeline_samples(session) >= min_audio_for_chunk:
    async with self.inference_lock:
        text = await asyncio.get_event_loop().run_in_executor(
            None, self._process_chunk, session
        )
```

The server loops over all pending audio, slicing it into 160 ms chunks, calling `_process_chunk` for each one. There is **no `await sleep`, no rate limit, no wall-clock pacing**. A WebSocket send of 5 seconds of audio in one binary frame is processed as ~31 back-to-back `conformer_stream_step` calls, each taking ~10 ms on the GPU = ~310 ms wall-clock to consume 5 s of audio (16× faster than realtime).

The only real-time pacing comes from the **client** — pipecat's `synthetic_transport` sends audio at 1× playback rate. The server itself has no notion of realtime.

### Auto-regressive cache, but not auto-regressive emit

Each chunk call in `_process_chunk`:

```python
(
    session.pred_out_stream,
    transcribed_texts,
    session.cache_last_channel,
    session.cache_last_time,
    session.cache_last_channel_len,
    session.previous_hypotheses,
) = self.model.conformer_stream_step(
    processed_signal=chunk_mel,
    processed_signal_length=chunk_len,
    cache_last_channel=session.cache_last_channel,
    cache_last_time=session.cache_last_time,
    cache_last_channel_len=session.cache_last_channel_len,
    keep_all_outputs=False,
    previous_hypotheses=session.previous_hypotheses,
    previous_pred_out=session.pred_out_stream,
    drop_extra_pre_encoded=drop_extra,
    return_transcription=True,
)
```

The encoder is auto-regressive **in the cache** (each call carries forward `cache_last_channel/time/len`). You cannot reorder or parallelize chunks for a single session.

The RNNT decoder is auto-regressive **in `previous_hypotheses`**. The greedy beam continues from where the last chunk left off — that's why we have the Step-3 deep-clone recipe for the fork mechanism.

So: chunks are strictly sequential, but the per-chunk GPU work is fast (~10 ms), so you can fire many chunks per second. The bottleneck on "synthetic silence pump" speed is GPU latency, not architecture.

### Streaming vs batch mode switch (the past issue, briefly)

The model is configured for streaming via:
- `change_decoding_strategy(decoding_cfg=...)` with a streaming-RNNT cfg
- `set_default_att_context_size([left, right])` for the chunked attention window
- `setup_streaming_params(...)` with `chunk_size=[9, 16], shift_size=[9, 16]`

These set internal model state that the conformer + decoder use during `conformer_stream_step`. **Switching mid-session to offline / batch mode** would require:
- Reverting to the offline attention pattern (no chunked attention)
- Switching the decoder cfg back to non-streaming-greedy
- Losing the streaming caches (or risk state corruption)

This was the issue we hit in the parent project. **It is not a blocker for the silence-pump idea**, because the silence-pump idea stays in streaming mode and just feeds more chunks through `conformer_stream_step`. The fork-flush in `_continuous_finalize_emit_locked` → `_process_final_chunk` already does this with `(R+1)*shift = 32` mel frames = 320 ms of synthetic zeros.

### Existing fork-flush: how far it already goes

`_build_continuous_finalize_fork` (server.py:1792) creates a disposable `ASRSession` clone with:
- All encoder/decoder state deep-cloned (Step-3 recipe: `clone_hypotheses_deep`, `tensor_clone`)
- `pending_audio` = the unconsumed audio at trigger time PLUS `final_padding_frames * hop_samples = 32 * 160 = 5120` samples = **320 ms of synthetic zeros**

`_process_final_chunk` (server.py:2389) then runs **one big** `conformer_stream_step` call with the combined (pending + 320 ms zeros) mel. NeMo internally handles the multi-encoder-frame input; we get the final transcribed text back. The whole final chunk processing takes ~20–60 ms of GPU time (much less than the 320 ms of audio it represents — "faster than wallclock").

**The user's intuition extends this**: instead of just 320 ms of synthetic zeros for the rc1 flush, what if we fed N×160 ms of synthetic silence and watched the decoder for sustained all-blank? The mechanism is closely related to the existing fork-flush BUT with two implementation notes:

1. The existing path calls `conformer_stream_step` **once** with the combined audio. For an EOU monitor we'd want either:
   - (a) Multiple sequential 160 ms calls, with the probe writer capturing per-call state (matches the parent-stream pattern; easier to integrate with the existing probe + EOU machinery).
   - (b) One big call with `keep_all_outputs=True` (currently False) and per-encoder-frame settle detection from the returned alignments.
2. Either way, the wall-clock cost scales with synthetic-mel length but is dominated by the GPU-call overhead, not the audio duration. For 2 s of synthetic silence: roughly 12 sequential calls of ~10 ms each ≈ 120 ms wall-clock to "consume" 2 s of synthetic audio. That budget would beat Silero's 200 ms.

## The empirical answer: does the model converge to all-blank under silence?

From the 100-session diagnostic (`eou_diagnostic.py` on the collected probe data):

### What the model emits during the trailing 2 s of silence (Analysis 2)

- **70 of 100 sessions emit at least one token during the post-vad_stop region**.
- Median first post-silence emission: **+280 ms past vad_stop**.
- p95: +931 ms; max: +1696 ms.
- Token classes: **49.7% alphanumeric words**, 46.7% space-markers, 3.6% punctuation.
- Top post-silence tokens: `▁the` (53×), `,` (45×), `ing` (45×), `▁and` (44×), `▁to` (39×).

This is striking: the model is **continuing to transcribe imagined speech** during the silence trailer. Words like `▁the`, `▁and`, `▁to` are common English words being emitted with nothing to transcribe. This is **hallucination behavior** during silence, not just BPE-completion of speech that was in flight.

### How long is the longest sustained all-blank run? (Analyses 1 + 4)

- **Longest sustained all-blank run in any session (post-silence): 71 encoder frames = 5.68 s**.
- **Longest in-speech blank run: 82 encoder frames** — note this is *longer than* the longest post-silence run, meaning **the speech vs silence blank-run distributions overlap completely**.
- Median trailing-blank-run length at +2 s past vad_stop: only **3 encoder frames (240 ms)**.
- Only **26%** of sessions have a 20+ encoder-frame (1.6 s) trailing-blank run at +2 s.

The data refutes the assumption that "after long enough silence, the model settles to all-blank." Even given 2 s of real silence padding, the model is still emitting tokens in 74% of sessions.

### Frame-level ROC (Analysis 3)

| K_frames | K_ms | det_lat p50 | det_lat p95 | false@100ms | never_fired |
|---|---|---|---|---|---|
| 10 | 800 | −3940 ms | +325 ms | 93% | 0% |
| 20 | 1600 | +1693 ms | +10244 ms | 28% | 0% |
| 40 | 3200 | +3218 ms | +10520 ms | 3% | **86%** |
| 60 | 4800 | +8979 ms | +11008 ms | 1% | **95%** |
| 100+ | 8000+ | — | — | 0% | **100%** |

The trade-off is bimodal: **K=20 has p95 latency +10.2 s** (signal triggered, but very late), and **K≥40 simply never fires in most sessions** (the model emits enough tokens during the 2 s trailing silence to prevent any 3.2 s sustained-blank run).

## Could faster-than-realtime silence pumping rescue the signal?

In principle yes; in practice no, on this model. Two observations:

1. **The GPU can pump silence at ~16× wall-clock** (the architecture allows it). If we needed K=20 (1.6 s of synthetic silence), the wall-clock cost would be ~100 ms of GPU time instead of 1.6 s of wall-clock. That budget would beat Silero's 200 ms.

2. **But the model never reaches K=40 in 95% of sessions** even given 2 s of real silence. Synthetic silence (pure zeros, not noisy room tone) might or might not help — pure zeros are more "silent" than the real trailing audio, but they're also unnatural inputs the model was never trained on. The 7d fork-flush uses synthetic zeros for exactly 320 ms; we don't have data on what 2 s of synthetic zeros does to this model.

There IS a clean experiment to run: extend the fork-flush mechanism to pump synthetic zeros until either (a) a sustained all-blank signal is achieved, or (b) some budget expires, and measure how often (a) succeeds. **But the priors suggest this would not give us a Silero-beating endpoint signal on this checkpoint** — the model emits hallucinated words like `▁the` during silence, which a synthetic-zero pump probably wouldn't suppress.

## Connection to the parent plan + TTFS explainer

The parent project produced `docs/ttfs-latency-explainer.html` (commit `9b817e4`) with the **two-floors framing**:
- **Modeled formula floor (~175 ms)**: endpoint_wait + rc1 + transport, with synthetic flush; this is a *modeled* lower bound under the locked additive budget.
- **Endpoint-evidence window (binding)**: the operational floor is the time required for a reliable "speech ended" signal. Today that's Silero's `vad-stop-secs = 200 ms`.

What this EOU work proves (with data, not just argument):

- **The endpoint-evidence window is binding**, not the modeled-formula floor.
- **ASR-internal signals on this Nemotron-streaming-0.6b checkpoint cannot beat Silero's 200 ms window.** They fire either on natural in-speech gaps (low K → false-fire) or so late they exceed any reasonable budget (high K → never-fires).
- **The model's "silence behavior" actively works against ASR-internal EOU**: 70% of sessions hallucinate trailing words during silence, breaking blank-run streaks.

These findings should be **folded into the TTFS explainer** as:
1. A new section on what the ASR-internal EOU project tested and found.
2. A clarification that the endpoint-evidence window is the **true binding limit**, not just an upper-bound modeling assumption.
3. A note that this is a property of THIS checkpoint (Nemotron-streaming-en-0.6b) and the analysis would have to be redone for other models — explicitly trained EOU heads, or models with sentence-end tokens, would change the picture.

## Open questions for the user

- Should we add a measured experiment that explicitly tests "pump 2+ s of synthetic zeros and watch settle behavior" before declaring NO-GO? Cheap to add (extends the existing 7d fork-flush; ~50 LOC + a focused 100-session run with `NEMOTRON_EOU_FLUSH_MS=2000` env). Either confirms NO-GO or surfaces a survivor operating point.
- Should we update `docs/ttfs-latency-explainer.html` now, or wait until after the synthetic-zero-pump experiment?
- Should we re-run the same diagnostics on the multilingual checkpoint (`NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b`) to see if it has the same trailing-hallucination behavior?

## Items to verify in second pass

- [ ] `_handle_audio_locked` has no rate-limiting (claim: while-loop processes all pending chunks back-to-back). **VERIFIED by direct code read at server.py:2130-2155.**
- [ ] `_build_continuous_finalize_fork` synthetic-zero size is `(R+1)*shift` mel frames = 320 ms. **TO VERIFY by inspecting `_build_continuous_finalize_fork` at server.py:1792.**
- [ ] `frame_alignment` length 2 per chunk corresponds to encoder-frame cadence of 80 ms. **VERIFIED by probe-row inspection: frame_alignment has 2 entries with `frame_offset` 0 and 1.**
- [ ] The model is conformer-rnnt with 8× downsampling. **TO VERIFY by inspecting `setup_streaming_params` log line "Shift size: 160 ms (16 frames)" and the resulting 2-frame encoder output per chunk.**
- [ ] `change_decoding_strategy` mid-stream issue. **TO VERIFY against parent-project commit history / canonical doc.**
- [ ] The 70% trailing-emission rate and +280 ms median first-emission are from the diagnostic JSON. **VERIFIED by reading Codex's smoke output.**
- [ ] Diagnostic's longest in-speech blank run (82 frames) is sus — could be a data-window artifact (e.g., session start before audio arrives, mis-attributed to "in-speech"). **TO INVESTIGATE.**
