# Synthetic-pad tuning for forking finalization

**Date**: 2026-05-20. **Status**: follow-up after EOU NO-GO #1 closure
(`docs/eou-asr-internal-finding.md`). Same data, same instrumentation, same Step-7d
fork-flush mechanism — different question. **Revised after Codex review** with
corrected understanding of the actual server flow.

## Goal

Build the **fastest possible forking finalization that is robust**: at `vad_stop`,
emit a transcript that contains all the tokens the model would emit for the audio
up to `vad_stop`, while preserving the ability to continue inference on the audio
stream until a true reset happens (the unfork mechanic).

**Assumption (user-stated)**: with Silero `stop_secs = 200 ms`, phonetics of speech
are complete at `vad_stop`. The goal isn't to detect end-of-turn (Silero does that);
the goal is to drive the decoder to commit any pending tokens for audio it has
already seen, as fast as possible. The benchmark corpus may not be representative
of real-world voice-agent VAD behavior, so we explicitly do not gate on
debounce-safety telemetry analysis.

## What the current production flow actually does (corrected per Codex review)

Earlier draft of this doc described the 150 ms debounce as "150 ms of real-silence
audio fed to the ASR before fork-flush." **That is incorrect.** Tracing the actual
code:

- `NEMOTRON_FINALIZE_SILENCE_MS=150` (default) is parsed at server.py:478 and used
  as an `asyncio` debounce sleep at server.py:1499.
- **During the debounce window, incoming audio is HELD in
  `continuous_post_stop_audio` (server.py:1531) and NOT fed to the ASR.** If
  `vad_start` arrives during the debounce, finalization is cancelled and the held
  audio is flushed back into the stream. If the debounce expires, the held audio
  is discarded and fork-flush runs from `parent_state_at_vad_stop`.
- Fork-flush (`_build_continuous_finalize_fork` at server.py:1792, then
  `_process_final_chunk` at server.py:2389):
  - Deep-clones parent ASR state with `clone_hypotheses_deep` + `tensor_clone`.
  - Appends `final_padding_frames * hop_samples = 32 * 160 = 5120` samples = **320 ms
    of synthetic zeros** to `pending_audio`.
  - Single `conformer_stream_step` call with `keep_all_outputs=True`.
  - Returns transcript; parent state is byte-identical before and after (verified by
    `NEMOTRON_FORK_ASSERT`).

So the **150 ms is a VAD-cancellation safety window**, not a model-settle margin.
The fork's input is `parent_state_at_vad_stop + 320 ms synthetic zeros` regardless
of debounce value. The user has explicitly asked NOT to gate the question on
debounce-cancellation telemetry analysis — we focus on the fork's behavior given
that we have committed to finalize.

## What the 310 ms median first post-vad_stop emission was actually measuring

The earlier `eou_diagnostic.py` analysis looked at the parent stream's emissions
during the 2-second post-vad_stop audio tail in the benchmark. That trailing audio
is what the parent processes **if no finalization happens** (i.e., session ends
naturally on disconnect). It tells us how the model behaves on real audio after a
Silero stop event, **NOT what tokens the fork-flush at vad_stop would emit**.

Production fork-flush at `vad_stop + 150 ms` uses `parent_state_at_vad_stop` (audio
is held) + 320 ms synthetic zeros. That's a different input to the model than 2
seconds of trailing real audio.

The question the replay harness needs to answer: **given
`parent_state_at_vad_stop` as the starting point, how much synthetic pad makes
the fork's transcript stable** (i.e., adding more pad doesn't change the
transcript)?

## Hypothesis (not proven)

Streaming RNNT with right-context = 160 ms (rc1) has a built-in commitment lag —
tokens from the last speech chunk need a subsequent input chunk to fully commit
through the attention pipeline. The current 320 ms = `(R+1) * shift_frames` of
synthetic should be just enough to close the rc1 window for the last speech chunk
plus one buffer chunk. Whether that's actually enough to capture all the trailing
tokens the model would emit is empirically open.

Alternative mechanisms to keep alive in interpretation:
- Model bias toward continuation (RNNT trained to expect more speech).
- Attention/cache state quirks that cause delayed commitment.
- BPE word-completion effects that depend on the audio that arrives.

We don't need to settle which mechanism is dominant — we just need to find the
minimum synthetic pad that produces stable transcripts matching what production
produces (the warm200 baseline).

## The experiment (Option C, revised)

Offline replay harness over existing per-chunk state snapshots
(`eou-collect/snapshots/*.pt`), no new collection, no server start.

### Inputs

- The 100 per-session sessions in the existing collection.
- For each session, the snapshot at the chunk closest to `vad_stop` (i.e., the
  snapshot just before audio was held aside).
- Baseline production transcripts from `stt-benchmark/stt_benchmark_data/test_results.db`
  (not the telemetry JSONL — telemetry doesn't include text).

### Method

1. Load the NeMo model (same as `server.py`'s `load_model`) once. Configure
   streaming params identically.
2. For each session:
   - Identify the chunk closest to `vad_stop` (anchor via probe row's
     `monotonic_done` vs telemetry's `vad_stop`; map session_id ↔ benchmark_batch_index
     via temporal order, same as `oracle_roc.py`).
   - Load that snapshot `.pt` file (cache, hypotheses, rings, counters).
   - Move all tensors back to CUDA (recursive walk).
   - Reconstruct an `ASRSession` (text fields default to empty since snapshots don't
     include them).
3. For each `synthetic_pad_ms` in {320 (current), 480, 640, 800, 1200}:
   - Override `server.final_padding_frames = synthetic_pad_ms / 10` (mel frames).
   - Build the fork (`_build_continuous_finalize_fork`).
   - Run `_process_final_chunk` on the fork.
   - Record the resulting transcript.
4. Compare each (session, pad_length) transcript pairwise + against the production
   baseline.

### Acceptance criteria

- The replay at `pad=320 ms` (current production value), starting from
  `snapshot_at_vad_stop`, should produce transcripts that **match the recorded
  production baseline within edit-distance ε** (not byte-exact — CUDA + cuFFT
  nondeterminism is real, see [[cufft-stft-plan-size-nondeterminism]] memory). A
  reasonable target: median edit distance ≤ 2 chars, p95 ≤ 5 chars, no token-level
  word substitutions in the body.
- If the replay can't reproduce production at the baseline configuration, the
  harness is wrong; fix before trusting other cells.
- Per-pad: report (a) fraction of sessions byte-identical to production, (b)
  fraction within edit-distance ≤ 2, (c) fraction with WER-relevant token changes,
  (d) median + p95 edit distance.

### Convergence as the answer

"Smallest pad such that the transcript stops changing as pad grows" is the
direction. Concretely:

| pad_ms | sessions where pad_ms → pad_ms+160ms changes the transcript |
|---|---|
| 320 | X% |
| 480 | Y% |
| 640 | Z% |
| 800 | ... |
| 1200 | ~0% (target) |

If `Z%` is small (say <2%), then 640 ms is "enough" — going beyond doesn't help.
If even 1200 ms still has nontrivial change rate, the rc1 commit-pipeline framing
is incomplete and we need to look harder.

## What this experiment does NOT decide

- **Whether debounce can go to 0**: that's a product question (false-finalize
  acceptability), not answered here.
- **The right SILENCE_MS for production**: this experiment tunes FORK_PAD, not
  SILENCE_MS.
- **Authoritative WER vs production warm200**: that requires a measured benchmark
  run with the new `FORK_PAD_MS` env. This experiment gives a *prediction* via
  offline replay; the measured run would confirm.

## Scope

- ~300–400 LOC of harness in
  `proj-2026-05-19-eou-endpointing/silence_pad_replay.py` (project scratch).
- No changes to `server.py` for the experiment itself — the harness sets
  `final_padding_frames` on a server instance it builds.
- If results suggest a production change, a follow-up commit would add
  `NEMOTRON_FORK_PAD_MS` env wiring + a measured benchmark run (separate concern).

## Snapshot ↔ telemetry join (Codex flagged as a missing piece)

The snapshots have `session_id` and `chunk_index`. The client telemetry has
`benchmark_batch_index` and `vad_stop` wall-clock. The mapping is:

1. `benchmark_batch_index → sample_id` via `test_results.db` (`samples` table).
2. `sample_id → session_id` is the indirection we need to bridge. Since the
   client telemetry **lacks** a `session_id` field (Codex flagged this), we use
   temporal-order fallback: probe sessions sorted by first-chunk wall-clock vs
   telemetry sorted by `benchmark_batch_index`. Same approach as
   `oracle_roc.py:join_sessions`. Document this limitation.
3. `session_id + vad_stop → chunk_at_vad_stop`: walk the session's probe rows in
   `chunk_index` order, find the chunk with `monotonic_done` closest to (but ≤)
   `vad_stop`. That's the snapshot we want.

For ground-truth transcripts: read from `test_results.db`'s `transcriptions`
table (or whichever holds the recorded final text). Match by `sample_id`.

## What the harness must do (engineering checklist)

- [ ] Load NeMo model once via `server.py`'s `load_model` (reuses streaming cfg).
- [ ] Override `final_padding_frames` per replay cell.
- [ ] Recursive CUDA migration of snapshot tensors (mirror of `snapshot_tree_cpu`
  but to GPU).
- [ ] Reconstruct `ASRSession` with all required fields (defaults for text fields
  not in snapshot).
- [ ] Snapshot selection: walk probe-JSONL for the chunk with `monotonic_done`
  ≤ `vad_stop` (anchored against telemetry's first `vad_stop`).
- [ ] Run `_build_continuous_finalize_fork` and `_process_final_chunk` on the
  reconstructed session (server methods are reusable — instantiate `ASRServer`
  without starting the listener).
- [ ] Baseline transcripts from `test_results.db`, joined via temporal order.
- [ ] Sequential execution per session (parallelizing against one model risks
  state corruption; NeMo mutates streaming config during calls).
- [ ] Per-cell edit-distance + WER tabulation (use the existing semantic-WER
  tooling if it's importable; otherwise plain Levenshtein).
- [ ] Output: a single JSON results file + a human-readable summary table per pad
  length.
