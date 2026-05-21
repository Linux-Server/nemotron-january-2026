# Claude direct review — round 1 (PLAN.md)

My own critical review, in parallel with the Codex review. Folded together in round-1-folded.

## 1. CRITICAL: `loop_labels=True` likely CHANGES transcripts (byte-exact violation)
The server runs greedy `loop_labels: False` (server.py:814). Step 6 switches the batched decode to
`loop_labels: True` (the batched label-looping computer). **These are different greedy algorithms** —
label-looping vs the frame-looping path can emit different token sequences (different max_symbols /
inner-loop handling), so the transcript may NOT be byte-identical to today's output. This silently
breaks the byte-exact gate against the *current* baseline.
**Fix:** ADD a probe (new Step ~2b): at B=1, compare `loop_labels=True` vs `loop_labels=False`
transcripts byte-for-byte on the fixed clip set. If they differ, the byte-exact baseline must be
re-defined as "loop_labels=True B=1" (and we must accept a one-time transcript change + re-validate WER),
OR we keep loop_labels=False and find a batched path that matches it. Decide this BEFORE the scheduler.

## 2. CRITICAL: per-stream chunk geometry can't naively stack (`drop_extra_pre_encoded` is scalar)
`conformer_stream_step(..., drop_extra_pre_encoded=drop_extra)` takes ONE scalar drop_extra for the
whole call (mixins.py:660). But first-chunk streams use `drop_extra=0` and a chunk_mel of width
`shift_frames`, while steady-state streams use `drop_extra=self.drop_extra` and chunk_mel =
`cat(mel_frame_ring, valid_new_mel)` (server.py:2462-2469) — **different T AND different drop_extra**.
So you CANNOT stack a first-chunk stream with a steady-state stream in one call.
**Fix:** the scheduler must batch only streams that share (drop_extra, chunk-T) — i.e. group
first-chunk streams separately from steady-state, or warm every new stream to steady-state before it
joins the main batch. Step 4 hand-waves "per-row drop or uniform-tick invariant" — per-row drop is
impossible (scalar). Make this explicit; Probe B (Step 2) MUST test first-chunk + steady-state mixing.

## 3. Probe B must cover the hard cases, not just "ragged length"
Step 2's gate should explicitly include: (a) first-chunk + steady-state in the same batch (per #2),
(b) different `target_lang` in the same batch must be FORBIDDEN (the prompt is model-global,
server.py:646 — batching mixed languages corrupts output), (c) hypothesis in-place mutation across the
batch (does the batched `rnnt_decoder_predictions_tensor` mutate `previous_hypotheses` per-row cleanly?
`clone_hypotheses_deep` exists at server.py:139 — confirm the batched path clones correctly).

## 4. Latency/TTFS tradeoff of tick-batching (the <400ms budget)
Batching collects ready streams each tick → a chunk may wait up to one tick + batch-assembly before
inference. The current path processes each chunk immediately under the lock. Tick batching can ADD
per-chunk + interim latency. Step 5 must bound the tick interval (≈ the chunk cadence, ~80-160ms) and
the validation must confirm TTFS stays within the silence0_warm200 budget (the keep-up sweep measures
proc-lag, but also check single-stream TTFS doesn't regress when batching is on with N=1).

## 5. Scheduler refactor is the riskiest change — Step 5 may be too big
Today each WS handler acquires `inference_lock` and runs `_handle_audio_locked`→`_process_chunk`
(server.py:2390/2429). Moving to a shared ready-queue + single drain/scheduler task is a substantial
concurrency-model change (back-pressure, fairness, error isolation per stream, finalize interleaving).
Consider SPLITTING Step 5: (5a) introduce the scheduler infra running B=1 per tick (no batching yet) —
prove it's byte-exact + no latency regression vs the per-handler path; (5b) switch the scheduler to
batched calls. This isolates the concurrency refactor from the batching correctness.

## 6. Finalize/fork: confirm B=1-outside-batch is safe + lock coordination
Step 6 says finalize may run B=1 outside the batch — good (it's a per-stream one-shot, not the hot
path). But specify: the fork's B=1 `conformer_stream_step` and the scheduler's batched call share the
model + lock — ensure they serialize correctly (no concurrent model calls). FORK_ASSERT must stay clean.

## 7. Minor / additions
- Step 1 (encoder compile): `mode="reduce-overhead"` uses CUDA graphs → static shapes required; the
  chunk shape is fixed (constant-plan) so OK, but the FIRST-chunk shape differs → compile may recompile
  or fail on the first chunk. Probe A should test both first-chunk and steady-state shapes.
- Add an explicit "baseline capture" step/sub-step: record the single-stream baseline transcripts for
  the fixed clip set ONCE (committed), so every byte-exact gate diffs against a stable reference.
- Modal re-sweep (Step 8): the B200 GPU failed to deploy (scarce/Blackwell) — exclude it; RTX-PRO-6000
  needs the patient 600s smoke (cold-start). Note these so the re-sweep driver doesn't trip.

## Verdict
The two-phase structure + front-loaded probes are sound. The plan UNDER-specifies three correctness
hazards that could silently corrupt transcripts: **loop_labels change (#1), drop_extra/chunk-shape
stacking (#2), and mixed-target_lang batching (#3)**. Add a loop_labels byte-exact probe, make Probe B
cover first-chunk+steady-state and same-lang grouping, and split the scheduler step (infra vs batching).
With those, GO.
