# Codex review - round 1

## 1. Batching-Correctness Risks

The design is directionally plausible: NeMo's `conformer_stream_step` accepts a batch of encoder inputs and per-row RNNT hypotheses (`mixins.py:592-612`, `mixins.py:653-712`). If every row's state is stacked on the real batch dimension, every row has the same streaming geometry, and the decoder receives one independent hypothesis object per row, there should not be inherent cross-talk. As written, though, PLAN.md leaves several transcript-corrupting details underspecified or wrong.

The biggest concrete bug is the cache stacking axis. PLAN.md says to stack `cache_last_channel/time/len` on dim 0 (`proj-2026-05-21-0410/PLAN.md:104-109`). That is wrong for the encoder caches. `get_initial_cache_state(batch_size=1)` returns `cache_last_channel` as `[layers, B, cache_T, d_model]` and `cache_last_time` as `[layers, B, d_model, time_T]` (`/home/khkramer/src/nemotron-nano-omni/NeMo/nemo/collections/asr/modules/conformer_encoder.py:1087-1107`). The encoder later indexes the layer dimension first and expects batch inside each layer (`conformer_encoder.py:698-710`), then returns next caches with the same `[layers, B, ...]` layout (`conformer_encoder.py:768-776`). The batch helper must concatenate `cache_last_channel` and `cache_last_time` on dim 1, concatenate `cache_last_channel_len` on dim 0, and scatter with `cache[:, i:i+1, ...]`. Stacking cache tensors on dim 0 would mix layers and sessions and can silently poison all later chunks.

The RNNT hypothesis shape also needs to be made explicit. In the current B=1 path, `session.previous_hypotheses` is the list returned by NeMo, length 1 (`src/nemotron_speech/server.py:2494-2501`, `mixins.py:707-713`). A batched call must pass a flat list of row hypotheses, e.g. `[session.previous_hypotheses[0] or None for session in sessions]`, not `[session.previous_hypotheses for session in sessions]`. After the call, it should scatter `best_hyp[i]` back as a one-element list so the existing B=1 finalization and fork code still see the same shape. The same applies to `pred_out_stream`: scatter one row, not the full returned batch list. This should be a named invariant in Step 4, with assertions for list length and object identity.

Hypothesis aliasing is a real hazard. The batched label-looping decoder mutates the passed `partial_hypotheses` in place via `hyp.merge_(hyp_continuation)` (`rnnt_greedy_decoding.py:825-831`), and `Hypothesis.merge_` extends token/timestamp lists in place (`rnnt_utils.py:153-181`). That is fine only if every batch row receives a unique per-session hypothesis object. Probe B must include an alias-sentinel check: no two rows may share the same `Hypothesis` object, and row order changes (`[A, B]` vs `[B, A]`) must not change either stream's output or state.

`drop_extra_pre_encoded` is scalar and model-global during the encoder call, so "per-row drop" is not available. PLAN.md currently leaves this as "handle differing emitted_frames/drop_extra via per-row drop or uniform-tick invariant" (`PLAN.md:104-109`). NeMo temporarily writes the scalar into `self.streaming_cfg.drop_extra_pre_encoded` (`streaming.py:53-55`), and the encoder applies it uniformly to the whole batch when cache is present (`conformer_encoder.py:657-659`). The server uses `drop_extra=0` and `chunk_mel` width `shift_frames` for first chunks, but `drop_extra=self.drop_extra` and `chunk_mel=mel_frame_ring + valid_new_mel` for steady-state chunks (`server.py:2475-2484`). First chunks and steady-state chunks must not share one `conformer_stream_step` call unless a probe proves an alternative representation is byte-identical. The scheduler should group by at least `(target_lang, keep_all_outputs, drop_extra, chunk_T, decoder_mode)`.

The plan's ragged-audio wording is dangerous. Probe B says to "pad shorter streams / mask" while advancing each stream (`PLAN.md:87-94`), and Step 5 says gather sessions with `>= shift` pending audio (`PLAN.md:113-119`). The current ready predicate is stricter: `_handle_audio_locked` waits for `(emitted_frames + shift_frames + 1) * hop_samples` on the session timeline (`server.py:2410-2415`), and `_process_chunk` refuses to advance if `len(pending_audio) < preprocess_new_audio_samples` (`server.py:2437-2441`). Mid-stream rows with less than the exact next chunk must not be padded with zeros and advanced; that would feed synthetic silence as real audio and corrupt caches/transcripts. "Uniform tick" is sound only as: each row included in a given batch advances exactly one real shift. Streams that are not ready stay out of the batch and retain state. Streams with backlog should process one shift, scatter, then be requeued if still ready.

The decoder plan has a major selection bug. The server currently sets `strategy: greedy` (`server.py:793-817`). In NeMo, `strategy=greedy` constructs `GreedyRNNTInfer` (`rnnt_decoding.py:384-400`), while the batched label-looping implementation is only selected by `strategy=greedy_batch` (`rnnt_decoding.py:438-455`). Setting `loop_labels=True` alone, as Step 6 proposes (`PLAN.md:122-127`), will not select the intended batched decoder if the strategy remains `greedy`. Conversely, changing to `greedy_batch` is a behavior change and must be byte-gated at B=1 before it is mixed with scheduler work.

The multilingual prompt constraint is correctly identified in PLAN.md (`PLAN.md:60-62`), but it needs a harder invariant. `_apply_inference_prompt` calls `self.model.set_inference_prompt(...)`, which is model-global (`server.py:639-646`). A batched model call must contain only one `target_lang`, and the scheduler must set the prompt exactly once immediately before that batch. Mixed-language concurrency must be tested as separate same-language batches, not as one batch with per-row prompts.

Finalization should remain a separate geometry until proven otherwise. `_process_final_chunk` can build a variable-length final mel span, uses `keep_all_outputs=True`, and may process more than one internal preprocessor window before a single final `conformer_stream_step` (`server.py:2686-2775`). That does not match the normal one-shift scheduler geometry. Keeping final/fork B=1 is the right initial safety posture.

## 2. Probe Adequacy

Probe A and Probe B are necessary de-riskers, but they are not sufficient as written.

Probe A should test both first-chunk and steady-state shapes. The first live chunk uses no mel pre-encode cache and `drop_extra=0`; later chunks prepend `mel_frame_ring` and use `self.drop_extra` (`server.py:2475-2484`). `torch.compile(..., mode="reduce-overhead")` may specialize or recompile across those shapes and attribute guards. If the compiled path is only safe for steady-state, the plan should say first chunks/final chunks remain uncompiled or are warmed into steady-state before using the compiled path. Probe A should also test `NEMOTRON_WARMUP_MS=200`, since `_run_session_warmup` runs a real `conformer_stream_step` and seeds caches/hypotheses before real audio (`server.py:1017-1074`).

Probe A's gate should measure full live `conformer_stream_step` latency and, if possible, encoder-only latency. The production bottleneck is `conformer_stream_step` as a combined encoder+decoder cost (`proj-2026-05-20-modal-cost/batching-design-notes.md:5-8`); compiling only the encoder is not decisive if decode remains dominant. The gate should include p50/p95 steady-state ms after compile warmup, compile/capture warmup cost, memory growth, and "no repeated graph recapture/recompile" logs. A local 5090 >=20% speedup is useful, but Modal is where the cost problem is; at minimum Step 8 should separately report whether Phase 1 helps Modal.

Probe B should be expanded into a state-correctness probe, not only a final transcript diff. It should compare per-step cumulative text, final text, `cache_last_channel/time/len` shapes and row-local values, `previous_hypotheses` token sequences/last_token/decoder state, `pred_out_stream`, `emitted_frames`, and emitted interim deltas. A final transcript can match while caches are already wrong and ready to diverge on the next chunk.

Probe B needs explicit hard cases:

- Same clips run separately at B=1 vs batched at B=2/4, using real server preprocessing and `_init_session` state.
- Row-order permutation `[A, B]` vs `[B, A]`.
- First-chunk rows separate from steady-state rows; if mixed first+steady is attempted, it should be expected NO-GO unless proven otherwise because of scalar `drop_extra`.
- Warmed sessions with `NEMOTRON_WARMUP_MS=200`, matching the silence0_warm200 path.
- Ragged total arrival: one stream absent for a tick, then rejoins later. Do not pad a not-ready stream.
- Mixed-language concurrency for prompted models, proving same-language grouping and prompt switching between batches.
- Current `strategy=greedy` B>1, and proposed `strategy=greedy_batch, loop_labels=True, use_cuda_graph_decoder=False` at B=1 and B>1. If `greedy_batch` differs at B=1, Step 6 cannot claim byte-exactness.
- Preserve-alignments/frame-confidence mode if `NEMOTRON_EOU_PROBE=1` remains supported (`server.py:818-826`), or the batch scheduler should reject that config.

I would add a separate front-loaded decoder probe before scheduler work: "Probe C - decoder strategy equivalence and speed." It should compare current `strategy=greedy` vs `strategy=greedy_batch/loop_labels=True/use_cuda_graph_decoder=False` at B=1 on the fixed clip set, then compare B=2/4. Gate: byte-identical transcripts and materially faster decode, or do not switch decoder strategy. If it is not byte-identical, the plan must either keep current greedy decoding for rc1 byte-exactness or explicitly accept a baseline change with a separate product decision.

I would also add a scheduler-plumbing probe before real batching: run a shared scheduler with B=1 only, preserving current decoder/config, and prove byte-exactness plus no latency regression. This isolates the concurrency refactor from the batch-state hazard.

## 3. Completeness Gaps

The silence0_warm200 finalize/fork path needs a sharper concurrency contract. `_build_continuous_finalize_fork` deep-clones parent cache and hypothesis state (`server.py:2068-2118`), `_continuous_finalize_emit_locked` runs `_process_final_chunk` under `inference_lock` (`server.py:2242-2264`), and `FORK_ASSERT` checks the parent stayed unchanged (`server.py:2141-2181`). With a scheduler, a queued normal chunk and a fork finalization for the same session must not race. The plan should require a per-session generation/in-flight flag and a single model-call lane so final/fork, warmup, normal batches, and prompt switches serialize. Initial implementation should keep fork finalization B=1 outside the scheduler but behind the same model-call lock.

The asyncio scheduler design is under-specified relative to the current event ordering. Continuous mode has one per-session worker that processes queued events under `session.state_lock` (`server.py:1702-1729`) and calls `_handle_audio_locked`, which currently both mutates state and awaits inference/send (`server.py:1802-1827`, `server.py:2390-2435`). A ready-queue scheduler cannot simply mutate sessions later unless the plan defines ownership. It needs one of these designs: hold `state_lock` until a scheduler future resolves, or move all ASR-state mutation into the scheduler and make the session worker only append audio/enqueue control events. The second is probably cleaner, but it requires cancel/close/reset handling, session generation tokens, and bounded queue growth.

The ready queue must handle backlog and fairness. Current `_handle_audio_locked` drains while a session remains ready (`server.py:2414-2435`). A batch scheduler that processes only one chunk per session per drain must requeue sessions that still satisfy the exact ready predicate. It also needs to avoid one backlogged stream monopolizing every batch and needs telemetry for queue wait and per-session lag.

Latency is not just a performance detail; it is a correctness/product gate for streaming UX. Waiting to form a larger batch can increase TTFS and interim latency. Step 5 should define a max batch wait, likely a small configurable budget (`NEMOTRON_BATCH_MAX_WAIT_MS`) with immediate dispatch for N=1/low load. Validation should include N=1 with batching enabled, p50/p95/p99 TTFS, interim emission delay, and finalization delay against the existing <400 ms target, not just throughput knee.

The multilingual path needs validation beyond "group by target_lang." `_validate_connection_query` sets the session target at connect time (`server.py:1547-1560`), `_ensure_session_target_lang` fills defaults (`server.py:639-642`), and `_extract_hypothesis_text` strips prompt language tags (`server.py:733-751`). Tests should cover concurrent `en-US` plus another supported language, plus `auto`, and prove batches are split and prompt tags remain stripped per session.

The plan should explicitly fail closed for unsupported decoding modes. `NEMOTRON_DECODING=beam` builds a different decoding config (`server.py:793-807`); the proposed scheduler/state helper is RNNT-greedy-specific. If `NEMOTRON_BATCH_SCHED=1` is set with beam, an unsupported model type, CTC/hybrid mode, or EOU settings not covered by probes, the server should refuse to enable batching rather than silently taking a half-tested path.

Batching memory limits and batch caps are missing. Encoder caches are per-layer tensors with batch dimension inside the cache (`conformer_encoder.py:1087-1107`), and RNNT label-looping can allocate batched hypothesis buffers based on `max_time * max_symbols` (`rnnt_label_looping.py:301-315`). The scheduler needs a configurable max batch size, observability for GPU memory, and a defined policy for overflow sessions.

## 4. Flag/Byte-Exact Adequacy

Defaulting everything off is the right safety posture. The current path is validated and should remain byte-identical when `NEMOTRON_ENCODER_COMPILE` and `NEMOTRON_BATCH_SCHED` are unset (`PLAN.md:64-68`). The plan should go further and require that flag-off startup constructs the exact same decoding config as today: `strategy=greedy`, `loop_labels=False`, `use_cuda_graph_decoder=False` (`server.py:793-817`). Do not globally switch to `greedy_batch` or `loop_labels=True` during load unless the batch flag is on and the decoder probe has passed.

The byte-exact gate should be based on a fixed captured baseline artifact, not "run baseline again" each time. Add a Step 0 that records baseline transcripts, interim sequences, final deltas, and relevant env/model commit metadata for the fixed clip set. Every later flag-on path should diff against that artifact.

The gates must cover streaming outputs, not only final transcripts. A scheduler can produce the same final text while changing interim timing, duplicated deltas, append-only finalize behavior, or multilingual tag stripping. For rc1 English, compare interim cumulative transcript sequence, final transcript, final delta, and no duplicate emissions. For continuous mode, run with `NEMOTRON_FORK_ASSERT=1` and the silence0_warm200 env.

For prompted models, byte-exactness must be checked both with one language alone and mixed-language concurrency. Because the prompt is model-global (`server.py:644-646`), a mixed-language test is the only practical way to catch accidental prompt cross-talk.

Feature flags should fail closed. If the scheduler cannot form a safe batch because rows differ in `target_lang`, `drop_extra`, `chunk_T`, `keep_all_outputs`, decoder mode, or unsupported config, it should split into smaller safe batches or fall back to B=1 for those rows. It should never pad or coerce a row into a batch just to improve occupancy.

## 5. Sequencing

The high-level order is reasonable, but several steps are too broad or have missing dependencies.

Add Step 0 before both probes: capture the byte-exact baseline artifact and define the fixed clip/multilingual/finalize test set. Without this, each gate can drift.

Split Step 2 into at least two probes: one for state batching with the current decoder strategy, and one for decoder strategy equivalence (`greedy` vs `greedy_batch/loop_labels=True`). The decoder strategy decision must happen before Step 4 state primitives and Step 6, because it changes what `previous_hypotheses` contains and whether the decoder mutates hypotheses in place.

Step 4 should be only normal-chunk state primitives and should correct the cache axis, flat hypothesis list, scatter shape, grouping key, and exact ready predicate. It should not include scheduler behavior yet.

Step 5 is too big for one delegation. Split it into:

- 5a: shared scheduler infrastructure with B=1 only, same decoder, same `_process_chunk` semantics, byte-exact and latency-gated.
- 5b: fixed-geometry batching for normal steady-state chunks only, grouped by target language and drop/shape, with requeue/backlog handling.
- 5c: variable-B/session join-leave and row-order permutation hardening.

Step 6 is also too broad. Separate "batched decoder strategy" from "finalize/fork integration." Decoder strategy should be decided by the new decoder probe before scheduler implementation. Finalize/fork should first remain B=1 behind the same model-call serialization and only later get its own optional batching step if it matters for throughput.

Phase 1 compile can proceed independently after Probe A, but final validation must test compile and scheduler independently and together. Compile can change shape guards and graph capture behavior; batching changes B and prompt scheduling. Their interaction should be an explicit matrix in Step 7.

## 6. Prioritized PLAN.md Edits

1. Correct Step 4 cache stacking: `cache_last_channel` and `cache_last_time` concatenate/scatter on dim 1; only `cache_last_channel_len` concatenates on dim 0. Add assertions for resulting shapes.

2. Replace "set loop_labels=True" with "switch to `strategy=greedy_batch`, `loop_labels=True`, `use_cuda_graph_decoder=False` only behind the batch flag after a B=1 byte-exact decoder probe." Cite NeMo factory behavior.

3. Add a new decoder probe before scheduler work: current greedy vs greedy_batch/loop_labels at B=1 and B=2/4, with transcript and state comparison. If not byte-identical, keep current greedy or stop for a product decision.

4. Remove "per-row drop" from Step 4. State that `drop_extra_pre_encoded` is scalar/global for a call. Batch only rows with the same `(target_lang, keep_all_outputs, drop_extra, chunk_T, decoder_mode)`.

5. Tighten ragged handling: not-ready streams are excluded and not mutated; no mid-stream zero padding. Backlogged streams process one real shift per batch and are requeued if still ready.

6. Add Step 0 baseline capture: fixed clip set, interim sequence, final text/delta, multilingual prompted cases, silence0_warm200 finalize/fork, env/model/NeMo commit metadata.

7. Expand Probe B to include row-order permutation, first-vs-steady grouping, warmup=200, absent/rejoin ragged arrival, alias detection, mixed-language grouping, and per-step state diffs.

8. Split Step 5 into B=1 scheduler plumbing, fixed-geometry normal batching, then variable-B join/leave/backlog. Gate each separately for byte-exactness and latency.

9. Keep finalize/fork B=1 initially, but add explicit serialization with the scheduler and run `NEMOTRON_FORK_ASSERT=1`. Do not batch final chunks until a separate final-specific probe passes.

10. Add latency gates: N=1 batching-on must not regress TTFS materially, and p95/p99 TTFS/interim/final latency must stay under the streaming budget while throughput improves.

11. Add fail-closed config checks for beam decoding, unsupported model/decoder types, unprobed EOU preserve-alignments mode, and unsafe mixed grouping. Split/fallback to B=1 rather than coercing rows.

12. Add scheduler telemetry: batch size, grouping reason, queue wait, model ms, preprocess ms, per-session lag, fallback count, and prompt language per batch.

Verdict: the plan is not safe as written. The two front-loaded probes are the right idea, but the current text misses cache tensor axes, decoder strategy selection, scalar `drop_extra`, flat hypothesis scattering, and scheduler/finalize concurrency. Those are exactly the areas that can silently corrupt transcripts while still looking like a throughput improvement.
