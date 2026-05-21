# Codex review - round 2

## 1. Round-1-fold verification

The round-1 critical correctness fixes are mostly incorporated correctly.

- **Cache axis:** fixed. PLAN.md now states that `cache_last_channel` and `cache_last_time` batch on dim 1 and `cache_last_channel_len` batches on dim 0, including scatter as `[:, i:i+1, ...]` and `[i:i+1]` (`proj-2026-05-21-0410/PLAN.md:24-29`, `:74-75`, `:130-132`, `:152-155`). That matches NeMo: initial caches are `[layers, B, cache_T, d_model]`, `[layers, B, d_model, time_T]`, `[B]` (`conformer_encoder.py:1087-1125`), layers index cache as `cache_last_channel[lth]`/`cache_last_time[lth]` (`conformer_encoder.py:696-710`), and returns the same stacked layout (`conformer_encoder.py:768-776`).
- **Decoder strategy:** fixed. PLAN.md now names `strategy=greedy_batch`, not just `loop_labels=True`, and gates it behind Probe C B=1 byte/state equivalence (`PLAN.md:34-36`, `:87-90`, `:139-144`, `:182-184`). This matches the factory: `strategy=greedy` builds `GreedyRNNTInfer` (`rnnt_decoding.py:384-400`), while `strategy=greedy_batch` builds `GreedyBatchedRNNTInfer` with `loop_labels` and `use_cuda_graph_decoder` options (`rnnt_decoding.py:438-455`).
- **Hypothesis shape/aliasing:** fixed. PLAN.md now requires the flat per-row list, unique hypothesis objects, one-element scatter back, alias sentinels, and row-order permutation tests (`PLAN.md:76-78`, `:129-137`, `:152-158`). This is necessary because `GreedyBatchedRNNTInfer` mutates partial hypotheses in place (`rnnt_greedy_decoding.py:825-831`) and `Hypothesis.merge_` extends token/timestamp/alignment lists in place (`rnnt_utils.py:153-181`).
- **Grouping/no padding:** fixed at the invariant level. PLAN.md now groups by `(target_lang, keep_all_outputs, drop_extra, chunk_T, decoder_mode)` and forbids not-ready padding/coercion (`PLAN.md:79-83`, `:152-157`, `:168-173`). This matches the code: `drop_extra_pre_encoded` is a scalar temporarily stored on `self.streaming_cfg` (`streaming.py:53-74`) and applied to the whole batch (`conformer_encoder.py:657-659`); first chunks use `drop_extra=0` and only `valid_new_mel`, while steady chunks use `drop_extra=self.drop_extra` and prepend `mel_frame_ring` (`server.py:2475-2484`).
- **Baseline, scheduler split, fail-closed, latency, finalize:** folded. PLAN.md now has Step 0 baseline capture (`PLAN.md:113-118`), split scheduler steps 5a/5b/5c (`PLAN.md:161-180`), fail-closed config handling (`PLAN.md:95-97`, `:175-180`), latency gates (`PLAN.md:103-106`, `:188-192`), and finalize/fork B=1 serialization (`PLAN.md:98-100`, `:182-185`).

Remaining inaccuracies or under-specification from the fold:

- **Stale dependency reference:** Probe A says `NO-GO => skip Step 6` (`PLAN.md:127`), but the Phase 1 implementation is Step 4 (`PLAN.md:147-150`). Step 6 is the B=1 scheduler and should not depend on encoder compile.
- **Exact ready predicate is named but not written.** Step 5 says "exact ready predicate" (`PLAN.md:156`) but should spell it out: include a row only when both `synthetic_prefix_samples + total_audio_samples >= (emitted_frames + shift_frames + 1) * hop_samples` and `len(pending_audio) >= preprocess_new_audio_samples`. These are the two current guards (`server.py:1088-1089`, `:2410-2415`, `:2437-2441`). This matters for `NEMOTRON_WARMUP_MS=200`, where `emitted_frames` and `synthetic_prefix_samples` are pre-advanced (`server.py:1072-1074`).
- **Step 7/Step 9 ordering conflict:** Step 7 already says it may use batched decode if Probe C GO (`PLAN.md:168-173`), but Step 9 is where `strategy=greedy_batch` is enabled (`PLAN.md:182-185`). Move decoder config selection before Step 7, or make Step 7 explicitly use current `greedy` only and reserve `greedy_batch` for a later throughput pass.

## 2. Deeper issues

### A. Probe-C-NO-GO fallback may not be worth the scheduler

If Probe C fails byte-exactness, PLAN.md says to "batch the encoder only, decode per-row" (`PLAN.md:142-144`, `:168-171`). That is a valid safety fallback, but the throughput ceiling may be much lower than the plan implies. With current `strategy=greedy`, the RNNT decoder explicitly loops over batch rows (`rnnt_greedy_decoding.py:401-409`). So a B-row `conformer_stream_step` saves encoder launch/compute work, but decode remains serialized per row.

Add an explicit fallback model:

`fallback_step_ms(B) = preproc_ms * B + encoder_batched_ms(B) + decode_ms * B + scheduler_overhead_ms`

The maximum speedup tends toward `1 / decode_fraction` once encoder launch overhead is amortized. If decode is 50% of the live step, the fallback ceiling is only 2x; if decode is 70%, it is 1.43x. The current profiling only splits preprocessing from `conformer_stream_step` (`proj-2026-05-20-modal-cost/batching-design-notes.md:6-8`), not encoder from decode, so the plan cannot yet tell whether this branch is worth a scheduler rewrite.

Concrete edit: extend Probe C or add a sub-probe to report encoder-only, current-greedy-decode-only, `greedy_batch` decode-only, and full scheduler-equivalent step time for B=1/2/4/8. If `greedy_batch` is NO-GO and the current-greedy fallback is less than 1.5x end-to-end at B=4 or B=8, stop the batching work rather than implementing a complex scheduler for marginal gain.

Also be precise about implementation: `conformer_stream_step` always decodes for RNNT even when `return_transcription=False` (`mixins.py:639-642`, `:707-712`). If "encoder-only then per-row decode" means bypassing `conformer_stream_step`, Step 7 must say it will call `model.encoder.cache_aware_stream_step` directly and then call the decoder per row. If it just calls `conformer_stream_step(B)` under current `greedy`, that is not an encoder-only path; it is batched encoder plus NeMo's row loop.

### B. Memory, OOM, and backpressure are not concrete enough

PLAN.md names `NEMOTRON_BATCH_MAX_SIZE` and `NEMOTRON_BATCH_MAX_WAIT_MS` but never gives defaults or high-water behavior (`PLAN.md:92-94`, `:168-172`). Current continuous mode uses an unbounded per-session event queue (`server.py:1631`), and the scheduler plan says "bounded queues" without a policy (`PLAN.md:161-165`).

Concrete edit:

- Set initial defaults before coding: `NEMOTRON_BATCH_MAX_WAIT_MS=5` and `NEMOTRON_BATCH_MAX_SIZE=4` for the first implementation; only raise the recommended max to 8/16 after B=8/16 OOM and latency sweeps pass on the smallest target GPU.
- Make the ready queue a deduplicated set/map of session IDs, not one entry per chunk, so backlogged sessions do not create unbounded duplicate queue nodes.
- Preserve byte-exactness under overload: never drop audio frames. Apply websocket/queue backpressure by awaiting bounded `put`, and fail closed with an overload close only if per-session lag exceeds a configured hard cap such as 2 seconds.
- Add a memory gate for B=1/2/4/8/16: `torch.cuda.max_memory_reserved`, active bytes before/after each batch, and no CUDA OOM. Use a startup/device-specific effective max batch size if the requested max would exceed a memory headroom threshold.

### C. Fork clone/snapshot work is outside the model-call lane today

The plan correctly serializes finalize/fork model calls, but it under-specifies the cost and lane ownership of the clone itself. `_continuous_finalize_emit_locked` snapshots and builds the fork before acquiring `inference_lock` (`server.py:2245-2253`). Those helpers clone CUDA tensors and deep-copy hypothesis state (`server.py:77-143`, `:2068-2118`, `:2120-2139`). With a scheduler, those CUDA clones can add latency and memory pressure outside the "single model-call lane" even if `_process_final_chunk` is serialized.

Concrete edit: Step 6 or Step 9 should say that parent snapshot, fork tensor cloning, fork final inference, and FORK_ASSERT comparison are all owned by the scheduler/model-call lane, or explicitly prove that clone outside the lane cannot interleave with normal batch CUDA work. Add telemetry for `fork_clone_ms`, `fork_assert_snapshot_ms`, `fork_assert_compare_ms`, fork memory delta, and `inference_lock/model_lane_wait_ms`. The silence0_warm200 gate should fail if FORK_ASSERT stays correct but clone work pushes final p95 over the <400ms budget.

### D. Global mutable streaming state needs exception cleanup

NeMo's encoder wrapper temporarily writes `self.streaming_cfg.drop_extra_pre_encoded` and restores it after the encoder call (`streaming.py:53-74`), but it is not protected by a `try/finally`. A CUDA OOM or other exception in a batched call can leave a stale global `drop_extra_pre_encoded`, corrupting the next B=1 or batch call. This is a deeper version of the scalar-drop hazard from round 1.

Concrete edit: the server-side model-call lane should snapshot and restore `model.encoder.streaming_cfg.drop_extra_pre_encoded` in `finally` around every normal, batched, warmup, and final call, and Probe B/Step 8 should include an injected-error test that proves the next B=1 chunk still uses the correct drop value. The same pattern should be considered for prompted models because `_apply_inference_prompt` mutates model-global prompt state (`server.py:639-646`), though prompt is set before every prompted call so it is less fragile.

### E. Baseline artifact needs dirty-tree identity

Step 0 records env/model/NeMo metadata (`PLAN.md:113-118`), but should also record the server repository commit, dirty status, and a diff hash. The current working tree is not clean, so "CURRENT server" is otherwise ambiguous. Add `git rev-parse HEAD`, `git status --short`, and a hash of `git diff -- src/nemotron_speech/server.py` to the artifact metadata.

### F. Plan size is acceptable, but Step 9 should be merged or narrowed

Do not cut Probes A/B/C; they are the safety net. The plan is big because the change is genuinely risky. The one step that should be merged or narrowed is Step 9: decoder switching must happen before Step 7 if Step 7 uses batched decode, and finalize/fork serialization is already required in Step 6 and Step 8 (`PLAN.md:161-180`, `:182-185`). Either move the decoder switch into Step 7 and make Step 9 only "finalize/fork B=1 validation", or delete Step 9 and put its gates in Steps 6/8/10.

## 3. Gate-threshold concreteness

Several GO/NO-GO gates are still not autonomous enough.

- **Probe A:** ">=20% faster steady-state per call" (`PLAN.md:124-127`) needs a measurement contract. Use the same fixed clip, same process, same right-context, same warmup, discard the first N warmup chunks, and record at least 200 steady chunks with CUDA synchronization or CUDA events. GO should mean full live step p50 is >=20% faster, p95 is not worse by more than 5%, no recapture/recompile after the allowed first/steady shapes, and memory reserved grows by less than a defined cap such as 10% or stays below 80% of device memory.
- **Probe C:** "materially faster decode batched" (`PLAN.md:141-142`) is too vague. Define it as: B=1 `greedy_batch` byte/state identical to `greedy`, and at B=4 or B=8 decode wall-time per stream is at least 25% lower than current `greedy` with the same encoded inputs, with full-step scheduler-equivalent throughput at least 1.5x. If this fails, the plan may still choose current-greedy fallback, but only if the fallback has its own >=1.5x end-to-end gate.
- **Step 7:** "keep-up knee materially > baseline" (`PLAN.md:172-173`) should be numeric. Local gate: baseline ~14 becomes at least 21 concurrent streams, or at least +4 streams if baseline noise is high. Modal gate: representative cheap GPU baseline ~5 becomes at least 10 before claiming streams/$ improvement.
- **Latency:** "within budget" (`PLAN.md:103-106`, `:188-192`) should compare against the Step-0 artifact and the old server. Suggested N=1 batch-on gate: TTFS/interim p95 <= baseline p95 + `NEMOTRON_BATCH_MAX_WAIT_MS` + 10ms, p99 <= baseline p99 + 2*`NEMOTRON_BATCH_MAX_WAIT_MS` + 20ms, and silence0_warm200 final p95 remains <400ms.
- **Memory/OOM:** add a gate: no OOM at configured default max batch size on T4/L4/local, and max memory reserved remains below a fixed headroom threshold. A failed memory gate must reduce effective max batch size or disable batching at startup; do not discover OOM in production traffic.

## 4. Compile x batch static-shape conflict check

The plan partly separates Phase 1 compile and Phase 2 batching, but Step 10's `{compile on/off} x {batch on/off}` matrix is not yet well-defined (`PLAN.md:188-192`). `torch.compile(..., mode="reduce-overhead")` is a CUDA-graph-oriented mode, and CUDA graphs require static shapes. Phase 1 can be safe for the fixed B=1 path if Probe A proves first-chunk, steady-state, warmup, and final handling (`PLAN.md:120-127`, `:147-150`). Continuous batching has variable B by design (`PLAN.md:175-180`), so one captured B=1 graph cannot be used for B=2/4/8 groups.

Concrete edit: define the both-on cell before implementation:

- Default semantics: `NEMOTRON_ENCODER_COMPILE=1` applies only to B=1 calls. When `NEMOTRON_BATCH_SCHED=1` forms B>1 groups, those groups use the uncompiled encoder path unless a bucketed compile probe passes. This makes Step 10's both-on cell mean "compile active for solo/fallback/final B=1, batching uncompiled for B>1."
- Optional later semantics: pre-capture exact static buckets such as B=1/2/4/8, with recapture telemetry and a byte/state/OOM gate per bucket. Do not pad real not-ready streams. If dummy padding is considered for graph buckets, it needs its own probe proving dummy rows cannot affect real rows and that the wasted work is still faster.

Without this edit, enabling both flags can silently degrade into repeated recapture/recompile, graph errors, or a fake benchmark that only tests B=1 compile and B>1 batching separately but not their runtime interaction.

## 5. Prioritized concrete edits

1. Fix Probe A dependency text: `NO-GO => skip Step 4 / Phase 1`, not Step 6 (`PLAN.md:127`).
2. Move the `greedy_batch` config switch before Step 7, or make Step 7 current-greedy-only and merge Step 9 into the later validation. Do not let Step 7 depend on a Step 9 artifact.
3. Add exact ready predicate text to Step 5: both the timeline predicate from `_handle_audio_locked` and the pending-audio guard from `_process_chunk` (`server.py:2410-2415`, `:2437-2441`), including `synthetic_prefix_samples` (`server.py:1088-1089`).
4. Define compile-plus-batch semantics: compiled B=1 only by default; B>1 uncompiled unless bucketed static-B graphs pass separate byte/state/OOM/recapture gates.
5. Quantify Probe C and Step 7 throughput gates, including a separate GO/NO-GO threshold for the Probe-C-NO-GO "batched encoder + current greedy row loop" fallback.
6. Add default batching knobs and overload policy: initial `NEMOTRON_BATCH_MAX_WAIT_MS=5`, `NEMOTRON_BATCH_MAX_SIZE=4`, deduped ready queue, bounded audio/event queues, no frame dropping, overload close only after a documented lag cap.
7. Add memory gates and telemetry for B=1/2/4/8/16 before raising default max batch size beyond 4.
8. Put fork snapshot/clone/final/FORK_ASSERT ownership under the scheduler/model-call lane, or explicitly prove clone outside the lane is safe; measure clone and FORK_ASSERT cost in the silence0_warm200 gate.
9. Add server-side `try/finally` restoration for `encoder.streaming_cfg.drop_extra_pre_encoded` around model calls and an injected-error test.
10. Extend Step 0 metadata with server git SHA, dirty status, and diff hash, not only env/model/NeMo commit.

## 5-line summary

1. Round-1's cache-axis, decoder-strategy, hypothesis, grouping, baseline, scheduler-split, fail-closed, and finalize fixes are largely incorporated.
2. The remaining blockers are sharper: stale dependencies, undefined compile+variable-B semantics, Step 7/9 ordering, and vague GO thresholds.
3. The Probe-C-NO-GO fallback may be a small win only, because current `greedy` decodes rows serially; measure the encoder/decode split before building the scheduler.
4. Memory limits, queue backpressure, fork clone cost, and global drop-extra cleanup need concrete defaults and gates.
5. Verdict: v2 is much safer than v1, but it still needs a v3 tightening pass before implementation.
