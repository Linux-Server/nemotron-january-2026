# Codex plan review - streaming ASR batching/compile

Scope: review only. I did not modify production code.

## CRITICAL - must fix before `/implement`

1. **Step 0 is marked done, but the hard multilingual baseline is still missing.**
   - Evidence: Step 0 requires rc1 English, multilingual prompted cases, and silence0_warm200 finalize/fork (`PLAN.md:106-111`), but Progress says only `baseline/english_baseline.json` exists and "multilingual baseline still TODO" (`PLAN.md:196`). `baseline_capture.py` also says the prompted baseline is "captured later" and writes only `english_baseline.json` (`baseline_capture.py:8-10`, `baseline_capture.py:124-126`).
   - Why this can derail/corrupt: the scheduler will change prompt switching and grouping around a model-global prompt (`server.py:639-646`) and tag stripping (`server.py:733-751`). Without a prompted baseline and mixed-language concurrency gate, the plan cannot enforce the hard constraint "don't break multilingual prompted path."
   - Recommended change: split Step 0 into `0a English baseline done` and `0b prompted baseline/blocking`. Mark Step 0 incomplete until `en-US`, one non-English supported language, and `auto` are captured with interim sequence, final text, final delta, tag stripping, and `target_lang`. Make Step 7/8 depend on `0b`, and add a concurrent mixed-language gate that proves prompt groups split correctly.

2. **Probe B is overcredited: the script does not implement the plan's state-equality or exception gates.**
   - Evidence: Step 2 says B=2/4, per-step text+state comparison, mixed-geometry rejection, warmup-200, absent/rejoin, alias sentinel, and injected exception (`PLAN.md:120-126`). The actual probe is B=2 only, compares text lists only, and does not compare scattered cache tensors, `previous_hypotheses`, `pred_out_stream`, or `emitted_frames` (`probe_batched_step.py:151-187`). It imports `scatter_cache_row` but never uses it for state validation (`probe_batched_step.py:26-28`). Progress also admits the in-server `drop_extra` exception test is still TODO (`PLAN.md:198`).
   - Why this can derail/corrupt: final transcripts can match while row-local caches or decoder state are already divergent. The first production scheduler bug is likely to be state drift, not immediate final-text divergence.
   - Recommended change: do not treat Probe B as fully satisfying Step 2. Add a blocking `test_batch_state.py`/Probe B2 before Step 7 that runs B=2 and B=4, uses the real server session initialization/warmup, scatters each row, and compares cache tensors, hypothesis token/state fields, `pred_out_stream`, `emitted_frames`, interim text, and final text against B=1. Include injected exception after NeMo sets `drop_extra_pre_encoded` and prove the next B=1 chunk is correct.

3. **`greedy_batch` is not proven for the actual Phase-2 multi-row streaming path.**
   - Evidence: Step 3 requires `greedy` vs `greedy_batch` at B=1 and then B=2/4/8, with byte+state and speed gates (`PLAN.md:128-133`). The actual Probe C streams each clip at B=1 under `greedy`, switches strategy, then streams each clip at B=1 under `greedy_batch` (`probe_decoder_strategy.py:59-75`, `probe_decoder_strategy.py:88-92`). Probe B, which does exercise B=2, explicitly configures current `strategy=greedy` (`probe_batched_step.py:55-57`). NeMo feeds `partial_hypotheses` into RNNT decoding per row (`mixins.py:707-712`).
   - Why this can derail/corrupt: Probe C proves B=1 strategy equivalence, not B>1 batched `partial_hypotheses` threading across many ticks with `max_symbols=10`. That is exactly the path Step 7 will ship.
   - Recommended change: add a blocking decoder/batch probe before Step 7: run N different clips through `strategy=greedy_batch`, B=2/4/8, full multi-chunk streams, row-order permutation, and mid-stream join/leave. Compare per-row interim sequence, final, caches, hypothesis token/state, and `emitted_frames` against current B=1 `greedy`. Include `keep_all_outputs=True` final/fork calls, not only normal interim chunks.

4. **The global decoder-strategy semantics are underspecified and may be impossible as written.**
   - Evidence: the server configures one global decoding strategy at load via `self.model.change_decoding_strategy(...)` (`server.py:793-827`), and `conformer_stream_step` later uses `self.decoding` (`mixins.py:707-712`). The plan says no `greedy_batch` at load except behind the batch flag (`PLAN.md:78-80`), while the implementation notes say use `greedy_batch` under the batch flag but keep `greedy` for flag-off and finalize/fork (`scheduler-implementation-notes.md:66-67`).
   - Why this can derail/corrupt: with one model object, normal batches, B=1 fallbacks, warmup, and finalize/fork cannot simultaneously use different decoder strategies unless the implementation switches decoding config on the model-call lane or keeps a second model. Per-call strategy switching is global mutation and may add latency or state hazards.
   - Recommended change: make Step 7 choose one explicit design. Preferred: flag off keeps today's load config byte-identical; flag on sets `strategy=greedy_batch, loop_labels=True, use_cuda_graph_decoder=False` globally only after B=1 normal, warmup, and final/fork are byte-exact. If the plan insists final/fallback use `greedy`, specify the per-call decoder context switch, serialize it on the lane, and measure the cost.

5. **Phase 1 compile is overvalidated relative to the server's real shapes and call path.**
   - Evidence: Step 1 requires first-chunk, steady-state, warmup-200, final shapes, full-step p50/p95, no recapture, and memory gates (`PLAN.md:113-118`). The actual probe compiles `m.encoder.cache_aware_stream_step` only, with one random input shape and `drop_extra_pre_encoded=0` (`probe_encoder_compile.py:32-43`, `probe_encoder_compile.py:52-60`). It times that encoder method only (`probe_encoder_compile.py:70-84`). The live server has different B=1 shapes: session warmup T=`warmup_frames` (`server.py:1017-1040`), first chunk T=`shift_frames` with drop 0, steady chunk T=`pre_encode_cache_size + shift_frames` with `self.drop_extra` (`server.py:2475-2485`), and final variable-length `keep_all_outputs=True` chunks (`server.py:2686-2775`).
   - Why this can derail/under-deliver: Step 4 cannot get the Probe A speedup by simply leaving `self.model.conformer_stream_step(...)` untouched, because the compiled object is the encoder method inside that call. Different T/drop/keep_all shapes can force graph breaks or recaptures, erasing the launch-overhead win.
   - Recommended change: downgrade Probe A to "conditional GO for one encoder shape." In Step 4, explicitly define how the compiled encoder is invoked: either patch `model.encoder.cache_aware_stream_step` behind the flag or split `conformer_stream_step` into compiled encoder plus existing RNNT decode. Warm/capture only exact static B=1 buckets that pass gates. Keep final/fork uncompiled unless a final-shape bucket probe passes. Add compile-hit/recapture telemetry and p95/memory gates.

6. **The plan still risks under-delivering the Modal knee because it batches only inference while the latest finding says the server layer is co-equal.**
   - Evidence: the latest measurements say isolated model span does not explain the Modal knee; server-layer CPU work is about half the per-stream budget on cloud, and batching must collapse N server-orchestration passes to 1 (`RESULTS.md:183-199`). The companion implementation recipe says "Only (b) is batched" and leaves preprocessing as a per-session loop (`scheduler-implementation-notes.md:13-23`). The current preprocessor is one GPU/CPU call per stream with a fixed B=1 tensor (`server.py:1091-1108`).
   - Why this can under-deliver: on slow-CPU cloud, leaving STFT/preprocess and most Python work per stream may improve the model term while still preserving a large serialized N-scaled term. That can miss the promised Modal knee lift even if `conformer_stream_step(B)` is fast.
   - Recommended change: add a Step 7a substep: stack fixed audio windows into `[B, K]`, stack valid lengths into `[B]`, call `self.model.preprocessor(...)` once per same-ready group, then slice per-row `valid_new_mel` and batch the model. Gate batched preprocessor byte-equality to per-row preprocessing. If batched preprocessing is not safe, explicitly measure remaining per-stream preprocess/orchestration cost and lower the knee target.

7. **Step 6 needs an explicit flag-off rollback path for the scheduler infrastructure itself.**
   - Evidence: the Rules say "Flag-gate everything" and default off equals today's exact config (`PLAN.md:87-88`), but Step 6 describes replacing the per-session worker path with a scheduler B=1 path before Step 7 introduces `NEMOTRON_BATCH_SCHED` (`PLAN.md:151-159`, `PLAN.md:161-165`). Today's path is per-session queue/worker plus `inference_lock` (`server.py:1702-1729`, `server.py:2414-2418`).
   - Why this can derail rollback: if Step 6 lands unconditionally, a scheduler bug affects production even with batching disabled. That violates the hard default-current constraint.
   - Recommended change: Step 6 must be behind the same `NEMOTRON_BATCH_SCHED=1` flag or a separate experimental `NEMOTRON_SCHEDULER_B1=1` flag. Flag off must execute the existing handler/worker/inference_lock path and build the exact current decoder config. Add startup logs showing `scheduler_enabled`, `batch_enabled`, `decoder_strategy`, and `encoder_compile_enabled`.

8. **Finalize/reset ordering under a decoupled scheduler is not specified as a barrier.**
   - Evidence: current continuous mode processes events in order under `session.state_lock` (`server.py:1708-1729`), and `_handle_audio_locked` drains all ready chunks before returning to the next reset/finalize event (`server.py:2390-2435`). Finalize/fork snapshots and runs final inference later (`server.py:2210-2264`). The plan says generation tokens prevent a queued chunk racing a fork (`PLAN.md:93-95`), but does not define reset/end/close as scheduler barriers.
   - Why this can corrupt output: a reset/finalize can overtake a ready normal chunk, or a normal batch can scatter stale output into a session after a fork/reset/close. This is especially dangerous when a middle row leaves an in-flight batch.
   - Recommended change: Step 6 should state: reset/end/close removes the session from ready sets, waits for or invalidates any in-flight generation, drains or folds all earlier audio events into the session state, then runs fork/finalize on the model-call lane. Scatter must check the generation token before mutating or sending. Add tests for finalize while queued, finalize while in-flight as the middle row of a batch, close while in-flight, and backlog plus reset.

9. **Batch scatter currently returns cache views, which can retain full-batch storage and destabilize variable-B/compile.**
   - Evidence: `scatter_cache_row` returns slices directly (`batch_primitives.py:60-63`). The plan correctly requires dim-1/dim-0 scatter (`PLAN.md:66-69`) and Step 8 covers streams joining/leaving (`PLAN.md:169-174`), but it does not require row tensors to own B=1 storage.
   - Why this can derail: each session may retain a view into the full `[layers, B, ...]` cache output until it is overwritten. That can inflate memory, keep stale rows alive after a stream leaves, and feed non-contiguous/view-strided caches into the B=1 compiled fallback, causing recapture or slow paths. If any downstream code mutates cache tensors in place, view aliasing can become cross-talk.
   - Recommended change: scatter caches as owned B=1 tensors, e.g. `clc[:, i:i+1, ...].detach().clone()` / same for time and len, or an explicitly proven cheaper contiguous copy. Add a unit test that scattered rows have independent storage and a variable-B test where a middle row leaves before the next tick.

10. **Exception rollback only covers `drop_extra_pre_encoded`, not in-place decoder hypothesis mutation.**
    - Evidence: the plan documents in-place hypothesis mutation (`PLAN.md:36`) and requires a `try/finally` restore for `drop_extra_pre_encoded` (`PLAN.md:75-76`). `stack_hypotheses` passes the session's existing hypothesis objects directly (`batch_primitives.py:66-82`). The server already has a tensor-aware hypothesis clone helper for fork safety (`server.py:139-143`). NeMo restores `drop_extra_pre_encoded` only after the encoder returns, with no `finally` (`streaming.py:53-74`).
    - Why this can corrupt output: if a batched decode mutates some `partial_hypotheses` and then raises before scatter/post-step, sessions can retain advanced decoder state while caches, pending audio, and emitted frames remain unadvanced.
    - Recommended change: either pass cloned hypotheses into the batched call and assign returned hypotheses only after success, or treat any model-call exception as fatal for all affected sessions and reset/close them instead of continuing. Add an injected decoder exception test, not only an encoder/drop-extra exception test.

## SHOULD-FIX

1. **Add the real cross-talk correctness gate.**
   - Existing `concurrency_test.py` is valuable because it uses 24 distinct audios (`concurrency_test.py:54-56`) and compares each session to its own N=1 baseline (`concurrency_test.py:272-331`), but it records final transcript only and allows edit-distance tolerance (`concurrency_test.py:22-27`, `concurrency_test.py:308-312`).
   - Recommended change: add a strict Phase-2 gate that streams N different clips concurrently with batch on and compares each per-stream interim sequence, final text, final delta, and no-duplicate-emission against same-binary flag-off baselines. Use a fast 24-clip gate on every step and a 1000-sample canary before Modal Step 10. This is the cheapest strong detector for 1-in-1000 cache/prompt cross-talk.

2. **Make the dispatch policy precise enough to preserve both latency and batch fill.**
   - Evidence: Step 7 says dispatch on timer, max size, or "all ready gathered," with "immediate at low load" (`PLAN.md:163-165`).
   - Risk: "all ready gathered" is not observable in an open server, and "immediate at low load" can accidentally turn most N=2/4 traffic into B=1 dispatches, undercutting the knee improvement. Waiting too aggressively can violate the N=1 latency gate.
   - Recommended change: define the exact rule: if only one active eligible session exists, dispatch immediately; otherwise start a 5 ms coalescing timer on first ready and dispatch same-group rows when max size hits or timer expires. Track effective batch-size histogram and queue-wait p95.

3. **Update Step 9/10 validation to compare flag-on against same-machine flag-off, not stale artifacts across environments.**
   - Evidence: Step 9 says byte+state diff vs Step-0 (`PLAN.md:176-180`), and Step 10 says Modal byte-exact (`PLAN.md:182-186`).
   - Risk: local artifacts are good for implementation gates, but cross-machine/cuFFT differences can confuse strict byte comparisons. The Modal correctness claim should be same image, same GPU class, same commit, flag off vs flag on.
   - Recommended change: Step 10 should run a short flag-off baseline in the deployed Modal container before flag-on, then compare exact outputs there. Keep the old RESULTS.md knee baseline for performance comparison.

4. **Tighten primitive assertions before server wiring.**
   - Evidence: `stack_processed` checks feature and T dimensions but not that every row is B=1, same dtype, same device, and contiguous enough for the intended fast path (`batch_primitives.py:35-45`). `stack_pred_out` returns `None` if any row lacks a value (`batch_primitives.py:85-89`), which is harmless for current RNNT but unsafe if CTC ever slips past fail-closed.
   - Recommended change: assert `cm.shape[0] == 1`, same dtype/device for mels and caches, and RNNT-only before using `stack_pred_out` semantics. Unsupported CTC/hybrid/beam should refuse batching at startup or force B=1 per row.

5. **Require a memory gate that includes retained session state after scatter.**
   - Evidence: the plan has a B=1/2/4/8/16 memory gate (`PLAN.md:100-102`), but the dangerous case is not only peak model memory; it is retained session cache storage after scatter and join/leave.
   - Recommended change: Step 8 should measure `max_memory_reserved`, active bytes before/after each batch, and retained memory after sessions leave/reset. Run with B=16 and churn sessions in/out.

## NICE-TO-HAVE

1. **Refresh the PLAN context with the 2026-05-21 split finding.**
   - `PLAN.md:14-18` still frames `conformer_stream_step` as about 91% of cost and preprocessing as about 1 ms/9%. The latest `RESULTS.md:183-199` says server orchestration is co-equal on Modal. Update the context so the implementer does not optimize only the model call and miss the cloud bottleneck.

2. **Add explicit telemetry for compile and scheduler effectiveness.**
   - In addition to Step 8 telemetry (`PLAN.md:169-174`), log compile path hit/miss/recapture count, effective batch size, eligible-ready count, fallback reason, per-batch prompt language, preprocessor batch ms, model batch ms, scatter/postprocess ms, and model-lane utilization.

3. **Add startup assertions for Blackwell decoder graph safety.**
   - The plan correctly keeps `use_cuda_graph_decoder=False` (`PLAN.md:78-80`). Add a startup log/assert under batch flag so a future config change cannot silently enable decoder CUDA graphs on Blackwell.

## Verdict

**GO-WITH-FIXES, not as-is.** The cache-axis design is now basically right, but `/implement` should not proceed until the critical fixes above are folded into the plan, especially the incomplete Probe B/C gates, missing multilingual baseline, explicit decoder-strategy semantics, scheduler rollback/barrier behavior, and the server-layer batching/performance requirement.
