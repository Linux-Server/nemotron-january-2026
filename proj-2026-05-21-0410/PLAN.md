# Plan: Streaming-ASR GPU throughput optimization (single-stream graphs + continuous batching)

Project directory: `./proj-2026-05-21-0410`

Review status: **round 4 folded (2026-05-21) — GO-WITH-FIXES, not as-is** (`claude-plan-review.md` +
`codex-plan-review.md`). Round 4 (post the captured 3-box GPU-vs-host split, RESULTS.md) found the **"done"
probes are over-credited** vs the gates they claim (Probe A = 1 shape; Probe B = text-only at B=2, no state
compare; Probe C = B=1 only) and that batching ONLY the model leaves a co-equal server-layer term on cloud.
The blocking fixes are consolidated in **Rules → "Round-4 review fixes"** and threaded into Steps 0,1-3,4-10.
DAG (acyclic): `0→{1,2,3}; 1→4; 2→5; 3→7; 5→6→7→8→9→10→11` plus round-4 blocking re-probes: **Probe B2
(state-equality) in Step 5 → Step 7; greedy_batch-B≥2-multichunk probe → Step 7; Step 0b multilingual
baseline → Steps 7,8** (Steps 5 & 6 independent; Step 4→9 optional compile edge).

## Context
The Nemotron streaming ASR server (`src/nemotron_speech/server.py`) runs `batch_size=1` under one
global `inference_lock`, so its realtime keep-up knee is **~5 concurrent streams on every Modal GPU
type** (T4≈L4≈A100≈H100≈RTX-PRO-6000; local bare-metal 5090 ~14 — an environment, not GPU-arch, gap,
proven by the same Blackwell silicon kneeing at ~5-6 on Modal). At B=1 the per-chunk bottleneck is
`conformer_stream_step` (~91% of cost, launch/dispatch-bound) vs preprocessing (~1ms/9%) — **but that 9%
is a B=1 framing.** The captured 3-box split (RESULTS.md, 2026-05-21) decomposes per-chunk cost into
GPU-active (scales with GPU clock; FLOPs irrelevant) + host idle-gaps (scales with CPU launch latency) +
a **co-equal SERVER-LAYER term** (per-stream STFT + asyncio + GIL + lock handoff) that is ~3ms local but
~21ms on Modal — roughly HALF the per-stream knee budget on slow-CPU cloud. So after the model is batched,
the un-batched per-stream preprocess/orchestration becomes the next serialized bottleneck (Step 7a). This
plan raises the per-GPU knee toward the batch-amortized ceiling (microbench: ~8-10× GPU headroom at B=1) via
**Phase 1** (CUDA-graph/`torch.compile` the encoder, B=1 launch-overhead) and **Phase 2** (a
continuous-batching scheduler running one `conformer_stream_step(B)` per tick). **The dominant risk is
silent transcript corruption from cache-aware batched state**; the plan is probe-gated and byte-exact-gated.

## Reference implementations
- **`conformer_stream_step`** — `…/NeMo/nemo/collections/asr/parts/mixins/mixins.py:592`: `encoder.cache_aware_stream_step(
  processed_signal[B,F,T], cache_last_channel, cache_last_time, cache_last_channel_len, drop_extra_pre_encoded)`
  (653) → `decoding.rnnt_decoder_predictions_tensor(encoder_output=encoded[B], encoded_lengths[B],
  partial_hypotheses=<per-row list>)` (707). It ALWAYS decodes for RNNT even with return_transcription=False (639-642).
- **Encoder cache layout (CRITICAL axis)** — `conformer_encoder.py:1087-1125`: `cache_last_channel`
  `[layers,B,cache_T,d_model]`, `cache_last_time` `[layers,B,d_model,time_T]`, `cache_last_channel_len` `[B]`.
  Layers indexed first, batch INSIDE (696-710), same layout returned (768-776). **Batch = dim 1 for
  channel/time, dim 0 for len.** Scatter row i: `[:, i:i+1, …]` (channel/time), `[i:i+1]` (len).
- **`drop_extra_pre_encoded` scalar/global + UNGUARDED** — NeMo writes it to `self.streaming_cfg.drop_extra_pre_encoded`
  and restores WITHOUT try/finally (`…/parts/submodules/streaming.py:53-74`); applied to the whole batch
  (conformer_encoder.py:657-659). An exception mid-call leaves it stale → corrupts the next call.
- **Decoder factory** — `rnnt_decoding.py:384-455`: `strategy=greedy` → `GreedyRNNTInfer` (current, decodes
  rows in a Python loop, 401-409); `strategy=greedy_batch` → `GreedyBatchedRNNTInfer` (loop_labels +
  use_cuda_graph_decoder options). `loop_labels=True` under `greedy` does NOT batch the decode.
- **In-place hyp mutation** — `rnnt_greedy_decoding.py:825-831` (`hyp.merge_`), `rnnt_utils.py:153-181`.
  Each row needs a UNIQUE Hypothesis; `clone_hypotheses_deep` at server.py:139.
- vLLM/SGLang continuous batching is the model; divergence: per-stream cache-aware state stacked as
  independent rows, one chunk/stream/tick, no token preemption. Byte-exact ref = current B=1 `greedy` path.

## Current state
- `ASRSession` — `server.py:364`: `target_lang`(369), `pending_audio`(376), `raw_audio_ring`(387),
  `mel_frame_ring`(390), `emitted_frames`(393), `cache_last_channel`(396), `cache_last_time`(397),
  `cache_last_channel_len`(398), `previous_hypotheses`(401, list len 1), `pred_out_stream`(402), `current_text`(405).
- `inference_lock` (463); per-session worker drains events under `state_lock` (~1702-1729) → `_handle_audio_locked`
  (2390); continuous-mode per-session event queue is UNBOUNDED (1631).
- `_process_chunk` (2429): pending guard `len(pending_audio) >= preprocess_new_audio_samples` (2437); first-chunk
  vs steady geometry (2462-2469); the call (2484). Ready timeline guard in `_handle_audio_locked`:
  `synthetic_prefix_samples + total_audio_samples >= (emitted_frames + shift_frames + 1)*hop_samples`
  (2410-2415; `synthetic_prefix_samples`/warmup pre-advance at 1072-1074, 1088-1089).
- Finalize/fork: `_continuous_finalize_emit_locked` snapshots+builds fork BEFORE `inference_lock` (2245-2253);
  `_build_continuous_finalize_fork` (2068) + clone helpers (77-143) deep-clone CUDA tensors+hyps; FORK_ASSERT (2141-2181);
  `_process_final_chunk` variable-length span, `keep_all_outputs=True` (2686-2775) — geometry ≠ normal tick.
- Decoding (793-827): `strategy=greedy`, `max_symbols=10`, `loop_labels: False`, `use_cuda_graph_decoder: False`
  (Blackwell-disabled). `NEMOTRON_DECODING=beam` differs (793-807); EOU adds preserve_alignments (818-826).
- Prompt: `set_inference_prompt(lang)` model-global (639-646); tag-strip `_extract_hypothesis_text` (733-751).
- Harnesses: `proj-2026-05-19-eou-endpointing/concurrency_test.py` (`--url`); Modal `asr_bench_modal.py`
  (env ASR_GPU/REGION/CONCURRENT/PROFILE); baseline knee/$ in `proj-2026-05-20-modal-cost/RESULTS.md`.
- Diagnosis/design: `proj-2026-05-20-modal-cost/{batching-design-notes.md,RESULTS.md}` (READ FIRST).

## Rules
### Correctness — cache-aware-state hazard (hard gates)
- **Byte-exact PER-STREAM** vs the Step-0 baseline artifact: interim cumulative sequence + final text +
  final delta + no dup emissions + multilingual tag-stripping. AND **state-equal**: each row's caches
  (correct axes), `previous_hypotheses` tokens/decoder-state, `pred_out_stream`, `emitted_frames` = B=1 ref.
- **Cache axis:** concat/scatter `cache_last_channel`+`cache_last_time` on **dim 1**, `…_len` on **dim 0**
  (`[:, i:i+1, …]` / `[i:i+1]`); assert shapes.
- **Hypothesis:** flat per-row list `[s.previous_hypotheses[0] or None for s in batch]`; UNIQUE objects
  (alias sentinel); scatter `best_hyp[i]` back as a one-element list; row-order permutation must not change output.
- **Grouping key:** `(target_lang, keep_all_outputs, drop_extra, chunk_T, decoder_mode)`. Never mix first-chunk
  + steady-state (scalar drop_extra). One `target_lang` per batched call (model-global prompt, set once before).
- **Never pad/coerce a not-ready stream.** Ready = BOTH `synthetic_prefix_samples + total_audio_samples >=
  (emitted_frames + shift_frames + 1)*hop_samples` AND `len(pending_audio) >= preprocess_new_audio_samples`.
  Backlogged streams do one real shift then requeue. No frame dropping ever (backpressure via bounded awaited put).
- **`try/finally` restore** `model.encoder.streaming_cfg.drop_extra_pre_encoded` (and snapshot prompt) around
  EVERY model call (normal/batched/warmup/final) so an exception can't poison the next call.
- rc1 English byte-identical + multilingual prompted path + silence0_warm200 finalize/fork (FORK_ASSERT clean) preserved.
### Decoder strategy
- No `greedy_batch`/`loop_labels=True` at load. Behind the batch flag only, after Probe C proves B=1
  `greedy_batch` byte+state identical to `greedy`. `use_cuda_graph_decoder` stays False (Blackwell).
### Compile × batch (static-shape) semantics
- `NEMOTRON_ENCODER_COMPILE=1` (reduce-overhead = CUDA graphs, needs static shape) applies to **B=1 calls
  only** by default. When `NEMOTRON_BATCH_SCHED=1` forms B>1 groups, those run the **UNcompiled** encoder
  unless a separate bucketed-static-B compile probe passes (per-bucket byte/state/OOM/recapture gates; no
  dummy padding of real streams). Step 10's "both-on" cell = compile for solo/fallback/final B=1, batching
  uncompiled for B>1. Primary ship target = **batch-only** (the bigger lever); compile is a separate B=1 lever.
### Safety, defaults, sequencing
- **Flag-gate everything**; default off = today's exact config (greedy/loop_labels=False/cuda_graph=False/B=1).
  Initial defaults: `NEMOTRON_BATCH_MAX_WAIT_MS=5`, `NEMOTRON_BATCH_MAX_SIZE=4` (raise to 8/16 only after
  OOM+latency sweeps pass on the smallest target GPU). Ready queue = deduped session-ID set (not per-chunk).
- **Fail closed:** unsupported row config (beam, CTC/hybrid-ctc, EOU preserve-alignments unprobed, unsafe
  grouping, model lacking the batch path) → split into safe sub-batches or B=1 fallback; never coerce.
- **Single model-call lane** serializes normal batches, warmup, finalize/fork, prompt switches; per-session
  generation/in-flight token prevents a queued chunk racing a fork. Fork snapshot/clone/FORK_ASSERT are
  owned by (or proven safe outside) the lane; measure their cost.
- No new heavy deps; don't perturb the constant-plan cuFFT/preprocessor invariant. A failed probe gate blocks dependents.
### Latency (streaming UX gate)
- N=1 batch-on: TTFS/interim p95 ≤ baseline p95 + `BATCH_MAX_WAIT_MS` + 10ms; p99 ≤ baseline p99 +
  2·`BATCH_MAX_WAIT_MS` + 20ms; silence0_warm200 final p95 stays <400ms (incl. fork clone cost).
### Memory
- Memory gate at B∈{1,2,4,8,16}: no CUDA OOM; `max_memory_reserved` < device headroom (e.g. 80%). Failed gate
  → reduce effective max batch at startup (device-specific), never discover OOM in prod.
### Round-4 review fixes (folded 2026-05-21 — BLOCKING; refs C#=codex-plan-review, S#/N#=claude-plan-review)
- **Probes are over-credited vs claims — re-validate before relying on them.** Probe A validated ONE encoder
  shape (`drop_extra=0`, one random input) ⇒ CONDITIONAL GO; Step 4 must cover the real server shapes
  (warmup/first/steady/final-variable). Probe B validated TEXT-equality + mid-stream dim-1 stack at B=2 only
  — it did NOT compare scattered caches / `previous_hypotheses` / `pred_out_stream` / `emitted_frames`; full
  STATE-equality + the injected-exception test are a BLOCKING **Probe B2** (Step 5) before Step 7. Probe C
  validated `greedy_batch`==`greedy` at **B=1 only** — `greedy_batch` at **B≥2 across a full multi-chunk
  stream** (per-row `partial_hypotheses` threading, `max_symbols=10`, row-permute, join/leave,
  `keep_all_outputs=True` final) is a BLOCKING probe before Step 7. (C2, C3, C5; S4.)
- **Scatter must OWN B=1 storage:** `scatter_cache_row` returns `.clone()` of the dim1/dim0 slice, not a view
  into the full `[layers,B,…]` batch — views retain whole-batch storage (memory), keep stale rows alive after
  a stream leaves, and feed view-strided caches into the compiled B=1 fallback (recapture/cross-talk).
  Unit-test that scattered rows have independent storage. (C9.)
- **Exception safety beyond drop_extra:** a batched decode mutates `partial_hypotheses` in place — if it
  raises mid-call, some rows have advanced decoder state while caches/audio/frames are unadvanced. Either pass
  CLONED hyps in and assign returned hyps only on success, OR treat ANY model-call exception as fatal for the
  affected rows (reset/close), never continue. Test an injected DECODER exception too, not only
  encoder/drop_extra. (C10; S5.)
- **Decoder strategy is ONE global on ONE model** (`change_decoding_strategy` at load; `conformer_stream_step`
  uses `self.decoding`). Step 7 picks an EXPLICIT design: flag-off = today's exact load config (byte-identical);
  flag-on = set `greedy_batch`/`loop_labels=True`/`cuda_graph=False` GLOBALLY, only after B=1-normal + warmup +
  final/fork are byte-exact under `greedy_batch`. If finalize/fallback must stay `greedy`, specify the
  per-call decoder-context switch, serialize it on the lane, and MEASURE its cost. No silent per-call global
  mutation. (C4.)
- **Scheduler rollback flag:** Step 6 infra is behind its own flag (`NEMOTRON_SCHEDULER_B1` or the batch flag);
  flag-OFF runs the EXISTING per-session worker + `inference_lock` path unchanged. Startup logs:
  `scheduler_enabled`, `batch_enabled`, `decoder_strategy`, `encoder_compile_enabled`. (C7.)
- **Barrier semantics for reset/end/close:** removing a session from the ready set waits-for-or-invalidates any
  in-flight generation, drains/folds earlier audio into state, THEN runs fork/finalize on the lane. Every
  scatter+send checks the per-session generation token before mutating/sending (no stale scatter into a
  session that forked/reset/closed — esp. a MIDDLE row leaving an in-flight batch). Tests: finalize-while-
  queued, finalize-as-middle-row-in-flight, close-while-in-flight, backlog+reset. (C8.)
- **Batch the preprocess too (Step 7a) — server layer is co-equal on cloud.** Stack same-ready-group fixed
  audio → `[B,K]` → ONE `preprocessor` call, slice per-row mel; behind its OWN byte-equality probe vs per-row
  preprocessing (batched cuFFT may pick a different plan — [[cufft-stft-plan-size-nondeterminism]]). If not
  byte-exact, keep preprocess serial and LOWER the knee target / cap MAX_SIZE (documented), don't ship a
  preprocess-bound knee labeled "8-10×". (C6; S1.)
- **Primitive assertions:** `stack_processed` asserts each row `shape[0]==1`, same dtype+device; assert RNNT
  before `stack_pred_out`; assert within a group that `previous_pred_out` AND `previous_hypotheses` are
  uniformly None-or-not (a fresh row must never batch with an established one — currently true only by the
  first-chunk/steady grouping split; make it explicit + fail-closed). (C-S4; S2.)
- **Strict cross-talk gate (dominant-risk detector):** every Phase-2 step streams N DISTINCT clips
  concurrently (flag-on) and compares each stream's interim sequence + final + final delta + no-dup vs the
  SAME-binary flag-off baseline — STRICT byte equality, not edit-distance (the existing `concurrency_test.py`
  is final-only + edit-distance, so extend it). Fast 24-clip gate each step; 1000-sample canary before Step 10.
  (C-S1; S3.)
- **Same-machine flag-off vs flag-on byte gates (Steps 9/10):** compare against a flag-off baseline captured on
  the SAME machine/image/GPU/commit (cross-env cuFFT differences confound strict byte comparison); Step 10
  runs a short Modal in-container flag-off baseline before flag-on. RESULTS.md knee numbers are for PERF
  comparison only. (C-S3.)
- **Memory gate includes retained state:** measure `max_memory_reserved` AND active bytes before/after each
  batch AND retained session-cache bytes after streams leave/reset, at B=16 with session churn. (C-S5.)
- **Telemetry + Blackwell assert:** log compile hit/miss/recapture, effective batch-size histogram,
  eligible-ready count, fallback reason, per-batch prompt lang, preprocessor-batch ms, model-batch ms,
  scatter/postprocess ms, model-lane utilization + wait p95, fork_clone_ms. Startup-assert `use_cuda_graph_
  decoder` stays False under the batch flag. (C-N2, C-N3; N2.)

## Steps

- [x] **0. Baseline artifact + fixed test set** (0a + 0b DONE; round-4 C1)
  Fixed clips: rc1 English (single + multi-segment), multilingual prompted (en-US + one other + auto), a
  silence0_warm200 finalize/fork case (`NEMOTRON_FORK_ASSERT=1`). Stream through the CURRENT server (B=1,
  greedy); record per-clip interim sequence + final text + final delta + metadata: env, model revision, NeMo
  commit, **server `git rev-parse HEAD` + `git status --short` + hash of `git diff -- src/nemotron_speech/server.py`**
  (tree is dirty — pin identity). Commit as the reference.
  - **0a (DONE):** rc1 English baseline → `baseline/english_baseline.json` (8 clips).
  - **0b (BLOCKING, still TODO — `baseline_capture.py` writes English only):** capture the MULTILINGUAL
    prompted baseline on the EA server — `en-US`, one supported non-English language, AND `auto` — with
    interim sequence + final + delta + **tag-stripping output** + per-session `target_lang`. Steps 7 & 8
    (which touch the model-global prompt + grouping) MUST NOT be signed off without 0b + a concurrent
    mixed-language gate proving prompt groups split correctly. Key files: `proj-2026-05-21-0410/baseline_capture.py`, `baseline/`

- [x] **1. Probe A — encoder compile/CUDA-graph feasibility (GO/NO-GO Phase 1, i.e. Step 4)**
  Standalone: real streaming loop, with vs without `torch.compile(model.encoder, mode="reduce-overhead")`
  (+ manual CUDA-graph variant); test first-chunk, steady-state, warmup-200, final shapes. Measurement
  contract: same clip/process/rc/warmup, discard first N, ≥200 steady chunks, CUDA-event timing. **GATE:**
  byte-identical transcript; full-step p50 ≥20% faster; p95 not worse >5%; no recapture after allowed shapes;
  reserved-mem growth <10% (or <80% device). NO-GO ⇒ skip **Step 4**. Key files: `proj-2026-05-21-0410/probe_encoder_compile.py`
  **Round-4 (C5): the as-run probe compiled `encoder.cache_aware_stream_step` for ONE random shape with
  `drop_extra=0` — it is a CONDITIONAL GO for that shape only. The real server has ≥4 B=1 shapes
  (warmup T=warmup_frames; first-chunk T=shift_frames drop 0; steady T=pre_encode_cache+shift drop self.drop_extra;
  final variable-length keep_all_outputs=True). The full-shape coverage + recapture/p95/memory gates move to Step 4.**

- [x] **2. Probe B — batched-step STATE correctness, current `greedy` (GO/NO-GO state batching)**
  N clips separately at B=1 capturing per-step text+state; then stacked B=2/4 via one `conformer_stream_step(B)`/tick
  with correct cache axes (dim1/dim0), flat per-row hyp list, real `_init_session` state. Compare per-row
  text+caches+hyps+pred_out+emitted_frames. Hard cases: row-order permute; alias sentinel; first-vs-steady
  (expect NO-GO if mixed); warmup-200; absent-then-rejoin (no padding); **injected exception mid-batch then a
  B=1 chunk** (proves drop_extra try/finally). **GATE:** byte+state identical within a group; mixed-geometry
  rejected; post-exception B=1 correct. Key files: `proj-2026-05-21-0410/probe_batched_step.py`
  **Round-4 (C2): the as-run probe is B=2 only and compares TEXT lists only — it does NOT scatter+compare
  caches / `previous_hypotheses` / `pred_out_stream` / `emitted_frames`, and the injected-exception test is
  unrun. Treat Step 2 as PARTIAL (text-equality + mid-stream stack proven). Full STATE-equality at B=2/4 +
  the injected encoder-AND-decoder exception tests are the BLOCKING Probe B2 folded into Step 5.**

- [x] **3. Probe C — decoder strategy equivalence + encode/decode split (GO/NO-GO greedy_batch + fallback)**
  Compare `greedy` vs `greedy_batch`(loop_labels=True, cuda_graph=False) at B=1 (byte+state) then B=2/4/8.
  ALSO report the per-step split: encoder-only ms, current-greedy decode-only ms (serial row loop), greedy_batch
  decode-only ms, full-step ms, at B=1/2/4/8 — to know the fallback ceiling (≈ 1/decode_fraction). **GATE:**
  (a) greedy_batch B=1 byte+state identical AND ≥25% lower decode/stream at B=4/8 + ≥1.5× full-step ⇒ use
  greedy_batch; ELSE (b) Probe-C-NO-GO fallback = batch the encoder (call `encoder.cache_aware_stream_step`
  directly) + current per-row greedy decode — only pursue if it still gives ≥1.5× end-to-end at B=4/8, else
  STOP batching (marginal). **Explicit GO/STOP decision point:** after Probe C, if greedy_batch is NO-GO
  AND the encoder-only fallback is <1.5× end-to-end at B=4/8, STOP the scheduler work (Steps 6-8) — don't
  sink the refactor for marginal gain; report and reconsider. Key files: `proj-2026-05-21-0410/probe_decoder_strategy.py`
  **Round-4 (C3): the as-run probe streams each clip at B=1 under `greedy` then B=1 under `greedy_batch` — it
  proves B=1 strategy equivalence, NOT the shipped path (`greedy_batch` at B≥2 with per-row `partial_hypotheses`
  threaded across many ticks, `max_symbols=10`). The B≥2 multi-chunk equivalence (+ row-permute, join/leave,
  `keep_all_outputs=True` final) is a BLOCKING probe before Step 7; the encode/decode split timing is still TODO.**

- [x] **4. Phase 1 — encoder graph/compile in server.py (`NEMOTRON_ENCODER_COMPILE`, B=1 only, default off)**
  If Probe A GO: wrap the `_process_chunk` encoder behind the flag (B=1 path only; warm/capture in `_warmup`;
  first/final per Probe A). `try/finally` drop_extra restore. **GATE:** flag-off rc1 byte-identical; flag-on
  byte+state identical + higher single-stream knee (local). Key files: `src/nemotron_speech/server.py`
  **Round-4 (C5): define the EXACT compiled-encoder invocation — either patch `model.encoder.cache_aware_stream_step`
  behind the flag, or split `conformer_stream_step` into {compiled encoder} + {existing RNNT decode}; leaving
  `self.model.conformer_stream_step(...)` untouched gets NO speedup (the compiled object is inside it). Warm/capture
  ONLY the exact static B=1 buckets that pass gates (warmup/first/steady); keep final/fork UNcompiled unless a
  final-shape bucket probe passes. Add compile hit/recapture telemetry + p95 + memory gates. NOTE: the (b) split
  shows idle-gaps are 46-68% of span on cloud vs 35% local → this B=1 lever is worth MORE on the slow-CPU target
  (solo/fallback/finalize/first-chunk paths) than Probe A's local 1.54× implies (N1).**

- [ ] **5. Batch state primitives — stack/unstack (correct axes), grouping key, exact ready predicate**
  Helpers: concat caches (dim1/dim0), flat unique-object hyp/pred_out lists, `processed_signal[B,F,T]`+`length[B]`,
  set the single drop_extra/prompt, inverse scatter (one-element lists). Grouping key + the exact two-guard ready
  predicate (above). Assertions for shapes + alias-freedom. `try/finally` drop_extra restore wrapper.
  **GATE:** stack→step→unstack on 2-4 same-group sessions byte+state identical to B=1 (reuses Probe B).
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-21-0410/test_batch_state.py`
  **Round-4 ADDITIONS (BLOCKING before Step 7):** (i) **Probe B2 — full STATE-equality** (the real Step-2
  gate, C2): B=2 AND B=4 using REAL `_init_session`/warmup state; scatter each row and compare cache tensors +
  `previous_hypotheses` token/decoder-state + `pred_out_stream` + `emitted_frames` + interim + final vs B=1;
  inject an exception AFTER NeMo sets `drop_extra_pre_encoded` (encoder) AND a DECODER exception mid-decode →
  prove the next B=1 chunk is correct (C10). (ii) `scatter_cache_row` returns `.clone()` (owned B=1 storage,
  C9) + a unit test for independent storage. (iii) `stack_processed` asserts row `shape[0]==1`/dtype/device;
  assert RNNT + uniform-None `pred_out`/`hyps` within a group (C-S4/S2).

- [ ] **6. Step 5a — shared scheduler infra, B=1 only (no batching)**
  Single drain/scheduler task + deduped per-session ready set + generation tokens + single model-call lane;
  sessions enqueue audio/control, scheduler owns ASR-state mutation; cancel/close/reset; bounded queues (awaited
  put, no frame drop); finalize/fork snapshot+clone+inference+FORK_ASSERT owned by the lane (telemetry:
  fork_clone_ms, model_lane_wait_ms). B=1 per tick (identical math). **GATE:** byte+state vs Step-0 baseline;
  no TTFS/interim regression vs per-handler path (N=1..4); FORK_ASSERT clean; final p95 <400ms.
  The WS protocol/handshake + Pipecat/`concurrency_test` client are UNCHANGED (the scheduler is internal) —
  do not touch the handlers' message flow. If one delegation is too big, split 6a (queue/lane/state-ownership
  refactor, byte-exact B=1) then 6b (fork/finalize migration onto the lane). Key files: `src/nemotron_speech/server.py`
  **Round-4 ADDITIONS:** (i) **Rollback flag (C7):** the scheduler is behind `NEMOTRON_SCHEDULER_B1` (or the
  batch flag) — flag-OFF executes the EXISTING per-session worker + `inference_lock` path UNCHANGED (a scheduler
  bug must not reach prod with batching off). Startup-log `scheduler_enabled`/`batch_enabled`/`decoder_strategy`/
  `encoder_compile_enabled`. (ii) **Barrier semantics (C8):** reset/end/close removes the session from the ready
  set, waits-for-or-invalidates any in-flight generation, drains/folds earlier audio into state, THEN runs
  fork/finalize on the lane; every scatter+send checks the per-session generation token before mutating/sending.
  GATE adds: finalize-while-queued, finalize-as-middle-row-in-flight, close-while-in-flight, backlog+reset.

- [ ] **7. Step 5b — steady-state batching (`NEMOTRON_BATCH_SCHED`, decoder per Probe C)**
  Group ready same-group sessions; ONE call/tick: encoder batched; decode = greedy_batch (if Probe C GO) or
  per-row via direct `cache_aware_stream_step` + greedy (fallback). Dispatch policy: on first ready start a
  `BATCH_MAX_WAIT_MS` timer; dispatch the largest safe same-group batch when timer elapses OR `BATCH_MAX_SIZE`
  hit OR all ready gathered; immediate at low load. Requeue backlog; defaults MAX_WAIT=5ms/MAX_SIZE=4.
  **GATE:** N same-lang streams byte+state identical to baseline; local knee ~14→≥21 (or +≥4); N=1 TTFS within
  the latency rule. Key files: `src/nemotron_speech/server.py`
  **Round-4 ADDITIONS:** (i) **decoder-strategy design (C4):** pick the explicit global design (flag-off = today's
  config; flag-on = `greedy_batch` GLOBALLY after B=1/warmup/final byte-exact; if finalize stays `greedy`, the
  per-call switch is lane-serialized + cost-measured). Gated by the BLOCKING `greedy_batch` B≥2 multi-chunk probe
  (C3). (ii) **dispatch policy precision (C-S2):** if only one eligible session → dispatch immediately; else start
  a `BATCH_MAX_WAIT_MS=5` coalescing timer on first-ready, dispatch the largest safe same-group batch when
  `MAX_SIZE` hits OR timer expires (don't rely on the unobservable "all ready gathered"; don't let "immediate at
  low load" silently collapse N=2/4 to B=1). Track effective-batch-size histogram + queue-wait p95.
  - [ ] **7a. Batch the preprocessor (C6/S1 — the co-equal server-layer term).** Stack same-ready-group fixed
    audio → `[B,K]` → ONE `preprocessor` call, slice per-row mel, then the batched model. **GATE:** batched-B
    preprocessor byte-equal to per-row B=1 (its own probe — batched cuFFT plan risk); if NOT byte-exact, keep
    serial preprocess, cap MAX_SIZE, and LOWER+document the knee target. Measure per-stream preprocess on L4
    first to confirm it's the post-model-batch bottleneck. Key files: `src/nemotron_speech/server.py`

- [ ] **8. Step 5c — variable-B + fail-closed + memory + telemetry**
  Streams join/leave mid-flight (rebuild batch/tick), row-order independence, fairness (no backlogged-stream
  monopoly), fail-closed config checks (beam/CTC/EOU/unsafe → split or B=1), memory gate B∈{1,2,4,8,16}
  (no OOM, startup device-specific max), telemetry (batch size, grouping reason, queue wait, model/preproc ms,
  per-session lag, fallback count, prompt lang, mem). **GATE:** mixed-lang + join/leave + permute byte+state
  identical; unsupported configs fall back safely; no OOM at default max. Key files: `src/nemotron_speech/server.py`
  **Round-4 ADDITIONS (C-S5, C9, C-N3):** memory gate measures `max_memory_reserved` AND active bytes
  before/after each batch AND RETAINED session-cache bytes after streams leave/reset (validates clone-on-scatter)
  at B=16 WITH session churn — not just peak. Telemetry adds: compile hit/miss/recapture, eligible-ready count,
  preprocessor-batch ms vs model-batch ms (the S1 check), scatter/postprocess ms, model-lane utilization;
  startup-assert `use_cuda_graph_decoder=False` under the batch flag.

- [ ] **9. Local validation — byte-exact + latency + knee (compile×batch matrix)**
  Matrix per the compile-batch semantics: {compile-only B=1}, {batch-only}, {compile(B=1)+batch(B>1 uncompiled)}.
  Byte+state diff vs Step-0; TTFS/interim/final p50/p95/p99 at N=1 (latency rule); `concurrency_test.py` knee
  (local ~14 baseline). Document the improvement. **GATE:** byte-exact everywhere; knee strictly up; latency
  within rule; no OOM. Key files: `proj-2026-05-19-eou-endpointing/concurrency_test.py`, `proj-2026-05-21-0410/` notes
  **Round-4 ADDITIONS (C-S1, C-S3):** the byte-exact gate compares flag-on vs a flag-OFF baseline captured on the
  SAME machine/commit (not cross-env artifacts). Add the **strict cross-talk gate**: N DISTINCT clips concurrent,
  each stream's interim sequence + final + delta + no-dup STRICT-byte-equal to its solo flag-off run (extend
  `concurrency_test.py` beyond its final-only/edit-distance compare); fast 24-clip every step + a **1000-sample
  canary** here before Step 10.

- [ ] **10. Modal re-sweep — knee/$ vs the batch=1 baseline**
  Deploy batched server (`asr_bench_modal.py`, flag on); re-sweep T4, L4, A100, H100, RTX-PRO-6000 (exclude
  B200; RTX-PRO-6000 needs the patient 600s smoke). Compare knee+$/stream to RESULTS.md baseline (T4 ~5/$0.12);
  update before/after. Stop apps between. **GATE:** documented per-GPU knee/$ improvement (Modal cheap GPU
  ~5→≥10), byte-exact. Key files: `src/nemotron_speech/modal/asr_bench_modal.py`, `proj-2026-05-20-modal-cost/RESULTS.md`
  **Round-4 ADDITIONS (C-S3, C6/S1):** run a short flag-OFF baseline IN the deployed Modal container before
  flag-on and compare exact outputs THERE (cross-env cuFFT differences confound strict byte comparison; RESULTS.md
  knee is for perf only). Report preprocessor-batch ms vs model-batch ms per tick to confirm whether preprocess
  became the post-batch bottleneck (the S1 empirical check).

- [ ] **11. Consolidate — docs + recommendation + memory**
  Design (scheduler, axes, grouping, flags, decoder/compile decisions), before/after knee/$ table, residual
  risks, the load-shape caveat (throughput scales with in-phase concurrency), production recommendation (batch
  size, GPU, self-host vs Modal). Update docs + memory. Key files: `proj-2026-05-21-0410/`, `docs/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 0 | Baseline artifact (+ git identity) | **done** | 2a2b96c | 0a: `baseline/english_baseline.json` 8 clips. 0b: `baseline/multilingual_baseline.json` — 24 recs (8×{en-US,es-ES,auto}), 19 non-empty, **0 tag-leak**, prompted [56,3] confirmed, git 1deac460. **FINDING: ml checkpoint geometry differs — `chunk_size=[25,32]` shift **320ms** (32 frames) + final-pad **1280ms** (vs en 160ms/16); chunk_T grouping keeps en/ml from ever batching together (correctness-safe), but Steps 7-10 ml knee/latency math must use 320ms.** |
| 1 | Probe A: encoder compile (GO/NO-GO Step 4) | **done — CONDITIONAL GO (R4 C5)** | (uncommitted) | `probe_encoder_compile.py`: torch.compile on encoder.cache_aware_stream_step + correct (Δ=2e-6) + **1.54× faster** (7.9→5.1ms) — but ONE shape (drop_extra=0). Real-shape coverage (warmup/first/steady/final) + invocation design moved to Step 4. Worth MORE on cloud (gaps 46-68% of span). |
| 2 | Probe B: batched-step STATE + drop_extra exception | **partial — TEXT-GO (R4 C2)** | (uncommitted) | `probe_batched_step.py`: batched(B=2)==separate(B=1) BYTE-IDENTICAL text, incl. row-permute AND mid-stream dim-1 stack. BUT text-only — caches/hyps/pred_out/emitted_frames NOT compared, exception test unrun → **full STATE-equality + enc&dec exception = Probe B2 (Step 5), BLOCKING Step 7**. |
| 3 | Probe C: decoder strategy + encode/decode split | **done — GO @B=1 ONLY (R4 C3)** | (uncommitted) | `probe_decoder_strategy.py`: `greedy_batch`==`greedy` BYTE-IDENTICAL at **B=1**, all clips. NOT proven at B≥2 multi-chunk (the shipped path) → **`greedy_batch` B≥2 multichunk probe BLOCKING Step 7**. encode/decode split timing still TODO. |
| 4 | Phase 1: encoder compile B=1 (flag) | **done** | _pending_ | Approach (b): compiled handle swapped into `encoder.cache_aware_stream_step` for static buckets {(20,0),(16,0),(25,2)} via `_conformer_stream_step`, restored in finally; **CUDA-graph cache outputs (idx 2,3,4) cloned** out of the static pool; final/fork uncompiled; prompted_model disables compile; dedicated 1-thread executor for graph consistency. GATE: flag-off 8/8 + flag-on 8/8 byte-exact, **1.45× step** (10.01→6.89ms), 0 recapture, FORK_ASSERT clean. (.venv-asr/torch2.11; bundles pre-existing NEMOTRON_PROFILE_CHUNK instrumentation.) |
| 5 | Batch state primitives (axes, ready predicate, try/finally) | **partial** | (uncommitted) | `src/nemotron_speech/batch_primitives.py` (pure stack/scatter/group-key/hyp-flatten) + `test_batch_primitives.py` PASS (dim1/dim0 round-trip, alias guard, ragged reject). In-server wiring + ready-predicate + try/finally = with scheduler (Steps 6-7). **R4: + Probe B2 state-equality (BLOCKING), clone-on-scatter (C9), B=1/dtype/uniform-None asserts (C-S4/S2).** |
| 6 | 5a: scheduler infra B=1 (+ fork lane) | pending | — | **R4: rollback flag (C7) + reset/close barriers + generation-token scatter (C8).** |
| 7 | 5b: steady-state batching (flag, decoder per Probe C) | pending | — | core. **R4: explicit global-decoder design (C4) + greedy_batch B≥2 multichunk probe (BLOCKING, C3) + precise dispatch (C-S2).** |
| 7a | Batch the preprocessor (byte-gated) | pending | — | **R4 (C6/S1): co-equal server-layer term — stack `[B,K]`→1 preprocessor call; own cuFFT byte-equality probe or cap MAX_SIZE.** |
| 8 | 5c: variable-B + fail-closed + memory + telemetry | pending | — | |
| 9 | Local validation (compile×batch matrix + latency + knee) | pending | — | |
| 10 | Modal re-sweep (knee/$ vs baseline) | pending | — | |
| 11 | Consolidate docs + recommendation | pending | — | |
