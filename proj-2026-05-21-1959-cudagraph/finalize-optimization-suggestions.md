# Finalize Optimization Suggestions

Date: 2026-05-22. Scope: analysis only. No runtime code changes.

Target config assumed here: L40S/Ada (`sm_89`) production shape with continuous scheduler, lanes=2, `NEMOTRON_ENCODER_CUDAGRAPH=1`, `NEMOTRON_WARMUP_MS=200`, rc1, and the `silence0_warm200` benchmark shape. The measured TTFS decomposition in the prompt is treated as ground truth: the target is the server-side finalize compute component, especially the P95 tail.

## Critical Path Trace

1. WebSocket control event routing:
   - Non-continuous mode calls `_reset_session(..., finalize=...)` directly on `reset`/`end` (`src/nemotron_speech/server.py:3171-3175`).
   - Continuous scheduler mode queues `("reset", finalize, msg_type)` (`server.py:3316-3319`), drains it through `_scheduler_process_event` under `session.state_lock` (`server.py:4496-4507`), then dispatches to `_scheduler_continuous_handle_reset_locked` (`server.py:4442-4466`).
   - Non-scheduler continuous mode has the same ordered queue shape (`server.py:3229-3251`, `server.py:4893-4924`).

2. `vad_stop` debounce:
   - Default server debounce is 150 ms (`server.py:54`, `server.py:763-773`).
   - `vad_stop` sets `PENDING_FINALIZE`, records `continuous_vad_stop_ts`, and starts `_continuous_debounce_timer` (`server.py:4693-4712` scheduler, `server.py:5165-5180` non-scheduler).
   - The timer sleeps for `finalize_silence_seconds`, queues `("debounce_expired", stop_seq)`, and wakes the scheduler (`server.py:4994-5002`).
   - Important: if the client sends explicit `reset/finalize=true` while the session is already `PENDING_FINALIZE`, the code only sets `continuous_reset_seen=True` and returns; it does not immediately finalize (`server.py:4761-4767` scheduler, `server.py:5224-5230` non-scheduler). The eventual finalize happens on `debounce_expired` (`server.py:4792-4824`, `server.py:5251-5279`).
   - `NEMOTRON_BATCH_FINALIZE` only batches `debounce_expired` events (`server.py:3608-3618`), and then prepares/finalizes all collected events in one pass (`server.py:3620-3679`).

3. Fork construction:
   - The final fork appends rc-dependent silence padding: `final_padding_frames = (right_context + 1) * shift_frames`; for rc1 and 16-frame shifts this is 32 frames / 320 ms (`server.py:1436-1450`, `server.py:5295-5299`).
   - `NEMOTRON_WARMUP_MS=200` makes fresh sessions non-first from the model's perspective by running `_run_session_warmup`, setting `session.emitted_frames = warmup_frames` and retaining a mel ring (`server.py:2587-2642`). In production warm200, the final path normally takes the "not first chunk" branch.
   - The fork deep-copies pending audio, raw ring, mel ring, encoder caches, previous hypotheses, pred state, and text bookkeeping (`server.py:5288-5338`). `fork_clone_ms` is already recorded around this call in both the batched and serial finalize paths (`server.py:5480-5487`, `server.py:6270-6277`).

4. Final preprocessing and model call:
   - Serial finalization computes `remaining_frames`, loops in chunks of at most `shift_frames`, runs `_preprocess_fixed_audio` once per loop, concatenates all final mels, prepends `mel_frame_ring`, and calls `_conformer_stream_step(... keep_all_outputs=True ...)` (`server.py:7226-7315`).
   - Batched finalization does the same per fork in `_prepare_final_fork_batch_row` (`server.py:5687-5755`) or, with `NEMOTRON_BATCH_FINALIZE_PREPROC=1`, groups preprocessor calls by exact `(valid_samples, frames_this_call)` (`server.py:5827-5956`). It then stacks rows and calls `_conformer_stream_step(... keep_all_outputs=True ...)` in `_process_final_batch_rows` (`server.py:6073-6133`).
   - The final response is sent after delta extraction (`server.py:5623-5670` batched, `server.py:6364-6404` serial). True-boundary cold reset happens after client-visible final emission (`server.py:6486-6521`) and should not be counted as this session's TTFS, though it can block other sessions if serialized.

5. CUDA graph gating:
   - The current encoder graph manager captures only `encoder.cache_aware_stream_step`, not RNNT decode (`src/nemotron_speech/cudagraph_encoder.py:1-4`).
   - It captures only one steady geometry per exact batch size: `steady_T = pre_encode_cache_size + shift_frames` and `drop_extra = streaming_cfg.drop_extra_pre_encoded` (`cudagraph_encoder.py:209-213`, `cudagraph_encoder.py:252-260`).
   - The server only installs the graph wrapper when `keep_all_outputs` is false, `bypass_pre_encode` is false, `chunk_frames == pre_encode_cache + shift`, and `drop_extra == self.drop_extra` (`server.py:1828-1860`). If not, `_conformer_stream_step` falls back to eager (`server.py:1942-1981`).
   - Therefore the final call bypasses the current encoder graph for two independent reasons: final calls set `keep_all_outputs=True` (`server.py:6122-6133`, `server.py:7304-7315`) and final `T` is not the steady 25-frame bucket.

## Where The Time Likely Goes

Measured:

- The current client-side cloud result decomposes L4 TTFS into about 200 ms harness VAD window, 23 ms network, and 67 ms median / 224 ms P95 server-side finalize compute. This document targets that final component.
- Prior single-session local warm200 telemetry showed the old no-contention final fork flush was only 13.4 ms median / 14.3 ms P95 with 0 ms lock wait (`proj-2026-05-17-1708/codex-jobs/step-8-warm200-run.out:57-64`). That means the new cloud tail is not inherent to the algorithm alone; it is cloud/MPS/load/path specific.
- The steady encoder graph cloud retest showed exactly the failure mode this finalize path still has: on L40S, graph-off K=4/MPS p95 was 349-767 ms, while graph-on held 64 streams with p95 216 ms and removed MPS bifurcation (`proj-2026-05-21-1959-cudagraph/cloud-retest.md:33-53`). That is measured for steady chunks, not final chunks, but it strongly supports launch/eager variance as a P95 cause.

Inferred from code:

- Final non-first chunk geometry is usually `T = pre_encode_cache_size + remaining_frames`. With warm200, `emitted_frames != 0`, so the final branch prepends the mel ring and uses `drop_extra=self.drop_extra` (`server.py:7286-7293`). Normal streaming leaves less than one ready chunk of real pending audio before finalization (`server.py:6564-6587`, `server.py:6980-6984`). With rc1 padding this implies roughly `remaining_frames = 33..49`, so final `T` is roughly `42..58`, not steady `T=25`.
- Final preprocessing is serial per stream in the normal path and can run three to four fixed-shape preprocessor calls for rc1 final tails (`server.py:7254-7283`). `NEMOTRON_BATCH_FINALIZE_PREPROC` batches these only across simultaneous finalize rows, not within a single final row (`server.py:5827-5956`).
- Final batch model execution has explicit CUDA synchronizations at entry, after model call, and after scatter/postprocess (`server.py:6027-6065`, `server.py:6134-6135`, `server.py:6187-6188`). Those make timing easy but can contribute to tail and reduce overlap.
- Final lane wait is real and already partially observable: `inference_lock_acquire_wait_ms` is included in `finalize_timing` (`server.py:5430-5444`, `server.py:5571-5607`, `server.py:6278-6351`). For batch-finalize with lanes, work uses the session's pinned lane (`server.py:5571-5589`, `server.py:6280-6310`); without `NEMOTRON_BATCH_FINALIZE`, the lane path falls back to the global exclusive model path plus `inference_lock` (`server.py:6313-6333`).

## Ranked Suggestions

### 1. Add finalize encoder CUDA graph buckets

Target:

- Final `_conformer_stream_step(... keep_all_outputs=True ...)` in `_process_final_chunk` and `_process_final_batch_rows` (`server.py:6073-6133`, `server.py:7226-7315`).
- Current graph exclusion gates (`server.py:1828-1860`) and steady-only manager (`cudagraph_encoder.py:209-213`).

Mechanism:

- Add a separate, default-off finalize encoder graph path, for example `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1`.
- Keep decode eager. Graph only the encoder cache step, matching the current manager pattern (`cudagraph_encoder.py:80-183`, `server.py:1881-1978`).
- Capture exact `(B, T, drop_extra, keep_all_outputs=True)` buckets. Start with L40S production-safe buckets: `B=1..2` or `B=1..4`, and final `T` values from telemetry/histogram. Avoid padding at first; exact buckets preserve the byte-exact story.
- Fail closed per bucket: if capture/replay fails or shape mismatches, use eager exactly as the current steady manager does (`cudagraph_encoder.py:284-337`).
- Add bucket telemetry: final `T`, final `B`, `drop_extra`, replay/fallback counts, capture memory.

Expected impact:

- Median: likely the largest single compute win. On L4, a plausible target is cutting the 67 ms server median toward the 40-60 ms range if final encoder launch overhead dominates. On L40S, expect a smaller median absolute win but still meaningful under K=4/MPS.
- P95: likely the highest-value tail lever. The steady cloud retest proved CUDA graphs remove MPS launch starvation on L40S (`cloud-retest.md:46-53`). Final calls currently remain eager, variable-T launch storms, so expect P95 reduction on the order of tens to low hundreds of ms depending on how much of the 224 ms L4 tail is eager encoder launch versus lock/lane wait.

Byte-exactness/risk/gating:

- Medium risk. `keep_all_outputs=True` can change encoder output shape and downstream decode behavior, so this must be a separate graph family, not a reuse of the steady graph.
- Gate default-off; English rc1 byte-identical final transcripts and final deltas versus current server; `NEMOTRON_FORK_ASSERT=1`; capture only exact buckets; no padding until exact buckets are proven.
- Do not change the fork semantics. Parent-state corruption is the core hazard protected by `_build_continuous_finalize_fork` (`server.py:5288-5338`) and fork assertions (`server.py:5340-5401`).

Effort:

- Medium-large. Reuses most of `BucketedCudaGraphEncoder`, but needs a new bucket key and capture call shape for final `keep_all_outputs=True`.

### 2. Collapse per-stream final preprocessing into one tail preprocessor call

Target:

- Serial final preprocessing loop (`server.py:7254-7283`) and equivalent batch-row loop (`server.py:5715-5744`).
- Constant preprocessor plan already includes `final_padding_frames` in the max shape (`server.py:1474-1482`) and accepts variable `valid_samples` with fixed tensor shape (`server.py:2659-2679`, `server.py:2681-2704`).

Mechanism:

- Add a flag-gated one-shot final-tail preprocessor path, for example `NEMOTRON_FINALIZE_SINGLE_PREPROC=1`.
- Instead of running the fixed preprocessor once per 16-frame slice, build one fixed audio buffer containing raw-ring context plus the whole pending+padding final tail, run `_preprocess_fixed_audio` once, and slice `remaining_frames` from `first_preprocess_mel_frame`.
- Preserve the current output contract: if `emitted_frames == 0`, no mel ring and `drop_extra=0`; otherwise prepend `mel_frame_ring` and use `drop_extra=self.drop_extra` (`server.py:5744-5755`, `server.py:7283-7293`).

Expected impact:

- Median: likely 5-20 ms on cloud, smaller on local. The serial path can do about three to four preprocessor invocations per rc1 final, so this removes repeated CUDA preprocessing launches and repeated NumPy buffer construction.
- P95: likely 20-60 ms if preprocessor launch/CPU staging is a visible part of the cloud tail. It also reduces event-loop/GIL allocation pressure during finalize bursts.

Byte-exactness/risk/gating:

- Low-medium risk if exact STFT frame equality holds; high enough to gate because prior batched final preprocessing dropped punctuation when final tails were mishandled (`proj-2026-05-21-inference-opt/round4-finalize-preproc.md:20-25`).
- Gate default-off; byte-identical mel hashes and final transcripts on rc1 English; include punctuation-heavy cases; keep multilingual prompted path because prompt application occurs before the model call (`server.py:6111-6113`, `server.py:7295-7296`) and should remain unchanged.

Effort:

- Medium. It is localized but needs careful frame-index proof and exact replay tests.

### 3. Make explicit client finalization bypass server debounce when configured

Target:

- Reset during `PENDING_FINALIZE` currently waits for the debounce timer (`server.py:4761-4767`, `server.py:5224-5230`).
- Debounce timer path (`server.py:4994-5002`) and expiry handlers (`server.py:4792-4824`, `server.py:5251-5279`).

Mechanism:

- Add a default-off forced-explicit-finalize mode, for example `NEMOTRON_RESET_FORCE_FINALIZE=1`, ideally requiring a request field like `{"type":"reset","finalize":true,"force":true}`.
- If `force` is present while `PENDING_FINALIZE`, cancel the debounce and run the same speculative finalize path immediately, retaining context as current `reset` does. Do not turn `reset` into a true-boundary cold reset; only `end`/`close` are true-boundary reasons today (`server.py:60`, `server.py:6486-6497`).
- For `silence0_warm200`, this mostly removes timer/queue jitter. For any accidental/default 150 ms deployment, it removes the full server-side debounce from TTFS.

Expected impact:

- Median/P95 if `NEMOTRON_FINALIZE_SILENCE_MS=150`: about 150 ms off both, plus timer scheduling jitter.
- Median/P95 in the stated `silence0_warm200` config: near zero median; maybe low-ms P95 improvement by avoiding the sleep(0) task and scheduler round trip.

Byte-exactness/risk/gating:

- Medium behavioral risk. The 150 ms window protects false VAD stops and barge-in. Only enable for clients that explicitly declare end-of-turn after they have sent all audio.
- Gate default-off; run barge-in tests where `vad_start` arrives after forced finalization; verify no duplicate final deltas and no parent-state corruption.

Effort:

- Small-medium.

### 4. Ensure production enables finalize batching and batched final preprocessing

Target:

- Feature gates: `NEMOTRON_BATCH_FINALIZE` and `NEMOTRON_BATCH_FINALIZE_PREPROC` (`server.py:570-575`, `server.py:3369-3380`).
- Batched finalize event path (`server.py:3608-3679`, `server.py:5555-5607`, `server.py:6016-6195`).
- Legacy lane finalize fallback when batch finalize is off (`server.py:6313-6333`).

Mechanism:

- If production does not already set both flags, turn them on behind the existing default-off gates after a cloud canary.
- With `NEMOTRON_BATCH_FINALIZE=1`, final work uses pinned model lanes instead of the global exclusive model path for lane deployments (`server.py:5571-5589`, `server.py:6280-6310`).
- With `NEMOTRON_BATCH_FINALIZE_PREPROC=1`, simultaneous final preprocessor calls are grouped by exact valid length and real frame count (`server.py:5827-5956`).

Expected impact:

- Median: small for isolated/out-of-phase single finalization.
- P95: medium to large when multiple streams finalize together or when finals contend with steady lane work. Local in-phase data already showed `NEMOTRON_BATCH_FINALIZE` improved p95 at every level but did not finish the job (`round3-finalize-storm.md:64-99`), and `NEMOTRON_BATCH_FINALIZE_PREPROC` moved the local strict in-phase knee from 130 to 140 (`round4-finalize-preproc.md:95-120`).
- For the measured internet out-of-phase TTFS, this is probably a tail robustness lever rather than the median lever.

Byte-exactness/risk/gating:

- Already designed as default-off and byte-exact locally (`round3-finalize-storm.md:27-45`, `round4-finalize-preproc.md:56-76`).
- Still run L40S production canary because K=4/MPS lane timing is the target.

Effort:

- Small if flags already exist in deployment wiring; medium if cloud canary automation and rollback need to be added.

### 5. Reduce duplicate fork-state cloning in the batched finalize path

Target:

- Fork deep-clones hypotheses and pred state (`server.py:5332-5333`).
- Batched finalize then clones those fork states again before stacking (`server.py:6091-6100`).
- Clone timing is already recorded as `fork_clone_ms` (`server.py:5480-5487`, `server.py:6270-6277`).

Mechanism:

- First, split `fork_clone_ms` into audio/padding, cache tensors, hypotheses, pred state, and optional assert snapshot/compare.
- If clone cost is visible at P95, remove one of the two deep-copy layers in the batched finalize path:
  - either build the fork with parent references for hypotheses/pred state and clone exactly once while stacking, or
  - keep the fork's deep clone and stack without another deep clone.
- Pool or reuse fixed silence padding arrays and final fork audio buffers where safe; avoid repeated `np.concatenate` allocation for padding (`server.py:5295-5300`).

Expected impact:

- Median: likely 1-10 ms only if decoder state has grown.
- P95: could be 10-30 ms in clone/GC-heavy tails, especially with batch finalize active and many simultaneous forks. Unknown until split telemetry exists.

Byte-exactness/risk/gating:

- Medium-high aliasing risk. This is exactly the corruption hazard the fork was built to avoid. Keep `NEMOTRON_FORK_ASSERT=1` in gates; do not ship clone elision unless parent cache, hypotheses, and pred state remain byte-identical after fork flush (`server.py:5340-5401`).
- Default-off, or internal canary-only until proven.

Effort:

- Small to instrument; medium to safely remove duplicate copies.

### 6. Trim finalize-path CUDA synchronizations and timing overhead

Target:

- `_process_final_fork_groups` synchronizes at entry and exit (`server.py:6027-6065`).
- `_process_final_batch_rows` synchronizes after model call and after scatter (`server.py:6134-6135`, `server.py:6187-6188`).
- Lane calls synchronize lane streams after every call (`server.py:2128-2141`).

Mechanism:

- Keep synchronizations needed for CPU-visible text extraction and safe stream ownership, but move pure measurement syncs behind a profiling flag or replace them with CUDA events.
- Avoid memory snapshots/telemetry on every batch unless enabled; keep summary counters cheap.

Expected impact:

- Median: probably small, 1-5 ms.
- P95: 5-20 ms if the syncs amplify MPS contention or prevent overlap between lane streams. This is a tail cleanup after the graph/preprocessor fixes.

Byte-exactness/risk/gating:

- Medium. Synchronization changes can surface latent async ordering bugs. Gate with CUDA_LAUNCH_BLOCKING on/off, `FORK_ASSERT`, and byte-identical final deltas.
- Default-off or profiling-gated first.

Effort:

- Medium.

### 7. Ada-only RNNT decoder CUDA graph experiment

Target:

- Decoder config explicitly sets `use_cuda_graph_decoder=False` for both `greedy_batch` and `greedy` (`server.py:1373-1403`).
- Batch startup disables batching if it detects a CUDA graph decoder (`server.py:892-915`).
- The current encoder graph manager explicitly leaves decode eager (`cudagraph_encoder.py:1-4`).

Mechanism:

- Add a separate experiment flag, not a default: `NEMOTRON_DECODER_CUDAGRAPH_ADA=1`.
- Allow only CUDA capability 8.9 / L40S-Ada at first; keep disabled on Blackwell. Do not remove the Blackwell assert globally; make the assert architecture-aware.
- Test both `greedy_batch` and final `keep_all_outputs=True` paths, because finalize decode may have different output lengths from steady decode.

Expected impact:

- Median: probably small. Existing split profiling with decoder graph off showed decode at about 3.17 ms of 48.17 ms on L4 and 1.15 ms of 13.83 ms on RTX PRO 6000 Blackwell (`proj-2026-05-20-modal-cost/profile_split_L4.out:197-203`, `proj-2026-05-20-modal-cost/profile_split_rtxpro6000.out:197-203`). L40S/Ada is likely in the low-single-ms range for steady chunks.
- P95: could still be useful if decoder has launch variance during final trailing-silence decode, but it is unlikely to beat finalize encoder graphs.

Byte-exactness/risk/gating:

- Medium-high. Decoder graphs touch autoregressive RNNT state, and previous code disabled them for Blackwell compatibility. Gate default-off, Ada-only, English rc1 byte-identical, multilingual prompted smoke, and fallback to eager on any graph error.

Effort:

- Medium, mostly validation risk rather than code size.

### 8. Build finalize-tail attribution telemetry before choosing clone/sync/GC work

Target:

- Current `finalize_timing` has `vad_stop`, `debounce_expiry`, `fork_flush_start`, `fork_flush_done`, `final_sent`, and `inference_lock_acquire_wait_ms` (`server.py:5430-5444`, `server.py:6241-6249`), but it does not split preprocessor, encoder, decoder, stack/scatter, clone, CUDA sync, or scheduler queue time.

Mechanism:

- Add profiling-only telemetry for:
  - reset/control event received time,
  - debounce wait,
  - scheduler queue wait,
  - lane/inference lock wait,
  - fork clone split,
  - final preprocessor wall time/count,
  - final model call wall time and final `T/B/drop_extra`,
  - decoder share if NeMo can be split as in `profile_split.py`,
  - CUDA sync time,
  - Python GC pauses.
- Keep off by default; sample or aggregate to avoid perturbing production.

Expected impact:

- No direct latency reduction, but it prevents spending engineering time on small components. It is the safest way to decide whether ranks 5 and 6 matter after ranks 1 and 2.

Byte-exactness/risk/gating:

- Low if profiling is default-off and non-invasive. Avoid unconditional `torch.cuda.synchronize()` in production telemetry.

Effort:

- Small-medium.

### 9. Do not prioritize removing the fork or shortening rc1 padding under the current constraints

Target:

- Fork isolation (`server.py:5288-5338`) and rc1 final padding (`server.py:1446-1450`, `server.py:5295-5299`).

Mechanism:

- Removing the fork would be risky because final padding and `keep_all_outputs=True` mutate cache/hypothesis state. The current code explicitly protects the live stream with a disposable fork and optional parent-state assertions (`server.py:5340-5401`).
- Shortening final padding or switching rc1/rc0 can reduce final `T`, but it is not byte-exact by definition. Treat it as a product-quality experiment, not a latency optimization under the current byte-exact rc1 constraint.

Expected impact:

- Potentially meaningful compute reduction if accepted, because rc1 contributes 32 final padding frames. But it violates the "rc1 English byte-identical" constraint unless kept as a separate baseline.

Byte-exactness/risk/gating:

- High. Not recommended for this optimization pass.

Effort:

- Small to test, high to productize safely.

## Seed Hypotheses

1. Finalize encoder call bypasses CUDA graph: confirmed. `_encoder_cudagraph_bucket_for_call` rejects `keep_all_outputs=True` and any `T != pre_cache + shift` (`server.py:1835-1855`); final calls use `keep_all_outputs=True` (`server.py:6128`, `server.py:7310`) and inferred `T ~= 42..58` in warm200 rc1, not steady `T=25`.

2. 150 ms VAD-cancel debounce on explicit reset path: conditionally confirmed. If `NEMOTRON_FINALIZE_SILENCE_MS=150`, explicit `reset` during `PENDING_FINALIZE` waits for debounce (`server.py:4761-4767`, `server.py:5224-5230`). In the stated `silence0_warm200` config, the 150 ms window should not be in the measured 67/224 ms server compute, except for low-ms timer/scheduler overhead.

3. Fork clone cost: plausible but unquantified from the provided data. Code deep-clones all relevant cache/decoder state (`server.py:5288-5338`) and logs `fork_clone_ms` (`server.py:5480-5487`, `server.py:6270-6277`). Batched finalize appears to clone decoder state twice (`server.py:5332-5333`, `server.py:6091-6100`), so it is worth measuring before optimizing.

4. Redundant final preprocessing/padding work: confirmed by code. Final paths loop over `remaining_frames` in `shift_frames` slices and run `_preprocess_fixed_audio` per slice (`server.py:5715-5744`, `server.py:7254-7283`). The fixed preprocessor plan appears large enough for the rc1 final tail (`server.py:1474-1482`), making one-shot final preprocessing a concrete candidate.

5. P95 tail causes:
   - Eager final encoder launch variance: confirmed as likely; final path bypasses graph and steady graph removed similar L40S MPS launch tails.
   - Lock/lane contention: applies; code records lane/inference wait and has a global exclusive fallback when finalize batching is off (`server.py:5571-5607`, `server.py:6313-6333`).
   - Scheduler finalize-batch grouping latency: not a major artificial wait; batch finalize collects already-queued `debounce_expired` events in the current drain pass, with no `batch_max_wait_ms` delay (`server.py:3459-3488`, `server.py:3608-3618`).
   - CUDA sync points: applies; finalize batch has multiple explicit synchronizations (`server.py:6027-6065`, `server.py:6134-6135`, `server.py:6187-6188`).
   - Python GC/asyncio scheduling: possible but not proven; clone-heavy finalize and event queues can contribute, but current telemetry does not isolate them.

6. RNNT greedy decode eager and Ada decoder graph: confirmed. Decode graphs are disabled in config (`server.py:1373-1403`) and batch startup disables batching if a graph decoder is detected (`server.py:892-915`). Because L40S is Ada, not Blackwell, an Ada-only experiment is reasonable, but expected impact is lower than encoder-final graphs because prior split profiles show decode as a small share of steady chunk wall time.

7. Other finding: if production has `NEMOTRON_BATCH_FINALIZE` or `NEMOTRON_BATCH_FINALIZE_PREPROC` off, enabling them is the fastest low-code tail hardening. The code already contains byte-exact local work for final model batching and batched final preprocessing; cloud canary is still required.

## Highest-Value Next Step

Implement and canary `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1` for exact final encoder buckets on L40S, starting with histogram-driven `B=1..2`/top-final-`T` buckets and fail-closed eager fallback.

Why: it targets the component most directly implicated by both the code and the cloud retest. The final path is still eager while the steady path's graph removed the L40S MPS p95 starvation. It preserves the fork and decoder semantics, can be default-off, and can be made byte-exact by exact bucket capture rather than padding.
