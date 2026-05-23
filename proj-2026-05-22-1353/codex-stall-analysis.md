# Steady Batch Stall Analysis

Date: 2026-05-23

Scope: code-path investigation only. I did not modify `src/nemotron_speech/server.py`.

## Bottom line

The observed 690-1178 ms steady `model_batch_ms` stalls are not explained by CUDA allocator retries, GC, or lazy encoder CUDA graph capture. The code and L40S log evidence point to a CUDA stream/device backlog inside `_conformer_stream_step`, exposed by the forced lane stream sync at `src/nemotron_speech/server.py:8329`. The most likely work inside that backlog is the eager RNNT batched label-loop decoder, which the server explicitly configures with `use_cuda_graph_decoder=False` at `src/nemotron_speech/server.py:1463-1474`.

In other words: the batch timer is mostly measuring `model.conformer_stream_step(...)` plus the immediate stream synchronization, and the encoder portion is already graph replaying with zero observed fallbacks. The remaining dynamic, high-launch, host-controlled part is RNNT decode. Two model lanes make the tail worse because both lanes submit GPU work concurrently and then block on their own stream; a device-level scheduling/backlog hiccup can make both lane workers report long model or scatter waits.

The SIGUSR1 stack in `scatter_cache_row` is consistent with this. `scatter_cache_row` is not CPU-only Python; it launches GPU clones at `src/nemotron_speech/batch_primitives.py:90-97`. A lane worker can appear wedged there when a clone or later sync is waiting behind previously queued GPU work. Scatter is still worth optimizing, but it is probably a symptom/amplifier, not the primary 1 s root.

## Evidence

- `model_batch_ms` includes only the model call and the post-model lane sync:
  - call: `src/nemotron_speech/server.py:8317-8328`
  - sync: `src/nemotron_speech/server.py:8329`
  - timer ends: `src/nemotron_speech/server.py:8330`
- The lane executor also synchronizes after `_process_ready_batch` returns:
  - `src/nemotron_speech/server.py:3175-3177`
- Internal helper syncs are placed at batch entry, after preproc, after model, and after scatter:
  - `src/nemotron_speech/server.py:8223`, `8243`, `8329`, `8399`
- Steady encoder CUDA graphs are startup-captured, not lazily captured:
  - startup capture: `src/nemotron_speech/server.py:1671-1683`, `1813-1986`
  - capture loop: `src/nemotron_speech/cudagraph_encoder.py:391-423`
  - replay returns `None` on mismatch instead of capturing: `src/nemotron_speech/cudagraph_encoder.py:507-560`
- The measured L40S run captured B=1..8 at startup for default + both lane models, and later `encoder_cuda_graph_status` reported `fallbacks=0` through more than 51k replays. This rules out steady lazy recapture/fallback for the conc-10 stalls in that log.
- The server disables RNNT decoder CUDA graphs:
  - `src/nemotron_speech/server.py:1463-1474`
- NeMo batched greedy decode selects label-looping decode and passes `allow_cuda_graphs=self.use_cuda_graph_decoder`:
  - `/home/khkramer/src/nemotron-nano-omni/NeMo/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py:630-648`
- The graph-disabled RNNT label-loop implementation uses host-controlled loops over GPU tensors:
  - torch eager loops: `/home/khkramer/src/nemotron-nano-omni/NeMo/nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:356-409`
  - graph/no-graph dispatch and `.item()` loop conditions in graph modes: `/home/khkramer/src/nemotron-nano-omni/NeMo/nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:714-731`
  - active-mask production: `/home/khkramer/src/nemotron-nano-omni/NeMo/nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:1058-1063`

## Ranked candidate causes

1. **Eager RNNT decode under two concurrent model lanes.**
   - Primary files: `src/nemotron_speech/server.py:1463-1474`, `src/nemotron_speech/server.py:8317-8329`, NeMo `rnnt_greedy_decoding.py:767-820`, NeMo `transducer_decoding/rnnt_label_looping.py:356-409`.
   - Why it fits: encoder replay fallbacks are zero, allocator retries are zero, preproc/scatter are separately timed, and the remaining eager portion is RNNT decode. Label-loop decode performs dynamic loops and many small GPU operations. The server then immediately syncs the lane stream, turning any queued work into a visible `model_batch_ms` stall.
   - Confidence: high for root-cause class; exact inner op still needs instrumentation.

2. **Cross-lane CUDA stream/device backlog and sync placement.**
   - Primary files: `src/nemotron_speech/server.py:3165-3177`, `src/nemotron_speech/server.py:3204-3211`, `src/nemotron_speech/server.py:3238-3422`.
   - Why it fits: `NEMOTRON_MODEL_LANES=2` runs two lane models on two streams. Each lane then blocks at `stream.synchronize()`. A slow decode or driver scheduling event on the GPU can stall both workers. The overload #2 dump with a lane worker stuck inside `_process_ready_batch` is consistent with a lane stream/device wait, not an event-loop deadlock.
   - Confidence: high as an amplifier and likely necessary condition for the worst tails.

3. **GPU cache stack/scatter copies becoming visible wait points.**
   - Primary files: `src/nemotron_speech/batch_primitives.py:59-97`, `src/nemotron_speech/server.py:8275-8286`, `8333-8400`.
   - Why it fits: `torch.cat` and `detach().clone()` are GPU work. Scatter spikes of 17-45 ms appear near some slow model batches. The SIGUSR1 frame at `scatter_cache_row` can mean the clone is waiting behind earlier model/decode work.
   - Confidence: medium as a secondary source; low as sole explanation for already-logged 600-812 ms `model_batch_ms`.

4. **Per-session warmup if enabled.**
   - Primary files: `src/nemotron_speech/server.py:3620-3667`.
   - Why it fits: the benchmark opens 1000 new sessions. If `NEMOTRON_WARMUP_MS > 0`, every new session runs an extra preprocess and `_conformer_stream_step` with `drop_extra_pre_encoded=0`, which is not the steady graph geometry. That can inject eager model/decode work into the same GPU.
   - Confidence: conditional. It is a strong contributor only if the flag was enabled for the stall run.

5. **Uncaptured encoder batch sizes or geometry.**
   - Primary files: `src/nemotron_speech/server.py:797-810`, `src/nemotron_speech/cudagraph_encoder.py:507-560`.
   - Why it fits: runtime B > captured max returns `None` and falls back eager. This would be very expensive at overload if B exceeds graph coverage.
   - Why it does not fit conc-10 log: observed graph status has `fallbacks=0`, and effective steady batches were mostly B=1-5. It remains a robustness risk for conc-24 or different batching.

6. **Startup graph capture or recapture.**
   - Primary files: `src/nemotron_speech/server.py:1813-1986`, `src/nemotron_speech/cudagraph_encoder.py:391-481`.
   - Why it does not fit: capture is done at startup for the default model and lane models. Replay never lazily captures. Per-run startup capture takes seconds, but it is not inside steady `_process_ready_batch`.

## Exact instrumentation to localize the 1 s stall

Add this behind one flag, e.g. `NEMOTRON_STEADY_BATCH_PROFILE=1`, and only emit full detail for batches where any stage exceeds 100-200 ms or every N batches. Log JSON lines so they can be joined with benchmark records.

### Batch/lane envelope

In `_process_ready_batch` at `src/nemotron_speech/server.py:8216`:

- batch id, wall timestamp, lane id, thread id, CUDA stream id, model id
- session ids, batch size, chunk T, `drop_extra`, `keep_all_outputs`, scheduler generation
- active lane count, ready queue length, per-row queue age if available
- encoder graph replay/fallback counters before and after the batch
- CUDA memory stats before/after, but keep this off unless the batch is already slow

### `_process_ready_batch` stage timers

Use both wall timers and CUDA events on the current lane stream. Wall time tells where the host blocked; event time tells how much GPU work was actually on the stream.

- `entry_sync_ms`: around `server.py:8223`
- `memory_snapshot_before_ms`: around `server.py:8224`
- `prepare_fixed_audio_ms`: `server.py:8226-8232`
- `batched_preprocess_total_ms`: `server.py:8239-8242`
- `post_preprocess_sync_ms`: around `server.py:8243`
- `row_build_ms`: `server.py:8246-8263`
- `stack_processed_ms`: `server.py:8276`
- `stack_caches_ms`: `server.py:8277-8286`
- `clone_previous_hypotheses_ms`: `server.py:8287-8290`
- `clone_previous_pred_out_ms`: `server.py:8291-8294`
- `stack_hypotheses_ms`: `server.py:8295`
- `stack_pred_out_ms`: `server.py:8296`
- `conformer_call_wall_ms`: `server.py:8317-8328`, no sync inside this timer
- `post_model_sync_ms`: around `server.py:8329`
- `model_cuda_event_ms`: event around `server.py:8317-8329`
- `scatter_cache_row_ms_by_row`: `server.py:8337-8342`
- `scatter_pred_out_ms_by_row`: `server.py:8343-8347`
- `scatter_hypotheses_ms_by_row`: `server.py:8348-8352`
- `extract_hypothesis_text_ms_by_row`: `server.py:8354-8355`
- `advance_session_ms_by_row`: `server.py:8389-8395`
- `post_scatter_sync_ms`: around `server.py:8399`
- `memory_snapshot_after_ms` and batch memory log time: `server.py:8401-8409`

Also split `_run_scheduler_model_lane_call_sync`:

- `lane_fn_wall_ms`: around `fn(*args)` at `server.py:3175-3176`
- `lane_outer_sync_ms`: around `server.py:3177`

If `post_model_sync_ms` is the full stall, the call enqueued work quickly and sync waited. If `conformer_call_wall_ms` itself is high before sync, host-side decode control flow, scalar syncs, or Python scheduling are likely involved.

### Encoder vs decoder split

Inside `_conformer_stream_step` at `server.py:2929`:

- Record whether the call selected steady encoder graph, finalize graph, compile bucket, or eager fallback.
- Record bucket B/T/drop and manager label.
- Add counters for replay success/failure reason from `_cudagraph_encoder_cache_step_installed`.

For exact split without changing NeMo permanently, add a profiling-only wrapper around:

- `model.encoder.cache_aware_stream_step`: wall and CUDA-event `encoder_ms`
- `model.decoding.rnnt_decoder_predictions_tensor`: wall and CUDA-event `rnnt_decode_ms`

The wrapper should be installed only during the profiled `_conformer_stream_step` call. It will distinguish:

- encoder graph replay/copy stall
- eager encoder fallback
- RNNT decode wall stall
- post-decode stream sync stall

For graph replay detail, instrument `BucketedCudaGraphEncoder.replay`:

- `copy_processed_ms`
- `copy_len_ms`
- `copy_cache_last_channel_ms`
- `copy_cache_last_time_ms`
- `copy_cache_last_channel_len_ms`
- `graph_replay_enqueue_ms`
- CUDA event span from first copy through replay completion

### Preprocessor split

Inside `_preprocess_scheduler_fixed_audio_batch` at `server.py:8050-8086`:

- `np_stack_ms`: `server.py:8071`
- `h2d_audio_ms`: `torch.from_numpy(audio_batch).cuda()` at `server.py:8075`
- `h2d_len_ms`: `server.py:8076`
- `preprocessor_call_wall_ms` and CUDA event span: `server.py:8077-8080`
- `mel_slice_clone_ms_by_row`: `server.py:8083-8085`

This is probably not the measured `model_batch_ms`, but it can create GPU backlog for another lane.

### Stall-trigger profiler

On the first batch where any segment exceeds 200 ms:

- emit the last 2 s of per-lane stage events from an in-memory ring buffer
- enable one-shot `torch.profiler` for the current or next batch with CPU+CUDA activities
- search profiler output for `aten::_local_scalar_dense`, `aten::item`, `aten::nonzero`, `cudaStreamSynchronize`, `cudaDeviceSynchronize`, and large memcpy/copy ops
- add NVTX ranges around `preprocess`, `encoder_graph_replay`, `rnnt_decode`, `post_model_sync`, `scatter_cache`, and `post_scatter_sync` for Nsight Systems

Run one diagnostic A/B with `CUDA_LAUNCH_BLOCKING=1` only to move the Python stack closer to the actual blocking op. Do not use it for latency measurement.

### Minimal A/B matrix

1. `NEMOTRON_MODEL_LANES=1` vs `2`: if the 700-1178 ms tail disappears or shrinks sharply, cross-lane GPU contention is confirmed.
2. `NEMOTRON_WARMUP_MS=0` vs current: if enabled, isolates per-session warmup load from the new-connection benchmark.
3. Decoder graph allowed vs current `use_cuda_graph_decoder=False`: isolates eager RNNT decode.
4. `NEMOTRON_ENCODER_CUDAGRAPH=0`: if stalls remain, encoder graph replay is exonerated further.
5. Increase/cap graph coverage: `NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=16` and/or cap scheduler batch size to captured max. This is mainly for conc-24 robustness.

## Optimization list

Ranked by expected payoff on the residual TTFS tail.

| Rank | Optimization | Expected payoff | Byte-exact risk | Flag gate |
|---|---|---:|---|---|
| 1 | Enable and validate RNNT decoder CUDA graphs for `greedy_batch` label-loop decode, or add a server flag that passes `use_cuda_graph_decoder=True`. | High. Best candidate to remove the 600-800 ms steady tail if decode is the culprit; also reduces normal decode launch overhead. | Medium to high. Dynamic hypotheses, partial state, alignments/confidence, and CUDA graph fallback behavior must be canaried. | `NEMOTRON_RNNT_DECODER_CUDAGRAPH=1`, fallback to current eager decode on error. |
| 2 | Serialize RNNT decode across lanes while keeping lane infrastructure, or test `NEMOTRON_MODEL_LANES=1`. | High for tail if concurrent decode streams cause device backlog. May trade P50/P95 throughput for P99 stability. | Low for byte exactness if only scheduling changes. Medium product risk from throughput changes. | `NEMOTRON_MODEL_LANES=1`, or `NEMOTRON_RNNT_DECODE_SEMAPHORE=1`. |
| 3 | Make lane scheduling adaptive: run two lanes only when observed model sync stays below a threshold; fall back to one active steady lane under tail pressure. | High for overload #2 and run variance. | Low byte-exact risk; medium scheduling complexity. | `NEMOTRON_ADAPTIVE_MODEL_LANES=1`, threshold envs for sync/model p95. |
| 4 | Remove or reposition redundant steady-path syncs after attribution is known. Keep one correctness sync, but do not sync at entry, after preproc, after model, after scatter, and again at lane-executor exit unless required. | Medium. Reduces forced serialization and makes overlap possible. | Medium. Session caches/hypotheses must not be reused before producer stream completion. | `NEMOTRON_RELAX_STEADY_SYNCS=1`, with byte-exact canary and stream lifetime asserts. |
| 5 | Replace internal sync timing with CUDA events, and make `stream.synchronize()` at lane-executor exit the single completion point. | Medium. Cleaner attribution and fewer waits. | Medium for same stream-lifetime reason. | `NEMOTRON_LANE_SINGLE_SYNC=1`. |
| 6 | Extend steady encoder CUDA graph coverage to the true scheduler max or cap batches to captured max. | Low for the current conc-10 log; high for conc-24 robustness if B>8 occurs. | Low byte-exact risk. Medium memory/startup risk because each B and lane captures static buffers. | `NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=16/32`, `NEMOTRON_BATCH_MAX_SIZE<=graph_max`. |
| 7 | Pre-capture or avoid per-session warmup geometry. If the new-connection benchmark has warmup enabled, set it to zero; otherwise capture the `drop_extra=0` warmup shape or use a pooled startup warm session. | High if `NEMOTRON_WARMUP_MS>0`; none if disabled. | Medium if disabling warmup changes first-token behavior. Low if graphing same warmup computation. | `NEMOTRON_WARMUP_MS=0`, `NEMOTRON_SESSION_WARMUP_MODE=pooled|graph`. |
| 8 | Preallocate per-lane batch buffers for `stack_processed` and `stack_caches`; use `copy_` into stable tensors instead of `torch.cat` allocations each tick. | Medium for scatter/copy variance and allocator pressure; likely not sole 1 s fix. | Low numerically; medium alias/lifetime risk. | `NEMOTRON_BATCH_BUFFER_POOL=1`. |
| 9 | Optimize `scatter_cache_row`: fast path B=1, and for B>1 copy into per-session cache buffers instead of slicing and cloning new tensors. | Low to medium. Should reduce 17-45 ms scatter outliers and overload symptoms. | Medium. Must avoid aliasing graph-owned static outputs that are overwritten on next replay. | `NEMOTRON_CACHE_SCATTER_POOL=1`, with alias asserts. |
| 10 | Keep cache outputs graph-owned only inside the batch and clone/copy once at the ownership boundary. Audit `_conformer_stream_step` clones at `server.py:3011-3014` plus scatter clones at `batch_primitives.py:93-96` to avoid double copies. | Low to medium. Reduces GPU copy work after graph replay. | Medium. Removing the wrong clone can corrupt later sessions because graph static outputs are reused. | `NEMOTRON_CACHE_CLONE_MODE=pooled|legacy`. |
| 11 | Use pinned host buffers and nonblocking H2D for batched preprocessor audio and lengths. | Low for `model_batch_ms`, medium for overall GPU backlog. | Low if shapes/dtypes unchanged. | `NEMOTRON_PINNED_PREPROC_H2D=1`. |
| 12 | Move preprocessor to a separate lower-priority stream or serialize it relative to model streams if H2D/preproc competes with decode. | Low to medium; depends on profiler. | Medium stream dependency risk. | `NEMOTRON_PREPROC_STREAM_MODE=separate|lane|legacy`. |
| 13 | Admission control for overload: cap active sessions/ready backlog per process and return retry/503 or shed before conc-24 creates 18-26 s stalls. | High for catastrophic overload. No P50 win. | Low byte-exact risk; high product behavior impact. | `NEMOTRON_MAX_ACTIVE_SESSIONS`, `NEMOTRON_MAX_READY_BACKLOG`. |
| 14 | Separate processes instead of in-process model lanes, or run one process per GPU partition/stream policy. | Medium if in-process streams contend badly. | Low byte-exact risk; high operational cost. | Deployment-level flag. |
| 15 | Driver/clock controls on cloud L40S: persistence mode, locked application clocks where allowed, and health telemetry for GPU clock/power throttling during stalls. | Unknown. Useful if profiler shows no code-local reason. | None to bytes; operational risk. | Deployment config, not server flag. |

## Immediate next probe order

1. Add the profiler flag and split `_conformer_stream_step` into encoder wrapper time, RNNT decode wrapper time, and post-model sync time.
2. Run the same L40S conc-10 benchmark with `NEMOTRON_MODEL_LANES=1`.
3. Run with `NEMOTRON_WARMUP_MS=0` if any per-session warmup was enabled.
4. Run one decoder-graph canary if NeMo accepts `use_cuda_graph_decoder=True` for this exact `greedy_batch` config.
5. If the stall is not in RNNT decode, use the profiler's CUDA events to decide between graph replay copy, post-model sync backlog, and scatter/cache copies.

