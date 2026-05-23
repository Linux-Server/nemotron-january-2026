# Finalize-batching investigation brief

## The question
At the FINALIZE stage, when multiple finalize requests coincide, they **serialize** instead of batching — and
that serialization is the P50→P95 tail driver. We want to know **why coincident finalizes are not batched
(B=1 despite `NEMOTRON_BATCH_FINALIZE` ON)** and how to fix it. Two ideas from the user to assess:

1. **Batch the work in these kernels when multiple requests are processed by the previous components in a lane** —
   i.e., turn N serial B=1 finalize model-calls into ONE B=N call (+ one gather/clone/scatter). The batching path
   already exists (`_process_final_batch_rows` runs B=len(rows)); the puzzle is why coincident finals don't land in
   one batch.
2. **A custom fused kernel** that takes the individual Python pieces of the last stage (stack/clone/scatter) and does
   them on-device in one kernel. (NOTE: the data below shows these Python pieces are TINY ~0.05-0.1ms — so weigh
   idea 2's value carefully vs idea 1.)

## The measured data (local RTX5090, prod config: lanes=2, BATCH_FINALIZE+_PREPROC, cudagraph, FINALIZE_PROFILE=1)
Records: `proj-2026-05-22-1353/local_probe2_records.txt.{len,burst}` (one JSON per finalize on each
`finalize_profile_record` log line). Driver: `local_finalize_probe2.sh`.

**LENGTH — REJECTED.** conc-1, clips 1-16s (T 44→56): every last-stage step FLAT — clone 0.05→0.05, gather
0.08→0.09, decode 1.44→1.47, model 11.0→11.1 ms. `clone_hypotheses_deep` is O(decoder-state), not O(tokens).

**BATCH TIMING — CONFIRMED (the driver).** In-phase conc-12 (same clip, JITTER=0 -> coincident finals): loadgen
TTFS 14 -> **86/176 ms**. Per-final breakdown:
| field | p50 | p95 | max |
|---|--:|--:|--:|
| **queue_wait_ms** | **39.5** | **102.7** | 116.8 |
| model_wall_ms | 10.5 | 12.9 | 19.6 |
| fork_flush_wall_ms | 13.4 | 26.4 | 36.6 |
| lock_wait_ms | 0.02 | 12.9 | 16.4 |
| clone_hyp_flush_ms | 0.05 | 0.08 | 0.58 |
| final_gather_ms | 0.08 | 0.12 | 0.65 |
| **B distribution** | **{1: 278, 2: 10}** | | |

So the blowup is **`queue_wait`** = finalize events serializing in the scheduler queue. The coincident finals are
processed **one at a time (B=1)**, NOT batched — 12 finals × ~13ms fork_flush ÷ 2 lanes ≈ ~78ms queue drain ≈ the
queue_wait p95. The model/clone/gather are tight; B>1 barely happened (10/288). Corroborated by the cloud K=4 gate
(`finalize-telemetry.md`): queue_wait ~100 + lock_wait ~95 p95 at high load — same serialization.

`queue_wait_ms` = `start_perf` (when `_new_finalize_profile` runs in `_continuous_prepare_finalize_item_locked`)
minus `debounce_event_queued_perf` (when the finalize/debounce event was queued). So it is the wait IN the scheduler
queue before the finalize even begins preparing.

## Where to look (server.py — grep these; line numbers approximate, my edits shifted them)
- Scheduler event loop + finalize-event handling: `reason = "reset_then_debounce" if reset_seen ...` (~4125);
  msg dispatch (~3640/3715/3786); `_scheduler_queue_event`.
- `_continuous_prepare_finalize_item_locked` (~6048) — builds one `SchedulerFinalizeItem` per event; sets queue_wait.
- `_continuous_flush_finalize_items_locked` (~6175) — collects `flush_items`, groups `by_lane`, runs
  `_process_final_fork_groups` per lane. **B = len(rows) in a lane group.** This is where batching does/doesn't happen.
- `_process_final_fork_groups` (~6720) -> `_process_final_batch_rows` (~6787) — the B-row gather + one
  `_conformer_stream_step(B)` + scatter.
- `_scheduler_pinned_model_lane_path` (~6121) — per-lane finalize reservation.
- Flags: `NEMOTRON_BATCH_FINALIZE`, `NEMOTRON_BATCH_FINALIZE_PREPROC`, and any barrier/drain
  (`NEMOTRON_BATCH_BARRIER_DRAIN`?) that would COLLECT coincident finalize events before flushing.

## The crux for idea 1
Why do coincident finalize events become separate B=1 flushes instead of being collected into one B>1
`_continuous_flush_finalize_items_locked` call? Is each finalize event drained in its own scheduler iteration
(so `flush_items` has 1 item each time)? Is there a barrier/drain that's off or ineffective? What would make
coincident finals batch (one B=N call ~15-20ms vs N×13ms serial through 2 lanes)?
