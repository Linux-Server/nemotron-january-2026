# Step 9 Local Validation

Date: 2026-05-21 local. Server: English `.nemo` from `/tmp/en-nemo-path`, rc1, port 8080.
Common env: `NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 NEMOTRON_FORK_ASSERT=1`.

## 1. High-N Harness Extension

Committed reusable harness extension: `7864feb Extend concurrency harness for high-N validation`.

- `proj-2026-05-19-eou-endpointing/concurrency_test.py` now selects `max(24, max(sweep), --sample-count)` distinct DB samples. The default N<=24 sweep still selects the historical 24 samples.
- Added `--strict-byte`: compares final transcript, interim sequence, final delta list, and duplicate-final condition against each sample's same-server solo baseline.
- Added strict baseline latency capture, per-level wall-clock timestamps for telemetry alignment, and `--run-all-concurrency` for bounded high-N canaries over larger sample sets.
- Verification: `python3 -m py_compile proj-2026-05-19-eou-endpointing/concurrency_test.py`.

## 2. TF32-Off Re-Baseline and Matmul Cost

Captured config-matched B=1 production baseline:
`proj-2026-05-21-0410/baseline/english_baseline_fp32.json`.

Startup verification:
`scheduler_enabled=True batch_requested=True batch_enabled=True decoder_strategy=greedy_batch encoder_compile_enabled=False`, plus `cuda.matmul.allow_tf32=False cudnn.allow_tf32=False`.

Comparison vs committed `baseline/english_baseline.json`:

| Check | Exact |
|---|---:|
| Final text | 8/8 |
| Final delta | 8/8 |
| Interim sequence | 8/8 |

Diffs: none.

Matmul/profile cost, same 16 s clip, same warmup, B=1 scheduler path with explicit PyTorch TF32 flags:

| Mode | `step(enc+dec)` | Preprocess | Total |
|---|---:|---:|---:|
| TF32 ON | 10.24 ms/chunk | 1.05 ms/chunk | 11.29 ms/chunk |
| TF32 OFF | 10.42 ms/chunk | 1.06 ms/chunk | 11.49 ms/chunk |

TF32-off slowdown: `+1.8%` on `step(enc+dec)`.

## 3. Compile x Batch Matrix

Keep-up rule used here: no errors/timeouts, strict byte-exact, TTFS p95 <500 ms, and processing-lag p95 <500 ms.
Each matrix cell used 60 distinct test clips and swept `N={16,24,40,48,60}`.

Startup checks:

| Config | Flags | Startup |
|---|---|---|
| A: B=1 scheduler | `NEMOTRON_SCHEDULER_B1=1` | scheduler on, batch off, `greedy`, compile off |
| B: batch-only | `NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1` | scheduler on, batch on, `greedy_batch`, TF32 off, compile off |
| C: compile-only | `NEMOTRON_SCHEDULER_B1=1 NEMOTRON_ENCODER_COMPILE=1` | scheduler on, batch off, `greedy`, compile on, 3 buckets warmed, recapture 0 |
| D: compile+batch | `NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_ENCODER_COMPILE=1` | scheduler on, batch on, `greedy_batch`, TF32 off, compile on, 3 buckets warmed, recapture 0 |

N=1 solo-baseline latency, ms, shown as p50/p95/p99:

| Config | First interim | Interim lag | Final TTFS |
|---|---:|---:|---:|
| A | 1453.1 / 2892.0 / 3213.2 | 11.9 / 12.5 / 13.7 | 13.8 / 15.0 / 16.1 |
| B | 1454.8 / 2894.3 / 3214.7 | 13.9 / 14.7 / 15.6 | 14.3 / 16.2 / 16.8 |
| C | 1449.5 / 2889.0 / 3209.5 | 8.7 / 9.1 / 9.4 | 14.1 / 15.5 / 16.1 |
| D | 1452.9 / 2892.3 / 3212.9 | 11.8 / 12.5 / 12.9 | 14.4 / 15.8 / 17.6 |

Latency gate: B/C/D p95 values are within A + 5 ms + 10 ms; final p95 is far below 400 ms.

Knee table:

| Config | Strict exact | Keep-up knee | N=40 TTFS95 / lag95 | N=48 TTFS95 / lag95 | Notes |
|---|---:|---:|---:|---:|---|
| A: B=1 scheduler | 188/188 | 16 | 7976 / 7996 | 13872 / 13892 | N=24 already falls behind. |
| B: batch-only | 188/188 | 40 | 404 / 424 | 1318 / 1339 | Best default: 2.5x local knee vs A. |
| C: compile-only | 188/188 | 24 | 2676 / 2696 | 5636 / 5656 | Helps B=1, but not enough for N=40. |
| D: compile+batch | 188/188 | 40 | 225 / 246 | 800 / 821 | Same knee as B; better overload, but long compile warmup. |

Batch histogram deltas by level:

| N | B: batch-only | D: compile+batch |
|---:|---|---|
| 16 | `{1:202,2:54,3:18,4:1}` | `{1:250,2:39,3:11}` |
| 24 | `{1:224,2:145,3:63,4:18}` | `{1:301,2:124,3:65,4:10}` |
| 40 | `{1:163,2:73,3:68,4:346}` | `{1:270,2:117,3:54,4:309}` |
| 48 | `{1:51,2:19,3:14,4:572}` | `{1:68,2:26,3:16,4:565}` |
| 60 | `{1:49,2:10,3:3,4:707}` | `{1:20,2:21,3:3,4:710}` |

Compile+batch coexistence: confirmed. D logged B>1 batches while compile telemetry ended at
`compiled_calls=6000 recapture_counter=0`. The compile bucket predicate only returns a bucket for
`processed_signal.shape[0] == 1`, so B>1 batches ran uncompiled while solo/fallback B=1 calls used compile.

Winner: B is the production default winner. D has better above-knee latency but does not move the strict knee
past 40 and paid a roughly 124 s compile warmup in this run.

## 4. 200-Sample Byte-Exact Canary

Full 1000 was not run locally because the required same-config solo re-baseline is realtime and would be a
multi-hour local pass. I ran the allowed 200-sample canary.

Config: batch-only production mode (`NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1`, `greedy_batch`, TF32 off).
Harness: `--db results.db --sample-count 200 --strict-byte --skip-sweep --run-all-concurrency 48`.

Result:

| Check | Result |
|---|---:|
| Strict byte-exact | 200/200 |
| Final text exact | 200/200 |
| Errors | 0 |
| Max edit distance | 0 |

Canary batch histogram delta: `{1:115,2:100,3:3,4:2187}`. This formed real batches heavily; B=4 dominated.
The canary was a correctness gate, not a keep-up run: sustained C=48 over 200 samples had TTFS p95 13388 ms
and lag p95 13407 ms.

## 5. Recommendation and Local Knee Story

Recommended production defaults for Step 10 Modal re-sweep:

- Decoder: `greedy_batch`, `loop_labels=True`, `use_cuda_graph_decoder=False`.
- Precision: TF32 off under batching (`cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`).
- Scheduler/batch: enable `NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1`.
- Batch policy: keep current `NEMOTRON_BATCH_MAX_WAIT_MS=5`, `NEMOTRON_BATCH_MAX_SIZE=4`.
- Compile: keep `NEMOTRON_ENCODER_COMPILE=0` by default for the Modal sweep. Treat compile as an optional
  follow-up cell: it helps B=1 and overload latency, but did not improve the strict batch knee locally and
  had a large compile+batch startup cost.

Local story to carry into Modal: the B=1 scheduler knee is 16; batch-only moves it to 40, a 2.5x local knee
increase with strict byte-exactness. Compile-only moves 16 to 24. Compile+batch ties batch-only at 40.

## 6. Cleanup

Servers were killed between runs by freeing port 8080. Final check: 8080 is free, no active GPU compute
process is present, temp logs under `/tmp/nemotron-step9` were removed, and the harness `__pycache__` from
`py_compile` was removed. Port 8088 was not touched and remains owned by the pre-existing listener.
