# Max Parallelism Sweep

Date: 2026-05-21 local. Host GPU: NVIDIA GeForce RTX 5090. Model:
`/home/khkramer/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/ef3bf40c90df5cd2de55cc07e06681e03d8e6ee4/nemotron-speech-streaming-en-0.6b.nemo`,
English `.nemo`, `--right-context 1`, `NEMOTRON_WARMUP_MS=200`,
`greedy_batch`, TF32 off.

No production defaults were changed. All MAX_SIZE results below were forced by per-run environment variables.
Startup logs for MAX_SIZE 4/8/16/32 confirmed:
`scheduler_enabled=True`, `batch_enabled=True`, `decoder_strategy=greedy_batch`,
`encoder_compile_enabled=False`, TF32 disabled, and `device_cap=46`.

Keep-up gate used for Part B: no final timeouts, `processing_lag_p95 < 500ms`, and final/TTFS p95
`< 400ms`. Client-bound gate: send-overrun p95/max and pacing violations.

## Part A - Forced-Batch Microbench

This drives the live server `_process_ready_batch` path in-process at a forced batch size:
batched fixed preprocessor `[B,K]`, one `conformer_stream_step(B)`, scatter/postprocess, and session advance.
Each requested B used 220 measured ticks after 30 discarded ticks. The B=40/46 rows are an extension because
B=32 was still climbing and Step 8 reported `device_cap=46`.

| B | T mean ms | T p50 ms | T p95 ms | realtime-stream-equivalents |
|---:|---:|---:|---:|---:|
| 1 | 10.90 | 10.73 | 11.56 | 14.7 |
| 2 | 12.08 | 11.87 | 13.05 | 26.5 |
| 4 | 13.59 | 13.59 | 14.23 | 47.1 |
| 8 | 19.44 | 19.43 | 20.22 | 65.9 |
| 16 | 22.11 | 22.05 | 23.22 | 115.8 |
| 32 | 27.80 | 27.77 | 29.26 | 184.2 |
| 40 | 31.75 | 31.73 | 33.11 | 201.6 |
| 46 | 33.43 | 33.41 | 35.04 | 220.2 |

Findings:

- B=1 sanity: `160 / 10.90 = 14.7` stream-equivalents, close to the measured B=1 scheduler knee near 16.
- The requested sweep max is B=32 at 184 realtime-stream-equivalents.
- The memory-cap extension max is B=46 at 220 realtime-stream-equivalents.
- There is no compute-saturation/linear-growth knee through B=46. T(B) is still sublinear; the practical
  tested ceiling is the memory cap, not a GPU compute saturation point.

## Part B - Realtime MAX_SIZE x N Sweep

Server: port 8080, `NEMOTRON_BATCH_MAX_WAIT_MS=8`, `NEMOTRON_SCHEDULER_B1=1`,
`NEMOTRON_BATCH_SCHED=1`, TF32 off. Client drove distinct realtime streams from `results.db`.
N=1 latency used four sequential single-stream clips per config.

### Coarse Sweep

`batch total avg ms` is `preprocessor_batch_ms + model_batch_ms + scatter_postprocess_ms` from
`scheduler_batch_memory`, averaged per server batch.

| MAX_SIZE | N | keep-up | proc-lag p95 ms | TTFS p95 ms | client overrun p95/max ms | violations | avg B | batch total avg ms | batch hist |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 1 | yes | 38.1 | 17.9 | 1.38 / 1.84 | 0 | 1.00 | 10.95 | `{1:231}` |
| 4 | 16 | yes | 53.1 | 33.2 | 1.09 / 2.32 | 0 | 1.17 | 10.52 | `{1:705,2:96,3:21}` |
| 4 | 40 | yes | 198.7 | 178.7 | 1.08 / 2.33 | 0 | 2.72 | 12.12 | `{1:213,2:188,3:113,4:370}` |
| 4 | 60 | no | 7437.8 | 7417.7 | 1.11 / 3.13 | 0 | 3.81 | 13.25 | `{1:34,2:22,3:3,4:715}` |
| 4 | 100 | no | 32893.8 | 32873.7 | 1.12 / 2.22 | 0 | 3.97 | 13.49 | `{1:4,2:5,3:2,4:802}` |
| 4 | 150 | no | 69115.1 | 69095.0 | 1.21 / 2.42 | 0 | 3.97 | 13.85 | `{1:4,2:5,3:1,4:803}` |
| 8 | 1 | yes | 36.4 | 17.3 | 1.34 / 2.01 | 0 | 1.00 | 11.01 | `{1:231}` |
| 8 | 16 | yes | 68.0 | 47.9 | 1.10 / 2.22 | 0 | 1.19 | 10.70 | `{1:674,2:119,3:16}` |
| 8 | 40 | yes | 156.5 | 136.5 | 1.10 / 2.63 | 0 | 2.80 | 12.45 | `{1:236,2:190,3:190,4:112,5:71,6:36,7:11,8:17}` |
| 8 | 60 | no | 1744.4 | 1725.1 | 1.11 / 2.32 | 0 | 5.09 | 15.35 | `{1:169,2:62,3:44,4:26,5:41,6:22,7:24,8:324}` |
| 8 | 100 | no | 12636.5 | 12616.5 | 1.14 / 2.78 | 0 | 7.50 | 18.36 | `{1:30,2:11,3:6,4:10,5:3,6:2,7:3,8:664}` |
| 8 | 150 | no | 43627.5 | 43607.7 | 1.31 / 3.79 | 0 | 7.80 | 19.29 | `{1:12,2:5,3:3,4:5,5:3,8:762}` |
| 16 | 1 | yes | 37.1 | 17.0 | 1.37 / 1.91 | 0 | 1.00 | 10.96 | `{1:231}` |
| 16 | 16 | yes | 56.6 | 37.0 | 1.09 / 2.80 | 0 | 1.17 | 10.52 | `{1:712,2:85,3:26}` |
| 16 | 40 | yes | 130.7 | 110.2 | 1.08 / 4.09 | 0 | 2.72 | 12.18 | `{1:228,2:228,3:212,4:99,5:65,6:37,7:11,8:4,9:4,10:1}` |
| 16 | 60 | no | 1088.0 | 1068.6 | 1.08 / 1.87 | 0 | 4.78 | 14.21 | `{1:197,2:149,3:126,4:41,5:31,6:15,7:17,8:41,9:4,10:14,11:7,12:28,13:40,14:3,15:1,16:46}` |
| 16 | 100 | no | 3654.1 | 3634.7 | 1.09 / 2.41 | 0 | 9.18 | 16.67 | `{1:114,2:115,3:35,4:14,5:5,6:7,7:8,8:9,9:8,10:10,11:5,12:14,13:9,14:22,15:22,16:263}` |
| 16 | 150 | no | 11403.3 | 11383.2 | 1.21 / 2.86 | 0 | 12.27 | 18.49 | `{1:130,2:16,3:7,4:10,5:7,6:5,7:3,8:5,9:4,10:4,11:7,12:5,13:8,14:3,15:9,16:507}` |
| 32 | 1 | yes | 37.4 | 17.3 | 1.38 / 1.84 | 0 | 1.00 | 10.96 | `{1:231}` |
| 32 | 16 | yes | 47.7 | 28.0 | 1.10 / 3.76 | 0 | 1.19 | 10.76 | `{1:661,2:136,3:9}` |
| 32 | 40 | yes | 163.4 | 143.3 | 1.09 / 3.61 | 0 | 2.83 | 12.58 | `{1:214,2:191,3:186,4:126,5:86,6:34,7:8,8:3,9:4,10:1}` |
| 32 | 60 | no | 1266.3 | 1245.9 | 1.11 / 6.13 | 0 | 5.10 | 14.25 | `{1:281,2:141,3:51,4:12,5:31,6:15,7:7,8:28,9:3,10:5,11:1,12:14,13:6,14:37,15:11,16:15,17:8,18:3,19:22,20:1,22:3,23:10,24:7}` |
| 32 | 100 | no | 3154.6 | 3134.8 | 1.12 / 2.06 | 0 | 9.47 | 16.33 | `{1:166,2:161,3:34,4:10,5:5,6:7,7:6,8:14,9:16,10:8,11:6,12:10,13:17,14:15,15:10,16:11,17:7,18:6,19:1,20:21,21:2,22:2,23:1,24:1,25:2,26:2,27:3,28:1,29:1,30:2,31:1,32:91}` |
| 32 | 150 | no | 8809.6 | 8789.5 | 1.16 / 2.60 | 0 | 12.45 | 16.99 | `{1:277,2:101,3:22,4:9,5:5,6:4,7:5,8:5,9:4,10:3,11:5,12:5,13:4,14:4,15:13,16:9,17:1,18:4,19:1,20:5,21:1,22:2,23:5,24:4,25:2,26:8,27:6,28:3,29:7,30:10,31:8,32:188}` |

### Knee Refinement

`batch total avg ms` has the same preprocessor + model + scatter definition as above.

| MAX_SIZE | N | keep-up | proc-lag p95 ms | TTFS p95 ms | client overrun p95/max ms | violations | avg B | batch total avg ms | batch hist |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 48 | no | 961.9 | 942.1 | 1.10 / 2.05 | 0 | 3.57 | 12.99 | `{1:96,2:17,3:10,4:646}` |
| 4 | 52 | no | 2726.0 | 2706.6 | 1.08 / 2.02 | 0 | 3.66 | 12.93 | `{1:66,2:34,3:1,4:675}` |
| 4 | 56 | no | 4571.1 | 4550.8 | 1.08 / 2.16 | 0 | 3.74 | 13.02 | `{1:55,2:13,3:10,4:700}` |
| 8 | 48 | yes | 366.6 | 346.4 | 1.07 / 2.18 | 0 | 3.39 | 12.89 | `{1:190,2:140,3:218,4:115,5:61,6:18,7:4,8:110}` |
| 8 | 52 | no | 652.8 | 632.4 | 1.07 / 2.25 | 0 | 4.03 | 14.23 | `{1:187,2:131,3:112,4:75,5:28,6:12,7:41,8:195}` |
| 8 | 56 | no | 837.6 | 817.5 | 1.08 / 1.63 | 0 | 4.51 | 14.45 | `{1:170,2:167,3:43,4:29,5:3,6:28,7:31,8:280}` |
| 16 | 48 | yes | 239.9 | 219.9 | 1.08 / 2.11 | 0 | 3.36 | 12.70 | `{1:185,2:157,3:183,4:166,5:66,6:54,7:19,8:7,9:4,11:2,12:2,13:7,14:7,15:1,16:4}` |
| 16 | 52 | yes | 405.3 | 385.1 | 1.08 / 2.17 | 0 | 3.83 | 13.43 | `{1:172,2:128,3:176,4:159,5:58,6:19,7:7,8:4,9:25,10:27,11:14,12:15,13:8,14:5,16:4}` |
| 16 | 56 | no | 442.4 | 422.5 | 1.08 / 1.90 | 0 | 4.22 | 13.37 | `{1:159,2:140,3:138,4:169,5:57,6:8,7:5,8:4,9:30,10:15,11:12,12:22,13:12,14:11,15:1,16:20}` |
| 32 | 48 | yes | 223.2 | 203.9 | 1.07 / 2.09 | 0 | 3.35 | 12.69 | `{1:189,2:158,3:170,4:177,5:89,6:29,7:6,8:21,9:3,10:2,11:1,12:2,13:8,14:6,15:2,17:1,18:2}` |
| 32 | 52 | yes | 377.2 | 357.1 | 1.07 / 2.17 | 0 | 3.82 | 13.10 | `{1:167,2:141,3:159,4:120,5:95,6:34,7:8,8:26,9:25,10:17,11:8,12:9,13:7,14:3,15:1,17:1,19:1,20:1}` |
| 32 | 56 | yes | 379.6 | 360.3 | 1.08 / 2.57 | 0 | 4.16 | 13.21 | `{1:166,2:112,3:119,4:167,5:98,6:59,7:11,8:9,9:9,10:8,11:11,12:14,13:10,14:6,15:3,17:1,19:1,21:5,22:6}` |
| 32 | 58 | no | 581.5 | 561.2 | 1.08 / 3.40 | 0 | 4.44 | 13.43 | `{1:144,2:169,3:133,4:142,5:66,6:32,7:8,8:8,9:11,10:2,11:1,12:4,14:14,15:7,16:9,17:10,18:3,19:11,20:7,21:6,22:1,23:1,25:1,26:1}` |

Knees:

| MAX_SIZE | strict realtime knee | first failing N | N=1 TTFS p95 | client-bound? |
|---:|---:|---:|---:|:---:|
| 4 | 40 | 48 | 17.9ms | no |
| 8 | 48 | 52 | 17.3ms | no |
| 16 | 52 | 56 | 17.0ms | no |
| 32 | 56 | 58 | 17.3ms | no |

Client-bound check: no run was client-bound. Across all coarse and refined runs, send-overrun p95 stayed
about 1.07-1.38ms, worst max was 6.13ms, and pacing violations were 0.

## Gap: Achievable vs Forced-Batch Ceiling

Best achieved strict realtime knee: 56 streams with `MAX_SIZE=32`, `MAX_WAIT=8ms`.

Forced-batch ceiling:

- Requested Part A ceiling, B=32: 184 realtime-stream-equivalents.
- Memory-cap extension, B=46: 220 realtime-stream-equivalents.

Dispatch efficiency:

- `56 / 184 = 30%` versus the requested B=32 forced-batch ceiling.
- `56 / 220 = 25%` versus the memory-cap B=46 extension.

The gap is not the Python client. The dominant observed reason is that independent realtime streams do not
arrive in full batches. At the strict knee for MAX_SIZE=32, effective average B is only 4.16 at N=56, even
though the cap is 32. Larger batches appear mainly under overload, where they reduce lag but do not restore
the final-latency budget once N crosses the knee.

## Recommendation

Recommend for this local box, subject to orchestrator review:

- `NEMOTRON_BATCH_MAX_SIZE=32`
- `NEMOTRON_BATCH_MAX_WAIT_MS=8`

Rationale:

- It raises the strict realtime knee from 40 streams at the current conservative MAX_SIZE=4 to 56 streams.
- N=1 latency remains effectively unchanged and far under budget: TTFS p95 about 17ms.
- The client was not the limiter, so the N=56 knee is a server/dispatch result.
- MAX_SIZE=32 is below the Step-8 startup `device_cap=46` and below the forced-batch memory-cap extension.
- Do not raise to the memory cap yet: Part B shows current dispatch rarely fills such large batches near
  the realtime knee, and MAX_SIZE=32 already starts to show higher p95 model batch times without making N=58
  keep up.

Modal Step 10 should push beyond the old N<=80 range. Suggested levels: keep the near-knee refinement
`48/52/56/58/60`, then add overload/client-bound points `100/150/200`. If send-overrun stays below 5ms at
N=150, push to at least N=200 so the cloud run can compare against the B=32 forced-batch ceiling rather than
only the old conservative cap.
