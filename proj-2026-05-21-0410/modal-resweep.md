# Modal Re-Sweep: Batched Cheap Subset

Date: 2026-05-21. Scope: T4 and L4 only. ASR apps and loadgen were co-located in
`us-east-1`; each ASR app was stopped before moving to the next GPU.

## Config

Deploy wrapper: `src/nemotron_speech/modal/asr_bench_modal.py`.

Server env:

- `NEMOTRON_CONTINUOUS=1`
- `NEMOTRON_SCHEDULER_B1=1`
- `NEMOTRON_BATCH_SCHED=1`
- `NEMOTRON_FINALIZE_SILENCE_MS=0`
- `NEMOTRON_WARMUP_MS=200`
- `NEMOTRON_BATCH_MAX_SIZE=32`
- `NEMOTRON_BATCH_MAX_WAIT_MS=8`

Startup logs confirmed `scheduler_enabled=True`, `batch_enabled=True`,
`decoder_strategy=greedy_batch`, `encoder_compile_enabled=False`, and TF32 off.

| GPU | startup memory cap |
|-----|--------------------|
| T4 | requested 32, effective 19, device_cap 19 |
| L4 | requested 32, effective 31, device_cap 31 |

## Smoke

Cheap 8-clip smoke ran before each sweep. Both GPUs returned sane English transcripts
with no looping and no language-tag leakage. The optional strict-byte N=8 smoke was
not exact while already overloaded: T4 3/8 exact, L4 4/8 exact. I treated this as
sanity-only per the scoped smoke requirement, not a cloud strict-byte signoff.

## Knee Results

Keep-up gate: no errors/timeouts and processing-lag p95 `<500ms`.

| GPU | N | keep-up | TTFS p95 ms | proc-lag p95 ms | avg B | batch histogram | send-overrun p95 ms |
|-----|---:|:-------:|------------:|----------------:|------:|-----------------|--------------------:|
| T4 | 4 | yes | 360 | 379 | 1.01 | `{1:124,2:1}` | 16 |
| T4 | 5 | no | 2925 | 2946 | 1.00 | `{1:225}` | 15 |
| T4 | 8 | no | 3894 | 3912 | 1.68 | `{1:141,2:115,3:44}` | 15 |
| L4 | 4 | yes | 356 | 375 | 1.00 | `{1:150}` | 15 |
| L4 | 5 | no | 856 | 877 | 1.03 | `{1:194,2:6}` | 15 |
| L4 | 6 | yes | 461 | 480 | 1.33 | `{1:175,2:25,3:25}` | 14 |
| L4 | 7 | no | 1238 | 1256 | 1.49 | `{1:178,2:29,3:36,4:7}` | 14 |
| L4 | 8 | no | 2996 | 3015 | 1.41 | `{1:207,2:143}` | 16 |

Client-bound flag: no. Send-overrun stayed around 14-16ms, matching the earlier
co-located finding that the client could pace realtime.

## Cost

| GPU | batch=1 baseline | batched knee | improvement | batched $/stream-hr |
|-----|------------------|--------------|-------------|---------------------|
| T4 | knee ~5, ~$0.12/stream-hr | 4 | 0.8x | $0.15 |
| L4 | knee ~5, ~$0.16/stream-hr | 6 | 1.2x | $0.13 |

## Headline

Batching does **not** close the gap to the local MAX_SIZE=32 result on the cheap
Modal subset. Batches form under overload, but near the realtime knee the effective
batch size is still close to 1. T4 is worse than the batch=1 baseline; L4 is only
modestly better. The cloud $/stream improves only on L4 and by much less than the
local ~3.5x knee gain.

## Cleanup

Stopped apps:

- `nemotron-asr-bench-t4`
- `nemotron-asr-bench-l4`

Final `modal app list` checks showed no running `nemotron-asr-bench-t4` or
`nemotron-asr-bench-l4` entries.
