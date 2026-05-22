# In-Phase Realtime Confirmation

Date: 2026-05-21 local. Host GPU: NVIDIA GeForce RTX 5090. Server:
`/home/khkramer/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/ef3bf40c90df5cd2de55cc07e06681e03d8e6ee4/nemotron-speech-streaming-en-0.6b.nemo`,
English `.nemo`, `--right-context 1`, port 8080.

Server env:

```text
NEMOTRON_CONTINUOUS=1
NEMOTRON_SCHEDULER_B1=1
NEMOTRON_BATCH_SCHED=1
NEMOTRON_BATCH_MAX_SIZE=32
NEMOTRON_BATCH_MAX_WAIT_MS=8
NEMOTRON_BATCH_MEMORY_TELEMETRY_EVERY=1
NEMOTRON_WARMUP_MS=200
NEMOTRON_FINALIZE_SILENCE_MS=0
```

Startup confirmed `scheduler_enabled=True`, `batch_enabled=True`, `decoder_strategy=greedy_batch`,
TF32 disabled, `requested_max=32 effective_max=32 device_cap=46`.

Driver: `proj-2026-05-21-0410/inphase_loadgen.py`. It connects all sessions first, waits for every `ready`,
sends `vad_start` as a burst, then sends 160 ms PCM chunks from one shared monotonic clock. The local DB only
has 58 samples at or above 10 seconds, so N above 58 repeats that long-duration pool. This keeps the full-N
phase-aligned portion long enough to test batching; streams still stop at their own aligned end boundary.

Keep-up gate: no final timeouts, `processing_lag_p95 < 500ms`, and `TTFS p95 < 400ms`.

## Requested Sweep

`avg B ready-pass` is from `scheduler_batch_memory` rows, i.e. the normal batched ready-pass path. `avg B incl
barrier` additionally counts `scheduler barrier drained X ready chunks before vad_stop` as B=1 model calls,
which matters in failed runs.

| N | keep-up | ok/final timeouts | TTFS p95 ms | proc-lag p95 ms | avg B ready-pass | full B32 batches | B=1 barrier chunks | avg B incl barrier | send-overrun p95/max ms | chunk-late p95/max ms |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | yes | 60/0 | 116.9 | 263.3 | 19.07 | 78/253 | 0 | 19.07 | 0.7 / 1.1 | 22.0 / 28.4 |
| 100 | yes | 100/0 | 189.0 | 330.5 | 20.29 | 223/388 | 1 | 20.24 | 0.8 / 1.5 | 34.4 / 36.2 |
| 150 | no | 150/0 | 21481.4 | 21640.4 | 31.35 | 294/308 | 2236 | 4.67 | 1.1 / 1.3 | 52.8 / 55.8 |
| 180 | no | 141/39 | 56520.2 | 56666.4 | 31.99 | 222/223 | 5822 | 2.14 | 1.1 / 1.1 | 59.7 / 66.5 |
| 200 | no | 59/141 | 61495.4 | 61612.5 | 31.65 | 75/78 | 6207 | 1.38 | 0.7 / 1.6 | 66.5 / 68.5 |

Batch histograms, ready-pass only:

```text
N=60  {1:73, 2:1, 4:6, 5:2, 6:1, 7:1, 8:1, 9:1, 10:2, 12:2, 13:1, 14:4, 15:2, 16:1, 18:1, 19:2, 20:2, 22:1, 23:1, 24:2, 25:1, 27:64, 28:1, 30:1, 31:1, 32:78}
N=100 {1:69, 2:1, 3:60, 4:5, 5:2, 6:1, 7:1, 8:1, 9:1, 10:1, 12:4, 13:4, 14:2, 15:2, 16:1, 19:1, 20:2, 23:1, 24:2, 26:1, 28:2, 30:1, 32:223}
N=150 {8:1, 10:4, 14:2, 18:1, 20:1, 24:1, 26:2, 28:1, 30:1, 32:294}
N=180 {30:1, 32:222}
N=200 {21:2, 27:1, 32:75}
```

## Knee Refinement

Because the requested sweep had a large gap between N=100 and N=150, I added refinement points.

| N | keep-up | TTFS p95 ms | proc-lag p95 ms | avg B ready-pass | full B32 batches | B=1 barrier chunks | avg B incl barrier | send-overrun p95/max ms |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 105 | yes | 205.4 | 343.3 | 21.44 | 227/388 | 1 | 21.39 | 1.1 / 1.6 |
| 110 | yes | 208.7 | 343.9 | 24.21 | 241/363 | 1 | 24.15 | 1.3 / 1.6 |
| 115 | yes | 194.4 | 330.7 | 25.17 | 250/369 | 1 | 25.11 | 1.4 / 1.4 |
| 120 | no | 2566.9 | 2727.1 | 24.53 | 231/375 | 452 | 11.67 | 0.8 / 0.9 |
| 130 | no | 12625.9 | 12786.5 | 31.30 | 273/287 | 1361 | 6.28 | 0.5 / 0.6 |
| 140 | no | 29378.9 | 29537.7 | 31.24 | 248/262 | 2910 | 3.50 | 0.4 / 1.1 |

Strict in-phase knee: N=115. First observed failing point: N=120.

## Contrast

| Case | strict realtime knee | avg B at/near knee | Notes |
|---|---:|---:|---|
| Out-of-phase MAX_SIZE=32 | 56 | 4.16 | From `max-parallelism-sweep.md`; independent 160 ms cycles are scattered. |
| In-phase MAX_SIZE=32 | 115 | 25.17 ready-pass | Full B32 batches dominate; one B=1 barrier chunk at N=115. |
| Forced-batch B=32 ceiling | 184 stream-equivalents | 32 forced | From `max-parallelism-sweep.md` Part A. |
| Forced-batch B=46 memory-cap ceiling | 220 stream-equivalents | 46 forced | Part A extension. |

## Verdict

In-phase alignment does fill the scheduler: at N=150 and above the normal ready-pass histograms are essentially
full batches (`avg B` about 31-32, with B32 dominating). That confirms the out-of-phase N=56 result was paying
a major phasing penalty: aligned streams raise the strict knee from 56 to 115, about 2.1x.

It does not reach the predicted 150-180 knee. The failure mode is not the client and not lack of full ready-pass
batches. At N=120+, once the server has any backlog at `vad_stop`, `_scheduler_drain_ready_barrier_locked`
drains pending chunks for that session through the B=1 barrier path before finalization. That collapses the
effective model-call size in the failed runs: N=150 has `avg B ready-pass=31.35`, but 2236 extra B=1 barrier
chunks bring the effective call average to 4.67 and TTFS p95 to 21.5s.

So the measurement partially confirms the hypothesis:

- Confirmed: phase alignment fixes the normal ready-pass batching problem; full batches form.
- Confirmed: the GPU is not the limiting factor in the clean passing region; the client also is not the limiter.
- Not confirmed: the practical realtime knee did not move to the B=32 forced-batch ceiling near 180.
- Refined conclusion: the queue is still the single serial model-call lane, but the current production scheduler
  has an additional nonbatched `vad_stop` barrier drain that becomes the practical high-N limiter before the
  forced-batch ceiling.

Client-bound caveat: no run was client-bound. Send-overrun p95 stayed <= 1.4 ms and max <= 1.6 ms. Per-chunk
burst lateness rose with N but stayed below one 160 ms interval (`chunk_late_max_ms` <= 68.5 ms), with zero
pacing violations.

Artifacts:

- Client results: `proj-2026-05-21-0410/inphase-artifacts/inphase-results.json`
- Refinement results: `proj-2026-05-21-0410/inphase-artifacts/inphase-refine-results.json`,
  `proj-2026-05-21-0410/inphase-artifacts/inphase-refine2-results.json`
- Parsed summary: `proj-2026-05-21-0410/inphase-artifacts/combined-summary.json`
- Server log: `proj-2026-05-21-0410/inphase-artifacts/server-clean.log`
