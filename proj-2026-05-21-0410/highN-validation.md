# High-N batching validation

Date: 2026-05-21 local. Server: English checkpoint from `/tmp/en-nemo-path`, `--right-context 1`, port 8080.

Configs:

- A: B=1 scheduler, no batching: `NEMOTRON_SCHEDULER_B1=1`
- B: batching default: `NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1`
  (`greedy_batch`, TF32 off, `MAX_WAIT=5ms`, `MAX_SIZE=4`)
- C: batching longer wait: B plus `NEMOTRON_BATCH_MAX_WAIT_MS=20 NEMOTRON_BATCH_MAX_SIZE=8`

Startup verification:

- A: `scheduler_enabled=True batch_enabled=False decoder_strategy=greedy encoder_compile_enabled=False`
- B: `scheduler_enabled=True batch_enabled=True decoder_strategy=greedy_batch encoder_compile_enabled=False`;
  `cuda.matmul.allow_tf32=False cudnn.allow_tf32=False`; `batch_max_wait_ms=5 batch_max_size=4`
- C: same as B, with `batch_max_wait_ms=20 batch_max_size=8`

Harness note: the checked-in `concurrency_test.py` hardcodes 24 selected audios and cannot run N=40/60/80
as-is. I kept repo files unchanged and used a temporary `/tmp` wrapper over the same harness logic that selects
`max(sweep)=80` distinct DB samples and adds client pacing counters. The temp wrapper/results/logs were removed
after this report.

Keep-up criterion used here: no final timeouts and both TTFS p95 and proc-lag p95 below 500ms. Values above
1s are treated as falling behind realtime. The raw p95s are included so the threshold can be reinterpreted.

## N sweep

Server batch histograms are per-level deltas from cumulative `scheduler_batch_telemetry`, aligned by client
level start/end timestamps. A has no effective batch histogram because batching is disabled; it is B=1 by
construction.

| N | A: B=1 scheduler | B: 5ms/max4 batching | C: 20ms/max8 batching |
|---:|---|---|---|
| 16 | keep-up yes; lag95 88ms; TTFS95 67ms; ok 16/16 | keep-up yes; lag95 85ms; TTFS95 64ms; hist `{1:231,2:45,3:9}`; B>=3 3%; queue avg/max 0.1/35ms | keep-up yes; lag95 86ms; TTFS95 66ms; hist `{1:227,2:47,3:9}`; B>=3 3%; queue avg/max 0.1/35ms |
| 24 | keep-up no; lag95 1626ms; TTFS95 1606ms; ok 24/24 | keep-up yes; lag95 169ms; TTFS95 149ms; hist `{1:165,2:153,3:35,4:12}`; B>=3 13%; queue avg/max 0.1/46ms | keep-up yes; lag95 157ms; TTFS95 137ms; hist `{1:173,2:143,3:46,4:3,5:2}`; B>=3 14%; queue avg/max 0.1/35ms |
| 40 | keep-up no; lag95 6680ms; TTFS95 6660ms; ok 40/40 | keep-up yes; lag95 132ms; TTFS95 112ms; hist `{1:226,2:85,3:75,4:239}`; B>=3 50%; queue avg/max 3.6/117ms | keep-up yes; lag95 128ms; TTFS95 109ms; hist `{1:245,2:160,3:93,4:52,5:73,6:13,7:5,8:9}`; B>=3 38%; queue avg/max 0.4/47ms |
| 60 | keep-up no; lag95 19494ms; TTFS95 19474ms; ok 60/60 | keep-up no; lag95 4979ms; TTFS95 4959ms; hist `{1:33,2:7,3:4,4:631}`; B>=3 94%; queue avg/max 60.7/1081ms | keep-up no; lag95 1136ms; TTFS95 1116ms; hist `{1:136,2:29,3:57,4:12,5:18,6:12,7:23,8:293}`; B>=3 72%; queue avg/max 6.4/113ms |
| 80 | keep-up no; lag95 28520ms; TTFS95 28501ms; ok 31/80, 49 final timeouts | keep-up no; lag95 18759ms; TTFS95 18739ms; hist `{1:19,2:6,3:4,4:771}`; B>=3 97%; queue avg/max 152.3/2024ms | keep-up no; lag95 4397ms; TTFS95 4377ms; hist `{1:57,2:34,3:20,4:13,5:8,6:9,7:7,8:554}`; B>=3 87%; queue avg/max 54.0/538ms |

Accuracy sanity: all completed sessions were exact vs their per-config N=1 baseline. A at N=80 completed
31/80 before final timeouts; those 31 were exact. B and C completed 80/80 at N=80 and were exact.

## Knees and batching onset

| Config | Strict keep-up knee | Delta vs A | Notes |
|---|---:|---:|---|
| A: B=1 scheduler | 16 | baseline | N=24 already falls behind by ~1.6s p95 proc-lag. |
| B: 5ms/max4 | 40 | +24 streams, 2.5x | Batches form enough by N=24, and B=4 dominates under overload. |
| C: 20ms/max8 | 40 | +24 streams, 2.5x | Does not move the strict knee to N=60, but greatly reduces overload lag. |

B>=3 onset:

- B default: first observed at N=16, but only 3% of batches. Non-trivial B>=3 starts at N=24 (13%), and
  N=40 has 50% B>=3 with many B=4 batches.
- C longer-wait: first observed at N=16, also 3%. Non-trivial B>=3 starts at N=24 (14%); B>=5 appears by
  N=24, and B=8 becomes common at N=40+.

Longer-wait tradeoff:

- At the strict knee (N=40), C is not meaningfully better than B: C lag95 128ms vs B 132ms.
- Above the knee, C is materially better: N=60 lag95 improves from 4979ms to 1136ms, and N=80 from 18759ms
  to 4397ms. This is real batching engagement, but still not realtime keep-up by the 500ms criterion.
- C's larger batches reduce queue-wait pressure versus B at overload: N=60 queue avg/max 6.4/113ms vs
  B's 60.7/1081ms; N=80 54.0/538ms vs B's 152.3/2024ms.

## Client-bound check

The client did not saturate in this run. Across all configs and N:

- send-overrun p95 stayed about 0.9-1.1ms.
- max send-overrun stayed below 5ms (`C/N80` max 4.7ms was the worst).
- max chunk-late stayed 0.0ms.
- pacing violations were 0 for every config and N.

So the failed knees here are server/GPU scheduler limits, not the single Python client failing to pace 80
realtime streams on the shared box. The shared-box caveat still applies for future runs, but it was not the
limiting factor in this measurement.

## Verdict

Yes, Step-7 continuous batching extends the local realtime knee at high N: strict keep-up moves from N=16
with the B=1 scheduler to N=40 with batching enabled (+24 streams, 2.5x on this local run).

Batching starts to engage visibly at N=24 and strongly by N=40. The default 5ms/max4 policy is enough to
extend the knee to N=40. The 20ms/max8 policy forms larger batches and substantially improves overloaded
N=60/N=80 lag, but it does not make N=60 a strict realtime keep-up point on this box.
