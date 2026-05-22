# Round 2 - Batched `vad_stop` Barrier Drain

Date: 2026-05-21 local. Host GPU: NVIDIA GeForce RTX 5090. Scope: local only, no commits.

## Change

Added `NEMOTRON_BATCH_BARRIER_DRAIN=1`, default off. It is active only when the continuous scheduler and
`NEMOTRON_BATCH_SCHED=1` are active.

Design:

- Default off keeps `_scheduler_drain_once` on the existing `_scheduler_drain_ready_barrier_locked` path.
- Flag on stores a non-audio control event (`vad_stop`, `reset`, `close`, etc.) as a per-session pending barrier
  when that session is still ready.
- The session remains eligible for the normal ready-batch scheduler. Those chunks go through
  `_scheduler_process_ready_batch_locked_sessions` and `_process_ready_batch`.
- The pending control event is dispatched only after `_scheduler_session_ready(session)` becomes false, under the
  session `state_lock`. The queue `task_done()` is also delayed until that point, so per-session event ordering is
  preserved.
- With `NEMOTRON_MODEL_LANES>1`, these chunks enter the same lane-dispatch path as normal steady ready batches.
  They do not call `_scheduler_process_one_ready_chunk_locked`, so steady barrier chunks no longer force
  `_scheduler_exclusive_model_path`.

Residual note: the measured `vad_stop`/`reset` path is batched. The separate forced-finalize close/end path that
flushes post-stop audio can still call the legacy drain after appending that post-stop audio; ordering is preserved,
but that rare close-with-post-stop storm was not the measured target here.

## Verification

Static/local checks:

- `PYTHONPYCACHEPREFIX=/tmp/nemotron-round2-pycache /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python -m py_compile src/nemotron_speech/server.py`
- `stt-benchmark/.venv/bin/python -m py_compile proj-2026-05-21-0410/inphase_loadgen.py`
- `git diff --check -- src/nemotron_speech/server.py`
- Lightweight scheduler simulations:
  - flag off selected `_scheduler_drain_ready_barrier_locked` before dispatch
  - flag on held `vad_stop`, drained one normal ready batch, then dispatched the event

Byte-exact gate:

- Harness: `proj-2026-05-21-0410/inphase_loadgen.py --include-interims`
- Levels: N=115, 120, 150
- Comparison fields: final transcript, full final delta list, full interim sequence, duplicate-final check
- Result: `BYTE_EXACT_PASS` for all 385 streams. No mismatches printed.

Default-off identity:

- Startup log: `batch_barrier_drain=False`
- Off run used the legacy barrier log path (`scheduler barrier drained ... ready chunks before ...`).
- No `scheduler_batch_barrier_drained` lines appeared in the off run.

FORK_ASSERT:

- Both off and on servers ran with `NEMOTRON_FORK_ASSERT=1`.
- No `ERROR`, `Traceback`, fork assertion failure, illegal memory, CUDA, or stream errors.
- Fork assertion pass lines: off `770`, on `2320` across all local runs.

## In-Phase Results

Common env:

```text
NEMOTRON_CONTINUOUS=1
NEMOTRON_SCHEDULER_B1=1
NEMOTRON_BATCH_SCHED=1
NEMOTRON_BATCH_MAX_SIZE=32
NEMOTRON_BATCH_MAX_WAIT_MS=8
NEMOTRON_BATCH_MEMORY_TELEMETRY_EVERY=1
NEMOTRON_WARMUP_MS=200
NEMOTRON_FINALIZE_SILENCE_MS=0
NEMOTRON_FORK_ASSERT=1
```

Paired byte-exact run, with interim capture enabled:

| Flag | N | strict | TTFS p95 ms | lag p95 ms | legacy B=1 barrier chunks | batched barrier chunks | avg B ready path | avg effective B | full B32 batches |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| off | 115 | yes | 207.6 | 342.5 | 0 | 0 | 23.58 | 23.58 | 239/394 |
| off | 120 | no | 7947.2 | 8107.0 | 1012 | 0 | 27.17 | 7.26 | 235/318 |
| off | 150 | no | 25170.2 | 25329.6 | 2556 | 0 | 31.33 | 4.17 | 284/298 |
| on | 115 | yes | 205.5 | 330.7 | 0 | 1 | 23.58 | 23.58 | 239/394 |
| on | 120 | no | 477.9 | 605.7 | 0 | 57 | 26.96 | 26.96 | 265/358 |
| on | 150 | no | 2363.6 | 2510.8 | 0 | 1660 | 31.46 | 31.46 | 363/378 |

Normal-output performance rerun, flag on:

| N | strict | TTFS p95 ms | lag p95 ms | legacy B=1 barrier chunks | batched barrier chunks | avg effective B |
|---:|:---:|---:|---:|---:|---:|---:|
| 115 | yes | 288.0 | 422.7 | 0 | 24 | 23.16 |
| 120 | yes | 207.5 | 342.9 | 0 | 3 | 26.44 |
| 130 | no | 673.2 | 810.5 | 0 | 184 | 23.46 |
| 140 | no | 1786.2 | 1921.4 | 0 | 1008 | 31.34 |
| 150 | no | 4642.2 | 4779.3 | 0 | 3275 | 31.46 |

Artifacts:

- `proj-2026-05-21-inference-opt/round2-artifacts/inphase-off-115-120-150.json`
- `proj-2026-05-21-inference-opt/round2-artifacts/inphase-on-115-120-150.json`
- `proj-2026-05-21-inference-opt/round2-artifacts/inphase-on-perf-115-120-150.json`
- `proj-2026-05-21-inference-opt/round2-artifacts/inphase-on-refine-130-140.json`
- `proj-2026-05-21-inference-opt/round2-artifacts/server-off.log`
- `proj-2026-05-21-inference-opt/round2-artifacts/server-on.log`

## Verdict

Correctness: GO. The batched barrier drain is byte-exact against the default-off path on the in-phase fixed clip
set, including interim sequence and final deltas, with `FORK_ASSERT=1` clean.

Performance: partial GO. The B=1 barrier-drain collapse is fixed: legacy barrier chunks dropped to zero with the
flag on, and effective batch size stayed high during the barrier region. N=120 can now pass the strict gate
(207.5 ms TTFS p95 in the normal-output run), so the cap moves from 115 to about 120 on this local run. N=130+
still fails, and N=150 remains seconds behind. The next visible limiter is no longer B=1 barrier drain; it is the
remaining serial final/fork/reset work under the aligned finalization storm.
