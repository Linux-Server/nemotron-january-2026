# Round 3 - Finalize Storm Prototype

Date: 2026-05-21 local. Host GPU: NVIDIA GeForce RTX 5090. Scope: local only, no commits.

## Change

Added `NEMOTRON_BATCH_FINALIZE=1`, default off. It is active only when the continuous scheduler and
`NEMOTRON_BATCH_SCHED=1` are active.

Shipped:

- A2 partial: flagged final fork calls on model-lane runs use the session's pinned model lane via a lane-local
  reservation path instead of `_scheduler_exclusive_model_path`. Same-session exclusion is kept with
  `_scheduler_inflight_sessions`. This was not exercised in the local sweep because `NEMOTRON_MODEL_LANES=1`.
- A3: scheduler debounce finalizations are collected across sessions, forked, grouped by
  `(target_lang, keep_all_outputs=True, drop_extra, final chunk_T, decoder mode, previous_hypotheses fresh,
  pred_out_stream fresh)`, then run through the existing batch stack/scatter invariants.

Not shipped:

- Batched final preprocessing. A local N=120 gate showed many byte diffs: terminal punctuation was dropped on
  final transcripts. I reverted it. Final preprocessing remains per-fork; only the final `conformer_stream_step`
  call is batched.
- Close true-boundary cleanup finalization and cold reset batching. Those still run on the legacy path after the
  client-visible speculative final and are not part of the measured TTFS path.

## Correctness

The batched path deep-clones each parent into a disposable fork before grouping. The parent snapshot used by
`NEMOTRON_FORK_ASSERT=1` is taken before the fork flush and checked after scatter. The batched model call clones
per-row hypotheses/pred state before stacking and scatters returned cache, hypotheses, and pred state back to the
fork only. Parent ASR state is not updated from the final fork.

Byte-exact gate used `proj-2026-05-21-0410/inphase_loadgen.py --include-interims`, levels
`N=120,130,140,150`, with `NEMOTRON_BATCH_BARRIER_DRAIN=1` on for both runs.

Result:

- Final transcript: exact for all 540 streams.
- Final delta list: exact for all 540 streams.
- Interim sequence: exact for all 540 streams.
- Duplicate final check: 0 duplicate-final streams.
- `FORK_ASSERT=1`: off 1080 pass lines, on 1080 pass lines. No fork assertion failures.
- Startup identity: flag off logged `batch_finalize_requested=False batch_finalize=False`, and no
  `scheduler_finalize_batch_telemetry` lines appeared in the off run.

## In-Phase Results

Common env:

```text
NEMOTRON_CONTINUOUS=1
NEMOTRON_SCHEDULER_B1=1
NEMOTRON_BATCH_SCHED=1
NEMOTRON_BATCH_BARRIER_DRAIN=1
NEMOTRON_BATCH_MAX_SIZE=32
NEMOTRON_BATCH_MAX_WAIT_MS=8
NEMOTRON_BATCH_MEMORY_TELEMETRY_EVERY=1
NEMOTRON_WARMUP_MS=200
NEMOTRON_FINALIZE_SILENCE_MS=0
NEMOTRON_FORK_ASSERT=1
```

| Flag | N | strict | TTFS p95 ms | lag p95 ms | debounced final rows | final model batches | B=1 final batches | avg final B |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| off | 120 | yes | 208.2 | 343.4 | 120 | 120 serial | 120 | 1.00 |
| off | 130 | no | 473.3 | 601.2 | 130 | 130 serial | 130 | 1.00 |
| off | 140 | no | 2189.9 | 2326.7 | 140 | 140 serial | 140 | 1.00 |
| off | 150 | no | 3725.8 | 3868.1 | 150 | 150 serial | 150 | 1.00 |
| on | 120 | yes | 147.5 | 281.1 | 120 | 70 | 50 | 1.71 |
| on | 130 | no | 419.2 | 548.0 | 130 | 67 | 45 | 1.94 |
| on | 140 | no | 1694.0 | 1822.0 | 140 | 49 | 15 | 2.86 |
| on | 150 | no | 3140.8 | 3263.4 | 150 | 36 | 1 | 4.17 |

Aggregate debounced-final telemetry with the flag on:

```text
rows=540 batches=222 avg_effective_B=2.43
effective_batch_hist={1:111, 2:44, 3:17, 4:19, 5:9, 6:10, 7:1, 8:4, 9:5, 10:1, 15:1}
serial_fallback_calls=0
```

Artifacts:

- `proj-2026-05-21-inference-opt/round3-artifacts/inphase-off-120-150.json`
- `proj-2026-05-21-inference-opt/round3-artifacts/inphase-on-120-150.json`
- `proj-2026-05-21-inference-opt/round3-artifacts/server-off.log`
- `proj-2026-05-21-inference-opt/round3-artifacts/server-on.log`

## Verdict

Correctness: GO for the shipped scope. The final fork batch path is byte-exact against flag off for finals,
final deltas, interims, and duplicate-final behavior, with `FORK_ASSERT=1` clean.

Performance: partial. The debounced final model-call count collapsed from 540 serial B=1 calls to 222 batched
calls across the sweep, with no serial fallbacks. TTFS improved at every level, but the strict in-phase knee did
not move past N=120: N=130 improved from 473.3 ms to 419.2 ms p95, still above the 400 ms strict gate. The next
visible limiter is the unbatched per-fork final preprocessing plus close/cold-reset cleanup work, not the
batched final model call itself.

Cleanup: local servers were stopped; port 8080 is free; GPU memory returned to idle.
