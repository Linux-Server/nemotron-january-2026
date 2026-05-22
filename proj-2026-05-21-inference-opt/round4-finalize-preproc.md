# Round 4 - Finalize Preprocessing

Date: 2026-05-22 local. Host GPU: NVIDIA GeForce RTX 5090. Scope: local only, no commits.

## Change

Added `NEMOTRON_BATCH_FINALIZE_PREPROC=1`, default off. It is active only when
`NEMOTRON_BATCH_FINALIZE=1`, the continuous scheduler is active, and `NEMOTRON_BATCH_SCHED=1`.

The final fork path now keeps the existing per-fork finalization semantics but batches the preprocessor calls
inside `_process_final_fork_groups`:

- Build each fork's final pending-audio loop exactly as the serial `_process_final_chunk` path does.
- Group each preprocessor call by exact `(valid_samples, frames_this_call)`.
- Run the fixed-shape preprocessor for groups up to `NEMOTRON_BATCH_MAX_SIZE`.
- Crop each row to its real `frames_this_call` before appending to that fork's `new_mels`.
- Keep the existing final model-call grouping by `(target_lang, keep_all_outputs=True, drop_extra, final chunk_T,
  decoder mode, previous_hypotheses fresh, pred_out_stream fresh)`.

Root cause of the prior punctuation drop: the final fork tail is variable length. A batched preprocessor attempt
that treats all final rows like steady-state rows can return/carry a full `shift_frames` slice or otherwise align
rows by the batch max instead of the row's real final tail. With `keep_all_outputs=True`, that tail is exactly
where the final word boundary and terminal punctuation are emitted. This round groups by exact valid audio length
and exact real output frames, then crops per row before concatenation, so no padded/truncated tail reaches the
final model call.

Close cleanup: under the same active flag, scheduler `reason="close"` keeps the ordering `final emit -> cleanup`,
but replaces the post-final cold model reset with a state-clear-only close cleanup. `end` still uses the ordered
cold reset path because the connection may continue with new audio.

## Verification

Static checks:

```text
PYTHONPYCACHEPREFIX=/tmp/nemotron-round4-pycache /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python -m py_compile src/nemotron_speech/server.py
git diff --check -- src/nemotron_speech/server.py
```

Common runtime env:

```text
NEMOTRON_CONTINUOUS=1
NEMOTRON_SCHEDULER_B1=1
NEMOTRON_BATCH_SCHED=1
NEMOTRON_BATCH_BARRIER_DRAIN=1
NEMOTRON_BATCH_FINALIZE=1
NEMOTRON_BATCH_MAX_SIZE=32
NEMOTRON_BATCH_MAX_WAIT_MS=8
NEMOTRON_BATCH_MEMORY_TELEMETRY_EVERY=1
NEMOTRON_WARMUP_MS=200
NEMOTRON_FINALIZE_SILENCE_MS=0
NEMOTRON_FORK_ASSERT=1
```

Byte-exact gate:

- Harness: `proj-2026-05-21-0410/inphase_loadgen.py --include-interims`
- Levels: `N=120,130,140,150`
- Compared fields: per-stream final transcript, full final-delta list, full interim sequence.
- Result: `BYTE_EXACT_PASS` for all 540 compared streams.
- Mismatches printed: 0.
- Duplicate-final streams: off 0, on 0.
- Max transcript length in this fixed set: 244 chars, so the loadgen transcript field was not truncated.

Default-off identity:

- Off startup logged `batch_finalize_preproc_requested=False batch_finalize_preproc=False`.
- Off run used the round-3 finalize path: final model batching active, final preprocessing still serial.

FORK_ASSERT:

- No `fork alias assertion FAILED`.
- No `Traceback`, illegal memory, or `scheduler_finalize_preproc_batch_fallback`.
- Pass lines: off 1073, on 1080. The off server was interrupted after the loadgen output while some post-final
  close cleanup was still draining; the measured finalization path had no assertion failures.

Close cleanup:

- Flag on logged 540 `continuous close cleanup complete` lines and 0 close cold resets.
- Flag off logged close cold resets on the legacy path.

Artifacts:

- `proj-2026-05-21-inference-opt/round4-artifacts/inphase-off-120-150.json`
- `proj-2026-05-21-inference-opt/round4-artifacts/inphase-on-120-150.json`
- `proj-2026-05-21-inference-opt/round4-artifacts/server-off.log`
- `proj-2026-05-21-inference-opt/round4-artifacts/server-on.log`

## In-Phase Results

Both runs used barrier-drain and finalize-storm. The only intended difference was
`NEMOTRON_BATCH_FINALIZE_PREPROC`.

| Flag | N | strict | TTFS p95 ms | lag p95 ms |
|---|---:|:---:|---:|---:|
| off | 120 | yes | 154.4 | 287.6 |
| off | 130 | yes | 312.3 | 439.3 |
| off | 140 | no | 1501.8 | 1632.3 |
| off | 150 | no | 2750.0 | 2872.5 |
| on | 120 | yes | 135.2 | 266.9 |
| on | 130 | yes | 132.1 | 266.0 |
| on | 140 | yes | 183.6 | 314.3 |
| on | 150 | no | 839.8 | 966.6 |

Finalize model batch telemetry:

```text
off: rows=540 batches=222 avg_effective_B=2.43 serial_fallback_calls=0
on:  rows=540 batches=254 avg_effective_B=2.13 serial_fallback_calls=0
```

## Verdict

Correctness: GO. With `NEMOTRON_BATCH_FINALIZE_PREPROC=1`, the in-phase fixed clip set is byte-identical to flag
off for finals, final deltas, interims, and duplicate-final behavior.

Performance: partial GO. The strict in-phase knee moved from `N=130` in the paired flag-off run to `N=140` with
the flag on, and TTFS improved at every measured level. It did not reach the ~180 forced-batch ceiling: `N=150`
still failed strict at 839.8 ms p95. The remaining limiter is beyond final preprocessing alone.

Cleanup: local servers were stopped; port 8080 is free; GPU memory returned to idle.
