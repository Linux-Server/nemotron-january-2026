# Foundation Invariance Gate

## Verdict

**GO: concurrent A/B oracle is valid for Steps 2/3.**

The current server with decode graph OFF (`greedy_batch`, `loop_labels=True`, `use_cuda_graph_decoder=False`) produced byte-exact per-session token sequences and decoded text across different eager batch compositions.

## Run

Command:

```bash
/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python \
  proj-2026-05-23-1731/decoder_graph_harness.py \
  --normal-chunks 20 --sessions 4 --final-tail-samples 4000
```

Server config forced by the harness:

- `NEMOTRON_CONTINUOUS=1`
- `NEMOTRON_SCHEDULER_B1=1`
- `NEMOTRON_BATCH_SCHED=1`
- `NEMOTRON_BATCH_FINALIZE=1`
- `NEMOTRON_BATCH_MAX_SIZE=32`
- `NEMOTRON_MODEL_LANES=1`
- `NEMOTRON_FINALIZE_SILENCE_MS=0`
- `NEMOTRON_WARMUP_MS=200`
- `right_context=1`

Decode graph remained off. Encoder compile / encoder CUDA graphs / finalize encoder CUDA graphs were explicitly disabled.

## Evidence

Compared `solo_b1` against:

- `co_batched_bn`
- `row_permutations`
- `shrink_grow`

Trace shape:

- 4 distinct real PCM clips from `proj-2026-05-20-modal-cost/loadgen_audio`
- 20 steady normal chunks per session
- 1 final event per session
- 84 per-session logical events compared by `(session_id, logical_event)`, not batch row

Result:

- token/text mismatches: `0`
- per-session token/text invariant: `true`
- float decoder state byte-equal: `false`
- float decoder state allclose at `atol=1e-4, rtol=1e-5`: `true`
- observed max abs float diff: `3.337860107421875e-06`
- observed max rel float diff: `4.949386857333593e-05`
- smallest checked absolute tolerance covering observed max abs: `1e-5`
- max-diff path: `co_batched_bn.s1.final:0000.previous_hypotheses[0].dec_state.predictor_state[1]`

This matches the expected oracle premise: user-visible tokens/text are byte-exact, while decoder float state is close but not bit-identical.

## NeMo

- imported path: `/home/khkramer/src/nemotron-nano-omni/NeMo/nemo/__init__.py`
- version: `2.8.0rc0`
- model: `nvidia/nemotron-speech-streaming-en-0.6b`

## Harness API

`proj-2026-05-23-1731/decoder_graph_harness.py` is standalone and reusable for Steps 2/3.

Main entry points:

- `build_server(model=..., lanes=1, batch_max_size=32, right_context=1)`: instantiates and loads the real `ASRServer`.
- `select_audio_clips(server, audio_dir=..., session_count=4, normal_chunks=20, final_tail_samples=4000)`: selects distinct real clips long enough for the fixed trace.
- `build_scenario_plans(session_ids, normal_chunks)`: builds B=1, B=N, row permutation, and shrink/grow traces.
- `run_scenario(server, clips, plan, final_tail_samples=..., lane_id=0)`: drives one trace through the real server methods and captures per-session events.
- `run_composition_suite(server, ...)`: runs all Step-1 scenarios and returns metadata, captures, and comparison summary.
- `compare_captures(reference_name, reference, candidates)`: compares token/text byte equality and decoder float-state allclose.

Methods exercised directly:

- session init: `_init_session_without_synthetic_warmup`
- real PCM feed: `_scheduler_append_audio_locked`
- steady batch path: `_process_ready_batch`
- finalize chain: `_continuous_prepare_finalize_item_locked` -> `_prepare_final_fork_batch_row` -> `_finalize_batch_group_key_for_row` -> `_process_final_batch_rows`
- lane TLS wrapper: `_run_scheduler_model_lane_call_sync` is used automatically when `--lanes > 1`

## Blockers / Concerns

No Step-1 blocker.

This run used `lanes=1`, which the plan allows for the foundation probe. The harness supports `--lanes 2`, but I did not run the optional lanes=2 spot-check in this pass.

## Next Step

Proceed to Step 2 using the concurrent A/B oracle for token/text correctness, while keeping float decoder state at allclose-only tolerance.
