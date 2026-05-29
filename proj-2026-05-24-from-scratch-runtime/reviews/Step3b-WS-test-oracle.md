# Step 3b WS Test Oracle

`tests/server_compat/run_compat.py` is the Part B pre-merge oracle for the Python
server and the C++ `ws_server`.

It launches:

- Python: `runtime/.venv/bin/python src/nemotron_speech/server.py --host 127.0.0.1 --port 8080`
- C++: `runtime/cpp/build/ws_server --port 8081 --admission-active-cap 64 --steady-batch-dir runtime/steady_b_artifacts`

Both subprocesses receive `HF_HUB_OFFLINE=1`, `NEMOTRON_CONTINUOUS=1`,
`NEMOTRON_FINALIZE_SILENCE_MS=0`, the same `NEMOTRON_ARTIFACT_DIR`, and density
batching env for the C++ path. The harness waits for `/health` to report
`model_loaded: true`, drives `utt0..utt7` from `runtime/artifacts/session_audio_bundle.ts`
as 16 kHz mono int16-LE PCM in 640-byte chunks, sends `vad_start` before audio
and `vad_stop` after audio, then waits for a final transcript.

Assertions:

- Ready frame is exactly `{"type":"ready"}` after JSON parse.
- Per utterance, Python and C++ transcript event counts match.
- Per event, `type`, `text`, `is_final`, and the optional `finalize` flag match.
- Final accumulated collector text matches.
- Final transcript `finalize_timing` contains the Step 1 audit's nine raw keys:
  `reason`, `vad_stop`, `vad_stop_recv`, `debounce_expiry`, `fork_flush_start`,
  `fork_flush_done`, `final_sent`, `inference_lock_acquire_wait_ms`,
  `gil_attrib_enabled`.
- `?model=bogus` returns the expected WebSocket error frame or HTTP rejection.
- `/stats?last=bogus` returns HTTP 400 with a JSON `error` body.

Canonicalization strips volatile fields before diagnostic diffs:

- Raw timestamps: `since_unix`, `until_unix`, and `finalize_timing` numeric values.
- Sequence/process identity: `finalize_seq`, `pid`, `process_label`.
- Native-only scheduler/admission extensions: scheduler telemetry, stale-gen/native
  scheduler blocks, and volatile admission counters.

The perf gate measures C++ WebSocket TTFS at `N=8` from `vad_stop` send to final
transcript receipt, then stops the servers and runs `density_main --mode density-sweep
--n-values 8 --batch-steady on`. The gate is:

`ws_overhead_p95 = ttfs_via_ws_p95 - ttfs_via_density_scheduler_p95 <= max(2ms, 0.10 * ttfs_via_density_scheduler_p95)`.

`vad_stop_recv` is accepted as numeric or null because the Step 1 audit documents
that Python only populates it under `NEMOTRON_FINALIZE_PROFILE=1`; the C++ runtime
currently preserves the key with a null value.
