# Manual CUDA Graph Encoder Probe

Command:

```bash
/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python proj-2026-05-21-0410/probe_manual_cudagraph.py
```

Setup:

- Model: `$(cat /tmp/en-nemo-path)` English streaming checkpoint
- `att_context=[70,1]`, `greedy`, decoder CUDA graph disabled, dither `0`
- `NEMOTRON_WARMUP_MS=200`
- TF32 backend defaults preserved as observed in-process: `matmul.allow_tf32=False`, `cudnn.allow_tf32=True`
- Manual graph bucket: B=1 steady encoder input `T=25` (`pre_cache=9 + shift=16`), `drop_extra_pre_encoded=2`
- Warmup/first/final shapes are not graphed in this proof. Final chunks run eager.

## Results

Byte-exact gate: **YES**

- Clips streamed: 4
- Normal steady chunks: 120
- Manual graph replays: 120
- Eager fallbacks during normal chunks: 0
- Normal interim text: byte-identical
- Final text after eager finalization: byte-identical
- State comparison: bit-identical/allclose, `max_abs=0`

Warmup/capture time: **251.3 ms**

- This includes static buffer allocation, 5 eager warmup calls on a side stream, graph capture, and synchronize.
- This is the important cloud/cold-start signal: manual capture is sub-second locally, not torch.compile-style multi-minute Inductor warmup.

Steady per-chunk timing, synced:

- Fully eager encoder+decode: `8.845 ms` avg, `8.763 ms` p50, `9.563 ms` p95
- Manual CUDA graph encoder replay + eager decode + cache copy-out: `6.510 ms` avg, `6.454 ms` p50, `6.856 ms` p95
- Speedup: **1.36x**
- Contention: no other GPU compute process was visible before the run, and none remained after cleanup. Timing still includes per-chunk synchronization and should be treated as approximate.

Stability: **PASS**

- Replay stayed stable across the full stream: no illegal memory access, no shape errors, no recapture path, and no text drift.

## Verdict

Manual CUDA graph capture of the steady Conformer encoder bucket is **feasible, byte-exact, and fast-warmup**. This is a viable cloud lever to wire into `server.py` next, with production work needed to capture or route the separate warmup/first/final buckets safely.
