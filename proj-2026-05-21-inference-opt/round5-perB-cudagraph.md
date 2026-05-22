# Round 5: per-B manual CUDA graphs

Date: 2026-05-22

Scope: local RTX 5090, standalone probe only. No `server.py` or NeMo edits.

Command:

```bash
/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python proj-2026-05-21-1959-cudagraph/probe_perB_cudagraph.py
```

Artifacts:

- Probe: `proj-2026-05-21-1959-cudagraph/probe_perB_cudagraph.py`
- JSON: `proj-2026-05-21-inference-opt/round5-artifacts/perB-cudagraph-results.json`

## Setup

- Model: `$(cat /tmp/en-nemo-path)` English streaming checkpoint.
- `att_context=[70,1]`, `greedy`, decoder CUDA graph disabled, dither `0`.
- `NEMOTRON_WARMUP_MS=200`.
- TF32 defaults preserved: `matmul.allow_tf32=False`, `cudnn.allow_tf32=True`.
- Steady encoder bucket: `T=25` (`pre_cache=9 + shift=16`), `drop_extra_pre_encoded=2`.
- Captured exact manual graph buckets for B=1..16, one static-buffer set per B, no padding.
- Clip set: 16 independent English clips selected from the benchmark DB with 64 normal steady chunks each.
- GPU contention: no other compute apps visible before the run; none remained after cleanup. Desktop graphics processes only.

CUDA-event timing below is event elapsed around the steady call path: eager batched
`conformer_stream_step(B)` vs manual graph encoder replay plus eager RNNT decode, including cache scatter/clone work.

## Correctness

Hard gate passed for every tested B=1..16:

- Interim per-stream text: byte-identical.
- Final per-stream text after eager finalization: byte-identical.
- State snapshots: bit-identical/allclose, `max_abs=0`.
- Graph engagement: 64 graph replays per B, 0 eager fallbacks.

## Curve

| B | byte/state | eager wall ms | graph wall ms | wall speedup | eager GPU ms | graph GPU ms | GPU speedup | GPU drop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | pass, max_abs=0 | 9.375 | 6.584 | 1.424x | 9.353 | 6.565 | 1.425x | 29.8% |
| 2 | pass, max_abs=0 | 10.672 | 7.907 | 1.350x | 10.648 | 7.887 | 1.350x | 25.9% |
| 3 | pass, max_abs=0 | 11.115 | 8.417 | 1.321x | 11.092 | 8.396 | 1.321x | 24.3% |
| 4 | pass, max_abs=0 | 12.099 | 9.977 | 1.213x | 12.077 | 9.952 | 1.214x | 17.6% |
| 5 | pass, max_abs=0 | 13.833 | 11.667 | 1.186x | 13.806 | 11.639 | 1.186x | 15.7% |
| 6 | pass, max_abs=0 | 14.114 | 11.920 | 1.184x | 14.089 | 11.895 | 1.184x | 15.6% |
| 7 | pass, max_abs=0 | 15.858 | 13.598 | 1.166x | 15.829 | 13.569 | 1.167x | 14.3% |
| 8 | pass, max_abs=0 | 16.359 | 14.164 | 1.155x | 16.326 | 14.131 | 1.155x | 13.4% |
| 9 | pass, max_abs=0 | 16.962 | 14.413 | 1.177x | 16.924 | 14.380 | 1.177x | 15.0% |
| 10 | pass, max_abs=0 | 17.497 | 14.941 | 1.171x | 17.459 | 14.907 | 1.171x | 14.6% |
| 11 | pass, max_abs=0 | 18.799 | 16.391 | 1.147x | 18.764 | 16.357 | 1.147x | 12.8% |
| 12 | pass, max_abs=0 | 19.688 | 17.348 | 1.135x | 19.653 | 17.311 | 1.135x | 11.9% |
| 13 | pass, max_abs=0 | 20.443 | 18.048 | 1.133x | 20.409 | 18.010 | 1.133x | 11.8% |
| 14 | pass, max_abs=0 | 21.011 | 18.704 | 1.123x | 20.975 | 18.666 | 1.124x | 11.0% |
| 15 | pass, max_abs=0 | 21.836 | 23.904 | 0.914x | 21.799 | 23.862 | 0.914x | -9.5% |
| 16 | pass, max_abs=0 | 22.832 | 20.314 | 1.124x | 22.791 | 20.273 | 1.124x | 11.1% |

Notes:

- B=15 had a graph-run average outlier; its graph p50 was 19.901 ms while avg was 23.904 ms. The adjacent B=14/B=16 points are back on the ~11% GPU-drop curve, so I do not treat B=15 as a separate architectural cliff.
- Capture cost for all B=1..16 was about 1.36 s total. B=1 was 289 ms; B=2..16 were 57-81 ms each after the first capture.

## K Recommendation

The first average wall-speedup point below the ~1.15x cutoff is B=11 (`1.147x`). Recommended graph max bucket:
`K=10`, meaning wire B=1..10 if the scheduler work proceeds. B=11 is marginal if an implementation wants a softer
threshold, but the clean cutoff is B=10.

## Verdict

Per-B manual graph capture is correctness-cleared: B=1..16 is byte-exact against eager batched B, including
`max_abs=0` state.

The GPU-active question is positive. CUDA-event elapsed drops with the graph, not just host wall time: ~30% at B=1,
~18% at B=4, ~14-16% around B=5..10, and ~11-13% beyond the cutoff. That is evidence the graph collapses launch
occupancy in the steady per-call path, which is exactly the mechanism that could raise a GPU-bound multi-process+MPS
ceiling.

Given the current scaling picture, full scheduler wiring is worth a bounded implementation only for B<=10 and only
behind fail-closed/default-off gating. Multi-process+MPS remains the primary scaling lever, but this result is strong
enough to justify a cloud GPU-bound retest after wiring, specifically L4 at its K=2/MPS ceiling and L40S around K=4/5
MPS. The question for that retest is whether the ~11-16% event-time reduction in the likely batched range actually
translates into higher per-box ceiling, not just lower single-process latency.
