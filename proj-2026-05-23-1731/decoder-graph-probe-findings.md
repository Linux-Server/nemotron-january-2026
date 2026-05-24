# Decoder Graph Probe Findings

Command:

```bash
/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python \
  proj-2026-05-23-1731/decoder_graph_harness.py \
  --step2-probe --sessions 4 --normal-chunks 20 --batch-max-size 32 \
  --final-sweep-normal-chunks 2 --decode-bench-reps 5 \
  --decode-case-samples-per-key 2 --write-findings
```

Model: `nvidia/nemotron-speech-streaming-en-0.6b`  
NeMo: `2.8.0rc0` at `/home/khkramer/src/nemotron-nano-omni/NeMo/nemo/__init__.py`  
GPU: `NVIDIA GeForce RTX 5090`

## Verdict

**NO-GO for Step 2.**

- FULL_GRAPH was confirmed: `cuda_graphs_mode=full_graph`.
- Warmed graph state: `B=32`, `max_time=375`.
- The first smaller-B graph replay after warm, `B=1, T=2`, failed with CUDA illegal memory access.
- `need_reinit` was `False` for that `B=1, T=2` replay, and `_graph_reinitialize` did not run. This was not a recapture; it was a replay failure.
- Because the CUDA context was poisoned by that failure, graph-on/off byte-exact comparison, lane-stream replay, inactive-row isolation, and graph-on P50 sizing could not be completed.

## FULL_GRAPH And Stream Safety

- Live computer: `GreedyBatchedRNNTLabelLoopingComputer`.
- Mode before warm: `full_graph`.
- Mode after warm: `full_graph`.
- Lane-stream verdict: **not reached**. Default-stream smaller-B replay failed first, so the lane-stream question remains blocked and should be treated as unsafe.

Failure evidence:

```text
FULL_GRAPH replay failed after warm: CUDA error: an illegal memory access was encountered
first replay shape after warm: input_shape=[1, 2, 640], state_batch_size=32, state_max_time=375, need_reinit=False
top reported frame: rnnt_utils.py:795, batched_hyps.scores.cpu()
```

## Correctness

Graph-on vs graph-off byte-exactness: **not proven / failed gate**.

- The graph-off fixed traces and finalize B sweep ran.
- The graph-on trace failed on the first `solo_b1` normal chunk immediately after max-B warm.
- Tokens/text/y-sequences could not be compared.
- Float decoder state allclose could not be evaluated.
- Inactive zero-length row isolation could not be evaluated.

This is a hard correctness/feasibility failure. Do not proceed to server wiring with max-B FULL_GRAPH replay.

## Recapture Characterization

- `INITIAL_MAX_TIME=375`.
- Warm reinitialize: 1 `_graph_reinitialize` at input shape `[32, 2, 640]`; resulting state max time was `375`.
- Warm finalize replay at shape `[32, 8, 640]`: `need_reinit=False`.
- Pre-replay shape probes:
  - same `B=32`, small `T`: `need_reinit=False`
  - smaller `B`, small `T`: `need_reinit=False`
  - larger `B=33`, small `T`: `need_reinit=True`
  - same `B=32`, larger `T=376`: `need_reinit=True`
- First varied-B replay at shape `[1, 2, 640]`: `need_reinit=False`, `_graph_reinitialize` calls after warm `0`, then CUDA illegal memory access.

Runtime precheck sketch remains: compare runtime `B` and projected encoder `max_time` to warmed maxima; if either would exceed warmed values, route to eager. This probe shows that precheck is not sufficient for max-B masking safety, because a smaller-B replay can fail without recapture.

## P50 Sizing

Conc-10 distribution from `ec2-bench/leaderboard_decomp_prod_l40s_full1000_c10.srvlog`:

- steady B histogram: `{1: 44018, 2: 5988, 3: 1086, 4: 264, 5: 68, 6: 1}`
- finalize B histogram: `{1: 1000}`
- finalize profile decode wall from records: p50 `3.696 ms`, p95 `6.100 ms`

Local decode-only eager CUDA-event timing at that small-B distribution:

- steady weighted p50: `0.829 ms`
- steady weighted p95: `0.829 ms`
- finalize B=1 p50: `1.703 ms`
- combined weighted p50: `0.829 ms`

FULL_GRAPH small-B timing was **not measured** because the first small-B graph replay failed. Therefore the recoverable host-loop/sync share and residual FULL_GRAPH copy/CPU-conversion share are unavailable from this run.

## ROI Gate

- PROVISIONAL FLOOR: **NO-GO**. FULL_GRAPH exists, but max-B warm followed by smaller-B replay is not feasible.
- PROVISIONAL UPSIDE: **NO-GO / not measurable**. No graph-on small-B timing was possible.
- Overall Step-2: **NO-GO**.

`conc10-pivot-findings.md` was written because the UPSIDE projection could not pass.

## Memory And Deploy Gates

Memory for `B=32, max_time=375` warm on the local 5090:

- before warm allocated/reserved: `2,567,023,616 / 2,753,560,576` bytes
- after warm allocated/reserved: `2,629,936,128 / 3,814,719,488` bytes
- delta allocated: `62,912,512` bytes per warmed model/lane
- delta reserved: `1,061,158,912` bytes per warmed model/lane
- graph state tensor bytes: `33,462,882`
- peak allocated/reserved during warm: `6,469,478,400 / 6,677,331,968` bytes

Deploy gates:

- `torch=2.11.0+cu130`
- `torch_cuda=13.0`
- `cuda-python=13.2.0`
- driver CUDA: `13.0`
- `check_cuda_python_cuda_graphs_conditional_nodes_supported`: passed

## EOU

EOU probe is out of the leaderboard configuration and was not exercised. Keep eou-on routed to eager unless a separate eou-on graph byte-exact check is added.

## Blockers

- FULL_GRAPH max-B replay is not safe for the smaller B values that dominate conc-10.
- Byte-exact graph-on/off output is unproven because graph-on replay failed before comparison.
- Lane-stream replay is unproven because default-stream replay already failed.
- Inactive zeroed-length row isolation is unproven.

## Suggested Next Step

Stop before server wiring. Reproduce the B=32 warm then B=1 replay failure under `CUDA_LAUNCH_BLOCKING=1`. Only revisit graph wiring if an exact-B/per-lane capture variant proves token/text/y-sequence byte-exact and stream-safe.
