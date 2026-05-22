# Parallel Lane Feasibility Probe

Date: 2026-05-21 local. Host GPU: NVIDIA GeForce RTX 5090. Model:
`/home/khkramer/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/snapshots/ef3bf40c90df5cd2de55cc07e06681e03d8e6ee4/nemotron-speech-streaming-en-0.6b.nemo`.

Setup:

- Python: `/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python`
- Direct `model.conformer_stream_step`, no server, no Modal
- English model, `att_context=[70,1]`, `greedy_batch`, TF32 off, `dither=0`
- `NEMOTRON_WARMUP_MS=200`
- Four independent B=1 steady-state rows from separate clips, each with its own input tensor, cache tensors,
  previous hypothesis, and previous pred-out state
- Steady chunk shape: `T=25` mel frames, `drop_extra_pre_encoded=2`
- Thread tests discarded 5 measured warmup trials and used 25 median trials per K

Important correctness caveat for future scheduler work: NeMo's `cache_aware_stream_step` mutates the shared
`encoder.streaming_cfg.drop_extra_pre_encoded` during the call. This probe forced all concurrent lanes to the
same steady-state `drop_extra=2` and set the baseline cfg to `2` before measuring. A production multi-lane
path must not concurrently run mixed `drop_extra` values through the same shared model without fixing or
guarding that global mutation.

## Part A - One B=1 Step

Low-overhead synced B=1 steady-state step:

| measure | median ms |
|---|---:|
| Wall span, call + synchronize | 9.730 |
| Python call/launch span, no extra sync wait | 9.719 |
| Thread CPU time around the call | 9.718 |
| CUDA device self-time from `torch.profiler` CUDA events | 6.765 |
| Non-GPU-active remainder, wall - CUDA self | 2.965 |

Interpretation: the local step is about 70% CUDA-active by this measurement (`6.77 / 9.73`) and about 30%
host/launch/gap (`2.96 / 9.73`). This matches the earlier launch-bound picture: the GPU work is not the whole
wall span, so some host-side overlap is possible.

Profiler note: `torch.profiler` CPU self-time was inflated by instrumentation (`19.7 ms` excluding explicit
sync), so I used the low-overhead call span plus profiler CUDA self-time for the split. The profiler still
showed the expected CPU launch cost shape, with `cudaLaunchKernel` self-time around `3.10 ms`.

## Part B - Serial vs Concurrent Threads

Medians, synchronized after all work:

| K | serial default stream ms | concurrent default stream ms | default overlap | concurrent K streams ms | stream overlap | K-stream thread CPU sum ms | CPU-sum / wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 21.824 | 21.392 | 1.02x | 19.173 | 1.14x | 34.258 | 1.79x |
| 3 | 32.728 | 30.349 | 1.08x | 21.932 | 1.49x | 53.966 | 2.46x |
| 4 | 43.918 | 38.157 | 1.15x | 31.140 | 1.41x | 85.790 | 2.75x |

Separate CUDA streams matter. Concurrent threads on the default stream were close to serialized, while one
explicit stream per lane improved the median overlap to about `1.4-1.5x` at K=3/4.

The GIL is not held for the whole call. At K=4 with separate streams, the worker threads accumulated `85.8 ms`
of thread CPU time inside a `31.1 ms` wall span. That cannot happen if all model-call work is Python bytecode
serialized under the GIL. The call spends substantial time in C/CUDA/PyTorch sections that release the GIL.

The overlap is also far below K-way scaling. Even with separate streams, K=4 produced only `1.41x` median
speedup over the serial lane, not `4x`. So threads are not blocked by a hard GIL wall, but the local 5090 path
still has driver/GPU/stream contention or internally serialized regions that cap the easy headroom.

## Part C - 2-Process Sanity Check

Two child processes each loaded the same model, warmed up, waited on a shared start file, then ran 40 B=1
steady-state steps on the same GPU. Model load time was excluded.

| process | median step ms | loop total ms | steps/s |
|---:|---:|---:|---:|
| 0 | 19.082 | 773.387 | 51.72 |
| 1 | 19.086 | 772.937 | 51.75 |

Using the max child loop time, aggregate throughput was `103.4 steps/s`. The single-process B=1 rate from
Part A was `102.8 steps/s`, so the 2-process aggregate was only `1.01x` of single-process throughput, not
`~2x`. On this local setup, separate processes appear to time-share the GPU work rather than doubling
throughput.

## Verdict

Parallel model-call lanes are viable with threads only in the limited sense: independent Python threads on
explicit CUDA streams do overlap, and this is not a pure GIL serialization failure. The measured local
headroom is modest, about `1.4-1.5x` at K=3/4, with no local evidence that two processes improve aggregate
throughput.

Concrete scheduler implication:

- Do not expect a simple `inference_lock -> 4 thread lanes` change to produce 4x. A feature-flagged small lane
  pool, probably N=2 or N=3 first, is the realistic experiment.
- Each lane must own an explicit `torch.cuda.Stream`; default-stream concurrency is mostly serialized.
- The shared mutable `drop_extra_pre_encoded` behavior must be fixed or guarded before allowing mixed first
  chunks and steady-state chunks to run concurrently on the shared model.
- Outputs must be synchronized per lane before session state is consumed.

Recommendation: thread lanes are worth a guarded prototype because they are much easier than process lanes and
did show real overlap. Processes are not justified by this local 5090 result. Because the cloud knee is the
actual problem and the cloud host gap is larger/single-core-bound, remeasure this exact probe on the target
cloud GPU/driver before using the local `1.4-1.5x` number for capacity planning.

## Prototype End-to-End Result

Date: 2026-05-21 local. Prototype flag: `NEMOTRON_MODEL_LANES=2`, with
`NEMOTRON_SCHEDULER_B1=1`, `NEMOTRON_BATCH_SCHED=1`, `NEMOTRON_BATCH_MAX_SIZE=32`,
`NEMOTRON_BATCH_MAX_WAIT_MS=8`, `NEMOTRON_WARMUP_MS=200`, and `NEMOTRON_FORK_ASSERT=1`.

Implementation note: the working prototype uses one worker thread, one CUDA stream, and one restored model
instance per lane. Sharing the same NeMo model across lane threads produced real transcript corruption in the
first attempt (`22/24` final exact, `13/24` strict exact), so the safe local prototype isolates mutable model
state per lane. Sessions are pinned before init/warmup and all exclusive non-steady calls run on that session's
pinned lane after draining other lanes. Concurrent dispatch is allowed only for steady normal chunks with the
same compatibility key: steady `drop_extra_pre_encoded=2`, steady chunk geometry, same prompt/language key.
First chunks, finalize/barrier drain, cold reset, and any non-steady geometry are exclusive.

Correctness gates:

| gate | result |
|---|---|
| Default-off / lanes=1 | `NEMOTRON_MODEL_LANES=1` strict canary: `24/24` final exact and `24/24` strict exact. Lane resources are not created on this path. |
| Byte exact, lanes=2 | `24/24` strict exact canary, then high-N sweep `N=56/72/80` all strict exact (`56/56`, `72/72`, `80/80`), max edit distance `0`. |
| Fork assert / CUDA | `NEMOTRON_FORK_ASSERT=1` clean; server log grep found no fork assertion failures, lane task failures, illegal-memory, or CUDA stream errors. |
| Cleanup | Servers were stopped after measurement; scratch artifacts stayed under `/tmp/nemotron-lanes`. |

End-to-end realtime keep-up sweep, out-of-phase `concurrency_test.py`, local RTX 5090, same server flags:

| config | keep-up points | first failing point | approximate knee |
|---|---:|---:|---:|
| canonical lanes=1 baseline from `max-parallelism-sweep.md` | `N=56` | `N=58` | `~56` |
| current lanes=1 quick same-80 throughput-only rerun | `N=40/44` | `N=48` (`lag95=594ms`) | `~44-48` |
| prototype lanes=2 strict sweep | `N=56/72/80` (`N=80 lag95=204ms`) | `N=96` throughput-only (`lag95=1353ms`) | `~80-95`, conservative reported knee `~80` |

The clean apples-to-existing-baseline comparison is `80 / 56 = 1.43x`, matching the feasibility estimate.
The same-session quick rerun made lanes=1 look lower (`~44-48`), so it should be treated as approximate/noisy
rather than a new canonical baseline. The useful conclusion is that the prototype realizes at least the
predicted `~1.4x` local end-to-end speedup while preserving strict byte identity through `N=80`.
