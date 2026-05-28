# B3-FU-2 Roofline Profiling Result

Date: 2026-05-28

## Summary

FU-2 ran on a separate `g6e.8xlarge` L40S instance, `i-09641e4d0167eed3e` / `34.211.39.91`, independent of the in-flight FU-1 instance. The instance is terminated.

The cleanest low-level result is Cell 3: isolated B=4 `ampere_sgemm_64x32_sliced1x4_tn` remains mostly memory-bound on the dominant GEMM shapes. Median DRAM throughput is 70.0%, median SM throughput is 34.4%, median achieved occupancy is 16.7% across the first 10 matching launches, and median estimated arithmetic intensity is 30.7 FLOP/byte versus an L40S machine balance of about 104-106 FLOP/byte. One B=4 grid shape reaches 29.6% achieved occupancy, but the dominant DRAM-heavy shapes still look like the prior B=1 profile.

The macro mechanism still delivered: the successful N=64 completion at the batched knee reproduced the production row with `throughput_rt=33.824`, TTFS p95/p99 `20.160/26.046 ms`, steady GPU p50/p95 `9.071/10.869 ms`, mean GPU util `55.1%`, and dispatcher stream util `61.3%`. The lift is therefore mostly from batching/amortizing per-row work and keeping the dispatcher stream busy, not from moving the dominant SGEMM kernel into a compute-bound regime.

## Artifacts

Text/CSV/JSON artifacts are under `runtime/artifacts/b3_fu2_profile_logs/`.

Key files:

- `cell1_n64_hiadmin_after_tmpcleanup_complete.out`
- `cell1_n64_after_tmpcleanup_density_summary.json`
- `cell1_n64_gate_delay430_nsys_stats.txt`
- `cell2_n88_hiadmin_complete.out`
- `ncu_sgemm_b4_skip966_lc40_details.csv`
- `ncu_sgemm_b4_skip966_lc40_raw.csv`
- `ncu_sgemm_b4_occupancy_skip966_lc10_details.csv`
- `ncu_sgemm_b4_summary.json`
- `ncu_sgemm_b4_summary.csv`

Large `.nsys-rep`, `.ncu-rep`, and SQLite files were not pulled into the repo per the task's binary-artifact safety note.

## Build And Smoke

Build/provisioning succeeded on L40S:

- Torch: `2.8.0+cu128`
- CUDA root: `/usr/local/cuda-13.0`
- `nsys`: `2024.6.2`
- `ncu`: `2025.2.1`
- `density_main` and `steady_batch_bench` built in `cpp/build_l40s_density`
- B packages loaded from S3 and verified:
  - B1 `457b27f4f7acac54afe6c63b2524759c5680515cf79aa167a3d000c230125eb1`
  - B2 `f9a4c9efe98169b072622bfe044974ac2b170130ef69b3919cca21d6baae1723`
  - B4 `b42323a87603d0c5c09cd00b04db55e583a94fab3a3cb6df40289e3482af1f4c`

Tooling smoke passed:

- `nsys` smoke generated stats from `steady_batch_bench`.
- `ncu` smoke profiled a single kernel with `sudo env ...`.
- `steady_batch_bench` smoke passed correctness and showed B4 per-row p50 around `2.16 ms` versus B1 `8.88 ms`.

## Cell 1: N=64 Nsys And Gate Telemetry

The first N=64 nsys captures landed in setup because the original `--delay` window was too early on this instance. The root cause of later AOTI loader failures was a full `/tmp` from repeated AOTI extraction directories; after cleaning `/tmp`, the N=64 completion passed.

Successful N=64 completion, same B=4 policy:

| Metric | Value |
|---|---:|
| SLO robust | true |
| Throughput realtime streams | 33.824 |
| TTFS p50 / p95 / p99 | 13.630 / 20.160 / 26.046 ms |
| Lag p50 / p95 / p99 | -127.703 / -94.040 / -84.461 ms |
| Steady GPU p50 / p95 / p99 | 9.071 / 10.869 / 12.484 ms |
| Finalize total p50 / p95 / p99 | 13.606 / 20.108 / 25.933 ms |
| GPU util mean / p95 | 55.1% / 90% |
| Peak GPU memory | 28.985 GiB |
| Token / event divergences | 0 / 2 |
| Errors | 0 |

Scheduler telemetry:

| Metric | Value |
|---|---:|
| Enqueued / completed | 14822 / 14822 |
| Dispatch cycles | 5503 |
| B1 / B4 dispatches | 2121 / 3382 |
| K4 dispatches | 2873 |
| K2/K3 padded to B4 | 318 / 191 |
| Backlog > Bmax | 2695 |
| Dispatcher CPU | 61.2% |
| Dispatcher stream util | 61.3% |
| Queue depth p50 / p95 / p99 | 4 / 53 / 59 |
| Gather wait p50 / p95 / p99 | 24.8 / 140.9 / 154.5 ms |
| Service wait p50 / p95 / p99 | 469 / 748 / 958 us |
| CUDA run p50 / p95 / p99 | 9.06 / 10.76 / 14.64 ms |
| Output sync p50 / p95 / p99 | 5.1 / 44.0 / 62.5 us |

The late recapture `cell1_n64_gate_delay430` still began during finalize-bucket preload, before warmup and measured gate. Its nsys stats are retained, but are not used as measured-gate evidence.

## Cell 2: N=88 Nsys

N=88 was attempted with the same policy. The completion run failed before warmup/measurement:

`create_func_( &container_handle_, num_models, device_str.c_str(), cubin_dir.empty() ? nullptr : cubin_dir.c_str()) API call failed at /pytorch/torch/csrc/inductor/aoti_runner/model_container_runner.cpp, line 122`

The failure occurred with peak GPU memory about `9.19 GiB`, so it was not GPU OOM. Based on the later N=64 recovery, the likely proximate cause was the full `/tmp` AOTI extraction state. I did not spend another full N=88 recapture after N=64 consumed the remaining profiling budget; Cell 2 remains an attempted/failed cell with raw stdout and setup nsys stats retained.

## Cell 3: NCU B=4 SGEMM Roofline

Target: `steady_batch_bench --warmup 0 --iters 1`, using an nsys order trace to choose `--launch-skip 966`, then `ncu --set roofline --launch-count 40` on `ampere_sgemm_64x32_sliced1x4_tn`. A second `--section Occupancy --launch-count 10` pass filled the achieved occupancy metrics.

Overall B=4 SGEMM summary:

| Metric | Value |
|---|---:|
| Profiled launches | 40 |
| Duration p50 / p95 | 34.91 / 47.90 us |
| DRAM throughput p50 / p95 | 70.0% / 72.8% |
| SM throughput p50 / p95 | 34.4% / 37.0% |
| FP32 peak p50 / p95 | 20% / 21% |
| Estimated FP32 p50 / p95 | 18.32 / 19.24 TFLOP/s |
| Estimated DRAM BW p50 / p95 | 604.8 / 629.0 GB/s |
| Estimated AI p50 / p95 | 30.7 / 131.5 FLOP/byte |
| Achieved occupancy p50 / max | 16.7% / 29.6% |
| Theoretical occupancy | 33.3% |

By grid shape:

| Grid | Launches | DRAM p50 | SM p50 | Occupancy p50 | Est. AI p50 |
|---|---:|---:|---:|---:|---:|
| `(16, 1, 8)` | 16 | 72.5% | 34.3% | 16.7% | 30.6 FLOP/B |
| `(64, 1, 4)` | 16 | 69.7% | 36.8% | 29.6% | 30.4 FLOP/B |
| `(16, 5, 1)` | 8 | 14.5% | 28.7% | 16.7% | 131.4 FLOP/B |

The two dominant DRAM-heavy shapes remain clearly below machine balance. The low-DRAM `(16, 5, 1)` shape has higher estimated AI but also lower SM throughput and is only 8/40 launches.

## Roofline Comparison

Prior baseline from `reviews/profiling-paired-verdict.md`: single-stream B=1 `ampere_sgemm_64x32_sliced1x4_tn` was DRAM 71-72%, SM 34-39%, achieved occupancy 15-17%, duration about 34-36 us, memory-bandwidth-bound.

| Metric | Prior B=1 single stream | This FU-2 B=4 isolated SGEMM |
|---|---:|---:|
| DRAM throughput | 71-72% | p50 70.0%, p95 72.8% |
| SM throughput | 34-39% | p50 34.4%, p95 37.0% |
| Achieved occupancy | 15-17% | p50 16.7%, max 29.6% |
| Duration | 34-36 us | p50 34.9 us |
| FP32 peak | not exported in baseline | p50 20%, p95 21% |
| Estimated AI | not exported in baseline | p50 30.7 FLOP/B |
| Verdict | memory-bound | still mostly memory-bound |

This does not support the strongest version of the hypothesis that B=4 shifts the dominant SGEMM close to compute-bound. B=4 improves the end-to-end row economics, but the kernel roofline remains close to the prior B=1 shape on the dominant GEMMs.

## Interpretation

At the N=64 knee, the system is not sync-bound: output sync p95 is only `44 us`, finalize wait p95 is `0`, and TTFS p95 is `20 ms` against a 175 ms budget. It is also not compute-bound at the dominant SGEMM level: B=4 median AI is about `31 FLOP/byte`, well below L40S machine balance, and DRAM throughput stays around 70%.

The measured bottleneck is mixed:

- Kernel level: dominant SGEMMs remain memory-bandwidth-bound.
- Runtime level: the dispatcher is materially busy (`61%` CPU, `61%` stream util), with large gather waits and queue depth p95 `53`.
- End-to-end: still comfortably within SLO at N=64, with roughly 8x TTFS p95 headroom.

The remaining lift opportunity is therefore not "make SGEMM compute-bound" by simple batching alone. Better headroom would come from reducing memory traffic in the encoder kernels, improving/fusing the memory-bound GEMM path, reducing scheduler gather backlog, or adding more work per dispatch without increasing gather wait enough to harm TTFS.

## Cell 4

Skipped. The nsys Cells 1-2 required retries and the NCU roofline plus occupancy passes consumed the budget. Multi-stream NCU replay would have added substantial overhead and risked another long run.

## EC2

Instance `i-09641e4d0167eed3e` was launched at `2026-05-28T18:43:44+00:00` and terminated at `2026-05-28T21:22:48+00:00`. Elapsed time was `2.651 h`, estimated cost `$6.63-$10.60` using the task's `$2.50-$4.00/hr` range.

Termination was done with AWS CLI because `ec2_down.py` could not import `boto3` in the default Python environment. State file `ec2-bench/.instance_b3_fu2_profile_l40s.json` is updated with `state=terminated`.
