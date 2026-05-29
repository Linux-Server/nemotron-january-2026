# B3-FU-5 L4 Knee Sweep Result

Date: 2026-05-28/29 UTC  
Instance: `i-0e2e61f9081de0e35`, `g6.4xlarge`, us-west-2, public IP `52.25.181.236`  
Run policy: `B_max=4`, `W=0`, `L=0`, `--density-start-stagger-ms 10000`, `sessions_per_worker=2`, `CUDA_MODULE_LOADING=EAGER`, admission caps `10000/10000`, events counted-not-gated.

## Verdict

Bounded last-pass L4 point: **N=24**. The adaptive sweep ran low-to-high through `N in {4,8,12,16,20,24}` and hit no SLO fail and no CUDA OOM before the contractual top of sweep. N=24 passed with TTFS p95/p99 `37.476 / 50.467 ms`, lag p95 `-110.329 ms`, zero token divergences, zero errors, and peak GPU memory `21.855 GiB`.

This means the actual SLO knee was **not found inside the bounded range**; the measured result is a lower bound of at least 24 streams. The likely next constraint is memory capacity/headroom rather than CPU: N=24 reached GPU util p95 `100%`, mean GPU util `75.4%`, dispatcher CPU `74.2%`, and only about `0.18 GiB` remained against CUDA-reported total memory in telemetry.

Economics at the bounded last-pass point: `g6.4xlarge` L4 at `$0.80/hr / 24 = $0.033/stream/hr`, versus L40S at `$3.00/hr / 64 = $0.047/stream/hr`. On this measurement L4 is about **29% cheaper per stream** than the L40S N=64 operating point, with the caveat that N=24 is top-censored rather than a failed-knee bracket.

## Artifacts

- Run logs: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/b3_fu5_l4_logs/`
- JSONL telemetry: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/b3_fu5_l4_jsonl/`

## Build And Correctness

The L4 reported `NVIDIA L4`, driver `580.159.04`, compute capability `8.9`, and the sm_89 AOTI path loaded on-device. The C++ runtime linked only `libcudart.so.12`.

The B=1 batched steady parity smoke passed tensor-exact in each measured cell (`max_enc_out/cache_ch/cache_t = 0`, no length mismatches). All sweep cells had `token_divergences=0` and `errors=0`. Strict event mismatches appeared at N=16+ but were counted-not-gated under the task policy; the summary still reported `correctness_at_knee=true`.

Operational note: to keep the L4 run bounded after the full artifact download, the measured run used a targeted drop=2 finalize bucket subset `T=43..58` with a 16-bucket manifest. The runtime manifest verification and serial oracle passed for every measured N.

## Sweep Table

| N | SLO | ttfs p50/p95/p99 ms | lag p50/p95/p99 ms | peak GiB | tok/event/err | admitted/offered | disp CPU % | q p95 | B1/B2/B4 | K2/K3/K4/backlog>B |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | PASS | 17.056 / 23.333 / 24.348 | -144.114 / -142.828 / -132.920 | 18.375 | 0 / 0 / 0 | 20 / 20 | 32.8 | 3 | 1565 / 0 / 261 | 154 / 107 / 0 / 1 |
| 8 | PASS | 18.555 / 31.874 / 33.513 | -143.405 / -132.021 / -128.834 | 19.074 | 0 / 0 / 0 | 20 / 20 | 42.6 | 4 | 1165 / 0 / 340 | 111 / 105 / 124 / 3 |
| 12 | PASS | 22.255 / 35.089 / 38.816 | -142.001 / -129.006 / -123.420 | 19.766 | 0 / 0 / 0 | 24 / 24 | 57.3 | 8 | 1067 / 0 / 536 | 258 / 73 / 205 / 159 |
| 16 | PASS | 22.682 / 47.955 / 54.923 | -134.586 / -122.305 / -113.929 | 20.443 | 0 / 1 / 0 | 32 / 32 | 65.5 | 11 | 1010 / 0 / 860 | 346 / 158 / 356 / 261 |
| 20 | PASS | 26.047 / 48.883 / 51.937 | -130.996 / -116.021 / -105.671 | 21.160 | 0 / 2 / 0 | 40 / 40 | 69.5 | 14 | 944 / 0 / 1205 | 418 / 197 / 590 / 535 |
| 24 | PASS | 27.043 / 37.476 / 50.467 | -127.581 / -110.329 / -94.988 | 21.855 | 0 / 2 / 0 | 48 / 48 | 74.2 | 20 | 633 / 0 / 1449 | 366 / 327 / 756 / 588 |

STOP behavior: no SLO fail, no OOM, and no `peak >= 23.0 GiB` trigger occurred. The sweep stopped because N=24 was reached and passed.

## Roofline Validation

B3-FU-2 projected L4 at about N=22 from the L40S N=64 point and the L4:L40S bandwidth ratio. This run passed N=24, so the projection is directionally validated and slightly conservative for the bounded range.

The result does not prove the true failure knee because the sweep stopped at the requested upper bound. SLO headroom at N=24 was still large: TTFS p95 was `37.476 ms` against `175 ms`, TTFS p99 was `50.467 ms` against `250 ms`, and lag p95 was still negative. CPU was not binding: dispatcher CPU was `74.2%` of one dispatcher thread's accounting and process CPU was about `0.98` cores on a 16-vCPU instance. The remaining risk is memory/capacity headroom plus GPU bandwidth/utilization, not host CPU.

## Memory Analysis

Peak memory by N:

| N | Peak GiB |
|---:|---:|
| 4 | 18.375 |
| 8 | 19.074 |
| 12 | 19.766 |
| 16 | 20.443 |
| 20 | 21.160 |
| 24 | 21.855 |

Linear fit over all six cells: **`peak_GiB = 17.679 + 0.1738 * N`**. Adjacent slopes were stable: `0.175`, `0.173`, `0.169`, `0.179`, `0.174 GiB/stream`.

The measured L4 per-stream slope essentially matches the L40S FU-1 slope of `0.173 GiB/stream`. The difference is the intercept: this L4 targeted-bucket run fit to about `17.68 GiB`, not the L40S OFF baseline of about `21.9 GiB`. That lower intercept is why the projected OOM at N=20-22 did not occur here. At N=24, telemetry reported `peak_gpu_mem_bytes=23467130880` and `total_gpu_mem_bytes=23659151360`, leaving only about `0.18 GiB` by CUDA telemetry, so memory remains the practical risk just above the bounded range.

## Funding And Deployment Math

Using the bounded last-pass point:

- L4 `g6.4xlarge`: `$0.80/hr / 24 = $0.033/stream/hr`.
- L40S comparator: `$3.00/hr / 64 = $0.047/stream/hr`.
- L4 advantage: about `0.71x` the L40S per-stream hourly cost, or about **29% lower**.

Deployment implication: L4 is economically attractive if the service can operate at or near N=24 and tolerate the smaller memory headroom. The observed slope says the memory model from FU-1 transfers well; the old OOM projection was too pessimistic because the L4 run's intercept was much lower than assumed. A production cap should still leave margin below the measured N=24 top unless the full finalize-bucket universe and production traffic mix reproduce the same low intercept.

## EC2 Cleanup And Cost

Artifacts were copied locally before cleanup. AWS SSO expired after the run; two device-code refresh attempts expired without approval, so I could not issue `terminate-instances` or confirm AWS `state=terminated` from the API in this session.

Cost-control action taken: in-guest `sudo shutdown -h now` was issued after artifact copy, and SSH subsequently timed out, consistent with the instance shutting down. Because `ec2_up.py`/the launch path does not set `InstanceInitiatedShutdownBehavior=terminate`, this should be treated as **AWS termination pending**, not a confirmed termination.

- Launch: `2026-05-28T21:51:12+00:00`
- Guest shutdown requested: approximately `2026-05-29T00:14Z`
- Elapsed to guest shutdown: approximately `2.38 h`
- Estimated active compute cost to guest shutdown at `$0.80/hr`: approximately **$1.90**
- Required final action once AWS SSO is available: `aws ec2 terminate-instances --region us-west-2 --instance-ids i-0e2e61f9081de0e35`, then `aws ec2 wait instance-terminated`.

State file `ec2-bench/.instance_b3_fu5_l4.json` is marked as guest-shutdown requested / AWS termination pending, not AWS-confirmed terminated.
