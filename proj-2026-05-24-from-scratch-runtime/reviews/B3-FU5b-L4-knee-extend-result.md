# B3-FU-5b L4 Knee Extension Result

Date: 2026-05-29 UTC  
Instance: `i-0b51f68cf9509bbc3`, `g6.4xlarge`, us-west-2, public IP `16.145.92.136`  
Run policy: `B_max=4`, `W=0`, `L=0`, `--density-start-stagger-ms 10000`, `sessions_per_worker=2`, `CUDA_MODULE_LOADING=EAGER`, admission caps `10000/10000`, events counted-not-gated.

## EC2 Termination

AWS CLI termination succeeded and `aws ec2 wait instance-terminated` returned successfully.

- AWS state: `terminated`
- Launch: `2026-05-29T01:04:15+00:00`
- Termination confirmed: `2026-05-29T02:52:35Z`
- Elapsed to confirmed termination: approximately `1.81 h`
- Estimated compute cost at `$0.80/hr`: approximately `$1.44`
- State file: `ec2-bench/.instance_b3_fu5b_l4.json`

No SSO-expiry fallback was needed.

## Verdict

The B3-FU-5b extension stopped at the first candidate, `N=25`. N=25 passed the latency and error SLOs (`ttfs_p95=47.677 ms`, `ttfs_p99=57.784 ms`, `lag_p95=-107.951 ms`, `errors=0`), but it exceeded the required L4 memory-safety line: `peak_gpu_mem_bytes=23637000192`, or `23.637 GB` decimal / `22.014 GiB` binary, against the task's practical `23.5 GB` safety stop and CUDA-reported total `23659151360` bytes.

Therefore the realized L4 last-pass knee remains **N=24** from B3-FU-5. The failure mode is memory capacity/safety margin, not SLO lag, TTFS, dispatcher CPU, or correctness errors. N=26, N=27, and N=28 were not run because the task required STOP on first OOM/memory-safety fail.

Economics at the realized knee: L4 `g6.4xlarge` at `$0.80/hr / 24 = $0.033/stream/hr`, versus L40S at `$3.00/hr / 64 = $0.047/stream/hr`. L4 remains about 29% cheaper per stream at the absolute knee, but production should cap at knee-minus-4, i.e. **N=20**, for memory margin (`$0.040/stream/hr`, about 15% cheaper than the L40S comparator).

## Artifacts

- Run logs: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/b3_fu5b_l4_logs/`
- JSONL telemetry: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/b3_fu5b_l4_jsonl/`
- Decision file: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/b3_fu5b_l4_logs/sweep_state.json`

## Build And Correctness

The run used one fresh `g6.4xlarge` L4. The guest reported `NVIDIA L4`, driver `580.159.04`, compute capability `8.9`, and the sm_89 AOTI path loaded on-device. The C++ runtime linked only `libcudart.so.12`.

As in B3-FU-5, the measured run used the targeted drop=2 finalize bucket subset `T=43..58` with a 16-bucket manifest. Runtime manifest verification and the serial oracle passed for N=25. The B=1 batched steady parity smoke passed tensor-exact. N=25 had `token_divergences=0`, `event_divergences=3`, `mismatches=3`, and `errors=0`; event divergences were counted-not-gated per policy.

## Sweep Table

| N | SLO | ttfs p50/p95/p99 ms | lag p50/p95/p99 ms | peak GiB | tok/event/err | admitted/offered | disp CPU % | q p95 | B1/B2/B4 | K2/K3/K4/backlog>B |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | PASS | 27.043 / 37.476 / 50.467 | -127.581 / -110.329 / -94.988 | 21.855 | 0 / 2 / 0 | 48 / 48 | 74.2 | 20 | 633 / 0 / 1449 | 366 / 327 / 756 / 588 |
| 25 | MEM FAIL | 25.685 / 47.677 / 57.784 | -127.133 / -107.951 / -95.527 | 22.014 | 0 / 3 / 0 | 50 / 50 | 69.3 | 21 | 589 / 0 / 1516 | 306 / 306 / 904 / 756 |

N=25 is marked `MEM FAIL` because latency SLOs passed but the contractual memory-safety stop triggered. The benchmark binary returned nonzero after `NO_PASS_TO_1B`, but the telemetry row was complete and parsed successfully.

## Memory Analysis

Peak memory at the knee boundary:

| N | Source | peak bytes | peak GB decimal | peak GiB binary | headroom vs CUDA total |
|---:|:---|---:|---:|---:|---:|
| 24 | B3-FU-5 baseline | 23467130880 | 23.467 | 21.855 | 0.192 GB |
| 25 | B3-FU-5b | 23637000192 | 23.637 | 22.014 | 0.022 GB |

The one-step N=24 to N=25 increment was `169869312` bytes, or `0.170 GB` decimal / `0.158 GiB` binary per stream. That is close to the B3-FU-5 all-cell L4 fit of `0.1738 GiB/stream` and the L40S FU-1 slope of `0.173 GiB/stream`. The practical difference is the tiny remaining L4 byte headroom at N=24: N=25 consumed the safety margin and landed only about `22 MB` below CUDA-reported total memory.

No CUDA OOM exception was logged at N=25, but the memory-safety stop is the binding result: `23.637 GB` exceeded the required `23.5 GB` safety line. Extrapolating the observed one-step slope puts N=26 near `23.807 GB` decimal, above the L4's CUDA-reported `23.659 GB` total, so N=26 would be expected to hit actual OOM or a harder allocation failure. Per task protocol, N=26+ were not attempted after the N=25 stop.

## Economics Update

Using the realized L4 knee:

- L4 absolute knee: `$0.80/hr / 24 = $0.033/stream/hr`.
- L4 production cap with knee-minus-4 margin: `$0.80/hr / 20 = $0.040/stream/hr`.
- L40S comparator: `$3.00/hr / 64 = $0.047/stream/hr`.
- L4 advantage at absolute knee: about `0.71x` L40S per-stream cost, or about 29% lower.
- L4 advantage at recommended production cap N=20: about `0.85x` L40S per-stream cost, or about 15% lower.

Recommendation: treat **N=24** as the lab absolute ceiling for this targeted-bucket L4 configuration and use **N=20** as the deployment cap unless a production-specific memory study proves equivalent headroom under the full finalize-bucket universe and real traffic mix.
