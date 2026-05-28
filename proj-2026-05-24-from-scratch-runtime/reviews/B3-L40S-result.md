# B3 L40S Result

Date: 2026-05-28

## Verdict

Fresh `g6e.8xlarge` L40S `i-09af5053d4200e799` (`44.254.81.253`) was provisioned, used for native sm_89 builds and B3 measurements, and terminated.

Under the cost-bounded scheduler sweep (`--density-sessions-per-worker 2`), the best staggered policy measured was `B_max=4, window_ms=0, lone_timeout_ms=0`, which passed through the top registered N point:

| Basis | OFF knee | Best ON knee | Lift |
|---|---:|---:|---:|
| Full-session OFF baseline observed before abort | 36 | not run full-session | n/a |
| Paired bounded sweep, sessions/worker=2 | 40 | >=64 | >=1.60x |

Important caveat: the `B=4` staggered rows have zero token divergences and zero runtime errors, but event-only divergences/mismatches appear from `N=40` onward. The harness marked them SLO-robust because F2-T ran with event divergences counted-not-gated. If the stricter "0 event mismatch" interpretation is enforced, the clean `B=4` measured point is only `N=36`; `B=2` stays clean through `N=40`.

The required burst variant at `N=64` failed keep-up: lag p95 `1355.181 ms`, with `enc_first_lock_p95=1606.801 ms`, despite zero token divergences.

## EC2

| Field | Value |
|---|---|
| Instance | `i-09af5053d4200e799` |
| Type | `g6e.8xlarge` |
| Region/profile | `us-west-2` / `AWSAdministratorAccess-419599258555` |
| Launch | `2026-05-28T12:17:10Z` |
| Terminated | `2026-05-28T18:05:54Z` |
| Elapsed | `5.812 h` |
| Cost estimate | `$14.53-$23.25` using the task's `$2.50-$4.00/hr` range |

## sm_89 B Buckets

Native compile used torch `2.8.0+cu128`, CUDA arch `8.9`, autotune off.

| B | Package | Bytes | SHA256 |
|---:|---|---:|---|
| 1 | `enc_steady_aoti_b1.pt2` | 2481063325 | `98a0f5b5f46f15fdb6046d18ce4c00364857644c0c956e628de08d5a45170cab` |
| 2 | `enc_steady_aoti_b2.pt2` | 2481238447 | `24a35a246c47aacc3d0adc7b6cc0a2502741fbc81a30773e35de6fea6b4a1e93` |
| 4 | `enc_steady_aoti_b4.pt2` | 2481340015 | `27d2724bb61b806538700d8402b6ce23f074d97b7b8c51360048f757cd8d724b` |

Shared weights SHA: `b96ba1d54701f43bd8c319685fb591bdec88a7d3affb86a40b3608a2ecae9819`.

A1 parity outcome: `B` (`production_sha=1765f8214eef8684c00ac78eb5a543e9e19530be9f36ff61965a6e87c6f211d0`, new B1 SHA above, package SHAs differ, tensor parity allclose true).

## F1 b2-t1

F1 PASS was obtained with the pre-F2-T B2 binary and `--batch-lone-timeout-ms 5`, because the F2-T binary and pre-F2T `lone_timeout=0` both deadlocked in forced-concurrency cases. Rows reference was full-corpus `1000`; the current harness still only exercises 2/3/4/4/4 rows in the forced/stagger/control cases.

| Case | Rows | Result | Token div | Event div | Max enc out | Max cache ch | Max cache t |
|---|---:|---|---:|---:|---:|---:|---:|
| `single_stream_scheduler_on` | 1000 | PASS | 0 | 0 | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `multi_stream_forced_K2_B2` | 2 | PASS | 0 | 0 | `6.596e-05` | `7.793e-05` | `1.041e-02` |
| `multi_stream_forced_K3_padded_B4` | 3 | PASS | 0 | 0 | `3.268e-04` | `6.287e-03` | `3.196e-01` |
| `multi_stream_forced_concurrency_B4` | 4 | PASS | 0 | 0 | `3.268e-04` | `6.287e-03` | `3.196e-01` |
| `multi_stream_staggered` | 4 | PASS | 0 | 0 | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `scheduler_on_Bmax1_control` | 4 | PASS | 0 | 0 | `0.000e+00` | `0.000e+00` | `0.000e+00` |

Overall: `B2_T1_RESULT PASS rows_reference=1000 cases=6 token_divergences=0 event_divergences=0 errors=0`.

## Sweep Table

Full-session OFF baseline:

| Cell | N36 | N40 | N44 |
|---|---|---|---|
| OFF B1, full-session | PASS | FAIL | partial |

Cost-bounded paired sweep (`sessions_per_worker=2`):

| Cell | N36 | N40 | N44 | N48 | N56 | N64 | Knee |
|---|---|---|---|---|---|---|---:|
| OFF B1 | PASS | PASS | FAIL | FAIL | FAIL | FAIL | 40 |
| ON Bmax=1, w10/l0 | FAIL | FAIL | partial | skipped | skipped | skipped | 0 |
| ON Bmax=2, w10/l0 | PASS | PASS | FAIL | skipped | skipped | skipped | 40 |
| ON Bmax=4, w0/l0 | PASS | PASS* | PASS* | PASS* | PASS* | PASS* | >=64 |
| ON Bmax=4, w4/l0 | PASS* | partial | skipped | skipped | skipped | skipped | not bracketed |
| Burst ON Bmax=4, w0/l0, N64, no stagger | n/a | n/a | n/a | n/a | n/a | FAIL* | 0 |

`*` means zero token divergences but nonzero event-only divergences/mismatches under counted-not-gated policy.

Key best staggered row, `Bmax=4,w0,l0,N64`: throughput_rt `33.823`, ttfs p95/p99 `21.087/27.237 ms`, lag p95 `-92.892 ms`, steady GPU p95 `11.056 ms`, GPU util mean `55.4%`, peak mem `33.641 GiB`, token/event/mismatch/errors `0/2/2/0`.

Burst row, `Bmax=4,w0,l0,N64,stagger=0`: throughput_rt `43.138`, ttfs p95/p99 `18.626/23.895 ms`, lag p95 `1355.181 ms`, steady GPU p95 `11.229 ms`, GPU util mean `75.5%`, peak mem `33.826 GiB`, token/event/mismatch/errors `0/3/3/0`.

## F2-T Telemetry

Best staggered `Bmax=4,w0,l0,N64`:

| Metric | Value |
|---|---:|
| Batch counts | `B1=371`, `B2=159`, `B4=1752`, `K4=1631`, `backlog_gt_bmax=1510` |
| Dispatcher CPU | `60.960%` |
| Dispatcher stream util | `60.749%` |
| Queue depth p50/p95/p99 | `7/22/26` |
| Fairness spread p50/p95/p99 | `3765.890/9904.550/14691.700 us` |
| Gather wait p50/p95/p99 | `19344.000/53824.400/63213.800 us` |
| Service wait p50/p95/p99 | `479.297/748.412/995.185 us` |
| CUDA run p50/p95/p99 | `9034.750/11056.100/12809.200 us` |
| Output sync p50/p95/p99 | `6.144/46.080/62.464 us` |

Burst `Bmax=4,w0,l0,N64`:

| Metric | Value |
|---|---:|
| Batch counts | `B1=584`, `B2=226`, `B4=1666`, `K4=1542`, `backlog_gt_bmax=1468` |
| Dispatcher CPU | `82.655%` |
| Dispatcher stream util | `82.359%` |
| Queue depth p50/p95/p99 | `43/60/60` |
| Fairness spread p50/p95/p99 | `3273.500/12302.200/18512.800 us` |
| Gather wait p50/p95/p99 | `129869.000/154035.000/405006.000 us` |
| `enc_first` lock p95 | `1606.801 ms` |

## F2-M Memory

Peak memory for the best measured B4 policy compared to the bounded OFF baseline:

| N | OFF GiB | B4 w0/l0 GiB | Delta GiB |
|---:|---:|---:|---:|
| 36 | 17.067 | 28.819 | 11.752 |
| 40 | 17.774 | 29.539 | 11.765 |
| 44 | 18.487 | 30.213 | 11.726 |
| 48 | 19.184 | 30.893 | 11.709 |
| 56 | 20.522 | 32.289 | 11.767 |
| 64 | 21.928 | 33.641 | 11.713 |

## Artifacts

- Logs: `runtime/artifacts/b3_l40s_logs/`
- F2-T JSONL: `runtime/artifacts/b3_l40s_jsonl/`
- Setup, F1, compile logs, and final sm_89 manifest: `runtime/artifacts/b3_l40s_setup/`
- EC2 state: `ec2-bench/.instance_b3_l40s.json`

## Findings

1. Batched steady on L40S does lift the staggered bounded knee to the top measured axis (`>=64`) under counted-not-gated event policy.
2. Synchronized burst arrival still fails badly at N=64; HOL is now visible as queue/gather wait plus `enc_first` lock tail.
3. `Bmax=1` scheduler-on is a useful negative control and is not viable: it fails keep-up even at N=36.
4. Strict zero-event-mismatch policy is not cleared by the high-N B4 rows. Token correctness is clean, but event drift must be either explicitly accepted as counted-not-gated or fixed before using this as a strict 0-mismatch gate.
5. The full window/lone grid was stopped after `B4,w0,l0` reached `N=64` and `B4,w4,l0,N36` already showed event-only mismatches, to stay near the EC2 budget. This report therefore contains a knee result, not an exhaustive window/lone optimization table.
