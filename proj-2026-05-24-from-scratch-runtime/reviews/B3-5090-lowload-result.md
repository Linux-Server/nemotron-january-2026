# B3 5090 Low-load Sweep Result

Date: 2026-05-28
Host: local RTX 5090, 31.32 GiB reported GPU memory
Build: `runtime/cpp/build_b2/density_main`, commit `5ef23af`
Artifacts: `runtime/artifacts/b3_5090_logs/20260528T150035Z/`
Summary JSONL: `runtime/artifacts/b3_5090_logs/20260528T150035Z/summary.jsonl`

`DENSITY_GOLD_EVENTS_TOLERANT` was not set. The telemetry confirms `event_divergences_gated=false` for all
cells, so event divergences are counted but not gated under the post-Fix-8 default policy.

All six cells completed without timeout. Each process returned `rc=1` because the single-N density-sweep
summary still reports `pass_to_1b=false`; the per-row telemetry is the binding result here, and every row is
`slo_robust=true`.

## Verdict

The "N=1 not penalized" hypothesis is **SUPPORTED**.

At N=1, scheduler ON has `ttfs_p95=8.296 ms` vs OFF `8.008 ms`, a `+0.287 ms` delta. That is far inside the
~5 ms support band and nowhere near the ~10 ms refutation threshold. Lag p95 is also effectively tied
(`+0.309 ms`, ON - OFF), with both paths far ahead of real time.

TTFS p95 break-even by linear interpolation is approximately **N=4.17** between N=4 (`+0.038 ms`) and N=8
(`-0.864 ms`). Lag p95 does not cross to ON benefit in the measured range; ON remains slightly less ahead
than OFF by less than 1 ms through N=8. Practically, this sweep says wrapper overhead is negligible through
low load, and first clear TTFS benefit appears by N=8.

## Per-cell Results

SLO-robust means `lag_p95 < 500 ms`, `ttfs_p95 <= 175 ms`, `ttfs_p99 <= 250 ms`, `errors <= 1%`, and
`token_divergences=0`.

| Cell | N | Scheduler | SLO | Token/event divergences | ttfs p50/p95/p99 (ms) | lag p50/p95/p99 (ms) | Peak mem |
|---|---:|---|---|---:|---:|---:|---:|
| `off_N1` | 1 | OFF | PASS | 0 / 0 | 7.432 / 8.008 / 8.015 | -153.816 / -153.181 / -150.789 | 12.052 GiB |
| `on_N1_B4_W10_L0` | 1 | ON | PASS | 0 / 0 | 7.608 / 8.296 / 9.355 | -153.603 / -152.872 / -150.623 | 23.859 GiB |
| `off_N4` | 4 | OFF | PASS | 0 / 0 | 7.425 / 9.202 / 10.038 | -154.070 / -153.329 / -150.888 | 12.654 GiB |
| `on_N4_B4_W10_L0` | 4 | ON | PASS | 0 / 0 | 7.285 / 9.241 / 9.957 | -154.026 / -153.116 / -150.848 | 24.476 GiB |
| `off_N8` | 8 | OFF | PASS | 0 / 0 | 7.269 / 9.871 / 10.842 | -154.014 / -151.518 / -149.869 | 13.438 GiB |
| `on_N8_B4_W10_L0` | 8 | ON | PASS | 0 / 0 | 7.299 / 9.007 / 11.971 | -153.915 / -150.778 / -142.743 | 25.246 GiB |

## ON - OFF Lift/Penalty

Positive ttfs/lag deltas are penalties. Negative ttfs/lag deltas are improvements. For lag, all values are
comfortably under the keep-up budget; the sub-ms positive deltas mean ON is slightly less ahead of schedule.

| N | ttfs p95 delta | lag p95 delta | Peak mem delta |
|---:|---:|---:|---:|
| 1 | +0.287 ms | +0.309 ms | +11.807 GiB |
| 4 | +0.038 ms | +0.213 ms | +11.823 GiB |
| 8 | -0.864 ms | +0.740 ms | +11.808 GiB |

## Scheduler Telemetry

| Cell | gather_wait p50/p95/p99 (ms) | service_wait p50/p95/p99 (ms) | output_sync p50/p95/p99 (ms) | worker_blocked p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|
| `on_N1_B4_W10_L0` | 0.012 / 0.021 / 0.024 | 0.095 / 0.116 / 0.126 | 0.000 / 0.002 / 0.003 | 5.577 / 5.694 / 5.785 |
| `on_N4_B4_W10_L0` | 0.009 / 0.018 / 0.021 | 0.050 / 0.096 / 0.108 | 0.000 / 0.001 / 0.003 | 5.340 / 5.583 / 5.685 |
| `on_N8_B4_W10_L0` | 0.011 / 2.925 / 5.332 | 0.042 / 0.084 / 0.148 | 0.000 / 0.002 / 0.004 | 5.363 / 8.359 / 15.204 |

| Cell | Dispatcher CPU | Stream util | Queue depth p50/p95/p99 | Fairness spread p50/p95/p99 (ms) | Bucket fill counts |
|---|---:|---:|---:|---:|---|
| `on_N1_B4_W10_L0` | 3.532% | 3.445% | 1 / 1 / 1 | 0.000 / 0.000 / 0.000 | `B1=1144`, `B2=0`, `B4=0`, `backlog_gt_bmax=0`, `dispatch_cycles=1144` |
| `on_N4_B4_W10_L0` | 10.176% | 9.989% | 1 / 1 / 1 | 0.000 / 0.000 / 0.000 | `B1=1792`, `B2=0`, `B4=0`, `backlog_gt_bmax=0`, `dispatch_cycles=1792` |
| `on_N8_B4_W10_L0` | 22.117% | 21.802% | 1 / 1 / 1 | 0.000 / 0.000 / 0.000 | `B1=3806`, `B2=8`, `B4=6`, `K4=6`, `backlog_gt_bmax=4`, `dispatch_cycles=3820` |

The fill counts explain the result shape: with 10s worker stagger and `lone_timeout_ms=0`, N=1 and N=4 are
pure B1 dispatches. At N=8 only 14 of 3820 dispatch cycles use B2/B4, so this is mostly measuring wrapper
overhead rather than sustained batching.

## Artifacts

| Cell | JSONL | Stdout log |
|---|---|---|
| `off_N1` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/off_N1.jsonl` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/off_N1.stdout.log` |
| `on_N1_B4_W10_L0` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/on_N1_B4_W10_L0.jsonl` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/on_N1_B4_W10_L0.stdout.log` |
| `off_N4` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/off_N4.jsonl` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/off_N4.stdout.log` |
| `on_N4_B4_W10_L0` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/on_N4_B4_W10_L0.jsonl` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/on_N4_B4_W10_L0.stdout.log` |
| `off_N8` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/off_N8.jsonl` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/off_N8.stdout.log` |
| `on_N8_B4_W10_L0` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/on_N8_B4_W10_L0.jsonl` | `runtime/artifacts/b3_5090_logs/20260528T150035Z/on_N8_B4_W10_L0.stdout.log` |
