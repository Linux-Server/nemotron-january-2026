# B3 5090 Focused Re-run Result

Date: 2026-05-28
Host: local RTX 5090, sm_120, 31.32 GiB reported GPU memory
Build: `runtime/cpp/build_b2/density_main`, torch 2.8.0+cu128, CUDA module loading EAGER
Baseline artifacts: `runtime/artifacts/b3_5090_logs/20260528T124526Z/`
Focused re-run artifacts: `runtime/artifacts/b3_5090_logs/20260528T131726Z/`
Summary JSONL: `runtime/artifacts/b3_5090_logs/20260528T131726Z/summary.jsonl`

## Corrected Verdict

The original `mismatches=1` STOP was the documented interim-event-timing drift class at
`utt198/chunk13/event[6]`, not a new correctness bug. Under the project policy, this is counted as an event
divergence and not used as the binding correctness gate. The binding gate remains token correctness: both
focused re-run cells had `token_divergences=0` and `errors=0`.

At N=40, both real batching cells are SLO-robust under that policy:

| Cell | N | Policy result | Token divergences | Event divergences | Notes |
|---|---:|---|---:|---:|---|
| ON `B_max=2`, W10/L0, tolerant | 40 | PASS | 0 | 1 | Same `utt198/chunk13/event[6]` interim drift, counted-not-gated |
| ON `B_max=4`, W10/L0, tolerant | 40 | PASS | 0 | 1 | Same documented drift class, counted-not-gated |

The 5090 lift evidence is constrained to N=40. The scheduler-ON cells peak at about 31.27-31.28 GiB of
31.32 GiB, so this box is memory-capped for this harness and should not be used to claim higher-N lift. The
higher-N lift demonstration is the L40S sweep's job.

## Per-cell Results

| Cell | SLO-robust | ttfs p50/p95/p99 (ms) | lag p50/p95/p99 (ms) | Peak GPU mem | Raw binary rc/mismatches |
|---|---|---:|---:|---:|---:|
| ON `B_max=2`, W10/L0 | PASS | 8.552 / 12.066 / 13.203 | -149.261 / -141.442 / -138.109 | 31.284 GiB | rc=1 / raw `mismatches=1` event-only |
| ON `B_max=4`, W10/L0 | PASS | 8.337 / 11.934 / 14.138 | -147.866 / -135.941 / -132.350 | 31.268 GiB | rc=1 / raw `mismatches=1` event-only |

SLO-robust policy: `lag_p95 < 500ms`, `ttfs_p95 <= 175ms`, `ttfs_p99 <= 250ms`, `errors <= 1%`,
and zero token divergence. The binary still returns rc=1 because its strict serial-oracle event mismatch is
wired into raw `mismatches`; the sidecar JSONL splits that into `token_divergences` and
`event_divergences`.

## Lift Table vs OFF Baseline

OFF baseline reused from `runtime/artifacts/b3_5090_logs/20260528T124526Z/smoke_off_N40.jsonl`:
`ttfs_p95=33.014ms`, `lag_p95=-129.149ms`, peak memory `19.737 GiB`, SLO-robust PASS.

| Cell | N=40 result vs OFF | ttfs p95 delta | lag p95 delta | Peak mem delta |
|---|---|---:|---:|---:|
| OFF B=1 baseline | PASS | 0.000 ms | 0.000 ms | 0.000 GiB |
| ON `B_max=2`, W10/L0 | PASS/PASS at N=40 | -20.948 ms | -12.293 ms | +11.547 GiB |
| ON `B_max=4`, W10/L0 | PASS/PASS at N=40 | -21.081 ms | -6.792 ms | +11.532 GiB |

No positive knee delta is credited on the 5090 because N>40 was intentionally not run under the memory cap.

## F2-T Scheduler Telemetry

| Cell | Dispatcher CPU | Stream util | Queue depth p95 | Fairness spread p95 | Batch fill counts |
|---|---:|---:|---:|---:|---|
| ON `B_max=2`, W10/L0 | 69.305% | 68.677% | 4 | 5.019 ms | `B1=6869`, `B2=6029`, `B4=0`, `backlog_gt_bmax=2814`, `dispatch_cycles=12898` |
| ON `B_max=4`, W10/L0 | 52.328% | 51.560% | 4 | 12.440 ms | `B1=5171`, `B2=248`, `B4=3522`, `K3_padded_to_B4=828`, `K4=2694`, `backlog_gt_bmax=211`, `dispatch_cycles=8941` |

Both cells had `dispatcher_exceptions=0`, `completed=18927`, and `enqueued=18927`.

## Artifacts

| Cell | JSONL | Stdout log |
|---|---|---|
| ON `B_max=2`, W10/L0 | `runtime/artifacts/b3_5090_logs/20260528T131726Z/on_B2_W10_L0_N40_TOL.jsonl` | `runtime/artifacts/b3_5090_logs/20260528T131726Z/on_B2_W10_L0_N40_TOL.stdout.log` |
| ON `B_max=4`, W10/L0 | `runtime/artifacts/b3_5090_logs/20260528T131726Z/on_B4_W10_L0_N40_TOL.jsonl` | `runtime/artifacts/b3_5090_logs/20260528T131726Z/on_B4_W10_L0_N40_TOL.stdout.log` |

