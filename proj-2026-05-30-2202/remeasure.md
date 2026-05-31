# Step 6 remeasure - cold boot after Steps 2-5 unify

**Date:** 2026-05-31 | **Binary:** `cpp/build_step10/ws_server` @ HEAD `ce0d2c6`.
**Raw:** `remeasure_runs/{summary.tsv,phases.txt,artifacts.txt,*.srvlog}`; runner `run_remeasure.sh`.

## Method

Deploy-mode only: `NEMOTRON_WS_BACKGROUND_WARMUP=1`, scheduler on, CAP/LANES=64, same env and launch args as `run_cold_baseline.sh`. Matrix:

- `encfirst_ts_{cold,warm}_on`: `NEMOTRON_WS_ENC_FIRST_TS=1` (default TS first encoder)
- `encfirst_aoti_{cold,warm}_on`: `NEMOTRON_WS_ENC_FIRST_TS=0` (AOTI first encoder borrowing shared constants)

COLD uses the same non-root method as Step 0: per-file `posix_fadvise(POSIX_FADV_DONTNEED)` over `artifacts/` + `steady_b_artifacts/`, not global `drop_caches`. WARM uses the same `cat artifacts/* steady_b_artifacts/* >/dev/null` page-cache fill. Current tree fadvise volume was `~41.22 GB` because it contains extra full bucket variants; the method and top-level artifact paths are unchanged from the baseline runner.

Each cell waited for `ws_server listening` plus `background_warm_complete`, sampled per-pid GPU MiB and host RSS, measured `/tmp` delta before shutdown, then sent `SIGINT` and reclaimed owned `/tmp/[A-Za-z0-9]{6}` AOTI extraction dirs.

## Headline

Baseline is Step 0 deploy-mode `cold_on` / `warm_on` from `baseline.md`.

| Cell | enc_first | cache | time-to-listening | background complete | peak GPU MiB | peak RSS MiB | `/tmp` delta MB | vs baseline listening | vs baseline GPU |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Step-0 `cold_on` | TS | cold | 25.2 s | 85.8 s | 16364 | 12275 | +2501 | - | - |
| Step-0 `warm_on` | TS | warm | 17.5 s | 78.1 s | 16364 | 12490 | +2501 | - | - |
| `encfirst_ts_cold_on` | TS | cold | 21.1 s | 73.4 s | 14014 | 8806 | +135 | -4.2 s | -2350 |
| `encfirst_ts_warm_on` | TS | warm | 16.4 s | 68.5 s | 14014 | 8888 | +134 | -1.1 s | -2350 |
| `encfirst_aoti_cold_on` | AOTI | cold | 16.5 s | 75.6 s | 11018 | 7510 | +138 | -8.8 s | -5346 |
| `encfirst_aoti_warm_on` | AOTI | warm | 14.1 s | 72.8 s | 11018 | 7537 | +138 | -3.4 s | -5346 |

## Phase Deltas

| Phase / evidence | Step-0 cold/warm | TS cold/warm | AOTI cold/warm | Interpretation |
|---|---:|---:|---:|---|
| `enc_steady_load` | 3142.7 / 812.1 ms | ABSENT / ABSENT | ABSENT / ABSENT | Step 2 worked: inline `enc_steady_aoti.pt2` no longer loads on the scheduler-on startup path. |
| `inline_enc_steady` lane line | loaded at startup | `skipped` / `skipped` | `skipped` / `skipped` | No hidden inline steady loader in lane build. |
| shared constants full blob | `finalize_shared_constants_load` 2999.8 / 772.2 ms | `shared_encoder_constants_load` 3278.8 / 997.0 ms | `shared_encoder_constants_load` 3215.6 / 1000.1 ms | The one kept full encoder constants blob remains; this is Step 9's prewarm target. |
| `enc_first_load` | 3330.7 / 733.0 ms | 3166.1 / 560.4 ms | 12.0 / 4.2 ms | AOTI mode removes the separate 2.48 GB `enc_first.ts` load. |
| first encoder line | raw TS path | `adapter=ts` | `adapter=aoti` | Env split is active. |
| `shared_big_modules` | 3 full blobs at Step 0 | `enc_first_ts,finalize_loaders` | `finalize_loaders` | TS default = shared constants plus TS first encoder; AOTI opt-in = one shared constants blob, with finalize borrowing it. |

Key raw lines:

- All four cells: `shared encoder constants loaded: entries=637 shared_delta_mib=2366.000 source=.../finalize_shared_weights.ts`
- All four cells: `runtime finalize shared constants ready: entries=637 shared_delta_mib=0.000 source=borrowed policy=ws_shared_finalize_pool`
- TS cells: `shared_big_modules=enc_first_ts,finalize_loaders inline_enc_steady=skipped first_encoder=ts`
- AOTI cells: `shared_big_modules=finalize_loaders inline_enc_steady=skipped first_encoder=aoti`
- Grep over `remeasure_runs/*.srvlog` finds no `COLD_START_PHASE phase=enc_steady_load`.

## Memory Accounting

The measured shared constants allocation is `2366 MiB`, matching the ~2.48 GB artifact size in decimal units.

| Startup shape | Full encoder blobs resident | Peak GPU MiB | Delta vs Step 0 | What changed |
|---|---:|---:|---:|---|
| Step 0 baseline | 3 | 16364 | - | `enc_first.ts` + inline `enc_steady_aoti.pt2` + finalize/shared weights. |
| TS default after Step 2-5 | 2 | 14014 | -2350 MiB | Inline `enc_steady` gone; TS first encoder still loads its own full blob. |
| AOTI opt-in after Step 2-5 | 1 | 11018 | -5346 MiB | Inline `enc_steady` gone and first encoder borrows the shared constants blob. |

Conclusion: Step 2 realizes the first ~2.48 GB reclaim in both modes (measured -2350 MiB peak GPU, `/tmp` +2501 MB -> +135 MB). The additional Step 5 reclaim is only realized with `NEMOTRON_WS_ENC_FIRST_TS=0`: measured TS->AOTI peak GPU drops another 2996 MiB, and baseline->AOTI drops 5346 MiB total. Blob accounting is the expected 3 -> 2 -> 1 full blobs, so item-1's ~-5 GB claim is true for AOTI opt-in; TS default gets only the Step-2 half.

## Residual Cold Floor

Post-unify AOTI cold time-to-listening is 16.5 s. The remaining serving-path floor is no longer duplicate encoder blobs:

| Residual phase (AOTI mode) | cold | warm | cold penalty / note |
|---|---:|---:|---|
| `bundle_tokenizer_preproc` | 2501.9 ms | 2267.4 ms | Mostly fixed startup work; CUDA init is not separately split and remains in the early floor. |
| `shared_encoder_constants_load` | 3215.6 ms | 1000.1 ms | Single kept 2.48 GB blob; primary Step 9 prewarm target. |
| `enc_first_load` | 12.0 ms | 4.2 ms | Collapsed; no TS blob read. |
| `finalize_manifest_verify` | 2073.9 ms | 2106.9 ms | Cache-insensitive ~2.1 s residual. |
| `finalize_bucket_bind_dlopen` | 102.9 ms | 105.6 ms | Small stripped package bind/dlopen cost. |
| `lane_build` | 737.0 ms | 688.9 ms | Fixed lane construction. |
| `scheduler_manifest_verify` + bucket bind | 155.0 ms | 143.7 ms | Small residual. |
| `scheduler_warmup_start` | 545.9 ms | 543.2 ms | First scheduler/finalize AOTI run cost. |
| sync `lane_warmup` | 7116.4 ms | 7197.8 ms | Compute, not cold-disk; background warmup keeps the rest off serving path. |

Step 9 should target only the single `shared_encoder_constants_load` cold penalty. It will not remove the bundle/tokenizer work, CUDA context/init effects, finalize manifest verify, finalize package bind/dlopen, first AOTI runs, or lane-warmup compute.

## Notes

- Background-warmup OFF cells were not run; deploy mode is bg-warmup ON and this was the requested priority matrix.
- `/tmp` cleanup completed: no owned `/tmp/[A-Za-z0-9]{6}` dirs remained after the run.
