# Step 0 — COLD baseline (pre-change), local RTX 5090 (sm_120)

**Date:** 2026-05-30 | **Binary:** `cpp/build_step10/ws_server` @ HEAD `6475bfe` (all landed cold-start commits 4cfbba1…6475bfe in). **CAP=64, LANES=64, scheduler on, BATCH_STEADY/MAX=4/WINDOW=10.**
**Raw:** `baseline_runs/{summary.txt,phases.txt,*.srvlog}`; runner `run_cold_baseline.sh`.

## Method (state which — see plan Rules)
4-cell matrix: {COLD-artifacts, WARM} × {bg-warmup OFF, ON}.
- **COLD method = per-file `posix_fadvise(POSIX_FADV_DONTNEED)` eviction of the artifact tree** (`artifacts/` + `steady_b_artifacts/`, ~7.5 GB of 2.48 GB blobs) — **non-root**, targets the dominant cold cost precisely. This is NOT a global `echo 3 > drop_caches` (that needs sudo here — password-gated — and would also evict torch libs/binary, a smaller component). A full-drop cold run can be added later with one user sudo if the lib-eviction component is wanted; for sizing the artifact-read levers (the plan's focus) the fadvise method is the right, reproducible instrument.
- WARM = artifact tree pulled into page cache before the run. Each cell: graceful start → poll to readiness/full-warm → sample peak GPU mem (per-pid) + host RSS + `/tmp` delta → graceful shutdown + reclaim leaked `/tmp` AOTI dirs.

## Headline numbers
| Cell | bg | cache | **time-to-serving** (`listening`) | full-warm total (`background_warm_complete`/`sync_warm_done`) |
|---|---|---|---|---|
| cold_off | 0 | cold | 85.2 s (sync; serves only after full warmup) | 85.2 s |
| warm_off | 0 | warm | 77.1 s (sync) | 77.1 s |
| **cold_on** | 1 | cold | **25.2 s** | 85.8 s (60.6 s bg) |
| **warm_on** | 1 | warm | **17.5 s** | 78.1 s (60.6 s bg) |

- **bg-warmup ON is the deploy mode** (serves at 17.5 s warm / 25.2 s cold; the 64-lane warmup finishes behind admission gating). Confirms the prior "~18 s warm time-to-serving" note (warm_on = 17.5 s).
- bg-warmup OFF blocks serving on the full ~69 s sync lane-warmup → 77–85 s. Not the deploy mode; shown for contrast.

## Phase breakdown — COLD vs WARM (the lever map)
`elapsed_ms` per phase, cold_on vs warm_on (bg-warmup ON; same shape as OFF):

| phase | COLD | WARM | cold penalty | cache-sensitive? | lever |
|---|---|---|---|---|---|
| bundle_tokenizer_preproc | 2545 | 2305 | +240 | mild | — |
| **enc_first_load** | **3331** | **733** | **+2598** | YES (2.48 GB read) | **unify (Steps 4-5)** |
| **enc_steady_load** | **3143** | **812** | **+2331** | YES (2.48 GB) | **lazy/skip (Step 2)** |
| finalize_manifest_verify | 2079 | 2083 | ~0 | NO (compute/hash) | non-disk residual ⚠️ |
| **finalize_shared_constants_load** | **3000** | **772** | **+2228** | YES (2.48 GB) | the kept blob → **prewarm (Step 9)** |
| finalize_bucket_bind_dlopen | 104 | 104 | ~0 | NO | — (stripped, tiny) |
| lane_build | 734 | 696 | +38 | NO | — |
| **lane_warmup** | 9688 (sync part) + 60568 (bg) | 9328 + 60639 | ~0 | **NO (compute)** | off serving-path via bg-warmup; **out of cold-disk scope** |
| scheduler_* (preload/verify/bind) | ~310 total | ~290 | ~0 | NO | stripped buckets + SHA-skip already done (4df9561, 8f99a5e) |

### What the cold penalty IS
The entire COLD−WARM time-to-serving delta (25.2 − 17.5 = **7.7 s**) is the **three 2.48 GB blob reads** (enc_first +2.6 s, enc_steady +2.3 s, finalize_shared +2.2 s ≈ 7.2 s) + ~0.5 s misc. Everything else (lane_warmup, manifest_verify, scheduler preload) is cache-insensitive compute.

## Memory + /tmp
- **peak GPU = 16364 MiB** (identical all cells). The three 2.48 GB blobs ≈ **7.4 GB** of this; **the unify should reclaim ~5 GB** (two duplicate full-weight copies → one shared constants map). This is item-1's ~−5 GB claim, measurable at Step 6.
- **peak host RSS ≈ 12.3–12.6 GB.**
- **`/tmp` +2501 MB per boot** (AOTI extraction of `enc_steady_aoti.pt2` + buckets to `/tmp/XXXXXX`). Identical cold/warm. This is the crash-loop `/tmp`-fill risk Step 7/8 guards; graceful shutdown reclaimed it each cell (0 net growth across 4 boots).

## Targets this sizes (for /implement)
1. **Step 2 (lazy/skip inline enc_steady):** removes the `enc_steady_load` phase entirely when scheduler-on → −3.1 s cold / −0.8 s warm time-to-serving + frees one 2.48 GB GPU copy.
2. **Steps 4-5 (unify enc_first → shared constants):** removes `enc_first_load`'s separate blob → −2.6 s cold / −0.7 s warm + frees the 2nd duplicate ~2.48 GB GPU copy. Net of Steps 2+5: time-to-serving cold ≈ 25.2 → ~**19 s**, warm ≈ 17.5 → ~**16 s**; GPU peak ≈ 16.4 → ~**11 GB**.
3. **Step 9 (prewarm):** the remaining single `finalize_shared_constants_load` blob (~3 s cold) overlaps disk read with the serialized `jit::load` → recovers most of its cold penalty. After unify+prewarm the cold floor ≈ warm (≈16–17 s), dominated by `bundle` (2.3 s) + `finalize_manifest_verify` (2.1 s, non-disk) + CUDA init + the sync lane portion.
4. **⚠️ Non-disk residual to flag (not a Step in this plan):** `finalize_manifest_verify` ~2.1 s is cache-independent and survived the scheduler SHA-skip — worth a follow-up look (is the finalize manifest still SHA-hashing the shared weights?). Out of cold-*disk* scope; note for later.

**Conclusion:** the plan's lever ordering is correct and well-sized — the unify (Steps 2-5) attacks ~5 s of the 7.7 s cold penalty + ~5 GB GPU, prewarm (Step 9) attacks the residual ~3 s. `lane_warmup` (the 69 s elephant) is compute, already mitigated by bg-warmup, and correctly out of this plan's cold-disk scope.
