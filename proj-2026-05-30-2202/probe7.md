# Step 7 Probe — post-unify AOTI extraction cost + cache mechanism (gates Step 8)

**Date:** 2026-05-31 | Binary @ `d8014b8` (Steps 2–6 landed). Read-only analysis; no code change.

## Question
Given Step 1c (no public pre-extracted-dir API) and the post-unify load set (Step 6), is an AOTI
extraction *speed* cache worth building, or does only `/tmp`-leak hygiene matter?

## The post-unify AOTI extraction set (what actually extracts to `/tmp/XXXXXX/...`)
The big ~2.48 GB encoder constants blob is **`torch::jit::load`** (`steady_batch_primitive.h:113`
`load_shared_constants` → `torch::jit::load`), **not** AOTI extraction — so it never touches the
AOTI temp-dir path. The only packages that extract are the small **stripped** AOTI `.pt2`s:

| AOTI package (stripped, constants-on-disk) | extracted size | when |
|---|---:|---|
| `enc_first_aoti.pt2` | 3.9 MB | AOTI enc_first mode only |
| scheduler steady buckets `enc_steady_aoti_b{1,2,4}.pt2` | 4.1 + 4.3 + 4.4 = **12.8 MB** | always (scheduler) |
| finalize buckets `stripped_finalize_buckets/` (33 buckets) | **125 MB** | always (finalize pool) |
| **total residual AOTI extraction** | **~142 MB** | |

(NB: `steady_b_artifacts/*.full.pt2` (2.48 GB ea.) and `enc_steady_t2a_b*.pt2` EPs are on-disk
backups/EPs the server does **not** load — the `MANIFEST.json` points at the 4 MB stripped buckets,
confirmed in the Step-6 srvlogs: `package_verified` uses the small `.pt2`s, `loader_delta_mib=0.000`
because they borrow the shared constants.)

## Measured extraction+bind cost (Step-6 cold, AOTI mode)
- `finalize_bucket_bind_dlopen` = **102.9 ms** cold (extract + dlopen of the 125 MB of stripped finalize buckets)
- `scheduler_bucket_bind_dlopen` = **7.5 ms** cold (the 12.8 MB scheduler buckets)
- `enc_first_load` (AOTI) = **12.0 ms** cold (bind of the 3.9 MB enc_first package)
- ⇒ **total residual AOTI extraction+bind ≈ 110 ms cold.** A speed cache could save at most ~110 ms.

## `/tmp` growth (Step-6 measured)
`/tmp` delta per boot collapsed from **+2501 MB (Step 0) → +138 MB** (post-unify), because the
dominant `/tmp` consumer was the inline `enc_steady_aoti.pt2` extraction (~2.4 GB), which Step 2
removed. The residual ~138 MB is the 142 MB stripped set above. So the crash-loop `/tmp`-fill risk
is **already 18× smaller**, but a dead process still leaks its `/tmp/XXXXXX/.../data/aotinductor*`
tree (torch 2.8 `create_temp_dir()` hard-codes `/tmp/XXXXXX`, cleaned only on graceful exit), so a
crash-restart loop can still accumulate.

## API constraint (Step 1c recap, re-confirmed)
`torch::inductor::AOTIModelPackageLoader` exposes only the package-path ctor; `temp_dir_` is private;
torch 2.8.0 `model_package_loader.cpp create_temp_dir()` hard-codes `/tmp/XXXXXX` with **no `TMPDIR`
honor**. A pre-extracted/extract-once cache would require a custom loader or a torch patch.

## DECISION
- **Steps 8 speed extraction cache: NO-GO.** The residual extraction is ~142 MB of small stripped
  `.so`s costing ~110 ms cold; the 2.48 GB dominant cost is `jit::load`, not extraction. A
  content-SHA/ABI-keyed extract-once cache would need a custom loader/torch patch to save ~110 ms —
  not worth the complexity or risk.
- **Step 8 `/tmp`-hygiene guard: GO (minimal).** Implement only the startup cleanup: scan and
  `rm -rf` stale, **owned** `/tmp/*/data/aotinductor*` (i.e. `/tmp/<mktemp>` trees containing an
  `aotinductor` extraction) left by prior dead processes, to bound the crash-loop `/tmp`-fill risk.
  Do **not** redirect `TMPDIR` (not honored) and do **not** delete dirs owned by live processes.

## Adversarial self-check (is the NO-GO wrong?)
- *Could the 125 MB finalize extraction be slow on a cold/slow disk (EBS gp3) where it'd be worth
  caching?* On a slow disk the **2.48 GB jit::load** dominates far more than the 125 MB extraction;
  Step 9 (prewarm) addresses the big read. The 125 MB extraction would scale up too, but caching it
  still needs a custom loader and saves a small fraction of the big-blob cost — the prewarm lever is
  the right one. NO-GO stands.
- *Does the cache have value for restart speed (2nd boot)?* Marginal (~110 ms) and the page cache
  already warms the 142 MB on a warm restart. Not worth a custom loader.
- *Is `/tmp` hygiene even needed given +138 MB?* Yes — a crash-restart loop accumulates leaked trees
  unbounded; the guard is cheap and removes a real ops foot-gun (the plan notes `/tmp` filled twice).

**Gate output for Step 8: NO-GO speed cache; GO minimal `/tmp`-hygiene guard (startup cleanup of
stale owned `/tmp/*/data/aotinductor*` trees).**
