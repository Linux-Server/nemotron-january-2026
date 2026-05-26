# W1 finalize-memory attribution

Date: 2026-05-26. Scope: read-only investigation plus a small loader-memory probe. Main evidence is the C++ density telemetry under
`runtime/artifacts_n200_mel/logs/20260526T193756Z/` and the Step-0 C++ logs; a Python AOTI probe was used only to sanity-check
`load_constants(user_managed=True)` sharing because it does not reproduce the C++ explicit-stream runner memory shape.

## Bottom line

The N=8 OOM is real, but the C++ stack and memory numbers do **not** support "loaded finalize buckets × runners/bucket" as the dominant resident allocation. The OOM occurs after the finalize pool is loaded, while constructing per-worker runtime context modules:

- `make_worker_context()` loads `session_bundle.ts`, then loads `enc_first.ts`, `joint_step.ts`, and `predict_step.ts` per worker (`runtime/cpp/density_main.cpp:624-631`).
- `load_module_on_device()` does `torch::jit::load(path); module.to(device); module.eval()` (`runtime/cpp/density_main.cpp:617-621`).
- The N=8 OOM stack frame #35 maps by `nm -anC` to `load_module_on_device(...)`, frame #36 to `make_worker_context(...)`; the OOM text reports only ~25 MiB free and ~30.50 GiB process memory in use.
- `enc_first.ts` is a 2.4 GB artifact (`ls -lhL runtime/artifacts_n200_mel/enc_first.ts`) and is loaded once per worker in this harness. That is the large N-scaling allocation.

Finalize memory still matters for headroom, but finalize-only levers are unlikely to move the 5090 knee from N=4 to N=8 until the per-worker `enc_first` duplication is removed or shared.

## Attribution

Measured C++ facts:

| Component | Measurement | Evidence |
|---|---:|---|
| Shared finalize weights | 2.300-2.311 GiB, one copy per `FinalizeBucketLoaderPool` | `shared_delta` in density logs; load at `runtime/cpp/density_main.cpp:944-953` |
| Per-bucket finalize loader/constants | 0.000 GiB per bucket on load | `loader_delta=0.000 GiB` for every bucket; `load_constants(..., user_managed=true)` at `density_main.cpp:1068-1075` |
| Steady AOTI weights | 2.309 GiB, flat across N | Step-0 0a `loader_delta_bytes=2478833664` for N=1/4/16 |
| Steady AOTI scratch | ~0.056 GiB/runner | Step-0 0a: N=4 `used_after_run-used_after_loader=0.225 GiB`; N=16 `=0.890 GiB` |
| N=4 full density peak | 24.352 GiB | `density_num_runners4...1a_full_session_density_sweep.jsonl` |
| N=4 after shared loaders, before contexts/run | 14.319 GiB | same log: `used_after_loaders_bytes` |
| N=4 runtime/context increment | 10.034 GiB | peak minus after-loaders; ~4 × 2.3 GiB `enc_first` copies plus scratch/session state |
| N=8 OOM point | ~30.5 GiB process memory before timed work | N=8 error JSON/stdout, after loading 15 needed finalize buckets |

Important nuance: the density sweep already does subset preload/warm, not all-bucket warm. It computes assigned-session buckets (`density_main.cpp:2783-2795`), preloads only `needed_buckets` (`density_main.cpp:2967-2994`), and warms only each worker's representative buckets (`density_main.cpp:3047-3087`). For the current corpus:

- Full `artifacts_n200_mel/session_bundle.ts`: 200/200 finalizes, 16 unique `(drop,T)` buckets, all `drop=2`, `T=43..58`.
- N=4 current workload (`sessions_per_worker=5`, raised to 20 sessions): 12/32 buckets.
- N=8 target workload (`40` sessions): 15/32 buckets.

## Reduction options

| Option | Memory saving | Density impact | T1/correctness risk |
|---|---:|---|---|
| Cap finalize runners to 1/bucket | Load-time saving is 0 GiB; runtime saving is only second-runner scratch for same-bucket concurrency. Current logs do not isolate it, but N=4 leaves <~0.6-0.8 GiB total for finalize/session scratch after accounting for per-worker `enc_first` and steady scratch. | May queue rare same-bucket finalizes. Does not fix N=8 because OOM happens before finalize warm/run. | Low for tokens; medium for tail latency if same bucket clusters. |
| Load+warm only workload buckets | Already implemented in density sweep. Avoids warming 17-20 unused buckets for this corpus, but loader delta is 0 GiB; saving is only any warm/run scratch for unused buckets. | Good hygiene and necessary for cold-start policy, but not enough for N=8 in current harness. | Low if fail-closed on unexpected `(drop,T)`. |
| Shared/streamed finalize activation via one buffer/lock | Could bound finalize scratch to one active finalize runner. Current data says this is a headroom lever, not the N=8 root cause. | Serializes finalizes; likely acceptable if finalizes are rare, but can hurt burst TTFS. | Low-medium: lock ordering/tail risk, token-safe if exact-T bucket unchanged. |
| Padded-bucket consolidation | Would reduce bucket count, but bucket load is already 0 GiB and consolidation is unsafe. | Fewer buckets but wrong target. | High/no-go: prior 1.3b found pad-to-single-bucket 39/40 token-exact with one deletion; depthwise-conv right context sees trailing zeros (`runtime/1.3b-finalize-encoder-findings.md:36-43`). |
| Free/reuse steady pool scratch during finalize | Steady scratch is only ~0.056 GiB/runner, so N=8 saves ~0.45 GiB. | Minimal. | Medium implementation risk for little memory. |
| Share `enc_first.ts` or convert first encoder to shared-weight/AOTI | Saves roughly 2.3 GiB per extra worker; at N=8 this is ~16 GiB versus per-worker copies. | This is the lever that should actually move the 5090 memory knee past N=8; compute may become the next limiter. | Needs T1 gate for thread safety/stream behavior, but it is read-only model state and should be tractable. |

## Recommendation

Do not spend W1 assuming finalize bucket count is the binding resident memory. The immediate fix to push past N=4->8 is to remove per-worker `enc_first.ts` GPU copies, then keep the finalize hygiene:

1. Share `enc_first` across workers or export it as an AOTI/shared-weights runner pool, with the same serial-oracle and explicit-stream gates used for steady.
2. Keep density's load+warm-needed-subset behavior and port that policy to production if production requires finalize warmup.
3. Cap general finalize runners to 1/bucket only if the post-`enc_first` memory probe still shows finalize scratch as material; otherwise keep the current `min(workers,2)` cap (`density_main.cpp:1114-1115`) to preserve rare same-bucket concurrency.

Expected knee: finalize-only subset + runner cap probably leaves the 5090 knee at N=4 in this harness, because N=8 OOMs while loading per-worker modules. Sharing `enc_first` should make N=8 memory-feasible with large headroom; the next knee then needs a fresh compute/tail sweep.

## Warmup vs subset loading

The reconciliation is: load and warm the exact workload subset, not all 32 manifest buckets. For the current density workload that means the assigned `(drop,T)` set above; for production it should be a declared/admitted workload bucket set. Unexpected buckets should fail closed: `FinalizeBucketLoaderPool::load_bucket_locked()` already throws on a missing key (`density_main.cpp:1060-1066`). A T1-approved eager fallback would be a separate policy decision, not a silent path.

