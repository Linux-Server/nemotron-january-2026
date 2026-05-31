# Step 9 - page-cache prewarm

**Date:** 2026-05-31 | **Binary:** `cpp/build_step10/ws_server`.
**Raw:** `prewarm_runs/{summary.tsv,phases.txt,artifacts.txt,*.srvlog}`.

## Implementation

`NEMOTRON_WS_PREWARM` is default-off. Values `1`, `true`, or `on` enable it. When enabled, `SharedRuntime` starts an owned `runtime_io::Prewarmer` immediately after stale `/tmp` hygiene and before the bundle/tokenizer/preproc load.

Queued files:

- Always: `artifacts/finalize_shared_weights.ts` (`2,477,725,779` bytes).
- TS first-encoder mode only: `artifacts/enc_first.ts` (`2,478,955,502` bytes).

The worker path is read-only: `open(O_RDONLY|O_CLOEXEC)`, `posix_fadvise(POSIX_FADV_WILLNEED)`, then sequential `read()` in 16 MiB blocks. Missing files and read/advice failures are logged and skipped; they do not fail construction. The prewarmer owns its worker threads and joins them before the constructor's final ready phase.

The kickoff marker is async and near-zero:

- TS: `PREWARM kicked files=2 bytes=4956681281 ...`; `prewarm_kickoff=0.2 ms`.
- AOTI: `PREWARM kicked files=1 bytes=2477725779 ...`; `prewarm_kickoff=0.1 ms`.

## Cold Results

COLD method matches Step 6: per-file `POSIX_FADV_DONTNEED` over `artifacts/` and `steady_b_artifacts/` before each cell (`~41.22 GB` advised out), deploy-mode background warmup on, scheduler on, `CAP/LANES=64`.

| Mode | Prewarm | `shared_encoder_constants_load` | Time-to-listening | Delta vs off |
|---|---:|---:|---:|---:|
| TS first encoder | off | 3215.3 ms | 21270.8 ms | - |
| TS first encoder | on | 1170.6 ms | 16557.2 ms | shared -2044.7 ms, listen -4713.6 ms |
| AOTI first encoder | off | 3285.5 ms | 16543.7 ms | - |
| AOTI first encoder | on | 1036.8 ms | 14373.0 ms | shared -2248.7 ms, listen -2170.7 ms |

The AOTI result is the post-unify target: the single retained `finalize_shared_weights.ts` cold read is mostly hidden under the earlier `bundle_tokenizer_preproc` phase. TS also improves because `enc_first.ts` is queued in parallel.

Correctness remains byte/token exact for this timing-only change:

- `runtime-smoke`: `token_divergences=0`, `event_divergences=0`, `errors=0`.
- `b2-t1 --correctness-rows 4`: `token_divergences=0`, `event_divergences=0`, `errors=0`.

## Storage Guidance

Instance-store/local NVMe has low first-read latency and high throughput, so prewarm still helps by overlapping the read but the absolute gain can be smaller.

EBS gp3 is network-attached. AWS documents gp3 baseline performance as `3,000 IOPS` and `125 MiB/s` throughput, with throughput provisionable higher. A fresh EBS volume restored from a snapshot lazy-loads blocks from Amazon S3 on first access; AWS recommends initializing volumes by reading every block when full performance is needed immediately. For fresh-from-snapshot gp3, expect the first cold boot to be slowest; prewarm plus one explicit read-through/volume initialization helps because the artifact blocks are pulled before or during startup.

Sources:

- https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html
- https://docs.aws.amazon.com/ebs/latest/userguide/ebs-initialize.html

## mmap Note

No mmap/JIT-loader change was implemented. If Torch exposes or grows a mmap-backed `jit::load` path for these artifacts, it could reduce userspace copies, but it would still rely on the same underlying page cache behavior. This step keeps the loader unchanged and only changes timing.
