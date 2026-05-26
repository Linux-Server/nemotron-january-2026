# Opus — Step-0 build review (density_main.cpp)

Reviewed `density_main.cpp` (1605 lines) line-by-line + the run results, vs PHASE2-PLAN Step 0 + the
ownership/topology contract.

## Faithful + correct (confirmed)
- **Topology contract met:** ONE shared steady loader `num_runners=N` (`:1109,1412`), explicit per-worker stream
  into `run(inputs, stream_handle)` (`:415,641,1081`), per-thread `SessionState` + per-thread
  `joint`/`predict`/`enc_first`/`preproc` via `make_worker_context` per worker (`:1416-1418`), and — the key
  overlap-correctness detail — **per-thread `CUDAStreamGuard` around the whole worker compute** (`:497-498,592,
  1072`) so the decode's `argmax().item()` / `to(CPU)` sync the WORKER's stream, not the default. Memory sampler
  + mem-ratio checks present.
- **0b assertion is STRICT** (not weakened): `strict_events_equal` (`:358-396`) compares kind + tokens +
  collector_tokens + text + collector_text; plus token equality. A real concurrent==serial oracle.
- **Default-stream negative control** implemented (`:1177-1180`).
- **`#include "session_main.cpp"` + `#define main…`** (`:7-9`) is clean — doesn't modify session_main; the N=200
  regression stays 0/200 token + 0/200 event.
- **Methodology sound:** StartGate-synchronized concurrent start (`:1134,1143`), warmup before timing (`:1122-
  1130`), `cudaEvent` GPU timing separated from host `run()` wall-time.

## Findings (2 are PLAN/metric-semantics, not harness bugs)

### F1 (MAJOR, metric) — `overlap_estimate = sum_gpu/wall_ms` is CONFOUNDED; it OVERSTATES overlap
`:1174`. Under contention each stream's `cudaEvent` `gpu_ms` STRETCHES (shares SMs/BW), so `sum_gpu` inflates →
the estimate reads **11.06 @N=16 while honest throughput is only 1.92×**. The metric conflates "more concurrent
GPU-time" with "contention-stretched GPU-time." → **Don't cite `overlap_estimate` as the overlap result; the
THROUGHPUT multiplier (1.69×/1.71×/1.92×) is the honest figure.** Relabel it "contention-confounded — not an
overlap proof" or drop it; 11.06 could be mis-read as "11× overlap" (a false strong signal).

### F2 (MAJOR, plan threshold) — the 0b scalar-sync "≤5%" bar is MIS-SPECIFIED → the reported FAIL is spurious
`scalar_sync_pct_of_gpu` = `.item()` wait / `gpu_ms` = 63.8% (`:572`). Because the worker runs decode on its
PER-THREAD stream (`:498`), this is PER-THREAD host-idle — exactly the GPU-idle window other threads fill (0a's
1.69× + the default-stream control being 22.6% worse confirm the fill works). **A high per-thread scalar fraction
is the overlap OPPORTUNITY, not a failure.** The real serialization risk (global/default-stream sync) is correctly
ruled out by the per-thread `CUDAStreamGuard` + the default-stream control. → **Fix the plan's 0b gate:** it should
be "(a) per-thread stream confirmed current for decode ✓ AND (b) default-stream control materially worse ✓" — NOT
"scalar p95 ≤5% of GPU." Remove/relabel the ≤5% bar. (Harness measured it correctly; the threshold semantics from
Round-4 were wrong.)

### F3 (MAJOR, interpretation) — 0a's ~1.7× is the ENCODER-run()-only ceiling, NOT the density multiplier
0a measures pure steady-encoder `run()` concurrency (no decode). The density figure (Step 1a) is where N threads
fill each other's DECODE `.item()` idle — a different, possibly higher number. So **0a confirms "encoder dispatch
overlaps + is correctness-safe" (the conjunct-2 binary YES), but 1.7× is NOT the streams/box multiplier.** The
early plateau (N=2 1.69× ≈ N=4 1.71×, N=16 1.92×) is consistent with the **AOTI execution lock serializing the
host-side dispatch** (`run()` is host-synchronous) while GPU work overlaps → a ~1.7× encoder-dispatch ceiling.
**This is the decisive conjunct-2 result, with a number: overlap is REAL but bounded by the lock's host-dispatch
serialization, not the GIL.** The plan should state 0a-multiplier ≠ Step-1a density multiplier, and Step 1a must
attribute the plateau (lock vs BW) with Nsight.

### F4 (MINOR) — Nsight/CUPTI kernel-overlap proof still pending (Codex flagged honestly)
Throughput + the default-stream control are strong circumstantial evidence overlap is real; the plan asks for the
profiler trace for definitive attribution (lock-serializes-host vs BW-saturates). For the Step-0 BINARY gate (does
it overlap at all?), throughput+control suffice; **defer the Nsight attribution to Step 1a.**

### F5 (MINOR) — 0b/0c were limited smoke (N=2, few rows); run full before the PASS is final
Run full 0b over all 200 rows + higher N (4/8/16) for concurrent==serial, and 0c over more buckets. The 0-mismatch
correctness result is only smoke-scale so far. (Codex's own next step.)

### F6 (MINOR) — record the mem-flat numbers at target N
The harness samples `peak_gpu_mem` + `mem_ratio_target_vs_n1`; confirm the actual telemetry shows ≈1.0× (one
weight copy) at N=16 before the PASS. Not yet reported in the summary line.

## Verdict
**Harness is SOUND and faithful to the topology contract; the correctness assertion is strict.** The decisive
preliminary finding is strong and honest: **concurrent native dispatch DOES overlap and IS correctness-safe
(conjunct-2 binary = YES), bounded ~1.7× for encoder-only on the 5090 — consistent with the AOTI execution lock
serializing host-dispatch, NOT the GIL.** Two issues (F1 overlap_estimate confound, F2 the ≤5% threshold) are
metric/plan-semantics fixes, not harness bugs. Before committing Step 0: relabel/drop `overlap_estimate`, fix the
plan's 0b threshold semantics, and run the full 0b/0c + record mem-flat. The Nsight attribution + the actual
density multiplier are Step-1a, not Step-0.
