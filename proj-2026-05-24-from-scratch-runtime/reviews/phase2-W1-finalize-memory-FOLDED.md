# W1 memory wall — paired investigation FOLDED (Codex + Opus-4.7 max-thinking)

Inputs: `codex-finalize-memory-W1.md`, `opus-finalize-memory-W1.md`. **Strong convergence — and both REFUTE the W1
premise (my framing).** The 5090 N=4 memory knee is NOT the finalize buckets. The lever is **`enc_first.ts`
deduplication.**

## Root cause (both, with MEASURED evidence)
- **The finalize buckets are cheap:** `loader_delta = 0.000 GiB per bucket` — they share ONE ~2.30 GiB
  `user_managed` constants set (`density_main.cpp:946,1072`). My "finalize per-runner activation × N" hypothesis
  was wrong.
- **The hog is `enc_first.ts` — a 2.48 GiB full-fp32 encoder loaded ONCE PER WORKER** (`make_worker_context`,
  `density_main.cpp:629`) to serve a single first-chunk forward per session. At N workers = N × 2.48 GiB ≈ the
  entire measured **~2.51 GiB/stream**. The N=4 context increment is ~10 GiB ≈ 4 × `enc_first`; the OOM stack frame
  is `load_module_on_device ← make_worker_context` (it OOMs *loading modules, before any finalize runs*).
- **Independent proofs (Opus):** the 0c control — 8 workers at 8 vs 16 finalize-runner-slots give the **same
  ~30.8 GiB** (finalize runner count doesn't move memory); activation scales as **N (workers)**, not
  `loaded_buckets × runners/bucket`. The steady AOTI encoder is correctly shared (one weight copy + ~0.31 GiB/
  runner arena, MEASURED).
- **Why:** the per-thread-module-handles decision (Step-0 concurrency safety) was applied to a **2.48 GiB** module.
  `joint`/`predict` (7/29 MB) duplicate fine; `enc_first` does not.
- **Cross-stack (Opus):** the **Python server has NO `enc_first`** — ONE shared encoder serves all geometries via
  `drop_extra`, no `num_runners` activation pool. Python never had this wall by design.
- **Finalize hygiene already correct:** the sweep already loads+warms only the workload's bucket subset (12–15/32),
  fail-closed on unknown `(drop,T)` — the warmup-vs-load-only tension is already resolved.

## The fix (both agree): stop duplicating the encoder per worker
- **Fix-1 (preferred, clean):** fold the first-chunk geometry into the already-shared **steady AOTI loader** as a
  second entry reusing the same shared constants (the finalize pool already proves the stripped-weights pattern),
  then delete `enc_first.ts` from the worker context. **Also closes the known "first-chunk-still-TorchScript"
  residual.** Concurrency-safe via the proven `num_runners` pool.
- **Fix-2 (cheap, ~10 lines, fast confirm):** load `enc_first` ONCE + share a reference across workers (with a lock
  around the rare first-chunk forward — restores the concurrency safety the per-thread handles gave). Removes
  (N−1)×2.48 GiB; keeps a second resident encoder copy. Confirmable in ~5 min by re-running 1a.
- **Reject** padded-bucket consolidation (saves ~0 — buckets already cost ~0 — and not token-safe: Conformer
  depthwise right-context bleeds pad zeros, the exact-T finding). Runner-cap / shared-finalize-activation are ~0 or
  2nd-order.

## Expected new knee (per-stream 2.51 → ~0.35–0.49 GiB; ESTIMATED ±~20%, confirmable in ~5 min)
- **5090: N=4 → ~40–45** (memory stops binding; the wall flips to GPU **compute/contention** — the transferable
  question, and the Step-0 ~N=4 encoder-contention may cap the *full-session* knee below the ~45 memory ceiling;
  the re-sweep settles it).
- **L40S: ~13 → ~60–69.**

## Two methodology/sequencing findings (Opus — important)
1. **Run the knee sweep FRESH-PROCESS-PER-N.** Same-process `used_before` grows 4.98→9.70 GiB across N=1→4 (CUDA
   modules/cuDNN workspace not returned by `emptyCache()`), inflating the OOM reading. A fresh-process N=8 projects
   to ~29.7 GiB (still edge-of-OOM on the 5090 *pre-fix*).
2. **The enc_first wall is NOT 5090-specific — it's a PREREQUISITE for W3 (the L40S sweep).** Under the current
   design the L40S would also hit a ~N=13 memory knee, right at its SLO-robust target — so **the L40S sweep would
   mis-measure the binding resource (reading the enc_first wall, not the real compute/host limit) unless enc_first
   is deduped first.**

## Net + next
The lever's name changes from "finalize-memory reduction" to **"dedup the per-worker `enc_first` (2.48 GiB)."** It
is **cheap, the critical path, and a prerequisite for the L40S gate**, with an ESTIMATED ~10× 5090 density jump
(N=4→~40-45) — confirmable in ~5 min. **Sequence: implement Fix-2 (cheap shared+locked enc_first) → re-run 1a
fresh-process-per-N to confirm the knee + correctness → then Fix-1 (AOTI fold) for production → then the L40S
sweep (W3) measures the real (compute/host) binding resource.**
