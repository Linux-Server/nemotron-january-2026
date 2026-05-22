# Plan: Per-batch-size manual CUDA-graph capture for the streaming encoder

Project directory: `./proj-2026-05-21-1959-cudagraph`

## Context
Replace the dead `torch.compile(reduce-overhead)` encoder path (Step 10b: inductor warmup never completes on
Modal T4/L4 — minutes-and-hung) with **manual CUDA-graph capture** of `encoder.cache_aware_stream_step`. Probe
(a) proved the primitive at B=1: byte-exact (text + state `max_abs=0`), **251 ms** warmup (no inductor codegen →
cloud/cold-start viable), **1.36×** synced per-chunk. The cheaper-call lever must compose with continuous
batching, so we capture a graph **per batch size B = 1…K** (no padding), replay the matching B, fall back to
eager for B>K and all non-steady buckets. This is the cloud lever (B=1 lifts the ~5 knee where batches don't
form) **and** the self-host lever (small-B graphs compound with batching; avg B≈4 at the N=56 knee, so the
realtime range is exactly where graphs help most). Flag-gated, default off, fail-closed, English-first.

## Reference implementations
- `proj-2026-05-21-0410/probe_manual_cudagraph.py` — the validated static-buffer record/replay (`ManualCudaGraphEncoder`)
  + the byte-exact methodology (per-step text bytes + `compare_snapshot` state `max_abs`). Extend its compare to B>1.
- vLLM / TRT-LLM bucketed CUDA-graph capture (graph-per-batch-bucket). **Divergence:** they pad actual B up to
  sparse power-of-2 buckets; we capture **every** B in 1…K (no padding) so byte-exactness stays a clean
  `graph(B)==eager(B)` gate and we never re-open the padded-row-independence question. K is small (~16) so the
  graph count is affordable.
- `src/nemotron_speech/batch_primitives.py` — `stack_caches` (dim1 channel/time, dim0 len), owned-clone scatter,
  `conformer_stream_step_restoring_drop_extra`. The per-B static buffers mirror these layouts.
- `proj-2026-05-21-0410/SUMMARY.md` — the lane analysis (host-gap ~3 ms ≈ constant in B → graph speedup 1.36×@B1,
  ~1.1×@B46; diminishing toward large B) and the batch-size-distribution finding (avg B≈4 at the knee).

## Current state
- `src/nemotron_speech/server.py`: the dead torch.compile path `_configure_encoder_compile` /
  `NEMOTRON_ENCODER_COMPILE` (to be superseded); the scheduler's batched model call (`_process_ready_batch` and
  the `conformer_stream_step(B)` dispatch) — **the wiring point**; per-session cache/hyps state; warmup path.
  NOTE: prototype (b) (`NEMOTRON_MODEL_LANES`, `tests/test_scheduler_model_lanes.py`) is landing changes to this
  same scheduler region — Step 3 must be finalized against (b)'s committed structure.
- `proj-2026-05-21-0410/probe_manual_cudagraph.py`: `ManualCudaGraphEncoder` (B=1), `compare_runs` (byte-exact).
- The eager batched path is already byte-exact at scale (200/200 canary, FORK_ASSERT clean).
- Encoder geometry: steady bucket T = pre_encode_cache(9) + shift(16) = 25, drop_extra=2; decode greedy,
  `use_cuda_graph_decoder=False` (Blackwell — hard constraint, decode stays eager).

## Rules
### Correctness (hard gates)
- **Byte-exact per B**: for every B in 1…K, graph(B) per-stream output must equal eager-batched(B) — interim
  text byte-identical at every step, final text byte-identical, and state `max_abs=0` (extend the probe's check).
  This is the cache-aware-state corruption hazard; gate before any commit.
- **English rc1 byte-identical** to the established baseline (existing project gate).
- **Default-off identity**: flag unset / `=0` → byte-identical to the current server (graph code fully bypassed).
### Safety & sequencing
- **Fail-closed**: any capture failure (any GPU arch, any bucket) → that B (or all) falls back to eager; the
  server still serves. Eager is never removed.
- **No padding**: capture every B in 1…K (real lengths). B>K → eager. Non-steady buckets (warmup/first-chunk
  drop_extra=0 / vad_stop finalize / barrier-drain) → eager.
- Only the **encoder** is graphed; decode stays eager (`use_cuda_graph_decoder=False`).
- Don't break the multilingual prompted path, the silence0_warm200 finalize/fork logic, or the warmup path.
- No new heavy deps. Flag-gated; default = current behavior until proven.

## Steps

- [ ] **1. Per-B byte-exact + speedup probe; pick K.**
  Extend `proj-2026-05-21-0410/probe_manual_cudagraph.py` (or a sibling) to capture B = 1…16 (configurable),
  building per-B stacked inputs from K independent clips. For each B: confirm byte-exact vs eager-batched
  (per-step text bytes + state `max_abs=0`, the existing compare extended to B>1) AND measure the synced per-B
  speedup. Output: the measured speedup-vs-B curve (resolves the projected 1.36×@B1 → ~1.1×@B46) and a
  recommended **K** = the B where speedup drops below ~1.15×. Standalone, no `server.py`. Gate: byte-exact for
  ALL tested B (any mismatch → diagnose the stacking/capture before proceeding).
  Key files: `proj-2026-05-21-1959-cudagraph/probe_perB_cudagraph.py`

- [ ] **2. Bucketed graph-manager module (standalone-tested).**
  New `BucketedCudaGraphEncoder` (e.g. in `src/nemotron_speech/cudagraph_encoder.py`): holds per-B captured
  graphs + per-B static buffers (cache `[layers,B,...]` channel/time, `[B]` len, mel `[B,F,T]`); `warmup()`
  captures B=1…K (side-stream warmup → capture, per probe); `replay(B, inputs)→outputs` does copy-in / replay /
  return-static (caller clones out); fail-closed (capture error on any B → mark uncaptured → eager). Unit test
  asserts byte-exact per B vs eager and that uncaptured-B cleanly returns "use eager". No `server.py` wiring yet.
  Key files: `src/nemotron_speech/cudagraph_encoder.py`, `tests/test_cudagraph_encoder.py`

- [ ] **3. Wire into the scheduler's batched call (FINALIZE AFTER (b) LANDS).**
  In `server.py`, gate behind `NEMOTRON_ENCODER_CUDAGRAPH` (default off) + `NEMOTRON_ENCODER_CUDAGRAPH_MAX_B`
  (default K). At the batched model call: if steady-bucket AND B≤K AND captured → `replay(B)`; else eager.
  Capture all 1…K at startup warmup (after model load). Supersede the dead `NEMOTRON_ENCODER_COMPILE` path
  (remove or redirect to manual). Route non-steady buckets to eager. **Integration point = the scheduler's
  batched `conformer_stream_step` dispatch; finalize against (b)'s committed scheduler structure** (lanes +
  graphs must compose — each lane replays on its own stream; the graph's static buffers are per-lane or guarded).
  Key files: `src/nemotron_speech/server.py`

- [ ] **4. Local byte-exact gate at scale (hard gate).**
  Stream a fixed multi-stream clip set with `NEMOTRON_ENCODER_CUDAGRAPH=1` vs `=0` (+ scheduler/batching on) and
  diff transcripts — must be byte-identical (200/200-style canary), `FORK_ASSERT=1` clean over a multi-minute
  run. Confirm default-off (`=0`/unset) is byte-identical to the pre-change server. Confirm graphs actually
  engage (replay counters > 0, no silent all-eager).
  Key files: existing harness (`proj-2026-05-19-eou-endpointing/`), `proj-2026-05-21-1959-cudagraph/`

- [ ] **5. Local knee measurement (first measured payoff).**
  Realtime keep-up sweep, graph-on vs graph-off (scheduler+batching on): does the knee lift at the small B
  realtime produces (baseline 56 → ?)? Also single-stream B=1 latency/knee (expect ≈ the old compile-only 24).
  Record avg B at the knee + the per-B engagement mix.
  Key files: `proj-2026-05-21-1959-cudagraph/local-knee.md`

- [ ] **6. Cloud knee test (T4 + L4) — the deliverable torch.compile couldn't run.**
  Deploy with manual capture; CONFIRM it engages at startup (~250 ms × K, no inductor hang — the Step-10b
  failure mode must be gone), smoke for correctness, then sweep T4/L4 and compare to the batch=1 baseline (~5).
  Does the cheaper call lift the cloud knee where batching couldn't? Billable, cost-conscious (smoke first, stop
  apps immediately, T4+L4 only). Write into `proj-2026-05-20-modal-cost/RESULTS.md` (Step 10c).
  Key files: `src/nemotron_speech/modal/asr_bench_modal.py`, `proj-2026-05-20-modal-cost/RESULTS.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Per-B byte-exact + speedup probe; pick K | pending | — | standalone; resolves the speedup curve + K |
| 2 | Bucketed graph-manager module | pending | — | standalone-tested; fail-closed |
| 3 | Wire into scheduler's batched call | pending | — | **finalize after (b) lands**; lanes+graphs compose |
| 4 | Local byte-exact gate at scale | pending | — | hard gate: graph-on==graph-off, FORK_ASSERT |
| 5 | Local knee measurement | pending | — | first measured payoff (56→?) |
| 6 | Cloud knee test (T4+L4) | pending | — | the torch.compile-couldn't-run deliverable |
