# Plan: Finalize-latency optimization — cut the P95 tail toward the streaming frontier

Project directory: `./proj-2026-05-22-1353`

## Context
Client-side TTFS (end-of-speech → final transcript), measured full-1000 @ conc 10 over the internet to a
production-config us-west-2 box (cudagraph ON, lanes=2, silence0_warm200, rc1): **L40S median 274 / P95 401 ms**
(L4 290/447). Decomposed: ~200 ms harness trailing-silence window (a benchmark constant, not ours) + ~23 ms
network RTT (measured, stable) + **~51 ms median / ~178 ms P95 server-side finalize compute = the target.** The
median is already frontier-competitive (Deepgram 247, Soniox 249); the **P95 tail** is the gap to the frontier
(Deepgram 298 / Soniox 281). Root cause (confirmed in code): the one model call on the finalize critical path,
`_conformer_stream_step(... keep_all_outputs=True ...)`, runs **eager** — it bypasses the steady CUDA-graph — so
it pays a full kernel-launch storm whose variance under MPS is the tail (the same failure mode steady graphs
already fixed: L40S K=4/MPS p95 349-767 → 216). This plan graphs the finalize encoder call (+ supporting levers),
flag-gated and byte-exact, to pull the P95 toward the frontier.

## Reference implementations
- `src/nemotron_speech/cudagraph_encoder.py` — `BucketedCudaGraphEncoder`: proven per-B static-buffer
  record/replay + fail-closed manager (steady bucket only). The finalize-graph **extends this pattern** to
  finalize buckets keyed by `(B, T, drop_extra, keep_all_outputs=True)` with variable final `T`.
- The steady CUDA-graph wiring in `server.py` is the exact pattern to mirror: `_conformer_stream_step`,
  `_encoder_cudagraph_bucket_for_call` (`server.py:1828`), `_cudagraph_encoder_cache_step_installed`, the
  per-replica + per-lane-stream manager dicts, and the fail-closed eager fallback.
- Byte-exact gate methodology: `proj-2026-05-21-1959-cudagraph/step4_byte_exact_canary.sh` (graph-on==graph-off,
  lanes=1 AND lanes=2, 100/100 transcripts; FORK_ASSERT clean).
- Cloud TTFS retest harness: `ec2-bench/bench_client_wan.sh` (client-side WAN, full 1000 @ conc 10, production
  config) + `ec2-bench/start_prod_server.sh`.
- Authoritative analysis: `proj-2026-05-21-1959-cudagraph/finalize-optimization-suggestions.md` (Codex
  critical-path trace + 9 ranked suggestions); data: `proj-2026-05-21-1959-cudagraph/cloud-retest.md`.
- Divergence from the steady graph: finalize buckets need **variable `T`** (not one steady `T=25`) and
  **`keep_all_outputs=True`**, so the finalize-graph is a *separate* bucket family + flag, never a reuse of the
  steady graph (different encoder output shape / downstream contract).

## Current state (exact code, confirmed)
- Finalize model call (EAGER, the target): batched `_process_final_batch_rows` → `_conformer_stream_step(...
  keep_all_outputs=True ...)` at `server.py:6128`; serial `_process_final_chunk` at `server.py:7226` → call at
  `server.py:7310`.
- CUDA-graph gate that excludes it: `_encoder_cudagraph_bucket_for_call` (`server.py:1828`) returns None when
  `keep_all_outputs` is truthy (`server.py:1835`) and when `chunk_frames != steady_T` (steady `T = pre_encode_cache
  + shift`). Steady manager: `cudagraph_encoder.py` (`get_initial_cache_state` + `empty_like` static buffers,
  side-stream capture, `replay`→None fail-closed).
- Finalize geometry: `final_padding_frames = (right_context + 1) * shift_frames` (`server.py:1450`); rc1 + 16-frame
  shift → 32 padding frames. warm200 makes sessions "non-first" → the final branch prepends the mel ring + uses
  `drop_extra=self.drop_extra`; inferred final `T ≈ 42..58` (to be confirmed by Step 1 telemetry).
- Fork (do NOT change): `_build_continuous_finalize_fork` (`server.py:5288`) deep-clones audio/cache/hyps/pred;
  `fork_clone_ms` recorded at `server.py:5472`/`6272`; fork assertions guard parent-state corruption.
- Final preprocessing loop (per-16-frame-slice): serial `server.py:7254-7283`; batch-row `server.py:5715-5744`;
  batched-by-shape variant `_prepare_final_fork_batch_rows_batched_preprocess` (`server.py:5827`);
  `_preprocess_fixed_audio` (`server.py:2659`); constant plan sizes for `final_padding_frames` (`server.py:1474-1482`).
- Finalize timing struct: `_continuous_finalize_timing` (`server.py:5430`) — currently vad_stop / debounce /
  fork_flush / final_sent / inference_lock_acquire_wait_ms; emitted as `finalize_timing` (`server.py:5664`, `6398`).
- Existing finalize-batching flags (byte-exact, default-off): `NEMOTRON_BATCH_FINALIZE`,
  `NEMOTRON_BATCH_FINALIZE_PREPROC` (`server.py:570-575`).

## Rules
### Correctness (hard gates — every step)
- **Byte-exact** per-stream final transcripts AND final deltas vs the current server (graph-off / flag-off). This
  is the cache-aware + fork-state corruption hazard; gate before any commit. rc1 English byte-identical.
- **Default-off identity**: every new flag unset → byte-identical to the current server (code fully bypassed).
- **Fail-closed**: any capture/replay/shape error on any finalize bucket → eager fallback; the server always serves.
### Safety & scope
- **No padding** in finalize graphs — capture EXACT `(B, T, drop_extra)` buckets so `graph==eager` stays a clean
  gate; uncaptured buckets → eager.
- **Decode stays eager** (`use_cuda_graph_decoder=False`, Blackwell-safe). Ada-only decoder graph is OUT of scope.
- **Do NOT change** fork semantics or rc1 padding (`server.py:5288-5338`, `5340-5401`) — byte-exact-unsafe.
- **Debounce bypass OUT of scope** (N/A for silence0 / `FINALIZE_SILENCE_MS=0`).
- Compose with the steady cudagraph + lanes + MPS; finalize graphs are per-replica + per-lane-stream and ADD to
  the steady graph pool — watch memory, fail-closed if it doesn't fit.
- No new heavy deps; <400 ms TTFS budget; flag-gated; default = current behavior until proven.
### Test protocol (per step)
- Local: byte-exact gate (graph-on==graph-off final transcripts + deltas, rc1, `FORK_ASSERT=1`,
  step4_byte_exact_canary.sh style) + default-off identity + graphs-actually-engage (replay counters, 0 silent
  fallbacks on captured buckets).
- Cloud: TTFS retest via `bench_client_wan.sh` (L40S + L4, conc 10, full 1000) — confirm the server-finalize P95
  (~178 ms L40S) drops + the client TTFS P95 moves toward the frontier. ALWAYS terminate boxes.

## Steps

- [ ] **1. Finalize-tail attribution telemetry (default-off, profiling-gated).**
  Extend `_continuous_finalize_timing` (`server.py:5430`) + the finalize emit (`server.py:5664`, `6398`) to split
  the finalize wall-time, behind a profiling flag (e.g. `NEMOTRON_FINALIZE_PROFILE=1`): scheduler-queue wait,
  debounce wait, lane/inference-lock wait, `fork_clone_ms` split (audio/cache/hyps/pred), final preprocessor
  wall-time + invocation count, final model-call wall-time, the final `(B, T, drop_extra)`, decode share if
  splittable, and CUDA-sync time. Emit a **final-`(B, T, drop_extra)` histogram** (aggregate log). Non-invasive:
  no unconditional `torch.cuda.synchronize()` in the default path; use CUDA events / guard syncs behind the flag.
  DELIVERABLE: a short run (local + a conc-10 cloud sample) that (a) confirms the P95 tail is the eager finalize
  encoder (vs clone / sync / lock), and (b) gives the final-`(B, T)` distribution that sizes Step 2's bucket set.
  Gate: default-off identity (flag unset → unchanged); the profile path adds no measurable overhead when off.
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-22-1353/finalize-telemetry.md`

- [ ] **2. Finalize-bucket graph manager + standalone byte-exact test.**
  Extend `BucketedCudaGraphEncoder` (or a `FinalizeBucketedCudaGraphEncoder` sibling in `cudagraph_encoder.py`)
  to capture exact finalize buckets keyed by `(B, T, drop_extra, keep_all_outputs=True)`. Reuse the static-buffer
  record/replay + fail-closed `replay→None` pattern; the buffers differ only in `T` (the mel `[B,F,T]` and the
  encoder output length for `keep_all_outputs=True`). `warmup(model, buckets)` captures a provided list of
  `(B, T)` buckets (from Step 1's histogram). Unit test (`tests/test_cudagraph_finalize_encoder.py`): for each
  captured `(B, T)`, assert `graph == eager` byte-exact (encoded + state `max_abs==0`) for `keep_all_outputs=True`,
  built from real clips whose final tail lands on that `T`; assert an uncaptured `(B, T)` returns the use-eager
  sentinel. No `server.py` wiring yet. Gate: byte-exact for every captured bucket.
  Key files: `src/nemotron_speech/cudagraph_encoder.py`, `tests/test_cudagraph_finalize_encoder.py`

- [ ] **3. Wire finalize graphs into the server + local byte-exact gate at scale.**
  Behind `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1` (default off; independent of steady `NEMOTRON_ENCODER_CUDAGRAPH`),
  capture the finalize buckets at startup (per replica + per lane-stream, mirroring the steady managers) and add a
  finalize-graph branch in `_conformer_stream_step`: when the call is a finalize call (`keep_all_outputs=True`,
  steady-bucket gate not matched) AND the `(B, T, drop_extra)` is captured → route `cache_aware_stream_step` to the
  finalize-graph replay (clone outputs as the steady path does); else eager. Per-lane-stream selection + fail-closed
  exactly as the steady path. Local hard gate: stream a fixed multi-stream clip set with the flag on vs off
  (+ scheduler/batching on), diff **final transcripts AND final deltas** byte-identical (lanes=1 AND lanes=2),
  `FORK_ASSERT=1` clean, default-off == pre-change, replay counters > 0 on captured finalize buckets, eager
  fallback on the long-T tail. Adapt `step4_byte_exact_canary.sh`.
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-22-1353/step3_finalize_canary.sh`

- [ ] **4. Cloud TTFS retest (L40S + L4, conc 10) + memory fit.**
  Confirm at startup the finalize buckets capture and FIT alongside the steady graph pool (per replica × lanes ×
  processes) on the 24 GB L4 and 48 GB L40S; fail-closed if not (and record what was captured/skipped). Run
  `bench_client_wan.sh` on g6e.8xlarge (L40S) and g6.4xlarge (L4), full 1000 @ conc 10, **finalize-graph ON vs
  OFF** (steady graph ON in both), and compare: server-finalize P95 (target: cut the ~178 ms L40S / ~224 ms L4
  tail) and client TTFS median/P95 vs the leaderboard. Re-run the Step-1 telemetry on-box to attribute the change.
  Write results into `cloud-retest.md` (+ a finalize section) and the leaderboard. ALWAYS terminate boxes.
  Key files: `ec2-bench/bench_client_wan.sh`, `ec2-bench/start_prod_server.sh`, `proj-2026-05-22-1353/finalize-cloud-retest.md`

- [ ] **5. One-shot final preprocessor (default-off, byte-exact).**
  `NEMOTRON_FINALIZE_SINGLE_PREPROC=1`: replace the per-16-frame-slice `_preprocess_fixed_audio` loop on the final
  tail (serial `server.py:7254-7283`; batch-row `server.py:5715-5744`) with ONE `_preprocess_fixed_audio` over the
  whole pending+padding tail, slicing `remaining_frames` from `first_preprocess_mel_frame` (the constant plan
  already sizes for `final_padding_frames`, `server.py:1474-1482`). Preserve the output contract (emitted_frames==0
  → no mel ring + drop_extra=0; else prepend mel ring + drop_extra=self.drop_extra). BYTE-EXACT RISK: a prior
  batched-final-preproc attempt dropped terminal punctuation (`round4-finalize-preproc.md`) — gate on byte-identical
  mel hashes AND final transcripts including punctuation-heavy clips. Only proceed if Step-1 telemetry shows final
  preprocessing is a material share of the tail; otherwise mark deferred.
  Key files: `src/nemotron_speech/server.py`

- [ ] **6. Verify + canary finalize-batching flags in the production config.**
  Confirm `NEMOTRON_BATCH_FINALIZE` + `NEMOTRON_BATCH_FINALIZE_PREPROC` are set in the production launch path
  (`deploy/`, `ec2-bench/start_prod_server.sh`) — they're byte-exact local-validated and harden the tail when
  finals contend on a pinned lane (vs the global-exclusive fallback). Add them to the prod config + a short L40S
  conc-10 canary (finalize-graph ON) confirming no byte-exactness regression and the P95 holds/improves.
  Key files: `ec2-bench/start_prod_server.sh`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Finalize-tail attribution telemetry | pending | — | confirm tail = eager finalize encoder + final-(B,T) histogram |
| 2 | Finalize-bucket graph manager + unit test | pending | — | extend BucketedCudaGraphEncoder; keep_all_outputs=True, variable T; byte-exact per bucket |
| 3 | Wire into server + local byte-exact gate | pending | — | NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE; final transcripts+deltas graph-on==off, lanes1+2 |
| 4 | Cloud TTFS retest (L40S+L4) + memory fit | pending | — | finalize-graph on vs off; server-finalize P95 drop; bucket pool fits |
| 5 | One-shot final preprocessor | pending | — | NEMOTRON_FINALIZE_SINGLE_PREPROC; byte-exact (punctuation); gated on Step-1 finding |
| 6 | Verify + canary finalize-batching flags | pending | — | NEMOTRON_BATCH_FINALIZE(_PREPROC) in prod config + L40S canary |
