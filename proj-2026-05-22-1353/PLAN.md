# Plan: Finalize-latency optimization — cut the P95 tail toward the streaming frontier

Project directory: `./proj-2026-05-22-1353`

## Context
Client-side TTFS (end-of-speech → final transcript), measured full-1000 @ conc 10 over the internet to a
production-config us-west-2 box (cudagraph ON, lanes=2, silence0_warm200, rc1): **L40S median 274 / P95 401 ms**
(L4 290/447). Decomposed: ~200 ms harness trailing-silence window (benchmark constant) + ~23 ms network RTT
(measured, stable) + **~51 ms median / ~178 ms P95 server-side finalize compute = the target.** The median is
already frontier-competitive (Deepgram 247, Soniox 249); the **P95 tail** is the gap (Deepgram 298 / Soniox 281).

Root cause hypothesis (to be CONFIRMED by Step 1, not assumed): the one model call on the finalize critical path,
`_conformer_stream_step(... keep_all_outputs=True ...)`, runs **eager** — it bypasses the steady CUDA-graph — so
it pays a full kernel-launch storm whose variance is a likely P95 driver. **IMPORTANT framing correction (review
R1):** the ~178 ms P95 was measured by `bench_client_wan.sh`, which runs **ONE server process (lanes=2, no MPS)**.
So at conc-10-one-process the tail driver is eager-launch + 2-lane-stream contention + GC/scheduling — NOT MPS
bifurcation. The steady-graph MPS result (K=4 p95 349-767 → 216) is an *analogy* (graphs collapse launch variance),
not the same scenario. The production multi-process+MPS finalize tail is a *separate, untested* regime — Step 5
measures both. This plan graphs the finalize encoder call (+ supporting levers), flag-gated and byte-exact.

## Reference implementations
- `src/nemotron_speech/cudagraph_encoder.py` — `BucketedCudaGraphEncoder`: per-B static-buffer record/replay +
  fail-closed manager (steady bucket only). The finalize-graph extends this to buckets keyed by `(B, T,
  drop_extra, keep_all_outputs=True, first/non-first)` with variable final `T`.
- Steady CUDA-graph wiring in `server.py` is the pattern to mirror — but note its couplings (see Current state):
  `_conformer_stream_step`, `_encoder_cudagraph_bucket_for_call` (`1828`), `_cudagraph_encoder_cache_step_installed`,
  the per-replica + per-lane-stream managers, the cudagraph **executor/thread** (`_run_inference_call`, `~2050`),
  the prompted-model disable (`~1659`), per-replica capture on the lane stream (`~1681`, `~1957`).
- Byte-exact gate methodology: `proj-2026-05-21-1959-cudagraph/step4_byte_exact_canary.sh` (graph-on==graph-off,
  lanes=1 AND lanes=2; FORK_ASSERT clean).
- Cloud retest: `ec2-bench/bench_client_wan.sh` (one-process WAN) + `ec2-bench/run_l4_ttfs_sweep.sh` /
  `ec2-bench/run_multiproc.sh` (multi-process+MPS) + `start_prod_server.sh`.
- Encoder/decode split profiling precedent: `proj-2026-05-20-modal-cost/profile_split.py:56` (manual encoder-then-
  decode split — whole-call CUDA events do NOT cleanly split CPU-heavy decode).
- Analysis: `proj-2026-05-21-1959-cudagraph/finalize-optimization-suggestions.md`; data: `.../cloud-retest.md`.
- Divergence from steady graph: finalize buckets need **variable T** + **keep_all_outputs=True** → a *separate*
  bucket family + flag, never a steady-graph reuse (different encoder output shape + downstream contract).

## Current state (exact code, confirmed)
- Finalize model call (EAGER target), TWO call sites — wire BOTH: batched `_process_final_batch_rows` →
  `keep_all_outputs=True` at `server.py:6128`; serial `_process_final_chunk` (`7226`) → `7310`.
- Gate that excludes it: `_encoder_cudagraph_bucket_for_call` (`1828`) returns None when `keep_all_outputs` truthy
  (`1835`) and when `chunk_frames != steady_T`.
- **Executor/flag coupling (R1-C3):** `_run_inference_call` routes non-lane inference through the cudagraph
  executor ONLY when `encoder_cudagraph_enabled` (`server.py:~2050`); captures are thread/stream-sensitive
  (`~1681`, `~1957`). A finalize-graph independent of the steady flag would have no executor/thread → must either
  require steady infra or stand up its own.
- **Prompted-model disable:** steady cudagraph is skipped for `prompted_model` (`server.py:~1659`) — carry forward.
- Finalize geometry: `final_padding_frames = (right_context+1)*shift_frames` (`1450`); rc1+16-shift → 32 frames.
  warm200 → "non-first" branch prepends the mel ring + `drop_extra=self.drop_extra`; final `T≈42..58` (CONFIRM in
  Step 1). NeMo output length depends on `keep_all_outputs` + pre-encode drop + cache-present
  (`nemo .../conformer_encoder.py` output-clamp ~567, pre-encode-drop ~657) — so the bucket KEY must capture
  first/non-first + drop + cache-present, not just (B,T).
- Fork (do NOT change): `_build_continuous_finalize_fork` (`5288`), `fork_clone_ms` (`5472`/`6272`), fork asserts
  (`5340-5401`). The finalize runs on a **disposable fork** → a wrong graph corrupts only that utterance's emitted
  final, NOT the live stream (no cross-talk) — lower blast radius than the steady graph.
- Final preprocessing loop (per-16-frame slice): serial `7254-7283`; batch-row `5715-5744`; by-shape variant
  `_prepare_final_fork_batch_rows_batched_preprocess` (`5827`); `_preprocess_fixed_audio` (`2659`); constant plan
  sizes for `final_padding_frames` (`1474-1482`).
- Finalize timing: `_continuous_finalize_timing` (`5430`), emitted as `finalize_timing` (`5664`,`6398`) — currently
  only vad_stop/debounce/fork_flush/final_sent/lock_wait, attached to the client-visible final (empty/suppressed
  finals — `5655`,`6389` — currently contribute no telemetry).
- Existing finalize-batching flags (byte-exact, default-off): `NEMOTRON_BATCH_FINALIZE`,
  `NEMOTRON_BATCH_FINALIZE_PREPROC` (`570-575`). NOT set in `start_prod_server.sh`; `deploy/launch_multiproc.sh`
  sets BATCH_FINALIZE but not PREPROC — so the measured 178 ms was WITHOUT them.

## Rules
### Correctness (hard gates — every step)
- **Byte-exact** vs the current server (flag-off): compare the **full final event stream** — final count, final
  text, final delta, empty-final suppression, ordering — not just the concatenated transcript. rc1 English
  byte-identical. FORK_ASSERT clean.
- **Default-off identity**: each new flag unset → byte-identical to the current server.
- **Fail-closed**: any capture/replay/shape/key-mismatch on any finalize bucket → eager fallback.
- **Prompted/multilingual**: disable finalize graphs for `prompted_model` (mirror the steady disable) until a
  separate prompted canary exists.
### Safety & scope
- **No padding** in finalize graphs (exact `(B,T,...)` buckets keep `graph==eager` clean); uncaptured → eager.
  Padded-bucket capture is a *deferred* fallback only if exact coverage can't fit memory.
- **Decode stays eager** (Blackwell-safe). Ada-only decoder graph OUT of scope.
- **Do NOT change** fork semantics or rc1 padding (byte-exact-unsafe). Debounce bypass OUT of scope (silence0).
- **Bucket budget** (hard): after Step-1 histogram, capture only the top-N `(B,T,...)` covering ≥ a coverage
  target; enforce a GPU-memory headroom limit + a capture-time budget; record per-manager allocated/reserved
  before/after; partial-capture is allowed (capture what fits, eager the rest). Finalize buckets ADD to the steady
  pool per replica × lanes × processes.
- No new heavy deps; <400 ms TTFS budget; flag-gated; default = current behavior.
### Test protocol (per step)
- Local: full-final-event-stream byte-exact (flag-on==off, lanes=1 AND lanes=2, FORK_ASSERT=1) + default-off
  identity + graphs-engage (replay counters, fallback counts) + first-final cold-latency check.
- Cloud: `bench_client_wan.sh` (one-process WAN) AND a multi-process+MPS retest (run_multiproc/run_l4_ttfs_sweep),
  L40S + L4, conc 10, finalize-graph ON vs OFF (steady ON in both); re-run Step-1 telemetry on-box. ALWAYS terminate.

## Steps

- [ ] **1. Finalize-tail telemetry + bucket-key characterization + DECISION GATE.**
  Behind `NEMOTRON_FINALIZE_PROFILE=1` (default-off, non-invasive — CUDA events / guarded syncs, no unconditional
  `torch.cuda.synchronize()` in the default path): split the finalize wall-time into scheduler-queue wait, debounce
  wait, lane/inference-lock wait, `fork_clone_ms` split (audio/cache/hyps/pred), final preprocessor wall-time +
  count, final **encoder** vs **decode** (via a profiling-only encoder wrapper, NOT whole-call events — see
  `profile_split.py`), and CUDA-sync time. Per final, log the full bucket key: `B, T, drop_extra, first/non-first,
  encoded shape, encoded_len, cache shapes, cache-present, att-context`. Emit an aggregate **histogram including
  empty/suppressed finals** (server-side log, not only client-visible finals). Run under BOTH topologies
  (one-process lanes=2; multi-process+MPS) and BOTH finalize-batching configs (decide the production BATCH_FINALIZE
  setting here, before sizing buckets). **DECISION GATE (explicit GO/PIVOT):** proceed to Steps 2-5 (graphs) ONLY
  if the eager finalize encoder is a dominant share of the P95 tail (set a threshold, e.g. ≥ ~40%); else PIVOT to
  the dominant component (clone/sync/lock/GC) and re-rank. Deliverable: `finalize-telemetry.md` with the breakdown,
  the `(B,T,...)` histogram, the topology/batching comparison, and the GO/PIVOT decision.
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-22-1353/finalize-telemetry.md`

- [ ] **2. Single-bucket finalize-graph feasibility probe (de-risk before the manager).**
  Standalone probe (no server.py wiring), mirroring the round-5 steady probe: pick ONE common finalize bucket from
  Step-1, capture a CUDA graph of `cache_aware_stream_step(keep_all_outputs=True)` at that exact `(B,T,drop_extra,
  first/non-first)`, and prove `graph == eager` byte-exact (encoded + encoded_len + state `max_abs==0`) on real
  fork states. CONFIRM the key fully determines the shapes: encoded length is deterministic given the key, cache
  shapes are fixed, no hidden variation from the mel-ring prepend / pre-encode drop / att-context. Gate the rest of
  the plan on this probe passing; if the key is insufficient, expand it here.
  Key files: `proj-2026-05-22-1353/probe_finalize_bucket.py`

- [ ] **3. Finalize-bucket graph manager + standalone test.**
  Extend `BucketedCudaGraphEncoder` (or a `FinalizeBucketedCudaGraphEncoder` sibling) for finalize buckets keyed by
  the Step-2 key, `keep_all_outputs=True`, per-T static output buffers. `warmup(model, buckets)` captures a provided
  bucket list (the Step-1 top-N within the memory/capture budget; partial-capture OK). Unit test
  (`tests/test_cudagraph_finalize_encoder.py`): build **synthetic fork states** for broad shape coverage (per
  Codex#11 — robust vs needing a real clip per T) PLUS a few real-clip end-to-end cases; assert `graph==eager`
  byte-exact per captured bucket and a clean use-eager sentinel for uncaptured. Record capture memory + time.
  Key files: `src/nemotron_speech/cudagraph_encoder.py`, `tests/test_cudagraph_finalize_encoder.py`

- [ ] **4. Wire finalize graphs into the server + executor resolution + local byte-exact gate at scale.**
  `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1` (default off). **Resolve the executor coupling (R1-C3):** either require
  the steady cudagraph infra (executor/thread/managers) when finalize-graph is on, OR stand up a finalize-specific
  executor; explicitly test `finalize=on, steady=off`. Capture finalize buckets at startup per replica + per
  lane-stream. Add a finalize branch in `_conformer_stream_step` covering BOTH call sites (batched `6128` + serial
  `7310`): if `keep_all_outputs=True` AND `(B,T,...)` captured → finalize-graph replay (clone outputs); else eager.
  Disable for `prompted_model`. Bound startup capture time + warm long-T finals per lane (first-final cold check).
  Hard gate: full-final-event-stream byte-identical flag-on==off (lanes=1 AND lanes=2), FORK_ASSERT clean,
  default-off==pre-change, replay>0 on captured buckets, eager on the long-T tail. Adapt `step4_byte_exact_canary.sh`.
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-22-1353/step4_finalize_canary.sh`

- [ ] **5. Cloud TTFS retest — BOTH topologies (one-proc WAN + multi-proc+MPS), L40S + L4.**
  Confirm finalize buckets capture + FIT alongside the steady pool (per replica × lanes × processes) on 24 GB L4 /
  48 GB L40S; record allocated/reserved; fail-closed + log skipped buckets if not. Retest finalize-graph ON vs OFF
  (steady ON both): (a) one-process WAN via `bench_client_wan.sh` (the apples-to-apples client TTFS) and (b)
  multi-process+MPS (production scaling) via `run_multiproc`/`run_l4_ttfs_sweep`. Compare server-finalize P95 (target:
  cut ~178 ms L40S / ~224 ms L4) + client TTFS median/P95 vs the leaderboard; re-run Step-1 telemetry on-box to
  attribute the change. Write `finalize-cloud-retest.md` + update the leaderboard. ALWAYS terminate boxes.
  Key files: `ec2-bench/bench_client_wan.sh`, `ec2-bench/run_multiproc.sh`, `proj-2026-05-22-1353/finalize-cloud-retest.md`

- [ ] **6. One-shot final preprocessor (default-off, byte-exact) — gated on Step-1.**
  ONLY if Step-1 telemetry shows final preprocessing is a material share of the tail (else mark deferred).
  `NEMOTRON_FINALIZE_SINGLE_PREPROC=1`: replace the per-16-frame-slice `_preprocess_fixed_audio` loop (serial
  `7254-7283`; batch-row `5715-5744`) with ONE call over the whole pending+padding tail, slicing `remaining_frames`
  out (constant plan already sizes for `final_padding_frames`, `1474-1482`). Preserve the output contract
  (emitted_frames==0 → no mel ring + drop_extra=0; else prepend + drop_extra=self.drop_extra). BYTE-EXACT RISK
  (prior attempt dropped terminal punctuation — `round4-finalize-preproc.md`): gate on byte-identical mel hashes +
  final transcripts incl punctuation-heavy clips.
  Key files: `src/nemotron_speech/server.py`

- [ ] **7. Finalize-batching production config + canary.**
  Per Step-1's decision, set `NEMOTRON_BATCH_FINALIZE` (+ `_PREPROC` if it wins) in the production launch path
  (`ec2-bench/start_prod_server.sh`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`) — they're byte-exact
  local-validated and harden the tail when finals contend on a pinned lane (vs the global-exclusive fallback).
  L40S conc-10 canary (finalize-graph ON) confirming no byte-exact regression + P95 holds/improves.
  Key files: `ec2-bench/start_prod_server.sh`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Telemetry + bucket-key chars + DECISION GATE | pending | — | both topologies + batching configs; GO/PIVOT if eager-encoder not dominant |
| 2 | Single-bucket feasibility probe | pending | — | de-risk variable-T keep_all_outputs=True capture; confirm the bucket key |
| 3 | Finalize-bucket graph manager + test | pending | — | extend BucketedCudaGraphEncoder; synthetic-fork + real-clip; bucket budget |
| 4 | Wire + executor resolution + byte-exact gate | pending | — | both call sites; finalize=on/steady=off; prompted disabled; full-event gate |
| 5 | Cloud retest — both topologies | pending | — | one-proc WAN + multi-proc+MPS; L40S+L4; memory fit; P95 drop |
| 6 | One-shot final preprocessor | pending | — | gated on Step-1; byte-exact (punctuation) |
| 7 | Finalize-batching prod config + canary | pending | — | BATCH_FINALIZE(_PREPROC) per Step-1; L40S canary |

## Review log
- **Round 1 (Codex `bfqf2k2ug` + self):** CRITICAL — (C1) MPS-framing wrong, the 178 ms was one-process/no-MPS →
  fixed Context + Step 5 tests both topologies; (C2) added the explicit GO/PIVOT decision gate after Step 1; (C3)
  finalize-graph can't be flag-independent of steady (shared executor/thread) → Step 4 resolves it + tests
  finalize=on/steady=off. MAJOR — concrete bucket budget (Rules); added Step 2 single-bucket feasibility probe;
  richer bucket key (first/non-first, drop, cache-present, encoded shape); full-final-event-stream gate; noted the
  disposable-fork reduced blast radius. ORDERING — moved the BATCH_FINALIZE decision into Step 1 (the 178 ms was
  measured without it). MINOR — encoder/decode split needs a profiling wrapper; cold-start/first-final warmup
  bound; synthetic-fork test; prompted-model disable; padded-bucket deferred fallback; empty/suppressed finals in
  telemetry. (Rounds 2-3 pending.)
