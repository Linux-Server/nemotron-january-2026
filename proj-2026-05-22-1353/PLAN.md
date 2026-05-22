# Plan: Finalize-latency optimization — cut the P95 tail toward the streaming frontier

Project directory: `./proj-2026-05-22-1353`

## Context
Client-side TTFS (end-of-speech → final transcript), measured full-1000 @ conc 10 over the internet to a
production-config us-west-2 box (cudagraph ON, lanes=2, silence0_warm200, rc1): **L40S median 274 / P95 401 ms**
(L4 290/447). Decomposed: ~200 ms harness trailing-silence window (benchmark constant) + ~23 ms network RTT
(measured, stable) + **~51 ms median / ~178 ms P95 server-side finalize compute = the target.** Median is already
frontier-competitive (Deepgram 247, Soniox 249); the **P95 tail** is the gap (Deepgram 298 / Soniox 281).

Root-cause hypothesis (CONFIRM in Step 1, do not assume): the finalize model call
`_conformer_stream_step(... keep_all_outputs=True ...)` runs **eager** (bypasses the steady CUDA-graph), paying a
full kernel-launch storm whose variance is a likely P95 driver. **Framing (R1):** the ~178 ms P95 was measured by
`bench_client_wan.sh` = **ONE process, lanes=2, no MPS**; so at conc-10-one-process the tail driver is eager-launch
+ 2-lane-stream contention + GC/scheduling, NOT MPS bifurcation (the steady-graph K=4/MPS 349-767→216 result is an
*analogy* — graphs collapse launch variance — not the same regime). Multi-process+MPS finalize is a separate,
untested regime — Step 5 measures both. Flag-gated, byte-exact, fail-closed.

## Reference implementations
- `src/nemotron_speech/cudagraph_encoder.py` — `BucketedCudaGraphEncoder` (per-B static-buffer record/replay +
  fail-closed manager, steady only). The finalize-graph extends it to buckets keyed by `(B, T, drop_extra,
  first/non-first)` with `keep_all_outputs=True` + variable `T`.
- Steady wiring + its COUPLINGS (see Current state): `_conformer_stream_step`, `_encoder_cudagraph_bucket_for_call`
  (`1828`), `_cudagraph_encoder_cache_step_installed`, the cudagraph executor/thread (`_run_inference_call ~2050`),
  per-replica/per-lane managers (capture on the lane stream, `~1681`/`~1726`/`~1957`/`~2128`), the manager
  completeness check (`~1592`), the prompted-model disable (`~1659`).
- Byte-exact gate: `proj-2026-05-21-1959-cudagraph/step4_byte_exact_canary.sh`.
- Cloud retest: `ec2-bench/bench_client_wan.sh` (one-proc WAN) + `run_multiproc.sh`/`run_l4_ttfs_sweep.sh`
  (multi-proc+MPS) + `start_prod_server.sh`.
- Encoder/decode split precedent: `proj-2026-05-20-modal-cost/profile_split.py:56` (manual split — whole-call CUDA
  events do NOT cleanly split CPU-heavy decode).
- Analysis: `proj-2026-05-21-1959-cudagraph/finalize-optimization-suggestions.md`; data: `.../cloud-retest.md`.
- Divergence: finalize buckets need variable T + `keep_all_outputs=True` → a *separate* bucket family + flag,
  never a steady-graph reuse.

## Current state (exact code, confirmed)
- Finalize model call (EAGER target), TWO call sites — wire BOTH: batched `_process_final_batch_rows` →
  `keep_all_outputs=True` `server.py:6128`; serial `_process_final_chunk` (`7226`) → `7310`.
- Excluding gate: `_encoder_cudagraph_bucket_for_call` (`1828`) → None when `keep_all_outputs` truthy (`1835`) or
  `chunk_frames != steady_T`.
- **Executor/flag coupling (R1/R2 CRITICAL):** the cudagraph executor + default manager are created ONLY inside
  steady `_configure_encoder_cudagraph()` when `NEMOTRON_ENCODER_CUDAGRAPH=1` (`~1654`/`~1681`); `_run_inference_call`
  routes through it only when `encoder_cudagraph_enabled` (`~2050`); lane graphs capture+replay on the lane stream
  via the lane executor (`~1726`/`~2128`). → a finalize graph independent of steady has NO executor/thread/manager.
- **Manager completeness (R2):** steady disables a manager if ANY requested bucket is uncaptured (`~1592`) — the
  finalize manager must NOT inherit this (it has many buckets; one OOM must not kill all).
- **Prompted-model disable** (`~1659`) — carry forward.
- Finalize geometry: `final_padding_frames=(right_context+1)*shift_frames` (`1450`); rc1+16-shift → 32 frames.
  warm200 → "non-first" branch prepends mel ring + `drop_extra=self.drop_extra`; final `T≈42..58` (CONFIRM Step 1).
  NeMo output length depends on `keep_all_outputs` + pre-encode drop (only when cache present) + first/non-first
  (`conformer_encoder.py` output-clamp ~542/567, pre-encode-drop ~638/657) → the bucket KEY must include
  first/non-first + drop + cache-present, and Step 2 must prove the key fully determines shapes.
- **Fork blast radius (R2 — claim downgraded):** `_build_continuous_finalize_fork` (`5288`) clones acoustic/decoder
  state (cache/hyps/pred/audio), and the fork asserts (`5340-5401`) guard the parent's *acoustic* state. BUT the
  chosen `final_text` is written back to the parent `committed_text`/`last_emitted_text` (`5652`/`6386`) and can
  advance `continuous_emitted_text` (`5671`); speculative finalize keeps ASR state (`6427`). So the fork isolates
  *acoustic/decoder mutation, NOT emitted-text state* — a wrong final affects future delta-suppression/corrections
  in the SAME live stream → byte-exact tests MUST span MULTI-FINAL continuous sessions, not one-shot finals.
- **Shared graph buffers (R2):** the per-replica graph static buffers are shared → finalize replay must be
  SERIALIZED per replica (no concurrent replay on the same buffers; the steady path gets this free via the lane).
- Final preprocessing loop (per-16-frame slice): serial `7254-7283`; batch-row `5715-5744`; by-shape variant
  `_prepare_final_fork_batch_rows_batched_preprocess` (`5827`); `_preprocess_fixed_audio` (`2659`); constant plan
  sizes for `final_padding_frames` (`1474-1482`).
- Finalize timing: `_continuous_finalize_timing` (`5430`), emitted `finalize_timing` (`5664`/`6398`) — only
  vad_stop/debounce/fork_flush/final_sent/lock_wait; attached to client-visible finals (empty/suppressed finals
  `5655`/`6389` contribute no telemetry today).
- Finalize-batching flags (byte-exact, default-off): `NEMOTRON_BATCH_FINALIZE`, `NEMOTRON_BATCH_FINALIZE_PREPROC`
  (`570-575`); B>1 only for batched `debounce_expired` events stacked in one drain (`3608`/`3459`) → `6073`. NOT set
  in `start_prod_server.sh`; `deploy/launch_multiproc.sh` sets BATCH_FINALIZE not PREPROC → the 178 ms was without.

## Rules
### Correctness (hard gates — every step)
- **Byte-exact** vs the current server (flag-off) over **multi-final continuous sessions**: compare the full final
  event stream — final count, final text, each final delta, the SET of empty/suppressed finals, ordering — AND the
  downstream delta-suppression/correction across subsequent turns. rc1 English byte-identical. FORK_ASSERT clean.
- **Default-off identity**; **fail-closed** (any capture/replay/shape/key mismatch → eager); disable for
  `prompted_model`.
### Safety & scope
- **No padding** (exact buckets keep `graph==eager`); uncaptured → eager. Padded buckets = deferred fallback only.
- **Decode stays eager** (Blackwell-safe; Ada decoder graph OUT of scope unless Step-1 shows decode dominates the
  tail). Do NOT change fork semantics / rc1 padding. Debounce bypass OUT of scope (silence0).
- **Finalize-specific completeness (R2):** manager usable if ≥1 requested bucket captured; each uncaptured bucket
  records a skip reason + falls back eager (do NOT mirror the steady all-or-nothing at `~1592`).
- **Bucket budget:** capture top-N `(B,T,...)` from the Step-1 histogram up to a GPU-memory-headroom limit + a
  capture-time budget; record per-manager allocated/reserved before/after; partial-capture expected. Finalize
  buckets ADD to the steady pool per replica × lanes × processes.
- **Replay serialization:** finalize-graph replay holds the per-replica/lane serialization (no concurrent replay on
  shared static buffers).
- No new heavy deps; <400 ms TTFS budget; flag-gated; default = current behavior.
### Test protocol (per step)
- Local: multi-final continuous-session full-event byte-exact (flag-on==off, lanes=1 AND lanes=2, FORK_ASSERT=1) +
  default-off identity + graphs-engage (replay/fallback counters) + a deterministic **forced-B>1** canary + a
  first-final cold-latency check.
- Cloud: `bench_client_wan.sh` (one-proc WAN) AND multi-proc+MPS (run_multiproc/run_l4_ttfs_sweep), L40S + L4,
  conc 10, finalize-graph ON vs OFF (steady ON both); re-run Step-1 telemetry on-box. ALWAYS terminate.

## Steps

- [ ] **1. Finalize-tail telemetry + bucket-key characterization + COUNTERFACTUAL decision gate.**
  Behind `NEMOTRON_FINALIZE_PROFILE=1` (default-off, non-invasive): record per-final
  `{finalize_wall, queue_wait, debounce_wait, lock_wait, fork_clone[audio/cache/hyps/pred], preproc_wall+count,
  encoder, decode, sync}` — split encoder vs decode via a profiling-only encoder wrapper (NOT whole-call events;
  see `profile_split.py`). Per final also log the full bucket key + shapes: `B, T, drop_extra, first/non-first,
  encoded shape, encoded_len, cache shapes, cache-present, att-context`. Aggregate a **histogram including
  empty/suppressed finals** (server-side). Sample under BOTH topologies (one-proc lanes=2; multi-proc+MPS) AND at
  BOTH conc 10 AND production concurrency (the knee, where finals batch → B>1) and decide the production
  BATCH_FINALIZE config here. **DECISION GATE (counterfactual, on the wall-time TAIL COHORT — top ~5% by
  post-debounce critical-path wall, NOT population-quantile ratios):** proceed to Steps 2-5 ONLY if
  `P95(W) - P95(W - E_eager_encoder + E_graph_est)` clears a concrete threshold (e.g. ≥ ~30 ms), where `E_graph_est`
  is the steady-graph per-call cost. Else PIVOT to the dominant tail component (decode → reconsider Ada decoder
  graph; clone/sync/lock/GC → those levers). Deliverable: `finalize-telemetry.md`.
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-22-1353/finalize-telemetry.md`

- [ ] **2. Finalize-graph feasibility PROBE MATRIX (de-risk before the manager).**
  Standalone (no server.py wiring), mirroring the round-5 steady probe. Capture + prove `graph==eager` byte-exact
  (encoded + encoded_len + state `max_abs==0`, `keep_all_outputs=True`) across a MATRIX, not one bucket:
  {B=1 first-final (drop=0); B=1 non-first short-T; B=1 non-first long/p95-T; ≥1 B>1 non-first if batch-finalize is
  in the production config}. For each key, test **multiple real fork states with different `cache_last_channel_len`**
  sharing that key. CONFIRM the `(B,T,drop_extra,first/non-first)` key fully determines all shapes (no hidden
  variation from mel-ring prepend / pre-encode drop / att-context / cache-present). Gate the rest on this; expand
  the key here if insufficient.
  Key files: `proj-2026-05-22-1353/probe_finalize_bucket.py`

- [ ] **3. Finalize-bucket graph manager + standalone test (partial-capture aware).**
  Extend `BucketedCudaGraphEncoder` (or a `FinalizeBucketedCudaGraphEncoder` sibling) for the Step-2 key,
  `keep_all_outputs=True`, per-T static output buffers. `warmup(model, buckets)` captures a provided list (the
  Step-1 top-N within the memory/capture budget); **finalize-specific completeness** — usable if ≥1 bucket
  captured, per-bucket skip reason + eager fallback (do NOT inherit the steady all-or-nothing). Unit test
  (`tests/test_cudagraph_finalize_encoder.py`): synthetic fork states for broad shape coverage + real-clip
  end-to-end; assert `graph==eager` per captured bucket and a clean use-eager sentinel for uncaptured; record
  capture memory + time per bucket.
  Key files: `src/nemotron_speech/cudagraph_encoder.py`, `tests/test_cudagraph_finalize_encoder.py`

- [ ] **4. Wire into server: executor/stream CONTRACT + both call sites + byte-exact gate at scale.**
  `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1` (default off). **Executor contract (R2 CRITICAL):**
  `CUDAGRAPH_FINALIZE` **requires steady cudagraph infra** (reuse its executor/thread + per-replica/per-lane
  managers); `finalize=on, steady=off` → assert **disabled, no replay** (logged), never silent. Capture finalize
  buckets at startup on the SAME executor/stream used for replay; hard-assert replay runs on the capture
  thread/stream; SERIALIZE replay per replica. Add a finalize branch in `_conformer_stream_step` covering BOTH call
  sites (batched `6128` + serial `7310`): `keep_all_outputs=True` AND key captured → finalize replay (clone
  outputs); else eager. Disable for `prompted_model`. Bound startup capture time + warm the per-lane finalize
  buckets (first-final cold check). HARD GATE — TEST MATRIX `lanes∈{1,2} × batch_finalize∈{on,off} × steady∈{on,off}`
  with expected replay/disable outcomes; multi-final continuous-session full-event byte-identical flag-on==off;
  a deterministic **forced-B>1** canary (synchronized finals → B=2..batch_max_size); FORK_ASSERT clean; default-off
  == pre-change; replay>0 on captured buckets, eager on the long-T/uncaptured tail. Adapt `step4_byte_exact_canary.sh`.
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-22-1353/step4_finalize_canary.sh`

- [ ] **5. Cloud TTFS retest — BOTH topologies (one-proc WAN + multi-proc+MPS), L40S + L4.**
  Confirm finalize buckets capture + FIT alongside the steady pool (per replica × lanes × processes) on 24 GB L4 /
  48 GB L40S; record allocated/reserved; partial-capture + log skips if not. Retest finalize-graph ON vs OFF (steady
  ON both): (a) one-proc WAN via `bench_client_wan.sh`; (b) multi-proc+MPS via `run_multiproc`/`run_l4_ttfs_sweep`
  (plumb the `CUDAGRAPH_FINALIZE` flag through, as was done for `CUDAGRAPH`). Compare server-finalize P95 (cut ~178 ms
  L40S / ~224 ms L4) + client TTFS vs the leaderboard; re-run Step-1 telemetry on-box AND **re-check the (B,T)
  histogram graph-on** (graphs shift drain timing → B distribution → may need bucket-set adjustment). Write
  `finalize-cloud-retest.md` + update the leaderboard. ALWAYS terminate.
  Key files: `ec2-bench/bench_client_wan.sh`, `ec2-bench/run_multiproc.sh`, `proj-2026-05-22-1353/finalize-cloud-retest.md`

- [ ] **6. One-shot final preprocessor (default-off, byte-exact) — gated on Step-1.**
  ONLY if Step-1 shows final preprocessing is a material tail share (else defer). `NEMOTRON_FINALIZE_SINGLE_PREPROC=1`:
  replace the per-16-frame-slice `_preprocess_fixed_audio` loop (serial `7254-7283`; batch-row `5715-5744`) with ONE
  call over the whole pending+padding tail, slicing `remaining_frames` (constant plan sizes for `final_padding_frames`
  `1474-1482`). Preserve the output contract (emitted_frames==0 → no mel ring + drop_extra=0; else prepend +
  drop_extra=self.drop_extra). BYTE-EXACT RISK (prior attempt dropped terminal punctuation —
  `round4-finalize-preproc.md`): gate on byte-identical mel hashes + final transcripts incl punctuation-heavy clips.
  Key files: `src/nemotron_speech/server.py`

- [ ] **7. Finalize-batching production config + canary.**
  Per Step-1's decision, set `NEMOTRON_BATCH_FINALIZE` (+ `_PREPROC` if it wins) in the production launch path
  (`start_prod_server.sh`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`). L40S conc-10 canary (finalize-graph
  ON) — no byte-exact regression + P95 holds/improves.
  Key files: `ec2-bench/start_prod_server.sh`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Telemetry + bucket-key + COUNTERFACTUAL gate | pending | — | per-final records; tail-cohort attribution; counterfactual GO/PIVOT; both topologies + concurrencies |
| 2 | Feasibility PROBE MATRIX | pending | — | first/drop0 + non-first short/long T + B>1; multiple cache_lens per key; confirm key determines shapes |
| 3 | Finalize-bucket manager + test (partial) | pending | — | ≥1-bucket completeness (NOT steady all-or-nothing); synthetic+real; capture mem/time |
| 4 | Wire + executor CONTRACT + gate at scale | pending | — | requires steady infra; finalize=on/steady=off→disabled; serialize replay; both call sites; matrix + forced-B>1 + multi-final gate |
| 5 | Cloud retest — both topologies | pending | — | one-proc WAN + multi-proc+MPS; memory fit; re-check histogram graph-on |
| 6 | One-shot final preprocessor | pending | — | gated on Step-1; byte-exact (punctuation) |
| 7 | Finalize-batching prod config + canary | pending | — | per Step-1; L40S canary |

## Review log
- **Round 1 (Codex `bfqf2k2ug` + self):** fixed MPS-framing (178 ms was one-proc/no-MPS) → both-topology Step 5;
  added GO/PIVOT gate; flagged executor coupling C3; concrete bucket budget; added Step 2 probe; richer key;
  full-event gate; prompted disable; moved BATCH_FINALIZE decision into Step 1.
- **Round 2 (Codex `b50rl9cfu` + self):** CRITICAL — executor/thread/stream **contract** made explicit
  (`CUDAGRAPH_FINALIZE` requires steady infra; `finalize=on/steady=off`→disabled; capture-on-replay-stream +
  assertions; **serialize replay per replica** — covers the shared-buffer race). MAJOR — **counterfactual** decision
  gate on the wall-time tail cohort (not P95-of-ratios); **disposable-fork claim downgraded** (final_text writes back
  to parent emitted-text → multi-final continuous-session byte-exact tests); Step 2 → **probe matrix** (first/non-
  first, short/long T, B>1, multiple cache_lens); **finalize-specific completeness** (≥1 bucket, NOT steady all-or-
  nothing); BATCH_FINALIZE **B>1 forced canary** + sample histogram at production concurrency + re-check post-graph-on.
  R1 framing fixes confirmed to hold. (Round 3 pending.)
