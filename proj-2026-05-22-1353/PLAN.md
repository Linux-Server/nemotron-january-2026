# Plan: Finalize-latency optimization — cut the P95 tail toward the streaming frontier

Project directory: `./proj-2026-05-22-1353`

## Context
Client-side TTFS (end-of-speech → final transcript), measured full-1000 @ conc 10 over the internet to a
production-config us-west-2 box (cudagraph ON, lanes=2, silence0_warm200, rc1): **L40S median 274 / P95 401 ms**
(L4 290/447). Decomposed: ~200 ms harness trailing-silence window (benchmark constant) + ~23 ms network RTT
(measured, stable) + **~51 ms median / ~178 ms P95 server-side finalize compute = the target.** Median is already
frontier-competitive (Deepgram 247, Soniox 249); the **P95 tail** is the gap (Deepgram 298 / Soniox 281).

Root-cause hypothesis (CONFIRM in Steps 1-2, do not assume): the finalize model call
`_conformer_stream_step(... keep_all_outputs=True ...)` runs **eager** (bypasses the steady CUDA-graph), paying a
full kernel-launch storm whose variance is a likely P95 driver. **Framing (R1):** the ~178 ms P95 was measured by
`bench_client_wan.sh` = **ONE process, lanes=2, no MPS**; the tail driver at conc-10-one-process is eager-launch +
2-lane-stream contention + GC/scheduling, NOT MPS bifurcation (the steady-graph K=4/MPS 349-767→216 result is an
*analogy*). Multi-process+MPS finalize is a separate, untested regime — Step 5 measures both. Flag-gated,
byte-exact, fail-closed.

## Reference implementations
- `src/nemotron_speech/cudagraph_encoder.py` — `BucketedCudaGraphEncoder` (per-B static-buffer record/replay +
  fail-closed manager, steady only). Finalize-graph extends it: key `(B, T, drop_extra, first/non-first)`,
  `keep_all_outputs=True`, variable `T`.
- Steady wiring + COUPLINGS (Current state): `_conformer_stream_step`, `_encoder_cudagraph_bucket_for_call`
  (`1828`), `_cudagraph_encoder_cache_step_installed`, cudagraph executor/thread (`_run_inference_call ~2050`),
  per-replica/per-lane managers (capture/replay on lane stream `~1681/1726/1957/2128`), manager completeness
  (`~1592`), prompted-model disable (`~1659`); the mismatch path only WARNS (`~1957`) — finalize must hard-disable.
- Byte-exact gate: `proj-2026-05-21-1959-cudagraph/step4_byte_exact_canary.sh`.
- Cloud retest: `ec2-bench/bench_client_wan.sh` (one-proc WAN; one-utterance/connection) +
  `run_multiproc.sh`/`run_l4_ttfs_sweep.sh` (multi-proc+MPS) + `start_prod_server.sh`. NOTE: NO existing harness
  does multi-final-per-connection or synchronized-B>1 finals — Steps 1/4 must BUILD them (see below).
- Encoder/decode split precedent: `proj-2026-05-20-modal-cost/profile_split.py:56`.
- Analysis: `.../finalize-optimization-suggestions.md`; data: `.../cloud-retest.md`.

## Current state (exact code, confirmed)
- Finalize model call (EAGER target), TWO call sites — wire BOTH: batched `_process_final_batch_rows` →
  `keep_all_outputs=True` `server.py:6128`; serial `_process_final_chunk` (`7226`) → `7310`.
- Excluding gate: `_encoder_cudagraph_bucket_for_call` (`1828`) → None when `keep_all_outputs` truthy (`1835`) or
  `chunk_frames != steady_T`.
- **Executor/flag coupling:** cudagraph executor + managers created ONLY in steady `_configure_encoder_cudagraph()`
  when `NEMOTRON_ENCODER_CUDAGRAPH=1` (`~1654/1681`); `_run_inference_call` routes there only when
  `encoder_cudagraph_enabled` (`~2050`); lane graphs capture/replay on the lane stream via the lane executor
  (`~1726/2128`, lane finalize paths `~5586/6306`). Steady disables a manager if ANY bucket uncaptured (`~1592`).
- **Fork blast radius (downgraded R2):** `_build_continuous_finalize_fork` (`5288`) isolates *acoustic/decoder*
  state, but the chosen `final_text` writes back to parent `committed_text`/`last_emitted_text` (`5652/6386`),
  advances `continuous_emitted_text` (`5671`), and speculative finalize keeps ASR state (`6427`) → a wrong final
  affects future delta-suppression/corrections in the live stream → byte-exact tests MUST span MULTI-FINAL
  continuous sessions. Suppressed/empty finals are NOT sent to the client (`~5680`) → tests capture them server-side.
- **Shared graph buffers:** per-replica static buffers are shared → finalize replay must be SERIALIZED per replica.
- Finalize geometry: `final_padding_frames=(rc+1)*shift` (`1450`); rc1 → 32 frames; warm200 → non-first branch
  prepends mel ring + `drop_extra=self.drop_extra`; `T≈42..58` (confirm). NeMo output length depends on
  `keep_all_outputs` + pre-encode-drop(only-if-cache) + first/non-first (`conformer_encoder.py ~542/638`) → key
  must include first/non-first + drop + cache-present; Step 2 proves it determines all shapes.
- Final preproc loop (per-16-frame): serial `7254-7283`; batch-row `5715-5744`; by-shape `5827`;
  `_preprocess_fixed_audio` `2659`; plan sizes for `final_padding_frames` `1474-1482`.
- Finalize timing: `_continuous_finalize_timing` (`5430`), emitted `5664/6398` — only vad_stop/debounce/fork_flush/
  final_sent/lock_wait; empty/suppressed finals contribute none today.
- Finalize-batching flags (byte-exact, default-off): `NEMOTRON_BATCH_FINALIZE`, `NEMOTRON_BATCH_FINALIZE_PREPROC`
  (`570-575`); B>1 only for batched `debounce_expired` events stacked in one scheduler drain (`3608/3459`) → `6073`.
  NOT in `start_prod_server.sh`; `deploy/launch_multiproc.sh` sets BATCH_FINALIZE not PREPROC → the 178 ms was without.

## Rules
### Correctness (hard gates)
- **Byte-exact** vs flag-off over **MULTI-FINAL continuous sessions**: full final event stream (count/text/each
  delta/the SET of empty+suppressed finals [captured server-side]/ordering) AND downstream delta-suppression across
  later turns. rc1 English byte-identical. FORK_ASSERT clean.
- **Default-off identity**; **fail-closed** (any capture/replay/shape/key/thread-stream mismatch → eager;
  hard-disable, not warn); disable for `prompted_model`.
### Safety & scope
- No padding (exact buckets); uncaptured → eager. Decode stays eager (Ada decoder graph OUT unless Step-1/2 show
  decode dominates). Do NOT change fork semantics / rc1 padding. Debounce bypass OUT (silence0).
- **Finalize-specific completeness:** manager usable if ≥1 bucket captured; per-bucket skip reason + eager (NOT the
  steady all-or-nothing `~1592`). **Bucket budget:** top-N from the histogram up to a memory-headroom + capture-time
  limit; record allocated/reserved; partial expected. **Replay serialized** per replica.
- New test harnesses required (no existing one fits): a same-websocket **multi-final** client + a **forced-B>1**
  scheduler-barrier test (below). No new heavy deps; <400 ms TTFS budget; flag-gated; default = current behavior.

## Steps

- [ ] **1. Finalize-tail telemetry + (B,T) histogram + BATCH_FINALIZE profiling (NO hard gate here).**
  `NEMOTRON_FINALIZE_PROFILE=1` (default-off, non-invasive). Per-final records `{finalize_wall, queue_wait,
  debounce_wait, lock_wait, fork_clone[audio/cache/hyps/pred], preproc_wall+count, encoder, decode, sync}` —
  encoder vs decode split via a profiling-only encoder wrapper (NOT whole-call events; see `profile_split.py`).
  Per-final key + shapes `{B,T,drop_extra,first/non-first,encoded shape/len,cache shapes,cache-present,att-context}`.
  Aggregate histogram **including empty/suppressed finals** (server-side). Profile the matrix:
  `BATCH_FINALIZE ∈ {0,1}` (+ `_PREPROC` if considered) × topology {one-proc lanes2; multi-proc+MPS} × concurrency
  {10; the knee}. Build the **multi-final same-websocket harness** here (prove the continuous protocol works —
  re-arm after finalize keeps state; NOT the reset-then-vad_start that timed out in `ec2_loadgen --rounds`) since
  later byte-exact gates need it. SOFT outcome: report `E_eager_encoder` (median + tail), the tail attribution, the
  `(B,T,...)` histogram per config, and a preliminary "encoder tail is worth probing" yes/no. Choose the production
  BATCH_FINALIZE config. Deliverable: `finalize-telemetry.md`. (The HARD go/pivot gate is Step 2, once E_graph is measured.)
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-22-1353/finalize-telemetry.md`, `.../multifinal_client.py`

- [ ] **2. Feasibility PROBE MATRIX + measured timing + COUNTERFACTUAL GO/PIVOT GATE.**
  Standalone (no server.py wiring), mirroring the round-5 steady probe. Capture + prove `graph==eager` byte-exact
  (encoded + encoded_len + state `max_abs==0`, `keep_all_outputs=True`) across a MATRIX: {B=1 first-final(drop=0);
  B=1 non-first short-T; B=1 non-first long/p95-T; ≥1 B>1 non-first if BATCH_FINALIZE is in the prod config}, each
  with multiple real fork states of differing `cache_last_channel_len` sharing the key — CONFIRM the key fully
  determines all shapes. ALSO measure the per-bucket **`E_finalize_graph`** (synced) vs eager. **HARD GATE
  (counterfactual, on the wall-time tail cohort from Step 1):** proceed to Steps 3-5 ONLY if
  `P95(W) - P95(W - E_eager_encoder + E_finalize_graph)` clears a concrete threshold (e.g. ≥ ~30 ms), using the
  MEASURED `E_finalize_graph` here (not the steady cost) and sanity-checked vs the measured eager cost. Else ABORT
  the graph track and PIVOT to the dominant tail component (decode → reconsider Ada decoder graph; clone/sync/lock/GC).
  Key files: `proj-2026-05-22-1353/probe_finalize_bucket.py`, `proj-2026-05-22-1353/finalize-gate.md`

- [ ] **3. Finalize-bucket graph manager + standalone test (partial-capture aware).**
  Extend `BucketedCudaGraphEncoder` (or a `FinalizeBucketedCudaGraphEncoder` sibling): Step-2 key,
  `keep_all_outputs=True`, per-T static output buffers. `warmup(model, buckets)` captures the Step-1 top-N within
  the memory/capture budget; **finalize-specific completeness** (≥1 bucket usable; per-bucket skip+eager). Unit test
  (`tests/test_cudagraph_finalize_encoder.py`): synthetic fork states (broad shape coverage) + real-clip end-to-end;
  `graph==eager` per captured bucket; clean use-eager sentinel for uncaptured; record capture mem/time/bucket.
  Key files: `src/nemotron_speech/cudagraph_encoder.py`, `tests/test_cudagraph_finalize_encoder.py`

- [ ] **4. Wire into server: executor/stream CONTRACT + both call sites + byte-exact gate at scale.**
  `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1` (default off). **Executor contract:** REQUIRES steady cudagraph infra
  (reuse its executor/thread + per-replica/per-lane managers); `finalize=on, steady=off` → assert **disabled, no
  replay** (logged). Capture finalize buckets at startup on the SAME executor/stream used for replay;
  **HARD-DISABLE finalize replay (fail-closed, not warn) on any thread/stream mismatch** (steady only warns,
  `~1957`); SERIALIZE replay per replica. Finalize branch in `_conformer_stream_step` covering BOTH sites (batched
  `6128` + serial `7310`): `keep_all_outputs=True` AND key captured → replay (clone outputs); else eager. Disable
  for `prompted_model`. Bound startup capture time + warm per-lane finalize buckets (first-final cold check).
  HARD GATE: TEST MATRIX `lanes∈{1,2} × batch_finalize∈{on,off} × steady∈{on,off}` (expected replay/disable);
  **multi-final continuous-session** full-event byte-identical flag-on==off (using the Step-1 multi-final harness);
  a **forced-B>1 scheduler-barrier** canary — a test-only hold flag that queues N `debounce_expired` events before
  waking the scheduler (concurrent clients alone may drain B=1), asserting the effective-batch histogram has
  B=2..batch_max_size and finalize replay counters increment; FORK_ASSERT clean; default-off==pre-change.
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-22-1353/step4_finalize_canary.sh`

- [ ] **5. Cloud TTFS retest — BOTH topologies (one-proc WAN + multi-proc+MPS), L40S + L4.**
  Confirm finalize buckets capture + FIT alongside the steady pool (per replica × lanes × procs) on 24 GB L4 /
  48 GB L40S; record allocated/reserved; partial-capture + log skips. Retest finalize-graph ON vs OFF (steady ON
  both): (a) one-proc WAN `bench_client_wan.sh`; (b) multi-proc+MPS via `run_multiproc`/`run_l4_ttfs_sweep` (plumb
  the `CUDAGRAPH_FINALIZE` flag through, as for `CUDAGRAPH`). Compare server-finalize P95 (cut ~178 ms L40S /
  ~224 ms L4) + client TTFS vs the leaderboard; re-run Step-1 telemetry on-box AND **re-check the (B,T) histogram
  graph-on** (graphs shift drain timing → B distribution → may adjust the bucket set). Write
  `finalize-cloud-retest.md` + update leaderboard. ALWAYS terminate.
  Key files: `ec2-bench/bench_client_wan.sh`, `ec2-bench/run_multiproc.sh`, `proj-2026-05-22-1353/finalize-cloud-retest.md`

- [ ] **6. (Optional, non-blocking) One-shot final preprocessor — gated on Step-1.**
  Independent of the graph track; do NOT let it block Steps 2-5. ONLY if Step-1 shows final preprocessing is a
  material tail share. `NEMOTRON_FINALIZE_SINGLE_PREPROC=1`: replace the per-16-frame-slice loop (serial `7254-7283`;
  batch-row `5715-5744`) with ONE `_preprocess_fixed_audio` over the whole pending+padding tail, slicing
  `remaining_frames` (plan sizes for `final_padding_frames` `1474-1482`); preserve the output contract. BYTE-EXACT
  RISK (prior attempt dropped terminal punctuation — `round4-finalize-preproc.md`): gate on byte-identical mel
  hashes + final transcripts incl punctuation-heavy clips.
  Key files: `src/nemotron_speech/server.py`

- [ ] **7. Finalize-batching production config + canary.**
  Per Step-1's decision, set `NEMOTRON_BATCH_FINALIZE` (+ `_PREPROC` if it wins) in the prod launch path
  (`start_prod_server.sh`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`). L40S conc-10 canary
  (finalize-graph ON) — no byte-exact regression + P95 holds/improves.
  Key files: `ec2-bench/start_prod_server.sh`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Telemetry + histogram + BATCH_FINALIZE profiling + multi-final harness | pending | — | NO hard gate; E_eager + tail attribution + (B,T) per config; build multi-final client |
| 2 | Probe matrix + measured E_graph + COUNTERFACTUAL gate | pending | — | byte-exact + timing; hard GO/PIVOT/ABORT with measured numbers |
| 3 | Finalize-bucket manager + test (partial) | pending | — | ≥1-bucket completeness; synthetic+real; mem/time |
| 4 | Wire + executor CONTRACT + gate at scale | pending | — | requires steady; hard-disable on thread/stream mismatch; multi-final + forced-B>1 barrier; matrix |
| 5 | Cloud retest — both topologies | pending | — | one-proc WAN + multi-proc+MPS; memory fit; histogram re-check |
| 6 | (Optional) one-shot preprocessor | pending | — | non-blocking; gated on Step-1; byte-exact (punctuation) |
| 7 | Finalize-batching prod config + canary | pending | — | per Step-1; L40S canary |

## Review log
- **R1 (Codex `bfqf2k2ug` + self):** fix MPS-framing (one-proc/no-MPS) → both-topology Step 5; GO/PIVOT gate;
  executor-coupling flagged; bucket budget; probe step; richer key; full-event gate; prompted disable;
  BATCH_FINALIZE decision → Step 1.
- **R2 (Codex `b50rl9cfu` + self):** executor/thread/stream CONTRACT explicit + serialize-replay (shared-buffer
  race); counterfactual gate on the tail cohort; disposable-fork claim downgraded (emitted-text writeback →
  multi-final tests); probe → matrix; finalize-specific completeness (not steady all-or-nothing); B>1 forced canary
  + production-concurrency histogram.
- **R3 (Codex `boqtfrp6a` + self):** CRITICAL — counterfactual gate was still circular → **moved the hard gate to
  after Step 2** (uses MEASURED `E_finalize_graph`); Step 1 is telemetry + a soft "worth probing" check. MAJOR —
  the multi-final + forced-B>1 tests are NOT buildable from existing harnesses → **build a same-websocket
  multi-final client** (prove the continuous protocol; the `--rounds` re-arm timed out) + a **test-only
  scheduler-barrier** for forced B>1 (concurrent clients may drain B=1); Step 1 must profile BATCH_FINALIZE 0/1 at
  conc-10 AND the knee. MINOR — finalize replay must HARD-disable (not warn) on thread/stream mismatch; Step 6 made
  optional/non-blocking. Both reviewers converged on the same two issues (gate ordering + harness buildability),
  severity dropped each round → **plan converged; ready to /implement** after these fixes (now folded in).
