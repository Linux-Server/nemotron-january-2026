# Plan: RNNT Label-Looping Decode CUDA-Graph (kill the overload stall + narrow the spread, byte-exact)

Project directory: `./proj-2026-05-23-1731`

## Context
The RNNT decode is the **last eager component on the finalize critical path** (encoder + finalize-encoder are already
CUDA-graphed). The server decodes with `greedy_batch` + `loop_labels=True` (NeMo's **label-looping** computer), whose
core is host-controlled `while active_mask_any.item():` — a per-frame D2H + `cudaStreamSynchronize` on the lane stream
(~16+/batch). Label-looping **FULL_GRAPH** (`use_cuda_graph_decoder=True`) replaces that host loop with a CUDA-graph
conditional-node while-loop, removing the syncs.

**Honest payoff framing:** the **SURE win is eliminating the conc-24 overload catastrophic stall** (cross-lane sync
contention; worst model_batch 13.3s) while keeping `lanes=2`. The conc-10 **leaderboard** win is **UNCERTAIN**: per-lane
batch is small (~1-5; ~5 sessions / 2 lanes), where graph-replay overhead can offset the few syncs saved, and
FULL_GRAPH still does eager input/state copies + CPU hyp→host conversion (`rnnt_utils.py:795`) — so the recoverable P50
is only the **sync/host-loop share**. Step 2 sizes it at the REAL conc-10 small-B distribution behind a floor/upside
ROI gate; an UPSIDE miss does not dead-end (it spawns a pivot doc and may still ship via the FLOOR). Keep `lanes=2`
(`lanes=1` regressed p50/p95 to 266/317). **Overriding constraint: byte-exact transcript correctness.**

**Target metric** = end-of-speech→final **TTFB** (`proj-2026-05-19-eou-endpointing/run_full1000_conc12.py`). Current
L40S conc-10 full-1000 WAN: **p50=246 / p95=279 / p99=474**, server-finalize p50=46 / p95=79 (beats Deepgram 247/298,
edges Soniox 249/281). **Acceptance is MEASURED in Step 8** (Step 2 is the PROVISIONAL/projected gate):
`no conc-10 p50/p95 regression AND (FLOOR OR UPSIDE)` where
- **FLOOR** (overload/tail-robustness GO): byte-exact, FULL_GRAPH-only, stream+memory safe, no fallback/OOM, AND the
  conc-24 stall eliminated OR a material repeated-A/B p99 below 474.
- **UPSIDE** (claim the conc-10 P50/spread goal): conc-10 p50≤236 AND (p95−p50)≤25.

The build is gated: **Steps 1-2 are a cheap LOCAL spike (no server changes, no cloud spend); Steps 4-8 proceed only on
a Step-2 GO.**

> **OUTCOME 2026-05-23 — BUILD HALTED at Step 2 (NO-GO).** (1) Max-B FULL_GRAPH **crashes at B=1** (illegal memory
> access; `need_reinit=False` so no recapture — a raw replay-shape failure), and B=1 is **86% of conc-10 decodes**. (2)
> Independently fatal to the conc-10 goal: the eager decode is only **~0.83ms steady / ~1.7ms finalize** at the real
> conc-10 B-distribution, so graphing it cannot move p50 (UPSIDE needs ≥10ms). Steps 3-8 NOT pursued. The decode-graph
> is the wrong lever for the conc-10 P50/spread goal. See `decoder-graph-probe-findings.md` + `conc10-pivot-findings.md`.

## The byte-exact oracle (read before every gate)
- **Foundation (Step 1):** graph-OFF eager, same session at B=1 vs co-batched — **tokens/text byte-exact** (proven:
  prior Probe B2 + a 200/200 strict-text canary); raw float state is only `allclose(~1e-4)`. Bar = "not worse than
  today's batched eager server."
- **Correctness gate (Steps 2/3, in-process, graph-ON vs graph-OFF on the SAME fixed batch trace):** **tokens / text /
  y-sequences byte-exact**; **float decode-state `allclose`** (the graph runs conditional-node ops, not the eager
  loop, so state is not bit-identical even same-trace). Optionally assert graph-replay determinism (graph vs graph,
  byte-equal) separately. The in-process path returns text + mutates state but does NOT emit WebSocket events.
- **Realism gate (Step 7, real-WS at concurrency):** compare **per-session emitted WS payloads — `text`/`is_final`/
  `finalize` only** (`_send_json_locked` @5930 does NOT carry token IDs), byte-identical; NOT per-batch / state /
  timestamps. (Token/y-seq byte-exactness is owned by Steps 2/3.)
- Never weaken the user-visible token/text gate; never over-strengthen float-state to byte-equal.

## Reference implementations
- **Live path** — `loop_labels=True` ⇒ `GreedyBatchedRNNTLabelLoopingComputer` (`rnnt_greedy_decoding.py:640-652`,
  `allow_cuda_graphs=use_cuda_graph_decoder`). Graph/state in **`transducer_decoding/rnnt_label_looping.py`**:
  `need_reinit` @167-173; `cuda_graphs_mode` @714-733 (**FULL_GRAPH @714-716 only mode without the host loop**);
  **silent FULL_GRAPH→NO_WHILE_LOOPS fallback @871-884**; `_graph_reinitialize` @814-822 (`max_time=max(.,~375)` → B
  is the recapture trigger; reinit inside `cuda_graphs_impl` @696); row-wise token @361/413; inactive rows masked
  @405/484; **own capture stream @958** (lane-stream replay-safety must be verified); residual eager copies @699 + CPU
  conversion @`rnnt_utils.py:795`; alignments/confidence @154/828/1111; biasing @236/704.
- **`cudagraph_encoder.py`** fail-closed manager (mirror only in the deferred bucket sub-project). **Finalize-graph
  wiring** `server.py` @766/1749/2318-2377.
- **Byte-exact gate** — `tests/test_cudagraph_encoder.py`, `tests/test_cudagraph_finalize.py`,
  `proj-2026-05-21-1959-cudagraph/step4_byte_exact_canary.sh`. graph-on/off share the identical preprocessor ⇒ cuFFT
  plan nondeterminism cancels (token/text comparison valid as exact equality).
- **Cloud** — `ec2-bench/bench_lanes_ab.sh` (adapt to vary the decoder-graph flag at lanes=2, or add a sibling graph
  A/B script); conc-24 overload harness built in Step 8.

## Current state
- **Decode cfg** `server.py:1463-1475` (`use_cuda_graph_decoder=False` @1473); `_decoding_cfg_for_lane_models` @1504;
  `_assert_batch_decoder_blackwell_safe()` @1507 **runs BEFORE lane creation/warmup (start() @1620)** → defer it.
- **INTERLOCK** @976-999 → `_disable_batching` @994 (def @924-937). **eou_probe** disables batching @633 + sets
  alignments/confidence @1476-1484. Env `NEMOTRON_MODEL_LANES` @611.
- **Reuse, don't reimplement**: steady `_process_ready_batch` starts **@8216** (stack @8287-8296 / scatter @8343-8352);
  finalize chain `_continuous_prepare_finalize_item_locked` @6611, `_prepare_final_fork_batch_row` @6897,
  `_finalize_batch_group_key_for_row` @7225, `_process_final_batch_rows` **@7350** (finalize splits by **`batch_max_size`**
  @7331 ⇒ B = 1..`batch_max_size`); fork deep-copy @6370 + hyp/pred clone @6480 + finalize re-clone @7371 (clone
  helpers @160/210); primitives `batch_primitives.py:59`. Session builder
  **`_init_session_without_synthetic_warmup` @3503** (`_init_session` @3574); lane wrapper (TLS model/stream)
  @3145/3165; emit `_send_json_locked` @5930. B-histograms in telemetry @5409 (steady) / @7268 (finalize). Lanes load
  LATE @3109-3137; per-lane streams (sync @3177/3209-3211).
- Verified `2026-05-23`: no `PLAN_RULES.md`; NeMo = `/home/khkramer/src/nemotron-nano-omni/NeMo` (venv `…/.venv-asr`).

## Rules
- **Byte-exact oracle** per the section above (in-process token/text/y-seq byte-exact + state allclose for 2/3;
  real-WS per-session text/is_final/finalize for 7). rc1 English, lanes1+lanes2, FORK_ASSERT=1. Default-off byte-identical.
- **Test protocol (no `PLAN_RULES.md`)**: every server-change step (4/5/6) runs, where CUDA is available,
  `pytest tests/test_decoder_cudagraph.py tests/test_cudagraph_encoder.py tests/test_cudagraph_finalize.py` (prove the
  decode graph does not regress the encoder/finalize graphs) + the relevant default-off/flag-on canary; step 6 also
  `bash -n` the touched shell scripts.
- **VERIFIED-only interlock** (replaces @976-994): keep batching when `use_cuda_graph_decoder` truthy ONLY if `flag==1`
  AND version gate AND all main+lane warmed AND every `cuda_graphs_mode==FULL_GRAPH`; any other truthy value ⇒
  `_disable_batching("cuda_graph_decoder_enabled_stray")`. Defer @1507 assert until after warmup.
- **Atomic-commit safety**: step 4 plumbs the flag but keeps it **INERT** (does NOT set `use_cuda_graph_decoder=True` —
  no warmup/verify yet); the graph only goes live at step 5. Each commit leaves the system safe AND flag-on-inert until 5.
- **FULL_GRAPH required** (others = NO-GO). **`need_reinit` PRECHECK** before each call → eager for would-reinit; never
  recapture under load; steady reinit count = 0. **Stream safety**: verify (Step 2) NeMo's own capture stream (@958)
  replays on a lane stream BEFORE step 5 captures per-lane; re-verify encoder+finalize graphs byte-exact AFTER decode
  capture; order encoder→finalize→decode then freeze. **eou_probe** ⇒ eager unless eou-on byte-exact proven;
  **biasing** off. Fail-closed on version/capture/would-reinit/OOM/over-budget.
- **Memory**: full-B × max_time hyp/state buffers per lane per process atop encoder+finalize; log before/after; enforce
  L4/L40S multiproc budget (`deploy/launch_multiproc.sh:36-39`); fail-closed if it doesn't fit.
- Deploy = Ada SageMaker; runtime version gate (cuda-python≥12.3 + driver CUDA≥12.6 + conditional nodes); log NeMo
  path+version. <400ms; 200ms VAD FIXED; no new heavy deps.
- ALWAYS terminate EC2 (traps + GPU leak check); local gate + cheap local latency pre-check before cloud; p99 noisy ⇒
  repeated same-box A/B (define min repeats/CI). Staged rollout: canary 1 replica → ramp 25/50/100%; rollback on
  non-FULL_GRAPH / fallback / OOM / byte mismatch / regression.

## Steps

- [x] **1. Foundation gate + in-process Session/state builder: eager batch-composition invariance**
  Standalone, local 5090, deploy venv; log NeMo path+version. Build `decoder_graph_harness.py` (shared by steps 2/3):
  load `ASRServer`; init sessions via `_init_session_without_synthetic_warmup` (@3503); populate caches /
  `previous_hypotheses` / `pred_out_stream` by running REAL warmup+ready chunks (do NOT synthesize state — NeMo mutates
  it); drive steady `_process_ready_batch` (@8216) + the finalize chain (@6611/6897/7225/7350) through the lane wrapper
  (@3145/3165). With the graph OFF, compare the SAME sessions under different compositions (B=1 vs co-batched, row
  permutations, shrink/grow) **by session-id + logical event**: tokens/text byte-exact; record float state allclose.
  **GO/NO-GO: if per-session tokens/text are NOT composition-invariant, make the FIXED batch-trace (same trace both
  arms) the authoritative oracle for steps 2/3** (robust regardless) and proceed.
  Key files: `proj-2026-05-23-1731/decoder_graph_harness.py`, `proj-2026-05-23-1731/foundation-invariance.md`

- [x] **2. Probe (PROVISIONAL GO/NO-GO + projected floor/upside ROI): FULL_GRAPH, row-state byte-exactness, P50 sizing** — VERDICT: NO-GO (see banner above)
  Extend the harness. (a) `use_cuda_graph_decoder=True`; reach `model.decoding.decoding.decoding_computer`; **assert
  `cuda_graphs_mode==FULL_GRAPH` after warmup** (reject silent NO_WHILE_LOOPS @884); determine if NeMo's own capture
  stream (@958) replays on a different (lane) stream. (b) graph-ON vs OFF on the in-process fixed batch trace:
  permuted/shrinking/growing subsets, continued non-`None` state, **a cold-start session right after warmup**, finalize
  **B=1..`batch_max_size`** — tokens/text/y-seq byte-exact, float state allclose; prove inactive zeroed rows can't
  affect active rows. (c) `need_reinit`/`INITIAL_MAX_TIME`: confirm B is the recapture trigger; design the runtime
  precheck; warm B=`batch_max_size` + finalize max-B. (d) **P50 sizing**: collect the REAL conc-10 steady+finalize
  B-histograms (telemetry @5409/@7268) and CUDA-event-time the decode-only at that small-B distribution; split
  recoverable **sync/host-loop share** from residual eager copies/CPU conversion (@`rnnt_utils.py:795`). **PROVISIONAL
  ROI GATE** (projection — measured acceptance is Step 8): GO if `correctness/feasibility AND (FLOOR projected: stall
  elimination / material p99 cut  OR  UPSIDE projected: p50≤236 AND spread≤25)`. **An UPSIDE-projection miss ⇒ write
  `conc10-pivot-findings.md`** (for the conc-10 goal: finalize fork/clone double-clone reduction @6370/6480/7371,
  one-shot finalize preprocessor @6927/7087, the reset-while-PENDING_FINALIZE debounce delay @5823, a global
  active-session/inflight admission cap @4163/4326) — **overall GO may still proceed via the FLOOR** as an
  overload-robustness project. (e) Memory per-(maxB,maxT) per lane; deploy gate values; eou-on byte-exactness if in scope.
  Key files: `proj-2026-05-23-1731/decoder_graph_harness.py`, `proj-2026-05-23-1731/decoder-graph-probe-findings.md`, (UPSIDE miss) `conc10-pivot-findings.md`

- [ ] **3. Promote the shared harness to a committed pytest gate**
  Wrap the Step-1/2 harness as `tests/test_decoder_cudagraph.py` (no rebuilt stack/scatter): assert FULL_GRAPH; the
  oracle-correct in-process suite (token/text/y-seq byte-exact + state allclose; permuted/shrinking/growing rows,
  continued state, cold-start, finalize B>1, fork-parent, + alignments/confidence if eou); assert ZERO `need_reinit`
  after warm-to-max. `pytest.skip` if no CUDA. Run it alongside the existing `tests/test_cudagraph_encoder.py` +
  `tests/test_cudagraph_finalize.py` to confirm no regression. (Deferred separate sub-project: a
  `BucketedCudaGraphDecoder` exact-B fallback — only if Step 2 shows max-B masking is non-byte-exact while exact-B passes.)
  Key files: `tests/test_decoder_cudagraph.py`

- [ ] **4. Wiring A — flag plumbing + VERIFIED-only interlock + deferred assert (atomic; graph stays INERT)**
  Add flag `NEMOTRON_RNNT_DECODER_CUDAGRAPH=1` (default OFF); **plumb only — do NOT set `use_cuda_graph_decoder=True`**
  (inert until step 5). Replace @976-994 with the VERIFIED-only control flow (stray ⇒ `_disable_batching("..._stray")`);
  grep ALL couplings; defer the @1507 assert. Both default-off and flag-on byte-identical to current at this commit.
  Test protocol (above). Key files: `src/nemotron_speech/server.py`

- [ ] **5. Wiring B — activate: per-lane(-stream) warm/capture + FULL_GRAPH verify + reinit-precheck (atomic)**
  Now set `use_cuda_graph_decoder=True` under the flag+version gate for main + per-lane cfg (@1504). Force lane creation
  and **warm EACH lane (and main) to FULL_GRAPH at max B (+ finalize max-B) before listening**, on the correct stream
  per Step-2's finding; after warmup assert `cuda_graphs_mode==FULL_GRAPH` on every model/lane — if ANY isn't, disable
  globally (batching+lanes eager). Add the runtime `need_reinit` precheck → eager. After decode capture, **re-verify
  encoder+finalize graphs still replay byte-exact** (order encoder→finalize→decode). Log before/after CUDA memory;
  enforce budget; fail-closed. Test protocol (above). Key files: `src/nemotron_speech/server.py`

- [ ] **6. Wiring C — telemetry + eou/biasing gates + flag passthrough (atomic)**
  Finalize telemetry: decode-replay share + `need_reinit` counter (==0 steady) + `cuda_graphs_mode` per model/lane.
  Gate eou_probe (eager unless proven) + biasing (off). Passthrough flag in `deploy/launch_multiproc.sh`,
  `ec2-bench/start_prod_server.sh`, `ec2-bench/bench_lanes_ab.sh`, `ec2-bench/bench_client_wan.sh`. Test protocol
  (above) + `bash -n` the touched scripts.
  Key files: `src/nemotron_speech/server.py`, `deploy/launch_multiproc.sh`, `ec2-bench/start_prod_server.sh`, `ec2-bench/bench_lanes_ab.sh`, `ec2-bench/bench_client_wan.sh`

- [ ] **7. Local byte-exact gate (CONCURRENT real-WS, per-session) + latency pre-check + default-off identity — HARD GATE**
  Canary graph ON vs OFF at **concurrency** over the REAL WS path (varying B = the row-mapping regime), lanes1 AND
  lanes2: compare **per-session emitted WS payloads — `text`/`is_final`/`finalize` only** (NOT token IDs/batch
  order/timestamps) byte-identical + fork-parent (+ alignments/confidence if eou); FORK_ASSERT=1; every model/lane
  FULL_GRAPH; zero runtime `need_reinit`; default-off byte-identical. Cheap LOCAL latency A/B to pre-check p50/spread
  before cloud. **GO/NO-GO: no cloud if any lane is partial/eager while reported "on", or any per-session mismatch.**
  Key files: `proj-2026-05-23-1731/decoder_graph_canary.sh`

- [ ] **8. Cloud MEASURED acceptance (repeated A/B, L40S→L4) + conc-24 overload harness + rollout readiness**
  L40S (g6e) same-box A/B graph OFF vs ON (finalize-graph ON in BOTH arms — isolate the marginal decode-graph win;
  vary ONLY the decoder-graph flag at lanes=2), conc-10 full-1000, **repeated (define min repeats/CI)**: apply the
  MEASURED acceptance — no conc-10 p50/p95 regression AND (FLOOR: conc-24 stall eliminated OR material p99 below 474 OR
  UPSIDE: p50≤236 AND spread≤25). Build the **conc-24 CLOUD overload harness** (`bench_overload_cloud.sh`: prod server
  + in-box `ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep 24 --rounds 8` jitter 400, graph OFF+ON same box, lanes=2,
  watchdog/USR1, srvlog pull, NeMo-version log, terminate): confirm the catastrophic stall is eliminated. Spot-check
  per-session byte-exact under load. **Then re-validate on L4 (g6.4xlarge)** before the L4 deploy lane ramps. Emit the
  staged-rollout readiness checklist. ALWAYS terminate; leak check.
  Key files: `ec2-bench/bench_lanes_ab.sh`, `ec2-bench/bench_overload_cloud.sh` (new), `proj-2026-05-23-1731/cloud-retest-decoder-graph.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Foundation gate + in-process Session/state builder (batch-composition invariance) | done | b8c000e | GO: 84 events, 0 token/text mismatches across B=1/B=N/perm/shrink-grow; float state allclose 1e-4 (max 3.3e-6); concurrent A/B oracle VALID; NeMo 2.8.0rc0; re-run reproduced |
| 2 | Probe (GO/NO-GO + projected floor/upside ROI) + P50 sizing at real conc-10 small-B | done — NO-GO | brlxvvedc | B=32 FULL_GRAPH crashes at B=1 (86% of workload); conc-10 decode only ~0.83ms steady/~1.7ms finalize ⇒ cannot move p50. Both FLOOR + UPSIDE fail. pivot doc written |
| 3 | Promote shared harness to committed pytest (+ run existing graph tests) | blocked (Step-2 NO-GO) | — | not pursued |
| 4 | Wiring A: flag plumb (INERT) + VERIFIED interlock + deferred @1507 assert | blocked (Step-2 NO-GO) | — | not pursued |
| 5 | Wiring B: activate per-lane(-stream) warm + FULL_GRAPH verify + reinit-precheck | blocked (Step-2 NO-GO) | — | not pursued |
| 6 | Wiring C: telemetry + eou/biasing gates + flag passthrough | blocked (Step-2 NO-GO) | — | not pursued |
| 7 | Local byte-exact gate (CONCURRENT real-WS per-session text/is_final) + latency pre-check | blocked (Step-2 NO-GO) | — | not pursued |
| 8 | Cloud MEASURED acceptance (repeated A/B, L40S→L4) + conc-24 overload harness + rollout | blocked (Step-2 NO-GO) | — | not pursued |
