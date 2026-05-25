# Plan: Near-term TAIL + DENSITY optimizations (admission, padded-T finalize bucket, host-sync trim, finalize priority, GIL probe)

Project directory: `./proj-2026-05-24-0859`

## Context
The roofline investigation (proj-2026-05-23-1731/roofline-COMBINED.md, two agents) proved the encoder is
memory-bandwidth-bound and **p50 TTFT is VAD+WAN-bound (only ~12-19ms movable by ANY engine)** — the real prize is
the **p95/p99 TAIL at load + DENSITY (in-budget streams/box) + overload-robustness**, all **GIL/scheduler-bound** (at
load the GPU is 46-65% idle while the single asyncio thread is GIL-starved by the eager decode → `vad_stop_recv_to_
process` 8ms@3/proc → 928ms@12/proc; in-budget L40S K=3 ~6.7/proc=~20/box, **L4 K=2 ~3.5/proc=~7/box** — L4 ~3× worse,
matching its ~2.9× lower mem-BW). **SUCCESS = tail (p95/p99 at the operating point) ↓, in-budget density (streams/box)
↑, overload cliff gone — NOT p50.** **Honest density math (R2): in-budget is ~6.7/proc; K=4 (Step 2b) gives ~K×7 ≈
28/box in-budget (vs K=3 ≈ 21, +~33%) — NOT 64 (64 was the over-budget keep-up KNEE); only Step 5 (per-proc keep-up)
could push higher, and it is probe-only.** OVERRIDING CONSTRAINT: byte-exact per-stream transcript correctness.

## Reference implementations
- **`src/nemotron_speech/cudagraph_encoder.py`** — exact-T capture/replay: static input alloc at bucket `time_steps`
  @193, `static_len` init @199-204 / overwrite-on-replay @272-278, **replay shape-equality reject @263/582-599** (so a
  padded probe needs NEW infra — the current manager rejects exact-T vs T_max). Finalize key = exact physical T from
  `processed_signal.shape[-1]` `server.py:2154-2208`.
- **Replay/fallback wrapper** `server.py:2318-2377` — **falls back to eager ONLY when `replay_finalize()` returns
  `None` @2345; it does NOT double-run eager and compare values.** ⇒ padded "fail-closed" must be a **startup/CI
  canary that disables the flag on mismatch**, not a per-call runtime compare.
- **NeMo masking + the physical-length hazards (Step 2)** — `max_audio_length = audio_signal.size(1)` drives
  `cache_keep_size`/returned cache length `conformer_encoder.py:664-668/771-776`, attention-mask dims @884-892,
  softmax/matmul shape `multi_head_attention.py:135-138`; subsampling length-mask `subsampling.py:655-683`; conv
  zeroes pad `conformer_modules.py:330-333`; BatchNorm OK in eval only (`server.py:1506/3137`), instance/group-norm
  would include pad `conformer_modules.py:296-307`.
- **Eager-decode GIL (Step 5)** — `loop_labels=True`,`use_cuda_graph_decoder=False` `server.py:1463`; NeMo Python
  loop conditions / `.item()` `rnnt_label_looping.py:357/409/718`; batching self-disables if cuda-graph-decoder on
  `server.py:993`. Dispatch is ALREADY in executors (`server.py:3165/5206`, generic @3087); the wall is GIL, not
  thread placement.
- **Byte-exact infra** — `tests/test_cudagraph_finalize.py` uses `torch.equal`/zero-max-diff @265/343 (exact-shape);
  the Step-2a probe needs a custom T_max path + crop-to-real + the relaxed gate. **Harnesses** —
  `ec2-bench/bench_prod_sweep.sh` (keep-up sweep; hardcodes `maxconn 12` @47 — parametrize for Step 1),
  `bench_prod_multiproc.sh`/`bench_lanes_ab.sh`. From-scratch C++/no-GIL core (roofline Q3) OUT of scope (Step 5 may
  hand off to it).

## Current state
- **Admission (Step 1)**: no gate — `self.sessions[id]=session` @4169 (cleanup @4243-4245); WS `prepare()`d @4152-4153
  (reject = WS-close, NOT 503). Queue depth 256 @4326-4341 (non-scheduler unbounded @4256-4289). **The backlog metric
  `vad_stop_recv_to_process_ms` is PROFILING-ONLY (gated `NEMOTRON_FINALIZE_PROFILE` @733/2469/4360, emitted @2790)** →
  admission needs an always-on signal. HAProxy `maxconn 12` @`haproxy.cfg.example:34`.
- **Lanes/scheduler (Steps 4/5)**: `inference_lock`/`state_lock` = asyncio.Lock @594/486; lane = single-worker
  ThreadPoolExecutor + 1 stream @3155-3163, availability removed before dispatch @3389, **lane-exit
  `stream.synchronize()` @3177**. Worker mutates session decode/cache state under event-loop-held locks @5190-5210
  (steady @8389-8397, serial @8487-8533, finalize @7451-7463). `self.sessions` mutated on loop @4169/4244, read by
  timers/scheduler @6070-6079/4516-4523. Finalize: collected before ready @4515-4547, generation-invalidated
  @4713-4725, lane wait (no same-session in-flight) @6711-6720, lane call @6765-6773; pinned-affinity @3295-3308 +
  violation check @5117-5128.
- **Host syncs (Step 3)**: removable = telemetry pre-syncs @8223/7300; **LOAD-BEARING** = lane-exit @3177, scatter
  completion @8399/7513, model/scatter host-visibility @8329-8355/7430-7494. Finalize-profiling adds syncs
  @7295-7341/7430-7518 (measure profiling OFF).
- **Finalize-graph memory (Step 2)**: per-T buckets (T=42..60)×lane×replica → ~11-12GB/proc → L40S K=3 (K=4 OOMs).
  Finalize write lands on the FORK @7451 (parent text updated separately @6844) → finalize cache does not carry forward.
- Verified 2026-05-24: no `PLAN_RULES.md`.

## Rules
### Correctness (hard gate — every behavior-changing step)
- **Byte-exact per-stream tokens/text/emitted vs current**, at CONCURRENCY, lanes1 AND lanes2, FORK_ASSERT=1,
  per-session; no dropped/reordered/dup emits. Default-off / below-cap identity.
- **Step 2 gate (relaxed, satisfiable)**: padded-T_max graph(realT) vs exact-T eager(realT) — **tokens/text/emitted
  byte-exact + `encoded_len` exact + `cache_last_channel_len` exact + real-frame encoder tensors allclose(tol)**,
  comparing ONLY the real region (crop the padded tail). Byte-exact TENSORS is OVERCONSTRAINED (different physical T
  shape → different reduction order) — do NOT require it; but tokens MUST stay byte-exact (allclose tensors can flip
  argmax). Assert norm==batch_norm + eval + real-length masks passed. **Fail-closed = startup/CI canary disables the
  padded flag on any mismatch (not per-call).**
- **Step 4**: **priority = queue-jump at SUBMIT only (cannot preempt a running lane call)**, CROSS-SESSION only — never
  jump a session's own ready/in-flight steady; obey pinned-affinity + no-same-session-in-flight.
- **Step 5**: a GIL-ATTRIBUTION probe (decode vs glue, buckets summing to thread-busy, + GPU-idle% + GIL-wait) whose
  ONE structured record decides the from-scratch project's `conjunct 2` (GIL/decode-bound → native helps; MPS/launch/
  bandwidth-bound → STOP). PROBE-ONLY: emit via the existing `_continuous_finalize_timing` schema (a logging add); do
  NOT attempt the native rewrite / NeMo-decode replacement / lock-domain redesign here (that is the from-scratch
  project, gated on this record).
### Measurement / capacity honesty (avoid double-counting)
- maxconn/admission **ENFORCE the in-budget operating point + shed** — they do NOT raise capacity. Step 2b adds the
  4th proc (K=4 → ~28/box in-budget, +~33% vs 21), NOT 64. Step 5 is the only per-proc keep-up lever (deferred).
- Step 1: report **attempted vs admitted**. Step 3: measure profiling **OFF**. Step 5 probe: attribute GIL between
  lane-thread decode AND event-loop-thread scheduler glue. p99 noisy → repeated runs / the sweep curve.
### Safety / deploy
- Flag-gated, default-off; fail-closed; independently committable. Ada (L4/L40S) SageMaker; prod multi-proc+MPS+HAProxy.
  ALWAYS terminate EC2 (traps + GPU leak check); us-west-2; `aws sso login --sso-session khk` (check before cloud). No
  new heavy deps. VAD+WAN out of scope.

## Steps

- [x] **1. Admission/backpressure on an ALWAYS-ON backlog signal + lower HAProxy maxconn**
  (a) Cheapest: lower HAProxy `maxconn` 12→ the in-budget value (~7 L40S, ~3-4 L4) in `haproxy.cfg.example` +
  `DEPLOYMENT.md` + parametrize `bench_prod_sweep.sh:47` (zero server code). (b) Server-side defense: a configurable
  cap keyed on an **always-on backlog signal = the composite** `sum(continuous_event_queue.qsize())` +
  `len(self._scheduler_ready)` (@698) + oldest `scheduler_ready_since` age (@4754) — NOT `vad_stop_recv_to_process`
  (profiling-only) and NOT session count. (Don't use ready-age alone — events sit in per-session queues before they
  become ready; add a queued-event timestamp at `_scheduler_queue_event` only if age-gating.) Reject past cap via
  WS-close before `self.sessions[id]=session` @4169 (or hit cleanup @4243-4245). Byte-exact below cap. **Gate**:
  default-off identity below cap; clean shed above. **Cloud**: `bench_prod_sweep.sh` past the operating point,
  reporting **attempted vs admitted** → admitted streams in-budget, cliff bounded.
  Key files: `src/nemotron_speech/server.py`, `deploy/haproxy.cfg.example`, `deploy/DEPLOYMENT.md`, `ec2-bench/{start_prod_server.sh,bench_prod_sweep.sh}`

- [x] **2a. Padded-T byte-exactness PROBE (GO/NO-GO, local, no cloud)** — VERDICT: GO (cache_len divergence empirically confirmed dead)
  Harness: REUSE `proj-2026-05-23-1731/decoder_graph_harness.py` (@528-630 builds real sessions + finalize forks +
  drives final rows) + NEW padded infra (the current manager rejects exact-T vs T_max @582-599; `stack_processed`
  @batch_primitives.py:59-75 sets length=physical T so the padded path needs custom length handling). For every real
  T in 42..60, B=1: **deep-clone hyp/pred state @7371-7382 before each run**, then run BOTH exact-T eager AND a T_max
  graph fed **zero-padded input + REAL `processed_signal_length`**; crop to the real region. Apply the relaxed Step-2
  gate: **full-decode tokens/text byte-exact** (not encoder-only) + `encoded_len` exact + `cache_last_channel_len`
  exact + real-frame encoder tensors allclose; assert norm==batch_norm+eval. If ANY T fails (esp. tokens or cache
  length) → **NO-GO: keep per-T; address K=4 OOM another way (trim T-range / accept K=3)**. Only GO → 2b.
  Key files: `proj-2026-05-24-0859/padded_t_probe.py`, `proj-2026-05-24-0859/padded-t-findings.md`

- [x] **2b. Single padded-T_max finalize bucket REPLACING the per-T buckets (recover K=4 ≈ 28/box) — only on 2a GO** — built + local-gated; K=4 cloud verify → Step 6
  Add padded capture/replay to `cudagraph_encoder.py`. **B POLICY (must-fix): capture ONE B=1 × T_max bucket only**
  — finalize is ~99.9% B=1 (measured across all configs incl. the 12/proc operating point; the current finalize graph
  is already B=1-only @1693/2180) — and **B>1 (rare storms, ~0.1%) falls back to eager** (this keeps the K=4 memory
  claim = 1 bucket; per-B × T_max would need 4-16 buckets + a revised memory claim, unjustified by the data). Replay
  zero-pads + overwrites `static_len` with REAL lengths @272-278. Flag `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1`
  (default off) **SWITCHES per-T→padded (does NOT also capture the 19 per-T buckets T=42..60 — verify absent, or the
  win is lost)**. **REQUIRED server plumbing (else the bucket is never hit / decode sees pad):** (i) change the
  finalize keying (@2154-2208) so any real T in range selects the single B=1 **T_max** bucket (today it keys on
  physical `processed_signal.shape[-1]` + requires that exact key); (ii) zero-pad the input to T_max + pass REAL length
  on replay; (iii) **pass the REAL `encoded_len`** — NeMo RNNT decode bounds by `encoded_len`, not physical T
  (mixins.py:707 / rnnt_label_looping.py:325), so **no mel-frame crop** (an optional `encoded[…,:encoded_len.max()]`
  slice only if the 2a probe shows it needed); (iv) **log both `graph_T=T_max` and `real_T`** (telemetry @2090/2112
  would otherwise collapse T to T_max, losing the distribution). Byte-exact enforced by: the local/CI real-audio 2a
  sweep (the GO/NO-GO) **plus** a **startup SYNTHETIC-SMOKE
  canary** (synthetic sessions via `_init_session`/`_run_session_warmup` @3574-3680, run **per captured lane manager**
  @1888-1942 not just `self.model`; on mismatch disable padded + log + fall back to per-T/eager) — neither is per-call.
  **Gate**: the relaxed Step-2 gate 100/100; memory acceptance via **total `finalize_capture_memory_bytes()`** (not
  per-key deltas) showing the ~16× finalize-pool drop (~2.5 GB/proc → per-proc ~9.5 GB). **Cloud**: K=4 on g6e.8xlarge
  no OOM + in-budget density ~28/box (4 × ~7). Update `launch_multiproc.sh` `auto_pick_K`→L40S=4 + `DEPLOYMENT.md`
  only after verified.
  Key files: `src/nemotron_speech/cudagraph_encoder.py`, `src/nemotron_speech/server.py`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`

- [x] **3. Remove the safely-removable host syncs (small, byte-exact)** — built + concurrent byte-exact (identical-SHA); keep-up delta → Step 6
  Remove ONLY the telemetry pre-syncs @8223/7300 under `NEMOTRON_SYNC_COMPRESS=1` (default off). Do NOT touch the
  load-bearing fences (@3177/8399/7513/8329/7430). STRETCH (separate, optional): replace lane-exit `stream.synchronize`
  @3177 with a CUDA-event + a lane-busy state so the scheduler does NOT release the lane until the event completes
  (else later work races on the same stream) — only if byte-exact + it helps. **Honest expectation: small.** **Gate**:
  byte-exact at concurrency, profiling OFF. **Cloud**: keep-up sweep delta.
  Key files: `src/nemotron_speech/server.py`

- [x] **4. Priority finalize-lane queue-jump (CROSS-SESSION, submit-time only)** — built + per-session byte-exact; tail measure → Step 6
  Let a due finalize jump AHEAD of OTHER sessions' not-yet-submitted steady work for the lane (it cannot preempt a
  running lane call — single-worker @3155, availability removed @3389; expected win ≈ one steady-batch wait, not
  many). Flag `NEMOTRON_FINALIZE_PRIORITY=1` (default off). MUST obey pinned-affinity (@3295-3308/5117-5128) + the
  no-same-session-in-flight wait (@6711-6720) + never jump the session's OWN ready/in-flight steady
  (@4515-4547/4713-4725/5558-5569). **Gate**: byte-exact + per-session-ordering at concurrency. **Cloud**: conc-10 +
  operating-point sweep → p95/p99 finalize lane/lock-wait share drops.
  Key files: `src/nemotron_speech/server.py`

- [x] **5. GIL-attribution PROBE — decode vs glue at the operating point (decides from-scratch `conjunct 2`; probe-only)** — VERDICT: **conjunct 2 SURVIVES (medium conf)**; steady path is DECODE-bound (decode 78.6/82.2/83.7% of thread-busy p50/p95/p99, dispatch <2%, host-sync ~0); finalize p95/p99 glue = pinned-lane/inference-lock wait (49-54%), not dispatch. GIL-wait via event-loop-lag PROXY (0.53/1.04/1.32ms — py-spy --gil ptrace-blocked locally) → recheck GIL-starvation with ptrace py-spy on the cloud box before treating that sub-claim as proven.
  GOAL: split the time the scheduler/event-loop (GIL-holding) thread spends holding the GIL into **decode vs glue at
  the operating-point load**, so `proj-2026-05-24-from-scratch-runtime`/0.1 can decide **conjunct 2** (residual is
  GIL/decode-bound → native dispatch + on-GPU decode FIXES it; MPS/launch/bandwidth-bound → STOP, native ≈ Python
  topology) WITHOUT re-deriving it from a fresh ablation.
  MEASURE (per chunk + per finalize, under load, **p50/p95/p99**) wall-time on the GIL-holding thread, **bucketed so
  the buckets SUM to thread-busy time**:
  - **decode** — the NeMo `greedy_batch` label-looping call (the per-frame `.item()` loop, rnnt_label_looping.py:357/409/718) — the hypothesized wall;
  - **model dispatch** — encoder graph replay / kernel launch;
  - **glue** — split into: scatter/gather (`batch_primitives` stack/scatter), host syncs (`stream.synchronize`), `inference_lock` acquire-wait, scheduling + socket I/O.
  PLUS **GPU-idle % while the thread is busy** and **GIL-wait seen by other lanes/coroutines** (`py-spy --gil`
  sampling) — this is what distinguishes "GIL-serialized" from "MPS/launch-serialized."
  OUTPUT (what 0.1 consumes): ONE structured **JSON record per operating point**, emitted through the EXISTING timing
  schema (`_continuous_finalize_timing` + batch/finalize telemetry — the same fields 0.10 mandates parity with), with
  the named buckets as **ms + % of thread-busy time**. A logging ADD, not new plumbing.
  DECISION the record must support: **decode ≫ glue + high GIL-wait + GPU-idle attributable to single-thread
  serialization → conjunct 2 SURVIVES** (native dispatch + on-GPU decode helps); **glue dominated by
  MPS/context-launch, or the work is bandwidth-bound, + GPU not idle-on-GIL → conjunct 2 DIES → STOP** (native ≈
  Python topology).
  KEEP IT CHEAP: prefer a **sampling profiler (`py-spy --gil`) + `perf_counter` brackets** around decode-vs-rest over
  heavy instrumentation; **ONE run at the operating point** is enough for the kill/keep direction. Probe-only — do NOT
  attempt the native rewrite / lock-domain redesign here (that is the from-scratch project, gated on this record).
  Key files: `proj-2026-05-24-0859/gil-attribution.md`, `src/nemotron_speech/server.py` (the timing-schema logging add)

- [x] **6. Combined cloud validation + deploy update + rollout readiness** — DONE (validation.md): G1 ✓ K=4 no-OOM (3×), G4 ✓ L4 K=2-padded no-OOM, G3 ✓ admission shed (backlog-count cap≈8-12). G2: clean density L40S **~16-20/box regardless of K** + L4 **~6/box** → ~28/box & 48/64 REFUTED (intake+BW-bound); **keep K=3**. No p50 regression; profiling ~2× tail tax on L40S quantified. Levers shipped byte-exact default-on; DEPLOYMENT.md + memory corrected; rollout checklist written. All 6 boxes terminated + leak-checked clean.
  Verified levers ON (each flagged) on L40S (+ L4): keep-up sweep (in-budget streams/proc), operating-point p95/p99
  tail, overload test (cliff gone, attempted-vs-admitted), **L40S K=4 in-budget density (~28/box, no OOM)** with the
  padded bucket. **ALSO: L4 K=2 with FINALIZE_PADDED=1 no-OOM check on g6.4xlarge** — confirm the padded bucket
  obsoletes the per-T finalize trim (full finalize-graph coverage fits 24 GB at K=2, ~1.5 GB headroom projected from
  the ~19× local drop); note L4 capacity stays keep-up-bound ~7/box regardless (NOT a capacity gain). No p50
  regression anywhere. Update `launch_multiproc.sh` (flags where proven; `auto_pick_K`→L40S=4 if 2b verified;
  drop the L4 finalize-T-trim recommendation in favor of padded) + `DEPLOYMENT.md` (in-budget/box honest numbers,
  K-sizing, padded-replaces-L4-trim, the L4-poor-for-SLO finding) + memory. Staged-rollout checklist. ALWAYS
  terminate; leak check.
  Key files: `ec2-bench/bench_prod_sweep.sh`, `deploy/launch_multiproc.sh`, `deploy/DEPLOYMENT.md`, `proj-2026-05-24-0859/validation.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Admission on always-on backlog signal + lower HAProxy maxconn | done (local) | 6a427fa | default-off identity runtime-confirmed; flag-on reject logic correct; always-on composite signal (qsize+ready+age); WS-close 1013 before admit; maxconn 7(L40S)/3-4(L4); HAPROXY_MAXCONN env. Cloud attempted-vs-admitted sweep → Step 6 |
| 2a | Padded-T byte-exactness PROBE (GO/NO-GO) | done — GO | 67f3ce7 | tokens/text/encoded_len byte-exact all T 42..60, tensors allclose 1.5e-7; ONLY cache_len diverged (fork [48] vs [46/47]) but it's on the DISPOSABLE fork — continuation probe CONFIRMED session keeps its own cache ([41]→[41]/[57]→[57]), post-finalize byte-exact. GO for 2b |
| 2b | Padded-T_max bucket REPLACING per-T (recover K=4 ≈ 28/box) | done (local) | 7cc434d | built: padded replay + single-key switch + B>1→eager + dual-T telem + per-manager startup canary. Local gate: padded graph byte-exact lanes1+2 all T + continuation (canary_ok self.model+lane0+lane1); per-T absent; ~19-21× mem drop (476→25MB/mgr); default-off identical. K=4 cloud verify → Step 6 |
| 3 | Remove safely-removable host syncs (small) | done (local) | 5c8dbc6 | NEMOTRON_SYNC_COMPRESS gates 2 entry pre-syncs (finalize @7687 elif, steady @8611); both telemetry/CPU-state only (real fences kept). Byte-exact: identical-SHA concurrent canary flag-on==flag-off, lanes1+2, FORK_ASSERT; default-off identical (elif preserves original). keep-up delta → Step 6. CUDA-event stretch noted, not built |
| 4 | Priority finalize-lane queue-jump (CROSS-SESSION, submit-time) | done (local) | 5ebec20 | NEMOTRON_FINALIZE_PRIORITY excludes OTHER sessions' steady from a pending finalize's pinned lane (excluded_lanes.difference); gated (empty when off → identical); intra-session ordering + no-same-session-in-flight guard kept. Byte-exact: identical per-session SHA flag-on==flag-off lanes1+2 (sessions finalizing mid-stream). tail measure → Step 6 |
| 5 | GIL-attribution probe (decode vs glue at operating point → from-scratch conjunct 2) | done — conjunct 2 SURVIVES (med conf) | 0ba026f | NEMOTRON_GIL_ATTRIB=1 default-off; one gil_attribution_record JSON at shutdown via existing timing surface; buckets SUM to thread-busy (delta 0.0ms). Decode-bound: chunk decode 78.6/82.2/83.7%, dispatch <2%, host-sync ~0; finalize p95/p99 glue=inference-lock/pinned-lane wait 49-54%. GPU-idle-while-busy chunk 17-26%. GIL-wait=event-loop-lag PROXY 0.53/1.04/1.32ms (py-spy --gil ptrace-blocked → recheck on cloud). Default-off identity by construction (timed_call pass-through, context-mgrs no-op-yield, output untouched); flag-on run ok=48/0err. → native dispatch+on-GPU decode direction kept for proj-2026-05-24-from-scratch-runtime/0.1 |
| 6 | Combined cloud validation + deploy update | done | (pending) | 6 boxes us-west-2, all terminated+leak-clean. G1✓K=4 no-OOM(40/46,3×) G4✓L4 K=2-padded no-OOM(19/23) G3✓admission shed(MAX_BACKLOG cap; ready-age signal wrong). G2: clean L40S **~16-20/box K=3≡K=4** (28/box & 48/64 refuted — intake+BW-bound; keep K=3, K=4 no density gain), L4 **~6/box** BW-bound (not regression/profiling-artifact). Profiling ~2× tail tax L40S quantified. Levers default-on byte-exact in launch_multiproc; DEPLOYMENT+memory honest-numbers; rollout checklist. Incident: same-ITYPE tag-reuse collision → NEMOTRON_EC2_NAME+STATE overrides |
