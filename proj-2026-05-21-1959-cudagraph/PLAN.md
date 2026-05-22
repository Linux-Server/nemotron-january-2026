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
Deployment target is **AWS SageMaker** (Ada GPU; reasoned default **G6/L4 sized for vCPUs**, not L40S — see
deployment memory), not Modal — Modal is only the benchmark
proxy, and since the knee is single-thread-CPU-bound the production knee must be validated on the target
instance (the Modal "batching doesn't help" result is CPU-allocation-specific).

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
- **Graph-pool memory (watch this)**: graphs are per-CUDA-context, so each lane/process replica captures its own
  1…K set; the cache tensors scale with B, so per-B (1…K) × lanes(2) × processes(K=2–4) replicas must fit GPU
  memory (24 GB L4 = the tight one, alongside ~2.4 GB × replicas of model). Fail-closed: OOM on capture of any B →
  that B (or that replica) falls back to eager; the server still serves. If the full 1…K set won't fit per
  replica, cap K (keep no-padding byte-exactness for whatever B *are* captured) rather than padding. The cloud
  retest (Step 6) must confirm the chosen K fits at the target K_proc×lanes on both L4 and L40S.
- Only the **encoder** is graphed; decode stays eager (`use_cuda_graph_decoder=False`).
- Don't break the multilingual prompted path, the silence0_warm200 finalize/fork logic, or the warmup path.
- No new heavy deps. Flag-gated; default = current behavior until proven.

## Steps

- [x] **1. Per-B byte-exact + speedup probe; pick K.**  (DONE — round5; `probe_perB_cudagraph.py`: per-B byte-exact B=1..16, GPU-active −12..30%, **K≈10**.)
  Extend `proj-2026-05-21-0410/probe_manual_cudagraph.py` (or a sibling) to capture B = 1…16 (configurable),
  building per-B stacked inputs from K independent clips. For each B: confirm byte-exact vs eager-batched
  (per-step text bytes + state `max_abs=0`, the existing compare extended to B>1) AND measure the synced per-B
  speedup. Output: the measured speedup-vs-B curve (resolves the projected 1.36×@B1 → ~1.1×@B46) and a
  recommended **K** = the B where speedup drops below ~1.15×. Standalone, no `server.py`. Gate: byte-exact for
  ALL tested B (any mismatch → diagnose the stacking/capture before proceeding).
  Key files: `proj-2026-05-21-1959-cudagraph/probe_perB_cudagraph.py`

- [x] **2. Bucketed graph-manager module (standalone-tested).**
  New `BucketedCudaGraphEncoder` (e.g. in `src/nemotron_speech/cudagraph_encoder.py`): holds per-B captured
  graphs + per-B static buffers (cache `[layers,B,...]` channel/time, `[B]` len, mel `[B,F,T]`); `warmup()`
  captures B=1…K (side-stream warmup → capture, per probe); `replay(B, inputs)→outputs` does copy-in / replay /
  return-static (caller clones out); fail-closed (capture error on any B → mark uncaptured → eager). Unit test
  asserts byte-exact per B vs eager and that uncaptured-B cleanly returns "use eager". No `server.py` wiring yet.
  Key files: `src/nemotron_speech/cudagraph_encoder.py`, `tests/test_cudagraph_encoder.py`

- [x] **3. Wire into the scheduler's batched call (FINALIZE AFTER (b) LANDS).**
  In `server.py`, gate behind `NEMOTRON_ENCODER_CUDAGRAPH` (default off) + `NEMOTRON_ENCODER_CUDAGRAPH_MAX_B`
  (default K). At the batched model call: if steady-bucket AND B≤K AND captured → `replay(B)`; else eager.
  Capture all 1…K at startup warmup (after model load). Supersede the dead `NEMOTRON_ENCODER_COMPILE` path
  (remove or redirect to manual). Route non-steady buckets to eager. **Integration point = the scheduler's
  batched `conformer_stream_step` dispatch; finalize against (b)'s committed scheduler structure** (lanes +
  graphs must compose — each lane replays on its own stream; the graph's static buffers are per-lane or guarded).
  Key files: `src/nemotron_speech/server.py`

- [x] **4. Local byte-exact gate at scale (hard gate).**
  Stream a fixed multi-stream clip set with `NEMOTRON_ENCODER_CUDAGRAPH=1` vs `=0` (+ scheduler/batching on) and
  diff transcripts — must be byte-identical (200/200-style canary), `FORK_ASSERT=1` clean over a multi-minute
  run. Confirm default-off (`=0`/unset) is byte-identical to the pre-change server. Confirm graphs actually
  engage (replay counters > 0, no silent all-eager). **ALSO run `NEMOTRON_MODEL_LANES=2` graph-on (the
  per-lane-stream path deferred from step 3) and confirm byte-identical + per-lane replay engagement.**
  Key files: existing harness (`proj-2026-05-19-eou-endpointing/`), `proj-2026-05-21-1959-cudagraph/`

- [x] **5. Local knee measurement (first measured payoff).**
  Realtime keep-up sweep, graph-on vs graph-off (scheduler+batching on): does the knee lift at the small B
  realtime produces (baseline 56 → ?)? Also single-stream B=1 latency/knee (expect ≈ the old compile-only 24).
  Record avg B at the knee + the per-B engagement mix.
  Key files: `proj-2026-05-21-1959-cudagraph/local-knee.md`

- [x] **6. Cloud GPU-bound retest on EC2 g6 (L4) + g6e (L40S) — tight TTFS budget (p50<250 / p95<300).**
  Deploy manual capture to **EC2 via `ec2-bench/`** (NOT Modal — Modal is the launch-bound proxy; EC2 g6/g6e is
  the SageMaker-representative target and our established vehicle). CONFIRM capture engages at startup (~250 ms × K
  per replica, no inductor hang — the Step-10b failure mode must be gone) and **fits memory** at the target
  K_proc×lanes (the graph-pool risk above), then smoke byte-exact. Then, with **multi-process + MPS** (the
  production scaling unit) under the **tight latency budget p50<250 / p95<300** (the `run_l4_ttfs_sweep.sh`
  methodology: staggered, sustained `--rounds` for stable p95), measure graph-ON vs graph-OFF in the GPU-bound
  regime and answer three questions:
  - **(a) per-process knee** — does collapsing launch dispatch raise the GIL-bound ~16 (→ fewer processes for the
    same box capacity; shifts the K-matrix)?
  - **(b) per-box GPU-bound ceiling** — graphs cut GPU-active 12–30%, so does the L4 hold >32 (K=2) and does the
    L40S 64 (K=4) become **robust** (it was variance-prone/fragile at the full per-process knee — see
    g6-vs-g6e-results.md TTFS section)?
  - **(c) p95 tail** — deterministic single-replay should tighten the tail that straddled 300 ms in the 20–32/box
    L4 zone → does the **tight-budget per-box max-streams** rise?
  Compare to the pre-cudagraph tight-budget baseline (the `run_l4_ttfs_sweep.sh` results). Billable,
  cost-conscious (smoke first; ALWAYS `ec2_down.py`). Write into
  `proj-2026-05-21-1959-cudagraph/cloud-retest.md` and fold the new max-streams + any K-matrix shift into
  `proj-2026-05-21-inference-opt/g6-vs-g6e-results.md`, the deploy docs (`deploy/`), and memory
  (`deployment-target-sagemaker`).
  Key files: `ec2-bench/run_l4_ttfs_sweep.sh`, `ec2-bench/run_multiproc.sh`, `proj-2026-05-21-inference-opt/g6-vs-g6e-results.md`

- [x] **7. Drop the coalescing tick (work-conserving batching) — measure, then flip if it wins.**  (RESULT: NEGATIVE — keep the tick; no default change. See below.)
  Hypothesis: the `NEMOTRON_BATCH_MAX_WAIT_MS=8` coalescing timer (server.py:635) was a *launch-bound*
  amortization — it forced bigger batches so fewer kernel-launch dispatches were paid (our note at
  server.py:632 records it raised the local knee 40->56). The scheduler loop (`_scheduler_drain_once`,
  server.py:3454) is ALREADY event-driven + drains every currently-ready chunk via `get_nowait`; the timer is
  the only non-work-conserving piece. CUDA-graphs collapse the per-launch cost (step 5: small batches now cheap,
  avg B~2-3, knee still up), so the timer's throughput benefit should shrink while its cost (up to 8 ms of
  *under-load* latency, exactly on the tight-budget path) stays. **Predict: `MAX_WAIT=0` recovers the old lower
  knee with graphs OFF, but ~matches `MAX_WAIT=8` with graphs ON, at lower p95.**
  Do: (a) make `MAX_WAIT` an env knob in `run_l4_ttfs_sweep.sh` (currently hard-coded 8 in the SRV env);
  (b) run the 2x2 tight-budget sweep `MAX_WAIT in {0,8} x CUDAGRAPH in {0,1}` on one box (reuse step 6's
  box/config), comparing per-process tight-budget max-N (p50<250/p95<300) + the keep-up knee + p95;
  (c) if `MAX_WAIT=0` with graphs ON holds the knee AND lowers p95, flip the default to 0 when cudagraph is
  enabled (small config change in server.py), flag-gated, default stays 8 until the measurement says otherwise.
  Correctness: `MAX_WAIT` changes only batch *grouping/timing*, not per-stream frames, so per-stream output is
  unchanged by the batch-independence property already validated in step 4 — spot-check byte-exact, expect free.
  Key files: `ec2-bench/run_l4_ttfs_sweep.sh`, `src/nemotron_speech/server.py`,
  `proj-2026-05-21-1959-cudagraph/cloud-retest.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Per-B byte-exact + speedup probe; pick K | done | round5 | probe_perB_cudagraph.py: per-B byte-exact B=1..16, GPU-active −12..30%, K≈10 |
| 2 | Bucketed graph-manager module | done | bf0a639 | cudagraph_encoder.py + test; byte-exact B=1..16 (encoded+state max_abs=0), fail-closed B=17->None/captured=False; capture ~60-82ms/bucket (~1.1s for K=16/replica) |
| 3 | Wire into scheduler's batched call | done | 22a817c | NEMOTRON_ENCODER_CUDAGRAPH; monkeypatch like compile path; steady-bucket-only per-B; per-replica + per-lane-stream managers; fail-closed; default-off identity; cudagraph supersedes compile. lanes=1 byte-identical smoked; lanes=2 impl fail-closed, runtime-verify in step4. NOTE: manager is all-or-nothing per replica (uncaptured B disables that replica) -> pick K to fit in step6 |
| 4 | Local byte-exact gate at scale | done | 023c99c | 100/100 byte-identical: on==off (lanes1 & lanes2), off_l2==off, off==historical baseline; replays 4650(l1)/5600(l2) fallbacks=0; 3 managers @ lanes2 (self+2 lanes, each B=1..16); FORK clean; capture ~1.35s/replica |
| 5 | Local knee measurement | done | ed53ff2 | knee 48->56 (+17%) on 5090; lag p95 @N48 229->151ms; avg B~2-3, 0 fallbacks; local-knee.md. Cloud expected to lift more (more launch-bound) |
| 6 | Cloud GPU-bound retest EC2 g6+g6e (tight budget) | done | 39956fa | L4 K=2: 16->24/box (+50%); L40S K=4: fails-at-32 (MPS-bifurcated) -> 64/box ROBUST (p95 216, ~4x p95 cut, bifurcation gone). cudagraph = tight-budget lever; maxB=8 fit both; byte-exact. cloud-retest.md |
| 7 | Drop coalescing tick (work-conserving) | done | fb51469 | NEGATIVE: graph-on MAX_WAIT 0==8 on capacity (24/box) + p95 not lower (slightly worse N10 214->297); graph-off 0 HURTS (20->16). Tick is adaptive (~0 cost) + dispatch still rewards bigger passes. KEEP MAX_WAIT=8, no code change |
