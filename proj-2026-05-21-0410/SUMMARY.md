# Streaming-ASR GPU-throughput optimization — final summary

Project: `proj-2026-05-21-0410`. Date: 2026-05-21. Goal: raise the per-GPU realtime keep-up knee of
`src/nemotron_speech/server.py` from `batch_size=1` toward the batch-amortized ceiling, improving streams/$.

## TL;DR

- **Built a correct, byte-exact, default-off continuous-batching scheduler.** Local realtime keep-up knee
  **16 → 56 streams (3.5×)** on the RTX 5090, byte-exact at scale (200/200 canary), rollback-safe.
- **Batching is a self-host / fast-CPU lever, NOT a cloud lever.** On Modal T4/L4 it gave **no improvement**
  (T4 0.8×, L4 1.2×) — the slow-CPU cloud knee (~5) is too low for batches to form.
- **The real bottleneck is the single serial model-call lane** (one launch-dispatch-bound call at a time,
  ~11–15ms, ~hundreds of tiny kernels). Batching only amortizes it when enough streams are *simultaneously*
  ready; the GPU itself is far from saturated (forced-batch ceiling ≥220 realtime-stream-equivalents).
- **Cloud throughput needs different levers** (next phase): CUDA-graphs (cheaper call) + parallel lanes (use
  the ~16 idle cores), not batching.

## What was built (all behind flags, default = today's exact behavior)

| Flag | Effect |
|---|---|
| `NEMOTRON_SCHEDULER_B1=1` | Central scheduler replaces per-session workers (B=1 per tick). Rollback-safe: off = the original `_continuous_session_worker` + `inference_lock` path, byte-identical. |
| `NEMOTRON_BATCH_SCHED=1` | Steady-state continuous batching: group same-key ready sessions → one `conformer_stream_step(B)`/tick → scatter. Implies `greedy_batch` + TF32-off. Requires `SCHEDULER_B1`. |
| `NEMOTRON_ENCODER_COMPILE=1` | torch.compile (CUDA-graphs) the encoder for static B=1 buckets. Off by default. |

Defaults when batching on: `greedy_batch` (loop_labels=True, cuda_graph_decoder=False), TF32 off,
`MAX_SIZE=32`, `MAX_WAIT=8ms`, device-aware startup cap on MAX_SIZE, fail-closed (CTC/EOU/beam/unsafe → B=1).

Correctness scaffolding: cache stacked dim1 (channel/time) / dim0 (len), owned-clone scatter, clone-in
hypotheses + assign-on-success, per-session generation tokens (no stale scatter), `try/finally` drop_extra,
device-aware memory cap (retained-after-churn = 0).

## Results

### Local (RTX 5090) — knee by config

| config | realtime keep-up knee | note |
|---|---:|---|
| B=1 scheduler (no batching) | 16 | the apples-to-apples baseline (old per-handler path ~14) |
| compile-only | 24 | 1.5× single-stream; helps B=1, not the batch knee; ~124s warmup |
| batch-only (MAX_SIZE=4) | 40 | 2.5× |
| + batched preprocessor (7a) | 48 | |
| **+ MAX_SIZE=32 (final)** | **56** | **3.5× vs B=1 scheduler**; N=1 TTFS p95 ~17ms (unchanged) |

- **TF32-off is byte-compatible** with the shipped (greedy + TF32-on) baseline — 8/8 exact + 200/200 canary —
  at only **+1.8%** per-chunk cost (launch-bound, so fp32 matmuls barely matter).
- **Forced-batch GPU ceiling: ≥220 realtime-stream-equivalents at B=46** (`160·B/T(B)`), still sublinear — the
  GPU is not compute-saturated; the cap was conservative memory, not compute.
- **In-phase knee 115 (2.1× vs out-of-phase 56)** — phase alignment fills full B=32 batches; confirms the
  out-of-phase knee is a dispatch/arrival artifact, not a GPU/client limit.

### Modal cloud (T4 + L4) — batching does NOT help

| GPU | batch=1 knee / $/stream-hr | batched knee | improvement |
|---|---|---:|---:|
| T4 | ~5 / $0.12 | 4 | **0.8× (regressed)** |
| L4 | ~5 / $0.16 | 6 | **1.2×** |

At the cloud knee (~5) the effective batch is ~1: independent 160ms cycles are out of phase, so you'd need
~250 concurrent streams to fill B=32 — but the slow CPU overloads at ~5. The local 3.5× does not transfer.

## The lane analysis (why, and what it means)

The per-chunk bottleneck is **host-side kernel-launch/dispatch on a single core** (not GPU FLOPs — H100 ≈ L4 ≈
RTX-PRO-6000 on Modal; the local 5090's edge is its 5.7GHz desktop core). The scheduler runs **one model call
at a time** (single `inference_lock`), each ~11–15ms, so its throughput = `avg_B / T(B)`. TTFS rises as a
single-server queue: gently below the knee, then a cliff as utilization → 1.

- **Out-of-phase realtime streams keep avg_B small (~4 at the N=56 knee)** → lane capacity ~`160·4/13 ≈ 49`
  streams → knee ~56. To fill B=32 you need ~250 concurrent streams (or phase-alignment, +latency).
- **In-phase fills batches (avg_B → 32)** → knee jumps to 115, *but* stalls there (not the ~180 ceiling)
  because of a **specific unbatched code path**: at high N, `_scheduler_drain_ready_barrier_locked` drains a
  session's backlog **one chunk at a time (B=1)** before `vad_stop` finalize. This is correct but slow — a
  fixable scheduler inefficiency, not a hardware limit.

## Production recommendation

| Deployment | Config | ~knee | ~$/stream-hr |
|---|---|---:|---|
| **Self-host bare-metal (fast CPU, e.g. RTX 5090)** | batching on (greedy_batch, TF32-off, MAX_SIZE=32), compile optional | **~56** | **~$0.02** (HW+power) |
| **Modal / slow-CPU cloud** | batching gives ~nothing → run the **cheapest GPU (T4) + scale horizontally** | ~5/GPU | **~$0.12** (T4) |

- Decoder/precision: `greedy_batch` + TF32-off only when batching; both byte-compatible with the baseline.
- Compile: off by default (helps B=1 / overload latency, not the batch knee; large warmup). It is the
  **promising cloud lever** (cloud is 46–68% launch-gaps → bigger collapse) — but untested on cloud.

## Residual risks / caveats

- **Throughput scales with *in-phase* concurrency.** Independent realtime streams cap the achievable knee well
  below the GPU's forced-batch ceiling (56 vs ~220). This is inherent to realtime independent arrivals.
- **`vad_stop` barrier-drain is unbatched** → caps high-N (in-phase 115 not 180). Fixable.
- **Cloud batching is a no-op / mild regression** — do not enable `NEMOTRON_BATCH_SCHED` on slow-CPU cloud
  without the cheaper-call levers; greedy_batch+TF32-off is marginally slower than plain greedy at B=1.
- **Byte-exactness:** batched per-stream output is byte-identical to single-stream on the tested sets, and
  state is fp32-equivalent (TF32 must be off — TF32 drifts the cache ~0.03, text still held but the gate fails).
- Unrelated: the multilingual checkpoint's front-drop is **H1 (model decode fragility), not our code**
  (`docs/multilingual-wer-deepdive/`).

## Next phase (lane-fixes — NOT part of this plan)

The fixes that would help the **cloud / low-N** regime (all attack the launch-dispatch lane, not the GPU):
1. **CUDA-graphs — via MANUAL capture, not torch.compile.** Local foothold = Step-4 encoder torch.compile
   (1.5×). But the cloud test (Step 10b) showed `torch.compile(reduce-overhead)` **warmup never completes on
   T4/L4** (inductor codegen/autotune too slow on the small/slow nodes; >6min, no completion) → not usable as
   a cloud lever and a non-starter for cold-start. **Next: manual CUDA-graph capture** (record+replay the
   launch sequence — no inductor codegen → fast warmup, production-viable). Hypothesis (graphs lift the cloud
   knee) remains UNTESTED — manual capture is the way to test it.
2. **Parallel lanes — PROBED (`parallel-lane-feasibility.md`): viable but modest.** Not a GIL wall (K=4
   threads = 2.75× CPU-in-wall), but Amdahl-bounded: the step is ~70% GPU-active (serial floor) / ~30% host,
   so threads + per-lane CUDA streams give only **~1.4× local** (measured 1.41×). Processes don't help (GPU
   time-shares). **Expected bigger on cloud** (host fraction 46–68% → ceiling ~1.9–3.1×) — needs a cloud
   re-measure. Realistic = a guarded N=2–3 thread-lane pool (own CUDA stream/lane; guard the global
   `drop_extra_pre_encoded`; per-lane sync). Note: lanes & CUDA-graphs are partial substitutes (both attack
   the host fraction).
3. **Batch the `vad_stop` barrier-drain** — quickest knee bump (in-phase 115 → toward 180); fixes the
   unbatched finalize path.
4. **Coarse phase-alignment** (global tick) — confirmed ~2× lever (in-phase 56→115), at a latency cost.

Reachability note: the GPU's ~220 ceiling is a **batching** number (one B=46 call), reachable only by
filling batches (offline/in-phase). Lanes (~1.4–3×) and CUDA-graphs (collapse the launch fraction) are the
realtime levers; neither reaches 220 with independent realtime traffic.

## Artifacts

`PLAN.md` (steps + per-step gates/commits) · `local-validation.md` · `max-parallelism-sweep.md` ·
`highN-validation.md` · `inphase-confirmation.md` · `modal-resweep.md` · `proj-2026-05-20-modal-cost/RESULTS.md`
(the batch=1 cost study + the SYNTHESIS) · probes (`probe_*`, `test_batch_*`) · `codex-jobs/` (delegation logs).
