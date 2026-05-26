# Opus W1 — native ASR runtime density-memory root-cause (the 5090 N=4 knee)

Independent ("Opus") half of the paired W1 investigation. Reasoned from the **measured warm-sweep telemetry**
(`artifacts_n200_mel/logs/20260526T193756Z/`, the trustworthy warm sweep), the 0a/0c probe logs, libtorch 2.x AOTI
headers, the model artifacts on disk, and the Python optimization history. **No GPU work run** — `nvidia-smi` showed the
5090 free (809 MiB desktop only, no compute apps), but the existing telemetry already decomposes the OOM unambiguously
and a probe would risk contending with the Codex half, so I reasoned from the rich measured data instead. Every GiB
below is tagged MEASURED or ESTIMATED.

---

## TL;DR — the prompt's premise is WRONG; the binding term is not the finalize buckets

The W1 brief (and `PHASE2-PLAN.md:237`) assert the 5090 knee is capped by **"the finalize buckets' per-runner
activation × N."** **The measured data refutes this.** The finalize buckets cost **~0 GiB** at load (shared-weight
stripped wrappers) and their per-runner-slot count does **not** move the peak (0c: 8-runners×1-bucket and
2-runners×8-buckets give the *same* ~30.8 GiB peak). The single dominant N-scaling term is a **2.48 GiB full-fp32
encoder module, `enc_first.ts`, that `make_worker_context` loads once per worker** (`density_main.cpp:629`). At N
workers that is **N × 2.48 GiB**, and it alone is ~2.5 of the measured ~2.51 GiB/stream that fills the GPU. The
recommended fix is therefore **not** a finalize-memory change at all — it is to **stop duplicating the encoder
per worker** (fold the first-chunk geometry into the already-shared steady AOTI loader, or share one enc_first
across workers). That moves the 5090 knee from N=4 to **~N=40+** and the L40S to **~N=60+** (memory ceases to bind;
the next wall becomes compute/contention).

---

## 1. Memory attribution at the OOM point — REAL GiB

### 1a. The warm sweep's measured numbers (MEASURED, `…/20260526T193756Z/`)

| N | finalize runners/bucket | loaded buckets | used_before | after_loaders | **peak** | Δloaders | Δactivation (peak−AL) | per-stream activation |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1 | 12 | 4.980 | 9.590 | **12.105** | 4.609 | 2.515 | **2.515** |
| 2 | 2 | 12 | 7.324 | 11.942 | **16.966** | 4.618 | 5.024 | **2.512** |
| 4 | 2 | 12 | 9.701 | 14.319 | **24.352** | 4.617 | 10.034 | **2.508** |
| 8 | (OOM) | — | — | — | **OOM @ 30.50 in-use** | — | — | — |

(GiB; total cap 31.32 GiB MEASURED. N=8 error: *"Tried to allocate 20.00 MiB … 25.19 MiB free … 30.50 GiB in use …
26.16 GiB allocated by PyTorch"*, `…density_num_runners8_…_error.jsonl`.) Summary verdict:
`knee_n=4, binding_slo=memory_oom, binding_resource=memory, pass_to_1b=true` (MEASURED, `…_summary.jsonl`).

### 1b. The three decomposed pools

**(a) SHARED finalize weights — verified ONE copy, not per-runner (MEASURED ~2.30 GiB).**
`finalize_loader_memory.shared_constants_delta_bytes = 2.301 / 2.311 / 2.309 GiB` at N=1/2/4 — **flat**, confirming
one shared set. Mechanism, confirmed in source: the pool loads the shared weights **once** in its ctor
(`density_main.cpp:946` `load_shared_constants`), and each bucket is given those *same* `at::Tensor` handles
(`finalize_main.cpp:608` `bucket_constants.values.emplace(fqn, *tensor)` — a shallow `at::Tensor` copy sharing
storage), then `load_constants(..., user_managed=true)` (`density_main.cpp:1072`, 4th arg `true`). `user_managed=true`
is the libtorch AOTI contract for *"caller owns the constant buffer; the container does not copy"*
(`aoti_package/model_package_loader.h` `load_constants(..., bool user_managed=false)`). **Per-bucket loader delta is
literally 0.000 GiB for all 12 buckets at every N** (`finalize_loader_memory.records[*].delta_bytes = 0`;
`loader_delta_bytes = 0`). The stripped buckets are 3.9 MB wrapper packages (`stripped_finalize_buckets/*.pt2`),
not 2.4 GB weight blobs. **So weights = 2.30 GiB once. Not a per-N or per-runner term.** ✔ verified shared.

**(b) STEADY pool per-runner activation — MEASURED ~0.31 GiB/runner (small).**
The 0a steady-only `shared_steady_loader_runner_pool` probes (no finalize buckets, no shared finalize constants)
measured peak vs `num_runners`: **1→4.53, 2→4.86, 4→5.45, 16→9.17–10.0 GiB** (MEASURED,
`…shared_steady_loader_runner_pool.jsonl`). Slope 1→16 ≈ **(9.2−4.53)/15 ≈ 0.31 GiB per steady runner**. The steady
AOTI loader is **shared** across workers (`density_main.cpp:2988`: one `enc_steady`, `num_runners=N`), so steady is
*one* 2.4 GiB weight copy + N cheap ~0.31 GiB activation arenas. **Steady is NOT the binding term.**

**(c) FINALIZE buckets' per-runner activation — MEASURED to be NOT the binding term.**
Decisive control: the **0c hot-vs-mixed** probes, both at 8 workers (MEASURED, `…shared_finalize_…`):
- `finalize runners=8/bucket × 1 bucket` (= 8 finalize runner-slots) → peak **30.81 GiB**
- `finalize runners=2/bucket × 8 buckets` (= 16 finalize runner-slots) → peak **30.79 GiB**

Doubling the finalize runner-slot count (8→16) changes the peak by **0.02 GiB**. If finalize per-runner activation
were the binding resource, 16 slots would cost ~2× of 8 slots. It does not. Equivalently, in 1a the activation scales
as **N** (1,2,4 → 2.5,5.0,10.0) — *not* as `loaded_buckets × runners/bucket` (which is 12, 24, 24). **The finalize
per-forward activation is a transient bounded by the number of concurrent finalizers (≤N), freed after each finalize;
it is real but second-order, and it is NOT what fills the GPU.** ✘ the premise that this is the binding term is false.

### 1c. What actually scales as N — the 2.48 GiB per-worker encoder (MEASURED + source-confirmed)

`make_worker_context` (`density_main.cpp:624-642`) loads, **once per worker**:
`enc_first.ts`, `joint_step.ts`, `predict_step.ts`, `preproc.ts`, `session_bundle.ts`. On disk (real targets via
symlink, MEASURED):

| per-worker module | size | role |
|---|--:|---|
| **`enc_first.ts`** | **2.48 GB** | **full fp32 streaming encoder, drop=0 @16-frame trace** |
| `session_bundle.ts` | 0.14 GB | n200 **corpus mel+gold buffers** (test fixtures — harness-only, not a prod cost) |
| `predict_step.ts` | 0.028 GB | RNNT prediction net |
| `joint_step.ts` | 0.007 GB | RNNT joint |

`enc_first.ts` is the **same encoder `e`** as the steady AOTI, just traced at the first-chunk geometry
(`export_stream_encoder.py:42-56` traces `Step(0)` = drop 0 for first, `Step(drop)` = steady; **same `e`**). It is a
TorchScript module → its parameters are private per `torch::jit::load` instance → **N workers hold N independent
2.48 GiB copies**. And it is invoked **exactly once per session** (only when `is_first`,
`density_main.cpp:847-849`) — a 2.48 GiB module duplicated N× to serve one forward at session start.

**This is the ~2.5 GiB/stream.** Model: per-stream = enc_first 2.48 + steady-runner 0.31 + joint/predict 0.035 ≈
**2.83 GiB** (the measured 2.51 is a touch lower — some cross-worker allocator reuse — but enc_first dominates either
way). Cross-check against 0c: 8 workers × 2.48 enc_first ≈ 19.8 + shared base ~4.7 + steady arenas + residue ≈ **~30.8
GiB = the measured 0c peak**, and it is the same whether finalize is 8×1 or 2×8 — exactly the observed behavior.

**Dominant term: the per-worker `enc_first` duplication (N × 2.48 GiB). The finalize buckets are ~0. The shared
weights are a flat 4.6 GiB (steady 2.4 + finalize 2.3, one copy each).**

### 1d. Second, independent finding — the OOM is partly a same-process sweep artifact (MEASURED)

The sweep runs N=1,2,4,8 **in one process** with `cleanup_cuda_cache()` between (`density_main.cpp:3505`, which is
`cudaDeviceSynchronize + CUDACachingAllocator::emptyCache`, `:335`). Yet `used_before` **grows 4.980 → 7.324 → 9.701
GiB** across N — i.e. ~4.7 GiB is **not returned to the driver** between runs (`gpu_used_bytes` reads
`cudaMemGetInfo`, the driver view, `:313`). Cause: AOTI CUDA modules / cuDNN-cuBLAS workspaces / lazily-loaded kernels
held across iterations that `emptyCache()` does not free. A **fresh-process** N=8 projects to ~9.6 + 2.51·8 ≈ **29.7
GiB** — still right at the 5090 edge (would likely still OOM on the 20 MiB margin + fragmentation), but the same-process
residue makes the observed knee **pessimistic**. Two consequences: (i) the *real* fix is the per-stream term, not the
residue; (ii) for a trustworthy knee, the sweep should run **fresh-process-per-N** (matches the 0a memory-gate
methodology already noted in `PHASE2-PLAN.md:128`). FLAG this so the N=8 OOM is not over-read.

---

## 2. Reduction options — saving × density × T1 risk

The brief's four options (a–d) are all aimed at the *finalize* pool. **Per §1 they target ~0–2nd-order memory** and
will barely move the knee. I evaluate them honestly, then add the option that actually matters (e).

| # | option | memory saving (5090) | new knee | T1 / correctness risk | verdict |
|---|---|--:|--:|---|---|
| **(a)** | cap finalize `num_runners`→1/bucket | **~0 GiB** (already capped at `min(N,2)`, `density_main.cpp:1115`; bucket load delta is 0; 0c shows runner-slot count doesn't move peak) | N=4 (unchanged) | none (serial finalize ⇒ may cost finalize concurrency/tail) | **near-useless for memory** |
| **(b)** | load+warm only the buckets the workload hits | **0 GiB more** — the 1a sweep **already** does this (`preload(needed_buckets)` loads 12/32, `:2994`); and buckets cost ~0 each anyway | N=4 (unchanged) | none | **already done; no memory headroom left here.** Still worth keeping for the warmup reconciliation (§4) |
| **(c)** | shared/streamed finalize activation (one buffer set + lock) | small (finalize transient is ≤N concurrent, 2nd-order) | ~N=4–5 | low if serialized correctly; adds a lock ⇒ finalize tail under load | **2nd-order; do later if finalize transient ever dominates (it doesn't now)** |
| **(d)** | padded-bucket consolidation (one bucket serves a T-range) | **0 GiB** (buckets are already ~0; fewer buckets saves ~nothing) AND introduces T1 risk | N=4 | **HIGH — NOT token-safe.** Confirmed below. | **REJECT** (cost with no benefit) |
| **(e)** | **dedup the per-worker encoder (`enc_first`)** — *the real lever* | **(N−1) × 2.48 GiB** | **~N=40+ (5090), ~N=60+ (L40S)** | low (see §3/§5) | **RECOMMENDED** |

**(d) padded-bucket T1 risk — CONFIRMED, reject.** The 1.3b/Python finding holds: the production-side cudagraph
guidance is explicit — *"Capture exact (B,T,drop_extra,keep_all_outputs=True) buckets … **Avoid padding at first;
exact buckets preserve the byte-exact story**"* and *"**capture only exact buckets; no padding until exact buckets are
proven**"* (`proj-2026-05-21-1959-cudagraph/finalize-optimization-suggestions.md:65,77`). The encoder is a Conformer
with depthwise convolutions whose right-context reaches into trailing frames; zero-padding a shorter finalize into a
larger-T bucket bleeds those zeros into the conv receptive field and is **not** token-exact. The Python code carries a
`replay_finalize_padded` path (`cudagraph_encoder.py:297-308, 643`) but it is **opt-in and unproven**, and the
*default* finalize CUDA graph is **off entirely** (`NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE`, `:320-321`). Since the native
buckets cost ~0 GiB, consolidating them buys nothing and risks T1. **Do not pad.** (Keeping ~32 exact buckets is the
*correct* per-the-Python-history design — see §5.)

---

## 3. RECOMMENDED FIX + expected new knee

**Fix: eliminate the per-worker encoder-weight duplication. The encoder must exist as ONE shared weight copy that all
N worker streams use for both first-chunk and steady-chunk geometries — exactly as the Python server already does.**
Two implementations, preferred first:

**Fix-1 (preferred): fold the first-chunk geometry into the shared steady AOTI loader.** The steady encoder is already
a shared AOTI loader with `num_runners=N` and one user-managed weight copy (`density_main.cpp:2988`). `enc_first` is
the *same encoder* `e` at a different input geometry (drop=0, T=16 vs drop=2, T=25). Export the first-chunk geometry as
a second AOTI **entry/bucket that reuses the same shared constants** (the finalize pool already proves this pattern:
many stripped packages → one shared 2.3 GiB constant set, `finalize_main.cpp:589-618` + `density_main.cpp:946,1072`).
Then drop `enc_first.ts` from `make_worker_context` entirely. Net per-stream cost → steady-runner arena (~0.31) +
small RNNT modules (~0.035) ≈ **~0.35 GiB/stream**, and the encoder weights stay **one** shared copy.

**Fix-2 (cheaper to land, smaller win): share one `enc_first` module across workers.** Load `enc_first.ts` once
(like `enc_steady`) and pass a reference into each `WorkerContext` instead of `load_module_on_device` per worker
(`density_main.cpp:629`). One forward per session means contention is negligible; if paranoid, guard the rare
first-chunk call. This removes (N−1)×2.48 GiB with a ~10-line change but keeps a *second* full encoder weight copy
(enc_first TS + steady AOTI) resident — so per-stream → ~0.35 GiB but the fixed base carries +2.48 GiB.

**Expected new knee (ESTIMATED from the measured per-stream slope; fresh-process, 90% headroom):**

| target | total | per-stream now | knee now | per-stream after fix | **knee after fix** |
|---|--:|--:|--:|--:|--:|
| 5090 | 31.3 GiB | 2.51 | N=4 (meas.) → ~8 fresh | ~0.35–0.49 | **~40–45** |
| L40S | ~44–46 GiB | 2.51 | ~13 | ~0.35–0.49 | **~60–69** |

After the fix, **memory stops binding well past any plausible compute knee** — the 5090 encoder-saturation knee was
already observed near N≈4–8 in the 0c contention probes (`PHASE2-PLAN.md:285`: kernel p50 5.28→13.63→28.54 ms with N).
So the practical effect is: **memory is removed as the 5090 floor; the binding resource flips to GPU
compute/contention (or the asyncio/host ceiling), which is the *transferable* density question** and what the L40S
sweep should now measure. Do not expect N=40 of *useful* throughput on the 5090 — expect the knee to move off "memory"
onto the real compute/scheduler wall, which is the point of W1.

**Honesty:** the post-fix per-stream (0.35–0.49) is ESTIMATED from the 0a steady-runner slope (0.31, MEASURED) + small
RNNT modules; the knee figures assume the §1d same-process residue is also fixed (fresh-process-per-N). The *direction
and magnitude* (≈5–10× more streams/box) are robust; the exact N is ±20%. A 5-minute confirmation: re-run 1a with
Fix-2 (share enc_first) and read `peak`/`used_after_loaders` — per-stream should collapse to <0.5 GiB.

---

## 4. Warmup-vs-load-only reconciliation

The brief notes a tension between the **mandatory per-bucket warmup** (every finalize bucket must be forward-warmed to
pay the CUDA-12 lazy-module-load cost up front — the fix for the earlier 234 ms cold-start,
`reviews/opus-finalize-234ms-investigation.md`) and **loading only the buckets the workload hits**. **There is no
real tension, and the code already resolves it correctly:** the sweep loads only the needed buckets
(`preload(needed_buckets)`, `density_main.cpp:2994`; 12/32 loaded) **and** warms exactly those — each worker
forward-warms its representative buckets in the warmup loop (`:3047-3086`, `run_finalize_density` per
`worker_bucket_reps`). So the policy is already **"load-and-warm the NEEDED subset"**, which is both memory-minimal and
cold-start-free. Two notes: (i) since buckets cost ~0 GiB to load, "load only needed" is a *latency/warm-time*
optimization, not a memory one — it does **not** help the OOM; (ii) `CUDA_MODULE_LOADING=EAGER`, if set, would raise
*load-time* memory by forcing all cubins resident — but the buckets are 3.9 MB wrappers, so even EAGER is negligible
here. **Recommendation: keep load-and-warm-needed; it is correct and orthogonal to the memory fix.**

---

## 5. Architectural insight + Python comparison

**Is the per-(drop,T)-bucket runner pool the root inefficiency? No — the pool is *fine*; the per-worker encoder
duplication is the root inefficiency.** Two separable design choices got conflated in the W1 framing:

1. **The exact-T finalize bucket set (good, Python-endorsed).** Keeping ~32 exact `(drop,T)` buckets is exactly what
   the Python optimization history prescribes (exact buckets, no padding —
   `finalize-optimization-suggestions.md:65,77`). The native runtime made these **memory-free** by stripping weights
   into one shared `user_managed` constant set (the `FinalizeBucketLoaderPool`, `density_main.cpp:914-1112`). This is a
   genuinely good design and is **not** the problem. The only sharp edge is `num_runners` per bucket, but it is already
   capped at `min(N,2)` (`:1115`) and 0c proves runner-slot count doesn't move the peak.

2. **The encoder split into two separately-serialized modules, one duplicated per worker (the bug).** The native
   runtime exports `enc_first` (TorchScript, drop=0) and `enc_steady` (AOTI, drop=2) as **two artifacts**
   (`export_stream_encoder.py:56`; `aot_compile.py` AOTI-compiles only steady). It shares `enc_steady`'s weights but
   **duplicates `enc_first` per worker** (`density_main.cpp:629`). That is the entire N-scaling memory wall.

**How Python bounds finalize (and encoder) memory — the lesson:**
- **One encoder module for all geometries.** The Python server has **no `enc_first`** (zero hits for `enc_first` in
  `src/nemotron_speech/server.py`). It calls the single shared `model.encoder.cache_aware_stream_step` with the
  appropriate `drop_extra` for both first and steady chunks. **One weight copy, period** — no per-stream encoder
  duplication. This is the design the native runtime should match (Fix-1).
- **No `num_runners` activation pool at all.** Python finalize is the same warm encoder module the steady path uses;
  concurrency is the batch dimension `B` (continuous batching / lanes), not N independent activation arenas. So Python
  has no "N × per-runner activation" multiplication on the finalize side to begin with.
- **The finalize CUDA graph (the only per-key static-buffer structure) is bounded by being default-OFF and
  capture-on-demand.** `BucketedCudaGraphEncoder` keys graphs by exact `(B,T,drop,keep_all_outputs)` and stores them in
  `self._finalize_buckets` (`cudagraph_encoder.py:357`), but it is gated behind `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE`
  (default off, `:320-321,341-345`) and only captures **explicitly-warmed keys** (`capture_finalize`, `:453`;
  `_finalize_requested_keys`, `:358`) — production-safe budgets are *"B=1..2 or 1..4 and final-T from telemetry"*
  (`finalize-optimization-suggestions.md:65`), a handful of graphs, not 32×N. There is **no eviction** — the cap is
  simply *"capture few, exact, on purpose"*. When the graph is off (the default), finalize is a single eager path with
  zero per-key static buffers. **That is how Python keeps finalize memory bounded: shared weights + no runner pool +
  opt-in, hand-capped graph capture.**

**Per-target math (does L40S even hit this wall?).** With the *current* ~2.51 GiB/stream and a ~5.7 GiB fixed base,
L40S (~44–46 GiB) would knee at **~N=13–14 on memory** — so **the wall is NOT 5090-32GB-specific; the L40S would also
be memory-bound at ~N=13** under the present per-worker-encoder design, just at a higher N. The MEMORY.md note that
"L40S has headroom" is true only relative to N=4; the L40S's *SLO-robust* density target (~16–20/box) sits right at
that ~N=13 memory wall, so **the duplication would cap the L40S below its compute potential too.** After the fix
(per-stream ~0.35–0.49 GiB) both boxes clear any plausible compute knee with room to spare. ⟹ **the encoder-dedup fix
helps both targets and is the prerequisite for the L40S sweep to measure the *real* (compute/host) binding resource
rather than this artifact.**

---

## Honesty ledger
- **MEASURED:** all peak/used_before/after_loaders/shared_constants_delta/per-bucket loader deltas (1a N=1,2,4,8);
  0a steady-only peak vs num_runners; 0c hot-vs-mixed peaks; the `binding_resource=memory` verdict; enc_first.ts disk
  size (2.48 GB) and that it traces the same encoder `e`; that enc_first runs once/session and is loaded per worker;
  that finalize buckets load with 0 delta via shared user_managed constants; that Python server has no enc_first and
  the finalize graph defaults off.
- **ESTIMATED:** the post-fix per-stream (0.35–0.49 GiB) and the resulting knees (5090 ~40–45, L40S ~60–69) — derived
  from the measured 0.31 GiB/steady-runner slope and the measured 2.51 GiB/stream; ±~20% on N. The 1.0 GiB
  CUDA-context/handles term is rough. enc_first in-memory == on-disk (2.48 GiB) is inferred from the serialized fp32
  size (could not load torch in the base python; the steady AOTI is the same 2.4 GiB encoder, consistent).
- **NOT measured (would need a probe / fresh-process run):** the exact finalize-forward transient per concurrent
  finalizer (shown ≤2nd-order by 0c, not isolated); the fresh-process-per-N knee (projected, not run).
