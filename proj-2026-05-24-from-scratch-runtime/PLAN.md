# Plan: From-scratch persistent serving runtime — break the scheduler/launch ceiling

Project directory: `./proj-2026-05-24-from-scratch-runtime`
Status: **v6 — REVIEWED** (5 paired adversarial rounds folded; `reviews/`). Verdict: sound + actionable;
**green-light Wave 1 only** (the cheap kill-shots); do not fund Wave 2 until Wave 1 clears the three-conjunction bet.
See the Review log at the bottom.

> **TL;DR.** A persistent native (Rust/C++) ASR serving runtime to lift the **p95/p99 tail + streams/box density** —
> **NOT p50** (VAD+WAN-bound; only ~12–19 ms is movable by any engine). The real bottleneck is the single asyncio
> thread + coarse `inference_lock`/exclusive-gate + GIL-bound `greedy_batch` label-looping decode — not "the GIL
> serializing CUDA launches." **Outcome space = native Rust/C++ (B1) or STOP** (user dropped free-threaded-Python B4
> and the in-tree-extension B5, 2026-05-24). **THE BET (both must hold):** (1) the cheap Python plan
> (`proj-2026-05-24-0859`) leaves a residual worth ~40–60 eng-wk + a second stack; (2) that residual is capturable by a
> native runtime (GIL/scheduler-bound AND a single native process can overlap, not MPS/bandwidth-bound). Early-exits:
> 0.0 / 0.1. **Green-light Wave 1 (cheap kill-shots) only.** Budget: prototype ~12–20 eng-wk, production ~26–42 eng-wk.

> This is the "Q3 from-scratch" endgame deliberately scoped OUT of `proj-2026-05-24-0859/PLAN.md` (the near-term
> Python-stack levers: admission, finalize priority, off-thread dispatch, host-sync compression, padded-T bucket). It
> develops in a separate subtree on separate EC2 instances / the local 5090. **But it is explicitly SEQUENCED AFTER
> that plan (see §0 Worth-it gate)** — we do not fund the rewrite until the cheap Python levers land and a *named*
> residual density/tail gap remains.

---

## 1. Why this project might exist (the thesis — corrected to the actual serializers)

Two independent investigations (`proj-2026-05-23-1731/roofline-COMBINED.md`; memory `roofline-and-real-limit`)
converged:

- The finalize **encoder is memory(weight-stream)-bound, ~3.4× above the L40S floor** (9.7 ms enc / 13.4 ms model /
  22 ms wall vs a ~2.9–3.0 ms floor — `roofline-COMBINED.md:13-23`). It is 24 sequential conformer layers of tiny
  (M≈7) kernels that can't saturate DRAM. **fp16 was 0.79× SLOWER** and "only helps once fused"
  (`roofline-COMBINED.md:20-23`). → **A faithful port of the kernels buys ~nothing on latency by itself.**
- **TTFT p50 ≈ 246 ms = ~200 ms fixed VAD trailing-silence + ~23 ms WAN + ~22 ms server finalize.** Even a perfect
  engine moves p50 by **only ~12–19 ms** (`roofline-COMBINED.md:29-33`). The biggest p50 lever (the VAD window) is
  **outside** this stack.
- **The real prize is the p95/p99 tail at load + streams/box density.**

### What actually serializes today (NOT "the GIL serializing CUDA launches")
Review round 1 corrected the v1 framing. Multi-lane concurrency *already exists*: per-lane single-worker
`ThreadPoolExecutor` + per-lane `torch.cuda.Stream` + a **separate full NeMo model replica per lane**
(`server.py:3109-3137`, `:3155-3163`), and model calls already run off the event loop via `run_in_executor`
(`:3087-3095`, `:3183-3192`) — PyTorch releases the GIL during CUDA dispatch, so steady kernel launches across lanes
are *not* purely GIL-serialized. The **actual** serializers, in source:

1. **Single asyncio scheduler/I/O thread** — one `_scheduler_loop` does socket I/O + all scheduling + dispatch; needs an
   explicit `sleep(0)` or socket I/O starves (`server.py:4456-4491`).
2. **`inference_lock` (an `asyncio.Lock`, `:593-594`) held across whole model calls** — single-lane batched call
   (`:4990-5005`), multi-lane non-batched chunk (`:5310-5323`), finalize fallbacks (`:6779-6785`, `:7698-7734`,
   `:7738-7771`).
3. **The exclusive pinned-lane gate** `_scheduler_exclusive_model_path` (`:3213-3233`): only *steady normal chunks* may
   share lanes; first chunks, finalize, and other geometries run exclusively after all in-flight lane tasks drain
   (`:3242-3249`, `:3271-3276`).
4. **Lane-end `stream.synchronize()`** inside each lane worker (`:3175-3178`) — a deliberately synchronous lane boundary.
5. **The Python RNNT decode** — **the DEPLOYED path is `greedy_batch` + `loop_labels=True` = LABEL-looping** (not
   frame-looping; corrected round 2): production sets `NEMOTRON_BATCH_SCHED=1`/`BATCH_FINALIZE=1`
   (`deploy/launch_multiproc.sh:42-45`) → `decoder_strategy="greedy_batch"` (`server.py:726-728`) →
   `loop_labels=True, use_cuda_graph_decoder=False` (`server.py:1463-1474`) → NeMo's label-looping computer
   (`rnnt_greedy_decoding.py:635-647`). It carries a `LabelLoopingStateItem` and merges partial hypotheses across chunks
   in place; data-dependent control flow + host syncs, GIL-held between launches. (The plain `greedy`/`loop_labels=False`
   *frame-looping* branch at `server.py:1486-1492` is the NON-batched fallback and **rejects `partial_hypotheses`**
   `rnnt_greedy_decoding.py:807-815` → cannot serve streaming continuation.)

**Corrected thesis:** a persistent native runtime that (a) does dispatch + decode on **real worker threads, off a
single event loop**, (b) replaces the coarse `asyncio.Lock`/exclusive-gate with **fine-grained per-session ownership**
so finalize overlaps steady, (c) **shares one read-only weight copy** across lane threads (vs today's K× full replicas),
and (d) keeps decode on-GPU with **no per-frame `.item()`** — could lift the tail + density. The launch-ceiling/GIL
story is *one* component, not the whole story, and may still require MPS/green-contexts (see Spike 0.1).

### Targets — **all are hypotheses to falsify, inherited from a first-principles roofline, not measurements**

| Metric | Today (Python) | From-scratch *target to validate* | Contingent on |
|---|---|---|---|
| server finalize | ~22 ms | parity→**6–10 ms** | **6–10 ms requires FUSION (B2/B3), NOT the default B1 port** (`roofline-COMBINED.md:56-60`) |
| steady throughput | 86% B=1 | up to **3–5×** | the workload *can* batch without added wait — **unproven; Spike 0.5 trace-sim** |
| reliable capacity | ~20/box today; **~28/box** after the Python plan (K=4, `proj-2026-05-24-0859:11-14`) | **40–48/box** (aspirational) | shared weights + real overlap **AND** one CUDA context isn't launch-serialized (Spike 0.1+0.11). The native delta to beat is **vs ~28, not vs 20** |
| **TTFT p50** | **246 ms** | **≈ unchanged** | VAD+WAN — **OUT OF SCOPE** |

**Non-goals:** moving p50; changing WER/accuracy; reimplementing/shrinking the VAD window; changing client protocol
semantics. **If a step's only justification is p50, it is out of scope.**

---

## 0. The "worth-it" gate (HARD — consumes Wave-1 outcomes; gates Wave-2 funding + all of Phase 1+)

The near-term Python plan (`proj-2026-05-24-0859`) already orders admission → padded-T bucket → host-sync compression →
finalize priority → off-event-loop dispatch under byte-exact gates, and the roofline recommended that order *before*
from-scratch work (`roofline-COMBINED.md:66-69`). Several from-scratch phases (admission, finalize priority, padded-T,
off-thread dispatch) **mirror** that plan. Therefore:

**THE BET (state it up front).** Outcome space is **native Rust/C++ (B1) or STOP** — the user removed B4 (free-threaded
Python) and B5 (in-tree extension) (2026-05-24). This project lives only if **BOTH** hold; each has an early-exit spike:
1. The Python plan (`proj-2026-05-24-0859`) leaves a **residual density/tail gap whose business value exceeds ~40–60
   eng-wk + a second stack to maintain** (early-exit: 0.0); AND
2. that gap is capturable by a **native runtime** — i.e. it's **GIL/single-thread-scheduler-bound AND a single native
   process can actually overlap finalize+steady**, not MPS/context-launch-/bandwidth-bound (else native ≈ the same
   MPS/multi-proc topology as Python and density doesn't improve) (early-exit: 0.1).

**B4 removal raises the bar.** With no free-threaded-Python shortcut, the *only* remedy for a real GIL/scheduler-bound
residual is the full native build (~40–60 eng-wk + second stack) — there is no cheap intermediate. So conjunct 1's
threshold must clear the *full* native cost, and conjunct 2 must be proven before committing (see 0.1, which now carries
a native launch-overlap microbench since the B4 end-to-end probe that used to validate this is gone).

**USER DECISIONS (2026-05-24 collaborative planning):**
1. **Is density the right problem? → YES.** The user affirmed density/tail is a strategic priority worth pursuing *if
   the residual is real*. So this is NOT a lean-STOP project; we proceed toward the gate. (The 0.0-pre ceiling arithmetic
   stays as an expectations check, not as a STOP prior.)
2. **The 0.0 threshold number → DEFERRED** until the Python plan lands and gives a measured baseline. The proposed
   numeric thresholds in `decision-template.md` are **reference-only** until then; freeze them *before* collecting
   Wave-1 data, set them *after* the Python baseline exists.
3. **Baseline → KEEPS MOVING (not frozen).** The native residual is measured against *whatever Python is at the time*;
   the native build must continuously out-perform the latest Python baseline. Accepted risk: the residual may shrink
   (even asymptote) as Python improves — see risk 13.
4. **B5 (in-tree native-decode extension) → OFF THE TABLE.** No middle path.
5. **B4 (free-threaded Python 3.13t) → OFF THE TABLE (2026-05-24).** The user is not interested in a free-threaded-Python
   approach. **The outcome space is strictly: native Rust/C++ (B1) or STOP.** Spike 0.3 is removed; its conjunct-2
   validation role moves into 0.1 (a native launch-overlap microbench). A decode-GIL-bound-but-full-runtime-not-worth-it
   result now resolves to **STOP** (no cheaper remedy remains).

Context still assembled (informs expectations, not a STOP prior now): p50 immovable; steady 86% B=1 (batching ~0
cloud-GPU benefit — `streaming-batching-outcome`); density ceiling ~28/box post-Python, native upside
triple-conditional.

- [ ] **0.0-pre — Residual-CEILING arithmetic (FREE / paper; the cheapest kill; do FIRST, before any spike spend).**
  Bound the prize from above *today*: best-case native delta = **~48 (aspirational) − ~28 (post-Python K=4) ≈ 20
  streams/box, AND only if 0.1+0.9+0.11 all clear** — a small, triple-conditional prize against an engineer-year **plus
  the ongoing dual-stack carry** (every NeMo upgrade / model swap / CUDA bump now hits two codebases — the ~40–60 eng-wk
  is BUILD only, not carry). If that ceiling can't clear the (written) threshold even assuming all native gates pass,
  **STOP without spending on spikes.** Write it into `reviews/decision.md`. **Strategic framing:** density is a COGS
  lever — quantify projected-scale COGS and compare against (a) just adding cheap L4 boxes, (b) spending the eng-year on
  multilingual quality / the front-drop bug [[multilingual-front-drop-bug]] / a p50-moving VAD.

- [ ] **0.0 Worth-it gate (value-based, NOT target-based).** Do NOT fund the expensive ports (Wave 2) until: (i) the
  Python levers have landed + been measured on L4/L40S; (ii) a **named, quantified residual** (p95/p99 ms gap AND
  streams/box delta) is established vs the *measured* Python result — **note the realistic density is ~28 in-budget
  streams/box at K=4, NOT the old 64/box headline** (`proj-2026-05-24-0859/PLAN.md:11-14`); (iii) an explicit
  **value-vs-cost** judgment says that residual justifies ~40–60+ eng-wk + a second stack. **The pause trigger is
  "residual value below threshold," NOT "Python reached some absolute streams/box number."** Record the residual
  numbers + the exact Python baseline commit in `reviews/decision.md`.

**Spike waves (round 4 — separate "can run early" from "should FUND early"; Budget-A ports are 12–20 eng-wk, §9):**
- **Wave 1 — cheap existence/path killers, in DECISION-VALUE order (path-forward review re-sequenced this to front-load
  the STOP evidence — the rounds put it last):**
  1. **0.0-pre ceiling arithmetic** (free; may STOP here with zero spend).
  2. **0.5 as a one-pass histogram + synthetic phase-sensitivity** (afternoon; existing "86% B=1 / phasing 115-vs-56"
     data + 160ms-cadence-vs-8ms-window arithmetic likely *already* kills the 3–5× batching + steady-graph claims —
     run the histogram before the full simulator; may not need the server instrumented at all for the kill direction).
  3. **Wire the Python plan's Step-5 GIL probe (`proj-2026-05-24-0859:148-156`) into 0.1** — its decode-vs-glue
     attribution pre-pays conjunct 2 for free; make it a required, 0.1-consumable deliverable, don't re-derive.
  4. **0.1 = (a) reduced binary** (single-process-single-context vs MPS overlap — the one question the tree turns on)
     **+ (b) a native launch-overlap microbench** (N OS threads, no GIL, each driving a CUDA stream replaying the
     captured encoder graph + a mock decode) — this is the conjunct-2 PROOF that used to be B4 stage-2. It's more than
     a py3.13t import probe but reuses the captured graph (no decode-equivalence needed) → still far cheaper than the
     full B1 build, and it's the cheapest remaining way to prove "GIL-free native dispatch lifts the knee" BEFORE
     committing to B1.
  5. **0.11 GPU memory** — only if 0.5 keeps steady-graphs alive. **0.9 / 0.11 analysis is already COMPLETE** (paper).
  6. **0.7 aarch64** — non-gating platform pre-check, whenever GB10 exists.
- **Wave 2 — expensive byte-exact ports (FUND ONLY IF Wave 1 ⇒ "residual exists AND it's native-capturable
  (GIL/scheduler-bound + single-process overlap proven in 0.1)"):** 0.6a (decode), 0.8 (preprocessor), 0.2 (encoder
  export), 0.10 (runtime contract). Baseline-independent (frozen fixtures) so they *can* start early, but should not be
  *spent* before Wave 1 clears.
- **PRE-REGISTER the Wave-1 pass/fail thresholds BEFORE collecting data** (they are kill decisions — defining them later
  in 0.4 is too late): for **0.1** — the required overlap factor vs Python/MPS, max queue/lane wait, max added latency;
  for **0.5** — the median/p95 B target ("≫1" made numeric), the minimum exact-B graph replay hit-rate, the max
  eager-fallback %, the max added wait, and the required L4/L40S graph-pool memory headroom. The raw data is measurable
  via the metric-schema parity required by 0.10 (`_continuous_finalize_timing` + batch/finalize telemetry); the only gap
  is the pass/fail line — register it up front.

**Two orthogonal lenses (don't confuse them):** *Track* = baseline-dependence (can it run before the Python plan
lands?); *Wave* = cost/sequencing (should it be funded before the cheap killers clear?). A spike can be Track-A
(baseline-independent) yet Wave-2 (expensive — don't fund early).

**Track A — baseline-independent feasibility (diffs frozen golden fixtures — oracle discipline in §6):** 0.2, 0.6a, 0.7,
  0.8, 0.9, 0.11. Answers **"can we build it faithfully."** Most are Wave 2 (expensive) except 0.7/0.9/0.11 (cheap/paper)
  — so they *can* run now but are *funded* per the wave ordering above.
- **Track B — post-Python residual validation (BLOCKED until the Python plan lands + traces/telemetry captured):**
  0.1 (overlap/MPS ablation + native launch-overlap microbench), 0.5 (trace-sim). These answer **"is it worth building"** and
  must use the post-plan baseline; pre-plan numbers are non-falsifiable for this purpose. `reviews/decision.md` must
  label every measurement as Track-A feasibility or Track-B post-plan residual.

---

## 2. The current system this must replace (corrected anchors)

Stock **NeMo `nvidia/nemotron-speech-streaming-en-0.6b`**, unmodified (no local NeMo patches): cache-aware streaming
**Conformer encoder** (relative-position attention with `rel_shift` — `multi_head_attention.py:259-270`; `rc0` crashes
upstream, memory `rc0-unsupported-nemo-relshift`) + **RNNT/transducer decoder**, **fp32, 128 mel bins**, ~0.6B params.

- Model load: `server.py:1399-1420`; per-lane replica load `:3109-3137`; dep `nemo_toolkit[asr]>=2.6.0`
  (`pyproject.toml:34-38`).
- **RNNT decode** — **deployed = `greedy_batch` LABEL-looping** (`server.py:726-728,1463-1474`;
  `rnnt_decoding.py:350-365`; computer select `rnnt_greedy_decoding.py:635-647`), `max_symbols=10`,
  `use_cuda_graph_decoder=False` for Blackwell. State = `LabelLoopingStateItem` (predictor state/output, label, decoded
  length, optional LM state — `label_looping_base.py:38-59`; **`time_jump` always `None` for RNNT**, TDT-only
  `rnnt_label_looping.py:494-509`); forced blank advance after `max_symbols`
  (`rnnt_label_looping.py:466-484`); timestamp/length carry across chunks (`:486-506`); batched split/merge for
  per-session carry (`:569-620`); partial-hyp in-place merge (`rnnt_greedy_decoding.py:783-804`); hypotheses packed by
  moving `y_sequence`/decoder-state to CPU (`:52-86`); `Hypothesis.merge_()` mutates score/sequence/state/timestamps/
  alignments/confidence + invalidates text (`rnnt_utils.py:146-173`). Reached via `conformer_stream_step` →
  `decoding.rnnt_decoder_predictions_tensor(..., partial_hypotheses=...)` (`mixins.py:646-660,705-710`). **The repo
  manages decoder *state* (alias-sensitive: `batch_primitives.py:109-126`; recursive clone helpers
  `server.py:147-207,249-263`; fork deep-clone `:6370-6425`) — the decode itself is NeMo's.** The non-batched plain
  `greedy` frame-looping branch (`:1486-1492`) rejects `partial_hypotheses` and is NOT the streaming model.
- Preprocessor: constant-plan STFT for cuFFT determinism `server.py:1509-1593`, `dither=0.0` (`:1510`,`:3139`); 128 mel,
  hop 160, win 400, 16 kHz. Memory `cufft-stft-plan-size-nondeterminism`.
- Streaming cfg: `shift_frames=16`, `pre_encode_cache_size=9`, `drop_extra_pre_encoded=2` (mutated/restored per call:
  `streaming.py:41-55,70-75`), att_context `[70,1]`. Encoder forward changes lengths/masks on `drop_extra>0`
  + builds masks from cache lengths/offsets (`conformer_encoder.py:629-667`), loops layers (`:677-695`), post-processes
  by `keep_all_outputs` (`:523-546`).
- Scheduler/lanes: `_scheduler_loop` `:4456-4509`; lane resources `:3104-3414`; **exclusive gate `:3213-3233`**;
  inference_lock `:593-594`; per-session `state_lock` `:486-490`; sorted lock acquisition `:5086-5089`/`:5190-5192`;
  generation-gated emits `:5201-5203`/`:5220-5225`/`:5227-5244`; in-flight bookkeeping `:5130-5134`/`:5272-5274`.
- Conformer step + graph routing: `_conformer_stream_step` `:2929-3078`; called as `conformer_stream_step`
  `:8317-8328`/`:8797-8809`.
- Finalize/fork: `_build_continuous_finalize_fork` `:6370-6425`; flush/emit `:6738-6895`; final-delta from cumulative
  text `:6816-6825`; stale-generation suppression `:6827-6842`; committed/emitted update `:6844-6866`; empty/dup
  suppression `:6881-6895`; fork-unchanged assert `:6792-6797`; finalize generation checks `:7621-7637`/`:7660-7679`.
- CUDA graphs: steady bucket gate `:2052-2084`; finalize per-T key gen `:1693-1705`, capture `:1707-1724`;
  `cudagraph_encoder.py` (`BucketedCudaGraphEncoder`; **exact-shape, static-buffer, never pads T** — `:6-8`, `:46-54`,
  `:193-204`, `:272-281`, `:293-296`, `:578-599`). Finalize CUDA-graph WIN already shipped in Python (246/279 —
  memory `finalize-graph-probed-rejected`). **The Python baseline is a moving target — record the exact commit each
  gate beats.**
- Transport: aiohttp WebSocket `:4148-4248`; binary audio in, JSON out; VAD **client-side**, scheduler receives
  `vad_start`/`vad_stop` at `:4359-4362`.
- Deploy: multi-proc + CUDA MPS + HAProxy (leastconn, maxconn≈12); `deploy/launch_multiproc.sh:6-9,19-24`
  (`auto_pick_K` L40S=3 because K=4 OOMs); target = AWS SageMaker Ada (L4/L40S) — memory `deployment-target-sagemaker`.
  **NB: the launcher's `~48/box`/`~64/box` comments are LEGACY keep-up/over-budget figures, NOT in-budget density;
  this plan's worth-it gate uses the Python plan's IN-BUDGET ~28/box (K=4). Don't read the launcher's 48/64 as
  "Python already hit the native target."**

**Target platforms:** RTX 5090 (local dev/CI), DGX Spark GB10 (aarch64, 273 GB/s — bandwidth-poor; user sets up later),
L4 (AWS g6, 300 GB/s), L40S (AWS g6e, 864 GB/s).

---

## 3. The central architectural decision (resolved by Phase-0 evidence, not assumed)

### Axis A — serving-core language: Rust vs C++
| | Rust | C++ |
|---|---|---|
| Networking/scheduler | tokio: memory-safe async, work-stealing | asio / hand-rolled; more footguns |
| libtorch interop | `tch-rs` FFI; lags versions; friction on CUDA-graph capture/stream control | **native** (`torch::jit`, `at::cuda::CUDAGraph`, `c10::cuda::CUDAStream`) |
| Raw CUDA | `cudarc`/`cust` or unsafe FFI | native |
| Concurrency safety (the whole point) | borrow checker prevents data races | manual; TSan/ASan needed |

Candidate shapes (pick in 0.4): (1) all-C++; (2) **Rust front + C++ model-worker `cdylib` over a thin off-hot-path FFI
(submit-batch/poll-completion)**; (3) all-Rust (`tch-rs`+`cudarc`).

**The Rust-vs-C++ decision is made on a concrete tch-rs evaluation, not preference (user direction 2026-05-24).
All-Rust (3) is the DEFAULT if-and-only-if `tch-rs`(+`cudarc`), *at a libtorch version that clears the version-selection
constraints*, exposes the hot-path surface — decisively, CUDA-graph capture/replay against libtorch-*allocated* tensors,
per-lane stream/event control, the capture-mode allocator, and the ATen ops the decode needs.** If any of those is
missing or tch-rs lags the required libtorch version, you'd write `unsafe extern "C"` shims to the missing C++ symbols
inside the Rust crate (hand-binding C++ anyway) → prefer shape (2) the thin C++ worker, or (1) all-C++. The decisive box
is the allocator-coupled graph capture (raw `cudarc` graphs over separately-allocated memory don't count). The full
checklist + libtorch version constraints (Blackwell/sm_120, NeMo range, same-version-for-export+fixtures+runtime, C++
ABI flag) live in `spikes/decision-template.md`; the probe is part of Spike 0.2; the call is recorded at 0.4.

### Axis B — model-execution backend
| Option | Fidelity | Effort | Notes |
|---|---|---|---|
| **B1a: libtorch C++ + export of the *encoder* `cache_aware_stream_step`** | high (same ATen) | medium | low payoff (encoder near-floor); still must test ALL geometries (0.2); byte-exact is a HARD gate (no "documented T1" escape) |
| **B1b: native re-implementation of the RNNT `greedy_batch` LABEL-looping decode** | must be *proven* vs NeMo (exact `Hypothesis`/state, not just text) | **HIGH** | the GIL-bound payoff AND the un-exportable part. **The real go/no-go (Spike 0.6a — EAGER, `use_cuda_graph_decoder=False`).** Must match `LabelLoopingStateItem`, partial-hyp merge, max_symbols saturation, fork. (The Blackwell CUDA-graph decoder = 0.6b research, NOT the gate.) |
| B2: TensorRT (encoder) + custom decode | medium (kernel/precision drift) | high | the *fusion* path; prerequisite for 6–10 ms finalize |
| B3: hand CUDA kernels | lowest | very high | deferred; only for fp8/fusion density later |
| ~~B4: free-threaded CPython 3.13t~~ | — | — | **REJECTED by the user (2026-05-24): not interested in a free-threaded-Python approach. Spike 0.3 removed.** Outcome space = B1 or STOP. (Recorded non-option.) |
| ~~B5: in-tree native-decode extension~~ | — | — | **REJECTED by the user (2026-05-24): no middle path.** (Recorded non-option.) |

**"From-scratch" = replacing the Python serving/dispatch/decode-orchestration layer**, driving the same NeMo
weights/kernels for the encoder (B1a) while **re-implementing the RNNT `greedy_batch` label-looping decode natively
(B1b)** — the decode does NOT come free from "export the model." B2/B3 (fusion) are a *later, separately-proven*
milestone, not part of the default v1.

**Why libtorch, not raw CUDA (cuBLAS/cuDNN/CUTLASS) or TensorRT, for the encoder (B1a):**
1. **Same kernels underneath.** libtorch dispatches the *exact* cuBLAS/cuDNN/cuFFT that PyTorch does — going lower-level
   doesn't change which kernels run, only who calls them; and "who calls them" (the GIL/scheduler) is the thing we're
   fixing, not the math.
2. **The roofline says the math isn't the bottleneck.** Encoder is memory-bandwidth-bound, ~near floor, fp16 was
   *slower*, p50 immovable → hand kernels buy ~nothing on the prize (tail+density). Most effort on the least-headroom axis.
3. **Byte-exactness (T2a) is free with libtorch, forfeited going lower.** Same ATen→cuBLAS/cuDNN with matched versions ⇒
   byte-exact by construction; raw kernels/CUTLASS/TRT pick different algos/reduction-orders/fusions → lose byte-exact AND
   inherit a full numerical re-derivation of rel-shift attn + conv-cache + RNNT joint (that's B3, "very high / rejected").
4. **Launch overhead is already solved by CUDA graphs** (`at::cuda::CUDAGraph`, native in libtorch) — a captured graph
   collapses the whole forward to one replay; raw kernels can't beat that. The usual reason to go lower-level is moot.
5. **The custom burden is the decode (control-flow/state), not GEMMs** — libtorch gives the tensor/joint primitives for
   it without hand-writing matrix math.
6. **TensorRT doesn't address the prize:** it's a kernel-execution engine — you'd still need the whole Rust/C++ serving
   core around it, and it breaks byte-exactness + struggles with the cache-aware-streaming + rel-shift + per-T finalize
   contract. Its only justified use is **fusion** (the 6–10 ms finalize), which is exactly why fusion is carved out as the
   deferred B2/B3 milestone (spike 3.3), not the default.

**Scope decision (round 2):** v1 native runtime targets the **English 0.6b checkpoint only**. The prompted/multilingual
variant (per-language prompt state `set_inference_prompt` `server.py:1288-1292`, `target_lang` batch grouping
`:4796-4803`, language-tag stripping `:1379-1397`, larger/multiple att-context `:429`) is a **later, separately-scoped
phase**. Where v1 is EN-only, the "drop-in" claim is dropped for prompted models.

---

## 4. Correctness bar (streaming-aware; the load-bearing constraint)

The project ran on byte-exact (memory `silence0-warm200-shippable`, `finalize-graph-probed-rejected`); the native path
can't guarantee byte-exact in general (cross-arch, reductions; memory `cufft-stft-plan-size-nondeterminism`). Tiered:

- **T0 — within-engine determinism (under concurrency):** self-consistent run-to-run, including with multiple lanes
  active. Requires an explicit numerics contract: **TF32 off** (the model is fp32; the Python server disables TF32 only
  under batching `server.py:655-660`), **deterministic algorithms**, **pinned cuBLAS/cuDNN workspace + per-lane
  handles**, constant FFT plan. Required before T2 is even attempted — sharing one weight copy across lane threads does
  NOT imply identical outputs unless these are pinned.
- **T1 — streaming-aware behavioral equivalence (the SHIP gate)** vs the current Python server:
  - **Exact per-session ordered event stream** (interim + final), **exact final-delta text** (`:6816-6825`), **exact
    duplicate/stale-generation suppression decisions** (`:6827-6842`, `:6881-6895`, `:5227-5244`) — aggregate WER cannot
    see interim flicker / delta-boundary / dup-suppression regressions.
  - **Per-utterance WER delta bound** (not just corpus mean) + **corpus full-1000 WER within a NAMED numeric CI width**
    (define it in 0.4; run `wer` *without* `--test` — memory `silence0-warm200-shippable`).
  - **State-machine trace suite** (continuous mode is more than "resume-after-stop"): exact event order + emitted JSON
    for audio-after-`vad_stop` (held in `continuous_post_stop_audio` `:5590-5598`), `vad_start`-before-debounce (cancels
    + flushes held audio `:5731-5744`), `reset`-before-debounce (sets `continuous_reset_seen`, waits `:5823-5829`),
    `end`/`close` while pending (`:5676-5684,5802-5815`), reset during in-flight finalize, stale finalize after newer
    audio (generation checks `:7301-7310,7660-7679,7795-7813,6827-6842`; cold-reset invalidation `:5907-5928`). Compare
    **generation transitions + post-stop audio byte counts**, not just final transcript. Property/fuzz random
    interleavings.
  - TTFB/finalize-timing distributions non-inferior.
- **T2 — split into two gates (round 3); both vs FROZEN golden fixtures from a pinned Python+torch+CUDA build, same
  arch (5090):**
  - **T2a — same-shape byte-exact (the B1a gate, no escape):** for the **B1a encoder per-B (same-B path)**, byte-exact
    real-frame encoder + cache outputs across the live geometry (first chunk `drop_extra=0`, steady, finalize
    `keep_all_outputs=True`). The wrapper calls the same `cache_aware_stream_step` with restored `drop_extra`
    (`cudagraph_encoder.py:231-241,56-64`), so non-byte-exactness means export changed behavior or the fixture isn't
    controlling inputs. **Whether byte-exact is attainable from libtorch at all is an OUTPUT of 0.2** (identical backend
    versions/flags required); if not, no-go B1a or reclassify it B2-risk. No "documented T1" pass for T2a.
  - **T2b — cross-B / session-invariance (a NAMED-tolerance gate, not byte-exact):** decoding a session alone vs batched
    with other rows changes GEMM/reduction shapes (mels `[B,F,T]` `batch_primitives.py:59-75`; caches stacked in the
    layer dim `:78-87`), so cross-B byte-exactness is NOT assumed even inside PyTorch. Require **exact token/event
    equivalence + `allclose` within a named tolerance**, not bit-equality. Cross-**arch** (L40S/L4/Spark) byte-exact is
    out — those rely on T1 + T2b.

Harnesses to reuse/extend: byte-exact finalize canary (`tests/test_cudagraph_finalize.py`,
`proj-2026-05-22-1353/finalize_graph_canary.sh`), full-1000 WER (`stt-benchmark`), keep-up/overload sweeps
(`ec2-bench/bench_prod_sweep.sh`, `bench_prod_multiproc.sh`, `bench_lanes_ab.sh`).

---

## 5. Rules

### Correctness (hard gate — every phase that runs the model)
- T1 streaming-aware equivalence vs the *current* Python baseline at **concurrency**, **per-session**; any drift
  diagnosed before proceeding. Record the baseline commit beaten.
- **Native state-ownership is a DESIGN, not a checker:** define a per-session **actor / immutable-snapshot →
  result-commit** protocol with explicit **generation/stale-output rules** and hypothesis-aliasing rules
  (NeMo mutates hyps in place — `batch_primitives.py:109-126`) BEFORE Phase 1. Borrow-checker / TSan / ASan are
  *secondary* validation, not the design.
- Preprocessor determinism preserved (constant cuFFT plan); finalize padding/masking must yield byte-identical encoder
  output for the REAL frames.

### Safety / sequencing
- **§0 worth-it gate precedes Phase 1.** Phase-0 spikes have explicit go/no-go criteria.
- The native runtime is a **parallel artifact**; the Python server stays production until the native one passes T1 +
  cloud sweeps. Each phase leaves a runnable, tested artifact.
- No change to client wire-protocol semantics (drop-in for existing clients/bots).

### Measurement / deploy
- **SUCCESS = finalize latency ↓, steady throughput ↑ (fill B), reliable streams/box ↑, overload cliff gone. NOT p50.**
- **Every platform re-measures** (knee is CPU/launch/scheduler-bound → platform-specific; memory
  `modal-asr-knee-launch-bound`, `deployment-target-sagemaker`). Order: 5090 (dev) → L4/L40S → DGX Spark. Local T1 gate
  BEFORE any cloud spend.
- Cloud: us-west-2; `aws sso login --sso-session khk` (expires ~hourly — check before each run); ALWAYS terminate EC2
  (traps + GPU-leak check).

---

## 6. Phases & steps

### Phase 0 — Spikes + decision

**Track A — early feasibility (run now, parallel with Python plan):**
**Track-A oracle discipline (round 3):** Track A is only baseline-independent if its byte-exact gates diff **FROZEN
golden fixtures captured once from a pinned Python+torch+CUDA build (commit X)**, NOT the live (moving) Python server.
If Track A runs before the Python plan lands, **rerun 0.2/0.8 against the final post-plan Python commit before 0.4**.
Padded-T equality must **align with the Python plan's relaxed gate** (`proj-2026-05-24-0859/PLAN.md:62-67`): tokens +
event stream byte-exact; real-frame *tensors* `allclose` unless the source path is exact-shape (do NOT demand
byte-identical padded-T tensors where the Python plan itself relaxed to `allclose`).

- [ ] **0.2 Encoder export fidelity spike (B1a).** Input oracle = **frozen Python mel fixtures** (decoupled from 0.8;
  the chained native-preproc→encoder pipeline is validated in Phase 1). Export `encoder.cache_aware_stream_step`
  (TorchScript/`torch.export`), load in libtorch C++, test **ALL live geometry**: first chunk `drop_extra=0`, steady
  `drop_extra=self.drop_extra`, finalize `keep_all_outputs=True`, B>1, exact cache round-trip shapes; handle
  `drop_extra_pre_encoded` mutate/restore (`streaming.py:41-55`) + `rel_shift` (`multi_head_attention.py:259-270`).
  **Go:** byte-exact vs the frozen same-shape fixtures (see §4 T2; whether byte-exact is *attainable* from libtorch is
  itself an output). **No-go:** export can't capture the control flow / not byte-exact → reclassify B1a as non-identical
  (B2-risk). **(No B4 fallback exists anymore — if B1a isn't byte-exact, the choice is B2-risk-sign-off or STOP.)**
  **ALSO IN 0.2 — the tch-rs / Rust-vs-C++ evaluation** (decides Axis A; user direction): first pin the **newest viable
  libtorch** per the version-selection checklist (Blackwell/sm_120 across 5090/L4/L40S/GB10, NeMo-supported torch range,
  same version for export+fixtures+runtime, C++ ABI flag — `spikes/decision-template.md`). Then run the **tch-rs
  coverage probe** against that version: does `tch-rs`(+`cudarc`) bind it (not lagging) and expose CUDA-graph capture
  **against libtorch-allocated tensors** (decisive), per-lane streams/events, the capture-mode allocator, and the
  decode's ATen ops? **All ✓ → all-Rust; any ✗ → Rust-front+C++-worker or all-C++.** Record the failing box at 0.4.
- [ ] **0.6a Native DEPLOYED (eager) `greedy_batch` LABEL-looping decode equivalence (THE go/no-go).** Scope = the
  **deployed config only**: `loop_labels=True`, **`use_cuda_graph_decoder=False`** (`server.py:1463-1474`),
  `eou_probe_enabled=False` ⇒ **NO alignments/confidence**, **no n-gram LM** (assert non-null `ngram_lm_model` is
  rejected — `rnnt_greedy_decoding.py:647-653`), **assert RNNT `time_jump is None`** (RNNT always returns `None`;
  time-jump is TDT scope creep — `rnnt_label_looping.py:494-509`). Re-implement and prove **exact `Hypothesis`/state
  equivalence** (not just text/WER) on fixed encoder-output fixtures: previous hypotheses + `LabelLoopingStateItem`
  (`label_looping_base.py:38-59`), forked finalization state, **`max_symbols=10` saturation** (forced blank advance
  `rnnt_label_looping.py:466-484`), all-blank chunks, partial-hyp in-place merge (`rnnt_greedy_decoding.py:783-804`).
  Compare `y_sequence`, score, timestamp/decoded-length, label/predictor state, decoder-state shapes/devices/dtypes, and
  the **parent-fork-unchanged** assertion (`server.py:6792-6797`). **Excludes** the Blackwell CUDA-graph-decoder (that's
  0.6b/3.x research, NOT the funding gate) and **mixed-continuation batches** (the server forbids mixed None/not-None
  rows: batch key `server.py:4789-4812`, `stack_hypotheses`/`stack_pred_out` assert uniform `batch_primitives.py:100-139`
  — allowing them is a NEW batching-semantics change with its own gate, not part of deployed equivalence). **This is the
  real funding gate.**
- [ ] **0.8 Native preprocessor byte-exactness spike.** Re-create the constant-plan STFT (`server.py:1509-1593`),
  `dither=0`, the **raw_audio_ring** (STFT boundary) + **mel_frame_ring** (9-frame pre-encode) buffering, ingest cadence
  (`preprocess_new_audio_samples=2720`), and the **finalize multi-pass preproc loop** (`:6927-6942`). Gate to **byte-
  exact mel vs the frozen Python fixtures on 5090** across steady + finalize (memory `cufft-stft-plan-size-
  nondeterminism`). 0.8 validates the native preprocessor *independently*; it is **not** a prerequisite for 0.2 (which
  uses frozen Python mels). The two compose in Phase 1.
- [ ] **0.11 CUDA-graph ownership-model spike.** Decide the native graph strategy: **per-lane graph+static-buffer pools
  (memory ×lanes) vs one shared set with a serialized replay mutex (measured lost overlap) vs none**. Specify capture
  thread/stream affinity (capture is not free-threaded; NeMo uses a dedicated capture stream + `capture_error_mode=
  "thread_local"` — `rnnt_label_looping.py:839-890`; the server captures on single-worker lane executors
  `server.py:1882-1942` and logs cross-thread replay `:2960-3004`), a graph-safe allocator + "no allocation during
  capture" rule, output clone/lifetime rules (static output pool — `cudagraph_encoder.py:15-17,272-281`), and **measured
  resident graph-pool memory at the target K×lanes** (today L40S is K=3 because K=4 OOMs — `deploy/launch_multiproc.sh:
  6-9`). **The 40–48/box density number is unsupported until this resolves.**
- [ ] **0.9 Model mutability audit.** Enumerate every mutable model surface touched per request and prove the native
  design passes them as **pure parameters** (no shared mutable state on the hot path): `streaming_cfg.drop_extra`
  (`cudagraph_encoder.py:56-64`, `batch_primitives.py:142-149`), prompt/language `set_inference_prompt`
  (`server.py:1288-1292,8306-8307,7394-7395`), decoder/joint eval↔train toggles (`rnnt_greedy_decoding.py:390-415,
  744-766`), graph static buffers, RNG/training flags. **This is the prerequisite for the shared-read-only-weights
  density lever** (today's per-lane replicas `server.py:3109-3137` dodge these races at a memory cost).
- [ ] **0.7 aarch64 toolchain pre-check (gates 4.2).** Build minimal libtorch+CUDA on aarch64, load a trivial exported
  module, run one CUDA-graph capture/replay; record CUDA/driver/libtorch/compiler/ABI versions + a cross-arch
  determinism note. **Must confirm the *pinned* libtorch+CUDA (version-selection checklist, `decision-template.md`) has a
  working aarch64 build that emits the GB10 arch** — i.e. the same pinned version spans 5090/L4/L40S *and* GB10, or the
  version pin is wrong. No DGX Spark work until this artifact exists.

**Track B — post-Python residual validation (BLOCKED on the Python plan landing + traces captured):**
- [ ] **0.1 Overlap/MPS ablation matrix (thesis test + "no MPS tax" resolution).** NOT a single comparison and NOT a
  generic graph-replay loop. Hold model/decode constant; on the **real post-Python finalize+steady path** separately
  toggle: batch-finalize on/off (deployed finalize uses `_scheduler_pinned_model_lane_path` `:6755-6773`, only the
  fallback takes `inference_lock` `:6779-6785`), exclusive gate on/off (`:3213-3233`), `inference_lock` on/off,
  lane-end `stream.synchronize()` (`:3175-3178`) vs CUDA-event dependency, **same-lane vs cross-lane** finalize/steady
  (lane affinity `:3295-3308`, pinned-lane wait `:6711-6720`), single-process/single-context vs MPS vs multi-process,
  and CPU thread affinity. **Report per-lane CUDA-event timelines + queue/lane wait, not just the throughput knee.**
  **Go:** isolates which serializer dominates AND shows a single native process overlaps finalize+steady ≥ a named
  factor. **No-go:** if only MPS/multi-proc overlaps, the single-process "no MPS tax"/40–48-box story is false → revise.
  **PLUS (0.1b) the native launch-overlap microbench** (replaces the deleted B4 probe as the conjunct-2 proof): N OS
  threads (no GIL — C++/Rust), each driving a CUDA stream that replays the captured encoder graph + a mock decode of the
  right shape; measure aggregate launches/sec, GPU util, and finalize/steady overlap vs N, on each target GPU
  (single-context vs MPS). Reuses the captured graph (no decode-equivalence needed) → far cheaper than the B1 build, and
  it's the cheapest remaining way to PROVE "GIL-free native dispatch lifts the knee" before committing to B1.
  **0.3 (free-threaded py3.13t) is REMOVED — the user rejected B4.**
- [ ] **0.5 Trace-driven batching simulator (validates/kills the 3–5× claim).** *Prereq:* instrument the current server
  to emit per-tick readiness traces (batch key `batch_primitives.py:24-31` incl. fresh/established decoder-state flag
  `server.py:4789-4812`, ready predicate `:34-56`, finalize state, lane affinity, `BATCH_MAX_WAIT_MS/MAX_SIZE` `:665-670`,
  dispatch sort `:4864-4918`). Replay **post-Python** traces into a simulator; report achievable **B distribution**
  without added latency **AND a graph-bucket capacity model**: per-lane graph memory per B, expected exact-B replay
  hit-rate from the B histogram, eager-fallback %, and a cap policy for rare B (the manager never pads B —
  `cudagraph_encoder.py:6-8,578-599`; out-of-range B → eager `server.py:2080-2084`). **Go:** median B ≫ 1 AND the graph
  pool fits L4/L40S memory at target K×lanes. If B stays ~1 (86% today) **drop the 3–5× target**; if the graph pool
  doesn't fit, **drop the steady-graph density claim**.

- [ ] **0.4 Decision memo (`reviews/decision.md`) — HARD GATE.** Pin libtorch/CUDA/ABI versions + export artifact
  format; define the named WER-CI width; v1 scope (EN-only). **Emit the filled-in DECISION TREE** (label each input
  Track-A feasibility vs Track-B post-plan residual):

  | Observed outcome | Decision |
  |---|---|
  | 0.0 residual value < threshold | **STOP** (archive Track-A learnings) |
  | 0.1: only MPS/multi-proc overlaps finalize+steady (not single-process), OR the native microbench shows GIL-free dispatch does NOT lift the knee | **STOP** (the only escape, *native-under-MPS*, is TAIL-ONLY = no density gain, and `proj-2026-05-24-0859` Step 5 already chases that tail in Python → re-run 0.0, value below threshold) |
  | 0.1 positive (single-process overlaps + microbench lifts the knee) AND 0.6a + 0.2 + 0.8 + 0.11 pass | **proceed B1** (full native) |
  | 0.6a fails byte/state equivalence | **STOP**, or accept a *named* T1-only (non-byte/state-exact) native-decode risk — explicit sign-off (no B4 fallback) |
  | 0.2 fails T2a / libtorch byte-exact unattainable | no B1a → **STOP** or explicit B2/T1-only-risk sign-off (no B4 fallback) |
  | 0.8 fails native-preprocessor byte-exact | no native preproc → **STOP**, or keep Python preproc as a named non-v1 topology |
  | 0.9 fails (can't make per-call config pure params) | drop **shared-weight density** → per-lane replicas; re-run 0.0 (likely **STOP**) |
  | 0.5: B stays ~1 | drop the **3–5× throughput** claim; re-run 0.0 (density-only justification) |
  | 0.5/0.11: B>1 but poor exact-B graph hit-rate / high eager fallback / no memory headroom | drop **steady-graph throughput+density** claims → B1-without-steady-graphs or revise topology; re-run 0.0 |
  | decode-GIL-bound but full-B1 residual not worth the cost (and B4 didn't win) | **STOP** (user rejected the in-tree-extension middle path 2026-05-24) |
  | 3.3 fusion unproven | the **6–10 ms finalize** headline is out of v1 scope (B1 target = parity); does NOT affect the core go/no-go |

### Phase 0.5 — Runtime contract (before Phase 1)
- [ ] **0.10 Runtime contract + acceptance tests.** A written contract + tests for: the WS wire protocol + exact
  interim/final JSON payload fields (`server.py:5234-5244,6849-6857`), connection validation + error/ready payloads
  (`:4148-4248`, language validation `:1350-1377`), close/reconnect behavior, health/readiness (`:8842-8847`),
  **metrics/log schema parity** (the benchmark-critical timing fields the sweeps consume — `_continuous_finalize_timing`
  `~:6594-6607`, batch/finalize telemetry `:5388-5424,7254-7281`; **needed from Phase 1 to even measure**),
  admission/backpressure signals (bounded queues `:4326-4341`), graceful drain/restart, model-load-failure + OOM
  fallback, and rollback triggers (the launcher's open TODOs: LB drain / alerting / MPS-context restart
  `deploy/launch_multiproc.sh:70-79`).

### Phase 1 — Minimal persistent runtime: single stream, steady + finalize, T1 on 5090
- [ ] **1.0 State-ownership design** (actor/snapshot + generation/aliasing rules) — written + reviewed before code.
- [ ] **1.1 Weight/graph asset pipeline.** Reproducible export of encoder modules + preprocessor params + streaming cfg;
  **one shared read-only weight copy**; pin source checkpoint hash; document constant-FFT-plan handling.
- [ ] **1.2 Single-stream steady path.** Native preprocessor (from 0.8) → encoder step → **native RNNT label-looping
  decode (from 0.6a), on-GPU, no per-frame `.item()`** → emit; carry encoder cache + decoder hyp state across chunks;
  emit the benchmark-critical metrics from 0.10. **Gate:** T1 single-stream on 5090 + T0; T2 encoder byte-exact.
- [ ] **1.3 Finalize path.** VAD-stop/debounce → padded finalize → `keep_all_outputs` encode → final decode → final
  delta, with fork/clone isolation (parent untouched; port the FORK_ASSERT `:6792-6797`). **Gate:** T1 finalize canary
  + resume-after-stop + stale-generation suppression equivalence.

### Phase 2 — Continuous batching + multi-thread scheduler (the core win)
- [ ] **2.1 Per-lane dispatcher threads + fine-grained ownership** (no single asyncio loop; no coarse exclusive gate).
- [ ] **2.2 Real continuous batching (fill B) — only if Spike 0.5 said it's reachable.** Per-session decode-state
  scatter/gather (port `batch_primitives` semantics incl. hyp-aliasing guards). **Gate:** T1 at concurrency; measured B
  distribution matches the 0.5 projection.
- [ ] **2.3 Admission + lane-priority + backpressure** (native; mirrors `proj-2026-05-24-0859` Steps 1/4). **Gate:**
  overload sweep — cliff bounded.

### Phase 3 — Density + (separately) fusion
- [ ] **3.1 Native CUDA graphs (steady B=1..K).** Reuse the existing per-T finalize graph approach; **do NOT** claim the
  padded-T_max bucket here — that proof belongs in the Python plan where the manager exists (require byte-exact
  real-frame outputs across T=42..60 first). Shared-weights density measured.
- [ ] **3.2 Graphed/fixed-trip decode + host-sync compression.** **Gate:** T1; finalize latency measured — target
  **parity-or-better** for B1. **6–10 ms is a SEPARATE later milestone gated on the fusion path (B2/B3)**, NOT a B1 gate.
- [ ] **3.3 (optional, gates the 6–10 ms headline) Fusion-viability spike.** Before promising 22→6–10 ms anywhere,
  prototype a fused/TensorRT version of **one** conformer layer at M≈7 with the rel-shift attention + streaming-cache
  contract intact and *estimate* achievable finalize ms. The roofline says fp16 was 0.79× until fused and "precision
  only helps once fused" (`roofline-COMBINED.md:20-23`) — this is research-grade at M≈7. If unfunded, the honest B1
  finalize target is **parity-to-modest** and the 6–10 ms headline is explicitly out of v1 scope.

### Phase 4 — Multi-platform + density
- [ ] **4.1 L4 + L40S (AWS).** Keep-up sweep, operating-point p95/p99, overload cliff, density (streams/box; whether
  shared weights + overlap raise K). Head-to-head vs the *current* Python baseline (record its commit). No p50
  regression.
- [ ] **4.2 DGX Spark GB10 (aarch64) — gated on 0.7.** Same sweeps. Expect bandwidth-poor (~9 ms floor): validate,
  don't assume it beats L40S.
- [ ] **4.3 5090 density datapoint** (dev box).
- [ ] **4.4 Shadow/mirror-traffic validation (SUPPLEMENTS, does not gate, the Phase-1 fixture-based T1).** Tee live or
  replayed audio to the native runtime and **diff its event stream vs Python without serving native output**; catches
  interim-cadence/delta/generation drift that offline corpus replay misses. Requires a tee in the client/bot or an LB
  mirror. (Phase 1–3 T1 already runs against offline golden fixtures; 4.4 is the stronger live check at scale.)

### Phase 5 — Hardening + rollout readiness
- [ ] **5.1 Protocol/parity hardening** (drop-in WS; metrics parity; graceful drain/restart) + **launcher hardening the
  current deploy still lacks: LB drain on restart, alerting, MPS-context restart after crash**
  (`deploy/launch_multiproc.sh:70-79`). Define **readiness = model loaded + graph/lane pools captured + T1 canary
  passing** (today's health endpoint only reports loaded-vs-loading `server.py:8842-8847` — insufficient for a runtime
  with graph capture + lane pools).
- [ ] **5.2 Deploy integration + cutover.** **Topology is an OUTPUT of 0.1** — single-process-multi-lane (and whether
  MPS stays or is removed) vs still-multi-proc; reconcile with HAProxy (leastconn, per-process maxconn) + MPS daemon
  (`deploy/launch_multiproc.sh:57-68`). Cutover: run Python + native **side-by-side behind the same LB**, canary by
  backend replica → ramp; **rollback = remove native backends, return all traffic to Python.** Update `deploy/` + memory.

---

## 7. Top risks (live)
1. **Thesis partly wrong** — single CUDA context may launch-serialize regardless of GIL → MPS/green-contexts still
   needed; "no MPS tax" + single-process 40–48/box may be false. → **Spike 0.1 (real-workload, MPS-compared) is the
   kill-switch.**
2. **The RNNT `greedy_batch` LABEL-looping decode is the un-exportable, byte-exact-risky payoff** (B1b — exact
   `Hypothesis`/`LabelLoopingStateItem` equivalence incl. partial-hyp merge + max_symbols saturation, **EAGER /
   `use_cuda_graph_decoder=False`**) → **Spike 0.6a is the funding gate**, not the encoder export. (The Blackwell
   CUDA-graph decoder is *separate* 0.6b/Phase-3 research — it is explicitly NOT part of the equivalence gate, since the
   deployed decode is eager.)
3. **Preprocessor/STFT cuFFT-plan nondeterminism** reintroduced natively, upstream of the encoder → Spike 0.8 byte-exact
   gate before 0.2 means anything.
4. **"Shared read-only weights" isn't read-only** — drop_extra/prompt/eval-train are per-call mutable shared state →
   Spike 0.9 mutability audit; pure-parameter design required before the density lever is bankable.
5. **T1 too weak** → mitigated by streaming-aware T1 (exact event/delta/generation behavior, state-machine trace suite)
   + T2 encoder byte-exact (no escape hatch).
6. **Fill-B may not be reachable** (86% B=1 today) → Spike 0.5 trace-sim (post-Python) before budgeting 3–5×.
7. **6–10 ms finalize requires research-grade fusion** at M≈7 (fp16 was 0.79× until fused) → separate milestone 3.3;
   B1 target = parity.
8. **Concurrency correctness of mutable per-session state** → actor/snapshot design + generation rules (1.0) first;
   checkers secondary.
9. **Two stacks; Python baseline moves** → each gate records the baseline commit beaten.
10. **Effort vs payoff / duplication of the Python plan** (~½–¾ eng-year, §9) → §0 worth-it gate; pause if Python
    already closes the gap.
11. **Multilingual/prompted serving** is a separate surface → v1 = EN-only; stated as explicit scope reduction.
12. **aarch64 toolchain** unknowns → Spike 0.7 pre-check gates 4.2.
13. **The baseline is a MOVING target — and the user DECIDED not to freeze it (2026-05-24).** The Python plan keeps
    shrinking the native residual (finalize graph landed 246/279; K=4 + finalize-priority coming). **Accepted risk:** a
    fast-improving Python baseline can shrink the residual (even asymptote) *mid-build* → the native build must
    continuously out-perform the *latest* Python baseline, and 0.0 must be re-checked against it as Python advances. Each
    gate records the baseline commit it beat (so the shrink is visible).
14. **Is this the right problem? — user DECIDED YES (2026-05-24): density/tail is a strategic priority.** So this is not
    a lean-STOP project. The 0.0-pre ceiling arithmetic remains an expectations check (density is a COGS lever; ~20
    best-case extra streams/box), but the directional call to pursue density *if the residual is real* is made.

## 9. Effort sizing (two budgets; ±50–100% — so §0 is an evaluable comparison)

Engineer-weeks, 1 senior eng familiar with the stack. Round 3 widened these: the v2 numbers were optimistic ~2×. Split
into a **research prototype** (may FAIL the 0.6a/0.1 gates — sunk if it does) and the **production replacement**.

**Budget A — research prototype (proves/kills the thesis):**
| Block | est (eng-wk) | risk |
|---|---:|---|
| Track A: 0.2 encoder export (frozen-mel oracle) | 1–2 | med |
| Track A: 0.6a native EAGER label-looping equivalence | **4–8** | **high** (funding gate; stateful, fork, pack/unpack to CPU) |
| Track A: 0.8 native preprocessor byte-exact | 2–3 | med (cuFFT determinism) |
| Track A: 0.9 mutability audit + 0.11 graph-ownership spike + 0.7 aarch64 | 2–3 | med (graph capture/alloc rules) |
| Track B: 0.1 ablation + native launch-overlap microbench + 0.5 trace-sim+graph-capacity | 3–4 | med (needs post-Python baseline; microbench needs the captured graph) |
| **Prototype subtotal** | **~12–20** | — |

**Budget B — production replacement (only if Budget A passes):**
| Block | est (eng-wk) | risk |
|---|---:|---|
| 0.10 runtime contract + acceptance tests + golden-fixture harness | 2–3 | low–med |
| Phase 1 (steady + finalize + state machine + ownership) | 5–8 | high |
| Phase 2 (multi-thread sched + batching + admission) | 5–8 | high (concurrency correctness) |
| Phase 3 (CUDA-graph infra + sync compression) | 3–5 | med (graph allocator/pools) |
| Cross-language FFI seam (if Rust+C++ shape) + determinism plumbing | 2–4 | med |
| Phase 4 (L4/L40S + aarch64 + sweeps + multi-GPU CI) | 3–5 | med (Spark unknowns) |
| Phase 5 (hardening: drain/restart/rollback/observability) | 3–4 | med |
| Long-tail byte-exact debugging buffer (history: cuFFT/fork ate weeks) | 3–5 | high |
| **Production subtotal** | **~26–42** | — |
| (3.3 fusion path, separate, if 6–10 ms pursued) | 4–8+ | very high (research) |

**Combined (excl. fusion): ~40–60+ eng-wk ≈ a full engineer-year, plus a second stack to maintain.** Compare against
`proj-2026-05-24-0859` (~6 byte-exact-gated steps, days-to-weeks, no new language/toolchain). **The §0 gate weighs ≈ an
engineer-year against whatever density/tail gap the Python plan leaves.** If that gap is small, this project should not
start.

## 8. Progress
| # | Step | Track | Status | Notes |
|---|------|------|--------|-------|
| 0.0 | Worth-it gate | gate | pending | named residual gap vs Python plan + baseline commit |
| 0.2 | Encoder export fidelity (frozen-mel oracle) | A | pending | B1a; T2a same-shape byte-exact hard gate |
| 0.6a | Native EAGER label-looping decode equivalence | A | pending | **the real go/no-go**; exact Hypothesis/state; no graph/LM/align/mixed-batch |
| 0.8 | Native preprocessor byte-exact | A | pending | independent of 0.2; frozen-fixture oracle |
| 0.9 | Model mutability audit | A | **scaffolded — analysis COMPLETE** | `spikes/0.9-mutability-audit.md`; 8 mutable surfaces enumerated + pure-param rule |
| 0.11 | CUDA-graph ownership-model spike | A | **scaffolded — analysis done; mem TBM on GPU** | `spikes/0.11-graph-ownership.md`; per-lane vs shared-mutex vs none |
| 0.7 | aarch64 toolchain pre-check | A | scaffolded (template) | `spikes/0.7-aarch64/`; run BLOCKED on GB10 box |
| 0.1 | Overlap/MPS ablation + native launch-overlap microbench | B | scaffolded (skeleton) | `spikes/0.1-overlap-ablation/`; 0.1b microbench now carries the conjunct-2 proof (was B4); run BLOCKED on GPU + post-Python baseline |
| ~~0.3~~ | ~~py3.13t probe~~ | — | **REMOVED 2026-05-24** | B4 rejected; conjunct-2 proof moved to 0.1b native microbench |
| 0.0-pre | Residual-ceiling arithmetic | gate | **DONE** | `reviews/decision.md`: ~20 streams/box ceiling (triple-conditional), no p50, vs eng-year + carry — thin prize |
| 0.5 | Trace-driven batching sim + graph-capacity model | B | **synthetic DONE; real traces pending** | `spikes/0.5-batching-sim/FINDINGS.md`: realistic mean B≈1.5–2.1 → **3–5× claim effectively dead**; density must come from shared-weights/overlap, not batch-fill |
| 0.4 | Decision memo | gate | scaffolded (template) | `spikes/decision-template.md`; decision tree + pre-registered thresholds block |
| 0.10 | Runtime contract + acceptance tests | pre-1 | pending | protocol/metrics/health/drain/rollback |
| 1.0 | State-ownership design | 1 | pending | actor/snapshot + generation rules |
| 1.1 | Weight/graph asset pipeline | 1 | pending | shared read-only weights (per 0.9) |
| 1.2 | Single-stream steady path | 1 | pending | T1+T0+T2-encoder on 5090 |
| 1.3 | Finalize path + state machine | 1 | pending | fork isolation + reset/resume trace suite |
| 2.1 | Multi-thread dispatch + ownership | 2 | pending | no asyncio loop / exclusive gate |
| 2.2 | Continuous batching (fill B) | 2 | pending | only if 0.5 positive |
| 2.3 | Admission + lane priority | 2 | pending | overload cliff bounded |
| 3.1 | Native CUDA graphs (steady) | 3 | pending | padded-T proof deferred to Python plan |
| 3.2 | Graphed decode + sync compression | 3 | pending | B1 target = parity |
| 3.3 | Fusion-viability spike (optional) | 3 | pending | gates the 6–10 ms headline |
| 4.1 | L4 + L40S sweeps | 4 | pending | density + tail + no p50 regression |
| 4.2 | DGX Spark aarch64 | 4 | pending | gated on 0.7; validate BW floor |
| 4.3 | 5090 density datapoint | 4 | pending | |
| 4.4 | Shadow/mirror-traffic validation | 4 | pending | strongest T1 check |
| 5.1 | Protocol/parity + launcher hardening | 5 | pending | drain/alerting/MPS-restart; readiness = pools captured + T1 canary |
| 5.2 | Deploy integration + cutover | 5 | pending | topology = output of 0.1; side-by-side LB; rollback = drop native backends |

## Review log
- **Round 1** (folded): `reviews/codex-round1.md` + `reviews/opus-round1.md`. Major changes: corrected the thesis to the
  real serializers (asyncio loop / `inference_lock` / exclusive gate / lane sync / Python decode — not "GIL serializes
  launches"); added the §0 worth-it gate (sequenced after the Python plan); split B1→B1a(encoder)/B1b(decode) and made
  the native RNNT decode the real go/no-go (NOTE: round 1 initially mislabeled the target as *frame-looping*
  `loop_labels=False`; **round 2 corrected this to the deployed `greedy_batch` label-looping**); rewrote T1 to be
  streaming-aware + added T2 encoder byte-exact gate; required Spike 0.1 to
  compare single-context/MPS/multi-proc on the real path before claiming "no MPS tax"/40–48/box; added the trace-driven
  batching sim (0.5) to validate 3–5×; made 6–10 ms a separate fusion milestone (not a B1 gate); added a native
  state-ownership design step (1.0) ahead of code; added the aarch64 toolchain pre-check (0.7); added the shared
  read-only weights density lever; corrected anchors (1693-1705/1707-1724, 2052-2084, 4359-4362, 3213-3233, 3109-3137,
  4990-5005).
- **Round 2** (folded): `reviews/codex-round2.md` + `reviews/opus-round2.md`. Major changes: **corrected the decode
  target — deployed = `greedy_batch` LABEL-looping (`loop_labels=True`), not frame-looping** (round 1's "frame-looping"
  was the non-batched fallback, which rejects `partial_hypotheses`); rewrote 0.6 around exact `Hypothesis`/
  `LabelLoopingStateItem` equivalence (partial-hyp merge, max_symbols saturation, all-blank, fork-unchanged); **split
  Phase 0 into Track A (early feasibility, parallel) vs Track B (post-Python residual validation)** — fixed the
  incoherence that 0.1/0.3/0.5 can't make go/no-go from pre-plan traces; turned 0.1 into an **ablation matrix**
  (batch-finalize/exclusive-gate/inference_lock/lane-sync/same-vs-cross-lane/MPS/affinity, CUDA-event timelines); added
  **0.8 native preprocessor byte-exact** + **0.9 model mutability audit** (drop_extra/prompt/eval-train shared state —
  shared-weights isn't read-only); removed the "or documented T1" escape from the **B1a byte-exact gate**; added the
  **state-machine trace suite** to T1 (audio-after-stop / vad_start-flush / reset-before-debounce / end-while-pending /
  stale-finalize) and a **prompted/multilingual** decision (**v1 = EN-only**); added **0.10 runtime contract +
  acceptance tests** (protocol/metrics-parity/health/drain/rollback) before Phase 1 and **metrics parity into Phase 1**;
  added **§9 effort sizing** so §0 is an evaluable comparison; added the **3.3 fusion-viability spike** gating the
  6–10 ms headline.
- **Round 3** (folded): `reviews/codex-round3.md` + `reviews/opus-round3.md`. Major changes: **split 0.6 → 0.6a
  (DEPLOYED *eager* label-looping equivalence, the funding gate) vs 0.6b (Blackwell CUDA-graph-decoder research,
  removed from the gate)** — the deployed decode is `use_cuda_graph_decoder=False`, so the graph-decoder must not gate
  semantic equivalence; **scoped 0.6a down** (no alignments/confidence — `eou_probe_enabled=False`; **reject n-gram LM**;
  **assert `time_jump is None`** (TDT-only); **removed mixed-continuation batch fixtures** — the server forbids mixed
  None/not-None rows, so that's a NEW batching-semantics change, not deployed equivalence); **froze the Track-A oracle**
  (diff golden fixtures from a pinned Python+torch+CUDA commit; rerun against final post-plan commit; **aligned padded-T
  equality with the Python plan's relaxed `allclose` gate** instead of demanding byte-identical padded tensors);
  **decoupled 0.2 from 0.8** (0.2 uses frozen Python mels); added the **0.11 CUDA-graph ownership-model spike**
  (per-lane pools vs shared replay-mutex; capture thread/stream/`thread_local` rules; graph-safe allocator; resident
  memory at K×lanes — the 40–48/box number depends on it); **split T2 → T2a same-shape byte-exact (B1a gate, attainability
  is an output) + T2b cross-B/session-invariance (named tolerance, not bit-equality)**; added a **numerics/determinism
  contract to T0** (TF32 off, deterministic algos, pinned cuBLAS workspace, per-lane handles); extended **0.5 with a
  graph-bucket capacity model**; **rewrote §9 into prototype (Budget A ~12–20 wk) vs production (Budget B ~26–42 wk)
  ≈ a full engineer-year**.
- **Round 4** (folded): `reviews/codex-round4.md` + `reviews/opus-round4.md`. Near-total convergence. Major changes:
  added the **integrated DECISION TREE to 0.4** (outcome combinations → STOP / B4 / B1 / native-under-MPS / fusion);
  finished the **0.6→0.6a rename** and **struck the Blackwell CUDA-graph-decoder from the funding gate** (it's 0.6b
  research) in the §3 B1b row, step 1.2, and risk #2; **reframed B4 as the PREFERRED cheapest-success branch** (skips
  the native ports) rather than an orphan fallback; added **spike Waves** (Wave 1 cheap killers gate Wave 2 expensive
  ports) distinct from Tracks (baseline-dependence); **made the worth-it gate value-based** (corrected the density
  expectation to ~28/box-at-K4, not 64; pause trigger = "residual value < ~40–60 eng-wk + second stack," not an absolute
  streams/box target); **stated THE BET** (triple conjunction: large residual ∧ GIL-bound ∧ B4 insufficient) up front
  with per-conjunct early-exits; **made rollout concrete** (4.4 shadow/mirror-traffic; 5.1 launcher hardening +
  pools-captured readiness; 5.2 side-by-side-LB cutover + rollback; topology as an output of 0.1).
- **Round 5** (folded, final consistency pass): `reviews/codex-round5.md` + `reviews/opus-round5.md`. Both verdicts:
  **sound + actionable, green-light Wave 1 only**; both independently named the same single most-likely-wrong thing —
  **the business premise (0.0 STOP is the probable, correct outcome)**. Changes: **completed the 0.4 decision tree**
  (added rows for 0.2 / 0.8 / 0.9 / 0.5-graph-capacity failures — no more fall-through); **required pre-registered
  numeric pass/fail thresholds for the Wave-1 kills (0.1, 0.5) before data collection**; annotated the **native-under-MPS
  branch as TAIL-ONLY** (no density; re-run 0.0); added a **TL;DR** + reconciled capacity numbers (~20 today / ~28
  post-Python / 40–48 aspirational) and flagged the launcher's legacy `48/64` comments as over-budget (not in-budget);
  clarified **4.4 shadow-traffic supplements (does not gate) the Phase-1 fixture-based T1**; fixed the stale round-1
  log frame-looping sentence. **No remaining blockers.**
- **Path-forward review** (folded): `reviews/codex-pathforward.md` + `reviews/opus-pathforward.md` (advisory, post-v6).
  Both: keep the gate, STOP is the likely+correct outcome, write numeric thresholds *before* measuring, cheapest kills
  first, B4-before-B1. Changes: added **0.0-pre residual-ceiling arithmetic** (free; ~20 streams/box triple-conditional
  upside — the cheapest kill, do first) + the **strategic prior** (density is a COGS lever; compare vs adding L4 boxes /
  multilingual / p50-VAD; price the **dual-stack carry**, not just build); **re-sequenced Wave 1** into decision-value
  order (0.0-pre → 0.5 histogram → 0.3 stage-1 → wire Python Step-5 into 0.1 → 0.1 reduced binary → 0.3 stage-2 → 0.11 →
  0.7; **0.3 before 0.1**); added **B5 (in-tree native-decode pybind extension)** as a 5th backend + decision-tree row
  (decode-throughput/tail only, no second stack); seeded **proposed numeric thresholds** in `decision-template.md`; added
  risks 13 (**moving baseline can obsolete B1 mid-build**) + 14 (**is this the right problem**). Five discussion topics
  surfaced for the user (see the session summary / collaborative-planning step).
- **User decisions (2026-05-24)** folded: density IS the priority (not lean-STOP); 0.0 number DEFERRED until Python
  plan lands; Python baseline NOT frozen (native beats the latest; risk 13); **B5 rejected** (no middle path).
- **B4 removed (2026-05-24, user: "not interested in a Python 3.13 approach; do Rust/C++ or do not proceed").**
  **Outcome space is now native Rust/C++ (B1) or STOP.** THE BET cut from 3 conjuncts to 2 (dropped "B4 insufficient");
  **Spike 0.3 (py3.13t) removed**, its conjunct-2 validation moved into **0.1b — a native launch-overlap microbench**
  (N no-GIL threads replaying the captured encoder graph + mock decode; reuses the graph, no decode-equivalence → far
  cheaper than B1; the cheapest remaining proof that GIL-free native dispatch lifts the knee). Decision-tree B4 branches
  removed (B1-fail → STOP, no fallback). **Consequence noted:** B4 also provided a cheap *end-to-end* conjunct-2 proof;
  0.1b is the replacement, and committing to B1 is a bigger leap if 0.1b is ambiguous. B4/B5 kept as recorded
  non-options in the backend table.
