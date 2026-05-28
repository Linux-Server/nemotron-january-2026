<task>
Produce a detailed investigation proposal for two encoder-kernel optimization levers identified
in the B3-FU-2 roofline analysis (2026-05-28):

1. **Reduce memory traffic in encoder kernels** (lower-precision weights, weight-stationary
   scheduling, quantization, activation-precision tricks).
2. **Improve/fuse the memory-bound GEMM path** (kernel fusion, custom tile sizes via CUTLASS or
   Triton, AOTI epilogue fusion, replacement libraries like TensorRT-LLM streaming, Marlin/Machete
   weight-only-4-bit kernels).

Write the proposal to `proj-2026-05-24-from-scratch-runtime/reviews/encoder-memory-traffic-investigation.md`.

DO NOT run experiments. DO NOT modify runtime code, exports, or AOTI artifacts. The output is a
research / proposal document that the human will review before any implementation work begins.
</task>

<context>

## The model being optimized

`nvidia/nemotron-speech-streaming-en-0.6b` — a streaming Conformer / Cascade-encoder ASR model.
- Encoder: fp32, 128-mel input.
- Streaming: ~80ms chunks; production cadence per stream is 160ms (after accounting for left-context).
- Compiled via `torch.export` + AOTI to per-batch-size `.pt2` packages (B ∈ {1, 2, 4} for steady;
  T ∈ {variable} for finalize).
- Production C++ runtime: `runtime/cpp/session_main.cpp` (monolith with AOTI runner, ~4668 lines).

Code pointers (read as needed):
- `runtime/cpp/session_main.cpp` — encoder forward call sites + AOTI invocation.
- `runtime/cpp/batched_steady_scheduler.{h,cpp}` — the cross-stream batched dispatcher.
- `runtime/cpp/steady_batch_primitive.h` — the steady B primitive.
- `runtime/cpp/density_main.cpp` — the density harness.
- `runtime/export_*.py` — the export scripts (`torch.export` → `aoti_compile_and_package` path).
- `proj-2026-05-24-from-scratch-runtime/PHASE2-PLAN.md` — Phase-2 master plan + lever inventory.

## The roofline data (the bottleneck we're trying to address)

Read these in order:

1. **`proj-2026-05-24-from-scratch-runtime/reviews/B3-FU2-roofline-result.md`** — the NEW NCU/nsys
   roofline measurement (2026-05-28). Dominant kernel: `ampere_sgemm_64x32_sliced1x4_tn`.
   - DRAM throughput p50 = **70.0%** (p95 72.8%) at N=64 production load.
   - SM throughput p50 = 34.4% (66% compute headroom WASTED on memory stalls).
   - Achieved occupancy = 16.7% (vs 33.3% theoretical — B=4 isn't enough to fill the GPU).
   - Arithmetic intensity p50 = **30.7 FLOP/byte** (vs L40S machine balance ~104-106 FLOP/byte
     → kernel is firmly memory-bound).
   - **B=4 did NOT shift the kernel into compute-bound regime** vs B=1 baseline (same profile).
   - Three grid shapes profiled: `(16, 1, 8)` 16 launches, `(64, 1, 4)` 16 launches with one
     reaching 29.6% occupancy, `(16, 5, 1)` 8 launches.
2. **`proj-2026-05-24-from-scratch-runtime/reviews/profiling-paired-verdict.md`** — earlier B=1
   single-stream baseline (the prior context Codex+Opus profiling work that informed Phase 2).
3. **`proj-2026-05-24-from-scratch-runtime/reviews/B3-FU1-result.md`** — knee + memory slope data.
   Per-stream activation slope = **0.173 GiB/stream**. N=64 peak ON = 28.99 GiB on L40S (46 GiB cap).
4. **`proj-2026-05-24-from-scratch-runtime/reviews/B3-L40S-result.md`** — earlier B3 sweep context.

## The knee data + the user-visible binding constraint

- L40S knee: **N=64 SLO-robust production cap**; N=72 ceiling (lag p95 +347ms = 69% of budget);
  N=80 first hard-fail with `lag_p95 = +1140ms` (per-stream wall-clock exceeds 160ms cadence due
  to DRAM saturation projected ~87% at N=80).
- Dispatcher CPU at N=64 = 61% (39% headroom; not the binding constraint, but a co-saturating one).
- Memory pressure: 28.99 GiB at N=64 on L40S (46 GiB cap = ~63% utilized at the knee).

## Latency budget + correctness contract

- TTFS p95 budget: 175ms (currently ~20ms at N=64 → 8× headroom; movability of TTFS is limited
  because user-visible latency is also gated by VAD+WAN; real density lever is per-finalize cost
  reduction, NOT TTFS reduction).
- Cadence: per-stream 160ms (steady encoder must keep up; at N=64 the per-finalize is ~11ms p95
  and 64 streams consume ~64×11 = 704ms of cumulative GPU time per 160ms wall-time; the system
  survives because B=4 amortizes per-stream work + dispatcher overlaps work).
- Byte-exact contract: token-exact + event/delta-exact equivalence vs `finalize_ref.py` (the
  reference Python encoder). AOTI is NOT byte-exact vs PyTorch eager (~1e-2 drift, "F'"); the
  bar is TOKEN-exact + EVENT/DELTA-exact vs `finalize_ref`. WER-within-CI is acceptable where
  token-exact is impossible (e.g., low-precision quantization).

## What's been tried + rejected (don't re-propose without justifying why this time would differ)

From project memory (`.claude/projects/.../memory/`):

1. **fp16 inference**: 0.79× SLOWER than fp32 in prior measurements. Likely because:
   - Activations fp16 + fp32 compute introduced additional cast overhead, OR
   - Full-fp16 hit accuracy issues forcing wider working precision.
   - **If you propose fp16 weights specifically (not full fp16), explain how this differs.**
2. **Bigger single-process GPU** (single huge stream): not a density lever.
3. **Decode-graph CUDA graph**: NO-GO for conc-10 P50 (different kernel; doesn't apply here).
4. **fp16 across the board** is in the dead-end list per memory.

## What HAS shipped + how the runtime fits together

- **CUDA graph for finalize encoder** (`NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE`): the big P95 win
  (finalize was launch-bound, ~1376 launches → graph collapses; byte-exact graph-on==graph-off).
- **Batched-steady scheduler** (`NEMOTRON_DENSITY_BATCH_STEADY`): the density win (B={1,2,4}
  buckets, central dispatcher, byte-exact per-row).
- **Tier-3 memory shrink** (per-bucket sealed loaders, share-ONE-bundle context): −4.65 GiB
  at N=64.
- **Encoder is AOTI-compiled at deploy-time**. Runtime cannot recompile (no torch.compile at
  startup — too slow). Any per-kernel change requires re-export through `torch.export` + AOTI
  compile + ship new `.pt2` packages.

## Constraints

1. **Streaming ASR profile**: small B (≤4), real-time cadence, single small-T tensor per chunk →
   CAN'T use "large-B amortization" tricks that vLLM/SGLang use.
2. **Token-exact contract**: prefer optimizations that preserve byte-exactness. If a lever
   requires WER/accuracy testing, flag explicitly + estimate the test cost.
3. **sm_89 (Ada) primary target**: L40S + L4 production; also consider sm_120 (5090 dev) and
   sm_90 (H100 future). Optimizations should be portable OR have explicit per-arch variants.
4. **AOTI deployment**: no JIT at runtime. Anything chosen must compile through `torch.export` +
   `aoti_compile_and_package` OR be a swap-in custom-op pre-registered before export.
5. **Memory budget**: per-stream slope 0.173 GiB; N=64 cap with 46 GiB L40S; **tighter on L4**
   (24 GiB, projected ~24.3 GiB at N=22 = at-cap). Optimizations that REDUCE memory footprint
   are DOUBLY valuable (push knee on L4 + L40S).

## Tools / libraries to evaluate (non-exhaustive — Codex may add others)

- **torch.compile / Inductor** (already in the pipeline via `torch.export`; check what fusions
  Inductor already produces vs what's left on the table).
- **CUTLASS** (custom GEMM kernels with controlled tile sizes / split-K / fused epilogue).
- **Triton** (custom kernel authoring; integrates with torch.export via custom ops).
- **TensorRT-LLM streaming** (NeMo has TRT export paths for streaming ASR; investigate the
  streaming-shape support and the lift over AOTI).
- **TorchAO** (weight-only quantization: int8, int4, fp8).
- **Marlin / Machete** (4-bit weight-only GEMM kernels; verify streaming-shape min-B support).
- **bitsandbytes** (mature int8 / 4-bit quantization; less optimized for streaming workloads).
- **FlashAttention v2/v3** (IF attention is in the GEMM-heavy hot path — verify from the AOTI
  graph whether the dominant SGEMM is attention QKV / output projection / FFN).
- **CUDA Graphs for steady encoder** (already used for finalize; consider for steady — B varies
  which complicates; possibly per-B graph capture).
- **CUDA MPS + per-stream private context** (orthogonal to BW; informs the L40S+L4 deploy paths).

## The two levers, in detail

### Lever 1: Reduce memory traffic in encoder kernels

**Mechanism**: the dominant kernel is at 70% DRAM utilization with arithmetic intensity 30.7
FLOP/byte ≪ machine balance 104. To shift toward compute-bound, EITHER:
- **Reduce bytes loaded per FLOP** (lower-precision weights → 2-8× weight bandwidth reduction;
  compute stays same OR grows due to dequant) → push AI up by ~2-8× → potentially compute-bound.
- **Reuse weights across more compute** (weight-stationary scheduling: same weights, multiple
  inputs in flight → reduce DRAM reads per FLOP) → push AI up.

Concrete candidates (Codex should evaluate each):

a. **int8 weight-only quantization** (weights int8, activations + compute fp16/bf16):
   - TorchAO / cuBLAS int8 GEMM paths. Mature.
   - Expected: 4× weight bandwidth reduction → ~2-3× AI lift → likely shifts dominant SGEMM
     toward compute-bound. Token-equivalence: WER likely within CI (calibration needed; well-
     established for ASR encoders).
   - Risk: dequant cost in epilogue might eat some gain; need careful kernel choice.
b. **int4 weight-only quantization** (Marlin/Machete):
   - 8× weight bandwidth reduction → higher AI lift; tighter accuracy budget.
   - Risk: Marlin/Machete may not have a streaming-shaped kernel for B≤4; VERIFY.
c. **bf16 weights, fp32 compute** (mixed-precision NOT the failed fp16 path):
   - 2× weight bandwidth reduction. Less aggressive than int8 but lower risk.
   - Compute stays fp32 → no accuracy concerns; byte-exactness possible if the model is fully
     bf16-stable at weight precision.
d. **fp8 weights** (Hopper-and-later only; sm_89 has limited fp8 support):
   - 4× weight bandwidth reduction; H100 only.
e. **Weight-stationary multi-stream scheduling**: redesign dispatcher to batch SAME-SHAPE work
   from multiple streams across more time slots, keeping weights in L2/L1 cache. (NOT the same as
   B≤4 batching, which is one tensor pass per dispatch.)
f. **Activation precision reduction** + selective casting (where numerically safe).

### Lever 2: Improve/fuse the memory-bound GEMM path

**Mechanism**: split-K, larger tile sizes, or fused epilogues can reduce DRAM traffic per FLOP
for small-B GEMMs. The current kernel `ampere_sgemm_64x32_sliced1x4_tn` is a cuBLAS choice for
B={1,2,4}; better tile choices may exist.

Concrete candidates:

a. **CUTLASS custom GEMM** with controlled tile + epilogue fusion (GEMM + bias + activation
   fused into one kernel). Smaller register pressure or larger shared-mem tiles → fewer global
   memory roundtrips.
   - Risk: hand-tuned per-shape; three grid shapes need separate tuning.
b. **Triton custom kernels** for the dominant GEMMs:
   - Easier authoring than CUTLASS; competitive perf on many shapes.
   - Risk: integration into AOTI pipeline (Triton ops must be registered as custom ops before
     `torch.export`).
c. **Fuse adjacent ops** (e.g., LayerNorm + GEMM, GEMM + bias + GELU, GEMM + residual):
   - Reduce intermediate tensor materializations to DRAM.
   - Requires patterns Inductor recognizes; or manual fusion in the exported graph.
   - **Verify what Inductor already fuses in the current export** before proposing.
d. **Replace cuBLAS path with FlashAttention** IF the dominant GEMM is attention-shaped:
   - Verify from the AOTI graph whether the SGEMM is attention QKV, attention output projection,
     or an FFN layer.
e. **TensorRT-LLM streaming export** (whole-encoder replacement):
   - High-effort but potentially the largest single win.
   - Risk: huge integration cost; loses the current AOTI byte-exactness contract; verify
     streaming-shape support.
f. **CUDA Graph capture for steady encoder** (separate question from epilogue fusion):
   - Steady has variable B (1/2/4) — capture per B; like finalize's per-T graphs.
   - Won't shift the BW wall but reduces launch overhead at small B.

## Output contract

Write `proj-2026-05-24-from-scratch-runtime/reviews/encoder-memory-traffic-investigation.md`
covering:

1. **Inventory table** — every candidate (in BOTH levers) with:
   - Expected speedup mechanism (one sentence).
   - Back-of-envelope roofline math: target AI lift (e.g., 30.7 → ?), target DRAM reduction
     (e.g., 70% → ?), per-finalize ms reduction estimate.
   - Maturity (production-proven / well-tested / research / experimental).
   - Eng-weeks estimate (low / medium / high — be specific in weeks).
   - Risk to byte-exactness / WER / latency / stability.

2. **Recommended sequence** — which to try first, why, what data would gate the NEXT step.
   Prefer bounded experiments with clear stop conditions. Identify the
   "easiest-cheapest-first" + the "biggest-potential-payoff" + the "lowest-risk".

3. **Specific measurement design** for each top candidate: what NCU/nsys counter would prove
   the mechanism worked? What success threshold (e.g., AI lift to >90 FLOP/B = compute-bound;
   DRAM <40%; per-finalize ms reduction)?

4. **Risk register** — what could go wrong; what to watch for. Include accuracy/WER risk
   estimation (cite ASR-encoder quantization literature briefly).

5. **Pivot conditions** — if early experiments show a candidate underperforming, what's the
   next-most-likely-to-work lever?

6. **Open questions / verification items** — things you couldn't determine from the docs that
   need to be checked (e.g., "verify from AOTI graph whether dominant SGEMM is attention QKV
   vs FFN" — the answer changes which Lever-2 candidate is best).

Be concrete. Cite roofline math (current AI 30.7 → target what? current DRAM 70% → target
what?). Be honest about uncertainty. Distinguish "high confidence" from "speculative."

Aim for thoroughness over brevity — this is a planning document the human will use to choose
the next 2-4 weeks of optimization work. Length budget: as long as needed.

</context>

<verification_loop>
This is a proposal document, NOT an implementation task. Don't run code or experiments. Verify
proposals by:
- Cross-checking the cited libraries' streaming-shape support (e.g., Marlin's min-B for B≤4).
- Citing the roofline math (AI target = ? FLOP/B; DRAM target = ?%).
- Reading the actual code (`runtime/cpp/session_main.cpp`, the export scripts) to verify
  what's already in the AOTI graph.
- Acknowledging where evidence is missing ("verify from AOTI graph whether dominant SGEMM is
  attention QKV / FFN / projection").
</verification_loop>

<action_safety>
Write only the review doc at
`proj-2026-05-24-from-scratch-runtime/reviews/encoder-memory-traffic-investigation.md`.
Do NOT modify runtime code, exports, or AOTI artifacts. Do NOT run experiments. Do NOT launch
any EC2 / GPU work.
</action_safety>

<compact_output_contract>
Report:
- Absolute path to the written doc.
- One-paragraph headline: top 1-2 recommended candidates + the "easiest-cheapest-first"
  recommendation + total eng-week range for the recommended sequence.
- Top 2 most-important open questions that gate the recommendation.
</compact_output_contract>
