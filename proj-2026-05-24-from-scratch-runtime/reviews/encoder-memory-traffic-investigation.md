# Encoder Memory-Traffic Optimization Investigation

Date: 2026-05-28

Scope: proposal only. I did not run GPU experiments, rebuild AOTI packages, modify runtime code, modify exports, or launch remote work. Evidence came from the cited review docs, code/artifact inspection, and current public docs for candidate libraries.

## Executive Verdict

The best next density lever is selective weight-only quantization of the encoder linear weights, starting with the feed-forward linear layers in the steady B=4 package. The B3-FU-2 roofline says the dominant `ampere_sgemm_64x32_sliced1x4_tn` launches are at AI p50 30.7 FLOP/B, DRAM p50 70.0%, and SM p50 34.4%, far below the L40S balance point of about 104-106 FLOP/B. The AOTI B=4 wrapper also shows the remaining hot path is not unfused pointwise work: Inductor already emits many Triton fused pointwise/reduction kernels, but still leaves 217 `aoti_torch_cuda_mm_out` calls and 72 `aoti_torch_cuda_bmm_out` calls. Constants in the B=4 package are about 2.31 GiB, with about 1.54 GiB in feed-forward linear weights and about 480 MiB in attention linear weights.

The first real implementation candidate should therefore be an int8 weight-only path for encoder linears, preferably via TorchAO if the packed quantized linears survive `torch.export` -> AOTI, otherwise as a custom registered W8A16/W8A32 linear op using cuBLASLt/CUTLASS. A bounded first pass should target AI 30.7 -> 75-95 FLOP/B, DRAM p50 70% -> below 45%, and steady GPU p95 about 10.9 ms -> 6.5-7.5 ms at N=64. The biggest potential payoff is int4 W4A16 via Marlin/Machete/CUTLASS-style kernels, but that should be second because B<=4 streaming shape and AOTI integration are still unproven. The lowest-risk work is graph attribution plus an Inductor fusion audit; it is cheap and gates the exact kernel target, but by itself probably does not move the bandwidth wall.

## Evidence Base

### Local Profiling Facts

- `reviews/B3-FU2-roofline-result.md`: B=4 isolated SGEMM remained memory-bound. Median DRAM throughput was 70.0%, median SM throughput 34.4%, achieved occupancy 16.7%, and estimated AI 30.7 FLOP/B versus L40S machine balance about 104-106 FLOP/B. The dominant DRAM-heavy grids were `(16,1,8)` and `(64,1,4)`.
- `reviews/profiling-paired-verdict.md`: the earlier B=1 profile had the same basic kernel character: 71-72% DRAM, 34-39% SM, 15-17% occupancy. B=4 improved row economics but did not make the kernel compute-bound.
- `reviews/B3-FU1-result.md`: production operating point is N=64, N=72 is already at the cliff, N=80 hard-fails keep-up. N=64 peak memory is 28.99 GiB and the measured peak slope is 0.173 GiB/stream.
- `reviews/B3-L40S-result.md`: the shipped batched-steady scheduler lifted the L40S knee to at least N=64, but burst synchronized starts still fail and high-N B=4 rows have counted-not-gated event-only drift.

### Runtime And Export Facts

- `runtime/cpp/session_main.cpp` is now a thin wrapper over `runtime/cpp/lib/session/session.cpp`.
- `runtime/cpp/lib/session/session.cpp` loads steady via `AOTIModelPackageLoader enc_steady(.../enc_steady_aoti.pt2, "model", false, 1, -1)` and calls `loader.run(inputs)` on contiguous inputs.
- `runtime/cpp/steady_batch_primitive.h` loads sealed B1/B2/B4 steady AOTI packages, injects shared constants, and runs `loader.run(inputs, stream_handle)` on the dispatcher stream.
- `runtime/cpp/batched_steady_scheduler.cpp` packs ready rows, runs one B bucket, then unpacks per-row outputs. The TODO at the telemetry sync notes a host sync remains for elapsed-time telemetry, but output sync p95 is tiny in B3-FU-2 and this is not the roofline wall.
- `runtime/export_steady_batched.py` exports fixed static B in {1,2,4} via `torch.export.export`, then `torch._inductor.aoti_compile_and_package`, with autotune off in the current path.
- AOTI artifact inspection of `runtime/steady_b_artifacts/enc_steady_aoti_b4.pt2` showed the generated wrapper still calls:
  - 217 `aoti_torch_cuda_mm_out`
  - 72 `aoti_torch_cuda_bmm_out`
  - 77 `aoti_torch_cuda_convolution`
  - many Triton fused pointwise/reduction kernels for layer norm, silu, conv/pad/relu, cat/clone, softmax/mask, and residual-style ops.
- The same wrapper's constants metadata sums to about 2.31 GiB:
  - feed-forward linear weights: about 1536 MiB
  - attention linear weights: about 480 MiB
  - conformer convolution weights: about 289 MiB
  - pre-encode linear weight: about 17 MiB
  - biases/norm/batchnorm are negligible by comparison.

Interpretation: the dominant remaining memory traffic is highly likely encoder linear weight streaming, especially FFN linear1/linear2, not unfused elementwise glue. This still needs kernel-to-FQN attribution from a profile with source annotations, but the current AOTI graph strongly favors FFN-first quantization.

### Public Library Checks

- PyTorch AOTInductor officially packages `torch.export` exported programs into `.pt2` with `torch._inductor.aoti_compile_and_package` and loads them with `aoti_load_package`: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_aot_inductor.html
- TorchAO exposes int8 weight-only, int4 weight-only, intx weight-only, float8 weight-only, and dynamic activation plus weight quantization configs for linear layers: https://docs.pytorch.org/ao/stable/api_reference/api_ref_quantization.html
- PyTorch custom C++/CUDA operators require proper registration and FakeTensor/meta kernels for compiler/export integration; `torch.library.triton_op` is preferred for Triton implementations that should remain visible to compiler subsystems: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html and https://docs.pytorch.org/docs/stable/library.html
- PyTorch developer discussion says user-defined Triton kernels can be precompiled and bundled by AOTI, while C++ CUDA custom kernels should be registered as custom ops with FakeTensor support: https://dev-discuss.pytorch.org/t/torch-export-with-aot-indutor-for-cute-dsl-kernels/3272
- CUTLASS 3.x supports explicit GEMM mainloop/epilogue composition and a broad type set including BF16, FP8, 4-bit, and 8-bit integer/floating formats across NVIDIA architectures: https://docs.nvidia.com/cutlass/4.3.5/media/docs/cpp/gemm_api_3x.html and https://docs.nvidia.com/cutlass/latest/overview.html
- Marlin is explicitly an FP16 x INT4 LLM inference kernel, with near-ideal speedups reported up to batch sizes 16-32 tokens; vLLM's Marlin config requires SM80+, half activations, output features at least 64, and input features at least 128: https://github.com/IST-DASLab/marlin and https://docs.vllm.ai/en/v0.10.1/api/vllm/model_executor/layers/quantization/marlin.html
- Machete targets W4A16/W8A16 mixed-input linear kernels on Hopper and beyond, with vLLM support; Red Hat's write-up explicitly says low-batch optimization remains future work, so B<=4 is not yet a safe assumption: https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel and https://docs.vllm.ai/en/latest/api/vllm/config/kernel/
- bitsandbytes provides `Linear8bitLt` and `Linear4bit`, but it is a Python CUDA wrapper stack rather than an AOTI-native deployment path: https://huggingface.co/docs/bitsandbytes/v0.43.0/en/index
- NeMo ASR docs support cache-aware streaming FastConformer and note export with cache support, but the documented TensorRT-LLM NeMo export path is LLM-focused; ASR likely goes through ONNX/TensorRT rather than TensorRT-LLM without custom work: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html and https://docs.nvidia.com/nemo-framework/user-guide/25.04/deployment/llm/nemo_models/optimized/tensorrt_llm.html
- TensorRT-LLM quantization docs support FP8 and quantized engine workflows, but emphasize calibration and quality checks; several features are Hopper/FP8/model-family-specific: https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html
- FlashAttention-2 improves attention by reducing non-matmul work and improving attention parallelism, but the local graph indicates the dominant weights are FFN linears, not attention score/value matmuls: https://arxiv.org/abs/2307.08691
- ASR quantization literature supports feasibility but not token exactness: Q-ASR reports negligible WER degradation for Conformer INT8 with up to 2.35x T4 speedup and 4x compression; a recent Conformer low-bit paper reports performance-lossless 2-bit/1-bit quantization only with co-training, not trivial PTQ: https://arxiv.org/abs/2103.16827 and https://arxiv.org/abs/2505.21245

## Roofline Math Used Below

Baseline:

- AI0 = 30.7 FLOP/B.
- DRAM0 = 70.0% of sustained peak.
- SM0 = 34.4% of peak.
- L40S balance = about 104-106 FLOP/B.
- Steady GPU p95 at N=64 = about 10.9 ms.
- Dominant SGEMM launch p50 = about 34.9 us.

If a kernel is memory-bound, reducing bytes by factor `r` gives first-order AI `AI1 = AI0 / r` and duration `t1 ~= t0 * r` until compute-bound behavior is reached. The missing term is the fraction of bytes that are reducible weights. For the dominant linear GEMMs I assume weight bytes are 75-90% of traffic. That gives:

- BF16/FP16 weight storage, 2x smaller weights: byte factor about 0.55-0.63, AI about 49-56 FLOP/B.
- INT8 weight-only, 4x smaller weights: byte factor about 0.33-0.44, AI about 70-93 FLOP/B.
- INT4 weight-only, 8x smaller weights: byte factor about 0.21-0.34, AI about 90-146 FLOP/B.

These are targets, not promises. Dequant overhead, activation dtype conversion, per-layer scale loads, and small-B kernel shape can erase part of the win. A true success should show both lower measured bytes and shorter kernel duration, not just a different kernel name.

## Inventory Table

| Candidate | Mechanism | Roofline target | Estimated p95 steady/finalize cost impact | Maturity | Eng estimate | Risks |
|---|---|---:|---:|---|---:|---|
| TorchAO int8 weight-only linears, FFN-first | Store linear weights int8 with per-channel/group scales; dequant inside linear kernel; reduce dominant weight reads by about 4x. | AI 30.7 -> 75-95 FLOP/B; DRAM 70% -> 35-45% if FFN linears dominate; package/shared constants -1.4 to -1.7 GiB. | 10.9 ms -> 6.5-7.5 ms if kernel is good; maybe only 8-9 ms if dequant/casts dominate. | Well-tested API, but AOTI export of packed linears must be verified. | 2-4 weeks prototype; 4-6 weeks production across buckets. | Not byte-exact; WER gate needed. May require FP16/BF16 activations, which differs from current fp32 path and prior full-fp16 failure. Packed tensor subclass may not lower through AOTI C++ runtime. |
| Custom int8 W8A16/W8A32 linear op via cuBLASLt or CUTLASS | Replace selected `aten.mm` linears with a registered quantized linear custom op before export. | Same as int8 above; success requires kernel-level measured bytes down >=2x and AI >75. | Similar to TorchAO int8, but more controllable. | Production-proven components, local integration custom. | 4-8 weeks. | Custom op registration, FakeTensor, C++ runtime loading, per-arch kernels, calibration, and drift/WER. |
| INT4 weight-only via Marlin | W4A16 kernel reduces weight bytes about 8x and uses prepacked 4-bit layout. | AI 30.7 -> 100-140 FLOP/B if weight bytes dominate; DRAM <35%; constants -1.7 to -1.9 GiB. | 10.9 ms -> 4.5-6.5 ms possible; B<=4 may underperform. | Mature in vLLM LLM decode/prefill, not proven for this AOTI ASR shape. | 4-8 weeks after int8 attribution. | Marlin supports half activations and is optimized around LLM shapes; B<=4 and fp32-current path are unproven. AOTI integration is custom-op work. Higher WER risk than int8. |
| INT4/W4A16 via Machete | Hopper-oriented mixed-input kernel that improves 4-bit linear performance in larger/compute-bound regimes. | H100/sm90 target AI >100; not primary for sm89 L40S. | Potentially 2x+ on H100; uncertain on L40S/L4. | vLLM backend, Hopper-focused. | 6-10 weeks, mostly future/H100 path. | Red Hat notes low-batch optimization is future work; primary production is sm89, not Hopper. Not first-line for L40S. |
| BF16 or FP16 stored weights with FP32 accumulation | Store constants at 16-bit and use fp32 accumulation or cast-on-load to cut weight bytes by 2x without full-fp16 activations. | AI 30.7 -> 45-60; DRAM 70% -> 50-60%; constants -1.0 to -1.2 GiB. | 10.9 ms -> 8.5-9.5 ms if casts are fused; could be slower if explicit casts materialize. | Low conceptual risk, but kernel path uncertain. | 1-3 weeks to test; 3-5 weeks to productionize if real. | Prior full-fp16 was 0.79x slower. This differs because activations/compute need not be full fp16, but if AOTI inserts casts before each GEMM it can recreate the old failure. Not byte-exact vs fp32 weights. |
| FP8 weights | 4x smaller weight storage with FP8 hardware paths on newer NVIDIA stacks. | AI similar to int8, 75-95; DRAM <45%. | Similar to int8 on supported arch; negligible value on L4/L40S unless support is proven. | Production in TensorRT-LLM/Hopper-focused stacks, less clear for sm89 AOTI. | 3-6 weeks for H100-only branch; more for cross-arch. | Architecture split; TensorRT-LLM docs are LLM-centric and quality checks are mandatory. Do not make this the sm89 primary path. |
| Activation precision reduction/selective casting | Reduce activation/cache/intermediate bytes in linear, attention, and conv sections; maybe enable tensor-core kernels. | AI 30.7 -> 35-50 unless paired with weight quant; DRAM 70% -> 55-65%. | 5-15% if casts are free; can regress if casts materialize. | Common, but this model has prior full-fp16 failure. | 2-4 weeks. | Accuracy and recurrent cache drift. Full-fp16 already failed, so this must be selective and tied to margin/WER gates. |
| Weight-stationary multi-stream scheduling beyond B=4 | Hold same weights in cache/shared memory across more same-shape inputs or dispatch slots instead of one B<=4 pass. | Effective weight bytes/stream could halve again if B_eff 8-16; AI 60-120 depending reuse. | 10.9 ms -> 6.5-8.0 ms if enough work can be held without gather latency. | Research/custom runtime. | 6-12 weeks. | Streaming cadence and B<=4 contract make large-B tricks hard. Added gather wait can break lag. Requires major dispatcher/export redesign and likely new kernels. |
| bitsandbytes int8/4-bit linears | Mature quantized Linear modules and CUDA kernels. | Same theoretical int8/int4 byte reduction. | Unknown for AOTI; likely poor fit without Python runtime. | Mature for Python/Transformers use. | 1 week to rule in/out; 4+ weeks if wrapping. | Less AOTI-native; likely not suitable for C++ deployment without custom wrapping. Treat as reference/calibration tool, not production path. |
| Inductor fusion audit / graph rewrite | Confirm what Inductor already fuses and remove remaining materialized bias/residual/activation tensors where possible. | AI 30.7 -> 35-45; DRAM 70% -> 55-65% if any large intermediates remain. | 0.5-1.5 ms if only epilogues; maybe less because many fusions already exist. | Existing compiler path. | 0.5-2 weeks for audit; 2-4 weeks for a safe rewrite. | Local wrapper already shows many Triton fusions; remaining win may be small. Must not over-focus here unless profiler proves intermediate traffic. |
| CUTLASS custom GEMM with fused epilogue | Replace cuBLAS `ampere_sgemm_64x32_sliced1x4_tn` for selected shapes with tuned tiles and fused bias/silu/residual. | AI 30.7 -> 40-60 without quant; DRAM 70% -> 45-60%; more if combined with W8/W4. | 10.9 ms -> 8-9.5 ms unquantized; 5-7 ms if combined with int8/int4. | Production-proven library; local kernels bespoke. | 4-8 weeks per useful shape family. | Shape-specific tuning, per-arch variants, custom op/AOTI integration, and maintenance. Without lower precision it may not move the byte floor enough. |
| Triton custom GEMM / fused dequant+linear | Easier custom kernels for small fixed shapes and fused dequant/epilogue. | Similar to CUTLASS; with quant target AI 75+. | Similar to CUTLASS but more uncertain peak perf. | Well-used compiler stack; custom kernels need tuning. | 3-6 weeks. | Need `torch.library.triton_op` or custom-op registration before export; sm89/sm90/sm120 tuning; risk slower than cuBLAS on GEMM. |
| FlashAttention v2/v3 replacement | Fuse attention score/softmax/value path and reduce attention intermediate traffic. | Only applies to BMM/softmax, not FFN weights. AI maybe +0-10% overall. | 0-0.5 ms likely unless profiler proves attention dominates. | Production-proven for transformer attention, not necessarily Conformer relative streaming shape. | 2-5 weeks to evaluate. | AOTI graph shows FFN linear weights dominate constants; attention BMMs are not the observed `ampere_sgemm` FFN-like weight stream. Custom masks/relative position may block direct use. |
| TensorRT or TensorRT-LLM whole-encoder path | Let TensorRT choose fused/quantized kernels and engine profiles for steady/finalize shapes. | Potential AI and fusion improvements, especially with INT8/FP8. | 20-50% possible if export and cache support fit; high variance. | TensorRT production-proven; TensorRT-LLM path is LLM-centric, NeMo ASR export is ONNX/cache-support oriented. | 8-14 weeks. | Loses current AOTI contract, needs new C++ runtime integration, new correctness/WER gates, dynamic finalize buckets, engine profile management, per-arch builds. |
| Torch-TensorRT subgraphs inside AOTI | Keep `.pt2` deployment while embedding TRT-convertible subgraphs into AOTI package. | Could improve GEMM/fusion if TRT captures linears/attention. | 10-30% possible; depends on subgraph capture. | Documented AOTI integration, but model-specific. | 4-8 weeks. | Export/retrace complexity, fallback subgraph behavior, shape profile setup, and token/WER drift. |
| CUDA Graph capture for steady encoder | Capture per-B steady AOTI run to remove launch overhead and driver variability. | No AI change; DRAM remains about 70%. | At N=64 probably 0-8%; larger tail benefit at low load. | Already shipped for finalize pattern, but steady B varies. | 2-4 weeks. | Does not solve bandwidth wall. Requires per-B static buffers and careful output lifetime. Good low-risk hygiene, not the main density lever. |
| CUDA MPS / per-stream private context | Deployment topology and isolation, not a kernel memory-traffic fix. | No AI or DRAM reduction. | 0% kernel win; may improve multi-process packing or hurt via memory duplication. | Production operational tool. | 1-3 weeks for deployment characterization. | Orthogonal to SGEMM wall. Do not count as encoder-kernel speedup. |

## Recommended Sequence

1. Attribution and graph audit, 0.5-1.0 week.
   Extract exact FQN/source mapping for the top `ampere_sgemm_64x32_sliced1x4_tn` launches. The wrapper strongly suggests FFN linear weights, but the next implementation choice depends on whether the two DRAM-heavy grids are FFN linear1/linear2, attention Q/K/V/out, or pre-encode/projection. This step also documents which bias/residual/silu/layernorm ops Inductor has already fused.

   Stop condition: if the dominant launches are not linears with large static weights, deprioritize weight-only quantization and pivot directly to fusion/tile work for the real source.

2. Easiest-cheapest-first real experiment: selective int8 FFN weight-only prototype on the steady B=4 package, 2-4 weeks.
   First attempt should use TorchAO because it is the least custom code if it exports. Restrict the first pass to FFN linear1/linear2 weights, because they are about 1.5 GiB of constants and dominate likely weight bytes. Keep attention/conv/pre-encode fp32 initially to reduce WER risk and isolate speed.

   Gates:
   - AOTI compile succeeds and C++ loader can run the package.
   - NCU shows AI p50 >=75 FLOP/B or DRAM p50 <=45% on the formerly dominant GEMM family.
   - N=64 steady GPU p95 improves at least 20% (10.9 ms -> <=8.7 ms); continue only if likely production improvement is >=25%.
   - T1 token/event gate passes or WER delta is within CI if token exactness is impossible.

3. If TorchAO export or performance fails, move to a custom registered quantized linear op, 4-8 weeks.
   Implement only the top one or two shape families first. Prefer CUTLASS/cuBLASLt W8A16/W8A32 over hand-written raw CUDA. Use the same measurement gates as step 2, but add an explicit custom-op packaging gate: registered op is loaded before export, has FakeTensor/meta support, and is callable from Python-less C++ AOTI runtime.

4. If int8 works and WER is clean, consider int4 as the biggest-potential-payoff branch, 4-8 additional weeks.
   Do not start with Marlin/Machete blindly. Verify B<=4 and M/N/K compatibility against the actual shapes first. Marlin's documented sweet spot is LLM FP16 x INT4 and batch sizes up to 16-32, with vLLM requiring half activations and SM80+. Machete is Hopper-oriented and still calls out low-batch optimization as ongoing work. If the shape fits poorly, build a CUTLASS W4A16 custom op instead of trying to force an LLM-serving kernel.

5. Run CUTLASS/Triton unquantized epilogue fusion only if attribution shows large intermediate traffic remains or int8 underperforms because of cuBLAS kernel choice, 3-8 weeks.
   The wrapper already shows many fused Triton kernels. Unquantized GEMM tile work is probably a 10-20% lever, not the 2x lever, unless it also fuses dequant and epilogue.

6. Treat TensorRT/Torch-TensorRT/TensorRT-LLM as the strategic rewrite branch, 8-14 weeks.
   This is the largest integration change and should only follow a failed AOTI quantized-linear path or a decision to target H100/Blackwell as the primary production architecture. For this ASR model, NeMo's documented streaming export path is cache-aware ONNX export; TensorRT-LLM docs are LLM-focused, so verify streaming cache and RNNT/Conformer support before budgeting it as a kernel replacement.

Recommended 2-4 week plan: attribution/audit plus selective FFN int8 weight-only prototype. Recommended full sequence to a production candidate: about 5-10 eng-weeks if TorchAO works, about 8-16 eng-weeks if a custom op is required. Biggest-payoff branch after that is int4, but it should be gated by the int8 result.

## Measurement Designs For Top Candidates

### Candidate A: Selective INT8 Weight-Only FFN Linears

Mechanism proof:

- NCU roofline on the same launch window used by B3-FU-2.
- Compare old vs new for the exact former hot kernel family:
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed`
  - `dram__bytes_read.sum`
  - `lts__t_bytes.sum`
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed`
  - tensor op counters if the new kernel uses tensor cores
  - kernel duration p50/p95
  - achieved occupancy
- Confirm kernel attribution: new quantized kernels should line up with FFN linears from the wrapper, not just move time elsewhere.

Success threshold:

- AI p50 >=75 FLOP/B, ideally >90.
- DRAM p50 <=45% or bytes/read per linear down >=2x.
- Former 34.9 us DRAM-heavy launch p50 <=25 us, or equivalent weighted sum improvement.
- N=64 steady GPU p95 <=7.5 ms for a strong GO; <=8.7 ms minimum to continue.
- Peak memory lower by >=1.0 GiB on L40S and projected L4 memory cap improves by at least 5 streams from base-weight savings if performance also passes.
- Correctness: token exact preferred. If not, full corpus WER delta within CI, with near-tie/margin analysis on divergences.

### Candidate B: BF16/FP16 Stored Weights With FP32 Accumulation

Mechanism proof:

- Verify generated wrapper does not materialize full fp32 copies of weights before each GEMM.
- NCU should show bytes/read down at least 25% on the same linear kernels.
- Kernel name and math mode should be documented: if the path silently becomes full fp16 activation compute, it is not this candidate.

Success threshold:

- AI >=45 FLOP/B.
- DRAM p50 <=60%.
- N=64 steady GPU p95 <=9.5 ms.
- Token exact or at least zero token divergences on existing T1 corpus shadow. This is the lowest-risk precision branch, so a WER-only pass is less attractive than for int8.

Stop condition:

- Any cast kernel or explicit pre-GEMM copy that consumes more than 5% of steady time.
- No more than 10% p95 improvement, because it would not justify carrying a second precision artifact family.

### Candidate C: Custom CUTLASS/Triton Fused Quantized Linear

Mechanism proof:

- One top shape only at first, matched to the attributed FFN shape.
- Compare against cuBLAS SGEMM and TorchAO if available.
- Measure bytes/read, duration, and output tensor materializations.
- Confirm the op compiles into the `.pt2` package and is callable in the C++ runtime without Python.

Success threshold:

- For int8: same as Candidate A.
- For int4: AI >=100 FLOP/B, DRAM <=35%, and steady GPU p95 <=6.5 ms.
- No more than +0.1 percentage point traditional WER delta, or within the pre-registered CI envelope.

Stop condition:

- Custom kernel is within 10% of cuBLAS/TorchAO but adds new runtime registration and maintenance.
- Per-arch tuning makes sm89 and sm120 diverge enough that artifacts become operationally fragile.

### Candidate D: CUTLASS/Triton Unquantized GEMM + Epilogue Fusion

Mechanism proof:

- Show separate post-GEMM bias/silu/residual/layernorm kernels disappear or output intermediate bytes drop.
- Show `dram__bytes_read.sum + dram__bytes_write.sum` per layer decreases by >=15%.

Success threshold:

- AI >=40 FLOP/B.
- DRAM p50 <=55%.
- N=64 steady GPU p95 <=9.0 ms.

Stop condition:

- If the only improvement is launch count and not bytes, route the same effort to steady CUDA graph instead.

### Candidate E: Steady CUDA Graph Capture

Mechanism proof:

- Per-B graph capture for B=1/B=2/B=4 with static input/output buffers.
- nsys launch/API overhead decreases; GPU kernel durations and DRAM percentages remain basically unchanged.

Success threshold:

- At N=64, steady p95 improves >=5% or p99/tail improves materially with no correctness drift.
- At low load, launch tail improves enough to justify operational complexity.

Stop condition:

- No high-N p95/p99 improvement and dispatcher/DRAM still dominate. Keep as tail hygiene only.

### Candidate F: TensorRT/Torch-TensorRT Whole Encoder

Mechanism proof:

- Demonstrate cache-aware streaming shape export, not just offline ASR export.
- Engine profiles cover B=1/2/4 steady and finalize T buckets.
- Compare kernel mix to AOTI: fewer cuBLAS SGEMMs, fused quantized linears or TRT fused subgraphs.

Success threshold:

- N=64 steady GPU p95 <=7.0 ms.
- Token/event exact if possible; otherwise full WER non-inferiority with event/delta validation.
- Artifact build/deploy process no more fragile than current AOTI bucket workflow.

Stop condition:

- No cache-aware streaming support with exact state contract, or engine profile explosion makes finalize unmanageable.

## Risk Register

| Risk | Why it matters | Mitigation |
|---|---|---|
| Token/event correctness loss | Quantization and custom kernels will not be byte-exact; the current contract is token exact plus event/delta exact unless WER is explicitly accepted. | Use the existing corpus shadow, argmax-margin reporting, and WER CI gates. Start FFN-only to reduce risk. |
| Prior full-fp16 failure repeats | Full fp16 was measured 0.79x slower. Weight-only or stored-weight-only is different, but accidental activation casting can recreate it. | Inspect generated graph for cast materialization and measure cast kernels separately. Do not call this a win without NCU bytes and p95 proof. |
| AOTI export incompatibility | Packed quantized tensor subclasses or custom ops may fail `torch.export` or AOTI C++ runtime. | Make export/AOTI/C++ load a first gate before performance work. Register FakeTensor/meta kernels. |
| Small-B kernel mismatch | Marlin/Machete/vLLM kernels are optimized around LLM token serving; B<=4 ASR chunks may not hit the intended regime. | Verify actual M/N/K and batch constraints before implementation. Prefer one-shape micro-prototype. |
| Dequant overhead eats byte savings | Weight-only quantization can shift the bottleneck to dequant instructions or scale loads. | NCU must show both bytes down and duration down. Compare int8 vs bf16 vs int4 on one shape. |
| Architecture split | Primary production is sm89 L40S/L4; sm120 dev and sm90 future differ in FP8, WGMMA, TensorRT, and Machete support. | Define per-arch artifact policy. Do not pick H100-only FP8/Machete as the L40S answer. |
| Memory savings overclaimed | Per-stream slope 0.173 GiB remains mostly activation/queue pressure; weight compression reduces base constants, not per-stream slope. | Report base-memory and slope separately. On L4, base savings still matter because it is near cap. |
| Dispatcher becomes binding after GPU win | B3-FU-2 dispatcher CPU is already about 61% at N=64. A large GPU win may expose dispatcher CPU/queue limits. | Continue to report dispatcher CPU, stream util, gather/service wait, queue p95. Pair kernel work with dispatcher telemetry. |
| Event-only drift policy ambiguity | B=4 high-N rows are token-clean but event drift counted-not-gated. Quantized paths may add more drift. | Pre-register whether WER-within-CI is acceptable for quantized experiments and keep event drift counted and surfaced. |
| TensorRT rewrite loses known contract | Current runtime has AOTI package SHAs, bucket manifests, shared constants, and finalize_ref alignment. | Treat TensorRT as separate backend with separate correctness gates, not a drop-in optimization. |

Accuracy risk estimate: INT8 ASR quantization is credible but still a WER-gated change. Q-ASR reports Conformer INT8 with negligible WER degradation and up to 2.35x T4 speedup, but that was a quantization method and model context, not this exact streaming RNNT. More aggressive 2-bit/1-bit Conformer results require co-training and should not be used to justify simple PTQ int4. For this model, int8 FFN-only is the safe first quantization point; int4 requires calibration and a full WER gate.

## Pivot Conditions

- If graph attribution shows the hot `ampere_sgemm` is not FFN/linear weight streaming, pivot from quantization-first to CUTLASS/Triton shape-specific kernels for the actual hot op.
- If TorchAO int8 does not export through AOTI, pivot to a custom registered quantized linear op rather than bitsandbytes.
- If int8 exports but AI stays below 50 FLOP/B or DRAM stays above 60%, quantized weights are not reducing the real traffic. Pivot to fusion/tile measurement and look for activation/intermediate traffic.
- If int8 improves NCU but not end-to-end p95, inspect dispatcher CPU/gather wait and post-GEMM cast/dequant overhead before trying int4.
- If int8 WER fails, try layerwise mixed precision: keep attention/conv/final projection fp32 and quantize only FFN, then add AWQ/SmoothQuant-style calibration. If still failing, fall back to BF16 stored weights or epilogue fusion.
- If Marlin/Machete fail B<=4 shape support, skip them for sm89 and build a CUTLASS/Triton W4/W8 kernel only for the top local shape.
- If unquantized CUTLASS/Triton improves less than 10%, do not continue tuning unquantized GEMM. The byte floor needs precision or reuse, not tile polish.
- If TensorRT cannot preserve cache-aware streaming state and bucket shapes, keep it as a future backend, not a Phase-2 kernel optimization.

## Open Questions / Verification Items

1. Exact hot-GEMM attribution: map the two DRAM-heavy grid shapes `(16,1,8)` and `(64,1,4)` to source FQNs. Wrapper evidence points to FFN linears, but the recommendation should be gated on hard attribution.
2. TorchAO export/AOTI viability: do `Int8WeightOnlyConfig` or `IntxWeightOnlyConfig` packed linears survive `torch.export` -> `aoti_compile_and_package` -> C++ `AOTIModelPackageLoader` for this graph?
3. Activation dtype requirement: can the int8 path keep fp32 activations/outputs with reduced weight traffic, or does it require W8A16/BF16 activations? If it requires half activations, it must be explicitly distinguished from the failed full-fp16 path.
4. Calibration data and WER gate: which corpus and CI threshold will govern quantized experiments? Existing full-1000 shadow and semantic WER tooling should be reused, but the acceptance threshold must be pre-registered.
5. L4 capacity effect: how much of the 24 GiB L4 pressure is base constants versus per-stream slope after Tier-3? Weight quantization reduces base memory and package size, but not the 0.173 GiB/stream slope unless activation/cache precision also changes.
6. Inductor fusion residual: are any GEMM bias/activation/residual epilogues still separate enough to justify CUTLASS epilogue fusion, or are they already fused into nearby Triton kernels?
7. Per-arch support matrix: sm89 L40S/L4, sm120 5090, and future sm90 H100 need separate feasibility rows for FP8, Machete, CUTLASS layouts, and TensorRT profiles.
8. Steady CUDA graph value at high load: finalize graphs were a major P95 win because finalize was launch-bound. Steady at N=64 is bandwidth-bound; graph capture should be measured as tail hygiene, not assumed to move the knee.

## Bottom Line

Do not start with a whole TensorRT rewrite or unquantized GEMM tile tuning. The roofline and AOTI wrapper both point to static linear weight traffic, especially FFN weights. The next 2-4 weeks should be: hard attribution, then a selective int8 FFN weight-only prototype through the exact AOTI deployment path. If that raises AI toward 75-95 FLOP/B and drops steady p95 below about 7.5-8.7 ms without WER regression, it becomes the production path. If it fails for export/runtime reasons, build the same idea as one custom quantized linear op. If it fails for accuracy reasons, fall back to BF16 stored weights and unquantized fusion while reserving int4 for a later calibrated branch.
