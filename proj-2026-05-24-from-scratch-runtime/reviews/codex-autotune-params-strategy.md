# AOTI/Inductor Autotune Parameters And T1 Search Strategy

## Recommendation

The best first compile is a precision-matched `max_autotune` run without coordinate descent, with the precision globals set before `aoti_compile_and_package`, and with the package written to a separate artifact directory. The exact first candidate is **R1a** below if the accepted eager reference is the task's TF32-reduced policy. If a preflight in the same environment that generated `t2a_io.pt` reports raw Torch defaults instead, use **R1b** first. This is not a broad search; it is the same "match eager precision, then tune schedule" rung instantiated to the measured eager policy.

I would spend one compile/T1 cycle on R1 and one fallback cycle on R2. I would not chase coordinate descent, exhaustive search, or CUTLASS unless R1/R2 both pass the package-specific T1 gate and show a real warm steady-density win over the autotune-off floor. The existing log already shows many small GEMMs and the visible conv have ATen as the fastest row, so the likely safe speed upside may be modest rather than transformative.

## Ground Truth From Existing Work

- The baseline steady compiler does not set TF32, matmul precision, or autotune flags. It calls `torch._inductor.aoti_compile_and_package(ep, package_path=pkg_path)` directly, then compares the package against `t2a_io.pt`: `runtime/aot_compile.py:41-64`.
- The baseline AOTI sanity log reports `cache_t` max_abs `1.663e-02`: `runtime/artifacts/logs/aot_compile.log:26-31`.
- The knob matrix was deliberately run one knob per fresh process because precision/config globals leak across compiles: `runtime/aot_knob_matrix.py:87-96`, `runtime/0.2b-aoti-findings.md:109-112`.
- The knob matrix says default is the closest tested AOTI policy, while `fp32_highest` and `emulate_precision_casts` both jump to about `1.027e+01` overall and `1.03e+01` on `cache_t`: `runtime/artifacts/logs/knob_matrix.log:1-16`, `runtime/0.2b-aoti-findings.md:114-128`.
- The aggressive autotune run used only `max_autotune=True` plus `coordinate_descent_tuning=True`: `runtime/artifacts_at_sm120/compile_steady_manifest.json:20-29`, `runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:16-23`.
- That package has SHA `4026b7d...542e8` and the single-step check reports `cache_t` max_abs `10.26782512664795`: `runtime/artifacts_at_sm120/compile_steady_manifest.json:33-34`, `runtime/artifacts_at_sm120/compile_steady_manifest.json:79-95`, `runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:319-324`.
- The logged matmul/bmm Triton candidates use `ACC_TYPE='tl.float32'`, `ALLOW_TF32=False`, and `USE_FAST_ACCUM=False`; examples are `runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:38-46`, `:61-69`, `:118-127`, `:142-150`, `:165-173`, `:206-214`, `:223-231`.
- The visible 2D conv autotune block has ATen `convolution` as fastest, and Triton conv alternatives carry `ALLOW_TF32=True`: `runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:175-186`.
- Some winning rows are ATen `mm`, `addmm`, or `convolution`, not explicit Triton templates, so the log alone cannot prove every selected kernel's internal precision policy: `reviews/codex-autotune-drift-verify.md:19-25`.
- `export_t2a.py` creates the eager reference without setting precision globals: `runtime/export_t2a.py:14-46`, `runtime/export_t2a.py:76-81`. Separately, the production Python server disables both matmul and cuDNN TF32 when batching is enabled: `../src/nemotron_speech/server.py:660-666`. Do not mix these two "eager" policies without saying which reference the T1 gate uses.
- The long-stream default AOTI drift probe plateaued rather than compounded and had zero token flips over 830 chunks: `runtime/artifacts/logs/drift_probe.log:134-167`. The full-1000 shadow had 1 token divergence and WER delta `+0.0042 pp`: `runtime/artifacts/logs/full1000_shadow.log:155-162`.
- The bucket compiler currently adds only the constants-on-disk AOTI config. Any bucket autotune candidate should merge the schedule/precision configs below with those existing bucket configs: `runtime/aot_compile_buckets.py:220-230`.
- Torch 2.8 Inductor exposes the relevant config keys and env names: `max_autotune`, `max_autotune_gemm`, GEMM/conv backends, search space, and coordinate descent at `/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/torch/_inductor/config.py:409-460`, coordinate descent envs at `:496-504`, epilogue fusion at `:241-248`, `conv_1x1_as_mm` at `:617-618`, and `emulate_precision_casts` at `:669-679`. `use_mixed_mm` is deprecated/ignored in this Torch 2.8 config: `/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/torch/_inductor/config.py:342-343`.

## Ranked Candidate Configs

All candidates must run in a fresh process. Apply precision globals before loading/compiling the `ExportedProgram`. Use a distinct `TORCHINDUCTOR_CACHE_DIR` and artifact directory per candidate so cache state and package SHA remain attributable.

For finalize buckets, merge the candidate's `inductor_configs` with:

```python
{
    "aot_inductor.package_constants_in_so": False,
    "aot_inductor.package_constants_on_disk": True,
}
```

### R1a: Precision-Matched Max Autotune, No Coordinate Descent

This is the first candidate if the T1 target is the task's TF32-reduced eager policy.

```python
# Precision globals, set before torch._inductor.aoti_compile_and_package(...)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

env = {
    "TORCHINDUCTOR_CACHE_DIR": "/work/.../torchinductor_cache/r1a_precision_match_max_no_coord",
    "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "0",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS": "ATEN,TRITON",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE": "DEFAULT",
}

inductor_configs = {
    "aot_inductor.package": True,
    "max_autotune": True,
    "max_autotune_gemm": True,
    "coordinate_descent_tuning": False,
    "coordinate_descent_check_all_directions": False,
    "coordinate_descent_search_radius": 1,
    "max_autotune_gemm_backends": "ATEN,TRITON",
    "max_autotune_conv_backends": "ATEN,TRITON",
    "max_autotune_gemm_search_space": "DEFAULT",
    "epilogue_fusion": True,
    "emulate_precision_casts": False,
    "force_same_precision": True,
    "conv_1x1_as_mm": False,
}
```

Rationale: this follows the Phase-2 ladder: match eager precision first, then autotune schedule, and remove the extra coordinate-descent variable that was present in the failing run. `force_same_precision=True` is conservative when ATen/cuBLAS and Triton are both in the GEMM menu; it avoids Triton using TF32 in shapes where cuBLAS would not. The expected speed is below the aggressive max+coord run but above autotune-off if GEMM or layout choices matter. Expected drift should fall near the default AOTI floor if the precision hypothesis is correct; a `cache_t` jump toward `10.27` means the precision match is wrong or a tuned schedule is still numerically unsafe.

### R1b: Raw-Export Eager Precision Variant

Use this as R1 if the preflight in the exact reference environment reports raw Torch defaults, which is plausible because `export_t2a.py` does not set precision globals.

```python
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = True

env = {
    "TORCHINDUCTOR_CACHE_DIR": "/work/.../torchinductor_cache/r1b_raw_export_precision_max_no_coord",
    "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "0",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS": "ATEN,TRITON",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE": "DEFAULT",
}

inductor_configs = {
    "aot_inductor.package": True,
    "max_autotune": True,
    "max_autotune_gemm": True,
    "coordinate_descent_tuning": False,
    "coordinate_descent_check_all_directions": False,
    "coordinate_descent_search_radius": 1,
    "max_autotune_gemm_backends": "ATEN,TRITON",
    "max_autotune_conv_backends": "ATEN,TRITON",
    "max_autotune_gemm_search_space": "DEFAULT",
    "epilogue_fusion": True,
    "emulate_precision_casts": False,
    "force_same_precision": True,
    "conv_1x1_as_mm": False,
}
```

Rationale: this matches the likely raw NeMo export process better than the production batch server policy. It keeps cuDNN/conv TF32 allowed while leaving matmul TF32 disabled. If R1a and R1b disagree strongly on single-step drift, the better match is the one whose no-autotune control reproduces the `1.66e-2` floor.

### R2: Precision-Matched GEMM-Only Autotune

```python
# Use the same precision globals as the winning R1 precision variant.

env = {
    "TORCHINDUCTOR_CACHE_DIR": "/work/.../torchinductor_cache/r2_gemm_only",
    "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": "1",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "0",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE": "DEFAULT",
}

inductor_configs = {
    "aot_inductor.package": True,
    "max_autotune": False,
    "max_autotune_gemm": True,
    "coordinate_descent_tuning": False,
    "max_autotune_gemm_backends": "ATEN,TRITON",
    "max_autotune_gemm_search_space": "DEFAULT",
    "epilogue_fusion": True,
    "emulate_precision_casts": False,
    "force_same_precision": True,
    "conv_1x1_as_mm": False,
}
```

Rationale: if full `max_autotune` perturbs conv or other non-GEMM choices, this keeps the optimization focused on the dense GEMM/bmm bottleneck. Expected speed is lower than R1 but the T1 risk is lower. This is the best fallback if R1 has a `cache_t` jump or token flips.

### R3: ATen-Only Autotune Diagnostic

```python
# Use the same precision globals as the winning R1 precision variant.

env = {
    "TORCHINDUCTOR_CACHE_DIR": "/work/.../torchinductor_cache/r3_aten_only",
    "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "0",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN",
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS": "ATEN",
}

inductor_configs = {
    "aot_inductor.package": True,
    "max_autotune": True,
    "max_autotune_gemm": True,
    "coordinate_descent_tuning": False,
    "max_autotune_gemm_backends": "ATEN",
    "max_autotune_conv_backends": "ATEN",
    "epilogue_fusion": True,
    "emulate_precision_casts": False,
    "force_same_precision": True,
    "conv_1x1_as_mm": False,
}
```

Rationale: this should be the closest numerical policy to eager/ATen while still letting Inductor benchmark ATen choices where applicable. It is unlikely to be the fastest, but it isolates whether Triton template selection is responsible for drift. If R3 passes T1 and R2 fails, the risk is probably Triton reduction scheduling rather than precision globals.

### R4: CUTLASS Targeted GEMM Experiment

```python
# Use the same precision globals as the winning R1 precision variant.

env = {
    "TORCHINDUCTOR_CACHE_DIR": "/work/.../torchinductor_cache/r4_cutlass_gemm",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": "1",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "0",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON,CUTLASS",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE": "DEFAULT",
    "TORCHINDUCTOR_CUTLASS_ENABLED_OPS": "mm,addmm,bmm",
}

inductor_configs = {
    "aot_inductor.package": True,
    "max_autotune": False,
    "max_autotune_gemm": True,
    "coordinate_descent_tuning": False,
    "max_autotune_gemm_backends": "ATEN,TRITON,CUTLASS",
    "max_autotune_gemm_search_space": "DEFAULT",
    "cuda.cutlass_enabled_ops": "mm,addmm,bmm",
    "cuda.cutlass_epilogue_fusion_enabled": False,
    "cuda.cutlass_max_profiling_configs": 64,
    "epilogue_fusion": True,
    "emulate_precision_casts": False,
    "force_same_precision": True,
    "conv_1x1_as_mm": False,
}
```

Rationale: CUTLASS is in the Torch 2.8 backend menu, but it is compile-costly and adds a third implementation family. Try this only after R1/R2 establish a T1-passing precision policy. Expected speed could improve on larger GEMMs, but many logged steady shapes are small or skinny and already favor ATen, so this is not the first dollar to spend.

### R5: Aggressive Max Plus Coordinate Descent

```python
# Use the same precision globals as the winning R1 precision variant.

env = {
    "TORCHINDUCTOR_CACHE_DIR": "/work/.../torchinductor_cache/r5_max_coord",
    "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "1",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS": "ATEN,TRITON",
}

inductor_configs = {
    "aot_inductor.package": True,
    "max_autotune": True,
    "max_autotune_gemm": True,
    "coordinate_descent_tuning": True,
    "coordinate_descent_check_all_directions": False,
    "coordinate_descent_search_radius": 1,
    "max_autotune_gemm_backends": "ATEN,TRITON",
    "max_autotune_conv_backends": "ATEN,TRITON",
    "epilogue_fusion": True,
    "emulate_precision_casts": False,
    "force_same_precision": True,
    "conv_1x1_as_mm": False,
}
```

Rationale: this is the closest controlled retest of the failed `max+coord` run after precision matching. Rank it low because the existing `max+coord` artifact hit `cache_t=10.27`. Only run it if R1 passes T1 and the speed gap to the floor is still too small.

### R6: Epilogue-Fusion-Off Diagnostic

```python
# Use the same precision globals as the winning R1 precision variant.

env = {
    "TORCHINDUCTOR_CACHE_DIR": "/work/.../torchinductor_cache/r6_no_epilogue",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM": "1",
    "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING": "0",
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS": "ATEN,TRITON",
}

inductor_configs = {
    "aot_inductor.package": True,
    "max_autotune": False,
    "max_autotune_gemm": True,
    "coordinate_descent_tuning": False,
    "max_autotune_gemm_backends": "ATEN,TRITON",
    "epilogue_fusion": False,
    "emulate_precision_casts": False,
    "force_same_precision": True,
    "conv_1x1_as_mm": False,
}
```

Rationale: not a performance candidate. Use it only to test whether fused epilogues are the remaining T1 problem after precision and backend scope are controlled. Expect slower kernels.

## Knobs To Skip Or Treat As Diagnostics

- Do not use `torch.set_float32_matmul_precision("highest")` plus `torch.backends.cuda.matmul.allow_tf32=False` and `torch.backends.cudnn.allow_tf32=False` as a performance candidate. The matrix already associates that fp32-precise policy with `cache_t` around `10.27`: `runtime/artifacts/logs/knob_matrix.log:5-12`.
- Do not turn on `emulate_precision_casts` for the first pass. It is intended for preserving lower-precision cast boundaries, and the tested `emulate_prec_casts` case paired with fp32 policy landed at the same bad `10.27`: `runtime/artifacts/logs/knob_matrix.log:9-12`.
- Do not treat `deterministic_all` as an optimization candidate. It did not recover byte-exactness and hit the same `3.54e-02` cache_t drift as the no-fusion fp32 case, with CuBLAS determinism warnings: `runtime/artifacts/logs/knob_matrix.log:17-40`, `runtime/artifacts/logs/knob_matrix.log:1791-1793`.
- Do not spend early cycles on `use_mixed_mm`; in this Torch 2.8 config it is deprecated and ignored.
- Do not start with `max_autotune_gemm_search_space="EXHAUSTIVE"` or broad CUTLASS instantiation. Use those only after a T1-passing precision policy has a measured speed gap worth chasing.

## Search Strategy

### 1. Prove The Eager Precision Policy First

Before compiling a candidate, print and record these values in the same kind of process that generates the eager reference and in the compile child:

```python
print("matmul_precision", torch.get_float32_matmul_precision())
print("cuda.matmul.allow_tf32", torch.backends.cuda.matmul.allow_tf32)
print("cudnn.allow_tf32", torch.backends.cudnn.allow_tf32)
```

If the reference is `t2a_io.pt` from `export_t2a.py`, raw export eager is the immediate target. If the reference is production batch server behavior, then `server.py`'s `allow_tf32=False` policy is the target. Do not compare an artifact compiled for raw export eager against a server-batch oracle and call the result a precision finding.

Run a no-autotune precision-only control for the selected precision policy. It should reproduce the default floor scale before schedule autotune is allowed. If the precision-only control is already near `10.27`, the policy is wrong and no schedule search should be trusted.

### 2. Tune Schedule In This Order

1. R1 precision-matched full `max_autotune`, no coordinate descent, ATEN/TRITON only.
2. R2 precision-matched GEMM-only autotune if R1 drifts or flips tokens.
3. R3 ATen-only if R2 fails T1 and you need to isolate Triton template risk.
4. R4 CUTLASS only after a T1-passing R1/R2 shows meaningful but insufficient speed.
5. R5 coordinate descent only after a precision-matched non-coordinate candidate passes T1 and the speed shortfall is large enough to justify the known drift risk.
6. R6 epilogue off only as a diagnostic if all speed candidates fail T1 with floor-like single-step drift.

This is not brute force: each rung changes one risk class - precision, autotune scope, backend family, coordinate search, or fusion.

### 3. T1 Filter Protocol

Every candidate gets a manifest with package SHA, torch/CUDA/driver/Triton, precision globals, `inductor_configs`, env, cache cold/warm state, compile log path, and artifact path. T1 attaches to that exact package SHA; a later recompile with the same apparent configs is a new candidate.

Use these gates in order:

1. **Single-step sanity:** run the `t2a_io.pt` check and record `enc_out`, `cache_ch`, and `cache_t` max_abs. This is a triage signal, not the ship gate. A jump toward `10.27` should normally stop the candidate before full corpus work.
2. **Recurrent drift probe:** run the E.1 style independent-cache stream. Require no token divergence and no clear growth trend in `cache_t`; compare against the default plateau result, `max=1.154e-01`, mean `1.573e-02`, grow ratio `0.93x`, zero flips over 830 chunks.
3. **E.2 shadow on the exact package SHA:** run the full corpus eager-vs-AOTI shadow with the candidate package. The objective for this task is 0 token divergences. WER must be within the Phase-2 bound, `WER_native <= max(WER_py + 0.5pp, 1.10 * WER_py)`: `PHASE2-PLAN.md:57`.
4. **Session/finalize token/event gate:** after compiling matching finalize buckets, run the production-shaped token/event/WER oracle, including finalize and collector fields. Warm all loaded buckets first; the 234 ms finalize finding was cold-start contamination, not a reason to judge autotune speed: `reviews/phase2-finalize-234ms-FOLDED.md:1-24`, `reviews/phase2-finalize-234ms-FOLDED.md:40-54`.
5. **Density benchmark:** only T1-passing packages enter speed comparison. Use the same EPs, model, target GPU, torch/CUDA/driver, topology, warmup, bounded workload, and repeat count for on/off. Report absolute streams/box delta and percent, not just each artifact's multiplier over its own N=1.

### 4. Per-Knob Attribution

Attribute failures and wins with paired comparisons:

- **Precision:** R1a versus R1b, plus each one's no-autotune control. If single-step `cache_t` tracks `10.27`, the precision policy is wrong.
- **Autotune scope:** R1 versus R2. If R1 fails and R2 passes, non-GEMM autotune, likely conv or pointwise/reduction selection, is the risk.
- **Backend family:** R2 versus R3. If R3 passes and R2 fails, Triton templates are implicated. If both pass but R2 is faster, keep R2.
- **CUTLASS:** R4 versus R2. Only keep R4 if it passes the same T1 gate and produces a speed win larger than measurement noise.
- **Coordinate descent:** R5 versus R1. Given the existing `max+coord` `10.27` result, coordinate descent must buy a large extra speed win to justify the risk.
- **Fusion:** R6 versus R2. Use only to explain drift, not as the first deployable speed config.

For each compile, parse the autotune log for the winning row per `AUTOTUNE` block, the backend family, `ALLOW_TF32`, `ACC_TYPE`, `USE_FAST_ACCUM`, and timing. Then join that with single-step drift, recurrent drift, token divergence count, WER delta, and density result.

### 5. Objective And Stop Criteria

The objective is:

```text
maximize warm steady-density throughput and warmed finalize performance
subject to:
  exact package SHA passes 0 token divergences on the recurrent T1 shadow,
  WER is within bound,
  production-shaped token/event/finalize oracle has 0 mismatches,
  benchmark repeatability is within the Phase-2 stability bar.
```

Stop and keep the autotune-off floor when any of these is true:

- The precision-matched R1 and R2 candidates fail T1.
- The best T1-passing autotune candidate does not beat the warmed autotune-off floor by more than run-to-run noise. As a practical bar, require at least a stable 10 percent density win or an absolute stream/box gain that changes the Step-1 pass/fail outcome.
- The warmed autotune-off floor already clears the relevant density/tail gate and the autotune win is within the repeat-run CV budget. The plan already warns that if the warmed floor is near the gate, marginal GEMM autotune speed may not be worth the T1 battle: `PHASE2-PLAN.md:90-96`.
- The candidate only improves cold finalize p95. Autotune does not fix the one-time bucket/module cold-start; warmup or `CUDA_MODULE_LOADING=EAGER` is the fix path there, not schedule tuning.

## Single Best First Candidate

Compile **R1a** first if the team is accepting the current "eager is TF32-reduced" mechanism. If the mandatory preflight shows the actual raw `t2a_io.pt` eager policy is `matmul_precision=highest`, `cuda.matmul.allow_tf32=False`, and `cudnn.allow_tf32=True`, compile **R1b** first instead. In both cases, keep coordinate descent off and do not include CUTLASS on the first compile.

Autotune is worth a bounded attempt because the steady encoder is the density bottleneck, but the evidence does not justify an open-ended tuning campaign. The current log shows ATen already winning many small/skinnier GEMMs and the visible conv, while the aggressive candidate hit the exact bad `cache_t` scale. The likely rational path is R1, then R2 if needed, then stop unless the measured warm floor is clearly below the gate and a T1-passing candidate gives a stable, material throughput gain.
