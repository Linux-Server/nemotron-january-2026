# Autotune Cache Drift Verification

## Scope

This review checks the proposed explanation for why the native AOTI steady encoder saw `cache_t max_abs=10.27` with autotune enabled versus `1.66e-2` with the baseline AOTI compile. It also traces the Python reference stack that existed before the native-runtime work to see whether its design corroborates the same numeric-sensitivity thesis.

## Claim 1: reduction/accumulation order, not lower precision

**Verdict: refined.** The autotune evidence supports "changed Inductor tuning/scheduling" and supports fp32 accumulation/no TF32 for the logged Triton matmul/bmm candidates, but the blanket "not precision" and "selected kernels keep `ALLOW_TF32=False`" wording is too strong.

Evidence:

- The steady autotune manifest records only two explicit Inductor tuning knobs: `TORCHINDUCTOR_MAX_AUTOTUNE=1` and `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1`, with matching `inductor_configs` of `max_autotune=true` and `coordinate_descent_tuning=true`: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/compile_steady_manifest.json:20-29`.
- The autotune compile log confirms the run used `{'max_autotune': True, 'coordinate_descent_tuning': True}` and warns that TF32 tensor cores exist but are not enabled for matmul: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:16-23`.
- Representative logged matmul/bmm autotune candidates use `ACC_TYPE='tl.float32'`, `ALLOW_TF32=False`, and `USE_FAST_ACCUM=False`: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:38-46`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:61-69`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:118-127`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:142-150`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:165-173`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:206-214`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:223-230`.
- The baseline AOTI compiler script does not set TF32, matmul precision, or Inductor tuning flags. It simply calls `torch._inductor.aoti_compile_and_package(...)`: `proj-2026-05-24-from-scratch-runtime/runtime/aot_compile.py:41-44`. Its postcompile comparison reports the non-autotuned AOTI drift against eager: `proj-2026-05-24-from-scratch-runtime/runtime/aot_compile.py:49-64`.
- The baseline AOTI log records `cache_t` drift of `1.663e-02`: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/logs/aot_compile.log:26-31`.

Important refinements:

- Not every autotuned kernel in the log is `ALLOW_TF32=False`. The convolution autotune entries show `ALLOW_TF32=True`: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:175-185`. So the precise claim should be scoped to the logged matmul/bmm Triton candidates, not all selected kernels.
- Some "best" rows are ATen `mm`/`addmm` baselines rather than explicit Triton configs, so the log does not let us prove every selected reduction kernel's accumulator and TF32 flags from the Triton config fields alone: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:37`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:60`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:89`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/logs/steady_autotune_compile_20260526T182709Z.log:119`.
- The earlier knob matrix shows that precision policy is part of the broader numeric story: forcing `fp32_highest` or `emulate_precision_casts` made `cache_t` drift jump to about `1.027e+01`, while the default was `1.663e-02`: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/logs/knob_matrix.log:1-16`, `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/logs/knob_matrix.log:1792-1793`. The written finding says forcing precise fp32 made drift roughly 600x worse and implies the eager reference likely used TF32/reduced-precision matmul somewhere: `proj-2026-05-24-from-scratch-runtime/runtime/0.2b-aoti-findings.md:109-128`.

So: for this autotune-on run, the recorded differentiator from the baseline script is `max_autotune + coordinate_descent_tuning`, not an explicit precision flag. But the stronger statement "not precision" should be softened to "not explained by a lower-precision Triton matmul accumulator in the logged matmul candidates; the global eager-vs-AOTI precision story is more nuanced."

## Claim 2: cache_t dynamic range amplifies drift

**Verdict: confirmed for the steady fixture; representativeness is unproven.**

Evidence:

- The `t2a_io.pt` fixture is generated from `datasets.load_dataset(...)[1]`: `proj-2026-05-24-from-scratch-runtime/runtime/export_t2a.py:23-25`.
- The fixture writer says it saves real steady-chunk inputs and eager reference outputs, using chunk 1 geometry as a representative steady step: `proj-2026-05-24-from-scratch-runtime/runtime/export_t2a.py:76-81`.
- Inspecting `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/t2a_io.pt` shows the eager output `cache_t` (`out[3]`) has shape `[24, 1, 1024, 8]`, `abs.mean=0.388613313`, `abs.max=54.3406601`, `min=-41.4834366`, and `max=54.3406601`. The input `clt` has the same shape, `abs.mean=0.235958502`, and `abs.max=54.3406601`.
- The autotune manifest reports the same `cache_t` output shape and the large autotune drift: `shape=[24, 1, 1024, 8]`, `max_abs=10.26782512664795`: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/compile_steady_manifest.json:79-94`.

The quoted `abs.mean 0.39` and `abs.max 54.3` are therefore accurate for the output cache tensor in the steady fixture. The ratio is about 140x (`54.34 / 0.3886`). What is not proven by the evidence is that this dynamic range is corpus-representative. The fixture is a real chunk and intentionally steady-shaped, but it is still one dataset item and one chunk.

## Claim 3: recurrence makes it bite

**Verdict: recurrence confirmed; systematic compounding/token-flip wording refuted or at least overstated.**

Evidence that `cache_t` is recurrent:

- `ASRSession` owns `cache_last_channel`, `cache_last_time`, and `cache_last_channel_len` as persistent streaming state: `src/nemotron_speech/server.py:450-473`.
- The single-session streaming path feeds the current session caches into `_conformer_stream_step` and writes returned caches back to the same session: `src/nemotron_speech/server.py:9669-9687`.
- The batched scheduler similarly passes batched `cache_last_channel`, `cache_last_time`, and `cache_last_channel_len` into `_conformer_stream_step`: `src/nemotron_speech/server.py:9476-9494`, then scatters returned row caches and assigns them back to each session: `src/nemotron_speech/server.py:9500-9508`, `src/nemotron_speech/server.py:9555-9559`.
- Finalization also feeds the session caches into the encoder and replaces them with the returned caches: `src/nemotron_speech/server.py:9981-9999`.
- The server wrapper ultimately calls `model.conformer_stream_step(...)`: `src/nemotron_speech/server.py:3701-3734`, with eager execution at `src/nemotron_speech/server.py:3793-3795` and the optional compile wrapper cloning cache outputs at `src/nemotron_speech/server.py:3824-3830`.
- NeMo's `conformer_stream_step` docstring says `cache_last_channel`, `cache_last_time`, and `cache_last_channel_len` are cache-state inputs and returns "next cache tensor to be used for next streaming step": `/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/mixins/mixins.py:592-631`.
- That method passes the caches to `self.encoder.cache_aware_stream_step(...)`: `/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/mixins/mixins.py:646-660`, and returns the next caches: `/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/mixins/mixins.py:715-727`.
- The NeMo encoder iterates layers, passes per-layer `cache_last_channel_cur` and `cache_last_time_cur` into each layer, collects `cache_last_time_next`, stacks the next caches, and returns them: `/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:669-697`, `/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conformer_encoder.py:750-758`.

Evidence that the "propagates -> flips tokens over a stream" part is overstated:

- The dedicated AOTI drift probe was explicitly written to test the dangerous recurrent-cache failure mode: `proj-2026-05-24-from-scratch-runtime/runtime/aoti_drift_probe.py:1-9`.
- It threads independent eager and AOTI recurrent caches across chunks and records per-chunk cache drift, token equality, and margin: `proj-2026-05-24-from-scratch-runtime/runtime/aoti_drift_probe.py:90-118`.
- The probe output reports bounded drift, not runaway compounding: `cache_t max=1.154e-01`, `mean=1.573e-02`, first-half mean `1.628e-02`, second-half mean `1.517e-02`, token divergence `0 / 830`, and final verdict `BOUNDED/PLATEAU`: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/logs/drift_probe.log:134-167`.
- The written finding makes the same point: recurrent drift was real but did not compound and tokens were stable in that probe: `proj-2026-05-24-from-scratch-runtime/runtime/0.2b-aoti-findings.md:133-145`.
- A larger full-1000 shadow run found one token divergence out of 1000 utterances, with tiny WER impact: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/logs/full1000_shadow.log:155-162`. The written finding attributes that single miss to a near-tie tail decision: `proj-2026-05-24-from-scratch-runtime/runtime/0.2b-aoti-findings.md:147-165`.

So the recurrent structure is real and important, but the available measured evidence does not support an unqualified "error propagates and flips tokens over a stream" claim. A safer statement is: recurrent cache state creates a plausible amplification/continuation path, but in the measured non-autotuned AOTI probes it plateaued and only caused rare near-tie token risk.

## Claim 4: not a bug, not precision; autotune reorders reductions; mild autotune may not help

**Verdict: refined.** The "kernel-selection/reduction-order risk" explanation is directionally correct, but "not precision" and "milder autotune may not help much" are overstated.

Evidence:

- The autotune run's recorded compile-time difference from the baseline is the Inductor autotune pair, not an explicit precision setting: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts_at_sm120/compile_steady_manifest.json:20-29`, `proj-2026-05-24-from-scratch-runtime/runtime/aot_compile.py:41-44`.
- The phase-2 plan itself described the intended tuning ladder as `max_autotune` without coordinate descent, then selected Triton knobs, then cache-sensitive exclusions: `proj-2026-05-24-from-scratch-runtime/PHASE2-PLAN.md:74-83`. That means "milder autotune may not help much" was a hypothesis, not a result.
- The earlier AOTI findings do not show evidence of a C++ runtime plumbing bug. They say Python-packaged AOTI and the C++ runtime matched exactly, narrowing the drift source to AOTI/Inductor versus eager: `proj-2026-05-24-from-scratch-runtime/runtime/0.2b-aoti-findings.md:167-180`.
- However, precision cannot be dismissed globally. The knob matrix and findings show that changing matmul precision policy can make the same cache drift much worse: `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/logs/knob_matrix.log:1-16`, `proj-2026-05-24-from-scratch-runtime/runtime/0.2b-aoti-findings.md:109-128`.

The corrected claim should be: there is no evidence here of a native runtime bug; the autotune-on regression is most consistent with Inductor tuning changing kernel choices/reduction schedules on a cache-sensitive encoder. But precision policy is part of the surrounding numeric landscape, and the value of milder autotune levels needs to be measured rather than assumed away.

## Python Reference Stack Findings

### TF32 policy

The production Python server disables TF32 when batching is enabled:

- `torch.backends.cuda.matmul.allow_tf32 = False`: `src/nemotron_speech/server.py:661`.
- `torch.backends.cudnn.allow_tf32 = False`: `src/nemotron_speech/server.py:662`.

The stated reason in the optimization docs was state-faithful batching, not native AOTI. The plan says TF32-on introduced batched-matmul reduction-order cache drift around `0.03`, while TF32-off reduced it to around `1e-4`, so batching should set `allow_tf32=False`: `proj-2026-05-21-0410/PLAN.md:279-286`. The same rationale appears in the batch-state test comments: `proj-2026-05-21-0410/test_batch_state.py:713-723`, with a gate note that TF32 drift failed the state allclose check while fp32 passed: `proj-2026-05-21-0410/test_batch_state.py:803-806`. The summary repeats the same conclusion: `proj-2026-05-21-0410/SUMMARY.md:93-95`.

Git history corroborates that this was an intentional production choice. `git blame` attributes `src/nemotron_speech/server.py:660-666` to commit `c5927b34`, whose message includes: "TF32 OFF when batching (Step-5 finding): state-faithful batched<->separate (~1e-4 vs ~0.03 drift)."

This directly supports the broader thesis that encoder cache numerics are sensitive to matmul/reduction behavior. It does not say the reason was the exact native AOTI autotune failure.

### torch.compile policy

The server gates encoder compile behind an environment variable and leaves it off by default:

- `encoder_compile_requested = os.environ.get("NEMOTRON_ENCODER_COMPILE", "") == "1"`: `src/nemotron_speech/server.py:787-790`.
- If not requested, `_configure_encoder_compile` logs that encoder compile is disabled: `src/nemotron_speech/server.py:1694-1702`.
- If requested, it compiles `model.encoder.cache_aware_stream_step` with `mode="reduce-overhead"`: `src/nemotron_speech/server.py:1723-1726`.

The historical docs show compile was explored and wired behind the flag: `proj-2026-05-21-0410/IMPLEMENTATION-STATUS.md:32-39`, and local validation measured compile-only and compile+batch variants: `proj-2026-05-21-0410/local-validation.md:49-63`. But the clearest reason found for retiring this path was not numeric divergence; it was warmup/cold-start behavior. The CUDA-graph plan says `torch.compile(reduce-overhead)` had "minutes-and-hung warmup" on Modal T4/L4 and was replaced by a manual CUDA graph with no Inductor codegen: `proj-2026-05-21-1959-cudagraph/PLAN.md:5-13`. It also calls the compile path dead and superseded: `proj-2026-05-21-1959-cudagraph/PLAN.md:31-39`.

So: I found evidence that compile was off by default and later replaced, but not evidence that the Python reference stack disabled `torch.compile` specifically because streaming encoder compile/autotune caused token drift or divergence. The direct drift evidence appears later in the native/AOTI investigation, not in the original Python compile gate rationale.

### CUDA graph finalize path

The finalize CUDA graph preserves eager kernel choices rather than invoking Inductor/autotune:

- The module describes itself as graphing `model.encoder.cache_aware_stream_step`, with decode still eager, exact batch capture, no padding in the manager, and static buffers: `src/nemotron_speech/cudagraph_encoder.py:1-17`.
- The graph key includes exact `batch_size`, `time_steps`, `drop_extra`, and `keep_all_outputs`: `src/nemotron_speech/cudagraph_encoder.py:46-53`.
- Each bucket stores one captured graph and static buffers for an exact encoder bucket: `src/nemotron_speech/cudagraph_encoder.py:157-180`.
- The captured call invokes the existing encoder wrapper with static buffers: `src/nemotron_speech/cudagraph_encoder.py:231-241`.
- Warmup and capture use `torch.cuda.graph(self.graph)` around that call, not `torch.compile` or AOTI: `src/nemotron_speech/cudagraph_encoder.py:246-259`.
- Replay copies inputs into the static buffers and calls `self.graph.replay()`: `src/nemotron_speech/cudagraph_encoder.py:286-295`.
- Finalize lookup is keyed by the exact graph key, and mismatches fall back rather than recompile: `src/nemotron_speech/cudagraph_encoder.py:591-632`.

The CUDA-graph plan explicitly contrasts this with Inductor: manual graph replaced `torch.compile`, was byte-exact for text and state, and used no Inductor codegen: `proj-2026-05-21-1959-cudagraph/PLAN.md:5-13`. Its correctness gates required byte-exactness and no padding in the initial graph manager: `proj-2026-05-21-1959-cudagraph/PLAN.md:42-53`.

This is strong support for the contrast in the explanation: the Python performance path collapsed launch overhead while preserving eager reduction order, whereas AOTI/autotune changes compiled kernel choices.

### Other reference-stack numeric findings

- The T2a export path was built around byte-exact export-vs-eager validation across a 20-chunk stream: `proj-2026-05-24-from-scratch-runtime/runtime/export_t2a.py:1-7`, `proj-2026-05-24-from-scratch-runtime/runtime/export_t2a.py:54-73`.
- The T2a findings say Python export was byte-exact and separated Python/export correctness from the unproven C++ runtime path: `proj-2026-05-24-from-scratch-runtime/runtime/T2a-findings.md:1-6`, `proj-2026-05-24-from-scratch-runtime/runtime/T2a-findings.md:21-36`.
- A prior review correctly flagged recurrent `cache_t` drift as a risk requiring long-stream probes: `proj-2026-05-24-from-scratch-runtime/reviews/codex-actionD-review.md:9-13`. The later E.1/E.2 probes refined that risk: recurrent drift existed but bounded and rarely affected tokens.

## Bottom Line

The explanation is directionally correct: enabling Inductor autotune changed the compiled kernel/tuning landscape, and the high-dynamic-range recurrent `cache_t` tensor is the output where numeric differences are most dangerous. The Python reference stack backs up the general thesis that encoder cache numerics are sensitive: it disabled TF32 for state-faithful batching, kept encoder compile behind an opt-in flag, and chose manual CUDA graph capture for the key Python performance win because it preserved eager execution order and byte-exact state.

But the original explanation needs these corrections:

- "Not precision" is overstated. The autotune log supports fp32 accumulation/no TF32 for many Triton matmul/bmm candidates, but convolution candidates show `ALLOW_TF32=True`, some winning rows are ATen baselines without those Triton fields, and earlier knob-matrix evidence shows precision policy can drastically change `cache_t` drift.
- The `abs.mean 0.39 / abs.max 54.3` dynamic-range claim is confirmed for the `t2a_io.pt` steady fixture output cache, not proven representative across all inputs.
- `cache_t` is absolutely recurrent, but measured non-autotuned AOTI long-stream drift plateaued rather than compounded, with zero divergences in the 830-token probe and one divergence in the full-1000 shadow run.
- "Milder autotune may not help much" remains a hypothesis. The phase-2 plan itself proposed measuring a ladder of milder autotune and cache-sensitive exclusions.

Final assessment: the root explanation should be kept as "autotune changed compiled reduction/kernel choices on a numerically sensitive recurrent encoder cache," not as the stronger "purely not precision, inevitably propagating, and structurally hopeless under milder autotune" version.
