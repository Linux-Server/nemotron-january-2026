# ~1s model-batch stall — COMBINED synthesis (Codex code-path + Claude subagent GPU-systems)

Two independent investigations (different angles) CONVERGED on the same root cause.

## ROOT CAUSE (both agree, high confidence): the EAGER RNNT decode's per-frame stream syncs
The server runs `greedy_batch` with **`use_cuda_graph_decoder=False`** (server.py:1463-1474). That routes decode to
NeMo's label-looping `torch_impl`, whose core is host-controlled `while active_mask.any():` / inner
`while advance_mask.any():` (rnnt_label_looping.py:356-409). **Each `.any()` used as a Python `while` condition forces
a D2H copy + `cudaStreamSynchronize` on the lane stream** — ~16+/batch — feeding the documented launch storm
(~1376 cudaLaunchKernel, ~20 syncs, ~250 copies, ~7ms real GPU math per call; finalize-telemetry.md). The encoder is
a single graph replay with **0 fallbacks over 51k replays** (exonerated). The decode is the ONLY dynamic,
high-launch, host-controlled part — so a steady batch can only park a CPU thread for ~1s inside a decode sync (or the
immediately-following scatter clone, queued behind the same stream). `model_batch_ms` includes the forced post-model
lane sync (server.py:8329), so it MEASURES this wait.

## AMPLIFIER (both agree): 2 lane streams + occasional launch-storm injectors
`NEMOTRON_MODEL_LANES=2` runs two lane models on two CUDA streams, each blocking on `stream.synchronize()`
(server.py:3175-3177). Concurrent decode storms + the **per-session eager warmup** (1000x, a non-steady-geometry
launch storm under the `_scheduler_exclusive_model_path` barrier that drains both lanes — server.py:3620-3667,
3213-3233) + finalize-graph replays all share ONE device (no MPS in the single-proc bench). When the device briefly
saturates, a decode `.any()` sync on one lane blocks until its stream drains behind everything queued -> ~1s at
conc-10, catastrophic 18-26s at conc-24. `scatter_cache_row` is the next eager op the SIGUSR1 dump happened to catch
(batch_primitives.py:90-97 launches GPU clones) -> a SYMPTOM, not the root.

## RULED OUT (both): allocator (num_alloc_retries=0/51k), GC (max gen-2 ~300ms, not coincident), lazy encoder
capture (0 fallbacks), cuFFT (constant plan), cuDNN benchmark (never set), MPS (absent in the reproducing bench).

## DECISIVE CONFIRM (cheapest first — disambiguates "what supplies the backlog")
1. **`py-spy dump --native` on a wedged lane thread during a live stall** — the one missing measurement: a C frame in
   `cudaStreamSynchronize` = waiting on backlog; in `cudaLaunchKernel` = launch-queue-full. (faulthandler's
   Python-only frame can't distinguish.)
2. **Env A/B matrix** (each one env var, binary outcome): `NEMOTRON_MODEL_LANES=1` (cross-lane contention?),
   `NEMOTRON_WARMUP_MS=0` (warmup trigger?), a decoder-graph-on TEST (eager decode?), `NEMOTRON_ENCODER_CUDAGRAPH=0`
   (exonerate encoder replay further).
3. **Codex per-op instrumentation** (NEMOTRON_STEADY_BATCH_PROFILE): split encoder-replay vs RNNT-decode vs
   post-model-sync vs scatter via wall+CUDA-event timers around server.py:8223/8243/8329/8399 + a profiling wrapper
   on `encoder.cache_aware_stream_step` vs `decoding.rnnt_decoder_predictions_tensor` -> quantitative attribution.

## OPTIMIZATIONS (ranked; combined)
1. **Graph the RNNT decode** (`use_cuda_graph_decoder=True` / NeMo FULL_GRAPH conditional-node mode) — removes the
   per-frame syncs (the MECHANISM); the decode analog of the finalize-encoder-graph win. HIGHEST payoff. **HIGH risk:**
   currently asserted-OFF for Blackwell-safety (`_assert_batch_decoder_blackwell_safe`, server.py:976-998). BUT the
   DEPLOY target is Ada (L4/L40S), which may support the conditional-node decode graph even though the local Blackwell
   5090 doesn't -> a dedicated byte-exact + Ada-support validation project (mirror the finalize-graph canary). Flag
   `NEMOTRON_RNNT_DECODER_CUDAGRAPH=1`, fail-closed to eager.
2. **Admission control / per-proc maxconn** (cap sessions/ready-backlog -> 503/shed) — kills the conc-24 catastrophic
   wedge. LOW risk, byte-exact. The pragmatic ship while #1 is validated.
3. **lanes=1 / decode-serialize / adaptive lanes** for the P99 tail (trade P50/P95 throughput for P99 stability).
   LOW byte-exact risk.
4. **Per-session warmup off the exclusive barrier / onto a captured bucket** — removes the 1000x backlog trigger.
   MEDIUM (prove the barrier isn't load-bearing; keep warm200 byte-exact).
5. **Sync reduction**: relax the redundant steady syncs (entry/after-preproc/after-model/after-scatter/lane-exit ->
   one completion sync), pinned-H2D non_blocking preproc. MEDIUM (stream-lifetime asserts + byte-exact canary).
6. **B>8 encoder graph coverage** (CUDAGRAPH_MAX_B=16 + cap BATCH_MAX_SIZE) — conc-24 robustness. LOW risk.
7. scatter/cache buffer pooling (fast-path B=1; copy into per-session buffers vs clone). MEDIUM (aliasing).

## Recommendation
The headline finding: **the decode is still eager and its per-frame syncs are the ~1s stall root — graphing the
decode is the next big lever (the decode analog of the finalize-encoder-graph win).** Next step = the cheap CONFIRM
matrix (py-spy-native + lanes=1/warmup=0/decoder-graph-test) to lock the root + size the decode-graph payoff BEFORE
the validation investment; ship admission control now for the conc-24 robustness. Full detail: codex-stall-analysis.md.
