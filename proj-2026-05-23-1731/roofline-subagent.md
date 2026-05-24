# Roofline + in-code levers + from-scratch potential — nemotron-speech-streaming-en-0.6b

Open-ended, first-principles optimization review. Read-only investigation (no production code changed). Primary
goal: TTFT (end-of-speech → final transcript). Secondary: parallelism (streams/box).

**TL;DR**
1. The finalize encoder (the single biggest TTFT-relevant compute chunk) is **deeply memory-bandwidth-bound**
   (arithmetic intensity ≈ **5 FLOP/byte fp32**, vs ridge points 58–114) on *every* platform: it must stream all
   **609M encoder weights (2.44 GB fp32)** through the GPU to do only **~12 GFLOP** at B=1.
2. We measure **9.66 ms of GPU time** for it on L40S vs a **2.84 ms weight-streaming floor** → we run at **29% of
   peak BW**, i.e. **~3.4× above roofline**. The gap is *not* launch dispatch (it's a graph replay) — it is **24
   sequential conformer layers of small (M≈7) kernels** that individually cannot saturate BW.
3. The **production scaling wall is NOT GPU compute or BW** — it is the **single-thread asyncio scheduler + GIL +
   eager-decode host syncs**. Measured: at the maxconn=12/proc operating point, the GPU is ~46–65% utilized but
   `vad_stop_recv→process` (end-of-speech received → finalize *starts*) blows from **8 ms (3/proc) to 938 ms
   (12/proc)** — head-of-line blocking on one scheduler thread. Real in-budget capacity ≈ **20/box**, not the
   documented 48.
4. Biggest realistic wins: **(a)** a fused encoder engine (TensorRT / custom fused conformer block) to close the
   29%→~70%+ BW gap → finalize encoder **9.7 → ~3–4 ms**; **(b)** break the single-scheduler-thread ceiling
   (move GPU dispatch + decode off the event-loop thread; one dispatcher thread per lane in C++/no-GIL) to lift the
   keep-up knee and kill the `vad_recv→process` tail. Together: TTFT server-compute **~22 → ~8–10 ms** and reliable
   capacity **~20 → ~40–48/box**.

---

## 0. Verified architecture (from the checkpoint, not the brief)

Extracted `model_config.yaml` + `model_weights.ckpt` from the shipped `.nemo`. **Total 618.1M params, fp32.**

| block | params | key dims |
|---|---:|---|
| encoder (ConformerEncoder) | **609.14 M** | 24 layers, d_model=1024, 8 heads × **head_dim 128**, d_ff=4096 (macaron ×2 FFN), conv depthwise k=9, **no bias** |
| decoder/predictor (RNNTDecoder) | 7.22 M | 2-layer LSTM, hidden **640**, embed (1025,640), blank_as_pad |
| joint (RNNTJoint) | 1.72 M | enc proj (1024→640), pred proj (640→640), out (640→**1025**) |
| preprocessor | 0.03 M | **128 mel** (NOT 80 — brief said 80), n_fft 512, win 25 ms/hop 10 ms, log-mel |

- Subsampling **dw_striding, factor 8**, 3 depthwise-separable conv stages, `pre_encode.out` Linear (4352→1024)
  where 4352 = 256 ch × 17 freq (128→64→32→17 after 3 stride-2 convs).
- att_context = **[70, 1]** (3rd of 4 trained modes), `chunked_limited` → `last_channel_cache_size = 70`,
  `cache_drop_size = 0`, `lookahead_steps = 1`.
- Streaming geometry (server-computed, confirmed in records): **shift_frames=16** (160 ms chunk), pre_encode_cache=9,
  drop_extra=2, final_padding=(1+1)×16=**32 frames (320 ms)**.
- Decode = `greedy_batch`, `loop_labels=True`, `use_cuda_graph_decoder=False` (eager), max_symbols=10
  (server.py:1463-1474). Cache tensor `cache_last_channel = [24, B, 70, 1024]`.

### The actual measured finalize workload (mined `ec2-bench/lanes_ab_1624/prod_l40s_c10_lanes2.records`, 2134 records)

- **B = 1 (100% of finalizes)**, **encoder_invocations = 1** per finalize (one forward over the padded tail).
- **T_in = 43–58 mel frames (mean 50.25)** → encoder produces **encoded_shape [1,1024,5–6]** valid output frames.
- `encoder_finalize_cudagraph = replay` (100%). preproc_invocations = **3** (mostly).
- p50 timings: **encoder_cuda_event 9.66 ms** (real GPU time, CUDA-event measured), decode_wall 3.64 ms,
  preproc_wall 2.44 ms, model_wall 13.38 ms, finalize_wall 22.1 ms. lock_wait p50 0.37 / p95 21.1 ms.

The single most important measured number: **encoder_cuda_event_ms p50 = 9.66 ms** — this is *GPU* time inside the
captured graph (launch overhead already collapsed), so the roofline must explain ~9.66 ms of genuine GPU work.

---

## 1. ROOFLINE (first-principles, quantitative)

### 1a. FLOPs of the finalize encoder forward (B=1, Lq≈7 conformer query frames, Lkv = 70 cache + 7 = 77)

Per conformer layer (GEMM = 2·m·k·n):
- **2× macaron FFN**: 2 × [ (Lq·1024·4096) + (Lq·4096·1024) ]·2 = **234.9 MFLOP**
- **rel-pos self-attention** (q,k,v proj for Lq + linear_pos over Lkv + q·kᵀ content + q·posᵀ + attn·v + out proj):
  **223.5 MFLOP**
- **conv module** (pointwise 1024→2048 GLU + depthwise k=9 + pointwise 1024→1024): **44.2 MFLOP**
- **per layer ≈ 502.6 MFLOP** → **× 24 = 12.06 GFLOP**; + subsampling/pre-encode ≈ 0.14 GFLOP → **≈ 12.2 GFLOP**.

### 1b. Bytes moved

At B=1 with only ~7 query frames, **activations are tiny; the traffic is dominated by reading the weights once.**
- encoder weights: 609.14M × 4 B (fp32) = **2.436 GB**; (fp16 = 1.218 GB).
- + attention KV-cache read+write [24,1,70,1024] fp32 ×2 ≈ 13.8 MB (negligible).

### 1c. Arithmetic intensity and bound

**AI = 12.2 GFLOP / 2.45 GB ≈ 5.0 FLOP/byte (fp32)**, ≈ 10 (fp16). Ridge points (peak FLOP ÷ BW):

| platform | mem BW | FP32 / TF32-TC / FP16-TC (TFLOP) | ridge fp32 | ridge tf32 | workload AI |
|---|---:|---|---:|---:|---:|
| RTX 5090 (Blackwell GB202) | 1792 GB/s | 105 / 105 / 210 | 58.6 | 58.6 | **5.0** |
| DGX Spark GB10 (Grace-Blackwell) | **273 GB/s** | ~31 / ~31 / ~100 (BF16, measured) | 113.6 | 113.6 | **5.0** |
| L4 (Ada AD104) | 300 GB/s | 30.3 / 60 / 121 | 100 | 200 | **5.0** |
| L40S (Ada AD102) | 864 GB/s | 91.6 / 183 / 362 | 106 | 212 | **5.0** |

Workload AI (5) ≪ every ridge point (59–114) ⇒ **the finalize encoder is memory-bandwidth-bound on all four
platforms.** (Specs: L40S 864 GB/s / 91.6 FP32 / 362 FP16-dense; L4 300 / 30.3 / 121; GB10 273 GB/s unified LPDDR5x
— the lowest BW; measured BF16 ~100 TFLOP, FP8 ~207, 1 PFLOP is fp4-sparse marketing; 5090 1792 GB/s GDDR7. Sources at bottom.)

### 1d. Roofline floors vs MEASURED

Memory floor = weight-bytes / BW (the achievable lower bound when perfectly BW-bound); compute floor = FLOP / peak.

| platform | mem floor fp32 | mem floor fp16 | comp floor fp32 | **MEASURED GPU** | distance from fp32 floor |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 1.37 ms | 0.68 ms | 0.12 ms | (not measured here) | — |
| DGX Spark GB10 | **8.98 ms** | 4.49 ms | 0.39 ms | — | (worst floor — low BW) |
| L4 | 8.17 ms | 4.08 ms | 0.41 ms | — | — |
| **L40S** | **2.84 ms** | 1.42 ms | 0.13 ms | **9.66 ms** | **3.4×** |

**The L40S finalize encoder runs at ~254 GB/s effective = 29% of its 864 GB/s peak.** This is the central roofline
finding for TTFT.

**Why only 29% of peak (the gap explanation):**
- The encoder is **24 sequential layers**, each a *chain* of small kernels: 2 FFN GEMMs, 5 layernorms, q/k/v/pos/out
  GEMMs, softmax, 3 conv kernels, several elementwise. At B=1/Lq≈7 each GEMM is **M=7** — far too small to saturate
  a 568-tensor-core GPU, so each kernel runs at a small fraction of peak BW and is **latency/occupancy-bound**, not
  BW-bound.
- The CUDA graph removes *launch* overhead (≈1376 launches/finalize collapse to one replay) but **cannot remove the
  data-dependency serialization** between the 24 layers, nor make any individual small kernel saturate BW.
- LayerNorm/Swish/GLU/elementwise glue (~dozens of kernels/layer) read+write activations with near-zero arithmetic →
  pure latency, dragging the *effective* BW down.
- So 29% of peak is the expected efficiency for a **B=1, short-T, deep, un-fused** transformer encoder. To approach
  the 2.84 ms floor you must **fuse** (fewer, larger kernels) or **raise B** (amortize the weight reads across rows).

### 1e. Steady per-chunk roofline

Steady chunk: B=1, Lq=2 frames (160 ms / 8 / ...), Lkv=72. ≈ **6.0 GFLOP**, but **bytes are unchanged (~2.44 GB
weights)** ⇒ still memory-bound, **same ~2.84 ms L40S weight-stream floor regardless of Lq.** 

Real-time budget = 160 ms/chunk. If dispatch were free and the encoder hit its floor, one stream would consume
~2.84 ms GPU per 160 ms ⇒ the **GPU alone could sustain ~56 streams**. **We sustain a keep-up knee of ~16/proc and a
reliable ~6.7/proc** — i.e. **3.5–8× below even the un-fused GPU ceiling.** That entire gap is CPU/dispatch/scheduler,
confirming the "launch-dispatch / single-thread-CPU bound" memory. (And it is exactly why **batching is the
self-host lever** — at B=N the 2.44 GB weight read amortizes across N rows, AI rises ~N×, the encoder moves toward
compute-bound and effective BW rises; on a fast CPU the knee climbs, on a slow/contended CPU batches never fill.)

### 1f. Decode roofline

Predictor (2-layer LSTM h=640) + joint, ~6 frames × ~7 tokens ≈ **0.11 GFLOP**, weights 8.94M (35.8 MB). Memory
floor sub-ms on every platform (L40S ~0.29 ms even counting ~7 weight re-reads). **MEASURED decode_wall = 3.6 ms** ⇒
~12× above its floor. This is **not** compute/BW — it is the **eager label-looping host loop**: `while
active_mask.any():` (rnnt_label_looping.py:357) and `while advance_mask.any():` (:409) each force a **D2H copy +
`cudaStreamSynchronize`** per iteration (~6–10 syncs/finalize observed: `cuda_sync_invocations` p50=10).

### 1g. The fp16-is-slower anomaly, explained by the roofline

The memory note records **fp16/bf16 inference measured 0.79× SLOWER** and treats it as proof of "not compute-bound."
The roofline both confirms *and refines* this:
- If the kernels were *cleanly* BW-bound, fp16 should be ~2× *faster* (half the weight bytes). It was slower.
- Reconciliation: at B=1/M≈7 the GEMMs are **not BW-saturated** — they are **latency/occupancy-bound on a fixed
  per-kernel-launch + fixed-overhead floor that fp16 does not shrink.** fp16 adds **per-kernel cast/convert** traffic
  and, on Ada with un-tuned kernels, can pick *slower* code paths; the FP32→FP16 accumulation reconfiguration and
  extra epilogue casts dominate the tiny payload. So fp16's *theoretical* BW win (relevant only near the floor)
  is swamped by per-op overheads that scale with the **number of small ops**, not bytes. **Lesson: precision alone
  doesn't help while we're at 29% of peak; you must first cut the op-count/fusion gap. Done right (fused fp16/fp8
  GEMMs in a TRT engine), fp16/fp8 *would* help — see §3.**

---

## 2. CURRENT-CODE LEVERS (within server.py + NeMo; bottleneck = dispatch/CPU + keep-up, not GPU)

Ranked by combined TTFT + parallelism impact. Citations are `server.py` unless noted.

### Tier 1 — attack the single-scheduler-thread ceiling (the parallelism wall + the TTFT *tail*)

**L1. Pull GPU dispatch + the eager decode off the asyncio event-loop thread (highest parallelism impact).**
The whole scheduler — event drain, batch assembly, per-row Python clones, lane dispatch — runs on **one event-loop
thread** (`_scheduler_loop` :4456, `_scheduler_drain_once` :4510). The lane executors are `ThreadPoolExecutor(max_workers=1)`
(:3156) but the **GIL** means the decode's per-frame `.any().item()` D2H syncs and all the Python glue still serialize
against the event loop. The smoking gun (load sweep, `prodsweep_0728/all_procs.records`):

| per-proc load | vad_recv→process p95 | lock_wait p95 | encoder p50 | finalize p50 | client TTFB p95 |
|---|---:|---:|---:|---:|---:|
| 3.3/proc (conc10) | 8 ms | 12 ms | 11 ms | 20 ms | 261 ms |
| 6.7/proc (conc20) | 89 ms | 25 ms | 13 ms | 26 ms | 359 ms (p99 6807!) |
| 10/proc (conc30) | 495 ms | 37 ms | 15 ms | 34 ms | 716 ms |
| **12/proc (conc36, maxconn)** | **938 ms** | 53 ms | 15 ms | 38 ms | **1230 ms** |

`vad_recv→process` p50 stays ~0.2 ms but p95 explodes to ~1 s: an arriving finalize waits behind a full
steady-batch drain cycle on the one thread. **This is head-of-line blocking, not compute.** Levers (byte-exact, no
math change): (i) a **dedicated high-priority finalize lane/thread** so a finalize never queues behind steady batches
(directly attacks `lock_wait` p95 21→? and `vad_recv→process` tail); (ii) move the per-frame decode-sync loop into a
**worker thread that releases the GIL during the `.item()`/sync** so the event loop keeps servicing I/O and starting
finalizes; (iii) longer-term, a **C-extension / no-GIL (3.13t) dispatcher**. Impact: lifts the keep-up knee and
collapses the `vad_recv→process` tail → reliable capacity ~20→~36–48/box and TTFT p95 at load 1230→~300 ms.

**L2. Admission control on in-flight finalizes + ready-backlog (parallelism robustness).**
At conc-24+ the box catastrophically stalls (Track-A: worst model_batch 13.3 s). A cap (shed/503 or defer) on
concurrent in-flight finalizes and a bounded ready-backlog converts a cliff into graceful degradation. Low risk,
byte-exact. (Already recommended in stall-COMBINED-VERDICT step 3.)

### Tier 2 — cut host syncs + Python dispatch on the steady path (raises the keep-up knee → density)

**L3. Remove redundant `_cuda_synchronize_for_current_model_lane()` calls on the steady path.**
`_process_ready_batch` issues **4 full host syncs per batch**: :8223 (pre), :8243 (after preproc), :8329 (after
model), :8399 (after scatter). The lane wrapper *already* does `stream.synchronize()` after each lane call
(:3177). Each sync is a host↔device round trip that stalls the (GIL-holding) scheduler thread. The pre/post syncs
are for *telemetry/memory snapshots* (`_cuda_memory_snapshot`, `_log_scheduler_batch_memory`) — gate them behind the
telemetry flag and drop to the minimum needed for correctness. Cuts dead time on the critical scheduler thread →
higher keep-up knee. Byte-exact.

**L4. Stop re-syncing in the eager decode (the conc-10 p99 tail + steady cost).**
The eager label-looping `while ... .any():` syncs (rnnt_label_looping.py:357/409) are the **only remaining eager
component on the critical path** and the documented driver of the conc-24 stall + conc-10 p99 (decode_wall 3.6 ms ≫
its 0.3 ms floor). The RNNT **decode CUDA graph is a NO-GO** as a finalize-p50 lever (max-B FULL_GRAPH crashes at B=1
= 86% of the workload; and at 0.8 ms steady it can't move p50 — decoder-graph-probe-findings.md). But two
non-graph mitigations remain: (i) **chunk-bounded decode** — the finalize tail is only ~6 frames, so cap the outer
loop by a host-known `encoded_len` (≤6) and a fixed `max_symbols` to make the loop a *fixed* trip count → no `.any()`
sync (the trip count is already bounded; the `.any()` is only needed for early-exit). (ii) Run decode on a side
stream so its syncs don't block the lane the *next* batch needs. Targets the p99 tail + steady keep-up. Must be
byte-exact-gated (this is the riskiest Tier-2 item).

**L5. Reduce per-row Python clone/stack/scatter glue (pure GIL-thread CPU).**
Per batch the steady path does Python loops of `clone_hypotheses_deep` (:8288), `clone_tree(pred_out_stream)` (:8292),
`stack_hypotheses`/`stack_pred_out` (:8295-96), and per-row `scatter_cache_row` + `_scatter_batch_list_item`
(:8336-8365). These run **on the scheduler thread under the GIL**. fork_clone is only 0.44 ms p50 at the finalize, but
these recur **every 160 ms per stream** and compound at the keep-up knee (they're part of what saturates the one
core). Micro-opts: vectorize the clone (one batched tensor op vs per-row Python), pre-stack into pinned reusable
buffers, avoid deep-copying hypotheses that are about to be overwritten. Modest each, additive to the knee.

### Tier 3 — modest TTFT-p50 compute trims

**L6. One-shot finalize preprocessor (≈1–2 ms p50).** `_preprocess_final_fixed_audio_batch` (:6980) runs the
preprocessor **3×** over the tail and computes a **64-frame** padded STFT to keep ~16 (4× compute waste, a deliberate
cuFFT-plan-determinism tradeoff per the memory note). Collapsing the 3 invocations into one batched preprocessor call
saves ~1–2 ms of finalize p50. **Byte-exact-gated** (a prior batched-final-preproc attempt dropped terminal
punctuation — handle the tail boundary carefully). This is the *only* meaningful p50 compute trim that doesn't touch
the encoder.

**L7. (NOT a lever, recorded to prevent re-litigation.)** fork/clone dedup (~0.44 ms), gather/scatter/clone_hyp
(<0.25 ms), decode-graph for p50 (NO-GO), fp16 (0.79× slower), bigger single-process GPU (GIL-capped). The encoder
finalize is *already* CUDA-graphed (the 246/279 win) — no further graphing headroom there.

**Ranking summary**

| lever | TTFT p50 | TTFT p95/p99 tail | parallelism (streams/box) | effort | risk |
|---|---|---|---|---|---|
| L1 dispatch/decode off event-loop thread | small | **large** (kills vad_recv tail) | **large** (20→~40+) | high | med |
| L2 admission control | — | large (no cliff) | robustness | low | low |
| L3 drop redundant steady syncs | — | small | medium (knee↑) | low | low |
| L4 fixed-trip / side-stream decode | small | medium (p99) | medium | med | **high (byte-exact)** |
| L5 vectorize clone/scatter glue | — | small | small–med | med | low |
| L6 one-shot finalize preproc | **1–2 ms** | small | — | med | med (byte-exact) |

---

## 3. FROM-SCRATCH (no constraints): how close to roofline, and how

Two independent ceilings to break: **(A)** the 29%-of-peak finalize-encoder BW gap (the TTFT-compute roofline), and
**(B)** the single-thread-launch/GIL scaling wall (the parallelism roofline). Both are addressable.

### 3A. A fused encoder engine → finalize encoder 9.7 ms → ~3–4 ms (toward the 2.84 ms floor)

The encoder is BW-bound but runs at 29% of peak because it is 24 layers × dozens of tiny kernels. The fix is to make
each layer **few, large, fused** kernels so the weight read approaches a single streaming pass.

- **TensorRT / TRT-LLM conformer engine, fp16/fp8, fixed shapes (B=1..K, T-buckets).** TRT's GEMM+epilogue fusion
  (bias/LN/Swish/GLU folded into the GEMM epilogue) and attention fusion collapse the per-layer kernel chain. Realistic
  effective-BW for a well-fused B=1 transformer block is **60–75% of peak** → L40S finalize encoder **~3.8–4.7 ms**;
  with **fp8 weights** (halve again the 2.44 GB → 0.6 GB, AI doubles, and Ada/Blackwell fp8 TC is plentiful) the
  *weight-stream floor itself* drops to **~0.7 ms** and a fused fp8 engine plausibly lands **~2–3 ms**. So **finalize
  encoder 9.7 → ~2–4 ms**, i.e. **2.5–5× faster**, pulling finalize_wall (22 ms) toward **~10–12 ms**.
  - fp8 is the *right* quantization here (the roofline says we're weight-bytes-bound). It needs a byte-exact-WER gate
    (post-training fp8 with per-channel scales + a held-out WER non-inferiority test, like the existing graph
    canaries). This directly overturns the "fp16 dead-end" — fp16 *eager* was dead; **fp8 in a fused engine is the
    live precision lever** because it cuts the dominant traffic *and* the kernel-count gap simultaneously.
- **Custom fused conformer block (CUTLASS / Triton)** if TRT shape-rigidity is awkward for the variable T-tail: one
  mega-kernel per layer (or per FFN/attn/conv sub-block) with persistent weights in shared/L2. Same target.
- **Keep it CUDA-graphed** (the engine call still wraps in a graph). The graph already removes launch overhead; the
  fusion is what removes the *intra-layer* serialization and small-kernel BW waste.

### 3B. Break the GIL / single-thread-launch ceiling → reliable 20 → ~40–48/box, kill the TTFT-at-load tail

This is the **most important parallelism move** and the current scaling wall. The knee is one CPU thread dispatching
launches + the eager-decode syncs, multiplied across processes via MPS. From scratch:

- **Decouple the network/event loop from GPU dispatch.** A dedicated **dispatcher thread (or process) per lane** that
  owns a CUDA stream and pulls assembled batches from a lock-free queue; the asyncio loop only does I/O + batch
  assembly. With the decode **fully graphed or fixed-trip** (no `.any()` syncs) the dispatcher issues *zero* host
  syncs in steady state → one core can feed far more streams, and a finalize is dispatched immediately on a priority
  lane (kills `vad_recv→process` 938 ms).
- **No-GIL Python (3.13 free-threaded) or a C++/Rust serving core.** The whole reason multi-process+MPS exists is to
  get GIL-independent dispatch lanes; a no-GIL runtime (or a Rust/C++ inference server calling the TRT engine) gets
  **true multi-thread dispatch in ONE process** → no MPS tax (the measured +15–40% MPS compute tax on encoder_wall
  disappears), no K×11 GB graph-pool duplication (today's K=3 *memory* cap on L40S), and one warm weight copy shared
  across lanes. Expected: the keep-up knee rises toward the **GPU ceiling (~56 un-fused, higher when fused)** instead
  of the ~16 GIL cap; **reliable in-budget capacity ~20 → ~40–48/box on L40S** at far lower memory.
- **Continuous batching that actually fills (B up to K).** Today 86% of finalizes and most steady batches are B=1, so
  the 2.44 GB weight read is *never* amortized. A real continuous-batching scheduler (à la vLLM) that co-batches
  steady chunks **and finalizes** across sessions raises B → AI rises ~B× → the encoder moves toward compute-bound →
  effective throughput multiplies. The existing scheduler has the machinery (lane keys, group keys) but the eager
  decode + per-frame syncs + GIL prevent big batches from paying off; fixing 3A+3B unlocks it. **Steady throughput
  potential: 3–5× (B-fill) on top of the per-call fusion win.**
- **Decode formulation.** Replace the host-controlled label-looping `.any()` loop with either (i) a **graphed
  fixed-trip decoder** keyed per-B (the decoder analog of the bucketed encoder graph — viable & byte-exact on
  Blackwell+Ada per Track B, the open work is the variable-B interlock), or (ii) a **TRT-LLM-style fused greedy RNNT
  decode** with device-side loop control (CUDA conditional nodes / a persistent decode kernel) so there are **zero
  per-frame D2H syncs**. Removes decode_wall's 3.6 ms→~0.3 ms floor *and* the cross-lane stall mechanism.

### 3C. Quantified from-scratch target (L40S, the density GPU)

| metric | today | from-scratch (fused engine + no-GIL dispatch + fp8 + B-fill) | how |
|---|---|---|---|
| finalize encoder GPU | 9.66 ms | **~2–4 ms** | TRT/fused fp8 conformer, 29%→~70% BW, fp8 weight bytes |
| decode | 3.6 ms | **~0.3–0.5 ms** | fixed-trip/graphed/fused decode, no `.any()` syncs |
| preproc | 2.4 ms | ~1 ms | one-shot batched preproc |
| **finalize_wall (server compute)** | **~22 ms** | **~8–10 ms** | sum of above + no lock_wait (priority finalize lane) |
| **TTFT p50** (≈200 ms VAD + 23 ms WAN + finalize) | ~246 ms | **~232–234 ms** | finalize compute is only ~22 of 246 — *p50 is VAD/WAN-bound*, so the compute win is bounded; the real TTFT prize is the **p95/p99 tail** |
| **TTFT p95 at load** | 1230 ms (12/proc) | **~300 ms** | kill `vad_recv→process` tail via off-thread dispatch + priority finalize |
| steady keep-up knee | ~16/proc (GIL) | **~40–56** | no-GIL dispatch + fixed-trip decode (no syncs) |
| **reliable in-budget /box (L40S)** | **~20** | **~40–48** | + remove MPS tax & K×11 GB graph-pool duplication via single-process multi-lane |

**Honest bound on TTFT p50:** ~200 ms of the 246 ms p50 is the **fixed VAD trailing-silence window** (a benchmark
constant, also semantically required for end-of-turn) + ~23 ms WAN. Server finalize compute is only ~22 ms. So even a
perfect engine moves **p50 only ~12–14 ms** (246→~232). **The from-scratch prize is overwhelmingly (a) the p95/p99
tail at load and (b) streams/box** — both gated by the scheduler/dispatch ceiling, not the encoder math. If the
product can shrink the VAD window (e.g. a faster/parallel end-of-turn detector), *that* is the largest remaining p50
lever and lives outside this stack.

### 3D. Effort / risk

| move | effort | risk | payoff |
|---|---|---|---|
| Off-event-loop dispatch + priority finalize lane (still CPython) | med-high | med | tail + knee, no model change |
| Fixed-trip / side-stream eager decode (byte-exact) | med | **high** (correctness) | p99 + knee |
| TRT/fused fp16 encoder engine, fixed B/T buckets | high | med (shape rigidity, capture cost) | encoder 2–2.5× |
| + fp8 weights (PTQ + WER gate) | high | med-high (accuracy gate) | encoder another ~1.5–2× + memory |
| No-GIL (3.13t) or C++/Rust serving core | **very high** | high (whole-stack rewrite) | removes MPS tax, K-memory cap, GIL knee — the durable scaling fix |
| Real continuous batching that fills B | high | med | steady throughput 3–5× |

Pragmatic order: **L1+L2 (off-thread dispatch + admission)** first (no model risk, biggest tail/knee win on the
current code) → **TRT fp16 encoder** (biggest single compute win, model-faithful via WER gate) → **fused/fixed-trip
decode** → **fp8 + B-fill** → **no-GIL core** as the endgame.

---

## 4. Other observations

- **Finalize graph T-bucketing is the L40S K=3 memory cap.** Per-T finalize buckets (T=43–58 ⇒ up to 16 graphs),
  each with its own static buffer pool (cudagraph_encoder.py:193-210), is the ~2–3 GB/proc that forced K=4→K=3 on the
  44 GB L40S. A **single padded finalize bucket** (capture at T_max=58, pad shorter tails, mask the extra outputs —
  byte-exactness permitting) would cut the finalize graph-pool ~10–16× → **recover K=4 (≈64/box)** without shrinking
  the budget. Worth a byte-exact probe; could be a cheaper density win than any compute change. (The memory note's
  "lower FINALIZE_T_MAX/_MAX_B to recover K=4" is the blunt version; padding to one bucket is the surgical version.)
- **The graph replay does 5 `copy_()` into static buffers every call** (cudagraph_encoder.py:273-277), incl. the
  6.9 MB `cache_last_channel`. Sub-ms today, but if the encoder gets ~3× faster these copies become a larger relative
  share — consider capturing with the cache already resident (persistent per-session static cache) to skip the copy.
- **`max_symbols=10`** (config:1162) bounds the inner decode loop; with `encoded_len≤6` at finalize the worst-case
  trip count is tiny and host-known — reinforcing that **L4's fixed-trip decode is feasible** (the `.any()` early-exit
  is an optimization, not a correctness requirement, when the bound is small).
- **Preprocessor computes 64 mel frames to keep 16** (4× waste) — a deliberate constant-FFT-plan choice for
  cuFFT-plan-size determinism (the byte-exactness memory). Don't "fix" by shrinking the plan to fixtures (the memory
  explicitly warns against this); the one-shot batched call (L6) is the safe trim.
- **att_context [70,1] right-context-1** means the encoder's *future* lookahead is 1 frame; the finalize pumps
  (1+1)×16 = 32 frames of silence padding (320 ms) to flush the lookahead + decoder. This 320 ms padding is part of
  the *audio* tail (faster-than-wallclock, per the latency-budget memory), not the 200 ms VAD window — both are in
  the TTFT but neither is server compute.
- **GPU is 46–65% utilized at the lane cap** (deployment memory) — directly consistent with the §1d 29%-of-peak
  finalize + §1e dispatch-bound steady: the silicon is idle waiting on one CPU thread. Every Tier-1/3B lever is about
  *feeding* the GPU, not making it faster.

---

### Method / sources
- Architecture from `/home/khkramer/.cache/huggingface/hub/models--nvidia--nemotron-speech-streaming-en-0.6b/.../nemotron-speech-streaming-en-0.6b.nemo`
  (`model_config.yaml` + `model_weights.ckpt`, loaded with torch in `.venv-asr`; 618.1M params, fp32).
- Code paths: `src/nemotron_speech/server.py` (scheduler :4456, ready-batch :8216, finalize :7350-7500, lanes
  :3145-3211, decode cfg :1463, syncs :8223/8243/8329/8399, finalize preproc :6980); `cudagraph_encoder.py`;
  NeMo `mixins.py:592` (conformer_stream_step), `conformer_encoder.py:574/616/977` (forward + streaming cfg),
  `transducer_decoding/rnnt_label_looping.py:357/409` (eager `.any()` loops).
- Measured: `ec2-bench/lanes_ab_1624/prod_l40s_c10_lanes2.records` (2134 finalize records — encoder_cuda_event 9.66 ms,
  T/B dist), `ec2-bench/prodsweep_0728/all_procs.records` + clientlogs (load sweep: vad_recv→process 8→938 ms).
- GPU specs: L40S 864 GB/s / 91.6 FP32 / 362 FP16-dense TFLOP; L4 300 GB/s / 30.3 / 121; DGX Spark GB10 **273 GB/s**
  unified LPDDR5x; RTX 5090 1792 GB/s GDDR7 / ~105 FP32 (local `nvidia-smi`: 5090 32 GB, driver 580.65 / CUDA 13.2).
  - https://www.nvidia.com/en-us/data-center/l40s/ , https://lenovopress.lenovo.com/lp1812-nvidia-l40s-48gb-pcie-gen4-passive-gpu
  - https://docs.nvidia.com/dgx/dgx-spark/hardware.html , https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/
</content>
