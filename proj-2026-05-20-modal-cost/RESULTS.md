# Modal per-GPU concurrency-knee + cost/stream study

**Goal:** deploy the EN Nemotron streaming ASR model on every Modal GPU type, find the
**realtime keep-up knee** (max concurrent live streams before one GPU can no longer consume audio
as fast as realtime — finalize/processing latency runs away), then build a **cost-per-stream**
table from Modal's per-hour pricing.

## Apples-to-apples with local

Deploys the **canonical `src/nemotron_speech/server.py`** (the updated implementation:
`NEMOTRON_CONTINUOUS=1` + `FINALIZE_SILENCE_MS=0` + `WARMUP_MS=200`, fork-flush finalize, continuous
context — silence0_warm200), **not** the older embedded reimplementation in `asr_server_modal.py`.
NeMo pinned to the exact local commit (`NVIDIA-NeMo/NeMo @ 056d937…`, 2.8.0rc0). One container, one
GPU, concurrent WS inputs (`max_containers=1` + `@modal.concurrent`) → same single `inference_lock`
+ `batch_size=1` serialization as local. Driven by the same harness
(`proj-2026-05-19-eou-endpointing/concurrency_test.py`, `--url <wss>`), rc1.

- Deploy file: `src/nemotron_speech/modal/asr_bench_modal.py` (GPU via `ASR_GPU` env).
- Local anchor (RTX 5090, [[silence0-warm200-shippable]]): server-finalize flat ~15 ms to N=12,
  knee ~N=16-18 (33 ms@16 → 664 ms@20 → 2082 ms@24). Capacity ~12-16 comfortable.
- torch on Modal = 2.12.0 (local 2.11.0+cu130); NeMo identical. Minor; forward-pass timing ~same.

⚠ **Measurement caveat:** sweeps are driven from the *local* client over WAN. For slow GPUs (low
knee) this is fine. For fast GPUs (high knee) the local client may bottleneck before the GPU
saturates → falsely-low knee; if observed, re-run with a same-region Modal-hosted load generator.
Per-GPU notes flag whether this bit.

## Modal GPU pricing (fetched 2026-05-20, modal.com/pricing)

| GPU | $/sec | $/hour |
|-----|-------|--------|
| T4 | 0.000164 | **$0.59** |
| L4 | 0.000222 | **$0.80** |
| A10(G) | 0.000306 | **$1.10** |
| L40S | 0.000542 | **$1.95** |
| A100 40GB | 0.000583 | **$2.10** |
| A100 80GB | 0.000694 | **$2.50** |
| RTX PRO 6000 | 0.000842 | **$3.03** |
| H100 | 0.001097 | **$3.95** |
| H200 | 0.001261 | **$4.54** |
| B200 | 0.001736 | **$6.25** |

## KEY FINDING: capacity is server-throughput-bound, ~GPU-independent

The realtime keep-up knee is **~5 streams/GPU on Modal** (T4≈L4≈H100), vs **~12-16 on the local
RTX 5090**. Root-caused (not GPU, not WAN, not CPU cores):

- **GPU-FLOPs ruled out:** H100 ($3.95) knee ~6-7 ≈ L4 ($0.80) ~5 ≈ T4 ($0.59) ~5. 5× price → ≤40%
  more streams (within ±1-2 instance noise). **H100 is the *worst* $/stream.**
- **WAN / client ruled out (co-located test):** an in-region (us-east-1) load-gen on the *same* L4
  gave the **same knee (~4-5)**; WAN only dropped the N=1 baseline (180→89 ms). Client send-pacing
  overrun stayed ≤19 ms even at N=16 → the client kept perfect realtime; **the server saturates.**
- **CPU cores ruled out:** Modal already gives **17 cores** by default; the critical path is
  single-threaded (one `inference_lock`, `batch_size=1`).
- **Instance noise:** ±1-2 streams (a bad L4 instance broke at ~3 vs a good one at ~5).

**Conclusion:** the bottleneck is the **single-threaded per-chunk critical path** (mel preprocessing
+ RNNT greedy decode + Python/kernel-launch), bound by single-core speed — the local 5090 box has a
faster desktop core + no virtualization. **The throughput lever is batching the streaming path
(currently `batch_size=1` by design — see [[silence0-warm200-shippable]]), NOT a bigger GPU.**
Cheapest GPU (T4) wins $/stream; scale streams horizontally (more cheap containers), not vertically.

### Batch-scaling microbench (L4, encoder forward, feat=128, T=200) — the batching prize ≈ 8-10×

| batch | ms/fwd | ms/stream | throughput vs b=1 |
|------|--------|-----------|-------------------|
| 1 | 47.0 | 47.0 | 1.0× |
| 4 | 49.0 | 12.3 | 3.8× |
| 8 | 49.8 | 6.2 | **7.6×** |
| 16 | 78.5 | 4.9 | **9.6×** |
| 32 | 160.6 | 5.0 | 9.4× |

8 streams' chunks cost the same wall-time as 1 (49.8 vs 47.0 ms) → **batch=1 wastes ~87% of the
GPU**. The 47 ms/forward is **kernel-launch/dispatch-bound, not compute-bound** (hundreds of tiny
conformer kernels; the math is µs) → explains H100≈L4 (launch latency is GPU-independent) and
reproduces the knee (≈320 ms chunk ÷ ≈60 ms/chunk ≈ 5 streams). **Batched streaming → knee ~5→~40-50
(~8-10× streams/$).** Secondary lever: CUDA-graph capture to collapse the launch overhead (raises
the single-stream knee without batching). Both are server changes; the cache-aware streaming state
makes batched streaming non-trivial (the documented "vLLM-style continuous batching" research item).

> ⚠ **CORRECTION (per-chunk profiling, 2026-05-21):** the 47ms above used the **full encoder forward
> on T=200 frames — too large**. The *real* cache-aware streaming step (`conformer_stream_step` on a
> 160ms chunk + cache) is **~10ms on the local 5090** (profiling below), so 47ms *overstates* the
> per-chunk cost and the 8-10× batch multiplier must be **re-measured on the real step** (pending).
> The qualitative point (batch=1 underutilizes the GPU → batching helps) likely holds; magnitude TBD.

### Per-chunk profiling (instrumented `server.py`, `NEMOTRON_PROFILE_CHUNK=1`)

Faithful timing of the live `_process_chunk` path (preprocess vs `conformer_stream_step`), single-stream.

| environment | preprocess | step (enc+dec) | total/chunk | implied keep-up (160ms ÷ total) |
|-------------|-----------|----------------|-------------|---------------------------------|
| **local RTX 5090** | 1.06 ms | 10.27 ms | **11.33 ms** | ~16-18 measured (naive 160÷11.3≈14) |
| Modal slow (L4) | _pending_ | | | |
| Modal mid (A100) | _pending_ | | | |

**Preprocessing is NOT the bottleneck (1ms, 9%); the `conformer_stream_step` (enc+dec, 91%) is** —
fix(1) targets the *step*, not preprocessing. The Modal points will localize the ~3× gap (local ~16-18
→ Modal ~5).

## Per-GPU results (realtime keep-up knee)

`$/stream-hr = $/hour ÷ knee_N`. Lower is better. Filled as each GPU is swept; app stopped between.

| GPU | $/hr | Knee N (keep-up) | $/stream-hr | $ / 1k-stream-hr | Notes |
|-----|------|------------------|-------------|------------------|-------|
| T4 | 0.59 | ~5 | **$0.12** | $118 | N≤4 keeps up (271ms); ≈L4. **Best $/stream.** |
| L4 | 0.80 | ~4-5 | $0.18 | $178 | GOOD instance N≤4 (239ms), cliff N6→1655ms. ⚠ 1st instance bad (~N3) → ±1-2 noise. byte-exact. |
| A10G | 1.10 | ~2-3 | $0.44 | $440 | N2 ok (200ms), N4 behind (1062ms). Ampere — lowest knee (or slow instance). |
| L40S | 1.95 | ~4 | $0.49 | $488 | N4 ok (217ms), N6→1935ms. ⚠ correctness unverified (0/0 baselines) — recheck. |
| A100 40GB | 2.10 | ~4 | $0.53 | $525 | N4 ok (240ms), N6→1915ms. |
| A100 80GB | 2.50 | ~5 | $0.50 | $500 | N4 ok (197ms), N6→1499ms. |
| H100 | 3.95 | ~6-7 | $0.61 | $610 | N4=217ms, N8=2189ms (R≈7). 5× price → ≤40% more = WORST $/stream. Proves server-bound. |
| RTX-PRO-6000 | 3.03 | ~5-6 | $0.55 | $550 | **5090-sibling (Blackwell GB202)**: clean re-verify N4=168ms ok, N8=1013ms behind → ~5-6 (CONFIRMED, no contention). **Same silicon as local-5090's ~16-18, but ~5-6 on Modal → the gap is the virtualized-cloud ENVIRONMENT, not GPU-arch.** Microbench encoder fwd 12.6ms (4× faster than L4) yet knee≈same → server per-call overhead, not GPU compute. |
| H200 | 4.54 | ~4-5 | $1.01 | $1010 | N4=207ms ok, N6=922ms behind. Hopper; ~ data-center. |
| B200 | 6.25 | n/a | — | — | FAILED to run twice (incl. patient 600s smoke) — GB200 scarce/won't schedule on Modal. Skipped (would be worst $/stream anyway). |

### Step 10 batched re-sweep, cheap subset only (T4 + L4)

Config: `NEMOTRON_CONTINUOUS=1 NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1
NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200`, greedy_batch, TF32 off,
defaults `MAX_SIZE=32` / `MAX_WAIT=8ms`. ASR and loadgen were co-located in `us-east-1`.
Apps were stopped immediately after each GPU.

Headline: **batching did not close the cloud gap on the cheap subset.** T4 regressed under the strict
realtime p95-lag gate, and L4 improved only modestly. The cloud runs did form batches under overload,
but near the realtime knee most chunks were still B=1, so the local MAX_SIZE=32 payoff (~16→56, ~3.5x)
did not transfer.

| GPU | batch=1 baseline knee | batched strict knee | improvement | batched $/stream-hr | client-bound? | startup cap |
|-----|-----------------------|---------------------|-------------|---------------------|---------------|-------------|
| T4 | ~5 ($0.59/hr → ~$0.12/stream-hr) | **4** | **0.8x** | **$0.15** | no (send-overrun p95 15-16ms) | requested 32 → effective 19 |
| L4 | ~5 ($0.80/hr → ~$0.16/stream-hr) | **6** | **1.2x** | **$0.13** | no (send-overrun p95 14-16ms) | requested 32 → effective 31 |

Per-level details (`keep-up` = no errors/timeouts and processing-lag p95 <500ms):

| GPU | N | keep-up | TTFS p95 ms | proc-lag p95 ms | avg effective B | batch histogram | send-overrun p95 ms |
|-----|---:|:-------:|------------:|----------------:|----------------:|-----------------|--------------------:|
| T4 | 4 | yes | 360 | 379 | 1.01 | `{1:124,2:1}` | 16 |
| T4 | 5 | no | 2925 | 2946 | 1.00 | `{1:225}` | 15 |
| T4 | 8 | no | 3894 | 3912 | 1.68 | `{1:141,2:115,3:44}` | 15 |
| L4 | 4 | yes | 356 | 375 | 1.00 | `{1:150}` | 15 |
| L4 | 5 | no | 856 | 877 | 1.03 | `{1:194,2:6}` | 15 |
| L4 | 6 | yes | 461 | 480 | 1.33 | `{1:175,2:25,3:25}` | 14 |
| L4 | 7 | no | 1238 | 1256 | 1.49 | `{1:178,2:29,3:36,4:7}` | 14 |
| L4 | 8 | no | 2996 | 3015 | 1.41 | `{1:207,2:143}` | 16 |

Smoke: both GPUs produced sane English transcripts with no tag leakage or looping. The optional strict-byte
N=8 smoke was not exact (T4 3/8, L4 4/8) while already well beyond the realtime knee, so this re-sweep should
be read as a throughput/cost result plus smoke sanity, not a full cloud strict-byte signoff.

## SYNTHESIS — answer to "why Modal ~5 vs local 5090 ~16-18?"

**It is host-side kernel-LAUNCH/DISPATCH overhead, bound by CPU single-thread speed — NOT the GPU and
NOT "virtualization" per se.** Isolation microbench (`iso_measure.py`, identical code local + Modal):

| | local 5090 | Modal RTX-PRO-6000 (same GB202 silicon) | Modal L4 |
|---|---|---|---|
| CPU single-thread | 5.69 GHz (Ryzen 9950X) | 3.54 GHz | 2.65 GHz |
| compute (fp16 matmul) | 223 TFLOP/s | **399** | 51 |
| launch latency (synced) | 6.6 µs | 13.0 µs | 15.8 µs |

- **GPU compute RULED OUT (decisively):** the Modal RTX-PRO-6000 is ~1.8× *faster* compute than the local
  5090 (399 vs 223 TFLOP/s) yet knees ~3× *worse*. Same silicon, faster GPU → not the GPU/clock.
- **The bottleneck is launch dispatch (~2× slower on Modal, 13 vs 6.6 µs)** — the conformer step is
  launch-bound (hundreds of tiny kernels/chunk). (Now decomposed precisely in the captured per-chunk
  SPLIT below: the model step is 9.3→13.8 ms sync'd, and a co-equal server-layer term takes the knee
  budget to ~30 ms on Modal — it is NOT a ~30 ms *model* step.)
- **That tracks CPU single-thread clock** (6.6/13.0/15.8 µs as the host core slows 5.69/3.54/2.65 GHz).
  The ~2× launch gap ≈ ~1.6× CPU-clock gap + a small (~0.25×) residual = the only true virtualization part.
- Also ruled out earlier: home→Modal WAN (in-region co-located test gave the same knee); CPU core count (17).

**Bottom line: the local box's edge is a 5.7 GHz desktop CPU dispatching launch-bound work fast; cloud
server CPUs (~2.6-3.5 GHz) dispatch ~2× slower. The fixes (CUDA-graphs/torch.compile, batching) all attack
LAUNCH overhead — not the GPU.** (Earlier "bare-metal vs virtualized" framing was imprecise; corrected here.)

**Sanity check (2026-05-21) — local sweep with the EXACT same code (server.py + concurrency_test.py,
only `--url`=localhost):** proc-lag flat ~32-34ms through N=16 (TTFS p50 ~12-14ms), tail blows up at
N=20 (p95 689ms), runaway at N=24 (1172ms) → **local knee ~16-18, byte-exact at every N**. This is the
apples-to-apples confirmation: same code, local ~16-18 vs Modal ~5 (~3×) = the environment. Local
proc-lag baseline ~33ms vs Modal ~150-240ms confirms WAN inflates only the BASELINE latency, not the
knee. (`concurrency_local_5090.json`.)

### Per-chunk GPU-vs-host SPLIT, all three boxes (2026-05-21, `profile_split.py` — CAPTURED)

Same clip, same code, batch=1, model-only (no server). `gpu_active` = torch.profiler kernel self-time
(pure execution, no idle); `step span` = CUDA-event wall around the synchronized enc+dec; `idle-gaps` =
span − active = the GPU waiting on the host between launches. Result written to the volume (the earlier
`modal app logs` issue is gone — `run()` now persists the dict + `vol.commit()`).

| per-chunk (160ms audio) | local 5090 | Modal RTX-PRO-6000 | Modal L4 |
|---|---|---|---|
| GPU clock / fp16 TFLOPs | 3090 MHz / 223 | 2430 MHz / **399** | 2040 MHz / 51 |
| launch latency (synced µs) | 6.6 | 13.0 | 15.8 |
| **GPU-active (kernel self-time)** | **6.06 ms** | **7.50 ms** | **15.57 ms** |
| step span (sync'd enc+dec) | 9.3 ms | 13.83 ms | 48.17 ms |
| **idle-gaps (span − active)** | **3.24 ms** | **6.33 ms** | **32.6 ms** |
| gaps as % of span | 35% | 46% | 68% |
| measured server knee | ~16-18 | ~5-6 | ~5 |

**The per-chunk cost has TWO components that scale with DIFFERENT hardware — and both microbench ratios
hit dead-on:**

1. **GPU-active execution scales with GPU CLOCK, not FLOPs.** 5090→Pro-6000 = 6.06→7.50 ms = **1.24×**,
   matching the **1.27× clock ratio** (3090/2430). The Pro-6000 has **1.8× the FLOPs** yet runs the step
   **1.24× slower** — the conformer kernels are small/latency-bound, so FLOPs are irrelevant and clock
   rules. **This confirms the user's hypothesis: GPU-exec is ~the same on same-class (Blackwell) silicon.**
   L4's GPU-active is 2.57× (15.57 ms) — clock explains only 1.51×, the rest is a genuinely *smaller* GPU.

2. **Idle-gaps scale with CPU single-thread launch latency.** 5090→Pro-6000 gaps = 3.24→6.33 ms = **1.95×**,
   matching the **1.97× synced launch-latency ratio** (13.0/6.6 µs) dead-on. These gaps ARE host
   launch/dispatch — the GPU starving while the CPU marshals the next of hundreds of tiny kernels. L4's
   gaps balloon **10×** (32.6 ms, 68% of the span) — far past the 2.4× trivial-launch ratio, because the
   slow virtualized CPU's *real*-kernel dispatch (PyTorch + NeMo Python wrapper per op) is the **"missing
   factor"** the trivial-kernel microbench under-measured.

**Why the knee is even lower than these numbers — the SERVER layer.** The isolated model step (Pro-6000
sync'd 13.8 ms; true GPU-active 7.5 ms) does NOT by itself explain the ~5-6 knee, which implies ~27-32 ms
of serialized per-stream budget (160 ÷ 5.5). The extra ~half is **server-layer CPU work** — per-chunk
STFT preprocess + asyncio scheduling + GIL contention across N threads + hypothesis/delta + lock handoff
— which on the slow virtualized cloud CPU costs ~15 ms but on the 5.7 GHz local desktop costs only ~3 ms.
Unified: the **CPU-bound portion** of the per-stream cost (launch-gaps + server Python) is **~3 ms local
vs ~21 ms Modal Pro-6000 (~7×)**; GPU execution is nearly identical (1.24×). **That CPU portion — not the
GPU — is what collapses the knee from ~17 to ~5-6.**

**Fix implications (both plan levers validated, and the cloud data says they help MORE there):**
- **Batching** amortizes GPU-active across B streams (microbench: batch-8 wall ≈ batch-1) AND collapses
  per-stream launches → attacks both components. The 8-10× lever.
- **CUDA-graphs / torch.compile** collapse the idle-gaps (Probe A: 1.54× locally by removing the 35%
  gaps). On Modal the gaps are **46-68% of the span**, so graph capture should help *more* on the
  slow-CPU cloud boxes than locally.
- The continuous-batching scheduler also collapses N server-orchestration passes → 1, attacking the
  ~15 ms server-layer overhead that co-dominates the cloud knee.

(Caveat: `profile_split` synchronizes per phase to attribute enc-vs-dec, which prevents CPU run-ahead and
*inflates* the gaps vs the async server's pipelined path; `gpu_active` is the sync-independent floor and
the same-class gap RATIO is robust. Knee ±1-2 streams. The earlier "Modal step ≈ 30ms" was inferred from
the knee and conflated model + server cost; the model step alone is 13.8 ms sync'd / 7.5 ms active.)

## Cost / deployment recommendation

| Scenario | Best option | ~$/stream-hr (batch=1) |
|----------|-------------|------------------------|
| Bursty / variable / no-ops | **Modal cheapest GPU (T4)**, scale horizontally, autoscale-to-zero | **~$0.12** |
| Steady, high-utilization, ops-capable | **Self-hosted bare-metal consumer GPU (RTX 5090)** — ~16-18 knee, ~3× Modal/GPU | **~$0.02** (HW ~$3-4k/3yr + power ≈ $0.25-0.30/hr ÷ ~16-18) |
| Cloud-rented 5090 (RunPod ~$0.67/hr) | likely VIRTUALIZED → ~5-6 knee (like Modal 6000), NOT ~16-18 | ~$0.11 (no bare-metal advantage) |

- **On Modal, GPU choice barely matters — cheapest wins; bigger/Blackwell GPUs are the WORST $/stream.**
- The **bare-metal ~16-18 vs cloud ~5-6** advantage requires owning the hardware (no virtualization); a
  *rented* cloud 5090 likely behaves like Modal (~5-6).
- **The real lever (~8-10×, applies to ANY environment): batching the streaming path.** `batch_size=1`
  is the cap; the optimization plan `proj-2026-05-21-0410/PLAN.md` implements continuous batching →
  expected knee ~5→~40-50 on Modal (~$0.012/stream on T4) and ~16-18→higher locally. **Batching dwarfs
  both GPU choice and self-host-vs-cloud.**

## Procedure (per GPU)
1. `ASR_GPU=<gpu> .venv/bin/modal deploy -m src.nemotron_speech.modal.asr_bench_modal` (image cached after L4 → fast).
2. 1-stream smoke (correctness) at `wss://daily--nemotron-asr-bench-asr.modal.run`.
3. `concurrency_test.py --url <wss> --sweep <N…>` → find where `processing_lag_ms`/`ttfs_ms` runs away.
4. `.venv/bin/modal app stop nemotron-asr-bench` (save $).
5. Record knee + $/stream.
