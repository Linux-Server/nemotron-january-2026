# Opus adversarial review — Round 1

Reviewing `PLAN.md` v1 against the actual source. Harsh by design.

## [BLOCKER] 1. The thesis names the wrong bottleneck — it is NOT "the GIL serializing CUDA launches"
The plan repeatedly says "single serial launch-dispatch lane" = GIL. The code shows a more specific (and more
attackable) picture, and getting it wrong risks building the wrong thing:
- Multi-lane already exists: each lane has its own `ThreadPoolExecutor(max_workers=1)` (`server.py:3155-3160`), its own
  `torch.cuda.Stream()` (`:3161-3163`), and its own **full model replica** loaded via a separate `restore_from().cuda()`
  (`_load_scheduler_model_lane_model` `:3109-3143`). Lane calls run in a worker thread via `run_in_executor`
  (`:3183-3192`) — and **PyTorch releases the GIL during CUDA kernel dispatch**, so steady kernel launches across lanes
  are *already* not GIL-serialized.
- The real serializers are: (a) the **single asyncio event-loop thread** that does socket I/O + all scheduling
  decisions + drives the dispatch (`_scheduler_loop`), (b) the **`_scheduler_exclusive_model_path` gate** (`:3213-3232`)
  which forces finalize, first-chunk, and B=1 barrier drains to run **exclusively after all in-flight lanes drain**
  (`:3242-3264`, `:3266-3283` — only steady normal-chunk keys may share lanes), and (c) the **GIL-bound eager RNNT
  decode** Python loop with per-frame host syncs.
- **Recommendation:** Rewrite §1 to name the three real serializers precisely. The GIL matters specifically for (a) the
  event-loop thread and (c) the Python decode loop — not for steady CUDA launches. Spike 0.1 must measure the *right*
  thing: not "launches/sec with vs without GIL" in the abstract, but whether **finalize can run concurrently with steady
  on a second real dispatcher thread** without the exclusive-path stall and without the event loop blocking I/O.

## [BLOCKER] 2. B1 export underestimates the RNNT decode — the GIL-bound part is exactly the part that won't export
The plan says "we are NOT re-deriving the model … driving the same NeMo weights via libtorch (B1) so fidelity is
preserved by construction." But:
- The encoder (`encoder.cache_aware_stream_step`) is the *easy* part to graph (already CUDA-graphed —
  `cudagraph_encoder.py`). It is **memory-bound and near-floor** (roofline) → exporting it buys ~nothing on latency.
- The **RNNT greedy decode is delegated entirely to NeMo** (`LabelLoopingStateItem`, referenced `:188/250/362` only for
  state-cloning; the loop itself is inside NeMo's `RNNTDecoding`), is a **data-dependent Python loop with per-frame
  `.item()`**, and **`use_cuda_graph_decoder=False` is forced for Blackwell safety** (`:1473/1492`, memory
  `stall-root-lanes-contention-decode-graph`). This loop **will not TorchScript/`torch.export` cleanly** (data-dependent
  control flow, `.item()` syncs) and is the *actual* GIL-bound hot path.
- So B1 reduces to: graph the encoder (≈already done) + **hand-reimplement the label-looping greedy decode in C++/CUDA**,
  byte-exact vs NeMo, AND solve the Blackwell cuda-graph-decoder problem NeMo punted on. That contradicts "fidelity by
  construction" and is the single largest hidden cost.
- **Recommendation:** Split B1 into B1a (encoder via libtorch/export — low risk, low payoff) and B1b (decode
  reimplementation — high risk, the actual payoff). Add a Phase-0 spike that **reimplements the greedy label-looping
  decode in the chosen native stack and proves byte/WER-CI equivalence on a fixed encoder-output fixture**, BEFORE
  committing. This spike, not the encoder export, is the real go/no-go.

## [MAJOR] 3. The K× model-replica memory cost is mis-stated, and the native win there is under-sold
The plan says density is capped by "K×11 GB graph-pool duplication." The code shows each lane loads a **separate full
fp32 model** (`:3120-3130`) — so the memory cost is *model replica (~2.4 GB weights, more resident) + per-lane graph
pool*, K times. A native runtime can hold **one read-only weight copy shared across all lane threads** (no GIL, no
separate NeMo objects) — this is a concrete, large density lever the plan should name explicitly as a primary
justification (it directly enables higher K / more streams/box, independent of the launch-ceiling argument).
- **Recommendation:** Add "shared read-only weights across lanes" as an explicit native-only density mechanism in §1/§3,
  and quantify the expected memory saving vs the measured K×replica footprint.

## [MAJOR] 4. B4 (no-GIL py3.13t) cannot keep "identical code" — it interacts badly with the asyncio architecture
The plan rates B4 "math identical, low–medium effort." But the current concurrency is **asyncio single-thread + lane
threadpools**, not free threads. Free-threaded CPython removes the GIL but does **not** make the asyncio event loop
multi-threaded — the head-of-line blocking on the single event-loop thread (the actual tail cause) remains unless the
architecture is restructured to use real threads for dispatch. Also: free-threaded PyTorch/NeMo wheels' maturity is
genuinely doubtful (C-extensions must opt into `Py_mod_gil`). B4 is therefore NOT "keep current Python" — it's "keep
Python but rewrite the scheduler off asyncio onto threads," which is most of the hard part anyway.
- **Recommendation:** Restate B4 honestly: its value is as a *thesis probe* (does removing the GIL + real dispatch
  threads lift the ceiling on the real stack?), and it still requires a scheduler rewrite. Don't sell it as low-effort
  drop-in.

## [MAJOR] 5. Correctness bar T1 is under-specified and genuinely risky for a *streaming* product
The whole project ran on byte-exact (memory `silence0-warm200-shippable`, `finalize-graph-probed-rejected`). Relaxing to
"full-1000 WER within CI + sequencing identical" is defensible *only if* the bar catches streaming-specific regressions
that aggregate WER hides: per-chunk **interim emit timing/flicker**, **partial→final delta correctness**, **finalize
boundary token splits**, and **EOU/finalize timing distribution** (the product is a voice agent — interim cadence
matters, not just final text). Aggregate WER on a 1000-utterance corpus can stay within CI while interim behavior
regresses badly.
- **Recommendation:** Make T1 concrete: (i) define the WER-CI width numerically; (ii) require **per-utterance** WER
  delta bounds, not just corpus mean; (iii) add an explicit **interim-emit-sequence diff** (timing-bucketed) and
  **partial/final delta** equivalence test; (iv) keep byte-exact as the *gate* for the encoder path where libtorch uses
  the same kernels (it should be achievable on a single 5090 with a constant FFT plan — if it's NOT, that's a finding to
  surface, not to wave away).

## [MAJOR] 6. The "single CUDA context serializes launches anyway" risk needs MPS/green-contexts in the plan as a first-class option, not a footnote
Even GIL-free, multiple host threads launching into multiple streams **in one CUDA context** share one launch queue;
true parallel kernel execution across lanes on one GPU typically needs **MPS** (which the current deploy already uses —
memory `deployment-target-sagemaker`) or CUDA green contexts / per-process contexts. The plan mentions MPS only in risk
#1. If MPS is required even for the native runtime, then the "single process, many lanes, no MPS tax" density argument
(§1 table "no MPS tax") may be partly false.
- **Recommendation:** Spike 0.1 must explicitly compare: single-context multi-stream vs MPS vs multi-process, GIL-free,
  and report which actually overlaps finalize+steady on each target GPU. Resolve the "no MPS tax" claim with data before
  asserting 40–48/box.

## [MINOR] 7. Anchors to verify / correct
- `inference_lock` is an **`asyncio.Lock`** (`:594`), not a threading lock — important because it serializes at the
  *coroutine* level on the event-loop thread, reinforcing finding #1. State this.
- Plan cites finalize fork `6370-6425` and flush `6738-6895`, conformer `2929-3078`, preproc `1509-1593`, ws
  `4148-4248`, lanes `3104-3414` — these match the explored ranges. Keep but add the exclusive-path anchor `:3213-3232`
  and the per-lane replica `:3109-3143`, which are load-bearing for the thesis and currently missing.

## [MINOR] 8. DGX Spark aarch64 + the moving Python baseline
- The plan correctly flags aarch64, but should add: libtorch aarch64 + CUDA-on-GB10 wheel availability is not guaranteed;
  add a "can we even get a working libtorch+CUDA toolchain on GB10?" pre-check before 4.2.
- The "Python baseline keeps moving" risk is noted but has no mechanism: add that each cloud gate records the *exact*
  Python baseline commit it beat, so the native win is attributable.

## [MINOR] 9. Throughput/finalize numbers are inherited, not re-derived
"3–5× throughput, 40–48/box, 6–10 ms finalize" come straight from the roofline memo's *projections*, which were
first-principles estimates, not measurements. The plan should label them **targets to validate**, and Spike 0.1/Phase 4
should be the falsification tests. The 6–10 ms finalize in particular assumes fused fp16/fp8 — but the roofline also
says fp16 was 0.79× SLOWER until fusion; so 6–10 ms is contingent on a *successful fusion* that is itself unproven.

## Top 5 things to fix
1. **Rewrite the thesis (§1)** to name the three real serializers (single asyncio thread, exclusive-model-path gate
   `:3213-3232`, GIL-bound Python decode loop) — not "GIL serializes CUDA launches."
2. **Add a Phase-0 decode-reimplementation spike** (B1b): the RNNT greedy label-looping decode is the GIL-bound payoff
   AND the un-exportable, byte-exact-risky part — make it the real go/no-go, not the encoder export.
3. **Make T1 streaming-aware**: per-utterance WER bounds + interim-emit-sequence + partial/final-delta equivalence;
   numerically define CI width; keep encoder byte-exact as a gate where kernels are identical.
4. **Resolve "no MPS tax"/single-context-launch with data** in Spike 0.1 (single-context multi-stream vs MPS vs
   multi-proc, GIL-free) before asserting 40–48/box.
5. **Re-label all throughput/finalize numbers as targets-to-validate** and tie 6–10 ms finalize to the (unproven)
   fusion; add the shared-read-only-weights density lever explicitly.
