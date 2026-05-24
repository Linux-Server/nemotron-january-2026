# Opus adversarial review — Round 2 (stressing v2; new findings only)

## [BLOCKER] 1. The "shared read-only weights" density lever collides with per-call MUTABLE global config — it is not read-only
v2 promotes "one shared read-only weight copy across lane threads" to a primary density mechanism (§1, §3.1, step 1.1).
But the encoder call is **not** stateless on the model object: `encoder_stream_step_restoring_drop_extra`
(`cudagraph_encoder.py:56-64`) **saves and restores `model.encoder.streaming_cfg.drop_extra_pre_encoded`** around every
call — proving NeMo reads that value from *shared module config*, not a call argument. First chunks use `drop_extra=0`,
steady uses `drop_extra=self.drop_extra` (2), finalize differs again. **Two lane threads sharing one encoder object and
running a first-chunk and a steady-chunk concurrently would race on that global** (one's restore clobbers the other's
set). The current design dodges this precisely because each lane has its **own** model replica (`server.py:3109-3137`).
- **Recommendation:** The native runtime must thread `drop_extra` (and any other per-call config: att-context,
  keep_all_outputs) as **pure call parameters**, never mutate shared module state. Make "shared weights ⇒ all per-call
  config passed explicitly, zero shared mutable state on the hot path" an explicit requirement of step 1.1 and a
  checked invariant in 0.6/1.2. Until that's designed, the shared-weights memory win is not bankable.

## [BLOCKER] 2. The preprocessor (STFT/mel + ring buffers + finalize multi-pass) is an unscoped reimplementation with its own determinism cliff
v2 treats preprocessing as a one-liner ("document constant-FFT-plan handling" in 1.1). But the native runtime must
re-create: the constant-plan STFT (`server.py:1509-1593`), `dither=0.0`, the **raw_audio_ring** (STFT boundary context)
and **mel_frame_ring** (9-frame pre-encode cache) buffering, the `preprocess_new_audio_samples=2720` ingest cadence, and
the **finalize multi-pass preproc loop** (`:6927-6942`). The project's hardest-won correctness lesson is
`cufft-stft-plan-size-nondeterminism`: incremental ≠ growing-full-reprocess bit-exactly on CUDA. A native STFT (libtorch
`torch.stft` vs raw cuFFT) can reintroduce exactly this nondeterminism, and it sits *upstream* of the encoder, so any
drift poisons the T2 encoder byte-exact gate.
- **Recommendation:** Add **Spike 0.8 — native preprocessor byte-exactness** (constant cuFFT plan, dither=0, ring-buffer
  semantics, finalize multi-pass) on the 5090, gated to T2 byte-exact vs Python BEFORE the encoder spike (0.2) is even
  meaningful. Make the audio-ingest/ring-buffer port an explicit step, not a footnote in 1.1.

## [MAJOR] 3. The finalize/reset/resume STATE MACHINE is the highest-risk port and is not its own step
T1 now mentions "stale-generation suppression" and "resume-after-stop," good — but the underlying state machine
(STREAMING → PENDING_FINALIZE → FINALIZED, `server.py:89-91/487`; debounce window; `continuous_reset_seen`; audio
arriving during the debounce window; reset during finalize) is subtle, concurrent, and where ordering bugs will hide
under the new fine-grained-ownership model (which *removes* the coarse `asyncio.Lock` that currently makes these
transitions atomic-ish on one thread). Porting it is not implied by "Phase 1.3 finalize path."
- **Recommendation:** Make the finalize/reset/resume state machine an explicit sub-step with its own state-transition
  table + a property/fuzz test (random interleavings of audio / vad_stop / reset / debounce-expiry) proving equivalence
  to Python. Tie it to the 1.0 ownership design.

## [MAJOR] 4. Multilingual / prompted model is essentially unscoped — but it changes the decode, batch key, and att-context
The repo supports a prompted multilingual variant (att-context fallback `[[56,0],[56,3],[56,6],[56,13]]`
`server.py:429`; `target_lang` in the batch key `batch_primitives.py:24-31`; memory
`multilingual-checkpoint-next-target` = SHIPPED). v2 only says "prompted/non-prompted if enabled" inside 0.2. If the
native runtime must serve the multilingual checkpoint, the prompt-conditioning + per-language batch grouping +
larger/multiple att-context is a whole additional surface; if not, that must be stated as an explicit non-goal.
- **Recommendation:** Add a scope line: **v1 native runtime = EN 0.6b only**; multilingual is a later, separately-scoped
  phase (or explicit non-goal). Don't leave it implicit.

## [MAJOR] 5. Observability/metrics parity is needed from Phase 1, not Phase 5 — the benchmarks depend on it
The success metrics (the keep-up sweep, `vad_stop_recv_to_process`, finalize timing) are computed from the server's
**own timing instrumentation** (`_continuous_finalize_timing` fields ~`server.py:6594-6607`; lock_wait etc.). The native
runtime cannot be compared on `bench_prod_sweep.sh` unless it emits the **same metric names/semantics**. v2 defers
"metrics parity" to 5.1 — but you need it in Phase 1/4 to even measure the thing the whole project is justified by.
- **Recommendation:** Move "emit the benchmark-critical timing metrics with identical semantics" into Phase 1 (a
  cross-cutting requirement), not Phase 5. List the exact fields the sweeps consume.

## [MAJOR] 6. "Cheap Phase-0 spikes run in parallel with the Python plan" undersells 0.2/0.6/0.1 — by the time they're done you've paid most of the cost
v2's §0 says Phase-0 spikes are "cheap" and parallelizable, gating only Phase 1. But Spike 0.6 (reimplement the RNNT
frame-looping decode byte-exact + solve the Blackwell cuda-graph-decoder), 0.2 (libtorch export across all geometries),
0.8 (native preprocessor), and 0.1 (a real-workload overlap harness in C++/Rust) are each substantial — collectively
they ARE most of Phase 1's core. The "worth-it gate" then fires *after* the expensive learning is sunk.
- **Recommendation:** Either (a) sequence the *order* of spikes so the cheapest kill-shots run first (0.1 overlap/MPS and
  0.3 py3.13t are the cheapest thesis kills; 0.5 trace-sim is cheap; do these FIRST and let them gate the expensive
  0.2/0.6/0.8), or (b) drop the "cheap" framing and give each spike an explicit effort estimate so the gate is honest.

## [MAJOR] 7. No effort/staffing/calendar sizing — so the §0 worth-it gate is unevaluable
A from-scratch Rust+C++ persistent runtime with a hand-written RNNT decode, native preprocessor, CUDA-graph management,
continuous batching, a drop-in WS protocol, multi-platform (incl. aarch64) bring-up, and observability parity is plainly
a multi-person-month-to-person-year effort. The plan has zero sizing. You cannot run a "worth-it" gate that weighs this
against the bounded Python plan without an order-of-magnitude cost on both sides.
- **Recommendation:** Add a rough sizing table (per phase: eng-weeks + risk) and an explicit staffing assumption, and
  put the Python-plan effort beside it in §0 so the gate is a real comparison.

## [MAJOR] 8. The 6–10 ms finalize is now correctly deferred to "fusion" — but fusion (B2 TensorRT / B3 kernels) has NO spike, NO geometry analysis, and the roofline says it's hard at M≈7
v2 demotes 6–10 ms to "a separate fusion milestone (B2/B3) with its own proof" — good — but then leaves B2/B3 as one-line
table rows with no Phase-0 viability probe, while the roofline is explicit that the encoder is 24 sequential tiny-M
layers where fp16 was *slower* and "precision only helps once you fuse" (`roofline-COMBINED.md:20-23`). Fusing a
cache-aware streaming Conformer with rel-shift attention (`multi_head_attention.py:259-270`) at M≈7 into TensorRT, with
the streaming cache contract intact, is a research-grade task, not a milestone.
- **Recommendation:** If 6–10 ms (and thus the headline finalize win) matters, add a **fusion viability spike** that
  estimates achievable finalize ms from a TensorRT/fused prototype of *one* conformer layer at M≈7 before promising it
  anywhere. If that spike isn't funded, state plainly that the realistic B1 finalize target is **parity-to-modest**, and
  the headline "22→6–10 ms" is aspirational/out-of-scope-for-v1.

## [MINOR] 9. Spike 0.5's trace data has an upstream dependency
"Feed real per-session readiness timestamps…" requires instrumenting the current server to log those traces first (batch
key, ready predicate, finalize state, lane affinity per tick). That's a small prerequisite task on the *Python* side.
- **Recommendation:** Add "instrument current server to emit per-tick readiness traces" as the first action of 0.5.

## [MINOR] 10. Coexistence/shadow-traffic for T1 validation is absent until 5.2
The cleanest T1 validation in the real world is **mirror/shadow traffic** (tee live audio to the native runtime, diff
its event stream vs Python without serving its output). v2 only integrates at 5.2.
- **Recommendation:** Add a shadow-traffic harness as a validation option in Phase 4 (diff native vs Python on live or
  replayed sessions) — it de-risks T1 far better than corpus replay alone.

## Top 5 things to fix
1. **Make "no shared mutable state on the hot path" an explicit invariant** — `drop_extra`/att-context/keep_all_outputs
   must be pure parameters (`cudagraph_encoder.py:56-64` proves today's global mutation); otherwise shared-weights
   density is unsafe.
2. **Add Spike 0.8 native preprocessor byte-exactness** (constant cuFFT plan + ring buffers + finalize multi-pass)
   *before* the encoder spike means anything; make audio-ingest/ring-buffer a real step.
3. **Reorder Phase 0 so the cheap kill-shots (0.1 overlap/MPS, 0.3 py3.13t, 0.5 trace-sim) run BEFORE the expensive
   ports (0.2/0.6/0.8)**, and drop the "all spikes are cheap" framing; add per-phase effort sizing so §0 is evaluable.
4. **Promote the finalize/reset/resume state machine + multilingual scope** to explicit decisions (state-transition
   table + fuzz test; EN-only-v1 statement), and move **metrics-parity into Phase 1**.
5. **Either fund a fusion-viability spike or stop implying 22→6–10 ms** — with B1 the honest finalize target is
   parity-to-modest; 6–10 ms is research-grade at M≈7.
