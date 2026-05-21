# Claude direct review — round 2 (PLAN.md v2)

Round-1 fixes look correctly incorporated (cache dim1, greedy_batch+Probe C, flat hyp list, grouping
key, no-pad-not-ready, Step 0, split 5a/5b/5c, fail-closed, latency gates). Deeper issues for round 2:

## A. CRITICAL: `torch.compile(reduce-overhead)` (CUDA graphs) is incompatible with variable batch B
`mode="reduce-overhead"` captures CUDA graphs → requires STATIC input shapes. Phase 1 (Step 4) at B=1
fixed chunk shape is fine. But Phase 2 batching has VARIABLE B per tick → a graph captured at one B
can't run at another; the compiled encoder would recapture per distinct B (thrash) or error. So
**Step 10's "compile × batch both-on" cell is latently broken.** Fix:
- State that Phase 1 compile targets the fixed-B path; with batching, EITHER (a) run the batched
  encoder UNcompiled (batch amortization is already the win — compile is redundant once B>1), OR
  (b) bucket B to a few fixed sizes (pad rows up to the next bucket, mask) so graphs are reused.
- Step 10 matrix: {compile-only (B=1)}, {batch-only}, and {compile+bucketed-batch} — NOT compile+arbitrary-B.
- Default recommendation: ship batch-only (simpler, the bigger lever); treat compile as a B=1/low-load optimization.

## B. The Probe-C-NO-GO fallback (batch encoder, decode per-row) — quantify the ceiling
If `greedy_batch` ≠ byte-exact (Probe C NO-GO), Step 7 decodes per-row. Per-row decode is a Python loop
over B `rnnt_decoder_predictions_tensor` calls → re-serializes the decode. Whether that still wins
depends on the encode:decode split of the ~10ms(local)/~30ms(Modal) step. The local microbench implies
the encoder dominates (decode is small per chunk: greedy, max_symbols=10, ~few tokens/160ms) → per-row
decode is cheap → encoder-only batching still captures most of the win. **Add to Probe B/C: measure the
encode-only vs decode-only ms at B=1** so we KNOW the fallback ceiling before committing. If decode is
>~30% of the step, the NO-GO fallback is weak and Probe C byte-exactness becomes load-bearing.

## C. Gate thresholds need concrete numbers (for autonomous /implement)
- Probe A: "≥20% faster" — specify: steady-state per-step p50 on local 5090, ≥20% AND no recapture.
- Probe C: "materially faster decode" — specify e.g. ≥1.3× decode throughput at B=4, byte-exact at B=1.
- Steps 7/10 knee: "materially > baseline" — specify e.g. ≥2× the batch=1 knee at the same GPU, OR
  "keeps up at ≥2× the baseline N". Make each GATE a number an agent can check.

## D. Scheduler TRIGGER/tick design is under-specified (Step 5a/6)
Specify the dispatch policy concretely: event-driven — when ≥1 session is ready, start a max-wait timer
(`NEMOTRON_BATCH_MAX_WAIT_MS`, default ~ a fraction of shift, e.g. 20-40ms); dispatch the largest safe
same-group batch when (a) max-wait elapses, or (b) batch hits `NEMOTRON_BATCH_MAX_SIZE`, or (c) all
known-ready streams are gathered. At N=1/low load → immediate dispatch (no added wait). This is needed
to bound the TTFS gate (Rule: latency) and to make 5a/5b implementable without guessing.

## E. Throughput win is load-shape-dependent (note as a caveat, not a gate)
With N realtime streams + tight start jitter (the concurrency_test harness), ~N are ready each tick →
batch ≈ N → full amortization. Anti-phased/Poisson real traffic yields smaller per-tick batches → less
amortization. The measured knee (harness) is a best-case-ish; note that production benefit scales with
in-phase concurrency. Not a blocker — but the recommendation (Step 12) should state it.

## F. Memory bound default (Step 8)
Encoder caches are [layers, B, cache_T, d_model] → linear in B; RNNT label-looping allocates batched
buffers. Set a conservative `NEMOTRON_BATCH_MAX_SIZE` default (e.g. 16) + log GPU mem; the 0.6b model is
small so B~16-32 fits 16-24GB, but make it explicit + OOM-safe (split the batch if alloc fails).

## G. Minor
- Step 0 baseline must pin the NeMo commit + model revision in the artifact (cross-env byte-exactness
  is only meaningful same-commit; cuFFT nondeterminism means the gate is same-machine byte-exact or
  WER-within-CI cross-machine — state which; for the LOCAL gate use same-machine byte-exact).
- 12 steps is justified (each a real unit); don't merge. The 3 probes are distinct de-riskers — keep.

## Verdict
v2 is strong. The one materially-new issue is **A (compile×variable-B CUDA-graph conflict)** — Step 10's
matrix needs correcting and the compile/batch relationship clarified (ship batch-only as primary; compile
is a separate B=1 lever or needs B-bucketing). Tighten gate thresholds (C), specify the scheduler trigger
(D), and add the encode/decode split measurement (B). With those, the plan is implementation-ready.
