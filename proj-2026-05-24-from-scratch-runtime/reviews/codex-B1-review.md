# Codex B1 Paired Review

Verdict: **PASS-by-policy on the B1 evidence, with one cheap coverage flag before B2**. The strict executable returned
FAIL because it gates two interim-event timing drifts, but the project policy already accepts this class when finals and
tokens are exact. B1 is better than the prior accepted L40S density bar: 2/1007 event-only drifts vs 5/1000, with
0 token divergences, 0 `enc_len` mismatches, and 0 `cache_ch_len` mismatches. I do not see evidence of a precision-policy
bug. The cheap pre-B2 fix is to add/run an explicit K=2/B=2 real-corpus decode/event case; the logged full run did not
exercise the B=2 package through the end-to-end path.

## Findings

1. **Coverage gap: B=2 is not covered by the logged end-to-end B1 T1.**
   `DensityArgs::b1_batch_size` defaults to 3, and the full result was 1007 rows over 336 cases. Subtract the two explicit
   cases (4 identical + 3 ragged rows) and the coverage portion is 1000 rows / 334 cases, exactly `ceil(1000/3)`, not
   B4 grouping. That exercises B4 for K=3 and B1 for the final singleton, plus explicit B4 cases; B2 is only preloaded and
   covered by the synthetic microbench. Since `bucket_for_k(2)` selects the distinct B=2 package, add an explicit
   two-row real-corpus case or run a bounded/full `--b1-batch-size 2` pass before B2.

2. **The executable policy is still strict-fail.**
   `run_b1_t1_gate` folds `event_divergences == 0` into `ok`, so the current `b1-t1` binary exits nonzero under the
   accepted counted-not-gated policy. That is fine for this manual audit if the commit notes say "PASS by policy", but
   do not wire this exact strict return code into CI as the B1 policy gate without either a tolerant mode or an explicit
   wrapper.

3. **`BatchedSteadyLoaderSet::get()` is not reentrant.**
   B1 calls `preload_all()` single-threaded, so this is not a B1 correctness bug. For B2, either keep the batching
   dispatcher as the only caller and guarantee startup preload, or put a mutex around lazy load/map access. The finalize
   pool already uses that pattern.

## Primitive Audit

Shared constants discipline looks correct for B1. `finalize_shared_weights.ts` is built from encoder parameters and
buffers under `encoder.*`, and `constants_for_bucket` fail-closes on missing FQNs with the `encoder.*`/`e.*` alias fallback.
Because `preload_all()` loaded B=1/2/4 without throwing, the constant set is complete for these packages. `user_managed=true`
is the right runtime contract for a single shared constants set. I would still add a steady-batch manifest/memory record in
B2/B3 because the steady packages are large and do not have the finalize buckets' manifest/loader-delta discipline.

Packing is correctness-safe but not yet a production hot path. At B=4, `torch::cat` copies about 29 MiB of cache inputs
per batch, unpack copies about the same back into row-shaped tensors, and `apply_encoder_outputs_density` clones the cache
again into session state. That is small versus HBM bandwidth but can show up as allocator/device-copy churn at density.
Preallocated pack/unpack workspaces are a performance cleanup, not a B1 blocker.

Unpack indexing is correct: `enc_out`/`enc_len`/`cache_ch_len` select batch dim 0, `cache_ch`/`cache_t` select batch dim 1,
then restore row-shaped tensors with `unsqueeze`/`reshape` and `contiguous`. The duplicate-row-0 pad policy is also correct
for this model class: pads are never unpacked, and the explicit identical B4 plus ragged K3 padded tests did not show any
real-row leakage. K=0 and K>4 throw; B2's scheduler must split larger ready sets.

The stream guard is placed at the right boundary. It covers pack allocations, AOTI `run(inputs, stream)`, and unpack copies;
the explicit stream handle is still passed into AOTI. That is sufficient for B1. The header is not self-contained, though:
it relies on `session_main.cpp` for `fs`, file helpers, AOTI type aliases, and shared-constant helpers. Acceptable for the
current include style, but make that dependency explicit if the primitive moves.

## Gate Audit

`prepare_b1_row_until_target` sets up the target correctly. It resets a fresh session, replays chunks before the target
through the production B=1 path, validates steady geometry, builds the target chunk, and computes `alone_out` from the
same pre-target state without applying it. That leaves the session ready to consume the batched target.

Ignoring the return value of `apply_encoder_outputs_density` in `apply_b1_target_chunk` is safe. The return is scalar-sync
timing telemetry only; the function mutates `state.clc`, `state.clt`, `state.clcl`, decoder state, and `state.hyp`.

`finish_b1_row_from_target` really continues from the batched cache state. After applying `target_out`, subsequent
`run_steady_chunk_density` calls consume `prepared.session.clc/clt/clcl`; there is no B=1 recomputation of the target that
would erase the batched effect. The final token checks therefore test propagation through later B=1 steady chunks and
finalize.

`build_serial_reference` is the right oracle for B1: production B=1 AOTI steady plus the existing finalize bucket path. In
the logged run it built 1000/1000 references with strict gold events and finals intact, so the unchanged B=1 path is not
contaminated by the primitive.

## Event Divergences

I did not add new instrumentation because this review was doc-only and the task forbids implementation edits. Classification
is therefore by pattern and magnitude, not by direct top-2 score margins.

The two drifts are the accepted interim-timing class:

- `utt198 chunk13`: batched interim contains one additional word, while final and steady token sequences end equal.
- `utt770 chunk1`: batched emits "A grid" where the reference first emits "A", with one fewer interim event overall.

This is a timing/coalescing difference in the interim stream, not a semantic decode regression. The final tokens are exact,
the steady tokens are exact at replay end, and event differences are localized to interim text cadence. The count is also
below the accepted prior density bar: 2/1007 vs 5/1000.

The missing margin probe is still worth doing. Add a debug-only label filter around `decode_range_density` that logs
top-1/top-2 margin, top IDs, emitted-token index, `utt`, `chunk`, and case label for `utt198/chunk13` and `utt770/chunk1`.
If the margins are tiny, it closes the loop. If margins are not tiny, revisit precision and B export policy before B3.

## Cache-T Magnitude

The microbench `cache_t` max of about `5e-3` was a synthetic fixed-cache case. The B1 full run's `4.711e-2` is not a
10x accumulation of repeated batched steps; each B1 coverage row applies exactly one batched target after a B=1-prepared
history. The larger number is consistent with corpus-diverse recurrent cache state and a larger real `cache_t` dynamic
range. Prior default AOTI drift evidence also saw bounded real-stream `cache_t` maxima up to `1.154e-1` over 830 chunks,
while tokens stayed exact.

This is far from the known precision-policy failure signature: autotune/precise-FP32 failures were around `cache_t=10.27`
and produced 995/1000 token divergences. B1's `4.711e-2`, with zero token divergences, looks like batch/reduction-order
noise inside the accepted T1 envelope.

## Precision Policy

`export_steady_batched.py` explicitly disables Inductor autotune via env and `inductor_configs`:
`max_autotune=False`, `max_autotune_gemm=False`, `max_autotune_pointwise=False`, and
`coordinate_descent_tuning=False`. The compile logs for B=1/2/4 confirm autotune OFF and show the same TF32 warning seen
in the production `enc_steady_aoti.pt2` compile. The production B=1 compiler (`aot_compile.py`) does not set explicit
precision globals either; it uses the default non-autotuned AOTI path. So I do not see a batched-export precision mismatch.

What is missing is artifact discipline, not a correctness fix: log `torch.get_float32_matmul_precision()`,
`torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, package SHAs, EP SHAs, and the exact
`inductor_configs` into a steady-batch manifest. In `--compile-only`, the script has no eager self-check, so the manifest
plus B1/B3 package-specific T1 are the traceability mechanism.

## Net

Do not HOLD B1 on the two strict event mismatches. Under the documented density policy, this is **PASS-by-policy**:
0 final/steady token divergences, exact encoder/cache lengths, accepted interim-event timing drift, and no precision-policy
smoking gun. Before folding into B2, add the cheap K=2/B=2 real decode/event case and record the event drift as
counted-not-gated in the commit/plan update.
