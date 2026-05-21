# Claude pre-implement review — proj-2026-05-21-0410/PLAN.md (2026-05-21)

Fourth-pass review after the captured three-box GPU-vs-host split (RESULTS.md). The plan is sound,
probe-gated, byte-exact-gated, fail-closed, flag-gated. No NO-GO. Findings below; verdict GO-WITH-FIXES.

## CRITICAL
(none — the cache-axis correctness, fail-closed grouping, compile×batch mutual exclusion, and fork-lane
ownership are all correctly specified and probe-backed.)

## SHOULD-FIX

### S1. Preprocess + ring-updates stay PER-STREAM SERIAL → the likely next bottleneck after the model is batched. (Steps 7/8/10)
The scheduler batches ONLY `conformer_stream_step`. `_process_chunk` (server.py:2437) ALSO does, per
session per tick: `_build_fixed_preprocess_audio` + `_preprocess_fixed_audio` (STFT/mel) + mel/raw ring
updates + emitted_frames. The "preprocess is negligible ~1ms/9%" framing is a **B=1 framing** and it
misleads: after the model collapses to ~1 call/tick, the un-batched preprocess becomes a serial tax that
scales with B. Back-of-envelope at B=8 on a slow cloud CPU (preprocess ~2-4ms/stream): N×preprocess ≈
16-32ms/tick vs one batched model call ~14ms → **preprocess dominates and caps the knee well below the
model-batched ceiling.** This squares with the new (b) finding: the server-layer term (preprocess +
orchestration) is ~21ms on Modal, co-equal with the model step — batching the model alone leaves half the
cost in place on the slow-CPU target.
**Recommendation:** (a) Add a measurement sub-step early in Phase 2: quantify per-stream preprocess +
ring-update cost on the smallest/slowest target (L4) at B=1, and project the post-model-batch knee bound.
(b) IF it's the bottleneck, batch the preprocess too — stack N `fixed_audio` → `[B,samples]` → one
`_preprocess_fixed_audio` (the cuFFT plan is over n_fft/hop, NOT batch, so the constant-plan invariant
holds). BUT this needs its OWN byte-exactness probe first: batched-B STFT vs B=1 STFT, bit-for-bit per row
— cuFFT may pick a different batched plan ([[cufft-stft-plan-size-nondeterminism]]); if not byte-exact,
keep preprocess serial and instead bound MAX_SIZE / document the cap. Don't silently ship a model-batched
server whose knee is preprocess-bound and call it "8-10×".

### S2. `stack_pred_out` all-or-nothing `None` is safe only by an IMPLICIT grouping invariant — assert it. (Step 5)
`batch_primitives.stack_pred_out` returns `None` for the WHOLE batch if ANY row's `previous_pred_out` is
None. That is correct ONLY because grouping currently separates first-chunk (drop_extra=0, chunk_T=16,
pred_out=None) from steady (drop_extra=self.drop_extra, wider chunk_T, pred_out set), so a group is
uniformly fresh or uniformly established. If a future grouping-key change ever mixed a fresh row with an
established one, stack_pred_out would return None → **silently restart the established rows' RNNT decode**
(corruption the byte gate might miss if rare). **Recommendation:** add a hard assertion in the stack path
that within a group `previous_pred_out` is uniformly None-or-not (and same for `previous_hypotheses`
None-ness), fail closed. Cheap, closes a latent footgun.

### S3. Scale the byte-exact gate up + add an explicit DISTINCT-clips cross-talk test. (Steps 7, 9)
The dominant risk is cache cross-talk between batched rows. The Step-0 baseline is **8 clips** — too small
to catch a 1-in-1000 crosstalk bug, and the gates don't explicitly state the real cross-talk test: stream
**N DIFFERENT clips concurrently** through the batched server and verify each session's FULL transcript ==
its own B=1 solo transcript. **Recommendation:** Step 9's byte-exact gate should run the 1000-sample
silence0_warm200 set (already in `results.db`) at concurrency ≥ `BATCH_MAX_SIZE` with per-session byte
comparison; and add a dedicated distinct-clips cross-talk test (each row byte-exact vs solo). This is the
single highest-value gate for the plan's stated dominant risk.

### S4. Close the greedy_batch transitivity gap at B≥2 ACROSS a full multi-chunk stream. (Step 7 gate)
Probe B proved batched==separate at B=2/4 using the CURRENT `greedy` (loop_labels=False, serial row
decode). Probe C proved `greedy_batch`==`greedy` at **B=1**. The live scheduler uses `greedy_batch` at
**B≥2 across many chunks** with `partial_hypotheses` threaded per row + max_symbols=10 — the transitive
closure (greedy_batch B≥2 multi-chunk == N separate B=1 greedy) SHOULD hold but was not directly validated
end-to-end (Progress row 3 notes only "B=1, all clips"). Step 7's gate "N same-lang streams byte+state
identical" DOES cover this **iff executed with greedy_batch + multi-chunk + distinct clips** — make that
explicit in the gate so it isn't satisfied by a B=1 or single-chunk run.

### S5. Don't defer the in-server `drop_extra` try/finally exception test. (Steps 5/6)
Rules require `try/finally` restore of `model.encoder.streaming_cfg.drop_extra_pre_encoded` around EVERY
model call; Probe B proved it at the PROBE level, but Progress rows 2 & 5 note the in-server wiring + the
"inject exception mid-batch → next B=1 chunk recovers" test are still TODO. Ensure the Step 5/6 gate
actually exercises this in-server (a poisoned scalar silently corrupts the NEXT unrelated stream's call).

## NICE-TO-HAVE

### N1. Step 4 (encoder compile) is worth MORE on cloud than Probe A's local 1.54× implies.
The (b) split shows idle-gaps are **46-68% of the span on Modal** vs 35% local, and CUDA-graphs collapse
exactly those gaps. The B=1 paths that never batch (solo streams, finalize/fork, first-chunk, fallback)
benefit disproportionately on the slow-CPU target. Worth a one-line note in Step 4/Context so the compile
lever isn't deprioritized based on the local-only 1.54×. (Motivational; no implementation change.)

### N2. Step 10 re-sweep should compare per-tick total-preprocess-ms vs model-ms (ties to S1).
Step 8 telemetry already records preproc ms + model ms. Step 10 should explicitly plot/compare them per
tick to confirm whether preprocess became the post-batch bottleneck — the empirical check on S1.

### N3. Verify the finalize-behind-max-batch worst case before raising MAX_SIZE.
Finalize/fork now shares the single model-call lane; at higher MAX_SIZE a finalize can wait behind a full
batch + its own fork-clone. Already gated (final p95 <400ms, model_lane_wait_ms telemetry) — just confirm
the worst case explicitly at the max batch size before raising MAX_SIZE from 4 → 8/16.

## Strengths (keep as-is)
- Cache-axis (dim1 channel/time, dim0 len) precisely specified + Probe-B validated incl. row-permute &
  mid-stream stack — the historical "batching corrupts cache state" fear is genuinely resolved.
- Compile×batch mutual exclusion (B=1 compiled / B>1 uncompiled unless a separate bucketed probe passes) is
  conservative and correct; respects use_cuda_graph_decoder=False.
- Fail-closed grouping + the explicit GO/STOP decision point after Probe C (don't sink the refactor for
  <1.5× gain) is excellent risk discipline.
- Fork-lane ownership + FORK_ASSERT + telemetry (fork_clone_ms, model_lane_wait_ms) is well thought through.

## Verdict: GO-WITH-FIXES
Fold S1-S5 into the plan (S1 + S3 are the high-value ones), note N1-N3. None block starting /implement at
Step 4; S1/S3/S4 must land before/with Steps 7-9.
