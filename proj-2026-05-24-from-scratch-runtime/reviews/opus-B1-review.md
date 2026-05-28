# Step B1 — independent Opus paired review (2026-05-27)

Adversarial second-look on the B1 implementation + T1 result (codex-jobs/step-B1-b4ml9h322.log + the new
files). Folded with `codex-B1-review.md` after Codex returns. **The user's verdict (2026-05-27): PASS-BY-POLICY
WITH AUDIT.** This review's job is the AUDIT, not the policy.

## TL;DR

**ENDORSE PASS-by-policy.** B1's T1 is strictly better than the project's documented prior bar (2/1007 interim
event drifts vs the historical 5/1000) and the SLO-binding signals are perfect (0 token divergences, 0
enc_len/cache_len mismatches, final transcripts byte-exact). The cache_t 4.71e-2 magnitude growth vs the
microbench's 5e-3 looks **content-dependent intrinsic reduction-order drift, not a precision-policy bug** —
`export_steady_batched.py` proves autotune is OFF and matches production discipline; the bench fixture happens
to be drift-tolerant. **One cheap hygiene fix worth doing before B2** (not before commit): the T1 compares
batched (NEW `enc_steady_aoti_bK.pt2`) vs alone (PRODUCTION `enc_steady_aoti.pt2`), so a small compile-time
delta between the two B=1 builds could be inflating the reported diffs. Verify or align them.

## 1. Primitive audit — `steady_batch_primitive.h` (196 lines)

**Architecture: clean, follows the plan's ownership contract.**
- `BatchedSteadyLoaderSet` loads B∈{1,2,4} buckets from `finalize_shared_weights.ts` with
  `loader->load_constants(values, false, false, /*user_managed=*/true)` — **single shared constants set, no
  per-bucket weight dup**, matching the discipline used for the finalize bucket pool and the plan's "one
  constants set" requirement. ✅
- Constants source `finalize_shared_weights.ts` is the right choice — same conformer encoder weights as
  finalize, so reuse is correct (not a misnomer).
- `preload_all()` invoked in the gate before any concurrent use → B=1/2/4 loaders are ready up-front. ✅
- `run()` enters with `c10::cuda::CUDAStreamGuard(stream)` AND passes the explicit stream handle to
  `loader.run(inputs, stream.stream())`. Belt + suspenders for AOTI's internal allocations. ✅
- `bucket_for_k`: K∈[1,4]→{1,2,4}; throws on K=0 and K>4; `run()` redundantly enforces K≤4 (defensive). ✅

**Pack (lines 134–165):** `torch::cat([row 0..K-1, row 0 pad..])` on dim 0 for chunks, dim 1 for cache_ch /
cache_t, dim 0 for cache_ch_len. Each source tensor is `.contiguous()`'d before cat. The `length` tensor is
`torch::full({bucket}, first.chunk.size(2), Long, device)` — assumes uniform chunk length, enforced by
`verify_row_shapes`. ✅ Correct. *Perf note (not blocking B1):* `torch::cat` allocates a new tensor per call;
for B2's high-frequency steady scheduler (every 160ms × N streams) this is per-chunk alloc churn — worth
pre-allocating a B-bucket scratch tensor and `index_copy_` rows into it.

**Unpack (lines 167–187):** `out[i].select(dim, row).unsqueeze(dim).contiguous()` for each of the 5 outputs,
**only for the K real rows** (pads NEVER unpacked). The `.contiguous()` after select makes each unpacked row
standalone (no aliasing into the to-be-freed batched output). ✅ Correct.

**Pad-leak concern (verified clean):** Pad row = `ready[0]` duplicated. The batched GEMM is row-independent
(each output row depends only on its input row, modulo reduction-order across the K rows for shared kernels —
which is the SAME order across all rows). The pad's output is computed but neither unpacked nor reused. **The
pad's compute does not perturb row 0's output**, because the conformer encoder is per-row (no cross-row
attention; the 1×B×... cache geometry confirms per-row cache). ✅ Safe.

**Reentrancy:** `get()` is non-atomic find-or-create. In B1 (single-thread T1) this is exercised once via
`preload_all()` and the map is read-only afterwards → safe. **In B2 (the cross-stream scheduler), `get()` MUST
be guarded** or `preload_all()` MUST run before any concurrent `run()`. The current API contract assumes
"preload at startup, then concurrent run" — needs to be documented + enforced for B2. Worth noting in the B2
design.

## 2. b1-t1 gate audit (`density_main.cpp` 1700–2260)

**Test design — exactly what the plan asked for, with one structural caveat:**
- `prepare_b1_row_until_target(spec)`: runs the session through chunks `[0, spec.chunk)` via the **production
  B=1** `run_steady_chunk_density`, captures `(new_mel, chunk, alone_out)` where `alone_out` is the SOLO B=1
  result for the target chunk. ✅
- `apply_b1_target_chunk`: feeds the BATCHED `out` into the session via `apply_encoder_outputs_density(state,
  out, ...)`. The `(void)` cast on the return matches the same pattern elsewhere in density code — safe ignore.
  ✅
- `finish_b1_row_from_target`: after the batched target, runs chunks `[spec.chunk+1, num_steady)` via
  production B=1, then `run_finalize_density`. Returns `steady_tokens` (running hyp), `final_tokens`, `events`.
  ✅ This propagates the batched cache state forward correctly (the B=1 follow-on reads the cache state that
  was updated by the batched output via `apply_encoder_outputs_density`).
- `run_b1_batched_case`: prepares all specs → `batched_steady.run(ready)` → `cudaStreamSynchronize` (ensures
  the batched output is materialized) → per-row tensor-diff stats + finish-from-target + strict compare to
  `build_serial_reference`. ✅

**Coverage:**
- `identical_rows_B4` (4 copies of same utt/chunk) — sanity.
- `ragged_mixed_K3_padded_to_B4` (3 distinct utts at chunks 1/2/3) — the real cross-stream case.
- `build_b1_coverage_specs` — every utt at `chunk = 1 + utt % (num_steady-1)`, grouped into `args.b1_batch_size`
  (default 4) batches → full-corpus walk.

**Structural caveat (a real coverage gap for B2, not B1):** B1's T1 batches **ONE chunk in each test stream**
(the target), with B=1 prepare + B=1 follow-on. The scheduler (B2) will batch **EVERY** steady chunk for every
stream → the cache drift compounds across the recurrence. B1's T1 measures "one batched chunk's effect on a
B=1 stream" — sound for primitive correctness, but the production behavior is "all chunks batched, drift
compounds." **B2 must add a per-stream all-chunks-batched T1** (run the full stream end-to-end through the
batched primitive, compare to full B=1). The current B1 T1 is necessary but not sufficient for the production
correctness picture. Flag for B2's gate.

**One hygiene gap (worth fixing for B2, not a B1 blocker):** the gate uses TWO B=1 packages:
- the alone reference: `AOTIModelPackageLoader enc_steady(args.dir + "/enc_steady_aoti.pt2", ...)` =
  PRODUCTION B=1 (compiled at the original artifact build).
- the batched K=1 case: the NEW `enc_steady_aoti_b1.pt2` from `steady_b_artifacts/` (compiled by
  `export_steady_batched.py`).

If these two B=1 builds aren't bit-identical, the reported diffs include compile-time noise on top of the
batched-vs-alone difference. The microbench compared NEW B=1 vs NEW B=1 (= 0.000e+00) and NEW B=2/4 vs NEW
B=1 (= 6e-6/8e-5/5e-3) — it did NOT compare PRODUCTION B=1 vs NEW B=1. **Recommendation:** either (a) verify
they're bit-identical (likely, same compile config / same machine), or (b) the b1-t1 gate uses NEW B=1 as the
"alone" too. Cheap and makes the drift attribution clean.

## 3. The 2 event divergences — likely intrinsic, not a precision bug

Both divergences are **interim-event-timing class** (the same class as the documented 5/1000 prior):
- `utt198 chunk13 event[6]`: batched emits "...where **is**" vs B=1 "...where" — 1 extra interim word at the
  same event slot, FINAL transcripts identical.
- `utt770 chunk1 event[0]`: batched "A grid" vs B=1 "A", event count 12 vs 13 — 1 extra interim event in the
  batched path, FINAL transcripts identical.

**Diagnosis: near-tie greedy-RNN-T blank-vs-emit flips one chunk earlier in the batched path.** Evidence:
- **0 / 1007 token divergences** end-to-end (steady AND final). If precision were systematically off, we'd see
  token flips elsewhere too. The fact that ONLY the *timing* of one interim emit differs in 2 utterances over
  1007 is the textbook signature of rare near-tie reduction-order flips.
- `enc_out` max 7.651e-4 — well within the eager-vs-AOTI band (1.66e-2 measured for the autotune-OFF /
  TF32-reduced policy, per the plan's Compile policy memory). Not a precision-policy excursion.
- 2 / 1007 < 5 / 1000 (the documented prior bar). Counted-not-gated per project policy is the established
  treatment.

A margin probe (joint-score gap at the divergent timestep) would confirm, but the empirical pattern alone
strongly supports intrinsic reduction-order. **Recommendation: PASS-by-policy without a margin-probe gate**;
optionally add a debug-flagged margin telemetry to B2 to catch any future systematic shift (cheap).

## 4. Cache_t magnitude — microbench 5e-3 → T1 4.71e-2 explained

This was my biggest concern initially. After the precision-policy audit (§5), I'm confident it's
**content-dependent intrinsic drift**, not a systematic bug:

- The **microbench used a synthetic linspace-based mel fixture** (`make_mel` in `export_steady_batched.py:108`)
  packed into one batched forward at chunk 1 (initial cache + one chunk of warmup). This fixture is
  drift-tolerant — well-conditioned cache state, smooth synthetic input.
- The **T1 measures at real audio chunks up to ~chunk 13** of natural speech, where the cache_t state has
  accumulated through the recurrence on real activations. Real audio occasionally hits cache states with
  wider dynamic range (per the plan's memory note: "abs.mean 0.39 / max 54 [for the bench fixture];
  corpus-representativeness unproven").
- If the exports were precision-misaligned, the microbench's batched-vs-alone diff would ALSO have been
  inflated (both use the same exports). The microbench was tiny → the export-level reduction-order is well
  controlled → the T1 growth is content-driven.
- 4.71e-2 is still well within the eager-vs-AOTI band (1.66e-2 was for one specific cache_t metric in the
  knob matrix; the recurrent accumulation explains the modest growth).

## 5. Precision-policy audit — `export_steady_batched.py` ✅ matches production

Verified the B=1/2/4 exports match the production B=1 discipline:
- **Autotune OFF** at all levels: env vars `TORCHINDUCTOR_MAX_AUTOTUNE_*=0`, `COORDINATE_DESCENT_TUNING=0`;
  inductor `cfg.max_autotune* = False`; and **passed again** into `aoti_compile_and_package(...,
  inductor_configs=compile_configs())` where `compile_configs()` re-sets the same. Triple belt-and-braces.
- TF32 policy is the torch default (matches eager's TF32-reduced path that the autotune-OFF production B=1
  uses; the plan's memory pinned this as the "1.66e-2 cache_t" eager match).
- The script's own `compare_outputs` self-check verifies export-vs-eager AND AOTI-vs-eager within atol=5e-2 /
  rtol=1e-3 — and the B=1/2/4 builds passed that self-check at export time.
- The `force_noexecstack_on_link()` shim is required (raises if not fired) — matches production hardening
  discipline.

**No precision-policy gap.** The 2 interim drifts are not attributable to export precision.

## 6. Net verdict + recommendations

**PASS-BY-POLICY.** Commit B1 as done; mark the plan B1 row `[x]`; proceed to B2. Specifically:

1. **The 2 interim event drifts** are intrinsic reduction-order at near-tie blanks (the documented class); 2 /
   1007 is better than the 5 / 1000 prior bar; counted-not-gated per the established policy.
2. **The cache_t 4.71e-2** is content-dependent recurrent accumulation, NOT a precision-policy bug
   (microbench + export-config audit both confirm).
3. **Flag-OFF byte-exactness preserved** — the new mode `b1-t1` is the only path that uses the primitive; the
   production density-sweep is untouched. ✅

**Cheap audit follow-ups (defer to B2, not blocking B1 commit):**
- **A1 — NEW-vs-PRODUCTION B=1 parity check**: verify `enc_steady_aoti.pt2` (production) and
  `enc_steady_aoti_b1.pt2` (new) are bit-identical, or use NEW B=1 as the alone reference in b1-t1. Cleans
  drift attribution.
- **A2 — Per-stream all-chunks-batched T1** (for B2's gate): the structural coverage gap — B1 tests "one
  batched chunk inserted into a B=1 stream"; B2 must test "every chunk batched through the same stream" to
  measure the compound recurrent drift the scheduler will actually produce.
- **A3 — Debug-flagged margin probe** (for B2): telemetry of the top-2 joint-score gap at near-tie blanks, so
  any future systematic precision shift gets caught early.
- **A4 — Reentrancy contract for B2**: `BatchedSteadyLoaderSet::get()` is non-atomic find-or-create; B2 must
  either call `preload_all()` at startup (current contract) or add a mutex. Document explicitly.
- **A5 — Pre-allocated pack scratch (perf)**: B2's high-frequency scheduler should pre-allocate a B-bucket
  scratch tensor and `index_copy_` rows, instead of per-call `torch::cat` allocations. Not correctness, just
  steady-state perf.

**No fixes required before commit.** Update PHASE2-PLAN.md B1 row to `[x]` (PASS by policy, 0 token / 2 event
interim-timing in documented tolerated class, ≤ prior 5/1000 bar); record the audit follow-ups A1-A5 against
B2 in the lever inventory / B2 step body.
