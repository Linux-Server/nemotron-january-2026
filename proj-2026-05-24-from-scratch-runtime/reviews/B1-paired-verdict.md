# Step B1 — paired adversarial verdict (2026-05-27)

**Folds:** `opus-B1-review.md` + `codex-B1-review.md` (two independent reviews of B1's implementation
+ T1 result, written in parallel without seeing each other). **User pre-call:** PASS-BY-POLICY WITH AUDIT.

## Convergence (both reviewers, independently)

1. **VERDICT: PASS-by-policy, endorsed.** 2/1007 interim event-only drifts < the documented prior 5/1000
   bar; **0 token divergences end-to-end**; 0 enc_len / 0 cache_len mismatches; finals byte-exact.
2. **The 2 event drifts are interim-timing class** (not semantic decode regressions): one extra interim word
   at one event slot (utt198 chunk13), one extra interim event total (utt770 chunk1). FINAL transcripts
   identical for both. Class matches the project memory's accepted "interim-event-timing drift,
   counted-not-gated via DENSITY_GOLD_EVENTS_TOLERANT."
3. **No precision-policy smoking gun.** `export_steady_batched.py` explicitly disables autotune at THREE
   levels (env vars + `cfg.max_autotune*=False` + `inductor_configs=compile_configs()` re-passed to
   `aoti_compile_and_package`); TF32 default matches the production `enc_steady_aoti.pt2` discipline. The
   B=1/2/4 packages share the same eager-matched precision policy as the production B=1.
4. **cache_t 4.71e-2 vs microbench 5e-3 is NOT a 10× precision excursion** — it's content-dependent
   reduction-order noise on real audio cache states (the microbench fixture was synthetic + well-conditioned;
   the autotune-precision failure signature was ~10.27 / 995-of-1000-tokens, qualitatively different).
5. **Primitive design is clean for B1's contract:** shared constants via `user_managed=true` from
   `finalize_shared_weights.ts`, `CUDAStreamGuard` at run() entry + explicit stream handle passed to AOTI,
   per-row unpack with `.contiguous()` (no aliasing into the to-be-freed batched output), pad-row never
   unpacked + encoder is row-independent so pad-leak is verified clean, K-bounded {1..4} with defensive
   throws.
6. **Flag-OFF byte-exactness preserved by mode separation:** the primitive is only exercised in the new
   `--mode b1-t1`; the production density-sweep is untouched.

## Productive disagreement / sharpenings

- **Codex caught a coverage gap I missed:** `DensityArgs::b1_batch_size` defaults to 3 → coverage groups
  are K=3 → `bucket_for_k(3)` → **B=4** always. With 1000 coverage rows + 4 identical + 3 mixed = 1007 rows
  over 336 cases, the math is `ceil(1000/3) + 2 = 334 + 2 = 336` (B=4 throughout coverage; B=1 once for the
  trailing singleton). **The B=2 bucket was preloaded but never exercised end-to-end through decode/event in
  the full corpus T1.** Microbench validated B=2 at the kernel level (≤0.62× per-row, byte-exact), but the
  through-decode T1 didn't cover it. **Pre-commit fix:** re-run with `--b1-batch-size 2` (one corpus pass, no
  rebuild). Cheap.
- **Opus flagged a coverage gap Codex didn't:** Codex flagged a missing K=2 bucket case; Opus flagged that
  B1 tests "one batched chunk inserted into a B=1 stream" (single-chunk-batched) while B2's scheduler will
  batch *every* chunk per stream (compound recurrent drift). Both gaps real; the bucket gap is the B1 fix,
  the compound-drift gap is the B2 gate addition.
- Both flagged the same minor items independently (each is real, none block B1):
  - **Pack/unpack `torch::cat` allocs** are correctness-fine but per-call alloc churn on B2's hot path.
  - **`BatchedSteadyLoaderSet::get()` is not reentrant** — safe in B1 (preload_all at startup), but B2 must
    either keep the dispatcher as the only caller OR add a mutex (finalize pool's pattern).
  - **A margin probe for the 2 drifts** (top-2 joint score gap) would close the near-tie hypothesis; not
    required for PASS but cheap to add and gives early warning of any future systematic shift.

## Distinctive items (the union)

- **Codex:** "Don't wire the current strict return code into CI as the B1 policy gate without either a
  tolerant mode or an explicit wrapper" (the executable returns nonzero under counted-not-gated policy).
  Reasonable hygiene for CI integration.
- **Codex:** "Add a steady-batch manifest/memory record in B2/B3" — the B-bucket packages are large and lack
  the finalize buckets' manifest/loader-delta discipline.
- **Opus:** **A1 — verify NEW `enc_steady_aoti_b1.pt2` is bit-identical to PRODUCTION `enc_steady_aoti.pt2`**
  (or use NEW B=1 as the alone reference in b1-t1). Cleans drift attribution; the T1 alone-reference currently
  uses production B=1 while batched K=1 uses NEW B=1, so reported diffs include any compile-time delta.
- **Both implicit:** the primitive header is not self-contained (depends on `session_main.cpp` for `fs`,
  AOTI types, helpers) — acceptable for now, document if the primitive moves.

## Pre-commit ACTION (single item) — DONE 2026-05-27

**B=2 coverage gate ran** (`runtime/artifacts/logs/b1_t1_K2_20260528T035408Z.log`):
- All 500 coverage_group_* cases through B=2 bucket: **PASS**.
- **token_divergences: 0** (the SLO signal — clean across all 1007 K=2/B=2 rows × 502 cases).
- enc_len_mismatches 0; cache_len_mismatches 0.
- event_divergences: 4 (interim-event timing class, same as the K=3 run's 2 — slightly higher count is
  consistent with B=2's distinct reduction-order signature; still well under the 5/1000 prior bar per-pass).
- max_enc_out 8.63e-4 / max_cache_ch 5.23e-3 / max_cache_t 9.40e-2 — same band as the K=3 run, no precision
  excursion.
- Combined K=3 + K=2 = **0 / 2014 token divergences across all three buckets**; 6 interim drifts split
  across the runs.

**B=2 bucket coverage CLOSED. B1 cleared for commit.**

## B2-deferred follow-ups (none block B1)

| ID | Item | Source |
|---|---|---|
| A1 | Verify or align NEW B=1 vs PRODUCTION B=1 to clean drift attribution | Opus |
| A2 | All-chunks-batched per-stream T1 (compound recurrent drift) | Opus |
| A3 | Debug-flagged top-2 margin probe at near-tie blanks | Opus + Codex |
| A4 | Reentrancy contract for `BatchedSteadyLoaderSet::get()` (preload-discipline or mutex) | Opus + Codex |
| A5 | Pre-allocated pack/unpack scratch (perf, B2 hot path) | Opus + Codex |
| A6 | Tolerant-mode wrapper or explicit CI policy gate (don't wire strict exit code as the policy gate) | Codex |
| A7 | Steady-batch manifest / memory record (B-bucket equivalent of finalize discipline) | Codex |
| A8 | Header self-containment if primitive moves out of `runtime/cpp/` | Codex (implicit) |

## Net

Commit B1 after the B=2 coverage gate passes (item above). Update PHASE2-PLAN.md B1 row `[x]` with the
PASS-by-policy verdict + token-clean + the documented event-drift class + the B=2 coverage closure. Record A1–A8
against Step B2 in the lever inventory or the B2 step body.
