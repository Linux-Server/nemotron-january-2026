<task>
Adversarial paired review of Step B1 (the batched-steady forward primitive + T1) implemented in commit-pending changes
to proj-2026-05-24-from-scratch-runtime. Skeptical, second-look: find bugs, edge cases, correctness risks, magnitude
anomalies. Write your fold-ready analysis to proj-2026-05-24-from-scratch-runtime/reviews/codex-B1-review.md (a sibling
opus-B1-review.md exists / is being written).
</task>

<context>
**What B1 implements** (PHASE2-PLAN.md Step B1, v3 block):
- runtime/cpp/steady_batch_primitive.h — new header. `BatchedSteadyLoaderSet` loads B in {1,2,4} steady AOTI packages
  with shared constants (user_managed=true via finalize_shared_weights.ts → ONE constants set), exposes `run(ready,
  stream)` that picks bucket≥K, packs rows, runs on the explicit stream via `CUDAStreamGuard`, unpacks per-row.
- runtime/cpp/density_main.cpp — adds `--mode b1-t1`, flag `NEMOTRON_DENSITY_BATCH_STEADY` (default OFF — note: the
  flag is informational telemetry in b1-t1 mode; b1-t1 always exercises the primitive; the normal density-sweep mode
  does NOT use the primitive, preserving flag-OFF byte-exactness by mode separation). New funcs `run_b1_t1_gate`,
  `run_b1_batched_case`, `prepare_b1_row_until_target`, `finish_b1_row_from_target`, `apply_b1_target_chunk`,
  `build_serial_reference` reuse, pad policy = duplicate row 0 into unused bucket rows, pads discarded on unpack.
- Test cases: identical_rows_B4, ragged_mixed_K3_padded_to_B4, full corpus coverage (groups of args.b1_batch_size=4).
- Test bar in code: ok = (token_divergences == 0 AND event_divergences == 0 AND len_mismatches == 0).

**Full T1 result** (codex-jobs/step-B1-b4ml9h322.log; gate run with DENSITY_GOLD_EVENTS_TOLERANT=0 = strict):
- identical PASS, ragged/mixed PASS.
- reference rows 1000/1000 matched gold (strict events) → B=1 path is untouched.
- batched row replays: 1007 rows over 336 cases.
- **token divergences: 0 / 1007** (both steady and final).
- **enc_len mismatches: 0; cache_len mismatches: 0.**
- **event divergences: 2 / 1007** — both interim-event-timing class:
  - utt198 chunk13 event[6]: batched "Could you translate the phrase where is" vs B=1 "Could you translate the phrase
    where" (1 extra interim word in the batched path).
  - utt770 chunk1 event[0]: batched "A grid" vs B=1 "A", event count 12 vs 13.
- max diffs: enc_out=7.651e-04, cache_ch=2.089e-03, cache_t=4.711e-02.
- Codex Job's verdict: "FAIL under strict event byte-exactness" (because it ran strict). It did not loosen tolerance.

**The user's call (2026-05-27):** PASS-BY-POLICY WITH AUDIT. The project's documented density T1 (per the memory
phase2-density-review + the run#13/run#14 W3 logs cited in PHASE2-PLAN.md row 1b) explicitly tolerates this exact
class — "1000/1000 finals byte-exact vs gold; 5/1000 interim-event-timing drift (WER-neutral, counted-not-gated via
DENSITY_GOLD_EVENTS_TOLERANT)." B1's 2/1007 is strictly better than that 5/1000 prior bar. So under project policy
this PASSES; this paired review should AUDIT (not relitigate the policy).

**Files to read (full)**:
- proj-2026-05-24-from-scratch-runtime/runtime/cpp/steady_batch_primitive.h (the new primitive, 196 lines).
- proj-2026-05-24-from-scratch-runtime/runtime/cpp/density_main.cpp lines 1700–2260 (the b1-t1 gate) + the include/
  arg-parse/mode wiring around lines 46/51/158/189/255 + the dispatcher around line 4591.
- proj-2026-05-24-from-scratch-runtime/runtime/export_steady_batched.py (the B=1/2/4 export; THE PRECISION-POLICY
  AUDIT: does it match the production B=1 enc_steady_aoti.pt2 compile settings — autotune off, TF32 policy, dtype?).
- proj-2026-05-24-from-scratch-runtime/PHASE2-PLAN.md (v3 block + Step B1 + the lever inventory + the Compile &
  artifact policy on autotune/precision).
- proj-2026-05-24-from-scratch-runtime/codex-jobs/step-B1-b4ml9h322.log (Codex's full build/run trail).
- The microbench reviews/steady-batch0-RESULT.md (microbench had enc_out 6e-6 / cache_t ~5e-3 — much smaller than
  the full-T1 4.71e-2; explain or flag the gap).

ASK / structure your fold-ready review:
1. **Implementation audit** of the primitive (steady_batch_primitive.h):
   - Shared-constants discipline: is `user_managed=true` correct here, and is the `finalize_shared_weights.ts`
     constant set complete for the steady encoder (it was wired for finalize buckets, is it complete for steady)?
   - The pack: torch::cat allocations per call — correctness fine but quantify if there's a hot-path issue.
   - The unpack: `select(0,row).unsqueeze(0).contiguous()` — correctness OK?
   - The stream guard placement: `CUDAStreamGuard` at `run()` entry — sufficient for AOTI internals?
   - Bucket selection K→B: any K=0 / K>4 paths? Concurrency safety (is `get()` reentrant?)?
   - Pad policy "duplicate row 0": correct (real distribution), but does it write any state that leaks into the real
     rows (cache_ch / cache_t)? Verify pads are write-only-then-discarded.
2. **b1-t1 gate audit** (density_main.cpp 1700–2260):
   - `prepare_b1_row_until_target` runs the session up to the target chunk via the production B=1 path → correct
     setup, no contamination?
   - `apply_b1_target_chunk` uses the BATCHED output: the (void)-cast of `apply_encoder_outputs_density` — is the
     discarded return value safe to ignore?
   - `finish_b1_row_from_target`: after applying the batched target, it runs subsequent chunks via the **B=1**
     `run_steady_chunk_density` path. Is the per-stream cache state correctly continued from the batched output (i.e.,
     does the B=1 follow-on consume the batched cache_ch/cache_t)? Or is there a subtle re-write that erases the
     batched effect?
   - Reference is `build_serial_reference` (the production B=1 path) — adequate gold?
3. **The 2 event divergences — MARGIN PROBE** (the audit's heart):
   - Are they near-tie greedy-argmax flips at the divergent chunks (intrinsic to the BW-amortized reduction-order →
     accept counted-not-gated), or are they systematic precision-policy gaps (fixable by recompiling the B exports to
     match B=1's autotune-OFF / TF32-reduced policy)?
   - Add a margin probe: at the divergent chunks (utt198/chunk13/event[6] and utt770/chunk1/event[0]), instrument the
     joint/predict score gap between the top-2 candidates at the diverging timestep — does the batched path land
     within fp32-noise of a tie (intrinsic) or shifts the score significantly (systematic)?
   - If you can't add new instrumentation cheaply, examine the cache_t diff per-utt at the divergence vs the corpus
     median to see if utt198/770 are outliers.
4. **Cache_t magnitude growth**: microbench 5e-3 → full T1 4.71e-2 (~10×). Is this consistent with reduction-order
   accumulation through ~13 recurrent chunks (utt198/chunk13), or is it a precision-policy mismatch in the B exports?
   Look at the per-chunk cache_t growth rate (cache_t evolves as a *streaming* state).
5. **Precision-policy audit of `export_steady_batched.py`**: confirm the B=1/2/4 packages are compiled with the SAME
   autotune-OFF + TF32-reduced (eager-matched) precision policy as the production `enc_steady_aoti.pt2`. The plan's
   memory and Compile policy explicitly warn about autotune precision divergence (cache_t 10.27 katmatrix). If the
   batched exports drifted policy, that's a fixable systematic bug (rather than intrinsic reduction-order).
6. **Net verdict**: PASS / PASS-with-cheap-fix / HOLD. If a cheap fix surfaces (e.g., precision-align the exports),
   recommend it; otherwise endorse PASS-by-policy and recommend the documentation / counted-not-gated framing for the
   commit + the PHASE2-PLAN.md update + the lever inventory.

Write your analysis to `proj-2026-05-24-from-scratch-runtime/reviews/codex-B1-review.md`. Adversarial, second-look,
specific to the code and the data. The user will fold this with the Opus pass.
</context>

<verification_loop>
Don't run the full corpus T1 again (it just ran). If you do a quick margin probe, keep it bounded (1-2 utterances).
Build is already known clean. The point is the audit, not a rerun.
</verification_loop>

<action_safety>
Write the review doc, nothing else. Do NOT modify the implementation files — fix recommendations go in the review doc
for me to action. If you decide to instrument a margin probe, gate it behind a debug flag so it doesn't pollute the
B1 commit.
</action_safety>

<compact_output_contract>
Report path of the review doc + a one-paragraph verdict summary (PASS-by-policy / PASS-with-cheap-fix / HOLD) + the
single most important fix-or-flag if any.
</compact_output_contract>
