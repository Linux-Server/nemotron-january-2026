# 1.3b-enc-scale paired-review FOLD (authoritative)

Two reviews — `opus-enc-scale-review.md` (me) + `codex-enc-scale-review.md` (Codex). High convergence; Codex escalates to
BLOCKER and adds the strip-inconsistency + right-context-invariant + alias-is-load-bearing findings. **Net: the per-T
bucket MECHANISM is validated for the observed drop=2, right-context=1 case (token-exact 1000/1000, shared-weights flat,
strip token-exact), but 1.3b-enc-scale is NOT production-complete.** Honest status: "drop=2 / rc=1 Python shadow is
token-exact for observed T=43..58; drop=0, out-of-range fallback, C++ corpus validation, strip consistency, and artifact
metadata remain open."

## BLOCKERs
- **B1 — drop=0 first-finalize untested + uncovered.** drop_extra=0 fires when emitted_frames==0 (utterance finalized
  before the first steady emit; server.py:8071/8317/9951). Real exact-T span = **T≈33..49** (server.py final-padding math
  1603/1607/7486; finalize_ref.prepare_finalize_inputs). Only a single drop0 T=49 fixture bucket exists; the corpus probe
  never creates a first-finalize case so it's UNTESTED. FIX: build+validate drop0 T=33..49 buckets AND/OR a proven eager
  fallback for drop0; add a synthetic short-audio sweep to the probe + shadow.
- **B2 — missing/out-of-range buckets silently PASS (lost transcript).** Python gate passes on `divergent==0`, ignoring
  `uncovered` (finalize_corpus_shadow.py:452/478); C++ Phase B prints "skipped" without `phase_b_ok=false`
  (finalize_main.cpp:416/452). A finalize with no matching bucket emits NO final tokens → dropped end-of-turn transcript.
  FIX: fail-closed in BOTH — require covered==total unless a NAMED, VALIDATED fallback path is exercised (and reported as
  fallback, not pass). Add the eager fallback for any (drop,T) without a bucket (finalize is off the hot path).
- **B3 — C++ corpus validation gap.** The 1000-shadow uses Python `aoti_load_package`/`load_constants`; the C++
  `AOTIModelPackageLoader` path (finalize_main.cpp Phase B) is validated only on 4 fixture rows. FIX: a C++ corpus runner /
  generated multi-T corpus fixture that routes all rows through the C++ loader and fails on any miss/mismatch/exception.

## MAJORs
- **M1 — T-range is an unasserted right-context=1 invariant.** Server supports rc ∈ {0,1,3,6,13}; prompted models default
  rc=3; final_padding = (rc+1)*shift, so the finalize T range SHIFTS with rc. finalize_ref hardcodes [70,1]; the bucket
  builder hardcodes drop2 T=43..58. FIX: embed model-id / att-context / right-context / shift / pre-cache / final-padding /
  drop / supported (drop,T) ranges in the bucket manifest + check at C++ startup; refuse out-of-contract buckets.
- **M2 — strip is two inconsistent implementations + drop0 strip unvalidated.** strip_bucket_weights.py SKIPS weight
  entries (validates, default T44 only); strip_all_buckets.py keeps entries but writes EMPTY bytes (no validation). The
  1000-shadow transitively validated all stripped drop2 T=43..58 (each occurred), but NEVER the stripped drop0 T49. FIX:
  ONE stripper; validate EVERY stripped bucket (incl drop0/synthetic) vs its unstripped original by full decoded tokens;
  write a per-bucket strip-validation manifest, fail if any stripped bucket is unrun.
- **M3 — shared-weights alias is LOAD-BEARING + stale-weight hole.** FinalizeStep names the module `encoder`
  (export_finalize_t2a.py:20) so bucket FQNs are `encoder.*`, but export_shared_weights keys `e.*`
  (export_shared_weights.py:32) — the `e.*`↔`encoder.*` alias (finalize_corpus_shadow.py:61, finalize_main.cpp:193) is what
  makes load_constants match (both fail-closed on missing, good). But export_shared_weights REUSES an existing .pt without
  identity check, so a stale .pt from another checkpoint/att-context would load silently wrong. FIX: make the wrapper
  naming consistent (pick one) or document the alias as a required contract; embed tensor shape/dtype/digest + model/ckpt/
  att-context in finalize_shared_weights and verify vs the bucket manifest before load_constants.
- **M4 — NeMo equivalence is canary-scale.** The corpus shadow compares bucket vs eager finalize_ref (1000), and
  finalize_ref vs NeMo is only 4 canaries (1.3a). FIX: run the 1000 finalize rows vs the NeMo stream+finalize oracle, OR
  downgrade the claim to "AOTI buckets == eager finalize_ref on observed drop2 corpus rows."
- **M5 — AOTI compile flakiness unguarded.** T57 hit KeyError('version') then compiled clean; no retry / temp-path /
  post-compile self-check (aot_compile_buckets.py). FIX: compile to temp, delete-on-fail, retry clean process, then
  immediately load+load_constants+run each package vs its captured example; record SHA + Torch/CUDA + token/enc deltas.
- **M6 — manifest is not a full inventory** (only T57 now); build_range writes only delta. FIX: a complete checked manifest
  (every expected bucket: source example, drop, T, pkg hash, strip hash, validation status, path); delta manifests separate.
- **M7 — findings doc overstates.** "production-proven" / "not novel risk" while drop0/fallback/C++/metadata are open.
  FIX: rewrite the status (done below).

## MINORs
- M8 probe always exits 0 (add --require-complete + required (drop,T) spec + synthetic cases). M9 harness ignores enc_len/
  shape/finite/dtype (record+assert alongside token equality).

## Iteration plan (priority)
1. **Fail-closed (B2) + findings-doc correction (M7)** — cheap, high-value, done now.
2. **drop=0 (B1) + out-of-range eager fallback (B2/B3)** — the real coverage; the eager fallback also handles M1's rc≠1
   and out-of-range T. Naturally lands with 1.4 (the live session invokes finalize + owns the fallback).
3. **C++ corpus runner (B3)** + strip-unify+validate-all (M2) + artifact metadata/contract (M1/M3) + build hardening (M5/M6).
4. **NeMo-corpus oracle (M4)** when convenient.
