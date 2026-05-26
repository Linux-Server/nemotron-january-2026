# 1.4b Step 5 paired-review FOLD — NOT MET; overclaims + a false coverage PASS to fix

Reviews: `codex-step5-hardening-review.md` (1 BLOCKER + 4 MAJOR + 2 MINOR) + my (Opus) concurrence. I AGREE with Codex
across the board — the hardening added real checks but several SUMMARIES OVERCLAIM what they prove, and one coverage entry
is a FALSE PASS. That's precisely the failure mode hardening is meant to eliminate, so Step 5 is correctly NOT MET. The
underlying binary is healthy (all gates pass, N=1000 0/1000, WER delta 0); the fixes are honesty/strengthening, not
re-architecting.

## BLOCKER
- **B1 — `vad_start_cancel=PASS` is a FALSE coverage claim.** SessionState has NO `continuous_post_stop_audio` buffer
  (confirmed: no such field); the "synthetic" path just flips mode PENDING_FINALIZE→STREAMING on a fresh empty session +
  checks counters (session_main.cpp:2977-2988), and append_audio rejects non-STREAMING (:2183). The reference behavior
  (finalize_ref.py:398-430 / the reset-resume test :1223-1267) HOLDS post-stop audio while pending, flushes it on
  vad_start, and asserts equality with a no-stop session + next steady + final tokens. The manifest reports the flip as
  lifecycle coverage. FIX: implement a REAL C++ vad_start-cancel (add the post-stop buffer; append-to-post-stop while
  PENDING; vad_start flush/drain; assert parent unchanged + equality vs a no-stop session + continuation tokens/events).
  If genuinely infeasible in the replay harness, REMOVE the false PASS and mark vad_start_cancel explicitly
  NOT-COVERED/deferred (like stale_generation) — do NOT report coverage we don't have.

## MAJORs (overclaim → relabel + strengthen)
- **M2 — provenance reuse is self-referential + config empty.** C++ is fail-closed on a hash mismatch (good,
  session_main.cpp:1831-1835) BUT export computes the expected hash from the EXISTING preproc.ts (export:476-480), so a
  stale preproc.ts + matching stale manifest is accepted; `preprocessor_config={}` (manifest:33) records nothing. FIX:
  regenerate preproc.ts unconditionally for audio exports OR derive the manifest contract from the LOADED model (real NeMo
  preprocessor _cfg + versions + geometry + dither + trace shape); ADD a corrupt-`preproc_ts_sha256` negative test that
  asserts C++ exits before running audio (analogous to the bucket-hash test).
- **M3 — determinism is SAME-PROCESS, not run-to-run.** The rerun reuses already-loaded loaders (session_main.cpp:3904-3919
  / :3639-3655) + same-process preproc twice (:1985-2003). FIX: a fresh-PROCESS determinism gate (run the binary twice,
  reload all artifacts, compare token/event + preproc-block fingerprints) OR relabel the current one
  `same_process_full_session_twice`.
- **M4 — `alias_after_clone=0` proves the CLONE BARRIER, not loader-output non-reuse.** The check compares the clone vs the
  immediate output (passes if clone() works); it doesn't prove raw out[2..4] don't alias inputs/prior raw outputs. FIX:
  rename to `cache_state_clone_barrier` + document raw-output reuse is tolerated, OR add pre-clone raw-output alias checks
  across consecutive run() calls.
- **M5 — audio oracle is now C++ vs shipped preproc.ts, not vs the eager model.** Sound for "matches the shipped artifact"
  but the token/event-exact gate is no longer eager-equivalent (TS-vs-eager drift only a separate CI envelope). FIX:
  document the audio gate as "vs shipped preproc.ts" AND add a separate eager token/event (subset) non-regression check; do
  not summarize audio as eager-equivalent from the one-sample self-check.

## MINORs
- **M6** — first-chunk margin (17.0) is observational (margin_checks=2000 = first ~2 frames/utt), not a global proof; narrow
  the wording ("TorchScript first chunk retained; tested first-chunk greedy decisions min margin 17; a pure-native runtime
  still needs an AOTI first chunk").
- **M7** — split the printed lifecycle counters: `fork_parent_unchanged` / `speculative_retained_state` /
  `true_boundary_cold_reset` (one flag conflates fork-isolation with post-finalize lifecycle).

## Decision
Step 5 NOT done. Fix pass: B1 (real vad_start-cancel OR honest removal+defer), M2 (provenance regenerate/model-derived +
corrupt-hash negative test), M3 (fresh-process determinism OR relabel), M4 (relabel clone-barrier OR raw-output alias
check), M5 (eager subset gate + document), M6/M7 (narrow + split). Re-run all gates (no regression) + the new/fixed checks,
then re-review. Honesty bar: every coverage/telemetry label must match exactly what it proves.
