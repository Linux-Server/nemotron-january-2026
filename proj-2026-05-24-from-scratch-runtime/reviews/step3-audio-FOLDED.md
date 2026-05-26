# 1.4b Step 3 paired-review FOLD — core MET (faithful audio front); fix tolerance-discipline + telemetry before done

Reviews: `codex-step3-audio-review.md` (4 MAJOR + 2 MINOR, NO BLOCKER) + `opus-step3-audio-review.md`. CONSENSUS: the audio
front is FAITHFUL — Codex verified line-by-line vs finalize_ref/server that the fixed-block assembly, valid-mel slice,
raw-ring advance, steady-chunk boundaries (ready predicate), and the finalize remainder geometry (a TRUE recompute from
live audio, not baked) all match; token/event/FORK/raw-ring exact (20/20) from PCM, geometry=PASS. The residuals are
TOLERANCE-DISCIPLINE + TELEMETRY, not correctness bugs.

## Fix before marking Step 3 done (tolerance discipline + the near-tie guard)
- **F1 (Codex M1+M4, me #1) — undocumented escape-hatch tolerances.** mel acceptance uses a hard-coded `mel_atol=5e-4`
  (session_main.cpp:1430) and the AUDIO-mode RETAINED encoder-cache comparison uses `retained_audio_atol=3e-2`
  (session_main.cpp:1936) — neither derived/documented. The 2.44e-4 mel + the cache drift are plausibly the documented
  cross-process cuFFT nondeterminism (`cufft-stft-plan-size` memory: validate streaming preproc by within-CI, never
  byte-equality), but the constants must be PRINCIPLED. FIX: measure the envelope (run preproc.ts over the bundle PCM at
  fixed K; report max/mean/p99 abs+rel mel diff AND retained-cache max_abs per tensor), set mel_atol + cache_atol from that
  envelope with a written rationale, and PRINT the observed max per run. Token/event-exact stays the semantic bar; mel/cache
  acceptance = "within documented preproc CI", not a magic constant.
- **F2 (Codex M2) — no argmax-margin guard.** Greedy decode is bare argmax (session_main.cpp:1314); token-exact on 20/20
  proves these clips weren't near a flip, not that 2.44e-4 mel drift is always safe. FIX: add min top1-top2 margin telemetry
  to the audio gate (reuse aoti_drift_probe's shape), report the min margin + count below a named warning threshold + where;
  acceptance = geometry/state exact ∧ mel within CI ∧ token/event exact ∧ no unsafe near-margin on the Step-3 corpus.
- **F4 (Codex M5) — separate geometry/mel counters.** `geometry=PASS` currently reuses the whole-session pass bit
  (session_main.cpp:2456). FIX: distinct geometry_checks/pass + mel_checks/pass counters; print geometry_recompute
  independently.

## Fold to Step 5 (hardening) — not blocking Step 3
- F3 (Codex M3) — preproc.ts PROVENANCE: reused without model-id/dither/K/geometry/hash check (export:249). FIX in Step 5's
  metadata/contract: a preproc sidecar manifest (model-id/config/dither/geometry/K/torch-cuda/hash) + an export-time
  TS-vs-eager self-check. (Same family as the bucket manifest contract.)
- F5 (Codex M6) — run-to-run determinism: harness compares one C++ run vs Python gold, not C++-twice. FIX in Step 5: a
  duplicate-run check (preproc.ts twice byte-equal in C++; session twice token/event-identical).

## Verdict
Step 3 audio-front equivalence is MET at the correct bar (token/event/geometry/raw-ring exact from PCM; mel/cache
within-CI per the cuFFT finding) — closing session-FOLDED B5 ("single native AUDIO stream"). Do F1/F2/F4 (tolerance
discipline + margin telemetry + counters), then mark done; F3/F5 fold into Step 5 hardening.
