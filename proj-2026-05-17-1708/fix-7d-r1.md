# Fix #2 for Step 7d — R1 BLOCKER (multi-emit duplication) + FORK_ASSERT gap + R4 scope label

PLAN: /home/khkramer/src/nemotron-january-2026/proj-2026-05-17-1708/PLAN.md (Step 7d block;
locked Rule lines ~100-145). Prior 7d work is uncommitted in the tree (server.py +225/-37
parent repo; stt-benchmark nemotron_local_stt.py +49/-3 and scripts/measure.py +221 nested
repo). Baseline 7c committed 7cbdf09 (cc7c GATE PASS). The speculative/true-boundary split
from fix #1 is correct — KEEP it. ONLY modify these 3 files:
- src/nemotron_speech/server.py
- stt-benchmark/src/stt_benchmark/nemotron_local_stt.py
- stt-benchmark/scripts/measure.py
No framework/NeMo/probe_alias/equiv_*/PLAN edits. No full-1000/benchmark (Claude runs it).
No new deps. No server left running. Preserve everything from 7c/7b/fix#1 (disposable fork
+ Step-3 deep-clone recipe, the speculative/true-boundary epilogue split, close-drain
_send_json_locked tolerate_closed, finalize_timing JSONL, default '' and phaseG byte-
unchanged & env-gated under NEMOTRON_CONTINUOUS=1, 6c constant-plan ring, CUDA hardening,
deadlock-free state_lock->inference_lock, NEMOTRON_FINALIZE_SILENCE_MS default 150).

## R1 — BLOCKER (must fix). Always-append-only continuous emission.
Root cause: the benchmark collector
stt-benchmark/src/stt_benchmark/observers/transcription_collector.py:72-75 is
FRAMEWORK/LOCKED and APPENDS every final TranscriptionFrame:
`self.transcriptions[id] += " " + text`. The 7d server emits one delta per speculative
finalize and, when a later fork final_text does NOT startswith committed_text (ASR self-
correction; ~40% of cumulative transitions are non-prefix), emits the FULL final_text
(server.py ~1358-1369). The client (nemotron_local_stt.py:_handle_transcript ~469) pushes
every armed finalize=true as a TranscriptionFrame. Net on the 653/1000 multi-segment
samples (1389 internal gaps all >150ms): old partial delta + full corrected text get
concatenated -> duplicated/garbled hypothesis -> inflated WER. 7b/7c never hit this
(single final/sample). Codex smoke-B used its own concat + clean sentences so it missed it.

Required contract (server-side, in src/nemotron_speech/server.py): the continuous
finalize path must be ALWAYS-APPEND-ONLY with respect to the locked append-only collector.
Maintain on the session an explicit record of what the collector has already accumulated,
e.g. `session.continuous_emitted_text` (the exact string the harness holds = prior emitted
deltas joined by single spaces, matching the collector's `+= " " + text`). On each
speculative finalize (the shared `_continuous_finalize_emit_locked` core):
1. compute `final_text` from the fork as today;
2. compute the emitted delta as the WORD-LEVEL suffix of `final_text` that extends
   `continuous_emitted_text` — i.e. split both into whitespace tokens, take the longest
   common leading token run, the delta is the remaining `final_text` tokens joined by
   spaces. If `final_text` does not extend (pure correction / shorter / divergent),
   emit the strictly-new trailing tokens only (never re-emit tokens already in
   `continuous_emitted_text`); if there are none, suppress (empty -> no frame, as today).
3. only when a non-empty delta is actually sent, update
   `continuous_emitted_text = (continuous_emitted_text + " " + delta).strip()` so it
   exactly mirrors the collector's post-append state; keep `committed_text`/
   `last_emitted_text = final_text` for the fork/correction-detection logic.
4. Net invariant: the collector's accumulated string is always a monotone, duplication-
   free prefix-growing reconstruction; corrections to ALREADY-emitted words are NOT
   retroactively applied (accepted, will be quantified by the measured paired-delta).
   Never send a full cumulative correction into the append-only collector.
On a true-boundary finalize (close/end) the same word-level-suffix rule applies for the
final residual emit, then the cold reset also resets `continuous_emitted_text=""`. The
speculative epilogue must RETAIN `continuous_emitted_text` (it is part of the continuous
emit state across pauses, like committed_text). Default ''/phaseG path must not use or be
affected by `continuous_emitted_text` (env-gated). Keep emit-once: exactly one frame per
finalize event or suppressed; early_final must still be able to reach 0.

## FORK_ASSERT gap — MAJOR (fold in). 
`_build_continuous_finalize_fork` deep-clones `pred_out_stream` but
`_snapshot_fork_assert_parent` / `_assert_fork_flush_parent_unchanged` only snapshot+assert
cache tensors + previous_hypotheses, NOT pred_out_stream. Add `pred_out_stream` to BOTH
the snapshot and the post-flush byte-identical assertion (use the same tensor-aware deep
compare already used for previous_hypotheses) so the under-load parent-unchanged proof is
complete. Keep it env-gated under NEMOTRON_FORK_ASSERT=1, default off.

## R4 — claim-scoping (no behavior change). 
In stt-benchmark/scripts/measure.py finalize-budget output, add a one-line explicit label
that the reported endpoint/finalize/transport p95 is single-session / sequential-benchmark
observer latency (the locked-Rule measurement), NOT a concurrent-production figure. Label
only; do not change the computed numbers or formula.

## Verification (NO full-1000). 
1. py_compile + ast.parse all 3 files via
   /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python with
   PYTHONPYCACHEPREFIX set to a mktemp dir.
2. Contained CUDA rc1 smoke, NEMOTRON_CONTINUOUS=1 NEMOTRON_FORK_ASSERT=1
   NEMOTRON_FINALIZE_SILENCE_MS=150, ~12s health:
   (a) single segment: exactly one non-empty final; finalize_timing present;
       fork alias assertion PASSED (now also covering pred_out_stream).
   (b) THE R1 REGRESSION TEST (critical, must fail on the OLD code and pass on the new):
       drive a MULTI-SEGMENT real >150ms-gap interaction where a later speculative
       final_text does NOT startswith the prior committed_text (force/observe a non-prefix
       cumulative transition — e.g. concatenate two fixtures whose boundary makes the
       streaming hypothesis revise). Feed the per-finalize deltas through the ACTUAL
       collector semantics — import and use
       stt_benchmark.observers.transcription_collector.TranscriptionCollectorObserver
       (or replicate its exact `+= " " + text` rule) — and assert the final collected
       hypothesis has NO duplicated run and equals the expected monotone always-append-
       only reconstruction (no `old + full_corrected` doubling). Print the collected
       string and the assertion result.
   (c) pre-debounce client close: final still emitted (close-drain) AND true-boundary
       cold reset ran; continuous_emitted_text reset.
   (d) measure.py finalize-budget parses the smoke JSONL, prints median/p95 + PASS/FAIL
       + the new single-session scope label.
   (e) default env-unset smoke: old hard-reset path, no finalize_timing, unaffected.
   Kill server; clean __pycache__ via PYTHONPYCACHEPREFIX; nothing left running.
   Iterate until ALL pass, especially (b).

## Output contract
Report: 1) the always-append-only algorithm (exact tokenization + longest-common-token-
prefix logic + the continuous_emitted_text lifecycle: where set/extended/retained on
speculative epilogue/reset on true boundary/never in default-phaseG); 2) proof no path
emits a full cumulative correction into the append-only collector + emit-once preserved;
3) the FORK_ASSERT pred_out_stream addition (snapshot + assert, env-gated); 4) the
measure.py scope label; 5) preservation proof (7c fork, speculative/true-boundary split,
close-drain, JSONL, default/phaseG, 6c ring, lock order, 150ms default); 6) py_compile/ast
+ smoke results with the (b) regression-test transcript + assertion output proving no
duplication through the real collector semantics; 7) scope (only 3 files); 8) blockers;
9) next (Claude re-review + measured fork full-1000).
