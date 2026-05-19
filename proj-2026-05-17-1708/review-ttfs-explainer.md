# Review brief — TTFS latency explainer (REVIEW ONLY, modify no files)

Target doc: /home/khkramer/src/nemotron-january-2026/docs/ttfs-latency-explainer.html

You are a harsh technical reviewer. Do NOT edit any file — report findings only.
Goal: verify the explainer is technically accurate, internally consistent, and does
not overstate or understate any claim relative to the evidence and the locked design.

## Context to read
- The doc itself (the HTML target above).
- PLAN locked Rule "Production latency budget" and latency taxonomy:
  /home/khkramer/src/nemotron-january-2026/proj-2026-05-17-1708/PLAN.md lines ~100-145.
- The 7d implementation in src/nemotron_speech/server.py: the speculative fork path
  (_continuous_finalize_emit_locked, _continuous_append_only_delta, the debounce timer
  _continuous_handle_vad_stop_locked / _continuous_handle_debounce_expired_locked),
  NEMOTRON_FINALIZE_SILENCE_MS default, the (R+1)*shift synthetic finalize padding in
  _build_continuous_finalize_fork.
- The measured budget reader stt-benchmark/scripts/measure.py (finalize-budget):
  the terms endpoint_wait(vad_stop->debounce_expiry), measured_finalize_flush_wallclock,
  transport, inference_lock_acquire_wait, rc1 modeled constant, the budget formula.
- Preliminary measured numbers (partial fork run, n approx 499/1000, 1214 finalize events):
  endpoint_wait p95 ~151.8 ms, finalize_flush p95 ~14.4 ms, transport p95 ~0.3 ms,
  inference_lock_acquire_wait ~0, budget total p95 ~325.9 ms PASS < 400 ms.

## Specifically attack these claims for correctness
1. The two-reference-frames split (budget measured from vad_stop vs perceived from true
   end-of-speech) and the claim that the Silero vad-stop-secs (~200 ms) is serial-before
   the 150 ms server debounce and is NOT included in the measured endpoint_wait term.
   Verify against how the client emits vad_stop and how measure.py defines endpoint_wait.
2. The rc1-overlaps-the-wait argument: is it correct that the encoder right-context
   lookahead is satisfied by the trailing silence consumed during the endpoint waits, so
   rc1 adds ~0 additional wall-clock once any wait >= 160 ms exists? Is the budget
   formula's additive rc1 therefore conservative, OR does the doc mis-handle that the
   finalize fork supplies the last-chunk right-context synthetically via (R+1)*shift
   padding (faster-than-wallclock) rather than by real trailing audio? Is the ~175 ms
   irreducible floor (rc1 + flush) defensible, or is the true floor different given the
   synthetic-padding mechanic?
3. The ML / model-internal endpointing section: are the RNNT blank-run confidence,
   hypothesis-stability, and joint-entropy claims technically sound for THIS model
   (cache-aware FastConformer-RNNT, greedy loop_labels=False), and correctly flagged as
   doable-within-constraints vs needing a model/framework change? Anything overstated?
4. measured/modeled/reasoned labels: is every quantitative claim labeled honestly and
   consistent with what is actually measured vs modeled vs reasoned?
5. Any missing first-order contributor to finalization latency, or any factor whose
   reducibility is mischaracterized.

## Output contract
Per item 1-5: verdict ACCURATE or DEFECT. For each DEFECT: the exact doc claim, why it
is wrong/over/understated, the corrected statement, with code/PLAN/measure references.
End with: overall is the doc safe to keep as a reference (YES/NO) and the top 1-3
corrections Claude must make. Review only; modify no files.
