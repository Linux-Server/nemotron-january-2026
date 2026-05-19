# Adversarial review brief — Step 7d (pre-run, REVIEW ONLY, do NOT modify any file)

You are a harsh adversarial reviewer. Do NOT edit/fix anything — report findings only.
Goal: find correctness/concurrency/aliasing/latency defects that would corrupt the
measured WER or the <400ms budget claim BEFORE a 3.7h measured full-1000 run.

## Scope
- PLAN: /home/khkramer/src/nemotron-january-2026/proj-2026-05-17-1708/PLAN.md
  (Step 7d block ~498-525; locked Rule ~100-145; Step 1 ~238-265).
- Parent repo diff: `git diff -- src/nemotron_speech/server.py`
- Nested repo diff: `cd stt-benchmark && git diff -- src/stt_benchmark/nemotron_local_stt.py scripts/measure.py`
- Baselines: 7c committed 7cbdf09 (cc7c GATE PASS: 1999/0 fork-alias, WER-neutral); 7b b757159.

## 7d design under test
Continuous mode (NEMOTRON_CONTINUOUS=1), short debounce NEMOTRON_FINALIZE_SILENCE_MS=150.
vad_stop arms a 150ms timer. Debounce expiry: _continuous_finalize_emit_locked
(build disposable deep-clone fork from parent pending + (R+1)*shift padding, flush under
inference_lock, NEMOTRON_FORK_ASSERT parent-byte-identical, emit ONE incremental delta
vs committed_text, then committed_text=final_text) then
_continuous_finish_speculative_finalize_locked (clears only continuous_state to STREAMING,
vad_stop_ts, debounce_expiry_ts, debounce_task, reset_seen; PARENT ASR context retained:
cache_last_channel/time/len, previous_hypotheses, pred_out_stream, raw/mel rings,
current_text, emitted_frames; NO _init_session). close or end: _continuous_finalize_and_reset_locked
= core + _continuous_cold_reset_after_finalize_locked (the only _init_session for continuous;
hard-raises if reason not in {close,end}). Thesis: one continuous ASR context per sample
across all VAD pauses reaches Phase-G/oracle ~1.3% WER; cold-reset only at true end.
This split FIXES a prior blocker (every 150ms expiry cold-reset the parent -> ~3% WER).
Step-1 VAD gaps: min154 p50380 p95800 p99 1220 max1700 ms; 650/1000 multi-segment.
7b(2500ms)=2.00% via a single end-of-sample finalize.

## Attack these risks hard
- R1 (top): multi-speculative-emit times ASR self-correction. Delta logic:
  if final_text.startswith(committed_text): delta = final_text[len(committed_text):].lstrip()
  else: delta = final_text ; then committed_text = final_text. With MANY speculative
  finals per sample, if a later fork final_text does NOT start with prior committed_text
  (streaming model revised earlier words) the FULL text is re-emitted with finalize=true.
  Trace nemotron_local_stt.py finalize_events / _handle_transcript (Step-5 split): do
  successive finalize=true deltas APPEND or REPLACE on the benchmark/client side? A
  full re-emit appended after partial finals = duplicated/garbled transcript = inflated
  WER. Is emit-once truly preserved across N per-sample emits? Quantify when startswith
  fails for continuous RNNT greedy.
- R2: post-stop audio. continuous_post_stop_audio retained on speculative epilogue,
  flushed into the parent on next vad_start (PENDING_FINALIZE branch reason vad_start;
  else branch reason vad_start_after_speculative_finalize) or _handle_audio. Verify no
  post-stop audio LOST or DOUBLE-fed into the parent stream across
  speculative-finalize -> next-vad_start. Fork consumed a COPY+padding (parent
  pending_audio untouched) but does the parent real (non-fork) stream advance over the
  same pending/post-stop audio exactly once? off-by-one / double-count / drop in ring or
  pending across the retained-context boundary.
- R3: audio arriving AFTER a speculative finalize but BEFORE the next vad_start
  (state STREAMING post-expiry). Where does _handle_audio_locked route it? appended to
  the retained parent context without re-triggering finalize or corrupting
  committed_text/delta?
- R4: latency/budget. N speculative fork-flushes per multi-segment sample, each takes
  inference_lock + ~13ms compute + JSON send. measure.py finalize-budget computes
  endpoint_wait(vad_stop->debounce_expiry) + rc1(160ms const) +
  finalize_flush_wallclock(fork_flush_start->final_sent incl lock wait) +
  transport(final_sent->final_received), p95 vs 400ms. Audit: are JSONL timestamps at
  the right points; final_received is client-side while fork_flush_* are server-side —
  same clock domain or skew that invalidates transport? Does inference_lock contention
  from concurrent benchmark sessions inflate real p95 beyond the synthetic n=3 smoke
  (329ms)? Any term mismeasured/missing (synthetic silence-padding duration wrongly
  counted, or lock-wait excluded)?
- Also: deadlock-free state_lock -> inference_lock order under the new epilogues;
  continuous_debounce_task cancel/None race vs the worker; stale-seq guard with rapid
  vad_stop/expiry; FORK_ASSERT still parent-byte-identical with context spanning many
  segments (longer previous_hypotheses/pred_out_stream clone depth/perf); default '' and
  phaseG byte-unchanged; 6c constant-plan ring intact with multi-segment retained
  context (unbounded growth / long-context bound).

## Output contract
Per risk (R1..R4 + the deadlock/seq/FORK_ASSERT/default/ring list): verdict SAFE or
DEFECT. For each DEFECT: file:line, severity BLOCKER/MAJOR/MINOR, exact failing
scenario, fix direction. For each SAFE: the code-referenced proof. End with a single
GO or NO-GO recommendation for the measured fork full-1000 run and the top 1-3 things
Claude must double-check. Review only; modify no files.
