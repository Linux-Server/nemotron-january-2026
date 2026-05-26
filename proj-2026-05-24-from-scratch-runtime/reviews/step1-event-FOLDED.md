# 1.4b Step 1 paired-review FOLD (authoritative) — bar NOT met; fix required

Reviews: `codex-step1-event-review.md` (4 BLOCKERs) + `opus-step1-event-review.md`. **Codex is right and corrects my
over-lenient take:** I claimed "word-vs-token delta PROVEN by the export fail-closed proxy-check." It is NOT — the export
proxy-check only validates the OBSERVED corpus rows, and every bundle row is a FRESH session with ONE finalize → the
collector (`continuous_emitted_text`) is EMPTY before the final → the delta is always the trivial "append-all," where word
and token agree. The general equivalence is FALSE.

## BLOCKERs (adopted)
- **B1 — token-level delta ≠ word-level reference (real bug).** finalize_ref/server `_continuous_append_only_delta` is
  WORD-level (`text.split()`, finalize_ref.py:144 / server.py:392). The C++ port (session_main.cpp:180) operates on
  token-ids. Counterexample (Codex): emitted "I live in New"=[…,_New], final "I live in Newark"=[…,_New,ark] → WORD algo
  SUPPRESSES (4 words→4, non-prefix last-word correction, `len(final)<=len(emitted)`); TOKEN algo EMITS [ark] (all emitted
  ids are a prefix). Token port mutates an already-emitted word — exactly what append-only prevents. FIX: compute the
  delta + suppression at the WORD/TEXT level in C++ (needs a tokenizer/id→piece path) and compare the C++ emitted TEXT
  against the exported text gold.
- **B2 — exported text gold is DEAD.** C++ `equal_events` compares only token-ids (session_main.cpp:244/284); the exporter
  writes `event_text_bytes`/`event_collector_text_bytes` (export:377/390) but C++ never reads them → the assertion CANNOT
  catch B1. FIX: load + compare the text payloads (decode C++ events with a tokenizer, compare to the text gold).
- **B3 — collector/suppression edge cases untested (empty-collector only).** Fresh session + one finalize per row →
  collector empty before the final → only "append-all" exercised. Duplicate-final, shortened-final, non-prefix correction,
  overlap-trim against a NON-empty collector are never hit. FIX: synthetic non-empty-collector unit rows (Newark, play→
  playing, duplicate, shortened) + the multi-turn non-empty-collector path (entangles with Step 2).
- **B4 — generation-suppression NOT modeled, but it's a SCHEDULER/Phase-2 concept.** Server stale-final suppression keys on
  `expected_generation != scheduler_generation` (server.py:8941-8959) — only arises with the ASYNC scheduler advancing the
  generation, which a SINGLE synchronous replayed stream has no analog for. RESOLUTION: NARROW Step 1's claim — it covers
  final-DELTA + empty/duplicate-final suppression (the append-only collector), NOT scheduler-generation-suppression, which
  is deferred to Phase-2 (multi-stream scheduler). Update the plan/claim; do not claim generation-suppression in 1.4b.

## MAJORs (adopted)
- **M5 — interim oracle is EAGER, not AOTI.** Gold interims from finalize_ref-eager; C++ uses AOTI steady → ±1-chunk
  interim timing drift (199/200) is encoder-drift, not logic. FIX: generate the gold steady tokens with the AOTI steady
  encoder (or feed captured AOTI per-chunk tokens through a pure-Python collector) → pure-logic interim check (expect
  200/200). Keep AOTI-vs-eager parity as the separate E.2 gate.
- **M6 — C++ interim on TOKEN-change; server on TEXT-change.** Subword retokenization could change ids without changing
  text → C++ would emit an extra interim the server suppresses. FIX: text-based interim emission (folds into B1's text path).

## Decision
Step 1 NOT done. Dispatch a FIX pass:
1. Add an id→token-piece table to the bundle; C++ reconstructs surface text (SentencePiece ▁ handling), computes the
   WORD-level append-only delta + TEXT-change interim, and compares emitted TEXT (delta/collector/interim) vs the exported
   text gold (use the dead text gold). Catches Newark + retokenization.
2. AOTI-consistent interim oracle (gold steady tokens from the AOTI steady encoder) → 200/200 pure-logic.
3. Synthetic non-empty-collector unit cases (Newark/play→playing/duplicate/shortened) asserting the word-level outcome.
4. Narrow the claim: 1.4b Step 1 = final-delta + empty/duplicate suppression + text-based interim equivalence; scheduler
   generation-suppression is Phase-2 (update PLAN). Non-empty-collector multi-turn corrections validated in Step 2.
