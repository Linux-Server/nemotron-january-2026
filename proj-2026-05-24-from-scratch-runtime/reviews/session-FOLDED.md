# 1.4 single-session paired-review FOLD (authoritative verdict)

Two reviews — `opus-session-review.md` (me) + `codex-session-review.md` (Codex). Strong convergence; Codex's framing is
the precise one and is ADOPTED: **the full "Phase-1 single-stream T1 behavioral-equivalence" gate is NOT met.** What IS
met (narrowed, real): the C++ session's MEL-level cumulative steady+final tokens equal `finalize_ref` over 200 corpus
streams, and the forked finalize does not mutate the checked parent state (200/200 + FORK_ASSERT). That proves the native
COMPUTE COMPOSITION (mel→steady-AOTI→decode→finalize-bucket, wired by the state machine + fork isolation) — the hard,
novel integration — and discharges the deferred C++-corpus validation (enc-scale review B3). It is NOT the full T1 bar.

**Correction:** the "Phase-1 exit gate MET" phrasing (commit 423b434, PHASE1-STEPS, memory) is OVER-CLAIMED and corrected
here to "compute-composition validated; full T1 behavioral gate has a named residual."

## What the gate MISSES vs the 0.4-decision T1 bar (residual; both reviewers)
- **BLOCKER — event-stream / final-delta / generation-suppression equivalence (0.4 §bar).** The gate asserts only the
  CUMULATIVE final token vector (session_main.cpp:159-178/912-915). A runtime could emit wrong partials, wrong final
  deltas, duplicate/stale/empty finals, or no events and still pass. Production is text-DELTA based
  (`_continuous_append_only_delta`, server.py:392-417; emit/suppress 8936-9012) with stale-generation suppression
  (server.py:8694-8729/8882-8959). The exported `finalize_new_tokens` (token suffix) is NOT the production text-delta and
  is unused by C++. FIX: an EVENT-STREAM oracle — capture ordered server events (type/text/is_final/delta/suppression
  reason/collector state) and compare event-for-event; keep cumulative tokens as one invariant, not the gate.
- **BLOCKER — mel-fed, not audio-fed.** C++ consumes Python-produced new_mel + final_chunk_mel (export_session_bundle.py
  63-123; session_main.cpp:826/882). The C++ audio→mel preprocessor, raw-ring/STFT-boundary/valid_samples/sample→mel
  alignment, AND the finalize remainder-from-audio assembly (server.py:4518-4567/9903-9932) are all trusted from Python.
  "Single native stream" overstates it — it's a single native MEL stream. FIX: audio-fed gate (PCM→C++ preproc+remainder),
  compare per-chunk mel hashes + final geometry vs Python, then the token/event assertions.
- **BLOCKER — multi-turn CONTINUOUS context-retention untested.** Each bundle row is an independent fresh Python session
  (export 98-127) and C++ cold-resets per row (session_main.cpp:968-969). Production's speculative finish RETAINS context
  across a finalize (server.py:9014-9035; cold reset only for close/end). The most production-relevant path —
  finalize→keep caches/decoder/collector→more audio→next steady+final — is never exercised. FIX: multi-turn bundles
  (turn A → speculative finalize → turn B continuation in the SAME session → 2nd finalize → true-boundary cold reset);
  assert retained caches/collector/deltas across the whole session; do NOT reset between turns for this test.
- **MAJOR — full-1000 WER + per-utterance WER not enforced** (0.4 §bar). The harness reports row pass-counts, no reference
  text / WER / CI bound (export default --n 20). FIX: run all 1000 with references + the agreed WER metric + CI/per-utt bounds.
- **MAJOR — oracle is finalize_ref (4-canary proxy for the server), not the shipping server.** "Equivalence to Python" =
  equivalence to our synchronous reference, not the server's emitted JSON (WS sends, send-failure, scheduler generations,
  continuous/cold branching). FIX: add a production-server oracle run for the same scripts; keep finalize_ref as the unit spec.
- **MAJOR — first chunk is TorchScript (enc_first.ts), not native AOTI** (session_main.cpp:784-795/950-952; ~1e-5 trace
  drift). Token-exact here is an argmax-margin observation, not native/byte-exact proof. FIX: a first-chunk AOTI path, or
  drop the unqualified "pure native" claim + add first-chunk drift/margin reporting.
- **MAJOR — finalize geometry pre-baked by the bundle; C++ doesn't recompute remainder math** (remaining_frames/
  padded_total exported but unchecked; session_main.cpp:881-887). FIX: recompute padded_total/total_mel/remaining/T from
  raw counters in C++, assert vs bundle.
- **MAJOR — AOTI steady output ownership / recurrent-cache aliasing assumed, not proven** (outputs installed directly as
  caches + fed back, session_main.cpp:772-810). FIX: clone-on-assign or explicit alias assertions + a long-stream cache
  stability check independent of tokens.
- **MAJOR — per-row reset masks lifecycle bugs; FORK_ASSERT omits `mode`/audio-ring counters.** A bug leaving the parent
  in the wrong post-finalize lifecycle/ring state passes because the next row starts from init_*. FIX: assert mode/state
  transitions + ring counters in ONE long-lived session (ties to multi-turn).
- **MINOR** — final_T==0 (no-pending) path under-validated; coverage isn't adversarially stratified (residual buckets /
  zero-many steady / corrections / duplicate finals / vad_start-cancel / stale-gen / cold reset).

## Verdict
**1.4 first pass = native COMPUTE-COMPOSITION equivalence achieved (token-exact vs finalize_ref, 200 corpus streams; fork
isolation; C++-corpus B3 discharged). The full Phase-1 single-stream T1 behavioral-equivalence gate is NOT yet met** — it
requires (priority order): event-stream/delta/suppression equivalence, audio-fed C++ front, multi-turn continuous
context-retention, full-1000 WER/per-utt bounds, and a server (not just finalize_ref) oracle. These are the Phase-1-finish
items (then Phase-2 = the multi-stream density win).
