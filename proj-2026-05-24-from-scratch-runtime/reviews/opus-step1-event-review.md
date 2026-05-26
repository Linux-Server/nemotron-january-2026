# Opus review — 1.4b Step 1 (event-stream / final-delta / generation-suppression equivalence)

Verdict: **Step 1 bar MET for the critical part** (final-delta + suppression + collector equivalence EXACT; word-vs-token
delta proven). One benign, well-understood residual: interim-partial TIMING can lead the eager reference by ≤1 chunk under
AOTI drift (1/200), finals always exact.

## 1. Word-vs-token delta equivalence — PROVEN, not assumed (the scariest concern is handled).
`export_session_bundle.py` computes finalize_ref's true WORD-level delta (`_continuous_append_only_delta` on text,
imported from finalize_ref) AND a token-level `_append_only_delta_tokens`, then FAIL-CLOSED asserts
`tokenizer.ids_to_text(token_delta) == text_delta` and `ids_to_text(collector_tokens) == collector_text` per event ("token/
text event proxy mismatch" → raise). So the token-id payloads the C++ compares are VERIFIED to decode exactly to the
word-level deltas finalize_ref emits — for every event in the bundle. The N=20 export succeeding => equivalence held for
all events. `session_main.cpp equal_events` compares kind+tokens+collector_tokens (token-id), which is sound given the
export's proof. (If a future corpus hits a subword case where word≠token delta, the EXPORT raises — it can't silently
ship a wrong gold. Good fail-closed design.)

## 2. Interim-timing drift (1/200) — benign AOTI artifact; finals exact. The real question is the ORACLE.
N=20: event_delta 20/20 (event counts 11-45/utt). N=200: 199/200 — sample 198 emits one interim partial one chunk EARLIER
in C++ than the gold, finals exact. Root cause: the gold is finalize_ref using the EAGER encoder; the C++ session uses the
AOTI steady encoder, whose ~1e-2 drift can shift a near-threshold token's FIRST appearance by ±1 chunk. So "interim stream
EXACT vs EAGER" is partly the wrong bar — the production runtime IS AOTI, and the AOTI-vs-eager cumulative/WER equivalence
is already validated (E.2, 1000/1000). The user-visible FINAL transcript + delta + suppression are exact; a 1-chunk-early
partial is behaviorally fine. RECOMMENDATION: ACCEPT with explicit documentation that interim partial timing may lead the
eager reference by ≤1 chunk (AOTI drift, finals/delta/suppression exact); OPTIONALLY make the N>20 interim oracle
AOTI-consistent (generate the gold steady tokens with the AOTI steady encoder) to turn the event check into a pure-LOGIC
check (then expect 200/200). Do NOT chase eager-exact interim timing by changing the runtime — the drift is intrinsic and
already accepted at the token/WER level.

## 3. Suppression + collector edge cases — faithful.
The token-port mirrors finalize_ref ~144-167: common-prefix append; `final<=emitted` → empty (suppress); else suffix +
overlap-trim. emit_event records interim on `hyp != last_interim`, final with the delta, suppressed on empty/duplicate;
collector (`continuous_emitted_tokens`) updated after emit; reset on utterance boundary. `equal_events` compares the FULL
ordered stream incl count (count-mismatch flagged) + per-event kind+payload+collector — not just final state. Looks correct.
(Edge cases to confirm in the paired check: non-prefix correction final, shortened final, duplicate final — present in the
finalize_ref logic; the export's fail-closed proxy check covers their token/text equivalence.)

## 4. No regression to token/FORK_ASSERT.
The event tracking is additive; N=20 still steady 20/20 + final-token 20/20 + FORK_ASSERT 20/20 (+ event_delta 20/20). The
N=200 token gate should still pass (event-tracking doesn't change the encode/decode/state path) — confirm in the loop.

## Net
Step 1's bar (event/delta/suppression equivalence) is MET for finals (the T1-critical, user-visible part) + word-vs-token
PROVEN. Residual = interim-partial timing ≤1-chunk lead vs the EAGER oracle (benign; 1/200), which is an oracle-choice
artifact not a logic bug. Accept + document (or optionally AOTI-consistent interim oracle for a pure 200/200 logic check).
