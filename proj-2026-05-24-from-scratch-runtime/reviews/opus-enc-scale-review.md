# Opus review — 1.3b-enc-scale (per-T finalize buckets + shared weights + strip + corpus token-exactness)

Verdict: the CORE mechanism is solidly validated (1000/1000 token-exact, shared-weights flat, strip token-exact). All
findings are COVERAGE/ROBUSTNESS gaps, not "the mechanism is wrong."

## 1. [MAJOR] drop=0 first-chunk finalize is UNTESTED at scale and the single T=49 bucket is almost certainly insufficient.
The probe + 1000-shadow only ever hit drop=2, T=43..58 (`finalize_corpus_shadow.py`, `finalize_t_distribution.py`).
drop=0 occurs when `emitted_frames==0` at finalize — i.e. the utterance is finalized before ANY steady chunk drained
(very short utterance, or a first finalize). For drop=0 the finalize chunk T = `remaining` = the WHOLE utterance's mel
frames (no pre_encode_cache prepend), which is VARIABLE and can be LARGE (the original export_finalize_t2a saw a drop0
T=261). A single drop0 T=49 bucket (the fixture's clipped case) covers essentially none of the real drop0 distribution.
FIX: characterize the drop0 T distribution (short-utterance probe), and either bucket the drop0 range too OR route drop0
to an EAGER finalize fallback (drop0 is rare + off the hot path). Until then, drop0 finalize in the native runtime is
unvalidated.

## 2. [MAJOR] No out-of-range-T fallback → silent loss of the final transcript.
If a finalize produces a (drop,T) with no bucket, `finalize_corpus_shadow.py` counts it "uncovered" (silently skips) and
`cpp/finalize_main.cpp` prints "no bucket for (drop,T); skipped" → the finalize emits NO tokens, i.e. the final hypothesis
is DROPPED. In production that's a correctness failure (lost end-of-turn transcript). T is bounded for the common case
(drop2: pre(9)+remainder where remainder ≤ ~tail(<16)+final_padding(32) ⇒ T≲57-58), BUT post-stop audio accumulated during
a long debounce window could push the remainder larger → T>58. FIX: an EAGER finalize fallback for any (drop,T) without a
bucket (off hot path, so acceptable), and assert/log when it fires. Verify the T bound against server.py's post-stop
accumulation + finalize_ref.prepare_finalize_inputs (does the speculative debounce path drain post-stop before finalize?).

## 3. [MAJOR] The C++ multi-bucket path is corpus-validated only by PROXY.
`finalize_corpus_shadow.py` is a PYTHON harness (aoti_load_package + load_constants); production is
`cpp/finalize_main.cpp`. The C++ Phase B was only run on the 4 FIXTURE rows. So the C++ routing/load at corpus scale (all
16 T, the e.*/encoder.* alias, the fail-closed missing check) is validated only transitively (same buckets + same
load_constants protocol, mirrored logic). FIX: run the C++ `finalize` target over a corpus-derived multi-T fixture bundle
(not just 4 rows) to directly exercise the C++ routing across all buckets, or state explicitly that the Python harness is
the corpus oracle and the C++ mirrors it.

## 4. [MINOR] Strip token-exactness for drop=0 bucket unvalidated; drop=2 buckets transitively OK.
The 1000-shadow loads the STRIPPED buckets and got 1000/1000 token-exact across T=43..58, so every stripped drop=2 bucket
WAS exercised + validated (each T occurred). The stripped drop0 T49 bucket was NEVER run (drop0 never occurred) → its strip
is unvalidated (ties to #1). Confirm the harness loads from `stripped_finalize_buckets/` (not the unstripped originals,
which were deleted) — if so, strip is transitively proven for all drop=2.

## 5. [MINOR] NeMo-equivalence is canary-scale, not corpus-scale.
The chain is: bucketed == eager finalize_ref (1000 samples, solid) ∧ eager finalize_ref == NeMo (1.3a, only 4 canaries +
algorithm faithfulness). So bucketed-finalize == NeMo is not directly corpus-validated; it rests on the 4-canary 1.3a
result + the encoder-substitution being token-neutral (which 1000-shadow shows). Acceptable, but the NeMo link is the
weaker one — a larger finalize_ref-vs-NeMo-streaming check would close it.

## 6. [MINOR] AOTI build flakiness (T57 KeyError('version')) — needs retry logic, not a correctness risk here.
T57 failed once then compiled clean; the 1000-shadow exercised T57 token-exact, so the retry-compiled bucket is correct
(no silent corruption for T57). But the automated full-range build should add per-bucket compile retry + a post-compile
self-check, else a transient failure silently leaves a missing bucket (→ #2's uncovered path).

## Net
Mechanism validated (token-exact 1000/1000, GPU-shared, 67MB). The gaps are: drop=0 (untested + needs eager fallback or
its own buckets), out-of-range-T (needs eager fallback to avoid silent transcript loss), and direct C++-path corpus
validation. None block the mechanism; all are needed for production robustness (and fold naturally into 1.4 session work).
