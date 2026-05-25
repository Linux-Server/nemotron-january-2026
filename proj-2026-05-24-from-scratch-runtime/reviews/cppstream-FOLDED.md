# Paired review FOLDED — C++ streaming cache/buffer milestone (Codex + Opus)

Sources: `codex-cppstream-review.md` + `opus-cppstream-review.md`. Strong convergence on the verdict.

## Verdict
**The cache/buffer/state-carry LOGIC is CORRECT** (both reviews, independently verified vs `server.py`):
- Ring update byte-faithful to `_update_mel_frame_ring` (cat → `[-PRE:]`; first-chunk init correct).
- First-vs-steady branch + drop (0/2) matches `server.py:8317-8322`.
- Cache threading correct (no clone needed for traced outputs; **WILL need clone when the encoder becomes CUDA-graph
  replay in Phase 3** — graph-owned static buffers).
- Decoder state-carry (g,h,c) + max_symbols + blank faithful to the verified ref.

**BUT the current `steady_main.cpp` is a one-clip / one-geometry / steady-only / token-exact PROOF-OF-CONCEPT, not a
robust component.** The paired review caught latent gaps the friendly 320=20×16 clip hid. So: **milestone (cache/buffer
logic) = PASS; production-readiness = NOT yet** — these gaps are the work before 1.2b is "done."

## Action list (folded, prioritized)
1. **[BLOCKER] Partial/short chunk** (Codex#1, Opus#2): `Tm % 16 != 0` (and `Tm<16`) feeds a wrong shape to the
   frozen-trace encoder → crash/garbage. **Fix:** dynamic-length encoder export (torch.export) OR pad-to-bucket + true
   `length` masking + fail-closed if no bucket. **Add parity tests: Tm = 1,15,16,17,31,319,320,321.**
2. **[BLOCKER@byte-exact / OK@T1] Token-exact-only encoder** (Codex#2, Opus#4): traced encoder drifts ~1e-5 across
   cache_len; a near-tie argmax flip → cascading divergence. Token-exact on ONE clip isn't a sound general gate.
   **Fix:** byte-exact encoder via `torch.export`/dynamic; add a per-chunk exact (enc_out/clc/clt/clcl) comparison vs
   eager + an **argmax-margin report** to catch unsafe near-ties. (At the T1 bar this clip passes; for a sound gate, do
   the byte-exact path.)
3. **[MAJOR] Finalize absent** (both): steady-only; transcript truncated ("...one lime", no "?"). Implement the
   server-faithful finalize (`keep_all_outputs=True`, dynamic final `chunk_mel`, `server.py` final path) → 1.3.
4. **[MAJOR] Incremental-STFT preprocessing** (Opus#1): the demo uses offline full-clip mel sliced into chunks, NOT
   server.py's incremental per-chunk STFT + `raw_audio_ring`. Port + validate the streaming preprocessing (0.8 only
   covered the fixed-size constant-plan buffer, not the across-chunk STFT continuity).
5. **[MAJOR] Per-session state object** (Codex#5): factor `StreamState` (cloned clc/clt/clcl, g/h/c, ring, emitted,
   hyp) + reset-per-utterance — the 1.0 state-ownership design; the demo's one-shot globals would contaminate across
   utterances. Add a 2-utterance test (2nd = silence → empty).
6. **[MAJOR] Metadata assertions** (Codex#4): export shift/pre/drop/blank/max_symbols + expected shapes into the
   bundle; assert at startup (don't hard-code + hide drop in artifact choice → silent model/artifact mismatch).
7. **[MINOR] Range-check `enc_len_out`** (Codex#6): assert `0 <= To <= eo.size(2)` with chunk/pos/clcl in the message.

## Takeaway
The paired review at this milestone earned its keep: the one-clip T1 pass hid a crash bug (partial chunks), a latent
nondeterminism (encoder near-tie flip), and two production-robustness gaps (session state, metadata). The structural
cache/buffer port is sound; the path to a robust 1.2b is the action list above (with the byte-exact encoder via
torch.export/dynamic and the incremental-STFT preproc being the two that also raise the bar from T1→T2a).
