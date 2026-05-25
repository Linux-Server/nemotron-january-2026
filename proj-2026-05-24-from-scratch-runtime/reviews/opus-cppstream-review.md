# Opus adversarial review — C++ streaming cache/buffer correctness (`steady_main.cpp`)

Reviewing the 1.2b C++ streaming port vs the verified Python ref (`stream_decode.py`) + `server.py` ground truth.

## [MAJOR] 1. The "streaming" preprocessing is OFFLINE full-clip mel, NOT true incremental streaming — a real fidelity gap
`steady_main.cpp` consumes `sb.mel` = the **full clip preprocessed once**, then slices it into chunks
(`mel.slice(2,pos,...)`). But `server.py` does **incremental per-chunk STFT** with a **`raw_audio_ring`** for STFT
boundary continuity (the real-time path: audio arrives in pieces, each chunk's mel is computed from accumulated raw
audio + the ring). **Full-clip-mel-then-slice ≠ incremental-STFT-then-mel at chunk boundaries** (STFT windowing across
the boundary differs). So this validates the encoder-cache + decode-state streaming, but **NOT the streaming
preprocessing** — on a true real-time stream the mel inputs would differ and the transcript could diverge.
- 0.8 validated the *fixed-size constant-plan* preprocessor (byte-exact, deterministic) — but that's the per-call
  buffer, **not** the incremental-across-chunks STFT + raw_audio_ring. **That path is unported + unvalidated.**
- **Fix/next:** port server.py's incremental STFT (raw_audio_ring + the fixed-K constant-plan buffer per chunk) and
  validate the streaming mel matches server.py's per-chunk mel. Until then, label this "streaming-encoder+decode
  validated; streaming-preprocessing pending," not "streaming pipeline."

## [MAJOR] 2. Partial last chunk is a latent crash/wrong-output bug (shape-frozen trace)
`new_mel = mel.slice(2, pos, std::min(pos+SHIFT, Tm))` (line ~36) handles a partial last chunk by shape — but
`enc_steady.ts` is a **trace frozen at 25 frames** and `enc_first.ts` at 16. If `Tm % 16 != 0`, the last chunk's
`chunk` is `cat(ring[9], new[<16])` ≠ 25 → **shape mismatch → runtime error or silently-wrong output.** The test clip
is 320 = 20×16 exactly, so this is **never exercised** — a real clip of arbitrary length WILL hit it.
- **Fix:** pad the last chunk to the trace shape (and set `length` to the true count so the encoder masks the pad), OR
  export a dynamic-length encoder (torch.export), OR special-case the tail. Add a non-multiple-of-16 test clip.

## [MAJOR] 3. Finalize is absent — the pipeline is steady-only (incomplete utterance)
The transcript is "How much juice is in one lime" (11 tok) — **missing the final token(s)/"?"** that come from the
finalize flush (trailing-padding encode + final decode). So this is the STEADY path only; a real utterance needs the
finalize path (1.3). Not a bug, but the "pipeline" is incomplete and the transcript is knowingly truncated. State it.

## [MINOR] 4. Token-exactness relies on no near-tie argmax flip under the ~1e-5 encoder drift
The encoder trace is token-exact but ~1e-5 off byte-wise (frozen cache_len ops). The greedy `argmax` is robust to 1e-5
**unless** a frame has two near-tied logits within ~1e-5 → a flip → divergent token → cascading divergence. Low
probability on clean speech, but a latent nondeterminism risk vs NeMo. The T2a fix (torch.export/dynamic → byte-exact)
removes it. Acceptable at the T1 bar; note it.

## [MINOR] 5. Cache-threading + ring: CORRECT vs server.py (verified)
- Ring (line ~57): `cum = ring? cat(ring,new):new; ring = cum[-PRE:]` matches `_update_mel_frame_ring` (cat then
  `[-9:]`); first-chunk init (ring undefined → new[-9:]) matches. ✓
- First/steady branch (`emitted==0`) + drop baked into enc_first(0)/enc_steady(2) matches server.py:8317-8322. ✓
- Cache threading (clc/clt/clcl reassigned from `out[2,3,4]`, fed next) — correct; **no clone needed** here because
  these are plain TorchScript-forward outputs (fresh tensors), NOT CUDA-graph static-pool outputs (which WOULD need
  cloning — important once the encoder moves to graph replay in Phase 3). ✓ (flag the clone requirement for Phase 3.)
- Decode state-carry (g,h,c) + max_symbols + blank — matches the verified ref (which is 18/18 + C++ 2-chunk-carry). ✓

## [MINOR] 6. `emitted` vs first-chunk: works, but coupled to "no partial first chunk"
`emitted==0` selects the first geometry; if `Tm < 16` the first chunk is partial → enc_first(16) shape mismatch (same
class as #2). Edge case; covered by the #2 fix.

## Top 5
1. **The "streaming" preprocessing is offline full-clip mel, not incremental STFT** — port + validate server.py's
   raw_audio_ring incremental preproc; relabel until done.
2. **Partial-last-chunk shape mismatch** (frozen trace) — pad/dynamic-export + add a non-16-multiple test clip.
3. **Finalize absent** — steady-only; transcript truncated; needs 1.3.
4. **Token-exact-only encoder (~1e-5 drift)** — near-tie argmax flip is a latent risk; T2a (torch.export/dynamic) fixes.
5. Cache/ring/state-carry logic is **correct vs server.py** (verified) — the structural port is sound; the gaps are
   preprocessing fidelity, partial chunks, finalize, and byte-exactness — not the cache/buffer logic itself.
