# Codex C++ Streaming Cache/Buffer Review

Verdict: not milestone-gate ready for byte-exact streaming correctness. The full 320-frame fixture path mostly ports the Python loop, but the implementation is only proven for one friendly geometry. Real streaming/finalization inputs expose hard correctness gaps.

## Findings

1. [BLOCKER] `steady_main.cpp:33-39` feeds partial final chunks into fixed-geometry traced encoders.

   The loop slices `new_mel` with `min(pos + SHIFT, Tm)` at `steady_main.cpp:34`, so any `Tm % 16 != 0` produces a short final chunk. That short tensor is then passed either to `enc_first` at `steady_main.cpp:36` or to `enc_steady` at `steady_main.cpp:37-39`. Those modules were traced only at first=`16` frames and steady=`9 + 16 = 25` frames (`export_stream_encoder.py:48-56`), and the exporter explicitly rejects non-multiple clips with `assert Tm % shift == 0` at `export_stream_encoder.py:35`. The passing clip is 320 frames, so this path is untested by construction.

   Impact: real utterances whose mel length is not a multiple of 16 can crash in TorchScript or, worse, run a trace whose cache/shape assumptions do not match the input. This is a correctness blocker for partial last chunks and for short utterances (`Tm < 16` hits the same issue on the first chunk).

   Concrete fix: replace the traced per-geometry encoder with a dynamic export/scripted encoder that accepts first lengths `1..16` and steady lengths `PRE+1..PRE+16`, including dynamic cache length. If fixed buckets are retained, add explicit final buckets and fail closed in C++ when no exact bucket exists. Add parity tests for `Tm = 1, 15, 16, 17, 31, 319, 320, 321`.

2. [BLOCKER] `steady_main.cpp:39,42,48,52-53` relies on a known non-byte-exact encoder trace across changing cache lengths.

   `steady_main.cpp:39` runs the traced module while `steady_main.cpp:42` threads `clc/clt/clcl` from the previous output into the next chunk. That is the right cache-threading shape, but the export already documents that the traced encoder is not byte-exact when `cache_last_channel_len` changes: `export_stream_encoder.py:84-86` says the trace freezes cache-length-dependent ops and drifts by about `1e-5`. `steady_main.cpp:48` then takes an unconstrained `argmax`, and any flipped near-tie token changes the predictor state at `steady_main.cpp:52-53`, making all later chunks divergent.

   Impact: token-exactness on one 20-chunk clip is not a sound correctness argument. The bug is latent until a chunk has a small enough logit margin, longer context, silence/noise, or different cache length trajectory. This fails the stated byte-exact vs NeMo streaming bar.

   Concrete fix: do not gate native streaming on this traced encoder. Move to `torch.export`/dynamic or another export path that is byte-exact against eager `cache_aware_stream_step` for every chunk and cache length. Add a long-run test that compares per-chunk `enc_out`, `enc_len`, `clc`, `clt`, and `clcl` exactly against the Python reference, plus an argmax-margin report to catch unsafe near ties.

3. [MAJOR] `steady_main.cpp:33-39` has no server-faithful finalization path (`keep_all_outputs=True`) and no final geometry.

   The exported `Step` hardcodes `keep_all_outputs=False` in `export_stream_encoder.py:45-46`, and `steady_main.cpp:39` has no way to select a final encoder behavior. The server final path can send a dynamically sized final `chunk_mel` and calls `_conformer_stream_step(... keep_all_outputs=True, ...)` (`server.py:9988-9999`). The C++ loop always processes `<=16` new frames per iteration and always uses the steady trace after the first chunk.

   Impact: this reproduces the narrow Python steady-loop fixture, not the server's final flush semantics. Real utterance finalization can have arbitrary remaining frames and must flush encoder outputs differently. Without a final branch, endpoint/final transcript behavior can diverge even if all steady chunks pass.

   Concrete fix: define and implement a native finalization mode: construct the final `chunk_mel` exactly like the server, export a dynamic final encoder path with `keep_all_outputs=True`, and validate final hypotheses against `server.py` on non-multiple and short utterances.

4. [MAJOR] `steady_main.cpp:10,36-38` hard-codes streaming geometry and hides `drop_extra` inside artifact choice.

   `SHIFT=16`, `PRE=9`, `BLANK=1024`, and `MAX_SYMBOLS=10` are compiled into `steady_main.cpp:10`. The first/steady branch at `steady_main.cpp:36-37` chooses between two modules with `drop_extra` baked in, and `steady_main.cpp:38` only passes the chunk length. The server derives `shift_frames`, `pre_encode_cache_size`, and `drop_extra` from the model streaming config (`server.py:1593-1601`), while the Python exporter also reads them from `e.streaming_cfg` (`export_stream_encoder.py:24-25`).

   Impact: a mismatched artifact/model/config silently corrupts ring construction, chunk length, blank handling, or max-symbol behavior. This is exactly the class of bug cache-aware streaming ports are prone to: the program still runs but is no longer implementing the model's streaming contract.

   Concrete fix: export metadata (`shift`, `pre`, `drop`, `blank`, `max_symbols`, expected first/steady/final input shapes) into the bundle and assert it at startup before decoding. Prefer passing `drop_extra` explicitly into a single dynamic encoder wrapper, or at least record it in each module's metadata and check it before use.

5. [MAJOR] `steady_main.cpp:24-31,42,53,56-58` is one-shot state, not a reusable session state machine.

   The native loop initializes encoder caches, decoder state, ring, `hyp`, and `emitted` once at `steady_main.cpp:24-32`, mutates them at `steady_main.cpp:42`, `steady_main.cpp:53`, and `steady_main.cpp:56-58`, and never exposes a reset boundary. That is fine for a command-line fixture that exits, but it is not safe as a native streaming pipeline under multi-utterance or multi-session use.

   Impact: if this logic is lifted into a long-lived runtime without a per-session state object, the next utterance can inherit `ring`, encoder caches, `g/h/c`, and previous tokens. That creates deterministic transcript contamination across utterances.

   Concrete fix: factor a `StreamState`/session object containing cloned initial `clc/clt/clcl`, cloned initial `g/h/c`, `ring`, `emitted_frames`, and hypothesis state. Reset it on every new utterance/session and add a two-utterance test where the second utterance is silence and must decode as empty.

6. [MINOR] `steady_main.cpp:41` trusts `enc_len_out` without shape/range checks.

   `steady_main.cpp:41` converts the second encoder output to `To`, then `steady_main.cpp:45-46` slices `eo` up to that value. The Python path assumes model sanity too, but the C++ port is now consuming serialized artifacts and dynamic cache state. If a bad trace, partial shape, or artifact mismatch returns `To > eo.size(2)`, the decoder will slice out of range or process invalid frames.

   Impact: this will show up as a late decoder crash or silent wrong output, far away from the encoder artifact mismatch that caused it.

   Concrete fix: assert `0 <= To && To <= eo.size(2)` after every encoder call, and include chunk index, `pos`, `chunk.size(2)`, `clcl`, and module kind in the failure message.

## Checks Without Findings

- Ring update is byte-faithful for full steady chunks. `steady_main.cpp:56-57` computes `cat(previous_ring, new_mel)` or `new_mel`, then slices the last `PRE` frames. That matches `_update_mel_frame_ring` in `server.py:4569-4575`. The first-chunk ring init is also correct: `ring` is undefined until after the first encoder call, then becomes the last 9 frames of the first `new_mel`.

- First-vs-steady branch is correct on the validated geometry. `steady_main.cpp:36-37` matches the server split at `server.py:8317-8322`: first chunk uses only `new_mel`/drop `0`, steady chunks prepend `mel_frame_ring`/drop `2`. In C++, the drop value is implicit in `enc_first.ts` vs `enc_steady.ts`; that is acceptable only if metadata checks are added as noted above.

- Cache threading is correct for the current traced TorchScript modules. `steady_main.cpp:39` passes `clc/clt/clcl`, and `steady_main.cpp:42` reassigns them from the encoder outputs before the next chunk. This mirrors `stream_decode.py:59-62`. No clone is required for ordinary traced functional outputs. If this is later replaced with CUDA graph/static-output replay, the session must clone graph-owned output buffers before the next replay overwrites them.

- Decoder carry is faithful to the verified Python reference. `steady_main.cpp:45-53` matches `ref_decode.py:24-35`: carry `g/h/c`, break on blank without changing state, update predictor state only after non-blank, and cap each frame at `MAX_SYMBOLS=10`. The max-symbol saturation behavior is the same as the Python reference: after 10 non-blanks, it advances the encoder frame without requiring a blank.

- Loop bounds and mel slicing are correct for `Tm` divisible by 16. `steady_main.cpp:33-34` covers `[0,Tm)` exactly once with no overlap in `new_mel`; the only loop-bound defect is the partial-geometry blocker above.

## Top 5

1. Block the gate until partial/final chunk geometries are supported or explicitly rejected.
2. Replace the cache-length-sensitive traced encoder; token-exact on one clip is not enough.
3. Add a server-faithful finalization path with `keep_all_outputs=True`.
4. Export and assert streaming metadata instead of hard-coding geometry/drop/blank constants.
5. Factor per-session state/reset before reusing this loop beyond the one-shot fixture.
