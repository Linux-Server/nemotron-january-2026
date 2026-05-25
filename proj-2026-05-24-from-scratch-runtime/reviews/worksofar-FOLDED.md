# Work-so-far review — FOLDED (accepted; claims corrected)

Source: `codex-worksofar-review.md`. **The review is correct: several claims were materially over-banked.** This folds
it by ACCEPTING the findings and stating the corrected, defensible claims. Net: this is a **promising narrow prototype
with a strong de-risking spine**, NOT a validated production runtime. The corrections below are now authoritative; the
individual docs are annotated to match.

## What is ACTUALLY proven (defensible)
- The **RNNT greedy decode ALGORITHM** is reproduced from scratch and matches NeMo's **token sequence** (`y_sequence`)
  on a **narrow fixture set** (3 real clips 12/20/74-tok + 3 all-blank), in Python and in C++ (the C++ self-test covers
  **one** bundled clip).
- A **steady streaming encoder** is **byte-exact vs eager via `torch.export`** for **one clip's cache_len trajectory**
  (steady geometry only; first-chunk run eager).
- The fixed-size preprocessor is run-to-run deterministic + TorchScript-byte-exact (NOT the incremental-STFT path).
- The components RUN composed in C++ (full non-streaming pipeline + a steady streaming loop) at the **token** level on
  one clip.
- The CUDA devel container gives a working nvcc/GPU build env (glibc fix).

## Corrected claims (the over-banked ones)
1. **Density "L40S ~2–2.5× (32–40/box)" → PRELIMINARY / likely-optimistic.** [Codex#1,#3] The microbench decode is a
   **pure host sleep with `decode_gpu_iters=0`** that *overlaps* the async encoder — it turns encoder time into free
   overlap the **real decode (joint/predict GPU calls + host argmax syncs, ordered AFTER encoder output) may not have**,
   and the "GPU-bound at 32–40 (lanes=12==8)" conclusion is under that zero-GPU-decode model. **Action:** rerun L40S
   with an **encoder-event-dependent realistic decode chain** (exported joint/predict) + `decode_gpu_iters`/host
   sensitivity before banking any density number.
2. **"0.4 GO / finalize cleared" → CONDITIONAL GO.** [Codex#2] The deploy-target (L40S) result is **steady-only**; the
   "finalize sensitivity YES" came from a **5090 synthetic** finalize (extra replays + sleep), not a real
   `keep_all_outputs` bucket. **Action:** rerun L40S with a faithful finalize bucket; until then GO is conditional.
3. **"T2a achieved" → Python `ExportedProgram` byte-exact; C++ RUNTIME integration UNPROVEN.** [Codex#13,#14] `.pt2`
   can't be `torch::jit::load`ed; AOTI C++ build/load/compare is not done (now unblocked by the container, still TBD).
   T2a validated steady-only, one clip, one cache_len trajectory, first-chunk eager.
4. **"byte-exact pipeline / audio→streaming byte-exact" → token-exact on ONE clip; preproc is full-clip-mel-slice, not
   incremental STFT.** [Codex#6,#7,#8,#11] C++ decode self-test covers one bundled fixture; streaming uses one clip;
   preprocessing isn't the server's incremental-STFT+raw_audio_ring path.
5. **"18/18 streaming continuation" → reference SPLIT SELF-CONSISTENCY** (split==full), not NeMo `partial_hypotheses`
   equivalence (only the one-clip `stream_decode.py` checks that). [Codex#9]
6. **"BYTE/STATE-exact decode" → token(`y_sequence`)-exact only.** [Codex#10] No `dec_state`/`predictor_state`/score/
   timestamp equality checks yet. **Action:** add `LabelLoopingStateItem`-field + emitted-metadata equality.
7. **"full-chunk-only loop is server-faithful" → only NO-CRASH for non-multiple clips**; the deferred remainder's final
   semantics are unvalidated (no finalize). [Codex#6]
8. **The density "no per-frame `.item()`" premise is NOT implemented** — the C++ decode calls `argmax().item()` per
   symbol. [Codex#16] The **semantic** decode port is not the **performance** (on-GPU, sync-compressed) decode the
   density thesis assumes; don't cite semantic C++ decode as density evidence.
9. **Shared-weights is assumed but unimplemented** (lanes=32 OOM'd on per-lane model copies). [Codex#5] "32–40 in one
   process" is conditional on a shared-weight/graph-buffer design.
10. **EN-only**, no prompted/multilingual/alignments — every "production runtime" line carries this. [Codex#17]
11. **Fixture base too narrow** (one clip + noise). [Codex#11,#12] Need a corpus matrix (varied lengths, non-multiple
    endings, silence lead/trail, fast speech, punctuation, near-tie logits, **max_symbols saturation** [not "minor"],
    state snapshots) + a full-1000 T1 shadow run before any "production byte-exact" language.
12. Minor: shallow metadata assertions (add model-id/tokenizer-hash/att-ctx/version/dtype/geometry); duplicated 1.3
    rows; "byte-exact free with libtorch" is wrong → "possible only under matched versions + exported graph form +
    geometry coverage + determinism + C++ integration." [Codex#18,#19,#20]

## Revised honest status (one line)
A **de-risked narrow prototype**: the decode algorithm + steady encoder are reproduced and byte/token-exact on a thin
fixture set, and run in C++ — but **production density, byte-exact multi-stream/finalize/multi-utterance behavior, and
the C++ T2a runtime are NOT yet proven.** The session's value is retiring the *algorithmic* risks + building the
verification scaffolding; the *systems/production* claims need the corpus matrix, realistic-decode density rerun,
finalize, AOTI C++, state machine, and on-GPU decode before they can be banked.

## Action list (folds into PHASE1-STEPS / the gates)
A. Rerun the density gate with realistic encoder-dependent decode + `decode_gpu_iters` sensitivity (L40S) → re-bank density.
B. L40S finalize sensitivity with a real `keep_all_outputs` bucket → conditional→firm GO.
C. C++ decode: iterate ALL fixtures + add max_symbols-saturation + state(`LabelLoopingStateItem`)-equality + multi-clip.
D. AOTI-compile the T2a encoder in the container → C++ byte-exact vs eager (close T2a-in-runtime).
E. Incremental-STFT preproc parity vs server; corpus fixture matrix + full-1000 T1 shadow.
F. Full Session state machine (generation suppression, finalize fork, audio rings) + finalize + reset/resume suite.
G. On-GPU / sync-compressed decode (the real "no-.item()" density premise) before density evidence from C++.
