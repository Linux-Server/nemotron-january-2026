# ASR-internal EOU on Nemotron-streaming-en-0.6b: negative finding

**Project**: `proj-2026-05-19-eou-endpointing` — Lower TTFS at non-inferior WER by replacing the static Silero `vad-stop-secs = 200 ms` debounce with an ASR-internal endpoint signal from the streaming RNNT decoder.

**Verdict (2026-05-20)**: NO-GO for the voice-agent use case. The parent project's `warm200` ship baseline (`docs/semantic-wer-finalization-finding.md`, commit `ef1a7a7`) remains the recommended configuration. The endpoint-evidence window — Silero's ~200 ms — is the binding floor on this checkpoint for end-of-turn detection.

## What we tested

The plan (`proj-2026-05-19-eou-endpointing/PLAN.md`) prescribed three families of EOU signals derived from the streaming RNNT decoder's emission behavior, with full per-chunk instrumentation captured under env gates `NEMOTRON_EOU_PROBE=1` + `NEMOTRON_EOU_SNAPSHOT_DIR=...` (additive, default-off, byte-identical when unset):

1. **`blank_run K`** — K consecutive 160 ms chunks where the model emits no new tokens (every encoder frame in every chunk is the RNNT blank id).
2. **`hyp_unchanged K`** — K consecutive chunks where the cumulative hypothesis length `len(y_sequence)` did not grow.
3. **`normalized_confidence τ for T ms`** — rolling-window mean of the entropy-based per-frame confidence over T ms must hit τ.

Per-chunk signals + per-session state snapshots were collected on 100 LibriSpeech samples (the `test_results.db` slice, duration-representative of the full 1000: mean 9.71 s, median 10.82 s, p90 14.75 s). The probe JSONL had 7,255 chunk rows.

## What we found

Three findings, each independently sufficient to conclude NO-GO for the voice-agent target.

### 1. Voice-agent EOU needs to discriminate inter-sentence pause from end-of-turn

Continuous-context conversational audio has a structure like:

```
sentence 1 ──[~280 ms gap]── sentence 2 ──[~280 ms gap]── ... ──[~2 s gap]── end of turn
```

Voice-agent finalize must fire only on the **last gap** (end of turn), not on the intermediate ones (sentence boundaries). Discriminating these requires either:

- A static threshold on gap duration (which is exactly what Silero `vad-stop-secs` does).
- A signal that distinguishes "end of speech" from "speech paused" by some property other than pause duration.

The plan's three signal families are all duration-thresholded on different counters (chunk-blank, hyp-unchanged, confidence-window). They are **structurally equivalent to "wait K chunks of quiet"** — they cannot do anything that a calibrated pause-duration threshold can't already do. As soon as you choose a K that ignores the 280 ms inter-sentence pauses, you've effectively reproduced Silero's ~200 ms threshold, and the wallclock latency floor is the same.

### 2. Empirically, no operating point hits the latency budget at affordable false-fire rate

A full ROC sweep across the three signal families and 28 operating points (`proj-2026-05-19-eou-endpointing/oracle_roc.py`, commit `b191e41`):

| Op point | det_lat p50 vs Silero | det_lat p95 | false_early_fire | never_fired |
|---|---|---|---|---|
| `blank_run K=1` | −4316 ms | −1213 ms | **100%** | 0% |
| `blank_run K=3` | −3940 ms | −538 ms | 97% | 0% |
| `blank_run K=5` | −3580 ms | +979 ms | 84% | 0% |
| `blank_run K=10` | +3090 ms | +11220 ms | 14% | 1% |
| `confidence τ=0.90, T=320 ms` | −4297 ms | −411 ms | 96% | 1% |

`hyp_unchanged K` is numerically identical to `blank_run K` at every K — they fire on the same chunk (a chunk that emits no new tokens IS both blank and unchanged-length).

**Viability table — every cell is empty**: at all 5 latency targets (p95 ≤ {0, 50, 100, 150, 200} ms) crossed with all 4 false-fire-rate targets ({0.5%, 1%, 2%, 5%}), the count of operating points satisfying both is **zero**.

### 3. The bimodal failure is structural, not a tuning problem

Each signal has the same shape: at low K it fires constantly during natural in-utterance gaps (high false-fire rate); at high K it doesn't fire at all in most sessions (high never-fired rate). The diagnostic (`eou_diagnostic.py`) found:

- The longest sustained all-blank run **in any session** in our 100-session collection is **71 encoder frames (~5.7 s)**.
- Only **5%** of sessions reach 60+ encoder frames of sustained-blank (4.8 s) at any point.
- Frame-level granularity (the encoder emits at 80 ms cadence; 8× downsampling from mel frames) does not change this — the model emits at 80 ms quanta, so the finer-than-chunk granularity available is only 2× finer, not 16×.

A signal that does discriminate end-of-turn from inter-sentence-pause would need to either:
- Wait long enough that natural inter-sentence pauses don't trigger it (effectively reproducing Silero with worse latency), or
- Have a non-duration feature that distinguishes the two — which the streaming RNNT decoder, on this checkpoint, doesn't expose.

## What we verified about the architecture (relevant to potential extensions)

The user asked whether we could feed silence to the decoder faster than realtime to compress the wait — connecting to the parent project's Step 7d fork-flush mechanism. Source inspection (with Codex independent verification) confirms:

- **The server has no per-chunk rate limiting**. `_handle_audio_locked` (server.py:2130) processes pending audio in 160 ms chunks back-to-back, bounded only by GPU compute time (~10 ms median per chunk on RTX 5090) and the global `inference_lock`. The CPU/GPU ratio gives a ~16× theoretical compute-vs-realtime ceiling for synthetic input.
- **The 7d fork-flush already uses synthetic-faster-than-wallclock**: `_build_continuous_finalize_fork` (server.py:1792) appends `(R+1) * shift_frames * hop_samples` = **320 ms of synthetic zeros** at rc1, then `_process_final_chunk` (server.py:2389) makes a **single** `conformer_stream_step` call with the combined mel (final flush already passes `keep_all_outputs=True` — server.py:2471). The 320 ms of synthetic zeros takes ~20–60 ms of GPU time.
- **Extending the fork-flush to longer synthetic silence is architecturally feasible** but does not solve the discriminative problem above. Even with an unbounded-K synthetic-silence pump at faster-than-wallclock, the K threshold required to **not** fire on inter-sentence pauses is the same as Silero's, and the wallclock saved is bounded by the per-step GPU overhead.
- **The model is auto-regressive in its cache** (`cache_last_channel/time/len` + `previous_hypotheses`). Chunks cannot be parallelized within a session, but they can be streamed sequentially as fast as the GPU allows.
- **No streaming↔batch decoder mode switch is needed** for any of these extensions; we stay in streaming mode and feed more chunks. The historical "streaming vs batch" discussion (`docs/parakeet-streaming-vs-batch-analysis.md` from the parent-of-parent project) was about a different question.

## Implications for the parent project's TTFS budget

The parent project established (`docs/ttfs-latency-explainer.html`, commit `9b817e4`) the two-floors framing:

- **Modeled-formula floor (~175 ms)**: `endpoint_wait + rc1_modeled + synthetic_flush + transport`, under the locked additive budget with synthetic prefix. This is the *analytical* lower bound assuming an oracle endpoint signal.
- **Endpoint-evidence floor (operational)**: the wall-clock time required to **observe** "speech has ended" with acceptable certainty.

This negative finding pins down what the operational floor actually is on this checkpoint:

- For **voice-agent end-of-turn detection**, the endpoint-evidence floor is **Silero's `vad-stop-secs = 200 ms`**, full stop. ASR-internal signals from this checkpoint's streaming RNNT decoder cannot distinguish inter-sentence pauses from end-of-turn without a calibrated pause-duration threshold; once you have a pause-duration threshold, you've reproduced Silero with a longer wallclock cost (no per-encoder-frame advantage; 80 ms decoder cadence isn't significantly finer than Silero's 30 ms VAD frame rate).
- For **streaming transcription** (per-sentence commit, no LLM gate on finalize), the picture would differ — ASR-internal signals at `K=3` give per-sentence commit at ~240–400 ms after the model settles, which is BELOW Silero's `vad-stop-secs`. But that is a different use case and the parent project's ship target was the voice-agent path.

The `warm200` shippable (parent commit `ef1a7a7`, full WER 2.07%, slice-A −1.29 [−1.84, −0.74], slice-B −1.39 [−1.96, −0.87] paired Δ vs default `''`, budget p95 325.8 ms, TTFS p95 366 ms) **is the recommended production configuration**. This EOU project does not change that recommendation.

## Caveats and limitations

- **Sample size**: 100 sessions, not 1000. The 100 in `test_results.db` are duration-representative of the full 1000 (mean 9.71 s vs 9.59 s; median 10.82 s vs 10.90 s; p90 14.75 s vs 14.90 s), so the ROC shape is statistically representative. Effect sizes are huge (100% false-fire at K=1; 95% never-fired at K=60); rescaling to 1000 would not flip the verdict.
- **Single checkpoint**: this is `nvidia/nemotron-speech-streaming-en-0.6b`. A different checkpoint with an **explicit EOU head** (a trained classifier on the encoder output predicting speech-ended) would change the picture. The multilingual checkpoint `NVIDIA-Nemotron-3.5-ASR-Streaming-Multilingual-0.6b` (queued for benchmarking — see [[multilingual-checkpoint-next-target]] memory) does not appear to have an EOU head either, but should be re-tested if/when this analysis is revisited.
- **Single language (English) and single corpus (LibriSpeech)**: speech-pattern variability across languages and corpora could change the inter-sentence-pause distribution. Conversational vs read speech would in particular have different gap structure.
- **session_id ↔ benchmark_batch_index join uses temporal-order fallback** (`oracle_roc.py`): the client finalize telemetry lacks a `session_id` field, so the join assumes the benchmark processes samples sequentially with one WebSocket session per sample. This held in our run but is a documented limitation.
- **Continuous-context multi-finalize correction**: the diagnostic initially used the FIRST `vad_stop` per session to define "post-silence." Codex's audit caught that 65 of 100 sessions have multiple `finalize_events`; re-analysis using the LAST `vad_stop` lowers the "post-silence emissions" rate from 70% to 16% and reveals those emissions are mostly continued speech from the next utterance in the same session, not hallucinated tokens during true silence. The model's silence behavior is well-behaved; the discriminative problem is between brief inter-sentence pauses and long end-of-turn pauses, not between speech and silence.

## What was built

All commits on branch `khk/20260516`. Project-scratch under `proj-2026-05-19-eou-endpointing/`.

| Step | Artifact | Commit |
|---|---|---|
| 1 | Server EOU probe instrumentation + client-acceptance bypass | `13a6846` parent + `a53fba8` nested |
| 2 | Per-chunk state snapshots + collect-signals manifest | `f6fc884` |
| 2b | `rc1_stability.py` — chunk-pair classification analyzer | `dcdee00` |
| 3 | `oracle_roc.py` — ROC sweep across signal families | `b191e41` |
| diagnostic | `eou_diagnostic.py` — frame-level analyses (Q4 in this finding) | (in this commit) |

Steps 4, 5, 6, 7 of the original plan are **not run**. Step 4 (fork-flush oracle proxy) and Step 5 (online prototype) were contingent on Step 3 passing the ROC GO/NO-GO #1, which it did not.

## Reproducibility

To reproduce the negative finding:

1. Build the project state at commit `b191e41` plus the diagnostic from this commit.
2. Start the server with `NEMOTRON_EOU_PROBE=1 NEMOTRON_EOU_SNAPSHOT_DIR=./eou-collect/snapshots NEMOTRON_EOU_CLIENT=1 NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=150 NEMOTRON_FORK_ASSERT=1 NEMOTRON_RUN_TAG=eou_step2_collect NEMOTRON_TELEMETRY_DIR=./eou-collect/telemetry` and `--right-context 1`. (See `proj-2026-05-19-eou-endpointing/runbook.md`.)
3. Run `stt-benchmark run --services nemotron_local --model eou_step2_collect --vad-stop-secs 0.2 --no-skip-existing`. Note: without `--test` to capture the full 1000; the analysis presented here is on the 100-sample test slice, which is duration-representative.
4. Run `oracle_roc.py` and `eou_diagnostic.py` against the produced telemetry. Numbers should match the tables above to within sample-noise.
