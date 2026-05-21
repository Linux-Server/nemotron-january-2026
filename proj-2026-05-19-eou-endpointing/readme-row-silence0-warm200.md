# README Results-Summary row: silence0_warm200 (full 1000)

Generated 2026-05-20 from the full-1000 `silence0_warm200_c12` run (commit `733da5e`)
+ Claude semantic-WER judge. Format matches `stt-benchmark/README.md` §"Results Summary".

## The row

```
| Service | Transcripts | Perfect | WER Mean | Pooled WER | TTFS Median | TTFS P95 | TTFS P99 |
|---------|-------------|---------|----------|------------|-------------|----------|----------|
| Nemotron 0.6b (silence0_warm200) | 100.0% | 76.1% | 1.94% | 1.95% | 220ms | 247ms | 263ms |
```

Column derivations (from `results.db`, model_name=`silence0_warm200_c12`):
- **Transcripts** 100.0% — 1000/1000 returned non-empty transcription, 0 errors.
- **Perfect** 76.1% — 761/1000 at 0% semantic WER (`wer_metrics.wer == 0`).
- **WER Mean** 1.94% / **Pooled WER** 1.95% — Claude semantic-WER judge, 1000 samples.
- **TTFS Median/P95/P99** 220/247/263 ms — `ttfb_seconds` = (final received) − (real-speech-end).

## Leaderboard context (vs the 11 services in the README)

- **TTFS P95 247 ms would be #1 (best)** — below current leader Soniox (281 ms) and Deepgram (298 ms).
- **WER Mean 1.94% ranks ~6th of 12** — Deepgram (1.71%) / AWS (1.68%) tier; behind Azure
  (1.21%) / Soniox (1.25%) / Speechmatics (1.40%); ahead of Smallest AI (2.30%) and below.
- Summary: **best-in-class latency at upper-mid-pack accuracy.**

## ⚠ CRITICAL CAVEATS — this row is NOT a clean apples-to-apples README measurement

**It did NOT go through the Pipecat benchmark pipeline.** It came from the custom raw-WebSocket
harness `run_full1000_conc12.py` (required because the standard `stt-benchmark run` CLI is
sequential-only and the framework is unchangeable, so "concurrency 12" forced a custom client).
The harness bypasses: `synthetic_transport`, **the Silero VAD analyzer**, `nemotron_local_stt.py`,
and the `TranscriptionCollectorObserver`.

1. **VAD differs (the big one).** The harness sends ONE `vad_stop` at the very end (after a fixed
   200 ms appended silence). The real pipeline runs Silero, which fires `vad_stop` at every
   detected pause → multi-segment finalization on ~65% of samples (the 650/1000 multi-segment
   cases). The append-only delta design *should* make "many segments concatenated" == "one big
   segment", so final transcripts should be equivalent — but **not guaranteed byte-identical** on
   multi-segment samples. Therefore **WER 1.94% is a close estimate, not the exact pipeline number.**
2. **TTFS was cross-validated; WER was not.** The harness's TTFB matched the real benchmark CLI to
   within noise on the same config (both ~213 ms on `silence_0_test`, a real-pipeline run). The
   harness's WER was never validated against a real-pipeline WER on the same samples.
3. **Measured under 12-way concurrency.** The 247 ms p95 is concurrency-inflated; single-session
   the same config measured **215 ms p95** (100-sample `silence_0_test`). The other 11 README
   services were measured sequentially, so a sequential Nemotron run would be ~215 ms — even
   further ahead of Soniox.

## To get a truly-comparable README row

Run the **real Pipecat pipeline** sequentially on the full 1000:
```
# server: NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 NEMOTRON_CONTINUOUS=1 --right-context 1
cd stt-benchmark && NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 \
  .venv/bin/stt-benchmark run --services nemotron_local --model silence0_warm200_seq \
    --vad-stop-secs 0.2 --no-skip-existing
# then: .venv/bin/stt-benchmark wer --services nemotron_local --model silence0_warm200_seq
```
~3.3 h (sequential), same VAD + observer + WER path as the other services. NOT yet run as of
2026-05-20 — the row above is the concurrency-12 raw-harness approximation.
