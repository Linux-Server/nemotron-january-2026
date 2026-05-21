# Step 6 — Multilingual full-1000 conc-12 + semantic WER (en-US)

Apples-to-apples with the English `silence0_warm200_c12` ship-gate, using the **same**
raw-WebSocket harness (`proj-2026-05-19-eou-endpointing/run_full1000_conc12.py`), the **same**
1000 samples / ground truth (`results.db`), the **same** concurrency (12) and finalize config
(`NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200`).

The only differences are the **model** (multilingual checkpoint vs English) and the **runtime**
(EA NeMo `.venv-ea` vs omni venv), plus the **rc** (multilingual rc3 `[56,3]` vs English rc1).
The benchmark language is forced to **en-US** via the `?language=en-US` handshake (per the user:
"for the benchmark run we do now we should specifically set english as the language").

## Configuration (this run)

| Knob | English ship-gate | Multilingual (this run) |
|------|-------------------|--------------------------|
| Checkpoint | NVIDIA-Nemotron Streaming-EN 0.6b | NVIDIA-Nemotron-3.5-ASR-Streaming-**Multilingual**-0.6b |
| Runtime | omni venv (NeMo 2.8.0rc0) | EA venv (`kingformatty/NeMo @ ...EA`, torch 2.12+cu130) |
| att_context_size | `[70,1]` (rc1) | `[56,3]` (rc3) |
| Language | n/a (no prompt) | `en-US` prompt (index 0), set per-session under lock |
| Finalize | silence0 + warm200 | silence0 + warm200 (identical) |
| Concurrency | 12 | 12 |
| Final synthetic pad | `(1+1)*shift = 320ms` | `(3+1)*shift = 640ms` (faster-than-wallclock) |
| model_name tag | `silence0_warm200_c12` | `ml_silence0_warm200_c12` |

## Results (run 2026-05-20; judge 997/1000, ~7 min)

| Metric | English `silence0_warm200_c12` | Multilingual `ml_silence0_warm200_c12` (en-US) | Δ |
|--------|-------------------------------|------------------------------------------------|---|
| Transcripts returned | 100.0% (1000/1000, 0 err) | 99.7% (997 non-empty, 0 err, **3 empty**) | −3 |
| Perfect (0% WER) | 76.1% | **57.7%** | −18.4 pp |
| ≤5% WER | 85.4% | 71.1% | −14.3 pp |
| >20% WER | 0.9% (9) | **4.3% (43)** | +34 clips |
| >50% WER (catastrophic) | 0.0% (0) | **1.0% (10)** | +10 clips |
| WER Mean | 1.94% | **4.72%** (5.0% if 3 empties count as deletions) | ~2.4× |
| Pooled WER | 1.95% | **4.84%** | ~2.5× |
| WER Median | 0.00% | 0.00% | 0 |
| TTFS Median | 220 ms | **219.3 ms** | −1 ms |
| TTFS P95 | 247 ms | **245.0 ms** | −2 ms |
| TTFS P99 | 263 ms | **264.7 ms** | +2 ms |
| Server finalize p50/p95 | ~15 / ~33 ms | 19.1 / 44.7 ms | small |
| timed_out/empty | 0 | 3 (recorded as empty, not hung) | +3 |

**Headline:** **latency is identical** (TTFS p95 245 vs 247 ms — rc3 is not a regression, as
predicted); **English WER is ~2.4× worse** (4.72% vs 1.94%). The median is 0% for *both* — the gap
is **entirely in the error tail**, not a uniform shift: multilingual produces 43 clips >20% (vs 9),
10 catastrophic clips >50% (vs **zero** for English), and 3 empties (vs zero). This is the expected
cost of a 0.6b model spreading capacity across 128 languages vs a 0.6b English-specialist.

## TTFS comparability note (rc3 is NOT a regression — CONFIRMED)

rc3's synthetic final-pad is `(R+1)*shift = (3+1)*160 = 640 ms` of silence fed to the decoder as
**one `conformer_stream_step` call**, faster than wall-clock (same mechanism as `silence_0` —
[[silence0-warm200-shippable]]). Measured finalize p50 19 ms / p95 45 ms and TTFS p95 245 ms
**confirm** this: the 640 ms pad costs only GPU time (~10–25 ms more than English's 320 ms flush),
not a 480 ms real-audio wait.

## Empties + error tail: NOT rc-related (diagnosed)

The user asked whether the empties were an rc0-vs-rc3 artifact. **They are not.** Diagnostics
(rc0 server `[56,0]` loads cleanly — multilingual rc0 is supported, unlike English `[70,0]`):

**The 3 empties — empty at BOTH rc0 and rc3:**
| sample | dur | RMS | rc3 | rc0 | English | root cause |
|--------|-----|-----|-----|-----|---------|------------|
| "Schedule my rent payment for the 28th…" | 4.8 s | 743 (quiet) | empty | empty | got it | **low energy** — at ×2 gain it transcribes ("…my rent payment for the twenty eighth of this month"); ×1 empty |
| "Send it." | 2.3 s | 807 | empty | empty | "Send it" | **very short** — empty at ×1/×2/×4/×8 gain (not energy) |
| "Mhm." | 1.0 s | 4982 (loud) | empty | empty | "Mmhmm." | **very short backchannel** — already full-scale, empty at all gains |

**The 10 catastrophic (>50%) clips — garbled at BOTH rc0 and rc3:** re-streaming at rc0 sometimes
helped ("I want to learn some busy phrases in Italia" vs rc3 "How else learns on busy freezes"),
sometimes hurt (one went empty), never a systematic fix. Several have degenerate/truncated *ground
truth* too ("At a told", "the um aren't the um", "I need to schedule a service appoint-").

**Conclusion:** rc0 ≈ rc3 on both the empties and the bad tail. The WER gap is the
multilingual-vs-specialist **model-capacity tradeoff**, plus a higher sensitivity to quiet/short
speech than the English checkpoint — not a config or right-context bug. rc3 stays the choice
(low-latency, and rc0 offers no accuracy win here).

## Correctness gate (from Step 5 + this run)

- No `<xx-XX>` language-tag leakage (regex-stripped before state; smoke + Step-5 subset clean).
- No looping / repetition / garbage.
- WER in a plausible band vs the English checkpoint on the same samples.

## CRITICAL caveat — carried from English (raw-harness ≠ Pipecat pipeline)

Identical to `proj-2026-05-19-eou-endpointing/readme-row-silence0-warm200.md`: this row comes from
the custom raw-WS harness, **not** the Pipecat benchmark pipeline. It bypasses Silero VAD (one
`vad_stop` at end vs Silero's multi-segment), `nemotron_local_stt.py`, and the observer. The
append-only delta design *should* make multi-segment == one-segment, so WER is a **close estimate,
not the exact pipeline number**. Both English and multilingual rows share this caveat, so the
**English-vs-multilingual comparison is apples-to-apples** (same harness, same bypass).

## Verdict

The multilingual checkpoint is **integration-correct and latency-competitive**, but **not an
English-accuracy substitute** for the dedicated English checkpoint:

- ✅ **Integration works end-to-end**: process-per-model + query-param handshake + per-session
  prompt (set under lock) + lang-tag stripping all functioned across 1000 concurrent-12 sessions;
  zero `<xx-XX>` tag leakage, zero errors, zero hangs (harness empty-final fix worked).
- ✅ **Latency parity**: TTFS p95 245 ms vs English 247 ms. rc3's 640 ms synthetic pad is
  faster-than-wallclock — confirmed, not a regression.
- ⚠️ **English WER ~2.4× worse** (4.72% vs 1.94%), all in the tail: 10 catastrophic clips and 3
  empties that English handled cleanly. Expected for a 128-language 0.6b vs an English-specialist
  0.6b.
- ✅ **rc ruled out** as the cause of empties/tail (rc0 ≈ rc3).

**Recommendation:** keep the **English checkpoint** as the default for English-only voice agents
(better accuracy, same latency). Use the **multilingual checkpoint** when multi-language coverage
is required — the latency is production-ready and the integration is sound. For its quiet/short-clip
empties in a voice-agent context, the definitive end-of-turn trigger (Smart Turn) plus input gain
normalization would mitigate the low-energy case; the very-short-utterance empties are inherent to
the checkpoint.
