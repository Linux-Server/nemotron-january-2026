# Semantic WER is dominated by the streaming/finalization strategy, not the model

**Status:** confirmed finding, 2026-05-17. Canonical reference for the
follow-on investigation.

## TL;DR

On the exact same checkpoint, audio, and semantic-WER judge, the **way we
stream audio and trigger finalization changes mean semantic WER by ~1.7
points — more than half the deployed error**:

| Path | Mean semantic WER | Method |
|---|---|---|
| Live benchmark (`nemotron_local`, VAD-driven streaming) | **3.08%** (n=1000) / 2.74% (n=200 slice) | real-time stream + a hard reset on every `VADUserStoppedSpeakingFrame` |
| Single-finalize **oracle** (fast harness) | **~1.3%** (n=200 slice; rc1ref 1.36%, warmup 1.26%) | whole clip → one hard reset; **no real-time pacing, no VAD, no Pipecat** (`/tmp/sweep.py`) |

Same model (`nvidia/nemotron-speech-streaming-en-0.6b`, March-2026
checkpoint), same clips/GT/judge. The only difference is
**segmentation/finalization behavior**. Model/decoder knobs do **not** help
(right-context flat rc1/6/13 = 1.36/1.60/1.45%; beam non-viable on Blackwell).
Estimated "addressable" WER after removing format/function-word/boundary
scoring noise ≈ **1.96%**.

> **Caveat (do not over-claim):** the ~1.3% is an **oracle upper bound**, not a
> demonstrated deployment number. The harness blasts the whole clip with no
> real-time pacing and no VAD, then one reset; its latency numbers are invalid
> (post-blast reset latency ≈ 628–639 ms median, not the live 211 ms). It
> proves *finalization strategy is the dominant lever* and sets the **target
> ceiling** — it does **not** prove we can reach ~1.3% *at low latency in real
> time*. That must be demonstrated by the gating phase below.

**Goal of the follow-on work:** finalize *complete* utterances both **very
fast** and with **very high accuracy** — currently we get one or the other,
not both.

## The mechanism (grounded in source)

### Client: `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`

- `run_stt()` forwards every audio frame to the server over WebSocket as it
  arrives (base `STTService.process_audio_frame` → `run_stt`).
- `process_frame()` (`:159-171`): on **every** `VADUserStoppedSpeakingFrame`
  it calls `request_finalize()` and `_send_reset(finalize=True)` — i.e. a
  **hard reset per VAD stop**. It never sends a soft reset.
- `_handle_transcript()` (`:193-209`): `is_final` → `confirm_finalize()` +
  push `TranscriptionFrame` (the server-deduped delta); interim → ignored by
  the benchmark observer.
- One WebSocket connection per benchmark sample (fresh server session).

This is a port of `pipecat_bots/nvidia_stt.py`, which used a **soft** reset on
`VADUserStoppedSpeakingFrame` and a **hard** reset only on the turn-level
`UserStoppedSpeakingFrame`. The benchmark pipeline has no turn aggregator
(only Silero VAD frames), so the port collapsed both into a single hard reset
on every VAD stop. **That collapse is the root cause.**

### Benchmark pipeline: `stt-benchmark/src/stt_benchmark/pipeline/`

- `Pipeline([SyntheticInputTransport, VADProcessor(SileroVAD, stop_secs=0.2), nemotron_local])`.
- `VADProcessor` emits only `VADUserStartedSpeakingFrame` /
  `VADUserStoppedSpeakingFrame`. Silero fires **multiple** start/stop pairs
  within one utterance whenever an internal pause exceeds `stop_secs` (0.2 s).
  Observed directly (multiple "User stopped speaking" within single clips);
  the smart-turn corpus is full of mid-utterance pauses by design.
- `transcription_collector` **concatenates** every final `TranscriptionFrame`
  delta with spaces.

### Server: `src/nemotron_speech/server.py` (cache-aware FastConformer-RNNT)

- `_handle_audio()` (`:368-398`): accumulates audio; runs `_process_chunk()`
  (`keep_all_outputs=False`) per `shift_frames`; emits interim
  `{is_final:false}`.
- `_reset_session(finalize)` (`:485-607`):
  - **soft** (`finalize=False`, `:518-535`): returns `current_text` as
    `{is_final:true, finalize:false}` immediately. No inference, **no state
    reset**, decoder/encoder/audio intact. Fast, lossless. **The client never
    uses this.**
  - **hard** (`finalize=True`, `:537-607`): pads audio with
    `final_padding_frames=(right_context+1)*shift_frames` of silence
    (`:541-546`), runs `_process_final_chunk()` with `keep_all_outputs=True`,
    sends the deduped delta, then **fully resets the session**
    (`_init_session` → fresh encoder cache, decoder reset, audio buffer
    cleared, `last_emitted_text=""`) (`:591-603`).

### Why the gap exists

In the **live** path, a single spoken utterance that contains any pause
> 0.2 s is split by Silero into N segments. Each segment boundary triggers a
**hard reset**, which:

1. Pads + force-decodes that segment in isolation (`keep_all_outputs=True`).
2. **Throws away encoder cache and decoder state** and re-initializes the
   session (`_init_session`).

So every segment after the first starts the cache-aware encoder **cold**.
This reproduces, once per internal pause, the cold-start onset garbling we
separately measured (≈25% of onset errors are streaming-cold-cache; client
preroll of 100–250 ms recovers ~65% of those). The benchmark then
concatenates the N independently-finalized, individually-degraded deltas →
boundary word loss/duplication + lost cross-segment context + repeated
cold-start onset errors → ~3.08%.

In the **single-finalize** path there are no mid-utterance resets: the whole
utterance streams through one warm session, then one clean
padded+`keep_all_outputs` finalize → ~1.3% on the same audio.

Corroborating evidence from this investigation:

- Error mass (1000-run): 71% substitutions, 18% insertions, 11% deletions —
  **not** a tail-loss problem (mean 0.21 trailing words lost; final-word
  recall 80%). Errors cluster at utterance/segment **onsets** and
  function-word/domain terms.
- Onset study: 75% of onset errors are corpus/GT artifacts (clips cut
  mid-utterance, median 0 ms lead-in); ~25% are streaming-cold-cache, fixable.
- Server-side warm-up as currently implemented (`NEMOTRON_ONSET_WARMUP_MS`,
  prepends zeros to the session buffer) is **ineffective** (onset fixable
  8/26 vs client-preroll 17/26): the bytes are in `accumulated_audio` but the
  session's `cache_last_channel/time/len` stay at `get_initial_cache_state()`
  (zeros) and `emitted_frames=0`, so the first real audio is still decoded via
  the cold first-chunk path. The correct fix is the Phase 3 recipe below.

## Hard constraints (locked)

- The **upstream benchmark framework is unchangeable**: pipeline is fixed at
  `[transport, VADProcessor(SileroVAD), stt_service]`. It uses **Silero VAD
  only — no Smart Turn / turn analyzer**; it emits only
  `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame` (no turn-level
  `UserStoppedSpeakingFrame`). The transcription collector **space-concatenates
  every final `TranscriptionFrame`** and ignores `InterimTranscriptionFrame`.
- We **cannot add Smart Turn** (or any new endpoint model) to our service.
- Changeable surface = **only Nemotron code**: `NemotronLocalSTTService` + its
  factory, the server `src/nemotron_speech/server.py`, and benchmark **CLI
  flags** (e.g. `--vad-stop-secs`). We must emit exactly the frames the
  framework expects: ideally **one** `TranscriptionFrame` per complete
  utterance.
- Design tension: fast finalize wants an early end-of-turn signal; high
  accuracy wants the whole utterance decoded in one warm pass. The missing
  capability — distinguishing *within-utterance pause* from *end of utterance*
  without cold-restarting the model — must be solved entirely inside our
  server + service.

## Recommended design: continuous context + disposable fork (speculative finalize)

The encoder-state problem **is** the cold restart (`_init_session` on every
hard reset). Eliminate it by never cold-restarting the live context:

The server keeps **one continuous, never-finalized streaming context per
connection**, fed the real audio in order straight through within-utterance
pauses (silence included) — the single-continuous-stream case the oracle
measured at **~1.3%**. It also keeps an explicit **`committed_text`** (the
text actually emitted to the client so far) that survives across speculative
flushes and is reset only at the true utterance boundary — *not* the existing
`last_emitted_text`/`current_text`, which `_init_session` clears (`:300-303`)
and which only deduped within a pre-reset session.

**All state lives server-side** (single source of truth, co-located with the
model state — more robust than a client-side hold; no client/server race;
reusable by any client). The client is a *thin protocol translator only*:

- Client responsibilities (no buffering, no timers, no decisions): forward
  audio; translate the only frames it can see into control messages — because
  the server sees the WebSocket protocol, not pipecat frames —
  `VADUserStoppedSpeakingFrame` → `{"type":"vad_stop"}`,
  `VADUserStartedSpeakingFrame` → `{"type":"vad_start"}`; relay whatever the
  server emits (server interim → `InterimTranscriptionFrame`; the server's
  single final → exactly one `TranscriptionFrame` + `confirm_finalize()`).
- Server owns the speculative finalize: on `vad_stop` → **fork** a deep copy
  of the session state, run the pad + `keep_all_outputs` flush **on the fork**
  for the candidate text, and start a **server-side** debounce timer
  (`NEMOTRON_FINALIZE_SILENCE_MS`, ~250 ms).
  - `vad_start` within the window → discard the fork; the continuous context
    was never touched and continues warm.
  - Timer expires (true end of turn) → emit the single final delta exactly
    once; reset for the next utterance.
- This is the "A/B" dual-context idea with A a *disposable fork* of the
  never-finalized context B (no replay / double-feed). "Speculative finalize"
  = compute-but-don't-emit until the server confirms end-of-turn.
- Composes with the benchmark transport: `SyntheticInputTransport` keeps
  sending silence until it sees a final `TranscriptionFrame`, then
  `post_transcription_delay`, with `max_silence_timeout`=10 s. During the
  server's debounce those silence frames simply continue feeding the
  continuous context (the natural tail); the server emits its one final well
  inside 10 s, so there is no deadlock/timeout. Keep an `_audio_send_lock` so
  a control message cannot interleave a binary audio send.

### Deep-stack risks that decide feasibility

1. **State aliasing/mutation in NeMo (the crux — confirmed real).** NeMo's
   batched greedy path mutates partial hypotheses in place
   (`rnnt_greedy_decoding.py:825-831`) and `Hypothesis.merge_()` mutates
   sequence/dec_state/timestamps/text (`rnnt_utils.py:153-181`). A shallow copy
   *will* corrupt the parent. Required clone recipe: `.detach().clone()` every
   cache tensor; recursively clone every tensor inside
   `previous_hypotheses.dec_state` (deepcopy the hypothesis objects); never copy
   websocket/model objects. Run the aliasing probe **per decoding strategy**
   (greedy packed-hyps is less alias-prone but config-dependent).
2. **Inference is not parallel-safe — fork flush must serialize.** NeMo
   mutates *model-global* `streaming_cfg.drop_extra_pre_encoded`
   (`streaming.py:53-74`) and toggles decoder/joint train/eval state
   (`rnnt_greedy_decoding.py:753-775`). The server has one `inference_lock`
   (`server.py:96`). The fork flush therefore **cannot run concurrently with
   the parent stream on the same model object** — it must hold `inference_lock`
   (and that latency goes on the critical path, counted honestly) or use a
   separate model instance/context (GPU cost).
3. **Continuous context is O(n²) without a ring buffer.** Today every chunk
   re-preprocesses *all* accumulated audio (`server.py:403-415`, `:615-623`);
   the hard reset is the only thing bounding it (`:591-603`). Remove the reset
   and a long utterance → quadratic preprocess + a fork that copies a growing
   buffer. The continuous design **requires** an incremental/ring-buffer
   preprocessor that retains only the needed raw/mel tail; clone only cache +
   minimal pending audio. (Prerequisite, not optional.)
4. **Emit-once discipline.** A `TranscriptionFrame` cannot be retracted and the
   collector concatenates finals — the speculative result is held **server-side**
   and emitted exactly once at confirmed end-of-turn; the client never buffers
   or decides. Also suppress the extra hard reset `stop()` sends
   (`nemotron_local_stt.py:79-83`) after a committed final, and ignore empty
   duplicate finals.
5. **Long-context is bounded.** Left context capped by
   `att_context_size[0]=70`; within a long pause the cache fills with silence
   (≫ cold-init, < speech context). Cross-*utterance* context is mostly moot
   here (one connection per sample, mostly single utterances) — the dominant,
   provable win is simply *never cold-resetting on within-utterance pauses*.

### Probe first — this gates everything

Before any architecture work, run an **aliasing probe**: fork the live session
mid-stream, run pad + `keep_all_outputs` on the fork, then assert the original
context's tensors/hypotheses are unchanged **and** its continued transcription
is bit-identical to the no-fork run. If NeMo mutates in place, determine the
required cloning depth before building.

### Phased plan

Each phase measured on the fixed 200-slice
(`uv run stt-benchmark run --services nemotron_local --model <tag>
--no-skip-existing` → `uv run stt-benchmark wer --services nemotron_local
--model <tag>`); report semantic WER + median/p95 latency **from the real
benchmark observer** (not harness reset latency) + hard-resets/sample.
Baseline rc1/greedy = **2.74% / TTFS 212 ms**.

**Measurement validity (fold these or the numbers mislead):**
- The 200-slice (`dataset_index` 0–199) is **biased low**: 2.74% vs 3.16% on
  the other 800; per-100 buckets range 2.26–4.18%. Use **paired** comparisons
  on identical sample IDs for deltas, **plus** one randomized/stratified
  200-slice with confidence intervals; never extrapolate slice absolutes to
  the corpus.
- `model_name` is only an experiment tag, not a checkpoint hash. **Record the
  HF revision / local `.nemo` path + model-config hash** in every run before
  treating any number as canonical.

- **Phase G — gating test (cheap, decides everything; do FIRST).** Real-time
  through the actual Pipecat benchmark transport, VAD observed for telemetry
  only, but the client issues **no mid-utterance hard reset** — exactly one
  hard reset at `EndFrame`/true sample end. If this real-time, single-reset
  run does **not** approach the oracle (rc1ref/warmup ~1.3%), the
  fork/continuous design is solving the wrong problem — stop and rethink. This
  also produces the first *real-time* (not harness) low-WER datapoint.
- **Phase 0 — telemetry + control:** in-service counters (VAD stops, soft/hard
  finalizes, segment count, real finalize latency). Run at **both
  `--right-context 0`** (~80 ms; finalize silence `(0+1)×16` = **160 ms**)
  **and `--right-context 1`** (~160 ms; finalize silence `(1+1)×16` =
  **320 ms**) — WER was flat across R (rc1/6/13 = 1.36/1.60/1.45%), so rc0
  should give the same accuracy while **halving the mandatory finalize
  padding**; Phase 0 must confirm WER parity on the slice and quantify that
  latency win. The `--vad-stop-secs` sweep {0.2,0.4,0.6,1.0} is a **control
  only** — it changes segmentation *and* endpoint delay together, so it does
  not isolate reset damage (Phase G does).
- **Phase 1 — client debounce:** collapse VAD stops into one finalize, hold
  window **empirically derived** from the measured intra-utterance stop→next-
  start gap distribution (sweep 250/500/1000/1500 ms; note Silero adds ~200 ms
  start + ~200 ms stop, so a 250 ms hold ≈ commits after ~450 ms silence and
  still splits longer pauses); + the `_handle_transcript` fix + `_audio_send_lock`.
- **Phase 2 — server continuous context + disposable fork** (gated by the
  aliasing probe **and** Phase G passing): the core fix; needs the ring-buffer
  preprocessor (risk 3) and serialized fork flush (risk 2).
- **Phase 3 — proper per-session warm-up** (residual onset; small aggregate
  WER, UX polish). Warm the encoder *cache state*, not the audio buffer. In
  `_init_session`, after `get_initial_cache_state()`:
  1. Synthesize ~150 ms silence → `model.preprocessor(...)` → mel.
  2. Call `model.conformer_stream_step(processed_signal=warm_mel, …,
     keep_all_outputs=False, return_transcription=False)` (the same call
     `_process_chunk`/the startup `_warmup()` use) and store the **returned**
     `cache_last_channel/time/len`, `previous_hypotheses`, `pred_out_stream`
     back into the session.
  3. **Consistency requirement (the bug the naive prepend missed):** the
     silence must also remain in `accumulated_audio` *and* `emitted_frames`
     advanced past it, so the first real chunk enters the warm
     `emitted_frames != 0` branch (`pre_encode_cache` lookback + `drop_extra`),
     not the cold first-chunk branch.
  4. Discard the (blank) warm-up text; do **not** seed
     `current_text`/`last_emitted_text`.
  This is the server-side equivalent of the client preroll measured at ~65%
  recovery (and what startup `_warmup()` already does globally). Validate on
  the onset-fixable set (target ≫ 8/26, ≈ client-preroll 17/26) + the slice;
  sweep `warmup_ms ∈ {100,150,200,250}` (500 ms **regressed** — do not
  over-pad). Becomes trivial once the risk-3 ring-buffer preprocessor lands
  (seed the cache; no `emitted_frames`/`accumulated_audio` bookkeeping).
- **Target:** ~1.3–1.96% semantic WER at finalize latency near 212 ms —
  *unproven until Phase G; the ~1.3% is a ceiling, not a guarantee.*

Smart Turn / a real end-of-turn detector remains the **production**
architecture + developer best-practice guidance, explicitly **out of scope for
the benchmark** (cannot change the framework or add the dependency) — recorded
separately, not measured here.

## Reference data / artifacts

- DB: `stt-benchmark/stt_benchmark_data/results.db` — `nemotron_local` results
  by `model_name`: `''` (1000-run, 3.08%), `rc1ref` 1.36%, `rc6` 1.60%,
  `rc13` 1.45%, `warmup` 1.26% (all on the fixed 200 `dataset_index`-ordered
  slice); `ground_truth` (1000, shared Gemini run); `wer_metrics`.
- Non-perfect review: `stt_benchmark_data/nemotron_local_semantic_wer_review.md`.
- Tail tool: `stt-benchmark/scripts/tail_accuracy.py`.
- Server toggles added (env-gated, defaults preserve behavior):
  `NEMOTRON_DECODING=beam` (non-viable on Blackwell), `NEMOTRON_ONSET_WARMUP_MS`
  (current impl ineffective — see above).

---

# Measured Outcome (2026-05-19) — authoritative addendum

The launching analysis above (2026-05-17) framed the investigation; this
addendum records what was actually built and measured. **The thesis is
confirmed at production latency.** Everything below is from full-1000
benchmark runs against the same HF model revision throughout.

## TL;DR — what we delivered

- **Recommended shippable config: `warm200`** — server-side continuous context
  + speculative disposable fork (Step 7d) + 150 ms in-budget server debounce +
  always-append-only emission + per-session warm-up at 200 ms.
- **Accuracy:** paired Δ vs `''` baseline **slice-A −1.29 pp CI[−1.84, −0.74]
  / slice-B −1.39 pp CI[−1.96, −0.87]** (both CIs exclude 0). **Matches or
  exceeds the non-shippable analytical 2.5 s-debounce ceiling** (cc7c) on
  both slice means.
- **Latency:** finalize budget p95 **325.8 ms** under the locked formula
  `endpoint_wait(151) + rc1(160 modeled) + finalize_flush(13.4) + transport(0.2)
  < 400 ms`; observer-TTFS p95 366 ms (single-session/sequential-benchmark scope
  — R4 caveat applies for concurrent production load).
- **The accuracy/latency trade-off is broken**: oracle-region accuracy at
  in-budget latency, in one shippable config.

## Canonical WER ↔ latency table (full 1000, paired same-ID Δ vs `''`, bootstrap=2000)

Legend: ✓ = `< 400 ms` real-observer p95 (shippable); ✗ = analytical-only
(reference). All paired Δs and CIs are 95% bootstrap on paired same-IDs.

| tag | full | slice-A mean | slice-A Δ vs `''` (CI95) | slice-B mean | slice-B Δ vs `''` (CI95) | TTFS p95 | finalize p95 | in-budget | role |
|---|---|---|---|---|---|---|---|---|---|
| `''` (legacy live) | 3.08% | 2.74% | 0 (def.) | 3.01% | 0 (def.) | 220 ms | n/a | ✓ | the deployed default |
| `rc1_base` | 3.18% | 2.84% | +0.10 [−0.13, +0.32] | 3.16% | +0.16 [−0.05, +0.37] | 220 ms | n/a | ✓ | rc1 control; ≈ baseline |
| `vad020` | ~2.95% | 2.89% | ≈ baseline (CI incl. 0) | 3.33% | ≈ baseline | 220 ms | n/a | ✓ | vad-stop=0.2 s control; ≈ baseline |
| `ringbuf` (6c) | ~3.10% | 2.93% | +0.20 [−0.06, +0.45] | 3.19% | +0.18 [−0.07, +0.46] | 221 ms | n/a | ✓ | constant-plan O(1)/chunk ring; **WER-neutral vs `''`** (Step 6c gate); enables Step 7 |
| `phaseG_single` | 2.04% | — | (analytical) | — | — | 2718 ms | n/a (2500 ms hold) | ✗ | **Gate Gp PASS** — proved single-reset oracle at ~Phase-G is achievable; gated Steps 7+ |
| `cc7b` (7b) | 2.00% | 1.47% | −1.27 [−1.71, −0.86] | 1.73% | −1.28 [−1.86, −0.79] | 2716 ms | n/a (2500 ms hold) | ✗ | analytical: continuous context + in-place finalize at the 2.5 s ceiling |
| `cc7c` (7c) | 2.02% | 1.47% | −1.27 [−1.73, −0.85] | 1.79% | −1.22 [−1.80, −0.77] | 2716 ms | n/a (2500 ms hold) | ✗ | analytical: **disposable-fork** finalize WER-neutral vs in-place (cc7b) → 0 fork-alias FAILED / 1999 PASSED |
| `fork` (7d) | 2.10% | 1.70% | −1.04 [−1.47, −0.64] | 1.97% | −1.04 [−1.50, −0.63] | 366 ms | **325.8** | ✓ | **first shippable** continuous-context config; 0 fork-alias FAILED / 3460 PASSED |
| `warm150` | 2.11% | 1.62% | −1.11 [−1.56, −0.67] | 1.71% | −1.30 [−1.82, −0.83] | 366 ms | 325.7 | ✓ | warm-up at 150 ms; **strict paired vs `fork`** slice-A −0.07 [−0.34, +0.20] (noise) / **slice-B −0.25 [−0.48, −0.05]** (CI excludes 0) |
| **`warm200`** ⭐ | **2.07%** | **1.45%** | **−1.29 [−1.84, −0.74]** | **1.62%** | **−1.39 [−1.96, −0.87]** | **366 ms** | **325.8** | ✓ | **RECOMMENDED**. Strict paired vs `fork`: slice-A −0.25 [−0.68, +0.13] / **slice-B −0.34 [−0.69, −0.07]** (sig). Trajectory `fork → warm150 → warm200` monotonic on slice-B. Matches/exceeds cc7c ceiling at in-budget latency. |
| *(rc0)* | — | — | — | — | — | — | — | — | **UNSUPPORTED** — NeMo `rel_shift` deterministically crashes on `att_context=[70,0]` (Step 4 finding; ~7.5 h burned, 0 transcriptions). Documented; do not run. |

Notes on the table:
- `phaseG_single`/`cc7b`/`cc7c` are analytical-only because their 2.5 s
  server debounce blows the < 400 ms wall-clock budget by ~7×. They are the
  *accuracy reference* the production-shippable configs aimed at.
- "in-budget" column = real-observer TTFS p95 ≤ 400 ms. The finalize-budget
  column applies only to the 7d-family configs (which emit the JSONL the
  budget reader consumes); `''`/`rc1_base`/`vad020`/`ringbuf` finalize via
  the legacy hard-reset path with no JSONL instrumentation but their TTFS is
  in-budget by ~180 ms margin.
- vad020's bootstrap CI vs `''` was not re-computed in the final canonical
  bootstrap run; means cited from the Step-4 sweep.
- `phaseG_single` slice means not re-computed in the final canonical bootstrap.

## Production recommendation (concise)

| ranked | config | when to use |
|---|---|---|
| 1 (default) | **`warm200`** — `NEMOTRON_CONTINUOUS=1`, `NEMOTRON_FINALIZE_SILENCE_MS=150`, `NEMOTRON_WARMUP_MS=200`, `NEMOTRON_FORK_ASSERT=1` (optional in prod; on for the gate-validation runs) | Default shippable: best measured accuracy at < 400 ms; matches the analytical ceiling. |
| 2 | `fork` (7d) — same as above without `NEMOTRON_WARMUP_MS` | If you cannot afford the ~13 ms warm-up GPU at every `_init_session` (rare; the savings are dominated by the per-flush ~13 ms anyway). Same shippable budget; slightly worse accuracy on the slice populations. |
| ref. | `''` (legacy live) | The deployed default before this work. Worst measured accuracy; in-budget. |
| do not ship | `cc7b`/`cc7c`/`phaseG_single` | 2.5 s server debounce → > 2.7 s observer TTFS. Reference-only. |
| unsupported | `rc0` | NeMo crash on `att_context=[70,0]`. Do not run. |

## Findings produced during the work (folded from the PLAN)

### rc0 is unsupported (Step 4)
`att_context_size=[70,0]` deterministically crashes upstream NeMo
`multi_head_attention.py:267 rel_shift` (0-element reshape). Across an
unattended sweep ~7.5 h of compute produced **35 M crashes and 0
transcriptions** before catch. The server swallowed the crash silently; the
finding is now documented and the orchestrator's Monitor patterns include
early-fail signatures to catch this class. **rc1 (~160 ms right-context) is
the only viable production context.** rc6/rc13 exceed the < 400 ms budget by
construction.

### cuFFT plan-size non-determinism (Step 6b)
Bit-exact equivalence between an incremental ring-buffer preprocessor and a
growing-full-reprocess preprocessor is **not achievable** on CUDA: cuFFT STFT
plans are batch-/length-sensitive (3/10 fixtures diverged ≤ 4 ULP after a
careful constant-plan design). The Step-6 byte-equivalence rule was relaxed
to **constant-plan ring + WER-within-CI validation + ≤ 1e-5 mel closeness on
an expanded fixture set**. Step 6c shipped on those criteria and is
WER-neutral on the full 1000.

### Production latency budget + taxonomy (Rule, locked 2026-05-18)
Hard target: end-to-end added latency **< 400 ms** in production. Budget
formula (evaluated at p95, not median):
`endpoint_wait + encoder_right_context(rc1 ≈ 160 ms) +
measured_finalize_flush_wallclock + transport < 400 ms`. The finalize *silence
padding's duration* is faster-than-wallclock (synthetic zeros + one
`conformer_stream_step`, ~13 ms compute); the *measured flush wall-clock* IS
budgeted. Required finalize-instrumentation JSONL (Nemotron-owned) extends
the Step-1 telemetry with `vad_stop` / `debounce_expiry` /
`fork_flush_start` / `fork_flush_done` / `final_sent` / `final_received` /
`inference_lock_acquire_wait` timestamps so the < 400 ms claim is measured,
not modeled. R4 caveat: the measured term is single-session/sequential
benchmark observer latency; under concurrent production load
`inference_lock` contention adds to it (separate measurement, out of scope
here).

**R4 caveat CLOSED (2026-05-20)** — measured directly. A realtime concurrent
sweep (`proj-2026-05-19-eou-endpointing/concurrency_test.py`, commit `737a1fd`)
opened N ∈ {1,4,8,12,16,20,24} concurrent realtime sessions against the
production config (`NEMOTRON_FINALIZE_SILENCE_MS=0` + `NEMOTRON_WARMUP_MS=200`
+ continuous). Findings: (a) **byte-exact correctness at every N** — 24/24
transcripts identical to the single-session baseline at N=24, max edit-distance
0, `FORK_ASSERT=1` clean throughout (no cross-session state leakage, no races;
the serialized-inference + per-session-cache design degrades gracefully —
latency only, never correctness). (b) **Server-side finalize latency
(`vad_stop→final`, the lock-contended term) is flat ~15 ms p95 through N=12**,
rises to 33 ms at N=16, then collapses (664 ms at N=20, 2.1 s at N=24) as the
single `inference_lock` saturates. End-to-end TTFS = ~200 ms Silero + this, so
**~12 concurrent live sessions per RTX 5090 are comfortably in the 400 ms
budget, ~16 at the edge.** Scales horizontally (capacity = per-GPU-ceiling ×
instances). No request batching exists (batch_size=1 by design); vLLM-style
continuous batching on the cache-aware streaming path is the unexplored lever
if the ~16/GPU ceiling is insufficient.

### Step 4 / Step 5 scope reductions
- **Step 4 (rc-and-VAD sweep) triaged**: rc0 dropped (crashes); vad-stop>0.2 s
  out-of-budget by construction; vad020 confirmed ≈ baseline (no WER lever);
  vad040/060/100 not run.
- **Step 5 (client-debounce sweep) GUTTED**: only the `_handle_transcript`
  finalize/interim plumbing fix was retained (reused by Step 7a); no `dbnc*`
  measurements (debounce-hold sweep out of < 400 ms budget by construction).

### Rigor wins (review-discipline-saved runs)
Four pre-run defects were caught by rigorous diff-review + dual adversarial
review on Step 7d alone — each would have produced a corrupt or misleading
multi-hour run:
1. **Step 7d cold-reset BLOCKER** (Claude review): `_init_session` was
   called on every 150 ms debounce expiry → mid-sample cold-restart
   segmentation → would have regressed to the ~3 % WER the project exists
   to eliminate.
2. **R1 multi-emit BLOCKER** (Codex + Claude dual review, converged): with
   short-debounce speculative emits feeding an append-only collector, the
   server's "full re-emit on ASR self-correction" path (~40 % of cumulative
   transitions are non-prefix) → duplicated/garbled hypothesis on the
   650/1000 multi-segment samples. Resolved by the `always-append-only`
   emission contract (frozen-by-count word suffix; corrections to already-
   emitted words dropped = accepted measured cost).
3. **FORK_ASSERT MAJOR gap**: the under-load aliasing assertion covered
   cache + `previous_hypotheses` but **not** `pred_out_stream` (which the
   fork *clones* but the assertion didn't *verify*). Added.
4. **R4 scope MAJOR**: the latency claim was implicit-concurrent until the
   single-session/sequential scope label was added to the budget reader.

Earlier in the project: Step 6b's first impl was fixture-tuned magic
windows (caught by Claude diff inspection); Step 7b's first impl had two
concurrency defects (caught by dual review pre-run).

# Corrections to the launching doc

The launching analysis above contains four claims that the implementation +
review process showed to be inaccurate. The corrections, with code references:

- **(B1) `_init_session` clears `current_text` but NOT `last_emitted_text`.**
  The original doc (≈ line 82) claims `_init_session` clears
  `last_emitted_text=""`. In fact, `_init_session` clears `current_text` and
  related ASR state; `last_emitted_text=""` is cleared by the **hard-reset
  path** (`server.py:~1857`, the legacy `_reset_session` epilogue), *not*
  `_init_session`. This matters for the 7b emit-once analysis.
- **(B5) `previous_hypotheses` is a *list of `Hypothesis`*, not an object
  with `.dec_state`.** The original doc (line 193) refers to
  "`previous_hypotheses.dec_state`" as if `previous_hypotheses` were a single
  object. It is `List[Hypothesis]`; each `Hypothesis` carries its own
  `.dec_state`. The Step-3 Gate-Ga deep-clone recipe correctly iterates the
  list and clones every `hyp.dec_state` tensor (verified by the probe).
- **(B12) `return_transcription=False` is invalid for RNNT (Transducer)
  models.** The launching warm-up recipe (line ~280) suggests calling
  `conformer_stream_step(..., return_transcription=False)`. NeMo logs that
  transcription cannot be disabled for Transducer models — the RNNT path
  still decodes. The correct Step-8 recipe (now implemented) runs
  `return_transcription=True` and **explicitly discards the returned
  hypotheses/text**, keeping only the warmed
  `cache_last_*`/`previous_hypotheses`/`pred_out_stream`.
- **(B-Ga) Deep-stack risk #1 cites `hyp.merge_` at
  `rnnt_greedy_decoding.py:825-831`, which lives in `GreedyBatchedRNNTInfer`
  (`strategy=greedy_batch` / `loop_labels=True`).** The configured server path
  (`server.py:446` sets `strategy: greedy` → NeMo instantiates
  `GreedyRNNTInfer`; `loop_labels` is only used for `greedy_batch`) does
  **not** fire that in-place merge. Step-3 probe + independent re-run
  confirmed: the server-path shallow+deep clones are clean; only the batched
  path's shallow clone corrupts. The deep-clone recipe is retained as
  defense-in-depth, and the serialization-under-`inference_lock` requirement
  is independent and still mandatory.

# Reproducibility / artifacts

- **Model:** `nvidia/nemotron-speech-streaming-en-0.6b`,
  HF revision **`ef3bf40c90df5cd2de55cc07e06681e03d8e6ee4`** (single resolved
  revision throughout this work; recorded per-tag in
  `stt-benchmark/stt_benchmark_data/run_metadata/<tag>.json`).
- **Benchmark:** `uv run stt-benchmark run --services nemotron_local --model <tag> --no-skip-existing`
  then `uv run stt-benchmark wer --services nemotron_local --model <tag>`.
- **Scoring (paired same-ID Δ + bootstrap CIs):**
  `uv run python stt-benchmark/scripts/measure.py score --tags '' rc1_base ringbuf cc7b cc7c fork warm150 warm200 --baseline-tag '' --bootstrap 2000`.
- **Finalize-budget (the < 400 ms gate):**
  `uv run python stt-benchmark/scripts/measure.py finalize-budget --tag <tag>`.
- **Per-step commits** (`proj-2026-05-17-1708/PLAN.md` Progress table):
  Phase G `c2fbe18`, Step 3 aliasing probe `eab05dd`, Step 4 `d23b067`,
  Step 5 plumbing `382c0cf` (stt-benchmark), Step 6c ringbuf `f3f37a1`,
  Step 7a `dfa4f13` + stt`32d4874`, Step 7b `b757159`, Step 7c `7cbdf09`,
  Step 7d `6181310` + stt-benchmark `e6cf064`, Step 8 `0462679`.
- **Archived Codex job logs + measured-run logs:**
  `proj-2026-05-17-1708/codex-jobs/step-*.log` (one per Codex delegation +
  the full-1000 run.out/score.log for the in-flight gate runs).
- **Next-stage feasibility plan** (post-this-project, draft):
  `proj-2026-05-19-eou-endpointing/PLAN.md` — ASR-internal endpointing toward
  a lower-latency finalize at non-inferior WER; twice dual-reviewed; carries
  a "final pre-flight re-review before `/implement`" gate.
- **TTFS explainer (companion doc):** `docs/ttfs-latency-explainer.html` —
  the finalization-latency bounds, contributing factors, and the
  greedy-append-only / rc1-stability nuance.
