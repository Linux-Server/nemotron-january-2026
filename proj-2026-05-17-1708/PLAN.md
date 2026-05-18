# Plan: Recover semantic-WER lost to the streaming/finalization strategy

Project directory: `./proj-2026-05-17-1708`

## Context
Canonical finding (`docs/semantic-wer-finalization-finding.md`): on the same
checkpoint/audio/judge, the live VAD-driven `nemotron_local` path scores ~3.08%
semantic WER while a single-finalize oracle scores ~1.3% — >½ the deployed
error is the *finalization strategy* (a hard reset on every
`VADUserStoppedSpeakingFrame` cold-restarts the server session), not the model.
This plan executes the doc's phased program to recover that gap **inside only
Nemotron code**, fastest- and cheapest-to-disprove steps first, with hard gates
so we do not build the expensive server redesign unless cheap evidence says it
will work.

## Reference implementations
- **Client precedent — `pipecat_bots/nvidia_stt.py`** (pipecat 0.0.98): the
  original soft-reset-on-`VADUserStoppedSpeakingFrame` / hard-reset-on-turn-end
  split, the `_audio_send_lock` (`:71-82`, `:225-238`), and the pending-frame
  hold. Our `nemotron_local_stt.py` is a 1.2.x port that *collapsed* this into
  hard-reset-per-VAD-stop — the root cause. Re-port the split, adapted to
  pipecat 1.2.x and to server-side state ownership.
- **Correct warm-up precedent — `src/nemotron_speech/server.py:_warmup()`
  (`:215`)**: already runs silence *through* `conformer_stream_step` to claim
  GPU memory. Phase 3's per-session warm-up does the same into session state.
- **Cache-aware streaming preprocessor**: NeMo ships streaming-preprocessor /
  `cache_aware_streaming` helpers (installed nemo package under
  `nemo/collections/asr/parts/utils/streaming_utils.py`); the ring-buffer step
  aligns to those rather than the current re-mel-everything wrapper. Divergence:
  we keep the existing WS protocol and `conformer_stream_step` call shape;
  only stage-1 feature extraction becomes incremental.
- No external (vLLM/SGLang/TRT-LLM) references apply; this is ASR streaming
  orchestration specific to NeMo cache-aware FastConformer-RNNT.

## Current state
Verified file:line (read 2026-05-17):

Server `src/nemotron_speech/server.py`:
- `DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"` (`:27`);
  `RIGHT_CONTEXT_OPTIONS` (`:30`); `set_default_att_context_size([70, R])`
  (`:139`); `final_padding_frames=(right_context+1)*shift_frames` (`:195`).
- `self.inference_lock = asyncio.Lock()` (`:97`); used at `:319`, `:383`,
  `:552`.
- `NEMOTRON_DECODING` env branch (`:144`); `NEMOTRON_ONSET_WARMUP_MS` env
  (`:289`, broken buffer-prepend).
- `_warmup()` (`:215`); `_init_session()` (`:260`, clears
  `previous_hypotheses`, `pred_out_stream`, `current_text` ~`:300`;
  **`last_emitted_text` is cleared by the hard reset at `:601`, not by
  `_init_session`**); `_handle_audio()`
  (`:368-398`, re-preprocesses all accumulated audio `:403-415`);
  `_process_chunk()` (`:400`, `conformer_stream_step keep_all_outputs=False`
  `:451`); `_reset_session()` (`:485-607`: soft `:518-535`, hard pad
  `:541-546`, `_init_session` cold reset `:591-603`); `_process_final_chunk()`
  (`:609`, `keep_all_outputs=True` `:657-668`).

Client `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`:
- `stop()` sends an extra hard reset (`:79-83`); `run_stt()` (`:138-146`);
  `_send_reset()` (`:148-158`); `process_frame()` hard-reset on
  `VADUserStoppedSpeakingFrame` (`:159-171`); `_handle_transcript()` treats any
  `is_final` as `TranscriptionFrame`, ignores the `finalize` flag (`:193-209`).
  No `_audio_send_lock`.

Benchmark framework (UNCHANGEABLE) `stt-benchmark/src/stt_benchmark/`:
- `pipeline/benchmark_runner.py`: `Pipeline([transport, vad_processor,
  stt_service])` (`:178`); `VADProcessor(SileroVADAnalyzer(VADParams(
  stop_secs=self.vad_stop_secs)))` (`:173-174`); `vad_stop_secs` default 0.2
  (`:34`, `:53`).
- `cli/benchmark.py`: `--limit` (`:38-40`), `--vad-stop-secs` (`:55-57`);
  selects first `limit` samples by `dataset_index` order.
- `pipeline/synthetic_transport.py`: silence-until-`transcription_received`
  then `post_transcription_delay`, `max_silence_timeout`=10 s.
- `observers/transcription_collector.py`: space-concatenates every final
  `TranscriptionFrame`; ignores interim.
- DB `stt_benchmark_data/results.db`: `nemotron_local` by `model_name`
  (`''`=3.08% 1000-run, `rc1ref` 1.36%, `rc6` 1.60%, `rc13` 1.45%,
  `warmup` 1.26%, all on the 200-slice); `ground_truth` (1000 shared Gemini);
  `wer_metrics`.

Run/measure: server via `.venv-asr` python on GPU port 8080; experiment per
`model_name` tag, `uv run stt-benchmark run --services nemotron_local --model
<tag> --no-skip-existing` → `uv run stt-benchmark wer --services
nemotron_local --model <tag>`.

## Rules

### Hard constraints (locked — from the finding doc)
- The upstream benchmark **framework is unchangeable**: `benchmark_runner.py`,
  `synthetic_transport.py`, observers, CLI internals. Pipeline stays
  `[transport, VADProcessor(SileroVAD), stt_service]`; Silero VAD only; only
  `VADUserStarted/StoppedSpeakingFrame` exist (no turn-level
  `UserStoppedSpeakingFrame`).
- Changeable surface = **only Nemotron code**: `nemotron_local_stt.py` + its
  factory in `services.py`, `src/nemotron_speech/server.py`, and benchmark
  **CLI flags**. **No Smart Turn, no new dependencies.**
- Must emit exactly the frames the framework expects — ideally **one**
  `TranscriptionFrame` per complete utterance; only `finalize=true` may become
  a `TranscriptionFrame`; soft/speculative → `InterimTranscriptionFrame` or
  withheld.

### Production latency budget (locked — added 2026-05-18, user)
- **Hard target: end-to-end added latency < 400 ms in production.** Any config
  whose added wall-clock latency exceeds this is **analytical only**, never
  shippable. Report real-observer **TTFS and finalize latency** against this
  bar for every candidate.
- Latency taxonomy — what counts against the 400 ms budget:
  - **Endpoint/decision wait** (Silero `vad-stop-secs`; any client/server
    debounce hold) = real wall-clock, **in budget**; the scarce resource —
    keep ~200 ms or less.
  - **Encoder right-context** (rc1 ~160 ms; rc6/rc13 ~560/1120 ms) = real
    wall-clock, **in budget**. **rc1 is the only viable production context**
    (rc0 crashes — see Step 4; rc6/rc13 exceed budget).
  - **Finalize silence padding** ((R+1)*shift ~320 ms @ rc1) = **synthetic,
    faster-than-wallclock, NOT in budget** — zeros appended + one
    `conformer_stream_step`, ~tens of ms GPU compute, no sleep. Steps 7/8
    **must instrument and prove** finalize is compute-bound (measure; never
    assume). Precisely: the synthetic silence **duration** is excluded from
    budget, but the **measured finalize-flush wall-clock** (fork-clone +
    `inference_lock` wait + conformer compute + JSON send + client receipt)
    **IS budgeted** — "not in budget" refers only to the appended audio's
    notional duration, never to real flush time.
- **Budget formula (evaluate at p95, not median):**
  `endpoint_wait + encoder_right_context(rc1≈160ms) +
  measured_finalize_flush_wallclock + transport/client_overhead < 400 ms`.
  Observer TTFS already includes encoder_right_context; per candidate state
  which terms are measured vs modeled. Endpoint allowance is realistically
  ≲ ~150 ms (not 200), since rc1 alone consumes ~160 ms of the 400.
- **Required finalize instrumentation (Nemotron-owned; prerequisite for
  Steps 7c/7d, not an assumption):** extend the Step-1 client/server JSONL
  (keyed by run tag + `benchmark_batch()` order) with timestamps for
  `vad_stop`, debounce-expiry, fork-flush-start, fork-flush-done,
  final-sent, final-received, plus `inference_lock` wait; report median/p95.
  Without this, 7c/7d's <400 ms / faster-than-wallclock claims are
  unverifiable.
- Every measured candidate records median/p95 observer latency + measured
  finalize wall-clock. **Intermediate / analytical configs** (6c `ringbuf`,
  7a, 7b long-debounce, Phase G) are explicitly labelled
  *analytical — not shippable*; only **7d/8 shippable candidates must pass
  the < 400 ms p95 bar**.
- Consequence: VAD-stop > 0.2 s and any client debounce hold of ~250 ms or
  more are out-of-budget by construction → measured for analysis, not
  deployed. The only design that reaches oracle accuracy < 400 ms is the
  **server-side continuous context + speculative disposable fork** (Steps
  6-7): commit the finalize optimistically (faster-than-wallclock padding),
  discard the fork if speech resumes — accuracy decoupled from the decision
  wait.

### Measurement validity (fold or the numbers mislead)
- **Baseline vs oracle (do not conflate):** the *live baseline* is
  `model_name=''` (3.08% on 1000 / 2.74% on slice-A). `rc1ref` and `warmup`
  are *single-finalize oracle* tags (~1.3%), **not** the baseline. Every phase
  reports **paired Δ vs the live baseline `model_name=''`** and separately
  compares its absolute level to the oracle tags.
- The fixed `dataset_index` 0–199 slice (slice-A) is **biased low** (2.74% vs
  3.16% on the other 800). The benchmark CLI cannot select an arbitrary
  subset (`--limit` only takes the first N by `dataset_index`, and the CLI is
  framework-locked), so **every config is run over the full 1000**; `measure.py`
  derives slice-A and a seeded/stratified slice-B, paired same-sample-ID Δ vs
  baseline, and bootstrap CIs. Never extrapolate slice absolutes to the corpus.
- Every run's success criteria report, alongside semantic WER and **real
  benchmark observer TTFS** (median/p95, never harness reset latency):
  **hard-resets/sample, soft-finalizes/sample, and final-`TranscriptionFrame`s/
  sample** (from Step 4 telemetry).
- Record HF revision / resolved `.nemo` path + model-config hash per tag in a
  sidecar `stt-benchmark/stt_benchmark_data/run_metadata/<tag>.json` (the
  `results`/`runs` schema is framework-locked — no migration) before any
  number is canonical. All `uv run stt-benchmark …` commands are run from the
  `stt-benchmark/` directory.
- Per-sample VAD gap data and counters come from an **offline Silero pass over
  the DB audio** + a client per-connection JSONL keyed by run tag and the
  sequential `benchmark_batch()` order (the locked framework does not pass
  `sample_id`/`model` into the service, so service-side per-sample keying is
  infeasible — correlate by batch order instead).

### Safety & sequencing
- **Gate Gp (Phase G):** if the real-time single-reset run does not land in
  the oracle (~1.3%) region — large paired improvement vs the **live baseline
  `model_name=''`**, absolute level near the `rc1ref`/`warmup` oracle tags,
  non-overlapping CIs on both slices, **and zero early finals** — STOP: the
  continuous/fork redesign is solving the wrong problem. **Gp gates Steps 6
  AND 7.** If Gp fails, Steps 4–5 still proceed (Step 4 only; Step 5 is plumbing-only — no standalone accuracy/latency claim);
  Steps 6–7 do not.
- **Gate Ga (probe):** **Step 7 (fork) only**. The ring buffer (Step 6) is a
  prerequisite for *any* continuous-context design (including a fork
  redesign), so it is **gated on Gp only, not Ga** — a failed aliasing probe
  must not block the still-needed preprocessing fix.
- **Ring buffer before continuous context:** Step 6 (incremental preprocessor)
  must land and be verified **chunk/mel/text byte-equivalent** (not merely
  WER-within-CI) before Step 7 removes the per-utterance reset.
- Fork flush must hold `inference_lock` (no concurrent parent-stream + fork on
  the same model object); count that latency honestly.
- Each step is independently committable and leaves the benchmark runnable.
- **Baseline hardening (2026-05-18, dual-reviewed, pre-Step-6b):** the
  pre-existing uncommitted `server.py` changes were reviewed (me + Codex).
  Folded: device CUDA->CPU silent fallback **removed** -> hard `raise` if
  CUDA unavailable (no silent-CPU benchmark hazard); the ineffective+buggy
  `NEMOTRON_ONSET_WARMUP_MS` buffer-prepend **removed** (Step 8 implements
  the correct warm-up from scratch); the env-gated `NEMOTRON_DECODING=beam`
  branch **kept inert** (default greedy byte-identical to prior). Verified
  behavior-preserving for the default/validated config (model-load line
  unchanged; smoke: server loads on CUDA, healthy) -> **Gp/Ga/Step4/6a
  results remain valid**. `tts_server.py` committed separately (unrelated
  TTS, not ASR).

### Resource constraints
- Shared GPU: another project's vLLM holds ~26 GB; ~5 GB headroom. Run the
  Nemotron server via `/home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/
  python` on port 8080, **one server at a time**; never disturb the vLLM PIDs;
  reclaim GPU between configs.

### Known bounds & out of scope (do not over-expect / do not build)
- **Long-context is bounded (risk 5):** model left context is capped by
  `att_context_size[0]=70`; within a long pause the continuous cache fills with
  *silence* (≫ cold-init, < speech context). Cross-*utterance* context is
  mostly moot here (one WS connection per sample, mostly single utterances).
  The provable win is *never cold-resetting on within-utterance pauses* — do
  not expect unbounded gains from "more context."
- **Smart Turn / any real end-of-turn detector is explicitly out of scope** for
  this benchmark work (cannot change the framework, no new deps). It remains
  the recommended *production* architecture + developer guidance, recorded
  separately in the finding doc — **not** built or measured in this plan.
- Beam decoding is Blackwell-non-viable; the rc sweep already showed
  right-context flat — do not re-litigate model/decoder knobs here.

## Steps

- [x] **1. Measurement scaffolding + VAD gap preflight + telemetry (prereq for every number AND for Phase G)**
  Add `stt-benchmark/scripts/measure.py` (our code, not framework):
  (a) persist canonical `dataset_index` 0–199 slice-A and a seeded
  (seed=1234) duration-stratified slice-B id-list as sidecar JSON;
  (b) a scorer: given two tags, mean semantic WER on slice-A/slice-B, **paired
  same-sample-ID Δ vs live baseline `model_name=''`**, 95% bootstrap CIs —
  on whatever rows exist (so every config runs the **full 1000**; the
  framework-locked CLI `--limit` only takes the first N by `dataset_index`);
  (c) run-metadata recorder → `stt-benchmark/stt_benchmark_data/run_metadata/
  <tag>.json` (HF revision / resolved `.nemo` path / model-config hash; no DB
  schema change); (d) observer-TTFS + counter reader for a tag;
  (e) **offline Silero VAD preflight**: run Silero over every DB audio sample
  (the same way the live `VADProcessor` would) and persist the per-sample VAD
  start/stop timestamps and the **stop→next-start gap distribution**; this
  empirically sets Phase G's hold (Step 2) — current data: 650/1000 multi-
  start, max observed gap ≈ 2.70 s;
  (f) a **client telemetry mechanism**: per-connection JSONL keyed by run tag
  + sequential `benchmark_batch()` index (the only reliable correlation under
  the locked framework), recording hard-resets, final-`TranscriptionFrame`
  count, and an early-final flag — the minimal counters Gate Gp needs, so this
  ships **before** Phase G, not in Step 4.
  Success: slice id-lists persisted; scorer reproduces 2.74% slice-A for `''`,
  prints slice-B + paired Δ + CIs; `run_metadata/<tag>.json` written; offline
  gap distribution persisted with its max; telemetry JSONL emitted for a smoke
  run and correlates to samples by batch order.
  Key files: `stt-benchmark/scripts/measure.py` (new),
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`,
  `stt-benchmark/stt_benchmark_data/`

- [x] **2. Phase G — gating real-time single-reset test (Gate Gp; decides Steps 6–7)** — **GATE Gp: PASS** (slice-A 1.43%≈oracle 1.36%, slice-B 1.85%, paired Δ −1.31/−1.16 pp both CIs exclude 0; full 2.04% vs 3.08%; hard_resets/sample 1.0; early_final 1/1000; TTFS ~2.7 s = debounce cost → Steps 6–7 justified to remove that latency). **Steps 6–7 PROCEED.**
  *B4 constraint:* the benchmark transport never emits `EndFrame` before it
  waits for a transcript (`synthetic_transport.py` sends silence until
  `transcription_received`, `max_silence_timeout`=10 s; the runner
  cancels/`cancel()` after), and `stop()`/`cancel()` happen *after* a
  transcript is needed — so "ignore all stops, reset on `EndFrame`" would
  deadlock and produce a false negative on the decisive gate.
  Instead, in `nemotron_local_stt.py` only, add env-gated mode
  `NEMOTRON_FINALIZE_MODE=single` = a **large-window debounce** that collapses
  to exactly one finalize at true end *within the live benchmark*: handle both
  `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`; on each stop
  (re)start a timer; any start cancels it (no reset → no cold restart
  mid-utterance). **Hold is set empirically from Step 1's offline gap
  distribution**: above (max observed stop→next-start gap + margin) and below
  (`max_silence_timeout` 10 s − finalize budget). Use Step 1's **persisted
  authoritative** distribution `stt_benchmark_data/vad_preflight_silero_stop0.2
  .json` (live Silero analyzer/params, 1000 samples, 653 multi-start):
  **max=1.7 s, p99=1.22 s, p95=0.8 s** → hold ≈2.5 s (max + margin, ≪ 10 s −
  budget). This supersedes the earlier ad-hoc ~2.70 s estimate; if Step 2
  observes ANY clip split, treat the sidecar as under-measuring, raise the
  hold and re-derive. A guessed 2000 ms is unsafe — the hold must exceed
  every real intra-utterance gap.
  Finalize-handshake (pipecat 1.2.x: `VADUserStartedSpeakingFrame` clears
  `_finalize_requested`/`_finalize_pending` at `stt_service.py:507`): call
  `request_finalize()` **for the stop whose timer is currently armed**
  (immediately before sending the hard reset), and clear it when a later start
  cancels the timer; `confirm_finalize()` immediately before the single
  `TranscriptionFrame`; only `finalize=true` → frame.
  **Move `_audio_send_lock` into this step** (do not defer to Step 5): both
  `run_stt()` audio sends and the timer-driven reset send must acquire it
  (the timer fires concurrently with ongoing benchmark silence sends).
  Real-time, in-benchmark, server unchanged. Run over the full 1000, tag
  `phaseG_single`.
  Success / **Gate Gp**: large paired-Δ **vs live baseline `model_name=''`**,
  absolute near oracle (`rc1ref`/`warmup` ~1.3%), non-overlapping CIs on both
  slices, hard-resets/sample ≈ 1, and **early-final count = 0** (no final
  before source audio complete / before the sample's last VAD stop — verified
  from Step-1 telemetry). If not → STOP; record why. **Gp gates Steps 6–7**;
  on failure Steps 4–5 still proceed, Steps 6–7 do not.
  Key files: `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`

- [x] **3. Aliasing probe — NeMo fork safety (Gate Ga; gates 7c/7d)**
  Standalone probe script (uses `.venv-asr`, no server): load the model as the
  server does (**greedy only** — beam is Blackwell-non-viable and explicitly
  out of scope; do not exercise it), stream a few seconds of real audio to
  populate `cache_last_channel/time/len` + `previous_hypotheses` +
  `pred_out_stream`. **Note `previous_hypotheses` is a *list* of `Hypothesis`
  objects returned by `conformer_stream_step`, not an object with
  `.dec_state`.** Candidate clone recipe: `.detach().clone()` each cache
  tensor; for the list, deepcopy each `Hypothesis` and recursively clone every
  tensor field, **especially each `hyp.dec_state`**; never copy model/ws
  objects. Run pad + `keep_all_outputs` on the **fork**, then assert the
  parent's cache tensors + every hypothesis field are byte-identical and the
  parent's continued transcription is bit-identical to a no-fork control.
  Inspect the NeMo in-place sites to set the required clone depth: hypothesis
  mutation `rnnt_greedy_decoding.py:825-831` and `rnnt_utils.py:153-181`;
  **model-global / decoder train-eval toggles
  `rnnt_greedy_decoding.py:753-775`**; model-global
  `nemo/collections/asr/parts/mixins/streaming.py:53-74`
  (`drop_extra_pre_encoded`) — the last two prove a fork flush **cannot run
  concurrently** with parent inference, so the probe must also assert
  correctness only under a serialized (lock-held) flush.
  Success / **Gate Ga**: a clone recipe under which parent state + continued
  output are provably unchanged **under serialized flush**, documented for
  Step 7. If none exists → redesign Step 7 (separate model context) before
  building.
  Key files: `proj-2026-05-17-1708/probe_alias.py` (new, scratch)

- [x] **4. Phase 0 — rc0-vs-rc1 + vad-stop-secs control (independent of Gp)**
  Telemetry already ships in Step 1 (client JSONL + offline Silero gaps); this
  step only runs configs and reports. Run baseline finalization at **both**
  `--right-context 0` (~80 ms; finalize silence `(0+1)×16`=160 ms) and
  `--right-context 1` (~160 ms; 320 ms); tags `rc0_base`, `rc1_base`
  (full 1000 each). Also run the `--vad-stop-secs` control sweep
  {0.2,0.4,0.6,1.0} (tags `vad020…vad100`) — labelled a **control only**
  (changes segmentation + endpoint delay together; does not isolate reset
  damage). This step does **not** depend on Gp/Ga.
  Success (SUPERSEDED — see the CONCLUDED note below; actual concluded scope: `rc1_base`+`vad020` measured, `rc0` unsupported, `vad040/060/100`+`vad100` NOT run as out-of-budget): originally "paired WER parity rc0 vs rc1 on both slices (expected within CI —
  confirms rc0 halves mandatory finalize padding at ~no accuracy cost);
  vad-sweep WER/latency frontier recorded with CIs.
  **rc0 UNSUPPORTED (decided 2026-05-18):** `att_context_size=[70,0]`
  deterministically crashes upstream NeMo `multi_head_attention.py:267
  rel_shift` ("reshape tensor of 0 elements into [1,8,-1,0]") — 0
  transcriptions; NeMo is not changeable. Per user decision rc0 is dropped
  and recorded **unsupported**; revised Step 4 scope = `rc1_base` +
  `vad020` ONLY were measured (concluded 2026-05-18); `vad040`/`vad060`/`vad100` were NOT run — out-of-budget, no insight beyond Gate Gp; `rc0` unsupported. The rc0 canonical-table row
  becomes "unsupported — NeMo rel_shift [70,0]" (carried to Step 9).
  **Revised 2026-05-18 (user):** the vad-stop-secs frontier is run
  incrementally — `rc1_base` → `vad020` → `vad100` first (orchestrator
  `/tmp/step4_vad100.sh`); `vad040`/`vad060` are deferred and run only if
  `vad100` results warrant the full {0.2,0.4,0.6,1.0} frontier. Partial
  scoring over `'' rc1_base vad020 vad100`.
  Key files: `src/nemotron_speech/server.py` (rc via existing
  `--right-context`), measurement via Step 1

- [x] **5. Phase 1 — client debounce (works against today's server)** — [GUTTED 2026-05-18]
  **SCOPE CUT (user, 2026-05-18):** the {250,500,1000,1500} ms debounce-hold
  sweep is **out of the <400 ms production budget by construction** (smallest
  hold + ~200 ms VAD ≈ 450 ms) and analytically redundant with Gate Gp. **NO
  `dbnc*` full-1000 runs.** Step 5 is reduced to ONLY the client
  correctness/plumbing fix that Step 7a reuses — *implement + code-verify, do
  not measure*: the `_handle_transcript` finalize/interim split (only
  `finalize=true`→`TranscriptionFrame`; soft/interim→`InterimTranscriptionFrame`),
  suppress the extra post-final `stop()` hard reset, ignore empty/duplicate
  finals, apply the Step-2 armed-stop `request_finalize` discipline, reuse the
  Step-2 `_audio_send_lock` (no client timers/buffering — Step 7 moves all
  hold/finalization logic server-side). Success: frame contract correct (≤1
  `TranscriptionFrame`/utterance, interims interim, no dup/empty) by code
  review; full exercise deferred to Step 7a (no standalone benchmark — CLI
  can't subset and the sweep is cut). **DEAD TEXT: everything from "Independent of Gp/Ga" through the closing
  "Key files:" line below is pre-2026-05-18 design — DO NOT execute, DO NOT
  run any `dbnc*` sweep or any Phase-1 full-1000 run. Step 5's ONLY
  deliverable is the plumbing fix described above. Step 5 Key files:
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`.**
  Independent of Gp/Ga. Using Step 1's offline gap distribution, collapse VAD
  stops into a single finalize, sweeping the hold over {250,500,1000,1500} ms
  (note Silero adds ~200 ms start + ~200 ms stop, so a 250 ms hold ≈ commits
  after ~450 ms silence — and per Step 1 data, holds below the ~2.70 s max
  intra-clip gap *will* split some clips; capture that on the frontier).
  Implement the `_handle_transcript` fix (only `finalize=true` →
  `TranscriptionFrame`; soft/interim → `InterimTranscriptionFrame`); reuse the
  `_audio_send_lock` added in Step 2 (do not re-port). Suppress the extra
  `stop()` hard reset after a committed final; ignore empty duplicate finals;
  apply the same armed-stop `request_finalize` discipline as Step 2.
  Tags `dbnc250…dbnc1500` (full 1000 each).
  Success: paired-Δ vs baseline `''` on both slices with CIs; the WER↔hold-
  latency knee identified; residual gap vs Phase G documented (this trades
  latency for accuracy against the *unmodified* server).
  Key files: `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`

  *(Steps 6a–6c = Phase 2a, the risk-3 ring-buffer preprocessor. Gated by
  Gate Gp only — needed for any continuous design regardless of the fork
  probe; do NOT gate on Ga.)*

- [x] **6a. Equivalence harness + fixture characterization (no behavior change)**
  Add a test harness that, on fixed audio fixtures, captures the *current*
  full-re-mel path's per-chunk mel tensors, chunk boundaries, and emitted text
  as golden references; document the exact preprocessor params used today
  (`window_stride`/`hop_samples` at `server.py:178-179`; `pre_encode_cache_size`;
  `:403-415`, `:615-623` re-preprocess sites). No server behavior change.
  Success: golden artifacts persisted; harness re-runs green against unchanged
  server (byte-identical).
  Key files: `proj-2026-05-17-1708/` harness (new), `src/nemotron_speech/server.py` (read-only)

- [ ] **6b. Incremental ring-buffer preprocessor implementation**
  In `src/nemotron_speech/server.py` replace the re-preprocess-all path with
  incremental preprocessing, two distinct retained states (do not conflate raw
  STFT overlap with NeMo's mel/pre-encode cache): (i) a **raw-audio ring** of
  `window_size − window_stride` samples for STFT boundary-correct mel; (ii) a
  **mel-frame ring** of `pre_encode_cache_size` frames the cache-aware chunker
  prepends (cf. NeMo `parts/utils/streaming_utils.py:1640,1778`). Keep the
  `conformer_stream_step` call shape and `emitted_frames` semantics unchanged.
  Success: 6a harness passes — **deterministic mel equality, chunk-boundary
  equality, emitted-text equality** vs golden, on all fixtures.
  Key files: `src/nemotron_speech/server.py`

- [ ] **6c. Perf + WER-parity validation (tag `ringbuf`)**
  Verify O(N) (not O(N²)) preprocessing on a long synthetic input; then a
  paired full-1000 run (tag `ringbuf`, hard reset still in place) is
  WER-identical (Δ within CI) to the matched baseline on both slices.
  Success: O(N) confirmed; paired Δ ≈ 0 within CI both slices.
  Key files: `stt-benchmark/scripts/measure.py`, `src/nemotron_speech/server.py`

  *(Steps 7a–7d = Phase 2b. 7a/7b gated by Gp; 7c/7d gated by Gp **and** Ga.
  All require 6c done.)*

- [ ] **7a. WS protocol + thin-client translator (gated Gp; needs 6c)**
  Add WS protocol `{"type":"vad_stop"}` / `{"type":"vad_start"}`. Refactor
  `nemotron_local_stt.py` to a thin translator: forward audio; emit those
  signals; **`request_finalize()` for the armed stop, `confirm_finalize()`
  immediately before pushing the one server final as a `TranscriptionFrame`**
  (no-op without a prior live request — see Step 2 handshake); relay server
  interim → `InterimTranscriptionFrame`; no client timers/buffering; reuse
  Step-2 `_audio_send_lock`. Server side: just accept/parse the new messages
  (no behavior change yet — still does today's hard reset on `vad_stop`).
  Success: protocol round-trips; client is purely declarative; a paired run
  reproduces the **live baseline `''`** hard-reset numbers within CI (post-6c also matches `ringbuf`) — Step 5 produces no measured "Phase 1 best" — (no regression — pure plumbing).
  Key files: `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`,
  `src/nemotron_speech/server.py`

- [ ] **7b. Server continuous-context state machine, NO fork yet (gated Gp; needs 7a)**
  Server: one continuous never-finalized context — **no `_init_session` on
  within-utterance pauses**; explicit persistent `committed_text` (survives
  flushes, reset only at the true utterance boundary at `:601`-equivalent);
  add a **per-session state lock + explicit ordering/state-machine** for queued
  audio, `vad_start`/`vad_stop`, timer expiry, reset (the global
  `inference_lock` does not order per-session control vs audio). Still finalize
  by the existing (non-forked) padded `keep_all_outputs` path for now.
  Success: with a long debounce, mid-utterance pauses no longer cold-restart
  (telemetry: cold-resets/sample → 0 within a turn); paired-Δ vs baseline `''`
  improves toward Phase G on both slices; no dup/empty finals.
  Key files: `src/nemotron_speech/server.py`

- [ ] **7c. Serialized disposable fork flush (gated Gp+Ga; needs 7b)**
  Replace the in-place finalize with the Step-3 clone recipe: on `vad_stop`,
  deep-clone cache tensors + the `previous_hypotheses` list of `Hypothesis`
  (incl. each `hyp.dec_state`), run pad+`keep_all_outputs` **on the fork while
  holding `inference_lock`** (serialized — never concurrent with parent
  stream; account that latency); parent context untouched. Build on 6c's ring
  buffer (clone cache + minimal pending audio only).
  Success: Step-3 aliasing assertions hold under benchmark load (parent
  continued output bit-identical); no WER regression vs 7b.
  Key files: `src/nemotron_speech/server.py`

- [ ] **7d. Server-side debounce + emit-once validation (gated Gp+Ga; needs 7c) — tag `fork`**
  Add the server-side `NEMOTRON_FINALIZE_SILENCE_MS` debounce: `vad_start`
  in-window → discard fork (parent untouched); timer expiry → emit single
  delta once + reset for next utterance. Hold derived from Step 1's gap
  distribution (same constraint as Phase G). Run full 1000, tag `fork`.
  Success: paired-Δ vs `''` on both slices reaches the Phase G/oracle region
  **at < 400 ms real-observer latency (TTFS + finalize, measured against the locked Production-latency-budget rule; finalize proven faster-than-wallclock by the Required-finalize-instrumentation JSONL defined in the Production-latency-budget Rule, NOT assumed)** (short debounce + ~1.3%-region WER — the
  trade-off broken); emit-once verified (one `TranscriptionFrame`/utterance,
  early-final count = 0, no dup/empty); CIs non-overlapping vs baseline.
  Key files: `src/nemotron_speech/server.py`,
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`

- [ ] **8. Phase 3 — correct per-session warm-up (residual onset, UX polish)**
  In `server.py._init_session`, implement the correct warm-up (the broken `NEMOTRON_ONSET_WARMUP_MS` buffer-prepend was already removed in the 2026-05-18 baseline hardening — `_init_session` is now clean): after
  `get_initial_cache_state()`, synthesize ~`warmup_ms` silence →
  `model.preprocessor` → `conformer_stream_step(..., keep_all_outputs=False,
  drop_extra_pre_encoded=0)`. **Do not pass `return_transcription=False`** —
  NeMo logs that transcription cannot be disabled for Transducer models and
  still decodes the RNNT path; instead run it and **explicitly discard the
  returned hypotheses/text**, storing only the returned cache/hyps/pred_out
  into the session and intentionally leaving `current_text`/`last_emitted_text`
  unseeded. Two variants by ordering (the plan runs Step 8 **after** Step 6):
  • *After ring buffer (expected path):* seed the session cache from the
  warm-up step and initialize the raw/mel ring cursors + `emitted_frames`
  consistently — **no `accumulated_audio` bookkeeping**. • *If ever run before
  Step 6 (fallback):* keep the silence in `accumulated_audio` AND advance
  `emitted_frames` past it so the first real chunk hits the warm
  `emitted_frames!=0` branch. Sweep `warmup_ms ∈ {100,150,200,250}` (500 ms
  regressed — do not over-pad). Combine only with the best finalization
  config; tags `warm100…warm250`.
  Success: onset-fixable set ≫ 8/26 (≈ client-preroll 17/26) and a small
  positive paired-Δ on both slices with CIs; **no regression vs the current
  best finalization config — Step 7d (`fork`) if built, else the best in-budget finalization config (`fork` if built & < 400 ms p95, else hard-reset `ringbuf`/`vad020`); Phase G is an analytical oracle reference only** (Step 7 may be invalidated by Gp/Ga).
  Key files: `src/nemotron_speech/server.py`

- [ ] **9. Consolidate: canonical WER↔latency table + doc update**
  Run every canonical tag over the **full 1000**; report **full corpus,
  slice-A, and slice-B** with paired Δ vs `''` + bootstrap CIs +
  real-observer latency. Produce the canonical table — rows: baseline `''` /
  oracle `rc1ref` / Phase G (analytical) / `rc1_base` & `vad020` (in-budget ~212 ms,
  ≈baseline) / `ringbuf` / *Phase 2b (`fork`) if built — must pass < 400 ms
  p95* / *+Phase 3 if built* / a non-metric note row "rc0 unsupported — NeMo
  rel_shift [70,0]" (no `Phase 1 best`/`dbnc*`/`vad040-100` rows — never run);
  **each row tagged in-budget (< 400 ms p95) vs analytical-only, with
  real-observer TTFS + finalize latency** — and the recommendation. Update
  `docs/semantic-wer-finalization-finding.md` with measured results — also documenting the rc0-unsupported finding, the < 400 ms production budget + latency taxonomy, and the Step 4/5 scope reductions — (replace
  "unproven until Phase G" with the outcome) and **also correct the four
  doc-level factual errors surfaced in review**: (B1) `_init_session` clears
  `current_text` but `last_emitted_text` is cleared by the hard reset at
  `:601`, not `_init_session`; (B5) `previous_hypotheses` is a *list* of
  `Hypothesis`, not an object with `.dec_state`; (B12) `return_transcription=
  False` is invalid for RNNT (discard returned text instead); **(B-Ga)**
  deep-stack risk #1 cites `rnnt_greedy_decoding.py:825-831` `hyp.merge_`,
  which lives in `GreedyBatchedRNNTInfer` (`strategy=greedy_batch` /
  `loop_labels=True`) — `server.py:158-166` runs `loop_labels=False` →
  `GreedyRNNTInfer`, so that in-place merge does **not** fire on the configured
  path (Step 3 probe + independent re-run: server-path shallow+deep clean;
  batched-path shallow corrupts, deep clean). The deep-clone recipe is kept as
  defense-in-depth and the serialization-under-`inference_lock` requirement is
  independent and still mandatory. Record model
  revision hashes per tag.
  Success: one reproducible table (full + slice-A + slice-B, CIs); doc
  reflects measured reality and the 3 corrections; no claim exceeds evidence.
  Key files: `docs/semantic-wer-finalization-finding.md`,
  `stt-benchmark/scripts/measure.py`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Measurement scaffolding + VAD gap preflight + telemetry | done | `67b01ae` | slice-A 2.7364% reproduced; VAD gap max 1.7s/p99 1.22s (authoritative; Step 2 hold ≈2.5s); sidecars regenerable |
| 2 | Phase G large-debounce (empirical hold, early-final guard) | done | `0076206` | **Gate Gp PASS** — sliceA 1.43%/sliceB 1.85% vs base 2.74/3.01; Δ −1.31/−1.16pp CIs<0; full 2.04% vs 3.08%; hard_resets 1.0; early_final 1/1000; TTFS 2.7s. Steps 6–7 proceed |
| 3 | Aliasing probe (NeMo fork safety, serialized) | done | `eab05dd` | **Gate Ga PASS** (probe + independent re-run, exit 0): detector-selftest PASS (injected cache + `root.y_sequence` corruption caught); server-path `GreedyRNNTInfer`/loop_labels=False shallow+deep all clean; batched `GreedyBatchedRNNTInfer` shallow corrupts (hyps+continuation) / deep clean → deep-clone recipe robust even on the hazard path. Serialized-only: fork flush holds `inference_lock`, no concurrent parent step. 7c/7d proceed w/ deep-clone recipe. DOC-CORRECTION (B-Ga) queued → Step 9 |
| 4 | Phase 0 rc0-vs-rc1 + vad-stop-secs control | done | `d23b067` | **rc0 UNSUPPORTED**: `att_context=[70,0]` deterministically crashes upstream NeMo `multi_head_attention.py:267 rel_shift` (0-element reshape; 7.5h burned, 35M crashes, 0 transcriptions, 1st crash 2s into sample 1). Per user decision: rc0 dropped + recorded unsupported. Scope → `rc1_base` + `vad020/040/060/100` (5 cfg, all rc1). Sweep relaunched (5cfg). **Revised 2026-05-18 (user): run incrementally — rc1_base→vad020→vad100; defer vad040/vad060, evaluate vad100 then decide whether to backfill full frontier.** rc1_base done=3.18% (≈ '' 3.08%, reproducible). rc0 row → Step 9 table/doc. **CONCLUDED 2026-05-18:** in-budget points ≈ baseline (rc1_base sliceA 2.84%/sliceB 3.16%, Δ +0.10/+0.16pp CIs incl 0; vad020 2.89%/3.33% ≈ base); real-observer TTFS median ~212ms (**IN <400ms budget**) but accuracy-damaged (2.45 hard-resets/sample, 654/1000 early-final). vad-stop>0.2s + vad100 NOT run (exceed <400ms, no insight beyond Gp); rc0 unsupported. Conclusion: VAD tuning ≠ WER lever — Steps 6–7 must deliver Phase-G accuracy at ~212ms-class latency |
| 5 | Phase 1 client plumbing fix (sweep CUT) | done | `382c0cf` (stt-bench) | **GUTTED 2026-05-18 (user):** debounce-hold sweep out of <400ms budget + redundant w/ Gate Gp. Only the `_handle_transcript` finalize/interim plumbing fix Step 7a reuses; **no `dbnc*` runs**, implement + code-verify only. ✓ DONE: finalize/interim split + empty/dup/unarmed suppression + confirm_finalize + post-final stop() skip + `_send_finalize_reset` (reuses `_audio_send_lock`, no 2nd lock); default `''`/phaseG_single preserved; py_compile+import OK; no measurement (7a exercises it) |
| 6a | Equivalence harness + fixture characterization | done | `c48f1c9` | golden oracle: `equiv_harness.py` + `equiv_golden/` (3 sha256-pinned fixtures: 6/68/99 chunks). Independent `assert` re-run byte-identical PASS exit 0 (deterministic). Production untouched. **6b must add a ring-buffer mode to the harness that reproduces this golden via the new server path** |
| 6b | Incremental ring-buffer preprocessor impl | pending | — | gated Gp; needs 6a (byte-equal) |
| 6c | Perf + WER-parity validation (`ringbuf`) | pending | — | gated Gp; needs 6b |
| 7a | WS protocol + thin-client translator | pending | — | gated Gp; needs 6c |
| 7b | Server continuous-context state machine (no fork) | pending | — | gated Gp; needs 7a |
| 7c | Serialized disposable fork flush | pending | — | gated Gp+Ga; needs 7b |
| 7d | Server-side debounce + emit-once (`fork`) | pending | — | gated Gp+Ga; needs 7c |
| 8 | Phase 3 per-session warm-up | pending | — | residual onset; regress vs current best |
| 9 | Consolidate table + doc update (+3 doc fixes) | pending | — | full+slice-A+slice-B, CIs |
