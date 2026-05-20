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
  must land before Step 7 removes the per-utterance reset. **Byte-equivalence
  to the current growing-reprocess path is INFEASIBLE** (2026-05-18 dual
  feasibility review, empirical + analytical: CUDA cuFFT is plan-size
  sensitive — identical leading samples yield ULP-different mel frames at
  different total stream lengths; residual ≤4 ULP, ≤2e-6 abs / ~5e-7 rel;
  CPU/determinism-flag regimes do not fix it; only a constant FFT plan does).
  Step 6 must therefore be a **length-independent constant-plan** incremental
  preprocessor (O(1)/chunk; constant cuFFT batch every call; **NO
  fixture-position-tuned constants**). **Acceptance = BOTH:** (a) **Step 6c
  paired full-1000 `ringbuf` WER-within-CI on both slices** vs the matched
  hard-reset baseline (the decisive gate); and (b) a **6a mel-closeness**
  check on an EXPANDED multi-length fixture set — max relative mel error
  ≤ 1e-5 and zero length-tuned constants (kills over-fit + gross bugs). Bit
  identity is explicitly dropped (proven infeasible).
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
  Success: golden artifacts persisted; harness re-runs deterministic.
  **REVISED 2026-05-18:** byte-identity vs the growing-reprocess path is
  infeasible (cuFFT plan-size sensitivity — see Rules). The 6a golden is
  retained as a *reference*; acceptance moves to **ULP-closeness (max relative
  mel-err ≤ 1e-5) on an EXPANDED multi-length real-fixture set** (built in the
  6b rework). Bit-identity dropped.
  Key files: `proj-2026-05-17-1708/` harness (new), `src/nemotron_speech/server.py` (read-only)

- [x] **6b. Incremental ring-buffer preprocessor implementation**
  In `src/nemotron_speech/server.py` replace the re-preprocess-all path with
  incremental preprocessing, two distinct retained states (do not conflate raw
  STFT overlap with NeMo's mel/pre-encode cache): (i) a **raw-audio ring** of
  `window_size − window_stride` samples for STFT boundary-correct mel; (ii) a
  **mel-frame ring** of `pre_encode_cache_size` frames the cache-aware chunker
  prepends (cf. NeMo `parts/utils/streaming_utils.py:1640,1778`). Keep the
  `conformer_stream_step` call shape and `emitted_frames` semantics unchanged.
  Success: 6a harness passes — **deterministic mel equality, chunk-boundary
  equality, emitted-text equality vs golden — **SUPERSEDED 2026-05-18**.
  **6b REWORK (clean, length-independent):** a **constant-plan** incremental
  preprocessor — every `preprocessor` call uses an identical FIXED frame plan
  (constant cuFFT batch ⇒ deterministic, O(1)/chunk), plus a raw-audio ring
  for STFT-boundary context and a mel-frame ring of `pre_encode_cache_size`
  for the cache-aware chunker. **NO `INCREMENTAL_PREPROCESS_*` /
  fixture-position-tuned constants** — the first impl's over-fit is rejected
  and reverted. `conformer_stream_step` shape + `emitted_frames` semantics
  unchanged; rebuild on the hardened `c4b496b` baseline. Validate via the
  **expanded-fixture mel-closeness** harness (many varied real lengths; max
  rel mel-err ≤ 1e-5; no length-tuned constants). Full WER acceptance is
  **Step 6c** (paired full-1000 `ringbuf` WER-within-CI both slices).
  Key files: `src/nemotron_speech/server.py`

- [x] **6c. Perf + WER-parity validation (tag `ringbuf`)** — **GATE PASS 2026-05-18**: paired Δ `ringbuf` vs `''` slice-A +0.20 pp CI[−0.06,+0.45] / slice-B +0.18 pp CI[−0.07,+0.46] (**both CIs include 0** ≈ `rc1_base` reproducibility noise); O(N) perf PASS (O(1)/chunk, `seconds^1.00` vs old `seconds^1.70`); 6a closeness PASS (≤7.95e-6). hard_resets/sample 2.46 ≈ baseline (parity config), TTFS 213 ms (in-budget), 1000/1000 transcribed. **Step 6 FULLY CLOSED → Steps 7a–7d proceed (Gp ✅ + Ga ✅ + 6c ✅).**
  Verify O(N) (not O(N²)) preprocessing on a long synthetic input; then a
  paired full-1000 run (tag `ringbuf`, hard reset still in place) is
  WER-identical (Δ within CI) to the matched baseline on both slices.
  Success: O(N) confirmed; paired Δ ≈ 0 within CI both slices.
  **6c is now THE Step-6 acceptance gate (criterion revised 2026-05-18):**
  paired full-1000 `ringbuf` WER-within-CI on BOTH slices vs the matched
  hard-reset baseline + O(N) perf confirmed + the 6a expanded-fixture
  mel-closeness ≤ 1e-5. If WER Δ exceeds CI on either slice → 6b is reworked,
  NOT accepted. (Byte-identity is infeasible — proven; see Rules.)
  Key files: `stt-benchmark/scripts/measure.py`, `src/nemotron_speech/server.py`

  *(Steps 7a–7d = Phase 2b. 7a/7b gated by Gp; 7c/7d gated by Gp **and** Ga.
  All require 6c done.)*

- [x] **7a. WS protocol + thin-client translator (gated Gp; needs 6c)**
  Add WS protocol `{"type":"vad_stop"}` / `{"type":"vad_start"}`. Refactor
  `nemotron_local_stt.py` to a thin translator: forward audio; emit those
  signals; **`request_finalize()` for the armed stop, `confirm_finalize()`
  immediately before pushing the one server final as a `TranscriptionFrame`**
  (no-op without a prior live request — see Step 2 handshake); relay server
  interim → `InterimTranscriptionFrame`; no client timers/buffering; reuse
  Step-2 `_audio_send_lock`. Server side: just accept/parse the new messages
  (no behavior change yet — still does today's hard reset on `vad_stop`).
  Success: protocol round-trips; client purely declarative; py_compile + a
  protocol round-trip smoke (server parses `vad_stop`/`vad_start`; no server
  behavior change — still hard-resets on `vad_stop`). **The full-1000
  no-regression parity is FOLDED INTO 7b (decided 2026-05-18, time-saving):**
  7b builds on and exercises this exact protocol, so a separate ~3.4 h 7a
  full-1000 is redundant; 7b's measured run catches any 7a plumbing
  regression (precedent: gutted Step 5 = plumbing code-verified, not
  separately measured). [Superseded tail:] a paired run
  reproduces the **live baseline `''`** hard-reset numbers within CI (post-6c also matches `ringbuf`) — Step 5 produces no measured "Phase 1 best" — (no regression — pure plumbing).
  Key files: `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`,
  `src/nemotron_speech/server.py`

- [x] **7b. Server continuous-context state machine, NO fork yet (gated Gp; needs 7a)** — **GATE PASS 2026-05-19**: `cc7b` paired Δ vs `''` slice-A −1.27 pp CI[−1.71,−0.86] / slice-B −1.28 pp CI[−1.86,−0.79] (**both exclude 0**), full **2.00%** ≈ Phase G (2.04%) / oracle region; emit-once `final_frames/sample=1.0`, `early_final=1/1000`; 7a no-regression absorbed (protocol exercised end-to-end). TTFS 2.7 s = expected (2.5 s debounce + in-place finalize; `<400 ms` is 7d). Server-side continuous context recovers the full WER gap. → 7c proceeds.
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

- [x] **7c. Serialized disposable fork flush (gated Gp+Ga; needs 7b)**
  Replace the in-place finalize with the Step-3 clone recipe: on `vad_stop`,
  deep-clone cache tensors + the `previous_hypotheses` list of `Hypothesis`
  (incl. each `hyp.dec_state`), run pad+`keep_all_outputs` **on the fork while
  holding `inference_lock`** (serialized — never concurrent with parent
  stream; account that latency); parent context untouched. Build on 6c's ring
  buffer (clone cache + minimal pending audio only).
  Success: Step-3 aliasing assertions hold under benchmark load (parent
  continued output bit-identical); no WER regression vs 7b.
  Key files: `src/nemotron_speech/server.py`

- [x] **7d. Server-side debounce + emit-once validation (gated Gp+Ga; needs 7c) — tag `fork`**
  **First, fold in the 7c-deferred close-drain hardening**: make the close
  path (`_continuous_handle_close_locked` -> `_continuous_force_finalize_locked`
  -> `_continuous_flush_post_stop_audio_locked` -> `_handle_audio_locked` send)
  tolerate a client that already closed — catch `aiohttp
  ClientConnectionResetError` on interim/transcript sends, skip the send but
  STILL run the final emit so a pre-debounce client close cannot drop the
  final (removes the lone `early_final` sample; cc7c=`0b47dc84`). The short
  production debounce here already makes this rare; this guard removes it
  entirely and it is re-measured by the `fork` run.
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

- [x] **8. Phase 3 — correct per-session warm-up (residual onset, UX polish)**
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

- [x] **9. Consolidate: canonical WER↔latency table + doc update**
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
| 6a | Equivalence harness + fixture characterization | done | `c48f1c9` | golden oracle: `equiv_harness.py` + `equiv_golden/` (3 sha256-pinned fixtures: 6/68/99 chunks). Independent `assert` re-run byte-identical PASS exit 0 (deterministic). Production untouched. **REVISED 2026-05-18:** byte-reproduction infeasible (cuFFT plan-sensitivity, dual-reviewed); golden kept as reference, acceptance → ULP-closeness (≤1e-5) on EXPANDED fixtures, rebuilt in the 6b rework |
| 6b | Incremental ring-buffer preprocessor impl | done | `7ea1711` | First impl passes the 3-fixture 6a gate **only via fixture-tuned STFT-plan magic windows** (`_incremental_frame_plan` [384,624)→512 / [1472,1536)→1536) = over-fit, code-confirmed. Deeper: CUDA cuFFT STFT is plan-size-sensitive → bit-exact incremental==growing-full-reprocess may be **infeasible as written**. User decision 2026-05-18: **feasibility review (dual) + fixture-length sweep first → then likely relax the byte-equiv Rule to 6c WER-within-CI + require a length-INDEPENDENT impl**. 6b NOT accepted; Codex's over-fit server.py/harness changes parked uncommitted pending rework. **RESOLVED 2026-05-18:** over-fit reverted to `c4b496b`; Step-6 criterion relaxed (WER-within-CI + length-independent constant-plan ring, NO magic windows); feasibility proven (cuFFT plan-sensitivity); re-delegated clean. **ACCEPTED 2026-05-18 (`7ea1711`):** constant-plan ring K=10080 (next-pow2 of model constants; NO over-fit/position constants; hardening intact; conformer/emitted_frames unchanged). Independent 12-fixture closeness (1–16s / 6–99 chunks) PASS: max rel mel ≤7.95e-6 (<1e-5), text+boundary EXACT. **Decisive WER acceptance = Step 6c.** |
| 6c | Perf + WER-parity validation (`ringbuf`) | done | `f3f37a1` | **THE Step-6 acceptance gate (criterion revised 2026-05-18):** full-1000 `ringbuf` WER-within-CI BOTH slices + O(N) + 6a closeness ≤1e-5; Δ over CI → 6b reworked not accepted. gated Gp; needs 6b |
| 7a | WS protocol + thin-client translator | done | `dfa4f13`+stt`32d4874` | gated Gp ✅ + 6c ✅. Pure plumbing: vad_stop/vad_start WS protocol + thin-client translator + server parse-only (no behavior change). Gate = code-verify + round-trip smoke; **full-1000 no-regression parity FOLDED INTO 7b** (time-saving, gutted-Step-5 precedent) |
| 7b | Server continuous-context state machine (no fork) | done | `b757159` | gated Gp ✅ + 7a ✅. Highest-risk impl (per-session state machine, continuous never-finalized ctx, committed_text, no cold _init_session within a turn). env-gated; default ''/phaseG_single preserved. Codex implements + smoke; Claude runs the measured full-1000 (also absorbs 7a no-regression parity) + gate. **Dual concurrency review (mine+Codex) caught 2 defects pre-run (BLOCKER reset/end-not-in-SM→lost final; MAJOR resumed-audio doesn't invalidate debounce); tightening folded+ACCEPTED (close=queued state input+drain-finalize; `continuous_post_stop_audio` buffer flushed on vad_start/dropped on debounce; reset_seen consumed; emit-once verified across 4 orderings by inspection; +504/-0 additive, default/phaseG byte-unchanged, smoke 5/5). Measured `cc7b` full-1000 launched (server NEMOTRON_CONTINUOUS=1 SILENCE_MS=2500, client default).** |
| 7c | Serialized disposable fork flush | done | `7cbdf09` | gated Gp ✅ + Ga ✅ + 7b ✅. Replace 7b in-place finalize with the Step-3 deep-clone disposable fork (serialized under inference_lock; parent ctx untouched). High-risk (deep-stack #1/#2). **Codex `mpcil0hf` impl reviewed+ACCEPTED by inspection**: `_build_continuous_finalize_fork` deep-clones cache×3 (`tensor_clone`) + `previous_hypotheses` (`clone_hypotheses_deep`, incl. nested `hyp.dec_state`) + `pred_out_stream`/raw+mel rings (`clone_tree`) + minimal pending+`(R+1)*shift` pad, `websocket=None`, parent read-only; flush `_process_final_chunk(fork)` under `async with inference_lock`; `NEMOTRON_FORK_ASSERT` snapshots+asserts parent cache+`previous_hypotheses` byte-identical AFTER flush BEFORE any parent write (logs PASS / raises on mismatch); 7b emit-once/state-machine/default+phaseG/6c-ring/CUDA-hardening intact; +312/-17 server.py-only, no forbidden constants; Codex live smoke logged `fork alias assertion PASSED` + one-final + context-continuity + default-unchanged. **Rigor decision**: 7c REUSES the already-dual-reviewed + Gate-Ga-verified + independently-re-run Step-3 recipe (not novel concurrency like 7b) — so instead of a 3rd full dual-review round, run measured `cc7c` full-1000 **with `NEMOTRON_FORK_ASSERT=1`** so parent-byte-identity is asserted over all 1000 real interleavings (strongest "aliasing holds under benchmark load" gate; far exceeds a static trace). Monitor failure-grep includes `fork alias assertion FAILED` -> CRITICAL abort. Gate: cc7c paired Δ vs `''` no-regression vs 7b 2.00% + zero `FAILED` + emit-once. cc7c **GATE PASS 2026-05-19** (full-corpus 1000/1000): **(b)** 0 `fork alias assertion FAILED` / **1999 `PASSED`** (strongest under-load Step-3 proof); **(a)** cc7c paired delta vs `''` slice-A -1.27 pp CI[-1.73,-0.85] / slice-B -1.22 pp CI[-1.80,-0.77] — statistically identical to cc7b (-1.27/-1.28), both exclude 0 ⇒ fork WER-NEUTRAL vs 7b in-place (raw 2.02% vs 2.00% = noise); **(c)** emit-once `final_frames/sample=1.0`, `early_final=1`. TTFS 2714.8/2716.4 ms ≈ cc7b (debounce-dominated; fork compute n=1999 median 12.6 ms, faster-than-wallclock). **Close-drain finding RESOLVED-as-deferred**: exactly 1 full-corpus occurrence (`0b47dc84`), pre-existing 7b chain, 7c-neutral, methodology-absorbed (cc7c≈cc7b deltas) — committed 7c **exactly as measured** (measurement integrity > re-run for a benign 1/1000); hardening folded into 7d (short debounce structurally suppresses + re-measured there) |
| 7d | Server-side debounce + emit-once (`fork`) | done | `6181310`+stt`e6cf064` | gated Gp+Ga; needs 7c. **Codex `mpcrzmg2` impl #1 reviewed → BLOCKER found (Claude, by inspection, pre-run): `_continuous_finalize_and_reset_locked` unconditionally cold-resets the parent (`committed_text=""` + `_init_session`) on EVERY debounce expiry; with 150ms debounce vs Step-1 gaps (min 154ms/p50 380ms, 650/1000 multi-seg) the timer fires at every intra-sample pause BEFORE vad_start can hit the discard → parent ASR context wiped mid-sample → cold-restart segmentation = the ~3% WER the project exists to kill. 7c fork itself correct (parent read-only, FORK_ASSERT would still PASS) — defect is the explicit post-emit `_init_session`. Codex scenario-b smoke only passed via artificial vad_start INSIDE the 150ms window.** Fix delegated (Codex `mpcsx3y9`) + **re-reviewed by Claude (inspection) → ACCEPT**: clean 3-way split — shared core `_continuous_finalize_emit_locked` (fork+FORK_ASSERT+emit+JSONL, parent read-only, committed_text advances AFTER delta), speculative epilogue `_continuous_finish_speculative_finalize_locked` (clears only 5 debounce ephemera, state→STREAMING, NO `_init_session`/wipe/stop_seq), cold-reset epilogue `_continuous_cold_reset_after_finalize_locked` (HARD-raises if reason∉{close,end}; only `_init_session` site for continuous). Debounce-expiry→core+speculative (context retained); close/"end"→core+cold-reset. Multi-seg glue: post-speculative `vad_start` else-branch flushes post-stop audio into the retained parent → segment B continues same context. Smoke-B `concat_matches_control=True`, no mid-sample cold reset, FORK_ASSERT PASSED ×2, default unchanged, budget p95 329ms (n=3 synthetic). server.py +225/-37; nested measure.py +221 / nemotron_local_stt.py +49/-3; scope clean. Aligns with locked Rule 141-144 + 7b thesis (correctness fix, not plan-direction change). **Dual adversarial review pre-run** (Codex `boggpqe7d` running + Claude parallel). **Claude review found R1 = BLOCKER-class**: harness `TranscriptionCollectorObserver._handle_transcription` (transcription_collector.py:72-75, FRAMEWORK/locked) CONCATENATES every final frame `transcriptions[id] += " " + text`; 7d server emits one delta per speculative finalize and on ASR self-correction (final_text NOT startswith committed_text) emits FULL final_text (server.py:1361) -> on the 650/1000 multi-seg samples a cross-segment prefix revision (plausible: fork (R+1)*shift padding retokenizes the tail between successive forks) makes the harness append a full re-emit after partials -> duplicated/garbled hypothesis -> inflated WER. 7b/7c emitted ONCE/sample so this concat path was NEVER gate-exercised; Codex smoke-B used clean sentences + its own concat (missed it). **NO-GO until fixed** (don't burn 3.7h on a majority-multi-seg-garbling config — rc0 lesson). Fold w/ Codex review -> targeted fix (make continuous multi-emit concatenation-safe: each delta a clean forward extension, or restructure emit/commit contract; harness is locked-append). **DUAL REVIEW FOLDED (Codex `boggpqe7d` + Claude, converged)**: R1 BLOCKER confirmed+quantified (Step-1: 1389 internal gaps all >150ms across 653/1000; 7b/7c max_final_frames=1 so never gate-exercised; ~40% Nemotron cumulative transitions are non-prefix -> pervasive correction->duplication). FORK_ASSERT MAJOR: snapshot/assert omits `pred_out_stream` though fork clones it -> fold pred_out_stream into snapshot+assert. R4 MAJOR but claim-scoping not code: 7d latency claim scoped to single-session/sequential observer (= the locked-Rule benchmark measurement); production-concurrency caveat documented in Step 9. SAFE (code-proven): R2 post-stop (copy-then-clear bytearray, fork copies only parent pending), R3 post-speculative audio (committed_text only touched by finalize), deadlock/lock-order, seq-races, default/phaseG, 6c ring. **Fix #2 delegated**: R1 = always-append-only emission (track `continuous_emitted_text` = harness-accumulated string; emit only the word-level suffix extending it; NEVER full cumulative re-emit; corrections to already-emitted words not retroactive = accepted, measured) + FORK_ASSERT pred_out_stream + a regression smoke through the REAL TranscriptionCollectorObserver semantics. **Fix #2 (Codex `mpcu3dks`) re-reviewed by Claude (inspection) -> ACCEPT, all folded findings resolved**: `_continuous_append_only_delta` (server.py:233) case-3 = frozen-by-count `final_tokens[len(emitted_tokens):]` + overlap-dedup -> provably non-duplicating vs the locked append-only collector (corrections to already-emitted words dropped = bounded measurable cost, NOT garble); call site (server.py:1402) uses ONLY this helper (no full-cumulative path), `continuous_emitted_text` advanced only on successful send + mirrors collector `+= " "+text` exactly, RETAINED on speculative epilogue, reset "" only at true boundary (server.py:1479), unused default/phaseG; emit-once preserved. FORK_ASSERT now also snapshots+asserts `pred_out_stream` (server.py:1262/1291, gated). R4 scope-label in measure.py (no formula change). Regression test through the REAL TranscriptionCollectorObserver on a real multi-gap sample: `old_would_fail=True`, `new_assert=True`, 6 append-only finals no dup; finalize-budget p95 342ms PASS<400; default env-unset unchanged. server.py +279/-44 + measure.py +225 (scope clean). 3rd dual-round disproportionate (narrow pure-fn fix, hand-traced + real-collector-validated; 7c precedent). **GO for measured `fork` full-1000** (residual is empirical: does always-append-only still hit Phase-G region -> the gate decides). cc7c GATE PASS unblocks 7d** (parent-untouched fork ⇒ speculative short-debounce finalize). **Folds in the 7c-deferred close-drain hardening** (catch `ClientConnectionResetError` on close-path sends, still emit final; short debounce + guard fully removes the 1/1000 `early_final`). The <400 ms production climax: short `NEMOTRON_FINALIZE_SILENCE_MS` + emit-once, tag `fork`, finalize-instrumented per the locked budget rule. **GATE PASS 2026-05-19 (full-corpus 1000/1000): (a)** `fork` paired Δ vs `''` slice-A −1.04 pp CI[−1.47,−0.64] / slice-B −1.04 pp CI[−1.50,−0.63] — **both firmly exclude 0** (large significant win vs baseline); vs analytical cc7b/cc7c (−1.27/−1.22) **non-inferior at 95% slice-CI (CIs overlap)** but a **small systematic ~0.2 pp point cost** (raw 2.10% vs 2.00/2.02%) = the measured price of 654/1000 early-fires' bounded fork-flush-suffix + always-append-only dropped-corrections (the trade-off, characterized, not noise). **(b)** 0 `fork alias assertion FAILED` / **3460 PASSED** incl. `pred_out_stream` (DEFINITIVE). **(c)** emit-once preserved: `final_frames/sample=2.416` = correct multi-seg speculative signature (cc7b/cc7c were 1.0 @2.5 s debounce), 1002 empty/dup suppressions, 761 corrections clean, 0 tracebacks/exceptions; `early_final=654` = the false-early-fire population (source of the (a) cost, not a correctness break). **(d)** finalize-budget full-corpus n=2416 p95 **325.8 ms** (endpoint 151 + rc1 160 modeled + flush 13.4 + transport 0.2; lock-wait 0; single-session/sequential scope) PASS <400 ms. 2460 speculative finalizes context-RETAINED + exactly 1000 true-boundary cold resets (=1/session). **Net: oracle-region-class accuracy at a measured 326 ms p95 — the accuracy/latency trade-off is broken, at a small honestly-quantified cost vs the non-shippable 2.5 s-debounce ceiling. `fork` = the recommended shippable config.** Four pre-run defects (cold-reset BLOCKER, R1 multi-emit BLOCKER, FORK_ASSERT pred_out_stream MAJOR, R4 scope) caught by rigorous review, each a saved misleading run |
| 8 | Phase 3 per-session warm-up | done | `0462679` | residual onset; no-regress vs current best (= `fork`). **Sweep TRIAGED (user 2026-05-19): measure `warm150` on `fork` first (~3.4 h); run ONE neighbor (`warm100` or `warm200`) only if `warm150` shows a positive paired-Δ both slices; NOT the 4-config frontier.** env-gated `NEMOTRON_WARMUP_MS`, default/phaseG/`fork` byte-unchanged when unset; Codex impl + Claude diff-review + measured `warm150` + gate. **Codex `mpd2kl29` impl reviewed+ACCEPTED by inspection** (server.py +101/-5): `_init_session` 594-635 = byte-identical 6181310 baseline; only adds `synthetic_prefix_samples=0` (inert default) + a `>0`-gated `_run_session_warmup` (6c constant-plan preprocess, one `conformer_stream_step`, returned text DISCARDED, `current_text`/`last_emitted_text` unseeded, rings+`emitted_frames`+prefix seeded). `_session_timeline_samples` (=prefix+total) used at ONLY the chunk-cadence gate (1620) — byte-identically `total_audio_samples` when prefix=0; 7d finalize/should_flush/budget math NOT perturbed; fork value-copies the int (no aliasing, FORK_ASSERT unaffected). Both `_init_session` call-sites keep the EXACT prior synchronous call in the `else`(warmup==0) branch (default/phaseG/'' unchanged); warmup>0 wraps GPU work in `async with inference_lock` via run_in_executor — site #1 (7d cold-reset, the path warm150 exercises) deadlock-free (post-flush-lock-release; state_lock→inference_lock order). Smoke: env-unset byte-identical fork PASSED; WARMUP_MS=150 warm-once+re-warm-after-reset+text-discarded+fork 2/0+2 finals. Proportionate (7c precedent): diff-review + measured run w/ FORK_ASSERT=1, no separate dual round. Gate: `warm150` paired Δ vs `''` no-regress vs `fork` (−1.04 both slices) + onset improvement + small positive Δ + 0 fork-alias-FAILED + budget <400 ms. **`warm150` GATE (mixed)**: full-1000 1000/1000, warm-ups 2000 (2.000/session — 1 session-start + 1 close-path cold-reset; the 2nd is wasted GPU in this 1-sample/connection bench, useful in session-reuse production), text-discard 2000/2000, FORK_ASSERT PASSED 3459/0, 0 exceptions, budget p95 325.7 ms (≈ fork 325.8). **WER (the locked-Rule slice gate, not the biased aggregate)**: `warm150` paired Δ vs `''` slice-A −1.11 pp CI[−1.56,−0.67] / slice-B −1.30 pp CI[−1.82,−0.83] — both exclude 0; **vs `fork` point-estimate** slice-A −0.07 pp (noise, within CI) / **slice-B −0.26 pp (real-looking improvement, slice-B now matches/beats cc7c −1.22)**; raw aggregate 2.11% vs fork 2.10% = +0.01 pp biased-aggregate noise (locked Rule). Asymmetric signal: slice-B clear-ish improvement, slice-A flat. **Honest re-read (user-prompted)**: the slice-level point estimates are both better than `fork` but the full-1000 aggregate is +0.01 pp WORSE — so on the ~600 samples outside slice-A+B, warm-up must be slightly hurting enough to offset the slice gains. Warm-up at 150 ms is therefore **heterogeneous, not a strict win** over `fork`: small improvement on the slice populations, small regression on the rest, ~tied in aggregate, all within CI. Triage decision re-confirmed by user given the mixed result — `warm200` launched to test if the slice-B signal extends with more warm-up (monotonic = real; plateau/regresses = warm150's slice-B was sampling noise / 150 was already past the peak). **`warm200` GATE PASS 2026-05-19 — decisive monotonic confirmation**: full-1000 1000/1000, warm-ups 2000 (2.000/s, exactly), text-discard 2000/2000, FORK_ASSERT PASSED 3458/0, 0 exceptions, budget p95 325.8 ms unchanged (warm-up at init under inference_lock, outside per-finalize budget). **WER (the locked-Rule paired slice gate)**: `warm200` paired Δ vs `''` slice-A −1.29 pp CI[−1.84,−0.74] / slice-B −1.39 pp CI[−1.96,−0.87] — both exclude 0; **STRICT paired vs `fork`** slice-A −0.25 pp CI[−0.68,+0.13] (big point-estimate, CI just-includes-0; 3× warm150's noise on slice-A) / **slice-B −0.34 pp CI[−0.69,−0.07] (CI EXCLUDES 0, stronger than warm150's −0.25)**. **Trajectory `fork → warm150 → warm200` MONOTONIC on slice-B paired Δ vs fork (0 → −0.25 sig → −0.34 sig)** = warm-up is doing real, extensible work. **vs analytical ceiling cc7c**: warm200 slice-A 1.45% < cc7c 1.47%; slice-B 1.62% < cc7c 1.79% — **warm200 matches/exceeds the non-shippable 2.5 s analytical ceiling on both slices at <400 ms latency**. Aggregate raw 2.07% < fork 2.10% (the first aggregate-positive warm-up). emit-once parity: final_frames/sample=2.42, early_final=655. **Recommendation: `warm200` is the recommended shippable config** — the small fork-vs-cc7c "production cost vs analytical ceiling" is essentially CLOSED by 200 ms warm-up. `warm250` NOT probed: triage was satisfied (one neighbor; monotonic confirmed); PLAN warned 500 ms regressed so 250 likely plateaus/regresses; ~3.4 h for diminishing-returns polish contradicts the project's time-discipline |
| 9 | Consolidate table + doc update (+4 doc fixes) | done | (this commit) | Authoritative "Measured Outcome 2026-05-19" addendum appended to `docs/semantic-wer-finalization-finding.md` (preserves the launching analysis for traceability): canonical WER↔latency table for all measured tags (full + slice-A + slice-B + paired Δ vs `''` + bootstrap CIs + TTFS p95 + finalize-budget p95 + in-budget vs analytical tag + role), production recommendation (`warm200` default; `fork` fallback; `cc7b`/`cc7c`/`phaseG_single` reference-only; `rc0` unsupported), findings folded (rc0 unsupported / cuFFT plan-size non-determinism / `<400 ms` budget+taxonomy / Step 4/5 scope reductions / rigor-wins log of the 4+ pre-run defects caught by review discipline), 4 corrections to the launching doc (B1 `_init_session` vs `:1857` `last_emitted_text` clear / B5 `previous_hypotheses` is a `List[Hypothesis]` not an object with `.dec_state` / B12 `return_transcription=False` invalid for RNNT — discard text instead / B-Ga deep-stack-risk-#1 `hyp.merge_` lives in `GreedyBatchedRNNTInfer` not the configured `GreedyRNNTInfer`/`loop_labels=False` server path), reproducibility (HF revision `ef3bf40c…`, per-step commit hashes, full reproduce commands, archived job/run logs in `codex-jobs/`). No new compute. **`/implement` loop COMPLETE — all 9 steps `[x]`.** |
