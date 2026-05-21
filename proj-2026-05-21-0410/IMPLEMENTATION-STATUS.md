# Implementation status — batching optimization (autonomous session handoff)

Written at the end of the autonomous run (started 2026-05-21 ~02:43). Summary of where the batching
optimization stands and how to continue.

## What's validated (the de-risking front — DONE)
The two **load-bearing probes both PASS**, so the entire batching approach is correctness-validated;
the rest is engineering on a proven foundation.

- **Step 0 — baseline artifact: DONE.** `baseline/english_baseline.json` — 8 English clips (interim
  sequences + final + delta) from the current B=1 greedy server, with git identity (HEAD 1deac460 +
  dirty status + server.py diff-sha). The byte-exact reference for all later gates. (Multilingual
  baseline still TODO — needs the EA-venv server.)
- **Probe B — batched-state correctness: GO.** `probe_batched_step.py` proved batched `conformer_stream_step(B=2)`
  is **byte-identical per-stream** to separate B=1 runs, including (a) row-order permutation invariance
  and (b) **mid-stream dim-1 cache stacking** — running two streams separately, then stacking their
  independent caches (channel/time on dim 1, len on dim 0) + flat hypothesis list and continuing as a
  batch. This is the exact scheduler scenario + the documented "batching corrupts cache-aware state"
  hazard → **RESOLVED**. (The transcripts are intentionally garbled — a simplified consistent chunk
  feed; the test is batched==separate consistency, which is exact.)
- **Probe C — decoder strategy: GO.** `probe_decoder_strategy.py` proved `strategy=greedy_batch`
  (loop_labels=True, use_cuda_graph_decoder=False) is **byte-identical** to the current `strategy=greedy`
  at B=1 across clips → **the decode can be batched too** (full encoder+decode ceiling ~8-10×, not just
  the encoder-only fallback).
- **Step 5 — batch state primitives: DONE (module + tested).** `src/nemotron_speech/batch_primitives.py`
  codifies the validated stacking recipe (cache concat/scatter on dim 1 channel/time + dim 0 len, flat
  per-row hypothesis list with alias guard, ragged-batch rejection, grouping key). `test_batch_primitives.py`
  PASSES (structural round-trip), and Probe B was refactored to USE these functions and still GOes
  (model-validated end-to-end). The in-server wiring (the `_process_batch` method + the scheduler) is
  Steps 6-7.

- **Probe A — encoder compile: GO.** `probe_encoder_compile.py` — `torch.compile(encoder.cache_aware_stream_step,
  mode="reduce-overhead")` compiles, is numerically correct (max|Δ|=2e-6), and is **1.54× faster** (7.9→5.1ms)
  on the streaming step. Phase 1 (single-stream compile) is viable + complementary to batching.

## What remains (engineering — for the /implement loop)
The reviewed plan (`PLAN.md`, 3× paired-reviewed, implementation-ready) covers it. **All 3 probes + Step 5
primitives + Step 0 baseline are DONE; the de-risking is complete.** Remaining:
- **Step 4**: wire the encoder compile into server.py behind `NEMOTRON_ENCODER_COMPILE` (B=1 only).
- **Steps 5(wire)-11**: in-server batch method using `batch_primitives`; the **scheduler**
  (5a infra B=1 → 5b steady-state batching → 5c variable-B/fail-closed/memory); decoder switch to
  greedy_batch; local validation; Modal re-sweep; consolidate. Steps 6-8 (the scheduler) are the bulk.
- In-server TODOs the probes flagged but didn't cover: the `drop_extra_pre_encoded` `try/finally`
  exception test inside server.py; the encode-vs-decode timing split (for the fallback-ceiling number,
  if greedy_batch had been NO-GO — moot now since Probe C is GO); the multilingual prompted baseline.

## How to continue
`/implement proj-2026-05-21-0410/PLAN.md` — it's a clean /implement-formatted plan (Codex per step + my
review). The two probes (Steps 2,3) are already GO, so the implementer can proceed to Step 4/5 with the
correctness foundation established. Expected payoff (RESULTS.md): knee ~5→~40-50 on Modal (~$0.012/stream
on T4), ~14→higher locally — batching dwarfs GPU choice.

## Resource state
Local GPU free (probes exited). Modal apps stopped (nemotron-asr-bench, nemotron-asr-profile). Nothing
billing. Probe scripts + baseline artifact are uncommitted files in this dir (not committed — no commit
was requested).

## The companion cost study
`proj-2026-05-20-modal-cost/RESULTS.md` is complete: full per-GPU knee/$ table, the decisive finding
(Modal ~5 vs local ~14 is the virtualized-cloud ENVIRONMENT not GPU-arch — proven by the same Blackwell
silicon RTX-PRO-6000 kneeing at ~5-6 on Modal), self-host-vs-Modal economics, and the recommendation
(cheapest GPU on Modal; batching is the real ~10× lever).
