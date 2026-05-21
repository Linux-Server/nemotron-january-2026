# Codex review - round 3

## 1. Round-2-fold verification

The round-2 fixes landed.

- Probe A NO-GO now skips Step 4 / Phase 1, not the scheduler (`PLAN.md:110-115`).
- Compile x batch semantics are explicit: compile is B=1-only by default; B>1 batches use the uncompiled encoder unless a separately gated static-bucket probe passes (`PLAN.md:78-83`, `:169-170`).
- The exact two-guard ready predicate is written out: timeline guard plus `len(pending_audio) >= preprocess_new_audio_samples` (`PLAN.md:69-71`).
- `drop_extra_pre_encoded` cleanup is now a hard rule and is covered by Probe B / Step 5 (`PLAN.md:72-73`, `:121-123`, `:141-143`).
- Probe C now requires encode/decode split timing and has the fallback >=1.5x end-to-end gate before pursuing encoder-only batching (`PLAN.md:125-132`).
- Step 7 uses `greedy_batch` only if Probe C GO; otherwise it uses the direct-encoder + current-greedy fallback (`PLAN.md:154-160`).
- Defaults/backpressure are concrete: `MAX_WAIT=5ms`, `MAX_SIZE=4`, deduped ready set, bounded awaited put, no frame drop (`PLAN.md:84-92`, `:146-158`).
- Memory gates, fork-lane ownership, dirty-tree Step-0 identity, quantified Probe A/C/Step 7/Modal thresholds, and Step 9 as local validation are all present (`PLAN.md:97-99`, `:103-108`, `:146-179`).

## 2. Remaining blockers

None. I found no remaining technical error that is likely to silently corrupt transcripts or mislead a gate. The previously dangerous areas now have hard invariants or probes: cache axes, flat unique hypotheses, same-group batching only, no padding/coercion, prompt/model-call serialization, drop-extra restoration, decoder equivalence, compile/static-shape behavior, latency, and memory.

No must-fix PLAN edits before implementation.

## 3. Step-sizing / DAG check

Steps 6/7/8 are large but scoped for one Codex delegation each:

- Step 6 is the concurrency refactor only, still B=1, with scheduler-owned ASR mutation, generation tokens, bounded queues, cancel/close/reset handling, and fork/finalize on the model-call lane.
- Step 7 is the first batched steady-state path only, gated by Probe C and same-group batching.
- Step 8 is the hardening pass: variable-B, join/leave, fail-closed cases, memory, fairness, and telemetry.

The asyncio scheduler risk that would break the current per-session worker/state_lock/event-queue model is called out sufficiently: Step 6 moves audio/control ownership into the scheduler and requires generation tokens plus a single model-call lane. Implementation should treat reset/close/debounce as ordered scheduler work or invalidation barriers, but that follows from the written ownership model and is not a new blocker.

The requested batching DAG has no cycle and is correct: `0 -> {1,2,3}; 1 -> 4; 2 -> 5; 3 -> 7; 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11`. One implicit optional validation edge remains: if Step 4 is implemented, Step 9 must validate it in the compile-only / compile+B=1 cells. Step 9 already names that matrix, so this is bookkeeping, not a blocking missing edge.

Probe GO/NO-GO gates are concrete enough to run autonomously. Failed probes block dependents or stop batching as written.

## 4. Implementation readiness

Implementation-ready: YES.

Must-fix edits if NO: none.

## 5-line summary

1. Round-2 fixes are folded correctly, including Step 4 skip, compile x batch semantics, ready guards, drop-extra cleanup, Probe C split/fallback, defaults, memory, fork lane, and validation.
2. Remaining blockers: none.
3. Steps 6/7/8 are still substantial but properly isolated into B=1 scheduler, steady-state batching, and hardening/telemetry.
4. DAG is acyclic and correct; treat Step 4 -> Step 9 as an optional compile-validation edge when Probe A GO.
5. Verdict: v3 is implementation-ready.
