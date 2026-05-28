# Codex Step 3b Plan Review - Round 4

Verdict: **MINOR_ONLY = CONVERGED**. v4's intended one-line fold landed cleanly: Step 5's key-files
parenthetical now instructs `runtime.cpp` to populate `SessionTiming` / `last_timing()` only, with
**no `StatsCollector::record()` call**, and explicitly says the WS worker owns `record()` in Step 9.
That matches Step 5's ownership prose and Step 9's lifecycle ordering. I do not see any remaining
Step 5 / Step 9 / cross-step contradiction that would force an implementer to ask before starting.

## Fold Check

- **Step 5 ownership vs key files:** consistent. `SessionRuntime::finalize_now()` returns the
  `WireEvent`, populates timing, and does not record. Step 5 still builds/tests `StatsCollector`
  directly via `--mode stats-smoke`.
- **Step 9 lifecycle vs Step 5:** consistent. The WS worker performs stale-gen check, final send/drop
  decision, stamps `was_suppressed` + `emitted`, then calls `StatsCollector::record(timing, emitted)`
  exactly once after the emit decision. That preserves the v4 architecture's final-suppression
  accounting.
- **No stale "wire record call" instruction remains:** searching the plan only finds the historical
  fold note, the explicit "no record()" Step 5 key-files fix, and the intended Step 9 production
  integration.

## Minor Only

1. **Step 11 port wording remains slightly split.** The v3/v4 fold notes say ports are configurable /
   free-allocated with default 8080/8081, while Step 11's body still says "Python server on port 8080,
   C++ ws_server on port 8081." This is an anti-flake implementation detail, not a plan blocker:
   `run_compat.py` can expose `--python-port` / `--cpp-port` or allocate free ports while keeping
   those defaults.
2. **Step 6 odd-length PCM note is not in the Step 6 bar.** Step 9 correctly owns the semantic
   validation and says odd-length binary payloads close WS-1003 before `PCMFrame` construction.
   Step 6's framing layer can remain byte-oriented; an implementation note or smoke assertion there
   would be clarifying only.

## Net

**MINOR_ONLY = CONVERGED.** The Round 3 must-fold is resolved, and the remaining items are bounded
implementation clarifications. The plan is ready for `/implement` under the stated stop condition.
