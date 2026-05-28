# STEP3B-WS-PLAN v3 — Opus Round 3 adversarial review (2026-05-28)

Reviewing v3 (committed `85a73c7`) from-scratch. Stop-condition check: did v3's tight fold of Codex
Round 2's must-fold land + does v3 reduce to minor-only?

## Verdict

**MINOR_ONLY → CONVERGED.** v3 lands the StatsCollector ownership/order fix correctly + the 3
minor cleanups. Ready for `/implement`.

## What v3 did right

1. **StatsCollector ownership/order pinned correctly.**
   - Step 5 prose explicitly says "`SessionRuntime::finalize_now()` does NOT call `record()`" +
     "Step 5's bar tests StatsCollector directly via `--mode stats-smoke`; the production-
     integration recording is Step 9's bar."
   - Step 9 prose now has the correct sequence: `finalize_now()` → produce WireEvent +
     `last_timing()` → stale-gen check → emit final (or drop) → stamp `was_suppressed` + `emitted`
     → `record(timing, emitted)` exactly once → close WS-1000.
   - The "exactly once" wording + the AFTER-emit-decision ordering both match v4 §V.
2. **Step 9 odd-length PCM WS-1003** added inline to the lifecycle prose — at the right layer
   (PCMFrame construction in the recv-loop), not premature at framing layer (Step 6).
3. **Bars-additive header Step 1 exemption** documented in the v3 fold-notes block (Step 1 is
   markdown-only; global protocol applies only when a build target is affected).

## Minor items (acceptable to fold during implementation)

### 1. Step 11 port-configurability not inlined into body
The v3 fold-notes header says ports are "configurable" but Step 11's prose still says "Python
server on port 8080, C++ ws_server on port 8081" without explicitly noting `--python-port` /
`--cpp-port` flags. The implementer will likely pick this up from the fold-notes + add flags
naturally. If they don't, anti-flake matters less than correctness.

### 2. Step 6 doesn't reference the odd-length WS-1003 rule
Codex Round 2 said "Step 6 OR Step 9." v3 chose Step 9 (correct — framing layer doesn't know PCM
semantics). Step 6's `lib/ws/framing` returns the binary frame bytes; Step 9's recv-loop validates
odd-length + closes WS-1003 before constructing PCMFrame. The split is right. No action needed,
but Step 6's bar could include "binary frame passes through framing layer unchanged; odd-length
validation is Step 9's responsibility" for clarity.

### 3. Bars-additive header Step 1 exemption isn't in the actual header text
The v3 fold-notes block says it but the in-body header `**Important rule (v2 fold ...)**`
predates v3. The implementer reading Steps in isolation might miss the exemption. Fix during
implementation: when Step 1 runs, the implementer notes "Step 1 is markdown-only, global protocol
N/A." Or fold inline as a tiny v4 edit.

## Cross-check (did v3 reopen Rounds 1-2 folds?)

- Step 11 commit semantics (Round 1 fold #1): unchanged. ✓
- Part A v1 salvage audit gate (Round 1 fold #2): unchanged. ✓
- Step 4 wrapper-equivalence harness (Round 1 fold #3a): unchanged. ✓
- Step 9 lifecycle oracle enumeration (Round 1 fold #3b): unchanged. ✓
- Step 11 bars + WS-overhead gate (Round 1 fold #3c): unchanged. ✓
- v4 contract restorations (Codex Round 1 #2): unchanged. ✓
- Step 5 admission Python-shape: unchanged + now correctly without record() wiring. ✓
- Step 7 v4 §XI/§XIII/§XVI additions: unchanged. ✓
- Step 8 Silero N=64 CPU probe: unchanged. ✓
- "Bars-additive" header (Opus Round 1 #2): unchanged + v3 adds Step 1 exemption in fold-notes.
- StatsCollector ownership (NEW from Round 2): folded correctly.

No reopened folds.

## What HOLDS

- 11-step decomposition; ordering; review-intensity tags.
- Reference implementations + PLAN_RULES.md compliance.
- Architecture v4 binding with all contract restorations.
- Every step has a concrete, executable bar after Round 1+2+3 folds.

## Net

**v3 = CONVERGED at Round 3.** Both reviewers expected to return MINOR_ONLY. Ready for
`/implement`.

After Codex Round 3 confirms convergence: declare paired-convergence + relaunch
`/implement STEP3B-WS-PLAN.md` (after in-flight Part A v1 + the L40S B3-FU sweeps land + are
audited; the implement loop iterates the 11 steps).

The 5-round budget had slack (converged at Round 3, 2 rounds under budget). The Round 1 fold was
substantive (3 must-folds); Round 2 found 1 specific bug (StatsCollector ownership); Round 3 was
the verification round.
