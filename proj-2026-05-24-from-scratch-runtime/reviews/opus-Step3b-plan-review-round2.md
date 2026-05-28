# STEP3B-WS-PLAN v2 — Opus Round 2 adversarial review (2026-05-28)

Reviewing v2 (committed `b5e4bd0`) from-scratch. Builds on Round 1; doesn't re-flag what's folded.
Stop condition check: does v2 reduce to minor-only?

## Verdict

**MINOR_ONLY → CONVERGED.** v2 folded Round 1's must-folds cleanly + restored the v4 contract
items. Remaining items are bounded clarifications that fold during implementation.

## What v2 did right

1. **Step 11 commit semantics** — clear and executable. Failure = integration regression (most
   likely Step 9), fix it via Step 9, don't un-mark prior steps. Resolves the circular dependency
   cleanly.
2. **Part A v1 salvage audit gate inline in Step 3** — explicit hunk-by-hunk classification +
   `reviews/part-a-v1-salvage-audit.md` deliverable. No ambiguity.
3. **Step 4 wrapper-equivalence harness** — concrete: real PCM through SessionRuntime + 4-row
   fixture + finalize_ref equality. Catches regressions a synthetic-frame smoke would miss.
4. **Step 5 admission Python-shape** — names the exact fields v4 §IV requires.
5. **Step 7 additions** — HTTP admin handler pool size/queue concrete; /health Python enum unified;
   pid/process_label; MPS-readiness selftest row; grouped operator config table.
6. **Step 8 Silero N=64 CPU probe** — concrete bar with measurement procedure.
7. **Step 9 lifecycle oracle** — 10 enumerated cases; concrete + executable.
8. **Step 11 WS-overhead perf gate** — quantified formula (max(2ms, 10%·baseline)).
9. **"Bars are ADDITIVE to global per-step protocol"** header — kills the "replaces vs augments"
   ambiguity Codex Round 1 flagged.

## Minor items (acceptable to fold during implementation)

### 1. Admission `signal{}` field mapping detail
Step 5 lists Python-shape fields including `signal{}`. The Python signal sub-object's exact keys/
shape isn't pinned in v2 — Step 1 audit will fill it in. Acceptable: Step 1 audit drives the
mapping, Step 5's implementation consumes the audit. Document in the implementation review.

### 2. Silero CPU threshold refinement
Step 8's bar says "FAIL if VAD path consumes enough CPU to threaten WS worker scheduling (e.g.,
> 50% of one core sustained)." At N=64 on g6e.8xlarge (32 vCPU), Silero on a bounded thread pool of
2 means total Silero CPU ≤ 2 cores sustained even at saturation = ~6% of the box. The "> 50% of one
core" threshold is permissive but reasonable for an early-warning signal. Implementation can refine
to "≤ pool_size × 0.7" if needed during execution. Defer.

### 3. Python server launch spec in Step 11
Step 11 says "Setup: Python server on port 8080, C++ ws_server on port 8081" but not HOW the
Python server is launched. Likely: `python -m nemotron_speech.server --port 8080 <args>` as a
subprocess managed by `run_compat.py`. Defer to Step 11 implementation; documented as part of
`run_compat.py`'s setup code.

### 4. Step 1 audit deliverable bar
Step 1's success criterion is "Output drives all subsequent steps' contract decisions" — not
strictly falsifiable. Implementation can add a concrete bar: `reviews/server-py-protocol-audit.md`
exists with sections for HTTP routes / WS handshake / control messages / WS close codes / VAD
trigger / finalize_timing keys / error frame format; each section cites
`src/nemotron_speech/server.py:line` references. Trivial fold.

### 5. WS-overhead perf gate at N=64
Step 11 measures at N=8. The production-shape ttfs at N=64 is the B3-FU-1 baseline. Step 11 could
add a N=64 gate too, but the N=8 measurement is sufficient for "no regression from scheduler-ON
margin" since the per-stream overhead is concurrency-independent (mostly WS framing + network).
Defer — N=8 covers it.

## Cross-check (did v2 introduce contradictions?)

- "Bars-additive" header vs each step's bar: cross-checked. Every step is compatible (bars add NEW
  assertions; smoke set always re-runs).
- Step 3's salvage audit vs Step 0 (Opus Round 1 suggested separate step) — v2 chose to inline in
  Step 3, consistent with Codex Round 1's "narrow scope, no separate Step 0." No contradiction.
- Step 11 commit semantics ("Steps 8-10 stay [x] on per-step bars") vs the global rule "build →
  test → mark [x]" — consistent. Step 11 is its own step with its own bar; failure is integration
  regression to fix.

No reopened Round 1 folds.

## What HOLDS (no change needed)

- 11-step decomposition.
- Step ordering / dependencies.
- PLAN_RULES.md compliance (with v2's explicit bars-additive header).
- Architecture v4 binding (v2 restored the missing contract items).
- Reference implementations cited correctly.

## Net

**MINOR_ONLY → CONVERGED at Round 2.** v2 is ready for `/implement`. The 5 minor items above can
fold during implementation as planned-in clarifications (not blocking).

**Recommend** declaring convergence if Codex Round 2 also returns MINOR_ONLY. If Codex finds new
must-folds, do Round 3; but my expectation is convergence — the 3 Round 1 must-folds + the v4
contract restorations were the real gaps.
