# STEP3B-WS-PLAN v1 — Opus Round 1 adversarial review (2026-05-28)

Reviewing `STEP3B-WS-PLAN.md` v1 (committed `c266771`) from-scratch, adversarially. Folded with the
parallel Codex Round 1 review after both land.

## Verdict (preview)

**GO-with-3-must-folds-to-v2.** The plan structure mirrors PHASE2-PLAN.md correctly + the 11 steps
are right-sized + the binding-to-v4-architecture is sound. Three specific issues need fixing:
(1) the Step 11 pre-merge gate creates a circular commit dependency; (2) the in-flight Part A v1
salvage is hand-wavy with no audit gate; (3) bars on Steps 4, 9, 11 are not concrete enough to be
falsifiable. Plus 5 medium and a few minor items.

## Must-folds to v2

### 1. Step 11 "pre-merge gate" creates circular commit dependency
Step 11's Bar says: "Steps WS-B1...WS-B5 are not committed-as-`[x]` until this oracle PASSes for
them in combination." But the implement-loop pattern is per-step-commit-and-mark. If Step 9 lands +
commits + marks `[x]`, then Step 11 fails the oracle on Step 9's behavior, we have an `[x]` step
that's actually broken. Either:
- Steps 8-10 stay `[~]` until Step 11 passes (delays their commit signal), OR
- Steps 8-10 mark `[x]` on their OWN per-step bars (smoke tests) + Step 11 is the INTEGRATION test
  whose failure is documented as an integration regression to fix, not as ungrading prior steps.

**Fold action**: pick the second framing. Step 11's bar = "C++ server's wire output matches Python's
canonicalized output on utt0..utt7." If it fails, Step 11 stays `[!]` blocked + the root cause is
investigated (most likely Step 9's lifecycle wiring). Don't retroactively un-mark Steps 8-10.

### 2. In-flight Part A v1 salvage needs an explicit audit gate
Plan says "Discard the v1 ws_server.cpp (from in-flight Part A `bdajesege`); start from v4. ...
salvage mechanical CMake/file-move pieces if useful." There's no concrete audit step that decides
"useful" or "discard." Risk: Step 3 starts before in-flight v1 commits; race conditions on
CMakeLists / density_main.

**Fold action**: insert **Step 0** = "Audit in-flight Part A v1 commit (when it lands); produce
`reviews/part-a-v1-audit.md` deciding keep/discard per file." Step 3 doesn't start until Step 0
commits. (If in-flight v1 fails to land cleanly, Step 0 documents what artifacts exist + Step 3
starts from clean v4-driven slate.)

### 3. Bars on Steps 4, 9, 11 aren't concrete enough
- **Step 4 bar**: "b2-t1 4-row still PASS" — but the public-API wrappers might pass b2-t1 while
  silently leaking state. Add: `grep -r "static [^ ]* [a-z]" lib/session/ | wc -l` documented +
  the static-global transfer audit (the `reviews/session-cpp-static-globals.md` Codex's review may
  also flag).
- **Step 9 bar**: "asserts the correct event sequence" — too hand-wavy. Concretize: `--mode
  ws-lifecycle-smoke` runs N=2 sessions; asserts exactly `[{"type":"ready"}, {"type":"transcript",
  "is_final":false, "text":"<...>"}, ..., {"type":"transcript","is_final":true, ...}]` per session
  + clean WS-1000 close. Reuse the existing density harness's event-equality machinery.
- **Step 11 bar**: "8 utts PASS canonicalized diff" — for each utt: assert event count equal,
  per-event `type` equal, per-event `text` equal, per-event `is_final` equal, per-event `finalize`
  flag equal, final `collector_text` equal. Volatile fields stripped explicitly listed:
  timestamps, finalize_timing values, sequence_ids, pid, process_label, native scheduler counters.

## Medium folds (recommend, not strictly blocking)

### 4. Static-global audit deserves a concrete artifact
Step 3 mentions the audit but doesn't produce a deliverable. **Fold**: Step 3's bar includes
producing `reviews/session-cpp-static-globals.md` listing every static identified + the disposition
(stays in `session.cpp`, transferred to `SharedRuntime` in Step 4, irrelevant). Step 4 consumes this
list as its checklist.

### 5. Step 2's Option A vs B is hedged
Plan says "Recommend Option A" but lists both. Pick A definitively; Option B is a fallback if
Dockerfile rebuild is blocked by the CI pipeline. Saves a decision-point that should already be made.

### 6. WS-overhead performance gate is missing
v4 §IV implicitly says WS overhead p95 ≤ 10% of TTFS. The plan has no step that VERIFIES this for
the production C++ WS server. Add to Step 11 (or new Step 12): measure `ttfs_via_ws_server` vs
`ttfs_via_density_main_with_scheduler` at a fixed concurrency (e.g., N=8 from the low-load sweep);
verify ws_overhead = ttfs_via_ws - ttfs_via_density-scheduler ≤ ~2-3 ms or ≤ 10% of TTFS.

### 7. Smoke set should include N=200 session gate per PLAN_RULES.md
PLAN_RULES.md test protocol step 3 says "Re-run the existing N=200 session gate
(cpp/session_main) to confirm no regression." The plan mentions the smoke set (b2-t1, density-sweep
N=4 OFF, stalegen-smoke, admission-smoke, stats-smoke) but doesn't explicitly call out the N=200
gate. Add to Steps 3, 4 (the refactor-impact steps) explicitly.

### 8. Multi-process MPS smoke is missing
v4 §XI/§XIII MPS-readiness. The plan has no step that verifies 2 ws_server processes on different
ports can both bind + both /health up. Add a row to Step 7's selftest matrix: "2 ws_server
instances on ports 8081 + 8082 simultaneously, both /health PASS."

## Minor folds (acceptable to defer or fold inline)

### 9. Step 6 "smoke binary OR --mode" — pick one
"Add a small `lib/ws/ws_lib_smoke` binary OR `--mode ws-lib-smoke` in density_main." Pick: small
standalone binary `ws_lib_smoke.cpp` linking only `nemotron_runtime` — proves the library can be
linked without dragging density_main's dep tree.

### 10. Step 11 Python server setup
The oracle needs a running Python server (port 8080) + a running C++ ws_server (port 8081). Spec
how the oracle sets up the Python side: spawn `python -m nemotron_speech.server --port 8080
<args>` as a subprocess. Tear down both at end.

### 11. Step 8 finalization signal logging
When server-side Silero triggers finalize, log the trigger reason + timing. Useful for operator
debugging post-deploy (the kind of thing the /stats `close_reason` distribution would surface).

### 12. Step 10 ping/pong env defaults
v4 §IX says defaults `NEMOTRON_WS_PING_INTERVAL_SEC=60` + `NEMOTRON_WS_PONG_TIMEOUT_SEC=30`.
Confirmed in the plan but worth verifying against the audit (Python server might use different
defaults; match Python).

## What HOLDS in the plan

- **11-step decomposition** — right size; each step is committable independently.
- **Step ordering** — Step 1 (audit) → 2-7 (foundation) → 8-10 (lifecycle) → 11 (oracle); deps
  resolve correctly.
- **Step boundaries** — each step has a "Bar" + a "Key files" list (per skill format).
- **Reference implementations** — Python server.py + /stats PRs + ws_tail_microbench correctly cited.
- **Rules section** — PLAN_RULES.md folded + 7 plan-specific rules added.
- **Decision-critical review tags** — Steps 1, 3, 6, 8, 9, 11 marked PAIRED; others Opus.
- **Architecture v4 binding** — each step cites the right § from v4.

## Net

**GO-with-3-must-folds-to-v2 + ~5 medium folds + ~4 minor.** Fold the 3 must-folds (Step 11 commit
semantics; Step 0 audit gate; concrete bars on Steps 4, 9, 11) + the 5 mediums; defer the minors
to v3 or fold inline. After Round 1 fold → v2 → Round 2 paired review.

If Codex Round 1 surfaces structural issues I missed, those get folded into v2 too.
