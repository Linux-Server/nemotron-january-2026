<task>
**Round 1 adversarial review of `STEP3B-WS-PLAN.md`** (v1, just committed). Mirrors the design-doc
review pattern: paired Codex + Opus reviews, fold to v2, continue until both reviews return only
minor improvements that fold during implementation.

Write your fold-ready review to `reviews/codex-Step3b-plan-review-round1.md`. Adversarial, specific,
terse.
</task>

<context>
**The plan under review**: `proj-2026-05-24-from-scratch-runtime/STEP3B-WS-PLAN.md` v1.
- 11 steps spanning Part A (foundational refactor + library carve + skeleton) and Part B
  (lifecycle, VAD, /stats wiring, shutdown, oracle).
- Binds to `reviews/Step3b-WS-architecture.md` v4 (CONVERGED post 4-round design review;
  commits `8b7e783` → `0e58e46` → `5e83a44` → `a7756d5` → `22df3d1`).
- Marks decision-critical steps for PAIRED review per `PLAN_RULES.md`.
- 11-step Progress table with paired/Opus-review tags.

**Related context to read**:
- `PHASE2-PLAN.md` — the format reference (the project's main plan doc this mirrors).
- `PLAN_RULES.md` — project-wide rules pulled into the plan.
- `reviews/Step3b-WS-architecture.md` v4 — the binding design.
- `reviews/codex-Step3b-design-round{1..4}.md` + `opus-Step3b-design-round{1..4}.md` for context on
  what's already CONVERGED in the design.

**ASK / structure your review around**:

1. **Step decomposition**: are the 11 steps right-sized? Any step too big (should split)? Any
   too small (should merge)? Are step boundaries committable (each leaves project in working
   state)?

2. **Step ordering / dependencies**: are dependencies resolved before dependents? Specifically:
   - Step 1 (audit) precedes 2-11 (audit drives contract decisions). ✓?
   - Step 3 (library carve) precedes 4-7 (which build in lib/). ✓?
   - Step 6 (lib/ws) precedes 7 (ws_server.cpp). ✓?
   - Step 8 (Silero) and 9 (lifecycle) — should Silero come AFTER lifecycle (since lifecycle
     defines where Silero triggers)? Or BEFORE (since lifecycle needs Silero's trigger)?
   - Step 11 (oracle) is a Part B pre-merge gate — does that mean Steps 8-10 are not
     COMMIT-as-`[x]` until 11 passes? Reconcile with the per-step commit rule.

3. **Risk + de-risking**: are there steps that should be PROBED first (diagnostic / research)
   before committing to an implementation approach? Specifically:
   - Static-global audit (in Step 3): does it warrant its own diagnostic step before the carve?
   - Silero CPU thread-cap at N=64: does this need an early measurement step (e.g., probe Silero
     per-frame cost on the target box) before committing to the integration in Step 8?

4. **Bars / gates**: each step has a "Bar" sentence. Are they concrete enough to be testable?
   Falsifiable? Especially:
   - Step 4's "b2-t1 4-row still PASS" — is that strict enough to catch a regression in the
     public-API wrappers?
   - Step 9's "asserts the correct event sequence" — too hand-wavy; concrete oracle?
   - Step 11's "8 utts PASS canonicalized diff" — what's the diff tolerance for finalize_timing?

5. **PLAN_RULES.md compliance**: the plan claims to follow them. Specifically:
   - Per-step test protocol (build → harness → existing N=200 + B2+Tier3 smoke set re-run).
   - PAIRED vs Opus review intensity by step.
   - Honesty / no-loosening clause.
   Are any steps missing a re-run of the existing smoke set?

6. **Architecture v4 alignment**: does each step cite the right § from v4? Any v4 spec items
   missing? Specifically:
   - v4 §X (Dockerfile fix + nlohmann::json) → Step 2.
   - v4 §XII (Part A revised scope) → Steps 3, 4, 5, 6, 7.
   - v4 §VI (Silero) → Step 8.
   - v4 §IX (graceful shutdown) → Step 10.
   - v4 §XIV (test oracle) → Step 11.
   - Anything in v4 NOT mapped to a step?

7. **In-flight Part A v1 (`bdajesege`) salvage**: the plan says "discard v1 ws_server.cpp;
   salvage mechanical CMake/file-move pieces if useful." Specific enough? What audit gate
   confirms "useful"? Should there be a Step 0 = audit Part A v1's output?

8. **What's missing entirely**? E.g.:
   - Performance gate on the WS server (e.g., the WS overhead must not regress the N=64
     scheduler-ON ttfs of 21ms by more than X)?
   - Production deploy steps (systemd unit, ASR.env, RUNBOOK update)? Likely Part C; not in
     this plan; acceptable?
   - Multi-process MPS smoke?

9. **Net verdict**:
   - **MINOR_ONLY = CONVERGED**: only minor improvements that fold during implementation; v1 is
     ready to `/implement`.
   - **GO-with-1-2-must-folds-to-v2**: small specific changes; one more round.
   - **GO-with-substantive-revisions**: substantive restructuring needed.
</context>

<verification_loop>
Doc/plan review only — NO BUILD, NO RUN. Read the plan + the cited related artifacts. Bounded —
the plan is ~250 lines.
</verification_loop>

<action_safety>
Write only the review doc. Do not modify the plan or any code.
</action_safety>

<compact_output_contract>
Report path of the review doc + one-paragraph verdict + top 1-3 must-folds (or "minor only —
converged").
</compact_output_contract>
