<task>
**Adversarial review of PHASE2-PLAN.md v4** (just committed). The v4 banner folds 10 insights from the
L40S knee sweep (≥N=64 SLO-robust with B_max=4 W=0 L=0) into actionable plan updates: 2 one-line defaults
updates, lift-shape implications for production setpoints, Step 3 sizing rules, and 4 B3 measurement
follow-ups. **Your job: skeptical second-look on the AMENDMENT — find missing items, mis-categorized
insights, defaults that should NOT change, or measurement debt I'm waving away.**

Write your fold-ready review to `proj-2026-05-24-from-scratch-runtime/reviews/codex-plan-v4-review.md`.
</task>

<context>
**Files to read (in this order)**:
1. `proj-2026-05-24-from-scratch-runtime/PHASE2-PLAN.md` — the v4 banner (after the v3 block), the
   updated Step 2 + Step 3 step bodies (with v4 follow-ups inline), the new "B3 measurement follow-ups"
   section before `## Rules`, and the updated progress table rows for Step 2 + Step 3.

**Context** (read selectively):
- `reviews/B3-5090-lowload-result.md` + `reviews/B3-5090-result.md` — the 5090 data underlying insights
  4 (dispatcher CPU%) + 9 (memory).
- `reviews/Step2a-invariant-design.md` + `reviews/Step3-scoping.md` — the Step 2/3 scoping the plan v4
  references.
- The L40S sweep is still in flight (codex job `bd7rd0m6n`). The plan v4 is based on its in-progress data
  ("`B_max=4, W=0, L=0` reached the top of the registered N axis N=64 with zero token divergences"); the
  exact numbers across all cells are not yet finalized. **Don't re-derive the L40S numbers from raw logs;
  trust the v4 banner's framing for the lift, but DO challenge whether the insights are correctly
  inferred from that framing.**

**The 10 insights folded into v4** (re-stated for review):

*Defaults that are now wrong (Step 2 v4 follow-ups — bundle with B3 verdict commit):*
1. Admission `active_cap` default of 40 is stale; change to deploy-required-no-default or 64+.
2. Scheduler default `window_ms=10` → `0` (matches L40S production winner W=0 L=0).

*Lift shape:*
3. B_max=2 barely lifts (+4 over OFF=36); B_max=4 lifts heavily (≥+28). B_max=4 is the right production
   default (already); B_max=2 is debug-only.
4. Known ceiling at ~N=80-100 = dispatcher single-thread saturates. Multi-dispatcher = Tier-4 future-work.

*Step 3 sizing rules:*
5. Worker thread pool must support 64+ concurrent streams (size for ~80-100, not 40).
6. WS-tail microbench (Step 3a RUN) should be exercised up to N=128 (user-set, bumped from N~80).

*B3 measurement follow-ups (new section):*
7. B3-FU-1: bracket the true L40S knee at N ∈ {72, 80, 88, 96, 112, 128}.
8. B3-FU-2: burst-injection at N=64 per §II.13.
9. B3-FU-3: L40S Tier-3 memory baseline (capture during B3-FU-1).
10. B3-FU-4: apply the 2 one-line defaults updates with the B3 verdict commit.

*Confirmations (no action):*
Tier 3 shrink fits L40S nicely; W=0 L=0 + B_max=4 = simplest config wins.

ASK / structure the review:
1. **Defaults changes (insights 1+2)**: are they correctly inferred from the data? Specifically:
   - Insight 1: `active_cap=40` → deploy-required OR 64+. Is "deploy-required-no-default" the right risk
     stance, or should we ship a sensible default that ops can override? Are there cases where 40 is the
     right default (smaller GPU targets, multi-tenancy, etc.)?
   - Insight 2: `window_ms=10` → `0`. The W=0 won on L40S at the operating point — but does W=0 still
     win at LOW load? The low-load 5090 sweep used W=10 L=0 and got break-even at N≈4 with +0.3ms
     wrapper. If W=0 vs W=10 differs at low N, the default flip might regress N=1-8 behavior.
2. **Lift shape (insights 3+4)**:
   - Insight 3: "B_max=2 barely lifts" — based on a single data point (B_max=2 W=10 L=0 N=40 pass, N=44
     fail). Is one data point enough to conclude "B_max=2 is debug-only" + skip future sweep cycles?
   - Insight 4: "known ceiling at ~N=80-100" is an extrapolation from 5090 dispatcher CPU% (52% at N=40
     B_max=4); extrapolation crosses 100% near N=80. Is that extrapolation safe? L40S has different
     dispatcher CPU dynamics (different per-call work, different stream characteristics) — should the
     ceiling claim be hedged?
3. **Step 3 sizing rules (insights 5+6)**:
   - Insight 5: "worker thread pool size for 80-100" — should it also call out memory headroom
     (Tier 3's +5 GiB scheduler footprint + per-stream ~35MB activation × 100 = ~9 GiB — still fits L40S
     48 GiB; flag explicitly for the multi-process MPS case where multiple instances compete for memory)?
   - Insight 6: N=128 WS-tail target — chose 128 to cover headroom. Is the sweep matrix `n_idle × m_streaming`
     well-designed? Are there pathological combinations missing (e.g., n_idle=128 m_streaming=0 to
     measure idle-socket overhead alone)?
4. **B3 follow-ups (insights 7-10)**:
   - B3-FU-1: bracket the knee at {72, 80, 88, 96, 112, 128}. Should it include a higher upper bound
     (e.g., 160, 192) in case the true knee is beyond 128 too? Sweep cost grows with N (per-cell run
     time scales).
   - B3-FU-2 burst-injection: is the N=64 burst-injection point right, or should it be at the true knee
     N (TBD by B3-FU-1)?
   - B3-FU-3 memory baseline: should it also capture per-stream activation explicit measurement (the
     "0.035 GiB/stream" from earlier Step 1b was a single estimate)?
   - B3-FU-4 defaults bundled with B3 verdict — is the bundling right, or should it be a separate commit
     for cleaner history?
5. **Missing insights / things the plan doesn't capture**:
   - Anything from the L40S sweep that should be folded but isn't?
   - Any inconsistency between the v4 banner and the existing v3 spec / B2 verdict / F1 funding gate?
   - The plan still cites pre-batching multipliers in Step 1b (~1.8-2.25× Python at S_py~16-20); should
     the v4 banner explicitly UPDATE that to the realized 3.2× ratio?
6. **Net verdict**: GO (the v4 amendment is faithful + complete), GO-with-changes (specific edits to fold
   before next commits), or HOLD (a real flaw needs addressing).

Write your fold-ready review. Adversarial, specific, terse where possible.
</context>

<verification_loop>
This is a doc/plan review, NO BUILD, NO RUN. Read the v4 amendment sections (banner + Step 2/3 + B3
follow-ups + progress table) + ground checks against the 5090 data. Bounded.
</verification_loop>

<action_safety>
Write only the review doc. Do not modify the plan or any other files. Fold decisions go through me after.
</action_safety>

<compact_output_contract>
Report path of the review doc + one-paragraph verdict summary (GO / GO-with-changes / HOLD) + the top 1-2
items to fold into the next plan amendment if any.
</compact_output_contract>
