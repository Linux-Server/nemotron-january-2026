<task>
**Step 3c — stale-generation validation via injection.** The "0 stale/mismatch" gate from PHASE2-PLAN.md
Step 3, validated without needing the real WS server (Step 3b is deferred). Bounded test scaffolding
that exercises the Step 2a stale-gen primitive in the existing density-sweep / b2-t1 modes via injected
close/reset/shed events.
</task>

<context>
**Pre-requisite**: Step 2a invariant work committed (provides the per-session generation counter +
check helper + the `stale_gen` telemetry block). This task BUILDS on Step 2a.

**The 4 scenarios** (per Step2a-invariant-design.md §II):
1. **close-while-inflight**: a session is closed while a finalize is being computed. Expected: 0
   FINAL events emitted; stale_drops_at_finalize_output++.
2. **reset-while-queued**: a session is reset while N work items are queued behind it. Expected:
   N stale_drops counted (one per stage where the queued work would have produced output).
3. **reset-while-finalizer-owns-runner**: a session is reset while the finalizer holds a runner.
   Expected: finalize completes but its FINAL event is suppressed (stale_drops_at_event_emit++).
4. **final-after-shed**: a session is shed mid-processing. Expected: no further events; stale_drops
   counted at the suppression points.

**Implementation approach**:
- Add a new mode `--mode stalegen-smoke` (or extend density-sweep with `--inject-stalegen-scenarios`).
- The mode spins up N=2 or 4 sessions, runs a few chunks per session, then triggers each scenario
  programmatically at known points:
  - Inject a `session.close()` call mid-processing.
  - Inject a `session.reset()` while work is queued.
  - Inject a `session.shed()` after admission completes.
- Assert post-condition: zero spurious events emitted (token sequences match expected truncated
  output); stale_drops counters increment exactly as expected per scenario.
- The "real WS" version of these triggers (in Step 3b) maps WS-close, WS-reset frames, admission-shed
  responses to the same internal `session.close/reset/shed()` calls; this test exercises the internal
  mechanism.

**Validation**:
- Container build clean.
- Run the new mode: all 4 scenarios PASS (0 spurious events; expected stale_drops counts).
- Run b2-t1 + density-sweep regression: NO stale_drops in happy-path runs (counter stays at 0).

**Code surface**:
- MODIFY: `runtime/cpp/density_main.cpp` — new mode dispatch, the 4 test scenarios, the injection
  helpers (`force_session_close(s)`, `force_session_reset(s)`, `force_session_shed(s)` that bump the
  generation counter).
- NO new files (keep this bounded; extend density_main).

**Out of scope**:
- Real WS server (Step 3b).
- Multi-turn lifecycle handling (Step 3b).
</context>

<verification_loop>
Container build. Run `--mode stalegen-smoke` + assert all 4 PASS. Re-run b2-t1 4-row + density-sweep N=4
to confirm no regression (stale_drops counter stays at 0 in happy paths).
</verification_loop>

<action_safety>
Local only. Don't disrupt the L40S sweep. Pre-requisite: Step 2a must be committed first.
</action_safety>

<compact_output_contract>
When done, report:
1. Files modified + line counts.
2. The 4 scenarios result (per-scenario stale_drops count + 0 spurious events).
3. b2-t1 + OFF smoke regression (stale_drops=0).
4. Code-snippet of the new mode entry + the injection helpers.
</compact_output_contract>
