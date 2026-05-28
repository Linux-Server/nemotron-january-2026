<task>
**Step 2a invariant work** — admission + stale-gen + telemetry extensions per the binding design at
`reviews/Step2a-invariant-design.md`. Bounded, single-step delivery; new files mostly (no overlap with Tier
3 / Tier 2 work). Single Opus review post-implementation (paired-review overkill for this bounded primitive
work; Step 3b will get the full paired treatment).
</task>

<context>
**The binding design**: `reviews/Step2a-invariant-design.md` §I-VII. Read it in full; §I admission, §II
stale-gen, §III telemetry schema, §IV WS-tail skeleton, §V priority-finalize-telemetry, §VI smoke tests,
§VII sequencing.

**Code surface (per §VII)**:
- NEW: `runtime/cpp/density_admission.{h,cpp}` — `DensityAdmission` class with active_cap +
  backlog_cap + atomic counters + try_admit + on_admit_complete + on_close + telemetry_snapshot. Thread-
  safe. CLI flags `--admission-active-cap`, `--admission-backlog-cap`. Env-var fallback
  `NEMOTRON_DENSITY_ADMISSION_{ACTIVE,BACKLOG}_CAP`.
- NEW: `runtime/cpp/ws_tail_microbench.cpp` — empty `main()` stub. Real impl is Step 3a.
- MODIFY: `runtime/cpp/density_main.cpp`:
  - Extend `SessionState` with `std::atomic<uint64_t> generation{0}`.
  - Extend the work-item types used by the scheduler (or session-internal queues) with `generation` field;
    add the check helper.
  - Wire admission into b2-t1 and density-sweep mode startup (a shim that calls `try_admit` per session
    construction; counts shed/admit).
  - Telemetry emit: extend the existing JSON output with `admission` + `stale_gen` + (placeholder) `ws_tail`
    blocks per §III schema. The placeholder ws_tail block emits `null` until Step 3a fills it.
- MODIFY: `runtime/cpp/CMakeLists.txt` — add density_admission.cpp + the stub binary target.

**Out of scope** (Step 3a or later):
- The real WS-tail microbench implementation (just the stub binary + JSON schema for now).
- The real WS server (Step 3b — explicitly deferred).
- Priority-finalize-lane implementation (Step 2a only adds telemetry to inform the decision; the lane
  itself is Step 2b).

**Smoke tests** (per §VI):
1. **Admission unit test**: a small C++ test (or a `--mode admission-smoke` in density_main) that
   synthesizes 100 try_admit calls + asserts the counters (active_cap=10 → 90 sheds; backlog_cap=5 → after
   the active+backlog fill, more sheds).
2. **Stale-gen unit test**: 4 scenarios from §II — close-while-inflight, reset-while-queued, reset-while-
   finalizer, final-after-shed. Use the b2-t1 mode harness with injected close/reset events.
3. **Telemetry emit smoke**: run b2-t1 4-row + a density-sweep --smoke N=4; verify JSON sidecars have the
   new blocks with sensible values (active=0/1/etc., shed=0, stale_drops=0 for happy-path runs).

**Container build clean. Don't disrupt the L40S sweep (different box).**
</context>

<verification_loop>
Build in container. Run the unit tests above + b2-t1 4-row + density-sweep N=4 OFF smoke (the new
telemetry blocks should appear in the JSON; admission counts admitted vs offered; stale_gen counters at 0
for happy paths).
</verification_loop>

<action_safety>
Local only. Don't touch the scheduler / primitive / manifest (Tier 3's territory). Don't change the WS
server (Step 3b deferred). The admission + stale-gen modules are standalone primitives.
</action_safety>

<compact_output_contract>
When done, report:
1. Files created/modified + line counts.
2. Admission unit test result (shed counts match expectations).
3. Stale-gen unit test result (4 scenarios, 0 stale events emitted).
4. b2-t1 + OFF smoke results with the new telemetry blocks (sample JSON snippet for each).
5. Anything you couldn't implement as specified + why.
</compact_output_contract>
