# Codex Step 3b Plan Review - Round 1

Verdict: **GO-with-1-2-must-folds-to-v2**. The 11-step shape is broadly right and the ordering is
mostly implementable. Do not start `/implement` from v1 unchanged: v2 needs a small but contract-
critical fold that reconciles Part B gating, tightens the executable bars, and restores several
binding v4 details that v1 currently weakens.

## Must Fold Before Implementation

1. **Reconcile Part B commit/review semantics.**
   `STEP3B-WS-PLAN.md:287-288` says WS-B1...WS-B5 are not committed-as-`[x]` until Step 11 passes,
   while `:87` and `:99` require per-step build/test/review before marking `[x]`. This is ambiguous
   enough to cause either over-claiming or blocked incremental commits. Fix by adding explicit
   WS-A/WS-B labels to every step and one rule:
   - implementation commits for Steps 8-10 may land after each step's local bar + required review;
   - progress-table `[x]` and any merge/verdict claim for Part B waits for Step 11 canonicalized
     oracle PASS over the combined Steps 8-10 path.
   Or, alternatively, require Step 11's oracle to run incrementally after each Part B step. Pick one.

2. **Restore v4 contract items dropped or weakened in the step text.**
   These are not optional polish:
   - `/stats` is missing the Python-shaped `admission` object in Step 5 (`:170-178`), despite v4
     section IV requiring it and mapping `DensityAdmission` counters into Python names.
   - `/health` in Step 7 only says `healthy|loading` (`:209-210`), while v4's banner unifies the
     enum to `loading|healthy|draining|degraded`; v4 section XIII also requires `pid` plus optional
     `process_label` in `/health` and `/stats`.
   - v4 section XI's fixed-size HTTP admin handler pool and bounded queue are not mapped. Step 10
     cites section XI but only covers WS send timeout/ping-pong/backpressure (`:255-269`).
   - v4 section XV's selftest matrix has "1 WS handshake + 1 PCM frame + clean close"; Step 7
     reduces that to "1 WS handshake" because lifecycle is not implemented yet (`:213-217`). If that
     deferral is intentional, say Step 7 carries the handshake-only row and Step 9 upgrades the same
     row to PCM+clean-close. Do not silently claim full v4 section XV coverage.

3. **Make the bars executable enough to catch the bugs the steps introduce.**
   - Step 4's bar (`:161-163`) does not prove the public `SessionRuntime` wrappers preserve density
     behavior. `b2-t1 4-row still PASS` can pass while the new wrapper path is wrong, and "1 synthetic
     PCM frame expects one interim emit" is likely a brittle/non-representative assertion. Replace
     with a fixture-driven wrapper equivalence harness: feed real 20ms chunks through
     `SessionRuntime`, call finalize, and assert token/event/delta equality vs `finalize_ref` for the
     same 4-row smoke.
   - Step 9's "asserts the correct event sequence" (`:249-251`) is hand-wavy. Spell the oracle:
     ready exact, binary PCM path, interim transcript ordering, `vad_start` ignored/informational,
     `vad_stop`/Silero finalize path, final transcript shape, stale interim drop, stale final drop
     with `was_suppressed=true`, `StatsCollector::record(timing, emitted)` behavior, and close code.
   - Step 11 should state the exact diff rule in the Bar, not only the body: `finalize_timing` key
     set and value types are compared after Step 1 pins them; numeric values are stripped/not
     compared; JSON keys sorted; only the listed volatile fields are stripped.

## Step Decomposition

The 11 steps are mostly right-sized for commit boundaries:
- Step 1 first is correct; protocol decisions drive every later public contract.
- Step 3 before Steps 4-7 is correct; `libnemotron_runtime` must exist before runtime/ws/server work.
- Step 6 before Step 7 is correct; the server skeleton should consume the library parser/framer.

Two sizing concerns need v2 text, not a full restructure:
- Step 3 is large but acceptable if it is explicitly "mechanical carve only" plus an audited static
  inventory. Add a concrete artifact such as `reviews/session-static-global-audit.md` or a checked
  table in the paired verdict. If the audit finds non-mechanical resource ownership changes, Step 3
  stops and Step 4 absorbs the ownership transfer.
- Step 4 is the highest risk local refactor. Treat the wrapper equivalence harness as part of Step 4,
  not a later cleanup. Otherwise Step 4 can be marked green while production code never exercised the
  real public API.

I would not split Step 8/9 by default. Silero before lifecycle is defensible because it belongs in
`SharedRuntime`/`SessionRuntime` and Step 9 only consumes its finalize trigger. But Step 8 must avoid
claiming WS lifecycle behavior; it should be a library-level VAD integration + measurement step.

## Ordering / Dependencies

The main dependency chain is correct:
- Step 1 -> all protocol-visible work: yes.
- Step 3 -> Steps 4-7: yes.
- Step 6 -> Step 7: yes.
- Step 8 -> Step 9: acceptable if Step 8 is scoped to `SessionRuntime` VAD and Step 9 wires it to WS.

The Step 8/9 tradeoff is not "Silero needs lifecycle" vs "lifecycle needs Silero"; it is "what can
be tested honestly before WS exists." v2 should say Step 8 proves VAD load/warm/per-frame trigger in
`SessionRuntime`; Step 9 proves the same trigger crosses the WS worker and stats/stale-gen emit path.

## Risk / De-risking

Static-global audit does not need a separate Step 0, but it needs a hard gate inside Step 3. The plan
currently says "identify any static globals" and leave them in place (`:138-141`), then transfers in
Step 4. That is fine only if Step 3 emits a durable inventory and Step 4 is blocked on consuming it.

Silero CPU cost does need an early probe. v4 deliberately pinned CPU Silero with bounded threads, but
the plan's Step 8 bar only runs N=4 synthetic streams (`:231-234`). That does not validate the N=64
thread-cap assumption. Add either a Step 8 pre-probe or a Step 8 bar:
- measure per-frame Silero eval cost on the target box;
- run an N=64-equivalent VAD loop with `torch::set_num_threads(2)`;
- report CPU utilization and p95 per-frame latency;
- fail/escalate if the VAD path consumes enough CPU to threaten WS worker scheduling.

## Bars / Gates

The global test protocol at `:87` is good, but individual bars read like replacements rather than
additions. Add one sentence before the step list: "Every Bar below is additive to the global
per-step protocol." Otherwise Steps 5, 6, 7, 9, and 10 can be misread as not requiring the N=200 gate
and B2+Tier3 smoke set.

Also add a production-shape WS overhead gate. Step 11's 8-utterance canonicalized diff is necessary
but not sufficient for a production WS server: it does not prove the WS path preserves the Phase 2
N=64 scheduler-ON margin. Add a bounded performance smoke after Step 9 or Step 11, e.g. scheduler ON
at N=64 with the real WS path, admitted error rate <=1%, `ttfs_p95 <= 175ms`, and an explicit
allowed overhead delta vs the density harness / Step 3a WS-tail baseline. Production deploy docs can
be Part C; this performance guard should not be deferred entirely.

## PLAN_RULES Compliance

Mostly compliant by the global rule block, but v2 should remove ambiguity:
- Label the review intensity by actual step number and WS-A/WS-B alias. Current text references
  WS-A1/A3/A6/B1/B2/B6 (`:88`) while the Progress table only has numbers 1-11.
- Restate that the N=200 gate + b2-t1 + density-sweep OFF + stalegen/admission/stats smokes apply
  to every code step, even when the step's local bar is a small selftest.
- Step 8 explicitly reruns only b2-t1 and density-sweep OFF (`:233-234`); by the global rule it also
  owes N=200, stalegen, admission, and stats smokes. Say that explicitly or point to the global rule.

## Architecture v4 Alignment

Mapped correctly: v4 X -> Step 2; v4 II/XII -> Steps 3-4 and 7; v4 VI -> Step 8; v4 IX -> Step 10;
v4 XIV -> Step 11.

Missing or weakened:
- v4 IV admission sub-object in `/stats`.
- v4 VII `/health` status enum and v4 XIII `pid`/`process_label`.
- v4 XI HTTP admin handler pool and bounded queue.
- v4 XV PCM-frame selftest row, unless explicitly deferred to Step 9.
- v4 XVI grouped operator config table. This can be a small Step 7 doc/artifact fold, not a new step.

## In-flight Part A v1 Salvage

The plan's "discard v1 ws_server.cpp; salvage mechanical CMake/file-move pieces if useful" is directionally
right but not auditable. Add a required salvage audit before Step 3 edits land:
- list every v1 hunk considered;
- classify it as `keep-mechanical`, `discard-protocol`, or `discard-public-api`;
- define `mechanical` narrowly as CMake target shape, file moves, include-path fixes, or no-behavior
  cleanup that matches v4;
- write the result to `reviews/part-a-v1-salvage-audit.md` or the Step 3 paired verdict.

No separate Step 0 is required unless this audit finds non-mechanical changes that are tempting to
keep. In that case, stop and review those changes explicitly.

## Missing Entirely

- **WS performance guard:** must fold, see above.
- **Production deploy artifacts** (`systemd`, `ASR.env`, runbook): acceptable to defer to Part C.
- **Full multi-process MPS smoke:** acceptable to defer, but v4 XIII's per-process observability
  fields and no-default `--port` behavior must be implemented in Step 7/11 now.
- **`/admission` route:** v4 calls it optional; not a blocker if `/stats.admission` is Python-shaped.

## Net

v1 is close, but not converged. Fold the Part B status semantics, v4 contract omissions, stronger
Step 4/9/11 bars, Silero N=64 probe, and WS overhead guard into v2. Then one more tight review should
be enough; no major redesign is needed.
