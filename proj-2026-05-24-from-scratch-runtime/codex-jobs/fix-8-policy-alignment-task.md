<task>
**Fix #8 — align density-sweep SLO predicate with documented project policy on event divergences.**
Bounded scope, single-file change (mostly), single Opus review (no paired delegation needed). Prevents
recurrence of the 5090 sweep misclassification where the harness STOPped on a documented interim-event-
timing drift (`utt198/chunk13/event[6]`) that policy classifies as counted-not-gated.
</task>

<context>
**The bug** (surfaced by the B3 5090 sweep, codex job bp3wc58uu):
- Project policy (memory + B1/B2 paired verdicts): interim-event-timing drift is **counted-not-gated**;
  the SLO gate is `token_divergences == 0`. Documented "5/1000 prior bar" is informational, not gating.
- Current code: density-sweep's `slo_robust` predicate (and related cell-pass gating) includes
  `event_divergences == 0` (or equivalent — `serial_oracle_match_pass` requires byte-exact event match).
  When `DENSITY_GOLD_EVENTS_TOLERANT=1` is set, the predicate softens; when it isn't set, the predicate
  fails on the documented drift class. This requires every task spec to remember to set the flag.

**The fix** (policy-aligned, single source of truth in code):
1. Make the SLO-gating predicate in `runtime/cpp/density_main.cpp` (density-sweep mode + b2-t1 mode)
   **independent of event_divergences**. The gate becomes:
   `slo_robust = (token_divergences == 0 ∧ errors == 0 ∧ lag_p95 < lag_budget ∧ ttfs_p95 ≤ ttfs_p95_budget ∧
                  ttfs_p99 ≤ ttfs_p99_budget ∧ admitted_err_rate ≤ err_budget)`
   Note: `token_divergences` already exists and is computed correctly; this fix just removes
   `event_divergences` from the SLO predicate.
2. **Keep event_divergences as a REPORTED count** in the JSON output + the stdout summary
   (`event_divergences: N`). The documented "5/1000 prior bar" is informational; readers can compare.
3. **`DENSITY_GOLD_EVENTS_TOLERANT` flag flips semantics**: previously it was opt-IN tolerance (default
   strict). Now it becomes **opt-IN STRICT** (default tolerant — matches policy). When set to 1, the gate
   includes the event-count match for byte-exact debug runs.
4. Add a 1-paragraph comment at the top of the gate function citing the project policy + B1/B2 verdicts.
5. Update `runtime/run_l40s_density.README.md` (or the equivalent runbook) to reflect that
   `DENSITY_GOLD_EVENTS_TOLERANT=1` is now opt-in strict, not opt-in tolerant.

**Why this is safe**:
- B1's commit (3887cb3) ran the full-corpus T1 with 0 token divergences and 6 interim event drifts across
  K=2 + K=3 grouped runs. B2 (0925fa6) re-confirmed token-clean at 4 rows. The token-divergence gate is the
  binding signal — never softened.
- The OFF path (production B=1) is unchanged byte-exact (per the B2 conditional construction). This fix
  only changes how the gate INTERPRETS event-count divergences, not how the production path runs.

**Validation** (bounded):
- Container build (`cmake --build cpp/build_b2 --target density_main -j$(nproc)`).
- Re-run b2-t1 at 4 rows (the B2 commit baseline): confirm pass=true with 0 token divergences, the existing
  event drifts now REPORTED but NOT GATING; the exit code is 0.
- Run a `density-sweep --smoke --n-values 4` (the previous OFF smoke from F2-T): confirm exit 0 (was 1
  before because of a separate "broader pass_to_1b multiplier summary" — verify the fix doesn't break
  that summary's own gate logic; if it does, scope the fix to the per-cell predicate only).

**Out of scope**:
- Don't touch the scheduler, primitive, manifest, A1 parity, or any other B2 logic.
- Don't change the gate predicates for `errors` or `ttfs/lag` budgets.
- Don't change the token-divergence reporting.
</context>

<verification_loop>
Build cleanly. Re-run b2-t1 4 rows + density-sweep N=4 smoke. Both should now exit 0 with 0 token divergences
and the documented event drift class no longer causing a non-zero exit. If the sweep harness's broader
`pass_to_1b` multiplier summary also needs adjustment, scope minimally — the per-cell predicate is the
primary fix.
</verification_loop>

<action_safety>
Only touch `density_main.cpp` (the gate predicate) + the runbook doc. Do NOT modify the scheduler, primitive,
or any test case structure. Keep the token-divergence gate strict (the SLO signal).
</action_safety>

<compact_output_contract>
When done, report:
1. The exact line numbers of the gate predicate change in `density_main.cpp`.
2. The b2-t1 4-row re-run: pass=true, token_divergences=0, event_divergences=N (REPORTED), exit code.
3. The density-sweep N=4 smoke: exit code 0 (or document why a separate gate still fails).
4. The runbook diff (if updated).
5. Confirmation that the change is policy-aligned with the B1/B2 paired-reviewed verdicts.
</compact_output_contract>
