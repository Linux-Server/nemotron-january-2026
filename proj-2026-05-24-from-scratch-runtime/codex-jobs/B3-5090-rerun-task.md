<task>
**Focused 5090 re-run** — the first sweep (codex-jobs B3-5090-sweep-task.md → reviews/B3-5090-result.md)
stopped early on a strict event-count mismatch at `utt198 chunk13 event[6]` that turns out to be the
**documented interim-event-timing drift class** (same exact utt/chunk/event as B1's drift; counted-not-gated
per project policy via `DENSITY_GOLD_EVENTS_TOLERANT=1`). The sweep didn't set that flag, so a tolerable
drift was misclassified as a STOP-this-lever correctness failure.

Re-run **two key cells at N=40** with the tolerance flag set, to confirm the lift mechanism on 5090 (the
B_max=2 cell already showed IMPROVED timing of ttfs_p95 −21ms / lag_p95 −11ms vs OFF baseline — confirm it
passes under correct policy + measure B_max=4 for the lift table).

The 5090 is **memory-capped at N=40** (+11.5 GiB scheduler ON → 31.3 of 31.32 GiB). Don't try higher N. The
L40S sweep (separate Codex task `bd7rd0m6n`) handles N=44/48/56/64 with its 48GB.
</task>

<context>
**The cells to run** (each fresh-process, 8 sessions/worker per existing pattern):
1. `on_B2_W10_L0_N40_TOL` — B_max=2, window=10, lone=0, N=40, `DENSITY_GOLD_EVENTS_TOLERANT=1`.
2. `on_B4_W10_L0_N40_TOL` — B_max=4, window=10, lone=0, N=40, `DENSITY_GOLD_EVENTS_TOLERANT=1`.

**Pre-conditions** (already met):
- F2-T committed (5155a96): scheduler emits `scheduler_telemetry` with hardened timers + aggregates.
- `runtime/F2T_READY.marker` exists.
- Container build available at `cpp/build_b2/density_main` (rebuild if needed).
- The OFF baseline at N=40 is `runtime/artifacts/b3_5090_logs/20260528T124526Z/off_N40.jsonl` (already
  done; reuse for lift comparison: `ttfs_p95=33.014ms`, `lag_p95=-129.149ms`).

**SLO + correctness contract**:
- `DENSITY_GOLD_EVENTS_TOLERANT=1` for both cells.
- 0 token divergences (the SLO signal — the binding correctness gate that B1+B2 paired-reviewed).
- Event drift count reported but not gated (consistent with B1's 5/1000 prior bar + B2's 0/0 + this drift
  class).
- SLO-robust = `lag_p95 < 500ms ∧ ttfs_p95 ≤ 175ms ∧ ttfs_p99 ≤ 250ms ∧ err≤1% ∧ 0 token divergence`.

**Output**:
- `runtime/artifacts/b3_5090_logs/<NEW_STAMP>/on_B2_W10_L0_N40_TOL.jsonl` + `on_B4_W10_L0_N40_TOL.jsonl`.
- Update `proj-2026-05-24-from-scratch-runtime/reviews/B3-5090-result.md` to include the corrected verdict
  + the 2 new cells + the lift table vs OFF baseline.
- The corrected verdict should explicitly note: (1) the original `mismatches=1` was the documented
  interim-event-timing drift, not a real correctness failure; (2) the 5090 lift is constrained to N=40 by
  memory; (3) the higher-N lift demonstration is the L40S sweep's job.

**Cost**: local, no EC2.
</context>

<verification_loop>
Rebuild density_main if needed (`container/enter.sh bash -lc 'cmake --build cpp/build_b2 --target
density_main -j$(nproc)'`). For each cell: fresh-process, 8 sessions/worker, 10s per-worker stagger,
`DENSITY_GOLD_EVENTS_TOLERANT=1`. Per-cell expected runtime ~25-35 min based on prior cell timing. STOP a
cell only on 0 token divergence violation OR `errors > 0`; otherwise let it complete and report the SLO
status.
</verification_loop>

<action_safety>
Local only. Don't commit binary artifacts. Don't change the scheduler/primitive code (F2-T's scope is
already committed; B2's design is locked).
</action_safety>

<compact_output_contract>
When done, report:
1. Per-cell result (B_max=2 and B_max=4 at N=40): SLO-robust pass/fail, token_divergences, event_divergences
   (counted-not-gated), `ttfs_p50/p95/p99`, `lag_p50/p95/p99`, peak GPU memory, the F2-T scheduler_telemetry
   summary (dispatcher CPU%, stream util%, queue depth p95, fairness spread p95, batch fill counts).
2. Lift table vs OFF baseline (knee delta = both pass/fail at N=40; ttfs delta + lag delta + memory delta).
3. The updated reviews/B3-5090-result.md path.
4. Confirmation that the original mismatch=1 was the policy-tolerated class (not a new bug).
</compact_output_contract>
