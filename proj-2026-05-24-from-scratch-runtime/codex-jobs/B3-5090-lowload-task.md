<task>
**5090 low-load sweep — verify the "N=1 not penalized" hypothesis.**

The B2 design hypothesized: at low concurrency, the `lone_timeout_ms=0` short-circuit means a lone-arrival is
dispatched immediately, so the scheduler-ON path is effectively "B=1 with a thin synchronization wrapper" —
no penalty at N=1. The 5090 B_max=1 control at N=40 collapsed lag, but that's a **service-rate failure at
full load with no batching**, not the low-load wrapper-overhead question.

This sweep measures the actual wrapper overhead at low load (N ∈ {1, 4, 8}) + scheduler ON vs OFF, to
validate or refute the hypothesis.

Bounded scope (~1 hour, local 5090, no EC2).
</task>

<context>
**Pre-conditions met**:
- F2-T committed (`5155a96`) → scheduler emits hardened telemetry.
- Fix #8 committed (`<this commit>`) → SLO predicate is policy-aligned (event drift counted-not-gated).
- The 5090 is idle (the prior 5090 sweep + Fix #8's b2-t1/smoke runs are done).
- Build dir: `cpp/build_b2/` (the post-F2-T, post-Fix-#8 binary).

**The cells** (each fresh-process, default `--density-sessions-per-worker 8`, 10s per-worker stagger via
`--density-start-stagger-ms 10000`):

| Cell label | N | scheduler | B_max | window | lone | Purpose |
|---|---|---|---|---|---|---|
| `off_N1` | 1 | OFF | — | — | — | OFF baseline at N=1 (single-stream best case) |
| `on_N1_B4_W10_L0` | 1 | ON | 4 | 10 | 0 | Wrapper overhead at N=1 (the hypothesis test) |
| `off_N4` | 4 | OFF | — | — | — | OFF baseline at low concurrency |
| `on_N4_B4_W10_L0` | 4 | ON | 4 | 10 | 0 | Wrapper overhead at N=4 (some batching opportunity) |
| `off_N8` | 8 | OFF | — | — | — | OFF baseline at moderate concurrency |
| `on_N8_B4_W10_L0` | 8 | ON | 4 | 10 | 0 | Where batching starts paying off vs wrapper overhead |

DO NOT set `DENSITY_GOLD_EVENTS_TOLERANT=1` — under the post-Fix-#8 default semantics, NOT setting it means
events are counted-not-gated (the policy default). Setting it would activate the new opt-in STRICT mode.

**Expected runtime**: ~10 min/cell × 6 cells = ~1 hour.

**SLO + correctness contract** (same as B3-5090-rerun):
- 0 token divergences (binding SLO signal).
- Events counted (per Fix #8 default).
- SLO-robust at each cell: `lag_p95 < 500 ∧ ttfs_p95 ≤ 175 ∧ ttfs_p99 ≤ 250 ∧ err≤1% ∧ 0 token divergences`.

**Output**:
- `runtime/artifacts/b3_5090_logs/<NEW_STAMP>/<cell_label>.jsonl` per cell.
- `runtime/artifacts/b3_5090_logs/<NEW_STAMP>/summary.jsonl` (one line per cell).
- A new report doc `reviews/B3-5090-lowload-result.md` with:
  - Per-cell SLO result + ttfs/lag p50/p95/p99 + peak memory.
  - The scheduler_telemetry summary per ON cell (gather_wait/service_wait/output_sync/worker_blocked
    p50/p95/p99, dispatcher CPU%, queue depth, fairness spread, bucket fill counts).
  - **The headline lift/penalty table**: for each N ∈ {1,4,8}, the (ON − OFF) ttfs_p95 delta + lag_p95
    delta + peak memory delta.
  - The verdict on the "N=1 not penalized" hypothesis:
    - SUPPORTS if ON N=1 has ttfs_p95 within ~5ms of OFF N=1 (wrapper overhead small enough to not matter).
    - REFUTES if ON N=1 has ttfs_p95 > OFF + ~10ms (wrapper overhead measurable + meaningful).
  - The break-even N where scheduler ON crosses from net cost to net benefit (interpolate from the 3
    measured points — useful for the future per-stream-context-bypass design discussion).
</context>

<verification_loop>
Local only. Rebuild if needed (`./container/enter.sh bash -lc 'cmake --build cpp/build_b2 --target
density_main -j$(nproc)'` — the binary should already be post-Fix-#8; Codex's last builds were on
cpp/build_b2). Run cells fresh-process per cell. If a cell wedges, kill + record + skip.
</verification_loop>

<action_safety>
Local only. Don't commit binary artifacts (gitignored under artifacts/). Don't touch the L40S sweep in
flight (codex job bd7rd0m6n) — that runs on a different box.
</action_safety>

<compact_output_contract>
When done, report:
1. Per-cell result (ttfs/lag p50/p95/p99, peak mem, SLO pass/fail, token + event counts).
2. The (ON − OFF) lift/penalty table at N ∈ {1, 4, 8}.
3. The scheduler_telemetry summary per ON cell (wrapper overhead breakdown).
4. The "N=1 not penalized" hypothesis verdict (SUPPORTS / REFUTES / AMBIGUOUS) with the data point.
5. The interpolated break-even N (where ON ≈ OFF).
6. The report doc path (`reviews/B3-5090-lowload-result.md`).
</compact_output_contract>
