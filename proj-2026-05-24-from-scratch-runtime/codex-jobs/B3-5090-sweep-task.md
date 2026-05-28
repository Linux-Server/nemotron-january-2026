<task>
**5090 knee re-measure (§II.13 of `reviews/B2-design-paired-verdict.md`)** — the local dev-box validation
of B2's projected lift mechanism (N=40 baseline → ~47+ via batching), in parallel with the L40S B3 sweep.
The 5090 is sm_120 (different arch than L40S sm_89); the *mechanism* (per-row weight amortization) transfers
but the *absolute* knee differs. Both numbers feed the B3 verdict.

This task assumes F2-T (telemetry hardening) has just been committed (the `runtime/F2T_READY.marker` file
exists; same gate the L40S task waits on) → the scheduler emits the hardened telemetry (timer p50/p95/p99 +
dispatcher CPU% + stream utilization + queue depth + fairness spread).
</task>

<context>
**The binding spec** §II.13 of `reviews/B2-design-paired-verdict.md` defines the 4-axis sweep:
- Axis 1 (control): scheduler OFF (production B=1 baseline) AND scheduler ON `B_max=1` (overhead control).
- Axis 2 (B_max): ON × {2, 4}.
- Axis 3 (window/lone): at the chosen B_max=4, `window_ms ∈ {0, 4, 8, 10, 12}` × `lone_timeout_ms ∈ {0, 1}`.
  If runtime is too expensive, narrow to `{0, 8, 10, 12}` × `{0, 1}` plus lone at chosen window.
- Axis 4 (N): {40, 44, 48, 56, 64} fresh-process-per-N, stagger-robust (existing 10s per-worker stagger).
- Plus: a **burst-injection variant** at N=64 (no stagger) to surface worst-case HOL tail.

**Existing baselines** (from `phase2-density-review` memory + run logs):
- Step 1a 5090 knee = N=40 SLO-robust at scheduler OFF (production B=1). This is the OFF baseline.
- Projected with scheduler ON Bmax=4: N=40 → ~47-64 (the B2 lift goal). Realized knee TBD by this sweep.

**Repo context**:
- `runtime/cpp/density_main.cpp` — the `--mode density-sweep` harness (already exists, used in 1a).
- `runtime/cpp/batched_steady_scheduler.{h,cpp}` — the scheduler.
- `runtime/cpp/steady_batch_primitive.h` — the primitive + manifest verify.
- `runtime/container/enter.sh` — the build environment (sm_120, torch 2.8.0+cu128).
- `runtime/steady_b_artifacts/enc_steady_aoti_b{1,2,4}.pt2` + `MANIFEST.json` — the sm_120 B-buckets.
- The prior 5090 sweep harness (Step 1a) — check `runtime/run_l40s_density.sh` for the sweep loop pattern
  (it's L40S-named but reusable for 5090).

**The sweep harness**:
- Use the existing `--mode density-sweep` + the new CLI args from B2 (`--batch-steady on/off`, `--batch-b-max`,
  `--batch-window-ms`, `--batch-lone-timeout-ms`).
- Wrap in a shell loop that iterates the 4 axes, fresh-process-per-N (process per (B_max, window, lone, N)
  cell).
- Output per-cell JSON to `runtime/artifacts/b3_5090_logs/<stamp>/<axis-cell>.jsonl`.
- For each N, the existing density-sweep telemetry already emits ttfs_p50/p95/p99, lag_p50/p95/p99,
  finalize_wait, etc. The F2-T'd scheduler emits the additional aggregates per cell.

**Stages**:
1. Wait for `runtime/F2T_READY.marker` to exist (poll every 30s; bounded to 90 minutes — if F2-T isn't done
   by then, STOP and report).
2. Build `density_main` in the container with the F2-T'd code (`cmake --build cpp/build_b2 --target
   density_main -j$(nproc)`).
3. Run the OFF baseline (the existing N=40 5090 knee, confirmed un-regressed) → 1 cell.
4. Run the ON `B_max=1` control at N=40 → 1 cell. Measures scheduler overhead with no real batching.
5. Run the ON `B_max=2` and `B_max=4` sweeps (Axes 2-4 = the bulk).
6. Run the burst-injection variant at N=64.
7. Emit a per-cell summary JSON (one line per cell) into `runtime/artifacts/b3_5090_logs/<stamp>/summary.jsonl`.
8. Write a structured results report to `proj-2026-05-24-from-scratch-runtime/reviews/B3-5090-result.md`
   with the SLO-robust knee per (B_max, window, lone), the lift table vs OFF baseline, the F2-T telemetry
   summary (dispatcher CPU%, stream util, queue depth p95, fairness spread per cell), the F2-M memory delta
   per cell, the burst-injection findings.

**SLO + correctness contract**:
- Per cell, SLO-robust = lag_p95 < 500ms ∧ ttfs_p95 ≤ 175ms ∧ ttfs_p99 ≤ 250ms ∧ err≤1% ∧ 0 mismatch.
- Stagger-robust at the knee N and N+4.
- Mismatch reports = STOP this lever (token correctness is the SLO signal).
- Lag p99 reports + tail telemetry per the F2-T schema.

**Cost discipline**: this is local (no EC2). Run as long as needed; if a cell wedges, kill + record + skip.
</context>

<verification_loop>
Build in container (sm_120). Validate the F2-T'd JSON emit on a single OFF + ON Bmax=1 cell BEFORE running
the full sweep (catch JSON schema regressions early). Then run the full sweep. Each cell is fresh-process so
a wedge in one cell shouldn't poison the next.
</verification_loop>

<action_safety>
- Local-only (no EC2).
- Don't commit the sweep .jsonl logs (gitignored under artifacts/).
- Don't change the scheduler / primitive code (that's F2-T's scope, already committed).
- Don't run the full 1000-corpus b2-t1 here (that's the L40S task's F1; 5090 OOMs at full corpus).
</action_safety>

<compact_output_contract>
When done, report:
1. The realized 5090 SLO-robust knee per cell + the chosen (B_max, window, lone) → max-knee setpoint.
2. The lift table vs OFF baseline (knee delta + ttfs_p95 delta + lag_p95 delta per cell).
3. The F2-T telemetry summary (dispatcher CPU% range, stream util range, queue depth p95 range, fairness
   spread per cell).
4. The F2-M memory delta (scheduler ON vs OFF peak GPU memory per N).
5. The burst-injection N=64 finding (worst-case HOL tail).
6. The path to the structured results report (`reviews/B3-5090-result.md`).
</compact_output_contract>
