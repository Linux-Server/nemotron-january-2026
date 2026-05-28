<task>
**B3-FU-1: bracket the true L40S knee + B3-FU-3 post-Tier-3 memory delta (combined task).**

The prior B3 sweep (codex job `bd7rd0m6n`) capped at N=64 (top of registered axis); the cell at N=64
staggered had ttfs p95=21ms / lag p95=−93ms (deep margin) — **ceiling not reached.** Bracket the true
production knee by extending the sweep to N ∈ {72, 80, 88, 96, 112, 128} with adaptive extension if N=128
still passes. Bundle the F2-M memory re-measurement since the prior L40S sweep used the pre-Tier-3 binary
(measured +11.7 GiB delta; Tier 3 since committed `0a4994f` and reduces overhead on 5090 by 4.6 GiB).

**Cost-bounded**: ~6-8 cells × ~25 min = ~2-3 hours. g6e.8xlarge ~$2.50-4/hr → ~$5-12. Terminate at end.
</task>

<context>
**Code state**: HEAD includes Tier 3 (`0a4994f`) + F2-T (`5155a96`) + Fix #8 (`5ef23af`) + Step 2a
(`7035d01`) + plan v5 (`c0a040a`). Use the current HEAD binary; the new L40S build is post-Tier-3.

**Reuse from prior B3 sweep**:
- S3 artifacts at `s3://nemotron-phase2-eps-419599258555/density/steady_b_artifacts/` (sm_89 B=1/2/4
  AOTI packages, finalize buckets, EPs, manifest) — pull-down should be fast (~5-10 min).
- Native sm_89 compile path documented in `runtime/run_l40s_density.README.md` (post-Fix-#8 — defaults
  now policy-tolerant; no need to set `DENSITY_GOLD_EVENTS_TOLERANT`).

**SLO + correctness contract** (same as B3):
- 0 token divergences (binding SLO signal).
- Events counted-not-gated per Fix #8 default (don't set `DENSITY_GOLD_EVENTS_TOLERANT`).
- SLO-robust per cell: `lag_p95 < 500ms ∧ ttfs_p95 ≤ 175ms ∧ ttfs_p99 ≤ 250ms ∧ err ≤ 1%`.
- Stagger-robust at the knee N and knee+4.

**Stages on the L40S box** (sequential):
1. Spin up fresh `g6e.8xlarge` in `us-west-2` with profile `AWSAdministratorAccess-419599258555`. Reuse
   the bench-key + AMI pattern from prior `ec2-bench/.instance_b3_l40s.json`. Write new instance ID to
   `ec2-bench/.instance_b3_fu1_l40s.json`.
2. Provision (Python venv + torch 2.8.0+cu128 + minimal deps; no nemo needed if using S3 EPs).
3. Pull pre-built sm_89 artifacts from S3 (`steady_b_artifacts/enc_steady_aoti_b{1,2,4}.pt2` +
   `MANIFEST.json` + finalize buckets + EPs).
4. Native cmake build of `density_main` (post-Tier-3 / post-F2-T / post-Fix-#8 / post-Step-2a HEAD).
5. **OPTIONAL parity check**: re-run `B_max=4 W=0 L=0 N=64` staggered as a single cell to verify the
   post-Tier-3 binary doesn't regress the prior N=64 ttfs/lag numbers (expected: same ttfs p50≈13.5ms /
   p95≈21ms but **lower peak memory** vs prior 33.64 GiB — F2-M measurement).
6. **THE BRACKET SWEEP** (B3-FU-1 primary): `B_max=4 W=0 L=0` staggered (10s per-worker stagger via
   `--density-start-stagger-ms 10000`) at:
   - N = {72, 80, 88, 96, 112, 128} — base bracket.
   - **Adaptive extension**: if N=128 PASSES SLO-robust, add N=160. If N=160 also passes, add N=192.
     Stop at first FAIL or after N=192 (whichever first).
   - Per cell: `sessions_per_worker=2` (cost-bounded harness, matching prior B3); fresh-process-per-N.
7. **Memory measurement (B3-FU-3)**: every cell records `peak_gpu_mem_gib`. Compute the OFF→ON delta at
   the OFF-baseline if available, OR document the ON-side absolute (the prior B3 OFF baselines maxed at
   N=64 OFF = 21.93 GiB; for N>64 we only have ON-side measurements).
8. **Terminate** the L40S at end (or after the result write).

**The deliverables**:
- `runtime/artifacts/b3_fu1_l40s_logs/` + `b3_fu1_l40s_jsonl/` (per-cell + summary).
- `reviews/B3-FU1-result.md` with:
  - Per-cell sweep table (N → ttfs p50/p95/p99 + lag p50/p95/p99 + SLO pass/fail + peak_gpu_mem_gib +
    token/event divergence counts + dispatcher CPU + queue depth + bucket fill).
  - The **realized true knee** + the cells that bracket it (last passing N + first failing N).
  - The F2-M memory delta: post-Tier-3 peak vs the prior B3 pre-Tier-3 peak at N=64 (and onwards if OFF
    baselines are re-run); per-N memory slope.
  - Updated funding math with the realized knee (vs the bounded 64/40 = 1.60×).
- Update `ec2-bench/.instance_b3_fu1_l40s.json` with terminated state.

**Out of scope** (defer):
- F1 b2-t1 full corpus re-run (already done in prior B3; 0 token + 0 event passed).
- Burst injection (per plan v5: characterization, not production-relevant).
- W/L sweep optimization (W=0 L=0 was the prior winner; just hold that policy at higher N).
- B_max=2 at higher N (separate small follow-up if needed for strict-event-clean characterization).
- F2-T deadlock investigation (B3-FU-5, separate).
</context>

<verification_loop>
Native build on the L40S. The sweep must surface any token divergence as a STOP for that cell. Use
adaptive bracketing: don't pre-commit cells beyond N=128 unless the prior cell PASSES. If a cell wedges
(no completion in 90 min), kill + record + skip; don't poison subsequent cells.
</verification_loop>

<action_safety>
- TERMINATE the L40S at end (cost discipline). Don't leave running after results synced.
- Sync code from repo to L40S; don't push L40S changes back.
- Don't commit binary AOTI artifacts (gitignored).
- Use the post-Fix-#8 default (tolerant events) — don't set `DENSITY_GOLD_EVENTS_TOLERANT`.
</action_safety>

<compact_output_contract>
When done, report:
1. EC2 instance ID + IP + cost so far (launch → terminate elapsed × rate).
2. Build result + the post-Tier-3 binary's parity-check cell (N=64 staggered) vs prior (B3) numbers.
3. The bracket sweep table per cell: N → ttfs p50/p95/p99 / lag p50/p95/p99 / pass/fail / peak_mem /
   token+event divergences / dispatcher CPU% / queue p95.
4. **The realized true knee** (last passing N + first failing N + the bracket interpretation).
5. The F2-M memory delta + per-N memory slope (the per-stream activation slope insight Codex's plan v4
   review flagged).
6. Updated funding math (vs S_py=20 high end + 0.83 haircut).
7. Path to the structured `reviews/B3-FU1-result.md`.
8. EC2 termination confirmation.
</compact_output_contract>
