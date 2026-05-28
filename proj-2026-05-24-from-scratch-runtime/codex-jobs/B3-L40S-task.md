<task>
**Step B3 — L40S batched-density sweep + F1 full-corpus b2-t1 prerequisite + L40S sm_89 B-bucket build.**
The decisive density measurement: spin up a fresh g6e.8xlarge L40S, compile the sm_89 B-bucket steady AOTI
packages natively (the sm_120 packages are 5090-only), run the F1 full-corpus b2-t1 (B2's pre-condition
follow-up — the 4-row B2 commit was scope-reduced due to 5090 OOM, L40S has 48GB so it should fit), then
the L40S knee sweep per the binding spec §II.13.

Parallel with the local 5090 knee sweep (separate task). Both feed the B3 verdict / F1 funding re-check
(PHASE2-PLAN.md F1 — provisionally cleared by the STEADY-BATCH-0 projection; B3's realized knee is the
binding re-check).
</task>

<context>
**The binding spec for the sweep** is `proj-2026-05-24-from-scratch-runtime/reviews/B2-design-paired-verdict.md`
§II.13 (the 5090 knee re-measure structure transferred to L40S). Read it in full.

**Read also**:
- `proj-2026-05-24-from-scratch-runtime/PHASE2-PLAN.md` — Step B3 (with the F1-F6 carry-over from B2's
  paired verdict at the bottom of the B3 step body).
- `proj-2026-05-24-from-scratch-runtime/reviews/B2-build-paired-verdict.md` — the F1-F6 follow-ups, the
  scope reductions B3 must close.
- `proj-2026-05-24-from-scratch-runtime/reviews/steady-batch0-RESULT.md` — the projected lift mechanism
  (37 → ~47-64 on L40S).
- `proj-2026-05-24-from-scratch-runtime/runtime/run_l40s_density.README.md` — existing L40S provisioning
  + density-sweep harness pattern (the prior Step 1b W3 run; reuse).
- The prior L40S sweep logs in `runtime/artifacts/l40s_w3_logs/` to understand the existing sweep harness
  + output format.

**EC2 setup** (AWS profile `AWSAdministratorAccess-419599258555`, region `us-west-2`):
- Spin up a fresh `g6e.8xlarge` instance with the prior bench AMI / key (`ec2-bench/nemotron-bench-key.pem`).
  Look at `ec2-bench/` for the prior spin-up script or `.instance_*.json` for the launch config pattern.
- After spin-up, write the new instance ID + IP to a fresh `ec2-bench/.instance_b3_l40s.json` so it's
  trackable + terminable.
- The L40S box hostname: SSH as `ubuntu@<ip>` with the bench key.
- Sync the repo to the L40S box (rsync excluding `.git`, `*.pt2`, `*.ts`, `__pycache__`, build dirs — use
  the prior W3 sync pattern).

**Stages on the L40S box** (sequential):
1. **Provision**: install Python venv (`/home/ubuntu/parakeet/venv` per the prior pattern; uses
   `parakeet/setup.sh` if it exists), install nemo (for the sm_89 export), torch 2.8.0+cu128 (matches the
   container; needed because the container is sm_120-only; sm_89 needs native compile).
2. **Compile sm_89 B-buckets**: `python runtime/export_steady_batched.py --out runtime/steady_b_artifacts
   --batches 1,2,4 --compile-only` (uses the pre-staged ExportedProgram .pt2 files; if not present, full
   export — but that needs nemo + the model). Verify the MANIFEST.json emits cleanly for sm_89.
3. **Native cmake build**: build `density_main` natively on the L40S box (the container path is sm_120-only;
   for L40S use the native CUDA / torch wheels). The runbook `run_l40s_density.README.md` documents the
   native build.
4. **F1 — full-corpus b2-t1**: run `density_main --mode b2-t1 --steady-batch-dir steady_b_artifacts artifacts`
   with `DENSITY_GOLD_EVENTS_TOLERANT=1` (per project policy; events counted-not-gated) on the FULL 1000-
   utterance corpus (not the 4-row B2 scope). If the L40S 48GB can't accommodate the original 6-case full-
   corpus, split per case (run identical / mixed / forced K=2 / forced K=3-padded / forced B=4 / staggered /
   Bmax=1 control SEPARATELY with fresh-process-per-case). Required: **0 token divergences** across all
   cases / all 1000 utts.
5. **L40S knee sweep** (§II.13 axes; **after F1 PASS**):
   - Axis 1 (control): scheduler OFF (production B=1 baseline) + scheduler ON `B_max=1` (overhead control).
   - Axis 2 (B_max): ON × {2, 4}.
   - Axis 3 (window/lone): at B_max=4, `window_ms ∈ {0, 4, 8, 10, 12}` × `lone_timeout_ms ∈ {0, 1}`. If
     runtime is too expensive, narrow to `{0, 8, 10, 12}` × `{0, 1}`.
   - Axis 4 (N): {36, 40, 44, 48, 56, 64} fresh-process-per-N, stagger-robust (the prior 10s per-worker
     stagger).
   - Plus: a **burst-injection variant** at N=64 (synchronized N-stream start, no stagger) → surface
     worst-case HOL tail.
   - **Per-N report**: SLO-robust knee + lift vs OFF baseline; the F2-T-hardened telemetry (timer p50/p95/p99,
     dispatcher CPU%, stream util, queue depth p95, fairness spread); peak memory (F2-M).
6. **Cost discipline**: terminate the L40S box at the END of the run (or after a verified result + log
   sync). EC2 cost ~$2.50-4/hr; budget ~6 hours = $15-24.

**SLO + correctness contract** (binding):
- **0 token divergences** in the F1 full-corpus b2-t1. FATAL if any.
- Events counted-not-gated per `DENSITY_GOLD_EVENTS_TOLERANT` (project policy).
- The L40S knee = the largest N with `ttfs_p95 ≤ 175ms / p99 ≤ 250ms`, `lag_p95 < 500ms`, err≤1%, 0
  mismatch, stagger-robust at the knee N and N+4.

**Output deliverables**:
- `runtime/artifacts/b3_l40s_logs/` (or similar): per-N density-sweep .log / .jsonl from each axis cell.
- `proj-2026-05-24-from-scratch-runtime/reviews/B3-L40S-result.md`: structured results report with the
  realized SLO-robust knee + lift vs OFF baseline + the F2-T telemetry summary + the F2-M memory delta
  table + the F1 b2-t1 full-corpus result + paired-review-ready findings.
- The fresh `ec2-bench/.instance_b3_l40s.json` (committed) so it's trackable; updated when terminated.

**Wait for F2-T to commit** (separate Codex task) before starting the knee sweep (step 5). Steps 1-4 can
start immediately. After F2-T commits, I'll signal you (or you can detect via a `runtime/F2T_READY.marker`
file I'll write) — then sync the F2-T'd density_main + scheduler to the L40S box + run step 5.
</context>

<verification_loop>
Native build on the L40S (no container): `cmake -S cpp -B cpp/build_b3 -DTORCH_ROOT=$(python -c
'import torch, pathlib; print(pathlib.Path(torch.__file__).parent)')` + `cmake --build cpp/build_b3 --target
density_main -j$(nproc)`. Validate F1 (0 token divergences) before any knee sweep. After F2-T sync, rebuild
before the sweep.

If steps 1-3 hit a real blocker (no spin-up script / nemo install failure / arch mismatch), STOP, report,
do not work around. The prior W3 logs document the path.
</verification_loop>

<action_safety>
- Sync code from the repo to the L40S; do NOT push changes from the L40S back to the repo (the L40S is
  ephemeral; commits stay on the dev box).
- TERMINATE the L40S at the end (whether success or fail) to stop the cost. Cost discipline > convenience.
- Don't commit AOTI binary artifacts (gitignored).
- Don't run the full knee sweep before F1 passes (correctness first).
</action_safety>

<compact_output_contract>
When done, report:
1. EC2 instance ID + IP + spin-up cost so far.
2. sm_89 B-bucket compile result (MANIFEST.json contents, byte sizes).
3. F1 full-corpus b2-t1 result (per-case PASS/FAIL, token + event divergence counts, max diffs).
4. The full knee sweep result table (N × B_max × window × lone → knee determination + SLO-pass/fail per
   cell + F2-T telemetry summary + F2-M memory delta).
5. The B3 verdict file path + the realized SLO-robust knee.
6. EC2 termination confirmation.
</compact_output_contract>
