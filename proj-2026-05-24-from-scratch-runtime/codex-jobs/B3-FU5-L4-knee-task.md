<task>
**B3-FU-5: L4 knee sweep + roofline validation** — measure the production knee on AWS `g6.4xlarge`
(L4, 24 GiB, ~300 GB/s, sm_89) to (1) validate the memory-bandwidth-bound roofline projection from
B3-FU-2 (predicting L4 knee ≈ 22 streams = L40S × 0.35 BW ratio), (2) close the deployment-economics
gap (L4 cheaper per stream vs L40S density), and (3) characterize the memory-cap risk (per-stream
slope 0.173 GiB → projected ~24.3 GiB at N=22 vs L4's 24 GiB cap — OOM is a real risk).

Run a bounded staggered knee sweep on a fresh `g6.4xlarge` EC2 instance, write results to
`proj-2026-05-24-from-scratch-runtime/reviews/B3-FU5-L4-knee-result.md`, and TERMINATE the instance
at the end of the run.

**This is a cost-controlled, bounded sweep — start LOW (N=4) and walk UP. STOP on the first SLO
failure OR OOM OR N=24 reached.**
</task>

<context>

## Why this measurement matters

From the project memory (`deployment-target-sagemaker.md`): production deploy = AWS SageMaker, with
the open question of L4 (g6) vs L40S (g6e). The earlier Python-server measurements suggested
L40S ~16-20/box and L4 ~6/box (ratio ~0.3-0.4). The native runtime might tilt the economics
differently. The B3-FU-2 roofline analysis (2026-05-28) confirmed the L40S knee is GPU-memory-
bandwidth-bound (DRAM 70% at N=64); L4 with 0.35× the BW should land at N ≈ 22 by linear
projection — IF memory bandwidth is the only binding constraint. Caveats:

- **Memory cap risk**: per-stream activation slope = 0.173 GiB/stream (measured B3-FU-1 on L40S).
  Projected at N=22: ~24.3 GiB peak ≈ L4's 24 GiB cap. **L4 may hit OOM at N=20-22 BEFORE the
  BW wall.** This is itself a useful finding.
- **Dispatcher CPU**: 32 vCPU Milan = SAME as L40S (g6e and g6 share the CPU socket). At L40S N=64
  dispatcher CPU = 61%; at L4 N=22 it should be ~21% (linear) → CPU headroom not the binding
  constraint.
- **Kernel arch**: sm_89 (Ada) on both → same AOTI bundles work; just slower per-launch on L4
  (fewer SMs: 58 vs 142).

## Predicted outcomes

| N | Projected DRAM | Memory peak (slope · N + OFF baseline ~21.9) | Reading |
|---|---:|---:|---|
| 4 | ~12% | ~22.6 GiB | comfortable |
| 8 | ~24% | ~23.3 GiB | comfortable |
| 12 | ~36% | ~24.0 GiB | tight on L4 cap |
| 16 | ~48% | ~24.7 GiB | likely OOM on L4 |
| 20 | ~60% | ~25.4 GiB | OOM |
| 22 | ~66% | ~25.7 GiB | OOM |

The OFF baseline ~21.9 GiB is from L40S — L4 may differ. The PROJECTION gives a starting hypothesis;
the actual measurement is what matters. **If memory holds, sweep up to N=24; if memory caps first,
the lowest-N cap is the actual binding constraint.**

## Run policy (mirror B3-FU-1 exactly except instance type + N range)

```
NEMOTRON_DENSITY_BATCH_STEADY=1
NEMOTRON_DENSITY_BATCH_MAX=4
NEMOTRON_DENSITY_BATCH_WINDOW_MS=0
NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS=0
--admission-active-cap 10000   (effectively unlimited; sweep tests the GPU knee, not admission)
--admission-backlog-cap 10000
--density-start-stagger-ms 10000
sessions_per_worker=2
CUDA_MODULE_LOADING=EAGER
events counted-not-gated (NEMOTRON_GOLD_EVENTS_TOLERANT=1, the default per Fix #8)
NO DENSITY_GOLD_EVENTS_TOLERANT flag (per Fix #8 semantic-flip lesson)
```

**Sweep**: N ∈ {4, 8, 12, 16, 20, 24} adaptive.

**Stop conditions** (any one triggers STOP — no further N attempts):
1. First SLO fail (`lag_p95 ≥ 500ms` OR `ttfs_p95 > 175ms` OR `ttfs_p99 > 250ms` OR `errors > 1%`).
2. OOM (CUDA out-of-memory exception OR peak GPU memory ≥ 23.0 GiB safety margin under L4's 24 GiB cap).
3. N=24 reached and passed → STOP at top of sweep.

**STOP behavior**: at first STOP trigger, do NOT continue higher N. Record the failure mode + the
last passing cell + the cell that failed.

## Instance + cost

- **Instance type**: `g6.4xlarge` (1× L4 24GB, 16 vCPU on g6.4xlarge — NOTE g6.4xlarge has 16 vCPUs
  not 32; verify whether to use g6.4xlarge or g6.8xlarge to match L40S g6e.4xlarge's 32 vCPU
  reference. The Python-server memory used g6e.4xlarge / 32 vCPU; for fair CPU comparison g6.8xlarge
  may be more apt. **Default to g6.4xlarge unless CPU-cores is suspected to be binding** — at
  predicted N≤22 with linear-projected dispatcher CPU ~20%, 16 vCPU should be ample. Document the
  choice in the result doc.)
- **AWS region**: `us-west-2` (same as B3-FU-1/2).
- **AWS profile**: `AWSAdministratorAccess-419599258555`.
- **Bench key**: `ec2-bench/nemotron-bench-key.pem`.
- **State file**: `ec2-bench/.instance_b3_fu5_l4.json`.
- **Cost estimate**: g6.4xlarge ~$0.80/hr × ~2h budget = **~$1.60-3.20 total**.

## Bundle source

sm_89 AOTI bundles from S3 (reuse B3-FU-1/2 path — they're sm_89 too). If sm_89 packages aren't
present locally on the L4 instance, compile them on the L4 in-instance (matches B3-FU-1 fallback
behavior). EPs reused from S3 expected.

## Build environment

- Container: `nemotron-aoti:cu128` (same as L40S).
- Build cmd: same as B3-FU-1 (`cmake --build` in-container via `runtime/container/enter.sh`).
- Density harness: `runtime/cpp/density_main.cpp` `--mode density-sweep`.

## Output document

Write `proj-2026-05-24-from-scratch-runtime/reviews/B3-FU5-L4-knee-result.md` covering:

1. **Verdict** — L4 knee (last-pass N), failure mode (lag / OOM / errors), $/stream comparison
   vs L40S.
2. **Sweep table** — same columns as B3-FU-1's table: N, SLO PASS/FAIL, ttfs p50/p95/p99, lag
   p50/p95/p99, peak GiB, tok/event/err, admitted/offered, dispatcher CPU%, queue p95, B1/B2/B4
   dispatches, K2/K3/K4 dispatches, backlog>Bmax.
3. **Roofline validation** — actual L4 knee vs the linear-BW projection of N≈22. If the L4 knee
   is at N≈22, roofline is confirmed; if much lower, identify the actual binding constraint
   (OOM, dispatcher CPU, cold cache misses, etc.).
4. **Memory analysis** — peak GiB per N + per-stream slope; compare to L40S slope of 0.173 GiB/stream.
   Did OOM hit at the predicted N? What's the OFF baseline on L4 (will likely differ from L40S's
   21.9 GiB)?
5. **Funding/deployment math** — at the realized L4 knee, compute `$/stream/hr`:
   - L4 cost ~$0.80/hr ÷ realized_knee = $/stream/hr.
   - Compare to L40S at $3.00/hr ÷ 64 = $0.047/stream/hr.
   - The headline economics number for the deploy-target decision.
6. **EC2 termination** — confirm AWS state=terminated; report cost (elapsed × hourly).

## Bundle re-use note

Per B3-FU-1's experience, the 668MB torch::jit::load on a shared bundle context can be slow if
not pre-shared. Reuse the `share-ONE-bundle` pattern (single bundle loaded once, shared across
workers via `SharedAOTIArtifacts` per Tier-3 work) to avoid the ~60min/N stall observed in earlier
L40S work.

</context>

<verification_loop>
For each N cell, verify SLO before proceeding to the next N:
- Read the JSON telemetry summary.
- Check `slo_robust`, `ttfs_p95_ms`, `lag_p95_ms`, `peak_gpu_memory_GiB`, `errors`, `token_divergences`.
- If any STOP condition (above) triggers, do NOT continue to higher N.

For OOM detection: monitor peak GPU memory; ALSO catch any CUDA runtime exception (instances may
crash before peak is recorded — handle gracefully with a non-zero exit + a logged "OOM_SUSPECTED"
marker).

Build/correctness verification: at start, run one N=4 cell with a quick smoke (e.g., a small
warmup pass) to confirm the AOTI bundles loaded successfully + are byte-equivalent vs L40S's
behavior (token-exact on a small fixture, since same sm_89 binary).
</verification_loop>

<action_safety>
- Bound this to `g6.4xlarge` (or `g6.8xlarge` if Codex justifies the larger size). Do NOT launch
  multiple L4 instances; bounded single instance.
- TERMINATE the instance at end-of-run regardless of result (cost discipline).
- Use AWS CLI termination if `ec2_down.py` fails (FU-2 had a boto3 import issue).
- DO NOT commit binary AOTI artifacts (gitignored).
- DO NOT silently substitute the OFF-path B=1 with NEW B=1 (preserve byte-exact contract per
  the A1 outcome handling pattern from B2).
- Container `nemotron-aoti:cu128` is canonical.
- Per PLAN_RULES.md: Python paths use `/home/khkramer/src/parakeet/venv/bin/python` with
  `HF_HUB_OFFLINE=1` for any host-side python.
</action_safety>

<compact_output_contract>
Report:
- Path to `reviews/B3-FU5-L4-knee-result.md`.
- One-paragraph headline: realized L4 knee, binding constraint (BW / OOM / CPU / other), $/stream
  comparison vs L40S, deploy-decision implication.
- EC2 state confirmation: instance ID, launch/terminate timestamps, cost estimate, state-file path.
- Build/correctness note: did the sm_89 bundle load cleanly on L4? Any token divergence vs L40S
  on a small smoke?
</compact_output_contract>
