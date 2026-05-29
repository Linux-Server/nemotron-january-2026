<task>
**B3-FU-5b: L4 knee extension above N=24** — find the L4 true knee, which B3-FU-5
showed is ABOVE the originally-planned N=24 stop. The N=24 result hit SLO PASS
but **left only ~0.18 GiB of GPU memory headroom on the 24 GiB L4**, so the
NEXT binding constraint is memory, not GPU compute. The true knee is probably
bracketed at N ∈ [25, 28].

Run a TIGHT bounded sweep on a fresh g6.4xlarge L4, write results to
`proj-2026-05-24-from-scratch-runtime/reviews/B3-FU5b-L4-knee-extend-result.md`,
TERMINATE the instance at the end.

**This task has a CRITICAL termination-discipline fix** (B3-FU-5 lost ~18min of
stopped-state EBS billing due to AWS SSO timeout). See "Termination protocol"
below.
</task>

<context>

## Why this measurement

The B3-FU-5 sweep (commit `f20e46e`) capped at N=24 — passing SLO comfortably
(TTFS p95 37.476ms, lag p95 -110ms) but with only 0.18 GiB GPU memory headroom
of 23.66 GiB total. The true L4 knee is somewhere above 24 but BELOW where
memory caps out. Memory slope = 0.174 GiB/stream.

Projection from N=24 result:
- N=25 peak ≈ 23.47 + 0.174 = ~23.65 GiB (right at cap; OOM possible)
- N=26 peak ≈ 23.65 + 0.174 = ~23.82 GiB (almost certainly OOM)
- N=27+ definitely OOM

So the true L4 knee is probably N=24 or N=25 (memory-bound), not SLO-bound.
Knowing this exactly informs the SageMaker deploy cap:
- L4 cap N=24: $0.80/24 = $0.033/stream/hr (29% cheaper than L40S $0.047)
- L4 cap N=20 (production margin): $0.040/stream/hr (15% cheaper)

The economic case for L4 vs L40S depends on where production caps below the
knee. This sweep tells us the absolute ceiling.

## Run policy

Same as B3-FU-5 except for the N values:

```
NEMOTRON_DENSITY_BATCH_STEADY=1
NEMOTRON_DENSITY_BATCH_MAX=4
NEMOTRON_DENSITY_BATCH_WINDOW_MS=0
NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS=0
--admission-active-cap 10000   (effectively unlimited; sweep tests GPU knee, not admission)
--admission-backlog-cap 10000
--density-start-stagger-ms 10000
sessions_per_worker=2
CUDA_MODULE_LOADING=EAGER
events counted-not-gated (NEMOTRON_GOLD_EVENTS_TOLERANT=1, the default per Fix #8)
```

**Sweep**: N ∈ **{25, 26, 27, 28}** adaptive — TIGHT range because memory is the
binding constraint and there's no point measuring N=32+ if N=27 OOMs.

**Stop conditions** (any one triggers STOP):
1. First SLO fail (`lag_p95 ≥ 500ms` OR `ttfs_p95 > 175ms` OR `ttfs_p99 > 250ms`
   OR `errors > 1%`).
2. **OOM** (CUDA out-of-memory exception OR peak GPU memory ≥ 23.5 GiB safety
   margin under L4's 23.66 GiB cap).
3. N=28 reached and passed → STOP at top of sweep (then mark "true knee >=28
   — exceeded sweep range; another extension needed if N=28 passed cleanly").

## Termination protocol (CRITICAL — fix the B3-FU-5 gap)

B3-FU-5 lost ~18 minutes of stopped-state EBS billing because AWS SSO expired
twice during termination. To prevent recurrence:

1. **Try AWS CLI termination first** (not boto3) — `ec2_down.py` hits boto3 SSO
   issues that the CLI avoids:
   ```bash
   aws ec2 terminate-instances --profile AWSAdministratorAccess-419599258555 \
       --region us-west-2 --instance-ids <ID>
   aws ec2 wait instance-terminated --profile AWSAdministratorAccess-419599258555 \
       --region us-west-2 --instance-ids <ID>
   ```
2. **If SSO has expired**: do NOT try `aws sso login --no-browser --use-device-code`
   from inside Codex (the device flow requires user interaction in a browser, and
   has consistently expired without approval). Instead:
   a. Issue `sudo shutdown -h now` on the guest immediately (stops compute).
   b. Write `cleanup_blocker: "AWS_SSO_NEEDS_USER_REFRESH"` to the state file.
   c. Surface a CLEAR message at the top of the compact_output_contract report:
      `⚠️ AWS SSO EXPIRED — INSTANCE STOPPED BUT NOT TERMINATED.
       Run after SSO refresh: aws ec2 terminate-instances --instance-ids <ID> ...`
   d. Do not retry SSO login.

The instance bill = compute hours × $0.80 while running, then EBS only ~
$0.10/hr while stopped. Surfacing the gap clearly lets the user terminate
in ~30 sec when convenient.

## Bundle source + build

sm_89 AOTI bundles from S3 (reuse B3-FU-5 path; bundle SHA validated in that run).

Container: `nemotron-aoti:cu128`. Build cmd: same as B3-FU-5.

## Output document

Write `proj-2026-05-24-from-scratch-runtime/reviews/B3-FU5b-L4-knee-extend-result.md`:

1. **Verdict** — true L4 knee (last-pass N), failure mode (memory OOM /
   SLO lag / dispatcher CPU), updated $/stream economics vs L40S at the
   confirmed knee.
2. **Sweep table** — same columns as B3-FU-5 (N, SLO PASS/FAIL, ttfs p50/p95/p99,
   lag p50/p95/p99, peak GiB, tok/event/err, admitted/offered, dispatcher CPU%,
   queue p95, B1/B2/B4 dispatches, K2/K3/K4 dispatches, backlog>Bmax).
3. **Memory analysis** — peak GiB per N + confirmed per-stream slope on L4;
   exact OOM point if hit; comparison vs L40S slope (0.173 GiB/stream).
4. **Economics update** — at the realized L4 knee, $/stream/hr; compare to
   L40S; recommend production cap with margin (e.g., knee minus 4 for memory
   safety).
5. **EC2 termination** — confirm AWS state=terminated; report cost. **If SSO
   issue forced stopped-state-only termination, surface as a TOP-LEVEL
   HEADLINE** (not buried).

</context>

<verification_loop>
For each N cell, verify SLO + memory before proceeding:
- Read the JSON telemetry summary.
- Check `slo_robust`, `ttfs_p95_ms`, `lag_p95_ms`, `peak_gpu_memory_GiB`, `errors`.
- If any STOP condition triggers, do NOT continue to higher N.
- **Watch memory CLOSELY**: if peak_GiB approaches 23.5, stop even if SLO holds.
</verification_loop>

<action_safety>
- Bound to ONE `g6.4xlarge`. Do NOT launch additional or larger instances.
- TERMINATE the instance on every exit path per the protocol above. SSO-expiry-
  to-shutdown should take <5 min, not the 30+ min B3-FU-5 spent retrying SSO.
- Use AWS CLI not boto3 for terminate (boto3 SSO is fragile).
- AOTI artifacts gitignored.
- AWS profile: `AWSAdministratorAccess-419599258555`. Region: `us-west-2`.
- SSH key: `ec2-bench/nemotron-bench-key.pem`.
- State file: `ec2-bench/.instance_b3_fu5b_l4.json`.
</action_safety>

<compact_output_contract>
Report (in this order — surface termination first):
- **EC2 TERMINATION**: ✅ AWS-confirmed terminated, OR ⚠️ stopped-not-terminated
  with the exact terminate command for the user to run.
- Path to `reviews/B3-FU5b-L4-knee-extend-result.md`.
- One-paragraph headline: realized L4 knee, binding constraint (memory / SLO /
  CPU), confirmed $/stream vs L40S, recommended production cap.
- Instance: ID, launch/terminate timestamps, cost estimate.
</compact_output_contract>
