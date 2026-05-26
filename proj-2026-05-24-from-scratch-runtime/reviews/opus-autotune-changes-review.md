# Opus — review of the autotune / EP-durability plan changes (one round)

Reviewing the new `## Compile & artifact policy` + Step 1a/1b edits in PHASE2-PLAN.md (autotune-on representative +
off floor; native-per-target; EP→S3 durability/Q1; token re-validation; bounded Step-1a workload).

## Sound (confirmed)
- **Autotune-on for a PERF measurement is the right call** — it's the AOT-time cost that buys runtime speed, and
  it lands on the steady encoder (the Step-0 contention bottleneck). Direction is correct.
- **Native-per-target is correctly reasoned** — `max-autotune` benchmarks on the present GPU, so cross-compile-
  autotune is invalid; sm_120-on-5090 / sm_89-on-L40S is right.
- **EP→S3 durability doubles as the g6e compile source** — since autotune-on-everything ships all EPs to the g6e
  anyway, the 75 GB upload IS the durable backup (Q1 fixed for free). Good synergy.
- **Bounded Step-1a workload** correctly fixes the unbounded-real-time footgun that stalled the killed sweep.

## GAPS / risks to fold (the changes are directionally right but under-specify the cost + the failure modes)

### A1 — Autotune-on must be CONTINGENT on passing T1 re-validation, with an explicit FALLBACK
The plan says "re-validate token-exactness before trusting the density numbers," but doesn't state what happens if
autotune drift **breaks** WER-neutrality (autotune changes reduction order → more near-tie flips than the current
1/1000; it could exceed `WER_bound`). → Make it a GATE with a fallback: **autotune-on artifacts are used ONLY IF
they re-validate T1/WER-neutral; if not, autotune-off becomes the headline** (and we report "autotune numerically
incompatible at the T1 bar"). Without the fallback stated, a failed re-validation leaves the plan undefined.

### A2 — Separate artifact dirs per variant (don't clobber the autotune-off / Step-0 baseline)
`enc_steady_aoti.pt2` (autotune-off, sm_120) is loaded by `session_main` AND the Step-0 harness — it's the
validated baseline. The **5090 sm_120-autotune-ON recompile must write to a SEPARATE path** (e.g.
`artifacts_at_sm120/`), NOT overwrite it, so the floor + the Step-0 baseline + the autotune-on headline coexist
and stay comparable. Same discipline as the `artifacts_sm89/` separation. The policy should state "one artifact
dir per (arch × autotune) variant; never clobber a validated baseline."

### A3 — Autotune compile TIME/COST on the g6e is a real, unflagged cost
`max-autotune` is much slower to compile (it benchmarks many kernel configs per op). **Autotuning the steady +
all 32 finalize buckets on the g6e could be hours of billable g6e time** — far more than the default compile
Codex's prep validated. → Flag it; and note the cheap high-value subset (autotune-on **steady** is where the
density win is; the 32 finalize buckets are once-per-utterance) as the fallback if the autotune-everything compile
balloons cost. (You chose autotune-everything for apples-to-apples — fine — but the g6e-hours should be a known,
not a surprise.)

### A4 — Pin autotune results for reproducibility
`max-autotune` config selection can be non-deterministic / machine-state-dependent. For a *measurement*, the
artifacts must be stable. → Pin/cache the chosen autotune configs (the inductor autotune cache) so the artifacts
are reproducible and a re-run doesn't silently pick different kernels (→ different perf/numerics).

### A5 — Verify the AUTOTUNE compile path works on the DL AMI (heavier than the default compile)
Codex's prep confirmed the *default* AOTI compile runs on a DL AMI + pip-torch-2.8 + system CUDA without the
container. **Autotune is heavier** — it needs Triton autotuning fully working (Triton + nvcc + the autotune
benchmark loop). → The g6e env check must verify the autotune path specifically (a quick autotune-on smoke of one
small bucket) before committing to the 32-bucket autotune run, so we don't discover a broken autotune env mid-run.

### A6 — The L40S autotune WIN is INFERRED from the 5090 unless we add an L40S floor
Step 1a (5090) does the off→on comparison (cheap, local). Step 1b (L40S) is autotune-on ONLY. So the **autotune
density win on the L40S is inferred from the 5090's off→on Δ, not measured** (an L40S autotune-off floor would
need a second 75 GB-EP compile). → State this explicitly: "the autotune win is quantified on the 5090 and assumed
to transfer; an L40S off-vs-on is optional/deferred." (Reasonable — just don't imply the L40S win was measured.)

### A7 — 75 GB S3 transfer is the L40S long pole — flag it
Upload 75 GB from this box → S3 → download to the g6e. Depending on the box's uplink this could be slow (tens of
minutes to hours). → It's the critical-path item for the L40S run; start it first; note it. (And it's the
durable-backup, so not wasted.)

## Internal consistency
No contradictions introduced. Step-1a PASS now references the autotune-on multiplier (consistent); the floor is
reported-not-gated (consistent); the ttfs budget is unaffected (autotune only makes kernels faster → easier to
meet the budget). The new policy section is consistent with Definitions + Rules.

## Verdict
The changes are **directionally sound and execute-able**, but should fold **A1 (contingent-on-revalidation +
fallback)** and **A2 (separate artifact dirs)** before execution — those are correctness/safety. A3–A7 (g6e
autotune cost, reproducibility pin, DL-AMI autotune-path check, L40S-win-inferred, 75 GB long pole) are
flag-and-proceed. With A1/A2 folded, green to execute the parallel steps.
