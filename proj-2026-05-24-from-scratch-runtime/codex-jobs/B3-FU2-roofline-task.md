<task>
**B3-FU-2-ROOFLINE: low-level profiling of the at-the-knee batched-steady configuration on a parallel
L40S instance.**

We have the OUTCOME (B3 v5: knee ≥64 staggered, ttfs p95=21ms, ~8× SLO headroom, 1.6-2.66× density lift)
but we don't have the MECHANISM CHARACTERIZATION at the realized operating point. The prior roofline
profiling (`reviews/roofline-COMBINED.md` / `reviews/profiling-paired-verdict.md` /
`reviews/{opus,codex}-l40s-profiling-analysis.md` from earlier this session) was **single-stream B=1**.
We're now running **multi-stream B=4 batched**, which has a fundamentally different roofline footprint.
This task closes the gap.

Spin a SEPARATE g6e.8xlarge L40S **in parallel with the in-flight B3-FU-1 sweep** (codex job `bm2ufeguw`,
different instance). Run nsys + ncu profiling. Compare against the prior single-stream B=1 baseline.
Output a structured roofline analysis. Terminate at end.

**Cost**: ~1-1.5 hour spin × $2.50-4/hr = ~$4-6.
</task>

<context>
**The prior baseline** to compare against — `reviews/profiling-paired-verdict.md` (this session's earlier
work, with the 5090 single-stream B=1 measurement):
- ampere_sgemm: DRAM 71-72%, SM 34-39%, achieved occupancy 15-17%, mem-BW-bound.
- 80% GPU time in SGEMM; conformer encoder weight-streaming dominates.
- Roofline verdict: BW-bound at single-stream; theoretical lift via batching = amortize weight load.

**Current state to characterize** (post-B2 scheduler + Tier 3):
- L40S knee ≥64 staggered with B_max=4 W=0 L=0.
- Dispatcher CPU 60.96%, queue depth p95=22, stream util 60.75%.
- Per-row encoder time dropped 2.6× (microbench) at B=4.
- Hypothesis: at B=4 the per-kernel arithmetic intensity rises ~4×, shifting toward compute-bound;
  multi-stream pressure should saturate DRAM at the aggregate.

**Three profiling cells**:

### Cell 1: nsys timeline at N=64 staggered (B_max=4 W=0 L=0)
- Run `density_main --mode density-sweep --n-values 64 --batch-steady on --batch-b-max 4
  --batch-window-ms 0 --batch-lone-timeout-ms 0 --density-start-stagger-ms 10000` + nsys wrapped.
- Capture window: `--delay=255s --duration=45s` (post-oracle, post-warmup, in measured gate; tune from
  the run's DENSITY_PHASE_TIMING lines per the runbook).
- Outputs: `nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum`.
- Specifically extract: dispatcher-stream kernel time%, worker-stream kernel time%, host-side launch
  API time %, gaps between batches, kernel-launch rate, total kernels executed.

### Cell 2: nsys timeline at N=88 (likely-interior-passing) staggered same policy
- Same setup as Cell 1, N=88. If B3-FU-1 reports the knee is below 88 by the time you run this, fall
  back to N at 80% of the realized knee (interior point in the production-relevant regime).
- Why: characterize whether dispatcher saturation has progressed; queue depth/gather wait should be
  bigger; cross-stream contention more visible.

### Cell 3: ncu on ampere_sgemm at B=4 batched (single-stream)
- Use a fixture that exercises the B=4 batched steady forward in isolation (the existing
  `runtime/cpp/steady_batch_bench.cpp` from STEADY-BATCH-0 already loads B=1/2/4 AOTI + runs a
  microbench; OR ncu the `--mode density-sweep --n-values 4` density_main run to capture B=4 cycles).
- ncu pattern from the existing `runtime/run_l40s_density.README.md` (updated this session): needs
  `sudo`; pass env via `sudo env LD_LIBRARY_PATH=... SELF_CHECK_ATOL=0.2 ...`. **Don't** set
  `DENSITY_GOLD_EVENTS_TOLERANT=1` — Fix #8 default is tolerant.
- Counters: roofline set (`--set roofline`), launch-count capped (`--launch-count 40-100`), full kernel
  metrics on `ampere_sgemm_64x32_sliced1x4_tn` (the dominant kernel in the B=1 baseline).
- Extract: DRAM throughput%, SM throughput%, achieved occupancy, achieved FP32 FLOPS, achieved
  FLOPS/byte vs machine balance (~104 FLOP/byte on L40S 864 GB/s + 91 TFLOPS).
- COMPARE to the B=1 baseline numbers (DRAM 71-72%, SM 34-39%, occupancy 15%): expectation = SM
  throughput rises (more work per kernel-launch), DRAM throughput may DECREASE per-call (per-stream
  but aggregate may rise), occupancy rises (bigger batch geometry exercises more SMs).

### Cell 4 (STRETCH — only if Cells 1-3 fit time/cost budget): multi-stream ncu at N=64
- The "1c-B gap" from PHASE2-PLAN — actual cross-stream contention.
- ncu --replay-mode=kernel adds significant overhead; limit to 10-20 kernel launches at N=64.
- This validates whether multi-stream batched is closer to DRAM saturation at the aggregate.

**Stages**:
1. Spin fresh `g6e.8xlarge` in `us-west-2` (profile `AWSAdministratorAccess-419599258555`). Write
   instance ID to `ec2-bench/.instance_b3_fu2_profile_l40s.json`.
2. Provision (Python venv + torch 2.8.0+cu128 + nsight-systems-2024.6.2 via apt;
   `chmod o+rx /opt/nvidia /opt/nvidia/nsight-systems` per the runbook).
3. Pull pre-built sm_89 artifacts from S3 (`s3://nemotron-phase2-eps-419599258555/density/
   steady_b_artifacts/`).
4. Native cmake build of `density_main` + `steady_batch_bench` (post-Tier-3 / post-F2-T / post-Fix-#8
   HEAD; same binary stack as B3-FU-1).
5. Run Cells 1-3 (4 if budget allows). Output per-cell summaries + the raw nsys/ncu artifacts.
6. **TERMINATE the L40S at end** (cost discipline).

**Output deliverable**:
- `runtime/artifacts/b3_fu2_profile_logs/` — raw nsys_stats.txt, ncu_sgemm.csv, etc.
- `reviews/B3-FU2-roofline-result.md` with:
  - The per-cell summary tables (nsys: kernel-time%, launch counts, stream util, dispatcher activity;
    ncu: DRAM/SM/occupancy/FLOPS metrics per kernel).
  - **The roofline comparison**: single-stream B=1 baseline (cited from
    `reviews/profiling-paired-verdict.md`) vs multi-stream B=4 batched (this measurement). Per-kernel
    DRAM throughput shift, SM throughput shift, occupancy shift, arithmetic intensity shift.
  - **The interpretation**: are we BW-bound, compute-bound, dispatcher-bound, or sync-bound at the knee?
    What's the theoretical headroom for further lift?
  - **Comparison to prior roofline-COMBINED conclusions**: did the BW-amortization mechanism deliver
    the predicted shift? Are there surprises?
- Update `ec2-bench/.instance_b3_fu2_profile_l40s.json` with terminated state.

**Out of scope**:
- Don't run a density sweep beyond what's needed for the profile captures (this is profiling, not a
  knee search).
- Don't re-run F1 b2-t1 (already done).
- Don't touch the in-flight B3-FU-1 instance (different box; parallel).
- Don't pursue extensive ncu sweeps — bounded scope.
</context>

<verification_loop>
Native build on the L40S. Pre-validate nsys + ncu tooling work BEFORE the actual measurement runs:
small smoke (nsys profile a 10s window of a trivial run; ncu on a single kernel) to confirm permissions /
LD_LIBRARY_PATH / sudo env wiring per the existing runbook documents in `run_l40s_density.README.md`.

If a measurement cell fails (permission error, OOM, kernel mismatch), STOP, record, skip — don't poison
subsequent cells. The deliverable is whatever measurements we can land + a clear note on what couldn't.
</verification_loop>

<action_safety>
- TERMINATE the L40S at end (cost discipline). Don't leave running.
- DON'T touch the B3-FU-1 L40S instance (codex job bm2ufeguw, different box, in flight).
- Sync code from repo; don't push back.
- Don't commit binary artifacts (.nsys-rep, .ncu-rep — large; gitignored). DO commit/sync the .txt /
  .csv exports + the result md doc.
</action_safety>

<compact_output_contract>
When done, report:
1. EC2 instance ID + IP + cost (launch → terminate elapsed × rate).
2. Build result + tooling smoke (nsys + ncu) result.
3. Per-cell measurement summary: Cell 1 (nsys @ N=64), Cell 2 (nsys @ N=88-or-fallback), Cell 3 (ncu
   single-stream B=4), Cell 4 (stretch multi-stream ncu).
4. **The roofline comparison table**: single-stream B=1 baseline vs multi-stream B=4 batched (DRAM/SM
   throughput, occupancy, FLOPS/byte, kernel-launch rate, etc.).
5. **The interpretation**: BW-bound / compute-bound / dispatcher-bound / sync-bound at the knee, with
   the data backing the verdict + the theoretical lift headroom remaining.
6. Path to `reviews/B3-FU2-roofline-result.md`.
7. EC2 termination confirmation.
</compact_output_contract>
