# Codex Autotune Plan Changes Review

Scope: adversarial read of the new `Compile & artifact policy` plus Step 1a/1b edits in
`PHASE2-PLAN.md` against HEAD.

## Sound

- Autotune-on is the right headline for Phase 2 density. The representative number should be the optimized artifact
  the project would actually deploy, with autotune-off retained as the performance floor.
- Native-per-target compile is the right rule. `max_autotune` benchmarks candidates on the present GPU, so
  cross-compile-autotune would produce an invalid or at least non-representative artifact.
- The 5090 autotune-on recompile is acceptable if it writes a separate artifact set and preserves hashes. The g6e
  native sm_89 compile is necessary for the L40S gate, not optional.
- Preserving EPs durably in S3 is the right Q1 fix. Do not externalize EP weights now; 75 GB is annoying but bounded,
  while changing the export representation now would add correctness and tooling risk before the density gate.
- T1 token/event plus WER-neutral revalidation belongs as a gate. Autotune numeric drift can flip near ties, and no
  density number should be trusted until the exact artifact being benchmarked is correctness-cleared.
- Step 1a's bounded workload requirement is directionally right for spend control; full-corpus-at-real-time is the
  wrong default for a 5090 proxy sweep.

## Defects / Gaps / Risks

1. **DEFECT: the native compile bullet ambiguously calls the floor run autotuned.**
   - Plan lines: `PHASE2-PLAN.md:68-71`.
   - Problem: line 69 says `sm_120-autotuned on the 5090` for "Step 1a headline + the floor re-run." The floor must be
     autotune-off, not an autotuned artifact.
   - Fix: say each target gets two native artifact sets: `<arch>-autotune-on` for headline and `<arch>-autotune-off`
     for floor, both compiled on the target GPU. The 5090 floor is `sm_120` native with autotune off.

2. **DEFECT: the L40S floor comparison is not operationalized in Step 1b.**
   - Plan lines: `PHASE2-PLAN.md:66-67`, `PHASE2-PLAN.md:115-121`, `PHASE2-PLAN.md:126-132`.
   - Problem: the policy defines autotune-off as the floor, but Step 1b only names the sm_89 autotune-on artifact. The
     decisive L40S gate should not leave the floor comparison implicit.
   - Fix: add a bounded sm_89 autotune-off floor run to Step 1b, or explicitly waive it with rationale. Define the
     win as same target, same EPs/model, same torch/CUDA/driver, same `n-values`, same
     `--density-sessions-per-worker`, same cadence/warmup/topology, and distinct artifact hashes. Report absolute
     SLO-robust streams/box delta and percent (`S_on` vs `S_off`), not only each artifact's multiplier over its own
     N=1. PASS remains based on autotune-on.

3. **RISK: autotune compile failure/time is understated.**
   - Plan lines: `PHASE2-PLAN.md:63-65`, `PHASE2-PLAN.md:68-71`, `PHASE2-PLAN.md:126-127`.
   - Problem: "No OOM concern (AOT)" is true for benchmark runtime, but max-autotune can still consume compile wall
     time, disk, GPU memory, and fail on g6e due to toolchain/driver/Triton config.
   - Fix: add compile acceptance criteria: timeout/retry policy, compile log capture, torch/CUDA/driver/Triton/
     Inductor config, cache state, and package SHA256. If autotune-on compile fails, do not silently substitute the
     off artifact for the headline; mark compile-blocked and use off only as diagnostic floor.

4. **RISK: autotune reproducibility is not pinned enough for an off->on claim.**
   - Plan lines: `PHASE2-PLAN.md:66-79`, `PHASE2-PLAN.md:115-121`.
   - Problem: correctness revalidation catches semantic drift, but the density delta can still be noisy if compile
     benchmarking chose a config under thermal/load/cache variance.
   - Fix: require the normal repeated-run stability bar for both headline and floor before reporting the off->on win
     as a finding, and log whether compilation used a warm or cold Inductor cache.

5. **GAP: S3 durability needs a manifest, not just a bucket.**
   - Plan lines: `PHASE2-PLAN.md:72-76`.
   - Problem: "Back them up to S3" does not define what makes the EP set restorable or auditable.
   - Fix: define the S3 prefix and archive a manifest with object path, byte size, SHA256, generating commit/command,
     model and fixture hashes, and the expected 32-bucket contract keys. The one-command regen should fail closed if
     the regenerated bucket set differs.

6. **GAP: token-exact revalidation should bind to artifact identities.**
   - Plan lines: `PHASE2-PLAN.md:77-79`, `PHASE2-PLAN.md:126-127`.
   - Problem: the gate is right, but it should explicitly run against the exact package hashes used for density.
   - Fix: require T1/E.2-style revalidation on the final sm_120 autotune-on and sm_89 autotune-on packages before
     Step 1a/1b density numbers are trusted. Prefer running the autotune-off floor artifacts through the same check
     as diagnostic coverage. A revalidation failure is a correctness STOP/recompile, not a perf caveat.

7. **GAP: the bounded Step 1a workload lacks a minimum.**
   - Plan lines: `PHASE2-PLAN.md:115-121`.
   - Problem: `--density-sessions-per-worker` prevents the 30min/N failure mode, but an arbitrary small bound can make
     WER, TTFS, and the off->on delta too noisy.
   - Fix: pre-register a minimum sessions-per-worker and repeat count for headline and floor, and use the same bound
     for both artifacts. Full-corpus real-time can remain deferred.

## Internal Consistency

No new threshold or TTFS inconsistency found. The `ttfs` budget is still defined in `Definitions`, G2 spread remains
reported rather than a STOP criterion, and Step 1a PASS now correctly keys off the autotune-on artifact. The only
textual inconsistency is the floor/native wording in item 1.

## Verdict

The plan changes are conceptually sound to execute: autotune-on headline, autotune-off floor, native-per-target
compilation, EP durability, and artifact-specific correctness revalidation are the right calls.

Apply the small fixes above before using the numbers, especially the floor wording, L40S floor protocol, compile
manifest, and S3 manifest. Without those, the headline direction is still right, but the off->on win and g6e artifact
provenance are underdefined.
