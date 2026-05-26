# Autotune / EP-durability plan changes — paired review FOLDED (Codex + Opus)

One round on the new `Compile & artifact policy` + Step 1a/1b edits. Strong convergence: direction is sound
(autotune-on headline + off floor + native-per-target + EP/S3 durability + token re-validation), with a text bug
and several under-specifications to fold before executing.

## Both agree (direction sound)
Autotune-on is the right headline for a perf/density measurement; autotune-off is the floor; native-per-target is
correctly reasoned (max-autotune benchmarks on the present GPU). **Both explicitly agree: do NOT externalize EP
weights now** — 75 GB is annoying but bounded; changing the export representation pre-gate adds correctness/tooling
risk. Back up as-is + log the tech-debt. T1 re-validation belongs as a gate. Bounded Step-1a is right.

## Fixes to fold (consolidated)
1. **TEXT BUG (Codex #1, Opus A2) — the floor is autotune-OFF, not "autotuned."** My policy line said
   "sm_120-autotuned … headline + the floor re-run." → Each target gets TWO native artifact sets:
   `<arch>-autotune-ON` (headline) + `<arch>-autotune-OFF` (floor), BOTH compiled natively on the target GPU.
2. **Separate artifact dir per (arch × autotune) variant; never clobber a validated baseline (Opus A2).**
   `enc_steady_aoti.pt2` (autotune-off sm_120) is used by `session_main` + the Step-0 harness — the autotune-on
   recompile writes a SEPARATE path (e.g. `artifacts_at_sm120/`, `artifacts_sm89/{on,off}`), preserving SHAs.
3. **Autotune-on is CONTINGENT — make the gates + fallbacks explicit (Codex #3/#6, Opus A1):**
   - **T1 re-validation binds to the EXACT package SHA being benchmarked**; a T1/WER-neutral FAIL is a
     **correctness STOP/recompile, NOT a perf caveat**.
   - If the autotune **compile** fails (toolchain/Triton/driver/timeout) → mark **compile-blocked**; use the
     autotune-off artifact **only as diagnostic floor — do NOT silently substitute it as the headline**.
4. **Operationalize the L40S floor (Codex #2, Opus A6):** add a bounded sm_89 **autotune-OFF** floor run to
   Step 1b, OR explicitly waive it (rationale: win inferred from the 5090). Define the **win rigorously**: same
   target / same EPs+model / same torch+CUDA+driver / same `n-values` / same `--density-sessions-per-worker` /
   same cadence+warmup+topology / **distinct artifact hashes**; report **absolute SLO-robust streams/box Δ and %**
   (`S_on` vs `S_off`), not each artifact's own multiplier-over-its-own-N=1. PASS still keys off autotune-on.
5. **S3 needs a MANIFEST, not just a bucket (Codex #5):** define the S3 prefix + archive a manifest (object path,
   byte size, SHA256, generating commit/command, model+fixture hashes, the 32-bucket contract keys). The
   one-command regen **fails closed** if the regenerated bucket set differs from the contract.
6. **Reproducibility for the off→on claim (Codex #4, Opus A4):** require the repeated-run stability bar (CV) for
   BOTH headline + floor before reporting the win; **pin/cache the autotune configs**; log warm/cold Inductor
   cache state.
7. **Bounded Step-1a needs a MINIMUM (Codex #7):** pre-register a minimum `--density-sessions-per-worker` +
   repeat count (same bound for headline + floor) so WER/TTFS/the off→on Δ aren't noise. Full-corpus-real-time
   stays deferred.
8. **Autotune compile cost/path (Opus A3/A5, Codex #3):** flag the g6e billable hours for autotuning 32 buckets;
   add compile acceptance criteria (timeout/retry, log capture, torch/CUDA/driver/Triton/Inductor config, cache
   state, package SHA256); **smoke the autotune path on the DL AMI with one small bucket** before the full
   32-bucket run (autotune is heavier than the default compile Codex's prep validated).
9. **75 GB S3 transfer is the L40S long pole (Opus A7)** — start it first; it doubles as the durable backup.

## Internal consistency
No threshold/ttfs/G2 inconsistency (Codex + Opus agree); the only textual error was the floor/native wording
(fix #1). ttfs budget unaffected (autotune only makes kernels faster → easier to meet).

## Verdict
Changes are sound to execute once #1 (floor=off) + #3 (contingency/fallback) + #2 (artifact-dir separation) +
#4 (L40S floor protocol) + #5 (S3 manifest) are folded into the plan. #6–#9 are flag-and-proceed (record them in
the plan). Applying these to PHASE2-PLAN.md now, then parallelizing execution.
