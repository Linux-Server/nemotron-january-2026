<task>
**Tier 3 memory shrink: +11.8 GiB → target <4 GiB scheduler-ON overhead on 5090.**

The B2 scheduler adds a constant +11.8 GiB GPU memory overhead (measured at N=1, 4, 8, 40 — independent of
concurrency; this is the scheduler's FIXED footprint, not per-stream activations). This caps the 5090 at
N=40 with scheduler ON and pushes the L40S closer to its 48GB ceiling than necessary. Shrink target: <4 GiB.
</task>

<context>
**Measured memory footprint** (from reviews/B3-5090-result.md + B3-5090-lowload-result.md):
- OFF baseline at N=1: 12.052 GiB peak.
- ON at N=1, 4, 8, 40: 23.86 / 24.48 / 25.25 / 31.29 GiB peak.
- Delta is **+11.8 GiB constant** across N → it's the scheduler's fixed setup cost.

**Likely sources** (from runtime/cpp/density_main.cpp scheduler construction path + steady_batch_primitive.h):
1. **Production B=1 loader** (`AOTIModelPackageLoader(args.dir + "/enc_steady_aoti.pt2", ...)`) — still loaded
   when scheduler is ON, but NEVER used in the scheduler-routed path (workers go through the scheduler's
   own B=1 bucket). ~2.4 GiB weights + per-runner activation state.
2. **Scheduler's 3 B-bucket loaders** (B=1, B=2, B=4) — already share ONE constants set via user_managed=true
   (good), so weights are ~2.4 GiB total. But each loader has its own per-runner activation buffers + AOTI
   runtime state. ~1-2 GiB × 3 loaders = ~3-6 GiB.
3. **Per-loader auxiliary state** (CUDA streams, scratch tensors, etc.) — small but real.
4. **Dispatcher scratch tensors** (per bucket: chunks/length/cache_ch/cache_t/cache_ch_len) — small (MB-scale).

**The shrink strategies, in order of risk/reward**:

**Strategy 1 — DROP production B=1 loader when scheduler is ON (HIGHEST PRIORITY).**
- The production `enc_steady` loader is loaded but unused in the scheduler-routed path. The scheduler's
  own B=1 bucket handles K=1 dispatches.
- Code change: in density_main.cpp's scheduler-ON construction path, skip loading `enc_steady` (set the
  pointer to nullptr or skip the AOTIModelPackageLoader construction); the worker integration already
  branches on `scheduler != nullptr`, so when scheduler is ON the worker never calls `enc_steady` directly.
- **Caveat**: serial reference construction (`build_serial_reference`) for b2-t1 + density-sweep uses the
  production B=1 to build the gold. So the production loader must still exist for those measurement paths.
  Solution: load it ON-DEMAND for serial reference construction; drop it after the reference is built; OR
  share the scheduler's B=1 bucket for the reference (this only works post A1-outcome-B verification — which
  Codex confirmed: tensors bit-identical, so safe to use scheduler's B=1 for the reference too).
- **Expected savings: ~3-4 GiB.**

**Strategy 2 — Make scheduler's B=1 bucket the SOLE B=1 source.**
- Use scheduler's B=1 bucket for both scheduler-routed K=1 dispatches AND (when scheduler is OFF) direct B=1
  forwards via a thin wrapper that bypasses the dispatcher (no enqueue, no future round-trip).
- Frees the production `enc_steady` entirely: never loaded.
- **Caveat**: changes the OFF-path's B=1 source from PRODUCTION (`enc_steady_aoti.pt2`) to NEW
  (`enc_steady_aoti_b1.pt2`). A1's outcome B already proved these are tensor-bit-identical (max diff 0) but
  SHA-different. So functionally equivalent but a byte-exact-contract change. Document explicitly.
- **Expected savings (additional): ~0 GiB if you're already on Strategy 1 (production loader was the cost);
  but this gets you to ONE B=1 source of truth + simpler code.**

**Strategy 3 — Drop scheduler's B=2 bucket (use B=4 for K=2 with 2 pads).**
- B=2 is only ~3% of dispatch cycles at N=40 B_max=4 (measured: B1=5171, B2=248, B4=3522). Tiny aggregate.
- Padding K=2 to B=4 wastes 50% of compute per K=2 dispatch (~2.5ms × 248 cycles ≈ 620ms aggregate). Tiny.
- **Expected savings: ~1-2 GiB per-loader activation state.**

**Strategy 4 — Drop scheduler's B=1 bucket (use B=4 for K=1 with 3 pads).** ⚠️ NOT RECOMMENDED.
- K=1 is 58% of dispatch cycles at N=40 B_max=4. Huge aggregate.
- Padding K=1 to B=4 wastes 75% of compute per K=1 dispatch (~3.75ms × 5171 cycles ≈ 19 seconds aggregate
  waste). Would significantly regress N=1/N=4 low-load performance (the wrapper overhead measured ~0.3ms
  vs predicted +3.75ms penalty).
- **Don't do this unless Strategies 1-3 fall short AND a perf trade-off is explicitly approved.**

**Approach**: implement Strategy 1 first; measure; if not enough, add Strategy 2 (with the byte-exact
contract change documented); if still not enough, add Strategy 3; STOP before Strategy 4.

**Validation** (bounded local smoke tests on 5090):
- Container build clean.
- b2-t1 4-row: PASS (0 token, 0 errors). The existing baseline.
- density-sweep N=4 OFF smoke: PASS (no regression of OFF path).
- **Low-load sweep (the perf preservation gate)**: N=1, 4, 8 ON+OFF — verify ttfs p95 delta ≤ +1ms vs OFF
  (the prior wrapper overhead was +0.3ms; padding adds shouldn't blow this).
- **Memory measurement at N=1 and N=4**: peak GPU memory before vs after. Report the delta.
- **Goal: scheduler-ON peak memory at N=1 ≤ 16 GiB** (= OFF baseline 12 GiB + <4 GiB overhead).

**Files to touch**:
- `runtime/cpp/density_main.cpp` — scheduler construction path, serial reference construction, optionally
  the scheduler-OFF worker path (if Strategy 2 is needed).
- Possibly `runtime/cpp/steady_batch_primitive.h` — minor (if B-bucket loading needs to be more selective).
- Possibly `runtime/cpp/batched_steady_scheduler.h/cpp` — if changes to bucket policy.

**Out of scope**:
- Don't touch the F2-T telemetry, A4 sealed-loader semantics, A1 parity check.
- Don't change the bidirectional CUDA sync.
- Don't pursue Strategy 4 (B=1 drop) without explicit perf re-justification.
</context>

<verification_loop>
Per strategy iteration: build + run the 3 smoke tests + measure GPU peak memory. Report intermediate
results. STOP early if Strategy 1 + 2 already hit <4 GiB (don't over-shrink at the cost of code complexity).
</verification_loop>

<action_safety>
Local only (5090). Don't disrupt the L40S sweep in flight (codex job bd7rd0m6n — different box). Don't
commit binary artifacts. Keep the production B=1 path code intact for the OFF mode unless Strategy 2 is
adopted with explicit byte-exact-contract documentation.
</action_safety>

<compact_output_contract>
When done, report:
1. Strategies applied + per-strategy memory delta + cumulative scheduler-ON peak at N=1.
2. b2-t1 4-row result (PASS/FAIL, token + event counts).
3. OFF-path smoke result (no regression).
4. Low-load sweep N=1/4/8 result (ttfs deltas vs prior baseline).
5. Whether target <4 GiB was hit; if not, what's blocking (and recommended next strategy).
6. Files modified + lines changed.
7. Any byte-exact-contract changes (if Strategy 2 was needed).
</compact_output_contract>
