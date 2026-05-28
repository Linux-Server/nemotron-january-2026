<task>
Implement **F2-T (telemetry hardening)** — the pre-knee-remeasure follow-up from B2's paired build verdict
(reviews/B2-build-paired-verdict.md). The scheduler's 5+1 timer buckets exist *in memory* but the density JSON
emit currently has only bucket counts, and two of the timer semantics are incorrect. F2-T fixes that + adds
aggregate telemetry the 5090/L40S knee sweeps need for attribution.

Bounded scope: a single PR that lands cleanly with the existing scheduler + density-sweep infrastructure.
Single Opus review post-implementation (paired-review overkill for a bounded telemetry change).
</task>

<context>
**The fixes** (per Codex's B2 build review finding + Opus's pass; both paired):
1. **Emit p50/p95/p99 summaries** for the 5 timer buckets (`gather_wait_us`, `service_wait_us`, `cuda_run_us`,
   `output_sync_us`, `worker_blocked_us`) AND `window_wakeup_jitter_us` into the density JSON sidecar (the
   existing telemetry path). Currently only counts are emitted.
2. **Fix `service_wait_us` semantics**: currently stops BEFORE scratch pack (line ~292 in
   `runtime/cpp/batched_steady_scheduler.cpp:dispatch_batch`). Should be pop → run-start (AFTER pack
   completes). Move the start-time marker.
3. **Fix or clarify `output_sync_us`**: currently measures CPU cost of *enqueueing*
   `cudaStreamWaitEvent` in `record_worker_wait`. Either (a) measure actual device-side wait time
   (record event before wait + sync, compute elapsed), OR (b) rename to `output_sync_cpu_us` and document.
   Recommend (a) for the knee re-measure to have meaningful device-side data; if too invasive, do (b) +
   document.
4. **Add aggregate telemetry**:
   - **Dispatcher CPU%**: sample `pthread_getcpuclockid` / `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` for the
     dispatcher thread; ratio over the measurement window.
   - **Dispatcher stream utilization%**: `sum(cuda_run_us) / wall_clock_window_us`. Data already collected,
     just emit the ratio.
   - **Queue depth p50/p95/p99**: sample `queue_.size()` at each dispatcher gather (under mutex).
   - **Per-stream fairness spread**: per dispatch cycle, `max(worker_blocked_us) - min(worker_blocked_us)`
     across the K streams in the cycle. Emit summary.

**Files to modify**:
- `runtime/cpp/batched_steady_scheduler.h` (add fields for the new telemetry + dispatcher CPU sampling
  state).
- `runtime/cpp/batched_steady_scheduler.cpp` (fix `service_wait_us` start, instrument output_sync,
  dispatcher CPU clock, queue depth sampling, fairness computation).
- `runtime/cpp/density_main.cpp` (emit the new fields into the density JSON sidecar in the existing
  `emit_telemetry` path — currently emits bucket counts in the b2-t1 JSON; add a separate
  `scheduler_telemetry` block in the density-sweep JSON output for both b2-t1 and density-sweep modes).

**SLO + correctness contract** (unchanged from B2):
- No regression in b2-t1: 0 token / 0 event divergences (the SLO signal).
- No regression in OFF-path byte-exactness (scheduler is null when off; no telemetry collected).
- The telemetry additions must be feature-on (scheduler ON only); zero overhead when scheduler OFF.

**Validation**:
- Container build clean.
- Re-run b2-t1 (same 4-row scope from B2; the existing run still passes); verify the new JSON fields are
  emitted with reasonable values (e.g., gather_wait_us p50 < 10ms for the staggered case; dispatcher CPU%
  in [0, 100]; queue_depth p50 plausible).
- Re-run density-sweep N=4 OFF-path smoke; verify no regression + no scheduler telemetry emitted.
- Single Opus review post-implementation (paired-review is overkill for bounded telemetry).

**Out of scope** (these are separate follow-ups F4/F5/F6):
- Don't touch the cosmetic `set_pending_exception_locked` unused `ep` param.
- Don't add EP SHA verification (provenance strengthening, not telemetry).
- Don't change the abandoned-future event cleanup.
</context>

<verification_loop>
Build in the container (`runtime/container/enter.sh bash -lc 'cmake --build cpp/build_b2 --target density_main
-j$(nproc)'`). Re-run b2-t1 with `--correctness-rows 4` (same scope as B2's commit run) + a density-sweep N=4
OFF-path smoke. Inspect the emitted JSON to confirm new fields are present with sensible values. Don't run the
full corpus (still memory-constrained).
</verification_loop>

<action_safety>
Only touch the scheduler + density_main + the JSON emit. Do NOT change the existing telemetry's in-memory
collection (the timer recordings are correct, only the emit is missing); just add summary computation +
emission. Do NOT change b2-t1's case structure or assertions.
</action_safety>

<compact_output_contract>
When done, report:
1. Files modified + the new fields added.
2. The b2-t1 re-run result (0/0 divergences; new JSON fields present).
3. The OFF-path smoke result (no regression).
4. A sample of the new JSON output (one b2-t1 case's `scheduler_telemetry` block).
5. Any spec items you couldn't implement as specified, with the reason.
</compact_output_contract>
