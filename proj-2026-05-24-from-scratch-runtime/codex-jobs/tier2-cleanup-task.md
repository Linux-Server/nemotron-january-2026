<task>
**Tier 2 cleanup: B2/B3 carry-over follow-ups F4 + F5 + F6.** Bounded hygiene work, no architecture
changes. Single bounded commit when done.
</task>

<context>
From `reviews/B2-build-paired-verdict.md` follow-ups:

**F4 — cosmetic cleanup**:
- `runtime/cpp/batched_steady_scheduler.cpp` line 397: `set_pending_exception_locked(std::exception_ptr ep,
  std::vector<...>* pending)` — the `ep` parameter is unused (`(void)ep;` discards it). Remove the parameter
  from the signature + update callers (close + dispatcher_loop catch). The actual exception is set by the
  caller via `set_item_exception(item, ep)` after the lock is released. Dead arg.
- `runtime/cpp/batched_steady_scheduler.cpp` `dispatch_batch`: `cudaEventSynchronize(ev_stop)` (line ~302)
  blocks the dispatcher thread synchronously for the CUDA timing measurement. Consider nonblocking timing:
  defer the `cudaEventElapsedTime` to a later poll, OR keep `cudaEventSynchronize` but make it a debug-only
  measurement path. For now: keep the sync (the measurement is useful) but add a comment explaining the
  trade-off + leaving a TODO marker.

**F5 — EP SHA verification in C++ loader**:
- `runtime/cpp/steady_batch_primitive.h` `verify_manifest()` currently verifies `package_sha256` +
  `shared_weight_sha256`. The manifest also contains `ep_sha256` (the ExportedProgram SHA). Add EP SHA
  verification: for each bucket, compute the SHA of the `enc_steady_t2a_b{B}.pt2` ExportedProgram file
  (alongside the .pt2 AOTI package) and verify it matches the manifest. If the EP file isn't present (in
  some deployment configurations), gracefully skip with a log line. Strengthens provenance; the runtime
  doesn't execute the EP (only the AOTI package), so the manifest's EP SHA is informational + audit.

**F6 — abandoned-future event cleanup**:
- `runtime/cpp/batched_steady_scheduler.cpp` `dispatch_batch` line ~322-336: creates a
  `cudaEventCreateWithFlags(&completion, cudaEventDisableTiming)` per row and embeds it in the
  `DispatchResult`. If the worker's `future.wait_for(timeout)` fires BEFORE the dispatcher delivers, the
  worker has no result to consume, and the completion event leaks (no one destroys it).
- Fix: add a destructor or RAII wrapper for the completion event in `DispatchResult`. When the
  `DispatchResult` is dropped without consumption, the event is destroyed. Alternative: a
  `std::shared_ptr<cudaEvent_t>` with a custom deleter; both approaches are bounded.
- Verify: the worker destroys the event correctly on the success path (currently destroyed after
  `cudaStreamWaitEvent` per the integration in density_main.cpp:1263-1271 — confirm + ensure RAII handles
  the worker's destroy path identically).

**Validation**:
- Container build clean.
- b2-t1 4-row PASS (0 token, 0 events).
- OFF-path smoke (N=4) PASS.
- A simple memory-leak / event-leak sanity: monitor `nvidia-smi --query-gpu=memory.used` before + after a
  b2-t1 run — should return to baseline within a few MB.

**Out of scope**:
- Don't touch the scheduler architecture or the bidirectional CUDA sync.
- Don't change the manifest schema or the export script (just add C++-side EP verification).
- Don't pursue the nonblocking-timing rewrite of dispatch_batch (too invasive for cleanup); just document.
</context>

<verification_loop>
Build + b2-t1 4-row + OFF smoke. Run the b2-t1 in a loop 2-3 times + check memory baseline stays stable
(event-leak sanity).
</verification_loop>

<action_safety>
Local only. Don't disrupt the L40S sweep (different box). Bounded cleanup; commit-ready.
</action_safety>

<compact_output_contract>
When done, report:
1. Files modified + line counts.
2. F4: signature change + callers updated; b2-t1 still PASS.
3. F5: EP SHA verification added + tested (with and without EP file present).
4. F6: completion event RAII added + verified no leak in 2-3 b2-t1 loops.
5. Build + smoke results.
</compact_output_contract>
