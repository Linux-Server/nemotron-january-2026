# Codex Step 3b Design Review - Round 3

Verdict: **GO-with-1-2-must-folds-to-v4. Not converged yet.** v3 folded the Round 1/2 architecture
blockers: Part A is superseded, server.py audit moved first, library shape is real, Python-exact
/stats is the stated target, shutdown ordering is mostly fixed, and malformed HTTP is specified.
The remaining issues are small, but they hit public contracts Part A will freeze.

## Must Fold To v4

1. **`WireEvent` still cannot represent the final transcript contract.**

   §II defines:
   ```cpp
   std::optional<std::map<std::string, double>> finalize_timing;
   ```
   and no `finalize` field. But §XIV says the oracle compares the final `finalize` flag, and §III
   says exact `finalize_timing` keys are deferred to Part A. Those two facts are incompatible with a
   concrete `map<string,double>` public type today.

   Fold action:
   - Add `std::optional<bool> finalize` to `WireEvent`, or explicitly state that the JSON serializer
     adds Python-only fields outside the DTO. Better: put it in the DTO.
   - Do not freeze `finalize_timing` as `map<string,double>` until the Part A audit proves the Python
     final payload is SLO-metric-only. If Python sends raw timing fields with strings/nulls/bools, use
     a JSON object/variant DTO. If v4 intentionally narrows to the 5 SLO metrics, document that as a
     compatibility decision and make the oracle assert that narrowed shape.
   - Deferring the exact key set to Part A is acceptable only if Part A's first audit happens before
     carving this header.

2. **StatsCollector completion gating is internally contradictory and would undercount.**

   §IV prose says the deque contains complete samples where `vad_stop` and `final_sent` are present,
   with per-metric `count` computed only from samples where that metric is populated. That matches the
   Python pattern. But the class comment says records missing `fork_flush_wall_ms` are incomplete and
   must not append.

   Fold action:
   - Completion predicate: `!was_suppressed && vad_stop_to_sent_ms.has_value()` (or raw
     `vad_stop && final_sent` before derivation). `fork_flush_wall_ms` is optional and participates
     only in its own per-metric count.
   - A complete sample with `emitted=false` is appended and increments `lifetime_suppressed`; it also
     contributes to `suppressed_in_window`.
   - Incomplete/stale/suppressed-before-send finalizes increment lifetime counters only and do not
     enter the deque.

## Minor Folds

3. **Silero VAD needs an execution-budget sentence.** Ownership is right: model in `SharedRuntime`,
   per-session state in `SessionRuntime`. Missing: where inference runs and how thread oversubscription
   is prevented. At 20 ms frames and N=64, this is up to 3,200 VAD eval calls/sec. Specify CPU vs CUDA
   (prefer CPU/off-ASR path unless measured), warm model at startup, use no-grad/inference mode, cap
   Torch intra/inter-op threads or use a small dedicated VAD pool, and expose VAD load failure in
   startup/selftest. `NEMOTRON_VAD_MODE=client_only` must be labeled non-compat/debug if server-side
   VAD is the Python-compatible default.

4. **Shutdown step 5 still has two defaults.** Pick one. Recommended default: on SIGTERM, enqueue a
   server-side `finalize_now(close_reason="shutdown")` for existing sessions, then wait up to
   `NEMOTRON_SHUTDOWN_DRAIN_SEC`; natural VAD-stop-only can be an alternate mode later. Also state
   that the accept/router keeps serving bounded admin requests during drain, while new WS upgrades get
   HTTP 503.

5. **Scheduler close ordering needs an enforceable fence.** §IX says `scheduler.close()` after workers,
   which is right. Add the operational invariant: once drain/force-close begins, workers must stop
   creating new scheduler enqueues before they are considered joined. If timeout hits while scheduler
   work is still outstanding, `close()` must drain or broadcast failure; it must not wait on workers
   that are still allowed to enqueue.

6. **Admin endpoint exposure needs one explicit policy.** `/scheduler_telemetry` and optional
   `/admission` leak queue depth, fairness, and load internals. For v1, either bind admin routes only
   on trusted networks / admin listener, or document "single public listener is only safe behind an
   authenticated LB/API gateway." This is not a protocol redesign, but it cannot remain implicit for
   multi-tenant deployments.

7. **`/health` shape is inconsistent.** §III says `healthy|loading`; §IX says `draining`. Fold the
   full enum in one place, e.g. `loading|healthy|draining|degraded`, and state readiness semantics for
   LBs. Liveness can remain out of scope if the deployment owns it.

8. **PCM frame parsing should avoid endian/alignment traps.** x86 and common aarch64/Spark targets are
   little-endian, but network bytes should still be decoded safely: reject odd binary payload lengths,
   decode/copy to aligned `int16_t` storage, and add a compile-time little-endian assertion or explicit
   byte-swap path. `PCMFrame::count` as samples is good.

9. **picohttpparser is acceptable, but v4 should name the edge cases it does not solve.** It is mature
   enough as a vendored parser, but the server still owns case-insensitive headers, comma-token
   `Connection: keep-alive, Upgrade`, duplicate headers, partial request return codes, max header
   bytes, and rejecting request bodies on upgrade/admin GETs.

10. **Test oracle is close; pin run policy.** `run_compat.py` should allocate or accept two explicit
    ports instead of assuming 8080/8081 are free. Volatile stripping should include timing values,
    timestamps, session/correlation IDs, pid/process_label, sequence IDs, and native scheduler/admission
    counters; do not strip timing key presence or final event order. Make this a Part B pre-merge gate;
    CI can be manual/expensive until artifact runtime is cheap enough.

11. **Config sprawl is acceptable for v1.** The prefixes are coherent. Add one operator-facing summary
    table grouping runtime, admission, stats, WS, shutdown, and VAD knobs. Do not add another config
    source unless deployment already has one.

## Holds

- Part A v1-superseded wording is clear enough for a mechanical auditor: keep CMake/file-move pieces,
  discard protocol/API/stats skeleton choices that predate v3.
- `SessionRuntime(const SharedRuntime& ...)` is buildable if `SharedRuntime` is noncopyable,
  outlives all sessions, and exposes only thread-safe const facades for shared mutable services.
- Server-side Silero in Part B does not require a public API change beyond the execution-budget notes
  above.
- Stats snapshot lock semantics are right: copy under mutex, sort/serialize outside.
- MPS readiness is sufficient at design level: required port, per-process windows, pid/process label,
  LB-owned aggregation.

## Net

No redesign. Fold the `WireEvent`/`finalize_timing` DTO and the StatsCollector completion predicate
before relaunching Part A. Everything else can fold as minor v4 text or implementation guardrails.
