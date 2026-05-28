# Step B2 build — paired adversarial verdict (2026-05-27)

**Folds:** `opus-B2-build-review.md` + `codex-B2-build-review.md` (two independent audits of the B2
implementation against the binding spec `B2-design-paired-verdict.md` §II, written without seeing each
other).

## Verdict

**PASS-with-followup. Commit B2.** Both reviews converged: no design-fork retreat, no correctness blocker.
The spec is faithfully implemented at the correctness level (§II.2 bidirectional CUDA sync, §II.4 explicit
nullable integration, §II.10 A4 sealed-loader fail-closed, §II.11 scratch confinement, §II.12 manifest
fail-closed). The b2-t1 result is the SLO signal — 0 token / 0 event divergences across 6 cases (including
the §II.6 forced-concurrency case that formed 20 actual B=4 batches + the §II.6 Bmax=1 control). A1 outcome
B handled per §II.9. OFF-path preservation by structural non-construction.

## Convergence (both reviewers, independently)

1. **Bidirectional CUDA sync (§II.2) is correctly implemented.** Producer event on worker stream before
   enqueue; `cudaStreamWaitEvent(dispatcher, producer)` before pack; event destroyed immediately after wait
   (correctly placed — dependency already enqueued on dispatcher stream); completion event recorded on
   dispatcher stream after unpack; worker waits before consume.
2. **A4 sealed-loader discipline enforced** (`steady_batch_primitive.h:403-409, 548-556`).
3. **A5 scratch confined to dispatcher stream**, pad rows duplicate row 0, workers only receive per-row
   `.contiguous()` copies.
4. **A7 manifest fail-closed at primitive constructor** — built-in SHA256 + JSON parser correctly handle
   the emitted schema; throws on missing file / malformed / missing bucket / package SHA mismatch /
   shared-weight SHA mismatch.
5. **No hidden global scheduler.** `run_steady_chunk_density()` takes a nullable `BatchedSteadyScheduler*`;
   reference / B1 prep / finalize parent prep / warmup all pass nullptr.
6. **A1 outcome B correctly handled** — SHAs differ, tensors bit-identical; OFF stays on PRODUCTION B=1;
   scheduler-ON path uses NEW B=1 and `b2-t1` case 1 verifies that path is token-equivalent.
7. **OFF-path preservation** — when batch_steady=off, scheduler is *never instantiated* (not just unused),
   the production `enc_steady_aoti.pt2` is used unchanged.
8. **Per-row completion event** is functionally equivalent to one event per batch (all rows complete at the
   same point on the same stream); minor waste, not a correctness issue.
9. **No mutex race** between `enqueue` and `gather_batch` (shared `mutex_`; capacity wakeups after pops).
10. **Fault path acceptable** — outer try/catch + set fault + propagate to pending futures + log + exit(1);
    a worker timeout that races dispatcher fault is benign (both terminate the process).

## Distinctive findings — Codex caught what I missed

1. **Telemetry semantics caveat (Codex F2 telemetry-hardening).** The 5 timer buckets EXIST in
   `BatchedSteadySchedulerTelemetry` per §II.7, but the measurements are not fully faithful to what the
   knee re-measure needs:
   - `service_wait_us` stops BEFORE scratch pack — so it's not true pop-to-run-start, it's pop-to-pack-
     start.
   - `output_sync_us` measures CPU cost of *enqueueing* `cudaStreamWaitEvent`, not actual device-side wait.
   - Density JSON emits bucket counts but NOT timer summaries, dispatcher CPU%, stream utilization, queue
     depth, or fairness spread.
   **Action for the knee re-measure (F2-T):** emit p50/p95/p99 for the 5 timers + dispatcher CPU% + stream
   utilization + queue depth + per-stream fairness spread; fix `service_wait_us` semantics to include pack;
   either fix `output_sync_us` to measure device-side wait OR document the CPU-cost framing.
2. **Forced-concurrency test race.** The barrier is BEFORE enqueue, not "all-items-queued-before-dispatcher-
   can-pop." With default `lone_timeout_ms=0`, the dispatcher could wake on the first enqueue before peers
   enqueue → could race to B=1. The bucket assertions catch it as a FAIL (not a false pass), and the
   reported run did form B>1/B4 batches, but for deterministic repeatability use a tiny test-only
   `lone_timeout_ms` (e.g., 5ms) OR add a stronger test-only enqueue gate.
3. **Abandoned-future event cleanup.** If a worker's `future.wait_for(timeout)` fires before the dispatcher
   completes the item, the completion event leaks (the worker is gone, no one destroys it). Error-path
   hygiene, not a normal-path blocker.
4. **A7 EP SHA not separately verified.** The C++ verifier checks `package_sha256` (the critical runtime
   path) but not `ep_sha256`. Since runtime executes the AOTI package, package SHA is the binding check;
   EP SHA would only strengthen provenance.

## Distinctive findings — Opus

1. **Memory headroom telemetry for the knee re-measure (F2-M, parallel to Codex's F2-T).** Codex's OOM at
   full-corpus b2-t1 is a real signal that B2 adds memory overhead the existing 5090 N=40 knee budget needs
   to accommodate. The §II.13 knee sweep MUST report scheduler-ON vs OFF peak-memory delta per N, so if
   N=40 isn't feasible with scheduler ON, the lift projection's denominator is wrong.

## Scope reductions both accepted

- **b2-t1 ran 4 reference rows, not 1000** — due to OOM at the full corpus. Both reviewers: acceptable for
  B2 acceptance (6 cases × multi-stream concurrent exercise × all 3 buckets × 0/0 divergences); flag a
  full-corpus-equivalent run as a **B3 pre-condition** (Codex F1 = Opus F1).
- **OFF-path smoke at N=4, 20 sessions** — acceptable given the B1 commit's full 1000/1000 production-B=1
  guarantee remains untouched.

## Folded follow-up list (none block B2 commit)

| ID | Item | When | Source |
|---|---|---|---|
| F1 | Full-corpus b2-t1 (split-case / fresh-process / streamed-reference) | B3 pre-condition | Both |
| F2-T | Telemetry hardening: timer p50/p95/p99 summaries, dispatcher CPU%, stream util, queue depth, fairness spread; fix service_wait semantics; clarify output_sync_us | Pre-knee-remeasure | Codex |
| F2-M | Scheduler-ON vs OFF peak-memory delta per N in the knee sweep | The knee re-measure (§II.13) | Opus |
| F3 | Test hardening: deterministic forced-concurrency (tiny test-only lone_timeout OR stronger enqueue gate) | Pre-B3 | Codex |
| F4 | Cosmetic: drop unused `ep` param in `set_pending_exception_locked`; consider nonblocking `cudaEventElapsedTime` plumbing | Anytime | Both |
| F5 | EP SHA verification in C++ loader (strengthen provenance, not correctness) | Anytime | Codex |
| F6 | Abandoned-future completion-event cleanup (error-path hygiene) | Anytime | Codex |

## Net

PASS B2. Commit. Mark `[x]` in PHASE2-PLAN.md. F1, F2-T, F3 are pre-B3 prep; F2-M is part of the §II.13
knee re-measure (the next real-work task). F4/F5/F6 are anytime-cleanup.

The B2 commit unblocks the **5090 knee re-measure** — that's where the projected N=40 → 47+ lift is either
demonstrated or not (and where F2-M memory headroom is answered).
