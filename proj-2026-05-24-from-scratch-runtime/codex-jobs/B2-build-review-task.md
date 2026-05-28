<task>
Adversarial paired BUILD review of Step B2 (the batching scheduler implementation just committed-pending). The
DESIGN was already paired-reviewed (verdict at `reviews/B2-design-paired-verdict.md` §II is the binding spec).
Your job here is to audit the IMPLEMENTATION against the spec + look for bugs, unsafe edges, and missed cases.
Write your fold-ready review to `proj-2026-05-24-from-scratch-runtime/reviews/codex-B2-build-review.md`.

**The user's call (2026-05-27):** PASS-by-policy if (a) the spec is faithfully implemented, (b) 0 token
divergences in b2-t1, (c) OFF-path byte-exactness preserved, (d) no design-fork retreat. Otherwise flag the
fix and we iterate.
</task>

<context>
**Files implemented or modified for B2** (read in full):
- `runtime/cpp/batched_steady_scheduler.h` (NEW, 148 lines).
- `runtime/cpp/batched_steady_scheduler.cpp` (NEW, 437 lines).
- `runtime/cpp/steady_batch_primitive.h` (substantially modified — now 631 lines with built-in SHA256 +
  JSON parser + manifest verify).
- `runtime/cpp/density_main.cpp` (modified — new policy parsing, args, b2-t1 mode, scheduler integration
  at `run_steady_chunk_density`, A1 parity check).
- `runtime/export_steady_batched.py` (modified — emits `runtime/steady_b_artifacts/MANIFEST.json`).
- `runtime/cpp/CMakeLists.txt` (modified — adds the new .cpp).

**The binding spec** to check faithfulness against:
- `proj-2026-05-24-from-scratch-runtime/reviews/B2-design-paired-verdict.md` §II.1 through §II.14.
- Also read `reviews/{opus,codex}-B2-design-review.md` for context on the design fork + the original
  attacks.

**Codex's build report** (already received):
- All 6 b2-t1 cases PASS, 0 token divergences, 0 event divergences over 4 reference rows × 6 cases.
- Bucket coverage: `single_stream`=B1=215; `K2_B2`=B1=20/B2=10/backlog=4; `K3_padded`=B1=7/B2=7/B4=13/K3=13;
  `forced_B4`=B1=2/B4=20/K3=2/K4=18/backlog=1; `staggered`=B1=80; `Bmax1_control`=B1=80/backlog=59.
- Max diffs: enc_out 7.83e-5 / cache_ch 1.22e-4 / cache_t 1.83e-2 — all *smaller* than B1's 8.63e-4/5.23e-3/
  9.40e-2 (good — compound multi-chunk drift is BETTER than single-chunk-insert in B1).
- A1 parity outcome **B** (SHAs differ, tensors bit-identical max=0) → OFF path stays on PRODUCTION B=1 per
  §II.9 outcome B.
- OFF-path smoke at N=4, 20 sessions, NEMOTRON_DENSITY_BATCH_STEADY=0 → mismatches=0, errors=0,
  serial_oracle_match_pass=true.
- **Scope reduction Codex flagged honestly**: b2-t1 used 4 reference rows (not 1000) due to memory pressure
  on the 5090; the OFF smoke was 20 sessions (not 1000).

ASK / structure the review:
1. **Scheduler implementation correctness** — read `batched_steady_scheduler.h` + `.cpp` in full:
   - §II.2 bidirectional CUDA sync: verify dispatcher waits on each producer event before pack/run AND the
     dispatcher records a completion event after unpack that the worker waits on. The implementation does
     `cudaStreamWaitEvent(dispatcher, producer)` then destroys the producer event right after — is the
     destroy correctly placed (the recorded dependency persists after destroy per CUDA docs)? Verify.
   - §II.7 telemetry split: are all 5+1 distinct buckets recorded (gather, service, cuda_run, output_sync,
     worker_blocked, jitter)?
   - §II.8 fault tolerance: dispatcher loop's try/catch + set fault + propagate exceptions + process exit(1).
     Reasonable for the runtime? Edge case: what if the worker future timeout (`future_timeout_ms()`) fires
     BEFORE the dispatcher's fault propagates? Worker would throw a timeout exception; dispatcher would
     still be in faulted state. Race? Is this benign or a bug?
   - §II.10 A4 sealed_ + fail-closed get(): correctly enforced?
   - §II.11 A5 scratch + index_copy_: verify scratch tensors are confined to dispatcher_stream (the
     `CUDAStreamGuard` in dispatch_batch covers index_copy_); verify pad-row src.chunk == ready[0]'s
     (correctly handled by the `row < ready.size() ? row : 0` indexing)?
   - **Per-row completion event vs one event per batch**: Codex creates an event PER ROW (line 322-324),
     even though all rows' work completes on the same dispatcher_stream at the same point. Is the per-row
     event necessary, or is it correctness-equivalent to one event per batch (and just slightly wasteful)?
   - The `cv_capacity_` wakes on close/fault but the `wait` predicate is `closing_ || fault_ || size < capacity`
     — does an enqueue blocked on capacity correctly fail-fast when the dispatcher dies?
   - Any race between `enqueue` (taking mutex_) and `dispatcher_loop` (also taking mutex_ inside
     `gather_batch`)?
2. **density_main.cpp integration** (focus on lines 1189-1290 + 1734-? for the scheduler wiring; the b2-t1
   case dispatcher + policy parsing):
   - §II.4 no globals: verify the scheduler is passed as a pointer parameter to `run_steady_chunk_density`,
     NOT a global; `nullptr` → unchanged B=1 path; reference/B1 T1/finalize/warmup paths pass `nullptr`.
   - The b2-t1 case design (6 cases) — does it actually cover §II.6 requirements
     (forced-concurrency-B4 + bucket-count assertions + scheduler-ON Bmax=1 control)?
   - The scope reduction (4 rows vs 1000): justified by OOM, OR could it be addressed with sequential per-N
     loader reuse (Codex noted in the log it switched to shared loaders for the K=3 fix — did this fully
     solve the OOM, or is the 4-row limit still memory-bound)? If the latter, a full-corpus b2-t1 run should
     be a B3-pre-condition follow-up.
   - The A1 parity check + outcome B handling: §II.9 says outcome B → OFF path stays PRODUCTION, scheduler-ON
     explicitly owns the risk via b2-t1 case 1. Verify implementation matches.
3. **steady_batch_primitive.h changes** (now 631 lines — substantial growth):
   - The built-in SHA256 + JSON parser: necessary (no easy std/Boost JSON; no SHA256 in libtorch)? Tested?
     Edge cases (empty file, malformed JSON)?
   - The manifest verify (line 486-524): fail-closed semantics correct? Throws on missing manifest, missing
     bucket, SHA mismatch, shared-weight SHA mismatch? §II.12 ✓ check.
   - The new `run_prepacked` / `run_raw_prepacked` / `unpack_prepacked_outputs` API: separation of concerns
     vs the original `run()` — scheduler uses raw+unpack (its own scratch); b1-t1 still uses original `run`
     (which uses torch::cat). Both correct?
4. **export_steady_batched.py manifest emit**:
   - Emits per-B `package_sha256`, `ep_sha256`, `shared_weight_sha256`, torch/cuda/arch, inductor configs,
     byte sizes. Schema versioned. Matches §II.12.
   - Are the SHAs computed correctly (file-content SHA256, matching the C++ verifier's algorithm)?
   - Does it run automatically as part of the export script, or only on demand? Should it always run?
5. **OFF-path preservation** (the binding byte-exact contract):
   - The 20-session N=4 smoke confirms no regression at one operating point.
   - Is the OFF-path code reachable WITHOUT the scheduler ever being constructed (per §II.3 conditional
     construction)? Trace: when `batch_steady = off`, is `BatchedSteadyScheduler` *never* instantiated, OR
     is it constructed-then-unused? Should be never-instantiated per the lifecycle requirement.
6. **Memory considerations for the 5090 knee re-measure** (§II.13):
   - Codex hit OOM on the full-corpus b2-t1 at the 5090. The scheduler adds: dispatcher stream, scratch
     tensors per bucket (small), B=1/2/4 packages preloaded (large — ~2.5GB each = 7.5GB across B-buckets,
     SHARED constants so not 3×), the loader pool. Is the existing N=40 5090 knee still feasible with the
     scheduler ON, or does the added memory push it down?
   - Worth flagging as a knee re-measure consideration (not a blocker for B2 commit).
7. **Edge cases / potential bugs**:
   - `set_pending_exception_locked` discards its `ep` parameter (`(void)ep;` line 397) — dead code, harmless
     but worth noting.
   - `dispatch_batch` calls `cudaEventSynchronize(ev_stop)` on dispatcher_stream — synchronous block, fine
     for measurement but is this introducing CPU blocking that defeats the dispatcher-thread-as-async-
     orchestrator goal? Could be replaced with the per-row completion event's elapsed-time approach.
   - The `unpack_outputs` returns tensors that are `.contiguous()` copies — these are scheduled on
     dispatcher_stream but consumed on worker stream. The completion event makes the worker wait for them.
     ✓
8. **Net verdict**: PASS / PASS-with-followup / HOLD-iterate.
   - If PASS, list any pre-B3 follow-ups (e.g., full-corpus b2-t1, scheduler memory profiling).
   - If HOLD, list the specific blocker.
</context>

<verification_loop>
Don't re-run b2-t1 (it just ran). Audit by reading. If you spot something that needs verification, propose a
focused micro-check; don't run the full corpus.
</verification_loop>

<action_safety>
Write only the review doc. Do not modify any implementation files. Fix recommendations go in the review for me
to fold + action.
</action_safety>

<compact_output_contract>
Report path of the review doc + one-paragraph verdict (PASS / PASS-with-followup / HOLD) + the highest-priority
fix-or-flag if any.
</compact_output_contract>
