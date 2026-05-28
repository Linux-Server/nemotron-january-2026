<task>
Implement Step B2 (batching scheduler + density integration) per the paired-reviewed binding spec at
`proj-2026-05-24-from-scratch-runtime/reviews/B2-design-paired-verdict.md` section II (II.1 through II.14).
That section supersedes `B2-design.md` v1 on every conflict — quote section II as the contract.

Build: the `BatchedSteadyScheduler` (central dispatcher), its integration into `density_main.cpp` via an
EXPLICIT pointer (no globals), the new `--mode b2-t1` gate, the A1 NEW-vs-PRODUCTION B=1 parity check, the
A4 sealed-loader fix, the A5 preallocated scratch with `index_copy_` on the dispatcher stream, and the A7
manifest emission + fail-closed consumption.
</task>

<context>
**THE BINDING SPEC** — read in full first:
- `proj-2026-05-24-from-scratch-runtime/reviews/B2-design-paired-verdict.md` (the fold; the §II.1-II.14 spec
  is the build contract).
- `proj-2026-05-24-from-scratch-runtime/reviews/B2-design.md` v1 (background context; superseded by §II
  wherever they conflict).
- `proj-2026-05-24-from-scratch-runtime/reviews/{opus,codex}-B2-design-review.md` (the two reviews that
  produced the fold).

**Code context**:
- `runtime/cpp/steady_batch_primitive.h` (B1 — the primitive; §II.10 modifies this with `sealed_`
  + fail-closed `get()`).
- `runtime/cpp/density_main.cpp` — Step 1a / B1 lives here. §II.4 integration point: the worker's
  `run_steady_chunk_density` continuation branch (NOT enc_first, NOT finalize); the scheduler pointer is
  explicit (nullptr → unchanged B=1 path). §II.6 new `--mode b2-t1` gate. §II.5 warmup via
  `scheduler.warmup_buckets()` outside the measured phase.
- `runtime/export_steady_batched.py` — §II.12 emits `runtime/steady_b_artifacts/MANIFEST.json`.
- `runtime/cpp/CMakeLists.txt` — add the new .cpp.
- `runtime/container/enter.sh` — the build environment (`nemotron-aoti:cu128`, torch 2.8.0, sm_120).
- The B1 commit `3887cb3` for the existing density_main, the B1 primitive, and the b1-t1 gate pattern (b2-t1
  mirrors b1-t1 but exercises real cross-stream batching with explicit barriers per §II.6).

**The single most critical correctness item: §II.2 bidirectional CUDA stream synchronization.** The dispatcher
MUST wait on each worker's producer-ready event before pack/run; the worker MUST wait on the dispatcher's
completion event before consuming. v1 only specified the output-side wait — that was a missed race.

**SLO + correctness contract** (binding, from the spec + the project policy):
- 0 token divergences in `b2-t1` (any case). FATAL.
- Events counted-not-gated per `DENSITY_GOLD_EVENTS_TOLERANT` (project policy).
- Byte-exact when `NEMOTRON_DENSITY_BATCH_STEADY=0` (the production B=1 path unchanged — verify by
  preserving the b1-t1 + serial reference + density-sweep paths exactly).
- Added batch-wait p95 ≤ window (10ms by default).

**Build deliverables** (per §II.14):
1. `runtime/cpp/batched_steady_scheduler.h` + `.cpp` (dispatcher, queue, futures, telemetry, scratch).
2. Modify `runtime/cpp/density_main.cpp`: conditional scheduler construction; explicit pointer integration
   in the worker's `run_steady_chunk_density` continuation branch; new `--mode b2-t1` gate + new CLI args
   (per §II.3); A1 parity check + log at scheduler construction.
3. Modify `runtime/cpp/steady_batch_primitive.h`: A4 `sealed_` + fail-closed `get()`.
4. Modify `runtime/export_steady_batched.py`: emit MANIFEST.json (§II.12). C++ loader CONSUMES it
   fail-closed.
5. Update `runtime/cpp/CMakeLists.txt` for the new .cpp.
6. (Optional, scope-permitting): `runtime/cpp/runtime_io.h` (A8 — only if it doesn't bloat B2; mechanical
   refactor).

**§II.13 5090 knee re-measure is OUT OF SCOPE** for this build task — that's a separate post-build run on
the green-lit B2 build, run by me (not Codex). Just deliver the build + the b2-t1 PASS.
</context>

<verification_loop>
Build in the container (`runtime/container/enter.sh bash -lc 'cmake -S cpp -B cpp/build_b2 -DTORCH_ROOT=...
&& cmake --build cpp/build_b2 --target density_main -j$(nproc)'`). Run the b2-t1 gate (§II.6: scheduler-on
single-stream + multi-stream forced-concurrency with barrier + multi-stream staggered + scheduler-on `B_max=1`
control; with bucket-count assertions; require token divergences == 0). Run the A1 parity check (§II.9). Run
ONE final `--mode density-sweep` smoke at N=4 with `NEMOTRON_DENSITY_BATCH_STEADY=0` to confirm the OFF path
is unchanged (token-equality against an existing N=4 result or 1000/1000 finals byte-exact). Report PASS/FAIL
per b2-t1 + the A1 parity outcome.

If §II.2 (input-side sync) or §II.4 (explicit integration, no globals) hits a blocker — STOP, report; don't
work around them. These are non-negotiable per the fold.

If the b2-t1 multi-stream forced-concurrency case forms <N_workers/B_max batches at B=B_max (i.e., the
scheduler short-circuited and the test ran B=1 paths), that is a TEST FAILURE — don't pass it.
</verification_loop>

<action_safety>
Do not modify the existing production B=1 path (`enc_steady_aoti.pt2` is the durable byte-exact reference).
Do not change `build_serial_reference` / b1-t1 / finalize / warmup paths to route through the scheduler —
they explicitly pass `nullptr` for the scheduler pointer per §II.4. Do not commit the binary AOTI artifacts
(gitignored).
</action_safety>

<compact_output_contract>
When done, report:
1. Files created/modified (per §II.14).
2. The b2-t1 result: per-case PASS/FAIL + bucket-count telemetry + token divergence count + event drift
   count + max enc_out/cache diffs.
3. The A1 parity outcome (A bit-identical / B different-but-tensor-parity-OK / C tensor-parity FAIL).
4. Density-sweep OFF-path smoke result (confirms byte-exact preserved).
5. Any §II item you couldn't implement as specified, with the reason; flag blockers on §II.2 or §II.4 as
   STOP-this-build.
6. Suggested next step (post-build paired review → 5090 knee re-measure).
</compact_output_contract>
