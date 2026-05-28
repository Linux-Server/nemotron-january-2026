<task>
Implement Step B1 of proj-2026-05-24-from-scratch-runtime/PHASE2-PLAN.md:
"batched-steady forward mechanism + T1 (5090 dev, topology-agnostic core)".

Build the cross-stream **batched steady-encoder forward primitive** for the native density runtime: pack K streams'
ready steady chunks into one batched AOTI run, then unpack per-stream — behind a flag, default OFF (B=1 byte-exact
preserved), with a T1 gate that proves per-row byte-exactness THROUGH THE FULL DECODE + EVENT PATH.

This is the green-lit #1 density lever: the L40S knee is DRAM-bandwidth-bound weight-streaming in the steady encoder,
and batching amortizes the weight load across B streams. The STEADY-BATCH-0 kill-gate already PASSED (see
reviews/steady-batch0-RESULT.md): per-row GPU time B=1 5.11ms -> B=2 3.18ms (0.62x) -> B=4 1.92ms (0.38x), and the
MICROBENCH already proved KERNEL-level per-row equality (enc_out max 6e-6). B1's NEW job is to (a) productionize the
primitive into the runtime's loader/session abstractions, and (b) prove the tiny enc_out diff does NOT flip any
greedy-decode token or event — that is the real correctness question for batching.
</task>

<context>
Read for full context:
- PHASE2-PLAN.md — the v3 block (top), Step B1 (the spec), the "Ownership / topology contract", and the lever
  inventory Tier-2. The SLO budget: server-side ttfs p95<=175 / p99<=250; keep-up lag p95<500.
- runtime/cpp/steady_batch_bench.cpp — THE REFERENCE for pack/run/unpack + the exact tensor geometry: steady input
  [B,128,25]; cache_ch [24,B,70,1024]; cache_t [24,B,1024,8]; outputs enc_out, new cache_ch, new cache_t, enc_len.
  It already loads B in {1,2,4} AOTI and packs B independent caches and checks per-row correctness vs B=1.
- runtime/export_steady_batched.py — exports the B-bucket steady AOTI. The B=1/2/4 packages are already compiled at
  runtime/steady_b_artifacts/enc_steady_aoti_b{1,2,4}.pt2 (sm_120, torch 2.8.0, validated byte-exact vs eager). If
  missing, regen per-B with `--compile-only` (TMPDIR + TORCHINDUCTOR_CACHE_DIR on /, not /tmp which is a small tmpfs).
- runtime/cpp/density_main.cpp — the density harness: how the steady encoder is currently run per-worker as a B=1
  `run(inputs, stream)`, the per-worker SessionState / AOTIModelPackageLoader usage, the worker context, and the
  existing T1 machinery (gold-session token/event compare; DENSITY_GOLD_EVENTS_TOLERANT).

DELIVERABLES:
1. A batched-steady **loader + forward primitive** (a new module/file, or in the session core): load the B in {1,2,4}
   steady AOTI buckets as a set (follow the one-shared-constants / one-bundle discipline used for the steady loader
   and finalize buckets — do NOT duplicate weights per bucket). Expose a primitive roughly:
   `run_batched_steady(span<SteadyInput> ready) -> vector<SteadyOutput>` that: picks the nearest bucket B >= K
   (K = ready.size(); pad rows K..B-1 — document the pad choice, pads are discarded on unpack), packs the K
   (mel-chunk, cache_ch, cache_t) into the batch dim, runs the batched AOTI on the worker's stream, and unpacks
   enc_out / new cache_ch / new cache_t / enc_len back to each of the K callers.
2. Flag **NEMOTRON_DENSITY_BATCH_STEADY** (default OFF). OFF => the existing per-worker B=1 path is used UNCHANGED
   (byte-exact). ON => the primitive is available (B1 need not yet wire the cross-stream COLLECTION — that's B2; B1
   may exercise the primitive in a degenerate K=1..4 self-batch or test-driver to prove correctness).
3. **T1 GATE — the heart of B1.** A test mode/binary that proves the batched path is byte-exact per-row through
   decode+events. REQUIRED coverage: **ragged / mixed batches** — pack rows from DIFFERENT gold sessions at DIFFERENT
   chunk indices into one batch (the real cross-stream case), and assert that EACH row's resulting decode tokens AND
   events are byte-identical to that same stream run ALONE on the B=1 path. (Not just enc_out tensor diffs — the
   microbench did that; the new risk is a 6e-6 enc_out diff flipping a greedy argmax tie.) Also include the simple
   identical-rows case. Report pass/fail + max abs enc_out/cache diffs + any token/event divergence (utterance, chunk,
   position). Start on a small subset for the dev cycle; the gate is the gold corpus (the existing density T1 ran
   1000/1000 finals byte-exact — match that bar, events counted-not-gated via the tolerant flag if a documented
   interim-timing drift recurs).
</context>

<verification_loop>
Build AND run in the container (runtime/container/enter.sh -> nemotron-aoti:cu128, torch 2.8.0, sm_120). Build with
the same cmake pattern as the bench (`cmake -S cpp -B cpp/build_<...> -DTORCH_ROOT=/usr/local/lib/python3.12/dist-packages/torch`).
Use runtime/steady_b_artifacts/enc_steady_aoti_b{1,2,4}.pt2. Run the T1 and confirm it PASSES (byte-exact per-row
tokens+events on the ragged/mixed case) before reporting. If a token flips, that is a STOP-this-lever finding — report
it with the exact divergence, do NOT paper over it with a loosened tolerance.
</verification_loop>

<action_safety>
Only touch the steady-encoder forward path + the new primitive + the T1 test. Do NOT change decode/finalize ownership
(stays per-stream, unchanged). Do NOT build the cross-stream scheduler/collection (that is B2). Keep the flag default
OFF so the existing B=1 path and all prior T1/gold guarantees are untouched. Do not commit binary AOTI artifacts (they
are gitignored).
</action_safety>

<compact_output_contract>
When done, report:
1. Work performed (files created/modified and why; how the pack/run/unpack primitive and the bucket loader work; the
   pad choice).
2. The T1 result: pass/fail, the cases covered (ragged/mixed + identical), max enc_out/cache diffs, and whether any
   decode token/event diverged per-row (with locations if so).
3. Current project status + how the flag-OFF byte-exactness was confirmed.
4. Any blockers or concerns (e.g., fixture capture for the mixed case).
5. Suggested next step (toward B2 the batching scheduler).
</compact_output_contract>
