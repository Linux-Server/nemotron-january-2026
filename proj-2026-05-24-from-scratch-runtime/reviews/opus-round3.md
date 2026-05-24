# Opus adversarial review — Round 3 (deeper; stressing v3)

## [BLOCKER] 1. CUDA-graph capture × continuous-batching × shared-weights is an unresolved three-way architectural tension
v3 asserts all three as wins independently but they fight each other, and the plan never reconciles them:
- The existing manager captures **one graph per exact B (1..K) with dedicated static input/output buffers**
  (`cudagraph_encoder.py:193-210` allocates per-bucket `static_processed/clc/clt/clcl`; replay copies into them
  `:272-281`; exact-shape, fail-closed `:578-599`). In a native runtime with **per-lane dispatcher threads**, each lane
  that wants graph replay needs **its own static-buffer set per B** → the graph-pool memory is **×lanes**, which is the
  very K×duplication the density story claims to remove. So "shared read-only *weights*" does NOT mean shared graph
  buffers; the per-(B,lane) static pools remain and may re-cap K.
- **CUDA graph *capture* is not freely thread-safe**: capture puts the stream into a capturing state, requires a
  graph-safe memory pool (PyTorch uses a private `graph_pool_handle` + the caching allocator's capture mode), and
  concurrent capture across threads in one context is unsafe. The native runtime must **serialize capture** and
  replicate a graph-safe allocator — non-trivial in C++/libtorch and absent from the plan.
- **Continuous batching (B varies 1..K per tick) keyed to exact-B graphs** means the hot path constantly switches
  buckets; with per-lane pools that's K×lanes captured graphs resident. The plan must choose: (a) per-lane graphs
  (memory ×lanes), (b) one shared graph set with serialized replay (throughput hit — re-introduces a serial lane), or
  (c) no steady graphs and rely on lower per-launch overhead from no-GIL. **This choice drives the whole density claim
  and is unmade.**
- **Recommendation:** Add a Phase-0 (Track A) "CUDA-graph ownership model" spike: decide per-lane vs shared-serialized
  vs none; measure the real resident graph-pool memory at the target K×lanes; and specify the graph-safe allocator
  approach. Until resolved, the 40–48/box density number is unsupported.

## [BLOCKER] 2. "Byte-exact vs Python" (T2) is gauged against a MOVING reference and is over-specified across batch size
- The Python plan (`proj-2026-05-24-0859`) is *actively changing encoder/preproc-adjacent numerics* (host-sync
  compression, padded-T finalize bucket, finalize graph). "Byte-exact vs the current Python server" therefore drifts
  under the Track-A gates. v3 says "record the baseline commit beaten" for *perf*, but the **correctness gate needs a
  FROZEN golden reference**: capture fixtures (mel, encoder out, hypotheses) **once from a pinned Python build + pinned
  torch/CUDA**, and diff against those fixtures — not the live server.
- T2 demands byte-exact "across the live geometry … B>1." **Cross-batch-size byte-exactness is not guaranteed even
  inside PyTorch** (cuBLAS/cuDNN pick different kernels + reduction orders per shape). The project *did* report
  batched byte-exactness (memory `streaming-batching-outcome`), but only with deterministic settings + constant plans —
  and reproducing **PyTorch's exact kernel/workspace selection from libtorch C++** requires identical backend
  versions/flags and is itself a finding to prove, not assume. Cross-**arch** byte-exact (5090 vs L40S vs Spark) is
  out entirely.
- **Recommendation:** Re-scope T2: "byte-exact vs **frozen fixtures from a pinned Python+torch+CUDA build, same arch
  (5090), per-B (same-B path), deterministic mode**." State that cross-arch and possibly batched byte-exact may only
  reach T1, and make *that determination itself* an output of Spike 0.2/0.8 — don't pre-promise byte-exact as a pass/fail
  gate without first proving it's attainable from libtorch.

## [MAJOR] 3. The Track A / Track B split moved the contradiction rather than resolving it for the correctness gates
Track A is "baseline-independent feasibility," but 0.2/0.6/0.8 are **byte-exact-vs-Python** gates — which are
*not* baseline-independent if the comparison is the live Python server (see #2). They become baseline-independent **only
if** they diff frozen fixtures. v3 doesn't say that, so Track A's "run now, parallel with the Python plan" is only valid
once the golden-fixture discipline is added.
- **Recommendation:** Make "Track A diffs against frozen golden fixtures captured from a pinned Python build" an
  explicit precondition of Track A; otherwise Track A is also blocked on the Python plan, collapsing the split.

## [MAJOR] 4. §9's ~22–35 eng-weeks is optimistic by roughly 2× — systematic under-counts
Credible omissions/under-counts: (a) **CUDA-graph ownership + graph-safe allocator** in C++ (finding #1) — not in the
table; (b) **golden-fixture harness + determinism plumbing** across the native stack; (c) **the cross-language seam**
(Rust↔C++ FFI marshalling of tensors/CUDA handles) if shape 2 is chosen; (d) **integration/debug time for byte-exact
chasing** — the project's history shows byte-exact debugging (cuFFT plan, fork aliasing) consumed disproportionate time;
(e) **observability/metrics parity** is one line but must mirror a large timing schema; (f) **multi-platform CI** (4
GPUs incl. aarch64) build/test infra. A from-scratch label-looping RNNT decode alone, proven byte-exact, is plausibly
4–8 weeks not 3–5. Realistic total excl. fusion is closer to **~40–60 eng-weeks**.
- **Recommendation:** Add the missing line items; widen the estimate to ~40–60 eng-wk and mark §9 ±50–100%. This makes
  the §0 worth-it gate even sharper (≈ a full engineer-year against the Python plan's days–weeks).

## [MAJOR] 5. Native multi-threaded determinism: per-thread cuBLAS handles / workspaces / TF32 can break the byte-exact gate even with shared weights
Sharing one weight copy is fine for correctness *per call*, but concurrent lanes mean **per-thread cuBLAS/cuDNN handles
and workspace** and stream-local algorithm heuristics; cuBLAS workspace size + algo can vary by handle/stream, and TF32
must be pinned off (the model is fp32 — note `roofline-COMBINED.md` "TF32 disabled under batching"). If the native
runtime lets the backend pick algos per-thread, byte-exactness (and even run-to-run determinism, T0) can break.
- **Recommendation:** Add to 0.9/1.0: pin cuBLAS workspace config, disable TF32, set deterministic algorithms, and use
  fixed per-lane handles; make T0 (run-to-run determinism under concurrency) an explicit gate before T2 is even
  attempted.

## [MAJOR] 6. The deployed decoder does NOT preserve alignments/confidence by default — state it (scope down 0.6), but note the EOU dependency
`preserve_alignments`/`preserve_frame_confidence` are set **only when `eou_probe_enabled`** (`server.py:1476-1484`,
`:1495-1503`), which is a NO-GO and off in the shipped warm200 config (memory `eou-asr-internal-nogo`). So v1's native
label-looping decode can **skip alignment/confidence**, simplifying 0.6 — but the plan should state this assumption
explicitly, because if EOU-probe is ever revived the native decode must add alignment/confidence (entropy/tsallis
method) and the equivalence fixtures change.
- **Recommendation:** Add to 0.6: "v1 assumes `eou_probe_enabled=False` (no alignment/confidence preservation); if EOU
  probe revives, alignment/confidence equivalence is a new required axis."

## [MINOR] 7. Internal consistency after edits
- §1 still lists "no MPS tax" historically? Verify it's now framed as *to-be-proven* (it is, in the corrected-thesis
  paragraph + Spike 0.1) — keep consistent; don't let the headline table re-imply it.
- "Phase 0.5 — Runtime contract" sits between Phase 0 and Phase 1 numerically as step 0.10 — fine, but ensure the
  worth-it gate §0.0 explicitly allows 0.10 (contract design) to proceed early too (it's baseline-independent).
- The §9 total and the §0 "½–¾ eng-year" phrasing should be reconciled with finding #4's wider estimate.

## Top 5 things to fix
1. **Add a CUDA-graph ownership-model spike** (per-lane vs shared-serialized vs none; graph-safe allocator; real
   resident memory at K×lanes) — the density number depends on it and it's currently unaddressed.
2. **Re-scope T2** to byte-exact vs **frozen golden fixtures from a pinned Python+torch+CUDA build, same-arch, per-B,
   deterministic mode**, and make "is byte-exact even attainable from libtorch?" an *output* of 0.2/0.8, not a
   pre-assumed gate; cross-arch/batched may only reach T1.
3. **Require golden-fixture diffing for Track A** so the A/B split is actually coherent against the moving Python plan.
4. **Add determinism plumbing** (TF32 off, deterministic algos, pinned cuBLAS workspace, per-lane handles) as a T0 gate
   under concurrency before T2.
5. **Widen §9 to ~40–60 eng-wk** with the missing line items (graph allocator, FFI seam, golden-fixture harness,
   byte-exact debugging, multi-GPU CI); scope 0.6 down via the `eou_probe_enabled=False` assumption.
