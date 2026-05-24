# Spike 0.4 — Decision memo (TEMPLATE — fill after spikes run)

This is the HARD-GATE output of Phase 0. Fill every `<…>` from measured spike results. **Label each input as Track-A
feasibility or Track-B post-plan residual.**

## Pinned versions / scope (fill once chosen)
- Language + process shape: `<all-C++ | Rust-front+C++-worker | all-Rust>` — **decided by the tch-rs evaluation below + 0.1/0.2.**
- libtorch / CUDA / driver / C++ ABI: `<pin exact versions>`
- Export artifact format: `<TorchScript | torch.export | …>`
- WER-CI width (T1): `<named numeric, e.g. ±X% abs WER on full-1000>`
- v1 scope: **EN 0.6b only** (multilingual/prompted = later phase).

### libtorch version-selection checklist — "newest VIABLE + pinned" (not "always latest")
Pick the **newest *stable* (not nightly)** libtorch that clears ALL of these; pin it exactly and use the SAME version
for export + golden fixtures + runtime (the T2a byte-exact gate breaks otherwise):
- [ ] **Blackwell/CUDA across all 4 targets:** built against a CUDA toolkit that emits **sm_120** (RTX 5090, Blackwell;
  CUDA ~12.8+) AND covers Ada (L4/L40S) AND aarch64 (GB10/Spark — ties to 0.7). *Newer is required here, not optional.*
- [ ] **NeMo-supported torch range:** ≤ the newest torch the `nemo_toolkit` that loads this checkpoint supports. *This is
  the real ceiling on "newest."*
- [ ] **Same version for export-producer + fixtures + runtime** (T2a). Record the exact build hash.
- [ ] **C++ ABI flag** (`_GLIBCXX_USE_CXX11_ABI`) matched across libtorch + compiler + linked libs (classic footgun).
- [ ] **If all-Rust: a tch-rs binding exists for this libtorch version** (see the gate below).

### tch-rs evaluation — DECIDES Rust vs C++ (user direction 2026-05-24)
The Rust-vs-C++ call is made on **concrete tch-rs binding coverage + version limits**, not preference. **All-Rust is the
default IF AND ONLY IF tch-rs (+ `cudarc`) clears every box at the libtorch version chosen above; otherwise the C++
model-worker (or all-C++) carries the hot path.**

**Ordering invariant (the hedge):** the libtorch **version is pinned by the C++/NeMo/CUDA constraints FIRST** (Blackwell
sm_120, NeMo torch range, same-version-for-export+fixtures+runtime, ABI) — *even in the all-Rust case*. **tch-rs
availability is a CONSEQUENCE of that pin, never an input to it: tch-rs can VETO Rust (if it doesn't bind the required
version / surface) but it can NEVER widen or move the version choice.** We do not downgrade libtorch to suit tch-rs; if
tch-rs only binds an older libtorch that fails the constraints, that is a Rust veto, not a reason to pick the older
libtorch.
- [ ] A **tch-rs release binds the chosen (constraint-clearing) libtorch version** — not lagging behind it.
- [ ] tch-rs/cudarc expose **CUDA-graph capture/replay against libtorch-ALLOCATED tensors** (the allocator-coupled path,
  not just raw `cudarc` graphs over separately-allocated memory). *This is the decisive box.*
- [ ] **per-lane CUDA stream + event** control (for the no-GIL dispatcher).
- [ ] the **graph-safe / capture-mode allocator** is reachable (no allocation during capture — 0.11).
- [ ] the **ATen ops the RNNT label-looping decode needs** (joint/pred forwards, argmax, state ops) are bound, or
  trivially FFI-shimmable.
- **Outcome:** all boxes ✓ → **all-Rust** (no seam, borrow-checker everywhere). Any ✗ → either write `unsafe extern "C"`
  shims to the missing libtorch symbols inside the Rust crate (you're hand-binding C++ anyway) → prefer **Rust-front +
  thin C++ worker**, or go **all-C++**. Record which box failed.

## PRE-REGISTERED Wave-1 thresholds (REGISTER BEFORE COLLECTING 0.1/0.5 DATA)
> These are kill decisions; defining them after seeing data is invalid. Fill the numbers, freeze, THEN run.
> The values below are **proposed STARTING points from the path-forward review** — replace with the user's actual
> business numbers before any data collection. If no one will write these down → STOP/defer.

**0.0-pre ceiling (do FIRST, free):** best-case native upside ≈ 48 − 28 ≈ **~20 streams/box**, triple-conditional, vs
~40–60 eng-wk BUILD **+ ongoing dual-stack carry**. If the *ceiling* can't clear the 0.0 threshold below assuming all
gates pass → STOP now.

**0.0 worth-it threshold — DEFERRED by the user (2026-05-24) until the Python plan lands + gives a measured baseline.**
The values below are reference-only proposals; freeze the real numbers *before* Wave-1 data, *after* the baseline exists.
Note the user chose NOT to freeze the Python baseline → 0.0 is re-checked against the *latest* Python result as it improves.
- Min in-budget streams/box delta vs Python's ~28/box (K=4): proposed `≥ +12 streams/box` at equivalent tail.
- OR min p99 server-side tail reduction at the same *admitted* load: proposed `≥ 150–200 ms`.
- Value-vs-cost: residual must justify `~40–60 eng-wk + a 2nd stack + carry`. **Distinguish attempted vs admitted load**
  — admission-only p99 wins (shedding traffic) do NOT justify a rewrite unless that's the intended tradeoff.

**0.1 overlap/MPS ablation (proposed):**
- Required single-process finalize+steady overlap factor vs Python/MPS baseline: `≥ 1.5×`
- Max acceptable queue/lane wait at the operating point: `<X ms>` (set from the operating point)
- Max added per-chunk latency from the new dispatch: `≤ 5–10 ms`
- **Input:** the Python plan's Step-5 GIL probe decode-vs-glue attribution (`proj-2026-05-24-0859:148-156`) — consume,
  don't re-derive.

**0.5 batching sim + graph capacity (proposed):**
- Median / p95 batch B target: `median ≥ 2, p95 ≥ 4`
- Min exact-B graph replay hit-rate: `≥ 90–95%`
- Max eager-fallback rate: `≤ 5–10%`
- Max added wait to form a batch: `≤ 5 ms` (unless explicitly traded for tail)
- Required L4 / L40S graph-pool memory headroom at target K×lanes: `≥ 2 GB or ≥ 10%, whichever larger`

## DECISION TREE (fill the "Observed" column with measured outcomes)

| Branch | Observed | Decision taken |
|---|---|---|
| 0.0 residual value < threshold | `< >` | **STOP** |
| 0.1 only MPS/multi-proc overlaps (not single-process) | `< >` | likely STOP; else native-under-MPS = **tail-only**, re-run 0.0 |
| (B4 / py3.13t path) | — | **REMOVED 2026-05-24 — user rejected free-threaded Python; outcome space is B1 or STOP** |
| 0.1 positive AND 0.6a + 0.2 + 0.8 + 0.11 pass | `< >` | **proceed B1** |
| 0.6a fails byte/state equivalence | `< >` | STOP, or named T1-only native-decode risk sign-off (no B4 fallback) |
| 0.2 fails T2a / libtorch byte-exact unattainable | `< >` | no B1a → STOP / B2-risk sign-off (no B4 fallback) |
| 0.8 fails native-preproc byte-exact | `< >` | STOP, or Python-preproc as named non-v1 topology |
| 0.9 fails (can't parameterize per-call config) | `< >` | drop shared-weight density → per-lane replicas; re-run 0.0 |
| 0.5 B stays ~1 | `< >` | drop 3–5× throughput; re-run 0.0 |
| 0.5/0.11 poor hit-rate / fallback / no memory headroom | `< >` | drop steady-graph density; B1-without-graphs or re-topology; re-run 0.0 |
| 3.3 fusion unproven | `< >` | 6–10 ms finalize out of v1 scope (B1 = parity); core go/no-go unaffected |

## Final decision
`<STOP | B1 | proceed-to-fusion>` — rationale: `<…>` (B4/B5 removed by the user 2026-05-24; outcome space is B1 or STOP)
