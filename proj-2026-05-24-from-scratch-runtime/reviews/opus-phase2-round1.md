# Opus — Phase-2 plan review, Round 1 (independent)

Scope: `PHASE2-PLAN.md` + the code/evidence it rests on. Foregrounding the two stated goals —
**(G1) system utilization / streams-box density** and **(G2) tightened P50↔P95 TTFT spread (the tail)**.
Method: line-by-line read of `session_main.cpp`, `finalize_main.cpp`, `microbench.cpp`, the 0.1b spec + RESULTS,
PLAN_RULES, and the folded Phase-1 reviews.

## Top-line
The plan's *structure* is right (Step 1 = the decisive conjunct-2 measurement; downstream steps gated on it) and
its honesty framing is good (no-mock, report util + SLO-robust knee). But it **understates how much NEW,
UNVALIDATED concurrency machinery Step 1 must introduce**, treats the "HARD GATE" as a vibe rather than a number,
omits a **concurrency-correctness gate**, and — most important for the user's framing — **does not make G2 (the
P50↔P95 spread) a first-class measured quantity** and risks a load model that structurally cannot produce a real
tail. Five BLOCKER/MAJOR items below would, uncorrected, allow either a **false GO** (a fast-but-unrepresentative
or fast-but-incorrect Step-1 number) or a **silent miss on G2**.

---

## BLOCKERs

### B1 — Dispatch-primitive mismatch: the 0.1b overlap was measured on a primitive the real session does NOT use
The 0.1b microbench measured overlap of a **captured CUDA graph** replayed on a **per-lane stream**:
`microbench.cpp:124` `graph.replay()` inside `c10::cuda::CUDAStreamGuard g(stream)` (capture at `:113-120`,
per-lane stream from `getStreamFromPool` at `:164`). The **real native session does neither**:
`session_main.cpp:1765` dispatches the steady encoder via `loader.run(inputs)` (AOTI), and a full-file grep shows
**no CUDA stream / `CUDAStreamGuard` / `CUDAGuard` anywhere** in `session_main.cpp` — every "stream" token is a
string/ID. The validated single stream therefore runs `AOTIModelPackageLoader::run()` **synchronously on the
default stream**.

These primitives are not interchangeable for an overlap measurement. Two things are now established from the E.3
harness (`aoti_encoder_main.cpp`) and partly DE-RISK this, but the core gap remains:
- **Stream plumbing EXISTS and is byte-exact (single-threaded).** `loader.run(inputs, (void*)stream.stream())`
  takes an explicit stream pointer (`aoti_encoder_main.cpp:64-65`) and E.3 proved it "stream-invariant" — output
  byte-identical to the default-stream run (`:106`). So per-thread streams ARE pluggable into `run()`; the
  native session simply doesn't use that arg (it calls the 1-arg `run()` at `session_main.cpp:1765` → default
  stream). This kills my earlier worry that `run()` might ignore the stream — it doesn't.
- **What is still UNPROVEN is the only thing that matters for G1:** (a) whether two `run()` calls on two streams
  from two threads actually **overlap on the GPU** (single-context kernel concurrency) vs the context serializing
  them; and (b) **aliasing/re-entrancy** — E.3's "no-alias" + "second call into the same runner" checks
  (`:79,106`) were *sequential on one runner*; concurrent `run()` on one runner is untested and likely unsafe
  (per-runner scratch). So overlap and concurrency-safety are the residual, not stream-honoring.
- `graph.replay()` (the mock) also elides `run()`'s per-call host work (input contiguity `:1759-1764`, output
  marshaling, the `out.size()<5` check `:1766`), so the mock's overlap is an *optimistic* proxy for `run()`.

**Risk:** the plan's premise ("replay the real per-session compute … a CUDA stream per worker") assumes
concurrent AOTI `run()` overlaps like `graph.replay()` did. The stream *plumbing* transfers; the *overlap* is the
unmeasured property the whole density thesis pivots on.
**Edit:** make Step 1's FIRST sub-measurement an isolated 2-thread `run(inputs, stream)` overlap check (wall-time
vs 2× serial + a profiler trace that kernels interleave). If they do NOT overlap, the plan needs an explicit
fork: per-thread **CUDA-graph-of-the-AOTI-steady** — real, unscheduled work that must be named as a Step-1
branch, not assumed away. Pair this with the aliasing check (B3: N runners vs one).

### B2 — No concurrency-CORRECTNESS gate; the HARD GATE is performance-only
All of Phase 1 was token/event-exactness. Step 1 measures throughput/util/tail and gates on "does the knee lift"
— but has **zero assertion that the concurrent compute is still correct**. Sharing one weight set across N
threads via `load_constants(..., user_managed=true)` is **unproven under concurrency**: it was validated
single-threaded at load time (`validate_shared_weights.py`; `finalize_main.cpp:1573-1575` loads each bucket's
constants once, serially). AOTI keeps internal constant buffers with an active/inactive notion (the
`use_inactive` / `check_full_update` args at `finalize_main.cpp:1575` and `session_main.cpp:1575`); concurrent
`run()` against a shared constant map — or any shared loader scratch — could race and silently corrupt outputs.
A fast-but-WRONG result reads as a GO.
**Edit:** add a mandatory **concurrency-correctness conjunct** to Step 1's gate: run the same bundle(s)
concurrently on N threads and assert each thread's token/event stream is identical to the serial N=1 result
(the FORK_ASSERT analog for threads). No throughput number is trusted until this passes. This is decision-
critical: a GO on an unverified-correct concurrent path funds the Wave-2 ports on a mirage.

### B3 — The loader-sharing mechanism (the crux) is unspecified
`session_main.cpp` uses exactly ONE `enc_steady` loader (`:3868`, ctor `AOTIModelPackageLoader(pkg,"model",false,
1,-1)` → `num_runners=1`) and a map of per-bucket finalize loaders (`:1568,1573`, also `num_runners=1`). The plan
says only "shared weights via `load_constants` user_managed." It never states which of three materially different
mechanisms Step 1 uses:
1. **One shared loader, N threads call `run()`** — requires `run()` re-entrancy; almost certainly unsafe
   (per-runner scratch). 
2. **N loaders, one shared constant-tensor map** (memory-feasible via `user_managed`) — the only path that both
   avoids the 0.1b OOM (L40S lanes=32 OOM'd at K×2.5GB, `RESULTS-L40S.md:40-43`) AND plausibly allows concurrent
   `run()`; but its concurrency safety is the **exact unproven thing**.
3. **One loader, `num_runners=N`** — AOTI's native concurrency pool (separate per-runner IO, shared weights);
   never mentioned in the plan, yet possibly the cleanest answer.
These differ in thread-safety, memory, and stream binding. The 0.1b OOM proves the plan implicitly needs the
shared-constants path — i.e. the unproven one.
**Edit:** Step 1 must pick + justify a mechanism, and B1/B2's gates must run against THAT mechanism. State the
memory math (N loaders × scratch + ONE 2.5GB weight set) and the fallback if `run()` isn't concurrency-safe.

---

## MAJORs

### M1 — "Meaningfully" is not a gate; and the 5090 number is not the L40S gate
Step 1: "HARD GATE: does concurrent native dispatch lift the knee **meaningfully**." Undefined. The pre-registered
gate is **≥1.5× on L40S → ≥~28/box** (`0.1b-microbench-spec.md:43-45`), but Step 1 runs on the **5090**, where
even the mock already showed **≥3×** vs L40S **~2–2.5×** (`RESULTS-5090.md:26` vs `RESULTS-L40S.md:25`). A 5090
PASS therefore cannot establish the L40S gate.
**Edit:** split the gate. Step-1 (5090) PASS = {overlap exists (B1) ∧ correctness holds (B2) ∧ mechanism works
(B3) ∧ 5090 real-decode multiplier ≥ a pre-registered X}. The **gating ≥1.5×** number is L40S-only, set in
Step 4/5. Label Step 1 the *conjunct-2 existence + mechanism* gate, not "the decisive gate" for the density
number.

### M2 — G2 (P50↔P95 spread) is barely in the plan, and the load model may preclude a real tail
The user's second goal is the **tail**. The plan mentions "latency tail" once (Step 1) and "SLO-robust" — but
makes no P95−P50 target a first-class metric, and never reconciles with the project's own roofline finding that
**TTFT P50 is VAD+WAN-bound (~12–19ms movable)** and the tail is single-thread intake. So:
- The native runtime cannot move the VAD/WAN part of TTFT; what it CAN move is the **server-side contribution to
  the tail** (queueing behind finalizes / intake serialization). The plan should explicitly decompose
  TTFT = VAD + WAN + server-side, state Phase 2 targets the server-side tail, and **measure P95−P50 of that
  component** as a named gate quantity.
- Worse, a tail only emerges from VARIANCE. Real decode is label-looping = **data-dependent variable iterations**
  per chunk; finalizes land **asynchronously** per stream. The 0.1b harness used identical zero inputs
  (`microbench.cpp:85 make_proto` → zeros) and a uniform-random finalize coin (`:182`) with a fixed host sleep
  (`:181`). A fixed-trace / constant-cost replay manufactures a SMOOTH load that **understates the tail** — i.e.
  measures the opposite of what G2 needs.
**Edit:** Step 1 must (a) run the real data-dependent decode (not a recorded constant-cost trace), (b) stagger
per-worker phase + use a MIX of utterance lengths, (c) model finalizes as per-stream async events, and (d) report
P95−P50 (and P99−P50) of the server-side first-token/chunk latency as an explicit gate, mapped to G2.

### M3 — Finalize is omitted from Step 1's scope (repeats the 0.1b's headline caveat)
Step 1 scope = "steady AOTI + decode." No real finalize path. Finalize is the heaviest GPU burst (lowers the
ceiling — G1), the `load_constants`/bucket-routing path that is the prime concurrency hazard (B2/B3), AND the
source of asynchronous bursts that drive the tail (G2). The 0.1b only APPROXIMATED it as extra replays
(`microbench.cpp:130-134`) and flagged that as a top caveat (`RESULTS-L40S.md:38-39`).
**Edit:** include the real finalize path (fork + bucket `load_constants`/route + heavier encoder + continuation
decode) in Step 1, or explicitly mark Step 1's number a steady-only UPPER bound and move the finalize-inclusive
number to a named sub-step before any GO.

### M4 — GPU util% is a misleading success signal for a BW-bound kernel
The plan's success cue is "GPU util > the Python stack's 46–65%." `nvmlDeviceGetUtilizationRates`
(`microbench.cpp:199`) reports *fraction of time ≥1 kernel ran* — it reads 80–98% while SMs stall on memory (see
the L40S 8×48 row: **98% util AND runaway**, `RESULTS-L40S.md:21`). For a mem-BW-bound encoder, high util can
coincide with SLO violation and is NOT evidence the idle GPU was usefully reclaimed.
**Edit:** report achieved DRAM throughput / SM-active-vs-memory-stall (Nsight/CUPTI or even a BW proxy), treat
**density-at-SLO** as the metric and util as a diagnostic. Otherwise G1 "we reclaimed the idle GPU" is unfalsifiable.

### M5 — The overlap thesis is in tension with the encoder being BW-bound; Step 1 must name the binding resource
Roofline says the steady encoder is ~3.4× above the L40S BW floor — it already consumes substantial memory BW at
B=1. Two concurrent encoders on two streams cannot fully overlap if they contend on the same BW; what overlaps is
the host/launch-bound decode + idle gaps. This *predicts* a ceiling near the mock's ~2–2.5× (mock decode = pure
host = perfectly overlappable; real decode has GPU ops that contend, so the real ceiling is ≤ mock). The plan
treats "does it overlap" as binary; the useful question is **which resource saturates at the knee** — if memory
BW, the ceiling is a hardware floor (and pre-confirms the L4/Spark negative in Step 5); if launch/host, there's
headroom.
**Edit:** state the mechanistic prediction and require Step 1 to attribute the knee to a resource (BW vs launch
vs host), not just report a knee.

---

## QUESTIONS / MINOR
- **Framing (now confirmed):** the native decode is **pure C++, no GIL** — `decode_range` runs
  `joint.forward`/`predict.forward` (TorchScript) with a per-label `argmax().item<int64_t>()` device→host sync
  (`session_main.cpp:1648,1657,1679`). So the spike has **no GIL at all**; Step 1's real variables are CUDA
  **context serialization + memory BW + cross-runner aliasing**, NOT "no-GIL intake." The plan's "GIL/scheduler-
  bound" language describes only the *Python baseline*. Rename Step 1's question accordingly.
  - *Mechanistic upside (supports the thesis):* that per-label `.item()` sync is exactly the host-bound,
    GPU-idle window the overlap thesis says N threads can hide → the mechanism is sound IF `run()` overlaps (B1).
  - *Concurrency surface to handle:* give each thread its OWN `joint`/`predict` module handles (the mock did this
    — `microbench.cpp:103` "each lane gets its own module handle; forward is stateless post-load"); concurrent
    `forward()` on one shared module is an avoidable hazard. Cheap (TorchScript modules are small).
- **Q (claim hygiene):** The plan says Step 1 uses "the validated single-stream native core." That core validated
  *compute correctness*; it has no streams and one loader. Say plainly: compute correctness transfers; the
  concurrency primitives (per-thread streams, N loaders/`num_runners`, stream-respecting `run()`) are NEW and
  unvalidated.
- **Q (Step 4):** Confirm "apples-to-apples" means SAME SLO + SAME semantic-WER tool + SAME hardware AND a
  **freshly re-measured** Python baseline (memory: baseline not frozen) — not the stale ~16–20 number.
- **Q (Step 5 linkage):** If Step 1 finds BW-binding at the knee even on 5090/L40S, that partially pre-confirms
  the L4/Spark "no-lift" hypothesis — note the linkage so Step 5 isn't treated as fully independent.
- **MINOR:** Pin the exact libtorch/AOTI version Step 1 tests; `run()`/`load_constants` concurrency semantics are
  version-sensitive (PLAN_RULES pins 2.8.0+cu128 — restate it in Step 1 since the whole result hinges on that
  API's behavior).
- **MINOR:** `run_single_threaded=false` is already passed to the ctor (`:3868`); document what that flag does
  for concurrent use (it gates internal runner locking) — it interacts directly with B3.

## What a corrected Step 1 proves (and doesn't)
With B1–B3 + M1–M5 folded, Step 1 (5090) can honestly establish: concurrent AOTI dispatch overlaps (or not) on a
single context; the mechanism is correctness-preserving; the real-decode multiplier and the binding resource on
the 5090; and a representative server-side P95−P50. It cannot establish the **L40S ≥1.5× gate** (Step 4/5) nor
G2's end-to-end TTFT (VAD/WAN are out of scope) — only the server-side tail component. The plan should claim
exactly that and no more.
