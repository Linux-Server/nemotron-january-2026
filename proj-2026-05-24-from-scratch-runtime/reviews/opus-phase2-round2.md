# Opus — Phase-2 plan review, Round 2 (adversarial-on-Round-1 + deeper concurrency code)

Read `phase2-round1-FOLDED.md` first. This round (a) attacks Round 1, (b) goes deeper into the concurrency code,
(c) adds what Round 1 missed. Goals foregrounded: **G1 density**, **G2 P50↔P95 tail**.

## A. Where I DISAGREE WITH / DOWNGRADE Round 1 (the honest part)

### A1 — DOWNGRADE R1's hot-bucket finalize collision; the STEADY loader is the primary, constant hazard
Round 1 (Codex M4) made the hot finalize-bucket collision a MAJOR. On re-read it's real but **secondary**. The
*acute* concurrent-`run()` hazard is the single `enc_steady` loader: **every stream calls `run_steady_encoder`
on the ONE shared `enc_steady` loader every ~160ms** (`session_main.cpp:1753-1765`, one loader at `:3868`). At a
~28-stream knee that's ~28 concurrent `run()` on one loader *continuously*; finalize-bucket collisions happen
~once per utterance spread across ~32 buckets (`session_main.cpp:1545-1580`) — far rarer. So the loader-topology
decision (R1-B2/B3) is **forced by the steady path**, not finalize. Re-rank: steady-loader concurrency is the
gate; hot-bucket is a secondary stress case. (This also means the topology must be solved for the *common* path
first — a num_runners pool or per-thread steady loaders.)

### A2 — REFINE R1's BW-bound-ceiling claim (mine, Opus R1-M5): it's a prediction, not a proven wall
"Encoder is mem-BW-bound" does NOT mean a single stream saturates BW. The roofline says the encoder runs **~3.4×
above the L40S BW floor** — i.e. one stream is well below the BW wall, leaving headroom for ~2–2.5 streams before
saturation, which is exactly the observed multiplier (`RESULTS-L40S.md:19-21`: util climbs 40→80→98% as streams
go 16→32→48, runaway at 98%). So the honest claim is: **the ceiling ≈ how many streams' worth of BW fit before
saturation**, and Step 1 must MEASURE the saturation point (achieved DRAM BW vs peak at the knee), not assume
"no overlap." This keeps M5 but corrects it from "BW kills overlap" to "BW sets the ceiling at ~N streams; find N."

### A3 — DOWNGRADE B3 from "invent a mechanism" to "test 3 candidates, one is built-in"
`num_runners` is a real ctor parameter (used as `1` at `session_main.cpp:3868`, `finalize_main.cpp:1573`,
`aoti_encoder_main.cpp:53`). So AOTI's runner-pool path EXISTS; the question is whether `run()` with
`num_runners=N` is concurrency-safe + shares weights, which is a **verify-against-2.8** task, not a from-scratch
design. B3 stands as a blocker (it must be tested) but its *cost* is lower than "design a new sharing scheme."

### A4 — Steelman: the plan's STEP STRUCTURE is sound
Steps 1→2→3→4→5 is the right progression (decisive measurement → scheduler → runtime+WS → apples-to-apples →
per-target). Round 1's findings are about the **precision/completeness of Step 1's measurement and the G2 gap**,
NOT a restructure. The bones are good; don't let the blocker count imply "rewrite the plan."

## B. Deeper concurrency code pass (Round 1 only sampled)

### B1 — CONFIRMED: per-stream state is fully isolated → narrows the correctness gate
`SessionState` (`session_main.cpp:~90-130`) holds all recurrent caches (`clc/clt/clcl/g/h/c`), `hyp`, audio rings
(`pending_audio/raw_audio_ring/post_stop_audio`), and text collectors as **instance members** — no shared/static
mutable state. The only file-scope `static` mutable-looking data is `static const std::string marker/unk_surface`
(`:569-570`), which is read-only/thread-safe. **So N threads can each own a `SessionState` safely.** This is good
news and it **focuses Step 1's correctness gate**: the only concurrency-correctness surface is the **shared model
objects**, not the per-stream state. Step 1's concurrent==serial assertion therefore tests exactly the shared
loaders/modules.

### B2 — Shared TorchScript modules are a real concurrency surface; require per-thread handles
The single-stream core loads ONE each of `enc_first`, `joint`, `predict` and reuses them (`session_main.cpp:3862-
3875`); decode calls `joint.forward`/`predict.forward` (`:1648,1679`). `torch::jit::Module::forward` on a *shared*
module from N threads is not reliably safe (the JIT profiling executor mutates per-graph state on first runs). The
mock avoided this by giving each lane its own module handle ("forward is stateless post-load", `microbench.cpp:103`).
→ **Step 1 must give each thread its own `joint`/`predict`/`enc_first` handles** (cheap; these are small `.ts`),
and the plan should state it. This is additional to Codex R1-m1 (which flagged `enc_first` only for AOTI-purity).

### B3 — The per-label `.item()` sync is the overlap engine — and a serialization trap to verify
`decode_range` does `argmax().item<int64_t>()` per emitted/blank symbol (`session_main.cpp:1657`), up to
`MAX_SYMBOLS` per frame. With a per-thread stream this syncs only that thread's stream → the GPU-idle window other
threads fill (the mechanism). BUT `.item()` on a tensor whose producing op was enqueued on the **default** stream
(if any input/intermediate isn't on the per-thread stream) forces a device-wide sync → would serialize ALL
threads. → Step 1 must confirm every decode op (joint/predict/argmax) runs on the worker's stream, and there's no
implicit default-stream dependency. This is the difference between the thesis holding and silently failing.

### B4 — Finalize fork is per-call clone (good), but assert it under concurrency
`clone_state`/`clone_session` (`finalize_main.cpp:123-134`, `session_main.cpp:953`) do real `.clone()` per
finalize → fresh allocations, no shared buffer. FORK_ASSERT proves parent-unchanged *serially*. Under N threads
each finalize clones its own — safe in principle, but Step 1's concurrency-correctness gate must include a
finalize-heavy trace so a cross-thread clone/alias bug can't hide.

## C. What Round 1 missed

### C1 — Front-load the cheap kill-shots as a pre-Step-1 gate (sequencing)
The 3 cheapest decisive micro-checks are each a potential STOP and cost ~hours, not the full sweep:
- **K1 Overlap:** 2 threads, 2 streams, real `run(inputs, stream)` — wall-time vs 2× serial + profiler kernel
  interleave. If no overlap on one context → STOP (or branch to per-thread CUDA-graph / MPS) before building more.
- **K2 Concurrency-correctness:** 2 threads, same bundle, assert token/event == serial. If it races → STOP.
- **K3 Memory-one-copy:** N loaders/runners sharing constants → assert peak GPU mem flat (not N×2.5GB). If it
  duplicates → fix the codisk-steady artifact (R1-B3) before any sweep.
Run K1–K3 FIRST (mirrors the original plan's "front-load cheap kills" path-forward re-sequencing). Only then the
full N-sweep + scheduler telemetry. The plan should name this Step-1.0 gate explicitly.

### C2 — G2 has a queueing subtlety the plan must own: the measured tail is scheduler-dependent
The P95−P50 tail under load is a **queueing** phenomenon, not just per-chunk service time. It depends on the
DISPATCH DISCIPLINE (per-stream thread? M-worker pool + shared queue, like `microbench.cpp:176-190`? priority
finalize lane?). Step 1 has no scheduler yet (that's Step 2). So **Step 1's tail number is the tail of whatever
placeholder discipline the harness uses** — not the eventual one. → Step 1 must (a) use a SIMPLE, documented
dispatch discipline, (b) label its P95−P50 as a *reference/floor* under that discipline, and (c) defer the binding
G2 number to Step 4 (real scheduler + real WS). Otherwise a good or bad Step-1 tail could be an artifact of the
placeholder. This is the deepest G2 point: you cannot fully measure the tail before you've designed the thing that
shapes it — so the plan must sequence G2 as "reference tail in Step 1 → binding tail in Step 4," and say which
scheduler knobs (finalize priority lane, admission) it expects to move it.

### C3 — Quantify finalize arrival rate (bears on G1 ceiling AND G2 tail)
At ~28 streams with ~1 finalize per utterance every few seconds, aggregate finalizes arrive ~every 100–200ms —
frequent enough that the heavier finalize bursts materially raise the steady-state GPU load (lowers the G1
ceiling) and cluster to drive the tail (G2). The plan should state the expected finalize rate at the knee so the
harness's finalize model is calibrated to reality (not the 0.1b's arbitrary `--finalize-every 15`).

## Net for Round 2
Round 1's 6 blockers survive, with two re-rankings: (1) the **steady loader**, not finalize buckets, is the
primary concurrency hazard (A1); (2) the BW ceiling is a *measure-the-saturation-point* task, not a proven wall
(A2). Good news: **per-stream state is cleanly isolated** (B1), so the correctness gate is narrowly about shared
model objects. New: front-load K1–K3 cheap kills (C1); G2's tail is scheduler-dependent so sequence it
reference→binding (C2). The plan's structure is sound; the work is precision in Step 1 + making G2 first-class.
