# Phase-2 plan — Round 1 FOLDED (Codex + Opus, independent)

Two independent adversarial reviews (`codex-phase2-round1.md`, `opus-phase2-round1.md`). Strong convergence: both,
without coordinating, landed the same 6 blockers — a high-confidence signal. Foregrounding the two goals:
**G1 = system utilization / streams-box density**, **G2 = tightened P50↔P95 TTFT spread (the tail)**.

## CONVERGENT BLOCKERS (both reviewers, independently) — highest confidence
1. **Gate is a vibe, and 5090 ≠ the L40S gate.** Step 1 gates on "meaningfully" (`PHASE2-PLAN.md:30`) on the
   5090, but the pre-registered gate is numeric + L40S-specific: **≥1.5× / ≥~28 SLO-robust streams/box**
   (`0.1b-microbench-spec.md:42-45`); 5090 already showed ≥3× vs L40S ~2–2.5× even with the mock. → **Split
   Step 1 into 1a (5090 overlap+correctness smoke; permits L40S, not a GO) and 1b (L40S numeric hard gate)**;
   replace "meaningfully" with the number; require P50/P95/P95−P50.
2. **AOTI concurrency primitive unspecified — Step 1 could test the wrong thing.** The mock used per-lane
   TorchScript + captured CUDA-graph `graph.replay()` on per-lane streams (`microbench.cpp:97-124,159-167`); the
   real session uses `AOTIModelPackageLoader::run(inputs)` on the **default stream, one loader, `num_runners=1`**
   (`session_main.cpp:1765,3868`). The explicit-stream API exists and is byte-exact *single-threaded*
   (`aoti_encoder_main.cpp:64-65,106`) but concurrent overlap/aliasing is untested. → **Pre-register + A/B the
   loader topology**: one shared loader (num_runners=1); one loader num_runners=N (if 2.8 supports it);
   per-worker loaders sharing one weight-tensor map; + a mutex-serialized negative control. For each: explicit
   per-worker stream + CUDA-event timing + "do kernels actually overlap?"
3. **Shared weights proven only serial; steady encoder isn't even in shared-weight format.** The proof loads+runs
   once (`validate_shared_weights.py:13-25`); no concurrent N-thread `run()` on shared `user_managed` constants
   was ever tested. **NEW (Codex):** the steady path loads `enc_steady_aoti.pt2` directly (`session_main.cpp:3868`)
   while shared-weight sharing needs a **constants-on-disk** package (`enc_steady_codisk.pt2`,
   `validate_shared_weights.py:6`) — that artifact doesn't exist for steady yet. → Build the codisk steady
   package; assert peak GPU mem proves ONE weight copy (not N); assert concurrent==serial outputs.
4. **Speed measured before concurrent CORRECTNESS proven.** Step 1 lists throughput/tail/util/knee but defers
   "correct per-stream events" to Step 3 (`PHASE2-PLAN.md:36`); PLAN_RULES requires token+event-exact, not
   WER-only (`PLAN_RULES.md:8-16`). A race in AOTI runner scratch / shared constants / `joint`/`predict` /
   cache ownership → fast-but-WRONG transcripts reads as a GO. → **Mandatory "correctness before perf" subgate:**
   N-thread run over the serial corpus, assert each stream's final tokens + ordered event text == serial
   `session_main` (the FORK_ASSERT analog for threads), before any throughput number is trusted.
5. **Real finalize path omitted.** Step 1 = "steady AOTI + decode" only; 0.1b explicitly had no real finalize
   (`RESULTS-L40S.md:38-39`; 5090 "finalize" was just extra replays + sleep). Finalize is the heaviest GPU burst
   (lowers the G1 ceiling), the `load_constants`/bucket path (the concurrency hazard), AND the async-burst source
   that drives the G2 tail. → Include real fork+bucket-route+heavier-encoder+continuation-decode in the gated
   workload; report steady-only as an ablation/upper-bound, not the gate.
6. **G2 (P50↔P95 spread) is not a first-class gate.** Mentioned once (`PHASE2-PLAN.md:27`), no TTFT/SLO/spread
   definition or target. 0.1b used a chunk-intake→done keep-up proxy, NOT client TTFT (`RESULTS-5090.md:3-6`). →
   Step 1 & 4 must report `TTFT_p50/p95/p99`, **`P95−P50`**, + queue/lane/finalize wait, with an explicit
   improvement-vs-Python-baseline target on the same hardware/load.

## DISTINCTIVE — Codex (preserve)
- **M4 Hot-bucket finalize collision.** One shared loader per `(drop,T)` bucket (`session_main.cpp:1545-1580`);
  under load many streams finalize into the SAME bucket → concurrent `run()` on one loader serializes/races.
  A trace with distinct buckets passes while production hot buckets stall (hits G1 *and* G2 P95). → Add a
  hot-bucket stress case; design a per-bucket runner pool if one-loader concurrency is unsafe.
- **M5 Step 2 can't be designed from a knee alone.** Scheduler needs queue depth, per-phase service time, CUDA-
  event durations, finalize wait, admission/shed counters, CPU-core util. → Step 1 must emit a **scheduler
  telemetry schema**; Step 2 is blocked on that telemetry, not just the knee number.
- **M6 Stale-generation suppression is DEFERRED to Phase 2** — the session itself prints
  `stale_generation=DEFERRED_PHASE2_SERVER_ORACLE` (`session_main.cpp:3533-3537`; `step1-event-FOLDED.md:24-28`).
  The plan's "token/event-exact" premise (`PHASE2-PLAN.md:5-14`) hides this. It's a *scheduler/tail* correctness
  issue: under overload, wrong stale-final suppression can make the tail look artificially better by dropping/
  misordering finals. → Reword the premise; add stale-final/generation suppression to Step 3 *before* Step 4.
- **m1 First chunk is still TorchScript** (`enc_first.ts`, `session_main.cpp:1715-1726,3034-3036`), not AOTI →
  mixed dispatch primitives in the density claim. → Either exclude+label or add an AOTI first-chunk path.
- **m3** No Phase-2 harness target in `CMakeLists.txt` yet.

## DISTINCTIVE — Opus (preserve)
- **M5(opus) Name the binding resource — the overlap thesis collides with a BW-bound encoder.** Roofline: steady
  encoder ~3.4× above the L40S BW floor → concurrent encoders contend on memory BW and can't fully overlap; what
  overlaps is the host/launch-bound decode (the per-label `.item()` sync, `session_main.cpp:1657`, is exactly the
  GPU-idle window threads fill) + idle gaps. This *predicts* a ceiling ≤ the mock's ~2–2.5× (mock decode = pure
  host = perfectly overlappable; real decode has contending GPU ops). → Step 1 must **attribute the knee to a
  resource (BW vs launch vs host)**, not just report a knee. If BW-bound, the ceiling is a hardware floor and
  this *pre-confirms the Step-5 L4/Spark "no-lift" hypothesis* — note the linkage.
- **G2 honesty: decompose TTFT.** The project's own roofline says TTFT P50 is **VAD+WAN-bound (~12–19ms
  movable)**; the native runtime cannot move VAD/WAN. So the plan must state TTFT = VAD + WAN + **server-side**,
  and that Phase 2 only moves the **server-side tail component** — measure P95−P50 of *that*. As written the plan
  could "succeed" on density while the user's literal G2 (end-to-end TTFT spread) is mostly outside its reach;
  say so explicitly to avoid promising what the runtime can't deliver.
- **Mechanism/framing:** native decode is **pure C++, no GIL**; Step 1's real variables are CUDA context
  serialization + BW + cross-runner aliasing, NOT "no-GIL intake." Give each thread its own `joint`/`predict`
  handles (mock did — `microbench.cpp:103`).

## CONSOLIDATED REQUIRED PLAN EDITS (Round-1 output)
1. Two-stage Step 1: **1a** 5090 overlap+correctness smoke (binary, permits L40S) → **1b** L40S numeric hard
   gate (≥1.5× / ≥~28 SLO-robust streams/box, real finalize, same SLO as Python, baseline re-measured).
2. Pre-register loader topology + explicit per-worker stream + shared-weight artifact (codisk steady) +
   memory-proof-of-one-copy + mutex negative control.
3. **Correctness-before-perf gate:** concurrent==serial token/event equality (incl. audio + multiturn +
   real finalize + hot-bucket collision) before any throughput number.
4. Make **G2** first-class: define TTFT/SLO, report P50/P95/P99 + **P95−P50** + per-phase waits; decompose
   TTFT (VAD+WAN out of scope; target the server-side tail); set an improvement-vs-baseline target.
5. Require the knee to be **attributed to a binding resource** (BW vs launch vs host) + CUDA-event/Nsight overlap
   evidence, not NVML averages.
6. Emit a **scheduler telemetry schema** in Step 1; block Step 2 on it. Add fallback (MPS/per-context) numeric
   subtest + a STOP threshold tied to L40S density AND tail.
7. Reword the Phase-1 "token/event-exact" premise to exclude stale-generation suppression; pull stale-final +
   first-chunk-AOTI into Step 3 before Step-4 benchmarking.

## OPEN FOR ROUND 2 (adversarial on this round + deeper code)
- Pressure-test these blockers: is the hot-bucket case actually reachable given the real (drop,T) distribution?
  Does libtorch 2.8 `num_runners>1` actually exist + give a runner pool? Is the BW-bound prediction right (or is
  there SM headroom)? Is the Step 1a/1b split the right cut?
- Deeper code Round 1 only sampled: per-stream **state/cache/audio-ring isolation** under threads (is any state
  static/shared?); the finalize fork/clone path's concurrency; whether `joint`/`predict` TorchScript `forward`
  is genuinely concurrent-safe; the `argmax().item()` per-label sync's interaction with N streams.
