# Phase-2 plan — Round 5 FOLDED + FINAL SYNTHESIS (Codex + Opus)

Inputs: `codex-phase2-round5.md`, `opus-phase2-round5.md`, and the four prior folds. This is the definitive
synthesis of the 5-round paired adversarial review. Goals: **G1 density**, **G2 P50↔P95 tail**.

## Convergence verdict
After 5 rounds, the two independent reviewers have **NO standing disagreement** and produced **near-identical**
must-have edit lists and sign-offs. Every cross-reviewer divergence was resolved by adoption within a later round
(BW-bound → hypothesis; hot-bucket → secondary; 5090 bar → ≥2.00×; tail bar → non-regression in the real regime;
Step-1b → ceiling ≥1.80×). That convergence is itself a sign-off signal.

## SIGN-OFF (both reviewers)
**GO to build Step 1 — starting with the Step-0 cheap kill-gates — CONDITIONAL on the 8 must-have plan edits
below and the 2 user ratifications.** No remaining reviewer-discovered engineering blocker to starting the
Step-0/1 harness. The thesis the user stated (one C++ process, true threads, one shared weight set) is correct AND
natively supported (`num_runners=N` shares one weight copy; per-stream state is cleanly isolated). The single
decisive unknown is whether the AOTI container's **execution lock** lets concurrent `run()` actually overlap on
the GPU — Step-0/K1 measures exactly that, cheaply, with a **de-risked fallback** (per-thread CUDA-graph-of-AOTI,
the primitive already shipping in the Python finalize-graph win) if it doesn't.

## THE 8 MUST-HAVE EDITS to PHASE2-PLAN.md (both lists merged; ranked)
1. **Replace Step 1's "meaningfully" with the staged numeric GO/STOP tree:** Step 0 kill-gates → Step 1a 5090
   (PASS ≥2.00×, STOP <1.50×, 1.50–2.00× tune) → Step 1b L40S ceiling (PASS ≥max(34, 1.80·S_py_L40S), STOP
   <G1_floor) → Step 4 realized (GO `S_native_step4 ≥ G1_floor = max(28, 1.50·S_py_L40S)` + reject/error/WER/SLO
   bounds). **Consistency fix (Codex R5):** Step-1b tail is a YELLOW/build-risk signal (GREEN if ≤1.10× Python
   spread), NOT an auto-STOP — the binding tail is Step 4; STOP only on absolute p95/p99/WER/correctness fail or a
   gross spread regression > max(1.25×, +10ms).
2. **Add the user-ratified goal contract** (the 2 user questions; instrument both TTFT and TTFS if undecided —
   don't call Step 4 a two-goal pass until ratified).
3. **Pin the ownership/topology contract:** one shared steady AOTI loader `num_runners=N` + explicit per-worker
   streams + per-worker `SessionState`/`AudioFrontend`/`enc_first`/`joint`/`predict`/`preproc`; `user_managed`/
   codisk constants for finalize buckets; mutex / default-stream / `num_runners=1` negative controls.
4. **Correctness-before-performance, non-negotiable + FIRST:** concurrent==serial token/event 0-mismatch over real
   decode + real finalize + same/mixed/hot-bucket + collector fields + stale-gen-sensitive events; semantic-WER ≤
   max(WER_py+0.5pp, 1.10·WER_py).
5. **Telemetry + artifact schema** (so Step 2 can be designed): throughput, p50/p95/p99, **P95−P50/P99−P50**,
   enqueue→first, enqueue→final, queue/runner/`.item()`/finalize waits, CUDA-event durations, Nsight/CUPTI
   counters, memory, CPU/loadgen health, `num_runners`/stream-mode/corpus-SHA/artifacts/topology in artifact names.
   Step 2 invalid if it cites only a scalar knee.
6. **False-STOP protocol:** a STOP-candidate is not a STOP until 3 runs (CV ≤10%) + negative controls behave +
   topology sweep/fallbacks tested + counters identify a real (hardware) limit + harness health logged. A passing
   fallback topology is a PIVOT, not a STOP.
7. **Step 2/3 preconditions written now** (they shape Step-1 telemetry): box-global active + backlog-count
   admission (shed=close; count admitted not offered; two curves; ready-age stays dead); a declared numeric
   priority-finalize-lane policy over the `num_runners=N` pool (e.g. `N_finalize_reserved≥1` + steady-starvation
   p95 ≤2× no-finalize, or weighted with finalize runner-wait ≤25% TTFS); Step 3 WS-tail microbench +
   stale-generation suppression gate before Step 4.
8. **Step 4 apples-to-apples contract:** re-measure Python back-to-back same L40S; pin corpus/commits/artifacts/
   loadgen-env/driver/server-flags (incl. `NEMOTRON_SYNC_COMPRESS`/`NEMOTRON_FINALIZE_PRIORITY`)/admission-caps/
   WER-config; report no-shed + admitted-through-shed; **count admitted-successful only**; rejects ≤10%, errors
   ≤1%. **Add the multi-turn decision:** either a multi-turn subcurve OR explicitly scope the gate to
   single-utterance (the loadgen is one-utterance-per-connection → multi-turn-under-load is otherwise unmeasured).

Nice-to-have (not a Step-1 blocker): Step 5 per-target confirmation table + Spark aarch64 preflight; "Step 3-lite"
(minimal WS echo + scheduler) for an early realized number before the full Step-3 build.

## COMPLETENESS — final reinforcements (no new blocker)
- **Per-runner ACTIVATION memory scales ×N even with shared weights** → re-check the memory gate at the TARGET N
  (16–40), not just N=4 (weights flat + N×activations must fit 46GB).
- **Density ceiling = min(GPU-BW, CPU-cores, launch/lock)** → a CPU-core-bound knee (32 vCPU box) is a
  "more-cores / fewer-host-syncs" finding, not a STOP; the telemetry's CPU-core util catches it.
- **K1-fail fallback is de-risked** (the CUDA-graph-of-AOTI primitive ships in the Python server) → a
  lock-serialization finding triggers the graph fallback BEFORE any STOP.
- **Multi-turn under load unmeasured** by the one-utterance loadgen → edit #8 covers it.

## WHAT ONLY THE USER CAN SETTLE (the 2 ratifications)
1. **Two-goal gate?** The frozen 0.0 threshold was density-ONLY. Adding G2 makes the technical GO conjunctive.
   Recommendation: adopt G2 as a **NON-REGRESSION guardrail** (don't widen the tail while densifying;
   strict-improvement = upside), because Python's tail is already ~33ms (an over-tight bar would be a wrong
   project-killer). If declined: report G2 separately; don't call a density-only pass a two-goal Phase-2 success.
2. **G2 = first-token TTFT or vad_stop→final TTFS?** The loadgen measures TTFS; first-token TTFT is un-instrumented
   and its P50 is VAD+WAN-dominated (only the server-side component is movable). Recommendation: bind on
   server-side TTFS spread (rename G2), OR add first-token-TTFT instrumentation in Step 3/4 and bind on the
   server-side component of `TTFT_spread`.

## Files
5 paired rounds in `reviews/`: `{codex,opus}-phase2-round{1..5}.md` + `phase2-round{1..5}-FOLDED.md` (this is the
final). Review briefs: `phase2-review-brief*.md`.
