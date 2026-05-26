# Opus — Phase-2 plan review, Round 5 (final convergence + sign-off)

Final round. Convergence, completeness, must-have edits, sign-off. Goals: **G1 density**, **G2 P50↔P95 tail**.

## 1. Residual-disagreement scan — NONE standing
Every point where the two reviewers diverged got resolved by adoption within a later round:
- **BW-bound:** Opus R1 stated it as near-fact → Codex R2 downgraded to a hypothesis-to-measure → Opus adopted.
  RESOLVED (it's a hypothesis; require counter attribution).
- **Hot-bucket vs steady loader:** Codex R1 made hot-bucket a MAJOR → Opus R2 argued steady loader is primary →
  Codex R2 downgraded its own M4. RESOLVED (steady loader primary, hot-bucket secondary stress).
- **5090 pass bar:** Opus R4 proposed a ≥1.5× floor → Codex R4 argued ≥2.00× (spend-control margin for the L40S
  haircut) → Opus adopted. RESOLVED (≥2.00× to pass to 1b; <1.50× STOP; 1.50–2.00× tune).
- **G2 tail bar:** Opus R4 "non-regression" vs Codex R4 piecewise (≥max(10ms,10%) better; if Py<50ms within +5ms)
  — these COINCIDE in the actual regime (Python tail ≈33ms < 50ms → bar = within +5ms, p95 no worse =
  non-regression). RESOLVED (converged).
- **Step 1b threshold:** both independently derived ≥1.80× / ≥34 as the ceiling (28/0.83), explicitly DISTINCT
  from the 1.5× realized gate. RESOLVED (converged).

**Bottom line: after 5 rounds, two independent adversarial reviewers have NO standing disagreement on substance.**
That convergence is itself a sign-off signal.

## 2. Completeness — three reinforcements, no new blocker
The decision tree (Round 4) gates G1 and G2 end-to-end. Final re-scan surfaces three items worth stating
explicitly (none overturns the tree):
- **C1 — Per-runner ACTIVATION memory scales with N, even with shared weights.** Round 2's K3 "memory-flat" check
  proves WEIGHTS are shared (one 2.5GB copy across `num_runners=N`), but each runner has its OWN IO/activation
  buffers — these scale ×N. The <1.35×N=1 bound is for N=4; **at the target N (16–40) the memory gate must be
  re-checked: weights flat + N×activation must fit the 46GB L40S.** Add the target-N memory check, not just N=4.
- **C2 — The density ceiling = min(GPU-BW, CPU-cores, launch).** The decode `.item()` loop + preproc + mel are
  host/CPU work; N threads need N non-contending cores (L40S box = 32 vCPU). So native density is bounded by
  `min(GPU ceiling, CPU-core ceiling, BW)`. Codex's telemetry schema already includes CPU-core util — good; the
  plan should STATE that the ceiling is a min over three resources, so a CPU-core-bound knee isn't misread as a
  GPU/lock failure (it would be a "buy more cores / fewer host-syncs" finding, not a STOP).
- **C3 — The K1-fail fallback is DE-RISKED, which protects against a false STOP.** If the AOTI execution lock
  serializes dispatch (K1 fail), the fallback is a per-thread **CUDA-graph-of-the-AOTI-steady** — and that exact
  primitive already ships in the Python server (the finalize-CUDA-graph win, `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE`).
  So the fallback is not speculative; a K1 fail is a "switch dispatch primitive," not a project STOP. State this in
  the STOP-corroboration protocol (a lock-serialization finding → try the graph primitive BEFORE declaring STOP).

## 3. Minimum-viable MUST-HAVE edits to PHASE2-PLAN.md (ranked)
Before Step 1 is built, the plan MUST change these (everything else is nice-to-have):
1. **Replace Step 1's "meaningfully" gate with the staged numeric tree** (Step 0 kill-gates → 1a 5090 ≥2.00× →
   1b L40S ≥1.80×/≥34 ceiling → 4 realized ≥1.5×/≥28). [Round 4 tree]
2. **Add the Step-0 cheap kill-gates** (0a steady `num_runners=N` overlap + concurrent==serial + memory-flat;
   0b decode per-thread-handles + `.item()`/default-stream check; 0c real finalize same/mixed-bucket equality) —
   run BEFORE the full harness / any EC2 spend.
3. **Pin the loader/object-ownership model:** one shared AOTI loader `num_runners=N` (shares weights; the residual
   risk is the container execution lock) + per-thread `SessionState`/`AudioFrontend`/streams/`joint`/`predict`/
   `enc_first`/`preproc`; `user_managed` shared constants for the finalize buckets only.
4. **Make the concurrency-correctness gate mandatory and FIRST:** concurrent==serial token/event-exact (incl.
   finalize + hot-bucket) before ANY throughput number is trusted.
5. **Make G2 first-class:** define the metric (TTFS spread now; TTFT iff instrumented), report
   p50/p95/p99/P95−P50 + per-phase waits in every table; decompose TTFT=VAD+WAN+server-side and scope Phase 2 to
   the server-side component.
6. **Require knee attribution to a binding resource** (BW vs launch vs lock vs CPU-core) via counters, not NVML
   util; "BW-bound" is a hypothesis.
7. **Label Step 1b CEILING vs Step 4 REALIZED**; require the 1.80× ceiling margin; add the corroborated-STOP
   protocol (3 runs + negative control + topology sweep + counter attribution + try-the-graph-fallback).
8. **Step 2/3/4 preconditions:** box-global active + backlog-count admission (shed=close, count admitted not
   offered, two curves); declared numeric priority-finalize-lane policy over the pool; WS-tail microbench +
   decomposition; stale-generation suppression as a Step-3 gate; apples-to-apples MANIFEST + re-measured Python
   baseline; Step 5 reframed as confirmation-of-Step-1 + Spark preflight.

## 4. Two items only the USER can settle (surface, don't decide)
1. **Ratify the two-goal gate.** The frozen 0.0 threshold was density-ONLY; adding G2 makes the technical GO
   conjunctive. Recommend adopting G2 as a **non-regression guardrail** (don't widen the tail while densifying),
   strict-improvement as upside.
2. **G2 = first-token TTFT or vad_stop→final TTFS?** The loadgen measures TTFS; first-token TTFT is currently
   un-instrumented (and its P50 is VAD+WAN-dominated, only the server-side component is movable). Pick: rename G2
   to server-side TTFS, or instrument first-token TTFT in Step 3/4.

## 5. SIGN-OFF
**GO to build — starting with the Step-0 kill-gates — CONDITIONAL on the 8 must-have edits and the 2 user
ratifications.** The plan's bones are sound (the step progression is right; the expensive WS build is correctly
gated behind the cheap L40S number). The thesis the user articulated — one C++ process, true threads, one shared
weight set — is correct AND natively supported (`num_runners=N` shares one weight copy; per-stream state is
cleanly isolated). The single decisive unknown is whether the AOTI container's execution lock lets concurrent
`run()` actually overlap on the GPU — and Step-0/K1 measures exactly that, cheaply, with a de-risked graph
fallback if it doesn't. No standing reviewer disagreement. The honesty bar from Phase 1 (no mock, SLO-robust not
keep-up, attribute the knee, report the spread) is now encoded in the gates.

Remaining risk is execution discipline, not plan soundness: build the Step-0 gates faithfully (real decode, real
finalize, per-thread streams wired, SLO-robust), and let a STOP be corroborated before it's believed.
