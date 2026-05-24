# Opus adversarial review — Round 4 (convergence; structural + consistency)

## [BLOCKER] 1. No integrated decision tree — ~10 spikes each with a local go/no-go, but nothing maps OUTCOMES → {kill / B4 / B1 / fusion}
v4 has 0.0, 0.1, 0.2, 0.3, 0.5, 0.6a, 0.7, 0.8, 0.9, 0.10, 0.11 — each with its own gate — but the reader cannot tell
what *combination* of outcomes means "proceed," "fall back to B4," or "abandon." 0.4 says "pick language+backend" but
not the logic. This is the single biggest remaining gap.
- **Recommendation:** Add an explicit decision tree to 0.4, e.g.:
  1. **0.0 residual gap small** (Python plan ~reaches 40/box + bounded tail) → **STOP** (Track-A learnings archived).
  2. **0.1 shows the ceiling is MPS/context-launch-/bandwidth-bound, not GIL/single-thread** (a single native process
     can't overlap finalize+steady without MPS/multi-proc) → **STOP / thesis dead** (native ≈ Python topology).
  3. **0.3 (py3.13t) closes the residual gap end-to-end AND free-threaded PyTorch/NeMo is stable** → **choose B4**
     (cheapest; NO native decode/encoder ports needed) — *this is the preferred WIN when it works*.
  4. **Else, if 0.1 positive AND 0.6a + 0.2 + 0.8 + 0.11 pass** → **proceed B1** (full native).
  5. **0.6a fails byte/state equivalence** → B1 byte-exact dead → either accept T1-only native decode (explicit added
     risk) or **stay B4 / STOP**.
  6. **3.3 fusion** is independent and gates ONLY the 6–10 ms headline, never the core go/no-go.
  Make 0.4 emit this tree filled in with the measured numbers.

## [BLOCKER] 2. Internal contradiction: risk #2 and the §3 B1b row still name "solve the Blackwell cuda-graph-decoder" as the FUNDING GATE — round 3 removed it
- §3 B1b row (line 168) and risk #2 (line ~406) still say the funding gate is "Spike 0.6" and include "the Blackwell
  cuda-graph-decoder NeMo punted on." But round 3 split 0.6→**0.6a (deployed EAGER, `use_cuda_graph_decoder=False`)** and
  explicitly **excluded** the Blackwell graph-decoder from the gate (it's 0.6b research). These lines now contradict the
  0.6a spike. Step 1.2 (line 359) and the §3 table also still say "0.6" not "0.6a."
- **Recommendation:** Replace all funding-gate references with **0.6a**, and strike "solve the Blackwell
  cuda-graph-decoder" from the gate/risk text (move it to a 0.6b/Phase-3 research note). Fix the §3 B1b row + step 1.2.

## [MAJOR] 3. B4 (free-threaded py3.13t) is mis-framed as a mere "fallback" — it is the PREFERRED outcome when it works, and it does NOT need the expensive native ports
The plan treats B4 as "fall back if export stalls." But B4 keeps NeMo's Python decode/encoder — so if 0.3 shows
py3.13t + a rethreaded scheduler closes the gap, B4 **avoids 0.6a/0.2/0.8 entirely** (the ~12–20 wk Budget-A core).
That makes B4 the cheapest *success* path, not a consolation prize. Burying it inverts the economics.
- **Recommendation:** Reframe B4 in §3 and the decision tree as **"the cheapest path; preferred if 0.3 + free-threaded
  maturity hold"**, and state clearly that the B1 native ports are funded ONLY if B4 is insufficient or immature.

## [MAJOR] 4. Spike ORDERING still front-loads ~12–20 eng-wk of Budget-A ports before the cheap kill-shots
v4 says Track A "runs now, parallel with the Python plan" — but Track A contains the *expensive* byte-exact ports
(0.6a 4–8 wk, 0.8 2–3 wk, 0.2). The cheap kill-shots are 0.0 (gate), 0.1 (ablation), 0.3 (py3.13t), 0.5 (sim), 0.9/0.11
(audits/specs). Running 0.6a before 0.0/0.1/0.3 resolve risks sinking 4–8 weeks into a decode port for a project that
0.0 or 0.1 would have killed, or that B4 would have made unnecessary.
- **Recommendation:** Re-order independent of track: **Wave 1 (cheap, decides existence + path):** 0.0, 0.1, 0.3, 0.5,
  0.7, 0.9, 0.11. **Wave 2 (expensive ports, funded only if Wave 1 ⇒ "alive AND B1-path"):** 0.6a, 0.8, 0.2, 0.10.
  Keep "baseline-independent" (frozen fixtures) so Wave 2 *can* start early, but don't *spend* it until Wave 1 clears.

## [MAJOR] 5. Rollout/coexistence (Phase 5) is hand-wavy about the topology change it implies
The density thesis is **single-process-multi-lane** (shared weights, no MPS tax). But production is
**multi-process + MPS + HAProxy leastconn maxconn≈12** (`deploy/launch_multiproc.sh:57-68`), and the launcher already
has open TODOs for **LB drain, alerting, MPS-context restart after crash** (`:70-79`). A single-process-multi-lane
native runtime is a *different* deploy shape: per-process maxconn, health checks, the MPS daemon, and restart semantics
all change. v4 doesn't address this or how traffic is cut over.
- **Recommendation:** Phase 5 must (a) state the target topology (single-proc-multi-lane vs still-multi-proc) as an
  *output of 0.1*, and reconcile it with HAProxy/MPS/the launcher; (b) add a **shadow/mirror-traffic validation**
  harness (tee live/replayed audio to the native runtime, diff its event stream vs Python without serving output) as
  the strongest T1 check — this was recommended in round 2 but never entered the plan; (c) define canary-by-replica
  behind the LB and **rollback = shift LB weights back to Python**.

## [MAJOR] 6. The plan never states the NARROW triple-conjunction bet up front — leadership can't see how thin the live path is
This project lives only if **ALL THREE** hold: (i) the Python plan leaves a *large* residual density/tail gap; AND
(ii) that gap is **GIL/single-thread-bound**, not MPS/context-launch-/bandwidth-bound (else native = Python topology);
AND (iii) **B4 (free-threaded Python) is insufficient or not production-ready** (else B4 wins far cheaper). Each
conjunct has a real chance of being false. That's a narrow bet for ~1 eng-year + a second stack.
- **Recommendation:** Put this triple-conjunction at the top of §1 (or §0) as "the bet," with the early-exit mapped to
  each conjunct (0.0 / 0.1 / 0.3). Honesty here is the plan's most valuable feature.

## [MINOR] 7. Over-build check — mostly clean, two watch-items
- 0.7 aarch64 pre-check before DGX Spark exists is cheap and gates a platform → keep.
- 0.11 graph-ownership spike is justified (density depends on it) → keep.
- Watch: the full **per-utterance + state-machine + cross-B + byte-exact** correctness matrix is large; for the *cheap*
  Wave-1 decision you do NOT need all of it — only enough to know B1 is *feasible*. Don't gold-plate the correctness
  harness before the worth-it gate clears.

## [MINOR] 8. Numbering/labels
- 0.11 (Track A, Phase 0) is numerically after 0.10 (Phase 0.5) but ordered before it — harmless but rename for
  reading order (e.g. 0.11→0.12 or just present 0.10 last consistently). Ensure the progress table order matches the
  wave ordering from finding #4.

## Top 5 things to fix
1. **Add the integrated decision tree to 0.4** (outcome combinations → kill / B4 / B1 / fusion).
2. **Fix the contradiction:** funding gate is **0.6a** (eager, `use_cuda_graph_decoder=False`); strike "solve the
   Blackwell cuda-graph-decoder" from the gate/risk #2/§3 B1b row; update step 1.2's "0.6"→"0.6a".
3. **Reframe B4 as the preferred cheapest path** (avoids the native ports) when 0.3 + free-threaded maturity hold.
4. **Re-order spikes into Wave 1 (cheap, decides existence+path) vs Wave 2 (expensive ports, funded only if Wave 1
   clears)** — don't sink 0.6a before 0.0/0.1/0.3.
5. **Make Phase 5 concrete:** topology change (single-proc vs multi-proc, output of 0.1) reconciled with HAProxy/MPS +
   shadow-traffic validation + canary/rollback; and **state the narrow triple-conjunction bet + per-conjunct early-exit
   up front.**
