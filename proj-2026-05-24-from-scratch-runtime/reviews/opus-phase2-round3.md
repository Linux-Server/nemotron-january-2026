# Opus — Phase-2 plan review, Round 3 (downstream Steps 2–5)

Rounds 1–2 hammered Step 1. Round 3 covers Steps 2–5 + whether the corrected plan actually MEASURES **G1
(density)** and **G2 (P50↔P95 tail)** end-to-end. Grounded in `local_lb.py`, `ec2_loadgen.py`, `semantic_wer.py`.

## Headline
The downstream infrastructure largely EXISTS and — importantly — **already measures G2**: `ec2_loadgen.py`
computes `ttfs_p50` and `ttfs_p95` of vad_stop→final (`:77,136-139`), so **Step 4's G2 number is simply
`ttfs_p95 − ttfs_p50`, native vs Python, on the same box.** That's a concrete, existing path to the user's second
goal — the plan should name it. The Step 2–5 risks are about FAITHFULNESS (the architecture flips from LB+K-procs
to one process+threads) and CONFOUNDS (the WS server's own tail; the one-utterance harness limitation), not
missing tooling.

## Step 2 — scheduler + admission

### S2-A (MAJOR) — "Faithful to the Python shed behavior" must be re-derived for one process, not copied
The Python deployment sheds via the **external LB**: when all K backends are at `maxconn`, the LB **closes** the
connection (`local_lb.py:45-47` "all backends at maxconn -> shed; cw.close()"). Aggregate admission = K × maxconn.
The native runtime is ONE process → there is no LB and no per-proc backlog; admission becomes a **single global
inflight/active-session cap**. So "faithful" means: (a) one global **backlog-COUNT** cap (the memory's proven
signal; **ready-age was proven NOT to work — don't re-litigate it**), set to the same aggregate as K×maxconn for
the comparison; (b) shed = **close** the connection (matching the LB), not silently queue. The plan says
"admission/backlog-cap shedding faithful to the Python stack's shed behavior" — make these three specifics
explicit, or the overload comparison in Step 4 is apples-to-oranges.

### S2-B (MAJOR) — The priority finalize lane must be defined against the Round-2 runner pool
Production ships a **priority finalize lane** (byte-exact, default-on). In the native runtime with one
`num_runners=N` pool (Round 2), steady and finalize `run()` calls contend for the SAME N runners + the container's
shared execution lock. A finalize burst can then **starve steady** (or steady starves finalize → final-TTFS tail
blows up = a direct G2 regression). The plan must specify how the priority lane maps onto the pool: a **dedicated
finalize runner / sub-pool**, or queue priority, or a separate finalize loader. This is decision-critical for G2:
the finalize lane is exactly the knob that protects `ttfs_p95`.

### S2-C (MAJOR) — Step 2 is blocked on Step 1's telemetry (carry Codex R1-M5 forward)
Step 2 says "From Step 1's knee" (`PHASE2-PLAN.md:33`) but a knee scalar can't size a scheduler. Step 2 needs
per-phase service time, runner-wait, `.item()`-wait, queue depth, finalize-wait — the telemetry schema Step 1
must emit. Make Step 2 explicitly blocked on that schema.

## Step 3 — multi-session runtime + real WS server

### S3-A (BLOCKER for G2) — The WS server's own tail can confound the very number Phase 2 exists to improve
Step 3 builds "a real WS server." A naive native WS server (single accept loop, blocking writes, no write
coalescing, head-of-line blocking on one event loop) has its OWN `ttfs` contribution — and if it adds tail, it
**confounds Step 4's G1/G2 numbers** and could make the native runtime look worse than Python for reasons that
have nothing to do with the inference scheduler. → Step 3 must (a) characterize the WS server's standalone
overhead (loopback echo `ttfs` with no model), so Step 4 can subtract WS-tail from runtime-tail; (b) state the
concurrency model of the WS layer itself (is the accept/IO loop separate from the inference worker threads?). This
is the Step-3→Step-4 version of the Round-2 "tail is dispatch-dependent" point.

### S3-B (MAJOR) — Stale-generation suppression must land in Step 3, before Step 4 benchmarking
Codex R1-M6: stale-generation suppression is the one Phase-1-deferred correctness item, and it's a
**scheduler/overload** behavior (`session_main.cpp` prints `stale_generation=DEFERRED_PHASE2_SERVER_ORACLE`). Under
the overload that Step 4 induces, wrong stale-final handling can make `ttfs_p95` look artificially BETTER (by
dropping/misordering finals) — a false G2 win. The plan currently has Step 3 = "correct per-stream events/finals"
but doesn't name stale-generation. Add it explicitly to Step 3, gated before Step 4.

## Step 4 — apples-to-apples density (the like-for-like number)

### S4-A (MAJOR) — Pin every knob identical; the harness already supports G2
For a fair comparison, native and Python runs must hold IDENTICAL: the dual SLO (keep-up `lag_p95 < 500ms`,
`ec2_loadgen.py:26,139` AND the `ttfs` distribution), `START_JITTER_MS`/`STREAM_JITTER_MS` (`:24-25`, the WAN-mimic
+ phase spread), the semantic-WER tool/version (`semantic_wer.py`), hardware, and a **freshly re-measured Python
baseline** (memory: baseline NOT frozen — don't compare to the stale 16–20). **G2 is then `ttfs_p95 − ttfs_p50`,
native vs Python.** Name this in the plan.

### S4-B (MAJOR) — Step 1b density is a CEILING; Step 4 is the realized number — label the gap
Step 1b measures density with a PLACEHOLDER dispatch (no real scheduler, no WS). Step 4 adds the real scheduler +
WS overhead + stale-gen + admission. So **Step 4 density ≤ Step 1b density**, possibly materially. If Step 1b
clears ≥1.5× but Step 4 (realized) drops below, that's a late, expensive surprise (post-Step-3 build). → Label
Step 1b a best-case CEILING and Step 4 the realized number; set the expectation (and ideally a Step-1b margin
above 1.5× to absorb the Step-4 haircut) so the GO at Step 1b isn't falsified at Step 4.

### S4-C (MAJOR) — Co-located harness measures the SERVER-SIDE tail (good for G2 scope) — state it
`ec2_loadgen.py` runs ON the box (`:6`), so `ttfs` EXCLUDES real WAN; `STREAM_JITTER_MS` is a synthetic WAN-MIMIC.
This is actually correct for Phase 2's scope: it isolates the **server-side tail component** — exactly the movable
part (Round 2's TTFT = VAD + WAN + server-side decomposition). State that Step 4's `ttfs` spread is the
server-side G2 component; real end-to-end TTFT adds VAD+WAN on top (out of scope). This keeps the G2 claim honest.

### S4-D (MAJOR) — Multi-turn under load is UNMEASURED (harness limitation)
`ec2_loadgen.py` is **one-utterance-per-connection** ("Multi-utterance-per-connection was rejected: re-arming
vad_start times out", `:47-49`). So the multi-turn continuous-context path (Phase-1 1.4b Step 2: speculative
finalize + retained context across turns) is **not load-tested** by Step 4. The density/tail numbers are for
single-utterance streams. → Flag this as a coverage residual (multi-turn density/tail unmeasured), and decide
whether it matters (multi-turn retains caches across finalizes → different memory/compute profile under load).

## Step 5 — per-target sweep

### S5-A (MAJOR) — Step 5 is a CONFIRMATION of Step 1's resource attribution, not independent
Round 2 made "BW-bound" a hypothesis to be MEASURED with counters at Step 1. That attribution **pre-determines
Step 5's outcome**: if the L40S knee is BW-bound, L4 (less BW) is more so → "no lift" is pre-confirmed; if it's
launch/execution-lock-bound, L4 might also lift. → Frame Step 5 as confirming the Step-1-predicted per-target
behavior + producing the L40S density number, NOT a fresh independent hypothesis. State that L4 is OUT as a fleet
target (memory) so Step 5's L4 run is confirmation, not a re-decision (unless L40S surprises).

### S5-B (MAJOR) — Spark aarch64: the runner-pool/AOTI path itself may not hold
The Round-2 mechanism (one loader, `num_runners=N`, explicit stream, shared constants) rests on libtorch 2.8 AOTI
behavior verified on x86. Spark is **aarch64** with "libtorch maturity risk" (plan's own note). The container's
runner pool + stream handling + CUDA-graph capturability may differ or be flaky on aarch64. → Step 5 must first
confirm the *mechanism* works on Spark before measuring density; budget for a build-from-source / fallback.

## Cross-cutting — does the corrected plan MEASURE G1 and G2?
- **G1 (density):** YES — Step 1b (L40S numeric gate, ceiling) + Step 4 (realized, apples-to-apples). Controls
  needed: re-measured baseline, identical SLO/jitter, ceiling-vs-realized labeling (S4-B).
- **G2 (server-side tail):** YES — `ttfs_p95 − ttfs_p50` from `ec2_loadgen.py`, native vs Python (S4-A). Controls
  needed: WS-server-tail subtraction (S3-A), priority-lane definition (S2-B), stale-gen correctness so a tail
  "win" isn't a dropped-final artifact (S3-B), and the honest scope (server-side only, S4-C). With those, G2 is
  genuinely measured — but ONLY if the plan makes `ttfs` spread a named gate quantity, which today it does not.

## Net for Round 3
The downstream steps are buildable and the tooling already measures both goals — the work is (1) re-deriving the
shed/priority-lane semantics for one-process-multi-thread (S2-A/B), (2) controlling the WS-server tail confound
(S3-A) and stale-gen correctness (S3-B), (3) pinning the apples-to-apples knobs + labeling Step1b-ceiling vs
Step4-realized (S4), (4) reframing Step 5 as confirmation of Step 1 + the Spark mechanism risk. With these, the
corrected plan measures G1 and G2 honestly; without S2-B/S3-A/S3-B, the G2 number is confoundable.
