# Phase-2 plan — Round 3 FOLDED (Codex + Opus, downstream Steps 2–5)

Inputs: `codex-phase2-round3.md`, `opus-phase2-round3.md`. Strong convergence again. Codex read the actual
`server.py` + deploy docs and surfaced the **primary-source production scheduler mechanisms** the native runtime
must replace; Opus grounded G2 in the existing loadgen metrics + found the multi-turn coverage gap. Goals:
**G1 density**, **G2 P50↔P95 tail**.

## The architecture flip both reviewers center on
Python = **K processes behind an external LB** (close-on-`maxconn`, `local_lb.py:44-48`). Native = **ONE process,
threads**. So three production controls must be re-derived for one process, not copied. Codex found production has
**TWO admission gates**, not one:
1. **External LB maxconn** — leastconn + per-backend maxconn at the SLO-robust point (`local_lb.py:19-27`,
   `deploy/DEPLOYMENT.md:39-42`).
2. **Server-local backlog admission** — `backlog_count > NEMOTRON_ADMISSION_MAX_BACKLOG` → **close WS with 1013**
   (`server.py:4974-5063`); backlog-count **8–12** is the proven signal, **ready-age did NOT track overload**
   (`DEPLOYMENT.md:43-48`).
The native runtime has neither unless Step 2 recreates both as **box-global** caps.

## Step 2 — scheduler + admission (BLOCKERS, both)
- **Two box-global caps (Codex B1 + Opus S2-A):** Step 2 must specify a box-global **active/admitted-session cap**
  (replacing LB maxconn) AND a box-global **backlog-COUNT cap** (replacing per-proc `NEMOTRON_ADMISSION_MAX_BACKLOG`;
  ready-age is proven dead — don't re-litigate). Shed = **close** (match the LB/1013). Report offered / admitted /
  rejected / close-codes / queue depth; **don't count rejected connections as capacity.**
- **Priority finalize lane is a real policy, not magic (Codex B3 + Opus S2-B):** production stages finalize before
  ready work, **excludes lanes with pending finalizes from steady selection** (`server.py:5340-5354,5945-5979`),
  runs finalize on the session's **pinned lane** (`:7788-7840`), and for non-parallel keys **reserves one lane +
  marks the model path exclusive** (`:4185-4213`); only steady same-drop continuation batches parallelize
  (`tests/test_scheduler_model_lanes.py:52-84`). → The native plan must define how this maps onto the Round-2
  **`num_runners=N` pool**: a partition (`N_steady + N_finalize_reserved`) or weighted priority with a hard bound
  on BOTH finalize-wait and steady-starvation. If finalize can take every runner → steady starves; if steady can →
  final `ttfs_p95` blows up. Either turns a valid Step-1b ceiling into a bad Step-4 realized number. **This is the
  knob that protects G2.**
- **Step 2 blocked on Step-1 telemetry (Codex B2 + Opus S2-C):** a scalar knee can't size a scheduler; reject any
  Step-2 design that cites only a knee. Required schema: active/offered/admitted/rejected, ready count, event-queue
  depth, runner-wait by phase, steady/finalize service time, `.item()`-wait, CPU-core util, WS recv/send overhead,
  and server-side TTFT/TTFS p50/p95/p99/**P95−P50**.

## Step 3 — multi-session + WS (BLOCKERS, both)
- **WS-tail confound (Codex B4 + Opus S3-A):** WS/event-loop work is not a rounding error — production needed an
  explicit **cooperative yield to avoid starving socket I/O under WAN timing** (`server.py:5537-5547`, the known
  event-loop-livelock fix). A naive native WS server can ADD the very tail Step 4 attributes to the runtime. →
  Step 3 must include a **WS-tail microbench** (accept→ready, send→recv, recv→queue, queue→scheduler, final
  serialize/send, client recv, event-loop lag under N idle + N streaming sockets) so Step 4 separates WS-tail from
  runtime-tail.
- **Stale-generation suppression is a Step-3 gate (Codex B5 + Opus S3-B):** production has concrete generation
  checks (`server.py:6297-6323,6440-6445,8797-8959`); native prints it DEFERRED. Under overload, wrong stale-final
  handling can make `ttfs_p95` look artificially BETTER (dropped/misordered finals) = a **false G2 win**, or skew
  WER. → Step-3 gate: per-session generation tokens; stale interim/final suppression; tests for close-while-inflight,
  reset-while-queued, reset-while-finalizer-owns-a-runner, final-after-shed; token/event equality vs serial oracle
  under concurrent load — BEFORE Step 4.

## Step 4 — apples-to-apples (BLOCKERS/MAJORS, both)
- **Good news — the harness already measures G2 (Opus S4-A):** `ec2_loadgen.py` computes `ttfs_p50`/`ttfs_p95` of
  vad_stop→final (`:77,136`). **G2 = `ttfs_p95 − ttfs_p50`, native vs Python.** It's co-located (no real WAN, `:6`),
  so it measures exactly the **server-side tail component** — the movable part (TTFT = VAD+WAN+server-side). State
  that scope; don't imply Phase 2 moves VAD/WAN.
- **BUT the loadgen must be EXTENDED (Codex M2):** it reports p50/p95 only — **no p99, no spread field**
  (`:128-139`). The corrected G2 gate needs p99 + P95−P50 + P99−P50 → extend the result schema.
- **Shed vs the "zero-errors knee" = false-STOP risk (Codex M3):** the loadgen knee = "max N with proc-lag p95 <
  500ms AND **zero errors**" (`:168-170`), but production **intentionally sheds** (WS 1013). A correctly-shedding
  native server would be marked FAILED by the zero-errors rule. → report **two curves**: no-shed SLO-robust knee +
  admitted-through-shed capacity; GO/STOP on **admitted successful streams/box** at SLO with a bounded reject rate,
  not offered-N-zero-errors.
- **Apples-to-apples MANIFEST (Codex B6 + Opus S4-A):** pin commit/artifact SHAs, audio corpus, loadgen
  args/env (`LOADGEN_JITTER_MS`, `LOADGEN_STREAM_JITTER_MS`, rounds, sweep), hardware/driver, server flags
  (incl. `NEMOTRON_SYNC_COMPRESS` + `NEMOTRON_FINALIZE_PRIORITY` default-on, `launch_multiproc.sh:45-55`),
  admission caps, semantic-WER model/version/prompt/retry. **Re-measure the Python baseline back-to-back** on the
  same box/date — NOT the stale 16–20/6. Pin a **semantic-WER pass/fail bound** at the knee (Codex Q4).
- **Step1b = CEILING, Step4 = REALIZED (both, M1/S4-B):** label them so; require a Step-1b margin above 1.5× to
  absorb the Step-4 scheduler+WS haircut, so the GO isn't falsified at Step 4.
- **DISTINCTIVE Opus (S4-D) — multi-turn under load is UNMEASURED:** `ec2_loadgen.py` is one-utterance-per-connection
  ("multi-utterance rejected: re-arming vad_start times out", `:47-49`). The Phase-1 multi-turn speculative-finalize/
  retained-context path is **not load-tested**. Flag as a coverage residual (multi-turn retains caches across
  finalizes → different memory/compute under load) and decide if it matters.

## Step 5 — per-target (MAJORS, both)
- **Confirmation, not rediscovery (Codex M4 + Opus S5-A):** Step-1 counter-based resource attribution
  PRE-DETERMINES Step 5 (BW-bound L40S → L4 more so → "no lift" pre-confirmed; lock/launch-bound → L4 might lift).
  Step-5 table: predicted binding resource, measured counters, predicted vs realized density, miss explanation.
  Purpose = confirm L40S number + confirm/falsify L4 negative + explore Spark; don't reopen the fleet decision
  without a counter mismatch.
- **Spark aarch64 mechanism risk is concrete (Codex M5 + Opus S5-B):** the AOTI `model_container.h` has an
  **aarch64-specific runner-reclamation branch** (`:718-731`) → the x86 `num_runners` conclusion may not carry.
  Step 5 needs a Spark **preflight** (build/load + overlap/concurrent==serial/memory-flat microgates + stream
  behavior) before the density sweep.

## DECISION STRUCTURE (Codex pre-empted Round 4's core — carry forward)
"Does the corrected plan measure G1/G2 end-to-end? **Not yet.**" Failure modes to design against:
- **G1 false GO:** Step-1b shows overlap, but Step-2 admission/priority or Step-3 WS loses enough that Step-4 can't
  realize it.
- **G1 false STOP:** Step-4 counts intentional shed-rejects as errors → undercounts admitted sustainable streams.
- **G2 false GO:** stale finals / shed / WS-tail suppress or shift slow finals → hidden P95/P99 spread.
- **G2 false STOP:** WS implementation overhead dominates but is charged to the scheduler/model runtime.

## OPEN FOR ROUND 4
Consolidate the decision structure into a pre-registered GO/STOP tree with NUMERIC thresholds (the 5090 Step-1
multiplier bar; the L40S ≥1.5× with a ceiling-vs-realized margin; the G2 P95−P50 target + non-regression bound;
the semantic-WER bound; the bounded reject rate). Trace each goal end-to-end through the corrected steps and
confirm no orphaned gap. Stress the thresholds themselves (are they the right bars?).
