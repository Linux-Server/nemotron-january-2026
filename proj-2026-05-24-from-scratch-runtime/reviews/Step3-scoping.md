# Step 3 — scoping for "bounded local smoke tests"

PHASE2-PLAN.md Step 3 calls for: "multi-session runtime + real WS server. Wrap the session core in the
scheduler + a real WS server (also closes the 1.4b interim-cadence residual). **Required before Step 4:** a
**WS-tail microbench** ... **Stale-generation suppression is a Step-3 gate** ... so a Step-4 tail 'win' can't
be a dropped-final artifact."

That's substantial work — likely 2-3 days of focused implementation if done in full (the WS server,
recv/send framing protocol, integration with the session core + admission + scheduler + stale-gen, the
WS-tail microbench, the production-shape concurrency, the gold-events compatibility, the multi-turn loop).

For the user's "bounded local smoke" constraint, **scope Step 3 into 3a + 3b + 3c**:

## Step 3a — WS-tail microbench standalone (bounded, ~hours)

**Goal**: characterize per-stage WS latency under N idle + M streaming sockets, to feed the `ws_tail`
telemetry block (defined in Step 2a-invariant-design.md §III).

**Scope**:
- A new standalone binary `runtime/cpp/ws_tail_microbench.cpp` using `boost::beast` (or `websocketpp` —
  Codex picks; boost::beast is the canonical pick, included in most torch boxes).
- Echo-only WS server: receives bytes, sends them back (no transcription, no session, no scheduler).
- Per-stage timestamping: accept, recv-frame, queue-to-handler, serialize-to-send, send-to-network.
- Loadgen client side: a small Python script (or boost::beast client) that opens N+M sockets and
  send-recv at a steady rate.
- Output: JSON sidecar with `ws_tail` block per Step 2a's schema.

**Validation**: the microbench itself produces p50/p95/p99 timings; the absolute numbers don't matter for
correctness (it's a characterization tool), but the schema must validate.

**Bounded**: yes. ~half day of focused work + smoke test.

## Step 3b — minimal WS server integration with session core (substantial)

**Goal**: a real WS server that accepts session connections, drives audio through the session core + scheduler,
sends transcription events back.

**Scope** (the FULL version):
- WS protocol: client sends 16-bit PCM mel-frames or raw PCM (probably PCM, matching the existing Python
  server's API); server sends interim + final transcription events as JSON.
- Per-connection: an admission check (Step 2a), session creation, scheduler routing for steady chunks,
  per-stream decode + finalize, event emission with stale-gen check.
- Concurrency: each connection is a worker thread (matches the existing density runtime's per-worker model).
- Lifecycle: connection open → admission → session start → streaming loop → close (triggers final + stale-
  gen bump).
- Error handling: WS-1013 shed-close, graceful client-disconnect, server-side fault.

**This is NOT bounded** — substantial implementation work + careful integration with all the existing
primitives (admission, stale-gen, scheduler, session core, finalize, telemetry).

**Recommendation**: do NOT scope this for "bounded local smoke." It's a 2-3 day focused implementation
that warrants its own paired review + significant smoke testing. **Defer to a dedicated Step 3b sprint**
(after Tier 3 ships + after L40S verdict).

## Step 3c — stale-generation gate validation (bounded, ~hours)

**Goal**: the "0 stale/mismatch" gate from the plan, validated in a real (or simulated-WS) context.

**Scope**:
- If Step 3b is deferred, simulate the WS lifecycle events (close, reset, shed) in the existing density-sweep
  mode via injection points (the 4 test cases in Step 2a-invariant-design.md §II).
- Verify `stale_drops` telemetry counts the right number of dropped work items per scenario.
- This validates the Step 2a stale-gen primitive without needing a real WS server.

**Bounded**: yes. Smoke-testable. The "real WS" version of this validation is part of Step 3b.

## What fits the user's "Prioritize through end of Tier 3" goal

The user wants: through Tier 3, with bounded local smoke. Here's the realistic scoping:

| Item | Scope | Bounded? | Priority |
|---|---|---|---|
| Tier 3 memory shrink | full | Yes (~1-3 hr) | **HIGH — running** |
| Tier 2 cleanup (F4/F5/F6) | full | Yes (~1-2 hr) | Medium (after Tier 3) |
| Step 2a invariant work | full (admission + stale-gen + telemetry + WS-tail stub) | Yes (~2-3 hr) | Medium |
| Step 3a WS-tail microbench | standalone | Yes (~4 hr) | Lower (after Step 2a) |
| Step 3b real WS server | full | **No (~2-3 days)** | DEFER until after L40S verdict |
| Step 3c stale-gen validation | simulated via injection | Yes (~1 hr) | After Step 2a + Step 3b's deferral |

**My recommended execution order while Tier 3 runs**:
1. Tier 3 (running now) → review + commit.
2. Tier 2 cleanup (delegate to Codex after Tier 3 commits to avoid file conflicts) → review + commit.
3. Step 2a invariant (delegate, paired review, commit).
4. Step 3a WS-tail microbench standalone (delegate, paired review, commit).
5. Step 3c stale-gen validation via injection (delegate, paired review, commit).
6. **Stop before Step 3b** (the real WS server). Surface to the user as the next-bounded sprint.

After all of the above + L40S sweep verdict in hand: assess if Step 3b is the right next focus, or if the
realized density already supports advancing to Step 4 directly.

## What's NOT in the bounded scope (the honest list)

- Step 3b (real WS server) — too big.
- Step 4 (apples-to-apples Python re-measure) — needs the L40S verdict + Step 3b.
- Step 5 (per-target confirmation: L4, DGX Spark) — EC2 spend, dependency on Step 4.
- A new B>4 bucket exploration — not requested.
- Multi-dispatcher / per-stream-pool scheduler — design exploration, not bounded code.
- CUDA-graph-of-batched-steady — design exploration.

These are the items the recap's "Tier 4-6" pointed at; they're not bounded local smoke material.

## Net

Bounded local-smoke checklist through "end of Tier 3" (the user's target):
- [ ] Tier 3 memory shrink (running)
- [ ] Tier 2 cleanup F4/F5/F6
- [ ] Step 2a invariant (admission + stale-gen + telemetry + WS-tail skeleton)
- [ ] Step 3a WS-tail microbench standalone
- [ ] Step 3c stale-gen validation via injection

Total ~half-day of orchestration + Codex compute (with smoke tests). Each piece is bounded; the parallel-
friendliness depends on file conflicts (Tier 3 + Tier 2 touch the same files → serialize; Step 2a + Step 3a
are new files → parallel-friendly).
