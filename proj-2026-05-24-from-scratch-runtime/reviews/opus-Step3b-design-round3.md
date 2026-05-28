# Step 3b WS server design v3 — Opus Round 3 adversarial review (2026-05-28)

Reviewing v3 (committed `5e83a44`) from-scratch. Builds on Rounds 1+2; doesn't re-flag what's
folded. Looking for the convergence signal: do the remaining items reduce to "fold during
implementation"?

## Verdict (preview)

**MINOR_ONLY with 1 specific clarification → effectively CONVERGED.** v3 made the substantive
decisions; the items below are clarifications + impl details. **ONE genuine concern worth pinning
before Part B**: Silero CPU-vs-GPU eval cost at N=64.

## The one substantive concern

### 1. Silero VAD eval cost at N=64

v3 §VI pins server-side Silero as authoritative. Practical math:
- 16 kHz mono, 20 ms frames = 50 frames/sec/stream.
- N=64 streams = **3200 Silero evals/sec aggregate**.
- Silero per-eval cost: CPU ~1-2 ms (depending on hardware) OR GPU ~0.1 ms.
- CPU path: 3200 × 1.5 ms ≈ **4800 ms of CPU work per wall-second** → saturates ~5 cores
  continuously (out of 32 on g6e.8xlarge). Real but bounded.
- GPU path: 3200 × 0.1 ms ≈ 320 ms of GPU work per wall-second → ~32% GPU util just for VAD →
  competes with ASR encoder for SM time.

**Audit needed**: does the Phase-1 prod Python server run Silero on CPU or GPU? (Probably CPU per
the typical Silero deployment pattern; GPU was the encoder.) If CPU is OK in Python, it's OK in C++
(same hardware budget). If GPU, the C++ port needs to schedule Silero alongside encoder via shared
streams — non-trivial.

**Fold action for v4 (minor)**: pin "CPU Silero, matching Python's deployment; per-VAD cost ~5
cores at N=64; budget accordingly." If Part A's audit reveals GPU, revise. This isn't a v3
architectural redesign — it's a 2-line clarification.

## Other minor folds (recommend, not blocking)

### 2. StatsCollector.record() emitted flag — clarify who decides
v3 §IV: `void record(SessionTiming timing, bool emitted)`. The worker thread calls this. How does
the worker know `emitted`? After calling `ws.send(json)` — if send returns success, `emitted=true`;
if generation-stale before send (the §V check), `emitted=false`.

**Fold action**: v4 §V table — add a column "Calls record() with what" so the wire-up is explicit.
Minor.

### 3. Graceful shutdown §IX step 5 — pick a default
v3 step 5: "enqueue a server-side finalize_now() on each existing session to accelerate drain OR
wait for natural VAD-stop" — pick one.

**Fold action**: v4 default = enqueue finalize_now() (faster drain). `NEMOTRON_SHUTDOWN_FORCE_FINALIZE=0`
opts back to natural wait if a deployment prefers it.

### 4. scheduler.close() during worker drain — semantics
v3 §IX step 8 says scheduler.close() AFTER workers complete. Edge case: a worker is in
`scheduler.enqueue(...).get()` when the close starts. What does enqueue return? B2's
BatchedSteadyScheduler::enqueue throws if `closing_` is set. Worker propagates the throw → session
finalize fails.

**Fold action**: v4 specifies — during drain, workers SHOULD NOT enqueue new work; they should be
in finalize_now()'s direct B=1 path or already past the scheduler. The scheduler.close() at step 8
happens only after the worker drain (step 6) is done, so no in-flight enqueues exist. Add an
assertion / log if this invariant breaks.

### 5. PCMFrame endianness
v3 §II says "signed 16-bit LE". x86 + aarch64-LE: always LE. aarch64-BE: nonexistent in practice for
Spark. Add a one-line `static_assert` or runtime check at first frame.

### 6. WireEvent finalize_timing keys deferred to Part A audit
v3 §III says the audit pins the keys. Risk: if Python has unexpected keys (e.g., includes per-chunk
breakdowns), SessionTiming may not map directly. **Mitigation**: v4 lists LIKELY keys (subset of
the 5 SLO metrics + finalize_seq + active_sessions_at_emit + emit_unix_ts) explicitly + flags
"confirm via audit." Reduces audit risk.

### 7. Test oracle — needs port management
v3 §XIV: Python server on 8080, C++ on 8081. step6_server_oracle.py runs one server. The run_compat.py
wrapper spawns both with `--port` args + manages lifecycle. Tiny addition; spec it.

### 8. /scheduler_telemetry endpoint trust
Exposes native dispatcher CPU% + queue depth — performance characteristics. Acceptable in trusted-
network deploys; could leak in multi-tenant. v4 note: "/scheduler_telemetry follows same trust model
as /stats (trusted-network); production multi-tenant should restrict via LB headers or auth proxy."

## What HOLDS in v3

- **Server-side Silero VAD authoritative** = correct decision (matches Python; Phase-1 production
  pattern).
- **StatsCollector Python-exact /stats + separate /scheduler_telemetry** = right split.
- **Concrete SessionRuntime + SharedRuntime + PCMFrame + WireEvent** = build-ready signatures.
- **Part A v1-superseded with clear keep/discard split** = pragmatic.
- **Graceful shutdown ordering** = correct after the §IX fix.
- **Malformed-request handling** = concrete.
- **picohttpparser** = right choice; widely used.
- **Test oracle canonicalized** = right approach.
- **Smoke matrix scheduler-ON case** = catches integration startup failures.
- **/health Python shape + admission Python shape** = operator dashboards compatible.

## Configuration sprawl (acceptable)

v3 has 16 env vars + 6 CLI flags. Operators familiar with the Phase-1 server already know most of
these. No consolidation needed.

## Items NOT addressed (acceptable to defer)

- Auth/authz: still trusted-network behind LB. Acceptable for current SageMaker target.
- TLS: handled by LB. Acceptable.
- Correlation IDs / per-connection metadata: small addition to Part B (1 UUID per WS connection,
  logged + optional in JSON). Not blocking.
- /metrics Prometheus: StatsCollector exposes `snapshot_prometheus()`; binary wires it as a route
  in Part B if needed.

## Net

**MINOR_ONLY → effectively CONVERGED.** The 8 items above are clarifications/impl-detail; none are
architectural redesigns. If the user wants Round 4 for completeness, the v4 fold is small (1-page
amendment). If we want to be efficient, **declare convergence at v3 + fold these 8 items into a v4
amendment OR into Part A's implementation** (Part A's task spec can call out items 2, 3, 4, 5, 6,
7, 8 as in-scope clarifications).

**Recommendation**: ONE more round (Round 4) to fold the 8 minor items into v4, since the user
directive was "5 rounds or until paired returns minor only." Round 4's outcome should be
straightforward — both reviewers should see v4 as build-ready.

**OR**: skip Round 4 and declare convergence at v3 + fold the 8 minor items directly into Part A's
new task spec (they're all impl-detail). Saves one round of compute.

The user has the call. My lean: declare convergence at v3 + fold minor items into Part A's spec.
The 5-round budget had slack to allow this convergence at Round 3.
