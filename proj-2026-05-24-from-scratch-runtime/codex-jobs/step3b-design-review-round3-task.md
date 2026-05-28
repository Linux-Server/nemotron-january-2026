<task>
**Round 3 adversarial review of `reviews/Step3b-WS-architecture.md` v3.** Rounds 1 + 2 folded; Round
3 attacks v3. Stop condition: if both Round 3 reviews come back with only minor improvements that can
fold during implementation, the design CONVERGES → next is Part A on v3 (or v4 minor-fold).

Write your fold-ready review to `reviews/codex-Step3b-design-round3.md`. Adversarial, specific, terse.
**Build on Rounds 1+2**: don't re-flag what's already folded (read
`reviews/codex-Step3b-design-round1.md` + `round2.md` for context).
</task>

<context>
**The design under review**: `proj-2026-05-24-from-scratch-runtime/reviews/Step3b-WS-architecture.md`
v3 (committed in the latest commit). v3's substantive decisions:
- Server-side Silero VAD authoritative (matches Python).
- StatsCollector Python-exact /stats; native scheduler_telemetry separate endpoint.
- Part A v1 work SUPERSEDED; relaunch on v3.
- Concrete public API: PCMFrame, WireEvent, SessionRuntime, SharedRuntime.
- Python protocol values folded from Codex Round 2's audit.
- HTTP parser: picohttpparser.
- Graceful shutdown ordering FIXED.
- Malformed/slowloris handling specified.

**ASK / structure your Round 3 attack around**:

1. **Did v3's substantive decisions land correctly?**
   - Server-side Silero VAD: any operational concern (model load cost, per-frame eval cost at
     N=64, multi-process MPS implications)?
   - StatsCollector Python-exact: are the Snapshot fields + the per-metric `count` semantics
     consistent? Any Python behavior we miss?
   - Part A v1-superseded: is the keep/discard split in §XII clear enough that the auditor can
     execute mechanically?

2. **Are the concrete signatures in §II actually buildable?**
   - SessionRuntime + SharedRuntime ownership: clean? Threading-safe? Does SessionRuntime hold a
     reference to SharedRuntime, or get the bits it needs passed in per call?
   - PCMFrame: int16 samples + count. Endianness assumption (LE) — is that always-true for x86?
     For aarch64 Spark?
   - WireEvent finalize_timing keys — v3 says "pinned per Part A's audit" but Part A is yet to
     run. Acceptable to defer this concrete spec to Part A, or force it now?

3. **VAD integration depth**: Part B includes Silero. Does the SessionRuntime API surface need to
   change to accommodate it? Where does the Silero model live (SharedRuntime owns it; per-session
   state in SessionRuntime)? Per-frame eval cost at 16kHz mono — is it negligible vs the ASR
   path?

4. **Graceful shutdown sequence §IX**: any remaining issue? Specifically:
   - Step 5 "enqueue a server-side finalize_now() on each existing session to accelerate drain"
     vs "wait for natural VAD-stop" — which is the default? Specify.
   - What if /stats is being polled during drain — does the admin handler pool keep serving?
   - Dispatcher close ordering: §IX step 8 says scheduler.close() AFTER workers. Is that
     enforceable (workers might be still trying to enqueue when close is initiated)?

5. **picohttpparser**: vendored single-header; permissive license. Confirm: is it
   maintained / battle-tested enough for production? Any known WS upgrade quirks?

6. **Test oracle §XIV**: now concrete (utt0..utt7, canonicalized diff). Practical concerns:
   - Running Python + C++ servers concurrently requires 2 ports. Is that supported by
     run_compat.py / step6_server_oracle.py?
   - Volatile-field-stripping logic — define which fields are volatile (timestamps, sequence
     numbers, finalize_timing values). Anything else?
   - When does this run? CI? Manual smoke? Block Part B's commit?

7. **The new endpoints**: /scheduler_telemetry and (optional) /admission — do they need
   admission/auth? Currently undefended. For multi-tenant deployments, exposing native scheduler
   internals to anyone is a leak.

8. **Configuration sprawl**: §XVI lists 16 env vars + 6 CLI flags. Operator memorability matters.
   Could any consolidate? Worth a sanity check.

9. **What's still missing entirely?** Codex Round 1's "Secondary Missing Items" — auth/authz, TLS/
   proxy, correlation IDs, language validation, readiness vs liveness — some are addressed (per-
   process /health distinguishes loading/draining), others still deferred. Re-survey: any of these
   now BLOCKING in light of v3's other decisions?

10. **Net verdict**:
    - **MINOR_ONLY = CONVERGED**: only minor improvements that fold during implementation; v3 is
      build-ready; proceed to Part A on v3.
    - **GO-with-1-2-must-folds-to-v4**: small specific changes; one more round.
    - **GO-with-substantive-revisions**: another substantive round needed.
    - **HOLD**: redesign needed.
</context>

<verification_loop>
Doc/design review only — NO BUILD, NO RUN. Read v3 + Rounds 1+2 reviews. Bounded — v3 is ~520 lines.
</verification_loop>

<action_safety>
Write only the review doc. Do not modify the design or any code.
</action_safety>

<compact_output_contract>
Report path of the review doc + one-paragraph verdict (MINOR_ONLY / 1-2-folds / substantive / HOLD) +
the top 1-3 must-folds (or "minor only — converged").
</compact_output_contract>
