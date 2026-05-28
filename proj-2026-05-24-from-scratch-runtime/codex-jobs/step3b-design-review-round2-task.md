<task>
**Round 2 adversarial review of `reviews/Step3b-WS-architecture.md` v2.** Round 1 folded; Round 2
attacks v2 to find what still needs sharpening or is wrong. Stop condition (5-round directive): if
both Round 2 reviews come back with only minor improvements that can fold in during implementation,
the rounds converge.

Write your fold-ready review to `reviews/codex-Step3b-design-round2.md`. Adversarial, specific, terse.
**Build on Round 1**: don't re-flag the items Round 1 already folded (read
`reviews/codex-Step3b-design-round1.md` for context).
</task>

<context>
**The design under review**: `proj-2026-05-24-from-scratch-runtime/reviews/Step3b-WS-architecture.md`
(v2, committed in the latest commit). v2 supersedes v1 on the items Round 1 flagged. Read v2 in full;
read Round 1's `reviews/codex-Step3b-design-round1.md` for context on what's already folded.

**ASK / structure your Round 2 attack around**:

1. **What did v2 fold INCORRECTLY?** Did v2 misinterpret any Round 1 finding or fold it in a way
   that creates a new problem? Specifically:
   - Library boundary §II: is the public/private split correct + complete? Anything missing or
     wrongly classified?
   - StatsCollector contract §III: does the Python compatibility hold? Any subtle behavior we
     missed?
   - WS protocol §IV: Python extractions accurate? Anything else the production server does that's
     not captured?
   - Close-code table §V: any gap or misuse?

2. **What new issues did v2 INTRODUCE?** The bigger doc has more surface area; review for new
   internal inconsistencies, mis-cited Python behavior, etc.

3. **The deferred-to-Part-B test oracle** (§XIII.2): a script that runs the same audio through
   Python + C++ servers and diffs the wire JSON. Is this concrete enough to actually build?
   Specifically what audio? what assertion?

4. **What remains TBD or under-specified?** Look for "to be confirmed via audit", "default if
   present", "implementation-bounded", "deployment-owned" phrases — are any of those actually
   blocking Part A code?

5. **Build risk re-check**: is the OpenSSL Option A fix actually safe? Will adding `libssl-dev` to
   the Dockerfile invalidate any in-flight CI/build state? Are there other dependency gaps not yet
   surfaced?

6. **The Part A in-flight risk** §XIII.1: v2 says "audit the diff against v2; redo if substantive."
   Is that practical, or should Part A be hard-aborted now and re-launched on v2? Take a position.

7. **Threading model §VI** is more specific now. Any race or starvation case not addressed?
   Specifically the "admin handler pool" — what's its size? Can /stats requests queue?

8. **Graceful shutdown §VII** sequence — anything missing? What about in-flight finalize that
   exceeds the 30s drain? What about the dispatcher thread (does it cleanly close or also force-
   abort)?

9. **§XII Part A revised scope** — is it tight? Anything missing? Anything over-scoped?

10. **The ONE-listener routing §IV**: the design relies on reading the HTTP request line + headers
    before deciding WS vs HTTP. What if a malformed client sends bytes that look neither HTTP nor
    WS? Define the timeout / parse-fail behavior.

11. **What's still missing entirely?** Codex Round 1's "Secondary Missing Items" listed: auth, TLS/
    proxy/header, correlation IDs, audio codec exact spec, language validation, readiness vs
    liveness, fragmentation/ping/pong, admin endpoint exposure, test oracle. v2 partially addresses
    some (test oracle is mentioned in §XIII.2) but others remain (auth, TLS, language validation,
    readiness vs liveness distinction, fragmentation). Which are blocking-now vs defer-to-Part-B?

12. **Net verdict**: GO-as-is-with-minor-fold-during-impl / GO-with-1-2-must-folds-to-v3 /
    GO-with-substantive-revisions-to-v3 / HOLD-redesign-needed.

If GO with ≤2 minor must-folds and no architectural changes → the round converges. The user's stop
condition.
</context>

<verification_loop>
Doc/design review only — NO BUILD, NO RUN. Read v2 + Round 1 + Python server.py if helpful for
verifying compatibility claims. Bounded — design isn't huge (v2 is ~400 lines).
</verification_loop>

<action_safety>
Write only the review doc. Do not modify the design or any code.
</action_safety>

<compact_output_contract>
Report path of the review doc + one-paragraph verdict + the top 1-3 must-folds (or "minor only —
converged").
</compact_output_contract>
