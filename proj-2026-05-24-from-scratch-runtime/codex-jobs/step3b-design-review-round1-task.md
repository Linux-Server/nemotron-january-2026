<task>
**Round 1 adversarial review of `reviews/Step3b-WS-architecture.md` (the design doc, committed at
`8b7e783`).** This is the first of up to 5 paired Codex+Opus review rounds on the WS server
architecture. Each round attacks the latest version of the design, finds substantive issues, and the
rounds stop when reviews return only minor improvements that can fold in during implementation.

Write your fold-ready review to `reviews/codex-Step3b-design-round1.md`. Adversarial, specific, terse.
</task>

<context>
**The design under review**: `proj-2026-05-24-from-scratch-runtime/reviews/Step3b-WS-architecture.md`.
Read it in full first. The headline is: production WS+HTTP server, library boundary carved from
current `#include "session_main.cpp"` monolith, /stats first-class via `StatsCollector` +
`SessionTiming`, startup-smoke discipline from the 1257d47 bug.

**Related context to read selectively**:
- `reviews/Step3-scoping.md` — the earlier Step 3 scope-narrowing that classified 3b as the substantive
  WS server implementation.
- `reviews/Step2a-invariant-design.md` — §I admission, §II stale-gen, §III telemetry schema (the
  modules the WS server integrates).
- `reviews/Step2a-invariant-design.md` §III's `ws_tail` block schema.
- Phase-1 `server.py` (in the broader repo, src/nemotron_speech/server.py — outside the proj-2026 dir
  but referenced) is the protocol reference. **You don't have to fully audit it (that's part B's work),
  but flag if the design's WS protocol section is too thin given that audit is deferred.**
- Recent commits 279f033 + 1257d47 (in main repo) are the /stats endpoint + the bug-fix lesson the
  design references.

**ASK / structure your review around these angles**:

1. **Library boundary precision**: §II says "Public headers carved from session_main.cpp" but the
   public API surface isn't enumerated. session_main.cpp is 4668 lines. Without an explicit list of
   symbols that go to the library headers vs stay private, the refactor is undefined — Part A's
   delegation has to make ad-hoc decisions. Specifically: is the public surface JUST the SessionState
   struct + the entry-point functions, or does it include all of EmittedEvent / Tokenizer / event-
   emission helpers / state-machine internals? List concretely.

2. **WS+HTTP listener design**: §IV.2 considers but doesn't commit between WS+HTTP on one listener vs
   separate ports. The `ws_tail_microbench.cpp` plumbing only does WS handshake. Production server.py
   does both on one port. Force a decision; if "one listener", explain how HTTP routing distinguishes
   `GET /health` from a WS handshake (it's the `Upgrade: websocket` header).

3. **StatsCollector API contract — null handling**: §III's `SessionTiming` uses `std::optional<double>`
   for each metric. The nearest-rank quantile algorithm doesn't naturally handle missing values. What's
   the contract? Skip records with any null? Skip per-metric (a record counts in TTFS p95 only if
   `vad_stop_to_sent_ms` is populated)? Specify.

4. **active_sessions_at_emit semantics**: this is the count AT FINALIZE TIME, not current. Where does
   the snapshot live? Captured in `run_finalize_density` and passed in? Read from DensityAdmission
   atomically? Race window between capture and record.

5. **Threading model**: §IV.5 says "thread-per-connection". With N=64 worker threads + 1 dispatcher
   thread + HTTP handler thread(s), what's the threading model for HTTP handlers? Blocking? Async via
   epoll/poll? If a /stats poll is in flight while N=64 finalizes are recording, is the mutex
   contention noticeable? Lock-free StatsCollector?

6. **Reset semantics**: §II.4 says "client sends a control message (TBD — match Python or a documented
   JSON command)" — TBD is a real ambiguity. The Python `server.py` audit is deferred to Part B but
   Part A's WS skeleton is being implemented NOW. Is there a risk Part A's WS skeleton makes choices
   that conflict with the audit?

7. **WS close-code mapping**: §II.4 lists 1000/1011/1013 generically. Specify the EXACT mapping:
   which scenarios trigger 1013 vs 1011 vs 1000 vs (if applicable) 1009 (message-too-big), 1003
   (unsupported-data), etc. Operator-facing semantics matter.

8. **Dispatcher concurrency**: at the realized N=64+ knee with B_max=4, finalize emission rate is
   ~throughput_rt × N / N_per_batch ≈ 33-34 finalizes/sec. Each finalize records to StatsCollector
   via mutex. Quantify if mutex contention is plausibly significant at this rate (probably no — but
   confirm). Future N=128+ with multi-dispatcher might change this.

9. **server.py audit is the actual first task of Part B**: §III-C says so. But Part A is being
   implemented IN PARALLEL. Risk: the Part A WS skeleton makes API choices that need to be revised
   after audit. Either move audit forward (out of Part B, into Part A) or accept the rework risk.
   Take a position.

10. **Prometheus /metrics**: §II briefly mentions it as future. If real production-target is Prom
    monitoring, designing the StatsCollector to expose Prom-format too NOW (cheap) is much easier than
    bolting on later. Or genuinely defer — but say so explicitly.

11. **Build env / dependencies**: §II uses raw RFC6455 inherited from `ws_tail_microbench.cpp` which
    uses `openssl/sha.h`. Confirm the `nemotron-aoti:cu128` container has OpenSSL; if not, build
    breaks. Cite the container Dockerfile.

12. **Graceful shutdown**: SIGTERM behavior is unspecified. Drain in-flight sessions? Force-close?
    Send graceful close frame? Important for cluster rolling deploys.

13. **Backpressure under load**: high-rate /stats polling while N=64 finalizes are recording — could
    poller starve recorders or vice versa? Worth a design decision.

14. **MPS-awareness**: production deploy is multi-process MPS per the memory. The WS server's design
    should at minimum NOT preclude this. Implications: per-process port binding, per-process
    StatsCollector window, cross-process aggregation in the LB layer? Or punt to deploy config?

15. **Stale-gen integration enumeration**: §II.3 says "downstream emit checks generation". But there
    are MULTIPLE emit points (WS interim event, WS final event, StatsCollector record, /stats response).
    Enumerate which need generation check + the exact check pattern.

16. **Missing items**: what's missing from the design entirely? E.g., authentication/authz? Rate
    limiting beyond admission? Per-connection metadata (correlation IDs for distributed tracing)?
    Audio codec assumptions (PCM 16-bit @ 16kHz vs Opus vs ...)? Health-check interval contract for
    load balancers?

17. **Net verdict**: GO-with-changes-then-implement / GO-with-major-revisions / HOLD (pause Part A).
    If GO-with-changes, list the changes concisely so they can fold in to v2 of the design doc.
</context>

<verification_loop>
This is a doc/design review only — NO BUILD, NO RUN. Read the design doc + the cited related artifacts.
Bounded — design isn't huge. Don't dig into the actual code unless a specific design claim needs
verification against existing implementation.
</verification_loop>

<action_safety>
Write only the review doc. Do not modify the design or any code. Fold decisions go through me after.
</action_safety>

<compact_output_contract>
Report path of the review doc + one-paragraph verdict summary + the top 3 most-important must-folds.
</compact_output_contract>
