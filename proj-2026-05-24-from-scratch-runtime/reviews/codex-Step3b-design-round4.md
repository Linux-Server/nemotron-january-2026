# Codex Step 3b Design Review - Round 4

Verdict: **MINOR_ONLY = CONVERGED.** v4 lands the two Round 3 public-contract must-folds. I do not
see a v5-blocking issue. Proceed to Part A on v4, with the minor banner items and the clarifications
below copied into the Part A task spec.

## Must-Fold Check

1. **`WireEvent` now has the missing `finalize` field.**

   This unblocks the public DTO. The one remaining ambiguity is wording, not structure: §II says
   `finalize` is "on transcript reset/end responses", while Python also emits `finalize: true` on
   ordinary continuous final transcript events, and `finalize: false` on soft-reset transcript events.
   The Part A protocol audit should pin this as: emit the optional `finalize` field exactly when
   Python emits it; do not restrict it to reset/end paths. Interim transcripts should continue to omit
   it unless the audit proves otherwise.

   This does **not** require v5 because the DTO can already represent the field. It is a serializer
   rule / audit-table row.

2. **`finalize_timing` as flexible JSON is the right temporary shape.**

   This is safer than `map<string,double>` because Python's timing object currently carries raw
   timing keys and non-SLO metadata during continuous finalize. The risk is dependency/API sprawl:
   §II names `nlohmann::json`, while §X does not add a JSON dependency and also says "or repo-local
   equivalent." Part A must choose one concrete representation before freezing the public header.
   If `nlohmann::json` is used, vendor/package it explicitly; otherwise define a repo-local JSON object
   wrapper or serialized-object type. Also require object-only serialization for `finalize_timing`, not
   arbitrary scalar/array JSON.

   This is minor because Part A's first audit already runs before header carve and can make the
   concrete type choice.

3. **StatsCollector completion predicate is pinned correctly.**

   The v4 predicate `!was_suppressed && vad_stop_to_sent_ms.has_value()` fixes the Round 3 bug: missing
   `fork_flush_wall_ms` no longer ejects a sample from the deque. The send-failed edge is handled by
   the separate `emitted` flag as long as the emit path stamps `final_sent`/`vad_stop_to_sent_ms` before
   the send attempt, matching Python's current pattern. Stale-generation suppressions still set
   `was_suppressed=true` and remain lifetime-only. No reopened blocker here.

## Minor Folds

The 9 Round 3 minor folds are adequately captured in the v4 banner for a Part A task-spec fold. The
section bodies still contain several stale v3 phrases, so the Part A spec should explicitly state that
the v4 banner overrides the unchanged body text for these items:

- Silero runs on CPU matching Python, warmed at startup, with bounded thread usage and a VAD-load
  selftest.
- SIGTERM default is server-side `finalize_now(close_reason="shutdown")`; natural wait is not the
  default.
- Joined workers must be past scheduler enqueue points before `scheduler.close()`.
- `/health` status enum is `loading|healthy|draining|degraded`.
- Admin endpoints use the trusted-network single-listener assumption unless a deployment puts auth/LB
  policy in front.
- PCM binary input rejects odd byte lengths and decodes safely as int16 LE.
- picohttpparser does parsing only; the server owns case-insensitive headers, comma-token
  `Connection`, duplicate headers, partial reads, and body-on-GET rejection.
- The compatibility oracle allocates/accepts ports, strips only volatile values, and is a Part B
  pre-merge gate.
- Add the grouped operator config table in the Part A spec rather than another design round.

No one of these needs v5. The only important process guard is to not let stale body text win over the
banner when Part A is launched.

## Implementability As Part A

Build-ready with the existing sequencing:

- Part A starts with `reviews/server-py-protocol-audit.md`.
- The audit pins `finalize` emission cases, exact `finalize_timing` keys/value types, invalid
  model/language behavior, `/health` shape, and `/stats` details before the header and skeleton become
  durable.
- The library carve can then expose `PCMFrame`, `WireEvent`, `SessionTiming`, `SessionRuntime`, and
  `SharedRuntime` without leaking harness APIs.

The only decisions Part A still has to make and document are intentionally audit-dependent: concrete
JSON representation for `finalize_timing`, exact Python compatibility rows for query validation, and
the concrete HTTP parsing edge-case behavior tests. Those are implementation choices inside the Part A
scope, not design blockers.

## Round 1-3 Cross-Check

No Round 1-3 architectural folds are reopened by v4. The library boundary, Python-first protocol
audit, Python-exact `/stats`, server-side VAD decision, shutdown ordering, malformed HTTP behavior,
OpenSSL/container fix, one-listener routing, and test-oracle direction all still hold.

## Net

**MINOR_ONLY = CONVERGED.** Proceed to Part A on v4. Carry forward three minor task-spec bullets:
clarify `finalize` field emission cases, choose the concrete JSON representation/dependency for
`finalize_timing`, and treat the v4 banner's 9 minor folds as authoritative over stale v3 body text.
