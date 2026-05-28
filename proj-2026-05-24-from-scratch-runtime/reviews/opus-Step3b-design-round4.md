# Step 3b WS server design v4 — Opus Round 4 adversarial review (2026-05-28)

Reviewing v4 (committed `a7756d5`) from-scratch. Stop condition check: does v4 reduce to minor-only?

## Verdict

**MINOR_ONLY → CONVERGED.** v4 lands Codex Round 3's 2 must-folds correctly (WireEvent.finalize +
StatsCollector completion predicate). The 9 minor folds documented in the v4 banner are tracked but
not in the section bodies — that's the one delta from a "perfect" v4. **Recommended next**: relaunch
Part A on v4 with the banner items pinned in Part A's task spec; declare design convergence.

## What v4 did right

1. **WireEvent.finalize** added as `std::optional<bool>` — matches Python's `{"finalize":...}` flag
   pattern.
2. **finalize_timing as `std::optional<nlohmann::json>`** — flexible until Part A's audit pins
   exact keys. Better than premature schema-freezing.
3. **StatsCollector completion predicate**: `!was_suppressed && vad_stop_to_sent_ms.has_value()` —
   correct. fork_flush_wall_ms is optional + per-metric, matching Python.

## Minor items (acceptable to defer to Part A task spec or fold during impl)

### 1. WireEvent.finalize population — when?
v4 says "matches Python's `{"finalize":...}` flag" — but on which events? Likely populated only on
responses to control-message reset/end (echoing the finalize state). Not on regular transcripts. Pin
in Part A's task spec from the audit.

### 2. nlohmann::json dep
v4 added `nlohmann::json` to WireEvent. Repo doesn't have it yet. Two options:
- Vendor nlohmann::json single-header (MIT) at `lib/runtime_io/json.hpp`. Cleanest.
- Extend the existing repo-local JSON parser (in `steady_batch_primitive.h`) to emit. More work.

**Recommend Option A** (vendor single-header) — consistent with picohttpparser vendoring. Pin in Part
A's task spec.

### 3. WireEvent serialization byte-order
Python's `json.dumps` produces insertion-order field output. C++ side needs to match for on-wire
compatibility (the test oracle canonicalizes for comparison, but production responses go to real
clients that may or may not care about field order). Recommend: WireEvent → JSON serializer uses
insertion-order matching Python's typical pattern (`type` first, then `text`, then `is_final`, then
`finalize` if present, then `finalize_timing`/`message`). Pin in Part A's audit.

### 4. emitted flag semantics
For COMPLETE samples (passed the predicate), `emitted=false` means "send returned failure" (TCP
closed, write timeout, etc.) — NOT stale-gen suppression (which makes was_suppressed=true and fails
the COMPLETE predicate first). Document explicitly in StatsCollector::record() docstring.

### 5. The 9 banner items are in the banner, not section bodies
v4's banner lists 9 minor folds (Silero budget, shutdown default, /health enum, etc.) but the
section bodies haven't been edited to reflect them. Part A's task spec must explicitly say "read
v4's banner items as part of the spec" OR these should be inlined.

**My recommendation**: don't bother inlining in v5 — Part A's task spec can call them out (saves a
round). The banner is durable + documented.

### 6. Banner items that actually shape Part A's implementation (re-emphasize for the task spec)
- **Silero CPU inference + thread cap** (§VI fold): Part A's SharedRuntime needs to wire Silero with
  `torch::set_num_threads(<small>)` to prevent oversubscription. Don't punt to Part B.
- **PCM endianness `static_assert`** (§II fold): trivial; Part A includes it in `lib/session/session.h`.
- **/health enum** (`loading|healthy|draining|degraded`): Part A's stub returns one of these.
- **picohttpparser edge-case ownership**: Part A's `lib/ws/handshake.cpp` handles case-insensitive
  headers + comma-token Connection + duplicate headers (the parser doesn't).
- **Test oracle port allocation**: Part B's run_compat.py; Part A can defer.

## What HOLDS in v4

- All Round 1-2-3 substantive folds (library boundary, Python-exact /stats, server-side Silero,
  Part A v1-superseded, graceful shutdown ordering, picohttpparser, malformed handling, smoke
  matrix, MPS-readiness).
- The v4 must-folds (WireEvent.finalize, StatsCollector completion predicate).
- No new architectural ambiguity introduced by v4's edits.

## Cross-check: did v4 reopen any Round 1-3 folds?
- Library boundary (Round 2): WireEvent change is a public-header field add, but didn't break the
  boundary discipline. ✓
- StatsCollector contract (Round 2): predicate clarification is consistent with the Python-exact
  framing. ✓
- Server-side Silero (Round 2): unchanged. ✓
- Graceful shutdown (Round 2): unchanged. ✓
- Malformed handling (Round 2): unchanged. ✓
- Test oracle (Round 2): unchanged. ✓
- All other folds: unchanged. ✓

No reopened folds.

## Net

**v4 = CONVERGED for design purposes.** Move forward:
1. Relaunch Part A on v4 with a task spec that:
   - References v4 sections AND the v4 banner items.
   - Pins the 9 banner items concretely (esp. Silero CPU+thread-cap, nlohmann::json vendoring,
     /health enum, endianness assert, picohttpparser edge ownership).
2. Skip Round 5 unless an unexpected ambiguity surfaces during Part A's execution.

The user's 5-round budget had slack; we converge at Round 4 (well within budget). Rounds 1-2-3 each
landed substantive folds; Round 4 was a tight clarification round; further rounds would yield
diminishing returns vs implementation.

**Recommendation**: declare CONVERGED at v4; queue Part A v3+v4-aware task spec; surface to user
for confirmation.
