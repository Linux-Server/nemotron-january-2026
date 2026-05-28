<task>
**Round 4 adversarial review of `reviews/Step3b-WS-architecture.md` v4.** Rounds 1-3 folded.

**Stop condition**: if both Round 4 reviews come back with MINOR_ONLY → CONVERGED → next is Part A
relaunched on v4 with the minor items in the task spec.

Write your fold-ready review to `reviews/codex-Step3b-design-round4.md`. Adversarial, specific.
Don't re-flag Rounds 1-3 folds.
</task>

<context>
**The design under review**: `reviews/Step3b-WS-architecture.md` v4 (committed in the latest
commit). v4 targeted-edit fold of Round 3:
- v4 §II: `WireEvent` adds `std::optional<bool> finalize`; `finalize_timing` is
  `std::optional<nlohmann::json>` (flexible until Part A audit pins keys).
- v4 §IV: `StatsCollector::record()` completion predicate = `!was_suppressed &&
  vad_stop_to_sent_ms.has_value()`; fork_flush_wall_ms is OPTIONAL + per-metric.
- v4 banner: 9 minor folds documented (Silero CPU eval cost, shutdown default, /health enum,
  endianness assert, etc.) — these are tracked but NOT yet edited into the section bodies; they'll
  be folded into Part A's task spec if Round 4 confirms convergence.

**ASK / structure your Round 4 attack**:

1. **Did v4's must-folds land correctly?**
   - WireEvent `finalize` field is now in the DTO. Any new ambiguity introduced (when is finalize
     populated — only on reset/end responses, or also on regular transcripts)?
   - finalize_timing as `nlohmann::json` is intentionally flexible — does that create more risk
     than the map<string,double> it replaced (e.g., loose typing → audit may surface keys that
     conflict with serialization)?
   - StatsCollector completion predicate pinned. Any edge case missed (e.g., timing where
     vad_stop_to_sent is present but emit failed; the `emitted` flag handles this)?

2. **Are the 9 minor folds adequately documented in the banner** to be safely folded into Part
   A's task spec without further review? Or do any need to be in v5?

3. **Is v4 implementable as Part A would build it?** Pretend you're the Part A engineer:
   - You read v4 + the protocol audit (Part A's first sub-task).
   - You build the library + skeleton.
   - Any decision in v4 that's still ambiguous enough to block you?

4. **Round 1-3 folds — any reopened by v4's changes?** Cross-check.

5. **What's left genuinely under-specified** that Part A would need to make a call on (and
   document)?

6. **Net verdict**:
   - **MINOR_ONLY = CONVERGED**: v4 is build-ready; proceed to Part A on v4.
   - **GO-with-1-2-must-folds-to-v5**: small specific changes; one more round.
   - **GO-with-substantive-revisions**: another substantive round needed (would surprise me).
</context>

<verification_loop>
Doc/design review only — NO BUILD, NO RUN. Read v4 + Rounds 1-3 reviews. v4 is mostly a targeted
edit of v3; bounded.
</verification_loop>

<action_safety>
Write only the review doc. Do not modify the design or any code.
</action_safety>

<compact_output_contract>
Report path of the review doc + one-paragraph verdict (MINOR_ONLY / 1-2-folds / substantive) +
top 1-3 must-folds (or "minor only — converged").
</compact_output_contract>
