# Adversarial review brief #4 — final pre-flight (post-fold) — REVIEW ONLY, modify no files

Target: /home/khkramer/src/nemotron-january-2026/proj-2026-05-19-eou-endpointing/PLAN.md

Status: round-3 (Codex `b510t8mq0` + Claude) folded all 5 converged items
into the plan at commit `0768cba`. This round-4 is a **focused
verification-pass** on the post-fold state. Three prior rounds + 3 probes
already happened; round-4 is *not* a structural pass. Two purposes:

1. **Verify the round-3 folds** — did the mechanical edits introduce any new
   inconsistency, stale reference, or internal contradiction? Are the new
   line numbers actually correct in `0462679`? Does the new dual-baseline
   Step 6 make sense and not scope-bloat? Is the new Step-1 dual-cursor
   (model-frame + real-audio-time) logging well-defined?
2. **Red-team for blind spots** — anything the three prior rounds (+ probes)
   missed that could derail Step-1/2/2b/3/4/5/6 once `/implement` starts.
   Specifically: cheap probes that *could* have been run pre-flight but
   weren't, framework assumptions that haven't been verified, gates that
   look right textually but might be empirically wrong.

Do NOT edit/create/fix any file — findings only.

## Read
- The post-fold PLAN.md at `0768cba` (the whole thing; ~310 lines).
- The committed companion docs: `docs/semantic-wer-finalization-finding.md`
  (the canonical doc at `ef1a7a7`) — verify the plan's baseline citations
  match the doc's authoritative numbers.
- The committed Step-7d/8 code: `src/nemotron_speech/server.py`,
  `stt-benchmark/src/stt_benchmark/nemotron_local_stt.py`,
  `stt-benchmark/scripts/measure.py` — verify every cited line number is
  current AND points to what the plan claims.
- Probe A's script (`probe_a_decoding_cfg.py`) — verify the plan's claims
  about what Probe A actually demonstrated.

## Attack hard (post-fold verification + red-team)

1. **Line-number drift on the now-frozen `0462679`.** Round 3 updated
   citations: parent-stream `:~1695`, fork-flush `:~1948`,
   per-session warm-up `:~641`, `change_decoding_strategy` `:461`,
   `_continuous_finalize_emit_locked` `:~1418`, `:~1466` (the model step
   inside it?), `:~1872` (`_process_final_chunk`?), `committed_text` advance
   `:~1511`, `continuous_emitted_text=""` `:~1563`. Verify EACH cites the
   correct site (open the file and check). Did Codex's round-3 say `:~1466`
   for "continuous fork call" — is that the right line? Any number > 5 lines
   off is a defect.

2. **Step 1 dual-cursor (model-frame + real-audio-time) — well-defined and
   computable?** The plan says: when running on `warm200`,
   `session.emitted_frames = warmup_frames` after warm-up, so per-token
   global-frame index must use that as the starting offset; persist both
   model-frame (for rc1 aging in Step 2b) and real-audio-time (for endpoint
   joins). Verify: (a) `session.emitted_frames` actually equals
   `warmup_frames` (15) after `_run_session_warmup` (check server.py:~695);
   (b) the dual-cursor logging is implementable from existing per-chunk
   state without adding new server state; (c) does Step 2b's classification
   need the dual cursor, or does model-frame alone suffice?

3. **Step 6 dual-baseline (`eou` + `eou_warm200`) — sound or scope-bloat?**
   Round 3 added a second full-1000 measured tag for attribution. Two
   measured runs = ~7 h. Is the attribution win worth the compute? Could the
   same attribution be done cheaper (e.g., one `eou_warm200` ship-gate run +
   a smaller-N ablation that doesn't violate the full-1000-for-scored-WER
   rule)? Or is dual-baseline the only methodologically sound option?

4. **Probe-A interpretation — did the plan read the probe results correctly?**
   The probe script outputs "alignments populated", "frame_confidence
   populated", "transcript matches baseline". The plan says Probe A
   "byte-identical to baseline" and "argmax-invariant to log_normalize=True"
   — but Probe A only tested ONE fixture, ONE config-acceptance turn (the
   probe shows alignments populated with `≈1 entries` per stream-step return,
   not per-frame deep populations). Is the plan over-claiming what Probe A
   demonstrated? Should the Gate (b) smoke target be tightened?

5. **Red-team for blind spots.** Top suspect areas:
   - The plan's Step 2 says "log the would-be fork-flush output per chunk
     boundary" as Option (i), "snapshot minimal fork inputs" as Option (ii).
     Implementing (i) requires *running* the fork at every chunk during
     collection (expensive — N flushes per sample); (ii) requires reproducing
     the exact fork inputs offline. Which is being recommended? Cost
     implications?
   - The Step-1 per-token "global encoder-frame index at first emission" —
     is there actually a NeMo API to get the *encoder frame index* a given
     token was emitted at? (Hypothesis.timestamp / alignment frame index?)
     Or does the plan implicitly require maintaining this in our wrapper?
   - Closed-loop concern: Step 1's instrumentation-only mode is supposed to
     be byte-identical to 7d `fork` (env-unset) and WER-equivalent
     (env-enabled). Is it OK that the new client-acceptance bypass in Step 1b
     is gated on `NEMOTRON_EOU_CLIENT=1` AND `continuous_context` mode?
     (i.e., default-client behavior preserved.)
   - Any race / state concern in 7d's worker that the new instrumentation
     reads from but doesn't synchronize? (state_lock holds when
     `previous_hypotheses` is mutated.)
   - The Probe-B nuance: the plan now says "(iii) ≈ 0 expected" more
     strongly. But Probe B was an EYEBALL; it could be wrong. Is the plan's
     wording overconfident? Should Step 2b's GO/NO-GO criteria stay
     defensive even if (iii) shows ~0?

6. **Any internal contradiction or stale stub.** After 3 rounds of editing,
   are there leftover references to things that have been superseded
   (e.g., "375 ms baseline" should now be "warm200 366 ms" everywhere)?

## Output contract
Per item 1-6: SOUND / NEEDS-EDIT / DEFECT. For NEEDS-EDIT/DEFECT items:
exact plan statement (with line number), suggested edit (concise), with
code/PLAN/probe refs. End with: overall **READY** / **READY-WITH-EDITS** /
**NEEDS-FIX-ROUND** for `/implement`, and the top 1-3 must-fix items
(if any). Review only; modify no files.
