# Adversarial review brief #2 — EOU-endpointing PLAN (revised) — REVIEW ONLY, modify no files

Target: /home/khkramer/src/nemotron-january-2026/proj-2026-05-19-eou-endpointing/PLAN.md
This is the SECOND adversarial pass. The plan was revised after review #1 (Codex `bi49ruh1q`)
folded 5 defects, then further corrected for a risk-model error. Do NOT edit/create/fix any
file — findings only. Be a harsh adversary.

## Read
- The target PLAN.md (note its "Risk model (corrected)" para, the Rules, Steps 1/2/2b/3, and
  the Dual-review record).
- Cited code: src/nemotron_speech/server.py ~446-454 (change_decoding_strategy, strategy:greedy),
  ~571, ~1604, ~1374/1851 (_process_final_chunk fork), ~1410 (_continuous_finalize_emit_locked),
  _continuous_append_only_delta; stt-benchmark/src/stt_benchmark/nemotron_local_stt.py ~453
  (finalize gating); stt-benchmark/scripts/measure.py (finalize-budget, vad preflight ~787/848).
- NeMo (read-only, in venv /home/khkramer/src/nemotron-nano-omni/.venv-asr or the NeMo source
  tree): rnnt_decoding.py (confidence_cfg/greedy placement, strategy→GreedyRNNTInfer),
  asr_confidence_utils.py (confidence method, normalized vs raw entropy),
  rnnt_greedy_decoding.py (greedy append-only y_sequence; confidence appended before blank),
  rnnt_utils.py Hypothesis (alignments/frame_confidence/y_sequence), mixins.py
  conformer_stream_step → rnnt_decoder_predictions_tensor(return_hypotheses, partial_hypotheses).
- Parent hard constraints: /home/khkramer/src/nemotron-january-2026/proj-2026-05-17-1708/PLAN.md.

## Attack hard
1. **Risk-model correctness (the core revision).** Is "greedy RNNT is append-only on the
   committed token sequence; tokens with full rc1 right-context are stable; therefore a
   false-early-fire cannot rewrite arbitrary earlier text and its cost is bounded to the
   provisional ≤rc1 tail flushed-against-synthetic-silence + a seam token + the segmentation
   family" actually correct for THIS path (cache-aware FastConformer EncDecRNNTBPE,
   conformer_stream_step, strategy:greedy, rc1)? Any mechanism (prediction-network state,
   partial_hypotheses carry-over, the fork's synthetic-padding flush, beam fallback,
   detok) by which an emitted, beyond-rc1 token could in fact change or by which the cost is
   NOT bounded as claimed? Cite code.
2. **Step 2b methodology.** Is the (i) provisional-≤rc1-tail / (ii) render-only-token-prefix
   -stable / (iii) genuine-beyond-rc1-edit classification well-defined and computable from the
   proposed Step-1 capture (cumulative token-id series, chunk→encoder-frame mapping, rc1 frame
   span R, token-level changed_positions)? Is the chunk→frame→rc1-age mapping actually
   recoverable from conformer_stream_step outputs without NeMo edits? Could (iii) be
   under- or over-counted (e.g., BPE re-segmentation changing token boundaries so a "render-only"
   change looks like an id edit, or vice versa)? Is the expectation (iii)≈0 a safe gating
   assumption or could it mask a real effect?
3. **Does Step 2b actually de-risk what it claims?** If (iii)≈0, is it truly valid to conclude
   "settled-past-rc1 ⇒ rewrite-safe" and let Step-3 set a looser false-fire threshold F? Or
   does the bounded seam/provisional-tail cost still dominate WER enough that F must stay tight
   regardless — i.e., is 2b decision-relevant or just descriptive?
4. **Residual review-#1 items.** Verify the 5 prior defects are genuinely resolved by the
   rewrite (config placement; client EOU-accept REQUIRED; Step-4 replays fork-flushed text;
   full-1000 rule / proxy non-authoritative; scratch-vs-source). Any regressions or
   newly-introduced inconsistencies from the revision?
5. **Constraint / phasing.** Any hard-constraint violation, any gate mis-ordered
   (expensive-before-cheap), any step that silently needs a NeMo/framework edit or a new dep,
   any place the plan still overclaims measurability or floor.

## Output contract
Per item 1-5: SOUND or DEFECT. For each DEFECT: exact plan statement, why wrong/risky,
corrected approach, code/PLAN refs. End with: overall YES / YES-WITH-FIXES / NO for handing
to `/implement` (after the parent project closes), and the top 1-3 must-fix items. Review
only; modify no files.
