# Codex Phase 2 Review - Round 5 Final Convergence

Scope: final convergence pass over `phase2-round1-FOLDED.md` through `phase2-round4-FOLDED.md`, plus the Round 5 brief. Goals remain G1 density and G2 P50/P95 tail.

## Residual-Disagreement Scan

No residual Codex-vs-Opus disagreement remains on the engineering findings. By Round 4, the folded review says both reviewers independently built the same numeric GO/STOP tree and converged on the 5090 `>=2.00x` bar, the TTFT-vs-TTFS orphan, and the goal-traceability table (`phase2-round4-FOLDED.md:3-5`, `phase2-round4-FOLDED.md:94-98`).

Settled disagreements or downgrades:

1. Round 1's "AOTI primitive unspecified" survives, but Round 2 resolves the intended topology: one AOTI loader, `num_runners=N`, one shared constants set, explicit per-worker streams (`phase2-round2-FOLDED.md:8-31`). The remaining question is empirical overlap, not API existence.
2. Round 1's steady codisk/shared-weight concern is re-scoped: steady can use the `num_runners=N` pool; codisk/user-managed sharing is mainly for finalize buckets (`phase2-round2-FOLDED.md:20-27`, `phase2-round2-FOLDED.md:38-41`).
3. SessionState/audio/fork isolation is downgraded from broad concern to targeted shared-model/per-thread-handle risk (`phase2-round2-FOLDED.md:58-80`).
4. BW-bound is downgraded from conclusion to hypothesis requiring counters (`phase2-round2-FOLDED.md:53-57`).
5. G2 is settled as Step 1 reference telemetry and Step 4 binding measurement, because scheduler/WS behavior shapes the tail (`phase2-round2-FOLDED.md:82-91`, `phase2-round3-FOLDED.md:52-70`).

One Round-4 threshold needs a consistency edit, not a relitigation:

- The Round-4 tree has Step 1b PASS requiring `TTFS_spread <=1.10x` Python spread (`phase2-round4-FOLDED.md:33-39`, `phase2-round4-FOLDED.md:59-60`), while the binding Step 4 rule switches to non-regression within `+5ms` when Python spread is already `<50ms` (`phase2-round4-FOLDED.md:43-49`, `phase2-round4-FOLDED.md:64-75`). Since Rounds 2-3 explicitly say Step 1 tail is a placeholder/reference and Step 4 is binding, make Step 1b tail a build-risk signal, not an automatic STOP, unless p95/p99 SLO, correctness, or WER fails. Recommended wording: Step 1b GREEN if `<=1.10x`; YELLOW if worse but within the Step 4 non-regression branch; STOP-candidate only if absolute p95/p99, WER, correctness, or gross spread regression `>max(1.25x, +10ms)` fails after corroboration.

Otherwise the numeric tree is internally consistent enough to freeze before Step 1: Step 0 cheap kill-gates, Step 1a spend-control, Step 1b L40S ceiling with margin, Step 4 realized technical GO (`phase2-round4-FOLDED.md:7-50`).

## Completeness Check

G1 density is covered end-to-end with the Round-4 tree, if the plan is edited to include it:

- Primitive feasibility: Step 0a proves `num_runners=N` overlap, memory-flat shared weights, explicit streams, and concurrent==serial (`phase2-round4-FOLDED.md:14-17`).
- Decode/finalize safety: Step 0b/0c cover per-thread handles, `.item()` sync, real finalize, hot buckets, and stale-generation leak checks (`phase2-round4-FOLDED.md:18-22`).
- Spend control and ceiling: Step 1a 5090 is not a GO; Step 1b L40S is the ceiling gate with `>=max(34, 1.80*S_py_L40S)` (`phase2-round4-FOLDED.md:24-41`).
- Realized result: Step 4 gates on admitted successful native streams/box against `G1_floor = max(28, 1.50*S_py_L40S)`, with reject/error bounds and a fresh Python baseline (`phase2-round4-FOLDED.md:9-12`, `phase2-round4-FOLDED.md:43-49`).
- False GO/STOP protection: Step 2/3 must implement box-global admission caps, priority finalize policy, WS-tail decomposition, and stale-generation tests before Step 4; STOP requires repeats, negative controls, topology sweep, profiler attribution, and harness health logs (`phase2-round3-FOLDED.md:19-50`, `phase2-round4-FOLDED.md:77-89`).

G2 tail is covered only after one user-level ambiguity is resolved:

- If G2 means server-side `vad_stop -> final` TTFS, the folded plan covers it: Step 1 reference telemetry, Step 4 binding `TTFS_spread`, p95/p99 guardrails, WS-tail decomposition, and stale-generation tests (`phase2-round3-FOLDED.md:53-58`, `phase2-round4-FOLDED.md:51-62`).
- If G2 means literal first-token TTFT, it is still orphaned until Step 3/4 add first-token timestamps and bind on `TTFT_spread`; Round 4 explicitly leaves this to the user (`phase2-round4-FOLDED.md:64-75`, `phase2-review-brief-round5.md:18-22`).

One remaining coverage gap can still make a Step 4 GO wrong if ignored: multi-turn under load. Round 3 notes `ec2_loadgen.py` is one utterance per connection and does not load-test retained-context multi-turn behavior (`phase2-round3-FOLDED.md:71-74`). This is not a Step 1 build blocker, but before Step 4 the plan must either add a multi-turn subcurve or explicitly state the Phase 2 gate is single-utterance only.

No other decision-critical G1/G2 gap remains ungated after the must-have edits below.

## Minimum-Viable Must-Have Edits To `PHASE2-PLAN.md`

Change these 8 things before Step 1 is built. These are the minimum; everything else is commentary or downstream detail.

1. **Replace the current Step 1 hard gate with the staged numeric decision tree.** Add Step 0a/0b/0c cheap kill-gates, Step 1a 5090 spend-control, Step 1b L40S ceiling, and Step 4 realized GO. Use the Round-4 thresholds: Step 0 overlap/correctness/memory gates; Step 1a PASS `>=2.00x`, STOP-candidate `<1.50x`; Step 1b PASS `>=max(34,1.80*S_py_L40S)`, STOP-candidate `<G1_floor`; Step 4 GO `S_native_step4 >= max(28,1.50*S_py_L40S)` with reject/error/WER/SLO bounds (`phase2-round4-FOLDED.md:7-50`). Also apply the Step 1b G2 consistency edit above.

2. **Add the user-ratified goal contract.** State whether Phase 2 is now a two-goal gate or density-only plus reported tail. State whether G2 is first-token TTFT or `vad_stop -> final` TTFS. If undecided, instrument both and do not call Step 4 a two-goal pass until the user ratifies the choice (`phase2-round4-FOLDED.md:64-75`).

3. **Specify the Step 0/1 ownership and topology contract.** One shared steady AOTI loader with `num_runners=N`; explicit per-worker CUDA streams; per-worker `SessionState`, `AudioFrontend`, `enc_first`, `joint`, `predict`, and `preproc`; user-managed/codisk constants for finalize buckets; mutex/default-stream/`num_runners=1` negative controls (`phase2-round2-FOLDED.md:20-31`, `phase2-round2-FOLDED.md:64-76`, `phase2-round2-FOLDED.md:93-108`).

4. **Make correctness-before-performance non-negotiable.** Before trusting throughput, require concurrent==serial token/event equality with `0` mismatches over real decode, real finalize, same/mixed/hot-bucket finalize, collector fields, and stale-generation-sensitive events. Semantic WER must be within `max(WER_py + 0.5pp, 1.10*WER_py)` on the benchmark corpus (`phase2-round1-FOLDED.md:27-37`, `phase2-round2-FOLDED.md:77-80`, `phase2-round4-FOLDED.md:9-12`).

5. **Add the telemetry and artifact schema that Step 2 needs.** Every Step 0/1 run must log throughput, p50/p95/p99, P95-P50 and P99-P50, enqueue-to-first, enqueue-to-final, queue wait, runner wait, `.item()` wait, finalize wait, CUDA-event durations, Nsight/CUPTI counters, memory, CPU/loadgen health, `num_runners`, stream mode, corpus SHA, model artifacts, and topology. Step 2 is invalid if it only cites a scalar knee (`phase2-round1-FOLDED.md:48-50`, `phase2-round2-FOLDED.md:103-108`, `phase2-round3-FOLDED.md:33-36`).

6. **Add the false-STOP protocol.** A STOP-candidate is not a STOP until three runs have CV `<=10%`, negative controls behave as expected, topology sweep/fallbacks are tested, profiler counters identify a real limit, and harness health is logged. If a fallback topology passes, it is a pivot, not a Phase 2 STOP (`phase2-round4-FOLDED.md:77-89`).

7. **Write the downstream Step 2/3 preconditions now, because they shape Step 1 telemetry.** Step 2 must define box-global active/admitted cap plus backlog-count cap, offered/admitted/rejected/close-code accounting, and priority-finalize mapping onto `num_runners=N`. Step 3 must include WS-tail microbenching and stale-generation suppression tests before Step 4 (`phase2-round3-FOLDED.md:8-50`, `phase2-round4-FOLDED.md:77-82`).

8. **Write the Step 4 apples-to-apples contract now.** Re-measure Python back-to-back on the same L40S; pin corpus, commits, artifacts, loadgen env, hardware/driver, server flags, admission caps, semantic-WER config; report no-shed and admitted-through-shed curves; count only admitted successful streams; bound intentional rejects at `<=10%` and non-intentional admitted errors at `<=1%`. Add the multi-turn decision: either Step 4 includes a multi-turn subcurve or explicitly scopes the gate to single-utterance sessions (`phase2-round3-FOLDED.md:52-74`, `phase2-round4-FOLDED.md:9-12`).

Nice-to-have, not a Step 1 build blocker: Step 5 per-target confirmation table and Spark aarch64 preflight. Keep it in the plan, but it should not block Step 0/1 (`phase2-round3-FOLDED.md:76-85`).

## Sign-Off

Plain sign-off: **GO-to-build Step 1, with Step 0 kill-gates first, after the eight must-have plan edits above are applied.** There are no remaining reviewer-discovered engineering blockers to starting the Step 0/1 harness.

What the reviews settle:

- The measurement design and sequencing.
- The native topology to test first.
- Correctness-before-performance.
- The numeric G1 density gates.
- The false-STOP protections.
- The Step 2/3/4 preconditions needed so a later GO/STOP is meaningful.

What only the user settles:

- Whether the final Phase 2 technical GO is conjunctive G1+G2, or the original density-only gate with G2 reported separately.
- Whether G2 means first-token TTFT or server-side `vad_stop -> final` TTFS.
- Whether product/funding policy wants stricter reject-rate or SLO bars than the technical review's recommended thresholds.
- Whether multi-turn under load is in scope for the Step 4 gate.

If the user ratifies the two-goal gate and disambiguates G2 before Step 4, the corrected plan is sound end-to-end. If the user does not settle those items, Step 0/1 can still be built if it instruments both TTFT and TTFS, but Step 4 cannot be honestly signed off as a two-goal Phase 2 decision.
