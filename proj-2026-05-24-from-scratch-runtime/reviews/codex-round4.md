# Codex adversarial review - Round 4

Review target: `proj-2026-05-24-from-scratch-runtime/PLAN.md` v4. New/convergence findings only.

1. [BLOCKER] No integrated decision tree maps spike outcomes to kill / B4 / B1 / fusion.

   v4 has many local gates, but 0.4 only says to "pick language + backend + process shape"; it does not define the logic that combines outcomes. That leaves real ambiguous combinations:

   - 0.1 says only MPS/multi-proc overlaps, while 0.6a/0.2/0.8 pass. Does the project continue as a native runtime behind the same MPS topology, or does the "single process shared weights/no MPS tax" density thesis die?
   - 0.3 py3.13t closes the post-Python residual, while 0.6a has not run or fails. Is B4 the chosen success path, or is the C++ decode still funded?
   - 0.5 shows B stays near 1, while 0.1 shows scheduler overlap improves tail. Does the plan proceed with B1 but drop the 3-5x throughput claim, or pause because density is not enough?
   - 0.11 says per-lane graph pools do not fit at target lanes, but non-graphed B1 could still reduce tail. Does that fall back to B4, proceed B1-without-graphs, or abandon the 40-48/box target?

   This matters because the source axes are independent. The deployed server uses `greedy_batch` with `loop_labels=True` and `use_cuda_graph_decoder=False` (`src/nemotron_speech/server.py:1463-1474`), strict fresh/established batch keying (`src/nemotron_speech/server.py:4789-4812`, `src/nemotron_speech/batch_primitives.py:100-139`), exact graph buckets/static buffers (`src/nemotron_speech/cudagraph_encoder.py:6-17`, `src/nemotron_speech/cudagraph_encoder.py:293-296`), and production runs multi-proc under MPS (`deploy/launch_multiproc.sh:57-68`). A pass/fail on one axis does not imply a decision on the others.

   Recommended change: make 0.4 produce an explicit outcome matrix:
   `0.0 residual too small -> stop`; `0.3 closes residual and is stable -> choose B4`; `0.1 negative for single-process overlap -> stop or explicitly choose native-under-MPS with reduced density target`; `0.6a/0.2/0.8 fail -> no B1 unless accepting a named T1-only risk`; `0.5 negative -> drop 3-5x and re-run worth-it`; `0.11 negative -> drop graph/density claim or revise topology`; `3.3 fusion only gates the 6-10 ms headline`.

2. [BLOCKER] The 0.6 -> 0.6a split is still incomplete, and one remaining line re-adds the graph-decoder work to the funding gate.

   v4 correctly defines 0.6a as deployed eager label-looping equivalence and explicitly excludes the Blackwell CUDA-graph decoder. But the B1b row still says "Spike 0.6" is the go/no-go, Phase 1.2 says native decode is "from 0.6", and Top risk #2 says 0.6 must solve the Blackwell cuda-graph-decoder NeMo punted on. That contradicts the deployed configuration: the server configures `greedy_batch` with `use_cuda_graph_decoder=False` (`src/nemotron_speech/server.py:1463-1474`), and the server disables batching if a CUDA graph decoder is present (`src/nemotron_speech/server.py:976-999`). NeMo's label-looping graph support is a separate mode family (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/transducer_decoding/label_looping_base.py:73-76`) with dedicated capture streams and `capture_error_mode="thread_local"` (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:839-890`).

   Recommended change: replace all funding-gate references with `0.6a`. Move "Blackwell cuda-graph-decoder" out of Top risk #2 and into Phase 3/0.6b research only. Phase 1.2 should say "native decode from 0.6a"; 3.2 can separately discuss graph/fixed-trip decode.

3. [MAJOR] B4 is still an orphaned fallback, not a coherent branch.

   B4 is only coherent for a specific outcome: the post-Python residual is mostly Python scheduler/GIL contention, and a py3.13t off-event-loop dispatcher closes it end-to-end. It is not a fallback for "native decode needed regardless." The current model calls are already offloaded through executors (`src/nemotron_speech/server.py:3087-3095`, `src/nemotron_speech/server.py:3183-3192`), while the live scheduler is still one asyncio loop that needs `sleep(0)` for I/O liveness (`src/nemotron_speech/server.py:4456-4491`). Separately, the deployed label-looping decode still has Python-controlled loops over GPU tensors (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:330-377`, `/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:466-508`) and in-place partial-hypothesis merge (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py:783-804`).

   Recommended change: make B4 a first-class branch in the decision tree. If 0.3 passes the real post-Python tail/density gate and py3.13t PyTorch/NeMo is production-stable, choose B4 and skip the native ports. If 0.3 fails and 0.6a fails, do not say "fall back to B4"; the honest outcomes are stop, or explicitly accept a non-byte/state-exact native decode risk. If 0.6a proves native decode is required for the residual gap, B4 no longer helps unless 0.3 has already proven otherwise.

4. [MAJOR] Phase 5 is not a real path from "native runtime passes T1" to serving production traffic.

   v4 says "HAProxy; deploy updates; canary 1 replica -> ramp; rollback triggers", but production today is a coupled launcher shape: it starts MPS, then K Python processes (`deploy/launch_multiproc.sh:57-68`), with deployed batching/finalize/graphs set through env (`deploy/launch_multiproc.sh:42-45`) and L40S K capped by memory (`deploy/launch_multiproc.sh:6-9`, `deploy/launch_multiproc.sh:19-24`). The supervisor explicitly still lacks LB drain, alerting, and MPS-context restart after a crash (`deploy/launch_multiproc.sh:70-79`). The server health endpoint only reports loaded vs loading (`src/nemotron_speech/server.py:8842-8847`), which is not enough readiness for a new runtime with graph capture, lane pools, and native decode state.

   Recommended change: add a concrete rollout phase before production cutover: run Python and native side by side behind the same LB substrate; shadow/mirror live or replayed audio to native and diff exact event streams without serving native output; define readiness as model loaded + graph/lane pools captured + T1 canary passing; canary by backend process/replica; rollback by removing native backends and returning all traffic to Python; specify whether MPS stays on or is removed as an output of 0.1. Also add the launcher work as real tasks: LB drain on restart, alerting, and MPS-context restart policy.

5. [MAJOR] Phase 0 is still over-built before the cheap kill decisions, despite the new effort table.

   v4 now admits Budget A is 12-20 engineer-weeks, but still says Track A can run now and includes the expensive ports: 0.6a native label-looping equivalence, 0.8 native preprocessor, and 0.2 encoder export. Those are not small probes. 0.6a must reproduce the deployed label-looping state machine (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/transducer_decoding/label_looping_base.py:51-59`), max-symbol forced advance (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:466-484`), state split/merge (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/transducer_decoding/rnnt_label_looping.py:569-620`), and NeMo's partial-hyp merge (`/home/khkramer/src/parakeet/venv/lib/python3.12/site-packages/nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py:783-804`). 0.8 must reproduce constant-plan preprocessing (`src/nemotron_speech/server.py:1509-1593`) and finalize's multi-pass preprocessor loop (`src/nemotron_speech/server.py:6927-6942`).

   Recommended change: separate "can run early" from "should fund early." Wave 1 should be the cheap existence/path killers: finish the Python plan and measure 0.0, then 0.1, 0.3, 0.5, plus 0.9/0.11 as paper/prototype audits. Only if Wave 1 says "residual exists, B4 insufficient, single-process/native still plausible" should 0.6a/0.8/0.2 be funded. Otherwise the plan can spend a quarter on a decode/preproc port for a project 0.0 or B4 would have killed.

6. [MAJOR] The most likely cancellation reason is "the remaining gap is not worth a second stack," and the early-exit threshold is too narrow.

   The roofline already says p50 can move only about 12-19 ms because VAD+WAN dominate (`proj-2026-05-23-1731/roofline-COMBINED.md:29-33`), and the near-term plan's honest density math is much lower than the old 64/box headline: K=4 is about 28 in-budget streams/box, not 64 (`proj-2026-05-24-0859/PLAN.md:11-14`). The current deploy is also memory/topology constrained by per-process model+graph pools (`deploy/launch_multiproc.sh:6-9`, `deploy/launch_multiproc.sh:36-45`). v4's §0 says pause if Python already reaches "~40/box and a bounded tail"; that is too high and may miss the real business decision. A Python result like "28/box, bounded p99, overload cliff gone" could make the native rewrite unjustifiable even though it does not hit 40/box.

   Recommended change: replace the single "~40/box" pause trigger with a minimum-worth threshold: named residual p95/p99 gap, named streams/box delta, and an estimated value/cost comparison against ~40-60+ engineer-weeks plus second-stack maintenance. The early exit should be "residual value below threshold", not only "Python reaches the aspirational native target."

7. [MINOR] Minor numbering/label cleanup remains, but it should be folded into the decision-tree edit.

   0.11 appears in Phase 0 before Phase 0.5's 0.10, while the progress table puts 0.10 after 0.4 and before Phase 1. This is not technically wrong, but it makes the already complex gate stack harder to read. More importantly, the stale "0.6" labels above make the numbering look less settled than it is.

   Recommended change: after adding the outcome matrix, present all pre-Phase-1 work in decision order, not numeric trivia order: 0.0/0.1/0.3/0.5 path decision, 0.9/0.11 topology feasibility, 0.6a/0.8/0.2 B1 feasibility, then 0.10 runtime contract.

## Top 5 things to fix

1. Add an explicit 0.4 decision tree mapping measured outcomes to stop / B4 / B1 / native-under-MPS / fusion.
2. Finish the 0.6a rename: remove graph-decoder work from the funding gate and update all "0.6" references that mean "0.6a".
3. Promote B4 from orphan fallback to a first-class cheapest-success branch, and state when it cannot help.
4. Make rollout real: shadow traffic, LB coexistence, readiness, canary, rollback, and MPS/HAProxy launcher hardening.
5. Change the early-exit from "Python hits ~40/box" to "residual value justifies ~40-60+ engineer-weeks and a second stack."
