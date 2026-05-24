# Codex path-forward recommendations

Date: 2026-05-24

Position: keep the v6 gate intact. Do not start Phase 1 or the Wave-2 native ports yet. The most valuable work is to make
the near-term Python plan produce a decision-grade residual packet for 0.0, with pre-registered kill thresholds before
any Wave-1 measurements are collected.

## Prioritized recommendations

1. **Make the post-Python residual packet the single next action.**

   The next funded unit of work should be the Python plan landing plus a concrete 0.0 decision packet, not another native
   scaffold. The packet should record the exact Python commit, flags, GPU type, process/MPS shape, admitted vs attempted
   load, p95/p99 server-side tail at the in-budget operating point, overload behavior, streams/box, GPU memory headroom,
   GPU utilization, CPU/event-loop pressure, natural batch-B distribution, and the GIL/decode-vs-scheduler attribution
   from the Python Step 5 probe.

   Work that should happen now: make sure the Python validation emits the fields needed for 0.0, 0.1-lite, and 0.5
   without another instrumentation pass. In particular, capture always-on queue/ready age, lane wait, finalize timing,
   batch key, fresh/established decoder-state flag, lane affinity, B distribution, and attempted/admitted counts. Also
   fill and freeze the pre-registered threshold block before collecting new Wave-1 evidence.

   Work that must wait: any native serving runtime, native decode implementation, native preprocessor, encoder export,
   Rust/C++ scheduler, or runtime contract build. Those are only justified after 0.0 says the residual is worth a second
   stack.

2. **Convert the worth-it gate from narrative to numbers before running Wave 1.**

   Recommended starting thresholds for discussion, to be replaced by the user's actual business values before data:

   - 0.0 residual: require a named monetizable residual, e.g. at least `+12` in-budget streams/box over the measured
     post-Python result at equivalent tail, or at least `150-200 ms` p99 server-side tail reduction at the same admitted
     load, plus an explicit value judgment that this beats `40-60+` eng-weeks and a second stack.
   - 0.1 overlap: require single-process finalize+steady overlap of at least `1.5x` over the measured Python/MPS
     baseline, with queue+lane wait below a named bound at the operating point and no more than `5-10 ms` added
     per-chunk latency from dispatch changes.
   - 0.5 batching/graphs: require median `B >= 2`, p95 `B >= 4`, exact-B graph replay hit rate `>= 90-95%`,
     eager fallback `<= 5-10%`, added batching wait `<= 5 ms` unless explicitly traded for tail, and at least `10%` GPU
     memory headroom or `>= 2 GB`, whichever is larger, after model plus graph pools at the target lane/process shape.

   If no one is willing to write these numbers down, the correct decision is STOP or defer. Undefined thresholds make the
   later decision vulnerable to rationalizing a rewrite after the fact.

3. **Run Wave-1 spikes in decision-value order, not numeric order.**

   Once the Python plan is validated, run:

   1. **0.0 worth-it gate.** This is the highest value per hour because it can kill the whole native project without
      touching native code.
   2. **0.5 quick trace kill.** If the emitted post-Python traces show that same-key sessions rarely become ready inside
      the allowed wait window, drop the 3-5x batching claim immediately. Do the one-pass histogram first; only run the
      richer simulator if the quick pass is not decisive.
   3. **0.3 stage-1 B4 feasibility.** Time-box the py3.13t/PyTorch/NeMo import and single-chunk sanity check. If the
      environment is unavailable or fragile, B4 may be killed cheaply. If it works, do not yet declare victory; stage 2
      still needs a real off-event-loop dispatcher.
   4. **0.1 overlap/MPS ablation, starting with a reduced matrix.** First compare post-Python single-process,
      MPS/multi-proc, lane wait, queue wait, and CUDA-event timelines using the cheapest toggles already available. Only
      build the full ablation matrix if the reduced run is ambiguous.
   5. **0.3 stage-2 B4 end-to-end.** Run this before any B1 native ports if 0.3 stage 1 works and 0.1 says the residual
      is plausibly Python scheduler/GIL-bound. B4 is the cheapest success path, not a fallback.
   6. **0.11 GPU memory measurement.** Run this only if 0.5 keeps steady graphs/density alive. Otherwise the graph-pool
      memory result is interesting but not decision-critical.
   7. **0.7 aarch64.** Keep it as a platform pre-check when the GB10 exists. It should not block the L4/L40S B4/B1/STOP
      decision.

4. **Use cheaper kill signals before full spike scope.**

   For 0.0, do not wait for native artifacts. The Python Step 6 validation should be enough to decide whether a residual
   exists.

   For 0.1, start with evidence already close to the current system: post-Python K/lane/MPS sweeps, CUDA-event timelines,
   lane wait, event-loop/ready age, and GIL attribution. The full lock/gate/sync toggle matrix is only worth building if
   these show a large residual but cannot isolate the serializer.

   For 0.3, split B4 into two gates: environment feasibility and end-to-end dispatcher proof. A failed import/build or
   unstable transitive extension stack is a cheap no-go. A successful import is not enough to fund B4; it only permits
   the stage-2 dispatcher measurement.

   For 0.5, run a histogram query before the simulator: same batch key, same fresh/established state, same T/drop_extra,
   ready within `5/10 ms`, and lane-compatible. If that natural coalescence is mostly B=1, the 3-5x target is dead.

5. **Keep the gating as written; allow only narrow parallel work.**

   The gating is correct. Phase 1+ and Wave-2 ports should not start before the Python residual is known.

   Acceptable parallel work now:

   - Pre-register thresholds and define the 0.0 residual packet schema.
   - Add low-impact trace fields to the Python validation path while that work is already in flight.
   - Run synthetic 0.5 simulations only to sanity-check thresholds, not as evidence.
   - Do 0.3 stage-1 environment reconnaissance if it is time-boxed to a day or two.
   - Prepare a frozen-fixture schema and comparator plan for 0.6a, but do not implement the native label-looping decode.

   Starting the 0.6a native decode implementation now would violate the worth-it gate in spirit even though the fixtures
   are baseline-independent. The expensive part is not the comparator; it is reproducing NeMo's deployed eager
   `greedy_batch` label-looping state machine. That spend should wait until Wave 1 says B1 is still alive.

6. **Default recommendation: STOP unless Wave 1 produces unusually strong evidence.**

   STOP if any of these are true:

   - The Python plan leaves only p50 movement or a small p95/p99 residual.
   - The remaining density delta is materially below the pre-registered value threshold.
   - 0.1 shows only MPS/multi-proc overlap and no single-process overlap. Native-under-MPS is tail-only; re-run 0.0 and
     expect STOP unless the tail value is exceptional.
   - 0.5 says B remains near 1 or graph replay hit/memory makes steady graphs impractical.
   - 0.3 B4 closes the residual with acceptable production stability. In that case choose B4, not B1.

   Choose **B4** if py3.13t plus a real off-event-loop dispatcher closes the post-Python residual end-to-end, preserves
   existing NeMo correctness behavior, and has an acceptable build/deploy/support story. B4 probably does not deliver the
   full shared-weight memory-density thesis by itself, so its acceptance must be based on measured tail and measured
   in-budget streams/box, not "no GIL" as a proxy.

   Choose **B1** only if all of the following are true: the residual is worth the cost, 0.1 proves single-process native
   overlap is plausible, 0.3 fails or is not production-stable, 0.5/0.11 keep the batching/graph-density claims alive,
   and the 0.9/0.11 ownership constraints remain viable. If B1 survives this far, fund Wave 2 in the order that kills
   fastest: fixture/oracle hardening, then 0.6a native decode equivalence, then 0.2 encoder export and 0.8 preprocessor.

7. **Add one middle option to keep in mind: a Python-hosted native decode extension.**

   The five review rounds mostly framed the choices as B4 Python scheduler rewrite, full B1 native runtime, or STOP. A
   possible middle option exists if the post-Python residual is strongly decode-GIL-bound, B4 is blocked by ecosystem
   maturity, and 0.1 does not justify a full runtime: implement only the deployed RNNT label-looping decode as a C++/CUDA
   or pybind extension under the current Python server.

   This is not cheap and still needs the 0.6a state-equivalence gate, but it avoids replacing the WebSocket protocol,
   deploy topology, admission, metrics, and most scheduler machinery. It would not solve shared-weight density,
   exclusive-gate topology, or MPS limits. Treat it as a fallback branch to discuss only if the evidence says "decode is
   the residual" and "full runtime is not worth it."

8. **Do not let admission-only wins hide the business decision.**

   The Python plan may improve p99 by shedding above-cap traffic. That is good product engineering, but for 0.0 the
   comparison must distinguish attempted load, admitted load, and quality at admitted load. A native rewrite should not
   be justified by a graph where p99 improves only because the system now rejects more sessions, unless that is the
   intended business tradeoff.

## Top 3 next actions

1. Finish the Python plan through combined validation and write a 0.0 residual packet with exact commit, flags,
   hardware, admitted/attempted load, p95/p99, streams/box, overload behavior, B distribution, and GIL attribution.
2. Before collecting Wave-1 data, fill and freeze the numeric thresholds in `spikes/decision-template.md` or an adjacent
   decision file: residual value, 0.1 overlap/wait limits, and 0.5 B/graph/memory limits.
3. Add the low-impact trace fields needed by 0.5 and 0.1 while the Python validation path is already being touched; do
   not start 0.6a/0.2/0.8 implementation.

## Topics to discuss with the user

1. What exact residual is worth `40-60+` eng-weeks plus a second production stack: how many p99 milliseconds, how many
   in-budget streams/box, and on which GPU?
2. Is a production py3.13t/PyTorch/NeMo stack acceptable if B4 works, including custom builds or less mature extension
   support?
3. Is v1 strictly EN 0.6b, no multilingual/prompted model, no EOU-probe alignments/confidence, and no p50 goal?
4. If the evidence says "decode-GIL-bound but full native runtime is too expensive," is a Python-hosted native decode
   extension an acceptable middle path, or should the decision be STOP?
