# Wave-1 spike scaffolding

Scaffolding for the **Wave-1 cheap kill-shot spikes** of `../PLAN.md`. Per the reviewed verdict, **only Wave 1** is
green-lit; Wave-2 byte-exact ports (0.6a/0.8/0.2/0.10) and all of Phase 1+ are blocked on the §0 worth-it gate (which
needs the near-term Python plan `proj-2026-05-24-0859` to land first).

**This scaffolding makes NO changes to the production `server.py` path and spins up NO cloud/EC2.** Where a spike needs
GPU / cloud / the post-Python baseline, the harness is a skeleton with the run-step marked `BLOCKED`.

| Spike | Artifact here | Status of the artifact | Blocked on |
|---|---|---|---|
| 0.9 mutability audit | `0.9-mutability-audit.md` | **COMPLETE** (pure code-archaeology) | nothing |
| 0.11 graph ownership | `0.11-graph-ownership.md` | **COMPLETE** (analysis + decision criteria; numbers TBM) | GPU for the memory measurement |
| 0.4 decision memo | `decision-template.md` | template + filled decision tree + pre-registered-thresholds block | the spike outcomes |
| 0.1 overlap/MPS ablation | `0.1-overlap-ablation/` | harness skeleton + ablation matrix + thresholds | GPU + post-Python baseline |
| 0.3 py3.13t probe | `0.3-py313t/` | probe-script skeleton + README | a py3.13t env + the stack |
| 0.5 batching sim | `0.5-batching-sim/` | trace schema + pure-Python simulator skeleton + instrumentation spec | server trace capture (spec only — NOT applied) |
| 0.7 aarch64 | `0.7-aarch64/` | version-matrix checklist template | a GB10/aarch64 box |

**Recommended run order once unblocked** (cheap kills first, per PLAN §0 waves): finish the Python plan → record its
baseline + residual (0.0) → 0.1, 0.3, 0.5 (post-Python) + the paper audits 0.9/0.11/0.7 → 0.4 decision memo. Only if 0.4
says "B1 path" do the Wave-2 ports get funded.
