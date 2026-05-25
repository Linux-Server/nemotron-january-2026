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
| ~~0.3 py3.13t probe~~ | `0.3-py313t/` (RETIRED) | tombstone — B4 rejected 2026-05-24 | — (proof moved to 0.1b) |
| 0.1b native launch-overlap microbench | `0.1-overlap-ablation/0.1b-microbench-spec.md` | spec'd; **GATE = ≥1.5× L40S (~28/box)** | the export from 0.2-pin + a GPU |
| 0.2 libtorch pin + tch-rs gate + export | `0.2-pin-and-export/` | **pin baseline measured** (torch 2.8/cu128, sm_120) | resolve NeMo ver / aarch64 / tch-rs-2.8 binding |
| 0.5 batching sim | `0.5-batching-sim/` | trace schema + pure-Python simulator skeleton + instrumentation spec | server trace capture (spec only — NOT applied) |
| 0.7 aarch64 | `0.7-aarch64/` | version-matrix checklist template | a GB10/aarch64 box |

**Run order (path-forward review re-sequenced to front-load STOP evidence — decision-value order, NOT numeric):**
**OUTCOME SPACE: native Rust/C++ (B1) or STOP.** The user removed B4 (free-threaded Python) and B5 (in-tree extension)
2026-05-24. Spike 0.3 (py3.13t) is RETIRED; its conjunct-2 proof moved into 0.1b (native launch-overlap microbench).

1. **0.0-pre residual-ceiling arithmetic** (free, paper; do FIRST — may STOP with zero spend; see `decision-template.md`).
2. **0.5 one-pass histogram + synthetic phase-sensitivity** (afternoon; existing 86%-B=1 data likely already kills 3–5×).
3. wire the Python plan's **Step-5 GIL probe** into 0.1.
4. **0.1 = (a) reduced binary** (single-process vs MPS overlap) **+ (b) native launch-overlap microbench** (the
   conjunct-2 proof; N no-GIL threads replaying the captured encoder graph — needs the graph + a GPU).
5. **0.11 GPU mem** (only if 0.5 keeps graphs) → **0.7** (non-gating).
6. **0.4 decision memo.** Only if it says "B1 path" do the Wave-2 ports get funded.

**Safe to start in parallel NOW (wastes nothing if 0.0 STOPs):** 0.0-pre arithmetic + threshold freeze; 0.5 synthetic
runs; making the Python Step-5 probe emit a 0.1-consumable decode-vs-glue split.
**Do NOT start:** any native port (0.6a/0.2/0.8/0.10), Rust/C++ scaffolding, the full 0.1 ablation matrix.
