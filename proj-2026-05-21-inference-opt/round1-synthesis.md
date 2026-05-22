# Round 1 — synthesis (self analysis + Codex analysis + the running g6e data)

Two independent round-1 analyses (`round1-self-analysis.md`, `round1-analysis.md`) + the EC2 g6 data converge.

## Strong agreements (both analyses, independently)
1. **The regime frame.** Per-instance behavior moves: (1) launch-dispatch-bound (1 lane, GPU idle, FLOPs
   irrelevant) → (2) filling the GPU (lanes) → (3) GPU-compute-bound (lanes filled it; FLOPs finally matter).
   **Which lever helps depends on the regime.** g6/L4 is at the 2→3 boundary at lanes=2 (GPU ~92%).
2. **#1 convergent lever = batch the `vad_stop` barrier-drain** (my T1/S4 = Codex A1/B2). Wins on BOTH axes:
   TTFS-under-load (in-phase N=120 currently TTFS p95 **2567 ms** with 452 B=1 barrier chunks → target <400 ms)
   AND the in-phase high-N cap (**115 → ~150-180**). Known location: `server.py:3732-3759`
   (`_scheduler_drain_ready_barrier_locked` B=1 loop). Measured failure signature. Quantified upside. **Top prototype.**
3. **Highest-value EXPERIMENT = the per-GPU lane-count sweep** (Codex B1) — **in progress on g6e now**. Sets the
   real production ceiling + whether L40S is worth ~2× the cost. Local 5090 lane data isn't predictive for Milan/L40S.
4. **TTFS ≈ scaling.** TTFS is already ~40-150 ms until saturation; the barrier-drain is the one TTFS-specific item.

## Codex additions worth keeping (beyond my pass)
- **A2 — finalize should NOT block all lanes.** Since lanes use per-lane model *replicas*, route final/fork work to
  the session's *pinned* lane (block only that lane), not `_scheduler_exclusive_model_path` which stalls every lane
  (`server.py:4785-4808`). Mixed steady+final win. Medium risk (prove replica-local mutation).
- **A3 — batch concurrent finalize storms** (O(N) final calls → O(N/B)) for simultaneous vad_stop/close bursts.
- **B6 — remove the 4 unconditional `cuda.synchronize()` + heavy telemetry from the hot batched path**
  (`server.py:5203-5225,5290-5311,5380-5390`). Extra syncs kill cross-lane stream overlap → ~5-15% + better lane
  scaling. **Low-risk, high-leverage for the lane regime.**
- **B7 — relax mixed-key lane concurrency**: replicas allow *different* keys on different lanes (disjoint sessions).
- **A5 — rc drives FINAL padding too**: final pad = (rc+1)·shift → rc1 = 320 ms, rc0 = 160 ms synthetic audio.
  A *final-only* shorter pad could speed finals (rc0 globally crashes — see memory; final-only is uncertain/risky).

## The fp16 nuance — resolve my-vs-Codex disagreement EMPIRICALLY
I ranked **fp16/bf16 #1 scaling** (~2× the GPU ceiling once lanes fill it). Codex ranked it **#5** and cautioned it
could be **~zero** if the workload stays launch/decoder-bound even at lanes=2 GPU-92% (the "92%" may be inflated by
launch-overhead occupancy, not pure compute). **This is testable and the test resolves a top-priority unknown:**
measure GPU-*active* vs total at lanes=2, and run fp16 directly — if GPU-active dominates → fp16 ~2×; if launch
gaps still dominate within each lane → fp16 ~nothing. So fp16 is a **round-2 cloud probe**, not an assumption.

## Round 2 plan (parallel; no conflicts)
- **2a — Codex (local, server.py): prototype the batched barrier-drain (A1/B2).** Flag-gated, byte-exact, the
  #1 convergent lever. Known location + measured target. Local in-phase N=115/120/150 + FORK_ASSERT + byte-exact finals.
- **2b — me (cloud g6e, once the lane sweep frees the box):** (i) **fp16/bf16 probe** (standalone — GPU-active fp32
  vs fp16 + transcript/WER delta; resolve the fp16 nuance), (ii) lock the **g6e optimal lane config** from the sweep,
  (iii) quick look at **B6** (sync removal) feasibility.
  *(Standalone probe on cloud + Codex's local server.py edit don't conflict — different machines/files.)*
- Gating result: the g6e lane sweep (running) → L40S ceiling + config + $/stream vs g6.

Deferred to the proj plan: A2, A3, B6, B7, phase-alignment (B4), per-B CUDA-graphs (the existing plan), adaptive
coalescing (B12), the per-GPU config matrix (B5).
