# Round 1 — self analysis: TTFS + scaling optimizations (g6/g6e)

(My own pass, parallel to the Codex round-1 analysis and the running g6e sweep. Fold all three together after.)

## The unifying frame: a bottleneck-REGIME transition

Everything we've measured says the per-instance behavior moves through **three regimes** as we apply levers:

1. **Launch-dispatch bound (1 lane).** One core issues kernels serially; GPU sits idle (L4: 46%). Knee ≈ how fast
   one core dispatches ≈ Modal (~5) / Milan, or 16 on the fast 5090 core. **GPU FLOPs are irrelevant here** — this
   is the whole "launch-bound" finding.
2. **Filling the GPU (add lanes).** Each lane = another core dispatching on its own stream/replica. Knee rises
   ~linearly with lanes UNTIL the GPU saturates. L4 saturates at **lanes=2** (46%→~92%, knee 16); lanes=4
   regresses (SM oversubscription). L40S (3-4× compute) should take **more lanes** before saturating (sweep running).
3. **GPU-compute bound (lanes filled the GPU).** Now the GPU-active time per chunk IS the ceiling. **This is the
   regime where GPU FLOPs finally matter** — and where a *different* set of levers applies.

**The key consequence: which lever helps depends on the regime.** This resolves a lot of apparent contradictions:
- Levers that attack **launch dispatch** (lanes, CUDA-graphs, phase-alignment-to-batch) help in regimes 1–2 and
  are **redundant once GPU-bound** (regime 3).
- Levers that attack **GPU-active time** (precision, smaller compute) do **nothing** in regime 1 (GPU idle) but
  become the **only** scaling lever in regime 3.

So the optimization roadmap is: **lanes to get to regime 3 (GPU-bound), then GPU-compute levers to push the
ceiling.** We are currently at the regime-2→3 boundary on L4 (lanes=2).

## TTFS and scaling largely CONVERGE

Measured TTFS at low N is **already ~40–150 ms** (far inside the 400 ms budget). TTFS only "breaks" at
**saturation**, where the single-server queue cliffs and lag explodes to seconds. So **"improve TTFS under load"
≈ "raise the knee"** — the two asks mostly converge. The *only* genuinely TTFS-specific item is the **finalize
path** (it has its own serialization independent of steady-state). Concretely:

- **The unbatched `vad_stop` barrier-drain** (`_scheduler_drain_ready_barrier_locked`) drains a session's backlog
  **one chunk at a time (B=1)** before the fork-flush finalize. Under load this both (a) adds finalize latency
  (TTFS) and (b) caps in-phase throughput (115 not ~180). **Batching/overlapping it is the one lever that wins on
  BOTH axes** → highest-priority convergent item.
- `warm200` connect warmup contends at concurrent connect → ready-latency spikes in a connect burst (a one-time
  per-session cost; spread across lanes helps). Secondary.
- Right-context lookahead (rc1 = 160 ms) is a fixed floor — **can't shrink** (rc0 crashes upstream NeMo on this
  checkpoint, see memory). Not optimizable here.

## Ranked candidates

### Scaling (ordered by expected value, given the regime frame)
| # | lever | regime | mechanism | expected | risk | test |
|---|---|---|---|---|---|---|
| S1 | **fp16/bf16 inference** | 3 (GPU-bound) | halve GPU-active/chunk → ~2× the GPU ceiling once lanes fill it | **L4 ~16→~24–32; L40S larger** | **NOT byte-exact → WER re-validate**; launch count unchanged so cores may re-bottleneck | **cloud g6/g6e** |
| S2 | **per-GPU optimal lane count** | 2 | L4=2 (measured); L40S=? (sweep running) | sets the per-instance knee | low (flag-gated) | cloud (running) |
| S3 | **CUDA-graphs (per-B manual)** | 1–2 | cheaper per-call dispatch → fewer lanes to fill GPU / higher per-lane knee; **compounds with lanes on CPU-bound boxes, redundant once GPU-bound** | ~1.3× per-lane | medium; byte-exact (proven) | local + cloud |
| S4 | **batch the barrier-drain** | 2 | raise in-phase high-N cap 115→~180 | high-N throughput | medium | local + cloud |
| S5 | **phase-alignment (global tick)** | 2 | align arrivals → fill batches → batching finally helps; compounds with lanes | ~2× (in-phase) | +20–40 ms TTFS (in budget) | local + cloud |

### TTFS (mostly converges with scaling)
| # | lever | mechanism | expected | test |
|---|---|---|---|---|
| T1 | **batch/overlap the barrier-drain** (= S4) | remove the B=1 finalize serialization | lower TTFS under load + higher cap | local + cloud |
| T2 | spread warm200 across lanes | cut connect-burst ready latency | lower p95 ready at connect bursts | cloud |
| T3 | earlier speculative-finalize commit | shave the finalize tail | small (already fast) | local |

## The single highest-value next experiment
**fp16/bf16 encoder+decode inference on g6/g6e (S1).** Rationale: lanes have moved us to the GPU-bound regime, so
GPU-active time is now the ceiling — and precision is the cheapest ~2× on that ceiling. It stacks *multiplicatively*
with lanes (lanes parallelize dispatch; fp16 shrinks each chunk's GPU work). The catch is it breaks byte-exactness,
so it needs a **WER non-inferiority check** (not the byte gate) — but for a *new* deployment that's an acceptable,
separately-validated trade. If fp16 gives ~2× the ceiling with non-inferior WER, the L40S+lanes+fp16 stack could
be the headline per-instance density win.

## Open questions for later rounds (need data)
- g6e optimal lane count + ceiling (sweep running) → is L40S's higher compute worth the ~2× cost ($/stream)?
- Does fp16 actually ~2× the ceiling, or do the cores re-bottleneck (launch count unchanged)? → measure GPU util
  + knee at lanes×fp16.
- Do CUDA-graphs add anything *on top of* lanes once GPU-bound? (Frame predicts: no — confirm.)
- Is per-stream cache memory ever the cap before the compute ceiling? (L4 24 GB at knee ~16: almost certainly not.)
