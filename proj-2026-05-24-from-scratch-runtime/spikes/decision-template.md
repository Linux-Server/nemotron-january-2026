# Spike 0.4 — Decision memo (TEMPLATE — fill after spikes run)

This is the HARD-GATE output of Phase 0. Fill every `<…>` from measured spike results. **Label each input as Track-A
feasibility or Track-B post-plan residual.**

## Pinned versions / scope (fill once chosen)
- Language + process shape: `<all-C++ | Rust-front+C++-worker | all-Rust>` — decided by 0.1/0.2.
- libtorch / CUDA / driver / C++ ABI: `<pin exact versions>`
- Export artifact format: `<TorchScript | torch.export | …>`
- WER-CI width (T1): `<named numeric, e.g. ±X% abs WER on full-1000>`
- v1 scope: **EN 0.6b only** (multilingual/prompted = later phase).

## PRE-REGISTERED Wave-1 thresholds (REGISTER BEFORE COLLECTING 0.1/0.5 DATA)
> These are kill decisions; defining them after seeing data is invalid. Fill the numbers, freeze, THEN run.

**0.1 overlap/MPS ablation:**
- Required single-process finalize+steady overlap factor vs Python/MPS baseline: `<≥ X×>`
- Max acceptable queue/lane wait at the operating point: `<X ms>`
- Max added per-chunk latency from the new dispatch: `<X ms>`

**0.5 batching sim + graph capacity:**
- Median / p95 batch B target ("≫1" made numeric): `<median ≥ X, p95 ≥ Y>`
- Min exact-B graph replay hit-rate: `<≥ X%>`
- Max eager-fallback rate: `<≤ X%>`
- Max added wait to form a batch: `<X ms>`
- Required L4 / L40S graph-pool memory headroom at target K×lanes: `<≥ X GB free>`

**0.0 worth-it threshold:**
- Min residual p95/p99 gap vs measured Python: `<X ms>`
- Min in-budget streams/box delta vs Python's ~28/box (K=4): `<+X streams/box>`
- Value-vs-cost: residual must justify `~40–60 eng-wk + a 2nd stack`.

## DECISION TREE (fill the "Observed" column with measured outcomes)

| Branch | Observed | Decision taken |
|---|---|---|
| 0.0 residual value < threshold | `< >` | **STOP** |
| 0.1 only MPS/multi-proc overlaps (not single-process) | `< >` | likely STOP; else native-under-MPS = **tail-only**, re-run 0.0 |
| 0.3 py3.13t closes residual + free-threaded stack stable | `< >` | **choose B4** (skip native ports) |
| 0.1 positive AND 0.6a + 0.2 + 0.8 + 0.11 pass | `< >` | **proceed B1** |
| 0.6a fails byte/state equivalence (0.3 didn't win) | `< >` | STOP, or named T1-only native-decode risk sign-off |
| 0.2 fails T2a / libtorch byte-exact unattainable | `< >` | no B1a → B4 if 0.3 won, else STOP / B2-risk sign-off |
| 0.8 fails native-preproc byte-exact | `< >` | STOP, or Python-preproc as named non-v1 topology |
| 0.9 fails (can't parameterize per-call config) | `< >` | drop shared-weight density → per-lane replicas; re-run 0.0 |
| 0.5 B stays ~1 | `< >` | drop 3–5× throughput; re-run 0.0 |
| 0.5/0.11 poor hit-rate / fallback / no memory headroom | `< >` | drop steady-graph density; B1-without-graphs or re-topology; re-run 0.0 |
| 3.3 fusion unproven | `< >` | 6–10 ms finalize out of v1 scope (B1 = parity); core go/no-go unaffected |

## Final decision
`<STOP | B4 | B1 | native-under-MPS-tail-only | proceed-to-fusion>` — rationale: `<…>`
