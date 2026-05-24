# Spike 0.7 — aarch64 / DGX Spark GB10 toolchain pre-check (TEMPLATE; run BLOCKED on a GB10 box)

**Goal (PLAN §6 / 0.7):** prove a libtorch+CUDA toolchain even works on aarch64 (GB10) BEFORE any DGX Spark bring-up
(Phase 4.2). The current stack is Python+NeMo; nothing here has been built for aarch64. Gate 4.2 on this artifact.

## Minimal pre-check (do on the GB10 once the user sets it up)
1. Build minimal **libtorch + CUDA** on aarch64; confirm a trivial C++ program links and runs a CUDA kernel.
2. Load a trivial **exported module** (TorchScript/torch.export artifact from 0.2) and run it.
3. Run **one CUDA-graph capture + replay** (the graph-safe allocator path from 0.11) — capture is the riskiest op.
4. Record a **cross-arch determinism note**: does the same exported module produce byte-identical output on GB10 vs the
   5090? (Expectation: NO across arch — PLAN T2 says cross-arch byte-exact is out; GB10 relies on T1 + T2b.)

## Version matrix to record (fill on the box)
| Item | Value |
|---|---|
| CUDA toolkit | `< >` |
| NVIDIA driver | `< >` |
| libtorch version + build | `< >` |
| C++ compiler + std | `< >` |
| C++ ABI (`_GLIBCXX_USE_CXX11_ABI`) | `< >` |
| Export artifact format/version | `< >` |
| Graph capture works? | `< yes/no + notes >` |
| GB10 vs 5090 same-module byte-exact? | `< expected no — record actual >` |

## Context (roofline)
GB10 is **273 GB/s LPDDR5x — bandwidth-poor** (~9 ms finalize floor) → capacity-rich but unlikely to beat L40S on
finalize latency despite 128 GB. **Validate, don't assume.** This pre-check is cheap; the full sweeps are Phase 4.2.

## Run prerequisites — **BLOCKED**
- A DGX Spark / GB10 (aarch64) box — user sets up when ready.
