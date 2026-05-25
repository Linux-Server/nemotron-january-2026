# Phase 1 decomposition — implementable steps (the plan-init for the native C++ runtime)

GO is granted (0.4-decision-FINAL). All-C++, libtorch 2.8.0+cu128, EN-only. Steps are scoped for delegation +
review; **review intensity is calibrated to risk** (not blanket paired-review-every-step), and **AWS only at phase
gates** (the 5090 covers local dev; L40S already validated the density thesis).

Legend: **REVIEW** = R0 (self-review) / R1 (Opus solo) / R2 (paired Codex+Opus adversarial). **GPU** = 5090 / L40S.

| # | Step | files | gate | REVIEW | GPU |
|---|---|---|---|---|---|
| **1.0** | State-ownership design (DONE — `1.0-state-ownership-design.md`) | — | reviewed | done | — |
| **1.1a** | **Verified Python REFERENCE decode** — ✅ **DONE 6/6 byte-exact** vs NeMo (`ref_decode.py`) | done | the real go/no-go — PASS | done | 5090 |
| 1.1b | streaming partial-hyp continuation — ✅ **DONE 18/18** (2/3/5-chunk carry == full). *Remaining minor:* max_symbols saturation fixture | done | PASS | done | 5090 |
| **1.2a** | C++ port of the decode — ✅ **DONE: BUILDS + BYTE-EXACT** (`cpp/decode_main.cpp`, loads exported `.ts`) | done | byte-exact vs gold — PASS | done | 5090 |
| 1.2a+ | C++ streaming-state-carry — ✅ **DONE** (2-chunk carry == full, byte-exact) | done | PASS | done | 5090 |
| 0.8 | native preprocessor byte-exact — ✅ **DONE** (T0 deterministic + .ts 0.000e+00) | done | PASS | done | 5090 |
| 0.2/T2a | encoder byte-exact within geometry — ✅ **DONE** (0.000e+00 all 5 outputs) | done | PASS | done | 5090 |
| 1.2b-pre | **FULL C++ pipeline** (audio→preproc→encoder→decode, non-streaming) — ✅ **DONE: BYTE-EXACT on real speech** (`cpp/pipeline_main.cpp`: 12 tok vs gold 12) | done | PASS | done | 5090 |
| 1.2b-py | **Python STREAMING pipeline** (cache-aware chunk loop + decode state-carry) — ✅ **DONE: BYTE-EXACT vs NeMo streaming** (`stream_decode.py`: "How much juice is in one lime", 11 tok steady) | done | PASS | done | 5090 |
| 1.2b-poc | C++ streaming loop POC — ✅ **logic PASS** (T1 token-exact on 320-frame clip; `cpp/steady_main.cpp`) + **paired R2 review** (`reviews/cppstream-FOLDED.md`): cache/ring/state-carry CORRECT vs server.py | done | logic PASS; gaps found | done (R2) | 5090 |
| 1.2b-T1 | **T1 hardening — ✅ DONE**: full-chunk-only steady loop (partial/short-chunk BLOCKER fixed server-faithfully; non-mult-16 NO CRASH) + per-session StreamState+reset + metadata assertions + range checks (`cpp/steady_main.cpp`) | done | T1 PASS + robustness | done (R2) | 5090 |
| 1.2b-T2a | **T2a byte-exact streaming encoder — ✅ ACHIEVED**: `torch.export` steady encoder byte-exact across cache_len (0.000e+00; trace was ~1e-5) — `export_t2a.py`/`T2a-findings.md`. Kills the near-tie-flip risk | done | byte-exact PASS | done | 5090 |
| 1.2b-wire | Wire the T2a byte-exact encoder into C++ via **AOTInductor** (.pt2→.so) — needs a glibc-compatible CUDA toolchain (**container / L40S**; local glibc 2.41 blocks AOTI nvcc). Also export the first-chunk geometry byte-exact. | `runtime/cpp/` | C++ byte-exact vs eager | R1 | container/L40S |
| 1.3 | C++ finalize path (handles the steady remainder) + STREAMING→PENDING→FINALIZED state machine + fork isolation | `runtime/cpp/finalize.*` | T1 finalize canary + reset/resume trace suite | **R2** | 5090 |
| 1.3 | C++ finalize path + the STREAMING→PENDING→FINALIZED state machine + fork isolation (FORK_ASSERT) | `runtime/cpp/finalize.*` | T1 finalize canary + reset/resume trace suite | **R2** | 5090 |
| 1.4 | Single-session end-to-end (WS ingest → steady+finalize → emit) drop-in vs Python on one stream | `runtime/cpp/session.*`, `ws.*` | T1 single-stream behavioral equivalence | R1 | 5090 |
| **GATE** | Phase-1 exit: one native stream byte/T1-equivalent to Python on the 5090 | — | T1 + T0 | **R2** | 5090 |

## Sequencing notes
- **1.1a (Python reference decode) is FIRST and the hard sub-gate.** It's cheap (no C++), locally runnable, and proves
  algorithmic understanding. If it can't hit byte-exact, STOP/reassess before any C++ — this is where a paired
  adversarial review (R2) earns its keep, not on scaffolding.
- **0.8 native preprocessor byte-exact** folds into 1.2b (it's upstream of the encoder); keep its own fixture gate.
- **0.2 encoder export fidelity (T2a)** folds into 1.2b; mechanical export already proven, byte-exact across geometries
  is the remaining check.
- **cx-delegate fit:** Codex can draft 1.1a/1.2a code and the C++ scaffolding; the BYTE-EXACT validation + the
  concurrency/state-machine correctness need Opus/human review (R2) and GPU runs I drive. The CUDA-kernel-build issue
  (glibc 2.41 local) means C++ builds happen via manual-link (graph-replay, like 0.1b) or in a CUDA devel container.
- **AWS:** none needed for Phase 1 (single-stream, 5090). L40S/EC2 returns in Phase 4 (density/tail at load) — Phase 2+.

## Phase 2+ (after the Phase-1 gate) — not decomposed yet
Multi-thread scheduler + continuous batching + admission (the density win); CUDA-graph ownership (0.11); shared-weights
(0.9, fixes the K×model-copy OOM seen on L40S); then Phase-4 multi-platform sweeps. Decompose after Phase 1 lands.
