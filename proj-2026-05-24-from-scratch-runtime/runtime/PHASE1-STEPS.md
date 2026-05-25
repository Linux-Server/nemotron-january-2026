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
| 1.2a+ | C++ streaming-state-carry (resumable decode across chunks) + max_symbols fixture | `cpp/decode.*` | matches the Python ref's 18/18 | R1 | 5090 |
| 1.2b | C++ steady path: native preprocessor (0.8) + encoder graph + decode → emit; carry cache+decoder state | `runtime/cpp/steady.*` | T1 single-stream on 5090 (WER-CI + event seq) ; T2a encoder byte-exact | **R2** | 5090 |
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
