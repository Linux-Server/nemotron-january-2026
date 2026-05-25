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
> **Status honesty (worksofar-FOLDED):** "✅" below = a narrow demo/token-exact result on ONE clip, NOT
> server-equivalent production parity. Gaps tracked in the A–G action list.
| 1.2b-T1 | T1 hardening (demo): full-chunk-only steady loop + StreamState(partial)+reset + metadata + range checks (`cpp/steady_main.cpp`) | demo | token-exact 1 clip + non-mult NO-CRASH (NOT semantic parity — no finalize) | R2 | 5090 |
| 1.2b-T2a | T2a — **Python `torch.export` byte-exact** (steady, 1 clip, 1 cache_len traj, first-chunk eager). **C++ runtime UNPROVEN** | partial | py-export byte-exact; C++ pending | done | 5090 |
| 1.2b-wire | **AOTI-compile the T2a .pt2 → .so** — ⚠ **PARTIAL / action-D STILL OPEN** (`0.2b-aoti-findings.md`, corrected by `reviews/codex-actionD-review.md`): established ONE result — **default AOTI FAILS the T2a byte-exact objective** (Triton fp reassoc; recurrent cache_t 1.66e-2, packaging-invariant). NOT "unreachable in any language" (no knob matrix tried). T1 = ONE smoke clip, steady-PREFIX token-exact (no finalize), NOT viability. **C++ loader NOT built → gap NOT closed.** Build emits clean noexecstack `.so` (fail-closed link shim, host-loadable). AOTI = CANDIDATE backend BLOCKED on E/F | `runtime/aot_compile.py`, `aoti_t1_check.py` | py byte-exact (FAILED) | R2 folded | container/5090 |
| 1.2b-wire-C++ | **C++ `AOTIModelPackageLoader` harness** — ✅ **DONE: action D CLOSED** (`cpp/aoti_encoder_main.cpp`): C++ `run()` output **byte-for-byte == Python `aoti_load_package`** (0.0 all 5); stream-invariant; device0 clean; no output aliasing. **Seam finding:** inputs MUST be `.contiguous()` (non-contig silently → garbage; `.contiguous()` fixes byte-exact) → carry into 1.3/1.4 | `runtime/cpp/aoti_encoder_main.cpp`, `export_t2a_io_bundle.py` | C++≡Python byte-exact + seams | done | container/5090 |
| F' | **AOTI accuracy-knob matrix** — ✅ **DONE: byte-exact NOT recoverable** (`aot_knob_matrix.py`, 5 knobs, isolated per-process). Default 1.66e-2 is best; forcing precise fp32 is ~600× WORSE (residual = matmul-accumulation-order, inherent). ⟹ compiled encoder is **T1-only**; corpus T1 gate now mandatory | `runtime/aot_knob_matrix.py` | byte-exact recovered? **NO** | done | container/5090 |
| 1.3a | **Python REFERENCE finalize** (executable spec) — ✅ **DONE** (`finalize_ref.py`, Codex pass-1 + paired-review fix): state machine (STREAMING/PENDING_FINALIZE/FINALIZED), keep_all_outputs=True remainder pass, fork/clone isolation + FORK_ASSERT, reset/resume (speculative vs cold). **TOKEN-EXACT 4/4 vs NeMo chunked stream+finalize oracle**; parent-byte-identical-after-speculative PASS; vad_start-cancel 5/5; 42-case boundary sweep PASS. 3-reviewer paired review folded (`finalize-ref-FOLDED.md`); spec §5 corrected | `runtime/finalize_ref.py`, `export_finalize_encoder.py` | token-exact vs chunked oracle | **R2 folded** | 5090 |
| 1.3b-A | **C++ finalize orchestration** (Codex pass-1) — ✅ **DONE**: `cpp/finalize_main.cpp` + `export_finalize_bundle.py`. Fork/clone (real `.clone()`), decode-continuation over the eager finalize enc_out, FORK_ASSERT (`at::equal` parent unchanged). In-container build PASS; **all rows TOKEN-EXACT vs finalize_ref AND the NeMo oracle, FORK_ASSERT PASS**. *Review note:* Phase-A FORK_ASSERT only proves clone-isolation (encoder not run in A); meaningful encoder-cache isolation needs Phase B | `runtime/cpp/finalize_main.cpp` | token-exact vs finalize_ref + oracle | R1 (Opus) | container/5090 |
| 1.3b-enc | **Finalize encoder substrate** — dynamic-T export INFEASIBLE (8× subsampling residue) AND pad-to-single-bucket NOT token-safe (1/40 flip, conv bleed up to 0.36) → **exact-T per-T buckets w/ SHARED weights**. ✅ **shared-weights mechanism PROVEN** (`validate_shared_weights.py`: constants-on-disk + `loader.load_constants(user_managed=True)` → 1.66e-2 == constants-in-so; N tiny wrapper .so share ONE 2.5GB weight set). `1.3b-finalize-encoder-findings.md` | `runtime/export_shared_weights.py`, `validate_shared_weights.py` | shared-weights run == in-so | done | container/5090 |
| 1.3b-enc-build | **Per-T finalize buckets + C++ multi-bucket loader** — export+AOTI fixed-T finalize (constants-on-disk) for T≈42-60 + drop0 outlier; C++ loads buckets, `load_constants(shared)`, routes by exact T, slices → wires finalize_main.cpp Phase B + real encoder-cache FORK_ASSERT | `runtime/aot_compile_finalize.py`, `cpp/finalize_main.cpp` | token-exact vs finalize_ref (corpus); FORK_ASSERT | **R2** | container/5090 |
| E.1 | **AOTI recurrent-drift probe** — ✅ **DONE: drift BOUNDED** (`aoti_drift_probe.py`): 830-chunk/132.8s stream, cache_t grow-ratio 0.93× (sliding-window → no compounding), tokens identical, min margin 0.004. `0.2b-aoti-findings.md` | `runtime/aoti_drift_probe.py` | drift bounded? YES | done | 5090 |
| E.2 | **Full-1000 T1 shadow (SHIP GATE)** — ✅ **DONE: AOTI WER-NEUTRAL** (`aoti_full1000_shadow.py`): 1/1000 divergence (one `easy→easier` near-tie flip, semantically trivial), trad-WER eager 3.681% vs aoti 3.685% (**delta +0.0042pp**). Compiled encoder is a validated T1 backend | `runtime/aoti_full1000_shadow.py` | WER delta ≈ 0 | done | 5090 |
| A–G | **worksofar-review action list** (realistic-decode density rerun; L40S finalize; all-fixture+state-exact decode; AOTI C++; incremental-STFT + corpus matrix + full-1000 T1; full Session state machine; on-GPU no-.item() decode) | — | de-bank the overclaims | R2 | 5090/L40S |
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
