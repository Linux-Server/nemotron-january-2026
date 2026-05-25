# 0.2 prerequisite — libtorch version pin + mechanical encoder export (Track-A; partially executed)

This is the runnable front half of Spike 0.2 and the **prerequisite for 0.1b** (the gating microbench). It does the
**libtorch version pin** + the **tch-rs Rust-vs-C++ gate** + a **mechanical encoder export** (load+run, not fidelity).
0.2's *full byte-exact fidelity validation across all geometries* (T2a) stays Wave-2 — 0.1b only needs the encoder to
RUN at the right shape/cost, not to be proven correct.

## Measured local baseline (2026-05-24, no cloud)
| thing | value | implication |
|---|---|---|
| GPU (local dev) | **RTX 5090, compute_cap sm_120 (Blackwell)**, driver 580.65.06 | sm_120 ⇒ needs CUDA **12.8** |
| torch (parakeet venv — the env that runs the model) | **2.8.0+cu128** | Blackwell-ready; the fixture/export producer |
| CUDA / cuDNN | **12.8 / 9.10.2** | matches the sm_120 floor |
| export/graph APIs in 2.8.0 | `torch.export` ✓, `torch.cuda.CUDAGraph` ✓, `torch.cuda.graph_pool_handle` ✓ | the APIs 0.1b/3.1 need all exist here |
| nemo | **2.4.1** | ⚠ pyproject says `nemo_toolkit[asr]>=2.6.0` — **RESOLVE which NeMo actually loads the shipped checkpoint** and pin against THAT (it sets the torch ceiling) |

## Recommended pin (confirm, then freeze in `decision-template.md`)
**libtorch 2.8.0 + cu128** — it matches the env that will produce the golden fixtures + export (the T2a same-version
rule), is Blackwell-ready, and exposes the export + native-graph APIs. Before freezing, confirm:
- [ ] the **NeMo version that actually loads the production checkpoint** supports torch 2.8 (resolve the 2.4.1-vs-≥2.6
  discrepancy first — this is the real ceiling).
- [ ] **2.8.0+cu128 has builds for all deploy targets:** L40S (Ada) ✓ expected, L4 (Ada) ✓ expected, **GB10/Spark
  aarch64** — verify in 0.7 (the aarch64 build is the risky one).
- [ ] C++ ABI flag (`_GLIBCXX_USE_CXX11_ABI`) chosen + matched across the toolchain.

## Blackwell forces a RECENT floor → elevated tch-rs risk
sm_120 (5090) requires CUDA 12.8 ⇒ libtorch **≥2.7/cu128**. That is a *recent* floor, and the ordering invariant holds:
**we do not move the pin down to suit tch-rs.** So the Rust path is viable ONLY IF a tch-rs release binds **2.8/cu128**.
If the newest tch-rs only binds ≤2.6 (no Blackwell), **Rust is VETOED** → C++ worker or all-C++.

### tch-rs coverage gate (run this — decides Axis A)
- [ ] a `tch-rs` release **binds libtorch 2.8.0 (cu128)** — not lagging at 2.6/2.7.
- [ ] tch-rs/`cudarc` expose **CUDA-graph capture against libtorch-ALLOCATED tensors** (decisive; not raw cudarc graphs).
- [ ] per-lane stream + event control; capture-mode/graph-safe allocator reachable.
- [ ] the ATen ops the decode needs are bound (or trivially FFI-shimmable).
→ all ✓ → **all-Rust**; any ✗ (esp. the 2.8 binding) → **Rust-front+C++-worker** or **all-C++**. Record the failing box.

## Mechanical encoder export for 0.1b (make it RUN; fidelity is Wave-2/0.2)
Goal: a libtorch-loadable encoder forward at the **steady bucket shape** so 0.1b can capture+replay a realistic graph.
- Inputs (steady, B=1): `processed_signal [1,128,25]` (128 mel × `pre_encode_cache_size 9 + shift 16`), `length [1]`,
  `cache_last_channel/_time/_channel_len` from `model.encoder.get_initial_cache_state(batch_size=1)`; `drop_extra=2`,
  `keep_all_outputs=False` (steady). (Mirror `cudagraph_encoder.py` static buffers.)
- Export `encoder.cache_aware_stream_step` via `torch.export` (preferred) or TorchScript; load in libtorch C++; capture
  the CUDA graph in-harness (same pattern as `cudagraph_encoder.py`), replay from M threads.
- **Scope:** load + run + graph-capture only. Do **NOT** validate byte-exactness here (that's 0.2/T2a, Wave-2). If
  export proves fiddly, 0.1b may fall back to a representative kernel-sequence stand-in (document the approximation).

## Hand-off to 0.1b
0.1b consumes: the exported encoder module + the steady shapes above + the cost-calibrated mock decode. Run 5090 (local)
→ L40S (cloud). **Gate: ≥1.5× L40S density (≥~28/box).**

## Open questions — RESOLVED (2026-05-24)
1. **Which NeMo? → RESOLVED.** The parakeet venv runs **torch 2.8.0+cu128 + nemo 2.4.1** and `nemo.collections.asr`
   imports/works; the EN checkpoint (`models--nvidia--nemotron-speech-streaming-en-0.6b`) is **cached locally**. The
   pyproject `>=2.6.0` is a stale floor, not what runs. **Pin = torch/libtorch 2.8.0+cu128 + nemo 2.4.1** (the
   proven-working pair, also the fixture/export producer). `torch.cuda.get_arch_list()` includes **sm_120** ✓.
2. **aarch64/GB10 build? → EXISTS but MATURITY RISK.** cu128 aarch64 libtorch was added (PyTorch PR #146378; targets
   sm_90/100/**120**) but has been **nightly/flaky** (issue #157548: "haven't been built for weeks"). A *stable* 2.8.0
   aarch64+cu128 libtorch may need build-from-source. **0.7 must verify a working build on GB10; treat as a risk, not a
   given.**
3. **Does tch-rs work? → NO for the hot path (the decisive box FAILS).** tch-rs is **version-current** (latest requires
   libtorch **2.11.0**, not lagging — an older tch-rs release targets 2.8), BUT **CUDA-graph capture is an OPEN feature
   request (tch-rs issue #631), NOT implemented**; the README documents no CUDA-graph / custom-stream / allocator
   control. The decisive box — **CUDA-graph capture against libtorch-allocated tensors** — therefore **fails for tch-rs
   out of the box.** Going all-Rust would require writing+maintaining **unsafe FFI shims** to libtorch's
   `CUDAGraph`/`c10::cuda` C++ symbols (defeats the borrow-checker-safety rationale for the hot path).
   **→ Axis A preliminary verdict: C++ for the model worker** (all-C++ shape 1, or Rust-front + C++-worker shape 2 if
   the team wants Rust for networking/scheduler). **All-Rust (shape 3) is effectively VETOED.** Confirm at 0.4 with a
   hands-on docs.rs/API check (in case a newer tch-rs added partial graph support) before finalizing.

Sources: [tch-rs](https://github.com/LaurentMazare/tch-rs) · [tch-rs#631 CUDA Graphs](https://github.com/LaurentMazare/tch-rs/issues/631) · [PyTorch PR #146378 aarch64 cu128](https://github.com/pytorch/pytorch/pull/146378) · [PyTorch #157548](https://github.com/pytorch/pytorch/issues/157548)
