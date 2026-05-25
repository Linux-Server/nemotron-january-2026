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

## Open questions (resolve before freezing the pin)
1. **Which NeMo loads the production checkpoint?** (2.4.1 in parakeet venv vs `>=2.6.0` in pyproject) — sets the torch ceiling.
2. **aarch64/GB10 build of 2.8.0+cu128?** (0.7).
3. **Does tch-rs bind 2.8/cu128?** (the Rust go/no-go).
