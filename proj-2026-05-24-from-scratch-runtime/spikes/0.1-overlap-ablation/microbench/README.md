# 0.1b microbench — build-ready harness (the GATING experiment)

Measures whether **GIL-free multi-thread intake** reclaims the 40–65%-idle L40S GPU above the ~16–20/box single-thread
ceiling. **GATE: ≥1.5× → ≥~28/box ⇒ GO** to fund the Wave-2 native ports; <1.5× ⇒ STOP. Spec: `../0.1b-microbench-spec.md`.

## Steps
1. **Export** (env with torch+nemo — parakeet venv): `python export_encoder.py --out ./artifacts`
   → `artifacts/encoder_steady_b1.ts` + `shapes.json`. Mechanical (load+run), NOT fidelity (that's 0.2).
2. **Get libtorch 2.8.0+cu128 C++ SDK** (cxx11-abi): download.pytorch.org/libtorch/cu128/...2.8.0+cu128.zip
3. **Build**: `mkdir build && cd build && cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release .. && make -j`
4. **Run the A/B + sweep** on the 5090 (local), then L40S (one EC2 run). **lanes = dispatcher threads (each owns one
   lane); A/B is lanes=1 vs lanes=nproc:**
   - baseline: `./microbench --lanes 1 --streams N` (sweep N → the single-thread-intake knee; must reproduce
     ~16–20/box-equivalent to validate calibration)
   - thesis:   `./microbench --lanes $(nproc) --streams N` (sweep N → the multi-thread-intake knee)
   - report knee(lanes=cores)/knee(lanes=1) + absolute streams/box at the SLO + GPU util (printed via NVML).

## STATUS (honest)
- `export_encoder.py` — **runnable, EXECUTED 2026-05-24: SUCCESS** (TorchScript trace clean; artifact produced; shapes
  confirmed — see `../../0.2-pin-and-export/README.md`).
- `microbench.cpp` — **complete + compile-targeted**: argv parsing, proto tensors at the confirmed shapes (zeros — a
  throughput bench, values irrelevant), per-lane stream + captured graph, **one dispatcher thread per lane (no
  cross-thread graph sharing → safe)**, mock decode (GPU dummy-GEMM iters + host-µs stall), NVML GPU-util sampling,
  p50/p95/p99 reporting. **Remaining build-time work:** verify the libtorch-2.8 C++ API signatures (`CUDAGraph::
  capture_begin` pool arg, `getStreamFromPool`, nvml link); optional TODOs left in-code (periodic finalize path,
  CUDA-event completion vs `stream.synchronize`, MPS/multi-proc variants).
- **Trust gate:** numbers are valid only after (a) it builds and (b) `--decode-host-us`/`--decode-gpu-iters` are
  calibrated so the **lanes=1 baseline reproduces the measured ~16–20/box**; then rerun decode-cost ±50% (sensitivity).
- **The build (libtorch C++ SDK) + GPU run is the hands-on step; the L40S confirm needs EC2.** The 5090 read is local.

## Build-safety caveat
The encoder `forward` must be CUDA-graph-safe (no host syncs / no new allocations post-warmup) for `capture()` to
succeed — NeMo's own `cudagraph_encoder.py` does exactly this, so it should hold, but a capture failure at build is a
real datapoint (would force the kernel-sequence stand-in fallback).

## Calibration is load-bearing
If the mock decode is too cheap, the native ceiling is overestimated. Calibrate `--decode-gpu-us/--decode-host-us` so
the **M=1 baseline reproduces the measured ~16–20/box L40S knee** before trusting the M=cores number. Then rerun with
decode cost ±50% for sensitivity (per the spec's faithfulness caveat).
