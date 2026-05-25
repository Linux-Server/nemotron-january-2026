# 0.1b microbench — build-ready harness (the GATING experiment)

Measures whether **GIL-free multi-thread intake** reclaims the 40–65%-idle L40S GPU above the ~16–20/box single-thread
ceiling. **GATE: ≥1.5× → ≥~28/box ⇒ GO** to fund the Wave-2 native ports; <1.5× ⇒ STOP. Spec: `../0.1b-microbench-spec.md`.

## Steps
1. **Export** (env with torch+nemo — parakeet venv): `python export_encoder.py --out ./artifacts`
   → `artifacts/encoder_steady_b1.ts` + `shapes.json`. Mechanical (load+run), NOT fidelity (that's 0.2).
2. **Get libtorch 2.8.0+cu128 C++ SDK** (cxx11-abi): download.pytorch.org/libtorch/cu128/...2.8.0+cu128.zip
3. **Build**: `mkdir build && cd build && cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPE=Release .. && make -j`
4. **Run the A/B + sweep** on the 5090 (local), then L40S (one EC2 run):
   - baseline: `./microbench --intake-threads 1 --lanes 3 --streams N` (sweep N → the single-thread knee; must
     reproduce ~16–20/box-equivalent to validate calibration)
   - thesis:   `./microbench --intake-threads $(nproc) --lanes 3 --streams N` (sweep N → the multi-thread knee)
   - report knee(M=cores)/knee(M=1) + absolute streams/box at the SLO + GPU util.

## STATUS (honest)
- `export_encoder.py` — runnable now (model cached). **Risk:** TorchScript-tracing the streaming step may warn/fail on
  the `drop_extra` global mutation + cache control flow; a failure is a real 0.2 datapoint (fall back to a
  kernel-sequence stand-in for 0.1b).
- `microbench.cpp` — **structurally complete for the core mechanic** (per-lane stream + captured graph + replay from
  M intake threads), but has marked TODOs: argv parsing, reading `shapes.json` for the proto tensors, NVML GPU-util
  sampling, percentile reporting, the finalize path, CUDA-event completion (vs `stream.synchronize`), and the
  MPS/multi-proc variants. **Numbers are not trustworthy until (a) it builds and (b) the mock-decode cost is calibrated
  against the Python `model_wall` (≤35 ms) + decode host time, and the M=1 baseline reproduces ~16–20/box.**
- **The build (libtorch C++ SDK) + GPU run is the hands-on engineering step; the L40S confirm needs EC2.** The 5090
  read is local.

## Calibration is load-bearing
If the mock decode is too cheap, the native ceiling is overestimated. Calibrate `--decode-gpu-us/--decode-host-us` so
the **M=1 baseline reproduces the measured ~16–20/box L40S knee** before trusting the M=cores number. Then rerun with
decode cost ±50% for sensitivity (per the spec's faithfulness caveat).
