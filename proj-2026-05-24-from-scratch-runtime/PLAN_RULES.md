# PLAN_RULES — 1.4b Phase-1 T1 gate completion

## Environment
- Python (export/oracle): HF_HUB_OFFLINE=1 ./.venv/bin/python <script> from runtime/ (has nemo+torch 2.8+cu128). The venv is created by `bash setup-venv.sh` (one-time, ~5-20 min, ~11 GiB). Pin: Python 3.12.10 (.python-version), torch 2.8.0+cu128, NeMo 2.4.1. Lockfile = requirements.txt (377 entries, full transitive freeze); top-level intents = requirements.in. The venv is gitignored (.venv/).
- C++ build+run: in-container nemotron-aoti:cu128. docker run --rm --gpus all -v /home/khkramer/src/nemotron-january-2026:/work -w /work/proj-2026-05-24-from-scratch-runtime/runtime ... ; TORCH_ROOT=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))"), CUDA_ROOT=/usr/local/cuda. Build via the cmake/make pattern in cpp/CMakeLists.txt; the established build scripts are /tmp/build_session_cpp.sh etc.
- Strip-validation + any nemo-dependent step: HOST only (container lacks nemo/omegaconf).

## Oracle + bars
- Per-step oracle = finalize_ref.py (extended), the validated executable spec (token-exact vs NeMo per 1.3a) — it holds the emit/delta logic + speculative/cold reset.
- AOTI is NOT byte-exact (~1e-2 drift, F'); the bar is TOKEN-exact + (Step 1+) EVENT/DELTA-exact vs finalize_ref. Do NOT loosen token/event checks to WER except where Step 4 explicitly measures corpus WER.
- Buckets: stripped_finalize_buckets/ (drop0 T34-49 + drop2 T43-58) + manifest.json contract (fail-closed). enc_steady_aoti.pt2 = AOTI steady.

## Test protocol (per step)
1. Build the affected C++ target in-container (must compile clean).
2. Run the step's harness/session; the relevant equivalence assertion (token / event-delta / mel-hash / WER) must PASS vs finalize_ref. Report real numbers; investigate divergence rather than loosening.
3. Re-run the existing N=200 session gate (cpp/session_main) to confirm no regression.
4. Artifacts (.ts/.pt2/bundles) are gitignored; commit code + docs + logs (force-add logs under runtime/artifacts/logs/).

## Review intensity
- Steps 1, 3, 5: PAIRED adversarial review (Codex /cx-delegate + an independent Opus agent), folded to reviews/, before marking [x].
- Steps 2, 4: my (Opus) review + independent re-run.
- Honesty: if a step's full bar isn't met, mark the residual explicitly (no over-claim); correct any prior over-claim.
