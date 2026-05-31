# Step 1 Probe Findings

Date: 2026-05-30 local RTX 5090, `.venv` torch `2.8.0+cu128`.

Probe script: `proj-2026-05-24-from-scratch-runtime/runtime/probe_step1.py`.

## Finding A: Weights Identity

Verdict: **PASS.** `enc_first`, inline `enc_steady`, stripped AOTI FQN sets, and `finalize_shared_weights` are all compatible with one shared encoder constants map.

Hard evidence:

| Source compared to `finalize_shared_weights.pt` | Matched | Missing | Alias/direct | Byte-equal | Max abs diff |
|---|---:|---:|---:|---:|---:|
| `finalize_shared_weights.ts` | 637/637 | 0 | 0 alias / 637 direct | 637/637 | 0 |
| `enc_first.ts` encoder parameters only | 636/636 | 0 | 636 alias / 0 direct | 636/636 | 0 |
| `enc_first.ts` parameters + buffers | 637/637 | 0 | 637 alias / 0 direct | 637/637 | 0 |
| `artifacts/enc_steady_aoti.pt2` inline constants | 637/637 | 0 | 637 alias / 0 direct | 637/637 | 0 |

Notes:

- The aliasing is exactly the C++ `constants_for_bucket` rule: `e.*` maps to `encoder.*`.
- The one shared extra when looking at **parameters only** is `encoder.pos_enc.pe`; including buffers gives full 637/637 coverage.
- `artifacts/enc_first_aoti.pt2` declares 637 constant FQNs, all present in the shared map.
- `steady_b_artifacts/enc_steady_aoti_b{1,2,4}.pt2` each declare 637 constant FQNs, all present in the shared map.
- Per-tensor evidence is in:
  - `step1-weights-enc-first-params.csv`
  - `step1-weights-enc-first-all.csv`
  - `step1-weights-enc-steady-inline.csv`
  - `step1-finalize-ts-vs-pt.csv`

Conclusion: **yes, the artifacts provably share the same encoder weights.**

## Finding B: AOTI First-Chunk Parity

Verdict: **NO-GO for default-on enc_first AOTI/unify.**

Harness:

- Reused the in-repo 1000-row `artifacts/session_bundle.ts` token/event oracle generated with shipped `enc_first.ts`.
- The probe swaps only first-chunk execution to `artifacts/enc_first_aoti.pt2`, binds all 637 constants from `finalize_shared_weights.pt`, then runs the full mel-bundle decode/event path with inline AOTI steady chunks and eager finalize logic from `RecordingContinuousFinalizeRef`.
- TS sanity on rows 0-3 passed: 0 token divergences, 0 event divergences.

Results:

| Scope | Rows | Final token divergences | Event divergences | First-output max abs vs bundle eager |
|---|---:|---:|---:|---:|
| b2-t1 subset rows 0-3 | 4 | 0 | 0 | `5.498528e-06` |
| 1000-row corpus | 1000 | 0 | 4 | `6.601214e-05` |

Event divergences:

| utt | sample_id | first event index | Difference | First-chunk TS-vs-AOTI max abs |
|---:|---|---:|---|---|
| 198 | `e018a533-4638-c75e-ef13-395ea8e2eb7b` | 6 | interim had extra partial word: got `"Could you translate the phrase where is"` vs gold `"Could you translate the phrase where"` | enc_out `5.54e-06`, cache_t `5.64e-03` |
| 759 | `5dd8def8-9ca0-aabf-8a22-e919d4945632` | 23 | interim dropped partial suffix: got ending `"box we"` vs gold ending `"box we del"` | enc_out `2.46e-06`, cache_t `6.12e-03` |
| 811 | `0a4b7986-46b9-b9da-9c61-3128e502ff63` | 11 | interim changed `"but I"` to `"but I've"`; event counts 36 vs 37 | enc_out `2.00e-06`, cache_t `8.70e-03` |
| 829 | `81717bc0-1ac9-f72e-1c41-da7db7b0dcb8` | 13 | interim dropped partial suffix: got ending `"an hour's"` vs gold ending `"an hour's dri"` | enc_out `1.42e-06`, cache_t `6.75e-03` |

Full parity JSON: `step1-enc-first-parity.json`. First-chunk numeric diffs for the four divergent rows: `step1-divergent-first-diffs.json`.

Conclusion: final tokens are stable, but the hard gate requires **0 token and 0 event divergences**. Because event divergences are `4/1000`, Steps 4-5 must keep TorchScript `enc_first` as default. AOTI first-chunk can only be opt-in/experimental unless a later export removes these interim-event differences.

## Finding C: Extraction-Cache API

Verdict: **NO-GO for a speed-motivated extract-once cache; GO only for `/tmp` hygiene.**

Hard evidence:

- Installed header: `.venv/lib/python3.12/site-packages/torch/include/torch/csrc/inductor/aoti_package/model_package_loader.h`
  - Public constructor surface is only `AOTIModelPackageLoader(const std::string& model_package_path, ...)`.
  - `temp_dir_` is private.
  - No public constructor accepting a pre-extracted directory.
- PyTorch v2.8.0 source fetched to `step1-torch-v2.8.0-model_package_loader.cpp`.
  - `create_temp_dir()` uses `std::string temp_dir = "/tmp/XXXXXX";`
  - It calls `mkdtemp(temp_dir.data())`.
  - No `TMPDIR` branch appears in the implementation.
- Installed `libtorch_cpu.so` strings include `/tmp/XXXXXX`, `aotinductor`, and `/pytorch/torch/csrc/inductor/aoti_package/model_package_loader.cpp`.

Post-unify residual sizing:

| Residual item | Size |
|---|---:|
| One shared weights JIT blob: `finalize_shared_weights.ts` | 2,477,725,779 bytes / 2,362.94 MiB |
| Stripped `enc_first_aoti.pt2` | 3,981,953 bytes / 3.80 MiB |
| Stripped steady-b packages total | 12,856,506 bytes / 12.26 MiB |
| Stripped finalize buckets total | 130,282,535 bytes / 124.25 MiB |
| All small stripped AOTI packages total | 147,120,994 bytes / 140.31 MiB |

Conclusion: after unify, the large remaining cold read is the single shared JIT weights blob, not AOTI extraction. A SHA-keyed AOTI extract-once cache would require a custom/torch-patched loader and is not worth building for speed. Step 8 should collapse to a startup cleanup guard for stale owned `/tmp/*/data/aotinductor*` trees.

## Dependent-Step Verdicts

- **Steps 4-5 (`enc_first` unify/default AOTI): NO-GO as default.** Weight identity passes, but the AOTI first-chunk event gate fails with 4/1000 event divergences. Keep TS default; AOTI first-chunk is opt-in only if implemented.
- **Steps 7-8 extraction cache: NO-GO for speed cache; GO for `/tmp` hygiene guard.** No public pre-extracted-dir API exists, and post-unify extraction is only ~140 MiB of stripped AOTI packages.

## Blockers

- The only correctness blocker found is AOTI first-chunk interim-event drift. There were no missing shared weights and no cache API ambiguity.
- No runtime/server/C++ source was modified.
