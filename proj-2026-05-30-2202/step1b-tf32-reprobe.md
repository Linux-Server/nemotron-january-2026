# Step 1b TF32 Reprobe

Date: 2026-05-31 local RTX 5090, `.venv` torch `2.8.0+cu128`.

Probe: `proj-2026-05-24-from-scratch-runtime/runtime/step1b_tf32_reprobe.py`.
Rows: 1000 from `artifacts/session_bundle.ts`.

## Setup

- Variant A: `artifacts/enc_first_aoti_fp32.pt2`, compiled with `--no-cudnn-tf32`.
- Variant B: `artifacts/enc_first_aoti_fp32_samep.pt2`, compiled with `--no-cudnn-tf32 --force-same-precision --emulate-precision-casts`.
- Both variants used the same EP/input/shared weights as the shipped build: `enc_first_t2a.pt2`, `enc_first_t2a_io.pt`, `finalize_shared_weights.pt`.
- `TORCHINDUCTOR_MAX_AUTOTUNE=0`, `TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0`, `cuda.matmul.allow_tf32=false`, `matmul_precision=highest`.
- Manifests were kept under `artifacts/step1b_fp32/` and `artifacts/step1b_fp32_samep/` to avoid overwriting the shipped `artifacts/compile_enc_first_manifest.json`.

Compile self-checks:

| Artifact | cuDNN TF32 at compile | Extra Inductor knobs | Package SHA256 | Package size | Self-check max_abs |
|---|---:|---|---|---:|---:|
| shipped `enc_first_aoti.pt2` | true | none | `a4d2a7fc83e104b66b95cb26832fc56f2e5401d7c90ab62dfc60d806d05b996a` | 3,981,953 | `3.963470e-03` |
| variant A `enc_first_aoti_fp32.pt2` | false | none | `b55d1e8243bdeb61b246fe58d4dcf4f3c90ca4cf681419fd943deb3aeedd845d` | 4,067,927 | `3.995132e-02` |
| variant B `enc_first_aoti_fp32_samep.pt2` | false | `force_same_precision`, `emulate_precision_casts` | `51c0ddc8045cbc7dc272e891eb6863c30c7772ad8c1b74c3846aef970ef4d66e` | 4,043,579 | `3.995037e-02` |

Reference sanity:

| Reference | Compared to shipped TF32 bundle oracle | Final-token divergences | Event divergences |
|---|---|---:|---:|
| TS with `cudnn.allow_tf32=true` | bundle oracle | 0 | 0 |
| TS with `cudnn.allow_tf32=false` | bundle oracle | 0 | 9 |

The fp32 TS reference is therefore not event-identical to shipped production behavior.

## Variant Results

First-chunk max_abs columns are max over all 1000 first chunks against the named TS reference.

| Candidate first chunk | TS reference | Final-token divergences | Event divergences | enc_out max_abs | cache_t max_abs | Event-divergent utts |
|---|---|---:|---:|---:|---:|---|
| shipped AOTI baseline | TF32 TS | 0 | 4 | `6.552041e-05` | `4.957008e-02` | 198, 759, 811, 829 |
| shipped AOTI baseline | fp32 TS | 0 | 3 | `1.021177e-04` | `1.510496e-01` | 549, 770, 798 |
| variant A, fp32 cuDNN | TF32 TS | 0 | 5 | `6.552041e-05` | `6.454849e-02` | 198, 515, 770, 811, 829 |
| variant A, fp32 cuDNN | fp32 TS | 0 | 3 | `1.021177e-04` | `1.510496e-01` | 549, 770, 798 |
| variant B, fp32 cuDNN + same precision | TF32 TS | 0 | 4 | `6.552041e-05` | `4.957008e-02` | 198, 759, 811, 829 |
| variant B, fp32 cuDNN + same precision | fp32 TS | 0 | 3 | `1.021177e-04` | `1.510496e-01` | 549, 770, 798 |

Baseline reconfirmed: shipped AOTI vs shipped TF32 TS is still `0` final-token divergences and `4` event divergences.

## Known-Divergent Rows

`cache_t` first-chunk max_abs for the four Step-1 divergent utterances, against shipped TF32 TS:

| Candidate | utt 198 | utt 759 | utt 811 | utt 829 |
|---|---:|---:|---:|---:|
| shipped AOTI baseline | `8.846283e-03` | `6.115913e-03` | `8.696556e-03` | `6.747246e-03` |
| variant A, fp32 cuDNN | `6.413460e-03` | `5.849838e-03` | `8.586884e-03` | `5.506516e-03` |
| variant B, fp32 cuDNN + same precision | `8.846283e-03` | `6.115913e-03` | `8.696556e-03` | `6.747246e-03` |

Against fp32 TS:

| Candidate | utt 198 | utt 759 | utt 811 | utt 829 |
|---|---:|---:|---:|---:|
| shipped AOTI baseline | `3.316212e-02` | `1.087008e-01` | `5.421114e-02` | `2.696037e-02` |
| variant A, fp32 cuDNN | `3.316212e-02` | `1.086965e-01` | `5.421329e-02` | `2.696276e-02` |
| variant B, fp32 cuDNN + same precision | `3.316212e-02` | `1.087008e-01` | `5.421114e-02` | `2.696037e-02` |

Result: no variant moved `cache_t` toward `~1e-6`. Variant A slightly changes the four known-row TF32-reference cache deltas, but they remain `~5e-3` to `~9e-3`; variant B is effectively identical to the shipped AOTI on these rows. Against fp32 TS the cache deltas are much larger (`~2.7e-2` to `~1.09e-1`).

## Verdict

Per the requested decision logic:

- No variant hits `0` final-token and `0` event divergences vs the shipped TF32 TS reference. Best cells are shipped AOTI and variant B at `0/4`; variant A is worse at `0/5`.
- No variant hits `0/0` vs the fp32 TS reference either; all three are `0/3`.
- The fp32 TS reference itself differs from shipped TF32 TS by `0` final-token and `9` event divergences, so even a hypothetical fp32-only `0/0` result would require a global `cudnn.allow_tf32=false` runtime flip and full oracle re-baseline across finalize/steady numerics.

Conclusion: a default-on, byte/event-exact AOTI `enc_first` drop-in is not achievable with these precision knobs. The divergence is not explained away by cuDNN TF32 conv precision; it remains consistent with Inductor kernel/fusion/reassociation ordering. Keep TorchScript `enc_first` as the default and leave AOTI `enc_first` opt-in.

## Files Created

- `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/enc_first_aoti_fp32.pt2`
- `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/enc_first_aoti_fp32_samep.pt2`
- `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/step1b_fp32/enc_first_aoti_fp32.pt2`
- `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/step1b_fp32/compile_enc_first_manifest.json`
- `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/step1b_fp32_samep/enc_first_aoti_fp32_samep.pt2`
- `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/step1b_fp32_samep/compile_enc_first_manifest.json`
- `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/torchinductor_cache_enc_first_step1b_fp32/` (compile cache)
- `proj-2026-05-24-from-scratch-runtime/runtime/artifacts/torchinductor_cache_enc_first_step1b_fp32_samep/` (compile cache)
- `proj-2026-05-24-from-scratch-runtime/runtime/step1b_tf32_reprobe.py`
- `proj-2026-05-30-2202/step1b-tf32-reprobe.json`
- `proj-2026-05-30-2202/step1b-tf32-reprobe.md`

No shipped artifact was overwritten: the original `artifacts/enc_first_aoti.pt2` SHA remains `a4d2a7fc83e104b66b95cb26832fc56f2e5401d7c90ab62dfc60d806d05b996a`, and the shipped manifest path was avoided during variant compiles. No C++ runtime/server source was modified.
