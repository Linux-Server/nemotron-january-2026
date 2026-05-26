# L40S Density Remote Prep

Run `run_l40s_density.sh` on the g6e.8xlarge after rsyncing this runtime subset to `~/density/`.
It creates `~/density/artifacts_sm89/`, symlinks read-only source artifacts from `artifacts/`, compiles native
sm_89 AOTI packages there, builds `cpp/density_main`, then runs:

```bash
./cpp/build_l40s_density/density_main --mode density-sweep --n-values 1,2,4,8,16,24 ./artifacts_sm89
```

Useful overrides:

```bash
SRC_ARTIFACTS=./artifacts ART_SM89=./artifacts_sm89 DENSITY_N_VALUES=1,2,4,8,16,24 ./run_l40s_density.sh
KEEP_UNSTRIPPED_BUCKETS=1 ./run_l40s_density.sh
DENSITY_TREAT_NO_PASS_AS_FAILURE=1 ./run_l40s_density.sh
```

## Transfer Manifest

Required existing files verified locally:

| Path under `runtime/` | Bytes | Size |
|---|---:|---:|
| `run_l40s_density.sh` | 16,023 | 0.016 MB |
| `aot_compile.py` | 3,960 | 0.004 MB |
| `aot_compile_buckets.py` | 14,907 | 0.015 MB |
| `strip_bucket_weights.py` | 24,930 | 0.025 MB |
| `cpp/CMakeLists.txt` | 4,831 | 0.005 MB |
| `cpp/density_main.cpp` | 155,490 | 0.155 MB |
| `cpp/session_main.cpp` | 209,192 | 0.209 MB |
| `artifacts/enc_steady_t2a.pt2` | 2,490,209,376 | 2,490.209 MB |
| `artifacts/t2a_io.pt` | 15,359,925 | 15.360 MB |
| `artifacts/finalize_buckets/buckets_manifest.json` | 28,412 | 0.028 MB |
| `artifacts/stripped_finalize_buckets/manifest.json` | 39,262 | 0.039 MB |
| `artifacts/session_bundle.ts` | 667,603,966 | 667.604 MB |
| `artifacts/finalize_shared_weights.pt` | 2,477,736,629 | 2,477.737 MB |
| `artifacts/finalize_shared_weights.ts` | 2,477,725,779 | 2,477.726 MB |
| `artifacts/enc_first.ts` | 2,478,955,502 | 2,478.956 MB |
| `artifacts/joint_step.ts` | 6,909,312 | 6.909 MB |
| `artifacts/predict_step.ts` | 28,890,948 | 28.891 MB |

Verified existing subtotal: **10,643,888,444 bytes = 10.644 GB = 9.913 GiB**.

Do not transfer the old compiled `artifacts/enc_steady_aoti.pt2` or old `artifacts/stripped_finalize_buckets/*.pt2`
for this workflow. The L40S script rebuilds native sm_89 packages into `artifacts_sm89/`.

Required finalize bucket ExportedPrograms are **missing locally** and must be restored or regenerated before rsync.
The runtime stripped-bucket manifest expects these 32 files:

```text
artifacts/finalize_buckets/enc_finalize_d0_T34_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T35_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T36_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T37_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T38_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T39_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T40_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T41_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T42_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T43_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T44_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T45_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T46_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T47_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T48_ep.pt2
artifacts/finalize_buckets/enc_finalize_d0_T49_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T43_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T44_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T45_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T46_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T47_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T48_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T49_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T50_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T51_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T52_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T53_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T54_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T55_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T56_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T57_ep.pt2
artifacts/finalize_buckets/enc_finalize_d2_T58_ep.pt2
```

Because those 32 files are absent in this checkout, their sizes cannot be summed here. The script fail-closes if
any are missing on the L40S.

Local note: `artifacts/finalize_buckets/buckets_manifest.json` currently exists but lists only 15 drop0 buckets.
The deploy contract in `artifacts/stripped_finalize_buckets/manifest.json` lists 32 buckets. The L40S script uses
the deploy contract for the required key set and lets `aot_compile_buckets.py` discover the EP files, so an
incomplete `buckets_manifest.json` is not fatal if all 32 EP files are present.

## Container Requirement

The native AOTI compile itself should run on the Ubuntu 22.04 DL AMI with pip `torch==2.8.0` plus system CUDA/NVCC;
the `nemotron-aoti` container is not required for this step.

Evidence:

- `aot_compile.py` imports only `os`, `torch`, and `torch._inductor.cpp_builder`.
- `aot_compile_buckets.py` imports stdlib plus `torch`, loads saved ExportedPrograms, calls
  `torch._inductor.aoti_compile_and_package`, then self-checks with `aoti_load_package` and shared weights.
- No Nemo/OmegaConf import is on the AOTI compile path.

`strip_bucket_weights.py --all` is different: its full validation path can import `finalize_ref`, `ref_decode`, and
`build_bucket_manifest`, and `build_bucket_manifest.py` derives the contract through `finalize_ref`/Nemo. The L40S
script therefore uses `strip_bucket_weights.py --strip-only` per compiled package and rebuilds the deploy manifest
from the transferred stripped-bucket contract manifest. Full strip validation remains a host/Nemo-capable step.

## Current Blocker

The only prep blocker found locally is the missing finalize bucket ExportedPrograms listed above. Without them the
L40S can compile `enc_steady_t2a.pt2`, but cannot rebuild native sm_89 finalize bucket packages.
