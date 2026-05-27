# L40S W3 Density Sweep Prep

Run `run_l40s_density.sh` on the g6e/L40S box after rsyncing only the code subset below to `~/density/`.
The script is an on-box workflow: it installs/checks the DL-AMI build environment, downloads artifacts from
`s3://nemotron-phase2-eps-419599258555/density/`, verifies `eps_manifest.json`, compiles native sm_89 AOTI packages
with autotune off, builds `cpp/density_main`, and runs the W3 sweep fresh-process-per-N.

It does not launch an instance. When run on the g6e, it does use `aws s3 cp` to fetch the artifacts.

## Run Command

```bash
cd ~/density
./run_l40s_density.sh
```

Default N sweep:

```text
1,8,16,24,32,40,48,64,80
```

Useful overrides:

```bash
DENSITY_N_VALUES=1,8,16,24,32,40,48,64,80 ./run_l40s_density.sh
FORCE_S3_DOWNLOAD=1 ./run_l40s_density.sh
DENSITY_TREAT_NO_PASS_AS_FAILURE=1 ./run_l40s_density.sh
```

## Script Flow

1. Installs/checks DL-AMI dependencies: `build-essential`, `cmake`, `ninja-build`, `python3-dev`, `python3-venv`,
   `awscli`, `curl`, and certs.
2. Locates system CUDA and `nvcc` under `/usr/local/cuda*`.
3. Creates a `torch==2.8.0` venv with `uv` when available, otherwise `venv` plus `pip`.
4. Confirms the box is x86_64 and `torch.cuda.get_device_capability()==(8, 9)`.
5. Ports the bare-AMI AOTI fixes: `python3-dev`, unversioned `libcuda.so` plus CUDA stub symlink for Triton link
   probes, and fail-closed `-Wl,-z,noexecstack` injection during AOTI shared-library links.
6. Downloads S3 artifacts into `artifacts_sm89/` and verifies SHA256s against `eps_manifest.json`.
7. Compiles `enc_steady_t2a.pt2` to `artifacts_sm89/enc_steady_aoti.pt2`, autotune off.
8. Compiles the 32 finalize bucket EPs with `aot_compile_buckets.py`, runs per-bucket self-checks at
   `SELF_CHECK_ATOL=0.1`, strips weights with `strip_bucket_weights.py`, and writes
   `artifacts_sm89/stripped_finalize_buckets/manifest.json`.
9. Builds the current shared+locked `enc_first.ts`, explicit-stream, `num_runners=N`, capped-finalize-pool
   `density_main`.
10. Runs each N in a fresh process and prints `L40S_DENSITY_ROW` plus `L40S_DENSITY_RESULT`.

## Rsync From This Box

Rsync these paths under `runtime/` to `~/density/` on the g6e:

```bash
rsync -av \
  runtime/run_l40s_density.sh \
  runtime/aot_compile.py \
  runtime/aot_compile_buckets.py \
  runtime/strip_bucket_weights.py \
  g6e:~/density/

rsync -av \
  runtime/cpp/CMakeLists.txt \
  runtime/cpp/density_main.cpp \
  runtime/cpp/session_main.cpp \
  g6e:~/density/cpp/
```

| Path under `runtime/` | Bytes | Size |
|---|---:|---:|
| `run_l40s_density.sh` | 32,468 | 0.032 MB |
| `aot_compile.py` | 3,960 | 0.004 MB |
| `aot_compile_buckets.py` | 14,907 | 0.015 MB |
| `strip_bucket_weights.py` | 24,930 | 0.025 MB |
| `cpp/density_main.cpp` | 178,074 | 0.178 MB |
| `cpp/session_main.cpp` | 209,192 | 0.209 MB |
| `cpp/CMakeLists.txt` | 4,831 | 0.005 MB |

Rsync subtotal: **468,362 bytes = 0.468 MB**.

## Download From S3

Bucket prefix:

```text
s3://nemotron-phase2-eps-419599258555/density/
```

The script downloads these keys into `artifacts_sm89/`. Sizes below are from the current local validated artifacts.
The on-box script fail-closes on SHA mismatches for the W3 required artifacts in `eps_manifest.json`; the small
helper artifacts are SHA-verified when the manifest lists them.

| S3 key under `density/` | Bytes | Size |
|---|---:|---:|
| `eps_manifest.json` | manifest | 0.009 MB expected |
| `buckets_manifest.json` | 2,435 | 0.002 MB |
| `enc_steady_t2a.pt2` | 2,490,209,376 | 2,490.209 MB |
| `session_bundle.ts` | 667,603,966 | 667.604 MB |
| `finalize_shared_weights.pt` | 2,477,736,629 | 2,477.737 MB |
| `finalize_shared_weights.ts` | 2,477,725,779 | 2,477.726 MB |
| `enc_first.ts` | 2,478,955,502 | 2,478.956 MB |
| `t2a_io.pt` | 15,359,925 | 15.360 MB |
| `joint_step.ts` | 6,909,312 | 6.909 MB |
| `predict_step.ts` | 28,890,948 | 28.891 MB |
| `finalize_buckets/enc_finalize_d0_T34_ep.pt2` | 2,490,205,928 | 2,490.206 MB |
| `finalize_buckets/enc_finalize_d0_T35_ep.pt2` | 2,490,206,440 | 2,490.206 MB |
| `finalize_buckets/enc_finalize_d0_T36_ep.pt2` | 2,490,206,952 | 2,490.207 MB |
| `finalize_buckets/enc_finalize_d0_T37_ep.pt2` | 2,490,207,464 | 2,490.207 MB |
| `finalize_buckets/enc_finalize_d0_T38_ep.pt2` | 2,490,207,976 | 2,490.208 MB |
| `finalize_buckets/enc_finalize_d0_T39_ep.pt2` | 2,490,208,488 | 2,490.208 MB |
| `finalize_buckets/enc_finalize_d0_T40_ep.pt2` | 2,490,209,000 | 2,490.209 MB |
| `finalize_buckets/enc_finalize_d0_T41_ep.pt2` | 2,490,209,512 | 2,490.210 MB |
| `finalize_buckets/enc_finalize_d0_T42_ep.pt2` | 2,490,210,280 | 2,490.210 MB |
| `finalize_buckets/enc_finalize_d0_T43_ep.pt2` | 2,490,210,792 | 2,490.211 MB |
| `finalize_buckets/enc_finalize_d0_T44_ep.pt2` | 2,490,211,304 | 2,490.211 MB |
| `finalize_buckets/enc_finalize_d0_T45_ep.pt2` | 2,490,211,816 | 2,490.212 MB |
| `finalize_buckets/enc_finalize_d0_T46_ep.pt2` | 2,490,212,328 | 2,490.212 MB |
| `finalize_buckets/enc_finalize_d0_T47_ep.pt2` | 2,490,212,840 | 2,490.213 MB |
| `finalize_buckets/enc_finalize_d0_T48_ep.pt2` | 2,490,213,352 | 2,490.213 MB |
| `finalize_buckets/enc_finalize_d0_T49_ep.pt2` | 2,490,213,864 | 2,490.214 MB |
| `finalize_buckets/enc_finalize_d2_T43_ep.pt2` | 2,490,216,104 | 2,490.216 MB |
| `finalize_buckets/enc_finalize_d2_T44_ep.pt2` | 2,490,216,616 | 2,490.217 MB |
| `finalize_buckets/enc_finalize_d2_T45_ep.pt2` | 2,490,217,128 | 2,490.217 MB |
| `finalize_buckets/enc_finalize_d2_T46_ep.pt2` | 2,490,217,640 | 2,490.218 MB |
| `finalize_buckets/enc_finalize_d2_T47_ep.pt2` | 2,490,218,152 | 2,490.218 MB |
| `finalize_buckets/enc_finalize_d2_T48_ep.pt2` | 2,490,218,664 | 2,490.219 MB |
| `finalize_buckets/enc_finalize_d2_T49_ep.pt2` | 2,490,219,176 | 2,490.219 MB |
| `finalize_buckets/enc_finalize_d2_T50_ep.pt2` | 2,490,219,688 | 2,490.220 MB |
| `finalize_buckets/enc_finalize_d2_T51_ep.pt2` | 2,490,220,200 | 2,490.220 MB |
| `finalize_buckets/enc_finalize_d2_T52_ep.pt2` | 2,490,220,712 | 2,490.221 MB |
| `finalize_buckets/enc_finalize_d2_T53_ep.pt2` | 2,490,221,224 | 2,490.221 MB |
| `finalize_buckets/enc_finalize_d2_T54_ep.pt2` | 2,490,221,736 | 2,490.222 MB |
| `finalize_buckets/enc_finalize_d2_T55_ep.pt2` | 2,490,222,248 | 2,490.222 MB |
| `finalize_buckets/enc_finalize_d2_T56_ep.pt2` | 2,490,222,760 | 2,490.223 MB |
| `finalize_buckets/enc_finalize_d2_T57_ep.pt2` | 2,490,223,272 | 2,490.223 MB |
| `finalize_buckets/enc_finalize_d2_T58_ep.pt2` | 2,490,224,040 | 2,490.224 MB |

S3 subtotal excluding `eps_manifest.json`: **90,330,271,568 bytes = 90.330 GB = 84.127 GiB**.

`t2a_io.pt`, `joint_step.ts`, and `predict_step.ts` are small runtime/helper artifacts required by the current
compile self-check, strip path, and `density_main`, even though the W3 gate artifacts are the EPs, bundle, shared
weights, bucket manifest, and `enc_first.ts`.

## Bare-AMI Risks

- `eps_manifest.json` must include SHA entries for the W3 required artifacts, including `enc_first.ts`; otherwise
  the script fail-closes before compile.
- The current `density_main` requires `joint_step.ts` and `predict_step.ts` at runtime. If those were not uploaded
  to S3 with the current artifact set, upload them or place them under `artifacts_sm89/` before running.
- The AOTI link path needs a visible unversioned `libcuda.so`. The script creates both the driver-lib symlink and
  CUDA stub symlink, then adds them to `LIBRARY_PATH`.
- Autotune is intentionally off. Do not set max-autotune env vars for this W3 run; the script asserts Inductor
  `max_autotune` and `coordinate_descent_tuning` are disabled.
