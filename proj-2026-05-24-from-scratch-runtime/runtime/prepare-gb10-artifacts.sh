#!/usr/bin/env bash
# Prepare GB10/aarch64 runtime artifacts without changing the legacy L40S/x86
# recipe. This is intentionally a proof-of-concept path for this platform.
#
# Expected use, inside the GB10 container:
#   ./prepare-gb10-artifacts.sh setup-export-env
#   ./prepare-gb10-artifacts.sh export-sources
#   ./prepare-gb10-artifacts.sh compile-steady
#   ./prepare-gb10-artifacts.sh compile-steady-batches
#   ./prepare-gb10-artifacts.sh export-session-bundle
#   ./prepare-gb10-artifacts.sh compile-buckets
#   ./prepare-gb10-artifacts.sh export-audio-bundle
#   ./prepare-gb10-artifacts.sh check
#
# Or from the repo root:
#   proj-2026-05-24-from-scratch-runtime/runtime/container/build-gb10-aarch64.sh \
#     bash -lc './prepare-gb10-artifacts.sh check-env'
set -euo pipefail
IFS=$'\n\t'

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd -P)"

ART_GB10="${ART_GB10:-$ROOT/artifacts_gb10}"
STEADY_B_GB10="${STEADY_B_GB10:-$ROOT/steady_b_artifacts_gb10}"
EXPORT_VENV="${EXPORT_VENV:-$ROOT/.venv-gb10-export}"
SELF_CHECK_ATOL="${SELF_CHECK_ATOL:-0.2}"
TORCH_CUDA_ARCH_LIST="${NEMOTRON_GB10_TORCH_CUDA_ARCH_LIST:-12.0}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$ART_GB10/torchinductor_cache}"
export TORCH_CUDA_ARCH_LIST TORCHINDUCTOR_CACHE_DIR
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=0
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=0
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=0

log() {
  printf '[gb10-artifacts %(%H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  printf '[gb10-artifacts ERROR] %s\n' "$*" >&2
  exit 2
}

need_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

check_env() {
  command -v python3 >/dev/null 2>&1 || die "python3 not found"
  python3 - <<'PY'
import platform
import torch
import torch._inductor.config as cfg

print("machine", platform.machine())
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda is not available")
print("device", torch.cuda.get_device_name(0))
print("cc", torch.cuda.get_device_capability(0))
print("arch_list", torch.cuda.get_arch_list())
if platform.machine() not in ("aarch64", "arm64"):
    raise SystemExit(f"expected ARM/aarch64 GB10 host, got {platform.machine()}")
if torch.cuda.get_device_capability(0)[0] < 12:
    raise SystemExit("expected a Blackwell/SM12x GPU for this GB10 artifact path")
if not hasattr(torch._inductor, "aoti_compile_and_package"):
    raise SystemExit("torch._inductor.aoti_compile_and_package is missing")
for name in ("max_autotune", "max_autotune_gemm", "max_autotune_pointwise", "coordinate_descent_tuning"):
    if hasattr(cfg, name) and bool(getattr(cfg, name)):
        raise SystemExit(f"Inductor autotune unexpectedly enabled: {name}")
PY
}

export_python() {
  if [[ -x "$EXPORT_VENV/bin/python" ]]; then
    printf '%s\n' "$EXPORT_VENV/bin/python"
  else
    printf '%s\n' "python3"
  fi
}

setup_export_env() {
  log "creating/updating local NeMo export venv: $EXPORT_VENV"
  python3 -m venv --system-site-packages "$EXPORT_VENV"
  "$EXPORT_VENV/bin/python" -m pip install --upgrade pip
  "$EXPORT_VENV/bin/python" -m pip install "nemo_toolkit[asr]==2.4.1"
  "$EXPORT_VENV/bin/python" - <<'PY'
import nemo
import nemo.collections.asr as _asr
import torch
print("nemo", nemo.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
}

export_sources() {
  local py
  py="$(export_python)"
  mkdir -p "$ART_GB10" "$ART_GB10/finalize_buckets" "$STEADY_B_GB10"
  log "exporting portable source artifacts locally into $ART_GB10"
  "$py" "$ROOT/export_stream_encoder.py" --out "$ART_GB10"
  "$py" "$ROOT/export_decode.py" --out "$ART_GB10"
  "$py" "$ROOT/export_shared_weights.py" --out "$ART_GB10"
  "$py" "$ROOT/export_t2a.py" --out "$ART_GB10"
  "$py" "$ROOT/export_finalize_encoder.py" --out "$ART_GB10"
  "$py" "$ROOT/export_finalize_buckets.py" \
    --out "$ART_GB10/finalize_buckets" \
    --fixture "$ART_GB10/finalize_fixture.pt"
  "$py" "$ROOT/build_range_examples.py" \
    --out "$ART_GB10/finalize_buckets" \
    --scan "${FINALIZE_RANGE_SCAN:-400}"
  "$py" "$ROOT/build_drop0_buckets.py" \
    --out "$ART_GB10/finalize_buckets"
  "$py" "$ROOT/export_steady_batched.py" \
    --out "$STEADY_B_GB10" \
    --batches 1,2,4 \
    --export-only \
    --shared-weights "$ART_GB10/finalize_shared_weights.ts" \
    --production-b1 "$ART_GB10/enc_steady_aoti.pt2"
}

bucket_eps_from_manifest() {
  python3 - "$ART_GB10/finalize_buckets" <<'PY'
import json
import sys
from pathlib import Path

bucket_dir = Path(sys.argv[1])
names = []
for manifest in (bucket_dir / "buckets_manifest.json", bucket_dir.parent / "buckets_manifest.json"):
    if not manifest.exists():
        continue
    data = json.loads(manifest.read_text())
    entries = data.get("buckets") or data.get("contract") or data.get("files") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        continue
    for item in entries:
        if not isinstance(item, dict):
            continue
        ep = item.get("ep")
        if ep is None and "drop" in item and "T" in item:
            ep = f"enc_finalize_d{int(item['drop'])}_T{int(item['T'])}_ep.pt2"
        if isinstance(ep, str) and ep.endswith("_ep.pt2"):
            names.append(ep)
names.extend(path.name for path in bucket_dir.glob("enc_finalize_d*_T*_ep.pt2"))
names = sorted(set(names))
if not names:
    raise SystemExit(f"no finalize bucket EPs found in {bucket_dir}")
for name in names:
    print(name)
PY
}

compile_steady() {
  need_file "$ART_GB10/enc_steady_t2a.pt2"
  need_file "$ART_GB10/t2a_io.pt"
  log "compiling steady encoder for GB10: enc_steady_t2a.pt2 -> enc_steady_aoti.pt2"
  python3 - "$ART_GB10" <<'PY'
import os
import sys
import torch

art = sys.argv[1]

def force_noexecstack_on_link():
    import torch._inductor.cpp_builder as cb
    orig = cb.CppBuilder.get_command_line
    seen = {"flagged": False}
    def patched(self):
        cmd = orig(self)
        if getattr(self, "_do_link", False) and "-shared" in cmd:
            if "-Wl,-z,noexecstack" not in cmd:
                cmd += " -Wl,-z,noexecstack"
            seen["flagged"] = True
            print("[noexecstack] injected into shared-lib link:", cmd[-160:], flush=True)
        return cmd
    cb.CppBuilder.get_command_line = patched
    return seen

print("torch", torch.__version__, "cc", torch.cuda.get_device_capability())
seen = force_noexecstack_on_link()
ep = torch.export.load(os.path.join(art, "enc_steady_t2a.pt2"))
pkg = os.path.join(art, "enc_steady_aoti.pt2")
out_path = torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
if not seen["flagged"]:
    raise SystemExit("noexecstack shim never fired on a shared-lib link")
runner = torch._inductor.aoti_load_package(out_path)
io = torch.load(os.path.join(art, "t2a_io.pt"), weights_only=False)
inputs = [io["chunk"].cuda(), io["L"].cuda(), io["clc"].cuda(), io["clt"].cuda(), io["clcl"].cuda()]
with torch.inference_mode():
    out = runner(*inputs)
outs = list(out) if isinstance(out, (list, tuple)) else [out]
ref = [t.cuda() for t in io["out"]]
names = ["enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"]
maxd = 0.0
for name, expected, actual in zip(names, ref, outs):
    diff = (expected.float() - actual.float()).abs().max().item() if expected.shape == actual.shape and expected.numel() else 0.0
    maxd = max(maxd, diff)
    print(f"{name}: shape={tuple(actual.shape)} max_abs_diff={diff:.3e}")
print(f"steady compile OK: {out_path} max_abs_diff={maxd:.3e}")
PY
}

compile_buckets() {
  local py
  py="$(export_python)"
  need_file "$ART_GB10/session_bundle.ts"
  need_file "$ART_GB10/finalize_shared_weights.pt"
  need_file "$ART_GB10/joint_step.ts"
  need_file "$ART_GB10/predict_step.ts"
  mkdir -p "$ART_GB10/stripped_finalize_buckets" "$ART_GB10/finalize_compile_work"

  local count=0
  while IFS= read -r ep_name; do
    [[ -n "$ep_name" ]] || continue
    local ep="$ART_GB10/finalize_buckets/$ep_name"
    local key="${ep_name%_ep.pt2}"
    local pkg="${ep_name/_ep.pt2/.pt2}"
    local work="$ART_GB10/finalize_compile_work/$key"
    need_file "$ep"
    if [[ -f "$ART_GB10/stripped_finalize_buckets/$pkg" && "${FORCE_BUCKET_COMPILE:-0}" != "1" ]]; then
      log "bucket exists, skipping: $pkg"
      count=$((count + 1))
      continue
    fi
    rm -rf "$work"
    mkdir -p "$work"
    ln -s "$(realpath "$ep")" "$work/$ep_name"
    log "compile bucket: $ep_name"
    if ! "$py" "$ROOT/aot_compile_buckets.py" \
        --dir "$work" \
        --shared-weights "$ART_GB10/finalize_shared_weights.pt" \
        --force \
        --self-check-atol "$SELF_CHECK_ATOL"; then
      if [[ "${ALLOW_PARTIAL_BUCKETS:-0}" == "1" ]]; then
        log "bucket failed self-check; skipping because ALLOW_PARTIAL_BUCKETS=1: $ep_name"
        rm -rf "$work"
        continue
      fi
      die "bucket compile/self-check failed: $ep_name"
    fi
    need_file "$work/$pkg"
    log "strip bucket: $pkg"
    "$py" "$ROOT/strip_bucket_weights.py" \
      --bucket "$work/$pkg" \
      --out-dir "$ART_GB10/stripped_finalize_buckets" \
      --shared-weights "$ART_GB10/finalize_shared_weights.pt" \
      --bundle "$ART_GB10/session_bundle.ts" \
      --joint "$ART_GB10/joint_step.ts" \
      --predict "$ART_GB10/predict_step.ts" \
      --strip-only \
      --force
    rm -rf "$work"
    count=$((count + 1))
  done < <(bucket_eps_from_manifest)

  "$py" "$ROOT/build_bucket_manifest.py" \
    --buckets-dir "$ART_GB10/stripped_finalize_buckets" \
    --weights "$ART_GB10/finalize_shared_weights.pt"
  log "compiled/verified bucket count: $count"
}

compile_steady_batches() {
  need_file "$ART_GB10/finalize_shared_weights.ts"
  need_file "$ART_GB10/enc_steady_aoti.pt2"
  for ep in "$STEADY_B_GB10"/enc_steady_t2a_b{1,2,4}.pt2; do
    need_file "$ep"
  done
  log "compiling steady batch scheduler artifacts into $STEADY_B_GB10"
  python3 "$ROOT/export_steady_batched.py" \
    --out "$STEADY_B_GB10" \
    --batches 1,2,4 \
    --compile-only \
    --shared-weights "$ART_GB10/finalize_shared_weights.ts" \
    --production-b1 "$ART_GB10/enc_steady_aoti.pt2" \
    --atol "$SELF_CHECK_ATOL"
}

export_audio_bundle() {
  local py
  py="$(export_python)"
  need_file "$ART_GB10/enc_first.ts"
  need_file "$ART_GB10/enc_steady_aoti.pt2"
  log "exporting ws_server PCM/audio bundle locally"
  "$py" "$ROOT/export_session_bundle.py" \
    --audio \
    --n "${SESSION_BUNDLE_N:-20}" \
    --out "$ART_GB10/session_audio_bundle.ts"
}

export_session_bundle() {
  local py
  py="$(export_python)"
  need_file "$ART_GB10/enc_first.ts"
  need_file "$ART_GB10/enc_steady_aoti.pt2"
  log "exporting mel/session bundle locally"
  "$py" "$ROOT/export_session_bundle.py" \
    --n "${SESSION_BUNDLE_N:-20}" \
    --out "$ART_GB10/session_bundle.ts"
}

check_runtime_artifacts() {
  for path in \
    "$ART_GB10/session_audio_bundle.ts" \
    "$ART_GB10/preproc.ts" \
    "$ART_GB10/preproc.ts.manifest.json" \
    "$ART_GB10/enc_first.ts" \
    "$ART_GB10/enc_steady_aoti.pt2" \
    "$ART_GB10/finalize_shared_weights.ts" \
    "$ART_GB10/joint_step.ts" \
    "$ART_GB10/predict_step.ts" \
    "$ART_GB10/stripped_finalize_buckets/manifest.json"; do
    need_file "$path"
  done
  local bucket_count
  bucket_count=$(find "$ART_GB10/stripped_finalize_buckets" -maxdepth 1 -type f -name 'enc_finalize_d*_T*.pt2' | wc -l)
  if [[ "${ALLOW_PARTIAL_BUCKETS:-0}" == "1" ]]; then
    [[ "$bucket_count" -gt 0 ]] || die "expected at least one stripped finalize bucket, found $bucket_count"
  else
    [[ "$bucket_count" -eq 32 ]] || die "expected 32 stripped finalize buckets, found $bucket_count"
  fi
  log "stripped finalize buckets present: $bucket_count"
  log "runtime artifact check PASS: $ART_GB10"
}

usage() {
  cat <<EOF
usage: $0 <stage> [stage...]

stages:
  check-env        Verify GB10/PyTorch/AOTI environment.
  setup-export-env Create/update a local NeMo export venv.
  export-sources   Export portable source artifacts locally from HF model/data,
                   including the full known finalize bucket EP contract.
  compile-steady   Compile enc_steady_t2a.pt2 -> enc_steady_aoti.pt2.
  compile-steady-batches
                   Compile steady_b_artifacts batch packages for scheduler.
  export-session-bundle
                   Export session_bundle.ts after steady AOTI exists.
  compile-buckets  Compile and strip the 32 finalize buckets.
                   Set ALLOW_PARTIAL_BUCKETS=1 only for proof-of-concept runs.
  export-audio-bundle
                   Export session_audio_bundle.ts after steady AOTI exists.
  check            Validate the ws_server runtime artifact set.
  all-local        Run clean local path: setup export env, export, compile, check.
  all              Alias for all-local. No AWS dependency.

env:
  ART_GB10=$ART_GB10
  STEADY_B_GB10=$STEADY_B_GB10
  EXPORT_VENV=$EXPORT_VENV
  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

for stage in "$@"; do
  case "$stage" in
    check-env) check_env ;;
    setup-export-env) setup_export_env ;;
    export-sources) export_sources ;;
    compile-steady) compile_steady ;;
    compile-steady-batches) compile_steady_batches ;;
    export-session-bundle) export_session_bundle ;;
    compile-buckets) compile_buckets ;;
    export-audio-bundle) export_audio_bundle ;;
    check) check_runtime_artifacts ;;
    all|all-local)
      check_env
      setup_export_env
      export_sources
      compile_steady
      compile_steady_batches
      export_session_bundle
      compile_buckets
      export_audio_bundle
      check_runtime_artifacts
      ;;
    -h|--help) usage ;;
    *) die "unknown stage: $stage" ;;
  esac
done
