#!/usr/bin/env bash
# Run on the g6e.8xlarge L40S box after this runtime directory is rsynced to
# ~/density. Builds native sm_89 AOTI artifacts in artifacts_sm89/ and runs the
# Phase-2 Step-1a density sweep without touching the source artifacts/.
set -euo pipefail
IFS=$'\n\t'

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=$(pwd -P)

SRC_ARTIFACTS=${SRC_ARTIFACTS:-"$ROOT/artifacts"}
ART_SM89=${ART_SM89:-"$ROOT/artifacts_sm89"}
VENV=${VENV:-"$HOME/torch280-sm89-venv"}
PYTHON_BIN=${PYTHON_BIN:-python3.11}
SELF_CHECK_ATOL=${SELF_CHECK_ATOL:-0.1}
DENSITY_N_VALUES=${DENSITY_N_VALUES:-"1,2,4,8,16,24"}
KEEP_UNSTRIPPED_BUCKETS=${KEEP_UNSTRIPPED_BUCKETS:-0}
DENSITY_TREAT_NO_PASS_AS_FAILURE=${DENSITY_TREAT_NO_PASS_AS_FAILURE:-0}
BUILD_DIR=${BUILD_DIR:-"$ROOT/cpp/build_l40s_density"}

log() {
  printf '[l40s-density %(%H:%M:%S)T] %s\n' -1 "$*"
}

die() {
  printf '[l40s-density ERROR] %s\n' "$*" >&2
  exit 2
}

need_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

need_dir() {
  [[ -d "$1" ]] || die "missing required directory: $1"
}

sudo_cmd() {
  if [[ ${EUID} -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    die "need root or sudo for: $*"
  fi
}

link_artifact() {
  local src=$1
  local dst=$2
  need_file "$src"
  mkdir -p "$(dirname "$dst")"
  ln -sfn "$(realpath "$src")" "$dst"
}

find_cuda_root() {
  if [[ -n "${CUDA_ROOT:-}" ]]; then
    [[ -d "$CUDA_ROOT" ]] || die "CUDA_ROOT is set but not a directory: $CUDA_ROOT"
    printf '%s\n' "$CUDA_ROOT"
    return
  fi

  local -a candidates=()
  while IFS= read -r path; do
    candidates+=("$path")
  done < <(find /usr/local -maxdepth 1 -type d -name 'cuda-12*' 2>/dev/null | sort)
  if ((${#candidates[@]} > 0)); then
    printf '%s\n' "${candidates[$((${#candidates[@]} - 1))]}"
  elif [[ -d /usr/local/cuda ]]; then
    printf '%s\n' /usr/local/cuda
  else
    die "could not find system CUDA under /usr/local/cuda-12* or /usr/local/cuda"
  fi
}

install_os_deps() {
  log "checking OS build deps"
  if command -v cmake >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    return
  fi
  sudo_cmd apt-get update -qq
  sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake g++ ninja-build python3-venv >/dev/null
}

setup_venv() {
  log "creating torch 2.8.0 venv at $VENV"
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  if command -v uv >/dev/null 2>&1; then
    uv venv --python "$PYTHON_BIN" "$VENV"
    uv pip install --python "$VENV" "torch==2.8.0"
  else
    "$PYTHON_BIN" -m venv "$VENV"
    "$VENV/bin/python" -m pip install -U pip
    "$VENV/bin/python" -m pip install "torch==2.8.0"
  fi
}

check_torch_cuda() {
  local py=$1
  log "checking torch CUDA device and wheel arch list"
  "$py" - <<'PY'
import sys
import torch

print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda is not available")
cc = torch.cuda.get_device_capability()
arch = torch.cuda.get_arch_list()
print("device", torch.cuda.get_device_name(0), "cc", cc)
print("arch_list", arch)
if tuple(cc) != (8, 9):
    raise SystemExit(f"expected L40S sm_89 device capability (8, 9), got {cc}")
if "sm_89" not in arch:
    raise SystemExit(f"torch CUDA arch list does not include sm_89: {arch}")
if not hasattr(torch._inductor, "aoti_compile_and_package"):
    raise SystemExit("torch._inductor.aoti_compile_and_package is missing")
print("aoti_compile_and_package OK")
print("_GLIBCXX_USE_CXX11_ABI", getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", "unknown"))
PY
}

prepare_artifacts() {
  log "preparing separate sm_89 artifact dir: $ART_SM89"
  need_dir "$SRC_ARTIFACTS"
  need_dir "$SRC_ARTIFACTS/finalize_buckets"
  need_file "$SRC_ARTIFACTS/stripped_finalize_buckets/manifest.json"

  mkdir -p "$ART_SM89"
  link_artifact "$SRC_ARTIFACTS/enc_steady_t2a.pt2" "$ART_SM89/enc_steady_t2a.pt2"
  link_artifact "$SRC_ARTIFACTS/t2a_io.pt" "$ART_SM89/t2a_io.pt"
  link_artifact "$SRC_ARTIFACTS/session_bundle.ts" "$ART_SM89/session_bundle.ts"
  link_artifact "$SRC_ARTIFACTS/finalize_shared_weights.pt" "$ART_SM89/finalize_shared_weights.pt"
  link_artifact "$SRC_ARTIFACTS/finalize_shared_weights.ts" "$ART_SM89/finalize_shared_weights.ts"
  link_artifact "$SRC_ARTIFACTS/enc_first.ts" "$ART_SM89/enc_first.ts"
  link_artifact "$SRC_ARTIFACTS/joint_step.ts" "$ART_SM89/joint_step.ts"
  link_artifact "$SRC_ARTIFACTS/predict_step.ts" "$ART_SM89/predict_step.ts"
  if [[ -f "$SRC_ARTIFACTS/preproc.ts" ]]; then
    link_artifact "$SRC_ARTIFACTS/preproc.ts" "$ART_SM89/preproc.ts"
  fi

  mkdir -p "$ART_SM89/finalize_buckets"
  if [[ -f "$SRC_ARTIFACTS/finalize_buckets/buckets_manifest.json" ]]; then
    cp -f "$SRC_ARTIFACTS/finalize_buckets/buckets_manifest.json" \
      "$ART_SM89/finalize_buckets/buckets_manifest.source.json"
  fi

  "$PY" - "$SRC_ARTIFACTS" "$ART_SM89" <<'PY'
import json
import os
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
manifest = json.loads((src / "stripped_finalize_buckets" / "manifest.json").read_text())
missing = []
linked = 0
for bucket in manifest["buckets"]:
    name = f"enc_finalize_d{bucket['drop']}_T{bucket['T']}_ep.pt2"
    ep = src / "finalize_buckets" / name
    if not ep.is_file():
        missing.append(str(ep))
        continue
    target = dst / "finalize_buckets" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass
    os.symlink(ep.resolve(), tmp)
    os.replace(tmp, target)
    linked += 1
if missing:
    print("missing exported finalize bucket programs:", file=sys.stderr)
    for path in missing:
        print("  " + path, file=sys.stderr)
    raise SystemExit(2)
print(f"linked {linked} finalize bucket ExportedPrograms")
PY

  rm -rf "$ART_SM89/stripped_finalize_buckets" "$ART_SM89/finalize_compile_work"
  mkdir -p "$ART_SM89/stripped_finalize_buckets" "$ART_SM89/finalize_compile_work/manifests"
  rm -f "$ART_SM89/enc_steady_aoti.pt2"
  find "$ART_SM89/finalize_buckets" -maxdepth 1 -type f \
    -name 'enc_finalize_d*_T*.pt2' ! -name '*_ep.pt2' -delete
}

compile_steady_sm89() {
  log "native sm_89 AOTI compile: enc_steady_t2a.pt2 -> enc_steady_aoti.pt2"
  "$PY" - "$ART_SM89" <<'PY'
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
            print("[noexecstack] injected into shared-lib link:", cmd[-120:])
        return cmd
    cb.CppBuilder.get_command_line = patched
    return seen

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_capability())
seen = force_noexecstack_on_link()
ep = torch.export.load(os.path.join(art, "enc_steady_t2a.pt2"))
pkg = os.path.join(art, "enc_steady_aoti.pt2")
out_path = torch._inductor.aoti_compile_and_package(ep, package_path=pkg)
if not seen["flagged"]:
    raise SystemExit("noexecstack shim never fired on a shared-lib link")
print("AOTI package:", out_path)

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
    if not torch.is_tensor(expected) or not torch.is_tensor(actual):
        continue
    if expected.shape != actual.shape:
        raise SystemExit(f"{name} shape mismatch: {tuple(expected.shape)} vs {tuple(actual.shape)}")
    diff = (expected.float() - actual.float()).abs().max().item() if expected.numel() else 0.0
    maxd = max(maxd, diff)
    print(f"  {name}: byte_equal={torch.equal(expected, actual)} max_abs_diff={diff:.3e} shape={tuple(actual.shape)}")
print(f"=== steady AOTI load+run OK: max_abs_diff={maxd:.3e} ===")
PY
}

compile_strip_bucket() {
  local ep=$1
  local base key pkg one_dir
  base=$(basename "$ep")
  key=${base%_ep.pt2}
  pkg=${base/_ep.pt2/.pt2}
  one_dir="$ART_SM89/finalize_compile_work/$key"

  rm -rf "$one_dir"
  mkdir -p "$one_dir"
  ln -sfn "$(realpath "$ep")" "$one_dir/$base"

  log "native sm_89 AOTI compile bucket: $base"
  "$PY" "$ROOT/aot_compile_buckets.py" \
    --dir "$one_dir" \
    --shared-weights "$ART_SM89/finalize_shared_weights.pt" \
    --force \
    --self-check-atol "$SELF_CHECK_ATOL"

  need_file "$one_dir/$pkg"
  cp -f "$one_dir/buckets_manifest.json" "$ART_SM89/finalize_compile_work/manifests/$key.json"

  log "strip bucket weights: $pkg"
  "$PY" "$ROOT/strip_bucket_weights.py" \
    --bucket "$one_dir/$pkg" \
    --out-dir "$ART_SM89/stripped_finalize_buckets" \
    --shared-weights "$ART_SM89/finalize_shared_weights.pt" \
    --bundle "$ART_SM89/session_bundle.ts" \
    --joint "$ART_SM89/joint_step.ts" \
    --predict "$ART_SM89/predict_step.ts" \
    --strip-only \
    --force

  if [[ "$KEEP_UNSTRIPPED_BUCKETS" == "1" ]]; then
    mv -f "$one_dir/$pkg" "$ART_SM89/finalize_buckets/$pkg"
  fi
  rm -rf "$one_dir"
}

compile_finalize_buckets_sm89() {
  log "native sm_89 AOTI compile + strip finalize buckets, self-check-atol=$SELF_CHECK_ATOL"
  local -a eps=()
  while IFS= read -r path; do
    eps+=("$path")
  done < <(find "$ART_SM89/finalize_buckets" -maxdepth 1 -type l -name 'enc_finalize_d*_T*_ep.pt2' | sort)
  ((${#eps[@]} > 0)) || die "no finalize bucket ExportedPrograms in $ART_SM89/finalize_buckets"
  for ep in "${eps[@]}"; do
    compile_strip_bucket "$ep"
  done

  "$PY" - "$SRC_ARTIFACTS" "$ART_SM89" <<'PY'
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
art = Path(sys.argv[2])
contract_manifest = json.loads((src / "stripped_finalize_buckets" / "manifest.json").read_text())
contract = contract_manifest["CONTRACT"]
expected = {(int(b["drop"]), int(b["T"])) for b in contract_manifest.get("buckets", [])}
bucket_re = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)\.pt2$")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

stripped_dir = art / "stripped_finalize_buckets"
buckets = []
seen = set()
for path in sorted(stripped_dir.glob("enc_finalize_d*_T*.pt2")):
    m = bucket_re.match(path.name)
    if not m:
        continue
    key = (int(m.group("drop")), int(m.group("T")))
    seen.add(key)
    buckets.append({
        "drop": key[0],
        "T": key[1],
        "pkg": path.name,
        "pkg_sha256": sha256_file(path),
    })
if expected and seen != expected:
    raise SystemExit(f"stripped bucket key mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")
manifest = {
    "schema_version": 1,
    "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "CONTRACT": contract,
    "buckets": buckets,
}
(stripped_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

compile_records = []
for manifest_path in sorted((art / "finalize_compile_work" / "manifests").glob("*.json")):
    compile_records.extend(json.loads(manifest_path.read_text()))
(art / "finalize_buckets" / "buckets_manifest_sm89.json").write_text(
    json.dumps(compile_records, indent=2, sort_keys=True) + "\n"
)
print(f"wrote stripped manifest: buckets={len(buckets)} path={stripped_dir / 'manifest.json'}")
print(f"wrote compile self-check manifest: buckets={len(compile_records)}")
PY
}

build_density_main() {
  log "building density_main with pip torch libtorch and system CUDA"
  TORCH_ROOT=$("$PY" - <<'PY'
import os
import torch
print(os.path.dirname(torch.__file__))
PY
)
  export TORCH_ROOT
  log "torch_root=$TORCH_ROOT"
  log "cuda_root=$CUDA_ROOT"
  cmake -S "$ROOT/cpp" -B "$BUILD_DIR" \
    -DTORCH_ROOT="$TORCH_ROOT" \
    -DCUDA_ROOT="$CUDA_ROOT" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" --target density_main -j"$(nproc)"
}

run_density_sweep() {
  log "running density sweep: N=$DENSITY_N_VALUES"
  mkdir -p "$ART_SM89/logs"
  local run_log
  run_log="$ART_SM89/logs/l40s_density_run_$(date -u +%Y%m%dT%H%M%SZ).log"
  local -a cmd=(
    "$BUILD_DIR/density_main"
    --mode density-sweep
    --n-values "$DENSITY_N_VALUES"
    "$ART_SM89"
  )
  if (($# > 0)); then
    cmd+=("$@")
  fi
  log "command: ${cmd[*]}"

  set +e
  "${cmd[@]}" 2>&1 | tee "$run_log"
  local density_rc=${PIPESTATUS[0]}
  set -e
  log "density_main exit code: $density_rc (0=PASS_TO_1B, 1=completed NO_PASS_TO_1B, 2=setup/runtime failure)"
  log "run log: $run_log"

  "$PY" - "$ART_SM89" <<'PY'
import glob
import json
import os
import sys

art = sys.argv[1]
manifests = sorted(glob.glob(os.path.join(art, "logs", "*", "density_sweep_manifest.json")))
if not manifests:
    print("L40S_DENSITY_RESULT no density_sweep_manifest.json found")
    raise SystemExit(0)
manifest_path = manifests[-1]
with open(manifest_path) as f:
    manifest = json.load(f)
log_dir = os.path.dirname(manifest_path)
summary_files = sorted(glob.glob(os.path.join(log_dir, "density_num_runners0_*1a_full_session_density_sweep_summary.jsonl")))
peak = 0
rows = []
if summary_files:
    with open(summary_files[-1]) as f:
        for line in f:
            if line.strip():
                summary = json.loads(line)
                rows = summary.get("rows", [])
                for row in rows:
                    peak = max(peak, int(row.get("peak_gpu_mem_bytes", 0)))
print(
    "L40S_DENSITY_RESULT "
    f"status={manifest.get('status')} "
    f"knee_N={manifest.get('knee_n')} "
    f"multiplier={manifest.get('multiplier')}x "
    f"binding_slo={manifest.get('binding_slo')} "
    f"binding_resource={manifest.get('binding_resource')} "
    f"peak_mem_GiB={peak / (1024**3):.3f} "
    f"manifest={manifest_path}"
)
for row in rows:
    print(
        "L40S_DENSITY_ROW "
        f"N={row.get('N')} "
        f"slo_robust={row.get('slo_robust')} "
        f"throughput_rt={row.get('throughput_realtime_streams')} "
        f"ttfs_p95={row.get('ttfs', {}).get('p95')} "
        f"lag_p95={row.get('lag', {}).get('p95')} "
        f"peak_mem_GiB={int(row.get('peak_gpu_mem_bytes', 0)) / (1024**3):.3f} "
        f"mismatches={row.get('mismatches')} "
        f"errors={row.get('errors')} "
        f"oom={row.get('oom')}"
    )
PY

  if [[ $density_rc -eq 2 ]]; then
    exit 2
  fi
  if [[ $density_rc -eq 1 && "$DENSITY_TREAT_NO_PASS_AS_FAILURE" == "1" ]]; then
    exit 1
  fi
}

main() {
  log "gpu"
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader || true

  install_os_deps
  CUDA_ROOT=$(find_cuda_root)
  export CUDA_ROOT CUDA_HOME="$CUDA_ROOT"
  export PATH="$CUDA_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
  unset TORCH_CUDA_ARCH_LIST
  command -v nvcc >/dev/null 2>&1 || die "nvcc not found under CUDA_ROOT=$CUDA_ROOT"
  log "nvcc: $(nvcc --version | tail -1)"

  setup_venv
  PY="$VENV/bin/python"
  export PY
  check_torch_cuda "$PY"

  prepare_artifacts
  export TORCHINDUCTOR_CACHE_DIR="$ART_SM89/torchinductor_cache"
  compile_steady_sm89
  compile_finalize_buckets_sm89
  build_density_main
  run_density_sweep "$@"
  log "DONE"
}

main "$@"
