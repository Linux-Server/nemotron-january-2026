#!/usr/bin/env python3
"""Strip dead packaged weights from an AOTI finalize bucket and validate it.

Constants-on-disk AOTI packages contain ``data/weights/weight_*`` files. The
runtime path proven by ``validate_shared_weights.py`` wires the same weights via
``loader.load_constants(..., user_managed=True)`` from
``artifacts/finalize_shared_weights.pt``. This script removes those packaged
weight blobs from a copy of one bucket and then validates load/constants/run +
decode-continuation against the fixture bundle.

Run validation in the AOTI container, from runtime/:
  python3 strip_bucket_weights.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
import zipfile
from dataclasses import dataclass
from typing import Any


BLANK = 1024
MAX_SYMBOLS = 10
BUCKET_RE = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)")
DEFAULT_BUCKET = "artifacts/finalize_buckets/enc_finalize_d2_T44.pt2"


@dataclass
class StripStats:
    src: str
    dst: str
    src_size: int
    dst_size: int
    removed_entries: int
    removed_uncompressed: int
    removed_compressed: int


def parse_bucket_name(path: str) -> tuple[int, int]:
    match = BUCKET_RE.match(os.path.basename(path))
    if not match:
        raise ValueError(f"cannot parse drop/T from bucket filename: {path}")
    return int(match.group("drop")), int(match.group("T"))


def fmt_bytes(size: int) -> str:
    return f"{size / 1_000_000_000:.3f} GB ({size / 1024 / 1024:.1f} MiB)"


def is_weight_entry(name: str) -> bool:
    return "/data/weights/" in name


def clone_zipinfo(info: zipfile.ZipInfo) -> zipfile.ZipInfo:
    out = zipfile.ZipInfo(info.filename)
    year, month, day, hour, minute, second = info.date_time
    if year < 1980 or not (1 <= month <= 12) or not (1 <= day <= 31):
        out.date_time = (1980, 1, 1, 0, 0, 0)
    else:
        out.date_time = (year, month, day, hour, minute, second)
    out.compress_type = info.compress_type
    out.comment = info.comment
    out.extra = info.extra
    out.internal_attr = info.internal_attr
    out.external_attr = info.external_attr
    out.create_system = info.create_system
    return out


def list_weight_entries(src: str) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(src, "r") as zf:
        return [info for info in zf.infolist() if is_weight_entry(info.filename)]


def strip_package(
    src: str,
    dst: str,
    *,
    remove_all_weights: bool = True,
    remove_weight_names: set[str] | None = None,
) -> StripStats:
    if remove_weight_names is None:
        remove_weight_names = set()
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".tmp"
    removed_entries = 0
    removed_uncompressed = 0
    removed_compressed = 0

    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(tmp, "w", allowZip64=True) as zout:
        for info in zin.infolist():
            remove = is_weight_entry(info.filename) and (
                remove_all_weights or info.filename in remove_weight_names
            )
            if remove:
                removed_entries += 1
                removed_uncompressed += int(info.file_size)
                removed_compressed += int(info.compress_size)
                continue

            cloned = clone_zipinfo(info)
            if info.is_dir():
                zout.writestr(cloned, b"")
                continue
            with zin.open(info, "r") as src_f, zout.open(cloned, "w", force_zip64=True) as dst_f:
                shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)

    os.replace(tmp, dst)
    return StripStats(
        src=src,
        dst=dst,
        src_size=os.path.getsize(src),
        dst_size=os.path.getsize(dst),
        removed_entries=removed_entries,
        removed_uncompressed=removed_uncompressed,
        removed_compressed=removed_compressed,
    )


def default_stripped_path(bucket: str, out_dir: str, suffix: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(bucket))
    return os.path.join(out_dir, f"{stem}_{suffix}{ext or '.pt2'}")


def attr_tensor(module: Any, name: str):
    try:
        return getattr(module, name)
    except AttributeError:
        return module._c.getattr(name)


def row_tensor(bundle: Any, row: int, name: str):
    return attr_tensor(bundle, f"row{row}_{name}")


def scalar_i64(tensor: Any) -> int:
    return int(tensor.detach().cpu().reshape(-1)[0].item())


def tensor_to_vec(tensor: Any) -> list[int]:
    import torch

    return [int(x) for x in tensor.detach().cpu().to(dtype=torch.long).reshape(-1).tolist()]


def find_fixture_row(bundle: Any, drop: int, T: int) -> int:
    num_rows = scalar_i64(attr_tensor(bundle, "num_rows"))
    for row in range(num_rows):
        row_drop = scalar_i64(row_tensor(bundle, row, "drop_extra"))
        row_T = int(row_tensor(bundle, row, "chunk_mel").shape[-1])
        if row_drop == drop and row_T == T:
            return row
    raise RuntimeError(f"no fixture row matches drop={drop} T={T}")


def resolve_shared_weight(weights: dict[str, Any], fqn: str):
    if fqn in weights:
        return weights[fqn]
    if fqn.startswith("encoder."):
        alt = "e." + fqn[len("encoder.") :]
        if alt in weights:
            return weights[alt]
    if fqn.startswith("e."):
        alt = "encoder." + fqn[len("e.") :]
        if alt in weights:
            return weights[alt]
    return None


def build_constants_for_loader(torch: Any, runner: Any, shared_weights_path: str) -> tuple[dict[str, Any], list[str], list[str]]:
    fqns = list(runner.loader.get_constant_fqns())
    weights = torch.load(shared_weights_path, map_location="cpu", weights_only=False)
    cmap: dict[str, Any] = {}
    missing: list[str] = []
    for fqn in fqns:
        tensor = resolve_shared_weight(weights, fqn)
        if tensor is None:
            missing.append(fqn)
        else:
            cmap[fqn] = tensor.cuda()
    return cmap, missing, fqns


def decode_range(torch: Any, joint: Any, predict: Any, enc_out: Any, enc_len: int, g: Any, h: Any, c: Any, hyp: list[int]) -> list[int]:
    if enc_len < 0 or enc_len > int(enc_out.shape[2]):
        raise RuntimeError(f"enc_len={enc_len} out of range for enc_out shape={tuple(enc_out.shape)}")
    f = enc_out.transpose(1, 2).contiguous()
    device = f.device
    for t in range(enc_len):
        f_t = f[:, t : t + 1, :]
        for _ in range(MAX_SYMBOLS):
            logits = joint(f_t, g)
            k = int(logits.reshape(-1).argmax().item())
            if k == BLANK:
                break
            hyp.append(k)
            y = torch.full((1, 1), k, dtype=torch.long, device=device)
            out = predict(y, h, c)
            g, h, c = out[0], out[1], out[2]
    return hyp


def validate_package(args: argparse.Namespace) -> dict[str, Any]:
    import faulthandler

    import torch

    faulthandler.enable()
    torch.set_grad_enabled(False)
    device = torch.device("cuda")

    drop, T = parse_bucket_name(args.bucket)
    print(f"validating package={args.bucket} drop={drop} T={T}", flush=True)

    bundle = torch.jit.load(args.bundle)
    row = find_fixture_row(bundle, drop, T)
    joint = torch.jit.load(args.joint).to(device).eval()
    predict = torch.jit.load(args.predict).to(device).eval()

    print("  aoti_load_package...", flush=True)
    runner = torch._inductor.aoti_load_package(args.bucket)
    fqns = list(runner.loader.get_constant_fqns())
    print(f"  package loaded; constant_fqns={len(fqns)}", flush=True)

    print("  loading shared constants + load_constants(user_managed=True)...", flush=True)
    cmap, missing, _ = build_constants_for_loader(torch, runner, args.shared_weights)
    if missing:
        raise RuntimeError(f"missing {len(missing)} shared constants; first={missing[:5]}")
    runner.loader.load_constants(cmap, False, False, True)
    print(f"  load_constants OK; matched={len(cmap)}", flush=True)

    inputs = (
        row_tensor(bundle, row, "chunk_mel").to(device).contiguous(),
        row_tensor(bundle, row, "cache_last_channel").to(device).contiguous(),
        row_tensor(bundle, row, "cache_last_time").to(device).contiguous(),
        row_tensor(bundle, row, "cache_last_channel_len").to(device).contiguous(),
    )
    print("  run...", flush=True)
    out = runner(*inputs)
    if not isinstance(out, (tuple, list)):
        out = (out,)
    if len(out) < 2:
        raise RuntimeError(f"bucket returned {len(out)} outputs, expected at least 2")
    enc_out = out[0]
    enc_len = scalar_i64(out[1])

    hyp = tensor_to_vec(row_tensor(bundle, row, "pre_final_tokens"))
    g = row_tensor(bundle, row, "pre_final_pred_out").to(device).contiguous()
    h = row_tensor(bundle, row, "pre_final_h").to(device).contiguous()
    c = row_tensor(bundle, row, "pre_final_c").to(device).contiguous()
    got = decode_range(torch, joint, predict, enc_out, enc_len, g, h, c, hyp)
    gold = tensor_to_vec(row_tensor(bundle, row, "finalize_ref_final_tokens"))
    nemo_gold = tensor_to_vec(row_tensor(bundle, row, "nemo_stream_finalize_tokens"))
    token_exact = got == gold
    nemo_exact = got == nemo_gold

    result = {
        "ok": bool(token_exact and nemo_exact),
        "package": args.bucket,
        "drop": drop,
        "T": T,
        "row": row,
        "constant_fqns": len(fqns),
        "matched_constants": len(cmap),
        "enc_len": enc_len,
        "tokens": got,
        "gold_tokens": gold,
        "token_exact_vs_finalize_ref": token_exact,
        "token_exact_vs_nemo": nemo_exact,
    }
    print(
        "  decoded "
        f"row={row} enc_len={enc_len} tokens={len(got)} "
        f"finalize_ref={'PASS' if token_exact else 'FAIL'} "
        f"nemo={'PASS' if nemo_exact else 'FAIL'}",
        flush=True,
    )
    print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return result


def validate_only_main(args: argparse.Namespace) -> int:
    try:
        result = validate_package(args)
        return 0 if result["ok"] else 1
    except BaseException as exc:
        payload = {
            "ok": False,
            "package": args.bucket,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print("VALIDATION_ERROR", type(exc).__name__, str(exc), flush=True)
        traceback.print_exc()
        print("RESULT_JSON " + json.dumps(payload, sort_keys=True), flush=True)
        return 1


def run_validation_child(bucket: str, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--validate-only",
        "--bucket",
        bucket,
        "--shared-weights",
        args.shared_weights,
        "--bundle",
        args.bundle,
        "--joint",
        args.joint,
        "--predict",
        args.predict,
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")

    parsed: dict[str, Any] | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            try:
                parsed = json.loads(line[len("RESULT_JSON ") :])
            except json.JSONDecodeError:
                parsed = None
    if parsed is None:
        parsed = {"ok": False, "package": bucket, "error": "validation child produced no RESULT_JSON"}
    parsed["returncode"] = proc.returncode
    parsed["process_ok"] = proc.returncode == 0
    if proc.returncode < 0:
        parsed["signal"] = -proc.returncode
    return parsed


def print_strip_stats(stats: StripStats) -> None:
    ratio = stats.dst_size / stats.src_size if stats.src_size else 0.0
    print(
        f"stripped {stats.removed_entries} weight entries from {stats.src}\n"
        f"  original: {fmt_bytes(stats.src_size)}\n"
        f"  stripped: {fmt_bytes(stats.dst_size)} ({ratio:.4%} of original)\n"
        f"  removed uncompressed: {fmt_bytes(stats.removed_uncompressed)}\n"
        f"  removed compressed:   {fmt_bytes(stats.removed_compressed)}\n"
        f"  output: {stats.dst}"
    )


def token_match(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    return bool(lhs.get("tokens") == rhs.get("tokens"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--out-dir", default="artifacts/stripped_finalize_buckets")
    parser.add_argument("--out", default=None)
    parser.add_argument("--shared-weights", default="artifacts/finalize_shared_weights.pt")
    parser.add_argument("--bundle", default="artifacts/finalize_bundle.ts")
    parser.add_argument("--joint", default="artifacts/joint_step.ts")
    parser.add_argument("--predict", default="artifacts/predict_step.ts")
    parser.add_argument("--strip-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fallback-largest",
        type=int,
        nargs="*",
        default=[1, 8, 32],
        help="if all-weight stripping fails validation, try packages with only the N largest weight blobs removed",
    )
    args = parser.parse_args()

    if args.validate_only:
        return validate_only_main(args)

    stripped = args.out or default_stripped_path(args.bucket, args.out_dir, "stripped")
    stats = strip_package(args.bucket, stripped, remove_all_weights=True)
    print_strip_stats(stats)

    if args.strip_only:
        return 0

    print("\n=== baseline validation: original package ===")
    baseline = run_validation_child(args.bucket, args)
    print("\n=== validation: all data/weights/* removed ===")
    stripped_result = run_validation_child(stripped, args)

    stripped_ok = bool(stripped_result.get("process_ok") and stripped_result.get("ok"))
    baseline_ok = bool(baseline.get("process_ok") and baseline.get("ok"))
    same_tokens = baseline_ok and stripped_ok and token_match(baseline, stripped_result)

    print("\n=== strip finding ===")
    print(f"baseline original token-exact: {baseline_ok}")
    print(f"all-weights-stripped token-exact: {stripped_ok}")
    print(f"stripped matches original tokens: {same_tokens}")
    print(f"size before/after: {fmt_bytes(stats.src_size)} -> {fmt_bytes(stats.dst_size)}")

    if baseline_ok and stripped_ok and same_tokens:
        print("FINDING: aoti_load_package + load_constants(user_managed=True) + run works with data/weights/* removed.")
        return 0

    print("FINDING: removing all data/weights/* did not validate cleanly; see validation output above.")
    if not stripped_ok and args.fallback_largest:
        weights = sorted(list_weight_entries(args.bucket), key=lambda info: info.file_size, reverse=True)
        for count in args.fallback_largest:
            if count <= 0:
                continue
            names = {info.filename for info in weights[:count]}
            fallback = default_stripped_path(args.bucket, args.out_dir, f"drop_largest{count}")
            fb_stats = strip_package(
                args.bucket,
                fallback,
                remove_all_weights=False,
                remove_weight_names=names,
            )
            print(f"\n=== fallback validation: removed largest {count} weight blobs ===")
            print_strip_stats(fb_stats)
            fb_result = run_validation_child(fallback, args)
            fb_ok = bool(fb_result.get("process_ok") and fb_result.get("ok"))
            fb_same = baseline_ok and fb_ok and token_match(baseline, fb_result)
            print(f"fallback largest {count}: token-exact={fb_ok} matches_original={fb_same}")
            if fb_ok and fb_same:
                print(f"FALLBACK FINDING: removing the largest {count} packaged weight blobs still works.")
                break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
