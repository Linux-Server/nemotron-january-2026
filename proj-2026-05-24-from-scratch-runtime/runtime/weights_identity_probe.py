#!/usr/bin/env python3
"""Read-only Step 1(a) probe: encoder weight identity and shared-constant FQN coverage.

Run from runtime/:
  HF_HUB_OFFLINE=1 ./.venv/bin/python weights_identity_probe.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def resolve_shared(weights: dict[str, torch.Tensor], fqn: str) -> tuple[str | None, bool]:
    if fqn in weights:
        return fqn, False
    if fqn.startswith("encoder."):
        alt = "e." + fqn[len("encoder.") :]
        if alt in weights:
            return alt, True
    elif fqn.startswith("e."):
        alt = "encoder." + fqn[len("e.") :]
        if alt in weights:
            return alt, True
    return None, False


def shape(t: torch.Tensor) -> list[int]:
    return [int(x) for x in t.shape]


def tensor_max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float | None:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return None
    if lhs.numel() == 0:
        return 0.0
    if lhs.is_floating_point() or rhs.is_floating_point():
        return float((lhs.float() - rhs.float()).abs().max().item())
    return float((lhs.to(torch.int64) - rhs.to(torch.int64)).abs().max().item())


def tensor_byte_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return False
    lhs_b = lhs.detach().cpu().contiguous().view(torch.uint8)
    rhs_b = rhs.detach().cpu().contiguous().view(torch.uint8)
    return bool(torch.equal(lhs_b, rhs_b))


def load_shared(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a dict")
    out: dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise TypeError(f"{path} has non string/tensor entry {type(key)} {type(value)}")
        out[key] = value.detach().cpu().contiguous()
    return out


def load_ts_named_tensors(path: Path) -> dict[str, torch.Tensor]:
    module = torch.jit.load(str(path), map_location="cpu").eval()
    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in module.named_parameters():
        tensors[name] = tensor.detach().cpu().contiguous()
    for name, tensor in module.named_buffers():
        tensors[name] = tensor.detach().cpu().contiguous()
    return tensors


def compare_named_tensors(
    source_name: str,
    source: dict[str, torch.Tensor],
    shared: dict[str, torch.Tensor],
    *,
    keep_per_tensor: bool,
) -> dict[str, Any]:
    missing: list[str] = []
    shape_dtype_mismatches: list[dict[str, Any]] = []
    nonzero_diff: list[dict[str, Any]] = []
    non_byte_equal: list[dict[str, Any]] = []
    per_tensor: list[dict[str, Any]] = []
    direct = 0
    alias = 0
    matched_shared: set[str] = set()
    max_abs_all = 0.0
    tensor_equal_count = 0
    byte_equal_count = 0

    for fqn in sorted(source):
        tensor = source[fqn]
        shared_key, used_alias = resolve_shared(shared, fqn)
        if shared_key is None:
            missing.append(fqn)
            row = {
                "fqn": fqn,
                "shared_key": None,
                "match": "missing",
                "shape": shape(tensor),
                "dtype": str(tensor.dtype),
            }
            if keep_per_tensor:
                per_tensor.append(row)
            continue

        direct += int(not used_alias)
        alias += int(used_alias)
        matched_shared.add(shared_key)
        other = shared[shared_key]
        max_abs = tensor_max_abs(tensor, other)
        tensor_equal = bool(torch.equal(tensor, other)) if max_abs is not None else False
        byte_equal = tensor_byte_equal(tensor, other)
        tensor_equal_count += int(tensor_equal)
        byte_equal_count += int(byte_equal)
        if max_abs is None:
            shape_dtype_mismatches.append(
                {
                    "fqn": fqn,
                    "shared_key": shared_key,
                    "shape": shape(tensor),
                    "dtype": str(tensor.dtype),
                    "shared_shape": shape(other),
                    "shared_dtype": str(other.dtype),
                }
            )
        else:
            max_abs_all = max(max_abs_all, max_abs)
            if max_abs != 0.0:
                nonzero_diff.append({"fqn": fqn, "shared_key": shared_key, "max_abs": max_abs})
        if not byte_equal:
            non_byte_equal.append({"fqn": fqn, "shared_key": shared_key})
        if keep_per_tensor:
            per_tensor.append(
                {
                    "fqn": fqn,
                    "shared_key": shared_key,
                    "match": "alias" if used_alias else "direct",
                    "shape": shape(tensor),
                    "dtype": str(tensor.dtype),
                    "shared_shape": shape(other),
                    "shared_dtype": str(other.dtype),
                    "max_abs": max_abs,
                    "tensor_equal": tensor_equal,
                    "byte_equal": byte_equal,
                }
            )

    shared_extras = sorted(set(shared) - matched_shared)
    return {
        "source": source_name,
        "source_tensors": len(source),
        "shared_tensors": len(shared),
        "source_found_in_shared": len(source) - len(missing),
        "shared_found_in_source": len(shared) - len(shared_extras),
        "missing_from_shared": missing,
        "shared_extras": shared_extras,
        "direct_matches": direct,
        "alias_fallbacks": alias,
        "shape_dtype_mismatches": shape_dtype_mismatches,
        "tensor_equal_count": tensor_equal_count,
        "byte_equal_count": byte_equal_count,
        "all_tensor_equal": (
            tensor_equal_count == len(source)
            and not missing
            and not shared_extras
            and not shape_dtype_mismatches
        ),
        "all_byte_equal": (
            byte_equal_count == len(source)
            and not missing
            and not shared_extras
            and not shape_dtype_mismatches
        ),
        "max_abs": max_abs_all,
        "nonzero_diff_count": len(nonzero_diff),
        "nonzero_diff_first10": nonzero_diff[:10],
        "non_byte_equal_count": len(non_byte_equal),
        "non_byte_equal_first10": non_byte_equal[:10],
        "per_tensor": per_tensor if keep_per_tensor else None,
    }


def fqn_coverage(name: str, fqns: list[str], shared: dict[str, torch.Tensor]) -> dict[str, Any]:
    missing: list[str] = []
    direct = 0
    alias = 0
    matched_shared: set[str] = set()
    for fqn in sorted(fqns):
        shared_key, used_alias = resolve_shared(shared, fqn)
        if shared_key is None:
            missing.append(fqn)
            continue
        direct += int(not used_alias)
        alias += int(used_alias)
        matched_shared.add(shared_key)
    shared_extras = sorted(set(shared) - matched_shared)
    return {
        "source": name,
        "constant_fqns": len(fqns),
        "shared_tensors": len(shared),
        "fqns_found_in_shared": len(fqns) - len(missing),
        "shared_found_in_fqns": len(shared) - len(shared_extras),
        "direct_matches": direct,
        "alias_fallbacks": alias,
        "missing_from_shared": missing,
        "shared_extras": shared_extras,
        "one_shared_map_covers_all_fqns": not missing and not shared_extras,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ART)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--include-per-tensor", action="store_true")
    args = parser.parse_args()

    started = time.time()
    shared_path = args.artifacts / "finalize_shared_weights.pt"
    enc_first_path = args.artifacts / "enc_first.ts"
    enc_steady_path = args.artifacts / "enc_steady_aoti.pt2"
    enc_first_aoti_path = args.artifacts / "enc_first_aoti.pt2"

    print(f"loading shared weights: {shared_path}", flush=True)
    shared = load_shared(shared_path)
    print(f"loading TorchScript first encoder: {enc_first_path}", flush=True)
    enc_first = load_ts_named_tensors(enc_first_path)

    enc_first_result = compare_named_tensors(
        "enc_first.ts named_parameters+named_buffers",
        enc_first,
        shared,
        keep_per_tensor=args.include_per_tensor,
    )

    print(f"loading inline steady AOTI package for FQNs: {enc_steady_path}", flush=True)
    enc_steady_runner = torch._inductor.aoti_load_package(str(enc_steady_path))
    enc_steady_fqns = [str(x) for x in enc_steady_runner.loader.get_constant_fqns()]
    enc_steady_result = fqn_coverage("enc_steady_aoti.pt2 get_constant_fqns", enc_steady_fqns, shared)

    print(f"loading stripped first AOTI package for FQNs: {enc_first_aoti_path}", flush=True)
    enc_first_aoti_runner = torch._inductor.aoti_load_package(str(enc_first_aoti_path))
    enc_first_aoti_fqns = [str(x) for x in enc_first_aoti_runner.loader.get_constant_fqns()]
    enc_first_aoti_result = fqn_coverage(
        "enc_first_aoti.pt2 get_constant_fqns",
        enc_first_aoti_fqns,
        shared,
    )

    result = {
        "elapsed_seconds": time.time() - started,
        "torch": str(torch.__version__),
        "shared_path": str(shared_path),
        "enc_first": enc_first_result,
        "enc_steady_inline_fqns": enc_steady_result,
        "enc_first_aoti_fqns": enc_first_aoti_result,
        "one_shared_constants_map_serves_all": (
            enc_first_result["all_tensor_equal"]
            and enc_steady_result["one_shared_map_covers_all_fqns"]
            and enc_first_aoti_result["one_shared_map_covers_all_fqns"]
        ),
    }

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("\n=== weights identity probe ===")
    r = enc_first_result
    print(
        "enc_first.ts vs finalize_shared_weights.pt: "
        f"{r['source_found_in_shared']}/{r['source_tensors']} enc_first tensors found, "
        f"{r['shared_found_in_source']}/{r['shared_tensors']} shared tensors covered, "
        f"direct={r['direct_matches']} alias={r['alias_fallbacks']}"
    )
    print(
        "  "
        f"tensor_equal={r['all_tensor_equal']} byte_equal={r['all_byte_equal']} "
        f"max_abs={r['max_abs']:.9g} shape_dtype_mismatches={len(r['shape_dtype_mismatches'])} "
        f"missing={len(r['missing_from_shared'])} shared_extras={len(r['shared_extras'])}"
    )
    for key, label in [
        ("enc_first_aoti_fqns", "enc_first_aoti.pt2 FQNs"),
        ("enc_steady_inline_fqns", "enc_steady_aoti.pt2 FQNs"),
    ]:
        c = result[key]
        print(
            f"{label}: {c['fqns_found_in_shared']}/{c['constant_fqns']} FQNs found, "
            f"{c['shared_found_in_fqns']}/{c['shared_tensors']} shared tensors covered, "
            f"direct={c['direct_matches']} alias={c['alias_fallbacks']} missing={len(c['missing_from_shared'])}"
        )
    print(f"one_shared_constants_map_serves_all={result['one_shared_constants_map_serves_all']}")
    if args.json_out:
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
