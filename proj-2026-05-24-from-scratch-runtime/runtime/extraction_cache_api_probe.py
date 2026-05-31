#!/usr/bin/env python3
"""Read-only Step 1(c) probe: AOTI package loader API and residual extraction sizing.

Run from runtime/:
  HF_HUB_OFFLINE=1 ./.venv/bin/python extraction_cache_api_probe.py
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
STEADY_B = ROOT / "steady_b_artifacts"
SOURCE_URL = (
    "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/"
    "torch/csrc/inductor/aoti_package/model_package_loader.cpp"
)


def numbered_hits(text: str, needles: tuple[str, ...], context: int = 0) -> list[dict[str, Any]]:
    lines = text.splitlines()
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for idx, line in enumerate(lines, 1):
        if any(needle in line for needle in needles):
            start = max(1, idx - context)
            end = min(len(lines), idx + context)
            for lineno in range(start, end + 1):
                if lineno in seen:
                    continue
                seen.add(lineno)
                out.append({"line": lineno, "text": lines[lineno - 1]})
    return out


def fetch_source() -> str:
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        return response.read().decode("utf-8")


def mib(nbytes: int) -> float:
    return nbytes / (1024.0 * 1024.0)


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def package_sizes() -> dict[str, Any]:
    enc_first = file_size(ART / "enc_first_aoti.pt2")
    shared_ts = file_size(ART / "finalize_shared_weights.ts")
    shared_pt = file_size(ART / "finalize_shared_weights.pt")
    enc_steady_inline = file_size(ART / "enc_steady_aoti.pt2")
    steady_stripped = sorted(
        p for p in STEADY_B.glob("enc_steady_aoti_b*.pt2") if ".full." not in p.name
    )
    finalize_stripped = sorted((ART / "stripped_finalize_buckets").glob("*.pt2"))
    steady_total = sum(file_size(p) for p in steady_stripped)
    finalize_total = sum(file_size(p) for p in finalize_stripped)
    small_total = enc_first + steady_total + finalize_total
    return {
        "bytes": {
            "enc_first_aoti": enc_first,
            "steady_stripped_total": steady_total,
            "steady_stripped_count": len(steady_stripped),
            "finalize_stripped_total": finalize_total,
            "finalize_stripped_count": len(finalize_stripped),
            "small_stripped_aoti_total": small_total,
            "finalize_shared_weights_ts": shared_ts,
            "finalize_shared_weights_pt": shared_pt,
            "current_inline_enc_steady_aoti_pre_unify": enc_steady_inline,
        },
        "mib": {
            "enc_first_aoti": mib(enc_first),
            "steady_stripped_total": mib(steady_total),
            "finalize_stripped_total": mib(finalize_total),
            "small_stripped_aoti_total": mib(small_total),
            "finalize_shared_weights_ts": mib(shared_ts),
            "finalize_shared_weights_pt": mib(shared_pt),
            "current_inline_enc_steady_aoti_pre_unify": mib(enc_steady_inline),
        },
        "files": {
            "steady_stripped": [str(p.relative_to(ROOT)) for p in steady_stripped],
            "finalize_stripped": [str(p.relative_to(ROOT)) for p in finalize_stripped],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    torch_root = Path(torch.__file__).resolve().parent
    header = torch_root / "include/torch/csrc/inductor/aoti_package/model_package_loader.h"
    header_text = header.read_text(encoding="utf-8")
    source_text = fetch_source()

    ctor_surface = numbered_hits(
        header_text,
        ("AOTIModelPackageLoader(", "load_constants(", "get_constant_fqns()", "temp_dir_"),
        context=0,
    )
    private_temp = bool(re.search(r"private:\s+std::string temp_dir_;", header_text, flags=re.S))
    public_preextracted_ctor = bool(
        re.search(
            r"AOTIModelPackageLoader\s*\([^;{}]*(?:pre.?extract|extract.*dir|directory|temp_dir)",
            header_text,
            flags=re.I | re.S,
        )
    )
    source_hits = numbered_hits(
        source_text,
        ("create_temp_dir()", '"/tmp/XXXXXX"', "mkdtemp", "temp_dir_ = create_temp_dir()"),
        context=2,
    )
    source_mentions_tmpdir = "TMPDIR" in source_text
    sizes = package_sizes()
    result = {
        "torch": str(torch.__version__),
        "header": str(header),
        "source_url": SOURCE_URL,
        "header_hits": ctor_surface,
        "temp_dir_private": private_temp,
        "public_preextracted_dir_ctor": public_preextracted_ctor,
        "source_hits": source_hits,
        "source_mentions_TMPDIR": source_mentions_tmpdir,
        "sizes": sizes,
        "recommendation": {
            "speed_cache_go": False,
            "tmp_hygiene_go": True,
            "reason": (
                "Post-unify extraction payload is only the stripped AOTI set; the 2.48 GB "
                "shared encoder blob is TorchScript-loaded, not AOTI-extracted."
            ),
        },
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print("=== AOTI extraction-cache API probe ===")
    print(f"torch={torch.__version__}")
    print(f"header={header}")
    print(
        "public_preextracted_dir_ctor="
        f"{public_preextracted_ctor} temp_dir_private={private_temp}"
    )
    print(f"source_url={SOURCE_URL}")
    print(f"source_mentions_TMPDIR={source_mentions_tmpdir}")
    print("header relevant lines:")
    for hit in ctor_surface:
        print(f"  {hit['line']}: {hit['text']}")
    print("source relevant lines:")
    for hit in source_hits:
        print(f"  {hit['line']}: {hit['text']}")
    b = sizes["bytes"]
    m = sizes["mib"]
    print(
        "post-unify stripped AOTI extraction payload: "
        f"enc_first={m['enc_first_aoti']:.1f} MiB, "
        f"steady_b={m['steady_stripped_total']:.1f} MiB/{b['steady_stripped_count']} files, "
        f"finalize={m['finalize_stripped_total']:.1f} MiB/{b['finalize_stripped_count']} files, "
        f"total={m['small_stripped_aoti_total']:.1f} MiB"
    )
    print(
        "shared weights are not residual AOTI extraction: "
        f"finalize_shared_weights.ts={m['finalize_shared_weights_ts']:.1f} MiB "
        f"(.pt={m['finalize_shared_weights_pt']:.1f} MiB)"
    )
    print("recommendation: speed_cache_go=False tmp_hygiene_go=True")
    if args.json_out:
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
