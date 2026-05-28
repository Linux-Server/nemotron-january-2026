#!/usr/bin/env python3
"""Report real continuous-finalize (drop_extra, chunk-T) demand over stt-benchmark.

This uses the verified ``finalize_ref.ContinuousFinalizeRef`` session machinery up
to ``prepare_finalize_inputs``. It intentionally does not run the finalize encoder;
it only measures which exact-T buckets the corpus would route to.

Run from runtime/:
  HF_HUB_OFFLINE=1 ./.venv/bin/python finalize_t_distribution.py 200
"""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter, defaultdict

import torch

from finalize_ref import ContinuousFinalizeRef, load_benchmark_dataset, load_model, load_wav


BUCKET_RE = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)\.pt2$")


def discover_existing_buckets(path: str) -> set[tuple[int, int]]:
    if not os.path.isdir(path):
        return set()
    buckets: set[tuple[int, int]] = set()
    for name in os.listdir(path):
        match = BUCKET_RE.match(name)
        if match:
            buckets.add((int(match.group("drop")), int(match.group("T"))))
    return buckets


def format_range(values: list[int]) -> str:
    if not values:
        return "none"
    return f"{min(values)}..{max(values)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="?", type=int, default=200, help="number of stt-benchmark samples to probe")
    parser.add_argument("--start", type=int, default=0, help="dataset start index")
    parser.add_argument("--buckets-dir", default="artifacts/finalize_buckets")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    print("loading model + stt-benchmark...")
    model = load_model()
    rt = ContinuousFinalizeRef(model)
    ds = load_benchmark_dataset()

    end = min(args.start + args.n, len(ds))
    if args.start < 0 or args.start >= len(ds) or end <= args.start:
        raise SystemExit(f"empty probe range start={args.start} n={args.n} dataset_len={len(ds)}")

    counts: Counter[tuple[int, int]] = Counter()
    by_drop: dict[int, list[int]] = defaultdict(list)
    no_inputs: list[int] = []

    with torch.inference_mode():
        for offset, sample_index in enumerate(range(args.start, end), 1):
            ex = ds[sample_index]
            wav = load_wav(ex)
            session = rt.new_session(f"dist-{sample_index}")
            rt.append_audio(session, wav)
            rt.vad_stop(session)
            fork = rt.build_continuous_finalize_fork(session)
            inputs = rt.prepare_finalize_inputs(fork)
            if inputs is None:
                no_inputs.append(sample_index)
            else:
                drop = int(inputs.drop_extra)
                T = int(inputs.chunk_mel.shape[-1])
                counts[(drop, T)] += 1
                by_drop[drop].append(T)

            if args.progress_every > 0 and offset % args.progress_every == 0:
                print(f"  probed {offset}/{end - args.start} samples...")

    needed = sorted(counts)
    existing = discover_existing_buckets(args.buckets_dir)
    missing = sorted(set(needed) - existing)
    extra_existing = sorted(existing - set(needed))

    print(f"\n=== finalize T distribution: samples {args.start}..{end - 1} (n={end - args.start}) ===")
    if no_inputs:
        print(f"no finalize inputs: {len(no_inputs)} samples; indices={no_inputs}")

    print("\nper-(drop,T) counts:")
    if counts:
        for drop, T in needed:
            print(f"  drop={drop} T={T}: {counts[(drop, T)]}")
    else:
        print("  none")

    drop2 = by_drop.get(2, [])
    drop0 = by_drop.get(0, [])
    print(f"\ndrop=2 continuation T range: {format_range(drop2)} count={len(drop2)}")
    print(f"drop=0 first-chunk T range: {format_range(drop0)} count={len(drop0)}")
    if drop0:
        print(f"drop=0 sorted T values: {sorted(set(drop0))}")

    print(f"\nfull sorted set needed: {needed}")
    if existing:
        print(f"existing buckets: {sorted(existing)}")
        print(f"missing vs existing buckets: {missing}")
        print(f"existing buckets not hit in this probe: {extra_existing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
