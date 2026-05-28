#!/usr/bin/env python3
"""Replay density FILL_TRACE lines through a max-wait batching policy."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


FILL_RE = re.compile(r"\bFILL_TRACE\b.*\bworker=(\d+)\b.*\bt_ready_ns=(\d+)\b.*\bkind=steady\b")
RUN_RE = re.compile(r"DENSITY 1a RUN START: N=(\d+)\b")


def read_lines(paths: list[Path]) -> Iterable[str]:
    if not paths:
        yield from sys.stdin
        return
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle


def parse_traces(paths: list[Path]) -> dict[str, list[int]]:
    by_run: dict[str, list[int]] = defaultdict(list)
    current_run = "all"
    for line in read_lines(paths):
        run_match = RUN_RE.search(line)
        if run_match:
            current_run = f"N={run_match.group(1)}"
            continue
        fill_match = FILL_RE.search(line)
        if fill_match:
            by_run[current_run].append(int(fill_match.group(2)))
    return dict(by_run)


def percentile_nearest(values: list[int], pct: float) -> int:
    if not values:
        return 0
    idx = max(0, min(len(values) - 1, math.ceil(pct * len(values)) - 1))
    return sorted(values)[idx]


def simulate_batches(times_ns: list[int], window_ms: float, bmax: int) -> tuple[list[int], list[int]]:
    times = sorted(times_ns)
    window_ns = int(round(window_ms * 1_000_000.0))
    batches: list[int] = []
    dispatch_ns: list[int] = []
    i = 0
    while i < len(times):
        oldest = times[i]
        j = i + 1
        while j < len(times) and j - i < bmax and times[j] - oldest <= window_ns:
            j += 1
        batch_size = j - i
        batches.append(batch_size)
        if batch_size >= bmax:
            dispatch_ns.append(times[j - 1])
        else:
            dispatch_ns.append(oldest + window_ns)
        i = j
    return batches, dispatch_ns


def format_dist(batches: list[int], bmax: int) -> str:
    counts = Counter(batches)
    return ",".join(f"{b}:{counts.get(b, 0)}" for b in range(1, bmax + 1))


def report_run(run_name: str, times_ns: list[int], windows_ms: list[float], bmax: int) -> None:
    times = sorted(times_ns)
    if not times:
        print(f"{run_name} chunks=0")
        return
    trace_span_s = max((times[-1] - times[0]) / 1_000_000_000.0, 1e-9)
    print(f"{run_name} chunks={len(times)} trace_span_s={trace_span_s:.3f} arrival_chunks_s={len(times) / trace_span_s:.3f}")
    median_pass = True
    for window_ms in windows_ms:
        batches, dispatch_ns = simulate_batches(times, window_ms, bmax)
        mean_b = statistics.fmean(batches) if batches else 0.0
        p50_b = statistics.median(batches) if batches else 0.0
        p95_b = percentile_nearest(batches, 0.95)
        pct_b1 = 100.0 * Counter(batches).get(1, 0) / len(batches) if batches else 0.0
        sim_span_s = max((dispatch_ns[-1] - times[0]) / 1_000_000_000.0, 1e-9) if dispatch_ns else 1e-9
        window_chunks_s = mean_b * 1000.0 / window_ms
        if window_ms in (8.0, 12.0) and p50_b < 2.5:
            median_pass = False
        print(
            f"  W_ms={window_ms:g} Bmax={bmax} batches={len(batches)} "
            f"mean_B={mean_b:.3f} p50_B={p50_b:.3f} p95_B={p95_b} "
            f"pct_B1={pct_b1:.1f} window_chunks_s={window_chunks_s:.3f} "
            f"sim_chunks_s={len(times) / sim_span_s:.3f} dist={format_dist(batches, bmax)}"
        )
    print(f"  answer_median_B_ge_2.5_at_8_12ms={'yes' if median_pass else 'no'}")


def parse_windows(text: str) -> list[float]:
    windows = [float(item) for item in text.split(",") if item]
    if not windows or any(window <= 0.0 for window in windows):
        raise argparse.ArgumentTypeError("--windows-ms must contain positive values")
    return windows


def run_sort_key(name: str) -> tuple[int, int | str]:
    match = re.fullmatch(r"N=(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, help="density run logs; reads stdin if omitted")
    parser.add_argument("--windows-ms", type=parse_windows, default=parse_windows("4,8,12,16"))
    parser.add_argument("--bmax", type=int, default=8)
    args = parser.parse_args()
    if args.bmax <= 0:
        parser.error("--bmax must be positive")

    by_run = parse_traces(args.logs)
    if not by_run:
        print("no FILL_TRACE lines found", file=sys.stderr)
        return 1
    for run_name in sorted(by_run, key=run_sort_key):
        report_run(run_name, by_run[run_name], args.windows_ms, args.bmax)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
