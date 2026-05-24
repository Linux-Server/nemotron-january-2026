#!/usr/bin/env python3
"""Spike 0.1 ablation-matrix harness — SKELETON (does not run any model).

Drives the existing benchmark scripts under a matrix of serializer toggles to isolate which one dominates the
finalize+steady overlap, and compares single-process / MPS / multi-process. See README.md.

BLOCKED: requires (1) the post-Python baseline landed, (2) a GPU (5090 / EC2 L4/L40S), (3) pre-registered thresholds.
This file intentionally contains NO model code and makes NO cloud calls; it is the orchestration skeleton only.
"""
from __future__ import annotations

import dataclasses
import itertools
from typing import Optional

# --- The ablation axes (env-toggle name -> off/on values). Filled to match the real flags at run time. ---
AXES: dict[str, tuple[str, str]] = {
    # "NEMOTRON_BATCH_FINALIZE": ("0", "1"),          # pinned-lane vs inference_lock fallback (server.py:6755 vs 6779)
    # "NEMOTRON_EXCLUSIVE_GATE":  ("off", "on"),       # server.py:3213-3233   (flag TBD)
    # "NEMOTRON_LANE_SYNC":       ("event", "stream"), # CUDA event vs stream.synchronize (server.py:3175)
    # "FINALIZE_LANE":            ("cross", "same"),   # affinity (server.py:3295 / 6711)
    # "PROCESS_SHAPE":            ("single", "mps"),   # + "multiproc" handled separately
}

PROCESS_SHAPES = ("single_context", "mps", "multiproc")  # deploy/launch_multiproc.sh:57-68


@dataclasses.dataclass
class PreRegisteredThresholds:
    """Frozen BEFORE data collection (see ../decision-template.md). None = not yet registered -> refuse to run."""
    min_overlap_factor_vs_python: Optional[float] = None
    max_queue_lane_wait_ms: Optional[float] = None
    max_added_latency_ms: Optional[float] = None

    def assert_registered(self) -> None:
        missing = [f.name for f in dataclasses.fields(self) if getattr(self, f.name) is None]
        if missing:
            raise SystemExit(f"REFUSING TO RUN: pre-register thresholds first (missing: {missing}) — see decision-template.md")


def enumerate_runs():
    """Cartesian product of the toggles × process shapes. One bench invocation per combo."""
    axis_names = list(AXES)
    combos = itertools.product(*(AXES[n] for n in axis_names)) if axis_names else [()]
    for shape in PROCESS_SHAPES:
        for combo in combos:
            yield {"process_shape": shape, **dict(zip(axis_names, combo))}


def run_one(config: dict) -> dict:
    """BLOCKED: shell out to ec2-bench/bench_prod_sweep.sh with `config`, parse the timing schema, return
    {per_lane_cuda_event_timeline, queue_wait, lane_wait, finalize_overlap_factor, knee}. Not implemented until
    GPU + post-Python baseline are available."""
    raise NotImplementedError("BLOCKED on GPU + post-Python baseline; see README.md")


def main() -> None:
    thresholds = PreRegisteredThresholds()  # fill from decision-template.md, then this stops refusing
    thresholds.assert_registered()
    results = [run_one(c) for c in enumerate_runs()]
    # TODO: write results table; flag the dominant serializer; evaluate single-process overlap vs thresholds.
    del results


if __name__ == "__main__":
    main()
