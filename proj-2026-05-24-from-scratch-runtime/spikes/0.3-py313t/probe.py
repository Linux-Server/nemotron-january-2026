#!/usr/bin/env python3
"""Spike 0.3 stage-1 feasibility probe — does the stack run free-threaded? SKELETON.

Run under a free-threaded CPython 3.13t interpreter. This only answers "does it import + run + fan out across real
threads without the GIL," NOT the thesis (stage 2 = off-event-loop dispatcher + tail measurement, see README.md).

BLOCKED: needs a py3.13t env with PyTorch+NeMo built for it.
"""
from __future__ import annotations

import sys


def check_free_threaded() -> bool:
    """True iff running on a free-threaded (no-GIL) CPython build."""
    # sys._is_gil_enabled() exists on 3.13t; absent => not free-threaded.
    is_ft = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()  # type: ignore[attr-defined]
    print(f"python={sys.version.split()[0]} free_threaded={is_ft}")
    return is_ft


def check_imports() -> None:
    """Import torch + NeMo and report whether their C-extensions declare GIL-free support."""
    import torch  # noqa: F401
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    # NeMo import is the real maturity risk — many transitive C-extensions.
    import nemo.collections.asr as nemo_asr  # noqa: F401
    print("nemo import: OK")


def check_thread_fanout() -> None:
    """Fan out N threads each doing a trivial CUDA op; under no-GIL these should truly overlap.
    BLOCKED: replace the placeholder with a real streaming-chunk call on the loaded model."""
    raise NotImplementedError("BLOCKED: load the model and run one streaming chunk per thread; needs the stack")


def main() -> None:
    ft = check_free_threaded()
    if not ft:
        print("WARNING: not a free-threaded interpreter — stage-1 result is not representative of B4")
    check_imports()
    check_thread_fanout()


if __name__ == "__main__":
    main()
