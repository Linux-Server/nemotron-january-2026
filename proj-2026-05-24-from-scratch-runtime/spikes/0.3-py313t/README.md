# Spike 0.3 — RETIRED (2026-05-24)

**B4 (free-threaded CPython 3.13t) was rejected by the user:** "not interested in a Python 3.13 approach; do Rust/C++
or do not proceed." The outcome space is now **native Rust/C++ (B1) or STOP** — there is no free-threaded-Python path.

This spike's role — cheaply validating conjunct 2 (is the residual GIL/scheduler-bound, and does removing the GIL
actually lift the knee?) — has moved to **0.1b: a native launch-overlap microbench** (N no-GIL OS threads in C++/Rust,
each driving a CUDA stream that replays the captured encoder graph + a mock decode). See `../0.1-overlap-ablation/` and
`../../PLAN.md` §6 (0.1).

Tombstone kept so the decision history is legible; `probe.py` removed.
