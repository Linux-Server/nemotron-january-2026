# Decision memo — from-scratch runtime (LIVING; not the final 0.4 memo)

Status: **pre-Python-plan.** This records the cheap Wave-1 evidence gathered with zero GPU/cloud spend. The binding 0.4
decision + the 0.0 numeric threshold are **DEFERRED until the Python plan (`proj-2026-05-24-0859`) lands** (user
decision 2026-05-24). Outcome space: **native Rust/C++ (B1) or STOP** (B4/B5 rejected).

---

## 0.0-pre — Residual-CEILING arithmetic (free; the cheapest kill)

**Upper-bounds the prize *before* any measurement.** Even assuming every native gate (0.1/0.6a/0.9/0.11) passes:

| | in-budget streams/box | source |
|---|---:|---|
| today (pre-Python) | ~20 | memory `roofline-and-real-limit` / `deployment-target-sagemaker` |
| **post-Python (K=4)** | **~28** | `proj-2026-05-24-0859/PLAN.md:11-14` — the baseline the native build must beat |
| native aspiration | ~40–48 | roofline projection, **triple-conditional** (shared weights AND single-context overlap AND graph-pool fits) |

**Best-case native upside ≈ 48 − 28 ≈ 20 streams/box** (~1.7× density), and only if 0.1 + 0.9 + 0.11 *all* clear —
otherwise less or zero.

**Cost against it:** ~40–60 eng-wk to BUILD (§9) **plus a permanent dual-stack carry** (every NeMo upgrade / model swap /
CUDA bump now hits two codebases — not in the 40–60). p50 is immovable, so there is **zero p50 upside**; the entire value
is density + tail.

**Reading:** a ~1.7× density ceiling (triple-conditional) with no p50 benefit, against an engineer-year + ongoing carry,
is a **thin** prize. L4 is already the cheapest $/stream and scales horizontally — adding boxes is the obvious
alternative to a density rewrite. This does **not** auto-STOP (the user affirmed density is a strategic priority), but it
sets the bar the post-Python residual must clear and says the threshold should be set with eyes open.

## 0.5 — Batching / 3–5× throughput (synthetic + existing-data; see `spikes/0.5-batching-sim/FINDINGS.md`)

At the deployed 8 ms window with realistic independent/bursty arrivals, **mean batch B ≈ 1.5–2.1, B=1 is 36–63%** — at
in-budget per-process concurrency this converges with the measured **86% B=1**. Filling B more requires either adding
latency (forbidden for steady) or more concurrency (capped). **→ the 3–5× steady-throughput claim is effectively dead**,
and the steady-graph *throughput* rationale with it. **Consequence for 0.0-pre:** a chunk of the headline value
evaporates — the remaining native value is **shared-weights density + tail only** (0.9/0.1/0.11), making the ~20
streams/box ceiling above *optimistic*. The real expected upside is likely **below** 20.

## Running conclusion (provisional)
- The prize is thin and shrinking under scrutiny: ~1.7× density ceiling (triple-conditional, likely lower after 0.5),
  no p50, vs an engineer-year + carry.
- **Not a STOP yet** (user: density is a priority; 0.0 number deferred), but the **threshold the post-Python residual
  must clear should be set before any GPU spend**, and the bar is high relative to the ceiling.
- **Next gating evidence:** the post-Python baseline (sets the real residual) + conjunct 2 (0.1 + the Python Step-5 GIL
  attribution: is the wall decode-GIL-bound/native-fixable, or MPS/bandwidth-bound/STOP).

## Pre-registered thresholds + the filled decision tree
See `../spikes/decision-template.md` — fill the numeric thresholds (proposed starting values there) and the tree at 0.4,
*after* the Python baseline exists.
