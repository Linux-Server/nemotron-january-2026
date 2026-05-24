#!/usr/bin/env python3
"""Spike 0.5 — trace-driven batching simulator (SKELETON; synthetic mode runnable now, no GPU).

Replays per-tick readiness traces through a simplified scheduler model and reports the achievable batch-B distribution
+ a graph-bucket capacity model. Real traces are BLOCKED on server instrumentation (see README.md / instrumentation
spec); --synthetic exercises the model on generated arrivals so thresholds can be pre-registered.

Pure Python, no model, no GPU, no cloud.
"""
from __future__ import annotations

import argparse
import collections
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchKey:
    target_lang: object
    keep_all_outputs: bool
    drop_extra: int
    chunk_T: int
    decoder_state_fresh: bool  # server.py:4789-4812 forbids mixing fresh/established in one batch


@dataclass
class Tick:
    t_ms: float
    session_id: str
    key: BatchKey
    deadline_ms: float
    lane_affinity: int | None


def gen_synthetic(n_sessions: int, ticks_per_session: int, arrival="poisson", seed=0) -> list[Tick]:
    """Generate synthetic readiness ticks. 'arrival' in {poisson, bursty, phase_aligned}."""
    rng = random.Random(seed)
    out: list[Tick] = []
    for s in range(n_sessions):
        # phase_aligned -> all sessions ready near the same t (best case for B-fill);
        # poisson -> independent; bursty -> clustered. This is the crux variable (memory: phasing 115 vs 56).
        base = 0.0 if arrival == "phase_aligned" else rng.uniform(0, 160.0)
        for c in range(ticks_per_session):
            jitter = rng.expovariate(1 / 20.0) if arrival == "poisson" else (rng.uniform(0, 5) if arrival == "phase_aligned" else rng.choice([0, 0, 0, 40]))
            t = base + c * 160.0 + jitter  # 160 ms steady cadence
            key = BatchKey(target_lang=None, keep_all_outputs=False, drop_extra=2, chunk_T=25, decoder_state_fresh=(c == 0))
            out.append(Tick(t_ms=t, session_id=f"s{s}", key=key, deadline_ms=t + 10.0, lane_affinity=None))
    out.sort(key=lambda x: x.t_ms)
    return out


def simulate(ticks: list[Tick], batch_max_size: int, batch_window_ms: float) -> dict:
    """Greedy window batching honoring the batch key (incl. fresh/established split) and a max-wait window.
    Returns the B distribution. (Lane assignment / graph hit-rate are layered on next.)"""
    b_hist: collections.Counter = collections.Counter()
    i, n = 0, len(ticks)
    while i < n:
        t0 = ticks[i].t_ms
        window = [ticks[i]]
        j = i + 1
        while j < n and ticks[j].t_ms <= t0 + batch_window_ms:
            window.append(ticks[j])
            j += 1
        # group by full batch key (cannot mix fresh/established — batch_primitives.py:100-139)
        groups: dict[BatchKey, int] = collections.Counter(t.key for t in window)
        for _key, count in groups.items():
            remaining = count
            while remaining > 0:
                b = min(remaining, batch_max_size)
                b_hist[b] += 1
                remaining -= b
        i = j
    total = sum(b_hist.values())
    weighted = sum(b * c for b, c in b_hist.items())
    return {
        "B_histogram": dict(sorted(b_hist.items())),
        "mean_B": weighted / total if total else 0.0,
        "frac_B1": b_hist.get(1, 0) / total if total else 0.0,
    }


def graph_capacity_model(b_hist: dict[int, int], per_b_graph_mb: dict[int, float], lanes: int, gpu_free_mb: float) -> dict:
    """Exact-B graph hit-rate + resident pool memory at K×lanes vs available. per_b_graph_mb is TBM on GPU (0.11)."""
    total = sum(b_hist.values()) or 1
    captured_bs = set(per_b_graph_mb)  # which B we'd actually capture graphs for
    hit = sum(c for b, c in b_hist.items() if b in captured_bs) / total
    resident_mb = lanes * sum(per_b_graph_mb.get(b, 0.0) for b in captured_bs)
    return {"hit_rate": hit, "eager_fallback": 1 - hit, "resident_pool_mb": resident_mb,
            "fits": resident_mb <= gpu_free_mb}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--arrival", default="poisson", choices=["poisson", "bursty", "phase_aligned"])
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--batch-max-size", type=int, default=32)
    ap.add_argument("--window-ms", type=float, default=10.0)
    args = ap.parse_args()
    if not args.synthetic:
        raise SystemExit("real-trace mode BLOCKED on server instrumentation; use --synthetic (see README.md)")
    ticks = gen_synthetic(args.sessions, ticks_per_session=20, arrival=args.arrival)
    res = simulate(ticks, args.batch_max_size, args.window_ms)
    print(f"arrival={args.arrival} mean_B={res['mean_B']:.2f} frac_B1={res['frac_B1']:.2%}")
    print(f"B_histogram={res['B_histogram']}")
    # graph capacity needs per-B graph MB from spike 0.11 (TBM on GPU); illustrative placeholder omitted.


if __name__ == "__main__":
    main()
