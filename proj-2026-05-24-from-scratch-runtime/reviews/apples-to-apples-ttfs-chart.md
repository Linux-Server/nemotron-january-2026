# Apples-to-apples TTFS chart — native runtime vs Python server (L40S, same box)

**Date:** 2026-05-27 | Same g6e/L40S, same metric (`ttfs = vad_stop→final`), same synchronized-burst (no-stagger) arrival. Native = 8 sessions/worker, no-stagger (`l40s_w3_logs/` + `rerun_logs/`). Python = single proc, LANES=2, all prod levers, **rounds=10** (n=80–280/level) (`spy_p99.json`). SLO budget: ttfs **p95≤175 ∧ p99≤250 ms**, keep-up lag p95<500.

## TTFS p50 / p95 / p99 (ms) by concurrency

| conc | Native p50 | Native p95 | Native p99 | Python p50 | Python p95 | Python p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 9.8 | 10.6 | 11.4 | — | — | — |
| 8  | 11.1 | 16.8 | 21.3 | 42 | 57 | 90 |
| 16 | 13.4 | 41.2 | 62.8 | 48 | 150 | 192 |
| 20 | 13.9 | 48.9 | 75.9 | 53 | **174** | 212 |
| 22 | — | — | — | 58 | **177** ✗ | 254 ✗ |
| 24 | 16.3 | 53.5 | 91.3 | 58 | 227 ✗ | 382 ✗ |
| 28 | — | — | — | 109 | 701 ✗ | 1085 ✗ |
| 32 | 20.6 | 75.6 | 109.8 | *(past knee)* | | |
| 36 | 26.0 | 90.9 | **146.8** | | | |
| 40 | 62.8 | 361 | 720 ✗ | | | |

## Knees (last conc passing p95≤175 ∧ p99≤250 ∧ lag p95<500)
- **Native: N=36** (p99 146.8, lag −35) — N=40 first fail.
- **Python: N=20** (p95 174, p99 212, lag 194) — N=22 first fail (p95 177, p99 254).
- **Density ratio = 36 / 20 = 1.8×.** (Python S_py=20 is now **robust** — rounds=10, p99, monotonic — confirming the earlier noisy repeats=2 estimate; *not* ≤18, *not* ≥22.)

## ⚠️ The measurement-plane caveat (red-team MF-1 — do not over-read the latency gap)
**Native ttfs is server-side / pre-WS** (Step-3 WS server not yet built); **Python ttfs is client-observed over the WebSocket** (`ec2_loadgen.py:77`, `now − vad_stop_t`). So the two columns are **not on the same plane**:
- The low-load gap (conc-8: native p50 **11** vs Python **42** = ~31ms) is ≈ the **WS round-trip + client/event-loop tax that native has not paid yet.** When native gets its WS server, its client-observed ttfs rises by ~that offset.
- ⟹ **The per-conc latency gap is INFLATED.** The robust, plane-insensitive headline is the **density ratio (36/20 = 1.8×)** — where each engine *saturates* is far less sensitive to a constant WS offset than the absolute latencies are.
- This is exactly why the paired red-team made **Step 1b.5 S_py_LOCK** (WS-matched or WS-subtracted) blocking: the honest, WS-paid multiplier is **≤1.8×**, and this chart's latency columns must not be cited as a 4–10× native latency win.

## What the chart DOES establish
1. **S_py = 20 is robust** (the noisy repeats=2 estimate holds under rounds=10 + p99) → the **1.8× at-bar / zero-margin** verdict stands; pushing >36 remains load-bearing for Step-4 (MF-1 branch S_py∈[19,21] → proceed to 1c).
2. **Native is genuinely denser** (36 vs 20) AND lower-latency at matched concurrency even before subtracting the WS tax — the GIL-break lets the GPU saturate ~1.8× further before the keep-up cliff.
3. Both curves are **smooth + monotonic** (native through 36, Python through 24) → the knees are real, not single-sample artifacts (the earlier repeats=2 non-monotonicity is gone).

## Still pending for the FULL MF-1 lock (this chart is the first half)
- WS-matched native (or WS-subtracted) so the columns share a plane.
- The multi-turn manifest (this is one-utterance-per-connection — the known coverage gap).
