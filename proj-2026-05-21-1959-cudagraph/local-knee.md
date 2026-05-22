# Step 5 — local keep-up knee, CUDA-graph encoder ON vs OFF

Date: 2026-05-22. Local RTX 5090 (Blackwell), `lanes=1`, continuous batching on
(`NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32`),
silence0_warm200, rc1. Loadgen = `ec2-bench/ec2_loadgen.py` (staggered, START_JITTER 400 ms,
24 distinct clips cycled to N). CUDA-graph K=16 (`NEMOTRON_ENCODER_CUDAGRAPH=1`).
Runner: `step5_local_knee.sh`. Knee = max N with proc-lag p95 < 500 ms, 0 errors.

## Result: knee 48 -> 56 (+17%), and lower tail latency at matched load

| N | OFF lag p95 | OFF keepup | ON lag p95 | ON keepup |
|--:|--:|:--:|--:|:--:|
| 40 | 331 | YES | 278 | YES |
| 48 | 229 | YES | **151** | YES |
| 56 | **608** | no | **254** | **YES** |
| 64 | 703 | no | 758 | no |

- **Keep-up knee: OFF=48, ON=56 (+17%).** N=56 flips from over-budget (608 ms) to keeping up (254 ms).
- **Lower tail at matched load:** e.g. N=48 lag p95 229 -> 151 ms (-34%); the cheaper encoder call
  reduces per-chunk latency, not just raises the ceiling.
- (The OFF knee here is 48 vs the ~56 cited from the earlier coloc/in-phase harness — different loadgen +
  staggering. The clean apples-to-apples is OFF-vs-ON on the *same* harness: 48 -> 56.)

## Engagement: graphs fire on essentially every call, mostly small B
Sampled B distribution (from the every-50-replays status log), `replays=8450 fallbacks=0`:

```
B=1:67  B=2:45  B=3:22  B=4:12  B=5:14  B=6:6  B=7..16: ~1-2 each
```

- **avg B ≈ 2-3**, with 0 eager fallbacks -> every steady-bucket encoder call hit a captured graph.
- The realtime regime is dominated by small B (1-5), which is exactly where launch-dispatch overhead is the
  largest fraction of the call -> where the graph speedup is biggest (the probe's ~1.36x@B1, diminishing toward
  large B). So the lever lands where it matters for realtime.

## Read-through to the cloud retest (Step 6)
The 5090 is a *fast* desktop core; its per-call launch overhead is already small, so +17% is the *floor* of the
benefit. The cloud L4/L40S have slower single-thread launch dispatch (more launch-bound), so collapsing the
launch with a graph should yield a **larger relative lift** there — and, under the tight p50<250/p95<300 budget,
the lower tail latency at matched load (seen here) is the property that should let more streams fit the budget.
