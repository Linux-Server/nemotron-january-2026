# Leaderboard decomposition run STALLED (b4wp7pcn0) — incomplete + a possible prod hang

## What happened
The cloud conc-10 leaderboard decomposition (L40S g6e, full PROD config: cudagraph + lanes=2 + BARRIER_DRAIN=1 +
BATCH_FINALIZE=1 + FINALIZE_PROFILE=1, real stt-benchmark client over WAN) **hung ~30 s in (after ~68 finalizes).**
Server went silent at 02:35:08 UTC and stayed dead ~1h39m (no watchdog → ~$4 + 1.5 hr burned before I caught it on
the user's "Progress?" prompt). Box terminated, no orphans. Full server log: `leaderboard_decomp_STALLED_srv.log`.

## Stall signature (CUDA-level hang, not a Python deadlock)
`model_batch_ms` over the last 12 steady batches: 19.9, 23.1, 14.1, 11.3, 11.3, 17.0, 20.7, 18.4, 21.7, 11.2, 12.2,
**1188.16** → then total silence. One steady batch (batch_size=1) jumped 50-100× to 1.2 s, then the server froze
completely. A 1.2 s *model* spike + freeze points to a CUDA/GPU-level hang (a kernel/graph-replay that didn't
complete, or a stream deadlock), NOT a scheduler/asyncio deadlock. No traceback, no OOM (cuda_reserved stable
~9.18 GB).

## Two readings (need a re-run to distinguish)
1. **Real interaction bug** — the full prod config (cudagraph + lanes + barrier-drain + batch-finalize) had never
   been stress-tested *together* under sustained WAN conc-10 (the 274/401 bench ran WITHOUT barrier-drain/batch-
   finalize; the K=4 gate ran WITHOUT barrier-drain). If it reproduces, this is a PROD reliability bug (launch_multiproc
   ships this exact config) and outranks the latency tail.
2. **Box/driver flake** — a one-off GPU glitch on that g6e instance.

## Consequences
- The leaderboard decomposition (queue_wait vs lock_wait vs model vs transmission) is NOT obtained.
- NEW open thread: a possible prod-config hang under load.

## Next (proposed)
- **Add a watchdog** to `bench_client_wan.sh`: if the server log is silent > N min (or the run exceeds a wall cap),
  kill + terminate immediately. (Prevents another 1.5 hr burn.)
- **Re-run** the decomposition with the watchdog + a smaller sample count (e.g., 300-400, enough for P95). Dual-purpose:
  reproduces the hang (→ real bug, pivot to debugging it) OR completes (→ flake, and we get the decomposition).
- If it hangs again: bisect the prod flags (cudagraph / barrier-drain / batch-finalize) to localize, with FORK_ASSERT
  and a CUDA-hang watchdog.

## UPDATE: did NOT reproduce locally (RTX5090)
`local_hang_repro.sh` ran the EXACT full prod config (cudagraph+lanes2+barrier-drain+batch-finalize) under
sustained conc-10 for 300 utterances / **668 finalizes — clean, no hang** (TTFS 14/20). So the cloud freeze is NOT
a deterministic config bug; it is cloud-specific (L40S/g6e GPU-driver transient) or a WAN-timing race — most likely
a one-off flake. NEXT: watchdogged cloud re-run (bench_client_wan.sh now has a server-silence watchdog + 25min
timeout + FAULTHANDLER passthrough so a hang is caught in ~4min AND its stacks are dumped). Completes -> flake +
decomposition obtained; hangs again -> reproducible cloud issue -> bisect the prod flags.
