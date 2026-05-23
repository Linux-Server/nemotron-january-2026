# Scheduler-hang fix: verification + a second (deeper) residual

## #1 SCHEDULER LIVELOCK — FIXED + VERIFIED (commit d8a98f3, cooperative yield)
Root cause: `_scheduler_loop` `if progressed: continue` with no yield; the batched-barrier path keeps
`progressed=True` synchronously (barrier-pending session perpetually re-marked ready -> ready-pass returns progress
every pass) -> the event loop is starved -> server freeze. WAN-timing-triggered (NOT a Python version issue: the
cloud hung on BOTH 3.11 and 3.12; local 3.12 loopback didn't trigger it until overload/jitter).

Verification:
- **Cloud (3.11 + WAN, the original reproducing env that froze 3x without the fix): COMPLETES THE FULL 1000 CLEAN**
  with the fix (1806 finalize records, box terminated, no orphans). ✓
- **Local A/B (same machine, same battery):** `NEMOTRON_SCHED_NO_YIELD=1` (pre-fix) froze at overload-32 (784
  finalizes), stuck in `_scheduler_drain_once_batched_barrier` ready-pass. **Fix-on cleared overload-32 + all
  in-phase configs (ran to 2796 finalizes).** ✓
The fix is necessary AND it resolves the production-load hang.

## #2 RESIDUAL — CUDA-level stall under EXTREME overload (separate bug, lower priority)
The fix-on battery hung at `WAN-jitter-24` (conc-24 + 400ms stream jitter, ~3x the conc-16 knee). Signature is
DIFFERENT from #1:
- `model_batch_ms`: 7.87 -> **18733 -> 26538** (sudden 8ms->18-26s jump = a CUDA op that stopped returning, not
  gradual saturation; not the busy-spin).
- Stacks: **event loop HEALTHY** (`Current thread` in `selectors.select` — the yield fix held), but a **lane worker
  thread is wedged in `scatter_cache_row` (batch_primitives.py:95) <- `_process_ready_batch` (server.py:7835) <-
  `_run_scheduler_model_lane_call_sync`**. A GPU/model-call stall on a lane, NOT the scheduler.

Severity: **does NOT trigger at the production load** — the cloud at conc-10+WAN completed clean. Needs ~3x the
per-proc knee + heavy jitter. A graceful-degradation-past-capacity gap, not a production-path hang (prod LB maxconn
caps per-proc connections). Repro: `local_stress_battery.sh` reaches it at WAN-jitter-24 (fix-on).

Hypotheses for #2 (to investigate if we pursue robustness past capacity): cudagraph replay under extreme
concurrency; a cross-lane CUDA stream/memory contention in the scatter; or raw GPU saturation wedging. Quick test:
re-run WAN-jitter-24 with NEMOTRON_ENCODER_CUDAGRAPH=0 (does cudagraph cause it?). Mitigation regardless: admission
control / per-proc maxconn so a process is never driven 3x past its knee.

## State
- #1 fixed + verified -> the production/leaderboard hang is resolved.
- Cloud fix-on run COMPLETED -> leaderboard decomposition data now in hand (1806 records +
  ec2-bench/leaderboard_decomp_prod_l40s_full1000_c10.{records,srvlog}) -> task #60 unblocked.
- #2 recorded as an extreme-overload robustness follow-up.
