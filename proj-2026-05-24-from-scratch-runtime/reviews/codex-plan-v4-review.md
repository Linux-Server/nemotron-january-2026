# PHASE2-PLAN v4 amendment review

Verdict: **GO-with-changes**.

The core v4 inference is directionally sound: under the counted-not-gated event policy, staggered `B_max=4,
W=0, L=0` reaching `N=64` with zero token divergences is enough to keep the F1/funding case alive. The
amendment is not fold-complete, though. It over-hardens defaults from one operating point and under-folds
burst/event/memory caveats that matter before changing production defaults.

Review note: `reviews/B3-L40S-result.md` now exists. I treat it as document evidence, not raw-log
re-derivation. If that artifact post-dates the v4 banner, the items below are still the next required fold.

## Must fold before the defaults/verdict commit

1. **Event policy and burst result are missing from the v4 framing.**

   `B3-L40S-result.md` says the high-N `B=4` staggered rows are token-clean but have event-only divergences
   from `N=40` onward; strict event-clean `B=4` is only `N=36`. The plan's Rules section still says no
   throughput number is trusted until token/event equality holds, while the v4 banner correctly narrows its
   claim to zero token divergences. That policy mismatch needs an explicit fold: either the B3 verdict
   accepts the known event-only drift class as counted-not-gated, or the plan must not use strict
   "0 mismatch" language for the `N>=64` claim.

   The same result file says the required synchronized burst at `N=64` failed keep-up badly (`lag_p95`
   about 1355 ms, `enc_first_lock_p95` about 1607 ms). That means B3-FU-2 is no longer just "small
   follow-up" if the result is final; it is a production-shape risk that must inform admission/start
   shaping and the default cap.

2. **Do not blindly change `active_cap` to a hard `64+` default.**

   `active_cap=40` is stale for the staggered L40S winner, but `64+` is not a safe universal default. Cases
   where 40 or lower can still be right: smaller GPUs, strict event-clean deployments, synchronized-start
   traffic, multi-tenant/MPS boxes, and multiple native instances sharing one L40S. The right production
   stance is closer to **deploy-required explicit cap** with a named L40S profile or dev fallback, not a
   global hard-coded 64.

   Fold suggestion: "production WS requires explicit admission cap; L40S single-instance default/profile may
   be 64 only after B3 verdict plus burst policy; smaller/multi-process targets must override."

3. **The Tier-3 memory conclusion is not yet measured on L40S and may be wrong as written.**

   v4 says Tier 3 shrink makes scheduler overhead `+~5 GiB` and estimates `N=64` ON around 22 GiB. The
   existing L40S result reports `N=64` OFF 21.928 GiB, B4 `W0/L0` ON 33.641 GiB, delta 11.713 GiB. Maybe
   that run did not include the Tier-3 shrink; either way, v4 should not present the `+5 GiB` or `~22 GiB`
   L40S conclusion as measured.

   B3-FU-3 should explicitly capture base resident memory, scheduler scratch/bucket overhead, per-stream
   activation slope, and projected multi-process/MPS packing headroom after the actual Tier-3 build.

4. **The `~N=80-100` dispatcher ceiling is overclaimed.**

   The 5090 `52% dispatcher CPU at N=40` linear extrapolation is useful, not binding proof. L40S dynamics
   differ, and the available L40S doc points at a mix of dispatcher CPU, dispatcher-stream utilization,
   queue depth, gather wait, burst HOL, and `enc_first` lock tail. Call the ceiling a hypothesis to bracket
   in B3-FU-1, not a "known ceiling."

## Defaults and setpoints

- **`window_ms=10 -> 0`: mostly OK, but scope it.** The L40S high-load winner supports `W=0` as the
  production high-load setpoint. The 5090 low-load sweep does not prove `W=0` low-load parity because it
  measured `W=10/L0`; with `lone_timeout=0`, N=1 and N=4 were pure B1 dispatches, so `window_ms` likely did
  not matter there. Add a small W0 low-load smoke or phrase the default as "L40S production high-load
  default; W/L remain configurable until Step 4 real-WS load testing."

- **`B_max=2` as "debug-only" is too strong.** One L40S point plus the microbench ratio is enough to stop
  spending broad sweep cycles on B2, but not enough to remove B2 from the serious toolbox. B2 is still a
  useful control/fallback, especially because the current L40S result says B2 stays event-clean through
  N=40 while B4 high-N rows are token-clean only.

- **Bundle defaults separately from the B3 verdict.** Keep B3 verdict as measurement/history, then land a
  tiny defaults commit after the plan text says exactly which policy is being accepted. That makes rollback
  and review cleaner than mixing "what we measured" with "what production now does."

## Step 3 sizing and WS-tail

- Worker pool sizing for `80-100` should include memory and traffic shape, not just thread count. On 32 vCPU,
  2-3x oversubscription may be fine for I/O-bound workers, but burst results show queue/HOL behavior can
  dominate before CPU thread count is the obvious limiter.

- The proposed WS-tail matrix is underspecified. It says `n_idle in {0,64,96,128}` x `m_streaming in
  {1,8,32,64}`, which can exceed 128 total sockets if interpreted literally and misses idle-only overhead.
  Add at least `n_idle=128,m_streaming=0`, `n_idle=0,m_streaming=128`, and a burst/connect-churn case. Define
  whether the target is total sockets or independent axes.

## B3 follow-ups

- **B3-FU-1:** `{72,80,88,96,112,128}` is a good first bracket. Make it adaptive: if 128 still passes, add
  160/192 or a sparse doubling point rather than pre-committing all high-N cells up front.

- **B3-FU-2:** run burst at both the proposed production cap (`N=64` today) and the true knee/proposed cap
  found by FU-1. If the existing burst failure is accepted, fold it now and rerun only after an admission,
  start-shaping, or HOL mitigation change.

- **B3-FU-3:** include per-stream activation slope, not only one absolute peak. The old `0.035 GiB/stream`
  estimate is useful but should not be the MPS planning basis.

- **B3-FU-4:** defaults should wait on the above policy edits. This is more than two harmless one-line
  changes because `active_cap` encodes deployment risk.

## Progress-table and banner consistency

- The progress table still has B3 as `todo`, F1 as provisional on the old `47-64` projection, and Step 4 as
  at-risk from the old `N=36` math. Once v4 claims realized `64/20 = 3.2x`, those rows should be synchronized
  or explicitly marked "pending final B3 paired verdict."

- Preserve the distinction between the cost-bounded B3 sweep and Step 4 apples-to-apples full-session/real-WS
  proof. The v4 ratio is a strong funding signal, not a substitute for Step 4.

- The B2/B3 full-corpus caveat is still not fully closed in the L40S result as written: the single-stream
  reference is 1000 rows, but the forced/stagger/control cases remain tiny. Do not state full-corpus b2-t1
  closure more broadly than the artifact supports.

## Net

**GO-with-changes.** Fold the event/burst policy and memory/cap corrections before changing production
defaults. The staggered token-clean lift is real enough to proceed with B3 verdict consolidation, but the
amendment should not imply that `active_cap=64+`, strict event correctness, burst robustness, or Tier-3 L40S
memory are already settled.
