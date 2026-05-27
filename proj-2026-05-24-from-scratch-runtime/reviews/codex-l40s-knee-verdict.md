# Codex adversarial verdict - L40S Step-1b density knee

## Verdict

**GO for Step-1b under the stated gate.** The certified native-runtime knee is **N=36**, with the true harness knee
bracketed at **[36, 39]** because N=37-39 were not measured. I would accept `L40S_DENSITY_RESULT status=PASS knee_N=36`
as a gate result, but I would word the headline carefully:

- "highest tested robust point = 36; first tested failing point = 40"
- "passes the pre-registered bar if `S_py_L40S <= 20`"
- "same-box Python remeasure recommended for multiplier accounting, not for invalidating this native measurement"

If the governance rule is changed to require a fresh same-box Python baseline, then this becomes **CONDITIONAL** because
the result is exactly at the bar when `S_py=20`. Under the stated input, `S_py_L40S ~16-20`, it is a defensible PASS.

## Evidence table

Primary evidence is the full per-N `DENSITY_TELEMETRY ... check=1a_density_sweep_full_session` row, not the final
wrapper summary. The final wrapper rows at `w3_run13.log:513-516` zero out TTFS/lag and say
`binding_resource=not_observed`; those aggregate fields are not reliable for metric details or binding attribution.

| N | status | sessions / finals | TTFS p50 / p95 / p99 | lag p50 / p95 | steady_gpu p50 / p95 | finalize_gpu p50 / p95 | finalize_wait p95 | evidence |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 32 | robust control | 256 / 256 | 20.6 / 75.6 / 109.8 ms | -140.1 / -66.5 ms | 14.1 / 39.0 ms | 14.0 / 32.1 ms | 0.0 ms | `w3_run13.log:220,223-224` |
| 36 | robust knee | 288 / 288 | 26.0 / 90.9 / 146.8 ms | -133.5 / -35.4 ms | 17.8 / 38.0 ms | 17.7 / 37.1 ms | 0.0 ms | `w3_run13.log:314,317-318` |
| 40 | not robust | 320 / 320 | 62.8 / 361.3 / 719.9 ms | 183.2 / 1337.3 ms | 28.5 / 38.4 ms | 28.4 / 38.7 ms | 0.0 ms | `w3_run13.log:408,411-412` |
| 44 | not robust | 352 / 352 | 64.3 / 483.9 / 943.2 ms | 700.5 / 2904.1 ms | 29.1 / 38.7 ms | 29.7 / 38.4 ms | 0.0 ms | `w3_run13.log:502,505-506` |

The harness also ran fresh-process-per-N (`w3_run13.log:135-136,230,324,418`), so N=36 is not inheriting warm state from
N=32.

## 1. Is knee=36 trustworthy?

Yes, for the pre-registered synchronized harness. The long bug history is real, but run #13 has the right controls:

- Warmup is full again. Code warms one steady path per worker and then every worker-local representative finalize
  bucket before the timed gate (`runtime/cpp/density_main.cpp:3177-3237`), and the log confirms full warmup at each
  measured N: 36/36 plus 223 finalize bucket-worker runs at N=36, 40/40 plus 258 at N=40
  (`w3_run13.log:314,408`).
- The measured gate starts only after warmup, via `resources.start(); gate.start_now();` after all workers are ready
  (`runtime/cpp/density_main.cpp:3314-3331`).
- N=32 is a meaningful control. Run #13 N=32 is nearly identical to the earlier full-warmup N=32 control: run #13 has
  TTFS 20.6/75.6/109.8 ms and finalize_gpu p50 14.0 ms (`w3_run13.log:223-224`); the earlier N=32 row had
  TTFS 21.3/72.6/112.8 ms and finalize_gpu p50 14.2 ms
  (`runtime/artifacts/l40s_w3_logs/l40s_density_N32_20260527T040756Z.stdout.log:70-71`).
- Correctness is not being waved through by the cross-arch tolerant path. The density loop requires `finalize.token_ok`,
  `fork_ok`, exact final tokens, and strict same-run serial-oracle event equality before counting a session as matched
  (`runtime/cpp/density_main.cpp:3302-3306`). The `DENSITY_GOLD_EVENTS_TOLERANT` code path is a separate cross-arch
  gold-event escape hatch and still requires steady/finalize token correctness (`runtime/cpp/density_main.cpp:1443-1454`).

Residual artifact risk is asymmetric. A warmup bug would more likely create a false fail, as the no-warm N=36 run shows:
with `warmup.enabled=false`, N=36 had finalize_gpu p50 498.9 ms and TTFS p95 7755.6 ms
(`runtime/artifacts/l40s_w3_logs/l40s_density_N36_20260527T103810Z.stdout.log:69-70`). Run #13's N=36 finalize_gpu p50
is 17.7 ms and p95 is 37.1 ms, so that failure mode is gone. I do not see a plausible warmup artifact that would
overstate N=36.

## 2. Is N=40 a real keep-up/compute collapse?

Yes for this harness. The N=36 -> N=40 transition is a capacity crossing, not finalize-pool contention, memory, or
CPU:

- Lag moves from healthy to collapsed: p95 -35.4 ms at N=36 to +1337.3 ms at N=40; p50 also crosses from -133.5 ms to
  +183.2 ms (`w3_run13.log:317-318,411-412`). Positive p50 lag means the problem is not just a few tail finalizes.
- TTFS fails both budgets at N=40: p95 361.3 ms > 175 ms, p99 719.9 ms > 250 ms (`w3_run13.log:411-412`).
- Finalize-pool wait is not the culprit. `finalize_wait` p95 is 0.0 ms at N=36, N=40, and N=44
  (`w3_run13.log:318,412,506`). The timing code records `finalize_runner_wait_ms` separately from GPU time
  (`runtime/cpp/density_main.cpp:1345-1364`).
- Warmup coverage is not the culprit. N=40 has full warmup, CUDA module loading is EAGER, 258 finalize bucket-worker
  warmups, and 16 loaded finalize buckets (`w3_run13.log:408,411`).
- Memory is not binding. Peak memory is 17.16 GiB at N=36 and 17.83 GiB at N=40 on a 44.39 GiB reported device, with
  per-worker context delta about 0.035 GiB. The code's memory-binding threshold would require about 92% peak/total
  (`runtime/cpp/density_main.cpp:3586-3589`); N=40 is about 40%.
- CPU is not binding. N=40 uses 5.87 / 32 cores (`w3_run13.log:411-412`), far below the code's 85% CPU threshold
  (`runtime/cpp/density_main.cpp:3591-3593`).

The resource signature is GPU/keep-up pressure plus post-finalize decode tail:

- steady_gpu p50 jumps from 17.8 ms to 28.5 ms at N=40 while p95 stays about 38 ms (`w3_run13.log:317,411`).
- finalize AOTI GPU p50 similarly jumps from 17.7 ms to 28.4 ms while p95 stays about 39 ms (`w3_run13.log:317,411`).
- `decode_wall` inside `finalize_phases` jumps from p95 17.4 ms at N=36 to p95 308.6 ms at N=40 (`w3_run13.log:317,411`).
- enc_first lock wait rises only modestly from p95 639.9 ms to 708.0 ms; it is high and pessimistic, but that delta
  does not explain a 1.37 s lag p95 or 720 ms TTFS p99 (`w3_run13.log:317,411`).

So the first failing point is best described as **synchronized keep-up/GPU scheduling collapse with TTFS tail**, not
"finalize bucket pool contention" and not "cold finalize".

## 3. Gate math

The pre-registered bar is `knee >= max(34, 1.80*S_py_L40S)`.

| assumed `S_py_L40S` | bar | result |
|---:|---:|---|
| 16 | 34 | pass by 2 streams |
| 18 | 34 | pass by 2 streams |
| 20 | 36 | pass exactly at bar |
| >20 | >36 | not established |

Calling this PASS is defensible only if the previously accepted production-confirmed `S_py_L40S ~16-20` is the baseline
input. The multiplier headline should be conservative: **36 is 1.8x vs 20 and 2.25x vs 16**. A fresh same-box Python
remeasure is useful audit work because the high-end case has zero margin, but it should not be retroactively required
unless "same-box Python" was part of the gate contract.

## 4. Sample sufficiency

For the pass/fail decision, the sample is sufficient.

At N=36, p99 is based on only about three tail samples out of 288 sessions, but the margin is large: p95 90.9 ms vs
175 ms, p99 146.8 ms vs 250 ms, and lag p95 -35.4 ms vs the 500 ms keep-up budget (`w3_run13.log:317-318`). There is a
rare max TTFS/finalize_total outlier around 628 ms in the full JSON (`w3_run13.log:317`), so this is not a production
p99.9 certificate. It is still a clear p99<=250 gate pass.

At N=40 the fail is unambiguous: TTFS p95 is already 361.3 ms, p99 is 719.9 ms, lag p50 is positive, and lag p95 is
1337.3 ms (`w3_run13.log:411-412`). N=44 confirms monotonic worsening (`w3_run13.log:505-506`).

The code's formal SLO decision is exactly these checks: enough finalize samples, lag p95 < 500 ms, TTFS p95 <= 175 ms,
TTFS p99 <= 250 ms, completed sessions, stream uniqueness, and zero mismatches/errors
(`runtime/cpp/density_main.cpp:3361-3372`).

## 5. Synchronized-burst pessimism

The harness is intentionally harsher than production arrivals. `StartGate` releases all workers together
(`runtime/cpp/density_main.cpp:518-552`), then the timed gate starts all worker loops at once
(`runtime/cpp/density_main.cpp:3314-3331`). Each worker starts the next session with `session_start = Clock::now()`,
feeds chunks at exact `160 ms * chunk` offsets, and calls finalize at the deterministic VAD deadline
(`runtime/cpp/density_main.cpp:3247-3286`). There is no stagger or jitter.

This makes first-chunk `enc_first` lock bursts and end-of-session finalize/decode bursts maximally synchronized. In a
staggered production arrival process, I would expect the practical knee to move upward, but I would not credit more than
one bracket without measuring it:

- N=40 might be salvageable with stagger because its failure is heavily tail-shaped and synchronized decode/finalize
  pressure is visible.
- N=44 is much harder to rescue: lag p50 is already +700 ms and lag p95 is +2904 ms (`w3_run13.log:505-506`).

My estimate: production-style staggering could plausibly move the usable point from 36 toward **38-40**, but not to 44
without another optimization. Treat **36 as a pessimistic lower bound**, not as a fully optimized production admission
number.

## 6. Skeptic objections that still matter

1. **The final aggregate summary has a parser/reporting bug.** `L40S_DENSITY_ROW` reports TTFS and lag as 0.000 for
   N=36/40/44, and the final line says `binding_resource=not_observed` even though N=40 and N=44 failed
   (`w3_run13.log:513-516`). This does not invalidate the knee because the full per-N telemetry rows are intact, but it
   must be fixed before publishing automated summaries.
2. **The exact knee is not pinned.** The result proves highest-tested robust N=36 and first-tested failing N=40. The
   true synchronized-harness knee may be 37, 38, or 39. This is fine for a PASS threshold of 36, but "native knee = 36"
   should be read as "certified knee_N=36 in this bracket."
3. **The multiplier is at-bar at the high end of Python baseline.** If `S_py_L40S` is 20, the bar is exactly 36. If a
   same-box Python rerun comes back above 20, the 1.8x claim fails unless N=37-39 passes. This is a multiplier audit
   issue, not a native telemetry issue.
4. **N=36 has rare tail outliers.** The p99 gate passes, but the max TTFS/finalize_total is much higher than p99. Do
   not use this run as a stricter tail-SLO certificate.
5. **The synchronized harness is pessimistic.** That helps defend against false PASS, but it also means N=40 may be a
   false fail relative to staggered production traffic. A staggered replay would be the right follow-up if the decision
   needs deployment admission, not just Step-1b gate clearance.

## Bottom line

**Step-1b gate verdict: GO.** Run #13 is trustworthy enough to certify **N=36 robust** on L40S native runtime, with
memory not binding and N=40 failing from real keep-up/TTFS collapse under an intentionally synchronized workload. The
only condition I would attach is to the multiplier prose: keep it as **1.8-2.25x vs the accepted 16-20 Python baseline**
until a same-box Python rerun either confirms `S_py <= 20` or forces an N=37-39 retest.
