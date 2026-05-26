# Step-0 build — paired review FOLDED (Codex + Opus)

Inputs: `codex-step0build-review.md`, `opus-step0build-review.md`. Both reviewed `density_main.cpp` + the 5090
smoke telemetry. **Combined verdict: the harness is a sound, faithful first diagnostic, but the Step-0 gates are
NOT yet sound for a PASS/STOP call.** Fix the gate soundness (+ one plan defect) before the decisive run.

## What's CONFIRMED good (both)
- Topology faithful: ONE shared steady loader `num_runners=N` (`:1109,1412`), explicit per-worker stream into
  `run(inputs,stream)` (`:415,1081`), per-thread `SessionState`+`joint`/`predict`/`enc_first`/`preproc`
  (`:1416-1418`), per-thread `CUDAStreamGuard` so decode `.item()`/`to(CPU)` sync the worker stream not default
  (`:497-498`). 0b's `strict_events_equal` is a real full-tuple oracle (`:358-396`). `#include session_main.cpp`
  + main-rename clean; N=200 regression preserved 0/200.
- **Decisive prelim finding (honest): concurrent native dispatch DOES overlap and IS correctness-safe at smoke
  scale (0 token/event mismatch) — conjunct-2 binary = YES — bounded ~1.7× encoder-only on the 5090, consistent
  with the AOTI execution lock serializing host-dispatch (NOT the GIL).**

## BLOCKERS (must fix before a Step-0 PASS/STOP)
1. **Scalar-sync ≤5% gate is a FALSE STOP (both: Opus F2 = Codex B1).** A high per-thread `.item()` fraction
   (63.8%) is the overlap OPPORTUNITY the design fills (0a's 1.69× + default-stream control 22.6%-worse confirm),
   not a failure. → **PLAN fix:** drop the ≤5% bar; split 0b into `identity_pass` (concurrent==serial, 0 mismatch)
   + `scalar_locality_pass` (gate on local-vs-global sync: per-thread stream confirmed + default-stream control
   worse + a sentinel/Nsight probe that an explicit-stream `.item()` does NOT drain unrelated streams). Keep
   `item_wait_pct` as reported telemetry. **(Harness measured it right; the Round-4 threshold was wrong.)**
2. **`overlap_estimate = sum_gpu/wall_ms` is BOGUS (both: Opus F1 = Codex B2).** It's contention-confounded
   (reads 11.06 @N=16 vs honest throughput 1.92×) AND reads >1 on the serialized default-stream control despite
   `unique_streams=1` — proving it's not an overlap measure. → relabel as diagnostic only; THROUGHPUT is the
   honest metric. **AND (Codex, new): 0a has NO correctness oracle** — it only checks `out.size()>=5` (`:1085`),
   never compares steady outputs to a serial oracle. → add a serial steady-output oracle for 0a; require an
   Nsight/CUPTI trace showing kernels on ≥2 non-default streams as the real overlap proof.
3. **Missing topology controls (Codex B3, new).** Harness always couples `workers==num_runners`; never tests
   `workers=N, num_runners=1`; no mutex-serialized mode; 0b's default-stream control doesn't exercise the
   `.item()` path. → add independent `--workers`/`--num-runners`, `--mutex-serialize-run`, and a 0b default-stream
   mode. Minimum matrix before PASS/STOP: `{N,N,explicit}`, `{N,1,explicit}`, `{N,N,default}`, `{N,N,explicit+mutex}`.

## MAJORS
4. **Memory-flat measurement is POLLUTED (Codex M1, new + Opus F6).** N=1 peak 6.71GB vs N=16 peak 5.34GB →
   `mem_ratio_target_vs_n1=0.796` (N=16 LOWER than N=1 = allocator/process-state pollution, NOT one-copy proof).
   → fresh process per N (or cleanup before the first measured loader); log loader-delta
   (`used_after_loader−used_before_loader`) separately from run peak; use target-N fresh-process deltas for the
   one-weight-copy gate.
5. **Workers reuse the SAME GPU input tensors (Codex M2, new).** `build_steady_cases` makes one vector; all
   workers loop it (`:1144-1147`) → weaker than production (distinct per-session buffers); can hide input/cache
   BW + aliasing. → per-worker (cloned/distinct) input+cache tensors.
6. **Smoke mislabeled as PASS (both: Opus F5 = Codex M3).** Two summaries exist (0a-only PASS @145342Z + failed
   smoke @145443Z); skip-modes still print `DENSITY STEP0 PASS` (`:1574-1576,1593-1600`). → if any gate
   skipped or rows<full-corpus, final status = `PARTIAL_DIAGNOSTIC`, not PASS; add `smoke/partial` flags + a run
   manifest; one stamp per dir or archive stale logs.
7. **0c is smoke-only (both).** Run full 200 rows + a hot-bucket target-N case; add stale-gen/event fixtures OR
   explicitly defer stale-gen out of Step 0 in the plan before calling 0c PASS.

## MINOR
- Stream uniqueness logged for 0a but not GATED (`unique_streams==workers` not asserted) and not logged for 0b/0c;
  `stream_for_worker` ignores the worker arg + relies on pool allocation (`:349-351`) → assert+log unique streams
  per explicit gate. (Codex m1)
- CMake target fine; an `add_custom_target(density …)` would be symmetric (Codex m2). Not a blocker.

## INTERPRETATION (both — carry to the Step-0 verdict + Step 1a)
0a's ~1.7× is the **encoder-run()-only** ceiling (no decode), NOT the density multiplier (Step 1a, where threads
fill each other's decode `.item()` idle). The early plateau (N=2 1.69× ≈ N=4 1.71×) ⇒ the execution lock
serializes host-dispatch; GPU work overlaps. Definitive lock-vs-BW attribution = Nsight at Step 1a.

## ACTION (this fix cycle, before commit + the decisive run)
- **PLAN edit:** fix the 0b threshold semantics (BLOCKER 1).
- **HARNESS edits (delegate to Codex):** 0a serial-output oracle + Nsight overlap proof (B2); independent
  workers/num_runners + mutex + 0b-default-stream controls (B3); fresh-process/loader-delta memory gate (M4);
  per-worker input tensors (M5); PARTIAL_DIAGNOSTIC status + run manifest (M6); stream-uniqueness gate (m1).
- Then re-review, run the full gates (200 rows + the control matrix + Nsight) → **PAUSE for the Step-0 PASS/STOP**
  (the decisive conjunct-2 measurement; needs the 5090; a real GO/STOP decision).
