# Step 1c — CLEAN-KNEE RE-MEASURE after the async-dispatch FLIP (A/B on one box/binary)

**Box:** ubuntu@34.214.169.199 — NVIDIA L40S sm_89, g6e.4xlarge (16 vCPU, ~45 GiB VRAM at idle), CUDA-13 system / cudart-12.9 runtime, torch 2.8.0+cu128. **Date:** 2026-05-31.
**Binary:** `~/density/cpp/build_l40s_density/ws_server`, **rebuilt** this session from the async-flip source (dev-box HEAD `ef64a0b`; rsync'd `cpp/` subtree). jit_load_guard PASSED in-build ("OK: exactly 1 serialized raw JIT load"); cudart fail-closed check PASSED (links only `libcudart.so.12`, no cudart-13).
**A/B method:** identical box, identical binary, identical launch config (scheduler ON, bg-warmup, CAP=128/LANES=128, TS enc_first, B_max=4, finalize_silence_ms=0). The ONLY difference is the dispatch-timing mode:
- **ASYNC** = `NEMOTRON_WS_DISPATCH_TIMING` unset → `DispatchTimingMode::Poll` (the new default; dispatcher no longer blocks on `cudaEventSynchronize` inline — timing events drained via `cudaEventQuery`).
- **SYNC CONTROL** = `NEMOTRON_WS_DISPATCH_TIMING=sync` (the OLD behavior; inline `cudaEventSynchronize(ev_stop)` on the dispatch path). Verified present in the server's `/proc/PID/environ` for the control run.

Gate (unchanged from Step 0): highest N that is **0-error ∧ TTFS-p95 < 400 ms ∧ lag-p95 < 500 ms**, robust across repeats. All windows `--procs 32 --window-s 45`.

---

## KNEE TABLE — async vs sync-control

### Adaptive sweeps (rt_loadgen, one row per N; TTFS/lag in ms)

**ASYNC (new default)** — sweep {56,60,64,68,72}:

| N | ok | err | TTFS p50 | TTFS p95 | TTFS p99 | TTFS max | lag p95 | maxjit | p95<400 |
|---|----|-----|----------|----------|----------|----------|---------|--------|---------|
| 56 | 277 | 0 | 15 | **23** | 35 | 151 | 43 | 0 | PASS |
| 60 | 296 | 0 | 16 | **27** | 45 | 588 | 46 | 0 | PASS |
| 64 | 319 | 0 | 17 | **117** | 371 | 834 | 137 | 0 | **PASS** |
| 68 | 334 | 0 | 332 | **1336** | 1737 | 2270 | 1356 | 0 | FAIL |
| 72 | 340 | 0 | 826 | **2817** | 3352 | 4152 | 2836 | 0 | FAIL |

**SYNC CONTROL (old behavior)** — sweep {56,60,64}:

| N | ok | err | TTFS p50 | TTFS p95 | TTFS p99 | TTFS max | lag p95 | maxjit | p95<400 |
|---|----|-----|----------|----------|----------|----------|---------|--------|---------|
| 56 | 277 | 0 | 15 | **22** | 39 | 477 | 42 | 0 | PASS |
| 60 | 295 | 0 | 16 | **37** | 135 | 411 | 58 | 0 | PASS |
| 64 | 304 | **3** | 17 | **485** | 923 | 1543 | 505 | 0 | **FAIL** |

### Full-telemetry captures (measure_one_n.sh: + /scheduler_telemetry + /stats + NVML dmon)

| mode | N | ok | err | TTFS p95 | TTFS p99 | lag p95 | disp_cpu%win | disp_util%win | qdepth p50/p95/p99 | backlog Δ | disp_exc | NVML sm% mean/max | gate |
|------|---|----|-----|----------|----------|---------|--------------|---------------|--------------------|-----------|----------|-------------------|------|
| SYNC | 60 | 295 | 0 | **42** | 167 | 62 | 70.1 | 67.6 | 8/16/16 | 3358 | 0 | 85.3 / 95 | PASS (sync robust knee) |
| SYNC | 64 r1 | 302 | **3** | **575** | 1317 | 596 | 69.8 | 67.2 | 7/16/16 | 3917 | 0 | 86.5 / 94 | FAIL |
| SYNC | 64 r2 | 307 | **2** | **704** | 1369 | 724 | 64.5 | 61.7 | 8/16/16 | 4312 | 0 | 86.5 / 93 | FAIL |
| ASYNC | 64 r1 | 317 | 0 | **58** | 238 | 78 | 47.3 | 61.9 | 12/12/13 | 4050 | 0 | 89.9 / 96 | PASS |
| ASYNC | 64 r2 | 317 | 0 | **66** | 166 | 86 | 54.1 | 70.9 | 11/12/13 | 4090 | 0 | 88.8 / 96 | PASS |
| ASYNC | 64 r3 | 316 | 0 | **176** | 324 | 197 | 55.8 | 73.6 | 12/12/15 | 4443 | 0 | 89.0 / 96 | PASS |
| ASYNC | 68 | 331 | 0 | **1519** | 2181 | 1539 | 54.5 | 72.9 | 12/12/15 | 4699 | 0 | 90.5 / 96 | FAIL (cliff) |

> disp_cpu%win / disp_util%win = per-window dispatcher CPU-fraction and GPU-work-fraction-on-dispatcher-timeline (us-deltas, the faithful windowed signals). qdepth ring capacity = 16 (16 = ring full). `future_wait`/`enqueue_wait` retain the Step-0 semantics caveat (multi-second absolute scale even when healthy — trend-only); not load-bearing here.

---

## RESULT

- **SYNC-control knee = N = 60** (cliff edge at 64). N=56/60 PASS (p95 22/37 ms); N=64 FAILS (p95 485–704 ms, 2–3 errors across runs). This **reproduces the Step-0 baseline** (clean SLO-robust knee N≈56–60, cliff edge 64) on the same box/binary → the A/B is valid; only the dispatch flip differs.
- **ASYNC knee = N = 64** (cliff edge at 68). N=56/60/64 all PASS; N=64 is robust across 3 repeats (p95 = 58 / 66 / 176 ms, all 0-error). N=68 FAILS hard (p95 1336–1519 ms). Under SYNC, N=64 fails; under ASYNC, N=64 passes with ~3–10× headroom under the 400 ms gate.

### **LIFT = async_knee − sync_control_knee = 64 − 60 = +4 streams.**

- **Clears the +4 keep-threshold? YES — exactly meets it (+4).** The lift is at the keep-threshold floor, not above it. Caveat below.
- The whole curve shifts out by ~4 streams AND the cliff softens at the boundary: at N=64 the async p95 (58–176 ms) is **~3.6–9.9× lower** than the sync p95 (485–704 ms) at the identical N, and async N=64 is 0-error vs sync N=64's 2–3 errors. The sharp cliff itself moved from 60→64 (sync) to 64→68 (async).

### Noise band
- **ASYNC N=64 (knee):** TTFS p95 = 58 / 66 / 176 ms across 3 repeats (band ~58–176 ms; all PASS, all 0-error). Wider than Step-0's near-zero band at the sync robust knee, but every repeat clears the gate with margin. ttfs_max on the worst repeat hit 1157 ms (occasional single-stream spike) but p95/p99 stayed in-gate (176/324 ms).
- **SYNC N=64 (control):** TTFS p95 = 485 / 575 / 704 ms across runs, 2–3 errors — consistently FAILS, matching Step-0's 482–531 ms straddle. The control firmly reproduces the OLD failing-at-64 behavior.
- **Knee uncertainty:** the async knee sits cleanly at 64 (robust pass) with the first hard fail at 68 — the [64,68) granularity means the true SLO-robust knee is in [64,67]; the +4 lift is solid at the measured grid. (Grid step was 4; a finer probe between 64 and 68 was not run.)

---

## TELEMETRY SHAPE CHANGE (what the flip did)

The async flip produced exactly the predicted dispatcher-unblocking signature, comparing the SAME N=64 under both modes:

1. **Dispatcher CPU fraction DROPPED.** disp_cpu%win at N=64: sync 64.5–69.8 % → async 47.3–55.8 % (a ~15–20-point drop). Removing the inline `cudaEventSynchronize` from the dispatch path freed the dispatcher CPU it was burning blocked-on-GPU. (At the sync knee N=60 it was already 70 %.)
2. **Queue depth UN-PINNED.** Under sync, queue_depth p95/p99 = 16/16 (ring FULL) at every N including the passing N=60 — the dispatcher couldn't drain fast enough. Under async, queue_depth p95 = 12 at N=64 (p50 11–12) — the ring no longer saturates; the dispatcher keeps up.
3. **GPU util went UP and is now the binding.** NVML sm% mean at N=64: sync 86.5 % → async ~89 %; at the async cliff N=68 it is **90.5 % mean / 96 % max**. The dispatcher CPU is no longer the limiter (it has headroom: ~45–56 % at the knee); the system now cliffs because the **GPU compute is saturating (~90 % mean, 96 % peak)**. backlog_gt_bmax keeps climbing past the knee (4050→4443→4699), and 0 dispatcher exceptions / 0 maxjit throughout (loadgen never the limiter).

**Binding constraint after the flip = the GPU (NVML ~90 % mean / 96 % peak at the async knee/cliff).** Pre-flip the binding was the dispatcher CPU (blocked on the inline sync, ring pinned at 16, GPU only ~86 %). This is the intended outcome: the flip converted a dispatcher-serialization wall into a GPU-compute wall.

**Residual headroom for Steps 3/4:** small. NVML at the async knee is already ~89–90 % mean / 96 % peak, so the GPU has only ~10 % mean headroom left. Steps 3/4 (further dispatch/host-sync compression) will face diminishing returns on streams-per-box unless they reduce per-stream GPU work itself — the cheap "free up the dispatcher" lever is now spent. The dispatcher still has CPU headroom (~45–56 %), so any remaining gains must come from the GPU side (batch efficiency / kernel work), not the dispatch front-end.

---

## Artifacts (on box, ubuntu@34.214.169.199; box + async server LEFT RUNNING)
- Rebuilt binary: `~/density/cpp/build_l40s_density/ws_server` (async source, ef64a0b). Build script: `~/density/build_ws_async.sh`.
- Async sweep stdout captured in this session (output-JSON write failed on a pre-existing missing dir; table data is the loadgen stdout above and is authoritative).
- Per-N full-telemetry captures: `~/density/step0_out/N{64,60,68}_{async_r1,async_r2,async_r3,async_edge,sync_r1,sync_r2,sync_knee}_<HHMMSS>.{telem_before,telem_after,stats_after,row}.json, .dmon.txt, .loadgen.{json,txt}`.
- Server logs: async sweep `~/density/step1c_async_ws_server.log`; sync control `~/density/step1c_sync_ws_server.log`; final running async `~/density/step1c_async_final_ws_server.log`.
- `/tmp` AOTI/inductor dirs cleaned at end. **Box NOT torn down; server re-launched on async default (DISPATCH_TIMING unset → Poll), warming.**
