# Opus investigation — native finalize TTFS ≈ 234 ms p95 at N=1 (the 10× gap vs Python)

Independent ("Opus") half of a paired investigation. Reasoned from static code + existing telemetry + the documented
Python optimization history. **GPU work was NOT run**: `nvidia-smi` showed two resident `C` (compute) python processes
(PIDs 288593 ≈5.4 GB + 330636 ≈2.9 GB) — the "other job" the prompt warned about — so Phase 4 (GPU micro-profile) was
correctly skipped to avoid contention. The telemetry already decomposes the 234 ms unambiguously, so no GPU was needed.

**Bottom line up front: the 234 ms is NOT a finalize bug. It is a per-AOTI-bucket FIRST-RUN cold-start cost
(~225 ms of GPU-stream time on the bucket's first `loader.run()`), which the density harness never amortizes because
each finalize bucket runs ≈1× in a 6-session sweep. The identical ~225-310 ms cold-start is visible on the STEADY
encoder too (one spike per worker stream), where it hides in `max` because it is amortized over ~292 chunks. It
contaminates the Phase-2 finalize TTFS measurement. The fix is warmup, not a kernel/graph change.**

---

## 1. What the PYTHON finalize actually costs at low concurrency, and the optimization-history lesson

### 1a. Python finalize decomposition (the <20 ms target)
Gold reference: `proj-2026-05-23-1731/finalize-p50-decomposition.md` (2134 finalize_profile_records, conc-10 L40S
lanes=2, finalize-graph ON, mined — no new run). Server-side finalize span:

| component | p50 (ms) | p95 (ms) | note |
|---|--:|--:|---|
| **finalize_wall (total)** | **22.1** | 41.3 | the whole server-finalize |
| lock_wait | 0.36 | 21.1 | ≈ the entire p95 spread (lane contention) — **0 at N=1** |
| model_wall | 13.4 | 17.6 | encoder + decode |
| ├ encoder_wall | 9.7 | 12.0 | **already CUDA-graph replay** |
| └ decode_wall | 3.6 | 6.4 | eager greedy |
| preproc_wall | 2.4 | 5.5 | 3 invocations |
| fork_clone | 0.44 | 2.1 | negligible |

At **N=1 / no contention** the lock_wait (the only contention term) collapses to ~0, so the Python server-side
finalize compute is **≈ model_wall(13.4) + preproc(2.4) + fork(0.44) ≈ 16 ms**, i.e. the **"< 20 ms at N=1"** the
prompt cites. Its single biggest piece is the **already-graphed finalize encoder at 9.7 ms**. The Python finalize
*encoder* therefore costs ~9.7 ms graphed / ~35 ms eager — **never ~200 ms.**

### 1b. The optimization-history lesson — did Python ever hit a ~100-200 ms finalize slowdown? What was it?
Yes — but it was a *steady-state launch-dispatch* slowdown, NOT a cold-start, and it was ~30 ms, not 200 ms:

- `proj-2026-05-22-1353/finalize-telemetry.md` (KERNEL PROFILE section) + `finalize-graph-WIN.md`: the eager finalize
  encoder was **LAUNCH-BOUND** — `NEMOTRON_FINALIZE_TORCH_PROFILE` measured **~1376 `cudaLaunchKernel`/finalize**,
  ~6.9 ms of actual GPU math but ~12 ms (5090 fast CPU) to ~35 ms (cloud slow Milan CPU) of CPU dispatch. The CUDA
  graph collapsed those launches → **encoder 39 → 9.6 ms** (`finalize-graph-WIN.md` line 16). Net win on p95 was large
  (274→246 p50, 401→279 p95) because **per-call savings COMPOUND across queued finalizes** — but the per-finalize
  encoder slowdown that was fixed was **~30 ms (eager dispatch), an order of magnitude short of 234 ms.**
- The same docs explicitly **ruled out a startup/cold cause for the residual tail**: allocator retries = 0 across 51k
  entries (`finalize-graph-WIN.md` line 35), GC only marginal, no lazy-load/recompile term ever appeared in the
  Python TTFS because the **production server warms the encoder at startup** (`NEMOTRON_WARMUP_MS=200`, see
  `finalize-optimization-suggestions.md:5`).
- `finalize-python-tail-analysis.md` ranked redundant clones/preproc/JSON/locks — all **sub-millisecond to single-ms**;
  none is a 200 ms item.

**Lesson for the native path:** the Python stack's big finalize lever was *launch collapse* (~30 ms), and it paid its
*one-time* kernel-warmup at process startup so it never showed in TTFS. **The native sweep is NOT repeating the
launch-bound mistake — it is hitting a DIFFERENT mistake the Python harness avoided: it never warms each finalize
bucket, so every finalize pays a one-time cold-start that Python folded into startup warmup.** There is no Python
precedent of a 200 ms finalize because warmup eliminated the analogue.

---

## 2. The native 234 ms decomposed from telemetry (per-phase, which dominates) — MEASURED

Source: the autotune-OFF floor sweep, 3 repeats: `artifacts_n200_mel/logs/{20260526T175948Z, 180757Z, 181603Z}/`
(`density_num_runners1_..._1a_full_session_density_sweep.jsonl`), plus the `off_floor_repeat_{1,2,3}` stdout logs.
Run config (stdout line 1): `stream_mode=explicit warmup=true cadence=160ms sessions_per_worker=6`, device =
**RTX 5090 (sm_120)**, artifacts = `artifacts_n200_mel` (NOT the L40S sm_89 set).

### 2a. The 234 ms is almost entirely `finalize_gpu`, and `finalize_gpu` is pure bucket-encoder GPU-stream time
What the timers measure (`density_main.cpp:run_finalize_density`, lines 1068-1188):
- `finalize_gpu_ms` = `cudaEventElapsedTime` between `ev_start` recorded **immediately before** and `ev_stop`
  **immediately after** the **bucket AOTI `run_aoti_loader(finalize_loader,…)`** call (lines **1122-1144**). It is the
  GPU-stream time of the **finalize encoder bucket forward ONLY** — it excludes fork/clone (line 1098), the
  `decode_range_density` continuation decode (line 1139, recorded after `ev_stop`), scalar `.item()` sync
  (`scalar_sync_wait_ms`, separate), and all glue.
- `finalize_runner_wait_ms` = `max(0, host_wall_around_run − gpu_ms)` (line 1184) — host dispatch overhead beyond GPU.
- `finalize_total_ms` = wall from `total_start` (line 1085) over the whole function.

N=1 measured (repeat 1 / 2 / 3):

| metric (N=1, n=6 finals) | r1 | r2 | r3 |
|---|--:|--:|--:|
| **finalize_gpu p50** | **225.3** | **227.7** | **225.0** |
| finalize_gpu p95 (=max, n=6) | 235.6 | 236.5 | 233.5 |
| **finalize_gpu mean** | **175.6** | **176.4** | **174.7** |
| finalize_runner_wait (host−gpu) p95 | 0.0016 | 0.0038 | 0.0033 |
| finalize_total p50 | 227.3 | 229.8 | 227.0 |
| steady_gpu p50 / p95 / **max** | 5.47 / 5.62 / **229.9** | 5.47 / 5.63 / **227.4** | 5.46 / 5.65 / **228.1** |
| item_wait (decode scalar-sync) p50 | 2.69 | 2.71 | 2.69 |

**Which phase dominates: `finalize_gpu` (the bucket encoder forward) IS the 234 ms.**
- `finalize_runner_wait ≈ 0` (≤0.004 ms) → the ~225 ms is genuinely on the GPU stream, **NOT** host glue, **NOT**
  queue, **NOT** runner-acquire, **NOT** the inference-lock. (`finalize_wait` percentile block is ~0 at every N.)
- fork/clone, decode-continuation, and scalar-sync are all separately accounted and tiny (item_wait ~2.7 ms; fork is
  off the GPU bracket entirely). So the 234 ms is **not** folded steady-chunk lag and **not** a slow per-token decode.
- The 234 ms `finalize_total` ≈ `finalize_gpu`(225) + decode/glue(~2-9). Confirmed: the encoder bucket forward is the
  whole story.

### 2b. It is a COLD-START (first-run-per-bucket), proven by mean<<median and the amortization curve — MEASURED
The encoder-forward GPU time being ~225 ms is absurd on its face: the **steady** encoder forward is **5.47 ms**, and
the Python eager finalize encoder is ~35 ms / graphed 9.7 ms. ~225 ms is ~40× the steady forward. The telemetry shows
why it is a first-run cost, not a per-call cost:

1. **mean (≈175) is far BELOW median (≈225) over n=6.** That is only possible if ≥2 of the 6 finalizes are *much*
   faster than 225 ms. Solving the n=6 distribution (p50≈225, p95=max≈235, mean≈175): the 4 cold finalizes ≈
   [225,228,232,236] and the **2 warm/repeat-bucket finalizes ≈ ~60-67 ms each.** (Inferred from the 6-sample
   mean/median, not directly logged — see honesty note.) So *some* finalizes are already ~3-4× faster — the 225 ms is
   not uniform.
2. **The warm minority maps exactly to the harness's warmup.** `assign_density_utts` (`density_main.cpp:2682`) gives
   the single N=1 worker utts [0,1,2,3,4,5]; warmup runs only `assigned[0][0]` = **utt 0** (line 2800). The 6 utts use
   **5 distinct buckets** (drop=2 T∈{45,46,49,57,58}; from `finalize_loader_memory.records`), so exactly one T is shared
   by 2 utts. With utt 0's bucket warmed, **2 finalizes are warm, 4 are cold** — matching the mean/median split.
3. **finals/bucket ≈ 1.2-1.85 across N → cold-start never amortizes → p50 stays pinned at ~225-230 ms regardless of N:**

   | N | loaded_buckets | finals/bucket | finalize_gpu p50 | finalize_gpu mean | finalize_wait p95 | steady_gpu p50 / max |
   |---|--:|--:|--:|--:|--:|--:|
   | 1 | 5 | 1.20 | 225 | 176 | 0.0016 | 5.47 / **229.9** |
   | 2 | 8 | 1.50 | 228 | 168 | 0.0010 | 5.42 / **268.9** |
   | 4 | 13 | 1.85 | 230 | 175 | 0.0004 | 5.42 / **310.9** |

   finalize_gpu **p50 does not degrade with N** (225→228→230) and finalize_wait stays ~0 → **this RULES OUT
   contention/queueing as the cause.** A contention story would show finalize_wait or p50 rising with N; instead the
   per-finalize GPU time is flat because it is dominated by a per-bucket first-launch cost that each N re-pays on its
   own fresh buckets.
4. **The same ~225-310 ms cold-start is on the STEADY path.** `steady_gpu` p50 = 5.47 ms but **max = 229.9 ms at N=1,
   268.9 at N=2, 310.9 at N=4** — i.e. exactly ≈N cold-start spikes (one per worker stream's first steady forward),
   each amortized over ~292 chunks so it only shows in `max`. **This is the clincher: the ~225 ms is a per-AOTI-loader
   first-launch cost that is universal (steady and finalize), not finalize-specific.** Finalize just never amortizes it
   (≈1 run/bucket) while steady drowns it (292 runs/loader).

### Honesty notes on Phase 2
- **Measured:** finalize_gpu p50/p95/mean/max; finalize_wait≈0; steady_gpu p50/max; item_wait; loaded_buckets;
  finals/bucket; warmup=true; the utt→bucket assignment logic. All from the JSONL + stdout + source.
- **Inferred (not directly logged):** the per-session split "2 warm @ ~60-67 ms + 4 cold @ ~225-236 ms" — the harness
  records only percentiles, not per-final values, so the exact warm-finalize cost is reconstructed from the n=6
  mean/median, not read off. The *direction* (≥2 finalizes far below the median) is mathematically forced by mean<<p50
  and is robust; the exact ~60 ms is an estimate.
- **p95 over n=6 is just `max`** — so "234 ms p95" is the slowest of 6 cold finalizes, but **p50 is also ~225 ms**, so
  this is NOT merely a single warmup outlier; the *typical* finalize in this sweep is cold.

---

## 3. Static suspects in the native finalize path (file:line) vs Python

The native finalize does several things the Python finalize does not — I checked each for a ~200 ms cost:

| # | native behavior | file:line | cost? | verdict |
|---|---|---|--:|---|
| **A** | **Per-T AOTI bucket, run cold (lazy CUDA module load on first `run()`)** | `density_main.cpp:1114` (`finalize_loaders.get(drop,T)` lazy-loads the bucket) → `:1128` (`run_aoti_loader`); pool builds an `AOTIModelPackageLoader` per bucket at `:1015`, `load_constants` at `:1017`; **package = 747 embedded `.cubin` files** (verified by `unzip -l enc_finalize_d2_T45.pt2`) | **~160-220 ms (first run only)** | **ROOT CAUSE.** Each bucket is a separate AOTI package with its own embedded cubins; with CUDA-12 default `CUDA_MODULE_LOADING=LAZY` (no override anywhere in the harness/scripts), the driver `cuModuleLoadData`s each cubin + cold-launches each of ~dozens of Triton/cuDNN kernels on first `run()`. `preload`/`load_constants` load *weights*, not kernel modules, and do **not** trigger a forward. |
| B | full `clone_session(parent)` per finalize (deep-clones caches + decoder state) | `density_main.cpp:1098`; FORK_ASSERT snapshot at `:1095,1177` | < 1 ms (off the GPU bracket; mirrors Python fork_clone ~0.44 ms) | not a lever; and it is **outside** the `finalize_gpu` window so cannot explain the 225 ms anyway |
| C | manifest + SHA256 contract verify | `density_main.cpp:884` (`verify_bucket_manifest`) | once per pool construction (stdout "manifest verified" prints once), **not per finalize** | NOT per-finalize → not the cause |
| D | scalar D2H syncs (`scalar_i64_timed`, `argmax_item_timed`) | `density_main.cpp:692-704,1133,1139` | accounted in `item_wait`/`scalar_sync` ≈ 2.7 ms | small; outside the gpu bracket |
| E | per-token continuation decode loop (joint+predict+argmax+`.item()` per token) | `density_main.cpp:706-735` (`decode_range_density`), called at `:1139` | ~few ms (item_wait 2.7 ms) | **recorded AFTER `ev_stop` (line 1130)** → not in finalize_gpu; not the 225 ms |
| F | `.to(device).contiguous()` of `final_chunk_mel` | `density_main.cpp:1107` | sub-ms H2D of a small mel | negligible |

Only **suspect A** is on the GPU stream, scales as "first-run per distinct bucket," and matches the magnitude — and it
is corroborated by the identical cold-start on the **steady** loader (§2b.4), which shares neither B, C, E, nor the
finalize bucket set.

**What Python does that the native does not (why Python never sees this):**
- Python's finalize encoder is **not a separate per-T AOTI package** — it is the **same warmed encoder module** the
  steady path uses, with a finalize **CUDA graph captured over already-warm eager kernels** (`cudagraph_encoder.py` +
  `server.py` finalize path). The kernels are loaded+warm by the time any finalize runs.
- The Python production server **warms at startup** (`NEMOTRON_WARMUP_MS=200`; `finalize-optimization-suggestions.md:5`),
  paying lazy-module-load and first-launch ONCE per process, before TTFS is ever measured.
- Net: Python folds the one-time cold-start into startup; the native density harness leaves it inside the timed
  finalize because it only warms 1 bucket per worker (`density_main.cpp:2797-2818`).

---

## 4. ROOT-CAUSE hypothesis for the 10× gap (quantified, with confidence)

**Primary cause (confidence: HIGH, ~0.9): per-AOTI-bucket FIRST-RUN cold-start (lazy CUDA module load + cold kernel
launch) that the harness never amortizes.**

Quantified contribution to the N=1 234 ms p95:
- **Cold-start first-run: ~160-220 ms** of the 225-234 ms. Evidence: (i) finalize_gpu mean (175) << median (225) ⇒
  warm-bucket finalizes run far faster (~60-67 ms, inferred); (ii) finals/bucket≈1.2-1.85 ⇒ p50 pinned ~225-230 ms at
  every N with finalize_wait≈0 ⇒ not contention; (iii) the steady loader shows the SAME ~230-311 ms cost exactly once
  per worker (in `max`), amortized away by 292 chunks. The cold-start is the difference between the warm finalize
  (~60 ms) and the cold finalize (~225 ms) ≈ **~160 ms**, plus warmup itself doesn't fully cover the steady first
  forward (so the residual warm finalize is still elevated vs steady).
- **Residual warm finalize ≈ ~60-67 ms** (inferred). This is the steady-state finalize-bucket forward cost. It is
  itself larger than the steady chunk (5.5 ms) and the Python eager finalize (~35 ms) — plausibly because a finalize
  bucket processes T≈45-58 frames through the full conformer with `keep_all_outputs=True` vs the steady 25-frame
  window, AND is a default-mode (autotune-OFF) artifact with un-tuned kernels. **This piece is a real per-call cost and
  IS legitimately part of the per-stream finalize budget** — but at ~60 ms it is ~3-4× the Python graphed finalize, not
  10×. (Confidence MEDIUM ~0.6 — the ~60 ms is inferred from the 6-sample mean, not directly logged; a warm probe is
  needed to confirm.)

**Ruled out (with evidence):**
- **Missing finalize CUDA graph as the 200 ms cause — NO** (confidence HIGH). The prompt's own framing is correct: a
  finalize graph collapses launch dispatch (~tens of ms in Python: 39→9.7), not a ~200 ms cold-start. finalize_wait≈0
  here means the host dispatch is NOT the bottleneck (unlike the Python launch-bound case), so a graph would help far
  less than warmup. The 225 ms is GPU-stream first-launch latency, which a graph does not remove (you still pay
  module-load + capture the first time).
- **Contention / lock / queue — NO** (HIGH): finalize_wait≈0 at all N; p50 flat across N.
- **A redundant clone / manifest re-verify / D2H copy — NO** (HIGH): all are off the `finalize_gpu` GPU bracket and/or
  once-per-pool, and are individually sub-ms to single-ms (suspects B-F).
- **SM/arch mismatch PTX→SASS JIT — UNLIKELY** (MEDIUM ~0.65): `aot_compile.py:41` states the artifacts target sm_120
  (the 5090) and the packages embed cubins (not just PTX), so the driver should not need to JIT-recompile from PTX.
  *Honest gap:* I could NOT directly confirm the cubins' SM target (`cuobjdump` is not installed on this host and the
  cubins carry no plaintext `sm_` strings), so a residual chance remains that some kernels fell back to PTX-JIT — but
  the lazy-module-load story explains the magnitude without needing a mismatch, and the steady path (same toolchain,
  same GPU) shows the same one-time cost, which is what you'd expect from lazy loading regardless of arch.

**Important orthogonality finding:** the cold-start is **independent of the autotune-OFF-vs-ON axis**
(`reviews/codex-autotune-changes-review.md`). `max_autotune` changes *which kernels* get compiled (faster steady-state
math), but does NOT remove first-launch module-load latency — if anything autotune-ON may emit more/larger specialized
kernels and pay a *similar or larger* cold-start. So switching to the autotune-ON "headline" artifact will **not** fix
the 234 ms unless warmup is also added. Conversely, the 234 ms is a property of the *harness's warmup policy*, not of
the floor artifact's quality — so it must not be read as "the autotune-OFF floor is 234 ms slow."

**Net answer to the 10× gap:** Python N=1 finalize ≈ 16 ms (encoder 9.7 graphed, warm). Native N=1 finalize p50/p95 ≈
225/234 ms = **~160 ms one-time cold-start (harness never warms the bucket) + ~60 ms warm finalize-bucket forward (real,
but only ~3-4× Python, partly because autotune-OFF + larger T).** The headline 10× is ~70% measurement artifact
(cold-start) and ~30% a real but modest per-call gap.

---

## 5. Recommended fix

**It is NOT the finalize graph, NOT the host-bound decode, NOT a redundant clone/manifest/copy. It is primarily a
MEASUREMENT ARTIFACT (missing per-bucket warmup), with a secondary real ~60 ms warm-finalize cost.**

1. **Warm every finalize bucket that the timed run will use, before timing (the decisive fix).** Today warmup runs only
   `assigned[worker][0]` (`density_main.cpp:2797-2818`) → 1 bucket/worker. Instead, after `finalize_loaders.preload(
   needed_buckets)` (`:2776`), run ONE throwaway forward through **each loaded bucket** (the pool already has
   `preload_all()`/`preload()` for *weights*; add a forward-warm pass — e.g. iterate `needed_buckets` and call a
   dummy `run_finalize_density` or a direct `loader.run()` on a padded chunk per (drop,T)). This pays lazy-module-load
   once per bucket up front — exactly what the Python `NEMOTRON_WARMUP_MS` startup warmup does — and the timed
   finalize_gpu should drop from ~225 ms to the warm ~60 ms (and the per-N spikes in steady_gpu `max` should vanish if
   the steady loader is likewise warmed per worker). This is the single change that de-contaminates the Phase-2 density
   numbers.
   - Cheaper/complementary: set `CUDA_MODULE_LOADING=EAGER` in the harness env to force all modules to load at package
     load instead of first launch — this moves the cost out of the timed window (though EAGER raises load-time memory;
     validate against the OOM seen at N=8).

2. **Re-run the N=1..4 floor sweep with per-bucket warmup and confirm finalize_gpu p50 ≈ the warm cost.** The current
   `finalize/TTFS tail` binding-resource verdict (NO_PASS_TO_1B, ttfs p95 234 >> 175 budget) is driven by the
   cold-start and is **not trustworthy** until warmup covers all buckets. Expect the budget question to change
   materially.

3. **THEN, if the warm finalize (~60 ms inferred) is still above the ~16 ms Python finalize and matters for density:**
   that residual is the legitimate target. Likely levers, in order: (a) compile the buckets **autotune-ON**
   (`max-autotune`) — the deploy artifact per `codex-autotune-changes-review.md` — to speed the steady-state kernels
   (this is where autotune-ON *does* help, unlike the cold-start); (b) only then consider a finalize CUDA graph to
   collapse launch dispatch, but note finalize_wait≈0 here means dispatch is not the current bottleneck, so expect a
   smaller win than Python got. Do **not** build the finalize graph to chase the 234 ms — that mis-attributes a
   cold-start to launch-dispatch.

4. **Guardrail:** report the finalize cost as a **warm** steady-state number for the density gate (cold-start excluded),
   and separately report the one-time per-bucket cold-start as a startup/warmup cost — mirroring how production folds
   it into `NEMOTRON_WARMUP_MS`. Mixing them (as the current sweep does) is the conflation that produced the 10×
   headline, directly analogous to the Python history's own corrected conflations
   (`finalize-telemetry.md`: "the 178 ms server finalize was a mis-attribution").
