# Plan: Further inference optimization — productionize multi-process scaling + finish the TTFS levers

Project directory: `./proj-2026-05-21-inference-opt`

## Context
The deep-dive (round1-synthesis.md, round2/3 docs, g6-vs-g6e-results.md) established the real scaling picture for
the Nemotron streaming ASR server on EC2 g6/g6e: **a single server process is GIL/scheduler-capped at a realtime
keep-up knee of ~16, independent of GPU.** The way to scale per-box is **multiple GIL-independent processes + CUDA
MPS** (L40S: K=3 processes × lanes=2 → 48/box, vCPU-bound with GPU headroom). This plan productionizes that
architecture and finishes the highest-value remaining optimizations. It also records **dead ends** so they aren't
re-explored.

## Established findings (do not re-litigate)
- Single process, `NEMOTRON_MODEL_LANES=2` = sweet spot (knee 16). lanes>2 regress (GIL); lanes=1 = knee 4.
- **2 lanes/process is ~2× more efficient than 1 lane/process** (amortizes per-process overhead; half the MPS clients).
- **Multi-process + MPS scales**: L40S K=3 → 48/box; **MPS essential** (K=2 GPU 90%→50%; unlocked K=3). The cap is
  **vCPU-bound, not GPU** (L40S ~65% at K=3) → more vCPUs (bigger instance size) → more processes → more density.
- **$/stream: g6/L4 (~32/box) cheapest; g6e/L40S (48+/box) = density play** (fewer boxes, higher $/stream).
- TTFS committed levers: `NEMOTRON_BATCH_BARRIER_DRAIN` (N=120 7947→207 ms), `NEMOTRON_BATCH_FINALIZE` (storm
  collapsed; knee 115→~120). Both byte-exact, default off.
- **DEAD ENDS:** fp16/bf16 inference (0.79× — *slower*; not compute-bound); a bigger GPU for a *single* process
  (GIL-capped regardless); rc0 (crashes upstream NeMo on this checkpoint).

## Reference implementations
- `ec2-bench/` (the benchmark toolkit + runbook); `run_multiproc.sh` (the multi-process test) is the prototype the
  production launcher derives from.
- `src/nemotron_speech/server.py` — the lanes (`NEMOTRON_MODEL_LANES`), barrier-drain, finalize-storm flags.
- `proj-2026-05-21-1959-cudagraph/PLAN.md` — the per-B manual CUDA-graph plan (a per-lane cheaper-call lever).
- vLLM/Triton multi-instance + MPS patterns; HAProxy/ALB `leastconn` + `maxconn` for connection-capacity routing.

## Current state
- `server.py`: `NEMOTRON_MODEL_LANES` (replica-per-lane), `NEMOTRON_BATCH_BARRIER_DRAIN`, `NEMOTRON_BATCH_FINALIZE`
  — all flag-gated, default off, byte-exact. Decoder greedy, `use_cuda_graph_decoder=False` (Blackwell-safe).
- `ec2-bench/run_multiproc.sh`: launches K servers (lanes=2, ports 8080+k) + K load-gens; pair with MPS.
- Measured: L40S K=3→48 (MPS, g6e.4xlarge/16 vCPU). L4 multi-process **inferred** ~32 (single-process 16@46%).

## Rules
- **Byte-exact per-stream output** vs the single-stream baseline — gate every server.py change (cache-aware-state hazard).
- Flag-gated; default = current behavior until proven. English rc1 path stays byte-identical.
- Cloud tests on EC2 via `ec2-bench/` (SSO profile `AWSAdministratorAccess-419599258555`, auto-refresh). ALWAYS
  `ec2_down.py` when done; never commit `.pem`/`.instance.json`.
- No new heavy deps; respect the <400 ms production TTFS budget.

## Steps

- [x] **1. Confirm the per-GPU multi-process+MPS matrix (cloud).**
  Measure what's currently inferred: (a) **L4 / g6** multi-process+MPS (`K_LIST=1,2,3`, lanes=2) → confirm ~32/box +
  where it GPU-saturates; (b) **L40S / bigger vCPU** (`g6e.8xlarge`, 32 vCPU) → does K=4–6 reach ~64–80 (is the
  prior K=3 cap really vCPU)? Record per (GPU, size): K*, GPU%, knee, $/stream. Note: co-located load-gen steals
  vCPUs → production cap is higher; if feasible, run the load-gen from a 2nd instance to remove that confound.
  Key files: `ec2-bench/run_multiproc.sh`, `proj-2026-05-21-inference-opt/g6-vs-g6e-results.md`

- [ ] **2. Production multi-process launcher + MPS (the deployable unit).**
  A launcher (systemd unit / container entrypoint) that: starts the MPS daemon, spawns K `server.py` processes
  (lanes=2, ports 8080+k) with readiness/health endpoints, and restarts a crashed process. K + lanes from config
  (per-GPU matrix from Step 1). Harden the **MPS blast-radius**: a CUDA fault in one client can corrupt the shared
  context — add per-process supervision + fast restart, and document the isolation tradeoff (vs MIG on A100/H100).
  Key files: `ec2-bench/` (derive from `run_multiproc.sh`), new `deploy/` launcher

- [ ] **3. Routing layer.**
  LB (HAProxy / nginx / ALB) with `leastconn` + per-backend `maxconn` ≈ 12 (~75% of the 16 knee, for the 400 ms TTFS
  headroom) + health-check + connection-drain, fronting all process-ports across boxes. WS streams are sticky-for-life
  (no mid-stream rebalance). Custom proxy only if lag-aware routing is later needed. Validate end-to-end keep-up +
  per-box knee through the LB.
  Key files: `deploy/` (LB config), `ec2-bench/`

- [ ] **4. Per-GPU config matrix + guarded auto-select.**
  Encode (detected GPU, vCPU) → (lanes=2, K, MAX_SIZE, MPS on) from Steps 1–2; the launcher picks at startup and
  refuses known-bad configs (e.g. K beyond the measured GPU/vCPU cap). Default off / explicit until validated.
  Key files: `src/nemotron_speech/server.py` (or the launcher), config

- [~] **5. Finish the finalize/TTFS chain.** (running in parallel with Step 1 — independent, local Codex)
  With barrier-drain + finalize-storm on, attack the *next* in-phase limiter (round 3's finding): batch finalize
  **preprocessing** (the reverted attempt dropped terminal punctuation — fix the per-fork preprocessing grouping to
  stay byte-exact) and the **close/cold-reset** cleanup path. Gate: in-phase byte-exact + knee past ~120 toward the
  ~180 forced-batch ceiling. (TTFS is already ~40–150 ms out-of-phase; this is burst robustness.)
  Key files: `src/nemotron_speech/server.py`, `proj-2026-05-21-0410/inphase_loadgen.py`

- [ ] **6. Per-B manual CUDA-graphs (re-scoped).**
  Execute `proj-2026-05-21-1959-cudagraph/PLAN.md` as the **per-lane cheaper-call** lever (collapse launch dispatch
  → each lane/process does more → fewer processes to fill the GPU / lower vCPU pressure). Re-rank vs multi-process:
  most valuable where vCPU-bound (Step 1 will show how vCPU-bound we are). Byte-exact per B, fail-closed, default off.
  Key files: `src/nemotron_speech/server.py`, `src/nemotron_speech/cudagraph_encoder.py`

- [ ] **7. Deployment substrate + SageMaker.**
  Target EC2/ECS/EKS + ALB for the WebSocket service (SageMaker real-time endpoints are request/response — wrong fit).
  If SageMaker is required, custom-container + the multi-process launcher behind the routing layer. Validate the
  per-box knee + $/stream on the actual production instance type (the knee is CPU-bound — Modal/local don't predict it).
  Key files: `src/nemotron_speech/modal/` (reference), `deploy/`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Per-GPU multi-process+MPS matrix (cloud) | done | (this commit) | L4 K=2->32 (GPU-bound); L40S 48@16vCPU, 64@32vCPU (ceiling); full matrix measured |
| 2 | Production multi-process launcher + MPS | pending | — | MPS blast-radius hardening |
| 3 | Routing layer (LB leastconn+maxconn) | pending | — | local+remote process-ports |
| 4 | Per-GPU config matrix + auto-select | pending | — | (GPU,vCPU)→(lanes,K,MAX_SIZE) |
| 5 | Finish finalize/TTFS chain | pending | — | finalize preprocessing + close/cold-reset, byte-exact |
| 6 | Per-B manual CUDA-graphs (re-scoped) | pending | — | per-lane cheaper-call; re-rank vs multi-process |
| 7 | Deployment substrate + SageMaker | pending | — | EC2/ECS + ALB; not SM real-time endpoints |
