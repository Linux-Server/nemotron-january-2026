# ec2-bench — benchmark the Nemotron streaming ASR server on EC2 GPU instances

Spin up an EC2 GPU box, deploy the **current local** server code (no GitHub needed), run the realtime
keep-up knee sweeps (baseline / batching / parallel-lanes), and tear it down. Built to measure the
**per-instance knee on the real target CPU+GPU** — Modal is only a proxy; the knee is single-thread-CPU-bound,
so it must be measured on the actual instance (see `memory/deployment-target-sagemaker`).

## Why this exists / key findings it produced
- The server's per-instance knee is **single-core launch-dispatch bound** — one core pegs at 100% while the GPU
  sits at ~46% (measured on g6/L4). Levers: **parallel lanes** (`NEMOTRON_MODEL_LANES`) use more cores to fill
  the GPU. L4 fills at **lanes=2 → knee ~16** (lanes=4 oversubscribes → regresses).
- Run server **and** load-gen on one box (no ingress / no WAN); the load-gen is ~7% CPU at the knee, so it does
  not confound the measurement.

## Apples-to-apples measurement notes (learned 2026-05-27, comparing Python vs the native runtime)
- **Same metric**: server-side **ttfs = vad_stop→final over the local WS** (no WAN, no endpoint-wait); `ec2_loadgen.py`
  measures exactly this. Budget: ttfs **p95≤175 / p99≤250ms**, keep-up **lag p95<500ms** (= 400 − VAD~200 − WAN~25).
- **Same arrival pattern**: the loadgen fires a **synchronized burst** (N concurrent conns, no stagger). Match the
  other side — the native density harness must also run **no-stagger** (a per-worker start-stagger improved native
  N=36 ttfs p99 to 50 vs 147 no-stagger; comparing staggered-vs-burst is a real confound).
- **Single-proc vs MPS**: run **one process on the full GPU** to isolate runtime/GIL efficiency; **K-proc+MPS** is
  the deployment config but adds an MPS latency tax at ≈no density gain (see `deploy/DEPLOYMENT.md`). Say which you ran.
  Same-box result: 1 Python proc ≈ **~20 streams @ ttfs p50 ~42ms**; native (1 process) ≈ **~36 @ p99 147ms**.
- **p99 needs samples**: pool **conc×rounds ≳ 100–200** per level before trusting a p99 (bump `--rounds`).
- **Coverage gap**: the loadgen is **one utterance per connection** — sustained *multi-turn* load per stream (where
  the per-proc asyncio intake bottleneck bites) is **not** exercised. It's the open caveat behind the single-proc lead.

## Prerequisites
- AWS creds — scripts default to the **auto-refreshing SSO profile** `AWSAdministratorAccess-419599258555`
  (set up once: `aws sso login --sso-session khk`; boto3 then mints fresh role creds on its own). Override with
  the `NEMOTRON_AWS_PROFILE` env var. (Pasted static temp creds also work but expire mid-run — prefer SSO.)
- `boto3` — use `stt-benchmark/.venv/bin/python` (already has it); no `aws` CLI needed.
- EC2 G-instance quota in the region (us-west-2: On-Demand G = 768 vCPU, wide open).
- Your workstation's public IP for the SSH security-group rule — set `MY_IP` (default is baked into `ec2_up.py`;
  re-detect with `curl -s https://checkip.amazonaws.com`).

## Workflow
```bash
PY=stt-benchmark/.venv/bin/python

# 1. launch (default g6.4xlarge; override the type)
NEMOTRON_EC2_ITYPE=g6e.4xlarge $PY ec2-bench/ec2_up.py      # writes ec2-bench/.instance.json (id+ip+key)

# 2. deploy code from THIS dir (committed server.py by default; --working for the working tree)
bash ec2-bench/ec2_push.sh                                  # rsyncs server.py + batch_primitives + scripts + audio

# 3. bootstrap the runtime ON the box (mirrors the Modal image: torch + NeMo@056d937 + checkpoint) — once
IP=$($PY -c "import json;print(json.load(open('ec2-bench/.instance.json'))['ip'])")
ssh -i ec2-bench/nemotron-bench-key.pem -o StrictHostKeyChecking=no ubuntu@$IP \
    'cd ~/nemotron && nohup bash bootstrap.sh > bootstrap.log 2>&1 & echo started; sleep 1; tail -f bootstrap.log'
# wait for "[bootstrap ...] DONE" (~3-5 min: pip + NeMo-from-git build + checkpoint download)

# 4. run the sweeps (on the box)
ssh ... ubuntu@$IP 'cd ~/nemotron && bash run_bench.sh'                       # B=1 baseline + batched
ssh ... ubuntu@$IP 'cd ~/nemotron && LANES=1,2,4,6,8 SWEEP=1,4,8,...,80 bash run_lanes.sh'  # lane sweep
#   results land as ~/nemotron/{baseline,batched,lanesN}.json (+ stdout knee tables)

# 5. ALWAYS tear down (stops billing; SG + key are kept for reuse)
$PY ec2-bench/ec2_down.py
```

## Scripts
| file | role |
|---|---|
| `ec2_up.py` | launch/reuse a `nemotron-bench-<itype>` instance (DLAMI Base GPU AMI, key pair, no-ingress-except-22 SG, public IP, 200 GB gp3). Writes `.instance.json`. Idempotent. |
| `ec2_push.sh` | rsync the bundle from this dir → `~/nemotron/` on the box. Default = **committed** `server.py`; `--working` = working-tree (for lanes/graphs runs). |
| `bootstrap.sh` | on-box: uv venv (py3.11), `torch` + deps + `nemo_toolkit[asr]@git+…@056d937`, download the public checkpoint, smoke. |
| `run_bench.sh` | on-box: B=1 baseline + batched knee sweeps → `baseline.json`, `batched.json`. |
| `run_lanes.sh` | on-box: B=1 baseline + `NEMOTRON_MODEL_LANES` sweep (env `LANES`, `SWEEP`). |
| `run_multiproc.sh` | on-box: multi-PROCESS scaling test — K server processes (each lanes=2) + K concurrent load-gens (env `K_LIST`, `N_PER`); pair with CUDA MPS for concurrent GPU sharing. |
| `ec2_loadgen.py` | standalone in-box load-gen (lifted from `coloc_loadgen.py`): N concurrent realtime WS streams from `loadgen_audio/*.pcm`, reports proc-lag + TTFS **p50/p95/p99 + sample count + max** + lag p99 + the keep-up knee. `--rounds R` pools N×R samples/level (raise it for a trustworthy p99). |
| `ec2_down.py` | terminate the instance. |

## Config knobs
- `NEMOTRON_EC2_ITYPE` (ec2_up): instance type, e.g. `g6.2xlarge`, `g6e.4xlarge`.
- `LANES`, `SWEEP`, `SKIP_BASELINE` (run_lanes): lane counts / concurrency levels.
- Server config baked into the run scripts: `silence0_warm200` (`NEMOTRON_CONTINUOUS=1`,
  `NEMOTRON_FINALIZE_SILENCE_MS=0`, `NEMOTRON_WARMUP_MS=200`), rc1, batching = `NEMOTRON_SCHEDULER_B1=1` +
  `NEMOTRON_BATCH_SCHED=1`.

## Gotchas (learned the hard way)
- **cuDNN**: the DLAMI's `LD_LIBRARY_PATH` points to system cuDNN 9.13, but pip-torch bundles 9.20 → version
  clash on model load. The run scripts launch the server with `env -u LD_LIBRARY_PATH` so torch uses its bundled
  cuDNN. (If you launch the server by hand, do the same.)
- **Checkpoint** `nvidia/nemotron-speech-streaming-en-0.6b` is public (no HF token needed); `server.py --model`
  takes the HF id directly.
- **Lanes need scheduler + batching on**, and load one model replica each (memory: ~2.4 GB × lanes).
- **Cost**: g6 ~$1/hr, g6e ~$2–4/hr (EC2 on-demand). Always `ec2_down.py` when done — the box bills while running.
- The `.pem` private key and `.instance.json` are git-ignored — do not commit them.
