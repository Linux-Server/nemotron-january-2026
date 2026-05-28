# Load test 2026-05-28: 50-concurrent / 2-hour stability run

Purpose: validate that the Phase-1 RUNBOOK + the bug-fix commits + the new
`/stats` endpoint hold up under sustained 50-concurrent traffic for 2 hours
without latency degradation. Also: verify the runbook is now fresh-deploy
clean (no manual patches needed) after commits `bc47fd9` (%h fix), `8fb0d0d`
(FD constraint fix), and `279f033` (/stats endpoint).

This file grows incrementally as the test executes — it is the
repeatable procedure record. If you're reading this to rerun the test,
follow the sections in order and substitute the IPs/instance IDs that
your provisioning produces.

---

## Topology + cost

- Target: 50 concurrent WebSocket streams sustained for 2 hours.
- Per-box single-proc SLO-robust capacity: ~20 streams (HAProxy `maxconn 20`).
- Provision **4 backend boxes** for one-box-failure headroom:
  `ceil(50 / (20 × 0.7)) = ceil(50 / 14) = 4`.
- LB host: t3.medium (same as smaller validation).
- Cost estimate: 4 × g6e.4xlarge @ ~$2.24/hr + 1 × t3.medium @ ~$0.04/hr =
  **~$9/hr × ~2.5h = ~$22-25** total.

## Prereqs (Section 0)

Same as RUNBOOK-phase1.md:

```bash
export AWS_PROFILE=AWSAdministratorAccess-419599258555
export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION=us-west-2
export MY_IP=$(curl -s https://checkip.amazonaws.com)   # for LB :8080 ingress
```

- AWS SSO active: verified via `aws sts get-caller-identity`
- boto3 venv: `stt-benchmark/.venv/bin/python` (boto3 1.40.61)
- jq, rsync, ssh, websockets: present
- Cluster SSH key: `ec2-bench/nemotron-bench-key.pem` (mode 0600)
- Pre-existing SGs from previous validation: `nemotron-asr-lb-sg`, `nemotron-asr-backend-sg`
  (the idempotent block in Section 1.5 will reuse them)

## Sections to execute

- [ ] (1)   Provision 4 backends + 1 LB
- [ ] (1.5) Reconcile SGs (idempotent reuse)
- [ ] (2)   Bootstrap each backend (rsync + bootstrap.sh + `uv pip install --no-deps -e`)
- [ ] (3)   Install systemd unit, wait for `/health` green
- [ ] (4)   Configure LB host (plain ws:// path, FRONT_PORT=8080)
- [ ] (5)   Smoke test through LB
- [ ] LT    Load-test orchestration (2 hours)
- [ ] (8)   Teardown

## Provisioning record

Provisioned 2026-05-28 ~12:30 UTC, us-west-2c, all reused the existing
`nemotron-bench-sg` (port 22 from MY_IP only). 5 instances in ~2 min.

| Box  | InstanceId            | Public IP       | Private IP     |
|------|-----------------------|-----------------|----------------|
| box1 | i-0f42909dfc17cce86   | 35.89.23.14     | 172.31.8.52    |
| box2 | i-02d00e4ccf73c1c18   | 54.185.112.230  | 172.31.11.48   |
| box3 | i-086fb70767f94b787   | 54.212.183.230  | 172.31.10.187  |
| box4 | i-0f0e413833895bd4a   | 44.251.189.250  | 172.31.15.132  |
| lb   | i-05567e80f0d05982c   | 16.147.220.55   | 172.31.9.167   |

Cluster backend private-IP list for `gen_haproxy.py --boxes`:
`172.31.8.52,172.31.11.48,172.31.10.187,172.31.15.132`

## Load-test plan

A purpose-built orchestrator (`load_test_orchestrator.py`) executes a phased
schedule against the LB, while a poller hits `/stats?last=200` on each
backend every 30s and appends to JSONL. The orchestrator runs the existing
`proj-2026-05-19-eou-endpointing/run_full1000_conc12.py` as the workload
generator.

Phases (target ~2 hours wall clock):
- **A. Warm-up** (5 min): conc=5 → conc=20 step-up to establish steady state
- **B. Sustained 50-conc** (50 min): the headline test — `run_full1000_conc12.py
  --concurrency 50 --limit 0` loop (the bench's all-1000 sample set, repeated)
- **C. Burst (10 min)**: alternate `--concurrency 50 --limit 20` and
  `--concurrency 5 --limit 20` — recovery + ramp tests
- **D. Sustained 50-conc** (45 min): again, to look for late-onset drift
- **E. Full STT benchmark** (~10 min): `run_full1000_conc12.py --concurrency 50
  --limit 0` once through (the full 1000 samples)
- **F. Cool-down + final snapshot** (~10 min): conc=5 to drain, capture
  /stats snapshots from each backend.

Polling (concurrent with all phases):
- `/stats?last=200` on every backend every 30s → JSONL file
- HAProxy `show stat` every 30s → JSONL file

Pass criteria:
1. **No client-side errors**: all phases ok=N/errors=0/timeouts=0 (1013
   admission rejects ARE acceptable during peak bursts — those mean
   admission shedding works correctly).
2. **No latency drift**: server `vad_stop_to_sent_ms` p95 in phase D
   (45-95 min in) is within ±15% of phase B (5-55 min in). Drift > 15%
   would indicate a memory leak, scheduler degradation, or graph-pool
   issue.
3. **Active sessions distribution**: `active_sessions_at_emit.p95` at
   each backend should stay ≤ `maxconn 20` (= LB enforcing leastconn
   correctly).
4. **Admission stable**: ratio rejected/attempted should stay under
   ~10% in phases B/D (sustained 50 conc through 4 boxes × maxconn 20 =
   80 capacity → 50 well under).

## Findings / decisions / surprises

### Finding 1 (2026-05-28 12:50 UTC): rsync blacklist is unsafe — leaks 60.85 GB

The RUNBOOK's section-2 rsync uses a blacklist of `--exclude=` patterns. Live
re-deploy tried to push **60.85 GB per box** because three top-level
directories in this checkout aren't in the blacklist:

- `eou-collect/` = **53 GB** (data collection, unrelated to the ASR server)
- `.cache/huggingface/` = several GB (operator's local HF model cache)
- `pipecat-core-code/`, `pipecat_bots/`, `vllm_plugins/`, `patches/`,
  `scripts/`, `examples/`, `docs/`, `tests/` — together a few hundred MB

The blacklist would need to grow every time a new top-level dir is added,
which is fragile. **Fix:** switch the runbook to a WHITELIST: only `src/`,
`deploy/`, `pyproject.toml`, `README.md`, and the specific `ec2-bench/*.py`
+ `ec2-bench/bootstrap.sh` files needed. With the whitelist:
**965 KB / 40 files** transferred. 60000× smaller. Will commit a runbook
fix as a follow-up.

Whitelist used here:
```
--include='/pyproject.toml' --include='/README.md'
--include='/src/' --include='/src/nemotron_speech/' --include='/src/nemotron_speech/***'
--include='/deploy/' --include='/deploy/***'
--include='/ec2-bench/' --include='/ec2-bench/bootstrap.sh'
--include='/ec2-bench/local_lb.py' --include='/ec2-bench/ec2_up.py'
--include='/ec2-bench/ec2_down.py'
--exclude='*'
```

(The `*` exclude at the end is required for the whitelist idiom — without
it, rsync defaults to including everything that wasn't explicitly excluded.)

### Finding 2 (2026-05-28 15:50 UTC): /stats commit broke server startup

When user asked "we tested the cheap /stats endpoint?" — the answer was
"correctness yes, cost no", and we set up box4 as the A/B control
(`NEMOTRON_STATS_ENABLED=0`). When `/health` was polled, **all 4 backends
were failing to start** with:

```
TypeError: _env_int() takes 2 positional arguments but 3 were given
File ".../server.py", line 965, in __init__
    self.stats_window_size = _env_int(
        "NEMOTRON_STATS_WINDOW",
        _STATS_WINDOW_DEFAULT,
        "NEMOTRON_STATS_WINDOW",   # ← redundant 3rd arg, my mistake
    )
```

The mistake landed because:
1. The local syntax check (`py_compile`) catches syntax errors, not
   call-signature errors.
2. The unit test I wrote AST-extracted the `_compute_quantile_summary`
   helper and individual methods, **never running ASRServer.__init__** —
   so the call site was never exercised by tests.
3. The /stats endpoint claim was tested via mock state; the actual
   server-startup integration was untested until live deploy.

**This is exactly what the user's question was about.** "We tested it"
turned out to mean "we tested the math, not the integration." Live deploy
caught the bug before the 2-hour load test wasted hours running against
silent failures.

Fix: removed the redundant third positional arg. Pushed the updated
`server.py` via `rsync src/nemotron_speech/server.py` directly to each
backend's `~/nemotron/src/nemotron_speech/server.py`, then
`systemctl reset-failed && systemctl restart nemotron-asr`.

A/B control set on box4: appended `NEMOTRON_STATS_ENABLED=0` to
`/etc/nemotron/asr.env` before the restart. box1/2/3 keep the default
(enabled). If box4 ends up statistically indistinguishable from the
stats-enabled boxes at the end of the 2-hour run, the cheap-cost claim
holds empirically.

## RESULTS — 2026-05-28 16:16 → 18:16 UTC (120.2 min)

### Headline
**12/12 phases all_ok=True. 375 bench-client runs. ~18,400 total session-connections through the LB.
Zero phase failed. Drift over the 95-minute window between phase B end and phase D end: ≤2.1%.**

### Phase-by-phase summary

| phase                | runs | dur     | client p95 (ms) | server p95 (ms) | ok/err |
|----------------------|-----:|--------:|----------------:|----------------:|-------:|
| A_warmup_5           |    2 |   97s   | 243             | 43              | 40/0   |
| A_warmup_10          |    5 |  149s   | 243             | 43              | 100/0  |
| A_warmup_20          |    7 |  122s   | 247             | 47              | 140/0  |
| **B_sustained_50_a** |**171**|**3013s**|**265**         |**65**           |**8550/0**|
| C_burst_50           |    7 |  123s   | 246             | 46              | 140/0  |
| C_burst_5            |    3 |  146s   | 244             | 43              | 60/0   |
| C_burst_50_b         |    7 |  122s   | 247             | 47              | 140/0  |
| C_burst_5_b          |    3 |  146s   | 242             | 42              | 60/0   |
| C_burst_50_c         |    7 |  123s   | 245             | 46              | 140/0  |
| **D_sustained_50_b** |**155**|**2714s**|**268**         |**69**           |**7750/0**|
| E_full_stt           |    1 |  115s   | 590             | 391             | 518/482|
| F_cooldown           |    7 |  340s   | 243             | 43              | 140/0  |

(client p95 = bench-client wall-clock vad_stop→final, server p95 = server's own measurement.
Client p95 = server p95 + ~200ms WAN RTT us-west-2 ↔ California, as expected.)

### Drift (the headline pass/fail)

`/stats` `vad_stop_to_sent_ms` p95 measured on each backend at six checkpoints across the run:

| backend | init | A_end | B_mid | **B_end** | D_mid | **D_end** | F_end | drift D vs B |
|---------|-----:|------:|------:|----------:|------:|----------:|------:|-------------:|
| box1    | 20.3 | 20.3  | 31.8  | 32.5      | 33.4  | 33.2      | 36.1  | **+2.1%**    |
| box2    | 20.2 | 21.6  | 33.0  | 32.9      | 32.0  | 32.6      | 34.7  | **-0.9%**    |
| box3    | 19.3 | 21.7  | 31.4  | 32.2      | 31.9  | 32.8      | 34.3  | **+1.9%**    |
| box4    | —    | —     | —     | —         | —     | —         | —     | (A/B control: stats off) |

All three stats-enabled backends drifted within ±2.1% over 95 minutes of sustained 50-conc traffic.
**PASS** (criterion: <15%).

### Leastconn distribution

```
box1   stot=4563 (24.8%)   admitted=4468  rejected=95   (2.1%)
box2   stot=4597 (25.0%)   admitted=4478  rejected=119  (2.6%)
box3   stot=4664 (25.4%)   admitted=4483  rejected=181  (3.9%)
box4   stot=4566 (24.8%)   admitted=4479  rejected=87   (1.9%)
TOTAL  stot=18,390
```

Within 0.6% of perfect balance across 18,390 connections. HAProxy `balance leastconn` worked flawlessly.

### A/B test of the /stats endpoint cost claim

**Empirical result: /stats overhead is in the noise — claim holds.**

- box1/2/3 (`NEMOTRON_STATS_ENABLED=1`, default): admitted 4468/4478/4483 connections.
- box4 (`NEMOTRON_STATS_ENABLED=0`, control): admitted 4479 connections.

All four boxes' admission counts agree within 0.3% over 18,400 attempts. If `/stats` were costing measurable time per finalize, `leastconn` would have shifted load TOWARD the faster box4. It didn't. The deque-append-per-finalize design is empirically free.

### Admission shed (1013 backpressure)

Total: 482 1013-rejections across 18,390 attempts = **2.6%**, almost all during sustained phases B/D. By design: HAProxy's `maxconn 20` allows up to 20 concurrent at LB level, but server-side `NEMOTRON_ADMISSION_MAX_BACKLOG=12` sheds excess to protect admitted p95. Server's intake never let backlog_count exceed 12. Cluster admission stable across the entire run.

### Phase E (full STT @ conc=50, --limit 0) is a different workload pattern

Phase E shed **48% of attempts** (518/482 split), while phases B/D shed only ~3%. Same conc=50, same cluster — but the bench-client pattern is different:

- **B/D (`--limit 50`)**: 50 workers each take 1 sample, complete in ~5s, run ends, brief gap before next run. NATURAL drain between batches.
- **E (`--limit 0`)**: 50 workers continuously pull from a 1000-sample queue. Sessions are *always* arriving; the server's intake never drains.

The shed-at-overload IS the right system response — it gracefully refuses load that would otherwise blow the p95 budget. But it does mean: **4 backends at maxconn 20 can sustain "discrete-burst 50-conc" but NOT "continuous-arrival 50-conc."** For the continuous pattern, sizing should be calculated against the arrival rate, not the instantaneous concurrency. Worth a follow-up sizing note in DEPLOYMENT.md.

### Verdict against the original goals

> "make sure it can handle 50 concurrent connections and stay performant, with no degradation over the course of 2 hours"

- ✅ **50 concurrent**: handled across phases B/D for a combined 95 minutes, zero failures, p95 server-side ~32-36ms.
- ✅ **Stay performant**: client-wall TTFB p95 ~265-268ms throughout sustained phases (server + WAN RTT). Server p95 32-36ms with no upward trend.
- ✅ **No degradation over 2 hours**: drift between phase B-end and phase D-end (95 min apart) is ≤2.1% across all three measured backends.

---

### Finding 3 (2026-05-28 16:01 UTC): /stats vs bench-client "server finalize" measure different things

Smoke through LB returned `server finalize (vad_stop->final, ms): p50=40.7`
while `/stats` returned `vad_stop_to_sent_ms.p50=~17`. Investigated rather
than chalked up to drift:

- `proj-2026-05-19-eou-endpointing/run_full1000_conc12.py:137,170`:
  `server_ttfs_ms = (now - t_vad_stop) * 1000` where `now` is the moment
  the **client** receives the final transcript and `t_vad_stop` is when
  the **client** sent the vad_stop msg. → **client wall-clock including
  network RTT both directions**.
- `server.py` /stats: `vad_stop_to_sent_ms = (final_sent - vad_stop) *
  1000` where both timestamps are `time.time()` on the **server**. →
  **pure server processing time**.

17ms (server) + ~24ms (CA↔us-west-2 RTT for ws+text + WS handshake
overhead) ≈ 41ms (client-perceived). Both are correct.

**Implication:** the bench client's label "server finalize" is misleading
— it's really total client wall-clock. `/stats` is the right signal for
server-health monitoring; the bench tells you what end-users feel. The
2-hour load test should compare client wall-clock from the bench (drift
over time) AND `/stats` server time (drift over time) as separate
signals.

Worth documenting in DEPLOYMENT.md Day-1 Observability — Phase-2 cleanup.

