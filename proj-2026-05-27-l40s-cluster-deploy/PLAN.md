# Plan: L40S cluster deploy — Phase 1 (single-proc Python server + leastconn LB)

Project directory: `./proj-2026-05-27-l40s-cluster-deploy`

## Context
Stand up a deployable cluster of L40S boxes serving the production Python Nemotron streaming-STT
server (`src/nemotron_speech/server.py`), fronted by a least-connections WebSocket load balancer.
Phase 1 ships **one `server.py` process per box, no CUDA MPS** (per `deploy/DEPLOYMENT.md:23`,
2026-05-27: a single proc on a full L40S holds ~20 streams at p50 ~42–54ms — matching K=3+MPS
density without the ~190ms median-latency tax). Deliverable = **production-ready deploy scripts +
runbook**, deployed by hand; **no IaC and no live cluster** stood up in this phase.

## Reference implementations
The existing `deploy/` artifacts are the templates to adapt — all are labeled "DESIGN ARTIFACT" and
must be promoted to runnable, single-proc form:
- `deploy/launch_multiproc.sh:13-96` — MPS + K-proc launcher. **Reuse** only the `SRV_ENV` stack
  (`launch_multiproc.sh:49-67`); **drop** MPS (`start_mps`/`stop_mps` `:71-72,81`), the K-loop
  (`:82`), AND the supervisor loop (`:87-96`) — for the single-proc variant, systemd is the
  supervisor (see Rules → Correctness/ops safety).
- `deploy/haproxy.cfg.example:1-48` — leastconn HAProxy template (`balance leastconn` `:31`,
  `httpchk GET /health` `:23`, per-server `maxconn` `:35-39`, drain notes `:42-48`). NB: the existing
  example uses `mode http` (`:17`), but for WebSocket the generator emits `mode tcp` — the example's
  shape and rationale carry over, the mode flips. The generator also emits **one backend line per
  box on :8080, `maxconn ~20`** (not 3 ports/box like the K=3+MPS example).
- `ec2-bench/local_lb.py:1-81` — working asyncio leastconn TCP balancer; the local stand-in used by the
  smoke test. Leave as-is; reference it in the runbook.
- `ec2-bench/bootstrap.sh:1-46` — proven box bootstrap (venv + torch + NeMo @ pinned commit `056d937` +
  checkpoint pre-download). Reused verbatim by the runbook.
- `ec2-bench/ec2_up.py:1-133` — g6e/L40S provisioner (`NEMOTRON_EC2_ITYPE=g6e.4xlarge`, writes
  `.instance.json`). Reused by the runbook; **not** wrapped in IaC this phase.

Server facts the scripts depend on (verified):
- WS handler on `/`, health on `/health` (`server.py:10075-10076`), default port 8080.
- `/health` **always returns HTTP 200** (`server.py:10035-10043`) with JSON `{"status":"healthy"|"loading", …}`.
  The server does not bind the port until *after* `load_model()` completes (`server.py:10045-10081`),
  so in practice HAProxy gets connection-refused during load and `healthy` once bound — but defensively
  the LB check uses `http-check expect rstring "\"status\"[ ]*:[ ]*\"healthy\""` (regex anchored
  to the JSON field) to distinguish.
- Admission reject closes the WS with **code 1013** (`server.py:5062`), gated by
  `NEMOTRON_ADMISSION_MAX_BACKLOG` (`server.py:900-905`, default effectively off = 1e9). `/health`
  surfaces `admission.attempted/admitted/rejected` (`server.py:5017-5025`).
- Launch: `python -m nemotron_speech.server` **or** `python server.py --model … --host 0.0.0.0
  --port 8080 --right-context 1` (`launch_multiproc.sh:77-78`). **Phase 1 uses the module form**
  (`python -m nemotron_speech.server`) so deploys can use a proper editable install (`uv pip install -e .`)
  instead of the legacy benchmark flat-copy pattern (`ec2_push.sh:14-26`).

**Env-var canonical namespace (lifted from `launch_multiproc.sh:49-67`, deduped):** server.py reads
`NEMOTRON_*` prefixed names *only*. The unprefixed aliases the multi-proc script accepts
(`FINALIZE_PADDED`, `SYNC_COMPRESS`, `FINALIZE_PRIORITY`, `FINALIZE_PROFILE`, `FAULTHANDLER`,
`ADMISSION_MAX_BACKLOG`, `ADMISSION_MAX_READY_AGE_MS`, `FINALIZE_T_MIN`, `FINALIZE_T_MAX`) are shell-level
indirection — multi-proc convenience that does not belong in single-proc. `launch_single.sh` and
`asr.env.example` use **`NEMOTRON_*` everywhere** (no aliases) for clarity.

No external (vLLM/SGLang) references apply — this is ops/serving glue around an existing server.

## Current state
- `deploy/DEPLOYMENT.md` — design doc; K=3+MPS is the primary topology, with the 2026-05-27 single-proc
  lead recorded only as an inline caveat (`:23`). Must be restructured so single-proc is the Phase-1
  primary and K=3+MPS is the documented fallback.
- `deploy/launch_multiproc.sh` — exists; the SRV_ENV (`:49-55`) is the reusable production config.
  In `launch_single.sh` we use the canonical NEMOTRON_*-prefixed names directly (server.py reads
  these — see the canonical-namespace rule in Reference implementations):
  `NEMOTRON_CONTINUOUS=1 NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200
  NEMOTRON_SCHEDULER_B1=1 NEMOTRON_BATCH_SCHED=1 NEMOTRON_BATCH_MAX_SIZE=32
  NEMOTRON_BATCH_MAX_WAIT_MS=8 NEMOTRON_MODEL_LANES=2 NEMOTRON_BATCH_BARRIER_DRAIN=1
  NEMOTRON_BATCH_FINALIZE=1 NEMOTRON_ENCODER_CUDAGRAPH=1 NEMOTRON_ENCODER_CUDAGRAPH_MAX_B=8
  NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE=1 NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1
  NEMOTRON_SYNC_COMPRESS=1 NEMOTRON_FINALIZE_PRIORITY=1`. Note `env -u LD_LIBRARY_PATH` (`:77`)
  is required for torch's bundled cuDNN.
- `deploy/haproxy.cfg.example`, `ec2-bench/local_lb.py`, `ec2-bench/bootstrap.sh`, `ec2-bench/ec2_up.py` —
  as described above.
- **Missing:** single-proc launcher, systemd unit, HAProxy config generator, drain automation, smoke-test
  script, updated runbook.

New files land in `deploy/` (production artifacts) with the runbook as `deploy/RUNBOOK-phase1.md`.

## Rules
(No `PLAN_RULES.md` at repo root — only the rules below apply.)

### Scope discipline
- **Single-proc, no MPS** is the Phase-1 topology. Do not add MPS, multi-proc, or K>1 paths to the new
  launcher — K=3+MPS stays only as the *existing* `launch_multiproc.sh` (untouched) + a documented fallback.
- **No IaC, no live infra**: do not write Terraform/CloudFormation/ASG/ALB definitions and do not launch
  any EC2 instance from these scripts. `ec2_up.py` is referenced from the runbook, not invoked by the plan.
- **Don't touch** `src/nemotron_speech/server.py` or the WIP C++ server. This is deploy glue only.
- Preserve the exact production `SRV_ENV` semantics (incl. `env -u LD_LIBRARY_PATH`); the deltas
  from `launch_multiproc.sh` are: removing MPS, the K-loop, AND the in-script supervisor (systemd
  is the sole supervisor — see Correctness/ops safety), dropping the log-file redirect (journald
  owns logs), and switching to module form (`python -m nemotron_speech.server`).

### Correctness / ops safety
- **systemd is the sole supervisor.** The launcher does NOT contain an in-script restart loop — it
  `exec`s into python so systemd's `Restart=on-failure` is the single restart mechanism. (Round 1
  caught the double-supervisor bug; don't reintroduce it during implementation.)
- Launcher must be SIGTERM-clean: with `exec`, SIGTERM goes straight to python which closes the WS
  server cleanly. No bash trap, no `pkill`, no log-redirect (journald owns logs).
- HAProxy config must be WebSocket-safe: `timeout client/server 1h`, `balance leastconn`,
  `httpchk GET /health`, per-box `maxconn` (default ~20, parameterizable), and a runtime
  socket (`stats socket`) so drain automation can issue `set server … state drain`.
- Drain must support both directions: (a) operator-initiated drain-before-deploy via the HAProxy runtime
  socket; (b) reacting to server admission shedding (WS close **1013**) — document that HAProxy
  `mode tcp` cannot read the WS close code, so the server-side `NEMOTRON_ADMISSION_MAX_BACKLOG` cap
  (default 12) is the backpressure mechanism and the LB role is health-check + operator drain.
- Generator output must be valid HAProxy config (validate with `haproxy -c -f` in the step's check if
  haproxy is available; otherwise structural assertions).

### Validation
- Each shell script: `bash -n` parse check + `shellcheck` if available.
- Python scripts: `python3 -m py_compile` + `python3 -m pyflakes` if available.
- systemd unit: `systemd-analyze verify deploy/nemotron-asr.service` if systemd installed.
- Generator: unit-style check — feed a 2-box fleet list, assert the emitted config has 2
  leastconn backend lines (`server box_…`) on :8080 with `maxconn`, `check`, `inter`/`fall`/`rise`,
  the 1h timeouts, the `option httpchk` + `http-check send meth GET uri /health` +
  `http-check expect rstring "\"status\"[ ]*:[ ]*\"healthy\""` triple, the stats socket,
  `option tcplog`. Negative assertion: `option dontlog-normal` must be ABSENT (it would silence
  production traffic logs).
- `drain.sh` fixture tests (Step 4) must pass.
- Smoke test must run **locally** (no GPU, no cloud) so a laptop run is sufficient; the cloud smoke
  is documented in the runbook, not executed here.

### Documentation (the primary Phase-1 deliverable)
This phase ships **scripts + documentation**, not infra. "Done" means an operator who has never seen the
system can deploy it from the docs without reading code. Every step must produce:
- **Header comment block** in every script (`launch_single.sh`, `gen_haproxy.py`, `drain.sh`, the systemd
  unit, the smoke harness) containing: (a) one-paragraph purpose, (b) invocation/usage with concrete
  examples, (c) every env var consumed with default + meaning + why-it's-set-that-way, (d) every CLI flag
  with type/default/meaning, (e) exit codes / failure modes, (f) relative-path link to
  `deploy/RUNBOOK-phase1.md` and `deploy/DEPLOYMENT.md`.
- **`--help`** output for every Python/CLI tool (`gen_haproxy.py`, `drain.sh` via `-h`) that is sufficient
  to use the tool without reading source.
- **`deploy/asr.env.example`** lists every honored env var (production defaults + comments explaining
  *why* — pointer to the validation that fixed the value).
- **`deploy/README.md`** is the navigation index: one-line description of every artifact in `deploy/`,
  links to RUNBOOK (how) and DEPLOYMENT (why), and a "start here" pointer.
- **Cross-linking:** any artifact that references another (e.g. drain.sh references the systemd unit)
  must link by relative path. No orphan files.
- **Documented failure modes & open risks** in RUNBOOK: server SIGTERM/drain behavior, HAProxy
  `mode tcp` cannot read WS close 1013 (server-side backpressure only), MPS-fallback trigger
  conditions, the un-validated sustained-multi-turn assumption, drain timeout behavior,
  what to do when `/health` never goes green.
- **Procedure vs rationale split:** `RUNBOOK-phase1.md` = step-by-step procedure (the *how*);
  `DEPLOYMENT.md` = topology/rationale/measurements (the *why*). Each links to the other; nothing
  important lives in only one.
- **Every step's "Documentation:" line below is mandatory output**, not optional polish — the step is
  not done until those docs exist and cross-link.

## Steps

- [x] **1. Single-proc launcher (`deploy/launch_single.sh`)**
  Adapt `launch_multiproc.sh` into a single-process, no-MPS launcher. Keep the full production `SRV_ENV`
  but **use canonical `NEMOTRON_*` names directly** (no unprefixed aliases — see "Env-var canonical
  namespace" in Current state). Drop `start_mps`/`stop_mps`, the `CUDA_MPS_*` exports, the
  `auto_pick_K`/K loop, **and the in-script `while true; kill -0; restart` supervisor** — systemd
  (Step 2) is the supervisor; two restart loops fighting each other is a real bug, not a defense-in-depth.
  Launch exactly one `python -m nemotron_speech.server --host 0.0.0.0 --port "${NEMOTRON_PORT:-8080}"
  --right-context "${NEMOTRON_RIGHT_CONTEXT:-1}" --model "$MODEL"` via
  `exec env -u LD_LIBRARY_PATH "${SRV_ENV[@]}" "$VENV/bin/python" -m nemotron_speech.server …`
  (the `exec` is important — systemd's SIGTERM goes to the launcher PID; with `exec` that *is* the
  python process). Module form requires the package be installed editable into `$VENV`
  (`uv pip install -e $NEMOTRON_APP_DIR`); the runbook does this during bootstrap. Drop the `server.log`
  redirect — journald owns logs via systemd. SIGTERM trap is no longer needed (exec hands the signal
  straight through). Honor `NEMOTRON_APP_DIR`, `NEMOTRON_VENV`, `NEMOTRON_MODEL`, `NEMOTRON_PORT`,
  `NEMOTRON_RIGHT_CONTEXT`, `HF_HOME`, and `NEMOTRON_ADMISSION_MAX_BACKLOG` (default to recommended 12).
  Documentation: header comment per the Documentation rule (purpose, invocation example, every env var
  with default + why, exit codes, link to RUNBOOK + DEPLOYMENT); explain why MPS/multi-proc are absent
  (the 2026-05-27 lead + the sustained-load fallback pointer); reference the lines in
  `launch_multiproc.sh` the env stack was lifted from so a future reader can diff them; document that
  SIGTERM = process kill, NOT a graceful WS drain — operator MUST use `drain.sh` first.
  Key files: `deploy/launch_single.sh`

- [x] **2. systemd unit template (`deploy/nemotron-asr.service`)**
  A `systemd` service that runs `launch_single.sh`. `Type=simple`, `Restart=on-failure`, `RestartSec=5`,
  `TimeoutStopSec=120` (a backstop, **not** a drain mechanism — operator uses `drain.sh` first to
  bound the wait; direct `systemctl restart` cuts active streams), `KillSignal=SIGTERM`, `User=ubuntu`,
  `WorkingDirectory=%h/nemotron`, `EnvironmentFile=-/etc/nemotron/asr.env` for overrides, plus inline
  `Environment=` defaults using **`%h`** (systemd specifier) not `$HOME` — `Environment=` does not
  shell-expand. Defaults: `Environment=HF_HOME=%h/hf NEMOTRON_VENV=%h/nemo-venv
  NEMOTRON_APP_DIR=%h/nemotron NEMOTRON_ADMISSION_MAX_BACKLOG=12`. The launcher is invoked
  `ExecStart=%h/nemotron/deploy/launch_single.sh` (chmod +x'd by the runbook). Health-gating note:
  systemd brings the proc up; the LB `/health` check is what gates traffic (the unit itself can't
  easily poll the WS readiness, so keep that at the LB).
  Validation: `systemd-analyze verify deploy/nemotron-asr.service` passes (when systemd is installed).
  Documentation: header comment with install procedure (`cp` to `/etc/systemd/system/`, `daemon-reload`,
  `enable --now`, `journalctl -u nemotron-asr -f` for logs, how to override env via
  `/etc/nemotron/asr.env`); explicit "SIGTERM is a process kill — operator drain first via drain.sh"
  paragraph; `asr.env.example` is structured into a **RECOMMENDED-EXPLICIT** block (values that
  match the launcher defaults but should appear explicitly in the per-box env file for operator
  visibility — `NEMOTRON_ADMISSION_MAX_BACKLOG=12` is the canonical example) and an **OPTIONAL**
  block (touch only when tuning). Nothing is strictly required to make the unit run; the launcher
  defaults are functional. Every `NEMOTRON_*` env var is enumerated with default + comment +
  provenance (which validation pinned it; for the optimization stack, note validation was K=3+MPS
  — see DEPLOYMENT.md for the carry-over assumption). A troubleshooting section at the top —
  common failure modes (model download stall, GPU not visible, `/health` never green, restart loop,
  NeMo install failed) with one-line diagnostics each.
  Key files: `deploy/nemotron-asr.service`, `deploy/asr.env.example`

- [x] **3. HAProxy config generator (`deploy/gen_haproxy.py`)**
  A small Python (stdlib-only) generator. CLI: `--boxes ip1,ip2,…` or `--boxes-file PATH` (one IP per
  line, accepts `ip` or `name=ip`), `--maxconn` default 20 (with `--maxconn-conservative` flag mapping
  to 12 — the safer pre-sustained-multi-turn-validation value), `--front-port` default 8080,
  `--tls-port` default 8443 (frontend uses `--tls-port` when `--tls-pem` is given, else `--front-port`),
  `--tls-pem` optional path, `--stats-socket` default `/run/haproxy/admin.sock`, `-o OUTPUT` (stdout
  if omitted), `--check` (only valid with `-o`; writes then runs `haproxy -c -f OUTPUT`),
  `--local-test` (laptop-safe variant for the smoke test: respects `--stats-socket` if given
  — smoke MUST pass an explicit path under `mktemp -d` — falls back to a generator-chosen temp
  path printed to stderr if not given; drops the `user haproxy`/`group haproxy` directives, drops
  `ulimit-n 200000` to a portable 4096, and uses `log stdout format raw daemon` — so a non-root
  current-user haproxy can run the generated config without needing the haproxy system user or a
  high `nofile` rlimit). Note: HAProxy config does NOT shell-expand `$TMPDIR` — the generator
  writes a fully-resolved socket path. **No `option dontlog-normal`** in defaults — that
  directive would silence all successful TCP connections including real production WS streams.
  HAProxy does not log successful health probes by default (`option log-health-checks` is OFF by
  default), so probe-spam is not a concern; the journald-volume answer is rotation
  (`SystemMaxUse=500M`), not log suppression.
  **Validation:** each IP through `ipaddress.ip_address()`; reject empty fleets, duplicate IPs, and
  any name not matching `^[a-zA-Z0-9_-]+$`. Backend server names default to `box_<ip-with-dashes>`
  (stable across input reordering — so `drain.sh drain box_10-0-1-10` always targets the same machine);
  `--boxes-file`'s `name=ip` form overrides for operator-chosen names.
  Emitted config:
  - `global`: `maxconn 100000`, `stats socket <sock> mode 660 level admin`, `user haproxy`,
    `group haproxy`, `log /dev/log local0`, `ulimit-n 200000`.
  - `defaults`: `mode tcp`, `option tcplog`, `log global`, `timeout connect 5s`, `timeout client 1h`,
    `timeout server 1h`, `timeout queue 5s`.
  - `backend asr_pool`: `balance leastconn`, `option httpchk`, `http-check send meth GET uri /health`,
    `http-check expect rstring "\"status\"[ ]*:[ ]*\"healthy\""` (per server.py:10035-10043, `/health`
    always returns 200; the regex anchored to the JSON field distinguishes `healthy` from `loading`
    AND tolerates future fields that might contain the substring "healthy"). One
    `server box_<ip-dashes> <ip>:8080 check inter 2s fall 3 rise 2 maxconn <maxconn>` per box.
  - `frontend asr_ws`: `bind *:<port>` (with `ssl crt <pem>` if TLS); `default_backend asr_pool`.
  Mirror the comment structure of `haproxy.cfg.example` but single-port-per-box.
  Documentation: `--help` is sufficient to use the tool without reading source (every flag has type,
  default, meaning, example); module docstring covers purpose, the leastconn-equals-streamcount
  rationale (lifted from `haproxy.cfg.example:7-10`), why maxconn 20 (the 2026-05-27 single-proc
  number, *one-utterance-per-connection*) and why `--maxconn-conservative` exists (sustained
  multi-turn unvalidated), the queue policy (`timeout queue 5s` means clients past `maxconn` wait up
  to 5s then fail — explicitly NOT a 1013-style server-side admission reject; HAProxy queues, the
  server sheds), an "operator examples" block (2-box dev, N-box prod with TLS), and a "graceful
  reload" recipe (`systemctl reload haproxy` or `haproxy -sf $(pidof haproxy) -f cfg`); emitted
  `haproxy.cfg` carries inline `#` comments naming the generator, the input fleet, the date, the
  rationale for each non-trivial directive, AND a stable name→IP table so an on-call engineer
  reading the file can match `drain.sh` arguments to real boxes.
  Key files: `deploy/gen_haproxy.py`

- [ ] **4. Drain automation (`deploy/drain.sh`)**
  Operator script wrapping the HAProxy runtime socket via `socat … stdio` (document `socat` as an
  LB-host prereq; the runbook installs it). Subcommands:
  - `drain <box>` → `set server asr_pool/<box> state drain`
  - `ready <box>` → `set server asr_pool/<box> state ready`
  - `status` → emits the parsed `show stat` rows for `asr_pool` (svname, status, scur, smax)
  - `wait-empty <box> [timeout=300]` → polls `show stat` (the Runtime API command — CSV is the
    default output format; `;csv` is stats-page URI syntax, not socket syntax) every 2s until
    `scur==0` for the given server, or timeout. **Parses CSV by header name** (`pxname=asr_pool`,
    `svname=<box>`, `scur`) — NOT by positional column (HAProxy CSV column order is not API-stable
    across versions). The first CSV header row starts with `# pxname,svname,…`; strip the leading
    `# ` before parsing. Use Python (`python3 - <<'PY'` heredoc) to parse so this isn't a fragile
    awk script.
  Exit codes: `0` = success / empty, `1` = timeout with N sessions remaining (print N), `2` =
  backend or server not found, `3` = socket unreachable / permission denied, `4` = bad arguments.
  Configurable socket path via `HAPROXY_SOCK` env (default `/run/haproxy/admin.sock`).
  **Fixture tests:** ship `deploy/_drain_fixtures/` with sample `show stat` CSV (server UP with
  scur>0, drained with scur=0, missing server) and a tiny harness that pipes each fixture through
  the script's parser to assert each exit code path. Runs in the smoke test (Step 5).
  Documentation: header includes the full rolling-deploy procedure as a runnable sequence
  (drain → wait-empty → systemd restart → wait `/health` green → ready), every subcommand has
  a usage line, every exit code is enumerated with operator action, and a "1013 backpressure" note
  explains why HAProxy `mode tcp` cannot react to the WS close code — the server-side
  `NEMOTRON_ADMISSION_MAX_BACKLOG` cap is the real backpressure, this script is the
  *operator-initiated* drain. Document permission setup: operator must be in `haproxy` group
  (socket mode 660), OR run drain.sh under sudo, OR the global `stats socket` uses an explicit
  `user/group` matching the operator. Cross-link to RUNBOOK and to the systemd unit.
  Key files: `deploy/drain.sh`, `deploy/_drain_fixtures/*.csv`

- [ ] **5. Local smoke test (`deploy/smoke_local.sh` + helpers)**
  Laptop-safe end-to-end check with no GPU/cloud. Four independent checks; each runs cleanly or skips:
  (a) **`local_lb.py` leastconn spread + shed-on-full** — start a trivial TCP echo backend on two ports,
  run `ec2-bench/local_lb.py --front … --backends p1,p2 --maxconn 2`, open several connections,
  assert spread is balanced and overflow is closed. Note in PASS message that local_lb.py
  *sheds* overflow while HAProxy *queues* (per `timeout queue`) — they are NOT equivalent and the
  test asserts local_lb's own documented behavior, not HAProxy's.
  (b) **Generator syntax** — `gen_haproxy.py --boxes 10.0.1.10,10.0.1.11 --maxconn 20 -o /tmp/...cfg`
  then `haproxy -c -f /tmp/...cfg` (skip with WARN if `haproxy` not on PATH).
  (c) **Generator health-check behavior (optional, only if `haproxy` installed)** — bring up real
  haproxy as a non-root local process. Smoke allocates `tmpdir=$(mktemp -d)` and runs
  `gen_haproxy.py --local-test --stats-socket "$tmpdir/haproxy.sock" -o "$tmpdir/cfg" …`,
  then `socat` polls `$tmpdir/haproxy.sock` for `show stat` (the smoke controls the socket
  path, no auto-discovery needed). Cleans up `$tmpdir` on exit (trap). With a Python stub backend
  serving `200 OK\n…"status":"healthy"…` on `/health`, assert `show stat` reports the backend UP
  within `inter*rise = 4s` (poll up to 10s). Then flip the stub to `"loading"`, assert DOWN within
  `inter*fall = 6s` (poll up to 12s). Flip back to `"healthy"`, assert UP again. This full
  healthy→loading→healthy cycle is the only test that proves the `http-check expect rstring`
  directive actually works. Skip with WARN if haproxy missing.
  (d) **drain.sh fixture parsing** — pipe each `deploy/_drain_fixtures/*.csv` through `drain.sh`'s
  parser (with `HAPROXY_SOCK` overridden to a mock), assert the expected exit code per fixture.
  Exit 0 only if every non-skipped check passed; print PASS/SKIP/FAIL per check with actual-vs-expected
  on failure.
  Documentation: header lists exactly what's covered, what's skipped under what conditions, and what's
  **not** covered at all (no real GPU/server, no actual WS protocol semantics — those are in the cloud
  smoke in RUNBOOK); how to run it (one invocation, no args); how to interpret each SKIP.
  Key files: `deploy/smoke_local.sh`, `deploy/_smoke_backend.py`, `deploy/_smoke_haproxy_check.py`

- [ ] **6. Rewrite `deploy/DEPLOYMENT.md` for single-proc Phase 1**
  Restructure so **single-proc/no-MPS is the Phase-1 primary topology** (architecture diagram: LB →
  N boxes, one `server.py` :8080 each; SLO-robust ~20/box at **p50 ~42–54ms, p95 ~158ms** at 20
  concurrent — cite the 2026-05-27 note in the current DEPLOYMENT.md; `maxconn 20` leastconn).
  Move K=3+MPS (the current `launch_multiproc.sh`) into a clearly-marked **fallback** section: use it
  if a *sustained multi-turn* load test shows the single asyncio intake thread saturating below
  ~16–20 (the validated single-proc number came from one-utterance-per-connection only). Keep the
  per-GPU matrix, MPS hardening, substrate (EC2+ASG+ALB recommended; SageMaker not a fit),
  autoscaling, and native-runtime roadmap sections, re-pointed at the single-proc baseline. Update
  the Artifacts list to include the new `launch_single.sh`, `nemotron-asr.service`, `gen_haproxy.py`,
  `drain.sh`, `RUNBOOK-phase1.md`, `README.md`.
  **`maxconn` caveat (everywhere it appears):** label `--maxconn 20` as the
  *one-utterance-per-connection* default; cite `--maxconn-conservative` (12) as the safer setting
  until sustained-multi-turn load testing lands.
  **`SRV_ENV` provenance note:** the optimization stack (encoder cudagraph, finalize-padded,
  sync-compress, finalize-priority) was validated in the K=3+MPS configuration, not single-proc.
  These are per-process optimizations and should carry over, but it's an untested assumption.
  If single-proc smoke shows unexpected p95, A/B-test with `NEMOTRON_SYNC_COMPRESS=0` or
  `NEMOTRON_FINALIZE_PRIORITY=0` to isolate. Cite this in the env-var reference so future operators
  understand the provenance.
  **Reproducibility risk callout:** `bootstrap.sh` pins NeMo by commit
  (`ec2-bench/bootstrap.sh:8,29-31`) but leaves `torch` unpinned (`:24-27`) and installs `uv` via a
  live `curl | sh` (`:17-19`). Document last-known-good torch version from validation; future Phase
  hardens these. This is the cost of reusing the validated bootstrap verbatim.
  **Network topology contract (Phase 1, scoped to what the reused provisioner supports):**
  `ec2-bench/ec2_up.py:65-68,96-97` provisions into PUBLIC subnets with public-IPv4 association — a
  private-subnet+NAT/bastion topology is Phase-2 hardening, not in scope here. Phase 1 contract:
  the LB uses each backend box's **private IP** (LB↔backend traffic stays inside the VPC);
  the backend SG MUST block public :8080 (allow only from the LB SG); the LB SG allows :8443 only
  from an approved client CIDR (corporate range, API gateway IP set, etc.) and :22 from MY_IP.
  App-level auth is deferred to a later phase, and that's only acceptable because the network
  allowlist is the access control. Cross-link the RUNBOOK section that walks operators through
  setting these SG rules. **Phase-2 hardening note** in DEPLOYMENT.md: move backends to private
  subnets + NAT/bastion to remove the "public-IP exists" surface even if SG blocks it.
  **LB is SPOF in Phase 1** (single haproxy host) — Phase 2 adds keepalived VIP / dual-LB / ALB front.
  Documentation: this file is **the WHY** — rationale, measurements, trade-offs. Every
  recommendation cites a source (validation file path:line OR a memory pointer). Add an explicit
  "Phase-1 scope + non-goals" section (no IaC, no live cluster, no autoscaling automation, no
  sustained-load validation, no app-level WS auth, no TLS cert automation, no LB HA, no monitoring
  beyond manual checks) so future readers know what's deliberately deferred. Add a
  "When to abandon single-proc and switch to the K=3+MPS fallback" decision rule with **externally
  observable** trip-wires (don't promise instrumentation we don't have): (a) sustained p95 above SLO
  at known concurrent-stream load below ~16, (b) repeated 1013 admission rejections from a single
  box under target load (count via `/health` admission counters), (c) any box's median TTFS climbing
  past ~150ms under steady multi-turn traffic. Add a **Day-1 observability** section: `curl
  http://box:8080/health` for admission counters, `socat … <<<"show stat"` for LB backend status,
  `journalctl -u nemotron-asr -f` for server logs, `nvidia-smi` for GPU memory — explicitly note
  what's NOT instrumented (no Prometheus, no log shipping). Top-of-file pointer: "for the HOW, see
  RUNBOOK-phase1.md."
  Key files: `deploy/DEPLOYMENT.md`

- [ ] **7. Hand-deploy runbook + deploy/README.md index (`deploy/RUNBOOK-phase1.md`, `deploy/README.md`)**
  RUNBOOK is the **HOW** — step-by-step operator procedure, deliberately verbose, copy-pasteable
  commands with concrete (anonymized) IPs, expected outputs at each step, and rollback for each.
  Sections:
  (0) **Prereqs** — AWS SSO profile + region, exported BEFORE any command in this runbook so
      `ec2_up.py` (`ec2_up.py:18-19`), the AWS CLI snippets, and the SG reconciliation cannot
      silently target the wrong account/region:
      ```
      export AWS_PROFILE=AWSAdministratorAccess-419599258555  # matches ec2_up.py default
      export AWS_REGION=us-west-2                              # matches ec2_up.py
      export AWS_DEFAULT_REGION=us-west-2                      # belt-and-braces
      export MY_IP=$(curl -s https://checkip.amazonaws.com)    # SG ingress for SSH
      ```
      (Override `AWS_PROFILE` via `NEMOTRON_AWS_PROFILE` if needed.) Workstation also needs:
      Python venv with `boto3` for `ec2_up.py` and `websockets` for the cloud smoke, `jq`,
      `rsync`, `ssh`. All commands in this runbook run from **repo root** (state files land at
      `ec2-bench/.instance_*.json` because `ec2_up.py:24-26` resolves paths relative to its own
      directory). LB host needs `haproxy`,
      `socat`, `python3`, `jq`, `aws-cli v2` (Ubuntu's `awscli` package is old — install
      AWS CLI v2 from the official tarball if SSO is required on the LB host; for Phase 1 the
      LB host only uses AWS CLI to query backend SGs during ops, so v1 works for basic
      ec2/sg operations).
  (0.5) **Sizing** — `boxes = ceil(target_streams / (maxconn × headroom))`; default `maxconn=20`,
      `headroom=0.7` (so one box failure doesn't blow the SLO budget). Worked example: 100 streams,
      20-maxconn, 0.7 headroom → `ceil(100 / 14)` = 8 boxes.
  (0.6) **Network topology** — diagram + concrete SG rules. **Important constraint:**
      `ec2_up.py:21-23,71-80` creates/reuses one fixed SG (`nemotron-bench-sg`) for every instance
      and only authorizes :22 from `MY_IP`. The Phase-1 backend↔LB isolation requires creating
      *additional* SGs and attaching them post-provision (step 1.5 below). Goal state:
      - `nemotron-asr-backend-sg`: ingress :8080 from `nemotron-asr-lb-sg` source-group (no CIDR).
      - `nemotron-asr-lb-sg`: ingress :8443 from approved client CIDR (corporate range, API gateway
        IP set, etc.).
      - `nemotron-bench-sg` (existing, from `ec2_up.py`): keep — provides :22 from `MY_IP` on all
        boxes (operator SSH/bootstrap path).
      Backend boxes use **private IPs** for LB→backend traffic (LB→backend traffic stays inside
      the VPC). This is the access control — without it, Phase 1 cannot ship.
  (1) **Provision N g6e.4xlarge L40S boxes** — all commands in this runbook run from **repo root**
      (state files land at `ec2-bench/.instance_*.json` because `ec2_up.py:24-26` resolves
      `NEMOTRON_EC2_STATE` relative to the script's own directory). For each box index N, run with
      **both** unique env vars to avoid instance reuse (`ec2_up.py:23,44-50` keys reuse on
      `tag:Name`):
      `NEMOTRON_EC2_ITYPE=g6e.4xlarge NEMOTRON_EC2_NAME=nemotron-asr-boxN
       NEMOTRON_EC2_STATE=.instance_boxN.json python ec2-bench/ec2_up.py`.
      Cost reminder up-front: g6e.4xlarge ~$2/hr per box — remember teardown in step 8.
      Then for each `ec2-bench/.instance_boxN.json`, fetch `PrivateIpAddress` via
      `aws ec2 describe-instances --instance-ids $(jq -r .instance_id ec2-bench/.instance_boxN.json)
       --query 'Reservations[].Instances[].PrivateIpAddress' --output text` and record the
      `name=ip` pair for step 4's `gen_haproxy.py --boxes-file` — the LB uses private IPs, not the
      public IPs the state file holds. **Provision the LB host** the same way with a small instance:
      `NEMOTRON_EC2_ITYPE=t3.medium NEMOTRON_EC2_NAME=nemotron-asr-lb
       NEMOTRON_EC2_STATE=.instance_lb.json python ec2-bench/ec2_up.py` (same VPC; public IP for
      client ingress).
  (1.5) **Reconcile security groups (REQUIRED — without this, the LB cannot reach backends on
      :8080 because `nemotron-bench-sg` only authorizes :22 from MY_IP per `ec2_up.py:71-80`).**
      `ec2_up.py` attaches the single fixed `nemotron-bench-sg` to every instance. The block below
      is **idempotent** (re-runnable on a partially-configured cluster — e.g. after a
      failed-box replacement in Step 7.6b). Set `CLIENT_CIDR` (the approved source range for
      ingress on the front port; corporate CIDR, API gateway IP set, etc.) and `FRONT_PORT`
      (8443 with TLS, 8080 without; must match the generator's `--tls-port`/`--front-port`):
      ```
      CLIENT_CIDR="203.0.113.0/24"   # set to your approved range
      FRONT_PORT=8443                # 8443 for TLS, 8080 plain — must match gen_haproxy.py
      VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
            --query 'Vpcs[0].VpcId' --output text)
      # describe-or-create both SGs (idempotent)
      ensure_sg(){ local name="$1" desc="$2"
        local id=$(aws ec2 describe-security-groups --filters Name=group-name,Values=$name \
              Name=vpc-id,Values=$VPC --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
        if [ -z "$id" ] || [ "$id" = None ]; then
          id=$(aws ec2 create-security-group --group-name "$name" --description "$desc" \
                --vpc-id $VPC --query GroupId --output text)
        fi
        echo "$id"; }
      LB_SG=$(ensure_sg nemotron-asr-lb-sg "Nemotron ASR LB ingress")
      BE_SG=$(ensure_sg nemotron-asr-backend-sg "Nemotron ASR backend ingress")
      # ingress rules — duplicates yield InvalidPermission.Duplicate (harmless, tolerate)
      aws ec2 authorize-security-group-ingress --group-id $LB_SG \
            --protocol tcp --port $FRONT_PORT --cidr "$CLIENT_CIDR" 2>/dev/null || true
      aws ec2 authorize-security-group-ingress --group-id $BE_SG \
            --protocol tcp --port 8080 --source-group $LB_SG 2>/dev/null || true
      ```
      Then attach the new SGs to instances (alongside `nemotron-bench-sg` so SSH still works).
      **`modify-instance-attribute --groups` REPLACES the SG list** — include existing groups
      explicitly, and de-dupe to avoid re-attaching on rerun:
      ```
      attach_sg(){ local id="$1" new_sg="$2"
        local existing=$(aws ec2 describe-instances --instance-ids $id \
              --query 'Reservations[].Instances[].SecurityGroups[].GroupId' --output text)
        # de-dupe: only add new_sg if not already attached
        local final=$(echo "$existing $new_sg" | tr ' \t' '\n' | sort -u | tr '\n' ' ')
        aws ec2 modify-instance-attribute --instance-id $id --groups $final; }
      for s in ec2-bench/.instance_box*.json; do
        attach_sg "$(jq -r .instance_id "$s")" "$BE_SG"
      done
      attach_sg "$(jq -r .instance_id ec2-bench/.instance_lb.json)" "$LB_SG"
      # verify
      aws ec2 describe-instances --instance-ids \
            $(jq -r .instance_id ec2-bench/.instance_lb.json) \
            --query 'Reservations[].Instances[].SecurityGroups'
      ```
  (2) **Bootstrap each box (idempotent)** — rsync the repo to `$HOME/nemotron` excluding bulky
      non-essential trees, local virtualenvs/git, AND secrets (`ec2_up.py` writes
      `ec2-bench/nemotron-bench-key.pem` — the cluster SSH key — into the repo; a plain rsync
      would leak it onto every box this deploy controls):
      `rsync -az --exclude='.git/' --exclude='.venv/'
      --exclude='stt-benchmark/.venv/' --exclude='stt-benchmark/' --exclude='proj-*/'
      --exclude='ec2-bench/prodsweep_*/' --exclude='ec2-bench/sweep_*/'
      --exclude='ec2-bench/prodmp_*/' --exclude='ec2-bench/lanes_*/'
      --exclude='ec2-bench/local_*/' --exclude='ec2-bench/.instance*.json'
      --exclude='ec2-bench/*.pem' --exclude='ec2-bench/*.key'
      --exclude='*.records' --exclude='*.srvlog' --exclude='__pycache__/' ./
      ubuntu@<pub-ip>:~/nemotron/`. (Trailing slashes on the patterns are intentional —
      `local_*/` excludes result directories but **not** the source file `ec2-bench/local_lb.py`
      we still need on the LB host. The `*.pem`/`*.key` excludes are non-negotiable security
      controls.) Then on the box, run `bash ~/nemotron/ec2-bench/bootstrap.sh`. Bootstrap creates `~/nemo-venv` and
      installs torch+NeMo+deps but does NOT install the local package — install it now (matching
      bootstrap's PATH-uv invocation style at `bootstrap.sh:25-31`):
      `export PATH="$HOME/.local/bin:$PATH" && uv pip install --no-deps --python
      "$HOME/nemo-venv" -e "$HOME/nemotron"` (`--no-deps` is REQUIRED — `pyproject.toml:22-32`
      declares pipecat-ai, riva-client, gdown, etc., that `bootstrap.sh:24-31` deliberately omits;
      a no-`--no-deps` install would pull the full unrelated dependency graph and may overwrite
      validated wheels with newer versions, breaking the validated runtime). After install,
      `python -m nemotron_speech.server` resolves. Expected
      smoke output is at the end of bootstrap; what to do if NeMo install fails
      (`bootstrap.sh:29-31` pinned commit may have moved on GitHub); torch is unpinned —
      reproducibility risk noted in DEPLOYMENT.md.
  (3) **Install systemd unit + env** — `chmod +x ~/nemotron/deploy/launch_single.sh`,
      `sudo cp ~/nemotron/deploy/nemotron-asr.service /etc/systemd/system/`, `sudo mkdir -p
      /etc/nemotron && sudo cp ~/nemotron/deploy/asr.env.example /etc/nemotron/asr.env`. Edit
      `asr.env`: the RECOMMENDED-EXPLICIT block at the top (values that match the launcher
      defaults but are worth setting explicitly for ops visibility) vs the OPTIONAL block (touch
      only when tuning) is laid out by step 2's `asr.env.example`. Nothing is strictly required
      to make the unit run.
      Recommended explicit: `NEMOTRON_ADMISSION_MAX_BACKLOG=12` (matches the launcher default;
      setting it explicitly makes it visible in the running config, which matters for ops
      visibility but isn't functionally required). Then `sudo systemctl daemon-reload && sudo systemctl
      enable --now nemotron-asr`, `journalctl -u nemotron-asr -f` until "ASR server listening on
      ws://" appears (expected first-boot ~minutes for model load), `curl localhost:8080/health
      | jq` → `"status":"healthy"`.
  (4) **Configure LB host** — first, get the deploy/ artifacts onto the LB host (Step 2's rsync
      only ran against backend boxes): repeat the same rsync (with the full exclusion list, esp.
      `--exclude='ec2-bench/*.pem'`) against the LB host's public IP — the LB host doesn't need
      the model/venv/bootstrap, just the `deploy/` directory and `ec2-bench/local_lb.py`. Then
      on the LB host, install prereqs (`sudo apt install -y haproxy socat python3 awscli jq`),
      then **in this exact order** (because /etc/haproxy/ requires sudo and `--check` will read
      the PEM):
      1. Install the TLS PEM: Phase 1 = operator-provided (concatenated `cert + chain + privkey`
         per haproxy's format), `sudo cp asr.pem /etc/haproxy/asr.pem && sudo chown
         haproxy:haproxy /etc/haproxy/asr.pem && sudo chmod 0600 /etc/haproxy/asr.pem`.
      2. Generate to a temp file, then install with sudo:
         `python3 ~/nemotron/deploy/gen_haproxy.py --boxes <priv-ip1>,<priv-ip2>,… --maxconn 20
          --tls-port 8443 --tls-pem /etc/haproxy/asr.pem -o /tmp/haproxy.cfg`
         then `sudo install -m 0644 -o root -g root /tmp/haproxy.cfg /etc/haproxy/haproxy.cfg`.
      3. Validate as root so the haproxy user can read the PEM:
         `sudo haproxy -c -f /etc/haproxy/haproxy.cfg`.
      4. `sudo systemctl stop haproxy 2>/dev/null || true` (in case the package's default config
         is running and would fail to reload with the new config), then `sudo systemctl enable
         --now haproxy`.
      5. Verify with `echo "show stat" | sudo socat /run/haproxy/admin.sock stdio | grep
         asr_pool` — every backend should show `UP` within ~4s (rise*inter).
      **Cert rotation:** swap the PEM at `/etc/haproxy/asr.pem` (same perms), `sudo systemctl
      reload haproxy` (zero-drop reload).
  (5) **End-to-end smoke through the LB** — run this from the **operator's workstation** (in a
      venv that has `websockets` installed; the full repo includes the smoke client at
      `proj-2026-05-19-eou-endpointing/`, which the deploy rsync exclusions drop from
      backend/LB hosts). Pick the URL based on TLS setup:
      - **With TLS** (`--tls-pem` set in step 4): the cert is issued for a DNS name, not the EC2
        public IP. Point a DNS record (corporate DNS, Route53, /etc/hosts for local testing) at
        the LB's public IP from `jq -r .ip ec2-bench/.instance_lb.json`, then
        `LB_URL=wss://lb.example.com:8443`. A raw `wss://<ip>:8443` will fail TLS hostname
        verification.
      - **Without TLS** (no `--tls-pem`, frontend on `--front-port 8080`):
        `LB_IP=$(jq -r .ip ec2-bench/.instance_lb.json); LB_URL=ws://$LB_IP:8080`. Acceptable for
        Phase-1 internal smoke if the LB SG ingress allows the operator's CIDR on :8080.
      Use `proj-2026-05-19-eou-endpointing/run_full1000_conc12.py` (verified flags: `--url`,
      `--concurrency`, `--limit`; default URL is `ws://127.0.0.1:8080` so `--url` must be passed
      explicitly). The client writes benchmark artifacts to `stt-benchmark/.../results.db` and a
      JSON name derived from `--model-tag`; pass a unique tag per run to avoid clobbering
      historical bench results: `python proj-2026-05-19-eou-endpointing/run_full1000_conc12.py
      --url "$LB_URL" --concurrency 5 --limit 10 --model-tag
      phase1_smoke_$(date +%Y%m%d_%H%M%S)`.
      Expected TTFS p50 ≤ ~60ms; if 1013 closes appear, the admission cap fired — lower
      `--concurrency` and/or raise `NEMOTRON_ADMISSION_MAX_BACKLOG`. Note: this is the
      closest-existing client, not a purpose-built production smoke (Phase-2 cleanup). Document a
      one-screen interpretation guide for the output. Also: this is the right moment to
      verify zero-drop reload — open one stream, `sudo systemctl reload haproxy`, confirm the
      stream survives.
  (6) **Rolling redeploy** — for each box:
      `drain.sh drain box_<ip-dashes> && drain.sh wait-empty box_<ip-dashes> 300 &&
       ssh box "sudo systemctl restart nemotron-asr" &&
       (until curl -sf box:8080/health | jq -e '.status=="healthy"' >/dev/null; do sleep 2; done) &&
       drain.sh ready box_<ip-dashes>`.
      Recovery: if `wait-empty` times out (active streams still open at 5min — they can legally
      last up to 1h per `timeout server`), operator decides: extend timeout, or force-kill the
      backend (active clients will reconnect through the LB to another box).
  (6b) **Replacing a failed box** (not a rolling redeploy — a single box died/was terminated):
      provision a replacement (step 1 with the same `NEMOTRON_EC2_NAME=nemotron-asr-boxN`; the
      reuse logic will see no running instance and create fresh). The operator's shell from the
      original deploy is long gone, so **rehydrate the SG IDs from name** before attaching:
      `BE_SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=nemotron-asr-backend-sg
       --query 'SecurityGroups[0].GroupId' --output text)`. Then re-run the `attach_sg`
      function from step 1.5 to attach `$BE_SG` to the new instance (the 1.5 block is idempotent,
      so re-running the whole block is also fine). Fetch its private IP, bootstrap and install
      systemd unit (steps 2–3), confirm `/health` green locally on the new box, **then** regen
      `haproxy.cfg` with the new IP and `systemctl reload haproxy`. Order is operational
      preference, not safety — HAProxy's `check inter 2s fall 3 rise 2` would mark a
      not-yet-ready backend DOWN within ~6s, so regen-first is safe; regen-after-healthy is just
      cleaner (no wasted reload, no brief LB DOWN-flap log noise).
  (7) **Switching to the K=3+MPS fallback** (if a trip-wire from DEPLOYMENT.md fires under
      sustained multi-turn load). **Known compatibility gap:** `launch_multiproc.sh:69-78` runs
      `python server.py` from `$NEMOTRON_APP_DIR` (the legacy flat-copy layout). Phase 1 installs
      the package and ships `src/nemotron_speech/server.py`, NOT `$HOME/nemotron/server.py`.
      Two operator options to bridge:
      (a) **One-line layout patch** — symlink the entrypoints into the app dir flat:
      `ln -s $HOME/nemotron/src/nemotron_speech/server.py $HOME/nemotron/server.py &&
       ln -s $HOME/nemotron/src/nemotron_speech/batch_primitives.py $HOME/nemotron/ &&
       ln -s $HOME/nemotron/src/nemotron_speech/cudagraph_encoder.py $HOME/nemotron/`. Then
      `launch_multiproc.sh` finds `server.py` at the expected path. Verify it imports cleanly
      against the editable-installed venv before flipping the unit.
      (b) **Edit the launcher** — replace the `python server.py` invocation with
      `python -m nemotron_speech.server` (matches `launch_single.sh`).
      Then: replace `launch_single.sh` with `launch_multiproc.sh` in the systemd unit's
      `ExecStart`, `systemctl daemon-reload && systemctl restart nemotron-asr`. Regenerate
      haproxy config with 3 ports per box (`gen_haproxy.py` will need a `--ports-per-box 3` flag
      — Phase-1 known-gap that requires either extending the generator or hand-editing the cfg
      to add 3 server lines per box on :8080/:8081/:8082).
      **systemd interaction note:** `launch_multiproc.sh` has its own in-script supervisor
      (`:87-96`); under systemd, the unit's PID is the supervisor's, so `Restart=on-failure`
      restarts the supervisor (which restarts MPS + all K procs) — accept the wider blast radius,
      which is the documented MPS trade-off.
  (8) **Teardown** — drain all boxes, `systemctl stop haproxy`, terminate each EC2 (state files
      live in `ec2-bench/`): `aws ec2 terminate-instances --instance-ids
      $(jq -r .instance_id ec2-bench/.instance_boxN.json)` (per box + the LB host's
      `.instance_lb.json`). Cost reminder: g6e.4xlarge ~$2/hr — running 8 boxes is meaningful spend.
  (9) **Troubleshooting matrix** — symptom → cause → fix, enumerated:
      - "Model never loads / `/health` stuck on `loading`" → check `journalctl` for NeMo import
        error or HF download stall; verify `nvidia-smi` shows GPU; confirm HF_HOME has space.
      - "`/health` flapping UP/DOWN at LB" → server crashing/restarting; `journalctl` for trace.
      - "LB shows backend DOWN, server says healthy" → SG rule wrong (LB SG not in backend ingress).
      - "`drain.sh wait-empty` always times out" → check `timeout server 1h` is acceptable; consider
        force-kill policy.
      - "`systemctl restart` loop" → `RestartSec=5` floods journald; check exit reason in journal.
      - "GPU not visible to systemd-launched process" → DLAMI default should work; check
        `nvidia-smi` under the user, then under `sudo -u ubuntu`; verify no `User=` mismatch.
      - "Port 8080 conflict on box" → another service bound; `ss -tnlp | grep 8080`.
      - "Server OOM kill" → check `dmesg` for OOM-killer; reduce `NEMOTRON_BATCH_MAX_SIZE` or
        re-verify `NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE_PADDED=1` (the L40S K=3 OOM fix per
        `launch_multiproc.sh:38-44`).
      - "`bootstrap.sh` NeMo install fails" → pinned commit drifted on GitHub or network blip; retry.
      - "TLS cert expired" → swap PEM, `systemctl reload haproxy` (zero-drop).
      - "1013 close-storm from one box" → that box's admission cap firing; check
        `/health` admission counters; consider lowering `--maxconn` for that box or raising
        `NEMOTRON_ADMISSION_MAX_BACKLOG`.
      - "Haproxy reload drops connections" → ensure using `systemctl reload haproxy` (which
        triggers `-sf`), not `restart`.
      - "All backends DOWN at LB" → clients get connection close. Check `show stat` for cause
        (every server.py crashed, network partition, etc.); operator runbook to recover each box.
      - "Pinned NeMo commit no longer fetchable from GitHub" → upstream force-pushed or removed the
        branch; document the next-known-good commit in DEPLOYMENT.md and update
        `bootstrap.sh:8` accordingly (Phase-2: keep a tarball mirror in S3).
      - "Timestamps disagree across boxes" → NTP drift; `timedatectl status` to confirm sync;
        DLAMI defaults to chrony so this should be self-healing within minutes.
      - "journald disk full" → set `SystemMaxUse=500M` in `/etc/systemd/journald.conf` (or run
        `journalctl --vacuum-size=500M`); restart journald.
  (10) **Open risks accepted by Phase 1** — listed explicitly with one-line rationale each:
       no sustained-multi-turn load validation, no IaC/ASG/ALB, single LB host (LB SPOF),
       no TLS automation/renewal, no app-level WS auth (network-allowlist substitutes), no
       monitoring/alerting beyond manual checks, no log shipping (journald local only), no GPU
       memory monitoring, torch version unpinned in bootstrap.
  Documentation: this is the procedure-of-record; every command is concretely runnable; every
  expected output is shown; every failure mode has a fix. Cross-link every artifact in `deploy/`
  inline by relative path.

  **deploy/README.md** — short index (≤2 screens): one-line description of every file in `deploy/`
  with a relative-path link; "Start here: RUNBOOK-phase1.md"; "Why these decisions: DEPLOYMENT.md";
  "Phase-1 scope + non-goals" pointer; "Sustained-load fallback: K=3+MPS via launch_multiproc.sh"
  pointer. A new engineer reading only the README must know which artifact to open next.
  Key files: `deploy/RUNBOOK-phase1.md`, `deploy/README.md`

## Progress
| # | Step | Status | Commit | Notes |
|---|------|--------|--------|-------|
| 1 | Single-proc launcher | done | 3d32d93 | 176 lines, bash -n OK |
| 2 | systemd unit template | done | b0b97cb | unit 64L, env 146L; %h not $HOME |
| 3 | HAProxy config generator | done | (pending) | 575L; 9 directive assertions + 6 validation tests pass |
| 4 | Drain automation | pending | — | |
| 5 | Local smoke test | pending | — | |
| 6 | Rewrite DEPLOYMENT.md for single-proc | pending | — | |
| 7 | Hand-deploy runbook + deploy/README.md | pending | — | |
