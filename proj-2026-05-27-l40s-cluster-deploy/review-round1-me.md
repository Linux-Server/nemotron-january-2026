# Round 1 self-review (Opus)

I'm reviewing my own PLAN.md adversarially. Bias to disclose: I wrote the plan and I want to ship.

## Critical (must-fix before implementation)

- **Double-supervisor bug.** Step 1 puts a `while true; sleep 10; kill -0; restart` supervisor inside
  `launch_single.sh`, and Step 2 sets `Restart=always` on the systemd unit. That's two restart loops
  fighting each other: if `server.py` crashes, the in-script supervisor restarts it, systemd never
  sees the death, and operator-initiated `systemctl stop` races against the in-script restart.
  Pick one: prefer systemd as the supervisor (drop the in-script loop, use
  `Restart=on-failure`/`RestartSec=5`) — the loop existed in `launch_multiproc.sh` because it had to
  supervise K children that systemd couldn't see individually; with K=1 that rationale is gone.

- **WebSocket auth/network exposure is unspecified.** `src/nemotron_speech/server.py` accepts any
  WS connection on `/` with no auth (`server.py:10076`). The plan has zero mention of authz or
  network controls. We must NOT ship a config that puts these boxes on a public subnet with
  permissive ingress. Phase 1 must DOCUMENT (even if not implement): backend boxes live in a private
  subnet, LB-only ingress on :8080 from the LB SG, LB front (:8443) accepts only the expected
  source range — plus a "no app-level auth in Phase 1" disclosure as an accepted risk.

- **TLS cert procurement is hand-waved.** Step 3 introduces `--tls-pem` but never says where the cert
  comes from, who owns rotation, what the PEM format must be, where it lives on disk, or what permissions
  it needs. "Operator provides" is fine for Phase 1, but it must be explicitly documented in RUNBOOK
  including rotation procedure (currently manual: stop haproxy, swap PEM, start — there's no acme/certbot
  integration).

- **HAProxy itself has no supervision.** "On the LB host, run gen_haproxy + start" — by what? Phase 1
  needs a systemd unit for haproxy too (or document that the OS package's unit is used) and a
  graceful-reload procedure (`haproxy -sf $(pidof haproxy)` or `systemctl reload haproxy`) so config
  regen doesn't drop active connections. Otherwise the rolling-redeploy story is incomplete.

- **`option httpchk` in `mode tcp` works, but needs the modern directives to actually require 2xx.**
  Plan says `option httpchk GET /health` (matches `haproxy.cfg.example:23`). In HAProxy 2.x the
  modern form is `option httpchk` + `http-check send meth GET uri /health` + `http-check expect status 200`.
  Without `expect status`, ANY HTTP response (including 503/500 from server.py) counts as healthy.
  This is a real silent-failure path.

## Material (should-fix; folding into the plan adds real value)

- **Two log paths — journald AND `server.log`.** Step 1 writes to `server.log`; Step 2's systemd unit
  captures stdout/stderr to journald. With Type=simple under systemd, the launcher's `>` redirect
  to `server.log` competes with journald capture. Pick one: drop the file redirect, let journald
  own the logs (`journalctl -u nemotron-asr -f` is the canonical view).

- **`TimeoutStopSec` is a number, not a vibe.** Step 2 says "long enough to drain in-flight streams"
  but `timeout server 1h` in HAProxy means streams can legally last 1h. Phase 1 should pick a concrete
  drain budget (e.g. 120s SIGTERM grace → SIGKILL) and document that planned ops use `drain.sh` ahead
  of `systemctl restart`, NOT systemd's stop-timeout, to bound the wait.

- **Source-of-truth for `src/` deploys is unspecified.** Runbook step 2 says "rsync `src/` to
  `$HOME/nemotron`" but never says from where. Operator's laptop? An S3 bucket? A git checkout on the
  box? Phase 1 should pick the simplest defensible answer (probably `git clone` on the box to a
  pinned commit, since `bootstrap.sh` already pins NeMo by commit) and document it.

- **Capacity / sizing formula missing.** RUNBOOK should include `boxes = ceil(target_streams /
  (per_box_maxconn × headroom))` with a default headroom (e.g. 70%) so operators don't have to
  rederive it. Without this, "how do I run 200 streams?" has no documented answer.

- **MPS-fallback switching procedure missing from RUNBOOK.** Step 6 promises decision trip-wires for
  abandoning single-proc; Step 7 has no procedure for what to DO when the trip fires. Add a
  RUNBOOK section: "Switching to K=3+MPS fallback" — swap launch_single.sh for launch_multiproc.sh,
  update systemd ExecStart, regen haproxy config with 3 ports per box (8080,8081,8082), reload LB.

- **`gen_haproxy.py` backend naming is positional and brittle.** "box1, box2, …" rebinds to whichever
  IPs the operator passes; reordering the input changes names that `drain.sh` references. Either
  derive names from the IP (`backend_10_0_1_10` style) or accept an explicit `--names` list. Document
  the choice.

- **Generator output validity is checked, but health-check behavior is not.** The smoke test asserts
  `haproxy -c` (syntax) and leastconn distribution (via `local_lb.py`). Nothing actually verifies that
  with the generated config, haproxy correctly marks a backend UP when it responds 200 to `/health`
  and DOWN otherwise. To close: in step 5, optionally bring up real `haproxy` against the generated
  config + a stub backend serving 200 on `/health`, assert `show stat` reports UP, then flip the stub
  to 500 and assert DOWN. Skips cleanly if haproxy isn't installed. This is THE check that catches the
  silent-failure `option httpchk` bug above.

- **socat prereq + LB permissions for `drain.sh` are implicit.** Plan assumes `socat` is on the LB
  host and the operator's user can write to `/run/haproxy/admin.sock`. Neither is guaranteed by the
  package install. RUNBOOK must list: install socat, set `stats socket … user haproxy group ops mode 660`
  in global, add operator to `ops` group.

- **HuggingFace checkpoint isn't revision-pinned.** `bootstrap.sh` pre-downloads
  `nvidia/nemotron-speech-streaming-en-0.6b` without a `--revision`. Future model updates would
  silently change the deployed checkpoint. Pin to a specific revision SHA in bootstrap and document
  it as an upgrade-knob in DEPLOYMENT.md.

- **`--right-context 1` is hardcoded in the launcher.** Operationally fine (rc0 is broken per
  `rc0-unsupported-nemo-relshift`), but make it an env var (`NEMOTRON_RIGHT_CONTEXT`, default 1) so
  the launcher doesn't have to be edited to test rc3/rc6 in a future phase.

- **HAProxy logging in `mode tcp` is empty without `option tcplog`.** Add it explicitly so connection
  logs have timing and backend info. Without it, post-incident analysis has nothing.

- **Network topology must be documented even though it's not implemented.** RUNBOOK should have a
  topology section (one diagram, ~5 lines) showing: public-ingress LB SG, private backend SG with LB
  SG as the only ingress on :8080, optional bastion for ops access. This is the missing security
  contract that auth-less WS depends on.

- **The smoke test's "CI-safe" framing is aspirational.** There's no CI in this repo (worth verifying).
  Either call it "laptop-safe" or stand up a minimal CI hook. Don't overclaim.

- **Step 6's "concrete trip-wires" promise may not be measurable.** `/health` exposes admission
  attempted/admitted/rejected/signal (`server.py:5017-5025`), not intake-thread saturation directly.
  Narrow the promise to "observable from outside" signals: sustained p95 above SLO at known
  concurrent-stream count, repeated 1013 admission rejections from a single box under target load,
  etc. Don't promise instrumentation we don't have.

- **Troubleshooting matrix should be enumerated in the plan, not promised.** Step 7 says "symptom →
  cause → fix" but doesn't list the symptoms. Enumerate them in the plan so the reviewer can spot
  missing ones: model never loads, health flapping, LB shows backend DOWN, drain times out, restart
  loop, GPU not visible, port conflict, OOM, NeMo install failed, certificate expired.

## Minor (defer to implementation)

- Per-restart log rotation (moot if we drop `server.log` and use journald).
- Cost reminder in RUNBOOK ("g6e.4xlarge is $X/hr — remember teardown").
- Time-sync/NTP note (DLAMI defaults are usually fine).
- `chmod +x` reminder on scripts.
- Mention ephemeral NVMe on g6e instances (200GB gp3 root is fine, no mount needed for Phase 1).
- A short sentence in DEPLOYMENT.md noting LB itself is SPOF in Phase 1 (LB-HA is a Phase-2 concern).

## Non-findings (checked, looks correct)

- The `SRV_ENV` lift from `launch_multiproc.sh:49-67` is the right source; nothing in those env vars
  is multi-proc-specific. The `env -u LD_LIBRARY_PATH` wrapper is correctly preserved (needed for
  torch's bundled cuDNN per `launch_multiproc.sh:77`).
- WS close code 1013 is correct (verified at `server.py:5062`).
- `/health` endpoint is correct (`server.py:10075`).
- `mode tcp` is the right choice for WS over HAProxy (not `mode http`).
- Single-proc topology decision itself is sound given the 2026-05-27 lead; the open risk
  (sustained multi-turn) is explicitly accepted and the fallback path is preserved.
- Reuse of `local_lb.py` unmodified for the local smoke is correct; it's a stable byte-pipe.
- `ec2_up.py` env-var override pattern (`NEMOTRON_EC2_STATE`, `NEMOTRON_EC2_NAME`) is correctly
  identified for multi-box provisioning.

## Verdict

MATERIAL_FINDINGS — five critical issues (double-supervisor, missing auth/network docs, hand-waved TLS,
unsupervised HAProxy, silent-failure health check) plus a dozen material gaps. None are fatal; all
are clearly fixable in one editing pass. Recommend folding and proceeding to round 2.
