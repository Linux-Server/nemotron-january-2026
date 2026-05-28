# Round 3 self-review (Opus)

The round 2 fold landed cleanly. Looking for what's left — internal consistency, layered-edit
artifacts, and gaps the prior rounds didn't notice.

## Resolved from round 2

- ✅ `show stat;csv` corrected to plain `show stat` with leading-`#` strip. Verified: HAProxy
  Runtime API's `show stat` returns CSV with `# pxname,svname,…` header line. Python csv module
  handles the strip cleanly.
- ✅ `uv pip install -e .` corrected to PATH-uv `uv pip install --python "$HOME/nemo-venv" -e
  "$HOME/nemotron"` — matches `bootstrap.sh:25-31` pattern.
- ✅ `http-check expect string` → `expect rstring` with JSON-anchored regex. The pattern
  `"status"[ ]*:[ ]*"healthy"` matches aiohttp's `json.dumps()` default (`{"status": "healthy", …}`).
- ✅ Rules section now has explicit "systemd is the sole supervisor" — won't reintroduce the
  double-supervisor bug during implementation.
- ✅ Network topology contract softened to match `ec2_up.py:65-68,96-97` reality (public subnets +
  SG-blocking-:8080), with private-subnet/NAT-bastion noted as Phase-2 hardening.
- ✅ `--local-test` mode added to Step 3 (gen_haproxy.py) wired into Step 5(c) smoke.
- ✅ asr.env REQUIRED/OPTIONAL split with `NEMOTRON_ADMISSION_MAX_BACKLOG` in REQUIRED.
- ✅ EC2 state-file paths now consistently `ec2-bench/.instance_*.json`.
- ✅ LB-host provisioning example (t3.medium) + unique name.
- ✅ rsync exclusions for `proj-*`, `prodsweep_*`, `sweep_*`, `*.records`, `*.srvlog`.
- ✅ Cloud smoke uses `--model-tag phase1_smoke_$(date +…)` to avoid clobbering historical bench
  artifacts.
- ✅ LB-host prereqs spell out socat + python3 + aws-cli.
- ✅ SRV_ENV provenance note (K=3+MPS-validated, carry-over assumption documented).

## Critical (must-fix before implementation)

(None.)

## Material (should-fix; folding into the plan adds real value)

- **Fallback procedure (Step 7.7) underspecified for the systemd interaction.** Switching to
  `launch_multiproc.sh` means systemd sees only the supervisor PID — when the supervisor crashes,
  systemd restarts the whole MPS+K-process tree. The plan correctly punts on `--ports-per-box 3` in
  the generator but doesn't note that swapping the launcher also restores the in-script supervisor
  pattern (acceptable, but the systemd unit's `Restart=on-failure` semantics shift meaning). Add a
  one-line note: "fallback switching requires the in-script supervisor; systemd restart applies to
  the supervisor, not individual procs — accept the wider blast radius (which is the documented MPS
  trade-off)."

- **Box-replacement vs rolling-redeploy aren't distinguished.** Step 7.6 covers rolling redeploy
  (same boxes, new code). It does not cover the recovery scenario of "box died, replace it" —
  which is provisioning + bootstrap + private-IP fetch + gen_haproxy regen + reload. Currently
  conflated. Add a Step 7.6b "Replacing a failed box" subsection — same building blocks but the
  ordering matters (regen MUST happen after the new box is healthy on `/health`, not before).

- **HAProxy log volume.** With `option tcplog`, every connection (including the `inter 2s` health
  probes) generates a log line. At 20 streams × N boxes plus 2 health probes per backend per
  second, journald gets noisy fast. Add `option dontlog-normal` to the generator defaults (or
  per-backend `no log` for health-probe noise reduction). Document the choice in the generator.

- **Pinned NeMo commit `056d937` could disappear from GitHub.** `bootstrap.sh:8,29-31` pins it,
  but if NVIDIA force-pushes or deletes the branch, fresh bootstraps fail. Document a recovery: keep
  a tarball mirror of the pinned commit in S3 (Phase-2), or document the "newest validated commit"
  in DEPLOYMENT.md and update the pin if upstream churns.

- **Time/clock assumptions.** Telemetry timestamps depend on box clocks being in sync. DLAMI NTP
  default works, but if drift causes confusing logs, operator needs to know what to check. One-line
  troubleshooting entry: "timestamps disagree across boxes → check `timedatectl status` for NTP
  sync."

## Minor (defer to implementation)

- **HAProxy "all backends DOWN" behavior** — clients get connection close. Worth one-line in the
  troubleshooting matrix: "all backends DOWN → clients get connection close; check `show stat` for
  cause (server.py crash on every box, network partition, etc.)."

- **journald disk usage** — default is unbounded by free space; if logs balloon, disk fills.
  Suggest `SystemMaxUse=500M` in `/etc/systemd/journald.conf`. One-line in step 7.3.

- **`--boxes` vs `--boxes-file` choice** — runbook uses `--boxes <comma list>`; for >10 boxes,
  `--boxes-file` is cleaner. One-line guidance in the generator's `--help`.

- **HF token note** — public model, no token required, but if HF rate-limits the bootstrap retries,
  `HF_TOKEN` set in `asr.env` would help. One-line OPTIONAL var.

- **Smoke test for `gen_haproxy.py` stable-naming invariant** — assert that reordering
  `--boxes 10.0.1.10,10.0.1.11` and `--boxes 10.0.1.11,10.0.1.10` produces the same backend names
  (both should produce `box_10-0-1-10` and `box_10-0-1-11`). Cheap check.

- **`systemctl enable --now haproxy` may fail if pre-existing config is invalid.** Runbook should
  precede with `sudo systemctl stop haproxy 2>/dev/null || true` to ensure the new config is what
  starts.

## Non-findings (things I checked that look correct)

- The full revised plan is internally consistent: `maxconn 20` consistently used; `--maxconn-conservative`
  cross-referenced in DEPLOYMENT.md and gen_haproxy.py docstring; every artifact reference uses the
  correct relative path; the deploy/README.md indexes all files mentioned in other steps.
- `http-check expect rstring "\"status\"[ ]*:[ ]*\"healthy\""` is valid HAProxy syntax (POSIX regex
  via `rstring`). The escape doubling is correct for the haproxy config (rstring expects a quoted
  string; the inner quotes are escaped). Aiohttp's `web.json_response()` uses `json.dumps()`
  default separators `(', ', ': ')`, so the body contains `"status": "healthy"` and the regex
  matches.
- `show stat` over the runtime socket returns CSV with `# pxname,svname,…` header. Python csv
  module + stripping the leading `# ` from the header line is standard. Parsing by header name is
  robust across haproxy versions.
- The `--local-test` mode design is sufficient for non-root local haproxy: temp socket under
  `$TMPDIR`, no system-user requirements, `log stdout` for visibility. The smoke can run haproxy
  in foreground and assert via the temp socket.
- rsync exclusions cover the bulky trees; nothing essential (deploy/, ec2-bench/bootstrap.sh,
  ec2-bench/ec2_up.py, ec2-bench/local_lb.py, src/, pyproject.toml) is excluded. Specifically:
  bootstrap.sh and ec2_up.py are at `ec2-bench/` top level, not in the excluded `prodsweep_*`/
  `sweep_*`/`prodmp_*` subdirs.
- The asr.env REQUIRED block correctly contains only `NEMOTRON_ADMISSION_MAX_BACKLOG`; everything
  else has a validated default in `launch_multiproc.sh:49-67` that the launcher inherits, so they
  belong in OPTIONAL.
- The plan correctly distinguishes Phase-1 scope from Phase-2 hardening (LB HA, private subnets,
  IaC, monitoring, log shipping, app auth, sustained-load validation) — these are listed as
  accepted risks with concrete pointers.
- The cloud smoke client choice (`run_full1000_conc12.py`) is fit-for-purpose for Phase 1 with the
  noted caveats about unique `--model-tag` and the fact that a purpose-built smoke is Phase-2.

## Verdict

MATERIAL_FINDINGS — no criticals; 5 material refinements (all 1-3 lines each, mostly doc additions
to RUNBOOK and the generator); 6 minor nice-to-haves. The plan is structurally sound and
implementation-ready; another round of folding would tighten the documentation further but is not
required for safety. If Codex's round 3 also returns no criticals, recommend going to /implement
after round 4 (one more iteration), with anything still remaining handled during implementation.
