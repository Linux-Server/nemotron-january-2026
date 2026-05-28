# Round 4 self-review (Opus)

Looking at the round-3 fold with fresh eyes. I found one item I added in round 3 that's actually
wrong, plus a few small consistency fixes.

## Resolved from round 3

- ✅ SG reconciliation step 7.1.5: `modify-instance-attribute --groups` REPLACES the list — but
  the snippet correctly fetches `$EXISTING` first and includes both. Bash word-splitting on
  `--output text` tab separators correctly expands `$EXISTING` to space-separated SG IDs for the
  `--groups` flag.
- ✅ rsync exclusions: directory-specific trailing slashes preserve `ec2-bench/local_lb.py`;
  `.git/`, `.venv/`, `stt-benchmark/` are excluded so the deploy isn't multi-GB.
- ✅ LB-host code-copy: Step 7.4 explicitly rsync's to the LB host.
- ✅ Repo-root invocations: every `python ec2-bench/ec2_up.py` is from repo root; state files
  consistently `ec2-bench/.instance_*.json`.
- ✅ Stale `expect string` references updated to `rstring` everywhere.
- ✅ Reference implementations no longer claims the supervisor loop is reused.
- ✅ SRV_ENV summary in Current state now uses NEMOTRON_* prefixes.
- ✅ Generator `--local-test` resolves a concrete temp socket path at generation time, drops
  ulimit-n and user/group, uses log stdout.
- ✅ `--no-deps` is now REQUIRED in the pip install command — prevents pipecat/riva/etc. from
  pulling.
- ✅ Box-replacement vs rolling-redeploy distinguished (Step 7.6b).
- ✅ Fallback procedure notes the systemd-supervisor interaction with launch_multiproc.sh.
- ✅ Troubleshooting matrix extended.

## Critical (must-fix before implementation)

- **`option dontlog-normal` is the wrong fix and would silence production traffic logs.** I added
  this in round 3 (Step 3 generator spec) to address health-probe log noise. But HAProxy doesn't
  log health-check probes by default at all — `log-health-checks` is OFF by default, so successful
  checks are already silent. `option dontlog-normal` instead suppresses logging of **all
  successful TCP connections** including real WS streams — we'd lose audit visibility into who
  connected, when, and for how long. Drop `option dontlog-normal` from the generator defaults;
  the health-probe noise concern was unfounded.

## Material (should-fix; folding into the plan adds real value)

- **Step 7.6b box-replacement order claim is overstated.** I said "regenning first would route
  streams to a not-yet-ready backend." But HAProxy's `check inter 2s fall 3 rise 2` directives
  mean a new backend stays in maintenance/down state until ≥2 consecutive successful health
  probes (~4s after the port opens). So order doesn't matter for *safety* — it matters for
  *operational cleanliness* (wasted reload, brief LB log noise). Soften the claim to: "operational
  preference — `check` handles the race safely, but regen-after-healthy is cleaner."

- **`--local-test` + `--stats-socket` interaction unclear.** Step 3 says the generator resolves
  `$TMPDIR` at generation time. But the smoke test (Step 5(c)) needs to *know* the resolved path
  to call `show stat`. Cleaner: smoke passes `--stats-socket "$tmpdir/haproxy.sock"` explicitly to
  the generator, so it controls the path. Document that `--local-test` works with an explicit
  `--stats-socket` override (and falls back to a generator-chosen temp path if not given,
  printing the chosen path to stderr).

- **Cloud smoke run location is ambiguous.** Step 7.5 doesn't specify where to run
  `run_full1000_conc12.py`. It's at `proj-2026-05-19-eou-endpointing/`, which the rsync
  exclusions drop on backend AND LB boxes. So the smoke runs from the operator's workstation
  (full repo). State this explicitly. Also: the operator needs the LB host's public IP for
  `--url wss://<lb-pub-ip>:8443` — pull from `ec2-bench/.instance_lb.json`.

- **The asr.env REQUIRED block needs honest framing.** Step 2 docs say REQUIRED contains
  `NEMOTRON_ADMISSION_MAX_BACKLOG`. The runbook (Step 7.3) says "Minimum required:
  NEMOTRON_ADMISSION_MAX_BACKLOG (12 is plan-default; set explicitly so it's visible in the
  running config)." This is misleading — the launcher default IS 12 (per Step 1), so setting
  asr.env's value to 12 is explicit-visibility-only, not functionally required. Reframe as
  "RECOMMENDED block — values that match the launcher default but should be explicit in the
  per-box env file for operator visibility." If nothing is truly REQUIRED, don't call any block
  REQUIRED.

## Minor (defer to implementation)

- **`modify-instance-attribute --groups` semantics not documented in the snippet.** The snippet
  works because we include `$EXISTING`, but an operator reading the AWS CLI block should be told
  "this REPLACES the SG list, so we include the existing SGs explicitly." One inline comment.

- **`option tcplog` log volume at scale.** Step 3 keeps `option tcplog` so we get connection
  durations. At N boxes × 20 streams + reconnects, this is ~hundreds of log lines/hour. Fine
  for Phase 1 but worth a note in the troubleshooting matrix's "journald disk full" entry.

- **Step 7.4 instructs to `apt install -y` on the LB host but Step 7.0 prereqs imply they should
  already be installed.** Slight redundancy — clearer to keep the install in 7.4 (per-host setup)
  and rephrase 7.0 prereqs as "the runbook installs these on the LB host in Step 7.4."

## Non-findings (things I checked that look correct)

- `--no-deps` correctness: pyproject.toml's runtime deps that bootstrap.sh:24-31 omits are
  `gdown`, `pipecat-ai`, `nvidia-riva-client`. None are imported by `nemotron_speech.server`
  (pipecat = pipeline framework, riva = TTS, gdown = google drive). `--no-deps` is correct.
- The `exec env -u LD_LIBRARY_PATH "${SRV_ENV[@]}" "$VENV/bin/python" -m …` invocation preserves
  $HOME and $USER from systemd's `User=ubuntu`; only LD_LIBRARY_PATH is stripped. Correct.
- `gen_haproxy.py --boxes-file` accepts `name=ip` (from step 3 spec) — runbook step 7.1 records
  `name=ip` pairs for this. Consistent.
- Step 7.1.5 SG reconciliation runs before Step 7.2 bootstrap; the original `nemotron-bench-sg`
  still provides :22 from MY_IP, so SSH/bootstrap path works after attaching the new SGs.
- The systemd `Environment=%h/...` paths resolve correctly for `User=ubuntu`.
- The plan is internally consistent on maxconn 20 (production default) vs `--maxconn-conservative`
  12 (safer alternative); all references match.
- After all edits, no section references a file or flag that doesn't exist in the plan elsewhere.

## Verdict

CRITICAL_FINDINGS (1 self-inflicted, easy to fix) — the `option dontlog-normal` I added in round
3 would silence production traffic logs and must be removed. The other findings are
1-line refinements. After this fold, round 5 likely returns clean.
