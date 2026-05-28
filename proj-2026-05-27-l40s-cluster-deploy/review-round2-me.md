# Round 2 self-review (Opus)

Verifying claims the revision now makes, evidence-based. Round 1 findings I checked and confirmed
resolved in fold are listed under "Resolved." New issues introduced or surfaced by the revision are
in Critical/Material/Minor.

## Resolved from round 1 (verified in fold)

- ✅ Double-supervisor — Step 1 now drops the in-script loop, Step 2 uses `Restart=on-failure`.
  Single supervisor (systemd). Correct.
- ✅ systemd `$HOME` non-expansion — Step 2 now uses `%h` specifier. Correct.
- ✅ EC2 instance reuse — Step 7.1 now mandates unique `NEMOTRON_EC2_NAME=nemotron-asr-boxN` per box
  AND unique state file. Matches `ec2_up.py:23,44-50`. Correct.
- ✅ App-dir layout — Step 1 now uses `python -m nemotron_speech.server` after editable install;
  Step 7.2 documents the install. Verified the package exists (`pyproject.toml:64-65` packages
  `src/nemotron_speech`) and `server.py:10127` has the `__main__` guard, so `-m` invocation works.
- ✅ Private vs public IPs — Step 7.1 now fetches `PrivateIpAddress` via AWS CLI snippet and uses
  those for the LB backend list. Public IP is only for SSH/bootstrap.
- ✅ Health-check semantics — verified `server.py:10035-10043` always returns 200 with
  `{"status":"healthy"|"loading", "model_loaded": …}`. The `http-check expect string "healthy"`
  directive distinguishes the two (since "healthy" is not a substring of "loading"). Plan correct.
- ✅ local_lb≠HAProxy queue — Step 5(a) now explicitly notes the divergence and asserts only
  local_lb's documented behavior.
- ✅ drain.sh CSV-by-name parsing — Step 4 now mandates header-name parsing via Python heredoc, with
  fixture tests. Correct.
- ✅ TLS PEM — Step 7.4 now spells out: operator-provided concatenated PEM,
  `/etc/haproxy/asr.pem`, `chown haproxy:haproxy`, `chmod 0600`, manual rotation via
  `systemctl reload haproxy`. Acceptable for Phase 1.
- ✅ Network access control — Step 7.0.6 now mandates concrete SG rules and Step 6 (DEPLOYMENT.md)
  treats the network topology as the Phase-1 access control contract.
- ✅ Cloud smoke client named — verified `proj-2026-05-19-eou-endpointing/run_full1000_conc12.py:279-285`
  has `--url`, `--concurrency`, `--limit`. Default URL is `ws://127.0.0.1:8080`; `websockets.connect()`
  at line 117 accepts wss:// natively, so `--url wss://lb:8443` works without code change.
- ✅ Env-var namespace canonicalization — Current state now says NEMOTRON_* everywhere, drops the
  unprefixed-alias indirection.
- ✅ Troubleshooting matrix enumerated in Step 7.9 — 12 specific symptoms with causes/fixes.
- ✅ Sizing formula — Step 7.0.5 has the formula and a worked example.
- ✅ Fallback switching procedure — Step 7.7 documents the K=3+MPS swap procedure.

## Critical (must-fix before implementation)

- **`uv pip install -e .` command in Step 7.2 is wrong.** Plan says `cd $HOME/nemotron &&
  $HOME/nemo-venv/bin/uv pip install -e .` but `uv` is not installed inside the venv — `bootstrap.sh:18-19`
  installs `uv` to `$HOME/.local/bin/uv`. The venv only contains `python`, `pip`, etc. (not `uv`). The
  correct invocation is `$HOME/.local/bin/uv pip install --python $HOME/nemo-venv -e .` (matches
  `bootstrap.sh:25-27` style) OR `$HOME/nemo-venv/bin/pip install -e .` (use venv's pip directly).
  As written, the bootstrap will fail with "command not found" on a fresh box.

## Material (should-fix; folding adds real value)

- **`http-check expect string "healthy"` is loose.** A future change to the `/health` JSON
  (adding `"queue_healthy_seconds": …` or similar) could create false positives. Defensive:
  `http-check expect rstring "\"status\":[ ]*\"healthy\""` — anchored to the JSON field, regex-tolerant
  to whitespace. Plan-level adjustment to Step 3's directive spec.

- **`SRV_ENV` defaults were validated for K=3+MPS, not single-proc.** Step 1 inherits the full
  optimization stack (`NEMOTRON_SYNC_COMPRESS=1`, `FINALIZE_PRIORITY=1`, `FINALIZE_PADDED=1`,
  encoder cudagraph). These are per-process optimizations and *should* carry over, but it's an
  untested assumption. Add a note in Step 6 (DEPLOYMENT.md): "Optimization stack inherited verbatim
  from K=3+MPS validation; if single-proc smoke shows unexpected p95, A/B with `SYNC_COMPRESS=0` or
  `FINALIZE_PRIORITY=0` to isolate." This is a 1-line addition, not new work.

- **Rsync the WHOLE repo will move large unrelated trees.** `ec2-bench/` alone has many GB of
  results (`.records`, `prodsweep_*` dirs from `git status`). Plan says "rsync the WHOLE repo" but
  needs an explicit `--exclude` list: `--exclude='proj-*' --exclude='ec2-bench/prodsweep_*'
  --exclude='ec2-bench/sweep_*' --exclude='ec2-bench/.instance*.json' --exclude='*.records'
  --exclude='*.srvlog'` (or just `--exclude='ec2-bench/'` since bootstrap is self-contained). Add
  the exclusion list to Step 7.2.

- **LB host provisioning is hand-waved.** Step 7.0 mentions "separate LB host (small EC2)" and 7.1
  says "provision the LB host the same way (smaller instance, public IP, in the same VPC)" — but
  `ec2_up.py` defaults to `g6.4xlarge`. Operator needs an explicit example:
  `NEMOTRON_EC2_ITYPE=t3.medium NEMOTRON_EC2_NAME=nemotron-asr-lb NEMOTRON_EC2_STATE=.instance_lb.json
  ec2-bench/ec2_up.py`. Add to Step 7.1.

- **`/etc/nemotron/asr.env` checklist.** Step 7.3 says "edit /etc/nemotron/asr.env" but doesn't say
  what MUST be set vs what's optional. `asr.env.example` should have a clear "REQUIRED" vs "OPTIONAL"
  split at the top, and the runbook should reference that split. Without this, operators will copy
  the example verbatim and miss e.g. setting `NEMOTRON_ADMISSION_MAX_BACKLOG` (12 is plan-default,
  but Phase 1 wants it explicit).

- **Smoke check (c) state-transition timing.** With `inter 2s fall 3 rise 2` (per Step 3), the
  healthy→loading flip takes ~6s for HAProxy to mark DOWN; loading→healthy recovery takes ~4s.
  Step 5(c) must build in ≥10s polling with a defined timeout per state transition, else flakes.
  Add timing requirement to Step 5(c).

- **socat + python3 + aws-cli are LB-host prereqs but only socat is documented.** Step 7.0 mentions
  socat. drain.sh uses Python heredoc (needs `python3`). Step 7.1 uses `aws ec2 describe-instances`
  (needs aws-cli configured with the same SSO profile as `ec2_up.py`). Add both to the Prereqs
  section.

## Minor (defer to implementation)

- **Cloud smoke client choice.** `run_full1000_conc12.py` is a sandbox bench script, not a
  production smoke. It works but the RUNBOOK should note this is "the closest existing client" and
  a future phase may want a purpose-built smoke. One-line note.

- **HAProxy `ExecReload` verification.** Step 7.4 promises zero-drop reload via
  `systemctl reload haproxy`. The Debian/Ubuntu package's unit does the right `-sf` thing, but the
  runbook should include a one-line verification ("test reload during step 5: open a stream, reload
  haproxy, verify the stream doesn't drop").

- **`--ports-per-box 3` punt in fallback procedure** (Step 7.7) is acceptable as documented; could
  add a one-line "use this as a starting point" example of the manual config edit (3 server lines
  per box on 8080/8081/8082).

- **Stable backend names in `asr.env`.** Operator might set per-box env vars (e.g. host-specific
  admission cap); the `asr.env` file is per-box but the deploy procedure copies the same example
  everywhere. Document that per-box overrides go in `/etc/nemotron/asr.env` (not in the systemd
  unit's `Environment=`).

- **Cost reminder is in step 7.8 but operators reading from the top might run for hours before
  seeing it.** Surface a one-line cost reminder near step 7.1 ("g6e.4xlarge ~$2/hr; remember to
  teardown").

## Non-findings (checked, looks correct)

- `python -m nemotron_speech.server` works: `server.py:10127` has `if __name__ == "__main__": main()`,
  package structure in `pyproject.toml:64-65` packages `src/nemotron_speech`. Editable install
  exposes it via `-m`.
- `exec env -u LD_LIBRARY_PATH "${SRV_ENV[@]}" "$VENV/bin/python" -m nemotron_speech.server …` —
  array expansion + exec is correct bash; systemd's SIGTERM goes to the python process directly.
- `websockets.connect(url, …)` at `run_full1000_conc12.py:117` accepts both ws:// and wss:// natively
  (library behavior), so cloud smoke through TLS LB works without client change.
- systemd `%h` specifier resolves to the home of `User=ubuntu` (`/home/ubuntu`), so the env paths
  expand correctly without shell.
- Fallback procedure's `--ports-per-box` punt is acceptable Phase-1 scope (the fallback is a
  recovery-only path, not the primary).
- bootstrap.sh runs on the box itself; it doesn't use any recorded IP, so the private-IP work in
  Step 7.1 doesn't break bootstrap.
- `gen_haproxy.py --check` writes to a temp file when `-o` is given, so it can validate before the
  file is symlinked into `/etc/haproxy/`.
- Cross-linking: Step 7's RUNBOOK explicitly cross-references all `deploy/` artifacts.

## Verdict

MATERIAL_FINDINGS — one critical (the `uv` command in runbook step 7.2 won't run as written) plus
six material refinements. The plan is substantially hardened by round 1; remaining items are
tightenings and one fixable typo-level bug. Recommend a Round 3 to verify the next fold, then likely
ship.
