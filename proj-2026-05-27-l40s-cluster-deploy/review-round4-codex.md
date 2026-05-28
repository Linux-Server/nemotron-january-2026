# Round 4 Codex review

## Resolved from round 3
- The round-3 critical SG shape is now executable in principle: Step 1.5 creates separate `nemotron-asr-lb-sg` and `nemotron-asr-backend-sg`, authorizes `client CIDR -> LB:8443` and `LB_SG -> backend:8080`, and attaches the new SGs alongside the existing `nemotron-bench-sg`.
- `aws ec2 modify-instance-attribute --groups` does **replace** the instance's SG list, not append to it. The plan's `EXISTING=$(describe-instances ...)` followed by `--groups $EXISTING $BE_SG` is the right basic pattern and preserves `nemotron-bench-sg` for SSH. Source: AWS CLI v2 docs for `modify-instance-attribute --groups` say it "Replaces the security groups of the instance with the specified security groups": https://docs.aws.amazon.com/cli/latest/reference/ec2/modify-instance-attribute.html
- The repo-root `python ec2-bench/ec2_up.py` examples and `ec2-bench/.instance_*.json` state-file paths are now consistent in the actual provisioning commands.
- The rsync exclusions now preserve `ec2-bench/local_lb.py`, exclude local venvs and `.git/`, and include an explicit LB-host code-copy step.
- The stale loose `"healthy"` health-check references were updated to the JSON-field-anchored `http-check expect rstring "\"status\"[ ]*:[ ]*\"healthy\""`.
- The `launch_multiproc.sh` reference bullet now correctly says to drop the supervisor loop for the single-proc launcher.
- The current-state `SRV_ENV` summary now uses canonical `NEMOTRON_*` names.
- `--local-test` now resolves a concrete socket path, removes the production `user/group haproxy` assumptions, drops the high `ulimit-n`, and uses stdout logging for non-root HAProxy.
- Box replacement is now separated from rolling redeploy, and the fallback procedure now mentions the `launch_multiproc.sh` in-script supervisor interaction under systemd.
- `uv pip install --no-deps --python "$HOME/nemo-venv" -e "$HOME/nemotron"` is the right install shape for this production server path. `pyproject.toml:22-32` declares `gdown`, `pipecat-ai`, and `nvidia-riva-client`, but `src/nemotron_speech/server.py` directly imports only `numpy`, `torch`, `aiohttp`, `loguru`, plus lazy `nemo`/`omegaconf`; those are covered by `ec2-bench/bootstrap.sh:24-31`.

## Critical (must-fix before implementation)
- `option dontlog-normal` is not health-probe-only. HAProxy documents it as disabling logs for normal, successful connections; in this plan, putting it in `defaults` would suppress normal successful production WebSocket/TCP client streams as well as any other normal frontend/listen traffic. The plan rationale at `PLAN.md:204-206` is therefore wrong, and it conflicts with the requirement to retain production traffic logs. Also, routine health-check success spam is not normally the thing `dontlog-normal` is for; `option log-health-checks` is the knob for health-check status-change logging. Fix: remove `option dontlog-normal` from generator defaults, keep `option tcplog`, and if log volume is a concern document journald/log rotation or `log-separate-errors` rather than suppressing all normal successes. Source: HAProxy 3.0 config manual, `option dontlog-normal` and `option log-health-checks`: https://docs.haproxy.org/3.0/configuration.html

## Material (should-fix; folding into the plan adds real value)
- Step 1.5 is not rerunnable as written. `create-security-group` fails on an existing `nemotron-asr-lb-sg` / `nemotron-asr-backend-sg`; duplicate ingress authorization can fail; and rerunning the attach snippet after a successful run makes `EXISTING` already include `$BE_SG` / `$LB_SG`, then appends the same group again. The runbook should use describe-or-create for both SGs, tolerate duplicate ingress rules, and de-dupe the final `--groups` list before calling `modify-instance-attribute`.
- Step 6b cross-references the right SG attachment block, but it relies on `$BE_SG` still existing in the operator's shell. Failed-box replacement may happen days later. Either Step 6b should explicitly rehydrate `BE_SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=nemotron-asr-backend-sg ...)`, or Step 1.5 should become a small idempotent snippet the operator can rerun during replacement.
- Smoke check 5(c) still lacks a precise socket ownership contract. Step 3 says `--local-test` resolves a concrete temp socket path, but the smoke test must also know that path to run `show stat`. Make 5(c) allocate `tmpdir=$(mktemp -d)`, pass `--local-test --stats-socket "$tmpdir/haproxy.sock"` to `gen_haproxy.py`, and use that same path for `socat` / `HAPROXY_SOCK`, with cleanup on exit. If the generator auto-creates the temp path instead, it must print or write the chosen path somewhere machine-readable.

## Minor (nice-to-have; OK to defer to implementation if low cost there)
- `PLAN.md:383-384` says that without Step 1.5, `:8080` is open from any source attached to the shared SG. That is not what `ec2_up.py` currently does: `nemotron-bench-sg` only authorizes SSH from `MY_IP`. The real failure mode without Step 1.5 is "LB cannot reach backend :8080", not "public/shared-SG :8080 is open." Fix the parenthetical so the security story is accurate.
- Step 1.5 should show an explicit `CLIENT_CIDR=...` assignment before the ingress command, and should note that if the generator is run without TLS on `--front-port 8080`, the LB SG ingress port must match that frontend port instead of `8443`.
- The generator validation list at `PLAN.md:109-113` should be updated after the logging fix: assert `option tcplog` is present and `option dontlog-normal` is absent unless the operator explicitly requests it.

## Non-findings (things I checked that look correct)
- `modify-instance-attribute --groups` replacing the full SG list is handled correctly by including `EXISTING` in the command. The remaining issue is idempotent de-duping, not the replace-vs-add semantics.
- The `--no-deps` editable install does not miss a direct runtime dependency for `python -m nemotron_speech.server`. The skipped `pyproject.toml` dependencies are broader project/bot/TTS deps, not required by the Phase-1 ASR server path.
- Step 6b's order is conservative and defensible. HAProxy health checks would eventually mark a not-ready backend DOWN, but a newly added checked server can still be considered available initially on common HAProxy versions unless explicitly configured otherwise. Waiting for local `/health` green before regenerating/reloading avoids that startup window.
- The Step 6b SG cross-reference points to the right operation: a replacement backend needs the backend SG attached alongside `nemotron-bench-sg`, not the LB SG.
- The `rstring` health check, `inter 2s fall 3 rise 2` smoke timing, and healthy-loading-healthy smoke cycle are internally consistent.
- The rsync pattern `ec2-bench/local_*/` excludes result directories without excluding `ec2-bench/local_lb.py`.

## Verdict
CRITICAL_FINDINGS - the plan would currently implement `option dontlog-normal` based on a false health-probe-only assumption and lose normal production success logs.
