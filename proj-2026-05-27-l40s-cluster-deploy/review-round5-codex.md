# Round 5 Codex review

## Resolved from round 4
- `option dontlog-normal` is removed from the generator defaults, and the validation section now has an explicit negative assertion that it must be absent. The plan also correctly documents that HAProxy does not log successful health checks by default.
- Step 7.1.5's SG block is now rerunnable in shape: describe-or-create SGs, tolerate duplicate ingress grants, and de-dupe the `modify-instance-attribute --groups` list before replacing it.
- The Step 7.1.5 failure-mode parenthetical is corrected: without the extra backend SG rule, the LB cannot reach backend `:8080`; the old "open from any source" wording is gone.
- `CLIENT_CIDR` and `FRONT_PORT` are now explicit in the SG snippet and tied back to `gen_haproxy.py`'s frontend port choice.
- Step 7.6b now rehydrates `BE_SG` by name before failed-box replacement, so it no longer assumes the original deployment shell variables still exist.
- Step 7.6b's ordering claim is softened correctly: HAProxy health checks make regen-first safe, while regen-after-healthy is cleaner operationally.
- Smoke step 5(c) now requires an explicit `mktemp -d` stats socket passed through `--stats-socket`, so the smoke can reliably query the same runtime socket it generated.
- Cloud smoke now explicitly runs from the operator workstation and derives `LB_IP` from `ec2-bench/.instance_lb.json`.
- Step 2 reframes `asr.env.example` as `RECOMMENDED-EXPLICIT` plus `OPTIONAL`, with no env var strictly required for the unit to start. There is one stale runbook reference left; see Material.

## Critical (must-fix before implementation)
- None.

## Material (should-fix; folding into the plan adds real value)
- The rsync command will copy the EC2 SSH private key to every backend and the LB host. After Step 1, `ec2_up.py` has created `ec2-bench/nemotron-bench-key.pem`, and Step 7.2's rsync excludes `.instance*.json` but not `ec2-bench/*.pem` (`PLAN.md:438-448`). That leaks the cluster SSH key onto the machines it controls. Add `--exclude='ec2-bench/*.pem'` by default, then make Step 7.6 explicit about where rolling redeploy commands run and how SSH is performed. If the LB host is meant to SSH into backends, copying the key there should be a deliberate, separately documented step with `0600` perms, not a side effect of repo sync.
- Step 7.4's HAProxy config command is not copy-pasteable as an Ubuntu operator. `python3 ... -o /etc/haproxy/haproxy.cfg --check` will not be able to write `/etc/haproxy/haproxy.cfg` without `sudo`; if the PEM is installed as `haproxy:haproxy 0600`, a non-root `haproxy -c` check also cannot read it; and the PEM must exist before `--check` runs (`PLAN.md:476-484`). Fold in an explicit order: install/chown the PEM first, generate to a temp file then `sudo install` and `sudo haproxy -c`, or run the generator/check under `sudo`.
- The TLS smoke URL still assumes an IP address can validate the certificate. Step 7.5 uses `wss://$LB_IP:8443` (`PLAN.md:490-497`), but a normal cert will be issued for a DNS name, not the raw EC2 public IP. The runbook should either require a DNS name whose A/ALIAS record points at the LB and use `wss://$LB_HOSTNAME:8443`, or explicitly document the no-TLS/plain `ws://` smoke path when TLS is not configured.
- AWS CLI snippets are not pinned to the same account/region as `ec2_up.py`. The provisioner uses `NEMOTRON_AWS_PROFILE` with default `AWSAdministratorAccess-419599258555` and hard-codes `us-west-2`, while the runbook's `aws ec2 ...` commands rely on ambient AWS CLI defaults. Add an early `export AWS_PROFILE=... AWS_REGION=us-west-2 AWS_DEFAULT_REGION=us-west-2` (or `--profile/--region` on every command) so SG reconciliation, private-IP fetch, and teardown cannot silently target the wrong account/region. Also add `jq` to workstation/LB prerequisites or replace the `jq` calls.
- The K=3+MPS fallback procedure is still not runnable with the Phase-1 package layout. Step 7.7 says to swap the systemd `ExecStart` to `launch_multiproc.sh`, but that script runs `python server.py` from `$NEMOTRON_APP_DIR`; the Phase-1 deploy layout installs the package and has `src/nemotron_speech/server.py`, not `$HOME/nemotron/server.py`. Either document the fallback compatibility step (copy/symlink a root `server.py` plus required sibling modules) or update the fallback procedure to patch/use a module-form multiproc launcher.
- Step 7.3 still says "`asr.env`: the REQUIRED block at the top" (`PLAN.md:463-464`), contradicting the round-4 reframing and Step 2's `RECOMMENDED-EXPLICIT` terminology. Rename that runbook text to `RECOMMENDED-EXPLICIT` so operators do not infer that `NEMOTRON_ADMISSION_MAX_BACKLOG=12` is functionally required to boot.

## Minor (nice-to-have; OK to defer to implementation if low cost there)
- The Scope rule still says "the only delta from `launch_multiproc.sh` is removing MPS + the K-loop" (`PLAN.md:86-87`). Step 1 correctly also drops the supervisor loop, file log redirect, traps, and switches to module form. Update the rule to avoid a stale "only delta" contradiction.
- Step 0 says `ec2_up.py` invocations run from repo root "or `cd ec2-bench/` first" (`PLAN.md:357-359`), while the rest of the runbook is now repo-root style. Dropping the alternate `cd` wording would prevent the old state-file path ambiguity from reappearing.
- Step 4 installs `awscli` on the LB host, but the plan does not show the LB host using AWS after that point. If it is only for operator convenience, say so; if SSO-backed AWS CLI use is required on the LB, prefer an AWS CLI v2 install note over Ubuntu's potentially stale `awscli` package.

## Non-findings (things I checked that look correct)
- The round-4 logging fix is correct: `option tcplog` remains present, `option dontlog-normal` is absent, and health-check success spam is not a default HAProxy logging problem.
- The SG idempotency snippet now handles the important `modify-instance-attribute --groups` replace semantics by including existing groups and de-duping before modification.
- `VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true ...)` matches `ec2_up.py`, which also provisions into the default VPC.
- `show stat` over the HAProxy runtime socket, strip leading `# ` from the CSV header, and parse `pxname/svname/scur` by name remains the right drain parsing contract.
- The `http-check expect rstring "\"status\"[ ]*:[ ]*\"healthy\""` directive is consistently specified and avoids the old loose `"healthy"` substring check.
- `--local-test` now has a coherent stats-socket contract for the smoke test.
- The public-subnet plus private-backend-IP plus SG-source-group access-control story is internally consistent for Phase 1.
- `maxconn 20` versus `--maxconn-conservative 12` is consistently caveated as one-utterance-per-connection validation versus safer sustained-multi-turn default.

## Verdict
MATERIAL_FINDINGS — no new critical architecture blocker, but the implementation pass should fold the runbook fixes above rather than copy the current commands literally.
