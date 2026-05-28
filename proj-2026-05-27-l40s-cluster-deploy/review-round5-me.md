# Round 5 self-review (Opus) — FINAL

Read the full revised PLAN.md top-to-bottom. Looking for: layered-edit inconsistencies, anything
the prior rounds let slip, anything that's still hand-waved. The plan has hardened a lot — this
review is mostly a final consistency sweep.

## Resolved from round 4

- ✅ `option dontlog-normal` removed from Step 3 generator defaults; explicit explanation that
  HAProxy doesn't log successful health probes by default anyway.
- ✅ Negative-assertion in validation (line 113): `option dontlog-normal` must be ABSENT.
- ✅ Step 7.1.5 is now idempotent: `ensure_sg` describe-or-create helper, `|| true` on duplicate
  ingress, `sort -u` de-dup before modify-instance-attribute. Re-runnable.
- ✅ Parenthetical corrected (line 390): "without this, the LB cannot reach backends on :8080"
  (accurate — `nemotron-bench-sg` only authorizes :22 from MY_IP).
- ✅ CLIENT_CIDR and FRONT_PORT explicit assignments with comments.
- ✅ Step 7.6b rehydrates `$BE_SG` from `describe-security-groups` by name.
- ✅ Step 7.6b order claim softened: regen-first is also safe (HAProxy `check` handles it);
  regen-after-healthy is operational preference only.
- ✅ Step 5(c): `mktemp -d` + explicit `--stats-socket "$tmpdir/haproxy.sock"` pass-through,
  cleanup on exit.
- ✅ Step 7.5 cloud smoke: runs from operator workstation; LB_IP fetched from
  `ec2-bench/.instance_lb.json`; explicit path to the client.
- ✅ asr.env reframed: RECOMMENDED-EXPLICIT vs OPTIONAL (Step 2); nothing is strictly required.

## Critical (must-fix before implementation)

(None.)

## Material (should-fix; folding into the plan adds real value)

- **Step 7.3 still uses stale "REQUIRED block" wording** (line 463–464) while Step 2 (line 184) now
  calls it RECOMMENDED-EXPLICIT. The runbook should say "the RECOMMENDED-EXPLICIT block at the top
  (set these explicitly for ops visibility, though they match launcher defaults) vs the OPTIONAL
  block (touch only when tuning)" — match Step 2's terminology.

- **Rules → Scope discipline (line 86–87) is stale.** It says "the only delta from
  `launch_multiproc.sh` is removing MPS + the K-loop" — but per Step 1 and the Reference
  implementations bullet, the supervisor loop is ALSO removed. Update to "removing MPS + the K-loop
  + the in-script supervisor loop (systemd is the sole supervisor — see Correctness/ops safety)."

- **Rules → Correctness/ops safety (line 101) says "recommended 8–12"** for
  `NEMOTRON_ADMISSION_MAX_BACKLOG`, but every other reference in the plan uses 12 specifically.
  Tighten to "default 12" to match the rest of the plan and the launcher default (Step 1, line 161).

## Minor (defer to implementation)

- **Progress table (line 591) step 7 title** doesn't include "+ deploy/README.md index" that the
  step description added. Cosmetic.

- **Step 7.5 implicitly assumes a Python venv with `websockets`** on the operator's workstation
  (the smoke client imports `websockets`). One-line note: "run from a venv that has `websockets`
  installed — `pip install websockets` if your operator venv doesn't already have it."

- **Step 7.1.5 verify command** only checks the LB host's SGs; could also verify one backend box
  has both `nemotron-bench-sg` and `nemotron-asr-backend-sg`. Nice-to-have.

## Non-findings (things I checked that look correct)

- The whole plan is internally consistent on the major decisions: single-proc, `maxconn 20`,
  systemd-as-sole-supervisor, `rstring` health check, `--no-deps` install, repo-root invocation,
  private-IP backends, idempotent SG block. No contradictions between sections on these.
- Step 7.4's "the LB host doesn't need the model/venv/bootstrap, just the deploy/ directory and
  ec2-bench/local_lb.py" — verified the rsync exclusion list preserves both: `deploy/` is not
  excluded; `ec2-bench/local_*/` excludes only directories (trailing slash) so `local_lb.py`
  survives.
- Step 7.5's smoke command is concretely runnable with the substitutions documented.
- Step 7.1.5's `modify-instance-attribute --groups` REPLACE semantics are handled correctly with
  the `existing $new_sg | sort -u` dedup.
- Step 7.6b correctly rehydrates `$BE_SG` from name (operator's shell from days/weeks earlier is
  gone).
- Generator's `--local-test` ↔ smoke (c) handoff via explicit `--stats-socket` is consistent
  between Step 3 spec and Step 5 smoke spec.
- Documentation rules section is complete and references every artifact the steps produce.
- The 10-subsection RUNBOOK (Step 7) covers prereqs → sizing → topology → provision → SG →
  bootstrap → systemd → LB → cloud-smoke → rolling-redeploy → box-replace → fallback → teardown →
  troubleshoot → open-risks. Comprehensive.

## Verdict

MATERIAL_FINDINGS — three layered-edit consistency cleanups (stale "REQUIRED" wording in Step 7.3,
stale "MPS + K-loop only" in Rules, "8–12" vs "12" in Rules). All trivial 1-line fixes. No
criticals. Per the user's stop condition ("five iterations and minimal findings can be addressed
during implementation"), ship to /implement after this fold — the residual cleanups can land
in-implementation if needed, but I'll fold them now since they're trivial.
