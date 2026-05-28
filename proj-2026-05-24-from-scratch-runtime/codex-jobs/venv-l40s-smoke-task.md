<task>
**Venv L40S smoke test** — validate that the new uv-native `setup-venv.sh`
works end-to-end on a clean L40S box (fresh OS, fresh CUDA stack, cold uv
cache). This is the reproducibility gate for the option-D conversion done
in commits 599f6d6 + 230d370 + 93e8208.

Launch a fresh `g6e.4xlarge` EC2 instance, copy ONLY the 4 files needed to
test the venv setup (pyproject.toml + uv.lock + setup-venv.sh +
.python-version), run setup-venv.sh, verify the venv works against the L40S
GPU, terminate the instance, write results to
`proj-2026-05-24-from-scratch-runtime/reviews/venv-l40s-smoke-result.md`.

**This is a bounded reproducibility check; not a model benchmark.** No AOTI
artifacts, no S3 pulls, no density sweeps. Just: does setup-venv.sh + the
committed uv.lock produce a working venv on a fresh box?
</task>

<context>

## Why this test matters

Local validation (RTX 5090) of the uv-native `setup-venv.sh` passed in 46
seconds — but that's because uv's local cache had every wheel from the prior
`requirements.txt` test in 599f6d6. A fresh L40S box has a cold uv cache and
a different CUDA stack. If the lockfile is missing platform markers, has
unresolved deps, or pulls torch from the wrong index, this test catches it.

The deploy target for this project is AWS SageMaker (L40S production per
`deployment-target-sagemaker` memory). Validating that a developer can clone
the repo on an L40S and run `bash setup-venv.sh` to get a working env is
the actual reproducibility contract.

## What's being tested

Commits:
- `599f6d6` — runtime venv: option D requirements.txt + bootstrap + path delocalization
- `230d370` — runtime venv: convert to uv-native (pyproject.toml + uv.lock; drop requirements.txt)
- `93e8208` — runtime venv: commit uv.lock (gitignore exception)

Files under test (all on current `khk/20260516` branch HEAD):
- `proj-2026-05-24-from-scratch-runtime/runtime/pyproject.toml` (98 lines; 75 top-level deps)
- `proj-2026-05-24-from-scratch-runtime/runtime/uv.lock` (5564 lines; 391 packages resolved)
- `proj-2026-05-24-from-scratch-runtime/runtime/setup-venv.sh` (uv-native, calls `uv sync`)
- `proj-2026-05-24-from-scratch-runtime/runtime/.python-version` (`3.12.10`)

Local fresh-install validation (RTX 5090, hot uv cache):
```
$ rm -rf .venv && bash setup-venv.sh
✓ Python 3.12.10
✓ torch 2.8.0+cu128
✓ nemo 2.4.1
✓ CUDA 12.8, NVIDIA GeForce RTX 5090
✓ nemo.collections.asr imports cleanly
Elapsed: 46 seconds. Venv: 9.7 GiB.
```

The L40S test expects much slower install (cold uv cache → real wheel
downloads). 10-20 min for download + extract is normal. **No other operations
should be slow.**

## Run plan

1. **Provision**: launch `g6e.4xlarge` in `us-west-2` via the existing EC2
   launch tooling (`ec2-bench/ec2_up.py` or AWS CLI; mirror B3-FU-5's
   launch pattern in `codex-jobs/B3-FU5-L4-knee-task.md`).
   - AMI: same as B3-FU-1/2/5 (Deep Learning AMI with CUDA 12.x + Ubuntu 22).
   - Instance type: `g6e.4xlarge` (1× L40S 48GB, 16 vCPU; cheaper than 8xlarge
     since we don't need 32 vCPU for a venv smoke).
   - State file: `ec2-bench/.instance_venv_l40s_smoke.json`.

2. **Copy only the 4 venv-setup files** (not the full repo) via rsync:
   ```
   rsync -av -e "ssh -i ec2-bench/nemotron-bench-key.pem -o StrictHostKeyChecking=no" \
       proj-2026-05-24-from-scratch-runtime/runtime/{pyproject.toml,uv.lock,setup-venv.sh,.python-version} \
       ubuntu@<ip>:~/runtime-test/
   ```

3. **Pre-flight on the L40S**:
   - Verify GPU: `nvidia-smi` shows L40S, CUDA 12.x.
   - Verify network: pip + uv index access.
   - Verify Python: should NOT have Python 3.12.10 pre-installed (uv should
     bootstrap it).
   - Confirm no uv cache: `~/.cache/uv` should be empty or minimal.

4. **Run setup-venv.sh** with timing:
   ```
   time bash ~/runtime-test/setup-venv.sh
   ```
   Time-budget for failure detection: 30 min hard timeout. Real expected time
   on a cold L40S box: 10-20 min (limited by wheel-download bandwidth, not
   CPU).

5. **Verification** (the bar):
   ```
   ~/runtime-test/.venv/bin/python -c "
   import sys, torch, nemo
   assert sys.version_info[:2] == (3, 12), f'Wrong Python: {sys.version_info}'
   assert torch.__version__.endswith('+cu128'), f'Wrong torch: {torch.__version__}'
   assert torch.cuda.is_available(), 'CUDA not available'
   assert 'L40S' in torch.cuda.get_device_name(0), f'Wrong GPU: {torch.cuda.get_device_name(0)}'
   import nemo.collections.asr  # noqa
   print(f'PASS: Python {sys.version.split()[0]}, torch {torch.__version__}, nemo {nemo.__version__}, GPU {torch.cuda.get_device_name(0)}')
   "
   ```
   Bar: this command exits 0 with "PASS:" line.

6. **Measure** + record:
   - Total `setup-venv.sh` wall time (cold-cache).
   - Venv size on disk (`du -sh ~/runtime-test/.venv`).
   - Number of unique wheels uv downloaded (`find ~/.cache/uv/wheels -name "*.whl" | wc -l`).
   - Network bytes pulled (rough — `du -sh ~/.cache/uv/wheels`).

7. **Terminate** the instance:
   - Try `ec2-bench/ec2_down.py` first.
   - Fallback to `aws ec2 terminate-instances` if boto3 issues (B3-FU-2 had
     this — Codex hit it before, AWS CLI is the workaround).
   - Wait for AWS-confirmed terminated state.
   - Update `ec2-bench/.instance_venv_l40s_smoke.json` with state=terminated.

8. **Write result** to `reviews/venv-l40s-smoke-result.md`:
   - PASS / FAIL verdict.
   - Setup-venv.sh wall time (cold-cache vs the 46s local hot-cache baseline).
   - Venv size + wheels downloaded.
   - Verification output (torch version, nemo version, GPU detected).
   - EC2 cost (instance ID, launch/terminate timestamps, elapsed × hourly).
   - Any errors encountered + their resolution (if applicable).

## Stop conditions

- **PASS**: verification command exits 0 with "PASS:" line. Write result, terminate, done.
- **FAIL — install error**: setup-venv.sh exits non-zero. Capture the full output,
  diagnose (likely: dep resolution failure on a transitive package, or torch
  wheel not pulling from cu128 index). Write FAIL result, terminate, done.
- **FAIL — verification error**: verification command exits non-zero. Likely
  causes: torch installed without +cu128 suffix, CUDA driver mismatch, NeMo
  failed to import its C extensions. Write FAIL result with diagnostic, terminate, done.
- **HARD TIMEOUT**: setup-venv.sh runs > 30 min. Kill it, capture state, terminate,
  write timeout result.

## Cost discipline

- Instance type: `g6e.4xlarge` (~$2.50-3.50/hr on-demand).
- Expected elapsed: ~30-45 min (boot + install + verify + terminate).
- Expected cost: **$1.50-$3.00 total**.
- TERMINATE on any exit path (success, failure, timeout, exception). Use
  trap/finally semantics if running interactively.

</context>

<verification_loop>
This is a single-box reproducibility check, not an iterative experiment. The
verify-command at step 5 is the single bar. If it passes, the lockfile +
bootstrap are validated for fresh L40S boxes. If it fails, the result doc
captures what broke + the L40S-specific diagnostic.

Do not retry on failure within this task — capture the failure, terminate,
write the FAIL result. Retries should happen in a separate follow-up task
after the failure mode is understood + the lockfile or script is fixed.
</verification_loop>

<action_safety>
- Bound to ONE `g6e.4xlarge`. Do NOT launch multiple instances or larger sizes.
- TERMINATE the instance on every exit path (PASS, FAIL, timeout, exception).
- Use AWS CLI for termination if `ec2_down.py` fails (FU-2 hit a boto3
  import issue; the workaround is `aws ec2 terminate-instances --instance-ids
  <id> --region us-west-2 --profile AWSAdministratorAccess-419599258555`).
- Do NOT copy the full repo to the L40S; only the 4 venv-setup files. Saves
  bandwidth + time + reduces the test's surface area.
- Do NOT install AOTI artifacts, do NOT pull from S3, do NOT run density
  sweeps. This is a venv-setup smoke test only.
- AWS profile: `AWSAdministratorAccess-419599258555`. Region: `us-west-2`.
- SSH key: `ec2-bench/nemotron-bench-key.pem`.
- State file: `ec2-bench/.instance_venv_l40s_smoke.json`.
</action_safety>

<compact_output_contract>
Report:
- Path to `reviews/venv-l40s-smoke-result.md`.
- One-paragraph headline: PASS/FAIL verdict + setup-venv.sh cold-cache wall time
  (vs the 46s local hot-cache baseline) + any deviations from the local result.
- EC2 state confirmation: instance ID, launch/terminate timestamps, cost estimate,
  state-file path.
- If FAIL: top-3 diagnostic lines.
</compact_output_contract>
