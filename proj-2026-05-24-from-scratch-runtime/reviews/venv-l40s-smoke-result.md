# Venv L40S Smoke Result

## Verdict

PASS. On a fresh `g6e.4xlarge` L40S DLAMI box with no pre-existing `uv`, no
Python 3.12, and no `~/.cache/uv`, the committed `setup-venv.sh` plus
`pyproject.toml`, `uv.lock`, and `.python-version` produced a working venv.

`setup-venv.sh` cold-cache wall time was 67.13s by `/usr/bin/time` (67s by
UTC timestamps), versus the local RTX 5090 hot-cache baseline of 46s. The
runtime versions matched the local result: Python 3.12.10, torch
2.8.0+cu128, NeMo 2.4.1. The GPU changed as expected from RTX 5090 to
NVIDIA L40S. The only CUDA-related deviation was that `nvidia-smi` reported
driver capability CUDA 13.0 on the fresh DLAMI, while torch runtime CUDA was
12.8 and the cu128 verification passed.

## Instance

- Instance: `i-0696ea37b2507dc95`
- Type/region: `g6e.4xlarge`, `us-west-2`
- AMI: `ami-0ad0f6da3aabae4c6`, `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260526`
- Name tag: `nemotron-bench-venv-l40s-smoke-20260528T225136Z`
- State file: `ec2-bench/.instance_venv_l40s_smoke.json`
- Launch: `2026-05-28T22:51:42+00:00`
- Terminated: `2026-05-28T22:59:43Z`
- AWS-confirmed final state: `terminated`
- EC2 elapsed: 481s, 0.1336h
- Price used: `$3.00424/hr` from AWS Pricing API for Linux shared `g6e.4xlarge` in US West (Oregon)
- Estimated instance cost: `$0.40`

## Preflight

- GPU: `NVIDIA L40S`, driver `580.159.04`, `46068 MiB`
- `nvidia-smi` CUDA capability: `13.0`
- Torch runtime CUDA after install: `12.8`
- Host Python before setup: `Python 3.10.12`
- `python3.12`: not present before setup
- `uv`: not present before setup
- `~/.cache/uv`: absent before setup
- Network checks: PyPI `uv` index reachable, `astral.sh` uv installer reachable, PyTorch `cu128` index reachable
- Remote copy: exactly 4 files in `~/runtime-test`: `.python-version`, `pyproject.toml`, `setup-venv.sh`, `uv.lock`

## Setup

- `setup-venv.sh` status: `0`
- Start: `2026-05-28T22:52:32Z`
- End: `2026-05-28T22:53:39Z`
- `/usr/bin/time`: `elapsed_seconds=67.13 user_seconds=96.91 sys_seconds=27.13 max_rss_kb=1325820 exit_status=0`
- `uv`: installed `uv 0.11.17`
- Python bootstrapped by uv: `CPython 3.12.10`
- `uv lock` resolution used: `Resolved 391 packages in 0.96ms`
- Packages prepared/installed: `389`
- Venv size: `9.7G`
- Requested wheel count command: `find ~/.cache/uv/wheels -name "*.whl" | wc -l` returned `0`
- `~/.cache/uv/wheels`: missing under uv 0.11.17 cache layout
- Total uv cache after run: `9.8G`, `85089` files

## Verification

Verification command exited `0` and printed:

```text
PASS: Python 3.12.10, torch 2.8.0+cu128, nemo 2.4.1, GPU NVIDIA L40S
```

The setup script's built-in verification also passed:

```text
  ✓ Python 3.12.10
  ✓ torch 2.8.0+cu128
  ✓ nemo 2.4.1
  ✓ CUDA 12.8 available, device: NVIDIA L40S
  ✓ nemo.collections.asr imports cleanly
```

Non-fatal warnings observed during NeMo import:

- `pynvml` deprecation warning from `torch.cuda`
- `pydub` warning that `ffmpeg`/`avconv` was not found

## Errors And Resolutions

- Initial local AWS CLI dry-runner attempt used the older `run-instances --min-count/--max-count` spelling. Current AWS CLI accepted `--count 1` instead. No instance was launched before this correction.
- `ec2_down.py` was tried first for teardown as requested, but failed with the known boto3 SSO `TokenRetrievalError`. The AWS CLI fallback terminated `i-0696ea37b2507dc95`, and `aws ec2 wait instance-terminated` confirmed final state `terminated`.
- Re-authorizing the SSH security group rule reported `InvalidPermission.Duplicate`; the existing rule was correct and reused.

Raw local run artifacts: `/tmp/venv-l40s-smoke-20260528T225136Z`.
