#!/bin/bash
# Bootstrap script for the native runtime export + oracle venv.
#
# Creates ./.venv at this directory, installs the full NeMo + torch+cu128 +
# AOTI compile-time dependency tree from requirements.txt, and verifies
# torch.cuda is functional + nemo loads.
#
# Prerequisites:
#   - uv (auto-installed if missing).
#   - Python 3.12.10 — pinned in .python-version (uv can bootstrap it).
#   - CUDA 12.x on host (for torch+cu128 to load at runtime).
#   - NVIDIA driver compatible with CUDA 12.8.
#   - ~11 GiB disk for the venv (NeMo + torch + transformers + bitsandbytes +
#     pynini + janome + faiss + nvidia/* CUDA wheels).
#
# Usage (from this directory):
#   bash setup-venv.sh
#
# Or to put the venv elsewhere:
#   VENV_PATH=/somewhere/else bash setup-venv.sh
#
# After setup, scripts run as (from this runtime/ directory):
#   HF_HUB_OFFLINE=1 ./.venv/bin/python export_steady_batched.py --out ./artifacts
#
# Or from the repo root:
#   HF_HUB_OFFLINE=1 proj-2026-05-24-from-scratch-runtime/runtime/.venv/bin/python \
#       proj-2026-05-24-from-scratch-runtime/runtime/export_steady_batched.py --out ...
#
# Why this venv lives next to runtime/: the export scripts + finalize_ref.py
# (Python oracle) reference it; co-locating keeps the dependency boundary
# clear. The repo's top-level pyproject.toml + uv.lock are for the production
# WebSocket server (a much smaller dep set: pipecat-ai, nvidia-riva-client,
# websockets, aiohttp, numpy, loguru) — kept separate intentionally to avoid
# bloating that lockfile with NeMo+torch+CUDA wheels.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_PATH="${VENV_PATH:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12.10}"
REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

echo "=== Nemotron native-runtime export venv setup ==="
echo "  venv path:   $VENV_PATH"
echo "  python:      $PYTHON_VERSION"
echo "  reqs file:   $REQUIREMENTS"
echo "  pytorch idx: $PYTORCH_INDEX_URL"
echo

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"
echo

# Create the venv with the pinned Python version (uv bootstraps Python if not
# already installed via pyenv or system).
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating venv at $VENV_PATH (Python $PYTHON_VERSION)..."
    uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
else
    echo "Reusing existing venv at $VENV_PATH"
fi
echo

# Install from the requirements lockfile, with the cu128 index for torch wheels.
# --index-strategy unsafe-best-match lets uv prefer the PyTorch index when
# torch/torchvision are explicitly pinned to +cu128 versions.
echo "Installing requirements (this can take 5-20 minutes; ~11 GiB of wheels)..."
VIRTUAL_ENV="$(cd "$VENV_PATH" && pwd)" \
    uv pip install \
        --index-strategy unsafe-best-match \
        --extra-index-url "$PYTORCH_INDEX_URL" \
        -r "$REQUIREMENTS"
echo

# Verify
echo "Verifying install..."
"$VENV_PATH/bin/python" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 12), f"Expected Python 3.12, got {sys.version_info}"
print(f"  ✓ Python {sys.version.split()[0]}")

import torch
assert torch.__version__.endswith("+cu128"), f"Expected torch+cu128, got {torch.__version__}"
print(f"  ✓ torch {torch.__version__}")

import nemo
print(f"  ✓ nemo {nemo.__version__}")

# Soft CUDA check (no GPU needed for export/oracle work on a CPU dev host, but warn)
if torch.cuda.is_available():
    print(f"  ✓ CUDA {torch.version.cuda} available, device: {torch.cuda.get_device_name(0)}")
else:
    print(f"  ⚠ CUDA not available on this host (export-only ok; oracle work needs a GPU)")

# Quick NeMo ASR import check (the most common failure mode if NeMo deps drift)
try:
    import nemo.collections.asr as nemo_asr  # noqa: F401
    print(f"  ✓ nemo.collections.asr imports cleanly")
except Exception as e:
    print(f"  ✗ nemo.collections.asr import failed: {e}")
    sys.exit(1)
PY

echo
echo "=== Setup Complete ==="
echo
echo "Run scripts from this directory (proj-2026-05-24-from-scratch-runtime/runtime/) as:"
echo "  HF_HUB_OFFLINE=1 ./.venv/bin/python <script>"
echo
echo "Or from the repo root as:"
echo "  HF_HUB_OFFLINE=1 proj-2026-05-24-from-scratch-runtime/runtime/.venv/bin/python <script>"
