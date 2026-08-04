#!/usr/bin/env python3
"""
Fix Gemma-4 MTP draft-head logit suppression under CUDA graph capture.

vLLM 0.26.0 stores the draft model's `suppress_tokens` (from its
generation_config) as a plain Python list and then indexes CUDA logits with it:

    logits[:, self._suppress_token_ids] = -float("inf")

Indexing a CUDA tensor with a Python list forces an implicit host->device copy
from pageable memory, which is illegal while the speculator's prefill CUDA graph
is being captured:

    RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
    capture unless the CPU tensor is pinned.

So any Gemma-4 MTP drafter whose generation_config sets `suppress_tokens` cannot
start unless CUDA graphs are disabled, which costs exactly the latency spec
decoding was added to buy.

The fix materializes the ids once, at model-init time, as a device tensor.
Not fixed upstream as of vllm main (checked 2026-08-03).

Usage: VLLM_DIR=<site-packages/vllm> python apply-vllm-gemma4-mtp-suppress-tokens.py
"""

import os
import sys

VLLM_DIR = os.environ.get("VLLM_DIR", "/usr/local/lib/python3.12/site-packages/vllm")

MARKER = "# patched: suppress_tokens as a registered buffer"

OLD_INIT = """        self._suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None"""

# It has to be a *buffer*, not a plain tensor attribute: nn.Module.to() moves
# parameters and buffers only, so a plain attribute built at __init__ time stays
# on whatever device the module was constructed on and reintroduces the copy.
# persistent=False keeps it out of the state_dict so weight loading ignores it.
NEW_INIT = f"""        _suppress_token_ids = gen_cfg.get("suppress_tokens") if gen_cfg else None
        {MARKER}: indexing CUDA logits with a
        # Python list (or a CPU tensor) forces a pageable host->device copy,
        # which is illegal while the speculator's CUDA graph is being captured.
        self.register_buffer(
            "_suppress_token_ids",
            torch.tensor(list(_suppress_token_ids), dtype=torch.long)
            if _suppress_token_ids
            else None,
            persistent=False,
        )"""

# A tensor is ambiguous in a boolean context, so the truthiness test has to go too.
OLD_LOGITS = """        if logits is not None and self._suppress_token_ids:
            logits[:, self._suppress_token_ids] = -float("inf")"""

NEW_LOGITS = """        if logits is not None and self._suppress_token_ids is not None:
            logits[:, self._suppress_token_ids] = -float("inf")"""


def patch_gemma4_mtp() -> bool:
    filepath = os.path.join(VLLM_DIR, "vllm/model_executor/models/gemma4_mtp.py")
    if not os.path.exists(filepath):
        # Try the layout where VLLM_DIR is the package itself rather than its parent.
        filepath = os.path.join(VLLM_DIR, "model_executor/models/gemma4_mtp.py")
    if not os.path.exists(filepath):
        print(f"ERROR: gemma4_mtp.py not found under {VLLM_DIR}")
        return False

    with open(filepath) as f:
        content = f.read()

    if MARKER in content:
        print(f"SKIP: {filepath} already patched")
        return True

    for old in (OLD_INIT, OLD_LOGITS):
        if old not in content:
            print(f"ERROR: expected source not found in {filepath}:\n{old}")
            return False

    content = content.replace(OLD_INIT, NEW_INIT).replace(OLD_LOGITS, NEW_LOGITS)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"PATCHED: {filepath}")
    return True


if __name__ == "__main__":
    sys.exit(0 if patch_gemma4_mtp() else 1)
