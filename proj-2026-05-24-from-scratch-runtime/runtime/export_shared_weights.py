#!/usr/bin/env python3
"""1.3b-enc shared-weights source: save the encoder's params+buffers keyed by the AOTI constant FQN ('e.<name>', from the
export Step naming the encoder 'e') so per-T finalize buckets (constants-on-disk) share ONE weight set via
loader.load_constants(user_managed=True). Saves BOTH a .pt (Python consumers) and a TorchScript .ts holding a
Dict[str,Tensor] attribute "weights" keyed by FQN (C++ consumers: FQNs have dots so they can't be buffer names).

Run: HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python export_shared_weights.py
"""
import os, itertools
from typing import Dict
import torch

ART = "artifacts"
PT = os.path.join(ART, "finalize_shared_weights.pt")


class SharedWeights(torch.nn.Module):
    def __init__(self, cmap: Dict[str, torch.Tensor]):
        super().__init__()
        self.weights: Dict[str, torch.Tensor] = cmap

    def forward(self) -> Dict[str, torch.Tensor]:
        return self.weights


def build_cmap() -> Dict[str, torch.Tensor]:
    if os.path.exists(PT):
        return torch.load(PT, weights_only=False)   # reuse (no model reload)
    from finalize_ref import load_model
    enc = load_model().encoder
    cmap = {}
    for name, t in itertools.chain(enc.named_parameters(), enc.named_buffers()):
        cmap["e." + name] = t.detach().cpu()
    torch.save(cmap, PT)
    return cmap


def main():
    cmap = build_cmap()
    sm = torch.jit.script(SharedWeights(dict(cmap)))
    sm.save(os.path.join(ART, "finalize_shared_weights.ts"))
    print(f"saved {len(cmap)} weights -> finalize_shared_weights.pt + .ts (Dict[str,Tensor] for C++)")


if __name__ == "__main__":
    main()
