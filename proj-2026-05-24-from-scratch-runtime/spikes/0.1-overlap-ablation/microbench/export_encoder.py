#!/usr/bin/env python3
"""0.1b prerequisite — MECHANICAL export of the steady streaming encoder for the C++ microbench.

Scope: make `encoder.cache_aware_stream_step` loadable + runnable in libtorch C++ at the steady B=1 bucket shape, so the
microbench can capture+replay a realistic CUDA graph from multiple threads. This is NOT the 0.2 fidelity/byte-exact gate
(that's Wave-2) — we only need it to RUN at the right shape/cost.

Run in the env that has torch+nemo (the parakeet venv):
  /home/khkramer/src/parakeet/venv/bin/python export_encoder.py --out ./artifacts

Outputs: artifacts/encoder_steady_b1.ts (TorchScript) + artifacts/shapes.json (static input shapes for the C++ harness).
If TorchScript tracing of the streaming step fails (known risk: the drop_extra global mutation + cache control flow),
the failure is itself a useful 0.2 datapoint — record it; the microbench can fall back to a kernel-sequence stand-in.
"""
from __future__ import annotations

import argparse
import json
import os

import torch


MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"


def build_steady_inputs(model):
    """Steady bucket, B=1: processed_signal [1, F, pre_encode_cache+shift], length, 3 cache tensors."""
    enc = model.encoder
    scfg = enc.streaming_cfg
    def _int(v):
        return int(v[1]) if isinstance(v, (list, tuple)) else int(v)
    shift = _int(scfg.shift_size)
    pre = _int(scfg.pre_encode_cache_size)
    drop_extra = int(scfg.drop_extra_pre_encoded)
    T = pre + shift
    feat = int(model.cfg.preprocessor.features)  # 128
    cache = enc.get_initial_cache_state(batch_size=1)
    dev = cache[0].device
    processed = torch.zeros((1, feat, T), device=dev, dtype=cache[0].dtype)
    length = torch.full((1,), T, device=dev, dtype=torch.long)
    return {
        "processed_signal": processed, "processed_signal_length": length,
        "cache_last_channel": cache[0], "cache_last_time": cache[1], "cache_last_channel_len": cache[2],
        "drop_extra": drop_extra, "keep_all_outputs": False,
        "shapes": {"feat": feat, "T": T, "shift": shift, "pre": pre, "drop_extra": drop_extra,
                   "clc": list(cache[0].shape), "clt": list(cache[1].shape), "clcl": list(cache[2].shape)},
    }


class SteadyEncoderStep(torch.nn.Module):
    """Wrapper with a fixed-signature forward, restoring NeMo's global drop_extra (mirrors cudagraph_encoder.py:56-64)."""
    def __init__(self, model, drop_extra, keep_all_outputs):
        super().__init__()
        self.model = model
        self.drop_extra = int(drop_extra)
        self.keep_all_outputs = bool(keep_all_outputs)

    def forward(self, processed_signal, processed_signal_length, clc, clt, clcl):
        scfg = self.model.encoder.streaming_cfg
        orig = scfg.drop_extra_pre_encoded
        try:
            return self.model.encoder.cache_aware_stream_step(
                processed_signal=processed_signal, processed_signal_length=processed_signal_length,
                cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                keep_all_outputs=self.keep_all_outputs, drop_extra_pre_encoded=self.drop_extra)
        finally:
            scfg.drop_extra_pre_encoded = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./artifacts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL, map_location="cpu").cuda().eval()
    try:
        model.encoder.set_default_att_context_size([70, 1])
    except Exception:
        pass
    try:
        model.preprocessor.featurizer.dither = 0.0
    except Exception:
        pass

    inp = build_steady_inputs(model)
    wrapper = SteadyEncoderStep(model, inp["drop_extra"], inp["keep_all_outputs"]).cuda().eval()
    args_tuple = (inp["processed_signal"], inp["processed_signal_length"],
                  inp["cache_last_channel"], inp["cache_last_time"], inp["cache_last_channel_len"])

    with torch.inference_mode():
        eager = wrapper(*args_tuple)  # sanity: runs eagerly
        print("eager outputs:", [tuple(t.shape) for t in eager if torch.is_tensor(t)])
        traced = torch.jit.trace(wrapper, args_tuple, check_trace=False)

    ts_path = os.path.join(args.out, "encoder_steady_b1.ts")
    traced.save(ts_path)
    with open(os.path.join(args.out, "shapes.json"), "w") as f:
        json.dump(inp["shapes"], f, indent=2)
    print(f"saved {ts_path} + shapes.json — MECHANICAL export OK (NOT a fidelity claim; that's 0.2)")


if __name__ == "__main__":
    main()
