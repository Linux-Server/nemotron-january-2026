#!/usr/bin/env python3
"""Scratch probe for the EA NeMo prompted streaming checkpoint.

Run with the dedicated EA venv:

  /home/khkramer/src/nemotron-ea-nemo/.venv-ea/bin/python probe_ea_streaming.py \
    --model-path /path/to/nemotron-asr-streaming-multilingual-0.6b.nemo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel


def _jsonable(value):
    if hasattr(value, "to_container"):
        return value.to_container(resolve=True)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return repr(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-lang", default="en-US")
    parser.add_argument("--map-location", default="cpu")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    print(f"model_path={model_path}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")

    model = ASRModel.restore_from(str(model_path), map_location=args.map_location)
    print(f"class={model.__class__.__name__}")
    print(f"module={model.__class__.__module__}")
    print(f"has_joint={hasattr(model, 'joint')}")
    print(f"has_ctc_decoder={hasattr(model, 'ctc_decoder')}")
    print(f"has_aux_ctc_cfg={'aux_ctc' in model.cfg}")
    if "aux_ctc" in model.cfg:
        print("aux_ctc_cfg=" + json.dumps(_jsonable(model.cfg.aux_ctc), sort_keys=True)[:1000])

    defaults = model.cfg.get("model_defaults", {})
    prompt_dict = defaults.get("prompt_dictionary", {})
    print(f"prompt_dictionary_size={len(prompt_dict)}")
    for key in ("en-US", "en-GB", "auto"):
        print(f"prompt_dictionary[{key!r}]={prompt_dict.get(key)}")

    print(f"has_set_inference_prompt={hasattr(model, 'set_inference_prompt')}")
    if hasattr(model, "set_inference_prompt"):
        model.set_inference_prompt(args.target_lang)
        print(f"_inference_prompt_index={getattr(model, '_inference_prompt_index', None)}")

    decoding = getattr(model, "decoding", None)
    print(f"has_decoding={decoding is not None}")
    print(f"decoding_class={decoding.__class__.__name__ if decoding is not None else None}")
    print(f"has_set_strip_lang_tags={hasattr(decoding, 'set_strip_lang_tags') if decoding is not None else False}")
    if decoding is not None and hasattr(decoding, "set_strip_lang_tags"):
        decoding.strip_lang_tags = True
        decoding.set_strip_lang_tags(True)
        print(f"strip_lang_tags={getattr(decoding, 'strip_lang_tags', None)}")
        print(f"lang_tag_pattern={getattr(getattr(decoding, 'lang_tag_pattern', None), 'pattern', None)}")

    encoder_cfg = model.cfg.get("encoder", {})
    print(f"att_context_size={encoder_cfg.get('att_context_size')}")
    print(f"self_attention_model={encoder_cfg.get('self_attention_model')}")
    print(f"subsampling_factor={encoder_cfg.get('subsampling_factor')}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
