#!/usr/bin/env python3
"""Step 1 read-only probes: weights identity, enc_first AOTI parity, cache API.

This is diagnostic tooling only. It does not modify runtime/server sources or
production artifacts. Outputs are written under proj-2026-05-30-2202 by default.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import textwrap
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
STEADY_B = ROOT / "steady_b_artifacts"
DEFAULT_OUT = ROOT.parent.parent / "proj-2026-05-30-2202"


EVENT_INTERIM = 0
EVENT_FINAL = 1
EVENT_SUPPRESSED = 2


def resolve_shared(weights: dict[str, torch.Tensor], fqn: str) -> tuple[str | None, bool]:
    if fqn in weights:
        return fqn, False
    if fqn.startswith("encoder."):
        alt = "e." + fqn[len("encoder.") :]
        if alt in weights:
            return alt, True
    if fqn.startswith("e."):
        alt = "encoder." + fqn[len("e.") :]
        if alt in weights:
            return alt, True
    return None, False


def tensor_byte_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return False
    a = lhs.detach().cpu().contiguous().view(torch.uint8)
    b = rhs.detach().cpu().contiguous().view(torch.uint8)
    return bool(torch.equal(a, b))


def tensor_max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float | None:
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype:
        return None
    if lhs.numel() == 0:
        return 0.0
    if lhs.is_floating_point() or rhs.is_floating_point():
        return float((lhs.float() - rhs.float()).abs().max().item())
    return float((lhs.to(torch.int64) - rhs.to(torch.int64)).abs().max().item())


def shape_str(t: torch.Tensor) -> str:
    return "x".join(str(int(x)) for x in t.shape)


def load_finalize_ts_weights(path: Path) -> dict[str, torch.Tensor]:
    mod = torch.jit.load(str(path), map_location="cpu")
    try:
        weights = mod.weights
    except Exception:
        weights = mod._c.getattr("weights")
    out: dict[str, torch.Tensor] = {}
    for key, value in dict(weights).items():
        if not torch.is_tensor(value):
            raise TypeError(f"{path} key {key!r} is not a tensor")
        out[str(key)] = value.detach().cpu().contiguous()
    return out


def load_finalize_pt_weights(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} did not contain a dict")
    out: dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{path} key {key!r} is not a tensor")
        out[str(key)] = value.detach().cpu().contiguous()
    return out


def compare_named_tensors(
    source_name: str,
    source: dict[str, torch.Tensor],
    shared: dict[str, torch.Tensor],
    csv_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    direct = 0
    alias = 0
    byte_equal_count = 0
    max_abs_all = 0.0
    meta_mismatch: list[str] = []
    matched_shared_keys: set[str] = set()

    for fqn in sorted(source):
        tensor = source[fqn]
        shared_key, used_alias = resolve_shared(shared, fqn)
        if shared_key is None:
            missing.append(fqn)
            rows.append(
                {
                    "source": source_name,
                    "fqn": fqn,
                    "shared_key": "",
                    "match_kind": "missing",
                    "shape": shape_str(tensor),
                    "dtype": str(tensor.dtype),
                    "shared_shape": "",
                    "shared_dtype": "",
                    "max_abs_diff": "",
                    "byte_equal": "false",
                }
            )
            continue
        matched_shared_keys.add(shared_key)
        direct += int(not used_alias)
        alias += int(used_alias)
        other = shared[shared_key]
        max_abs = tensor_max_abs(tensor, other)
        byte_equal = tensor_byte_equal(tensor, other)
        if max_abs is None:
            meta_mismatch.append(fqn)
        else:
            max_abs_all = max(max_abs_all, max_abs)
        byte_equal_count += int(byte_equal)
        rows.append(
            {
                "source": source_name,
                "fqn": fqn,
                "shared_key": shared_key,
                "match_kind": "alias" if used_alias else "direct",
                "shape": shape_str(tensor),
                "dtype": str(tensor.dtype),
                "shared_shape": shape_str(other),
                "shared_dtype": str(other.dtype),
                "max_abs_diff": "" if max_abs is None else f"{max_abs:.9g}",
                "byte_equal": "true" if byte_equal else "false",
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "source": source_name,
        "total": len(source),
        "matched": len(source) - len(missing),
        "missing": missing,
        "direct_matches": direct,
        "alias_matches": alias,
        "byte_equal": byte_equal_count,
        "all_byte_equal": byte_equal_count == len(source) and not missing and not meta_mismatch,
        "max_abs": max_abs_all,
        "meta_mismatch": meta_mismatch,
        "shared_extras": sorted(set(shared) - matched_shared_keys),
        "csv": str(csv_path),
    }


def build_aoti_extract_extension():
    from torch.utils.cpp_extension import load_inline

    src = r'''
#include <torch/extension.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <pybind11/stl.h>

pybind11::dict extract_constants_cpu(const std::string& pkg, int64_t device_index) {
  torch::NoGradGuard ng;
  torch::inductor::AOTIModelPackageLoader loader(
      pkg, "model", false, 1, static_cast<c10::DeviceIndex>(device_index));
  auto constants = loader.get_runner()->extract_constants_map(false);
  pybind11::dict out;
  for (const auto& kv : constants) {
    at::Tensor t = kv.second.detach().to(at::kCPU).contiguous().clone();
    out[pybind11::str(kv.first)] = t;
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("extract_constants_cpu", &extract_constants_cpu);
}
'''
    return load_inline(
        name="probe_step1_aoti_extract",
        cpp_sources=[src],
        verbose=False,
        extra_cflags=["-O0"],
    )


def load_inline_aoti_constants(pkg_path: Path) -> dict[str, torch.Tensor]:
    mod = build_aoti_extract_extension()
    constants = mod.extract_constants_cpu(str(pkg_path), 0)
    return {str(k): v.detach().cpu().contiguous() for k, v in constants.items()}


def probe_weights(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    shared_pt = load_finalize_pt_weights(ART / "finalize_shared_weights.pt")
    shared_ts = load_finalize_ts_weights(ART / "finalize_shared_weights.ts")
    ts_vs_pt = compare_named_tensors(
        "finalize_shared_weights.ts",
        shared_ts,
        shared_pt,
        out_dir / "step1-finalize-ts-vs-pt.csv",
    )
    del shared_ts
    gc.collect()

    enc_first = torch.jit.load(str(ART / "enc_first.ts"), map_location="cpu")
    enc_first_params = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in enc_first.named_parameters()
        if name.startswith("e.") or name.startswith("encoder.")
    }
    enc_first_buffers = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in enc_first.named_buffers()
        if name.startswith("e.") or name.startswith("encoder.")
    }
    enc_first_all = {**enc_first_params, **enc_first_buffers}
    enc_first_param_cmp = compare_named_tensors(
        "enc_first.ts parameters",
        enc_first_params,
        shared_pt,
        out_dir / "step1-weights-enc-first-params.csv",
    )
    enc_first_all_cmp = compare_named_tensors(
        "enc_first.ts parameters+buffers",
        enc_first_all,
        shared_pt,
        out_dir / "step1-weights-enc-first-all.csv",
    )
    del enc_first, enc_first_params, enc_first_buffers, enc_first_all
    gc.collect()

    enc_steady_inline = load_inline_aoti_constants(ART / "enc_steady_aoti.pt2")
    enc_steady_cmp = compare_named_tensors(
        "enc_steady_aoti.pt2 inline constants",
        enc_steady_inline,
        shared_pt,
        out_dir / "step1-weights-enc-steady-inline.csv",
    )
    del enc_steady_inline
    gc.collect()

    enc_first_aoti_fqns = []
    enc_steady_stripped_fqns: dict[str, list[str]] = {}
    if torch.cuda.is_available():
        runner = torch._inductor.aoti_load_package(str(ART / "enc_first_aoti.pt2"))
        enc_first_aoti_fqns = sorted(str(x) for x in runner.loader.get_constant_fqns())
        del runner
        for b in (1, 2, 4):
            runner = torch._inductor.aoti_load_package(str(STEADY_B / f"enc_steady_aoti_b{b}.pt2"))
            enc_steady_stripped_fqns[str(b)] = sorted(str(x) for x in runner.loader.get_constant_fqns())
            del runner

    result = {
        "elapsed_seconds": time.time() - started,
        "torch": str(torch.__version__),
        "shared_pt_count": len(shared_pt),
        "finalize_ts_vs_pt": ts_vs_pt,
        "enc_first_params_vs_shared": enc_first_param_cmp,
        "enc_first_all_vs_shared": enc_first_all_cmp,
        "enc_steady_inline_vs_shared": enc_steady_cmp,
        "enc_first_aoti_constant_fqns": {
            "count": len(enc_first_aoti_fqns),
            "missing_from_shared": [
                f for f in enc_first_aoti_fqns if resolve_shared(shared_pt, f)[0] is None
            ],
        },
        "steady_b_stripped_constant_fqns": {
            b: {
                "count": len(fqns),
                "missing_from_shared": [f for f in fqns if resolve_shared(shared_pt, f)[0] is None],
            }
            for b, fqns in enc_steady_stripped_fqns.items()
        },
    }
    with (out_dir / "step1-weights-summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return result


class SharedWeights:
    def __init__(self, path: Path):
        self.path = path
        self.cpu: dict[str, torch.Tensor] | None = None
        self.cuda: dict[str, torch.Tensor] = {}

    def load_cpu(self) -> dict[str, torch.Tensor]:
        if self.cpu is None:
            obj = torch.load(self.path, map_location="cpu", weights_only=False)
            if not isinstance(obj, dict):
                raise TypeError(f"{self.path} did not contain a dict")
            self.cpu = {str(k): v.detach().cpu().contiguous() for k, v in obj.items()}
        return self.cpu

    def constants_for_runner(self, runner: Any) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        fqns = [str(x) for x in runner.loader.get_constant_fqns()]
        weights = self.load_cpu()
        cmap: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        direct = 0
        alias = 0
        for fqn in fqns:
            source_key, used_alias = resolve_shared(weights, fqn)
            if source_key is None:
                missing.append(fqn)
                continue
            if source_key not in self.cuda:
                self.cuda[source_key] = weights[source_key].cuda().contiguous()
            cmap[fqn] = self.cuda[source_key]
            direct += int(not used_alias)
            alias += int(used_alias)
        if missing:
            raise RuntimeError(f"missing {len(missing)} constants for {runner}: {missing[:5]}")
        return cmap, {
            "fqns": len(fqns),
            "matched": len(cmap),
            "direct": direct,
            "alias": alias,
        }


@dataclass
class Event:
    kind: int
    text: str
    tokens: list[int]
    collector_text: str
    collector_tokens: list[int]


def _tensor_attr(module: Any, name: str) -> torch.Tensor:
    try:
        value = getattr(module, name)
    except Exception:
        value = module._c.getattr(name)
    if not torch.is_tensor(value):
        raise TypeError(f"{name} is not a tensor")
    return value


def _scalar_i64(module: Any, name: str) -> int:
    return int(_tensor_attr(module, name).cpu().reshape(-1)[0].item())


def _i64_list(module: Any, name: str) -> list[int]:
    return [int(x) for x in _tensor_attr(module, name).cpu().to(torch.int64).reshape(-1).tolist()]


def _bytes_to_strings(bytes_tensor: torch.Tensor, offsets_tensor: torch.Tensor) -> list[str]:
    data = bytes(int(x) for x in bytes_tensor.cpu().to(torch.uint8).reshape(-1).tolist())
    offsets = [int(x) for x in offsets_tensor.cpu().to(torch.int64).reshape(-1).tolist()]
    return [data[offsets[i] : offsets[i + 1]].decode("utf-8") for i in range(len(offsets) - 1)]


def bundle_events(module: Any, prefix: str) -> list[Event]:
    kinds = _i64_list(module, f"{prefix}_event_kinds")
    tokens_flat = _i64_list(module, f"{prefix}_event_tokens")
    token_offsets = _i64_list(module, f"{prefix}_event_token_offsets")
    collector_flat = _i64_list(module, f"{prefix}_event_collector_tokens")
    collector_offsets = _i64_list(module, f"{prefix}_event_collector_token_offsets")
    texts = _bytes_to_strings(
        _tensor_attr(module, f"{prefix}_event_text_bytes"),
        _tensor_attr(module, f"{prefix}_event_text_offsets"),
    )
    collector_texts = _bytes_to_strings(
        _tensor_attr(module, f"{prefix}_event_collector_text_bytes"),
        _tensor_attr(module, f"{prefix}_event_collector_text_offsets"),
    )
    out: list[Event] = []
    for i, kind in enumerate(kinds):
        out.append(
            Event(
                kind=kind,
                text=texts[i],
                tokens=tokens_flat[token_offsets[i] : token_offsets[i + 1]],
                collector_text=collector_texts[i],
                collector_tokens=collector_flat[collector_offsets[i] : collector_offsets[i + 1]],
            )
        )
    return out


def event_payload(events: list[Event]) -> list[dict[str, Any]]:
    return [
        {
            "kind": e.kind,
            "text": e.text,
            "tokens": e.tokens,
            "collector_text": e.collector_text,
            "collector_tokens": e.collector_tokens,
        }
        for e in events
    ]


def append_only_delta_tokens(final_tokens: list[int], emitted_tokens: list[int]) -> list[int]:
    common = 0
    for emitted_token, final_token in zip(emitted_tokens, final_tokens):
        if emitted_token != final_token:
            break
        common += 1
    if common == len(emitted_tokens):
        return list(final_tokens[common:])
    if len(final_tokens) <= len(emitted_tokens):
        return []
    delta = list(final_tokens[len(emitted_tokens) :])
    max_overlap = min(len(emitted_tokens), len(delta))
    for overlap in range(max_overlap, 0, -1):
        if emitted_tokens[-overlap:] == delta[:overlap]:
            return delta[overlap:]
    return delta


def continuous_append_only_delta(final_text: str, emitted_text: str) -> str:
    final_words = final_text.split()
    emitted_words = emitted_text.split()
    common = 0
    for emitted, final in zip(emitted_words, final_words):
        if emitted != final:
            break
        common += 1
    if common == len(emitted_words):
        delta = final_words[common:]
    elif len(final_words) <= len(emitted_words):
        delta = []
    else:
        delta = final_words[len(emitted_words) :]
        max_overlap = min(len(emitted_words), len(delta))
        for overlap in range(max_overlap, 0, -1):
            if emitted_words[-overlap:] == delta[:overlap]:
                delta = delta[overlap:]
                break
    return " ".join(delta)


class BundleMelParityRunner:
    def __init__(self, rt: Any, bundle: Any, enc_first: Any):
        from ref_decode import ref_greedy_range

        self.rt = rt
        self.bundle = bundle
        self.enc_first = enc_first
        self.ref_greedy_range = ref_greedy_range
        self.events: list[Event] = []
        self.collector_tokens: list[int] = []
        self.continuous_emitted_text = ""

    @torch.inference_mode()
    def run_first(self, chunk: torch.Tensor, state: Any) -> tuple[torch.Tensor, ...]:
        device = self.rt.device
        length = torch.tensor([chunk.shape[-1]], device=device, dtype=torch.long)
        out = self.enc_first(
            chunk.contiguous(),
            length.contiguous(),
            state.cache_last_channel.contiguous(),
            state.cache_last_time.contiguous(),
            state.cache_last_channel_len.contiguous(),
        )
        return tuple(out)

    @torch.inference_mode()
    def run_steady(self, chunk: torch.Tensor, state: Any) -> tuple[torch.Tensor, ...]:
        device = self.rt.device
        length = torch.tensor([chunk.shape[-1]], device=device, dtype=torch.long)
        out = self.rt.enc_steady_aoti(
            chunk.contiguous(),
            length.contiguous(),
            state.cache_last_channel.contiguous(),
            state.cache_last_time.contiguous(),
            state.cache_last_channel_len.contiguous(),
        )
        return tuple(out)

    def record_event(
        self,
        *,
        kind: int,
        text: str,
        tokens: list[int],
        collector_text: str,
        collector_tokens: list[int],
    ) -> None:
        self.events.append(Event(kind, text, list(tokens), collector_text, list(collector_tokens)))

    @torch.inference_mode()
    def run_row(self, utt: int) -> dict[str, Any]:
        session = self.rt.new_session(f"step1-parity-utt{utt}")
        self.events = []
        self.collector_tokens = []
        self.continuous_emitted_text = ""
        prefix = f"utt{utt}"
        device = self.rt.device
        num_steady = _scalar_i64(self.bundle, f"{prefix}_num_steady")

        first_output_diffs: list[dict[str, Any]] = []
        for chunk_idx in range(num_steady):
            new_mel = _tensor_attr(self.bundle, f"{prefix}_chunk{chunk_idx}_new_mel").to(device).contiguous()
            old_text = session.current_text
            first = session.emitted_frames == 0
            if first:
                chunk_mel = new_mel
                out = self.run_first(chunk_mel, session)
            else:
                chunk_mel = torch.cat((session.mel_frame_ring, new_mel), dim=-1).contiguous()
                out = self.run_steady(chunk_mel, session)

            if first:
                eager = _tensor_attr(self.bundle, f"{prefix}_chunk{chunk_idx}_first_eager_enc_out").to(device)
                max_abs = float((out[0].float() - eager.float()).abs().max().item()) if out[0].numel() else 0.0
                first_output_diffs.append(
                    {
                        "output": "enc_out",
                        "max_abs_vs_bundle_eager": max_abs,
                        "byte_equal_vs_bundle_eager": bool(torch.equal(out[0], eager)),
                    }
                )

            tokens, decoder_state, pred_out = self.ref_greedy_range(
                self.rt.decoder,
                self.rt.joint,
                out[0].transpose(1, 2).contiguous(),
                0,
                int(out[1][0]),
                session.decoder_state,
                session.pred_out_stream,
            )
            session.hyp_tokens.extend(tokens)
            session.decoder_state = decoder_state
            session.pred_out_stream = pred_out
            session.cache_last_channel = out[2].clone()
            session.cache_last_time = out[3].clone()
            session.cache_last_channel_len = out[4].clone()
            self.rt._update_mel_frame_ring(session, new_mel)
            session.emitted_frames += self.rt.geometry.shift_frames
            session.current_text = self.rt.tokenizer.ids_to_text(session.hyp_tokens)
            if session.current_text != old_text:
                self.record_event(
                    kind=EVENT_INTERIM,
                    text=session.current_text,
                    tokens=list(session.hyp_tokens),
                    collector_text=self.continuous_emitted_text,
                    collector_tokens=list(self.collector_tokens),
                )

        steady_tokens = list(session.hyp_tokens)
        final_t = _scalar_i64(self.bundle, f"{prefix}_final_T")
        if final_t > 0:
            final_chunk = _tensor_attr(self.bundle, f"{prefix}_final_chunk_mel").to(device).contiguous()
            final_drop = _scalar_i64(self.bundle, f"{prefix}_final_drop_extra")
            final_len = torch.tensor([final_chunk.shape[-1]], device=device, dtype=torch.long)
            enc_out, enc_len, clc, clt, clcl = self.rt.encoder.cache_aware_stream_step(
                processed_signal=final_chunk,
                processed_signal_length=final_len,
                cache_last_channel=session.cache_last_channel.contiguous(),
                cache_last_time=session.cache_last_time.contiguous(),
                cache_last_channel_len=session.cache_last_channel_len.contiguous(),
                keep_all_outputs=True,
                drop_extra_pre_encoded=int(final_drop),
            )
            tokens, decoder_state, pred_out = self.ref_greedy_range(
                self.rt.decoder,
                self.rt.joint,
                enc_out.transpose(1, 2).contiguous(),
                0,
                int(enc_len[0]),
                session.decoder_state,
                session.pred_out_stream,
            )
            final_tokens = list(session.hyp_tokens) + list(tokens)
        else:
            final_tokens = list(session.hyp_tokens)

        final_text = self.rt.tokenizer.ids_to_text(final_tokens)
        delta_text = continuous_append_only_delta(final_text, self.continuous_emitted_text)
        token_delta = append_only_delta_tokens(final_tokens, self.collector_tokens)
        if delta_text:
            self.continuous_emitted_text = (self.continuous_emitted_text + " " + delta_text).strip()
            collector_tokens = self.collector_tokens + token_delta
            kind = EVENT_FINAL
            text = delta_text
        else:
            collector_tokens = list(self.collector_tokens)
            kind = EVENT_SUPPRESSED
            text = ""
        self.record_event(
            kind=kind,
            text=text,
            tokens=token_delta,
            collector_text=self.continuous_emitted_text,
            collector_tokens=collector_tokens,
        )
        self.collector_tokens = list(collector_tokens)
        return {
            "steady_tokens": steady_tokens,
            "final_tokens": final_tokens,
            "events": event_payload(self.events),
            "first_output_diffs": first_output_diffs,
        }


def first_diff(lhs: list[Any], rhs: list[Any]) -> int | None:
    n = min(len(lhs), len(rhs))
    for i in range(n):
        if lhs[i] != rhs[i]:
            return i
    if len(lhs) != len(rhs):
        return n
    return None


def make_aoti_first(shared: SharedWeights) -> tuple[Any, dict[str, int]]:
    runner = torch._inductor.aoti_load_package(str(ART / "enc_first_aoti.pt2"))
    cmap, stats = shared.constants_for_runner(runner)
    runner.loader.load_constants(cmap, False, False, True)
    return runner, stats


def probe_parity(out_dir: Path, rows: int, b2_rows: int, ts_sanity_rows: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    started = time.time()

    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    from export_session_bundle import RecordingContinuousFinalizeRef

    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b",
        map_location="cpu",
    ).cuda().eval()
    try:
        model.preprocessor.featurizer.dither = 0.0
    except Exception:
        pass
    model.encoder.set_default_att_context_size([70, 1])
    model.change_decoding_strategy(
        decoding_cfg=OmegaConf.create(
            {
                "strategy": "greedy_batch",
                "greedy": {
                    "max_symbols": 10,
                    "loop_labels": True,
                    "use_cuda_graph_decoder": False,
                },
            }
        )
    )

    rt = RecordingContinuousFinalizeRef(model, artifacts_dir=str(ART), warm_encoder=False)
    bundle = torch.jit.load(str(ART / "session_bundle.ts"), map_location="cpu")
    total_rows = _scalar_i64(bundle, "num_utts")
    rows = min(rows, total_rows)
    b2_rows = min(b2_rows, rows)
    ts_sanity_rows = min(ts_sanity_rows, rows)

    shared = SharedWeights(ART / "finalize_shared_weights.pt")
    aoti_first, aoti_stats = make_aoti_first(shared)
    ts_first = rt.enc_first

    def compare_runner(label: str, enc_first: Any, count: int) -> dict[str, Any]:
        runner = BundleMelParityRunner(rt, bundle, enc_first)
        token_divs: list[dict[str, Any]] = []
        event_divs: list[dict[str, Any]] = []
        steady_divs = 0
        max_first_eager_abs = 0.0
        first_eager_byte_equal = 0
        first_eager_checks = 0
        row_records: list[dict[str, Any]] = []
        t0 = time.time()
        for utt in range(count):
            got = runner.run_row(utt)
            gold_final = _i64_list(bundle, f"utt{utt}_gold_tokens")
            gold_steady = _i64_list(bundle, f"utt{utt}_steady_tokens")
            gold_events = event_payload(bundle_events(bundle, f"utt{utt}"))
            sample_id = _bytes_to_strings(
                _tensor_attr(bundle, f"utt{utt}_sample_id_bytes"),
                _tensor_attr(bundle, f"utt{utt}_sample_id_offsets"),
            )[0] if hasattr(bundle, f"utt{utt}_sample_id_bytes") else str(utt)

            if got["steady_tokens"] != gold_steady:
                steady_divs += 1
            token_idx = first_diff(got["final_tokens"], gold_final)
            event_idx = first_diff(got["events"], gold_events)
            if token_idx is not None:
                token_divs.append(
                    {
                        "utt": utt,
                        "sample_id": sample_id,
                        "first_diff": token_idx,
                        "got": None if token_idx >= len(got["final_tokens"]) else got["final_tokens"][token_idx],
                        "gold": None if token_idx >= len(gold_final) else gold_final[token_idx],
                        "got_len": len(got["final_tokens"]),
                        "gold_len": len(gold_final),
                    }
                )
            if event_idx is not None:
                event_divs.append(
                    {
                        "utt": utt,
                        "sample_id": sample_id,
                        "first_diff": event_idx,
                        "got": None if event_idx >= len(got["events"]) else got["events"][event_idx],
                        "gold": None if event_idx >= len(gold_events) else gold_events[event_idx],
                        "got_len": len(got["events"]),
                        "gold_len": len(gold_events),
                    }
                )
            for diff in got["first_output_diffs"]:
                first_eager_checks += 1
                max_first_eager_abs = max(max_first_eager_abs, float(diff["max_abs_vs_bundle_eager"]))
                first_eager_byte_equal += int(bool(diff["byte_equal_vs_bundle_eager"]))
            row_records.append(
                {
                    "utt": utt,
                    "sample_id": sample_id,
                    "token_match": token_idx is None,
                    "event_match": event_idx is None,
                    "steady_token_match": got["steady_tokens"] == gold_steady,
                    "final_token_len": len(got["final_tokens"]),
                    "event_len": len(got["events"]),
                }
            )
            if (utt + 1) % 50 == 0 or utt + 1 == count:
                print(
                    f"{label}: {utt + 1}/{count} token_div={len(token_divs)} "
                    f"event_div={len(event_divs)} elapsed={(time.time() - t0):.1f}s",
                    flush=True,
                )
        return {
            "label": label,
            "rows": count,
            "steady_token_divergences": steady_divs,
            "token_divergences": len(token_divs),
            "event_divergences": len(event_divs),
            "token_divergence_details": token_divs[:50],
            "event_divergence_details": event_divs[:50],
            "first_output_max_abs_vs_bundle_eager": max_first_eager_abs,
            "first_output_byte_equal_vs_bundle_eager": first_eager_byte_equal,
            "first_output_checks": first_eager_checks,
            "row_records": row_records,
            "elapsed_seconds": time.time() - t0,
        }

    sanity = compare_runner("ts_sanity_vs_bundle", ts_first, ts_sanity_rows) if ts_sanity_rows else None
    b2 = compare_runner("aoti_first_b2_t1_rows", aoti_first, b2_rows)
    corpus = compare_runner("aoti_first_corpus", aoti_first, rows)

    result = {
        "elapsed_seconds": time.time() - started,
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "bundle_rows": total_rows,
        "aoti_first_constants": aoti_stats,
        "ts_sanity": sanity,
        "b2_t1": b2,
        "corpus": corpus,
        "go_enc_first_unify": (
            b2["token_divergences"] == 0
            and b2["event_divergences"] == 0
            and corpus["token_divergences"] == 0
            and corpus["event_divergences"] == 0
        ),
    }
    detail = dict(result)
    for section in ("b2_t1", "corpus", "ts_sanity"):
        if detail.get(section):
            # Keep the main JSON compact enough to inspect while preserving row CSV below.
            detail[section] = {k: v for k, v in detail[section].items() if k != "row_records"}
    with (out_dir / "step1-enc-first-parity.json").open("w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, sort_keys=True)
        f.write("\n")
    for name, section in (("ts_sanity", sanity), ("b2_t1", b2), ("corpus", corpus)):
        if not section:
            continue
        with (out_dir / f"step1-parity-{name}-rows.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(section["row_records"][0].keys()))
            writer.writeheader()
            writer.writerows(section["row_records"])
    return result


def source_url_fetch(url: str, dst: Path) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=20) as response:
            dst.write_bytes(response.read())
        return True
    except Exception as exc:
        print(f"source fetch failed for {url}: {exc}", file=sys.stderr)
        return False


def inspect_cache_api(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch_root = Path(torch.__file__).resolve().parent
    header = torch_root / "include/torch/csrc/inductor/aoti_package/model_package_loader.h"
    header_text = header.read_text(encoding="utf-8")
    ctor_lines = [
        line.strip()
        for line in header_text.splitlines()
        if "AOTIModelPackageLoader(" in line or "temp_dir_" in line
    ]
    public_preextracted_ctor = bool(
        re.search(r"AOTIModelPackageLoader\s*\([^;{}]*(?:dir|directory|extracted)", header_text)
    )
    temp_private = "private:" in header_text and "std::string temp_dir_;" in header_text

    src_path = out_dir / "step1-torch-v2.8.0-model_package_loader.cpp"
    url = "https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/csrc/inductor/aoti_package/model_package_loader.cpp"
    fetched = source_url_fetch(url, src_path)
    src_text = src_path.read_text(encoding="utf-8") if fetched else ""
    create_temp_dir_mentions = [
        line.strip()
        for line in src_text.splitlines()
        if "create_temp_dir" in line or "/tmp/XXXXXX" in line or "TMPDIR" in line or "mkdtemp" in line
    ]

    strings_hits: list[str] = []
    libtorch_cpu = torch_root / "lib/libtorch_cpu.so"
    if libtorch_cpu.exists():
        try:
            proc = subprocess.run(
                ["strings", str(libtorch_cpu)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for line in proc.stdout.splitlines():
                if (
                    "/tmp/XXXXXX" in line
                    or "aotinductor" in line
                    or "model_package_loader.cpp" in line
                ):
                    strings_hits.append(line)
        except Exception as exc:
            strings_hits.append(f"strings failed: {exc}")

    package_sizes = {}
    for path in [
        ART / "enc_first_aoti.pt2",
        ART / "enc_steady_aoti.pt2",
        ART / "finalize_shared_weights.ts",
        ART / "finalize_shared_weights.pt",
    ]:
        if path.exists():
            package_sizes[str(path.relative_to(ROOT))] = path.stat().st_size
    for path in sorted((ART / "stripped_finalize_buckets").glob("*.pt2")):
        package_sizes[str(path.relative_to(ROOT))] = path.stat().st_size
    for path in sorted(STEADY_B.glob("enc_steady_aoti_b*.pt2")):
        if ".full." not in path.name:
            package_sizes[str(path.relative_to(ROOT))] = path.stat().st_size

    stripped_finalize_total = sum(
        path.stat().st_size for path in (ART / "stripped_finalize_buckets").glob("*.pt2")
    )
    stripped_steady_b_total = sum(
        path.stat().st_size for path in STEADY_B.glob("enc_steady_aoti_b*.pt2") if ".full." not in path.name
    )
    result = {
        "torch": str(torch.__version__),
        "torch_root": str(torch_root),
        "header": str(header),
        "header_constructor_lines": ctor_lines,
        "public_preextracted_dir_ctor": public_preextracted_ctor,
        "temp_dir_private": temp_private,
        "source_url": url,
        "source_fetched": fetched,
        "source_create_temp_dir_mentions": create_temp_dir_mentions,
        "installed_libtorch_cpu_strings_hits": strings_hits[:50],
        "package_sizes_bytes": package_sizes,
        "post_unify_residual_bytes": {
            "shared_weights_ts": package_sizes.get("artifacts/finalize_shared_weights.ts", 0),
            "enc_first_stripped_aoti": package_sizes.get("artifacts/enc_first_aoti.pt2", 0),
            "steady_b_stripped_total": stripped_steady_b_total,
            "finalize_buckets_stripped_total": stripped_finalize_total,
            "small_stripped_aoti_total": (
                package_sizes.get("artifacts/enc_first_aoti.pt2", 0)
                + stripped_steady_b_total
                + stripped_finalize_total
            ),
        },
        "cache_speed_go": False,
    }
    with (out_dir / "step1-cache-api.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return result


def cleanup_tmp_aoti() -> list[str]:
    user_uid = os.getuid()
    removed: list[str] = []
    root = Path("/tmp")
    pat = re.compile(r"^[A-Za-z0-9]{6}$")
    for child in root.iterdir():
        try:
            st = child.stat()
        except FileNotFoundError:
            continue
        if not child.is_dir() or st.st_uid != user_uid or not pat.match(child.name):
            continue
        if not (child / "data" / "aotinductor").exists():
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed.append(str(child))
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["weights", "parity", "cache", "all"], default="all")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parity-rows", type=int, default=1000)
    parser.add_argument("--b2-rows", type=int, default=4)
    parser.add_argument("--ts-sanity-rows", type=int, default=4)
    parser.add_argument("--skip-tmp-cleanup", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {}
    if args.mode in ("weights", "all"):
        print("=== Step1 weights probe ===", flush=True)
        result["weights"] = probe_weights(args.out_dir)
    if args.mode in ("parity", "all"):
        print("=== Step1 enc_first parity probe ===", flush=True)
        result["parity"] = probe_parity(args.out_dir, args.parity_rows, args.b2_rows, args.ts_sanity_rows)
    if args.mode in ("cache", "all"):
        print("=== Step1 cache API probe ===", flush=True)
        result["cache"] = inspect_cache_api(args.out_dir)
    if not args.skip_tmp_cleanup:
        removed = cleanup_tmp_aoti()
        result["tmp_cleanup_removed"] = removed
        if removed:
            print(f"removed tmp AOTI dirs: {removed}", flush=True)
    with (args.out_dir / "step1-probe-run-summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
