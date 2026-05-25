#!/usr/bin/env python3
"""Export finalize-geometry encoder attempts + eager fixtures.

The C++ finalize port needs the keep_all_outputs=True encoder geometry, whose T
is dynamic at debounce time:

* first finalize:      chunk_mel T = remaining, drop_extra=0
* continuation final: chunk_mel T = pre_encode_cache_size + remaining, drop_extra=2

This script traces both scalar drop geometries to TorchScript when possible and
always writes an eager fixture bundle containing inputs, eager reference outputs,
carried decoder state, and gold final token sequences.

Run:
  HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python export_finalize_encoder.py --out ./artifacts
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any

import torch

from finalize_ref import (
    ContinuousFinalizeRef,
    FinalizeInputs,
    load_benchmark_dataset,
    load_model,
    load_wav,
)


class FinalizeStep(torch.nn.Module):
    def __init__(self, encoder, drop_extra: int):
        super().__init__()
        self.encoder = encoder
        self.drop_extra = int(drop_extra)

    def forward(self, chunk, length, cache_last_channel, cache_last_time, cache_last_channel_len):
        return self.encoder.cache_aware_stream_step(
            processed_signal=chunk,
            processed_signal_length=length,
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
            keep_all_outputs=True,
            drop_extra_pre_encoded=self.drop_extra,
        )


def _cpu_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, list):
        return [_cpu_tree(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_cpu_tree(item) for item in obj)
    if isinstance(obj, dict):
        return {key: _cpu_tree(value) for key, value in obj.items()}
    return obj


def _input_tuple(inputs: FinalizeInputs):
    return (
        inputs.chunk_mel,
        inputs.chunk_len,
        inputs.cache_last_channel,
        inputs.cache_last_time,
        inputs.cache_last_channel_len,
    )


def _tensor_outputs_equal(lhs, rhs) -> tuple[bool, float]:
    ok = True
    max_diff = 0.0
    for a, b in zip(lhs, rhs):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            ok = ok and (a == b)
            continue
        if not torch.equal(a, b):
            ok = False
            max_diff = max(max_diff, float((a.float() - b.float()).abs().max().item()))
    return ok, max_diff


def _make_row(rt: ContinuousFinalizeRef, ds, sample_index: int, kind: str) -> dict[str, Any]:
    ex = ds[sample_index]
    wav = load_wav(ex)
    session = rt.new_session(f"fixture-{kind}-{sample_index}")

    if kind == "continuation":
        rt.append_audio(session, wav)
    elif kind == "first":
        session.pending_audio = wav.copy()
        session.total_audio_samples = int(wav.shape[0])
    else:
        raise ValueError(f"unknown fixture kind {kind!r}")

    fork = rt.build_continuous_finalize_fork(session)
    pre_final_tokens = list(fork.hyp_tokens)
    pre_final_decoder_state = _cpu_tree(fork.decoder_state)
    pre_final_pred_out = _cpu_tree(fork.pred_out_stream)
    flush = rt.flush_finalize_fork(fork)
    inputs = flush["inputs"]
    eager_outputs = flush["encoder_outputs"]
    if inputs is None or eager_outputs is None:
        raise RuntimeError(f"empty finalize fixture for sample_index={sample_index}")

    full_tokens = rt.full_greedy_tokens(wav)
    final_tokens = list(flush["final_tokens"])
    new_tokens = final_tokens[len(pre_final_tokens) :]
    return {
        "kind": kind,
        "sample_index": sample_index,
        "sample_id": ex.get("sample_id"),
        "transcription": ex.get("transcription"),
        "drop_extra": int(inputs.drop_extra),
        "chunk_T": int(inputs.chunk_mel.shape[-1]),
        "remaining_frames": int(inputs.remaining_frames),
        "padded_total_samples": int(inputs.padded_total_samples),
        "chunk_mel": inputs.chunk_mel.detach().cpu().clone(),
        "chunk_len": inputs.chunk_len.detach().cpu().clone(),
        "cache_last_channel": inputs.cache_last_channel.detach().cpu().clone(),
        "cache_last_time": inputs.cache_last_time.detach().cpu().clone(),
        "cache_last_channel_len": inputs.cache_last_channel_len.detach().cpu().clone(),
        "eager_outputs": [item.detach().cpu().clone() for item in eager_outputs],
        "pre_final_tokens": torch.tensor(pre_final_tokens, dtype=torch.int64),
        "gold_new_tokens": torch.tensor(new_tokens, dtype=torch.int64),
        "gold_final_tokens": torch.tensor(final_tokens, dtype=torch.int64),
        "full_greedy_tokens": torch.tensor(full_tokens, dtype=torch.int64),
        "pre_final_decoder_state": pre_final_decoder_state,
        "pre_final_pred_out": pre_final_pred_out,
        "steady_text": rt.text(pre_final_tokens),
        "final_text": rt.text(final_tokens),
        "full_text": rt.text(full_tokens),
    }


def _row_inputs_cuda(row: dict[str, Any], device):
    return (
        row["chunk_mel"].to(device),
        row["chunk_len"].to(device),
        row["cache_last_channel"].to(device),
        row["cache_last_time"].to(device),
        row["cache_last_channel_len"].to(device),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./artifacts")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("loading model + building finalize fixtures...")
    model = load_model()
    rt = ContinuousFinalizeRef(model)
    ds = load_benchmark_dataset()
    device = rt.device

    rows = [
        _make_row(rt, ds, 5, "first"),
        _make_row(rt, ds, 4, "continuation"),
        _make_row(rt, ds, 9, "continuation"),
        _make_row(rt, ds, 2, "continuation"),
    ]

    fixture_path = os.path.join(args.out, "finalize_fixture.pt")
    meta = {
        "model": "nvidia/nemotron-speech-streaming-en-0.6b",
        "att_context_size": [70, 1],
        "keep_all_outputs": True,
        "geometry": asdict(rt.geometry),
        "notes": (
            "Rows contain finalize encoder inputs, eager encoder outputs, "
            "carried decoder state, and gold final/new token sequences."
        ),
    }
    torch.save({"meta": meta, "rows": rows}, fixture_path)
    print(f"saved fixture: {fixture_path}")
    for row in rows:
        print(
            f"  row kind={row['kind']} idx={row['sample_index']} "
            f"drop={row['drop_extra']} T={row['chunk_T']} "
            f"pre/final/full tok={len(row['pre_final_tokens'])}/"
            f"{len(row['gold_final_tokens'])}/{len(row['full_greedy_tokens'])}"
        )

    report_lines: list[str] = []
    all_trace_byte_exact = True
    for drop_extra in (0, 2):
        drop_rows = [row for row in rows if row["drop_extra"] == drop_extra]
        if not drop_rows:
            continue
        example = drop_rows[0]
        module = FinalizeStep(rt.encoder, drop_extra).cuda().eval()
        ts_name = f"enc_finalize_drop{drop_extra}.ts"
        ts_path = os.path.join(args.out, ts_name)
        try:
            with torch.inference_mode():
                traced = torch.jit.trace(
                    module,
                    _row_inputs_cuda(example, device),
                    check_trace=False,
                )
            traced.save(ts_path)
            print(f"saved trace: {ts_path}")
        except Exception as exc:
            all_trace_byte_exact = False
            line = (
                f"drop={drop_extra}: torch.jit.trace FAILED: "
                f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            print(line)
            report_lines.append(line)
            continue

        drop_ok = True
        drop_max_diff = 0.0
        for row in drop_rows:
            cuda_inputs = _row_inputs_cuda(row, device)
            eager = [item.to(device) for item in row["eager_outputs"]]
            try:
                with torch.inference_mode():
                    traced_out = traced(*cuda_inputs)
                ok, max_diff = _tensor_outputs_equal(eager, traced_out)
            except Exception as exc:
                ok = False
                max_diff = float("inf")
                report_lines.append(
                    f"drop={drop_extra} T={row['chunk_T']}: traced call FAILED: "
                    f"{type(exc).__name__}: {str(exc)[:500]}"
                )
            drop_ok = drop_ok and ok
            drop_max_diff = max(drop_max_diff, max_diff)
            print(
                f"  trace check drop={drop_extra} T={row['chunk_T']}: "
                f"byte_exact={ok} max_abs_diff={max_diff:.3e}"
            )

        all_trace_byte_exact = all_trace_byte_exact and drop_ok
        if drop_ok:
            report_lines.append(
                f"drop={drop_extra}: TorchScript trace byte-exact for "
                f"T={[row['chunk_T'] for row in drop_rows]}"
            )
        else:
            report_lines.append(
                f"drop={drop_extra}: TorchScript trace NOT byte-exact/dynamic-faithful "
                f"for T={[row['chunk_T'] for row in drop_rows]} "
                f"(max_abs_diff={drop_max_diff:.3e}); C++ port should use "
                "torch.export/AOTI or native encoder step for byte-exact finalize."
            )

    report_path = os.path.join(args.out, "finalize_export_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"saved report: {report_path}")
    print(
        "=== FINALIZE_ENCODER_EXPORT "
        f"{'PASS byte-exact trace' if all_trace_byte_exact else 'PASS fixture; trace needs torch.export/AOTI'} "
        "==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
