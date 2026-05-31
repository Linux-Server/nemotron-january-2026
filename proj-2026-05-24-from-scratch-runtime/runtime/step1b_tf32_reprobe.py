#!/usr/bin/env python3
"""Step 1b enc_first AOTI precision-variant parity reprobe.

This reuses the Step-1 session_bundle oracle machinery, but materializes two
TorchScript first-chunk references:
  - cudnn.allow_tf32=True, using the shipped bundle token/event oracle
  - cudnn.allow_tf32=False, generated from the same bundle inputs

Candidate runs swap only first-chunk execution to a selected AOTI package bound
to finalize_shared_weights.pt.  Non-first chunks and finalization follow the
same Step-1 probe path.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from probe_step1 import (
    ART,
    DEFAULT_OUT,
    SharedWeights,
    _bytes_to_strings,
    _i64_list,
    _scalar_i64,
    _tensor_attr,
    append_only_delta_tokens,
    bundle_events,
    continuous_append_only_delta,
    event_payload,
    first_diff,
)


KNOWN_DIVERGENT_UTTS = (198, 759, 811, 829)


def configure_precision(*, cudnn_tf32: bool) -> None:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = bool(cudnn_tf32)


def resolve_shared(weights: dict[str, torch.Tensor], fqn: str) -> str | None:
    if fqn in weights:
        return fqn
    if fqn.startswith("encoder."):
        alt = "e." + fqn[len("encoder.") :]
        if alt in weights:
            return alt
    if fqn.startswith("e."):
        alt = "encoder." + fqn[len("e.") :]
        if alt in weights:
            return alt
    return None


def load_aoti_first(shared: SharedWeights, package_path: Path) -> tuple[Any, dict[str, Any]]:
    runner = torch._inductor.aoti_load_package(str(package_path))
    cmap, stats = shared.constants_for_runner(runner)
    runner.loader.load_constants(cmap, False, False, True)
    stats = dict(stats)
    stats["package_path"] = str(package_path)
    return runner, stats


def sample_id_for(bundle: Any, utt: int) -> str:
    prefix = f"utt{utt}"
    try:
        return _bytes_to_strings(
            _tensor_attr(bundle, f"{prefix}_sample_id_bytes"),
            _tensor_attr(bundle, f"{prefix}_sample_id_offsets"),
        )[0]
    except Exception:
        return str(utt)


def bundle_reference_row(bundle: Any, utt: int) -> dict[str, Any]:
    prefix = f"utt{utt}"
    return {
        "final_tokens": _i64_list(bundle, f"{prefix}_gold_tokens"),
        "events": event_payload(bundle_events(bundle, prefix)),
        "sample_id": sample_id_for(bundle, utt),
    }


def tensor_max_abs(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.shape != rhs.shape:
        return float("inf")
    if lhs.numel() == 0:
        return 0.0
    if lhs.is_floating_point() or rhs.is_floating_point():
        return float((lhs.float() - rhs.float()).abs().max().item())
    return float((lhs.to(torch.int64) - rhs.to(torch.int64)).abs().max().item())


class BundleFirstRunner:
    def __init__(self, rt: Any, bundle: Any, enc_first: Any):
        from ref_decode import ref_greedy_range

        self.rt = rt
        self.bundle = bundle
        self.enc_first = enc_first
        self.ref_greedy_range = ref_greedy_range
        self.events: list[Any] = []
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
        from probe_step1 import Event

        self.events.append(
            Event(
                kind=kind,
                text=text,
                tokens=list(tokens),
                collector_text=collector_text,
                collector_tokens=list(collector_tokens),
            )
        )

    @torch.inference_mode()
    def run_row(self, utt: int) -> dict[str, Any]:
        session = self.rt.new_session(f"step1b-utt{utt}")
        self.events = []
        self.collector_tokens = []
        self.continuous_emitted_text = ""
        prefix = f"utt{utt}"
        device = self.rt.device
        num_steady = _scalar_i64(self.bundle, f"{prefix}_num_steady")
        first_outputs: dict[str, torch.Tensor] | None = None

        for chunk_idx in range(num_steady):
            new_mel = _tensor_attr(self.bundle, f"{prefix}_chunk{chunk_idx}_new_mel").to(device).contiguous()
            old_text = session.current_text
            first = session.emitted_frames == 0
            if first:
                chunk_mel = new_mel
                out = self.run_first(chunk_mel, session)
                first_outputs = {
                    "enc_out": out[0].detach().cpu().contiguous(),
                    "cache_t": out[3].detach().cpu().contiguous(),
                }
            else:
                chunk_mel = torch.cat((session.mel_frame_ring, new_mel), dim=-1).contiguous()
                out = self.run_steady(chunk_mel, session)

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
                from probe_step1 import EVENT_INTERIM

                self.record_event(
                    kind=EVENT_INTERIM,
                    text=session.current_text,
                    tokens=list(session.hyp_tokens),
                    collector_text=self.continuous_emitted_text,
                    collector_tokens=list(self.collector_tokens),
                )

        final_t = _scalar_i64(self.bundle, f"{prefix}_final_T")
        if final_t > 0:
            final_chunk = _tensor_attr(self.bundle, f"{prefix}_final_chunk_mel").to(device).contiguous()
            final_drop = _scalar_i64(self.bundle, f"{prefix}_final_drop_extra")
            final_len = torch.tensor([final_chunk.shape[-1]], device=device, dtype=torch.long)
            enc_out, enc_len, _clc, _clt, _clcl = self.rt.encoder.cache_aware_stream_step(
                processed_signal=final_chunk,
                processed_signal_length=final_len,
                cache_last_channel=session.cache_last_channel.contiguous(),
                cache_last_time=session.cache_last_time.contiguous(),
                cache_last_channel_len=session.cache_last_channel_len.contiguous(),
                keep_all_outputs=True,
                drop_extra_pre_encoded=int(final_drop),
            )
            tokens, _decoder_state, _pred_out = self.ref_greedy_range(
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
            from probe_step1 import EVENT_FINAL

            kind = EVENT_FINAL
            text = delta_text
        else:
            collector_tokens = list(self.collector_tokens)
            from probe_step1 import EVENT_SUPPRESSED

            kind = EVENT_SUPPRESSED
            text = ""
        self.record_event(
            kind=kind,
            text=text,
            tokens=token_delta,
            collector_text=self.continuous_emitted_text,
            collector_tokens=collector_tokens,
        )
        if first_outputs is None:
            raise RuntimeError(f"row {utt} had no first chunk")
        return {
            "final_tokens": final_tokens,
            "events": event_payload(self.events),
            "first_outputs": first_outputs,
        }


def collect_ts_true_first_outputs(rt: Any, bundle: Any, ts_first: Any, rows: int) -> dict[int, dict[str, torch.Tensor]]:
    configure_precision(cudnn_tf32=True)
    runner = BundleFirstRunner(rt, bundle, ts_first)
    out: dict[int, dict[str, torch.Tensor]] = {}
    t0 = time.time()
    for utt in range(rows):
        session = rt.new_session(f"step1b-ts-true-first-utt{utt}")
        new_mel = _tensor_attr(bundle, f"utt{utt}_chunk0_new_mel").to(rt.device).contiguous()
        first = runner.run_first(new_mel, session)
        out[utt] = {
            "enc_out": first[0].detach().cpu().contiguous(),
            "cache_t": first[3].detach().cpu().contiguous(),
        }
        if (utt + 1) % 100 == 0 or utt + 1 == rows:
            print(f"ts_tf32_first_outputs: {utt + 1}/{rows} elapsed={time.time() - t0:.1f}s", flush=True)
    return out


def run_reference(
    *,
    label: str,
    rt: Any,
    bundle: Any,
    enc_first: Any,
    rows: int,
    cudnn_tf32: bool,
) -> dict[str, Any]:
    configure_precision(cudnn_tf32=cudnn_tf32)
    runner = BundleFirstRunner(rt, bundle, enc_first)
    records: dict[int, dict[str, Any]] = {}
    first_outputs: dict[int, dict[str, torch.Tensor]] = {}
    token_divs_vs_bundle = 0
    event_divs_vs_bundle = 0
    t0 = time.time()
    for utt in range(rows):
        got = runner.run_row(utt)
        records[utt] = {
            "final_tokens": got["final_tokens"],
            "events": got["events"],
            "sample_id": sample_id_for(bundle, utt),
        }
        first_outputs[utt] = got["first_outputs"]
        gold = bundle_reference_row(bundle, utt)
        token_divs_vs_bundle += int(first_diff(got["final_tokens"], gold["final_tokens"]) is not None)
        event_divs_vs_bundle += int(first_diff(got["events"], gold["events"]) is not None)
        if (utt + 1) % 50 == 0 or utt + 1 == rows:
            print(
                f"{label}: {utt + 1}/{rows} vs_bundle token_div={token_divs_vs_bundle} "
                f"event_div={event_divs_vs_bundle} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return {
        "label": label,
        "cudnn_allow_tf32": bool(cudnn_tf32),
        "records": records,
        "first_outputs": first_outputs,
        "vs_bundle": {
            "token_divergences": token_divs_vs_bundle,
            "event_divergences": event_divs_vs_bundle,
        },
        "elapsed_seconds": time.time() - t0,
    }


def compare_candidate(
    *,
    label: str,
    rt: Any,
    bundle: Any,
    enc_first: Any,
    rows: int,
    cudnn_tf32: bool,
    reference_label: str,
    reference_records: dict[int, dict[str, Any]],
    reference_first_outputs: dict[int, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    configure_precision(cudnn_tf32=cudnn_tf32)
    runner = BundleFirstRunner(rt, bundle, enc_first)
    token_divs: list[dict[str, Any]] = []
    event_divs: list[dict[str, Any]] = []
    first_max = {"enc_out": 0.0, "cache_t": 0.0}
    known_cache_t: dict[str, float] = {}
    known_enc_out: dict[str, float] = {}
    t0 = time.time()
    for utt in range(rows):
        got = runner.run_row(utt)
        ref = reference_records[utt]
        token_idx = first_diff(got["final_tokens"], ref["final_tokens"])
        event_idx = first_diff(got["events"], ref["events"])
        if token_idx is not None:
            token_divs.append(
                {
                    "utt": utt,
                    "sample_id": ref["sample_id"],
                    "first_diff": token_idx,
                    "got_len": len(got["final_tokens"]),
                    "ref_len": len(ref["final_tokens"]),
                    "got": None if token_idx >= len(got["final_tokens"]) else got["final_tokens"][token_idx],
                    "ref": None if token_idx >= len(ref["final_tokens"]) else ref["final_tokens"][token_idx],
                }
            )
        if event_idx is not None:
            event_divs.append(
                {
                    "utt": utt,
                    "sample_id": ref["sample_id"],
                    "first_diff": event_idx,
                    "got_len": len(got["events"]),
                    "ref_len": len(ref["events"]),
                    "got": None if event_idx >= len(got["events"]) else got["events"][event_idx],
                    "ref": None if event_idx >= len(ref["events"]) else ref["events"][event_idx],
                }
            )
        ref_first = reference_first_outputs[utt]
        enc_out_abs = tensor_max_abs(got["first_outputs"]["enc_out"], ref_first["enc_out"])
        cache_t_abs = tensor_max_abs(got["first_outputs"]["cache_t"], ref_first["cache_t"])
        first_max["enc_out"] = max(first_max["enc_out"], enc_out_abs)
        first_max["cache_t"] = max(first_max["cache_t"], cache_t_abs)
        if utt in KNOWN_DIVERGENT_UTTS:
            known_cache_t[str(utt)] = cache_t_abs
            known_enc_out[str(utt)] = enc_out_abs
        if (utt + 1) % 50 == 0 or utt + 1 == rows:
            print(
                f"{label} vs {reference_label}: {utt + 1}/{rows} "
                f"token_div={len(token_divs)} event_div={len(event_divs)} "
                f"cache_t_max={first_max['cache_t']:.6g} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return {
        "variant": label,
        "reference": reference_label,
        "cudnn_allow_tf32": bool(cudnn_tf32),
        "rows": rows,
        "token_divergences": len(token_divs),
        "event_divergences": len(event_divs),
        "token_divergence_details": token_divs[:50],
        "event_divergence_details": event_divs[:50],
        "first_chunk_max_abs": first_max,
        "known_utt_first_chunk_max_abs": {
            utt: {"cache_t": known_cache_t[utt], "enc_out": known_enc_out[utt]}
            for utt in sorted(known_cache_t, key=int)
        },
        "elapsed_seconds": time.time() - t0,
    }


def tensorless_reference(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": ref["label"],
        "cudnn_allow_tf32": ref["cudnn_allow_tf32"],
        "vs_bundle": ref["vs_bundle"],
        "elapsed_seconds": ref["elapsed_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT / "step1b-tf32-reprobe.json")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    del args.progress_every  # Fixed at 50 to keep command lines stable.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    configure_precision(cudnn_tf32=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    from export_session_bundle import RecordingContinuousFinalizeRef

    started = time.time()
    print("loading NeMo model", flush=True)
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

    print("loading Step-1 reference runtime and session bundle", flush=True)
    rt = RecordingContinuousFinalizeRef(model, artifacts_dir=str(ART), warm_encoder=False)
    bundle = torch.jit.load(str(ART / "session_bundle.ts"), map_location="cpu")
    total_rows = _scalar_i64(bundle, "num_utts")
    rows = min(int(args.rows), int(total_rows))
    ts_first = rt.enc_first

    print("collecting TS/cudnn_tf32=true first-output references", flush=True)
    ts_true_first = collect_ts_true_first_outputs(rt, bundle, ts_first, rows)
    ts_true_records = {utt: bundle_reference_row(bundle, utt) for utt in range(rows)}
    ts_true_ref = {
        "label": "ts_cudnn_tf32_true_bundle",
        "cudnn_allow_tf32": True,
        "vs_bundle": {"token_divergences": 0, "event_divergences": 0},
        "elapsed_seconds": 0.0,
    }

    print("running TS/cudnn_tf32=false reference", flush=True)
    ts_false = run_reference(
        label="ts_cudnn_tf32_false",
        rt=rt,
        bundle=bundle,
        enc_first=ts_first,
        rows=rows,
        cudnn_tf32=False,
    )

    shared = SharedWeights(ART / "finalize_shared_weights.pt")
    variants = [
        ("original_tf32_aoti", ART / "enc_first_aoti.pt2"),
        ("variant_a_fp32_cudnn", ART / "enc_first_aoti_fp32.pt2"),
        ("variant_b_fp32_samep", ART / "enc_first_aoti_fp32_samep.pt2"),
    ]
    comparisons: list[dict[str, Any]] = []
    variant_stats: dict[str, Any] = {}
    for variant_label, package_path in variants:
        print(f"loading {variant_label}: {package_path}", flush=True)
        candidate, stats = load_aoti_first(shared, package_path)
        variant_stats[variant_label] = stats
        comparisons.append(
            compare_candidate(
                label=variant_label,
                rt=rt,
                bundle=bundle,
                enc_first=candidate,
                rows=rows,
                cudnn_tf32=True,
                reference_label="ts_cudnn_tf32_true",
                reference_records=ts_true_records,
                reference_first_outputs=ts_true_first,
            )
        )
        comparisons.append(
            compare_candidate(
                label=variant_label,
                rt=rt,
                bundle=bundle,
                enc_first=candidate,
                rows=rows,
                cudnn_tf32=False,
                reference_label="ts_cudnn_tf32_false",
                reference_records=ts_false["records"],
                reference_first_outputs=ts_false["first_outputs"],
            )
        )
        del candidate
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "elapsed_seconds": time.time() - started,
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(0),
        "rows": rows,
        "references": {
            "ts_cudnn_tf32_true": ts_true_ref,
            "ts_cudnn_tf32_false": tensorless_reference(ts_false),
        },
        "variant_constants": variant_stats,
        "comparisons": comparisons,
        "settings": {
            "cuda.matmul.allow_tf32": False,
            "matmul_precision": "highest",
            "TORCHINDUCTOR_MAX_AUTOTUNE": os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE", ""),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
