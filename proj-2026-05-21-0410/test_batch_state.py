"""Probe B2: full state equality for cache-aware batched streaming.

This is the blocking Step-5 probe. It compares separate B=1 streams against
batched B=2 and B=4 streams over the same full multi-chunk clips, including
cache tensors, RNNT hypothesis state, pred_out_stream, and emitted frame cursors.

Run with:
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python proj-2026-05-21-0410/test_batch_state.py
"""

from __future__ import annotations

import dataclasses
import gc
import os
import sqlite3
import sys
from typing import Any, Optional

import numpy as np
import torch


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from nemotron_speech.batch_primitives import (  # noqa: E402
    conformer_stream_step_restoring_drop_extra,
    ready_predicate,
    scatter_cache_row,
    stack_caches,
    stack_hypotheses,
    stack_pred_out,
    stack_processed,
)


DB = os.path.join(REPO, "stt-benchmark/stt_benchmark_data/results.db")
EN_NEMO = open("/tmp/en-nemo-path").read().strip()
SR = 16000
ATOL = 1.0e-4
RTOL = 1.0e-5

# Four distinct clips with the same normal streaming chunk count (30) under the
# server's fixed preprocessor geometry. That lets B=4 run full-stream lockstep
# without padding, final partial chunks, or row leave/rejoin.
CLIP_IDS = [
    "0ff9664b-2d12-7d3d-eb35-10a4b57061ee",
    "9b157ac1-e896-9ed7-3446-d9a342b72ea8",
    "bc53d5b8-99a8-f66f-c005-d6a984ec8158",
    "82855049-2871-b512-c456-d1366f23b97f",
]
PERMUTED_ORDER = [2, 0, 3, 1]


def log(*args: Any) -> None:
    print(*args, flush=True)


@dataclasses.dataclass
class Geometry:
    shift_frames: int
    pre_encode_cache_size: int
    drop_extra: int
    hop_samples: int
    window_size_samples: int
    raw_audio_ring_samples: int
    preprocess_align_pad_samples: int
    preprocess_new_audio_samples: int
    first_preprocess_mel_frame: int
    constant_preprocess_samples: int
    warmup_ms: int
    warmup_frames: Optional[int]


@dataclasses.dataclass
class StreamState:
    sample_id: str
    audio: np.ndarray
    pending_audio: np.ndarray
    total_audio_samples: int
    raw_audio_ring: np.ndarray
    mel_frame_ring: Optional[torch.Tensor]
    emitted_frames: int
    synthetic_prefix_samples: int
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    previous_hypotheses: Any
    pred_out_stream: Any
    current_text: str = ""


@dataclasses.dataclass
class CompareSummary:
    label: str
    tensor_count: int = 0
    bit_equal: bool = True
    allclose: bool = True
    max_abs: float = 0.0
    max_path: str = ""


def clone_tree(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    if memo is None:
        memo = {}
    oid = id(obj)
    if oid in memo:
        return memo[oid]
    if torch.is_tensor(obj):
        return obj.detach().clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        out: list[Any] = []
        memo[oid] = out
        out.extend(clone_tree(x, memo) for x in obj)
        return out
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        out_tuple = tuple(clone_tree(x, memo) for x in obj)
        memo[oid] = out_tuple
        return out_tuple
    if isinstance(obj, dict):
        out_dict: dict[Any, Any] = {}
        memo[oid] = out_dict
        for key, value in obj.items():
            out_dict[clone_tree(key, memo)] = clone_tree(value, memo)
        return out_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out_obj = dataclasses.replace(obj)
        memo[oid] = out_obj
        for field in dataclasses.fields(obj):
            setattr(out_obj, field.name, clone_tree(getattr(obj, field.name), memo))
        return out_obj
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
        import copy

        out_obj = copy.copy(obj)
        memo[oid] = out_obj
        for key, value in vars(obj).items():
            setattr(out_obj, key, clone_tree(value, memo))
        return out_obj
    return obj


def snapshot_tree_cpu(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    if memo is None:
        memo = {}
    oid = id(obj)
    if oid in memo:
        return memo[oid]
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        out: list[Any] = []
        memo[oid] = out
        out.extend(snapshot_tree_cpu(x, memo) for x in obj)
        return out
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        out_tuple = tuple(snapshot_tree_cpu(x, memo) for x in obj)
        memo[oid] = out_tuple
        return out_tuple
    if isinstance(obj, dict):
        out_dict: dict[Any, Any] = {}
        memo[oid] = out_dict
        for key, value in obj.items():
            out_dict[snapshot_tree_cpu(key, memo)] = snapshot_tree_cpu(value, memo)
        return out_dict
    return obj


def clone_state(state: StreamState) -> StreamState:
    return StreamState(
        sample_id=state.sample_id,
        audio=state.audio.copy(),
        pending_audio=state.pending_audio.copy(),
        total_audio_samples=int(state.total_audio_samples),
        raw_audio_ring=state.raw_audio_ring.copy(),
        mel_frame_ring=clone_tree(state.mel_frame_ring),
        emitted_frames=int(state.emitted_frames),
        synthetic_prefix_samples=int(state.synthetic_prefix_samples),
        cache_last_channel=clone_tree(state.cache_last_channel),
        cache_last_time=clone_tree(state.cache_last_time),
        cache_last_channel_len=clone_tree(state.cache_last_channel_len),
        previous_hypotheses=clone_tree(state.previous_hypotheses),
        pred_out_stream=clone_tree(state.pred_out_stream),
        current_text=str(state.current_text),
    )


def hypothesis_snapshot(hyp: Any) -> Any:
    if hyp is None:
        return None
    return {
        "text": getattr(hyp, "text", None),
        "y_sequence": snapshot_tree_cpu(getattr(hyp, "y_sequence", None)),
        "dec_state": snapshot_tree_cpu(getattr(hyp, "dec_state", None)),
        "last_token": snapshot_tree_cpu(getattr(hyp, "last_token", None)),
    }


def state_snapshot(state: StreamState) -> dict[str, Any]:
    hyp = None
    if state.previous_hypotheses is not None:
        hyp = [hypothesis_snapshot(h) for h in state.previous_hypotheses]
    return {
        "text": state.current_text,
        "cache_last_channel": snapshot_tree_cpu(state.cache_last_channel),
        "cache_last_time": snapshot_tree_cpu(state.cache_last_time),
        "cache_last_channel_len": snapshot_tree_cpu(state.cache_last_channel_len),
        "previous_hypotheses": hyp,
        "pred_out_stream": snapshot_tree_cpu(state.pred_out_stream),
        "emitted_frames": int(state.emitted_frames),
    }


def text_from_hyp(hyp: Any) -> str:
    if hyp is None:
        return ""
    if isinstance(hyp, str):
        return hyp
    text = getattr(hyp, "text", None)
    if isinstance(text, str):
        return text
    return str(hyp)


def load_clip(sample_id: str) -> np.ndarray:
    con = sqlite3.connect(DB)
    row = con.execute("SELECT audio_path FROM samples WHERE sample_id=?", (sample_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"sample_id not found in {DB}: {sample_id}")
    with open(os.path.join(REPO, "stt-benchmark", row[0]), "rb") as f:
        pcm = f.read()
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def build_geometry(model: Any) -> Geometry:
    scfg = model.encoder.streaming_cfg
    shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, (list, tuple)) else scfg.shift_size
    pre_cache = scfg.pre_encode_cache_size
    pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, (list, tuple)) else pre_cache
    preprocessor_cfg = model.cfg.preprocessor
    hop_length_sec = preprocessor_cfg.get("window_stride", 0.01)
    window_size_sec = preprocessor_cfg.get("window_size", 0.025)
    featurizer = model.preprocessor.featurizer
    hop_samples = int(getattr(featurizer, "hop_length", int(hop_length_sec * SR)))
    window_size_samples = int(getattr(featurizer, "win_length", int(window_size_sec * SR)))
    raw_audio_ring_samples = window_size_samples - hop_samples
    align_pad = (hop_samples - (raw_audio_ring_samples % hop_samples)) % hop_samples
    preprocess_new_audio_samples = (int(shift_frames) + 1) * hop_samples
    prefix_samples = align_pad + raw_audio_ring_samples
    first_preprocess_mel_frame = prefix_samples // hop_samples
    right_context = 1
    final_padding_frames = (right_context + 1) * int(shift_frames)
    min_plan_frames = (
        first_preprocess_mel_frame
        + int(pre_encode_cache_size)
        + int(shift_frames)
        + final_padding_frames
        + 1
    )
    constant_preprocess_frames = 1 << (min_plan_frames - 1).bit_length()
    constant_preprocess_samples = (constant_preprocess_frames - 1) * hop_samples
    warmup_ms = int(os.environ.get("NEMOTRON_WARMUP_MS", "0") or "0")
    warmup_frames = None
    if warmup_ms > 0:
        target_samples = int(round(SR * warmup_ms / 1000))
        warmup_frames = max(int(pre_encode_cache_size), int(round(target_samples / hop_samples)))
    return Geometry(
        shift_frames=int(shift_frames),
        pre_encode_cache_size=int(pre_encode_cache_size),
        drop_extra=int(scfg.drop_extra_pre_encoded),
        hop_samples=hop_samples,
        window_size_samples=window_size_samples,
        raw_audio_ring_samples=raw_audio_ring_samples,
        preprocess_align_pad_samples=align_pad,
        preprocess_new_audio_samples=preprocess_new_audio_samples,
        first_preprocess_mel_frame=first_preprocess_mel_frame,
        constant_preprocess_samples=constant_preprocess_samples,
        warmup_ms=warmup_ms,
        warmup_frames=warmup_frames,
    )


def build_fixed_preprocess_audio(geom: Geometry, raw_audio_ring: np.ndarray, new_audio: np.ndarray) -> tuple[np.ndarray, int]:
    if len(raw_audio_ring) != geom.raw_audio_ring_samples:
        raise ValueError(f"raw ring length mismatch: {len(raw_audio_ring)}")
    prefix_len = geom.preprocess_align_pad_samples + geom.raw_audio_ring_samples
    valid_samples = prefix_len + len(new_audio)
    if valid_samples > geom.constant_preprocess_samples:
        raise ValueError(f"valid span {valid_samples} exceeds K={geom.constant_preprocess_samples}")
    audio = np.zeros(geom.constant_preprocess_samples, dtype=np.float32)
    cursor = geom.preprocess_align_pad_samples
    audio[cursor : cursor + geom.raw_audio_ring_samples] = raw_audio_ring
    cursor += geom.raw_audio_ring_samples
    audio[cursor : cursor + len(new_audio)] = new_audio
    return audio, valid_samples


def preprocess_fixed_audio(model: Any, audio: np.ndarray, valid_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).cuda()
    audio_len = torch.tensor([valid_samples], device="cuda", dtype=torch.long)
    return model.preprocessor(input_signal=audio_tensor, length=audio_len)


def update_mel_frame_ring(geom: Geometry, state: StreamState, new_mel: torch.Tensor) -> None:
    if state.mel_frame_ring is None:
        combined = new_mel.detach()
    else:
        combined = torch.cat((state.mel_frame_ring, new_mel.detach()), dim=-1)
    state.mel_frame_ring = combined[:, :, -geom.pre_encode_cache_size :].detach()


def init_state(model: Any, geom: Geometry, sample_id: str, audio: np.ndarray) -> StreamState:
    cache = model.encoder.get_initial_cache_state(batch_size=1)
    state = StreamState(
        sample_id=sample_id,
        audio=audio,
        pending_audio=audio.copy(),
        total_audio_samples=len(audio),
        raw_audio_ring=np.zeros(geom.raw_audio_ring_samples, dtype=np.float32),
        mel_frame_ring=None,
        emitted_frames=0,
        synthetic_prefix_samples=0,
        cache_last_channel=cache[0],
        cache_last_time=cache[1],
        cache_last_channel_len=cache[2],
        previous_hypotheses=None,
        pred_out_stream=None,
    )
    if geom.warmup_frames is not None:
        run_session_warmup(model, geom, state)
    return state


def run_session_warmup(model: Any, geom: Geometry, state: StreamState) -> None:
    assert geom.warmup_frames is not None
    warmup_samples = geom.warmup_frames * geom.hop_samples
    preprocess_samples = warmup_samples + geom.hop_samples
    warmup_audio = np.zeros(preprocess_samples, dtype=np.float32)
    fixed_audio, valid_samples = build_fixed_preprocess_audio(geom, state.raw_audio_ring, warmup_audio)
    mel, _ = preprocess_fixed_audio(model, fixed_audio, valid_samples)
    warmup_mel = mel[
        :,
        :,
        geom.first_preprocess_mel_frame : geom.first_preprocess_mel_frame + geom.warmup_frames,
    ]
    chunk_len = torch.tensor([warmup_mel.shape[-1]], device="cuda")
    (
        state.pred_out_stream,
        _discarded_texts,
        state.cache_last_channel,
        state.cache_last_time,
        state.cache_last_channel_len,
        state.previous_hypotheses,
    ) = conformer_stream_step_restoring_drop_extra(
        model,
        processed_signal=warmup_mel,
        processed_signal_length=chunk_len,
        cache_last_channel=state.cache_last_channel,
        cache_last_time=state.cache_last_time,
        cache_last_channel_len=state.cache_last_channel_len,
        keep_all_outputs=False,
        previous_hypotheses=None,
        previous_pred_out=None,
        drop_extra_pre_encoded=0,
        return_transcription=True,
    )
    consumed_audio = warmup_audio[:warmup_samples]
    if len(consumed_audio) >= geom.raw_audio_ring_samples:
        state.raw_audio_ring = consumed_audio[-geom.raw_audio_ring_samples :].copy()
    else:
        keep = geom.raw_audio_ring_samples - len(consumed_audio)
        state.raw_audio_ring = np.concatenate([state.raw_audio_ring[-keep:], consumed_audio]).astype(
            np.float32, copy=False
        )
    update_mel_frame_ring(geom, state, warmup_mel)
    state.emitted_frames = geom.warmup_frames
    state.synthetic_prefix_samples = warmup_samples
    state.current_text = ""


def state_is_ready(geom: Geometry, state: StreamState) -> bool:
    return ready_predicate(
        synthetic_prefix_samples=state.synthetic_prefix_samples,
        total_audio_samples=state.total_audio_samples,
        emitted_frames=state.emitted_frames,
        shift_frames=geom.shift_frames,
        hop_samples=geom.hop_samples,
        pending_audio_len=len(state.pending_audio),
        preprocess_new_audio_samples=geom.preprocess_new_audio_samples,
    )


def prepare_row(model: Any, geom: Geometry, state: StreamState) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not state_is_ready(geom, state):
        raise RuntimeError(f"state is not ready: {state.sample_id}")
    new_audio = state.pending_audio[: geom.preprocess_new_audio_samples]
    fixed_audio, valid_samples = build_fixed_preprocess_audio(geom, state.raw_audio_ring, new_audio)
    mel, _mel_len = preprocess_fixed_audio(model, fixed_audio, valid_samples)
    valid_new_mel = mel[
        :,
        :,
        geom.first_preprocess_mel_frame : geom.first_preprocess_mel_frame + geom.shift_frames,
    ]
    if state.emitted_frames == 0:
        chunk_mel = valid_new_mel
        drop_extra = 0
    else:
        assert state.mel_frame_ring is not None
        chunk_mel = torch.cat((state.mel_frame_ring, valid_new_mel), dim=-1)
        drop_extra = geom.drop_extra
    return chunk_mel, valid_new_mel, drop_extra


def process_group(model: Any, geom: Geometry, group: list[StreamState]) -> None:
    prepared = [prepare_row(model, geom, state) for state in group]
    chunk_mels = [item[0] for item in prepared]
    valid_new_mels = [item[1] for item in prepared]
    drop_values = [item[2] for item in prepared]
    assert len(set(drop_values)) == 1, f"mixed drop_extra in group: {drop_values}"
    chunk, chunk_len = stack_processed(chunk_mels)
    clc, clt, clcl = stack_caches(
        [(state.cache_last_channel, state.cache_last_time, state.cache_last_channel_len) for state in group]
    )

    # Decoder exception policy for future scheduler wiring: clone mutable inputs
    # and assign returned hypotheses only after the whole model call succeeds.
    per_session_hyps = [clone_tree(state.previous_hypotheses) for state in group]
    stacked_hyps = stack_hypotheses(per_session_hyps)
    previous_hypotheses = None if all(h is None for h in stacked_hyps) else stacked_hyps
    per_session_pred = [clone_tree(state.pred_out_stream) for state in group]
    previous_pred_out = stack_pred_out(per_session_pred, rnnt=True)

    pred_out, transcribed_texts, next_clc, next_clt, next_clcl, best_hyp = conformer_stream_step_restoring_drop_extra(
        model,
        processed_signal=chunk,
        processed_signal_length=chunk_len,
        cache_last_channel=clc,
        cache_last_time=clt,
        cache_last_channel_len=clcl,
        keep_all_outputs=False,
        previous_hypotheses=previous_hypotheses,
        previous_pred_out=previous_pred_out,
        drop_extra_pre_encoded=drop_values[0],
        return_transcription=True,
    )

    for row, state in enumerate(group):
        state.cache_last_channel, state.cache_last_time, state.cache_last_channel_len = scatter_cache_row(
            next_clc, next_clt, next_clcl, row
        )
        state.previous_hypotheses = [best_hyp[row]] if best_hyp is not None else None
        state.pred_out_stream = [pred_out[row]] if pred_out is not None else None

        consumed_audio = state.pending_audio[: geom.shift_frames * geom.hop_samples]
        if len(consumed_audio) >= geom.raw_audio_ring_samples:
            state.raw_audio_ring = consumed_audio[-geom.raw_audio_ring_samples :].copy()
        else:
            keep = geom.raw_audio_ring_samples - len(consumed_audio)
            state.raw_audio_ring = np.concatenate([state.raw_audio_ring[-keep:], consumed_audio]).astype(
                np.float32, copy=False
            )
        state.pending_audio = state.pending_audio[geom.shift_frames * geom.hop_samples :]
        update_mel_frame_ring(geom, state, valid_new_mels[row])
        state.emitted_frames += geom.shift_frames
        if transcribed_texts and len(transcribed_texts) > row and transcribed_texts[row] is not None:
            state.current_text = text_from_hyp(transcribed_texts[row])


def run_streams(
    model: Any,
    geom: Geometry,
    clips: dict[str, np.ndarray],
    *,
    max_batch: int,
    order: Optional[list[int]] = None,
) -> dict[str, list[dict[str, Any]]]:
    sample_ids = list(clips)
    if order is not None:
        sample_ids = [sample_ids[i] for i in order]
    states = [init_state(model, geom, sample_id, clips[sample_id]) for sample_id in sample_ids]
    captures: dict[str, list[dict[str, Any]]] = {sample_id: [] for sample_id in clips}
    while True:
        ready = [state for state in states if state_is_ready(geom, state)]
        if not ready:
            break
        for start in range(0, len(ready), max_batch):
            group = ready[start : start + max_batch]
            process_group(model, geom, group)
            for state in group:
                captures[state.sample_id].append(state_snapshot(state))
    return captures


def compare_value(path: str, expected: Any, actual: Any, summary: CompareSummary) -> None:
    if torch.is_tensor(expected) or torch.is_tensor(actual):
        if not (torch.is_tensor(expected) and torch.is_tensor(actual)):
            raise AssertionError(f"{path}: tensor/non-tensor mismatch")
        if expected.shape != actual.shape:
            raise AssertionError(f"{path}: shape mismatch {tuple(expected.shape)} != {tuple(actual.shape)}")
        if expected.dtype != actual.dtype:
            raise AssertionError(f"{path}: dtype mismatch {expected.dtype} != {actual.dtype}")
        summary.tensor_count += 1
        equal = torch.equal(expected, actual)
        summary.bit_equal = summary.bit_equal and bool(equal)
        if expected.numel() == 0:
            return
        if expected.is_floating_point() or actual.is_floating_point():
            exp = expected.to(torch.float32)
            act = actual.to(torch.float32)
            max_abs = float((exp - act).abs().max().item())
            close = torch.allclose(exp, act, atol=ATOL, rtol=RTOL)
            summary.allclose = summary.allclose and bool(close)
            if max_abs > summary.max_abs:
                summary.max_abs = max_abs
                summary.max_path = path
        elif not equal:
            summary.allclose = False
            raise AssertionError(f"{path}: integer/bool tensor mismatch")
        return

    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if type(expected) is not type(actual):
            raise AssertionError(f"{path}: sequence type mismatch {type(expected)} != {type(actual)}")
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: sequence length mismatch {len(expected)} != {len(actual)}")
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual)):
            compare_value(f"{path}[{idx}]", exp_item, act_item, summary)
        return

    if isinstance(expected, dict) or isinstance(actual, dict):
        if not (isinstance(expected, dict) and isinstance(actual, dict)):
            raise AssertionError(f"{path}: dict/non-dict mismatch")
        if set(expected) != set(actual):
            raise AssertionError(f"{path}: dict keys mismatch {set(expected)} != {set(actual)}")
        for key in sorted(expected):
            compare_value(f"{path}.{key}", expected[key], actual[key], summary)
        return

    if expected != actual:
        raise AssertionError(f"{path}: value mismatch {expected!r} != {actual!r}")


def compare_snapshot(label: str, expected: dict[str, Any], actual: dict[str, Any]) -> CompareSummary:
    summary = CompareSummary(label=label)
    for key in (
        "text",
        "cache_last_channel",
        "cache_last_time",
        "cache_last_channel_len",
        "previous_hypotheses",
        "pred_out_stream",
        "emitted_frames",
    ):
        compare_value(f"{label}.{key}", expected[key], actual[key], summary)
    return summary


def merge_summary(dst: CompareSummary, src: CompareSummary) -> None:
    dst.tensor_count += src.tensor_count
    dst.bit_equal = dst.bit_equal and src.bit_equal
    dst.allclose = dst.allclose and src.allclose
    if src.max_abs > dst.max_abs:
        dst.max_abs = src.max_abs
        dst.max_path = src.max_path


def compare_captures(
    label: str,
    reference: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
) -> CompareSummary:
    summary = CompareSummary(label=label)
    for sample_id in CLIP_IDS:
        ref_steps = reference[sample_id]
        cand_steps = candidate[sample_id]
        if len(ref_steps) != len(cand_steps):
            raise AssertionError(f"{label} {sample_id}: step count mismatch {len(ref_steps)} != {len(cand_steps)}")
        for step, (ref_snapshot, cand_snapshot) in enumerate(zip(ref_steps, cand_steps), start=1):
            if ref_snapshot["text"] != cand_snapshot["text"]:
                raise AssertionError(
                    f"{label} {sample_id} step={step}: text mismatch "
                    f"{ref_snapshot['text']!r} != {cand_snapshot['text']!r}"
                )
            if ref_snapshot["emitted_frames"] != cand_snapshot["emitted_frames"]:
                raise AssertionError(
                    f"{label} {sample_id} step={step}: emitted_frames mismatch "
                    f"{ref_snapshot['emitted_frames']} != {cand_snapshot['emitted_frames']}"
                )
            step_summary = compare_snapshot(f"{label}.{sample_id[:8]}.step{step}", ref_snapshot, cand_snapshot)
            merge_summary(summary, step_summary)
    return summary


def mutate_hypothesis_for_decoder_exception(hyp: Any) -> None:
    if hyp is None:
        return
    token = 987654321
    y_sequence = getattr(hyp, "y_sequence", None)
    if torch.is_tensor(y_sequence):
        hyp.y_sequence = torch.cat([y_sequence, y_sequence.new_tensor([token])])
    elif isinstance(y_sequence, list):
        y_sequence.append(token)
    else:
        hyp.y_sequence = [token]
    hyp.text = (getattr(hyp, "text", "") or "") + "__MUTATED_BY_PROBE__"


def run_exception_tests(model: Any, geom: Geometry, clips: dict[str, np.ndarray]) -> None:
    ids = list(clips)

    # Encoder-phase exception: patch encoder.forward so NeMo has already set
    # streaming_cfg.drop_extra_pre_encoded, then force an exception before NeMo's
    # unguarded restore line. The primitive wrapper must restore it.
    encoder_start = init_state(model, geom, ids[0], clips[ids[0]])
    encoder_expected = clone_state(encoder_start)
    process_group(model, geom, [encoder_expected])
    encoder_expected_snapshot = state_snapshot(encoder_expected)
    encoder_victim = clone_state(encoder_start)
    encoder_peer = init_state(model, geom, ids[1], clips[ids[1]])

    streaming_cfg = model.encoder.streaming_cfg
    actual_drop_extra = int(streaming_cfg.drop_extra_pre_encoded)
    sentinel_drop_extra = actual_drop_extra + 97
    streaming_cfg.drop_extra_pre_encoded = sentinel_drop_extra
    original_forward = model.encoder.forward
    seen_drop_extra: dict[str, Optional[int]] = {"value": None}

    def raising_forward(*args: Any, **kwargs: Any) -> Any:
        seen_drop_extra["value"] = int(streaming_cfg.drop_extra_pre_encoded)
        raise RuntimeError("injected encoder failure after drop_extra set")

    model.encoder.forward = raising_forward
    try:
        try:
            process_group(model, geom, [encoder_victim, encoder_peer])
            raise AssertionError("encoder exception injection did not raise")
        except RuntimeError as exc:
            assert "injected encoder failure" in str(exc)
        # NeMo overwrote the sentinel with the call's drop_extra during forward
        # (0 on a first chunk, geom.drop_extra on a steady chunk) — proving the restore
        # below is non-trivial. (The literal value is geometry/warmup-dependent.)
        assert seen_drop_extra["value"] != sentinel_drop_extra, seen_drop_extra
        assert int(streaming_cfg.drop_extra_pre_encoded) == sentinel_drop_extra
    finally:
        model.encoder.forward = original_forward
        streaming_cfg.drop_extra_pre_encoded = actual_drop_extra

    process_group(model, geom, [encoder_victim])
    encoder_recovery = compare_snapshot("encoder_exception_recovery", encoder_expected_snapshot, state_snapshot(encoder_victim))
    assert encoder_recovery.allclose

    # Decoder exception: patch the RNNT decoder entry point to mutate the cloned
    # partial hypothesis and raise. The original stream state must remain exactly
    # unchanged, and the next normal B=1 chunk must match the expected baseline.
    decoder_state0 = init_state(model, geom, ids[0], clips[ids[0]])
    decoder_state1 = init_state(model, geom, ids[1], clips[ids[1]])
    process_group(model, geom, [decoder_state0])
    process_group(model, geom, [decoder_state1])
    decoder_expected = clone_state(decoder_state0)
    process_group(model, geom, [decoder_expected])
    decoder_expected_snapshot = state_snapshot(decoder_expected)
    decoder_victim0 = clone_state(decoder_state0)
    decoder_victim1 = clone_state(decoder_state1)
    decoder_before = state_snapshot(decoder_victim0)

    original_decode = model.decoding.rnnt_decoder_predictions_tensor
    mutation_seen = {"value": False}

    def raising_decode(*args: Any, **kwargs: Any) -> Any:
        partial_hypotheses = kwargs.get("partial_hypotheses")
        if partial_hypotheses:
            mutate_hypothesis_for_decoder_exception(partial_hypotheses[0])
            mutation_seen["value"] = True
        raise RuntimeError("injected decoder failure mid-decode")

    model.decoding.rnnt_decoder_predictions_tensor = raising_decode
    try:
        try:
            process_group(model, geom, [decoder_victim0, decoder_victim1])
            raise AssertionError("decoder exception injection did not raise")
        except RuntimeError as exc:
            assert "injected decoder failure" in str(exc)
        assert mutation_seen["value"], "decoder mutation injection did not touch the cloned hypothesis"
    finally:
        model.decoding.rnnt_decoder_predictions_tensor = original_decode

    unchanged = compare_snapshot("decoder_exception_no_partial_assign", decoder_before, state_snapshot(decoder_victim0))
    assert unchanged.bit_equal and unchanged.allclose
    process_group(model, geom, [decoder_victim0])
    decoder_recovery = compare_snapshot("decoder_exception_recovery", decoder_expected_snapshot, state_snapshot(decoder_victim0))
    assert decoder_recovery.allclose

    log("  exceptions: encoder drop_extra restored + B=1 recovery OK; decoder cloned-hyp no-partial-assign + B=1 recovery OK")


def load_model() -> Any:
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    # Batching state-faithfulness is validated under fp32 (TF32 OFF) — TF32 batched-matmul
    # reduction-order noise drifts the cache ~0.03 (text still byte-identical, but state allclose
    # fails). fp32 drifts only ~1e-4 (allclose passes). So this gate runs TF32-off by default;
    # set NEMOTRON_KEEP_TF32=1 to reproduce the TF32 drift for comparison.
    if os.environ.get("NEMOTRON_KEEP_TF32") == "1":
        log(f"TF32 KEPT: matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
            f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} — expect state drift > allclose tol")
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        log("TF32 DISABLED (allow_tf32=False) — fp32 matmuls; batching state-faithful precision")

    log(f"loading model: {EN_NEMO}")
    model = nemo_asr.models.ASRModel.restore_from(EN_NEMO, map_location="cuda")
    model.encoder.set_default_att_context_size([70, 1])
    model.change_decoding_strategy(
        decoding_cfg=OmegaConf.create(
            {
                "strategy": "greedy",
                "greedy": {
                    "max_symbols": 10,
                    "loop_labels": False,
                    "use_cuda_graph_decoder": False,
                },
            }
        )
    )
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    return model


def test_probe_b2_batch_state() -> None:
    torch.backends.cudnn.benchmark = False
    model = load_model()
    try:
        geom = build_geometry(model)
        clips = {sample_id: load_clip(sample_id) for sample_id in CLIP_IDS}
        chunk_counts = {
            sample_id: max(0, (len(audio) - geom.hop_samples) // (geom.shift_frames * geom.hop_samples))
            for sample_id, audio in clips.items()
        }
        log(
            "geometry: "
            f"shift_frames={geom.shift_frames} pre_cache={geom.pre_encode_cache_size} "
            f"drop_extra={geom.drop_extra} hop={geom.hop_samples} "
            f"new_audio={geom.preprocess_new_audio_samples} K={geom.constant_preprocess_samples} "
            f"warmup_ms={geom.warmup_ms}"
        )
        log("clips:")
        for sample_id in CLIP_IDS:
            log(f"  {sample_id[:8]} samples={len(clips[sample_id])} normal_chunks={chunk_counts[sample_id]}")
        assert len(set(CLIP_IDS)) == len(CLIP_IDS) and len(CLIP_IDS) >= 3
        assert len(set(chunk_counts.values())) == 1, chunk_counts

        with torch.inference_mode():
            reference = run_streams(model, geom, clips, max_batch=1)
            n_chunks = len(reference[CLIP_IDS[0]])
            assert n_chunks == next(iter(chunk_counts.values()))
            log(f"B=1 reference captured: clips={len(CLIP_IDS)} chunks_per_clip={n_chunks}")

            b2 = run_streams(model, geom, clips, max_batch=2)
            b2_summary = compare_captures("B2", reference, b2)
            log(
                f"B=2: text byte-identical=True emitted_exact=True "
                f"torch.equal(all tensors)={b2_summary.bit_equal} "
                f"allclose(atol={ATOL})={b2_summary.allclose} "
                f"max_abs={b2_summary.max_abs:.6g} path={b2_summary.max_path or 'n/a'}"
            )

            b4 = run_streams(model, geom, clips, max_batch=4)
            b4_summary = compare_captures("B4", reference, b4)
            log(
                f"B=4: text byte-identical=True emitted_exact=True "
                f"torch.equal(all tensors)={b4_summary.bit_equal} "
                f"allclose(atol={ATOL})={b4_summary.allclose} "
                f"max_abs={b4_summary.max_abs:.6g} path={b4_summary.max_path or 'n/a'}"
            )

            permuted = run_streams(model, geom, clips, max_batch=4, order=PERMUTED_ORDER)
            perm_summary = compare_captures("B4_permuted", reference, permuted)
            log(
                f"B=4 row-permute {PERMUTED_ORDER}: invariant=True "
                f"torch.equal(all tensors)={perm_summary.bit_equal} "
                f"allclose(atol={ATOL})={perm_summary.allclose} "
                f"max_abs={perm_summary.max_abs:.6g} path={perm_summary.max_path or 'n/a'}"
            )

            run_exception_tests(model, geom, clips)

        # GATE: text byte-identity is hard-enforced in compare_captures (raises on any mismatch).
        # State equivalence uses torch.allclose(atol=1e-4, rtol=1e-5) — the principled fp32-CUDA
        # FP-equivalence test (it FAILS under TF32 drift ~0.03 and PASSES under fp32 ~1e-4). The raw
        # max_abs is reported as info only (a scalar atol-only cap double-counts what rtol already covers).
        failures = []
        if not b2_summary.allclose:
            failures.append(
                f"B=2 state not allclose; max_abs={b2_summary.max_abs:.6g} at {b2_summary.max_path}"
            )
        if not b4_summary.allclose:
            failures.append(
                f"B=4 state not allclose; max_abs={b4_summary.max_abs:.6g} at {b4_summary.max_path}"
            )
        if not perm_summary.allclose:
            failures.append(
                f"B=4 permuted state not allclose; max_abs={perm_summary.max_abs:.6g} at {perm_summary.max_path}"
            )
        if failures:
            raise AssertionError("PROBE_B2 NO-GO: " + "; ".join(failures))
        log(
            "PROBE_B2 PASS: B=2/B=4 full-stream text byte-identical; "
            "state allclose drift bounded; row permutation invariant; exceptions recover"
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    test_probe_b2_batch_state()
