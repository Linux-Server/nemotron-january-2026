#!/usr/bin/env python3
"""Gate Ga probe: NeMo streaming fork-state aliasing.

This is a scratch verification script for Step 3 of
proj-2026-05-17-1708/PLAN.md. It intentionally mirrors
src/nemotron_speech/server.py's greedy streaming path and final-chunk flush,
without starting a server or running a benchmark.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
RIGHT_CONTEXT = 1
SAMPLE_RATE = 16000
PREFIX_SECONDS = 4.0
CONTINUE_SECONDS = 3.0

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO_ROOT / "src/nemotron_speech/server.py"
DEFAULT_AUDIO_DIR = REPO_ROOT / "stt-benchmark/stt_benchmark_data/audio"

NEMO_MUTATION_SITES = [
    (
        "rnnt_greedy_decoding.py:825-831",
        "partial_hypotheses entries are mutated/replaced during greedy decode",
    ),
    (
        "rnnt_utils.py:153-181",
        "Hypothesis.merge_ mutates score/y_sequence/dec_state/timestamp/length/text/etc.",
    ),
    (
        "rnnt_greedy_decoding.py:753-775",
        "decoder and joint train/eval flags are model-global toggles",
    ),
    (
        "nemo/collections/asr/parts/mixins/streaming.py:53-74",
        "encoder.streaming_cfg.drop_extra_pre_encoded is temporarily mutated on the model",
    ),
]


@dataclass
class ProbeSession:
    """Minimal server-session state needed by the streaming path."""

    name: str
    accumulated_audio: np.ndarray
    emitted_frames: int
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    previous_hypotheses: Any = None
    pred_out_stream: Any = None
    current_text: str = ""


@dataclass
class ProbeConfig:
    device: torch.device
    shift_frames: int
    pre_encode_cache_size: int
    drop_extra: int
    hop_samples: int
    final_padding_frames: int
    decoder_class: str


@dataclass
class ProbeResult:
    recipe: str
    cache_unchanged: bool
    hyps_unchanged: bool
    continued_bit_identical: bool
    before_text: str
    fork_delta: str
    parent_continued_text: str
    control_continued_text: str
    changed_paths: list[str]

    @property
    def clean(self) -> bool:
        return self.cache_unchanged and self.hyps_unchanged and self.continued_bit_identical


@dataclass
class DetectorSelfTestResult:
    cache_detected: bool
    hyps_detected: bool
    hyp_tensor_path: str
    changed_paths: list[str]

    @property
    def passed(self) -> bool:
        return self.cache_detected and self.hyps_detected


def sha_short(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def pick_audio_file(audio_dir: Path) -> Path:
    requested = os.environ.get("PROBE_ALIAS_AUDIO")
    if requested:
        path = Path(requested)
        if not path.exists():
            raise FileNotFoundError(f"PROBE_ALIAS_AUDIO does not exist: {path}")
        return path

    candidates = sorted(audio_dir.glob("*.pcm"), key=lambda p: (-p.stat().st_size, p.name))
    if not candidates:
        raise FileNotFoundError(f"No .pcm files found under {audio_dir}")
    return candidates[0]


def load_pcm_s16le(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.int16)


def to_float_audio(audio_i16: np.ndarray) -> np.ndarray:
    return audio_i16.astype(np.float32) / 32768.0


def text_from_hypotheses(transcribed_texts: Any, fallback: str) -> str:
    if transcribed_texts and transcribed_texts[0]:
        hyp = transcribed_texts[0]
        if hasattr(hyp, "text"):
            return hyp.text
        if isinstance(hyp, str):
            return hyp
        return str(hyp)
    return fallback


def tensor_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def clone_tree(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Tensor-aware deepcopy.

    Every torch.Tensor is detached and cloned. Dataclasses such as NeMo
    Hypothesis are copied field-by-field so nested dec_state tensors/tuples/lists
    get independent storage without copying model objects.
    """

    if memo is None:
        memo = {}

    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if torch.is_tensor(obj):
        return tensor_clone(obj)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        cloned_list: list[Any] = []
        memo[oid] = cloned_list
        cloned_list.extend(clone_tree(item, memo) for item in obj)
        return cloned_list
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        cloned_tuple = tuple(clone_tree(item, memo) for item in obj)
        memo[oid] = cloned_tuple
        return cloned_tuple
    if isinstance(obj, dict):
        cloned_dict: dict[Any, Any] = {}
        memo[oid] = cloned_dict
        for key, value in obj.items():
            cloned_dict[clone_tree(key, memo)] = clone_tree(value, memo)
        return cloned_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for field in dataclasses.fields(obj):
            setattr(cloned_obj, field.name, clone_tree(getattr(obj, field.name), memo))
        return cloned_obj

    # Some NeMo helper objects may live inside optional fields. Avoid copying
    # modules or other heavy objects; copy plain object state only when present.
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for key, value in vars(obj).items():
            setattr(cloned_obj, key, clone_tree(value, memo))
        return cloned_obj

    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def clone_hypotheses_deep(previous_hypotheses: Any) -> Any:
    """Deep-copy each Hypothesis and recursively clone all tensor fields."""

    if previous_hypotheses is None:
        return None
    return [clone_tree(hyp) for hyp in previous_hypotheses]


def clone_session_deep(session: ProbeSession, name: str) -> ProbeSession:
    return ProbeSession(
        name=name,
        accumulated_audio=session.accumulated_audio.copy(),
        emitted_frames=session.emitted_frames,
        cache_last_channel=tensor_clone(session.cache_last_channel),
        cache_last_time=tensor_clone(session.cache_last_time),
        cache_last_channel_len=tensor_clone(session.cache_last_channel_len),
        previous_hypotheses=clone_hypotheses_deep(session.previous_hypotheses),
        pred_out_stream=clone_tree(session.pred_out_stream),
        current_text=session.current_text,
    )


def clone_session_shallow(session: ProbeSession, name: str) -> ProbeSession:
    """Hazard recipe: the fork shares cache tensors and Hypothesis objects."""

    return ProbeSession(
        name=name,
        accumulated_audio=session.accumulated_audio,
        emitted_frames=session.emitted_frames,
        cache_last_channel=session.cache_last_channel,
        cache_last_time=session.cache_last_time,
        cache_last_channel_len=session.cache_last_channel_len,
        previous_hypotheses=session.previous_hypotheses,
        pred_out_stream=session.pred_out_stream,
        current_text=session.current_text,
    )


def freeze(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Value snapshot for recursive equality checks."""

    if memo is None:
        memo = {}
    oid = id(obj)
    if oid in memo:
        return ("cycle", oid)
    if torch.is_tensor(obj):
        return ("tensor", tuple(obj.shape), str(obj.dtype), obj.detach().cpu().clone())
    if isinstance(obj, np.ndarray):
        return ("ndarray", obj.shape, str(obj.dtype), obj.copy())
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        memo[oid] = True
        return [freeze(item, memo) for item in obj]
    if isinstance(obj, tuple):
        memo[oid] = True
        return tuple(freeze(item, memo) for item in obj)
    if isinstance(obj, dict):
        memo[oid] = True
        return {
            repr(key): (freeze(key, memo), freeze(value, memo))
            for key, value in sorted(obj.items(), key=lambda item: repr(item[0]))
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        memo[oid] = True
        return {
            field.name: freeze(getattr(obj, field.name), memo)
            for field in dataclasses.fields(obj)
        }
    if hasattr(obj, "__dict__"):
        memo[oid] = True
        return {
            "__class__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
            "attrs": {
                key: freeze(value, memo)
                for key, value in sorted(vars(obj).items())
                if not key.startswith("__")
            },
        }
    return ("repr", repr(obj))


def equal_snapshot(left: Any, right: Any, path: str = "root", diffs: Optional[list[str]] = None) -> bool:
    if diffs is None:
        diffs = []

    if torch.is_tensor(left) and torch.is_tensor(right):
        ok = left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        if not ok:
            diffs.append(path)
        return ok
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        ok = left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)
        if not ok:
            diffs.append(path)
        return ok
    if type(left) is not type(right):
        diffs.append(path)
        return False
    if isinstance(left, list):
        if len(left) != len(right):
            diffs.append(path)
            return False
        ok = True
        for idx, (l_item, r_item) in enumerate(zip(left, right)):
            ok = equal_snapshot(l_item, r_item, f"{path}[{idx}]", diffs) and ok
        return ok
    if isinstance(left, tuple):
        if len(left) != len(right):
            diffs.append(path)
            return False
        ok = True
        for idx, (l_item, r_item) in enumerate(zip(left, right)):
            ok = equal_snapshot(l_item, r_item, f"{path}[{idx}]", diffs) and ok
        return ok
    if isinstance(left, dict):
        if left.keys() != right.keys():
            diffs.append(path)
            return False
        ok = True
        for key in left:
            ok = equal_snapshot(left[key], right[key], f"{path}.{key}", diffs) and ok
        return ok
    ok = left == right
    if not ok:
        diffs.append(path)
    return ok


def snapshot_cache(session: ProbeSession) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        tensor_clone(session.cache_last_channel),
        tensor_clone(session.cache_last_time),
        tensor_clone(session.cache_last_channel_len),
    )


def cache_equal(before: tuple[torch.Tensor, ...], session: ProbeSession) -> bool:
    after = (
        session.cache_last_channel,
        session.cache_last_time,
        session.cache_last_channel_len,
    )
    return all(torch.equal(left, right) for left, right in zip(before, after))


def iter_tensor_leaves(
    obj: Any,
    path: str = "root",
    memo: Optional[set[int]] = None,
):
    if memo is None:
        memo = set()

    if torch.is_tensor(obj):
        yield path, obj
        return
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return

    oid = id(obj)
    if oid in memo:
        return
    memo.add(oid)

    if isinstance(obj, list):
        for idx, item in enumerate(obj):
            yield from iter_tensor_leaves(item, f"{path}[{idx}]", memo)
        return
    if isinstance(obj, tuple):
        for idx, item in enumerate(obj):
            yield from iter_tensor_leaves(item, f"{path}[{idx}]", memo)
        return
    if isinstance(obj, dict):
        for key, value in sorted(obj.items(), key=lambda item: repr(item[0])):
            yield from iter_tensor_leaves(key, f"{path}.key[{repr(key)}]", memo)
            yield from iter_tensor_leaves(value, f"{path}[{repr(key)}]", memo)
        return
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            yield from iter_tensor_leaves(getattr(obj, field.name), f"{path}.{field.name}", memo)
        return
    if hasattr(obj, "__dict__"):
        for key, value in sorted(vars(obj).items()):
            if not key.startswith("__"):
                yield from iter_tensor_leaves(value, f"{path}.{key}", memo)


def find_mutable_tensor_leaf(obj: Any) -> tuple[str, torch.Tensor]:
    candidates = [
        (path, tensor)
        for path, tensor in iter_tensor_leaves(obj)
        if tensor.numel() > 0 and tensor.dtype is not torch.bool
    ]
    if not candidates:
        raise RuntimeError("No mutable tensor leaf found inside previous_hypotheses[0]")

    preferred_names = ("y_sequence", "dec_state")
    for preferred in preferred_names:
        for path, tensor in candidates:
            if preferred in path:
                return path, tensor
    return candidates[0]


def mutate_tensor_first_element(tensor: torch.Tensor) -> None:
    if tensor.dtype is torch.bool:
        raise RuntimeError("Refusing to mutate a bool tensor in detector self-test")

    delta: int | float = 1.0 if tensor.is_floating_point() or tensor.is_complex() else 1
    with torch.no_grad():
        if tensor.dim() == 0:
            tensor.add_(delta)
        else:
            tensor[(0,) * tensor.dim()].add_(delta)


def run_detector_selftest(base: ProbeSession) -> DetectorSelfTestResult:
    cache_victim = clone_session_deep(base, "detector-selftest-cache-victim")
    before_cache = snapshot_cache(cache_victim)
    with torch.no_grad():
        cache_victim.cache_last_channel.view(-1)[0].add_(1.0)
    cache_detected = not cache_equal(before_cache, cache_victim)

    hyps_victim = clone_session_deep(base, "detector-selftest-hyps-victim")
    before_hyps = freeze(hyps_victim.previous_hypotheses)
    hyp_tensor_path, hyp_tensor = find_mutable_tensor_leaf(hyps_victim.previous_hypotheses[0])
    mutate_tensor_first_element(hyp_tensor)
    changed_paths: list[str] = []
    hyps_detected = not equal_snapshot(
        before_hyps,
        freeze(hyps_victim.previous_hypotheses),
        "previous_hypotheses",
        changed_paths,
    )

    result = DetectorSelfTestResult(
        cache_detected=cache_detected,
        hyps_detected=hyps_detected,
        hyp_tensor_path=hyp_tensor_path,
        changed_paths=changed_paths[:12],
    )
    if result.passed:
        print(
            "DETECTOR-SELFTEST PASS: injected cache_last_channel corruption and "
            f"previous_hypotheses[0] tensor corruption ({hyp_tensor_path}) were both detected."
        )
    else:
        print(
            "DETECTOR-SELFTEST FAIL: "
            f"cache_detected={cache_detected}, hyps_detected={hyps_detected}, "
            f"hyp_tensor_path={hyp_tensor_path}, changed_paths={changed_paths[:6]}"
        )
    return result


def new_session(model: Any, name: str) -> ProbeSession:
    cache = model.encoder.get_initial_cache_state(batch_size=1)
    return ProbeSession(
        name=name,
        accumulated_audio=np.array([], dtype=np.float32),
        emitted_frames=0,
        cache_last_channel=cache[0],
        cache_last_time=cache[1],
        cache_last_channel_len=cache[2],
        previous_hypotheses=None,
        pred_out_stream=None,
        current_text="",
    )


def process_one_chunk(model: Any, cfg: ProbeConfig, session: ProbeSession) -> Optional[str]:
    audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).to(cfg.device)
    audio_len = torch.tensor([len(session.accumulated_audio)], device=cfg.device)

    with torch.inference_mode():
        mel, _mel_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)

        available_frames = mel.shape[-1] - 1
        new_frame_count = available_frames - session.emitted_frames
        if new_frame_count < cfg.shift_frames:
            return session.current_text

        if session.emitted_frames == 0:
            chunk_start = 0
            chunk_end = cfg.shift_frames
            drop_extra = 0
        else:
            chunk_start = session.emitted_frames - cfg.pre_encode_cache_size
            chunk_end = session.emitted_frames + cfg.shift_frames
            drop_extra = cfg.drop_extra

        chunk_mel = mel[:, :, chunk_start:chunk_end]
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=cfg.device)

        (
            session.pred_out_stream,
            transcribed_texts,
            session.cache_last_channel,
            session.cache_last_time,
            session.cache_last_channel_len,
            session.previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=False,
            previous_hypotheses=session.previous_hypotheses,
            previous_pred_out=session.pred_out_stream,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True,
        )

    session.emitted_frames += cfg.shift_frames
    return text_from_hypotheses(transcribed_texts, session.current_text)


def feed_audio(model: Any, cfg: ProbeConfig, session: ProbeSession, audio_float: np.ndarray) -> list[str]:
    session.accumulated_audio = np.concatenate([session.accumulated_audio, audio_float])
    texts: list[str] = []
    min_audio_for_chunk = (session.emitted_frames + cfg.shift_frames + 1) * cfg.hop_samples

    while len(session.accumulated_audio) >= min_audio_for_chunk:
        text = process_one_chunk(model, cfg, session)
        if text is not None and text != session.current_text:
            session.current_text = text
            texts.append(text)
        min_audio_for_chunk = (session.emitted_frames + cfg.shift_frames + 1) * cfg.hop_samples

    return texts


def flush_final_on_fork(model: Any, cfg: ProbeConfig, session: ProbeSession) -> str:
    """Mirror server.py _process_final_chunk after appending final silence."""

    original_text = session.current_text
    if len(session.accumulated_audio) > 0:
        padding_samples = cfg.final_padding_frames * cfg.hop_samples
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        session.accumulated_audio = np.concatenate([session.accumulated_audio, silence_padding])

    if len(session.accumulated_audio) == 0:
        return ""

    audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).to(cfg.device)
    audio_len = torch.tensor([len(session.accumulated_audio)], device=cfg.device)

    with torch.inference_mode():
        mel, _mel_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)

        total_mel_frames = mel.shape[-1]
        remaining_frames = total_mel_frames - session.emitted_frames
        if remaining_frames <= 0:
            return ""

        if session.emitted_frames == 0:
            chunk_start = 0
            drop_extra = 0
        else:
            chunk_start = session.emitted_frames - cfg.pre_encode_cache_size
            drop_extra = cfg.drop_extra

        chunk_mel = mel[:, :, chunk_start:]
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=cfg.device)

        (
            session.pred_out_stream,
            transcribed_texts,
            session.cache_last_channel,
            session.cache_last_time,
            session.cache_last_channel_len,
            session.previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=True,
            previous_hypotheses=session.previous_hypotheses,
            previous_pred_out=session.pred_out_stream,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True,
        )

    final_text = text_from_hypotheses(transcribed_texts, session.current_text)
    session.current_text = final_text
    if final_text.startswith(original_text):
        return final_text[len(original_text) :].lstrip()
    return final_text


def apply_greedy_decoding_strategy(model: Any, loop_labels: bool, strategy: str = "greedy") -> str:
    from omegaconf import OmegaConf

    decoding_cfg = OmegaConf.create(
        {
            "strategy": strategy,
            "greedy": {
                "max_symbols": 10,
                "loop_labels": loop_labels,
                "use_cuda_graph_decoder": False,
            },
        }
    )
    model.change_decoding_strategy(decoding_cfg=decoding_cfg)
    model.eval()
    return type(model.decoding.decoding).__name__


def load_model_and_config() -> tuple[Any, ProbeConfig]:
    import nemo.collections.asr as nemo_asr

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe; server.py uses CUDA for streaming inference.")
    device = torch.device(os.environ.get("NEMOTRON_SPEECH_DEVICE", "cuda"))
    if device.type != "cuda":
        raise RuntimeError(f"This probe is intended for CUDA, got NEMOTRON_SPEECH_DEVICE={device}")

    print(f"Loading {MODEL_NAME} on {device} with server.py greedy streaming config")
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu")
    model = model.to(device)
    model.encoder.set_default_att_context_size([70, RIGHT_CONTEXT])
    decoder_class = apply_greedy_decoding_strategy(model, loop_labels=False)
    model.preprocessor.featurizer.dither = 0.0

    scfg = model.encoder.streaming_cfg
    preprocessor_cfg = model.cfg.preprocessor
    hop_length_sec = preprocessor_cfg.get("window_stride", 0.01)
    hop_samples = int(hop_length_sec * SAMPLE_RATE)
    shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, list) else scfg.shift_size
    pre_cache = scfg.pre_encode_cache_size
    pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, list) else pre_cache
    final_padding_frames = (RIGHT_CONTEXT + 1) * shift_frames
    drop_extra = scfg.drop_extra_pre_encoded

    cfg = ProbeConfig(
        device=device,
        shift_frames=shift_frames,
        pre_encode_cache_size=pre_encode_cache_size,
        drop_extra=drop_extra,
        hop_samples=hop_samples,
        final_padding_frames=final_padding_frames,
        decoder_class=decoder_class,
    )

    print(
        "Config: "
        f"att_context=[70,{RIGHT_CONTEXT}], shift_frames={cfg.shift_frames}, "
        f"pre_encode_cache_size={cfg.pre_encode_cache_size}, drop_extra={cfg.drop_extra}, "
        f"hop_samples={cfg.hop_samples}, final_padding_frames={cfg.final_padding_frames}, "
        f"decoder={cfg.decoder_class}"
    )
    return model, cfg


def warmup_like_server(model: Any, cfg: ProbeConfig) -> None:
    warmup_samples = SAMPLE_RATE + (cfg.final_padding_frames * cfg.hop_samples)
    warmup_audio = np.zeros(warmup_samples, dtype=np.float32)
    session = new_session(model, "warmup")
    session.accumulated_audio = warmup_audio

    audio_tensor = torch.from_numpy(warmup_audio).unsqueeze(0).to(cfg.device)
    audio_len = torch.tensor([len(warmup_audio)], device=cfg.device)
    with torch.inference_mode():
        mel, mel_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)
        _ = model.conformer_stream_step(
            processed_signal=mel,
            processed_signal_length=mel_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=True,
            previous_hypotheses=None,
            previous_pred_out=None,
            drop_extra_pre_encoded=0,
            return_transcription=True,
        )


def run_recipe(
    recipe: str,
    model: Any,
    cfg: ProbeConfig,
    base: ProbeSession,
    continuation_audio: np.ndarray,
) -> ProbeResult:
    parent = clone_session_deep(base, f"{recipe}-parent")
    control = clone_session_deep(base, f"{recipe}-control")
    if recipe == "shallow":
        fork = clone_session_shallow(parent, f"{recipe}-fork")
    elif recipe == "deep":
        fork = clone_session_deep(parent, f"{recipe}-fork")
    else:
        raise ValueError(recipe)

    before_cache = snapshot_cache(parent)
    before_hyps = freeze(parent.previous_hypotheses)
    before_text = parent.current_text

    # Serialized fork flush: this returns before the parent takes any next step.
    fork_delta = flush_final_on_fork(model, cfg, fork)

    cache_unchanged = cache_equal(before_cache, parent)
    after_hyps = freeze(parent.previous_hypotheses)
    hyp_diffs: list[str] = []
    hyps_unchanged = equal_snapshot(before_hyps, after_hyps, "previous_hypotheses", hyp_diffs)

    parent_texts = feed_audio(model, cfg, parent, continuation_audio)
    control_texts = feed_audio(model, cfg, control, continuation_audio)
    continued_bit_identical = (
        parent_texts == control_texts and parent.current_text == control.current_text
    )

    return ProbeResult(
        recipe=recipe,
        cache_unchanged=cache_unchanged,
        hyps_unchanged=hyps_unchanged,
        continued_bit_identical=continued_bit_identical,
        before_text=before_text,
        fork_delta=fork_delta,
        parent_continued_text=parent.current_text,
        control_continued_text=control.current_text,
        changed_paths=hyp_diffs[:12],
    )


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def print_result_table(label: str, results: list[ProbeResult]) -> None:
    print(f"\nProbe assertions: {label} (fork flush is serialized):")
    print("| recipe | parent cache byte-identical | parent Hypothesis fields unchanged | parent continuation bit-identical vs no-fork |")
    print("|---|---:|---:|---:|")
    for result in results:
        print(
            f"| {result.recipe} | {status(result.cache_unchanged)} | "
            f"{status(result.hyps_unchanged)} | {status(result.continued_bit_identical)} |"
        )

    for result in results:
        if result.changed_paths:
            joined = ", ".join(result.changed_paths[:6])
            print(f"{result.recipe} changed Hypothesis paths: {joined}")
        print(
            f"{result.recipe} text: before={result.before_text!r} "
            f"fork_delta={result.fork_delta!r} parent_continue={result.parent_continued_text!r} "
            f"control_continue={result.control_continued_text!r}"
        )


def main() -> int:
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)

    start = time.perf_counter()
    print(f"Server reference: {SERVER_PATH}")
    print("NeMo mutation sites that set clone depth / serialization requirement:")
    for cite, reason in NEMO_MUTATION_SITES:
        print(f"- {cite}: {reason}")

    model, cfg = load_model_and_config()
    warmup_like_server(model, cfg)

    audio_path = pick_audio_file(DEFAULT_AUDIO_DIR)
    audio_i16 = load_pcm_s16le(audio_path)
    required = int((PREFIX_SECONDS + CONTINUE_SECONDS) * SAMPLE_RATE)
    if len(audio_i16) < required:
        raise RuntimeError(
            f"Audio too short for probe: {audio_path} has {len(audio_i16) / SAMPLE_RATE:.2f}s, "
            f"need {(PREFIX_SECONDS + CONTINUE_SECONDS):.2f}s"
        )

    prefix_samples = int(PREFIX_SECONDS * SAMPLE_RATE)
    continue_samples = int(CONTINUE_SECONDS * SAMPLE_RATE)
    prefix_audio = to_float_audio(audio_i16[:prefix_samples])
    continuation_audio = to_float_audio(audio_i16[prefix_samples : prefix_samples + continue_samples])

    print(
        f"Audio: {audio_path} ({len(audio_i16) / SAMPLE_RATE:.2f}s), "
        f"prefix={PREFIX_SECONDS:.1f}s, continuation={CONTINUE_SECONDS:.1f}s, "
        f"sha256_prefix={sha_short(audio_i16[:prefix_samples].tobytes())}"
    )

    base = new_session(model, "base")
    prefix_texts = feed_audio(model, cfg, base, prefix_audio)
    if base.previous_hypotheses is None:
        raise RuntimeError("Prefix did not populate previous_hypotheses; cannot probe aliasing.")
    print(
        f"Base populated: emitted_frames={base.emitted_frames}, "
        f"hypotheses={len(base.previous_hypotheses)}, "
        f"interim_updates={len(prefix_texts)}, current_text={base.current_text!r}"
    )
    print(
        "Deep clone recipe: cache tensors use .detach().clone(); "
        "previous_hypotheses is a new list of tensor-aware deep-copied Hypothesis "
        "objects; every nested tensor field, including tuple/list dec_state, is "
        "detached and cloned; pred_out_stream is recursively cloned; model/websocket "
        "objects are never copied."
    )

    detector_selftest = run_detector_selftest(base)
    if not detector_selftest.passed:
        print(
            "DOC-CORRECTION (queue Step 9): finding doc deep-stack risk #1 cites "
            "rnnt_greedy_decoding.py:825-831 hyp.merge_, which is in GreedyBatchedRNNTInfer "
            "(loop_labels=True); server.py:158-166 uses loop_labels=False -> GreedyRNNTInfer, "
            "so that specific in-place merge does NOT occur on the configured path. The "
            "deep-clone recipe is retained as defense-in-depth + the serialization requirement "
            "is independent and still mandatory."
        )
        print("\nGate-Ga FAIL: detector-selftest=FAIL; probe detector is invalid.")
        print(f"Elapsed: {time.perf_counter() - start:.1f}s")
        return 3

    server_results = [
        run_recipe("shallow", model, cfg, base, continuation_audio),
        run_recipe("deep", model, cfg, base, continuation_audio),
    ]
    server_decoder_class = cfg.decoder_class
    print_result_table("loop_labels=False (server config)", server_results)

    print("\nSwitching decoder to loop_labels=True (batched, NOT server config) without reloading model.")
    cfg.decoder_class = apply_greedy_decoding_strategy(model, loop_labels=True, strategy="greedy_batch")
    print(f"Batched decoder active: decoder={cfg.decoder_class}, strategy=greedy_batch, loop_labels=True")
    batched_base = new_session(model, "batched-base")
    batched_prefix_texts = feed_audio(model, cfg, batched_base, prefix_audio)
    if batched_base.previous_hypotheses is None:
        raise RuntimeError("Batched prefix did not populate previous_hypotheses; cannot probe aliasing.")
    print(
        f"Batched base populated: emitted_frames={batched_base.emitted_frames}, "
        f"hypotheses={len(batched_base.previous_hypotheses)}, "
        f"interim_updates={len(batched_prefix_texts)}, current_text={batched_base.current_text!r}"
    )
    batched_results = [
        run_recipe("shallow", model, cfg, batched_base, continuation_audio),
        run_recipe("deep", model, cfg, batched_base, continuation_audio),
    ]
    batched_decoder_class = cfg.decoder_class
    print_result_table("loop_labels=True (batched, NOT server config)", batched_results)

    print("\nRestoring decoder to loop_labels=False (server config) after batched-path control.")
    cfg.decoder_class = apply_greedy_decoding_strategy(model, loop_labels=False)
    print(f"Restored decoder: decoder={cfg.decoder_class}, loop_labels=False")

    server_shallow = next(result for result in server_results if result.recipe == "shallow")
    server_deep = next(result for result in server_results if result.recipe == "deep")
    batched_shallow = next(result for result in batched_results if result.recipe == "shallow")
    batched_deep = next(result for result in batched_results if result.recipe == "deep")

    if not server_deep.clean:
        verdict = "Gate-Ga FAIL"
        server_reason = (
            "server-path deep-clone=FAIL; deep clone changed parent state or continuation "
            "under loop_labels=False"
        )
    else:
        verdict = "Gate-Ga PASS"
        server_reason = (
            f"server-path deep-clone=PASS under {server_decoder_class}, loop_labels=False"
        )

    if not batched_shallow.clean and batched_deep.clean:
        batched_observation = (
            f"batched-path observation=shallow FAIL / deep PASS under {batched_decoder_class}, "
            "showing the in-place batched hazard is caught by the probe"
        )
    elif batched_shallow.clean and batched_deep.clean:
        batched_observation = (
            f"batched-path observation=shallow PASS / deep PASS under {batched_decoder_class}; "
            "the cited batched in-place merge did not corrupt this sampled state"
        )
    elif not batched_deep.clean:
        batched_observation = (
            f"batched-path observation=deep FAIL under {batched_decoder_class}; "
            "not a Gate-Ga blocker because server config uses loop_labels=False, but it is a hazard note"
        )
    else:
        batched_observation = (
            f"batched-path observation=shallow PASS / deep PASS={status(batched_deep.clean)} "
            f"under {batched_decoder_class}"
        )

    print(
        "DOC-CORRECTION (queue Step 9): finding doc deep-stack risk #1 cites "
        "rnnt_greedy_decoding.py:825-831 hyp.merge_, which is in GreedyBatchedRNNTInfer "
        "(loop_labels=True); server.py:158-166 uses loop_labels=False -> GreedyRNNTInfer, "
        "so that specific in-place merge does NOT occur on the configured path. The "
        "deep-clone recipe is retained as defense-in-depth + the serialization requirement "
        "is independent and still mandatory."
    )
    print(
        f"\n{verdict}: detector-selftest={status(detector_selftest.passed)}; "
        f"{server_reason}; {batched_observation}. "
        "Correctness claim is serialized-only: fork flush must hold the existing inference_lock; "
        "there can be no concurrent parent step because NeMo mutates model-global decoder/joint "
        "train/eval state and encoder.streaming_cfg.drop_extra_pre_encoded."
    )
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")
    return 0 if detector_selftest.passed and server_deep.clean else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
