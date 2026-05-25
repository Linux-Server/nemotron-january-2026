#!/usr/bin/env python3
"""1.3 finalize reference: single-stream continuous FINALIZE executable spec.

This intentionally mirrors the production continuous-finalize path in
src/nemotron_speech/server.py, but keeps it synchronous and single-stream:

* STREAMING / PENDING_FINALIZE / FINALIZED state transitions.
* normal streaming uses keep_all_outputs=False.
* finalize forks the live state, appends right-context silence, runs one
  keep_all_outputs=True remainder encoder pass, and continues greedy RNNT decode
  with the carried decoder state.
* FORK_ASSERT proves the parent cache/decoder/hyp state is byte-identical after
  flushing the fork.

Run:
  HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python finalize_ref.py
"""
from __future__ import annotations

import copy
import dataclasses
import io
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

import nemo.collections.asr as nemo_asr
from ref_decode import ref_greedy, ref_greedy_range


STREAMING = "STREAMING"
PENDING_FINALIZE = "PENDING_FINALIZE"
FINALIZED = "FINALIZED"

BLANK = 1024
MAX_SYMBOLS = 10
FINALIZE_SILENCE_MS = 150
RIGHT_CONTEXT = 1
CANARY_INDICES = (4, 9, 2, 3)


def tensor_clone(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if tensor is None else tensor.detach().clone()


def clone_tree(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Tensor-aware deepcopy for disposable ASR state."""
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
        out: list[Any] = []
        memo[oid] = out
        out.extend(clone_tree(item, memo) for item in obj)
        return out
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        out = tuple(clone_tree(item, memo) for item in obj)
        memo[oid] = out
        return out
    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        memo[oid] = out
        for key, value in obj.items():
            out[clone_tree(key, memo)] = clone_tree(value, memo)
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = copy.copy(obj)
        memo[oid] = out
        for fld in dataclasses.fields(obj):
            setattr(out, fld.name, clone_tree(getattr(obj, fld.name), memo))
        return out
    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def assert_tree_equal(name: str, expected: Any, actual: Any) -> None:
    """Byte-exact recursive equality check for FORK_ASSERT state."""
    if torch.is_tensor(expected) or torch.is_tensor(actual):
        if not (torch.is_tensor(expected) and torch.is_tensor(actual)):
            raise AssertionError(f"{name}: tensor/non-tensor mismatch")
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(
                f"{name}: tensor metadata mismatch {expected.shape}/{expected.dtype} "
                f"vs {actual.shape}/{actual.dtype}"
            )
        if not torch.equal(expected, actual):
            raise AssertionError(f"{name}: tensor values differ")
        return
    if isinstance(expected, np.ndarray) or isinstance(actual, np.ndarray):
        if not (isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray)):
            raise AssertionError(f"{name}: ndarray/non-ndarray mismatch")
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(f"{name}: ndarray metadata mismatch")
        if not np.array_equal(expected, actual):
            raise AssertionError(f"{name}: ndarray values differ")
        return
    if isinstance(expected, (list, tuple)) or isinstance(actual, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise AssertionError(f"{name}: sequence mismatch")
        for index, (lhs, rhs) in enumerate(zip(expected, actual)):
            assert_tree_equal(f"{name}[{index}]", lhs, rhs)
        return
    if isinstance(expected, dict) or isinstance(actual, dict):
        if not (isinstance(expected, dict) and isinstance(actual, dict)):
            raise AssertionError(f"{name}: dict/non-dict mismatch")
        if expected.keys() != actual.keys():
            raise AssertionError(f"{name}: dict keys differ")
        for key in expected:
            assert_tree_equal(f"{name}.{key}", expected[key], actual[key])
        return
    if dataclasses.is_dataclass(expected) or dataclasses.is_dataclass(actual):
        if type(expected) is not type(actual):
            raise AssertionError(f"{name}: dataclass type mismatch")
        for fld in dataclasses.fields(expected):
            assert_tree_equal(
                f"{name}.{fld.name}",
                getattr(expected, fld.name),
                getattr(actual, fld.name),
            )
        return
    if expected != actual:
        raise AssertionError(f"{name}: {expected!r} != {actual!r}")


def _continuous_append_only_delta(final_text: str, emitted_text: str) -> str:
    """Return the collector-safe word suffix to append for a cumulative final."""
    final_tokens = final_text.split()
    emitted_tokens = emitted_text.split()

    common = 0
    for emitted_token, final_token in zip(emitted_tokens, final_tokens):
        if emitted_token != final_token:
            break
        common += 1

    if common == len(emitted_tokens):
        delta_tokens = final_tokens[common:]
    elif len(final_tokens) <= len(emitted_tokens):
        delta_tokens = []
    else:
        delta_tokens = final_tokens[len(emitted_tokens) :]
        max_overlap = min(len(emitted_tokens), len(delta_tokens))
        for overlap in range(max_overlap, 0, -1):
            if emitted_tokens[-overlap:] == delta_tokens[:overlap]:
                delta_tokens = delta_tokens[overlap:]
                break

    return " ".join(delta_tokens)


def _load_normalizer():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        return lambda text: normalizer(text).strip()
    except Exception:
        return lambda text: re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def load_model():
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
                    "max_symbols": MAX_SYMBOLS,
                    "loop_labels": True,
                    "use_cuda_graph_decoder": False,
                },
            }
        )
    )
    return model


@dataclass(frozen=True)
class RuntimeGeometry:
    shift_frames: int
    pre_encode_cache_size: int
    drop_extra: int
    final_padding_frames: int
    hop_samples: int
    window_size_samples: int
    raw_audio_ring_samples: int
    preprocess_align_pad_samples: int
    preprocess_new_audio_samples: int
    stream_preprocess_valid_samples: int
    first_preprocess_mel_frame: int
    constant_preprocess_frames: int
    constant_preprocess_samples: int


@dataclass
class FinalizeInputs:
    chunk_mel: torch.Tensor
    chunk_len: torch.Tensor
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    drop_extra: int
    new_mel: torch.Tensor
    remaining_frames: int
    padded_total_samples: int


@dataclass
class FinalizeResult:
    final_text: str
    delta_text: str
    steady_text: str
    final_tokens: list[int]
    steady_tokens: list[int]
    fork_assert_passed: bool
    state_after: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinuousSession:
    id: str
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    decoder_state: Any
    pred_out_stream: torch.Tensor
    pending_audio: np.ndarray
    raw_audio_ring: np.ndarray
    state: str = STREAMING
    debounce_armed: bool = False
    continuous_stop_seq: int = 0
    continuous_reset_seen: bool = False
    continuous_post_stop_audio: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    total_audio_samples: int = 0
    synthetic_prefix_samples: int = 0
    mel_frame_ring: Optional[torch.Tensor] = None
    emitted_frames: int = 0
    hyp_tokens: list[int] = field(default_factory=list)
    current_text: str = ""
    last_emitted_text: str = ""
    committed_text: str = ""
    continuous_emitted_text: str = ""
    last_finalize_fork: Optional["ContinuousSession"] = None


class ContinuousFinalizeRef:
    def __init__(self, model):
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.joint = model.joint
        self.tokenizer = model.tokenizer
        self.device = next(model.parameters()).device
        self.geometry = self._derive_geometry()

    def _derive_geometry(self) -> RuntimeGeometry:
        scfg = self.encoder.streaming_cfg
        to_int = lambda value: int(value[1]) if isinstance(value, (list, tuple)) else int(value)
        shift_frames = to_int(scfg.shift_size)
        pre_encode_cache_size = to_int(scfg.pre_encode_cache_size)
        drop_extra = int(scfg.drop_extra_pre_encoded)
        featurizer = self.model.preprocessor.featurizer
        hop_samples = int(getattr(featurizer, "hop_length", 160))
        window_size_samples = int(getattr(featurizer, "win_length", 400))
        raw_audio_ring_samples = window_size_samples - hop_samples
        preprocess_align_pad_samples = (
            hop_samples - (raw_audio_ring_samples % hop_samples)
        ) % hop_samples
        preprocess_new_audio_samples = (shift_frames + 1) * hop_samples
        stream_preprocess_valid_samples = (
            preprocess_align_pad_samples
            + raw_audio_ring_samples
            + preprocess_new_audio_samples
        )
        prefix_samples = preprocess_align_pad_samples + raw_audio_ring_samples
        if prefix_samples % hop_samples != 0:
            raise RuntimeError("fixed preprocessor prefix is not hop-aligned")
        first_preprocess_mel_frame = prefix_samples // hop_samples
        final_padding_frames = (RIGHT_CONTEXT + 1) * shift_frames
        min_plan_frames = (
            first_preprocess_mel_frame
            + pre_encode_cache_size
            + shift_frames
            + final_padding_frames
            + 1
        )
        constant_preprocess_frames = 1 << (min_plan_frames - 1).bit_length()
        constant_preprocess_samples = (constant_preprocess_frames - 1) * hop_samples
        return RuntimeGeometry(
            shift_frames=shift_frames,
            pre_encode_cache_size=pre_encode_cache_size,
            drop_extra=drop_extra,
            final_padding_frames=final_padding_frames,
            hop_samples=hop_samples,
            window_size_samples=window_size_samples,
            raw_audio_ring_samples=raw_audio_ring_samples,
            preprocess_align_pad_samples=preprocess_align_pad_samples,
            preprocess_new_audio_samples=preprocess_new_audio_samples,
            stream_preprocess_valid_samples=stream_preprocess_valid_samples,
            first_preprocess_mel_frame=first_preprocess_mel_frame,
            constant_preprocess_frames=constant_preprocess_frames,
            constant_preprocess_samples=constant_preprocess_samples,
        )

    def new_session(self, session_id: str = "s0") -> ContinuousSession:
        cache = self.encoder.get_initial_cache_state(batch_size=1)
        state = self.decoder.initialize_state(
            torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        )
        pred_out, state = self.decoder.predict(
            None,
            state,
            add_sos=False,
            batch_size=1,
        )
        return ContinuousSession(
            id=session_id,
            cache_last_channel=cache[0].detach().clone(),
            cache_last_time=cache[1].detach().clone(),
            cache_last_channel_len=cache[2].detach().clone(),
            decoder_state=clone_tree(state),
            pred_out_stream=pred_out.detach().clone(),
            pending_audio=np.array([], dtype=np.float32),
            raw_audio_ring=np.zeros(
                self.geometry.raw_audio_ring_samples,
                dtype=np.float32,
            ),
        )

    def _session_ready(self, session: ContinuousSession) -> bool:
        g = self.geometry
        timeline_samples = session.synthetic_prefix_samples + session.total_audio_samples
        needed_samples = (session.emitted_frames + g.shift_frames + 1) * g.hop_samples
        return (
            timeline_samples >= needed_samples
            and len(session.pending_audio) >= g.preprocess_new_audio_samples
        )

    def append_audio(self, session: ContinuousSession, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"expected mono audio, got shape={audio.shape}")
        if session.state == PENDING_FINALIZE:
            session.continuous_post_stop_audio = np.concatenate(
                [session.continuous_post_stop_audio, audio]
            ).astype(np.float32, copy=False)
            return
        if session.state != STREAMING:
            raise RuntimeError(f"append_audio in state={session.state}")
        session.pending_audio = np.concatenate([session.pending_audio, audio]).astype(
            np.float32,
            copy=False,
        )
        session.total_audio_samples += int(audio.shape[0])
        self.drain_steady(session)

    def drain_steady(self, session: ContinuousSession) -> None:
        while self._session_ready(session):
            self._process_one_steady_chunk(session)

    def vad_stop(self, session: ContinuousSession) -> None:
        if session.state == FINALIZED:
            raise RuntimeError("vad_stop called before finalize finish")
        session.continuous_stop_seq += 1
        session.state = PENDING_FINALIZE
        session.debounce_armed = True
        session.continuous_reset_seen = False

    def vad_start(self, session: ContinuousSession) -> None:
        if session.state == PENDING_FINALIZE:
            session.state = STREAMING
            session.debounce_armed = False
            session.continuous_reset_seen = False
            if len(session.continuous_post_stop_audio) > 0:
                held = session.continuous_post_stop_audio
                session.continuous_post_stop_audio = np.array([], dtype=np.float32)
                self.append_audio(session, held)
            return
        if session.state == STREAMING and len(session.continuous_post_stop_audio) > 0:
            held = session.continuous_post_stop_audio
            session.continuous_post_stop_audio = np.array([], dtype=np.float32)
            self.append_audio(session, held)

    def debounce_expire(self, session: ContinuousSession) -> FinalizeResult:
        if session.state != PENDING_FINALIZE:
            raise RuntimeError(f"debounce_expire in state={session.state}")
        session.debounce_armed = False
        session.state = FINALIZED
        result = self._finalize_and_emit(session, reason="debounce_expired")
        self._finish_speculative(session, result)
        return result

    def force_finalize_end(self, session: ContinuousSession) -> FinalizeResult:
        if len(session.continuous_post_stop_audio) > 0:
            held = session.continuous_post_stop_audio
            session.continuous_post_stop_audio = np.array([], dtype=np.float32)
            session.pending_audio = np.concatenate([session.pending_audio, held]).astype(
                np.float32,
                copy=False,
            )
            session.total_audio_samples += int(held.shape[0])
            self.drain_steady(session)
        session.debounce_armed = False
        session.state = FINALIZED
        result = self._finalize_and_emit(session, reason="end")
        self._cold_reset_after_finalize(session)
        return result

    def _build_fixed_preprocess_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        g = self.geometry
        if len(raw_audio_ring) != g.raw_audio_ring_samples:
            raise ValueError(
                f"expected raw ring {g.raw_audio_ring_samples}, got {len(raw_audio_ring)}"
            )
        prefix_len = g.preprocess_align_pad_samples + g.raw_audio_ring_samples
        valid_samples = prefix_len + len(new_audio)
        if valid_samples > g.constant_preprocess_samples:
            raise ValueError(
                f"fixed preprocessor valid span {valid_samples} exceeds "
                f"K={g.constant_preprocess_samples}"
            )
        audio = np.zeros(g.constant_preprocess_samples, dtype=np.float32)
        cursor = g.preprocess_align_pad_samples
        audio[cursor : cursor + g.raw_audio_ring_samples] = raw_audio_ring
        cursor += g.raw_audio_ring_samples
        audio[cursor : cursor + len(new_audio)] = new_audio
        return audio, valid_samples

    @torch.inference_mode()
    def _preprocess_fixed_audio(
        self,
        audio: np.ndarray,
        valid_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.geometry
        if len(audio) != g.constant_preprocess_samples:
            raise ValueError(
                f"expected fixed preprocessor input {g.constant_preprocess_samples}, "
                f"got {len(audio)}"
            )
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(
            self.device
        )
        audio_len = torch.tensor([valid_samples], device=self.device, dtype=torch.long)
        return self.model.preprocessor(input_signal=audio_tensor, length=audio_len)

    def _fixed_mel_from_new_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> torch.Tensor:
        fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
            raw_audio_ring,
            new_audio,
        )
        mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
        return mel

    def _update_mel_frame_ring(
        self,
        session: ContinuousSession,
        new_mel: torch.Tensor,
    ) -> None:
        g = self.geometry
        if session.mel_frame_ring is None:
            combined = new_mel.detach()
        else:
            combined = torch.cat((session.mel_frame_ring, new_mel.detach()), dim=-1)
        session.mel_frame_ring = combined[:, :, -g.pre_encode_cache_size :].detach()

    def _advance_raw_ring(self, raw_ring: np.ndarray, consumed_audio: np.ndarray) -> np.ndarray:
        g = self.geometry
        if len(consumed_audio) >= g.raw_audio_ring_samples:
            return consumed_audio[-g.raw_audio_ring_samples :].copy()
        if len(consumed_audio) == 0:
            return raw_ring
        keep = g.raw_audio_ring_samples - len(consumed_audio)
        return np.concatenate([raw_ring[-keep:], consumed_audio]).astype(
            np.float32,
            copy=False,
        )

    @torch.inference_mode()
    def _process_one_steady_chunk(self, session: ContinuousSession) -> None:
        g = self.geometry
        new_audio = session.pending_audio[: g.preprocess_new_audio_samples]
        mel = self._fixed_mel_from_new_audio(session.raw_audio_ring, new_audio)
        valid_new_mel = mel[
            :,
            :,
            g.first_preprocess_mel_frame : g.first_preprocess_mel_frame
            + g.shift_frames,
        ]

        if session.emitted_frames == 0:
            chunk_mel = valid_new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
            drop_extra = g.drop_extra

        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
        enc_out, enc_len, clc, clt, clcl = self.encoder.cache_aware_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_extra,
        )
        tokens, decoder_state, pred_out = ref_greedy_range(
            self.decoder,
            self.joint,
            enc_out.transpose(1, 2).contiguous(),
            0,
            int(enc_len[0]),
            session.decoder_state,
            session.pred_out_stream,
        )
        session.hyp_tokens.extend(tokens)
        session.decoder_state = decoder_state
        session.pred_out_stream = pred_out
        session.cache_last_channel = clc
        session.cache_last_time = clt
        session.cache_last_channel_len = clcl

        consumed_audio = session.pending_audio[: g.shift_frames * g.hop_samples]
        session.raw_audio_ring = self._advance_raw_ring(
            session.raw_audio_ring,
            consumed_audio,
        )
        session.pending_audio = session.pending_audio[g.shift_frames * g.hop_samples :]
        self._update_mel_frame_ring(session, valid_new_mel)
        session.emitted_frames += g.shift_frames
        session.current_text = self.tokenizer.ids_to_text(session.hyp_tokens)

    def _snapshot_fork_assert_parent(self, session: ContinuousSession) -> dict[str, Any]:
        return {
            "cache_last_channel": tensor_clone(session.cache_last_channel),
            "cache_last_time": tensor_clone(session.cache_last_time),
            "cache_last_channel_len": tensor_clone(session.cache_last_channel_len),
            "decoder_state": clone_tree(session.decoder_state),
            "pred_out_stream": tensor_clone(session.pred_out_stream),
            "hyp_tokens": list(session.hyp_tokens),
        }

    def _assert_fork_flush_parent_unchanged(
        self,
        session: ContinuousSession,
        snapshot: dict[str, Any],
    ) -> None:
        assert_tree_equal("cache_last_channel", snapshot["cache_last_channel"], session.cache_last_channel)
        assert_tree_equal("cache_last_time", snapshot["cache_last_time"], session.cache_last_time)
        assert_tree_equal(
            "cache_last_channel_len",
            snapshot["cache_last_channel_len"],
            session.cache_last_channel_len,
        )
        assert_tree_equal("decoder_state", snapshot["decoder_state"], session.decoder_state)
        assert_tree_equal("pred_out_stream", snapshot["pred_out_stream"], session.pred_out_stream)
        assert_tree_equal("hyp_tokens", snapshot["hyp_tokens"], session.hyp_tokens)

    def build_continuous_finalize_fork(
        self,
        session: ContinuousSession,
        append_padding: bool = True,
    ) -> ContinuousSession:
        pending_audio = session.pending_audio.copy()
        padding_samples = 0
        if append_padding and session.total_audio_samples > 0:
            padding_samples = (
                self.geometry.final_padding_frames * self.geometry.hop_samples
            )
            pending_audio = np.concatenate(
                [pending_audio, np.zeros(padding_samples, dtype=np.float32)]
            ).astype(np.float32, copy=False)

        fork = ContinuousSession(
            id=f"{session.id}:fork",
            cache_last_channel=tensor_clone(session.cache_last_channel),
            cache_last_time=tensor_clone(session.cache_last_time),
            cache_last_channel_len=tensor_clone(session.cache_last_channel_len),
            decoder_state=clone_tree(session.decoder_state),
            pred_out_stream=tensor_clone(session.pred_out_stream),
            pending_audio=pending_audio,
            raw_audio_ring=session.raw_audio_ring.copy(),
            state=session.state,
            debounce_armed=session.debounce_armed,
            continuous_stop_seq=session.continuous_stop_seq,
            continuous_reset_seen=session.continuous_reset_seen,
            total_audio_samples=session.total_audio_samples + padding_samples,
            synthetic_prefix_samples=session.synthetic_prefix_samples,
            mel_frame_ring=clone_tree(session.mel_frame_ring),
            emitted_frames=session.emitted_frames,
            hyp_tokens=list(session.hyp_tokens),
            current_text=session.current_text,
            last_emitted_text=session.last_emitted_text,
            committed_text=session.committed_text,
            continuous_emitted_text=session.continuous_emitted_text,
        )
        return fork

    def prepare_finalize_inputs(self, fork: ContinuousSession) -> Optional[FinalizeInputs]:
        g = self.geometry
        if len(fork.pending_audio) == 0:
            return None
        padded_total_samples = fork.emitted_frames * g.hop_samples + len(
            fork.pending_audio
        )
        total_mel_frames = padded_total_samples // g.hop_samples + 1
        remaining_frames = total_mel_frames - fork.emitted_frames
        if remaining_frames <= 0:
            return None

        pending = fork.pending_audio
        raw_ring = fork.raw_audio_ring.copy()
        new_mels: list[torch.Tensor] = []
        frames_collected = 0
        while frames_collected < remaining_frames:
            frames_this_call = min(
                g.shift_frames,
                remaining_frames - frames_collected,
            )
            needed_new_samples = min(len(pending), g.preprocess_new_audio_samples)
            new_audio = pending[:needed_new_samples]
            mel = self._fixed_mel_from_new_audio(raw_ring, new_audio)
            start = g.first_preprocess_mel_frame
            new_mels.append(mel[:, :, start : start + frames_this_call])

            if frames_this_call == g.shift_frames:
                consumed_samples = min(g.shift_frames * g.hop_samples, len(pending))
                consumed_audio = pending[:consumed_samples]
                raw_ring = self._advance_raw_ring(raw_ring, consumed_audio)
                pending = pending[consumed_samples:]
            frames_collected += frames_this_call

        new_mel = torch.cat(new_mels, dim=-1)
        if fork.emitted_frames == 0:
            chunk_mel = new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((fork.mel_frame_ring, new_mel), dim=-1)
            drop_extra = g.drop_extra

        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
        return FinalizeInputs(
            chunk_mel=chunk_mel,
            chunk_len=chunk_len,
            cache_last_channel=tensor_clone(fork.cache_last_channel),
            cache_last_time=tensor_clone(fork.cache_last_time),
            cache_last_channel_len=tensor_clone(fork.cache_last_channel_len),
            drop_extra=int(drop_extra),
            new_mel=new_mel,
            remaining_frames=int(remaining_frames),
            padded_total_samples=int(padded_total_samples),
        )

    @torch.inference_mode()
    def flush_finalize_fork(self, fork: ContinuousSession) -> dict[str, Any]:
        g = self.geometry
        inputs = self.prepare_finalize_inputs(fork)
        if inputs is None:
            return {
                "final_text": fork.current_text,
                "final_tokens": list(fork.hyp_tokens),
                "inputs": None,
                "encoder_outputs": None,
            }

        enc_out, enc_len, clc, clt, clcl = self.encoder.cache_aware_stream_step(
            processed_signal=inputs.chunk_mel,
            processed_signal_length=inputs.chunk_len,
            cache_last_channel=inputs.cache_last_channel,
            cache_last_time=inputs.cache_last_time,
            cache_last_channel_len=inputs.cache_last_channel_len,
            keep_all_outputs=True,
            drop_extra_pre_encoded=inputs.drop_extra,
        )
        tokens, decoder_state, pred_out = ref_greedy_range(
            self.decoder,
            self.joint,
            enc_out.transpose(1, 2).contiguous(),
            0,
            int(enc_len[0]),
            fork.decoder_state,
            fork.pred_out_stream,
        )
        fork.hyp_tokens.extend(tokens)
        fork.decoder_state = decoder_state
        fork.pred_out_stream = pred_out
        fork.cache_last_channel = clc
        fork.cache_last_time = clt
        fork.cache_last_channel_len = clcl
        fork.current_text = self.tokenizer.ids_to_text(fork.hyp_tokens)

        finalized_audio = fork.pending_audio
        fork.emitted_frames += inputs.remaining_frames
        fork.pending_audio = np.array([], dtype=np.float32)
        self._update_mel_frame_ring(fork, inputs.new_mel)
        if len(finalized_audio) >= g.raw_audio_ring_samples:
            fork.raw_audio_ring = finalized_audio[-g.raw_audio_ring_samples :].copy()
        elif len(finalized_audio) > 0:
            fork.raw_audio_ring = self._advance_raw_ring(
                fork.raw_audio_ring,
                finalized_audio,
            )

        return {
            "final_text": fork.current_text,
            "final_tokens": list(fork.hyp_tokens),
            "new_tokens": tokens,
            "inputs": inputs,
            "encoder_outputs": (enc_out, enc_len, clc, clt, clcl),
        }

    def _finalize_and_emit(
        self,
        session: ContinuousSession,
        *,
        reason: str,
    ) -> FinalizeResult:
        parent_snapshot = self._snapshot_fork_assert_parent(session)
        fork = self.build_continuous_finalize_fork(session)
        steady_tokens = list(session.hyp_tokens)
        steady_text = session.current_text
        flush = self.flush_finalize_fork(fork)
        self._assert_fork_flush_parent_unchanged(session, parent_snapshot)

        final_text = flush["final_text"]
        delta_text = _continuous_append_only_delta(
            final_text,
            session.continuous_emitted_text,
        )
        session.committed_text = final_text
        session.last_emitted_text = final_text
        if delta_text:
            session.continuous_emitted_text = (
                session.continuous_emitted_text + " " + delta_text
            ).strip()

        session.last_finalize_fork = fork
        return FinalizeResult(
            final_text=final_text,
            delta_text=delta_text,
            steady_text=steady_text,
            final_tokens=list(fork.hyp_tokens),
            steady_tokens=steady_tokens,
            fork_assert_passed=True,
            state_after=session.state,
            meta={
                "reason": reason,
                "remaining_frames": (
                    None if flush["inputs"] is None else flush["inputs"].remaining_frames
                ),
                "final_chunk_T": (
                    None if flush["inputs"] is None else int(flush["inputs"].chunk_mel.shape[-1])
                ),
                "drop_extra": (
                    None if flush["inputs"] is None else int(flush["inputs"].drop_extra)
                ),
            },
        )

    def _copy_finalized_fork_state(
        self,
        session: ContinuousSession,
        fork: ContinuousSession,
    ) -> None:
        session.cache_last_channel = tensor_clone(fork.cache_last_channel)
        session.cache_last_time = tensor_clone(fork.cache_last_time)
        session.cache_last_channel_len = tensor_clone(fork.cache_last_channel_len)
        session.decoder_state = clone_tree(fork.decoder_state)
        session.pred_out_stream = tensor_clone(fork.pred_out_stream)
        session.mel_frame_ring = clone_tree(fork.mel_frame_ring)
        session.emitted_frames = fork.emitted_frames
        session.raw_audio_ring = fork.raw_audio_ring.copy()
        session.pending_audio = fork.pending_audio.copy()
        session.total_audio_samples = fork.total_audio_samples
        session.hyp_tokens = list(fork.hyp_tokens)
        session.current_text = fork.current_text

    def _finish_speculative(
        self,
        session: ContinuousSession,
        result: FinalizeResult,
    ) -> None:
        if session.last_finalize_fork is not None:
            self._copy_finalized_fork_state(session, session.last_finalize_fork)
        session.state = STREAMING
        session.debounce_armed = False
        session.continuous_reset_seen = False
        result.state_after = session.state

    def _cold_reset_after_finalize(self, session: ContinuousSession) -> None:
        committed = session.committed_text
        emitted = session.continuous_emitted_text
        fresh = self.new_session(session.id)
        session.cache_last_channel = fresh.cache_last_channel
        session.cache_last_time = fresh.cache_last_time
        session.cache_last_channel_len = fresh.cache_last_channel_len
        session.decoder_state = fresh.decoder_state
        session.pred_out_stream = fresh.pred_out_stream
        session.pending_audio = fresh.pending_audio
        session.raw_audio_ring = fresh.raw_audio_ring
        session.mel_frame_ring = None
        session.emitted_frames = 0
        session.hyp_tokens = []
        session.current_text = ""
        session.total_audio_samples = 0
        session.synthetic_prefix_samples = 0
        session.continuous_post_stop_audio = np.array([], dtype=np.float32)
        session.committed_text = ""
        session.last_emitted_text = ""
        session.continuous_emitted_text = ""
        session.last_finalize_fork = None
        session.state = STREAMING
        session.debounce_armed = False
        session.continuous_reset_seen = False
        _ = committed, emitted

    @torch.inference_mode()
    def full_greedy_tokens(self, wav: np.ndarray) -> list[int]:
        audio = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)
        audio_len = torch.tensor([wav.shape[0]], dtype=torch.long, device=self.device)
        enc, enc_len = self.model.forward(
            input_signal=audio,
            input_signal_length=audio_len,
        )
        return ref_greedy(self.decoder, self.joint, enc, enc_len)

    def text(self, tokens: list[int]) -> str:
        return self.tokenizer.ids_to_text(tokens)


def load_benchmark_dataset():
    import datasets

    return datasets.load_dataset(
        "pipecat-ai/stt-benchmark-data",
        split="train",
    ).cast_column("audio", datasets.Audio(decode=False))


def load_wav(example: dict[str, Any]) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(example["audio"]["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        n = int(len(wav) * 16000 / sr)
        wav = np.interp(
            np.linspace(0, len(wav), n, endpoint=False),
            np.arange(len(wav)),
            wav,
        ).astype(np.float32)
    return np.asarray(wav, dtype=np.float32)


def run_single_finalize(rt: ContinuousFinalizeRef, wav: np.ndarray, session_id: str) -> FinalizeResult:
    session = rt.new_session(session_id)
    rt.append_audio(session, wav)
    rt.vad_stop(session)
    return rt.debounce_expire(session)


def validation_gate() -> bool:
    print("loading model + stt-benchmark canaries...")
    model = load_model()
    rt = ContinuousFinalizeRef(model)
    norm = _load_normalizer()
    ds = load_benchmark_dataset()
    g = rt.geometry
    print(
        "geometry "
        f"shift={g.shift_frames} pre={g.pre_encode_cache_size} drop={g.drop_extra} "
        f"final_padding_frames={g.final_padding_frames} "
        f"fixed_preproc_samples={g.constant_preprocess_samples}"
    )

    canary_ok = True
    exact_count = 0
    wer_equiv_count = 0
    recovered_count = 0
    print("\nT1 finalize canary:")
    for sample_index in CANARY_INDICES:
        ex = ds[sample_index]
        wav = load_wav(ex)
        result = run_single_finalize(rt, wav, f"canary-{sample_index}")
        full_tokens = rt.full_greedy_tokens(wav)
        full_text = rt.text(full_tokens)
        exact = result.final_tokens == full_tokens
        wer_equiv = norm(result.final_text) == norm(full_text)
        recovered = result.final_tokens != result.steady_tokens
        exact_count += int(exact)
        wer_equiv_count += int(wer_equiv)
        recovered_count += int(recovered)
        canary_ok = canary_ok and (exact or wer_equiv)
        status = "PASS" if (exact or wer_equiv) else "FAIL"
        print(
            f"  [{status}] idx={sample_index} id={ex.get('sample_id')} "
            f"steady/final/full tok={len(result.steady_tokens)}/"
            f"{len(result.final_tokens)}/{len(full_tokens)} "
            f"exact={exact} wer_equiv={wer_equiv} recovered={recovered} "
            f"T={result.meta['final_chunk_T']} drop={result.meta['drop_extra']}"
        )
        print(f"    steady  : {result.steady_text!r}")
        print(f"    finalize: {result.final_text!r}")
        print(f"    full    : {full_text!r}")

    print(
        f"  summary: canary={'PASS' if canary_ok else 'FAIL'} "
        f"token_exact={exact_count}/{len(CANARY_INDICES)} "
        f"wer_equiv={wer_equiv_count}/{len(CANARY_INDICES)} "
        f"visible_recovery={recovered_count}/{len(CANARY_INDICES)}"
    )

    print("\nFORK_ASSERT:")
    fork_assert_ok = True
    ex = ds[CANARY_INDICES[0]]
    result = run_single_finalize(rt, load_wav(ex), "fork-assert")
    fork_assert_ok = result.fork_assert_passed
    print(f"  {'PASS' if fork_assert_ok else 'FAIL'} parent byte-identical after fork flush")

    print("\nreset/resume:")
    turn_a = load_wav(ds[CANARY_INDICES[0]])
    turn_b = load_wav(ds[CANARY_INDICES[1]])
    full_a = rt.text(rt.full_greedy_tokens(turn_a))
    full_b = rt.text(rt.full_greedy_tokens(turn_b))

    spec = rt.new_session("speculative")
    rt.append_audio(spec, turn_a)
    rt.vad_stop(spec)
    spec_a = rt.debounce_expire(spec)
    rt.append_audio(spec, turn_b)
    rt.vad_stop(spec)
    spec_b = rt.debounce_expire(spec)
    spec_ok = (
        spec_a.state_after == STREAMING
        and spec_b.state_after == STREAMING
        and bool(spec_a.delta_text)
        and bool(spec_b.delta_text)
        and norm(spec_a.delta_text) == norm(full_a)
        and norm(spec_b.delta_text) == norm(full_b)
    )
    print(
        f"  [{'PASS' if spec_ok else 'FAIL'}] speculative context-retained "
        f"delta tok={len(spec_a.delta_text.split())}/{len(spec_b.delta_text.split())} "
        f"state={spec.state}"
    )
    print(f"    turn1 delta: {spec_a.delta_text!r}")
    print(f"    turn2 delta: {spec_b.delta_text!r}")

    cold = rt.new_session("cold")
    rt.append_audio(cold, turn_a)
    rt.vad_stop(cold)
    cold_a = rt.force_finalize_end(cold)
    rt.append_audio(cold, turn_b)
    rt.vad_stop(cold)
    cold_b = rt.force_finalize_end(cold)
    cold_ok = (
        cold.state == STREAMING
        and norm(cold_a.delta_text) == norm(full_a)
        and norm(cold_b.delta_text) == norm(full_b)
    )
    print(
        f"  [{'PASS' if cold_ok else 'FAIL'}] cold true-boundary reset "
        f"delta tok={len(cold_a.delta_text.split())}/{len(cold_b.delta_text.split())} "
        f"state={cold.state}"
    )
    print(f"    turn1 delta: {cold_a.delta_text!r}")
    print(f"    turn2 delta: {cold_b.delta_text!r}")

    ok = canary_ok and fork_assert_ok and spec_ok and cold_ok
    print(f"\n=== FINALIZE_REF {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if validation_gate() else 1)
