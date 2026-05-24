#!/usr/bin/env python3
"""In-process decoder batch-composition invariance harness.

Step 1 for proj-2026-05-23-1731. This intentionally imports the real server and
drives its private in-process batch methods; it does not reimplement
stack/scatter or edit server/NeMo code.

Run with:
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python \
    proj-2026-05-23-1731/decoder_graph_harness.py
"""

from __future__ import annotations

import argparse
import asyncio
import ast
import contextlib
import copy
import dataclasses
import gc
import hashlib
import inspect
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Iterable
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent
SRC = REPO / "src"
DEFAULT_AUDIO_DIR = REPO / "proj-2026-05-20-modal-cost" / "loadgen_audio"
DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_SESSION_COUNT = 4
DEFAULT_NORMAL_CHUNKS = 20
DEFAULT_FINAL_TAIL_SAMPLES = 4000
DEFAULT_CONC10_RECORDS = REPO / "ec2-bench" / "leaderboard_decomp_prod_l40s_full1000_c10.records"
DEFAULT_CONC10_SRVLOG = REPO / "ec2-bench" / "leaderboard_decomp_prod_l40s_full1000_c10.srvlog"
ATOL_CANDIDATES = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3)
RTOL = 1.0e-5

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclasses.dataclass(frozen=True)
class ClipSpec:
    session_id: str
    path: Path
    pcm_i16: np.ndarray
    total_samples_used: int


@dataclasses.dataclass(frozen=True)
class ScenarioPlan:
    name: str
    normal_groups_by_chunk: list[list[list[str]]]
    final_groups: list[list[str]]


@dataclasses.dataclass
class SessionRuntime:
    session: Any
    clip: ClipSpec
    audio_cursor: int = 0
    next_chunk_index: int = 0


@dataclasses.dataclass
class FloatCompareSummary:
    tensor_pairs: int = 0
    byte_equal: bool = True
    allclose_1e_4: bool = True
    max_abs: float = 0.0
    max_rel: float = 0.0
    max_path: str = ""
    min_observed_atol: Optional[float] = 0.0
    mismatches: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class CompareSummary:
    reference: str
    candidates: list[str]
    event_count: int
    token_text_invariant: bool
    token_mismatches: list[str]
    float_state: FloatCompareSummary


@dataclasses.dataclass
class NeedReinitStats:
    need_reinit_calls: int = 0
    need_reinit_true: int = 0
    graph_reinitialize_calls: int = 0
    need_reinit_shapes: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    graph_reinitialize_shapes: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DecodeCase:
    name: str
    phase: str
    encoder_output: torch.Tensor
    encoded_lengths: torch.Tensor
    partial_hypotheses: Any

    @property
    def batch_size(self) -> int:
        return int(self.encoder_output.shape[0])

    @property
    def encoded_time(self) -> int:
        return int(self.encoder_output.shape[-1])


def configure_environment(*, lanes: int, batch_max_size: int) -> dict[str, str]:
    """Force the Step-1 server config before ASRServer is instantiated."""
    forced = {
        "NEMOTRON_CONTINUOUS": "1",
        "NEMOTRON_SCHEDULER_B1": "1",
        "NEMOTRON_BATCH_SCHED": "1",
        "NEMOTRON_BATCH_FINALIZE": "1",
        "NEMOTRON_BATCH_MAX_SIZE": str(batch_max_size),
        "NEMOTRON_MODEL_LANES": str(lanes),
        "NEMOTRON_FINALIZE_SILENCE_MS": "0",
        "NEMOTRON_WARMUP_MS": "200",
        "NEMOTRON_DECODING": "greedy",
        "NEMOTRON_TARGET_LANG": "en-US",
        "NEMOTRON_FORK_ASSERT": "1",
    }
    disabled = (
        "NEMOTRON_EOU_PROBE",
        "NEMOTRON_ENCODER_COMPILE",
        "NEMOTRON_ENCODER_CUDAGRAPH",
        "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE",
        "NEMOTRON_BATCH_FINALIZE_PREPROC",
    )
    for name in disabled:
        os.environ.pop(name, None)
    os.environ.update(forced)
    torch.backends.cudnn.benchmark = False
    return forced


def import_server_classes() -> tuple[Any, Any]:
    from nemotron_speech.server import ASRServer, ASRSession

    return ASRServer, ASRSession


def nemo_identity() -> dict[str, str]:
    import nemo

    return {
        "path": inspect.getfile(nemo),
        "version": str(getattr(nemo, "__version__", "unknown")),
    }


def build_server(
    *,
    model: str = DEFAULT_MODEL,
    lanes: int = 1,
    batch_max_size: int = 32,
    right_context: int = 1,
) -> Any:
    """Instantiate and load the real ASRServer with graph-OFF greedy_batch."""
    configure_environment(lanes=lanes, batch_max_size=batch_max_size)
    ASRServer, _ASRSession = import_server_classes()
    server = ASRServer(model=model, host="127.0.0.1", port=0, right_context=right_context)
    server.load_model()
    server.model_loaded = True
    if server.model_lanes > 1:
        server._ensure_scheduler_model_lane_resources()
    return server


def _read_manifest(audio_dir: Path) -> list[tuple[str, float]]:
    manifest_path = audio_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing audio manifest: {manifest_path}")
    data = json.loads(manifest_path.read_text())
    out: list[tuple[str, float]] = []
    for row in data:
        if not isinstance(row, list) or len(row) < 2:
            continue
        out.append((str(row[0]), float(row[1])))
    return out


def select_audio_clips(
    server: Any,
    *,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    session_count: int = DEFAULT_SESSION_COUNT,
    normal_chunks: int = DEFAULT_NORMAL_CHUNKS,
    final_tail_samples: int = DEFAULT_FINAL_TAIL_SAMPLES,
    allow_reuse: bool = False,
    session_prefix: str = "s",
) -> list[ClipSpec]:
    """Pick distinct real clips long enough for the fixed normal+final trace."""
    if session_count < 1:
        raise ValueError("session_count must be >= 1")
    first = int(server.preprocess_new_audio_samples)
    shift = int(server.shift_frames) * int(server.hop_samples)
    total_needed = first + max(0, int(normal_chunks) - 1) * shift + int(final_tail_samples)
    candidates: list[tuple[float, str, Path, int]] = []
    for sample_id, duration in _read_manifest(audio_dir):
        path = audio_dir / f"{sample_id}.pcm"
        if not path.exists():
            continue
        sample_count = path.stat().st_size // np.dtype(np.int16).itemsize
        if sample_count >= total_needed:
            candidates.append((duration, sample_id, path, sample_count))
    if len(candidates) < session_count and not allow_reuse:
        raise RuntimeError(
            f"only found {len(candidates)} clips with >= {total_needed} samples; "
            f"need {session_count}"
        )
    if not candidates:
        raise RuntimeError(f"found no clips with >= {total_needed} samples")

    # Prefer longer clips so the chosen prefix is real speech/audio, not a tiny
    # edge case. Keep session IDs stable and short for comparison output.
    base_selected = sorted(candidates, reverse=True)
    if allow_reuse and len(base_selected) < session_count:
        selected = [base_selected[index % len(base_selected)] for index in range(session_count)]
    else:
        selected = base_selected[:session_count]
        selected = sorted(selected, key=lambda row: row[1])
    clips: list[ClipSpec] = []
    for index, (_duration, _sample_id, path, _samples) in enumerate(selected):
        pcm = np.fromfile(path, dtype=np.int16)
        clips.append(
            ClipSpec(
                session_id=f"{session_prefix}{index}",
                path=path,
                pcm_i16=pcm,
                total_samples_used=total_needed,
            )
        )
    return clips


def build_scenario_plans(session_ids: list[str], normal_chunks: int) -> list[ScenarioPlan]:
    """Build B=1, B=N, row-permuted, and shrink/grow traces."""
    if len(session_ids) < 4:
        raise ValueError("the shrink/grow plan expects at least four sessions")

    baseline = [
        [[sid] for sid in session_ids]
        for _chunk in range(normal_chunks)
    ]
    co_batched = [
        [list(session_ids)]
        for _chunk in range(normal_chunks)
    ]

    base_perm = [session_ids[2], session_ids[0], session_ids[3], session_ids[1]]
    permuted: list[list[list[str]]] = []
    for chunk in range(normal_chunks):
        rotation = chunk % len(base_perm)
        order = base_perm[rotation:] + base_perm[:rotation]
        permuted.append([order])

    a, b, c, d = session_ids[:4]
    shrink_patterns = (
        [[a, b, c, d]],
        [[a, b], [c, d]],
        [[c, a, d], [b]],
        [[d, b], [a, c]],
        [[a], [b, c, d]],
    )
    shrink_grow = [
        [list(group) for group in shrink_patterns[chunk % len(shrink_patterns)]]
        for chunk in range(normal_chunks)
    ]

    return [
        ScenarioPlan(
            name="solo_b1",
            normal_groups_by_chunk=baseline,
            final_groups=[[sid] for sid in session_ids],
        ),
        ScenarioPlan(
            name="co_batched_bn",
            normal_groups_by_chunk=co_batched,
            final_groups=[list(session_ids)],
        ),
        ScenarioPlan(
            name="row_permutations",
            normal_groups_by_chunk=permuted,
            final_groups=[base_perm],
        ),
        ScenarioPlan(
            name="shrink_grow",
            normal_groups_by_chunk=shrink_grow,
            final_groups=[[b, d], [c, a]],
        ),
    ]


def _audio_bytes_for_samples(runtime: SessionRuntime, sample_count: int) -> bytes:
    start = runtime.audio_cursor
    end = start + int(sample_count)
    if end > runtime.clip.total_samples_used:
        raise RuntimeError(
            f"{runtime.clip.session_id}: trace requested {end} samples, "
            f"but clip budget is {runtime.clip.total_samples_used}"
        )
    runtime.audio_cursor = end
    return np.ascontiguousarray(runtime.clip.pcm_i16[start:end]).tobytes()


async def _append_real_audio(server: Any, runtime: SessionRuntime, sample_count: int) -> None:
    await server._scheduler_append_audio_locked(
        runtime.session,
        _audio_bytes_for_samples(runtime, sample_count),
    )


async def feed_until_ready(server: Any, runtime: SessionRuntime) -> None:
    """Append real PCM slices until the next normal scheduler row is ready."""
    session = runtime.session
    pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
    if runtime.next_chunk_index == 0:
        needed = int(server.preprocess_new_audio_samples) - pending_len
    else:
        needed = int(server.preprocess_new_audio_samples) - pending_len
    if needed > 0:
        await _append_real_audio(server, runtime, needed)
    if not server._scheduler_session_ready(session):
        raise RuntimeError(
            f"{session.id}: not ready after feeding {needed} samples; "
            f"pending={len(session.pending_audio)} emitted={session.emitted_frames} "
            f"total={session.total_audio_samples}"
        )


async def feed_final_tail(
    server: Any,
    runtimes: dict[str, SessionRuntime],
    *,
    final_tail_samples: int,
) -> None:
    for runtime in runtimes.values():
        await _append_real_audio(server, runtime, final_tail_samples)


def _call_model_path(server: Any, fn: Any, *args: Any, lane_id: int = 0) -> Any:
    if int(getattr(server, "model_lanes", 1)) > 1:
        server._ensure_scheduler_model_lane_resources()
        return server._run_scheduler_model_lane_call_sync(lane_id, fn, args)
    return fn(*args)


def _token_ids_from_hyp(hyp: Any) -> tuple[int, ...]:
    if hyp is None:
        return ()
    y_sequence = getattr(hyp, "y_sequence", None)
    if y_sequence is None:
        return ()
    if torch.is_tensor(y_sequence):
        return tuple(int(item) for item in y_sequence.detach().cpu().reshape(-1).tolist())
    if isinstance(y_sequence, np.ndarray):
        return tuple(int(item) for item in y_sequence.reshape(-1).tolist())
    if isinstance(y_sequence, Iterable) and not isinstance(y_sequence, (str, bytes)):
        return tuple(int(item) for item in y_sequence)
    return (int(y_sequence),)


def _session_hypothesis(session: Any) -> Any:
    hyps = getattr(session, "previous_hypotheses", None)
    if not hyps:
        return None
    return hyps[0]


def _snapshot_float_tensors(root: Any, prefix: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    seen: set[int] = set()

    def visit(obj: Any, path: str) -> None:
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if torch.is_tensor(obj):
            if obj.is_floating_point() or obj.is_complex():
                out[path] = obj.detach().cpu().clone()
            return
        if isinstance(obj, np.ndarray):
            if np.issubdtype(obj.dtype, np.floating) or np.issubdtype(obj.dtype, np.complexfloating):
                out[path] = torch.from_numpy(np.ascontiguousarray(obj)).detach().cpu().clone()
            return
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(value, f"{path}.{key}")
            return
        if isinstance(obj, (list, tuple)):
            for index, value in enumerate(obj):
                visit(value, f"{path}[{index}]")
            return
        if hasattr(obj, "__dict__"):
            for key, value in vars(obj).items():
                visit(value, f"{path}.{key}")

    visit(root, prefix)
    return out


def _snapshot_decoder_floats(session: Any) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    hyp = _session_hypothesis(session)
    if hyp is not None and hasattr(hyp, "dec_state"):
        tensors.update(_snapshot_float_tensors(getattr(hyp, "dec_state"), "previous_hypotheses[0].dec_state"))
    tensors.update(_snapshot_float_tensors(getattr(session, "pred_out_stream", None), "pred_out_stream"))
    return tensors


class DecodeCaseRecorder:
    """Capture decoder-only inputs while the real server path is running."""

    def __init__(self, model: Any, *, max_cases_per_key: int = 2):
        self.model = model
        self.max_cases_per_key = max_cases_per_key
        self.phase = "unknown"
        self.cases: list[DecodeCase] = []
        self._counts: Counter[tuple[str, int, int]] = Counter()
        self._original = None

    def __enter__(self) -> "DecodeCaseRecorder":
        from nemotron_speech.server import clone_hypotheses_deep

        decoding = self.model.decoding
        self._original = decoding.rnnt_decoder_predictions_tensor

        def wrapper(
            *,
            encoder_output: torch.Tensor,
            encoded_lengths: torch.Tensor,
            return_hypotheses: bool = False,
            partial_hypotheses: Optional[list[Any]] = None,
        ):
            phase = str(self.phase)
            batch_size = int(encoder_output.shape[0])
            encoded_time = int(encoder_output.shape[-1])
            key = (phase.split(":", 1)[0], batch_size, encoded_time)
            if self._counts[key] < self.max_cases_per_key:
                self._counts[key] += 1
                cloned_partials = (
                    None
                    if partial_hypotheses is None
                    else clone_hypotheses_deep(partial_hypotheses)
                )
                self.cases.append(
                    DecodeCase(
                        name=f"{phase}:case{self._counts[key]}",
                        phase=phase,
                        encoder_output=encoder_output.detach().clone(),
                        encoded_lengths=encoded_lengths.detach().clone(),
                        partial_hypotheses=cloned_partials,
                    )
                )
            return self._original(
                encoder_output=encoder_output,
                encoded_lengths=encoded_lengths,
                return_hypotheses=return_hypotheses,
                partial_hypotheses=partial_hypotheses,
            )

        object.__setattr__(decoding, "rnnt_decoder_predictions_tensor", wrapper)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._original is not None:
            object.__setattr__(
                self.model.decoding,
                "rnnt_decoder_predictions_tensor",
                self._original,
            )


def capture_event(
    *,
    scenario: str,
    session_id: str,
    logical_event: str,
    text: Optional[str],
    state_session: Any,
) -> dict[str, Any]:
    hyp = _session_hypothesis(state_session)
    y_sequence = _token_ids_from_hyp(hyp)
    return {
        "scenario": scenario,
        "session_id": session_id,
        "event": logical_event,
        "text": "" if text is None else str(text),
        "tokens": y_sequence,
        "y_sequence": y_sequence,
        "float_tensors": _snapshot_decoder_floats(state_session),
        "emitted_frames": int(getattr(state_session, "emitted_frames", 0)),
    }


def _validate_chunk_groups(session_ids: list[str], chunk: int, groups: list[list[str]]) -> None:
    flat = [sid for group in groups for sid in group]
    if sorted(flat) != sorted(session_ids):
        raise RuntimeError(
            f"chunk {chunk}: groups must contain every session once; "
            f"got {flat}, expected {session_ids}"
        )


async def run_scenario(
    server: Any,
    clips: list[ClipSpec],
    plan: ScenarioPlan,
    *,
    final_tail_samples: int,
    lane_id: int = 0,
    decode_recorder: Optional[DecodeCaseRecorder] = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Run one composition trace and return captures keyed by session/event."""
    _, ASRSession = import_server_classes()
    runtimes: dict[str, SessionRuntime] = {}
    for clip in clips:
        session = ASRSession(id=clip.session_id, websocket=None, target_lang=server.target_lang)
        session.continuous_event_queue = asyncio.Queue()
        server._init_session_without_synthetic_warmup(session)
        runtimes[clip.session_id] = SessionRuntime(session=session, clip=clip)

    server.sessions = {sid: runtime.session for sid, runtime in runtimes.items()}
    if int(getattr(server, "model_lanes", 1)) > 1:
        for sid in runtimes:
            server._scheduler_session_model_lane_affinity[sid] = lane_id

    captures: dict[tuple[str, str], dict[str, Any]] = {}
    session_ids = [clip.session_id for clip in clips]

    try:
        for chunk_index, groups in enumerate(plan.normal_groups_by_chunk):
            _validate_chunk_groups(session_ids, chunk_index, groups)
            for group_ids in groups:
                for sid in group_ids:
                    await feed_until_ready(server, runtimes[sid])
                sessions = [runtimes[sid].session for sid in group_ids]
                if decode_recorder is not None:
                    decode_recorder.phase = (
                        f"normal:{plan.name}:chunk{chunk_index:04d}:B{len(group_ids)}"
                    )
                texts = _call_model_path(
                    server,
                    server._process_ready_batch,
                    sessions,
                    lane_id=lane_id,
                )
                for sid in group_ids:
                    runtime = runtimes[sid]
                    text = texts.get(sid)
                    if text is not None:
                        runtime.session.current_text = text
                    logical_event = f"chunk:{runtime.next_chunk_index:04d}"
                    captures[(sid, logical_event)] = capture_event(
                        scenario=plan.name,
                        session_id=sid,
                        logical_event=logical_event,
                        text=runtime.session.current_text,
                        state_session=runtime.session,
                    )
                    runtime.next_chunk_index += 1

        await feed_final_tail(
            server,
            runtimes,
            final_tail_samples=final_tail_samples,
        )

        flat_final = [sid for group in plan.final_groups for sid in group]
        if sorted(flat_final) != sorted(session_ids):
            raise RuntimeError(
                f"final groups must contain every session once; got {flat_final}"
            )
        for group_ids in plan.final_groups:
            items = [
                server._continuous_prepare_finalize_item_locked(
                    runtimes[sid].session,
                    reason=f"harness:{plan.name}",
                    expected_generation=runtimes[sid].session.scheduler_generation,
                )
                for sid in group_ids
            ]
            rows = []
            row_to_sid: dict[int, str] = {}
            for item, sid in zip(items, group_ids, strict=True):
                row = server._prepare_final_fork_batch_row(item)
                if row is not None:
                    row_to_sid[id(row)] = sid
                    rows.append(row)

            grouped: dict[tuple[Any, ...], list[Any]] = {}
            for row in rows:
                grouped.setdefault(server._finalize_batch_group_key_for_row(row), []).append(row)

            texts = {item.session.id: item.final_text for item in items}
            for key, key_rows in grouped.items():
                if decode_recorder is not None:
                    decode_recorder.phase = f"final:{plan.name}:B{len(key_rows)}"
                texts.update(
                    _call_model_path(
                        server,
                        server._process_final_batch_rows,
                        key_rows,
                        key,
                        lane_id=lane_id,
                    )
                )
            for item in items:
                if item.session.id in texts and texts[item.session.id] is not None:
                    item.final_text = texts[item.session.id]
                state_session = item.fork if item.fork is not None else item.session
                captures[(item.session.id, "final:0000")] = capture_event(
                    scenario=plan.name,
                    session_id=item.session.id,
                    logical_event="final:0000",
                    text=item.final_text,
                    state_session=state_session,
                )
    finally:
        server.sessions = {}
        if int(getattr(server, "model_lanes", 1)) > 1:
            server._scheduler_session_model_lane_affinity.clear()
        gc.collect()
        torch.cuda.empty_cache()

    return captures


def _tensor_max_abs_rel(expected: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    if expected.numel() == 0 and actual.numel() == 0:
        return 0.0, 0.0
    exp = expected.to(torch.float32)
    act = actual.to(torch.float32)
    diff = (exp - act).abs()
    max_abs = float(diff.max().item())
    denom = exp.abs().clamp_min(1.0e-12)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


def _observed_min_atol(max_abs: float) -> Optional[float]:
    for atol in ATOL_CANDIDATES:
        if max_abs <= atol:
            return atol
    return None


def compare_captures(
    reference_name: str,
    reference: dict[tuple[str, str], dict[str, Any]],
    candidates: dict[str, dict[tuple[str, str], dict[str, Any]]],
    *,
    float_atol: float = 1.0e-4,
) -> CompareSummary:
    token_mismatches: list[str] = []
    float_summary = FloatCompareSummary()
    ref_keys = set(reference)
    for scenario_name, captures in candidates.items():
        if set(captures) != ref_keys:
            token_mismatches.append(
                f"{scenario_name}: event key mismatch missing={sorted(ref_keys - set(captures))[:5]} "
                f"extra={sorted(set(captures) - ref_keys)[:5]}"
            )
            continue
        for key in sorted(ref_keys):
            ref = reference[key]
            cand = captures[key]
            if ref["tokens"] != cand["tokens"] or ref["text"] != cand["text"]:
                token_mismatches.append(
                    f"{scenario_name} {key[0]} {key[1]}: "
                    f"tokens {ref['tokens']} != {cand['tokens']} text {ref['text']!r} != {cand['text']!r}"
                )

            ref_tensors = ref["float_tensors"]
            cand_tensors = cand["float_tensors"]
            if set(ref_tensors) != set(cand_tensors):
                float_summary.byte_equal = False
                float_summary.allclose_1e_4 = False
                float_summary.mismatches.append(
                    f"{scenario_name} {key[0]} {key[1]} tensor keys differ: "
                    f"missing={sorted(set(ref_tensors) - set(cand_tensors))[:5]} "
                    f"extra={sorted(set(cand_tensors) - set(ref_tensors))[:5]}"
                )
                continue

            for path in sorted(ref_tensors):
                exp = ref_tensors[path]
                act = cand_tensors[path]
                full_path = f"{scenario_name}.{key[0]}.{key[1]}.{path}"
                float_summary.tensor_pairs += 1
                if exp.shape != act.shape or exp.dtype != act.dtype:
                    float_summary.byte_equal = False
                    float_summary.allclose_1e_4 = False
                    float_summary.mismatches.append(
                        f"{full_path}: shape/dtype {tuple(exp.shape)}/{exp.dtype} "
                        f"!= {tuple(act.shape)}/{act.dtype}"
                    )
                    continue
                equal = bool(torch.equal(exp, act))
                close = bool(torch.allclose(exp, act, atol=float_atol, rtol=RTOL))
                float_summary.byte_equal = float_summary.byte_equal and equal
                float_summary.allclose_1e_4 = float_summary.allclose_1e_4 and close
                max_abs, max_rel = _tensor_max_abs_rel(exp, act)
                if max_abs > float_summary.max_abs:
                    float_summary.max_abs = max_abs
                    float_summary.max_rel = max_rel
                    float_summary.max_path = full_path
                if not close:
                    float_summary.mismatches.append(
                        f"{full_path}: not allclose max_abs={max_abs:.6g} max_rel={max_rel:.6g}"
                    )

    float_summary.min_observed_atol = _observed_min_atol(float_summary.max_abs)
    return CompareSummary(
        reference=reference_name,
        candidates=list(candidates),
        event_count=len(ref_keys),
        token_text_invariant=not token_mismatches,
        token_mismatches=token_mismatches,
        float_state=float_summary,
    )


def _merge_float_summary(dst: FloatCompareSummary, src: FloatCompareSummary) -> None:
    dst.tensor_pairs += src.tensor_pairs
    dst.byte_equal = dst.byte_equal and src.byte_equal
    dst.allclose_1e_4 = dst.allclose_1e_4 and src.allclose_1e_4
    if src.max_abs > dst.max_abs:
        dst.max_abs = src.max_abs
        dst.max_rel = src.max_rel
        dst.max_path = src.max_path
    dst.mismatches.extend(src.mismatches)
    dst.min_observed_atol = _observed_min_atol(dst.max_abs)


def compare_capture_groups(
    reference_label: str,
    candidate_label: str,
    reference_by_scenario: dict[str, dict[tuple[str, str], dict[str, Any]]],
    candidate_by_scenario: dict[str, dict[tuple[str, str], dict[str, Any]]],
    *,
    float_atol: float = 1.0e-4,
) -> CompareSummary:
    token_mismatches: list[str] = []
    float_summary = FloatCompareSummary()
    event_count = 0
    ref_names = set(reference_by_scenario)
    cand_names = set(candidate_by_scenario)
    if ref_names != cand_names:
        token_mismatches.append(
            f"scenario mismatch missing={sorted(ref_names - cand_names)} "
            f"extra={sorted(cand_names - ref_names)}"
        )
    for scenario_name in sorted(ref_names & cand_names):
        comp = compare_captures(
            f"{reference_label}:{scenario_name}",
            reference_by_scenario[scenario_name],
            {f"{candidate_label}:{scenario_name}": candidate_by_scenario[scenario_name]},
            float_atol=float_atol,
        )
        event_count += comp.event_count
        token_mismatches.extend(comp.token_mismatches)
        _merge_float_summary(float_summary, comp.float_state)
    return CompareSummary(
        reference=reference_label,
        candidates=[candidate_label],
        event_count=event_count,
        token_text_invariant=not token_mismatches,
        token_mismatches=token_mismatches,
        float_state=float_summary,
    )


async def run_composition_suite(
    server: Any,
    *,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    session_count: int = DEFAULT_SESSION_COUNT,
    normal_chunks: int = DEFAULT_NORMAL_CHUNKS,
    final_tail_samples: int = DEFAULT_FINAL_TAIL_SAMPLES,
    lane_id: int = 0,
) -> tuple[dict[str, Any], dict[str, dict[tuple[str, str], dict[str, Any]]], CompareSummary]:
    clips = select_audio_clips(
        server,
        audio_dir=audio_dir,
        session_count=session_count,
        normal_chunks=normal_chunks,
        final_tail_samples=final_tail_samples,
    )
    plans = build_scenario_plans([clip.session_id for clip in clips], normal_chunks)
    captures_by_scenario: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    timings: dict[str, float] = {}
    for plan in plans:
        start = time.perf_counter()
        captures_by_scenario[plan.name] = await run_scenario(
            server,
            clips,
            plan,
            final_tail_samples=final_tail_samples,
            lane_id=lane_id,
        )
        timings[plan.name] = (time.perf_counter() - start) * 1000.0

    reference_name = "solo_b1"
    reference = captures_by_scenario[reference_name]
    candidates = {
        name: captures
        for name, captures in captures_by_scenario.items()
        if name != reference_name
    }
    comparison = compare_captures(reference_name, reference, candidates)
    metadata = {
        "model": server.model_name_or_path,
        "nemo": nemo_identity(),
        "clips": [
            {
                "session_id": clip.session_id,
                "path": str(clip.path),
                "sha1_12": hashlib.sha1(clip.path.read_bytes()).hexdigest()[:12],
                "samples_used": clip.total_samples_used,
            }
            for clip in clips
        ],
        "normal_chunks": normal_chunks,
        "final_tail_samples": final_tail_samples,
        "shift_frames": int(server.shift_frames),
        "hop_samples": int(server.hop_samples),
        "preprocess_new_audio_samples": int(server.preprocess_new_audio_samples),
        "batch_max_size": int(server.batch_max_size),
        "model_lanes": int(server.model_lanes),
        "timings_ms": timings,
    }
    return metadata, captures_by_scenario, comparison


async def run_plans(
    server: Any,
    clips: list[ClipSpec],
    plans: list[ScenarioPlan],
    *,
    final_tail_samples: int,
    lane_id: int = 0,
    decode_recorder: Optional[DecodeCaseRecorder] = None,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    captures_by_scenario: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for plan in plans:
        captures_by_scenario[plan.name] = await run_scenario(
            server,
            clips,
            plan,
            final_tail_samples=final_tail_samples,
            lane_id=lane_id,
            decode_recorder=decode_recorder,
        )
    return captures_by_scenario


def _jsonable_summary(metadata: dict[str, Any], comparison: CompareSummary) -> dict[str, Any]:
    floats = comparison.float_state
    return {
        "metadata": metadata,
        "reference": comparison.reference,
        "candidates": comparison.candidates,
        "event_count": comparison.event_count,
        "token_text_invariant": comparison.token_text_invariant,
        "token_mismatch_count": len(comparison.token_mismatches),
        "token_mismatches": comparison.token_mismatches[:20],
        "float_state": {
            "tensor_pairs": floats.tensor_pairs,
            "byte_equal": floats.byte_equal,
            "allclose_1e_4": floats.allclose_1e_4,
            "max_abs": floats.max_abs,
            "max_rel": floats.max_rel,
            "max_path": floats.max_path,
            "min_observed_atol": floats.min_observed_atol,
            "mismatch_count": len(floats.mismatches),
            "mismatches": floats.mismatches[:20],
        },
        "oracle": "concurrent_ab_valid" if comparison.token_text_invariant else "fixed_batch_trace_required",
    }


def _jsonable_compare(comparison: CompareSummary) -> dict[str, Any]:
    floats = comparison.float_state
    return {
        "reference": comparison.reference,
        "candidates": comparison.candidates,
        "event_count": comparison.event_count,
        "token_text_exact": comparison.token_text_invariant,
        "token_mismatch_count": len(comparison.token_mismatches),
        "token_mismatches": comparison.token_mismatches[:20],
        "float_state": {
            "tensor_pairs": floats.tensor_pairs,
            "byte_equal": floats.byte_equal,
            "allclose_1e_4": floats.allclose_1e_4,
            "max_abs": floats.max_abs,
            "max_rel": floats.max_rel,
            "max_path": floats.max_path,
            "min_observed_atol": floats.min_observed_atol,
            "mismatch_count": len(floats.mismatches),
            "mismatches": floats.mismatches[:20],
        },
    }


def greedy_batch_decoding_cfg(*, use_cuda_graph_decoder: bool) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "strategy": "greedy_batch",
            "greedy": {
                "max_symbols": 10,
                "loop_labels": True,
                "use_cuda_graph_decoder": bool(use_cuda_graph_decoder),
            },
        }
    )


def set_decoder_cuda_graph(server: Any, *, enabled: bool) -> None:
    """Toggle the live greedy_batch decoder graph in-process."""
    cfg = greedy_batch_decoding_cfg(use_cuda_graph_decoder=enabled)
    server._decoding_cfg_for_lane_models = cfg
    server.decoder_strategy = "greedy_batch"
    models = [server.model]
    models.extend(getattr(server, "_scheduler_model_lane_models", [])[1:])
    seen: set[int] = set()
    for model in models:
        if id(model) in seen:
            continue
        seen.add(id(model))
        model.change_decoding_strategy(decoding_cfg=copy.deepcopy(cfg))
        model.eval()
        with contextlib.suppress(Exception):
            model.preprocessor.featurizer.dither = 0.0


def get_decoding_computer(model: Any) -> Any:
    decoding = getattr(model, "decoding", None)
    greedy = getattr(decoding, "decoding", None)
    computer = getattr(greedy, "decoding_computer", None)
    if computer is None:
        raise RuntimeError(
            "could not reach model.decoding.decoding.decoding_computer"
        )
    return computer


def graph_mode_name(computer: Any) -> Optional[str]:
    mode = getattr(computer, "cuda_graphs_mode", None)
    return None if mode is None else str(mode)


def is_full_graph(computer: Any) -> bool:
    mode = getattr(computer, "cuda_graphs_mode", None)
    return mode is getattr(computer, "CudaGraphsMode").FULL_GRAPH


def memory_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "max_allocated": int(torch.cuda.max_memory_allocated()),
        "max_reserved": int(torch.cuda.max_memory_reserved()),
    }


def tensor_bytes_unique(root: Any) -> int:
    total = 0
    seen: set[int] = set()

    def visit(obj: Any) -> None:
        nonlocal total
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        if torch.is_tensor(obj):
            total += int(obj.numel() * obj.element_size())
            return
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return
        if isinstance(obj, dict):
            for value in obj.values():
                visit(value)
            return
        if isinstance(obj, (list, tuple)):
            for value in obj:
                visit(value)
            return
        if hasattr(obj, "__dict__"):
            for value in vars(obj).values():
                visit(value)

    visit(root)
    return total


@contextlib.contextmanager
def instrument_label_looping() -> Iterable[NeedReinitStats]:
    from nemo.collections.asr.parts.submodules.transducer_decoding import rnnt_label_looping

    stats = NeedReinitStats()
    label_state_cls = rnnt_label_looping.LabelLoopingState
    computer_cls = rnnt_label_looping.GreedyBatchedRNNTLabelLoopingComputer
    original_need_reinit = label_state_cls.need_reinit
    original_graph_reinitialize = computer_cls._graph_reinitialize

    def need_reinit_wrapper(self, encoder_output_projected: torch.Tensor) -> bool:
        result = bool(original_need_reinit(self, encoder_output_projected))
        stats.need_reinit_calls += 1
        if result:
            stats.need_reinit_true += 1
        if len(stats.need_reinit_shapes) < 40:
            stats.need_reinit_shapes.append(
                {
                    "state_batch_size": int(getattr(self, "batch_size", -1)),
                    "state_max_time": int(getattr(self, "max_time", -1)),
                    "input_shape": [int(dim) for dim in encoder_output_projected.shape],
                    "input_device": str(encoder_output_projected.device),
                    "result": result,
                }
            )
        return result

    def graph_reinitialize_wrapper(self, encoder_output_projected: torch.Tensor):
        stats.graph_reinitialize_calls += 1
        if len(stats.graph_reinitialize_shapes) < 20:
            stats.graph_reinitialize_shapes.append(
                {
                    "input_shape": [int(dim) for dim in encoder_output_projected.shape],
                    "input_device": str(encoder_output_projected.device),
                }
            )
        return original_graph_reinitialize(self, encoder_output_projected)

    label_state_cls.need_reinit = need_reinit_wrapper
    computer_cls._graph_reinitialize = graph_reinitialize_wrapper
    try:
        yield stats
    finally:
        label_state_cls.need_reinit = original_need_reinit
        computer_cls._graph_reinitialize = original_graph_reinitialize


def reset_need_reinit_stats(stats: NeedReinitStats) -> None:
    stats.need_reinit_calls = 0
    stats.need_reinit_true = 0
    stats.graph_reinitialize_calls = 0
    stats.need_reinit_shapes.clear()
    stats.graph_reinitialize_shapes.clear()


def jsonable_need_reinit_stats(stats: NeedReinitStats) -> dict[str, Any]:
    return {
        "need_reinit_calls": stats.need_reinit_calls,
        "need_reinit_true": stats.need_reinit_true,
        "graph_reinitialize_calls": stats.graph_reinitialize_calls,
        "need_reinit_shapes": stats.need_reinit_shapes,
        "graph_reinitialize_shapes": stats.graph_reinitialize_shapes,
    }


def clone_clip_specs(
    clips: list[ClipSpec],
    *,
    count: int,
    prefix: str,
) -> list[ClipSpec]:
    if not clips:
        raise RuntimeError("cannot clone an empty clip set")
    out: list[ClipSpec] = []
    for index in range(count):
        clip = clips[index % len(clips)]
        out.append(
            ClipSpec(
                session_id=f"{prefix}{index:03d}",
                path=clip.path,
                pcm_i16=clip.pcm_i16,
                total_samples_used=clip.total_samples_used,
            )
        )
    return out


async def warm_decoder_graph_to_max(
    server: Any,
    *,
    audio_dir: Path,
    batch_max_size: int,
    final_tail_samples: int,
    lane_id: int = 0,
) -> dict[str, Any]:
    clips = select_audio_clips(
        server,
        audio_dir=audio_dir,
        session_count=batch_max_size,
        normal_chunks=2,
        final_tail_samples=final_tail_samples,
        allow_reuse=True,
        session_prefix="warm",
    )
    ids = [clip.session_id for clip in clips]
    plan = ScenarioPlan(
        name="warm_max_b",
        normal_groups_by_chunk=[[ids], [ids]],
        final_groups=[ids],
    )
    await run_scenario(
        server,
        clips,
        plan,
        final_tail_samples=final_tail_samples,
        lane_id=lane_id,
    )
    computer = get_decoding_computer(server.model)
    state = getattr(computer, "state", None)
    return {
        "mode": graph_mode_name(computer),
        "full_graph": is_full_graph(computer),
        "state_batch_size": None if state is None else int(getattr(state, "batch_size", -1)),
        "state_max_time": None if state is None else int(getattr(state, "max_time", -1)),
        "state_tensor_bytes": 0 if state is None else tensor_bytes_unique(state),
    }


async def run_finalize_b_sweep(
    server: Any,
    *,
    clips_pool: list[ClipSpec],
    batch_max_size: int,
    normal_chunks: int,
    final_tail_samples: int,
    lane_id: int = 0,
    decode_recorder: Optional[DecodeCaseRecorder] = None,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    captures: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for batch_size in range(1, batch_max_size + 1):
        clips = clone_clip_specs(clips_pool, count=batch_size, prefix=f"fb{batch_size:02d}_")
        ids = [clip.session_id for clip in clips]
        plan = ScenarioPlan(
            name=f"final_B_{batch_size:02d}",
            normal_groups_by_chunk=[[ids] for _ in range(normal_chunks)],
            final_groups=[ids],
        )
        captures[plan.name] = await run_scenario(
            server,
            clips,
            plan,
            final_tail_samples=final_tail_samples,
            lane_id=lane_id,
            decode_recorder=decode_recorder,
        )
    return captures


@contextlib.contextmanager
def lane_style_stream(server: Any, stream: torch.cuda.Stream):
    previous_stream = getattr(server._scheduler_model_lane_tls, "stream", None)
    previous_model = getattr(server._scheduler_model_lane_tls, "model", None)
    server._scheduler_model_lane_tls.stream = stream
    server._scheduler_model_lane_tls.model = server.model
    try:
        with torch.cuda.stream(stream):
            yield
        stream.synchronize()
    finally:
        server._scheduler_model_lane_tls.stream = previous_stream
        server._scheduler_model_lane_tls.model = previous_model


async def run_lane_stream_replay_check(
    server: Any,
    *,
    audio_dir: Path,
    final_tail_samples: int,
) -> dict[str, Any]:
    clips = select_audio_clips(
        server,
        audio_dir=audio_dir,
        session_count=4,
        normal_chunks=4,
        final_tail_samples=final_tail_samples,
        session_prefix="ls",
    )
    ids = [clip.session_id for clip in clips]
    plan = ScenarioPlan(
        name="lane_stream_replay",
        normal_groups_by_chunk=[[ids], [[ids[2], ids[0]], [ids[3], ids[1]]], [ids], [ids]],
        final_groups=[ids],
    )
    default_captures = await run_scenario(
        server,
        clips,
        plan,
        final_tail_samples=final_tail_samples,
    )
    stream = torch.cuda.Stream()
    try:
        with lane_style_stream(server, stream):
            stream_captures = await run_scenario(
                server,
                clips,
                plan,
                final_tail_samples=final_tail_samples,
            )
    except Exception as exc:
        return {
            "ok": False,
            "verdict": "different_stream_replay_failed",
            "error": repr(exc),
        }
    comparison = compare_captures(
        "graph_on_default_stream",
        default_captures,
        {"graph_on_lane_style_stream": stream_captures},
    )
    return {
        "ok": comparison.token_text_invariant and comparison.float_state.allclose_1e_4,
        "verdict": (
            "capture_stream_graph_replays_on_lane_style_stream"
            if comparison.token_text_invariant and comparison.float_state.allclose_1e_4
            else "different_stream_replay_mismatch"
        ),
        "comparison": _jsonable_compare(comparison),
    }


def _parse_last_hist_from_text(path: Path, marker: str) -> dict[int, int]:
    if not path.exists():
        return {}
    hist: dict[int, int] = {}
    pattern = re.compile(r"effective_batch_hist=({[^}]+})")
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            match = pattern.search(line)
            if not match:
                continue
            parsed = ast.literal_eval(match.group(1))
            hist = {int(key): int(value) for key, value in parsed.items()}
    return hist


def _iter_finalize_profile_records(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    marker = "finalize_profile_record "
    with path.open("r", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            payload = line.split(marker, 1)[1].strip()
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue


def conc10_distribution(records_path: Path, srvlog_path: Path) -> dict[str, Any]:
    steady_hist = _parse_last_hist_from_text(
        srvlog_path,
        "scheduler_batch_telemetry",
    )
    finalize_hist = _parse_last_hist_from_text(
        srvlog_path,
        "scheduler_finalize_batch_telemetry",
    )
    finalize_profile_hist: Counter[int] = Counter()
    finalize_t_hist: Counter[int] = Counter()
    finalize_decode_wall_ms: list[float] = []
    for record in _iter_finalize_profile_records(records_path):
        if "B" in record:
            finalize_profile_hist[int(record["B"])] += 1
        if "encoded_len" in record:
            finalize_t_hist[int(record["encoded_len"])] += 1
        if record.get("decode_wall_ms") is not None:
            finalize_decode_wall_ms.append(float(record["decode_wall_ms"]))
    if not finalize_hist and finalize_profile_hist:
        finalize_hist = dict(finalize_profile_hist)
    combined = Counter(steady_hist)
    combined.update(finalize_hist)
    return {
        "records_path": str(records_path),
        "srvlog_path": str(srvlog_path),
        "steady_hist": dict(sorted(steady_hist.items())),
        "finalize_hist": dict(sorted(finalize_hist.items())),
        "combined_hist": dict(sorted(combined.items())),
        "finalize_profile_B_hist": dict(sorted(finalize_profile_hist.items())),
        "finalize_profile_encoded_T_hist": dict(sorted(finalize_t_hist.items())),
        "finalize_profile_decode_wall_ms_p50": (
            None
            if not finalize_decode_wall_ms
            else float(statistics.median(finalize_decode_wall_ms))
        ),
        "finalize_profile_decode_wall_ms_p95": (
            None
            if not finalize_decode_wall_ms
            else percentile(finalize_decode_wall_ms, 95.0)
        ),
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return float(ordered[lower] * (1.0 - frac) + ordered[upper] * frac)


def weighted_percentile(values_by_key: dict[int, float], weights: dict[int, int], pct: float) -> Optional[float]:
    total = sum(int(weight) for key, weight in weights.items() if key in values_by_key)
    if total <= 0:
        return None
    threshold = total * (pct / 100.0)
    running = 0
    for key, value in sorted(values_by_key.items(), key=lambda item: item[1]):
        weight = int(weights.get(key, 0))
        if weight <= 0:
            continue
        running += weight
        if running >= threshold:
            return float(value)
    fallback_values = list(values_by_key.values())
    return float(fallback_values[-1]) if fallback_values else None


def _clone_decode_partials(partials: Any) -> Any:
    if partials is None:
        return None
    from nemotron_speech.server import clone_hypotheses_deep

    return clone_hypotheses_deep(partials)


def run_decode_case_once(model: Any, case: DecodeCase) -> tuple[float, float, list[tuple[int, ...]], list[str]]:
    encoder_output = case.encoder_output.detach().clone()
    encoded_lengths = case.encoded_lengths.detach().clone()
    partials = _clone_decode_partials(case.partial_hypotheses)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start_event.record()
    hyps = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoder_output,
        encoded_lengths=encoded_lengths,
        return_hypotheses=True,
        partial_hypotheses=partials,
    )
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    event_ms = float(start_event.elapsed_time(end_event))
    y_sequences = [_token_ids_from_hyp(hyp) for hyp in hyps]
    texts = ["" if getattr(hyp, "text", None) is None else str(hyp.text) for hyp in hyps]
    return event_ms, wall_ms, y_sequences, texts


def select_cases_for_distribution(
    cases: list[DecodeCase],
    distribution: dict[str, Any],
) -> dict[str, dict[int, DecodeCase]]:
    by_phase_b: dict[tuple[str, int], list[DecodeCase]] = defaultdict(list)
    for case in cases:
        phase = case.phase.split(":", 1)[0]
        by_phase_b[(phase, case.batch_size)].append(case)
    selected = {"steady": {}, "finalize": {}}
    for batch_size in distribution.get("steady_hist", {}):
        candidates = by_phase_b.get(("normal", int(batch_size)), [])
        if candidates:
            selected["steady"][int(batch_size)] = candidates[-1]
    for batch_size in distribution.get("finalize_hist", {}):
        candidates = by_phase_b.get(("final", int(batch_size)), [])
        if candidates:
            selected["finalize"][int(batch_size)] = candidates[-1]
    return selected


def benchmark_decode_distribution(
    model: Any,
    selected_cases: dict[str, dict[int, DecodeCase]],
    distribution: dict[str, Any],
    *,
    reps: int,
    warmup_reps: int = 2,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for bucket_name, hist_name in (("steady", "steady_hist"), ("finalize", "finalize_hist")):
        hist = {int(key): int(value) for key, value in distribution.get(hist_name, {}).items()}
        event_by_b: dict[int, float] = {}
        wall_by_b: dict[int, float] = {}
        case_meta: dict[int, dict[str, Any]] = {}
        for batch_size, case in sorted(selected_cases.get(bucket_name, {}).items()):
            for _ in range(warmup_reps):
                run_decode_case_once(model, case)
            event_samples: list[float] = []
            wall_samples: list[float] = []
            for _ in range(reps):
                event_ms, wall_ms, _seqs, _texts = run_decode_case_once(model, case)
                event_samples.append(event_ms)
                wall_samples.append(wall_ms)
            event_by_b[batch_size] = float(statistics.median(event_samples))
            wall_by_b[batch_size] = float(statistics.median(wall_samples))
            case_meta[batch_size] = {
                "case": case.name,
                "phase": case.phase,
                "encoded_time": case.encoded_time,
                "event_ms_samples": event_samples,
                "wall_ms_samples": wall_samples,
            }
        results[bucket_name] = {
            "hist": hist,
            "covered_B": sorted(event_by_b),
            "event_ms_by_B": event_by_b,
            "wall_ms_by_B": wall_by_b,
            "event_ms_p50_weighted": weighted_percentile(event_by_b, hist, 50.0),
            "event_ms_p95_weighted": weighted_percentile(event_by_b, hist, 95.0),
            "wall_ms_p50_weighted": weighted_percentile(wall_by_b, hist, 50.0),
            "wall_ms_p95_weighted": weighted_percentile(wall_by_b, hist, 95.0),
            "case_meta": case_meta,
        }
    combined_hist = Counter({int(k): int(v) for k, v in distribution.get("steady_hist", {}).items()})
    combined_hist.update({int(k): int(v) for k, v in distribution.get("finalize_hist", {}).items()})
    combined_event: dict[int, float] = {}
    combined_wall: dict[int, float] = {}
    for batch_size, value in results.get("steady", {}).get("event_ms_by_B", {}).items():
        combined_event[int(batch_size)] = float(value)
    for batch_size, value in results.get("steady", {}).get("wall_ms_by_B", {}).items():
        combined_wall[int(batch_size)] = float(value)
    # When finalize has a B not present in steady, include it. For shared B=1,
    # steady dominates the combined stream and finalize is reported separately.
    for batch_size, value in results.get("finalize", {}).get("event_ms_by_B", {}).items():
        combined_event.setdefault(int(batch_size), float(value))
    for batch_size, value in results.get("finalize", {}).get("wall_ms_by_B", {}).items():
        combined_wall.setdefault(int(batch_size), float(value))
    results["combined"] = {
        "hist": dict(sorted(combined_hist.items())),
        "event_ms_p50_weighted": weighted_percentile(combined_event, combined_hist, 50.0),
        "event_ms_p95_weighted": weighted_percentile(combined_event, combined_hist, 95.0),
        "wall_ms_p50_weighted": weighted_percentile(combined_wall, combined_hist, 50.0),
        "wall_ms_p95_weighted": weighted_percentile(combined_wall, combined_hist, 95.0),
    }
    return results


def compare_decode_benchmarks(eager: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for bucket in ("steady", "finalize", "combined"):
        e = eager.get(bucket, {})
        g = graph.get(bucket, {})
        bucket_out: dict[str, Any] = {}
        for metric in (
            "event_ms_p50_weighted",
            "event_ms_p95_weighted",
            "wall_ms_p50_weighted",
            "wall_ms_p95_weighted",
        ):
            ev = e.get(metric)
            gv = g.get(metric)
            bucket_out[f"eager_{metric}"] = ev
            bucket_out[f"full_graph_{metric}"] = gv
            bucket_out[f"recoverable_delta_{metric}"] = (
                None if ev is None or gv is None else float(ev - gv)
            )
        out[bucket] = bucket_out
    return out


def deploy_gate_values() -> dict[str, Any]:
    values: dict[str, Any] = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        from cuda.bindings import __version__ as cuda_python_version
        from cuda.bindings import driver as cuda

        values["cuda_python_version"] = str(cuda_python_version)
        error, driver_version = cuda.cuDriverGetVersion()
        values["driver_query_error"] = str(error)
        values["driver_cuda_raw"] = int(driver_version)
        values["driver_cuda"] = f"{driver_version // 1000}.{(driver_version % 1000) // 10}"
    except Exception as exc:
        values["cuda_python_version"] = None
        values["driver_cuda"] = None
        values["driver_error"] = repr(exc)
    try:
        from nemo.core.utils.cuda_python_utils import (
            check_cuda_python_cuda_graphs_conditional_nodes_supported,
        )

        check_cuda_python_cuda_graphs_conditional_nodes_supported()
        values["conditional_nodes_supported"] = True
        values["conditional_nodes_error"] = None
    except Exception as exc:
        values["conditional_nodes_supported"] = False
        values["conditional_nodes_error"] = repr(exc)
    return values


def need_reinit_characterization(computer: Any) -> dict[str, Any]:
    state = getattr(computer, "state", None)
    if state is None:
        return {"available": False}
    batch_size = int(getattr(state, "batch_size"))
    max_time = int(getattr(state, "max_time"))
    encoder_dim = int(getattr(state, "encoder_output_projected").shape[-1])
    dtype = getattr(state, "encoder_output_projected").dtype
    device = getattr(state, "encoder_output_projected").device

    def probe(shape: tuple[int, int, int]) -> bool:
        tensor = torch.zeros(shape, dtype=dtype, device=device)
        return bool(state.need_reinit(tensor))

    small_t = max(1, min(8, max_time))
    return {
        "available": True,
        "warmed_batch_size": batch_size,
        "warmed_max_time": max_time,
        "initial_max_time": int(getattr(computer, "INITIAL_MAX_TIME", -1)),
        "same_B_small_T": probe((batch_size, small_t, encoder_dim)),
        "smaller_B_small_T": probe((max(1, batch_size - 1), small_t, encoder_dim)),
        "larger_B_small_T": probe((batch_size + 1, small_t, encoder_dim)),
        "same_B_larger_T": probe((batch_size, max_time + 1, encoder_dim)),
        "runtime_conclusion": (
            "after warm-to-max, varied runtime T below INITIAL_MAX_TIME does not reinit; "
            "B above warmed batch or T above warmed max_time would reinit"
        ),
    }


def inactive_row_isolation_check(
    model: Any,
    cases: list[DecodeCase],
    *,
    padded_batch_size: int,
) -> dict[str, Any]:
    source = next(
        (
            case
            for case in cases
            if case.batch_size == 1 and case.partial_hypotheses is not None
        ),
        None,
    )
    if source is None:
        source = next((case for case in cases if case.batch_size == 1), None)
    if source is None:
        return {"ok": False, "reason": "no_B1_decode_case"}
    solo_event, solo_wall, solo_seq, solo_text = run_decode_case_once(model, source)
    padded_output = source.encoder_output.new_zeros(
        (padded_batch_size, source.encoder_output.shape[1], source.encoder_output.shape[2])
    )
    padded_output[0:1].copy_(source.encoder_output)
    padded_lengths = source.encoded_lengths.new_zeros((padded_batch_size,))
    padded_lengths[0:1].copy_(source.encoded_lengths[0:1])
    if source.partial_hypotheses is None:
        padded_partials = None
    else:
        active_partial = _clone_decode_partials(source.partial_hypotheses)[0]
        padded_partials = [active_partial] + [None] * (padded_batch_size - 1)
    padded_case = DecodeCase(
        name=f"{source.name}:padded_idle_rows",
        phase="inactive_isolation",
        encoder_output=padded_output,
        encoded_lengths=padded_lengths,
        partial_hypotheses=padded_partials,
    )
    padded_event, padded_wall, padded_seq, padded_text = run_decode_case_once(model, padded_case)
    return {
        "ok": bool(solo_seq[0] == padded_seq[0] and solo_text[0] == padded_text[0]),
        "source_case": source.name,
        "solo_event_ms": solo_event,
        "padded_event_ms": padded_event,
        "solo_wall_ms": solo_wall,
        "padded_wall_ms": padded_wall,
        "solo_y_sequence": solo_seq[0],
        "padded_y_sequence": padded_seq[0],
        "solo_text": solo_text[0],
        "padded_text": padded_text[0],
        "padded_batch_size": padded_batch_size,
        "inactive_lengths": [int(x) for x in padded_lengths.detach().cpu().tolist()],
    }


def _fmt_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def write_probe_findings(summary: dict[str, Any], path: Path) -> None:
    full_graph = summary.get("full_graph", {})
    correctness = summary.get("correctness", {})
    need_reinit = summary.get("need_reinit", {})
    sizing = summary.get("p50_sizing", {})
    roi = summary.get("provisional_roi", {})
    memory = summary.get("memory", {})
    gates = summary.get("deploy_gates", {})
    lines = [
        "# Decoder Graph Probe Findings",
        "",
        f"Run timestamp: `{summary.get('run_timestamp_unix')}`",
        f"Model: `{summary.get('model')}`",
        f"NeMo: `{summary.get('nemo', {}).get('version')}` at `{summary.get('nemo', {}).get('path')}`",
        f"GPU: `{gates.get('device_name')}`",
        "",
        "## Verdict",
        "",
        f"- FULL_GRAPH confirmed: **{_fmt_bool(full_graph.get('confirmed'))}** (`cuda_graphs_mode={full_graph.get('mode')}`)",
        f"- Lane-stream replay: **{full_graph.get('lane_stream', {}).get('verdict')}**",
        f"- Byte-exact graph-on vs graph-off: **{_fmt_bool(correctness.get('overall_ok'))}**",
        f"- Need-reinit after max-B warm: **{need_reinit.get('replay', {}).get('need_reinit_true')} true need_reinit**, **{need_reinit.get('replay', {}).get('graph_reinitialize_calls')} recaptures**",
        f"- PROVISIONAL FLOOR: **{_fmt_bool(roi.get('floor_go'))}**; PROVISIONAL UPSIDE: **{_fmt_bool(roi.get('upside_go'))}**; overall Step-2 GO: **{_fmt_bool(roi.get('overall_go'))}**",
        "",
        "## FULL_GRAPH And Stream Safety",
        "",
        f"- Decoder computer: `{full_graph.get('computer_class')}`",
        f"- Mode before warm: `{full_graph.get('mode_before_warm')}`",
        f"- Mode after warm: `{full_graph.get('mode')}`",
        f"- Warm state: B=`{full_graph.get('warm', {}).get('state_batch_size')}`, max_time=`{full_graph.get('warm', {}).get('state_max_time')}`",
        f"- Stream verdict: `{full_graph.get('lane_stream', {}).get('verdict')}`",
        "",
        "## Correctness",
        "",
        f"- Step-1 scenario graph ON vs OFF: tokens/text/y-sequence exact `{correctness.get('main', {}).get('token_text_exact')}`, float state allclose(1e-4) `{correctness.get('main', {}).get('float_state', {}).get('allclose_1e_4')}`, events `{correctness.get('main', {}).get('event_count')}`.",
        f"- Finalize B=1..batch_max_size graph ON vs OFF: tokens/text/y-sequence exact `{correctness.get('finalize_b_sweep', {}).get('token_text_exact')}`, float state allclose(1e-4) `{correctness.get('finalize_b_sweep', {}).get('float_state', {}).get('allclose_1e_4')}`, events `{correctness.get('finalize_b_sweep', {}).get('event_count')}`.",
        f"- Inactive zero-length row isolation: `{correctness.get('inactive_row_isolation', {}).get('ok')}` using `{correctness.get('inactive_row_isolation', {}).get('source_case')}` padded to B=`{correctness.get('inactive_row_isolation', {}).get('padded_batch_size')}`.",
        f"- Worst float diff path: `{correctness.get('worst_float_path')}` max_abs=`{correctness.get('worst_float_max_abs')}`.",
        "",
        "## Recapture Characterization",
        "",
        f"- INITIAL_MAX_TIME: `{need_reinit.get('initial_max_time')}`",
        f"- Warm instrumentation: `{need_reinit.get('warm')}`",
        f"- Replay instrumentation after warm: `{need_reinit.get('replay')}`",
        f"- Shape probes: `{need_reinit.get('characterization')}`",
        "",
        "Runtime precheck sketch: before a graph decode call, compare `B` and projected encoder `max_time` to the warmed graph state's `batch_size` and `max_time`; if either would exceed the warmed values, route that call to eager and count it instead of letting NeMo recapture under load.",
        "",
        "## P50 Sizing",
        "",
        f"- Conc-10 steady B hist: `{sizing.get('distribution', {}).get('steady_hist')}`",
        f"- Conc-10 finalize B hist: `{sizing.get('distribution', {}).get('finalize_hist')}`",
        f"- Eager decode benchmark: `{sizing.get('eager')}`",
        f"- FULL_GRAPH decode benchmark: `{sizing.get('full_graph')}`",
        f"- Recoverable share vs residual: `{sizing.get('comparison')}`",
        f"- Projected p50: `{roi.get('projected_p50_ms')}` ms; projected spread: `{roi.get('projected_p95_minus_p50_ms')}` ms.",
        "",
        "Interpretation: the eager-minus-FULL_GRAPH delta is the recoverable host-loop/sync share. The measured FULL_GRAPH decode time is the residual graph replay plus eager input/state copies and CPU hypothesis conversion that this project does not remove.",
        "",
        "## Memory And Gates",
        "",
        f"- Memory before warm: `{memory.get('before_warm')}`",
        f"- Memory after warm: `{memory.get('after_warm')}`",
        f"- Per-lane graph warm delta: allocated `{memory.get('delta_allocated_bytes')}` bytes, reserved `{memory.get('delta_reserved_bytes')}` bytes, state tensor bytes `{memory.get('state_tensor_bytes')}`.",
        f"- Deploy gates: `{gates}`",
        "",
        "## EOU",
        "",
        "EOU probe is out of the leaderboard configuration and was not exercised here; keep eou-on routed to eager until a separate byte-exact eou-on graph check is added.",
        "",
        "## Blockers",
        "",
    ]
    blockers = summary.get("blockers") or []
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None for Step 2 if the verdict above is GO.")
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            str(summary.get("suggested_next_step")),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_conc10_pivot_findings(summary: dict[str, Any], path: Path) -> None:
    roi = summary.get("provisional_roi", {})
    sizing = summary.get("p50_sizing", {})
    lines = [
        "# Conc-10 Pivot Findings",
        "",
        "The Step-2 UPSIDE projection missed the conc-10 p50/spread target, so the decode graph should be treated primarily as the FLOOR overload-robustness project unless later measured A/B data proves otherwise.",
        "",
        f"- Projected p50: `{roi.get('projected_p50_ms')}` ms",
        f"- Projected p95-p50 spread: `{roi.get('projected_p95_minus_p50_ms')}` ms",
        f"- Decode sizing comparison: `{sizing.get('comparison')}`",
        "",
        "Alternative conc-10 p50/spread levers to pursue:",
        "",
        "- Reduce finalize fork/clone double-clone cost at `server.py:6370`, `server.py:6480`, and `server.py:7371`.",
        "- Add or promote one-shot finalize preprocessor work around `server.py:6927` and `server.py:7087`.",
        "- Fix or retune reset-while-`PENDING_FINALIZE` debounce delay around `server.py:5823`.",
        "- Add a global active-session/inflight admission cap around `server.py:4163` and `server.py:4326`.",
        "",
    ]
    path.write_text("\n".join(lines))


def _worst_float(comparisons: list[CompareSummary]) -> tuple[str, float]:
    worst_path = ""
    worst_abs = 0.0
    for comparison in comparisons:
        if comparison.float_state.max_abs > worst_abs:
            worst_abs = comparison.float_state.max_abs
            worst_path = comparison.float_state.max_path
    return worst_path, worst_abs


async def run_step2_probe(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Step 2 probe")

    summary: dict[str, Any] = {
        "run_timestamp_unix": time.time(),
        "model": args.model,
        "nemo": nemo_identity(),
        "deploy_gates": deploy_gate_values(),
        "blockers": [],
    }
    distribution = conc10_distribution(args.conc10_records, args.conc10_srvlog)

    print("step2_probe: loading graph-OFF ASRServer", flush=True)
    server = build_server(
        model=args.model,
        lanes=args.lanes,
        batch_max_size=args.batch_max_size,
        right_context=1,
    )
    set_decoder_cuda_graph(server, enabled=False)
    torch.cuda.empty_cache()

    main_clips = select_audio_clips(
        server,
        audio_dir=args.audio_dir,
        session_count=args.sessions,
        normal_chunks=args.normal_chunks,
        final_tail_samples=args.final_tail_samples,
        session_prefix="s",
    )
    main_plans = build_scenario_plans([clip.session_id for clip in main_clips], args.normal_chunks)
    sweep_pool = select_audio_clips(
        server,
        audio_dir=args.audio_dir,
        session_count=min(max(args.sessions, 4), 8),
        normal_chunks=args.final_sweep_normal_chunks,
        final_tail_samples=args.final_tail_samples,
        allow_reuse=True,
        session_prefix="pool",
    )

    print("step2_probe: running graph-OFF fixed traces and collecting decode cases", flush=True)
    with DecodeCaseRecorder(server.model, max_cases_per_key=args.decode_case_samples_per_key) as recorder:
        off_main = await run_plans(
            server,
            main_clips,
            main_plans,
            final_tail_samples=args.final_tail_samples,
            lane_id=args.lane_id,
            decode_recorder=recorder,
        )
        off_final_sweep = await run_finalize_b_sweep(
            server,
            clips_pool=sweep_pool,
            batch_max_size=args.batch_max_size,
            normal_chunks=args.final_sweep_normal_chunks,
            final_tail_samples=args.final_tail_samples,
            lane_id=args.lane_id,
            decode_recorder=recorder,
        )
        decode_cases = list(recorder.cases)

    selected_cases = select_cases_for_distribution(decode_cases, distribution)
    print(
        "step2_probe: timing graph-OFF decoder-only cases "
        f"cases={sum(len(v) for v in selected_cases.values())}",
        flush=True,
    )
    eager_bench = benchmark_decode_distribution(
        server.model,
        selected_cases,
        distribution,
        reps=args.decode_bench_reps,
    )

    print("step2_probe: toggling decoder graph ON in-process and warming max B", flush=True)
    set_decoder_cuda_graph(server, enabled=True)
    computer = get_decoding_computer(server.model)
    mode_before_warm = graph_mode_name(computer)
    torch.cuda.empty_cache()
    with contextlib.suppress(Exception):
        torch.cuda.reset_peak_memory_stats()
    before_warm = memory_snapshot()

    with instrument_label_looping() as stats:
        warm = await warm_decoder_graph_to_max(
            server,
            audio_dir=args.audio_dir,
            batch_max_size=args.batch_max_size,
            final_tail_samples=args.final_tail_samples,
            lane_id=args.lane_id,
        )
        after_warm = memory_snapshot()
        warm_stats = copy.deepcopy(jsonable_need_reinit_stats(stats))
        computer = get_decoding_computer(server.model)
        full_graph_confirmed = is_full_graph(computer)
        characterization_before_replay = (
            need_reinit_characterization(computer) if full_graph_confirmed else {}
        )
        if not full_graph_confirmed:
            summary["blockers"].append(
                f"FULL_GRAPH not achieved after warm: cuda_graphs_mode={graph_mode_name(computer)}"
            )
            summary["full_graph"] = {
                "confirmed": False,
                "mode_before_warm": mode_before_warm,
                "mode": graph_mode_name(computer),
                "computer_class": type(computer).__name__,
                "warm": warm,
                "lane_stream": {"ok": False, "verdict": "skipped_no_full_graph"},
            }
            summary["memory"] = {
                "before_warm": before_warm,
                "after_warm": after_warm,
                "delta_allocated_bytes": after_warm.get("allocated", 0) - before_warm.get("allocated", 0),
                "delta_reserved_bytes": after_warm.get("reserved", 0) - before_warm.get("reserved", 0),
                "state_tensor_bytes": warm.get("state_tensor_bytes", 0),
            }
            summary["need_reinit"] = {
                "initial_max_time": getattr(computer, "INITIAL_MAX_TIME", None),
                "warm": warm_stats,
                "replay": {},
                "characterization": {},
            }
            summary["correctness"] = {"overall_ok": False}
            summary["p50_sizing"] = {
                "distribution": distribution,
                "eager": eager_bench,
                "full_graph": {},
                "comparison": {},
            }
            summary["provisional_roi"] = {
                "floor_go": False,
                "upside_go": False,
                "overall_go": False,
                "projected_p50_ms": None,
                "projected_p95_minus_p50_ms": None,
            }
            summary["suggested_next_step"] = "Stop Step 2 and inspect CUDA conditional-node/FULL_GRAPH compile failure before any server wiring."
            return 2, summary

        reset_need_reinit_stats(stats)
        print("step2_probe: running graph-ON fixed traces", flush=True)
        graph_replay_exception = None
        try:
            on_main = await run_plans(
                server,
                main_clips,
                main_plans,
                final_tail_samples=args.final_tail_samples,
                lane_id=args.lane_id,
            )
            print("step2_probe: running graph-ON finalize B sweep", flush=True)
            on_final_sweep = await run_finalize_b_sweep(
                server,
                clips_pool=sweep_pool,
                batch_max_size=args.batch_max_size,
                normal_chunks=args.final_sweep_normal_chunks,
                final_tail_samples=args.final_tail_samples,
                lane_id=args.lane_id,
            )
        except Exception as exc:
            graph_replay_exception = repr(exc)
        replay_stats = copy.deepcopy(jsonable_need_reinit_stats(stats))

    if graph_replay_exception is not None:
        computer = get_decoding_computer(server.model)
        summary["blockers"].append(
            f"FULL_GRAPH replay failed after warm: {graph_replay_exception}"
        )
        summary.update(
            {
                "full_graph": {
                    "confirmed": True,
                    "mode_before_warm": mode_before_warm,
                    "mode": graph_mode_name(computer),
                    "computer_class": type(computer).__name__,
                    "warm": warm,
                    "lane_stream": {
                        "ok": False,
                        "verdict": "skipped_cuda_context_poisoned_after_replay_failure",
                    },
                },
                "correctness": {
                    "overall_ok": False,
                    "main": {},
                    "finalize_b_sweep": {},
                    "inactive_row_isolation": {
                        "ok": False,
                        "reason": "skipped_cuda_context_poisoned_after_replay_failure",
                    },
                    "worst_float_path": "",
                    "worst_float_max_abs": None,
                },
                "need_reinit": {
                    "initial_max_time": int(getattr(computer, "INITIAL_MAX_TIME", -1)),
                    "warm": warm_stats,
                    "replay": replay_stats,
                    "characterization": characterization_before_replay,
                },
                "p50_sizing": {
                    "distribution": distribution,
                    "selected_cases": {
                        phase: {
                            int(batch): {
                                "name": case.name,
                                "encoded_time": case.encoded_time,
                                "phase": case.phase,
                            }
                            for batch, case in cases.items()
                        }
                        for phase, cases in selected_cases.items()
                    },
                    "eager": eager_bench,
                    "full_graph": {},
                    "comparison": {},
                },
                "provisional_roi": {
                    "feasibility_ok": False,
                    "floor_go": False,
                    "upside_go": False,
                    "overall_go": False,
                    "baseline_p50_ms": 246.0,
                    "baseline_p95_minus_p50_ms": 33.0,
                    "projected_p50_ms": None,
                    "projected_p95_minus_p50_ms": None,
                    "recoverable_combined_event_p50_ms": None,
                    "recoverable_combined_event_p95_ms": None,
                },
                "memory": {
                    "before_warm": before_warm,
                    "after_warm": after_warm,
                    "delta_allocated_bytes": after_warm.get("allocated", 0)
                    - before_warm.get("allocated", 0),
                    "delta_reserved_bytes": after_warm.get("reserved", 0)
                    - before_warm.get("reserved", 0),
                    "state_tensor_bytes": warm.get("state_tensor_bytes", 0),
                    "per_lane_note": "same cost applies per warmed model/lane process",
                },
                "suggested_next_step": (
                    "Stop before server wiring. Reproduce the B=max warm then smaller-B FULL_GRAPH replay failure under CUDA_LAUNCH_BLOCKING=1, or force exact-B/per-lane captures only if that proves byte-exact."
                ),
            }
        )
        return 2, summary

    characterization = characterization_before_replay
    print("step2_probe: checking lane-style stream replay", flush=True)
    lane_stream = await run_lane_stream_replay_check(
        server,
        audio_dir=args.audio_dir,
        final_tail_samples=args.final_tail_samples,
    )
    print("step2_probe: checking inactive zero-length row isolation", flush=True)
    inactive = inactive_row_isolation_check(
        server.model,
        decode_cases,
        padded_batch_size=args.batch_max_size,
    )
    print("step2_probe: timing FULL_GRAPH decoder-only cases", flush=True)
    graph_bench = benchmark_decode_distribution(
        server.model,
        selected_cases,
        distribution,
        reps=args.decode_bench_reps,
    )
    bench_comparison = compare_decode_benchmarks(eager_bench, graph_bench)

    main_compare = compare_capture_groups("graph_off", "graph_on", off_main, on_main)
    final_compare = compare_capture_groups(
        "graph_off_finalize",
        "graph_on_finalize",
        off_final_sweep,
        on_final_sweep,
    )
    worst_path, worst_abs = _worst_float([main_compare, final_compare])
    correctness_ok = (
        main_compare.token_text_invariant
        and main_compare.float_state.allclose_1e_4
        and final_compare.token_text_invariant
        and final_compare.float_state.allclose_1e_4
        and bool(inactive.get("ok"))
    )
    reinit_ok = (
        int(replay_stats.get("need_reinit_true", -1)) == 0
        and int(replay_stats.get("graph_reinitialize_calls", -1)) == 0
    )
    stream_ok = bool(lane_stream.get("ok"))
    feasibility_ok = full_graph_confirmed and correctness_ok and reinit_ok and stream_ok

    delta_p50 = (
        bench_comparison.get("combined", {}).get("recoverable_delta_event_ms_p50_weighted")
        or 0.0
    )
    delta_p95 = (
        bench_comparison.get("combined", {}).get("recoverable_delta_event_ms_p95_weighted")
        or delta_p50
    )
    baseline_p50 = 246.0
    baseline_spread = 279.0 - 246.0
    projected_p50 = baseline_p50 - max(0.0, float(delta_p50))
    projected_spread = baseline_spread - max(0.0, float(delta_p95) - float(delta_p50))
    upside_go = feasibility_ok and projected_p50 <= 236.0 and projected_spread <= 25.0
    floor_go = feasibility_ok and full_graph_confirmed
    overall_go = bool(feasibility_ok and (floor_go or upside_go))

    if not main_compare.token_text_invariant:
        summary["blockers"].append("main fixed-trace token/text/y-sequence mismatch")
    if not final_compare.token_text_invariant:
        summary["blockers"].append("finalize B sweep token/text/y-sequence mismatch")
    if not main_compare.float_state.allclose_1e_4 or not final_compare.float_state.allclose_1e_4:
        summary["blockers"].append("float decoder state exceeded allclose tolerance")
    if not inactive.get("ok"):
        summary["blockers"].append("inactive zero-length row isolation failed")
    if not reinit_ok:
        summary["blockers"].append("need_reinit/recapture occurred after max-B warm")
    if not stream_ok:
        summary["blockers"].append("lane-style stream replay failed")

    summary.update(
        {
            "full_graph": {
                "confirmed": full_graph_confirmed,
                "mode_before_warm": mode_before_warm,
                "mode": graph_mode_name(get_decoding_computer(server.model)),
                "computer_class": type(get_decoding_computer(server.model)).__name__,
                "warm": warm,
                "lane_stream": lane_stream,
            },
            "correctness": {
                "overall_ok": correctness_ok,
                "main": _jsonable_compare(main_compare),
                "finalize_b_sweep": _jsonable_compare(final_compare),
                "inactive_row_isolation": inactive,
                "worst_float_path": worst_path,
                "worst_float_max_abs": worst_abs,
            },
            "need_reinit": {
                "initial_max_time": int(getattr(get_decoding_computer(server.model), "INITIAL_MAX_TIME", -1)),
                "warm": warm_stats,
                "replay": replay_stats,
                "characterization": characterization,
            },
            "p50_sizing": {
                "distribution": distribution,
                "selected_cases": {
                    phase: {
                        int(batch): {
                            "name": case.name,
                            "encoded_time": case.encoded_time,
                            "phase": case.phase,
                        }
                        for batch, case in cases.items()
                    }
                    for phase, cases in selected_cases.items()
                },
                "eager": eager_bench,
                "full_graph": graph_bench,
                "comparison": bench_comparison,
            },
            "provisional_roi": {
                "feasibility_ok": feasibility_ok,
                "floor_go": floor_go,
                "upside_go": upside_go,
                "overall_go": overall_go,
                "baseline_p50_ms": baseline_p50,
                "baseline_p95_minus_p50_ms": baseline_spread,
                "projected_p50_ms": projected_p50,
                "projected_p95_minus_p50_ms": projected_spread,
                "recoverable_combined_event_p50_ms": delta_p50,
                "recoverable_combined_event_p95_ms": delta_p95,
            },
            "memory": {
                "before_warm": before_warm,
                "after_warm": after_warm,
                "delta_allocated_bytes": after_warm.get("allocated", 0) - before_warm.get("allocated", 0),
                "delta_reserved_bytes": after_warm.get("reserved", 0) - before_warm.get("reserved", 0),
                "state_tensor_bytes": warm.get("state_tensor_bytes", 0),
                "per_lane_note": "same cost applies per warmed model/lane process",
            },
            "suggested_next_step": (
                "Proceed to Step 3 pytest promotion if Step-2 GO is accepted; otherwise stop before server wiring and resolve the blockers above."
            ),
        }
    )
    return (0 if overall_go else 2), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSION_COUNT)
    parser.add_argument("--normal-chunks", type=int, default=DEFAULT_NORMAL_CHUNKS)
    parser.add_argument("--final-tail-samples", type=int, default=DEFAULT_FINAL_TAIL_SAMPLES)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--batch-max-size", type=int, default=32)
    parser.add_argument("--lane-id", type=int, default=0)
    parser.add_argument("--step2-probe", action="store_true")
    parser.add_argument("--final-sweep-normal-chunks", type=int, default=2)
    parser.add_argument("--decode-bench-reps", type=int, default=12)
    parser.add_argument("--decode-case-samples-per-key", type=int, default=2)
    parser.add_argument("--conc10-records", type=Path, default=DEFAULT_CONC10_RECORDS)
    parser.add_argument("--conc10-srvlog", type=Path, default=DEFAULT_CONC10_SRVLOG)
    parser.add_argument(
        "--findings-path",
        type=Path,
        default=PROJECT / "decoder-graph-probe-findings.md",
    )
    parser.add_argument(
        "--pivot-path",
        type=Path,
        default=PROJECT / "conc10-pivot-findings.md",
    )
    parser.add_argument("--write-findings", action="store_true")
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this harness")

    if args.step2_probe:
        status, summary = await run_step2_probe(args)
        print("decoder_graph_step2_probe_summary")
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
        if args.write_findings:
            write_probe_findings(summary, args.findings_path)
            if not summary.get("provisional_roi", {}).get("upside_go", False):
                write_conc10_pivot_findings(summary, args.pivot_path)
        return status

    print("decoder_graph_harness: loading ASRServer", flush=True)
    print(f"nemo={nemo_identity()}", flush=True)
    server = build_server(
        model=args.model,
        lanes=args.lanes,
        batch_max_size=args.batch_max_size,
        right_context=1,
    )
    print(
        "server_loaded "
        f"model={server.model_name_or_path} lanes={server.model_lanes} "
        f"batch_max_size={server.batch_max_size} decoder_strategy={server.decoder_strategy} "
        "use_cuda_graph_decoder=False",
        flush=True,
    )

    metadata, _captures, comparison = await run_composition_suite(
        server,
        audio_dir=args.audio_dir,
        session_count=args.sessions,
        normal_chunks=args.normal_chunks,
        final_tail_samples=args.final_tail_samples,
        lane_id=args.lane_id,
    )
    summary = _jsonable_summary(metadata, comparison)
    print("decoder_graph_harness_summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if comparison.token_text_invariant and comparison.float_state.allclose_1e_4 else 2


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
