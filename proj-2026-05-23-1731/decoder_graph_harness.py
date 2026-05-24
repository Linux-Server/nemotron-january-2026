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
import dataclasses
import gc
import hashlib
import inspect
import json
import os
import sys
import time
from collections.abc import Iterable
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
) -> list[ClipSpec]:
    """Pick distinct real clips long enough for the fixed normal+final trace."""
    if session_count < 2:
        raise ValueError("session_count must be >= 2")
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
    if len(candidates) < session_count:
        raise RuntimeError(
            f"only found {len(candidates)} clips with >= {total_needed} samples; "
            f"need {session_count}"
        )

    # Prefer longer clips so the chosen prefix is real speech/audio, not a tiny
    # edge case. Keep session IDs stable and short for comparison output.
    selected = sorted(candidates, reverse=True)[:session_count]
    selected = sorted(selected, key=lambda row: row[1])
    clips: list[ClipSpec] = []
    for index, (_duration, _sample_id, path, _samples) in enumerate(selected):
        pcm = np.fromfile(path, dtype=np.int16)
        clips.append(
            ClipSpec(
                session_id=f"s{index}",
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


def capture_event(
    *,
    scenario: str,
    session_id: str,
    logical_event: str,
    text: Optional[str],
    state_session: Any,
) -> dict[str, Any]:
    hyp = _session_hypothesis(state_session)
    return {
        "scenario": scenario,
        "session_id": session_id,
        "event": logical_event,
        "text": "" if text is None else str(text),
        "tokens": _token_ids_from_hyp(hyp),
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
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this harness")

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
