#!/usr/bin/env python3
"""Offline fork-flush replay sweep for synthetic finalization padding.

This harness replays the existing EOU snapshot collection without starting the
websocket server. It loads one ASRServer/model instance, reconstructs sessions
from per-chunk snapshots, overrides final_padding_frames, and runs the existing
fork-finalize path sequentially.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nemotron_speech.server import (  # noqa: E402
    ASRServer,
    ASRSession,
    DEFAULT_MODEL,
    clone_hypotheses_deep,
    clone_tree,
    snapshot_tree_cpu,
    tensor_clone,
)


DEFAULT_COLLECTION_DIR = REPO_ROOT / "eou-collect"
DEFAULT_PROJECT_DIR = REPO_ROOT / "proj-2026-05-19-eou-endpointing"
DEFAULT_DB = REPO_ROOT / "stt-benchmark" / "stt_benchmark_data" / "test_results.db"
DEFAULT_OUTPUT = DEFAULT_PROJECT_DIR / "silence_pad_replay_results.json"
DEFAULT_RUN_TAG = "eou_step2_collect"
DEFAULT_PADS_MS = (320, 480, 640, 800, 1200)
SAMPLE_RATE_HZ = 16_000
CLOCK_COMPATIBLE_SECONDS = 24 * 60 * 60


# Keep the imports requested by the design spec anchored to the server module.
_SERVER_HELPERS = (clone_hypotheses_deep, clone_tree, tensor_clone, snapshot_tree_cpu)


@dataclass(frozen=True)
class ProbeChunk:
    source_line: int
    run_tag: str
    session_id: str
    chunk_index: int
    monotonic_start: float | None
    monotonic_done: float | None
    wall_time_start: float | None
    wall_time_done: float | None
    real_audio_cursor_seconds: float | None


@dataclass
class ProbeSession:
    run_tag: str
    session_id: str
    chunks: list[ProbeChunk]

    @property
    def first_chunk(self) -> ProbeChunk:
        return self.chunks[0]

    @property
    def first_sort_time(self) -> float:
        first = self.first_chunk
        for value in (
            first.wall_time_start,
            first.wall_time_done,
            first.monotonic_start,
            first.monotonic_done,
        ):
            if value is not None:
                return value
        return float(first.chunk_index)


@dataclass(frozen=True)
class TelemetryRow:
    source_line: int
    run_tag: str | None
    benchmark_batch_index: int | None
    sample_id: str | None
    vad_stop: float | None
    final_received: float | None
    vad_stops_count: int | None
    order_time: float | None


@dataclass(frozen=True)
class BaselineTranscript:
    sample_id: str
    service_name: str
    model_name: str
    transcript: str
    duration_seconds: float | None
    dataset_index: int | None


@dataclass(frozen=True)
class ReplayTarget:
    probe: ProbeSession
    telemetry: TelemetryRow
    baseline: BaselineTranscript
    chunk: ProbeChunk
    snapshot_path: Path
    snapshot_time_field: str
    snapshot_done_time: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION_DIR)
    parser.add_argument(
        "--probe-jsonl",
        type=Path,
        default=DEFAULT_COLLECTION_DIR / "telemetry" / f"{DEFAULT_RUN_TAG}.eou_probe.jsonl",
    )
    parser.add_argument(
        "--telemetry-jsonl",
        type=Path,
        default=DEFAULT_COLLECTION_DIR / "telemetry" / f"{DEFAULT_RUN_TAG}.jsonl",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=DEFAULT_COLLECTION_DIR / "snapshots",
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--service-name", default="nemotron_local")
    parser.add_argument("--model-name", default=DEFAULT_RUN_TAG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--right-context", type=int, default=1)
    parser.add_argument(
        "--server-log-level",
        default="INFO",
        help="Loguru level for server/model logs during replay.",
    )
    parser.add_argument(
        "--pads-ms",
        type=int,
        nargs="+",
        default=list(DEFAULT_PADS_MS),
        help="Synthetic silence pad lengths to replay, in milliseconds.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        help="Debug limiter. Default replays every joined session.",
    )
    parser.add_argument("--silent", action="store_true")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        if math.isfinite(number):
            return number
    return None


def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def first_event_value(record: dict[str, Any], key: str) -> Any:
    value = record.get(key)
    if value is not None:
        return value
    events = record.get("finalize_events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and event.get(key) is not None:
                return event.get(key)
    return None


def telemetry_order_time(record: dict[str, Any]) -> float | None:
    for key in (
        "start_time",
        "started_at",
        "first_audio_time",
        "first_audio_wall_time",
        "vad_start",
        "vad_stop",
        "final_received",
    ):
        value = as_float(first_event_value(record, key))
        if value is not None:
            return value
    return None


def safe_tag_filename(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._-") or "tag"


def load_probe_sessions(path: Path) -> tuple[list[ProbeSession], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ProbeChunk]] = defaultdict(list)
    diagnostics: dict[str, Any] = {
        "path": str(path),
        "rows_read": 0,
        "rows_loaded": 0,
        "invalid_rows": 0,
        "skip_reasons": Counter(),
    }

    with path.open("r", encoding="utf-8") as f:
        for source_line, line in enumerate(f, start=1):
            if not line.strip():
                continue
            diagnostics["rows_read"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                diagnostics["invalid_rows"] += 1
                diagnostics["skip_reasons"]["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                diagnostics["invalid_rows"] += 1
                diagnostics["skip_reasons"]["not_object"] += 1
                continue

            run_tag = record.get("run_tag")
            session_id = record.get("session_id")
            chunk_index = as_int(record.get("chunk_index"))
            if (
                not isinstance(run_tag, str)
                or not isinstance(session_id, str)
                or chunk_index is None
            ):
                diagnostics["invalid_rows"] += 1
                diagnostics["skip_reasons"]["missing_required_probe_field"] += 1
                continue

            real_audio_cursor_seconds = as_float(record.get("real_audio_cursor_seconds"))
            if real_audio_cursor_seconds is None:
                cursor_samples = as_float(record.get("real_audio_cursor_samples"))
                if cursor_samples is None:
                    cursor_samples = as_float(record.get("timeline_cursor_samples"))
                if cursor_samples is not None:
                    real_audio_cursor_seconds = cursor_samples / SAMPLE_RATE_HZ

            chunk = ProbeChunk(
                source_line=source_line,
                run_tag=run_tag,
                session_id=session_id,
                chunk_index=chunk_index,
                monotonic_start=as_float(record.get("monotonic_start")),
                monotonic_done=as_float(record.get("monotonic_done")),
                wall_time_start=as_float(record.get("wall_time_start")),
                wall_time_done=as_float(record.get("wall_time_done")),
                real_audio_cursor_seconds=real_audio_cursor_seconds,
            )
            grouped[(run_tag, session_id)].append(chunk)
            diagnostics["rows_loaded"] += 1

    sessions: list[ProbeSession] = []
    for (run_tag, session_id), chunks in grouped.items():
        chunks.sort(
            key=lambda item: (
                item.chunk_index,
                item.wall_time_start if item.wall_time_start is not None else float("inf"),
                item.source_line,
            )
        )
        sessions.append(ProbeSession(run_tag=run_tag, session_id=session_id, chunks=chunks))

    sessions.sort(key=lambda item: item.first_sort_time)
    diagnostics["sessions_loaded"] = len(sessions)
    diagnostics["skip_reasons"] = dict(diagnostics["skip_reasons"])
    return sessions, diagnostics


def load_sample_maps(db_path: Path) -> tuple[dict[int, str], dict[str, float | None], dict[str, int | None]]:
    sample_id_by_batch_index: dict[int, str] = {}
    duration_by_sample_id: dict[str, float | None] = {}
    dataset_index_by_sample_id: dict[str, int | None] = {}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT sample_id, duration_seconds, dataset_index FROM samples ORDER BY dataset_index"
        ).fetchall()
    for ordinal, row in enumerate(rows):
        sample_id = row[0]
        duration_seconds = as_float(row[1])
        dataset_index = as_int(row[2])
        if not isinstance(sample_id, str):
            continue
        duration_by_sample_id[sample_id] = duration_seconds
        dataset_index_by_sample_id[sample_id] = dataset_index
        sample_id_by_batch_index.setdefault(ordinal, sample_id)
        if dataset_index is not None:
            sample_id_by_batch_index.setdefault(dataset_index, sample_id)
    return sample_id_by_batch_index, duration_by_sample_id, dataset_index_by_sample_id


def load_telemetry_rows(
    path: Path,
    sample_id_by_batch_index: dict[int, str],
) -> tuple[list[TelemetryRow], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "path": str(path),
        "rows_read": 0,
        "rows_loaded": 0,
        "invalid_rows": 0,
        "skip_reasons": Counter(),
        "rows_with_batch_index": 0,
    }
    rows: list[TelemetryRow] = []

    with path.open("r", encoding="utf-8") as f:
        for source_line, line in enumerate(f, start=1):
            if not line.strip():
                continue
            diagnostics["rows_read"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                diagnostics["invalid_rows"] += 1
                diagnostics["skip_reasons"]["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                diagnostics["invalid_rows"] += 1
                diagnostics["skip_reasons"]["not_object"] += 1
                continue

            batch_index = as_int(record.get("benchmark_batch_index"))
            if batch_index is not None:
                diagnostics["rows_with_batch_index"] += 1
            run_tag = record.get("run_tag") if isinstance(record.get("run_tag"), str) else None

            rows.append(
                TelemetryRow(
                    source_line=source_line,
                    run_tag=run_tag,
                    benchmark_batch_index=batch_index,
                    sample_id=sample_id_by_batch_index.get(batch_index)
                    if batch_index is not None
                    else None,
                    vad_stop=as_float(first_event_value(record, "vad_stop")),
                    final_received=as_float(first_event_value(record, "final_received")),
                    vad_stops_count=as_int(record.get("vad_stops")),
                    order_time=telemetry_order_time(record),
                )
            )
            diagnostics["rows_loaded"] += 1

    diagnostics["skip_reasons"] = dict(diagnostics["skip_reasons"])
    return rows, diagnostics


def load_baselines(
    db_path: Path,
    service_name: str,
    model_name: str,
    duration_by_sample_id: dict[str, float | None],
    dataset_index_by_sample_id: dict[str, int | None],
) -> dict[str, BaselineTranscript]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT sample_id, service_name, model_name, transcription
            FROM results
            WHERE service_name = ? AND model_name = ? AND error IS NULL
            ORDER BY sample_id
            """,
            (service_name, model_name),
        ).fetchall()

    baselines: dict[str, BaselineTranscript] = {}
    for sample_id, svc, model, transcription in rows:
        if not isinstance(sample_id, str):
            continue
        baselines[sample_id] = BaselineTranscript(
            sample_id=sample_id,
            service_name=str(svc),
            model_name=str(model),
            transcript=transcription if isinstance(transcription, str) else "",
            duration_seconds=duration_by_sample_id.get(sample_id),
            dataset_index=dataset_index_by_sample_id.get(sample_id),
        )
    return baselines


def sort_telemetry_for_temporal_join(rows: list[TelemetryRow]) -> list[TelemetryRow]:
    if all(row.benchmark_batch_index is not None for row in rows):
        return sorted(rows, key=lambda row: (row.benchmark_batch_index, row.source_line))
    if all(row.order_time is not None for row in rows):
        return sorted(rows, key=lambda row: (row.order_time, row.source_line))
    return sorted(rows, key=lambda row: row.source_line)


def join_sessions(
    probe_sessions: list[ProbeSession],
    telemetry_rows: list[TelemetryRow],
) -> tuple[list[tuple[ProbeSession, TelemetryRow]], dict[str, Any], Counter[str]]:
    skipped: Counter[str] = Counter()
    ordered_sessions = sorted(probe_sessions, key=lambda item: item.first_sort_time)
    ordered_rows = sort_telemetry_for_temporal_join(telemetry_rows)
    pair_count = min(len(ordered_sessions), len(ordered_rows))
    joined = list(zip(ordered_sessions[:pair_count], ordered_rows[:pair_count]))

    if len(ordered_sessions) > pair_count:
        skipped["no_telemetry_join"] += len(ordered_sessions) - pair_count
    if len(ordered_rows) > pair_count:
        skipped["no_probe_join"] += len(ordered_rows) - pair_count

    telemetry_order_basis = "source_line"
    if telemetry_rows and all(row.benchmark_batch_index is not None for row in telemetry_rows):
        telemetry_order_basis = "benchmark_batch_index"
    elif telemetry_rows and all(row.order_time is not None for row in telemetry_rows):
        telemetry_order_basis = "telemetry_time"

    diagnostics = {
        "method": "temporal_order_fallback",
        "probe_session_order_basis": "first chunk wall_time_start/done, then monotonic_start/done",
        "telemetry_order_basis": telemetry_order_basis,
        "probe_sessions_seen": len(probe_sessions),
        "telemetry_rows_seen": len(telemetry_rows),
        "joined_before_filters": len(joined),
        "limitation": (
            "client finalize telemetry lacks session_id; session_id->benchmark_batch_index "
            "is inferred by ordering probe sessions by first chunk and telemetry rows by "
            "benchmark_batch_index/source order. This assumes one websocket session per sample "
            "and sequential benchmark processing."
        ),
    }
    return joined, diagnostics, skipped


def choose_snapshot_chunk(
    session: ProbeSession,
    vad_stop: float,
) -> tuple[ProbeChunk | None, str | None, float | None, str | None]:
    """Choose closest chunk done at or before vad_stop on a compatible clock."""
    fields = ("wall_time_done", "monotonic_done")
    for field_name in fields:
        values = [
            getattr(chunk, field_name)
            for chunk in session.chunks
            if getattr(chunk, field_name) is not None
        ]
        if not values:
            continue
        min_delta = min(abs(vad_stop - value) for value in values)
        if min_delta > CLOCK_COMPATIBLE_SECONDS:
            continue

        candidates = [
            chunk for chunk in session.chunks if (getattr(chunk, field_name) or float("inf")) <= vad_stop
        ]
        if not candidates:
            return None, field_name, None, "no_probe_chunk_before_vad_stop"
        best = max(candidates, key=lambda chunk: getattr(chunk, field_name) or float("-inf"))
        return best, field_name, getattr(best, field_name), None

    return None, None, None, "no_probe_clock_comparable_with_vad_stop"


def build_targets(
    joined: list[tuple[ProbeSession, TelemetryRow]],
    baselines: dict[str, BaselineTranscript],
    snapshot_dir: Path,
) -> tuple[list[ReplayTarget], list[dict[str, Any]], Counter[str]]:
    targets: list[ReplayTarget] = []
    skipped_records: list[dict[str, Any]] = []
    skipped_counts: Counter[str] = Counter()

    for probe, telemetry in joined:
        base_skip = {
            "session_id": probe.session_id,
            "run_tag": probe.run_tag,
            "benchmark_batch_index": telemetry.benchmark_batch_index,
            "sample_id": telemetry.sample_id,
        }
        if telemetry.vad_stop is None:
            skipped_counts["no_vad_stop"] += 1
            skipped_records.append({**base_skip, "reason": "no_vad_stop"})
            continue
        if telemetry.sample_id is None:
            skipped_counts["no_sample_id"] += 1
            skipped_records.append({**base_skip, "reason": "no_sample_id"})
            continue
        baseline = baselines.get(telemetry.sample_id)
        if baseline is None:
            skipped_counts["no_baseline_transcript"] += 1
            skipped_records.append({**base_skip, "reason": "no_baseline_transcript"})
            continue

        chunk, time_field, done_time, reason = choose_snapshot_chunk(probe, telemetry.vad_stop)
        if chunk is None or time_field is None or done_time is None:
            skipped_counts[reason or "no_snapshot_chunk"] += 1
            skipped_records.append({**base_skip, "reason": reason or "no_snapshot_chunk"})
            continue

        snapshot_path = (
            snapshot_dir
            / f"{safe_tag_filename(probe.run_tag)}_{safe_tag_filename(probe.session_id)}"
            f"_chunk{chunk.chunk_index:06d}.pt"
        )
        if not snapshot_path.exists():
            skipped_counts["snapshot_missing"] += 1
            skipped_records.append(
                {
                    **base_skip,
                    "reason": "snapshot_missing",
                    "chunk_index": chunk.chunk_index,
                    "snapshot_path": str(snapshot_path),
                }
            )
            continue

        targets.append(
            ReplayTarget(
                probe=probe,
                telemetry=telemetry,
                baseline=baseline,
                chunk=chunk,
                snapshot_path=snapshot_path,
                snapshot_time_field=time_field,
                snapshot_done_time=done_time,
            )
        )

    return targets, skipped_records, skipped_counts


def cuda_tree(obj: Any, memo: dict[int, Any] | None = None) -> Any:
    """Recursively move snapshot tensors to CUDA while preserving object shape."""
    if memo is None:
        memo = {}

    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if torch.is_tensor(obj):
        return obj.detach().cuda()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        cloned_list: list[Any] = []
        memo[oid] = cloned_list
        cloned_list.extend(cuda_tree(item, memo) for item in obj)
        return cloned_list
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        cloned_tuple = tuple(cuda_tree(item, memo) for item in obj)
        memo[oid] = cloned_tuple
        return cloned_tuple
    if isinstance(obj, dict):
        cloned_dict: dict[Any, Any] = {}
        memo[oid] = cloned_dict
        for key, value in obj.items():
            cloned_dict[cuda_tree(key, memo)] = cuda_tree(value, memo)
        return cloned_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for field in dataclasses.fields(obj):
            setattr(cloned_obj, field.name, cuda_tree(getattr(obj, field.name), memo))
        return cloned_obj

    module = obj.__class__.__module__
    if hasattr(obj, "__dict__") and (module.startswith("nemo.") or module.startswith("nemo_")):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for key, value in vars(obj).items():
            setattr(cloned_obj, key, cuda_tree(value, memo))
        return cloned_obj

    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def ensure_np_float32(value: Any) -> np.ndarray:
    if value is None:
        return np.array([], dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=True)
    return np.asarray(value, dtype=np.float32).copy()


def reconstruct_session(session_id: str, snapshot: dict[str, Any]) -> ASRSession:
    session = ASRSession(id=session_id, websocket=None)
    session.cache_last_channel = snapshot.get("cache_last_channel")
    session.cache_last_time = snapshot.get("cache_last_time")
    session.cache_last_channel_len = snapshot.get("cache_last_channel_len")
    session.previous_hypotheses = snapshot.get("previous_hypotheses")
    session.pred_out_stream = snapshot.get("pred_out_stream")
    session.pending_audio = ensure_np_float32(snapshot.get("pending_audio"))
    session.accumulated_audio = session.pending_audio
    session.raw_audio_ring = ensure_np_float32(snapshot.get("raw_audio_ring"))
    session.mel_frame_ring = snapshot.get("mel_frame_ring")
    session.emitted_frames = int(snapshot.get("emitted_frames") or 0)
    session.synthetic_prefix_samples = int(snapshot.get("synthetic_prefix_samples") or 0)
    session.total_audio_samples = int(snapshot.get("total_audio_samples") or 0)
    session.current_text = ""
    session.committed_text = ""
    session.last_emitted_text = ""
    session.continuous_emitted_text = ""
    return session


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def levenshtein(a: str | list[str], b: str | list[str]) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, a_item in enumerate(a, start=1):
        current = [i]
        for j, b_item in enumerate(b, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (0 if a_item == b_item else 1)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def percentile(values: list[int | float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def text_excerpt(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def replay_one_cell(
    server: ASRServer,
    target: ReplayTarget,
    cpu_snapshot: dict[str, Any],
    pad_ms: int,
) -> str:
    server.final_padding_frames = int(round(pad_ms / 10))
    snapshot = cuda_tree(cpu_snapshot)
    session = reconstruct_session(target.probe.session_id, snapshot)
    fork = server._build_continuous_finalize_fork(session)
    text = server._process_final_chunk(fork)
    if text is None:
        raise RuntimeError("_process_final_chunk returned None")
    return text


def replay_target(
    server: ASRServer,
    target: ReplayTarget,
    pads_ms: list[int],
) -> list[dict[str, Any]]:
    cpu_snapshot = torch.load(target.snapshot_path, map_location="cpu", weights_only=False)
    if not isinstance(cpu_snapshot, dict):
        raise TypeError(f"snapshot did not load as dict: {target.snapshot_path}")

    pad320_text: str | None = None
    records: list[dict[str, Any]] = []
    baseline = target.baseline.transcript
    baseline_words = normalize_words(baseline)

    for pad_ms in pads_ms:
        replay_text = replay_one_cell(server, target, cpu_snapshot, pad_ms)
        baseline_comparable = target.telemetry.vad_stops_count == 1
        if baseline_comparable:
            baseline_comparable_reason = "single_vad_stop"
        elif target.telemetry.vad_stops_count is None:
            baseline_comparable_reason = "unknown_vad_stop_count"
        else:
            baseline_comparable_reason = "multi_vad_stop_baseline_is_aggregate"
        if pad_ms == 320:
            pad320_text = replay_text
        char_edit_distance = levenshtein(replay_text, baseline)
        word_edit_distance = levenshtein(normalize_words(replay_text), baseline_words)
        reference_words = len(baseline_words)
        wer = word_edit_distance / reference_words if reference_words else (0.0 if not replay_text else 1.0)
        records.append(
            {
                "session_id": target.probe.session_id,
                "run_tag": target.probe.run_tag,
                "sample_id": target.baseline.sample_id,
                "benchmark_batch_index": target.telemetry.benchmark_batch_index,
                "dataset_index": target.baseline.dataset_index,
                "vad_stop": target.telemetry.vad_stop,
                "vad_stops_count": target.telemetry.vad_stops_count,
                "baseline_comparable": baseline_comparable,
                "baseline_comparable_reason": baseline_comparable_reason,
                "snapshot_path": str(target.snapshot_path),
                "snapshot_chunk_index": target.chunk.chunk_index,
                "snapshot_time_field": target.snapshot_time_field,
                "snapshot_done_time": target.snapshot_done_time,
                "snapshot_lag_ms": (
                    round((target.telemetry.vad_stop - target.snapshot_done_time) * 1000.0, 3)
                    if target.telemetry.vad_stop is not None
                    else None
                ),
                "snapshot_real_audio_cursor_seconds": target.chunk.real_audio_cursor_seconds,
                "pad_ms": pad_ms,
                "final_padding_frames": int(round(pad_ms / 10)),
                "baseline_transcript": baseline,
                "replay_transcript": replay_text,
                "edit_distance": char_edit_distance,
                "word_edit_distance": word_edit_distance,
                "wer": wer,
                "byte_identical": replay_text == baseline,
                "within_2": char_edit_distance <= 2,
                "within_5": char_edit_distance <= 5,
                "wer_relevant_diff": word_edit_distance > 0,
                "edit_distance_vs_pad320": None,
                "different_from_pad320": None,
                "length_mismatch_flag": False,
            }
        )

    if pad320_text is not None:
        for record in records:
            replay_text = record["replay_transcript"]
            record["edit_distance_vs_pad320"] = levenshtein(replay_text, pad320_text)
            record["different_from_pad320"] = replay_text != pad320_text

    for record in records:
        replay_len = len(record["replay_transcript"])
        baseline_len = len(baseline)
        record["length_mismatch_flag"] = is_obvious_length_mismatch(replay_len, baseline_len)

    return records


def is_obvious_length_mismatch(replay_len: int, baseline_len: int) -> bool:
    if baseline_len == 0:
        return replay_len > 20
    length_delta = abs(replay_len - baseline_len)
    shorter = min(replay_len, baseline_len)
    longer = max(replay_len, baseline_len)
    return length_delta >= 120 and shorter / longer < 0.35


def fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_by_pad(records: list[dict[str, Any]], pads_ms: list[int]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for pad_ms in pads_ms:
        pad_records = [record for record in records if record["pad_ms"] == pad_ms]
        edit_distances = [int(record["edit_distance"]) for record in pad_records]
        n = len(pad_records)
        summary.append(
            {
                "pad_ms": pad_ms,
                "n": n,
                "byte_identical": sum(1 for record in pad_records if record["byte_identical"]),
                "byte_identical_fraction": fraction(
                    sum(1 for record in pad_records if record["byte_identical"]), n
                ),
                "within_2": sum(1 for record in pad_records if record["within_2"]),
                "within_2_fraction": fraction(sum(1 for record in pad_records if record["within_2"]), n),
                "within_5": sum(1 for record in pad_records if record["within_5"]),
                "within_5_fraction": fraction(sum(1 for record in pad_records if record["within_5"]), n),
                "wer_relevant_diff": sum(1 for record in pad_records if record["wer_relevant_diff"]),
                "wer_relevant_diff_fraction": fraction(
                    sum(1 for record in pad_records if record["wer_relevant_diff"]), n
                ),
                "median_edit_distance": statistics.median(edit_distances)
                if edit_distances
                else None,
                "p95_edit_distance": percentile(edit_distances, 95),
                "median_wer": statistics.median([record["wer"] for record in pad_records])
                if pad_records
                else None,
                "p95_wer": percentile([record["wer"] for record in pad_records], 95),
            }
        )
    return summary


def summarize_by_pad_filtered(
    records: list[dict[str, Any]],
    pads_ms: list[int],
    *,
    baseline_comparable: bool,
) -> list[dict[str, Any]]:
    return summarize_by_pad(
        [record for record in records if record.get("baseline_comparable") is baseline_comparable],
        pads_ms,
    )


def convergence_summary(records: list[dict[str, Any]], pads_ms: list[int]) -> list[dict[str, Any]]:
    by_pad = {pad_ms: [record for record in records if record["pad_ms"] == pad_ms] for pad_ms in pads_ms}
    pad320 = by_pad.get(320, [])
    n320 = len(pad320)
    summary: list[dict[str, Any]] = []
    for pad_ms in pads_ms:
        if pad_ms == 320:
            continue
        pad_records = by_pad.get(pad_ms, [])
        diffs = sum(1 for record in pad_records if record.get("different_from_pad320"))
        edit_distances = [
            int(record["edit_distance_vs_pad320"])
            for record in pad_records
            if record.get("edit_distance_vs_pad320") is not None
        ]
        summary.append(
            {
                "pad_ms": pad_ms,
                "n": len(pad_records),
                "baseline_pad_ms": 320,
                "sessions_different_from_pad320": diffs,
                "different_from_pad320_fraction": fraction(diffs, n320 if n320 else len(pad_records)),
                "median_edit_distance_vs_pad320": statistics.median(edit_distances)
                if edit_distances
                else None,
                "p95_edit_distance_vs_pad320": percentile(edit_distances, 95),
            }
        )
    return summary


def top_divergent_pad320(records: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    pad_records = [record for record in records if record["pad_ms"] == 320]
    pad_records.sort(key=lambda record: (record["edit_distance"], record["sample_id"]), reverse=True)
    return [
        {
            "session_id": record["session_id"],
            "sample_id": record["sample_id"],
            "benchmark_batch_index": record["benchmark_batch_index"],
            "edit_distance": record["edit_distance"],
            "word_edit_distance": record["word_edit_distance"],
            "replay_excerpt": text_excerpt(record["replay_transcript"]),
            "baseline_excerpt": text_excerpt(record["baseline_transcript"]),
        }
        for record in pad_records[:limit]
    ]


def print_summary(
    summary: list[dict[str, Any]],
    comparable_summary: list[dict[str, Any]],
    convergence: list[dict[str, Any]],
    top_divergent: list[dict[str, Any]],
    n_sessions_analyzed: int,
    n_baseline_comparable_sessions: int,
    skipped_counts: Counter[str],
    output_json: Path,
) -> None:
    def fmt(value: Any, width: int, precision: int | None = None) -> str:
        if value is None:
            return f"{'NA':>{width}}"
        if precision is None:
            return f"{value:>{width}}"
        return f"{float(value):>{width}.{precision}f}"

    print(f"n_sessions_analyzed: {n_sessions_analyzed}")
    print(f"n_baseline_comparable_sessions: {n_baseline_comparable_sessions}")
    print(f"n_skipped: {sum(skipped_counts.values())}")
    if skipped_counts:
        print("skipped_reasons:")
        for reason, count in sorted(skipped_counts.items()):
            print(f"  {reason}: {count}")

    def print_pad_table(title: str, rows: list[dict[str, Any]]) -> None:
        print(f"\n{title}")
        print(
            "pad_ms  n   byte_identical  within_2  within_5  "
            "wer_diff  median_ed  p95_ed"
        )
        for row in rows:
            print(
                f"{row['pad_ms']:>6}  {row['n']:>3}  "
                f"{row['byte_identical_fraction'] * 100:>13.1f}%  "
                f"{row['within_2_fraction'] * 100:>8.1f}%  "
                f"{row['within_5_fraction'] * 100:>8.1f}%  "
                f"{row['wer_relevant_diff_fraction'] * 100:>7.1f}%  "
                f"{fmt(row['median_edit_distance'], 9)}  "
                f"{fmt(row['p95_edit_distance'], 6, 1)}"
            )

    print_pad_table(
        "Per-pad summary (all sessions; multi-VAD baselines are aggregate transcripts):",
        summary,
    )
    print_pad_table(
        "Per-pad summary (baseline-comparable single-VAD sessions):",
        comparable_summary,
    )

    print("\nConvergence vs pad=320:")
    print(
        "pad_ms  n   sessions_diff  diff_fraction  median_ed_vs_320  p95_ed_vs_320"
    )
    for row in convergence:
        print(
            f"{row['pad_ms']:>6}  {row['n']:>3}  "
            f"{row['sessions_different_from_pad320']:>13}  "
            f"{row['different_from_pad320_fraction'] * 100:>12.1f}%  "
            f"{fmt(row['median_edit_distance_vs_pad320'], 16)}  "
            f"{fmt(row['p95_edit_distance_vs_pad320'], 14, 1)}"
        )

    print("\nTop-5 divergent sessions at pad=320:")
    for row in top_divergent:
        print(
            f"- session_id={row['session_id']} sample_id={row['sample_id']} "
            f"edit_distance={row['edit_distance']} word_edit_distance={row['word_edit_distance']}"
        )
        print(f"  replay:   {row['replay_excerpt']}")
        print(f"  baseline: {row['baseline_excerpt']}")

    print("\nCaveats:")
    print("- CUDA and cuFFT nondeterminism can prevent byte-exact reproduction.")
    print(
        "- Baselines came from the original streaming run; replay uses snapshot reconstruction, "
        "so small differences are expected."
    )
    print(
        "- The session_id to benchmark_batch_index join is a temporal-order fallback; "
        "length_mismatch_flag is recorded per cell for obviously suspicious pairs."
    )
    print(f"\nWrote JSON: {output_json}")


def load_server(args: argparse.Namespace) -> ASRServer:
    logger.remove()
    logger.add(sys.stderr, level=args.server_log_level.upper())
    os.environ.setdefault("NEMOTRON_RUN_TAG", args.run_tag)
    os.environ.setdefault("NEMOTRON_EOU_PROBE", "1")
    os.environ.setdefault("NEMOTRON_DECODING", "greedy")
    server = ASRServer(
        model=args.model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
    )
    server.load_model()
    server.model_loaded = True
    return server


def main() -> int:
    args = parse_args()
    start_time = time.perf_counter()

    sample_id_by_batch_index, duration_by_sample_id, dataset_index_by_sample_id = load_sample_maps(args.db)
    probe_sessions, probe_diag = load_probe_sessions(args.probe_jsonl)
    telemetry_rows, telemetry_diag = load_telemetry_rows(args.telemetry_jsonl, sample_id_by_batch_index)
    baselines = load_baselines(
        args.db,
        args.service_name,
        args.model_name,
        duration_by_sample_id,
        dataset_index_by_sample_id,
    )
    joined, join_diag, join_skips = join_sessions(probe_sessions, telemetry_rows)
    targets, skipped_records, skipped_counts = build_targets(joined, baselines, args.snapshot_dir)
    skipped_counts.update(join_skips)

    if args.max_sessions is not None:
        targets = targets[: args.max_sessions]

    if not args.silent:
        print(
            f"Loaded inputs: probe_sessions={len(probe_sessions)} "
            f"telemetry_rows={len(telemetry_rows)} baselines={len(baselines)} "
            f"targets={len(targets)}"
        )
        print("Loading ASR model without starting websocket server...")

    server = load_server(args)

    all_records: list[dict[str, Any]] = []
    session_errors: list[dict[str, Any]] = []
    pads_ms = list(args.pads_ms)

    for index, target in enumerate(targets, start=1):
        if not args.silent:
            print(
                f"[{index:03d}/{len(targets):03d}] session={target.probe.session_id} "
                f"sample={target.baseline.sample_id} chunk={target.chunk.chunk_index}"
            )
        try:
            all_records.extend(replay_target(server, target, pads_ms))
        except Exception as exc:
            skipped_counts["replay_error"] += 1
            session_errors.append(
                {
                    "session_id": target.probe.session_id,
                    "sample_id": target.baseline.sample_id,
                    "benchmark_batch_index": target.telemetry.benchmark_batch_index,
                    "snapshot_path": str(target.snapshot_path),
                    "reason": "replay_error",
                    "error": repr(exc),
                }
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    n_sessions_analyzed = len({record["session_id"] for record in all_records if record["pad_ms"] == pads_ms[0]})
    summary = summarize_by_pad(all_records, pads_ms)
    comparable_summary = summarize_by_pad_filtered(
        all_records,
        pads_ms,
        baseline_comparable=True,
    )
    multi_vad_summary = summarize_by_pad_filtered(
        all_records,
        pads_ms,
        baseline_comparable=False,
    )
    convergence = convergence_summary(all_records, pads_ms)
    top_divergent = top_divergent_pad320(all_records)

    length_mismatch_cells = [record for record in all_records if record.get("length_mismatch_flag")]
    length_mismatch_sessions = sorted({record["session_id"] for record in length_mismatch_cells})
    n_baseline_comparable_sessions = len(
        {record["session_id"] for record in all_records if record.get("baseline_comparable")}
    )

    output = {
        "metadata": {
            "created_at_unix": time.time(),
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "repo_root": str(REPO_ROOT),
            "collection_dir": str(args.collection_dir),
            "probe_jsonl": str(args.probe_jsonl),
            "telemetry_jsonl": str(args.telemetry_jsonl),
            "snapshot_dir": str(args.snapshot_dir),
            "db": str(args.db),
            "run_tag": args.run_tag,
            "service_name": args.service_name,
            "model_name": args.model_name,
            "pads_ms": pads_ms,
            "n_sessions_analyzed": n_sessions_analyzed,
            "n_baseline_comparable_sessions": n_baseline_comparable_sessions,
            "n_skipped": sum(skipped_counts.values()),
            "n_records": len(all_records),
            "length_mismatch_cells": len(length_mismatch_cells),
            "length_mismatch_sessions": length_mismatch_sessions,
            "caveats": [
                "CUDA and cuFFT nondeterminism can prevent byte-exact reproduction.",
                "Baselines came from the original streaming run; replay uses snapshot reconstruction.",
                "The session_id to benchmark_batch_index join is a temporal-order fallback.",
            ],
        },
        "diagnostics": {
            "probe": probe_diag,
            "telemetry": telemetry_diag,
            "join": join_diag,
            "skipped_counts": dict(skipped_counts),
        },
        "summary_by_pad": summary,
        "summary_by_pad_baseline_comparable": comparable_summary,
        "summary_by_pad_multi_vad_aggregate_baseline": multi_vad_summary,
        "convergence_vs_pad320": convergence,
        "top_divergent_pad320": top_divergent,
        "skipped": skipped_records + session_errors,
        "records": all_records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    if not args.silent:
        print_summary(
            summary,
            comparable_summary,
            convergence,
            top_divergent,
            n_sessions_analyzed,
            n_baseline_comparable_sessions,
            skipped_counts,
            args.output_json,
        )

    return 0 if not session_errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
