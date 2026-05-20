#!/usr/bin/env python3
"""Offline ROC analyzer for Step-3 EOU candidate endpoint signals.

Pure CPU/JSONL analysis over Step-1/2 probe rows and client finalize telemetry.
No server, GPU, framework, or benchmark source interaction is required.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BLANK_K_VALUES = (1, 2, 3, 4, 5, 6, 8, 10)
HYP_K_VALUES = (1, 2, 3, 4, 5, 6, 8, 10)
CONFIDENCE_TAUS = (0.50, 0.60, 0.70, 0.80, 0.90)
CONFIDENCE_WINDOWS_MS = (80, 160, 240, 320)
GO_LATENCY_TARGETS_MS = (0, 50, 100, 150, 200)
GO_FALSE_RATE_TARGETS = (0.005, 0.01, 0.02, 0.05)
SAMPLE_RATE_HZ = 16_000
ACOUSTIC_FRAME_MS = 20
ACOUSTIC_UNCERTAINTY_MS = 40


@dataclass(frozen=True)
class ProbeChunk:
    source_line: int
    run_tag: str
    session_id: str
    chunk_index: int
    y_len: int
    chunk_blank: bool
    confidence_values: tuple[float, ...]
    audio_elapsed_s: float | None
    wall_time_start: float | None
    wall_time_done: float | None
    monotonic_start: float | None
    monotonic_done: float | None


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

    @property
    def audio_start_wallclock(self) -> float | None:
        """Approximate raw-audio time origin on the same clock as vad_stop.

        The probe stream records chunk processing timestamps plus the cumulative
        real-audio cursor. Anchoring audio t=0 at first_chunk.wall_time_done -
        first_chunk.real_audio_cursor_seconds makes acoustic-stop estimates
        comparable to client `time.time()` Silero/finalize telemetry.
        """

        first = self.first_chunk
        if first.wall_time_done is None or first.audio_elapsed_s is None:
            if first.wall_time_start is not None:
                return first.wall_time_start
            return None
        return first.wall_time_done - first.audio_elapsed_s


@dataclass(frozen=True)
class TelemetryRow:
    source_line: int
    run_tag: str | None
    benchmark_batch_index: int | None
    sample_id: str | None
    session_id: str | None
    vad_stop: float | None
    final_received: float | None
    finalize_events: list[dict[str, Any]]
    order_time: float | None


@dataclass
class JoinedSession:
    probe: ProbeSession
    telemetry: TelemetryRow
    estimated_acoustic_stop: float | None
    acoustic_status: str
    audio_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-jsonl", type=Path, required=True)
    parser.add_argument("--finalize-jsonl", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument(
        "--early-margin-ms",
        type=float,
        default=100.0,
        help="False-early-fire margin before Silero vad_stop in milliseconds.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=500.0,
        help="Minimum real-audio elapsed before any signal may fire.",
    )
    parser.add_argument(
        "--samples-db",
        type=Path,
        help=(
            "Optional samples sqlite DB. Defaults to test_results.db, with results.db "
            "used as a fill-in if the checkout's test DB is truncated."
        ),
    )
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
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


def flatten_numbers(value: Any) -> list[float]:
    numbers: list[float] = []
    if isinstance(value, bool):
        return numbers
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            numbers.append(number)
        return numbers
    if isinstance(value, list):
        for item in value:
            numbers.extend(flatten_numbers(item))
    return numbers


def is_chunk_blank(record: dict[str, Any]) -> bool:
    new_tokens = record.get("new_tokens")
    no_new_tokens = isinstance(new_tokens, list) and len(new_tokens) == 0
    frame_alignment = record.get("frame_alignment")
    alignment_blank = (
        isinstance(frame_alignment, list)
        and len(frame_alignment) > 0
        and all(isinstance(frame, dict) and frame.get("all_blank") is True for frame in frame_alignment)
    )
    return no_new_tokens or alignment_blank


def parse_probe_chunk(record: Any, source_line: int) -> tuple[ProbeChunk | None, str | None]:
    if not isinstance(record, dict):
        return None, "not_object"
    session_id = record.get("session_id")
    run_tag = record.get("run_tag")
    chunk_index = as_int(record.get("chunk_index"))
    y_sequence = record.get("y_sequence")
    new_tokens = record.get("new_tokens")
    if (
        not isinstance(session_id, str)
        or not isinstance(run_tag, str)
        or chunk_index is None
        or not isinstance(y_sequence, list)
        or not isinstance(new_tokens, list)
    ):
        return None, "missing_or_bad_required_field"

    audio_elapsed = as_float(record.get("real_audio_cursor_seconds"))
    if audio_elapsed is None:
        samples = as_float(record.get("real_audio_cursor_samples"))
        if samples is None:
            samples = as_float(record.get("timeline_cursor_samples"))
        if samples is not None:
            audio_elapsed = samples / SAMPLE_RATE_HZ

    return (
        ProbeChunk(
            source_line=source_line,
            run_tag=run_tag,
            session_id=session_id,
            chunk_index=chunk_index,
            y_len=len(y_sequence),
            chunk_blank=is_chunk_blank(record),
            confidence_values=tuple(flatten_numbers(record.get("frame_confidence"))),
            audio_elapsed_s=audio_elapsed,
            wall_time_start=as_float(record.get("wall_time_start")),
            wall_time_done=as_float(record.get("wall_time_done")),
            monotonic_start=as_float(record.get("monotonic_start")),
            monotonic_done=as_float(record.get("monotonic_done")),
        ),
        None,
    )


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
            chunk, reason = parse_probe_chunk(record, source_line)
            if chunk is None:
                diagnostics["invalid_rows"] += 1
                diagnostics["skip_reasons"][reason or "invalid"] += 1
                continue
            grouped[(chunk.run_tag, chunk.session_id)].append(chunk)
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


def telemetry_session_id(record: dict[str, Any]) -> str | None:
    value = record.get("session_id")
    if isinstance(value, str) and value:
        return value
    events = record.get("finalize_events")
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict):
                event_value = event.get("session_id")
                if isinstance(event_value, str) and event_value:
                    return event_value
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


def load_sample_id_map(samples_dbs: list[Path]) -> tuple[dict[int, str], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "paths": [str(path) for path in samples_dbs],
        "available": False,
        "rows_by_path": {},
        "mapped_indices": 0,
        "error": None,
        "errors_by_path": {},
    }
    mapping: dict[int, str] = {}
    for samples_db in samples_dbs:
        if not samples_db.exists():
            diagnostics["errors_by_path"][str(samples_db)] = "missing"
            continue
        try:
            with sqlite3.connect(samples_db) as conn:
                rows = conn.execute(
                    "SELECT sample_id, dataset_index FROM samples ORDER BY dataset_index"
                ).fetchall()
        except sqlite3.Error as exc:
            diagnostics["errors_by_path"][str(samples_db)] = str(exc)
            continue

        diagnostics["available"] = True
        diagnostics["rows_by_path"][str(samples_db)] = len(rows)
        for ordinal, row in enumerate(rows):
            sample_id = row[0]
            dataset_index = as_int(row[1])
            if not isinstance(sample_id, str):
                continue
            mapping.setdefault(ordinal, sample_id)
            if dataset_index is not None:
                mapping.setdefault(dataset_index, sample_id)
    if not diagnostics["available"]:
        diagnostics["error"] = "no_samples_db_available"
    diagnostics["mapped_indices"] = len(mapping)
    return mapping, diagnostics


def parse_finalize_events(record: dict[str, Any]) -> list[dict[str, Any]]:
    events = record.get("finalize_events")
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


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
        "rows_with_session_id": 0,
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
            session_id = telemetry_session_id(record)
            if session_id is not None:
                diagnostics["rows_with_session_id"] += 1
            run_tag = record.get("run_tag") if isinstance(record.get("run_tag"), str) else None
            rows.append(
                TelemetryRow(
                    source_line=source_line,
                    run_tag=run_tag,
                    benchmark_batch_index=batch_index,
                    sample_id=sample_id_by_batch_index.get(batch_index) if batch_index is not None else None,
                    session_id=session_id,
                    vad_stop=as_float(first_event_value(record, "vad_stop")),
                    final_received=as_float(first_event_value(record, "final_received")),
                    finalize_events=parse_finalize_events(record),
                    order_time=telemetry_order_time(record),
                )
            )
            diagnostics["rows_loaded"] += 1
    diagnostics["skip_reasons"] = dict(diagnostics["skip_reasons"])
    return rows, diagnostics


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_samples_dbs() -> list[Path]:
    repo_root = repo_root_from_script()
    data_dir = repo_root / "stt-benchmark" / "stt_benchmark_data"
    return [data_dir / "test_results.db", data_dir / "results.db"]


def load_subset_ids(path: Path | None) -> tuple[set[str] | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"path": str(path) if path else None, "enabled": path is not None, "count": None}
    if path is None:
        return None, diagnostics
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("sample_ids")
    else:
        values = None
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError(f"{path} must contain a string list or a 'sample_ids' string list")
    ids = set(values)
    diagnostics["count"] = len(ids)
    return ids, diagnostics


def sort_telemetry_for_temporal_join(rows: list[TelemetryRow]) -> list[TelemetryRow]:
    if all(row.benchmark_batch_index is not None for row in rows):
        return sorted(rows, key=lambda row: (row.benchmark_batch_index, row.source_line))
    if all(row.order_time is not None for row in rows):
        return sorted(rows, key=lambda row: (row.order_time, row.source_line))
    return sorted(rows, key=lambda row: row.source_line)


def join_sessions(
    probe_sessions: list[ProbeSession],
    telemetry_rows: list[TelemetryRow],
    subset_ids: set[str] | None,
) -> tuple[list[tuple[ProbeSession, TelemetryRow]], dict[str, Any], Counter[str]]:
    skipped: Counter[str] = Counter()
    joined: list[tuple[ProbeSession, TelemetryRow]] = []
    direct_by_session_id = {row.session_id: row for row in telemetry_rows if row.session_id}
    used_telemetry_lines: set[int] = set()
    method = "direct_session_id" if direct_by_session_id else "temporal_order_fallback"

    if direct_by_session_id:
        for session in probe_sessions:
            row = direct_by_session_id.get(session.session_id)
            if row is None:
                continue
            joined.append((session, row))
            used_telemetry_lines.add(row.source_line)

    remaining_sessions = [session for session in probe_sessions if session.session_id not in {item[0].session_id for item in joined}]
    remaining_rows = [row for row in telemetry_rows if row.source_line not in used_telemetry_lines]
    fallback_count = 0
    if remaining_sessions and remaining_rows:
        if direct_by_session_id:
            method = "direct_session_id_plus_temporal_order_fallback"
        ordered_sessions = sorted(remaining_sessions, key=lambda item: item.first_sort_time)
        ordered_rows = sort_telemetry_for_temporal_join(remaining_rows)
        pair_count = min(len(ordered_sessions), len(ordered_rows))
        for session, row in zip(ordered_sessions[:pair_count], ordered_rows[:pair_count]):
            joined.append((session, row))
            fallback_count += 1

    joined_by_session = {session.session_id for session, _row in joined}
    for session in probe_sessions:
        if session.session_id not in joined_by_session:
            skipped["no_telemetry_join"] += 1

    filtered: list[tuple[ProbeSession, TelemetryRow]] = []
    for session, row in joined:
        if subset_ids is not None:
            if row.sample_id is None:
                skipped["no_sample_id_for_subset"] += 1
                continue
            if row.sample_id not in subset_ids:
                skipped["not_in_subset"] += 1
                continue
        if row.vad_stop is None:
            skipped["no_silero_endpoint"] += 1
            continue
        filtered.append((session, row))

    telemetry_order_basis = "source_line"
    if telemetry_rows and all(row.benchmark_batch_index is not None for row in telemetry_rows):
        telemetry_order_basis = "benchmark_batch_index"
    elif telemetry_rows and all(row.order_time is not None for row in telemetry_rows):
        telemetry_order_basis = "telemetry_time"
    diagnostics = {
        "method": method,
        "telemetry_order_basis": telemetry_order_basis,
        "probe_session_order_basis": "first_chunk wall_time_start/done, falling back to monotonic_start/done",
        "probe_sessions_seen": len(probe_sessions),
        "telemetry_rows_seen": len(telemetry_rows),
        "joined_before_filters": len(joined),
        "analyzable_after_filters": len(filtered),
        "direct_session_id_rows": len(direct_by_session_id),
        "temporal_fallback_pairs": fallback_count,
        "limitation": None,
    }
    if fallback_count:
        diagnostics["limitation"] = (
            "client finalize telemetry lacks session_id for at least some rows; "
            "session_id->benchmark_batch_index is inferred by ordering probe sessions by first chunk "
            "and telemetry rows by benchmark_batch_index/source order. This assumes the benchmark "
            "uses one websocket session per sample and processes samples sequentially."
        )
    return filtered, diagnostics, skipped


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def percentile_ms(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), percentile)), 3)


def chunk_trigger_time(chunk: ProbeChunk) -> float | None:
    for value in (chunk.wall_time_done, chunk.wall_time_start, chunk.monotonic_done, chunk.monotonic_start):
        if value is not None:
            return value
    return None


def chunk_audio_elapsed(chunk: ProbeChunk, session: ProbeSession) -> float | None:
    if chunk.audio_elapsed_s is not None:
        return chunk.audio_elapsed_s
    session_start = session.first_sort_time
    trigger_time = chunk_trigger_time(chunk)
    if trigger_time is None:
        return None
    return max(0.0, trigger_time - session_start)


def trigger_blank_run(session: ProbeSession, k: int, warmup_s: float) -> float | None:
    streak = 0
    for chunk in session.chunks:
        elapsed = chunk_audio_elapsed(chunk, session)
        if elapsed is None or elapsed < warmup_s:
            streak = 0
            continue
        if chunk.chunk_blank:
            streak += 1
        else:
            streak = 0
        if streak >= k:
            return chunk_trigger_time(chunk)
    return None


def trigger_hyp_unchanged(session: ProbeSession, k: int, warmup_s: float) -> float | None:
    streak = 0
    prev_y_len: int | None = None
    for chunk in session.chunks:
        elapsed = chunk_audio_elapsed(chunk, session)
        if elapsed is None or elapsed < warmup_s:
            streak = 0
            prev_y_len = chunk.y_len
            continue
        if prev_y_len is not None and chunk.y_len == prev_y_len:
            streak += 1
        else:
            streak = 0
        prev_y_len = chunk.y_len
        if streak >= k:
            return chunk_trigger_time(chunk)
    return None


def trigger_confidence(session: ProbeSession, tau: float, window_ms: int, warmup_s: float) -> float | None:
    window_s = window_ms / 1000.0
    rolling: deque[tuple[float, tuple[float, ...]]] = deque()
    for chunk in session.chunks:
        elapsed = chunk_audio_elapsed(chunk, session)
        if elapsed is None or elapsed < warmup_s:
            rolling.clear()
            continue
        if chunk.confidence_values:
            rolling.append((elapsed, chunk.confidence_values))
        while rolling and rolling[0][0] < elapsed - window_s:
            rolling.popleft()
        values: list[float] = []
        for _when, chunk_values in rolling:
            values.extend(chunk_values)
        if values and float(np.mean(np.asarray(values, dtype=np.float64))) >= tau:
            return chunk_trigger_time(chunk)
    return None


def sweep_triggers(session: ProbeSession, warmup_s: float) -> dict[str, dict[str, float | None]]:
    return {
        "blank_run": {f"K={k}": trigger_blank_run(session, k, warmup_s) for k in BLANK_K_VALUES},
        "hyp_unchanged": {f"K={k}": trigger_hyp_unchanged(session, k, warmup_s) for k in HYP_K_VALUES},
        "normalized_confidence": {
            f"tau={tau:.2f},T={window_ms}ms": trigger_confidence(session, tau, window_ms, warmup_s)
            for tau in CONFIDENCE_TAUS
            for window_ms in CONFIDENCE_WINDOWS_MS
        },
    }


def audio_path_for(audio_dir: Path, session: ProbeSession) -> Path:
    return audio_dir / f"{session.run_tag}_{session.session_id}_audio.bin"


def estimate_acoustic_stop(
    audio_path: Path,
    audio_start_wallclock: float | None,
) -> tuple[float | None, str]:
    if audio_start_wallclock is None:
        return None, "no_audio_time_anchor"
    if not audio_path.exists():
        return None, "missing_audio_bin"
    raw = audio_path.read_bytes()
    if len(raw) < 2:
        return None, "empty_audio_bin"
    if len(raw) % 2:
        raw = raw[:-1]
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if audio.size == 0:
        return None, "empty_audio_bin"
    frame_samples = int(round(SAMPLE_RATE_HZ * ACOUSTIC_FRAME_MS / 1000.0))
    frame_count = audio.size // frame_samples
    if frame_count <= 0:
        return None, "audio_shorter_than_one_frame"
    trimmed = audio[: frame_count * frame_samples].reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(trimmed * trimmed, axis=1, dtype=np.float64))
    max_rms = float(np.max(rms)) if rms.size else 0.0
    if max_rms <= 0.0:
        return None, "silent_audio_bin"
    threshold = max(max_rms * 0.01, max_rms * 0.001)
    above = np.flatnonzero(rms >= threshold)
    if above.size == 0:
        return None, "no_frame_above_threshold"
    last_frame = int(above[-1])
    stop_offset_s = (last_frame + 1) * (ACOUSTIC_FRAME_MS / 1000.0)
    return audio_start_wallclock + stop_offset_s, "ok"


def build_joined_sessions(
    pairs: list[tuple[ProbeSession, TelemetryRow]],
    audio_dir: Path,
) -> tuple[list[JoinedSession], Counter[str]]:
    acoustic_status_counts: Counter[str] = Counter()
    joined: list[JoinedSession] = []
    for session, row in pairs:
        path = audio_path_for(audio_dir, session)
        acoustic_stop, status = estimate_acoustic_stop(path, session.audio_start_wallclock)
        acoustic_status_counts[status] += 1
        joined.append(
            JoinedSession(
                probe=session,
                telemetry=row,
                estimated_acoustic_stop=acoustic_stop,
                acoustic_status=status,
                audio_path=str(path) if path.exists() else None,
            )
        )
    return joined, acoustic_status_counts


def summarize_operating_point(
    *,
    family: str,
    operating_point: str,
    trigger_by_session: dict[str, float | None],
    sessions: list[JoinedSession],
    early_margin_s: float,
) -> dict[str, Any]:
    lat_silero_ms: list[float] = []
    lat_acoustic_ms: list[float] = []
    false_early = 0
    never = 0
    acoustic_sessions = 0
    acoustic_never = 0

    for session in sessions:
        trigger = trigger_by_session.get(session.probe.session_id)
        if trigger is None:
            never += 1
        else:
            lat_silero_ms.append((trigger - float(session.telemetry.vad_stop)) * 1000.0)
            if trigger < float(session.telemetry.vad_stop) - early_margin_s:
                false_early += 1

        if session.estimated_acoustic_stop is not None:
            acoustic_sessions += 1
            if trigger is None:
                acoustic_never += 1
            else:
                lat_acoustic_ms.append((trigger - session.estimated_acoustic_stop) * 1000.0)

    n_sessions = len(sessions)
    return {
        "signal_family": family,
        "operating_point": operating_point,
        "n_sessions": n_sessions,
        "triggered_sessions": n_sessions - never,
        "never_fired_count": never,
        "never_fired_rate": round(safe_rate(never, n_sessions), 6),
        "false_early_fire_count": false_early,
        "false_early_fire_rate": round(safe_rate(false_early, n_sessions), 6),
        "detection_latency_p50_ms": percentile_ms(lat_silero_ms, 50),
        "detection_latency_p95_ms": percentile_ms(lat_silero_ms, 95),
        "detection_latency_vs_acoustic_p50_ms": percentile_ms(lat_acoustic_ms, 50),
        "detection_latency_vs_acoustic_p95_ms": percentile_ms(lat_acoustic_ms, 95),
        "acoustic_endpoint_sessions": acoustic_sessions,
        "acoustic_never_fired_count": acoustic_never,
        "acoustic_never_fired_rate": round(safe_rate(acoustic_never, acoustic_sessions), 6),
    }


def roc_summary(
    sessions: list[JoinedSession],
    warmup_s: float,
    early_margin_s: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, float | None]]]]:
    trigger_grid: dict[str, dict[str, dict[str, float | None]]] = {}
    for session in sessions:
        trigger_grid[session.probe.session_id] = sweep_triggers(session.probe, warmup_s)

    family_to_ops: dict[str, list[str]] = {
        "blank_run": [f"K={k}" for k in BLANK_K_VALUES],
        "hyp_unchanged": [f"K={k}" for k in HYP_K_VALUES],
        "normalized_confidence": [
            f"tau={tau:.2f},T={window_ms}ms" for tau in CONFIDENCE_TAUS for window_ms in CONFIDENCE_WINDOWS_MS
        ],
    }
    summary: dict[str, list[dict[str, Any]]] = {}
    for family, ops in family_to_ops.items():
        summary[family] = []
        for op in ops:
            trigger_by_session = {
                session.probe.session_id: trigger_grid[session.probe.session_id][family].get(op)
                for session in sessions
            }
            summary[family].append(
                summarize_operating_point(
                    family=family,
                    operating_point=op,
                    trigger_by_session=trigger_by_session,
                    sessions=sessions,
                    early_margin_s=early_margin_s,
                )
            )
    return summary, trigger_grid


def build_viability_table(roc: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_ops = [item for rows in roc.values() for item in rows]
    table: list[dict[str, Any]] = []
    for latency_target in GO_LATENCY_TARGETS_MS:
        for false_target in GO_FALSE_RATE_TARGETS:
            matches = []
            for op in all_ops:
                p95 = op.get("detection_latency_p95_ms")
                false_rate = op.get("false_early_fire_rate")
                if p95 is None or false_rate is None:
                    continue
                if float(p95) <= latency_target and float(false_rate) <= false_target:
                    matches.append(
                        {
                            "signal_family": op["signal_family"],
                            "operating_point": op["operating_point"],
                            "detection_latency_p95_ms": p95,
                            "false_early_fire_rate": false_rate,
                            "never_fired_rate": op["never_fired_rate"],
                        }
                    )
            table.append(
                {
                    "latency_p95_target_ms": latency_target,
                    "false_early_fire_rate_target": false_target,
                    "count": len(matches),
                    "operating_points": matches,
                }
            )
    return table


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def fmt_ms(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.1f}"


def fmt_rate(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value) * 100:.2f}%"


def render_roc_table(family: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"{family}:",
        "  signal_op              det_lat_p50  det_lat_p95  acoustic_p95  false_early  never",
    ]
    for row in rows:
        lines.append(
            "  "
            f"{row['operating_point']:<22}"
            f"{fmt_ms(row['detection_latency_p50_ms']):>11}"
            f"{fmt_ms(row['detection_latency_p95_ms']):>13}"
            f"{fmt_ms(row['detection_latency_vs_acoustic_p95_ms']):>14}"
            f"{fmt_rate(row['false_early_fire_rate']):>13}"
            f"{fmt_rate(row['never_fired_rate']):>8}"
        )
    return "\n".join(lines)


def render_viability_table(table: list[dict[str, Any]]) -> str:
    lines = [
        "GO/NO-GO #1 viability table:",
        "  p95<=ms  false<=  count  operating_points",
    ]
    for row in table:
        op_names = ", ".join(
            f"{op['signal_family']}:{op['operating_point']}" for op in row["operating_points"]
        )
        if not op_names:
            op_names = "-"
        lines.append(
            f"  {row['latency_p95_target_ms']:>7}"
            f"  {fmt_rate(row['false_early_fire_rate_target']):>7}"
            f"  {row['count']:>5}  {op_names}"
        )
    return "\n".join(lines)


def render_human_report(summary: dict[str, Any], roc: dict[str, list[dict[str, Any]]], table: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append("Join:")
    join = summary["join"]
    lines.append(f"  method: {join['method']}")
    lines.append(f"  telemetry_order_basis: {join['telemetry_order_basis']}")
    if join.get("limitation"):
        lines.append(f"  limitation: {join['limitation']}")
    lines.append("")
    lines.append("Endpoint extraction:")
    endpoint = summary["endpoint_extraction"]
    lines.append(f"  silero_vad_stop: client telemetry vad_stop; skipped no_silero_endpoint sessions")
    lines.append(
        "  estimated_acoustic_stop: 20 ms int16 RMS frames; threshold=max(1% session max RMS, "
        "60 dB below max); uncertainty_band_ms=+/-40"
    )
    lines.append(f"  acoustic_status_counts: {endpoint['acoustic_status_counts']}")
    lines.append("")
    lines.append("ROC:")
    for family, rows in roc.items():
        lines.append(render_roc_table(family, rows))
        lines.append("")
    lines.append(render_viability_table(table))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.early_margin_ms < 0:
        raise ValueError("--early-margin-ms must be >= 0")
    if args.warmup_ms < 0:
        raise ValueError("--warmup-ms must be >= 0")

    subset_ids, subset_diag = load_subset_ids(args.subset)
    samples_dbs = [args.samples_db] if args.samples_db else default_samples_dbs()
    sample_map, samples_db_diag = load_sample_id_map(samples_dbs)
    telemetry_rows, telemetry_diag = load_telemetry_rows(args.finalize_jsonl, sample_map)
    probe_sessions, probe_diag = load_probe_sessions(args.probe_jsonl)
    joined_pairs, join_diag, session_skip_reasons = join_sessions(probe_sessions, telemetry_rows, subset_ids)
    joined_sessions, acoustic_status_counts = build_joined_sessions(joined_pairs, args.audio_dir)
    roc, trigger_grid = roc_summary(
        joined_sessions,
        warmup_s=args.warmup_ms / 1000.0,
        early_margin_s=args.early_margin_ms / 1000.0,
    )
    viability_table = build_viability_table(roc)

    n_skipped = sum(session_skip_reasons.values())
    summary = {
        "n_sessions_analyzed": len(joined_sessions),
        "n_sessions_skipped": {"total": n_skipped, "reasons": dict(session_skip_reasons)},
        "probe_rows_available": probe_diag["rows_read"],
        "probe_rows_loaded": probe_diag["rows_loaded"],
        "finalize_rows_loaded": telemetry_diag["rows_loaded"],
        "config": {
            "early_margin_ms": args.early_margin_ms,
            "warmup_ms": args.warmup_ms,
            "latency_units": "ms",
        },
        "inputs": {
            "probe_jsonl": str(args.probe_jsonl),
            "finalize_jsonl": str(args.finalize_jsonl),
            "audio_dir": str(args.audio_dir),
            "subset": str(args.subset) if args.subset else None,
        },
        "subset": subset_diag,
        "samples_db": samples_db_diag,
        "probe_diagnostics": probe_diag,
        "telemetry_diagnostics": telemetry_diag,
        "join": join_diag,
        "endpoint_extraction": {
            "silero_source": "client finalize telemetry vad_stop",
            "estimated_acoustic_stop": {
                "audio_format": "int16 little-endian PCM, 16 kHz",
                "frame_ms": ACOUSTIC_FRAME_MS,
                "threshold": "last 20 ms RMS frame >= max(1% of session max RMS, 60 dB below max)",
                "uncertainty_band_ms": ACOUSTIC_UNCERTAINTY_MS,
                "clock_anchor": "first probe wall_time_done - first real_audio_cursor_seconds",
            },
            "acoustic_status_counts": dict(acoustic_status_counts),
        },
        "per_signal_family_roc_summary": roc,
        "viable_operating_points_table": viability_table,
    }

    print(compact_json(summary))
    if not args.silent:
        print(render_human_report(summary, roc, viability_table))

    if args.output_json is not None:
        details = {
            "summary": summary,
            "per_session": [
                {
                    "run_tag": session.probe.run_tag,
                    "session_id": session.probe.session_id,
                    "benchmark_batch_index": session.telemetry.benchmark_batch_index,
                    "sample_id": session.telemetry.sample_id,
                    "silero_vad_stop": session.telemetry.vad_stop,
                    "estimated_acoustic_stop": session.estimated_acoustic_stop,
                    "acoustic_status": session.acoustic_status,
                    "audio_path": session.audio_path,
                    "triggers": trigger_grid.get(session.probe.session_id, {}),
                }
                for session in joined_sessions
            ],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
