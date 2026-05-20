#!/usr/bin/env python3
"""Offline EOU signal diagnostic over Step-2 probe telemetry.

This script reads existing JSONL/audio sidecars only. It does not start a
server, use a GPU, or collect new data.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_RATE_HZ = 16_000
MODEL_FRAME_MS = 10.0
RUN_K_VALUES = (10, 20, 40, 60, 80, 100, 120, 160, 200)
ROC_K_VALUES = (10, 20, 40, 60, 80, 100, 120, 160, 200, 240, 320)
FALSE_EARLY_MARGINS_MS = (100, 200)
TAIL_OFFSETS_MS = (250, 500, 750, 1000, 1500, 2000)
TAIL_K_VALUES = (20, 50, 100, 200)
PUNCT_RE = re.compile(r"[.,?!;:()\[\]{}\"'`/\\|<>@#$%^&*+=~-]")
ALNUM_RE = re.compile(r"[A-Za-z0-9]")


@dataclass(frozen=True)
class ProbeChunk:
    source_line: int
    run_tag: str
    session_id: str
    chunk_index: int
    chunk_model_frame_start: int
    shift_frames: int
    real_audio_cursor_seconds: float | None
    monotonic_done: float | None
    wall_time_done: float | None
    wall_time_start: float | None
    frame_alignment: list[dict[str, Any]]
    new_tokens: list[dict[str, Any]]


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
        for value in (first.wall_time_start, first.wall_time_done, first.monotonic_done):
            if value is not None:
                return value
        return float(first.chunk_index)

    @property
    def audio_start_wallclock(self) -> float | None:
        first = self.first_chunk
        if first.wall_time_done is None or first.real_audio_cursor_seconds is None:
            return None
        return first.wall_time_done - first.real_audio_cursor_seconds

    @property
    def last_audio_cursor_s(self) -> float | None:
        values = [chunk.real_audio_cursor_seconds for chunk in self.chunks if chunk.real_audio_cursor_seconds is not None]
        return max(values) if values else None


@dataclass(frozen=True)
class TelemetryRow:
    source_line: int
    run_tag: str | None
    benchmark_batch_index: int | None
    session_id: str | None
    vad_stop: float | None
    order_time: float | None


@dataclass(frozen=True)
class ProbeFrame:
    session_id: str
    chunk_index: int
    frame_offset: int
    model_frame_index: int
    audio_time_s: float
    available_wall_time: float | None
    is_blank: bool
    has_non_blank: bool
    labels: tuple[int, ...]
    tokens: tuple[str, ...]


@dataclass
class AnalyzedSession:
    run_tag: str
    session_id: str
    benchmark_batch_index: int | None
    vad_stop_wallclock: float
    vad_stop_audio_s: float
    frames: list[ProbeFrame]
    frame_audio_times_s: list[float]
    audio_path: str | None
    last_audio_cursor_s: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-jsonl", type=Path, required=True)
    parser.add_argument("--finalize-jsonl", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
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


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def round_float(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def median(values: list[float]) -> float | None:
    return percentile(values, 50)


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
    for key in ("start_time", "started_at", "first_audio_wall_time", "vad_stop", "final_received"):
        value = as_float(first_event_value(record, key))
        if value is not None:
            return value
    return None


def parse_probe_chunk(record: Any, source_line: int) -> tuple[ProbeChunk | None, str | None]:
    if not isinstance(record, dict):
        return None, "not_object"
    run_tag = record.get("run_tag")
    session_id = record.get("session_id")
    chunk_index = as_int(record.get("chunk_index"))
    chunk_model_frame_start = as_int(record.get("chunk_model_frame_start"))
    shift_frames = as_int(record.get("shift_frames"))
    frame_alignment = record.get("frame_alignment")
    new_tokens = record.get("new_tokens")
    if (
        not isinstance(run_tag, str)
        or not isinstance(session_id, str)
        or chunk_index is None
        or chunk_model_frame_start is None
        or shift_frames is None
        or not isinstance(frame_alignment, list)
        or not isinstance(new_tokens, list)
    ):
        return None, "missing_or_bad_required_field"
    return (
        ProbeChunk(
            source_line=source_line,
            run_tag=run_tag,
            session_id=session_id,
            chunk_index=chunk_index,
            chunk_model_frame_start=chunk_model_frame_start,
            shift_frames=shift_frames,
            real_audio_cursor_seconds=as_float(record.get("real_audio_cursor_seconds")),
            monotonic_done=as_float(record.get("monotonic_done")),
            wall_time_done=as_float(record.get("wall_time_done")),
            wall_time_start=as_float(record.get("wall_time_start")),
            frame_alignment=[frame for frame in frame_alignment if isinstance(frame, dict)],
            new_tokens=[token for token in new_tokens if isinstance(token, dict)],
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
        "frame_alignment_lengths": Counter(),
        "shift_frames": Counter(),
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
            diagnostics["frame_alignment_lengths"][len(chunk.frame_alignment)] += 1
            diagnostics["shift_frames"][chunk.shift_frames] += 1

    sessions: list[ProbeSession] = []
    for (run_tag, session_id), chunks in grouped.items():
        chunks.sort(key=lambda item: (item.chunk_index, item.source_line))
        sessions.append(ProbeSession(run_tag=run_tag, session_id=session_id, chunks=chunks))
    sessions.sort(key=lambda item: item.first_sort_time)

    diagnostics["sessions_loaded"] = len(sessions)
    diagnostics["skip_reasons"] = dict(diagnostics["skip_reasons"])
    diagnostics["frame_alignment_lengths"] = dict(diagnostics["frame_alignment_lengths"])
    diagnostics["shift_frames"] = dict(diagnostics["shift_frames"])
    return sessions, diagnostics


def load_telemetry_rows(path: Path) -> tuple[list[TelemetryRow], dict[str, Any]]:
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
            session_id = telemetry_session_id(record)
            if batch_index is not None:
                diagnostics["rows_with_batch_index"] += 1
            if session_id is not None:
                diagnostics["rows_with_session_id"] += 1
            run_tag = record.get("run_tag") if isinstance(record.get("run_tag"), str) else None
            rows.append(
                TelemetryRow(
                    source_line=source_line,
                    run_tag=run_tag,
                    benchmark_batch_index=batch_index,
                    session_id=session_id,
                    vad_stop=as_float(first_event_value(record, "vad_stop")),
                    order_time=telemetry_order_time(record),
                )
            )
            diagnostics["rows_loaded"] += 1
    diagnostics["skip_reasons"] = dict(diagnostics["skip_reasons"])
    return rows, diagnostics


def sort_telemetry_for_temporal_join(rows: list[TelemetryRow]) -> list[TelemetryRow]:
    if rows and all(row.benchmark_batch_index is not None for row in rows):
        return sorted(rows, key=lambda row: (row.benchmark_batch_index, row.source_line))
    if rows and all(row.order_time is not None for row in rows):
        return sorted(rows, key=lambda row: (row.order_time, row.source_line))
    return sorted(rows, key=lambda row: row.source_line)


def join_sessions(
    probe_sessions: list[ProbeSession],
    telemetry_rows: list[TelemetryRow],
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

    joined_session_ids = {session.session_id for session, _row in joined}
    remaining_sessions = [session for session in probe_sessions if session.session_id not in joined_session_ids]
    remaining_rows = [row for row in telemetry_rows if row.source_line not in used_telemetry_lines]
    fallback_count = 0
    if remaining_sessions and remaining_rows:
        if direct_by_session_id:
            method = "direct_session_id_plus_temporal_order_fallback"
        ordered_sessions = sorted(remaining_sessions, key=lambda item: item.first_sort_time)
        ordered_rows = sort_telemetry_for_temporal_join(remaining_rows)
        for session, row in zip(ordered_sessions, ordered_rows):
            joined.append((session, row))
            fallback_count += 1

    joined_by_session = {session.session_id for session, _row in joined}
    for session in probe_sessions:
        if session.session_id not in joined_by_session:
            skipped["no_telemetry_join"] += 1

    filtered: list[tuple[ProbeSession, TelemetryRow]] = []
    for session, row in joined:
        if row.vad_stop is None:
            skipped["no_vad_stop"] += 1
            continue
        if session.audio_start_wallclock is None:
            skipped["no_audio_wallclock_anchor"] += 1
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
        "probe_session_order_basis": "first probe wall_time_start/done, falling back to monotonic_done",
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
            "finalize telemetry lacks session_id; sessions are joined by probe order "
            "and benchmark_batch_index/source order, matching oracle_roc.py's fallback."
        )
    return filtered, diagnostics, skipped


def frame_labels(frame: dict[str, Any]) -> tuple[int, ...]:
    labels = frame.get("labels")
    if not isinstance(labels, list):
        return ()
    parsed = [as_int(label) for label in labels]
    return tuple(label for label in parsed if label is not None)


def token_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return str(value)


def build_frames(session: ProbeSession, vad_stop_wallclock: float) -> AnalyzedSession:
    audio_start_wallclock = session.audio_start_wallclock
    if audio_start_wallclock is None:
        raise ValueError(f"session {session.session_id} has no audio wallclock anchor")
    vad_stop_audio_s = vad_stop_wallclock - audio_start_wallclock
    frames: list[ProbeFrame] = []

    for chunk in session.chunks:
        token_strings_by_frame: dict[int, list[str]] = defaultdict(list)
        for token in chunk.new_tokens:
            model_frame_index = as_int(token.get("model_frame_index"))
            text = token_text(token.get("token"))
            if model_frame_index is not None and text is not None:
                token_strings_by_frame[model_frame_index].append(text)

        chunk_span_s = chunk.shift_frames * (MODEL_FRAME_MS / 1000.0)
        chunk_start_audio_s: float | None = None
        if chunk.real_audio_cursor_seconds is not None:
            chunk_start_audio_s = chunk.real_audio_cursor_seconds - chunk_span_s

        for frame in chunk.frame_alignment:
            frame_offset = as_int(frame.get("frame_offset"))
            model_frame_index = as_int(frame.get("model_frame_index"))
            if frame_offset is None or model_frame_index is None:
                continue
            if chunk_start_audio_s is None:
                audio_time_s = model_frame_index * (MODEL_FRAME_MS / 1000.0)
            else:
                audio_time_s = chunk_start_audio_s + frame_offset * (MODEL_FRAME_MS / 1000.0)
            frames.append(
                ProbeFrame(
                    session_id=session.session_id,
                    chunk_index=chunk.chunk_index,
                    frame_offset=frame_offset,
                    model_frame_index=model_frame_index,
                    audio_time_s=audio_time_s,
                    available_wall_time=chunk.wall_time_done,
                    is_blank=frame.get("all_blank") is True,
                    has_non_blank=frame.get("has_non_blank") is True,
                    labels=frame_labels(frame),
                    tokens=tuple(token_strings_by_frame.get(model_frame_index, [])),
                )
            )

    frames.sort(key=lambda item: (item.audio_time_s, item.model_frame_index, item.chunk_index, item.frame_offset))
    audio_path = Path(f"{session.run_tag}_{session.session_id}_audio.bin")
    return AnalyzedSession(
        run_tag=session.run_tag,
        session_id=session.session_id,
        benchmark_batch_index=None,
        vad_stop_wallclock=vad_stop_wallclock,
        vad_stop_audio_s=vad_stop_audio_s,
        frames=frames,
        frame_audio_times_s=[frame.audio_time_s for frame in frames],
        audio_path=str(audio_path),
        last_audio_cursor_s=session.last_audio_cursor_s,
    )


def build_analyzed_sessions(
    joined_pairs: list[tuple[ProbeSession, TelemetryRow]],
    audio_dir: Path,
) -> tuple[list[AnalyzedSession], dict[str, Any]]:
    analyzed: list[AnalyzedSession] = []
    audio_sidecars = Counter()
    frame_counts = []
    for session, row in joined_pairs:
        assert row.vad_stop is not None
        item = build_frames(session, row.vad_stop)
        item.benchmark_batch_index = row.benchmark_batch_index
        audio_path = audio_dir / f"{session.run_tag}_{session.session_id}_audio.bin"
        if audio_path.exists():
            audio_sidecars["found"] += 1
            item.audio_path = str(audio_path)
        else:
            audio_sidecars["missing"] += 1
            item.audio_path = None
        frame_counts.append(len(item.frames))
        analyzed.append(item)
    diagnostics = {
        "audio_sidecars": dict(audio_sidecars),
        "observed_frames_total": sum(frame_counts),
        "observed_frames_per_session_min": min(frame_counts) if frame_counts else 0,
        "observed_frames_per_session_max": max(frame_counts) if frame_counts else 0,
    }
    return analyzed, diagnostics


def estimate_observed_signal_frame_ms(probe_sessions: list[ProbeSession]) -> dict[str, Any]:
    frame_alignment_lengths: list[int] = []
    cursor_deltas_ms: list[float] = []
    shift_frame_values: list[int] = []
    for session in probe_sessions:
        previous_cursor = None
        for chunk in session.chunks:
            frame_alignment_lengths.append(len(chunk.frame_alignment))
            shift_frame_values.append(chunk.shift_frames)
            if previous_cursor is not None and chunk.real_audio_cursor_seconds is not None:
                delta = chunk.real_audio_cursor_seconds - previous_cursor
                if delta > 0:
                    cursor_deltas_ms.append(delta * 1000.0)
            if chunk.real_audio_cursor_seconds is not None:
                previous_cursor = chunk.real_audio_cursor_seconds
    median_alignment_len = median([float(value) for value in frame_alignment_lengths if value > 0])
    median_shift_frames = median([float(value) for value in shift_frame_values if value > 0])
    median_chunk_step_ms = median(cursor_deltas_ms)
    observed_ms = None
    if median_alignment_len and median_chunk_step_ms:
        observed_ms = median_chunk_step_ms / median_alignment_len
    expected_entries = median_shift_frames
    warning = None
    if median_alignment_len is not None and expected_entries is not None and median_alignment_len != expected_entries:
        warning = (
            "probe rows are sparse: median frame_alignment length differs from shift_frames; "
            "run lengths count observed decoder alignment frames, not filled 10 ms gaps"
        )
    return {
        "model_frame_ms_from_task": MODEL_FRAME_MS,
        "median_shift_frames_per_chunk": round_float(expected_entries),
        "median_frame_alignment_entries_per_chunk": round_float(median_alignment_len),
        "median_audio_cursor_step_ms": round_float(median_chunk_step_ms),
        "estimated_observed_signal_frame_ms": round_float(observed_ms),
        "warning": warning,
    }


def histogram(counter: Counter[int]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def blank_run_distribution(sessions: list[AnalyzedSession]) -> dict[str, Any]:
    in_speech_lengths: list[int] = []
    post_silence_lengths: list[int] = []
    in_speech_hist: Counter[int] = Counter()
    post_silence_hist: Counter[int] = Counter()

    for session in sessions:
        run_len = 0
        run_region: str | None = None
        for frame in session.frames:
            if frame.is_blank:
                if run_len == 0:
                    run_region = "in_speech" if frame.audio_time_s < session.vad_stop_audio_s else "post_silence"
                run_len += 1
                continue
            if run_len:
                if run_region == "in_speech":
                    in_speech_lengths.append(run_len)
                    in_speech_hist[run_len] += 1
                else:
                    post_silence_lengths.append(run_len)
                    post_silence_hist[run_len] += 1
            run_len = 0
            run_region = None
        if run_len:
            if run_region == "in_speech":
                in_speech_lengths.append(run_len)
                in_speech_hist[run_len] += 1
            else:
                post_silence_lengths.append(run_len)
                post_silence_hist[run_len] += 1

    longest = max(in_speech_lengths) if in_speech_lengths else 0
    longest_post = max(post_silence_lengths) if post_silence_lengths else 0
    return {
        "in_speech_histogram": histogram(in_speech_hist),
        "post_silence_histogram": histogram(post_silence_hist),
        "in_speech_blank_run_count": len(in_speech_lengths),
        "post_silence_blank_run_count": len(post_silence_lengths),
        "longest_in_speech_blank_run_frames": longest,
        "longest_post_silence_blank_run_frames": longest_post,
        "zero_in_speech_false_fire_min_k_frames": longest + 1 if in_speech_lengths else 1,
        "in_speech_blank_run_median_frames": round_float(median([float(value) for value in in_speech_lengths])),
        "in_speech_blank_run_p95_frames": round_float(percentile([float(value) for value in in_speech_lengths], 95)),
        "in_speech_blank_run_ge_k_fraction": {
            str(k): round_float(safe_div(sum(1 for value in in_speech_lengths if value >= k), len(in_speech_lengths)), 6)
            for k in RUN_K_VALUES
        },
    }


def classify_token(token: str) -> str:
    if PUNCT_RE.search(token):
        return "punctuation"
    if token.startswith("\u2581") or token.startswith("_") or token.strip() == "":
        return "space-marker"
    if ALNUM_RE.search(token):
        return "alphanumeric-word"
    return "other"


def silence_trailer_emissions(sessions: list[AnalyzedSession]) -> dict[str, Any]:
    class_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    post_non_blank_frames = 0
    post_token_total = 0
    sessions_with_emission = 0
    first_audio_offsets_ms: list[float] = []
    first_wall_latencies_ms: list[float] = []
    per_session_first_offsets: dict[str, float] = {}

    for session in sessions:
        first_audio_offset: float | None = None
        first_wall_latency: float | None = None
        for frame in session.frames:
            if frame.audio_time_s < session.vad_stop_audio_s:
                continue
            if not frame.has_non_blank:
                continue
            post_non_blank_frames += 1
            for token in frame.tokens:
                class_counts[classify_token(token)] += 1
                token_counts[token] += 1
                post_token_total += 1
                audio_offset_ms = (frame.audio_time_s - session.vad_stop_audio_s) * 1000.0
                if first_audio_offset is None or audio_offset_ms < first_audio_offset:
                    first_audio_offset = audio_offset_ms
                    if frame.available_wall_time is not None:
                        first_wall_latency = (frame.available_wall_time - session.vad_stop_wallclock) * 1000.0
        if first_audio_offset is not None:
            sessions_with_emission += 1
            first_audio_offsets_ms.append(first_audio_offset)
            if first_wall_latency is not None:
                first_wall_latencies_ms.append(first_wall_latency)
            per_session_first_offsets[session.session_id] = round_float(first_audio_offset)

    classes = {}
    for name in ("punctuation", "space-marker", "alphanumeric-word", "other"):
        count = class_counts[name]
        classes[name] = {
            "count": count,
            "percentage": round_float(safe_div(count * 100.0, post_token_total), 3),
        }
    return {
        "n_sessions": len(sessions),
        "post_silence_non_blank_frame_count": post_non_blank_frames,
        "post_silence_token_emission_count": post_token_total,
        "classes": classes,
        "top_20_token_strings": [{"token": token, "count": count} for token, count in token_counts.most_common(20)],
        "sessions_with_post_silence_emission": sessions_with_emission,
        "session_fraction_with_post_silence_emission": round_float(safe_div(sessions_with_emission, len(sessions)), 6),
        "first_post_silence_token_audio_offset_ms": {
            "median": round_float(median(first_audio_offsets_ms)),
            "p95": round_float(percentile(first_audio_offsets_ms, 95)),
            "min": round_float(min(first_audio_offsets_ms) if first_audio_offsets_ms else None),
            "max": round_float(max(first_audio_offsets_ms) if first_audio_offsets_ms else None),
        },
        "first_post_silence_token_wall_latency_ms": {
            "median": round_float(median(first_wall_latencies_ms)),
            "p95": round_float(percentile(first_wall_latencies_ms, 95)),
        },
        "per_session_first_audio_offset_ms": per_session_first_offsets,
    }


def first_blank_run_trigger(session: AnalyzedSession, k_frames: int) -> ProbeFrame | None:
    run_len = 0
    for frame in session.frames:
        if frame.is_blank:
            run_len += 1
        else:
            run_len = 0
        if run_len >= k_frames:
            return frame
    return None


def frame_level_roc(
    sessions: list[AnalyzedSession],
    observed_signal_frame_ms: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for k in ROC_K_VALUES:
        latencies_ms: list[float] = []
        false_by_margin = {margin: 0 for margin in FALSE_EARLY_MARGINS_MS}
        never = 0
        fired = 0
        for session in sessions:
            trigger = first_blank_run_trigger(session, k)
            if trigger is None or trigger.available_wall_time is None:
                never += 1
                continue
            fired += 1
            latency_ms = (trigger.available_wall_time - session.vad_stop_wallclock) * 1000.0
            latencies_ms.append(latency_ms)
            for margin in FALSE_EARLY_MARGINS_MS:
                if trigger.available_wall_time < session.vad_stop_wallclock - margin / 1000.0:
                    false_by_margin[margin] += 1
        rows.append(
            {
                "K_frames": k,
                "K_ms": round_float(k * observed_signal_frame_ms if observed_signal_frame_ms is not None else None),
                "det_lat_p50_ms": round_float(median(latencies_ms)),
                "det_lat_p95_ms": round_float(percentile(latencies_ms, 95)),
                "false_fire@100ms": round_float(safe_div(false_by_margin[100], len(sessions)), 6),
                "false_fire@200ms": round_float(safe_div(false_by_margin[200], len(sessions)), 6),
                "never_fired_rate": round_float(safe_div(never, len(sessions)), 6),
                "triggered_sessions": fired,
                "n_sessions": len(sessions),
            }
        )
    return rows


def trailing_blank_run_at(session: AnalyzedSession, target_audio_s: float) -> int | None:
    if not session.frames or not session.frame_audio_times_s:
        return None
    if session.last_audio_cursor_s is not None and target_audio_s > session.last_audio_cursor_s + MODEL_FRAME_MS / 1000.0:
        return None
    if target_audio_s < session.frame_audio_times_s[0]:
        return None
    index = bisect.bisect_right(session.frame_audio_times_s, target_audio_s) - 1
    if index < 0:
        return None
    run_len = 0
    while index >= 0 and session.frames[index].is_blank:
        run_len += 1
        index -= 1
    return run_len


def silence_tail_convergence(sessions: list[AnalyzedSession]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset_ms in TAIL_OFFSETS_MS:
        lengths: list[int] = []
        for session in sessions:
            target_audio_s = session.vad_stop_audio_s + offset_ms / 1000.0
            value = trailing_blank_run_at(session, target_audio_s)
            if value is not None:
                lengths.append(value)
        rows.append(
            {
                "offset_ms": offset_ms,
                "eligible_sessions": len(lengths),
                "p50_trailing_blank_run_frames": round_float(median([float(value) for value in lengths])),
                "p95_trailing_blank_run_frames": round_float(percentile([float(value) for value in lengths], 95)),
                "fraction_ge_k": {
                    str(k): round_float(safe_div(sum(1 for value in lengths if value >= k), len(lengths)), 6)
                    for k in TAIL_K_VALUES
                },
            }
        )
    return rows


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def fmt_value(value: Any, digits: int = 1) -> str:
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct_fraction(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value) * 100.0:.1f}%"


def fmt_pct_percent(value: Any) -> str:
    if value is None:
        return "None"
    return f"{float(value):.1f}%"


def render_analysis_1(analysis: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    frame_meta = diagnostics["frame_granularity"]
    lines = [
        "",
        "Analysis 1 - Blank-Run Length Distribution",
        f"  sessions analyzed: {diagnostics['n_sessions_analyzed']}",
        f"  probe rows loaded: {diagnostics['probe_rows_loaded']}",
        f"  join method: {diagnostics['join']['method']}",
        "  observed frame granularity: "
        f"{fmt_value(frame_meta['estimated_observed_signal_frame_ms'])} ms "
        f"({fmt_value(frame_meta['median_frame_alignment_entries_per_chunk'])} "
        "frame_alignment entries per chunk)",
        f"  in-speech histogram: {analysis['in_speech_histogram']}",
        f"  post-silence histogram: {analysis['post_silence_histogram']}",
        f"  longest in-speech blank run: {analysis['longest_in_speech_blank_run_frames']} frames",
        f"  longest post-silence blank run: {analysis['longest_post_silence_blank_run_frames']} frames",
        f"  zero-in-speech-false-fire minimum K: {analysis['zero_in_speech_false_fire_min_k_frames']} frames",
        "  in-speech blank-run median/p95: "
        f"{fmt_value(analysis['in_speech_blank_run_median_frames'])}/"
        f"{fmt_value(analysis['in_speech_blank_run_p95_frames'])} frames",
        "  fraction of in-speech blank-runs >= K:",
    ]
    if frame_meta.get("warning"):
        lines.insert(6, f"  warning: {frame_meta['warning']}")
    fractions = analysis["in_speech_blank_run_ge_k_fraction"]
    for k in RUN_K_VALUES:
        lines.append(f"    K={k:<3} {fmt_pct_fraction(fractions[str(k)])}")
    return "\n".join(lines)


def render_analysis_2(analysis: dict[str, Any]) -> str:
    lines = [
        "",
        "Analysis 2 - Silence Trailer Emissions",
        f"  post-silence non-blank frames: {analysis['post_silence_non_blank_frame_count']}",
        f"  post-silence token emissions: {analysis['post_silence_token_emission_count']}",
        "  token classes:",
    ]
    for name, item in analysis["classes"].items():
        lines.append(f"    {name:<18} {item['count']:>5}  {fmt_pct_percent(item['percentage'])}")
    lines.append("  top-20 token strings:")
    for item in analysis["top_20_token_strings"]:
        lines.append(f"    {item['token']!r:<18} {item['count']}")
    first_audio = analysis["first_post_silence_token_audio_offset_ms"]
    first_wall = analysis["first_post_silence_token_wall_latency_ms"]
    lines.extend(
        [
            "  first post-silence token audio offset median/p95/min/max: "
            f"{fmt_value(first_audio['median'])}/"
            f"{fmt_value(first_audio['p95'])}/"
            f"{fmt_value(first_audio['min'])}/"
            f"{fmt_value(first_audio['max'])} ms",
            "  first post-silence token availability latency median/p95: "
            f"{fmt_value(first_wall['median'])}/"
            f"{fmt_value(first_wall['p95'])} ms",
            "  sessions with >=1 post-silence emission: "
            f"{analysis['sessions_with_post_silence_emission']}/{analysis['n_sessions']} "
            f"({fmt_pct_fraction(analysis['session_fraction_with_post_silence_emission'])})",
        ]
    )
    return "\n".join(lines)


def render_analysis_3(rows: list[dict[str, Any]]) -> str:
    lines = [
        "",
        "Analysis 3 - Frame-Level Blank-Run ROC",
        "  K_frames  K_ms   det_lat_p50_ms  det_lat_p95_ms  false@100  false@200  never  n",
    ]
    for row in rows:
        lines.append(
            f"  {row['K_frames']:>8}  "
            f"{fmt_value(row['K_ms'], 0):>4}  "
            f"{fmt_value(row['det_lat_p50_ms']):>16}  "
            f"{fmt_value(row['det_lat_p95_ms']):>16}  "
            f"{fmt_pct_fraction(row['false_fire@100ms']):>9}  "
            f"{fmt_pct_fraction(row['false_fire@200ms']):>9}  "
            f"{fmt_pct_fraction(row['never_fired_rate']):>6}  "
            f"{row['n_sessions']:>3}"
        )
    return "\n".join(lines)


def render_analysis_4(rows: list[dict[str, Any]]) -> str:
    lines = [
        "",
        "Analysis 4 - Silence-Tail Convergence",
        "  offset_ms  n   p50_frames  p95_frames  >=20    >=50    >=100   >=200",
    ]
    for row in rows:
        fractions = row["fraction_ge_k"]
        lines.append(
            f"  {row['offset_ms']:>9}  "
            f"{row['eligible_sessions']:>3}  "
            f"{fmt_value(row['p50_trailing_blank_run_frames']):>10}  "
            f"{fmt_value(row['p95_trailing_blank_run_frames']):>10}  "
            f"{fmt_pct_fraction(fractions['20']):>6}  "
            f"{fmt_pct_fraction(fractions['50']):>6}  "
            f"{fmt_pct_fraction(fractions['100']):>7}  "
            f"{fmt_pct_fraction(fractions['200']):>7}"
        )
    return "\n".join(lines)


def render_report(
    diagnostics: dict[str, Any],
    analysis1: dict[str, Any],
    analysis2: dict[str, Any],
    analysis3: list[dict[str, Any]],
    analysis4: list[dict[str, Any]],
) -> str:
    lines = [render_analysis_1(analysis1, diagnostics)]
    lines.append(render_analysis_2(analysis2))
    lines.append(render_analysis_3(analysis3))
    lines.append(render_analysis_4(analysis4))
    return "\n".join(lines)


def build_summary(
    diagnostics: dict[str, Any],
    analysis1: dict[str, Any],
    analysis2: dict[str, Any],
    analysis3: list[dict[str, Any]],
    analysis4: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "n_sessions_analyzed": diagnostics["n_sessions_analyzed"],
        "probe_rows_loaded": diagnostics["probe_rows_loaded"],
        "finalize_rows_loaded": diagnostics["finalize_rows_loaded"],
        "observed_frame_granularity": diagnostics["frame_granularity"],
        "blank_runs": {
            "longest_in_speech_frames": analysis1["longest_in_speech_blank_run_frames"],
            "longest_post_silence_frames": analysis1["longest_post_silence_blank_run_frames"],
            "zero_false_min_k_frames": analysis1["zero_in_speech_false_fire_min_k_frames"],
            "median_in_speech_frames": analysis1["in_speech_blank_run_median_frames"],
            "p95_in_speech_frames": analysis1["in_speech_blank_run_p95_frames"],
        },
        "post_silence_emissions": {
            "token_count": analysis2["post_silence_token_emission_count"],
            "session_fraction": analysis2["session_fraction_with_post_silence_emission"],
            "first_token_audio_offset_median_ms": analysis2["first_post_silence_token_audio_offset_ms"]["median"],
            "first_token_wall_latency_median_ms": analysis2["first_post_silence_token_wall_latency_ms"]["median"],
        },
        "roc": analysis3,
        "tail_convergence": analysis4,
    }


def main() -> int:
    args = parse_args()
    telemetry_rows, telemetry_diag = load_telemetry_rows(args.finalize_jsonl)
    probe_sessions, probe_diag = load_probe_sessions(args.probe_jsonl)
    joined_pairs, join_diag, skip_reasons = join_sessions(probe_sessions, telemetry_rows)
    analyzed_sessions, analyzed_diag = build_analyzed_sessions(joined_pairs, args.audio_dir)
    frame_granularity = estimate_observed_signal_frame_ms(probe_sessions)
    observed_frame_ms = frame_granularity.get("estimated_observed_signal_frame_ms")
    observed_frame_ms_float = float(observed_frame_ms) if observed_frame_ms is not None else None

    analysis1 = blank_run_distribution(analyzed_sessions)
    analysis2 = silence_trailer_emissions(analyzed_sessions)
    analysis3 = frame_level_roc(analyzed_sessions, observed_frame_ms_float)
    analysis4 = silence_tail_convergence(analyzed_sessions)

    diagnostics = {
        "inputs": {
            "probe_jsonl": str(args.probe_jsonl),
            "finalize_jsonl": str(args.finalize_jsonl),
            "audio_dir": str(args.audio_dir),
        },
        "n_sessions_analyzed": len(analyzed_sessions),
        "n_sessions_skipped": {"total": sum(skip_reasons.values()), "reasons": dict(skip_reasons)},
        "probe_rows_loaded": probe_diag["rows_loaded"],
        "finalize_rows_loaded": telemetry_diag["rows_loaded"],
        "probe": probe_diag,
        "telemetry": telemetry_diag,
        "join": join_diag,
        "analyzed": analyzed_diag,
        "frame_granularity": frame_granularity,
        "time_alignment": {
            "vad_stop_audio_time": "vad_stop_wallclock - (first_chunk.wall_time_done - first_chunk.real_audio_cursor_seconds)",
            "frame_audio_time": "chunk.real_audio_cursor_seconds - shift_frames*10ms + frame_offset*10ms",
            "trigger_wallclock": "chunk.wall_time_done for the chunk that exposed the triggering frame",
        },
    }
    summary = build_summary(diagnostics, analysis1, analysis2, analysis3, analysis4)

    print(compact_json(summary))
    print(render_report(diagnostics, analysis1, analysis2, analysis3, analysis4))

    if args.output_json is not None:
        payload = {
            "summary": summary,
            "diagnostics": diagnostics,
            "analysis_1_blank_run_distribution": analysis1,
            "analysis_2_silence_trailer_emissions": analysis2,
            "analysis_3_frame_level_roc": analysis3,
            "analysis_4_silence_tail_convergence": analysis4,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
