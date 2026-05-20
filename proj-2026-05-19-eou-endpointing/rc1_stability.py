#!/usr/bin/env python3
"""Measure rc1 stability from Step-1/2 EOU probe JSONL files.

This is a pure offline analyzer. It reads the per-chunk cumulative token series
captured by `NEMOTRON_EOU_PROBE=1` and classifies consecutive chunk-pairs by
whether prior token positions changed before or after their rc1 right-context
window closed.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CLASS_I = "i"
CLASS_II_A = "ii-a"
CLASS_II_B = "ii-b"
CLASS_III = "iii"
CLASS_APPEND_ONLY = "append-only"
CLASS_UNKNOWN = "unknown"
CLASS_KEYS = [
    CLASS_I,
    CLASS_II_A,
    CLASS_II_B,
    CLASS_III,
    CLASS_APPEND_ONLY,
    CLASS_UNKNOWN,
]

SANITY_SHORTENING = "shortening"
BOUNDARY_MARKERS = ("\u2581", "\u0120")


@dataclass(frozen=True)
class TokenInfo:
    token_id: int
    token: str
    model_frame_index: int | None


@dataclass(frozen=True)
class ProbeRow:
    source_path: str
    source_line: int
    run_tag: str
    session_id: str
    chunk_index: int
    y_sequence: list[int]
    emitted_frames_after: int
    right_context: int
    new_tokens: list[dict[str, Any]]


@dataclass
class SessionState:
    token_history: dict[int, TokenInfo]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-jsonl",
        type=Path,
        action="append",
        required=True,
        help="Step-1/2 .eou_probe.jsonl path. May be repeated.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path for the full structured result.",
    )
    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=5,
        help="Maximum concrete examples to retain per classification bucket.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Only emit the machine-readable one-line JSON summary to stdout.",
    )
    return parser.parse_args()


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


def load_right_context(record: dict[str, Any]) -> int | None:
    """Return the rc1 span in encoder frames.

    The task contract names this field `right_context`. The landed Step-2
    writer at f6fc884 emits the same value as `R` and `right_context_frames`.
    Accept all three spellings so the analyzer works against both contracts.
    """

    for key in ("right_context", "R", "right_context_frames"):
        value = as_int(record.get(key))
        if value is not None:
            return value
    return None


def is_int_list(value: Any) -> bool:
    return isinstance(value, list) and all(as_int(item) is not None for item in value)


def parse_probe_row(record: Any, source_path: Path, source_line: int) -> tuple[ProbeRow | None, str | None]:
    if not isinstance(record, dict):
        return None, "not_object"

    required = [
        "chunk_index",
        "session_id",
        "y_sequence",
        "emitted_frames_after",
        "new_tokens",
        "run_tag",
    ]
    missing = [key for key in required if key not in record]
    if load_right_context(record) is None:
        missing.append("right_context/R/right_context_frames")
    if missing:
        return None, "missing:" + ",".join(missing)

    chunk_index = as_int(record.get("chunk_index"))
    emitted_frames_after = as_int(record.get("emitted_frames_after"))
    right_context = load_right_context(record)
    y_sequence = record.get("y_sequence")
    new_tokens = record.get("new_tokens")
    run_tag = record.get("run_tag")
    session_id = record.get("session_id")
    if (
        chunk_index is None
        or emitted_frames_after is None
        or right_context is None
        or not isinstance(session_id, str)
        or not isinstance(run_tag, str)
        or not is_int_list(y_sequence)
        or not isinstance(new_tokens, list)
        or not all(isinstance(item, dict) for item in new_tokens)
    ):
        return None, "bad_type"

    return (
        ProbeRow(
            source_path=str(source_path),
            source_line=source_line,
            run_tag=run_tag,
            session_id=session_id,
            chunk_index=chunk_index,
            y_sequence=[int(item) for item in y_sequence],
            emitted_frames_after=emitted_frames_after,
            right_context=right_context,
            new_tokens=new_tokens,
        ),
        None,
    )


def load_probe_rows(paths: Iterable[Path]) -> tuple[list[ProbeRow], dict[str, Any]]:
    rows: list[ProbeRow] = []
    diagnostics: dict[str, Any] = {
        "input_files": [],
        "rows_read": 0,
        "rows_loaded": 0,
        "missing_or_invalid_rows": 0,
        "invalid_json_rows": 0,
        "skip_reasons": Counter(),
    }
    for path in paths:
        diagnostics["input_files"].append(str(path))
        with path.open("r", encoding="utf-8") as f:
            for source_line, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                diagnostics["rows_read"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    diagnostics["invalid_json_rows"] += 1
                    diagnostics["missing_or_invalid_rows"] += 1
                    diagnostics["skip_reasons"]["invalid_json"] += 1
                    continue
                row, reason = parse_probe_row(record, path, source_line)
                if row is None:
                    diagnostics["missing_or_invalid_rows"] += 1
                    diagnostics["skip_reasons"][reason or "invalid"] += 1
                    continue
                rows.append(row)
                diagnostics["rows_loaded"] += 1
    diagnostics["skip_reasons"] = dict(diagnostics["skip_reasons"])
    return rows, diagnostics


def lcp_len(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def diff_positions(prev_y: list[int], curr_y: list[int]) -> list[int]:
    positions: list[int] = []
    for index in range(max(len(prev_y), len(curr_y))):
        prev_token = prev_y[index] if index < len(prev_y) else None
        curr_token = curr_y[index] if index < len(curr_y) else None
        if prev_token != curr_token:
            positions.append(index)
    return positions


def int_from_token(token: dict[str, Any], key: str) -> int | None:
    return as_int(token.get(key))


def token_string_from_record(token: dict[str, Any], token_id: int) -> str:
    value = token.get("token")
    if isinstance(value, str):
        return value
    return f"<TOKEN:{token_id}>"


def new_token_map(row: ProbeRow) -> dict[int, TokenInfo]:
    mapped: dict[int, TokenInfo] = {}
    for token in row.new_tokens:
        token_index = int_from_token(token, "token_index")
        token_id = int_from_token(token, "token_id")
        if token_index is None or token_id is None:
            continue
        mapped[token_index] = TokenInfo(
            token_id=token_id,
            token=token_string_from_record(token, token_id),
            model_frame_index=int_from_token(token, "model_frame_index"),
        )
    return mapped


def token_info_for_position(
    index: int,
    token_id: int,
    *,
    current_new_tokens: dict[int, TokenInfo],
    history: dict[int, TokenInfo],
) -> TokenInfo:
    token = current_new_tokens.get(index)
    if token is not None:
        return token
    token = history.get(index)
    if token is not None and token.token_id == token_id:
        return token
    if token is not None and token.model_frame_index is not None:
        return TokenInfo(
            token_id=token_id,
            token=f"<TOKEN:{token_id}>",
            model_frame_index=token.model_frame_index,
        )
    return TokenInfo(token_id=token_id, token=f"<TOKEN:{token_id}>", model_frame_index=None)


def history_for_row(row: ProbeRow, prior_history: dict[int, TokenInfo]) -> dict[int, TokenInfo]:
    current_new_tokens = new_token_map(row)
    rebuilt: dict[int, TokenInfo] = {}
    for index, token_id in enumerate(row.y_sequence):
        rebuilt[index] = token_info_for_position(
            index,
            token_id,
            current_new_tokens=current_new_tokens,
            history=prior_history,
        )
    return rebuilt


def render_from_sequence(y_sequence: list[int], history: dict[int, TokenInfo]) -> str:
    tokens = [
        history.get(index, TokenInfo(token_id=token_id, token=f"<TOKEN:{token_id}>", model_frame_index=None)).token
        for index, token_id in enumerate(y_sequence)
    ]
    rendered = "".join(tokens)
    for marker in BOUNDARY_MARKERS:
        rendered = rendered.replace(marker, " ")
    return rendered.strip()


def semantic_key(text: str) -> str:
    chars: list[str] = []
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        chars.append(char.casefold())
    return "".join(chars)


WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)


def word_list(text: str) -> list[str]:
    return WORD_RE.findall(text.casefold())


def token_starts_word(token: str) -> bool:
    return token.startswith(BOUNDARY_MARKERS) or token[:1].isspace()


def trailing_word_start(y_sequence: list[int], history: dict[int, TokenInfo]) -> int:
    if not y_sequence:
        return 0
    index = len(y_sequence) - 1
    while index > 0:
        token = history.get(index)
        if token is not None and token_starts_word(token.token):
            return index
        index -= 1
    return 0


def is_trailing_subword_change(
    existing_changed_positions: list[int],
    prev_y: list[int],
    curr_y: list[int],
    prev_history: dict[int, TokenInfo],
    curr_history: dict[int, TokenInfo],
) -> bool:
    if not existing_changed_positions:
        return False
    prev_tail_start = trailing_word_start(prev_y, prev_history)
    curr_tail_start = trailing_word_start(curr_y, curr_history)
    tail_start = min(prev_tail_start, curr_tail_start)
    return min(existing_changed_positions) >= tail_start


def format_tail(sequence: list[int], width: int = 10) -> list[int]:
    return sequence[-width:]


def changed_position_details(
    positions: list[int],
    prev_y: list[int],
    curr_y: list[int],
    history: dict[int, TokenInfo],
    threshold_frame: int,
) -> list[dict[str, Any]]:
    details = []
    for index in positions:
        info = history.get(index)
        model_frame_index = info.model_frame_index if info is not None else None
        details.append(
            {
                "token_index": index,
                "prev_token_id": prev_y[index] if index < len(prev_y) else None,
                "curr_token_id": curr_y[index] if index < len(curr_y) else None,
                "token": info.token if info is not None else None,
                "model_frame_index": model_frame_index,
                "rc1_state": (
                    "unknown"
                    if model_frame_index is None
                    else "settled"
                    if model_frame_index < threshold_frame
                    else "provisional"
                ),
            }
        )
    return details


def make_example(
    *,
    classification: str,
    reason: str,
    prev: ProbeRow,
    curr: ProbeRow,
    positions: list[int],
    history: dict[int, TokenInfo],
    threshold_frame: int,
    prev_rendered_text: str,
    curr_rendered_text: str,
    prefix_len: int,
) -> dict[str, Any]:
    return {
        "class": classification,
        "reason": reason,
        "source_path": curr.source_path,
        "run_tag": curr.run_tag,
        "session_id": curr.session_id,
        "prev_chunk_index": prev.chunk_index,
        "curr_chunk_index": curr.chunk_index,
        "prefix_len": prefix_len,
        "prev_y_sequence_tail": format_tail(prev.y_sequence),
        "curr_y_sequence_tail": format_tail(curr.y_sequence),
        "threshold_frame": threshold_frame,
        "R": curr.right_context,
        "emitted_frames_after": curr.emitted_frames_after,
        "changed_positions": changed_position_details(
            positions,
            prev.y_sequence,
            curr.y_sequence,
            history,
            threshold_frame,
        ),
        "prev_rendered_text": prev_rendered_text,
        "curr_rendered_text": curr_rendered_text,
    }


def classify_pair(
    prev: ProbeRow,
    curr: ProbeRow,
    state: SessionState,
) -> tuple[str | None, dict[str, Any] | None]:
    prefix_len = lcp_len(prev.y_sequence, curr.y_sequence)
    current_new_tokens = new_token_map(curr)
    render_changed_positions = []
    for index, info in current_new_tokens.items():
        if index >= min(len(prev.y_sequence), len(curr.y_sequence)):
            continue
        if prev.y_sequence[index] != curr.y_sequence[index]:
            continue
        prior = state.token_history.get(index)
        if prior is not None and prior.token != info.token:
            render_changed_positions.append(index)
    positions = sorted(set(diff_positions(prev.y_sequence, curr.y_sequence) + render_changed_positions))

    if not positions:
        return None, {
            "skip_reason": "no-change",
            "prefix_len": prefix_len,
        }

    curr_history = history_for_row(curr, state.token_history)
    prev_history = history_for_row(prev, state.token_history)
    prev_rendered_text = render_from_sequence(prev.y_sequence, prev_history)
    curr_rendered_text = render_from_sequence(curr.y_sequence, curr_history)
    threshold_frame = curr.emitted_frames_after - curr.right_context

    if len(curr.y_sequence) < len(prev.y_sequence):
        return SANITY_SHORTENING, make_example(
            classification=SANITY_SHORTENING,
            reason="current cumulative y_sequence is shorter than the previous chunk",
            prev=prev,
            curr=curr,
            positions=positions,
            history=curr_history,
            threshold_frame=threshold_frame,
            prev_rendered_text=prev_rendered_text,
            curr_rendered_text=curr_rendered_text,
            prefix_len=prefix_len,
        )

    existing_changed_positions = [index for index in positions if index < len(prev.y_sequence)]
    appended_positions = [index for index in positions if index >= len(prev.y_sequence)]
    if not existing_changed_positions:
        return CLASS_APPEND_ONLY, make_example(
            classification=CLASS_APPEND_ONLY,
            reason="token-id prefix unchanged; only new positions appended",
            prev=prev,
            curr=curr,
            positions=positions,
            history=curr_history,
            threshold_frame=threshold_frame,
            prev_rendered_text=prev_rendered_text,
            curr_rendered_text=curr_rendered_text,
            prefix_len=prefix_len,
        )

    classification_positions = sorted(set(existing_changed_positions + appended_positions))
    frame_by_position: dict[int, int | None] = {}
    for index in classification_positions:
        token_id = curr.y_sequence[index] if index < len(curr.y_sequence) else prev.y_sequence[index]
        info = token_info_for_position(
            index,
            token_id,
            current_new_tokens=current_new_tokens,
            history=state.token_history,
        )
        frame_by_position[index] = info.model_frame_index

    if any(frame is None for frame in frame_by_position.values()):
        return CLASS_UNKNOWN, make_example(
            classification=CLASS_UNKNOWN,
            reason="one or more changed positions lacked a model_frame_index",
            prev=prev,
            curr=curr,
            positions=classification_positions,
            history=curr_history,
            threshold_frame=threshold_frame,
            prev_rendered_text=prev_rendered_text,
            curr_rendered_text=curr_rendered_text,
            prefix_len=prefix_len,
        )

    settled_positions = [
        index for index, frame in frame_by_position.items() if frame is not None and frame < threshold_frame
    ]
    settled_existing_id_changes = [
        index
        for index in existing_changed_positions
        if index < len(curr.y_sequence)
        and prev.y_sequence[index] != curr.y_sequence[index]
        and frame_by_position.get(index) is not None
        and frame_by_position[index] < threshold_frame
    ]
    all_provisional = all(
        frame is not None and frame >= threshold_frame for frame in frame_by_position.values()
    )
    all_settled = bool(classification_positions) and all(
        frame is not None and frame < threshold_frame for frame in frame_by_position.values()
    )
    semantically_same = semantic_key(prev_rendered_text) == semantic_key(curr_rendered_text)
    words_changed = word_list(prev_rendered_text) != word_list(curr_rendered_text)
    trailing_subword = is_trailing_subword_change(
        existing_changed_positions,
        prev.y_sequence,
        curr.y_sequence,
        prev_history,
        curr_history,
    )

    if settled_existing_id_changes:
        return CLASS_III, make_example(
            classification=CLASS_III,
            reason="a settled token position changed token id",
            prev=prev,
            curr=curr,
            positions=classification_positions,
            history=curr_history,
            threshold_frame=threshold_frame,
            prev_rendered_text=prev_rendered_text,
            curr_rendered_text=curr_rendered_text,
            prefix_len=prefix_len,
        )

    if all_settled and words_changed and (prefix_len == len(prev.y_sequence) or trailing_subword):
        return CLASS_II_B, make_example(
            classification=CLASS_II_B,
            reason="settled render changes decoded words with prefix unchanged or trailing-subword change",
            prev=prev,
            curr=curr,
            positions=classification_positions,
            history=curr_history,
            threshold_frame=threshold_frame,
            prev_rendered_text=prev_rendered_text,
            curr_rendered_text=curr_rendered_text,
            prefix_len=prefix_len,
        )

    if all_settled and semantically_same and prev_rendered_text != curr_rendered_text:
        return CLASS_II_A, make_example(
            classification=CLASS_II_A,
            reason="settled render changed only by casing, punctuation, or whitespace",
            prev=prev,
            curr=curr,
            positions=classification_positions,
            history=curr_history,
            threshold_frame=threshold_frame,
            prev_rendered_text=prev_rendered_text,
            curr_rendered_text=curr_rendered_text,
            prefix_len=prefix_len,
        )

    if all_provisional:
        return CLASS_I, make_example(
            classification=CLASS_I,
            reason="all changed positions are within the current rc1 provisional tail",
            prev=prev,
            curr=curr,
            positions=classification_positions,
            history=curr_history,
            threshold_frame=threshold_frame,
            prev_rendered_text=prev_rendered_text,
            curr_rendered_text=curr_rendered_text,
            prefix_len=prefix_len,
        )

    reason = (
        "mixed or render-only edge case outside the explicit rc1-stability classes; "
        f"settled_positions={settled_positions}"
    )
    return CLASS_UNKNOWN, make_example(
        classification=CLASS_UNKNOWN,
        reason=reason,
        prev=prev,
        curr=curr,
        positions=classification_positions,
        history=curr_history,
        threshold_frame=threshold_frame,
        prev_rendered_text=prev_rendered_text,
        curr_rendered_text=curr_rendered_text,
        prefix_len=prefix_len,
    )


def update_session_state(state: SessionState, row: ProbeRow) -> None:
    state.token_history = history_for_row(row, state.token_history)


def rate_dict(counts: Counter[str], total_events: int) -> dict[str, float]:
    return {
        key: (counts.get(key, 0) / total_events if total_events else 0.0)
        for key in CLASS_KEYS
    }


def analyze(rows: list[ProbeRow], examples_per_class: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[ProbeRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.run_tag, row.session_id)].append(row)

    counts: Counter[str] = Counter({key: 0 for key in CLASS_KEYS})
    sanity_counts: Counter[str] = Counter()
    row_r_distribution: Counter[str] = Counter()
    event_r_distribution: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {key: [] for key in CLASS_KEYS}
    examples[SANITY_SHORTENING] = []
    per_session: dict[str, dict[str, Any]] = {}
    skipped_pairs = Counter()
    non_increasing_pairs = 0

    for row in rows:
        row_r_distribution[str(row.right_context)] += 1

    for (run_tag, session_id), session_rows in sorted(grouped.items()):
        session_rows.sort(key=lambda item: (item.chunk_index, item.source_path, item.source_line))
        state = SessionState(token_history={})
        session_key = f"{run_tag}:{session_id}"
        session_counts: Counter[str] = Counter({key: 0 for key in CLASS_KEYS})
        session_sanity: Counter[str] = Counter()
        session_skips: Counter[str] = Counter()

        if not session_rows:
            continue
        update_session_state(state, session_rows[0])
        prev = session_rows[0]
        for curr in session_rows[1:]:
            if curr.chunk_index <= prev.chunk_index:
                non_increasing_pairs += 1
                session_skips["non-increasing-chunk-index"] += 1
                update_session_state(state, curr)
                prev = curr
                continue

            classification, example = classify_pair(prev, curr, state)
            if classification is None:
                reason = "no-change"
                if example and isinstance(example.get("skip_reason"), str):
                    reason = example["skip_reason"]
                skipped_pairs[reason] += 1
                session_skips[reason] += 1
            elif classification == SANITY_SHORTENING:
                sanity_counts[SANITY_SHORTENING] += 1
                session_sanity[SANITY_SHORTENING] += 1
                if len(examples[SANITY_SHORTENING]) < examples_per_class and example is not None:
                    examples[SANITY_SHORTENING].append(example)
            else:
                counts[classification] += 1
                session_counts[classification] += 1
                event_r_distribution[str(curr.right_context)] += 1
                if len(examples[classification]) < examples_per_class and example is not None:
                    examples[classification].append(example)

            update_session_state(state, curr)
            prev = curr

        per_session[session_key] = {
            "run_tag": run_tag,
            "session_id": session_id,
            "rows": len(session_rows),
            "counts": {key: session_counts.get(key, 0) for key in CLASS_KEYS},
            "sanity_counts": dict(session_sanity),
            "skipped_pairs": dict(session_skips),
        }

    total_events = sum(counts.get(key, 0) for key in CLASS_KEYS)
    summary = {
        "total_events": total_events,
        "by_class": {key: counts.get(key, 0) for key in CLASS_KEYS},
        "rates": rate_dict(counts, total_events),
        "R_distribution": dict(sorted(event_r_distribution.items(), key=lambda item: int(item[0]))),
    }
    return {
        "summary": summary,
        "counts": {key: counts.get(key, 0) for key in CLASS_KEYS},
        "rates": summary["rates"],
        "examples": examples,
        "sanity_counts": dict(sanity_counts),
        "row_R_distribution": dict(sorted(row_r_distribution.items(), key=lambda item: int(item[0]))),
        "skipped_pairs": dict(skipped_pairs),
        "non_increasing_pairs": non_increasing_pairs,
        "per_session": per_session,
    }


def truncate_text(text: str, width: int = 220) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= width:
        return one_line
    return one_line[: width - 3] + "..."


def render_counts_table(result: dict[str, Any]) -> str:
    total = result["summary"]["total_events"]
    lines = ["Counts:", "  class        count    rate"]
    for key in CLASS_KEYS:
        count = result["counts"].get(key, 0)
        rate = count / total if total else 0.0
        lines.append(f"  {key:<11} {count:>6}  {rate:>7.2%}")
    return "\n".join(lines)


def render_example(example: dict[str, Any]) -> str:
    positions = example.get("changed_positions", [])
    position_text = ", ".join(
        (
            f"idx={item.get('token_index')} "
            f"prev={item.get('prev_token_id')} curr={item.get('curr_token_id')} "
            f"frame={item.get('model_frame_index')} {item.get('rc1_state')}"
        )
        for item in positions
    )
    return "\n".join(
        [
            (
                f"  - {example.get('run_tag')}:{example.get('session_id')} "
                f"chunks {example.get('prev_chunk_index')}->{example.get('curr_chunk_index')} "
                f"R={example.get('R')} threshold={example.get('threshold_frame')}"
            ),
            f"    reason: {example.get('reason')}",
            f"    positions: {position_text}",
            f"    prev_y[-10:]: {example.get('prev_y_sequence_tail')}",
            f"    curr_y[-10:]: {example.get('curr_y_sequence_tail')}",
            f"    prev text: {truncate_text(str(example.get('prev_rendered_text', '')))}",
            f"    curr text: {truncate_text(str(example.get('curr_rendered_text', '')))}",
        ]
    )


def render_human_report(result: dict[str, Any], diagnostics: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(render_counts_table(result))
    lines.append("")
    lines.append(f"Loaded rows: {diagnostics['rows_loaded']} / {diagnostics['rows_read']}")
    lines.append(f"Skipped invalid rows: {diagnostics['missing_or_invalid_rows']}")
    if diagnostics.get("skip_reasons"):
        lines.append(f"Row skip reasons: {diagnostics['skip_reasons']}")
    lines.append(f"Skipped no-change pairs: {result['skipped_pairs'].get('no-change', 0)}")
    lines.append(f"Non-increasing chunk-index pairs skipped: {result['non_increasing_pairs']}")
    lines.append(f"Row R distribution: {result['row_R_distribution']}")
    lines.append(f"Event R distribution: {result['summary']['R_distribution']}")

    sanity_counts = result.get("sanity_counts", {})
    if sanity_counts:
        lines.append(f"Sanity buckets: {sanity_counts}")
    if sanity_counts.get(SANITY_SHORTENING, 0):
        lines.append("WARNING: shortening events were observed; greedy RNNT cumulative output shrank.")

    lines.append("")
    lines.append("Examples:")
    for key in CLASS_KEYS + [SANITY_SHORTENING]:
        class_examples = result["examples"].get(key, [])
        if not class_examples:
            continue
        lines.append(f"{key}:")
        for example in class_examples:
            lines.append(render_example(example))

    if result["counts"].get(CLASS_III, 0) > 0:
        lines.append("")
        lines.append(
            "WARNING: Step 2b approach-gating finding: genuine beyond-rc1 token-id edits "
            "were observed. Inspect the class iii examples before treating the no-arbitrary-rewrite "
            "mechanism as confirmed on the full operator-driven subset."
        )

    return "\n".join(lines)


def compact_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, sort_keys=True, separators=(",", ":"))


def main() -> int:
    args = parse_args()
    if args.examples_per_class < 0:
        raise ValueError("--examples-per-class must be >= 0")

    rows, diagnostics = load_probe_rows(args.probe_jsonl)
    result = analyze(rows, args.examples_per_class)
    result["diagnostics"] = diagnostics

    print(compact_summary(result["summary"]))
    if not args.silent:
        print(render_human_report(result, diagnostics))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
