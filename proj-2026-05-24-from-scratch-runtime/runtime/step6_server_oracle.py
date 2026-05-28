#!/usr/bin/env python3
"""First-pass production-server event-stream oracle.

This harness drives ``src/nemotron_speech/server.py`` over its real WebSocket
protocol in continuous mode and compares emitted transcript JSON against the
session bundle's text-event oracle.  The bundle oracle is the event stream that
``cpp/session_main.cpp`` already checks event-for-event, so this closes the
remaining transitivity edge against the shipping server without modifying the
server or C++ harness.

Run from this directory with the parakeet env:

  HF_HUB_OFFLINE=1 ./.venv/bin/python step6_server_oracle.py --n 8 --start-server

The client intentionally follows the same wire protocol used by
``stt_benchmark.nemotron_local_stt``: raw 16 kHz mono int16 PCM binary frames
plus ``vad_start``/``vad_stop`` JSON control messages.  It does not use the
Pipecat benchmark pipeline directly because that service intentionally drops
raw interim/final JSON details that this oracle must inspect.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import datetime as dt
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(__file__).resolve().parent
ART = RUNTIME / "artifacts"
LOG_DIR = ART / "logs"
SERVER = ROOT / "src" / "nemotron_speech" / "server.py"
DEFAULT_BUNDLE = ART / "session_audio_bundle.ts"
DEFAULT_PYTHON = "./.venv/bin/python"

EVENT_INTERIM = 0
EVENT_FINAL = 1
EVENT_SUPPRESSED = 2


@dataclasses.dataclass
class TextEvent:
    kind: str
    text: str
    collector_text: str
    is_final: bool
    raw: dict[str, Any] | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _bundle_attr(bundle: torch.jit.ScriptModule, name: str) -> torch.Tensor:
    return getattr(bundle, name)


def _scalar_i64(bundle: torch.jit.ScriptModule, name: str) -> int:
    return int(_bundle_attr(bundle, name).detach().cpu().reshape(-1)[0].item())


def _tensor_i64(bundle: torch.jit.ScriptModule, name: str) -> list[int]:
    tensor = _bundle_attr(bundle, name).detach().cpu().to(torch.int64).reshape(-1)
    return [int(v) for v in tensor.tolist()]


def _tensor_f32(bundle: torch.jit.ScriptModule, name: str) -> np.ndarray:
    tensor = _bundle_attr(bundle, name).detach().cpu().to(torch.float32).contiguous()
    return tensor.numpy().copy()


def _unpack_utf8(bundle: torch.jit.ScriptModule, bytes_name: str, offsets_name: str) -> list[str]:
    data = bytes(int(v) for v in _bundle_attr(bundle, bytes_name).detach().cpu().reshape(-1).tolist())
    offsets = _tensor_i64(bundle, offsets_name)
    if not offsets or offsets[0] != 0 or offsets[-1] != len(data):
        raise ValueError(f"invalid UTF-8 offsets for {bytes_name}/{offsets_name}")
    return [
        data[offsets[i] : offsets[i + 1]].decode("utf-8")
        for i in range(len(offsets) - 1)
    ]


def _one_utf8(bundle: torch.jit.ScriptModule, prefix: str, name: str) -> str:
    values = _unpack_utf8(
        bundle,
        f"{prefix}_{name}_bytes",
        f"{prefix}_{name}_offsets",
    )
    if len(values) != 1:
        raise ValueError(f"{prefix}_{name} expected one value, got {len(values)}")
    return values[0]


def _kind_name(kind: int) -> str:
    if kind == EVENT_INTERIM:
        return "interim"
    if kind == EVENT_FINAL:
        return "final"
    if kind == EVENT_SUPPRESSED:
        return "suppressed"
    return f"unknown:{kind}"


def _bundle_events(bundle: torch.jit.ScriptModule, utt: int) -> list[TextEvent]:
    prefix = f"utt{utt}"
    kinds = _tensor_i64(bundle, f"{prefix}_event_kinds")
    texts = _unpack_utf8(
        bundle,
        f"{prefix}_event_text_bytes",
        f"{prefix}_event_text_offsets",
    )
    collectors = _unpack_utf8(
        bundle,
        f"{prefix}_event_collector_text_bytes",
        f"{prefix}_event_collector_text_offsets",
    )
    if not (len(kinds) == len(texts) == len(collectors)):
        raise ValueError(f"{prefix} event payload count mismatch")
    out: list[TextEvent] = []
    for kind, text, collector in zip(kinds, texts, collectors):
        name = _kind_name(kind)
        out.append(
            TextEvent(
                kind=name,
                text=text,
                collector_text=collector,
                is_final=name in {"final", "suppressed"},
            )
        )
    return out


def _wire_gold(events: list[TextEvent]) -> list[TextEvent]:
    """Return events the server can actually emit on the WebSocket."""
    return [event for event in events if event.kind != "suppressed"]


def _float_audio_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    # Match stt-benchmark's PCM conversion and the server's /32768 decode.
    return (clipped * 32767.0).astype("<i2").tobytes()


def _final_collector(events: list[TextEvent]) -> str:
    finals = [event.collector_text for event in events if event.kind == "final"]
    if finals:
        return finals[-1]
    return ""


def _event_payload(event: TextEvent) -> dict[str, Any]:
    payload = {
        "kind": event.kind,
        "text": event.text,
        "collector_text": event.collector_text,
        "is_final": event.is_final,
    }
    if event.raw is not None:
        payload["raw"] = event.raw
    return payload


def _compare_events(server_events: list[TextEvent], gold_events: list[TextEvent]) -> dict[str, Any]:
    first_diff: dict[str, Any] | None = None
    n = min(len(server_events), len(gold_events))
    mismatches = 0
    for i in range(n):
        server = server_events[i]
        gold = gold_events[i]
        fields: dict[str, dict[str, Any]] = {}
        for key in ("kind", "text", "collector_text", "is_final"):
            lhs = getattr(server, key)
            rhs = getattr(gold, key)
            if lhs != rhs:
                fields[key] = {"server": lhs, "gold": rhs}
        if fields:
            mismatches += 1
            if first_diff is None:
                first_diff = {"index": i, "fields": fields}
    if len(server_events) != len(gold_events):
        mismatches += abs(len(server_events) - len(gold_events))
        if first_diff is None:
            first_diff = {
                "index": n,
                "fields": {
                    "event_count": {
                        "server": len(server_events),
                        "gold": len(gold_events),
                    }
                },
            }

    server_final = _final_collector(server_events)
    gold_final = _final_collector(gold_events)
    return {
        "event_match": len(server_events) == len(gold_events) and mismatches == 0,
        "final_match": server_final == gold_final,
        "event_mismatches": mismatches,
        "event_count_server": len(server_events),
        "event_count_gold": len(gold_events),
        "server_final_text": server_final,
        "gold_final_text": gold_final,
        "first_diff": first_diff,
    }


def _normalize_server_transcript(data: dict[str, Any], collector_text: str) -> tuple[TextEvent | None, str]:
    text = str(data.get("text") or "")
    is_final = bool(data.get("is_final"))
    if not is_final:
        return (
            TextEvent(
                kind="interim",
                text=text,
                collector_text=collector_text,
                is_final=False,
                raw=data,
            ),
            collector_text,
        )

    if data.get("finalize") is True:
        next_collector = collector_text
        if text:
            next_collector = text if not collector_text else f"{collector_text} {text}"
        return (
            TextEvent(
                kind="final",
                text=text,
                collector_text=next_collector,
                is_final=True,
                raw=data,
            ),
            next_collector,
        )

    return (
        TextEvent(
            kind="final_non_finalize",
            text=text,
            collector_text=collector_text,
            is_final=True,
            raw=data,
        ),
        collector_text,
    )


async def _wait_http_health(url: str, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_error: str | None = None
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                    text = await resp.text()
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        data = {"raw": text}
                    if resp.status == 200 and data.get("model_loaded") is True:
                        return data
                    last_error = f"status={resp.status} body={text[:200]}"
            except Exception as exc:  # noqa: BLE001 - readiness probe
                last_error = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(1.0)
    raise TimeoutError(f"server did not become healthy within {timeout_s}s; last={last_error}")


async def _capture_one(
    *,
    url: str,
    pcm: bytes,
    chunk_bytes: int,
    final_timeout_s: float,
    close_timeout_s: float,
    realtime: bool,
    chunk_ms: int,
) -> tuple[list[TextEvent], list[dict[str, Any]]]:
    events: list[TextEvent] = []
    raw_messages: list[dict[str, Any]] = []
    collector_text = ""
    final_seen = asyncio.Event()

    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url, max_msg_size=10 * 1024 * 1024) as ws:
            ready = await ws.receive(timeout=final_timeout_s)
            if ready.type != aiohttp.WSMsgType.TEXT:
                raise RuntimeError(f"expected ready text frame, got {ready.type}")
            ready_payload = json.loads(ready.data)
            raw_messages.append(ready_payload)
            if ready_payload.get("type") != "ready":
                raise RuntimeError(f"expected ready payload, got {ready_payload!r}")

            async def recv_loop() -> None:
                nonlocal collector_text
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        raw_messages.append(data)
                        if data.get("type") == "transcript":
                            event, collector_text = _normalize_server_transcript(
                                data,
                                collector_text,
                            )
                            if event is not None:
                                events.append(event)
                                if event.kind == "final":
                                    final_seen.set()
                        elif data.get("type") == "error":
                            final_seen.set()
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break

            recv_task = asyncio.create_task(recv_loop())
            try:
                await ws.send_json({"type": "vad_start"})
                sleep_s = chunk_ms / 1000.0
                for offset in range(0, len(pcm), chunk_bytes):
                    await ws.send_bytes(pcm[offset : offset + chunk_bytes])
                    if realtime:
                        await asyncio.sleep(sleep_s)
                await ws.send_json({"type": "vad_stop"})
                await asyncio.wait_for(final_seen.wait(), timeout=final_timeout_s)
            finally:
                await ws.close()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(recv_task, timeout=close_timeout_s)
                if not recv_task.done():
                    recv_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await recv_task

    return events, raw_messages


def _server_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["NEMOTRON_CONTINUOUS"] = "1"
    env["NEMOTRON_FINALIZE_SILENCE_MS"] = str(args.finalize_silence_ms)
    for name in (
        "NEMOTRON_SCHEDULER_B1",
        "NEMOTRON_BATCH_SCHED",
        "NEMOTRON_BATCH_FINALIZE",
        "NEMOTRON_BATCH_FINALIZE_PREPROC",
        "NEMOTRON_FINALIZE_PRIORITY",
    ):
        env.pop(name, None)
    return env


async def _run(args: argparse.Namespace) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"step6_server_oracle_{stamp}"
    server_log_path = LOG_DIR / f"{run_name}.server.log"
    events_path = LOG_DIR / f"{run_name}.events.jsonl"
    summary_path = LOG_DIR / f"{run_name}.summary.json"

    if not args.bundle.exists():
        raise FileNotFoundError(f"bundle not found: {args.bundle}")

    bundle = torch.jit.load(str(args.bundle), map_location="cpu")
    rows = _scalar_i64(bundle, "num_utts")
    if args.start < 0 or args.start >= rows:
        raise ValueError(f"--start {args.start} outside bundle rows={rows}")
    n = min(args.n, rows - args.start)

    process: subprocess.Popen[bytes] | None = None
    server_started_here = False
    server_start_error: str | None = None
    health: dict[str, Any] | None = None
    url = f"ws://{args.host}:{args.port}"
    health_url = f"http://{args.host}:{args.port}/health"

    if args.start_server:
        python = args.python
        cmd = [
            python,
            str(SERVER),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model",
            args.model,
        ]
        if args.right_context is not None:
            cmd.extend(["--right-context", str(args.right_context)])
        server_log = server_log_path.open("wb")
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env=_server_env(args),
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            server_started_here = True
            health = await _wait_http_health(health_url, args.server_start_timeout_s)
        except Exception as exc:  # noqa: BLE001 - reported as feasibility
            server_start_error = f"{type(exc).__name__}: {exc}"
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=10)
            process = None
        finally:
            server_log.close()
    else:
        try:
            health = await _wait_http_health(health_url, args.health_timeout_s)
        except Exception as exc:  # noqa: BLE001 - reported as feasibility
            server_start_error = f"{type(exc).__name__}: {exc}"

    results: list[dict[str, Any]] = []
    feasible = server_start_error is None
    try:
        if feasible:
            for offset in range(n):
                utt = args.start + offset
                prefix = f"utt{utt}"
                sample_index = _scalar_i64(bundle, f"{prefix}_sample_index")
                sample_id = _one_utf8(bundle, prefix, "sample_id")
                audio = _tensor_f32(bundle, f"{prefix}_audio")
                pcm = _float_audio_to_pcm16(audio)
                gold_all = _bundle_events(bundle, utt)
                gold = _wire_gold(gold_all)

                started = time.time()
                server_events, raw_messages = await _capture_one(
                    url=url,
                    pcm=pcm,
                    chunk_bytes=args.chunk_bytes,
                    final_timeout_s=args.final_timeout_s,
                    close_timeout_s=args.close_timeout_s,
                    realtime=args.realtime,
                    chunk_ms=args.chunk_ms,
                )
                elapsed = time.time() - started
                comparison = _compare_events(server_events, gold)
                suppressed_gold = sum(1 for event in gold_all if event.kind == "suppressed")
                final_reasons = [
                    event.raw.get("finalize_timing", {}).get("reason")
                    for event in server_events
                    if event.kind == "final" and event.raw
                ]
                row = {
                    "utt": utt,
                    "sample_index": sample_index,
                    "sample_id": sample_id,
                    "audio_samples": int(audio.shape[0]),
                    "elapsed_s": elapsed,
                    "suppressed_internal_gold_events": suppressed_gold,
                    "server_finalize_reasons": final_reasons,
                    **comparison,
                    "gold_events": [_event_payload(event) for event in gold],
                    "server_events": [_event_payload(event) for event in server_events],
                    "raw_server_messages": raw_messages,
                }
                results.append(row)
                with events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                status = "MATCH" if comparison["event_match"] else (
                    "FINAL_MATCH" if comparison["final_match"] else "DIFF"
                )
                print(
                    f"{status} utt={utt} sample={sample_index} id={sample_id} "
                    f"events={comparison['event_count_server']}/{comparison['event_count_gold']} "
                    f"final={comparison['final_match']} elapsed={elapsed:.2f}s",
                    flush=True,
                )
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=20)

    event_matches = sum(1 for row in results if row.get("event_match"))
    final_matches = sum(1 for row in results if row.get("final_match"))
    event_diffs = [
        {
            "utt": row["utt"],
            "sample_index": row["sample_index"],
            "sample_id": row["sample_id"],
            "event_count_server": row["event_count_server"],
            "event_count_gold": row["event_count_gold"],
            "final_match": row["final_match"],
            "first_diff": row["first_diff"],
        }
        for row in results
        if not row.get("event_match")
    ]
    final_diffs = [
        {
            "utt": row["utt"],
            "sample_index": row["sample_index"],
            "sample_id": row["sample_id"],
            "server_final_text": row["server_final_text"],
            "gold_final_text": row["gold_final_text"],
        }
        for row in results
        if not row.get("final_match")
    ]
    finalize_reasons = sorted(
        {
            reason
            for row in results
            for reason in row.get("server_finalize_reasons", [])
            if reason is not None
        }
    )
    summary = {
        "run_name": run_name,
        "timestamp_utc": stamp,
        "bundle": args.bundle,
        "server_py": SERVER,
        "server_started_here": server_started_here,
        "server_runs_headless": feasible,
        "server_start_error": server_start_error,
        "health": health,
        "url": url,
        "env_control": {
            "NEMOTRON_CONTINUOUS": "1",
            "NEMOTRON_FINALIZE_SILENCE_MS": args.finalize_silence_ms,
            "explicit_vad_stop": True,
            "sent_reset_or_end": False,
            "chunk_bytes": args.chunk_bytes,
            "chunk_ms": args.chunk_ms,
            "realtime_pacing": args.realtime,
            "pcm_wire": "int16 little-endian, stt-benchmark-compatible float*32767 quantization",
        },
        "subset": {
            "start": args.start,
            "requested_n": args.n,
            "attempted_n": n if feasible else 0,
            "completed_n": len(results),
        },
        "comparison": {
            "mode": "event-for-event transcript JSON vs C++/bundle text-event oracle",
            "gold_suppressed_events_dropped_for_wire_compare": True,
            "event_matches": event_matches,
            "final_matches": final_matches,
            "event_divergences": len(results) - event_matches,
            "final_divergences": len(results) - final_matches,
            "finalize_reasons": finalize_reasons,
            "event_diffs": event_diffs,
            "final_diffs": final_diffs,
        },
        "artifacts": {
            "summary": summary_path,
            "events": events_path,
            "server_log": server_log_path,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    print(f"wrote summary: {summary_path}")
    print(f"wrote events: {events_path}")
    if args.start_server:
        print(f"wrote server log: {server_log_path}")
    if not feasible:
        print(f"SERVER_FEASIBILITY_BLOCKER {server_start_error}")
        return 2
    if len(results) != n:
        return 3
    return 0 if event_matches == len(results) else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--start-server", action="store_true")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--model", default="nvidia/nemotron-speech-streaming-en-0.6b")
    parser.add_argument("--right-context", type=int, default=1)
    parser.add_argument("--finalize-silence-ms", type=int, default=0)
    parser.add_argument("--server-start-timeout-s", type=float, default=240.0)
    parser.add_argument("--health-timeout-s", type=float, default=5.0)
    parser.add_argument("--final-timeout-s", type=float, default=45.0)
    parser.add_argument("--close-timeout-s", type=float, default=5.0)
    parser.add_argument("--chunk-ms", type=int, default=20)
    parser.add_argument("--chunk-bytes", type=int, default=640)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
