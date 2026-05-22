#!/usr/bin/env python3
"""Phase-aligned local load generator for Nemotron realtime ASR sweeps.

This driver is intentionally local-only. It opens all WebSocket sessions first,
waits until they have all received the server `ready` handshake, then sends
160 ms PCM chunks from one shared monotonic clock. The goal is to remove arrival
phase jitter while keeping 1x realtime pacing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import websockets


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "stt-benchmark" / "stt_benchmark_data" / "test_results.db"
DEFAULT_URL = "ws://127.0.0.1:8080"
SAMPLE_RATE = 16000
TRAILING_SILENCE_MS = 200


def pct(values: list[float | None], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]


def fmt_ms(v: float | None) -> str:
    return "nan" if v is None else f"{v:.1f}"


@dataclass
class Stream:
    stream_id: str
    sample_id: str
    audio_path: str
    duration_seconds: float
    pcm: bytes
    stream: bytes
    expected_stream_s: float
    ws: Any = None
    receiver_task: asyncio.Task | None = None
    final_event: asyncio.Event = field(default_factory=asyncio.Event)
    transcript: str = ""
    interims: list[str] = field(default_factory=list)
    final_deltas: list[str] = field(default_factory=list)
    interim_count: int = 0
    ready_latency_ms: float | None = None
    first_interim_ms: float | None = None
    ttfs_ms: float | None = None
    processing_lag_ms: float | None = None
    stream_overrun_ms: float | None = None
    chunk_late_ms: list[float] = field(default_factory=list)
    vad_start_t: float = 0.0
    vad_stop_t: float = 0.0
    last_audio_sent_t: float = 0.0
    error: str | None = None
    tag_leak: bool = False


def select_specs(db_path: Path, count: int, min_duration_s: float) -> list[dict[str, Any]]:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT sample_id, audio_path, duration_seconds FROM samples "
        "WHERE duration_seconds IS NOT NULL AND duration_seconds >= ? "
        "ORDER BY duration_seconds, sample_id",
        (min_duration_s,),
    ).fetchall()
    if not rows:
        rows = con.execute(
            "SELECT sample_id, audio_path, duration_seconds FROM samples "
            "WHERE duration_seconds IS NOT NULL ORDER BY duration_seconds, sample_id"
        ).fetchall()
    con.close()
    if not rows:
        raise SystemExit(f"no samples found in {db_path}")

    specs = []
    # Repeat the duration-stratified pool if N exceeds the 100-sample benchmark DB.
    for i in range(count):
        if count <= len(rows):
            idx = round(i * (len(rows) - 1) / max(1, count - 1))
        else:
            idx = i % len(rows)
        sid, audio_path, dur = rows[idx]
        specs.append(
            {
                "stream_id": f"{sid}:{i:04d}",
                "sample_id": sid,
                "audio_path": audio_path,
                "duration_seconds": float(dur),
            }
        )
    return specs


def load_pcm(audio_path: str) -> bytes:
    path = REPO_ROOT / "stt-benchmark" / audio_path
    if not path.exists():
        path = REPO_ROOT / audio_path
    return path.read_bytes()


def build_stream(pcm: bytes, chunk_bytes: int) -> tuple[bytes, float]:
    trailing = bytes(int(SAMPLE_RATE * TRAILING_SILENCE_MS / 1000) * 2)
    stream = pcm + trailing
    if len(stream) % chunk_bytes:
        stream += bytes(chunk_bytes - (len(stream) % chunk_bytes))
    return stream, len(stream) / 2 / SAMPLE_RATE


def load_streams(db_path: Path, count: int, min_duration_s: float, chunk_ms: int) -> list[Stream]:
    chunk_bytes = int(SAMPLE_RATE * chunk_ms / 1000) * 2
    streams = []
    for spec in select_specs(db_path, count, min_duration_s):
        pcm = load_pcm(spec["audio_path"])
        stream, expected_s = build_stream(pcm, chunk_bytes)
        streams.append(
            Stream(
                stream_id=spec["stream_id"],
                sample_id=spec["sample_id"],
                audio_path=spec["audio_path"],
                duration_seconds=spec["duration_seconds"],
                pcm=pcm,
                stream=stream,
                expected_stream_s=expected_s,
            )
        )
    return streams


def clone_stream(s: Stream) -> Stream:
    return Stream(
        stream_id=s.stream_id,
        sample_id=s.sample_id,
        audio_path=s.audio_path,
        duration_seconds=s.duration_seconds,
        pcm=s.pcm,
        stream=s.stream,
        expected_stream_s=s.expected_stream_s,
    )


async def connect_stream(url: str, stream: Stream) -> None:
    try:
        t0 = time.monotonic()
        stream.ws = await websockets.connect(
            url,
            max_size=16 * 1024 * 1024,
            open_timeout=120,
            ping_interval=None,
        )
        try:
            raw = await asyncio.wait_for(stream.ws.recv(), timeout=120)
            if isinstance(raw, str):
                data = json.loads(raw)
                if data.get("type") != "ready":
                    stream.error = f"unexpected first message: {data.get('type')}"
        except Exception as e:  # noqa: BLE001
            stream.error = f"ready: {e}"
        stream.ready_latency_ms = (time.monotonic() - t0) * 1000.0
        stream.receiver_task = asyncio.create_task(receive_loop(stream))
    except Exception as e:  # noqa: BLE001
        stream.error = f"connect: {e}"


async def receive_loop(stream: Stream) -> None:
    final_parts: list[str] = []
    try:
        async for raw in stream.ws:
            if isinstance(raw, bytes):
                continue
            data = json.loads(raw)
            if data.get("type") != "transcript":
                continue
            now = time.monotonic()
            text = data.get("text", "")
            if "<|" in text or "|>" in text:
                stream.tag_leak = True
            if data.get("is_final") and data.get("finalize"):
                stream.final_deltas.append(text)
                if text:
                    final_parts.append(text)
                stream.transcript = " ".join(final_parts).strip()
                stream.ttfs_ms = (now - stream.vad_stop_t) * 1000.0
                stream.processing_lag_ms = (now - stream.last_audio_sent_t) * 1000.0
                stream.final_event.set()
            else:
                stream.interims.append(text)
                stream.interim_count += 1
                if stream.first_interim_ms is None and stream.vad_start_t > 0:
                    stream.first_interim_ms = (now - stream.vad_start_t) * 1000.0
    except asyncio.CancelledError:
        raise
    except Exception:
        # The main task owns timeout/error reporting. Receiver close races are
        # expected when the session is closed after final receipt.
        pass


async def close_stream(stream: Stream) -> None:
    if stream.receiver_task is not None:
        stream.receiver_task.cancel()
        try:
            await stream.receiver_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
    if stream.ws is not None:
        try:
            await stream.ws.close()
        except Exception:  # noqa: BLE001
            pass


async def run_level(
    url: str,
    streams: list[Stream],
    n: int,
    *,
    chunk_ms: int,
    vad_lead_ms: int,
    start_delay_ms: int,
    final_timeout_s: float,
    include_interims: bool,
) -> dict[str, Any]:
    selected = [clone_stream(s) for s in streams[:n]]
    chunk_bytes = int(SAMPLE_RATE * chunk_ms / 1000) * 2
    print(f"=== N={n} in-phase: connecting {n} sessions ===", flush=True)
    level_start_epoch = time.time()
    level_start_monotonic = time.monotonic()
    await asyncio.gather(*(connect_stream(url, s) for s in selected))
    connected = [s for s in selected if s.ws is not None and s.error is None]
    print(f"    ready {len(connected)}/{n}", flush=True)

    vad_start_at = time.monotonic() + start_delay_ms / 1000.0
    audio_start_at = vad_start_at + vad_lead_ms / 1000.0

    async def sleep_until(t: float) -> None:
        delay = t - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    await sleep_until(vad_start_at)
    vad_send_start = time.monotonic()
    await asyncio.gather(
        *(s.ws.send(json.dumps({"type": "vad_start"})) for s in connected),
        return_exceptions=True,
    )
    vad_send_done = time.monotonic()
    for s in connected:
        s.vad_start_t = vad_send_done

    chunk_counts = {
        s.stream_id: math.ceil(len(s.stream) / chunk_bytes)
        for s in connected
    }
    max_chunks = max(chunk_counts.values(), default=0)
    print(
        f"    audio_start={audio_start_at:.6f} chunks={max_chunks} "
        f"chunk_ms={chunk_ms} vad_burst_ms={(vad_send_done - vad_send_start) * 1000.0:.1f}",
        flush=True,
    )
    await sleep_until(audio_start_at)
    audio_start_epoch = time.time()
    audio_start_mono = time.monotonic()

    stopped: set[str] = set()

    async def stop_one(s: Stream, stop_t: float) -> None:
        if s.stream_id in stopped or s.error is not None or s.ws is None:
            return
        stopped.add(s.stream_id)
        s.stream_overrun_ms = (
            stop_t - audio_start_mono - s.expected_stream_s
        ) * 1000.0
        s.vad_stop_t = time.monotonic()
        try:
            await s.ws.send(json.dumps({"type": "vad_stop"}))
            await s.ws.send(json.dumps({"type": "reset", "finalize": True}))
        except Exception as e:  # noqa: BLE001
            s.error = s.error or f"stop: {e}"

    for chunk_i in range(max_chunks + 1):
        target = audio_start_at + chunk_i * (chunk_ms / 1000.0)
        await sleep_until(target)

        due_to_stop = [
            s for s in connected
            if chunk_counts.get(s.stream_id, 0) == chunk_i and s.stream_id not in stopped
        ]
        if due_to_stop:
            stop_t = time.monotonic()
            await asyncio.gather(
                *(stop_one(s, stop_t) for s in due_to_stop),
                return_exceptions=True,
            )
        if chunk_i >= max_chunks:
            break

        sends = []
        send_streams = []
        for s in connected:
            offset = chunk_i * chunk_bytes
            if offset >= len(s.stream):
                continue
            send_streams.append(s)
            sends.append(s.ws.send(s.stream[offset : offset + chunk_bytes]))
        if not sends:
            continue
        results = await asyncio.gather(*sends, return_exceptions=True)
        sent_at = time.monotonic()
        late_ms = max(0.0, (sent_at - target) * 1000.0)
        for s, result in zip(send_streams, results, strict=True):
            if isinstance(result, Exception):
                s.error = s.error or f"send: {result}"
                continue
            s.last_audio_sent_t = sent_at
            s.chunk_late_ms.append(late_ms)

    audio_done_mono = time.monotonic()

    async def wait_final(s: Stream) -> None:
        if s.error is not None:
            return
        try:
            await asyncio.wait_for(s.final_event.wait(), timeout=final_timeout_s)
        except asyncio.TimeoutError:
            s.error = "timeout waiting for final transcript"

    await asyncio.gather(*(wait_final(s) for s in connected))
    await asyncio.gather(*(close_stream(s) for s in connected), return_exceptions=True)
    level_end_epoch = time.time()

    summary = summarize_level(selected, chunk_ms)
    summary.update(
        {
            "n": n,
            "started_epoch": level_start_epoch,
            "audio_start_epoch": audio_start_epoch,
            "ended_epoch": level_end_epoch,
            "started_at": datetime.fromtimestamp(level_start_epoch).isoformat(),
            "audio_start_at": datetime.fromtimestamp(audio_start_epoch).isoformat(),
            "ended_at": datetime.fromtimestamp(level_end_epoch).isoformat(),
            "level_wall_s": time.monotonic() - level_start_monotonic,
            "audio_send_wall_s": audio_done_mono - audio_start_mono,
            "vad_lead_ms": vad_lead_ms,
            "start_delay_ms": start_delay_ms,
        }
    )
    return {
        "summary": summary,
        "results": [
            stream_json(s, include_interims=include_interims)
            for s in selected
        ],
    }


def summarize_level(streams: list[Stream], chunk_ms: int) -> dict[str, Any]:
    ok = [s for s in streams if s.error is None and s.ttfs_ms is not None]
    errors = [s for s in streams if s.error is not None]
    ttfs = [s.ttfs_ms for s in ok]
    lags = [s.processing_lag_ms for s in ok]
    ready = [s.ready_latency_ms for s in streams]
    over = [s.stream_overrun_ms for s in streams]
    chunk_late_all = [v for s in streams for v in s.chunk_late_ms]
    chunk_late_max_per_stream = [max(s.chunk_late_ms) for s in streams if s.chunk_late_ms]
    violations = sum(1 for v in chunk_late_all if v > chunk_ms)
    keep_up = (
        len(ok) == len(streams)
        and (pct(lags, 0.95) or float("inf")) < 500.0
        and (pct(ttfs, 0.95) or float("inf")) < 500.0
    )
    strict_keep_up = (
        len(ok) == len(streams)
        and (pct(lags, 0.95) or float("inf")) < 500.0
        and (pct(ttfs, 0.95) or float("inf")) < 400.0
    )
    return {
        "ok": len(ok),
        "errors": len(errors),
        "error_samples": [
            {"stream_id": s.stream_id, "sample_id": s.sample_id, "error": s.error}
            for s in errors[:10]
        ],
        "tag_leaks": sum(1 for s in streams if s.tag_leak),
        "keep_up_500ms": keep_up,
        "strict_keep_up": strict_keep_up,
        "ttfs_p50_ms": pct(ttfs, 0.50),
        "ttfs_p95_ms": pct(ttfs, 0.95),
        "processing_lag_p50_ms": pct(lags, 0.50),
        "processing_lag_p95_ms": pct(lags, 0.95),
        "ready_p95_ms": pct(ready, 0.95),
        "send_overrun_p50_ms": pct(over, 0.50),
        "send_overrun_p95_ms": pct(over, 0.95),
        "send_overrun_max_ms": max((v for v in over if v is not None), default=None),
        "chunk_late_p95_ms": pct(chunk_late_all, 0.95),
        "chunk_late_max_ms": max(chunk_late_all) if chunk_late_all else None,
        "chunk_late_stream_max_p95_ms": pct(chunk_late_max_per_stream, 0.95),
        "pacing_violations": violations,
    }


def stream_json(s: Stream, *, include_interims: bool = False) -> dict[str, Any]:
    record = {
        "stream_id": s.stream_id,
        "sample_id": s.sample_id,
        "audio_path": s.audio_path,
        "duration_seconds": s.duration_seconds,
        "expected_stream_s": s.expected_stream_s,
        "transcript": s.transcript[:500],
        "final_deltas": s.final_deltas,
        "interim_count": s.interim_count,
        "ready_latency_ms": s.ready_latency_ms,
        "first_interim_ms": s.first_interim_ms,
        "ttfs_ms": s.ttfs_ms,
        "processing_lag_ms": s.processing_lag_ms,
        "send_overrun_ms": s.stream_overrun_ms,
        "chunk_late_max_ms": max(s.chunk_late_ms) if s.chunk_late_ms else None,
        "chunk_late_p95_ms": pct(s.chunk_late_ms, 0.95),
        "error": s.error,
        "tag_leak": s.tag_leak,
    }
    if include_interims:
        record["interims"] = s.interims
    return record


async def main_async(args: argparse.Namespace) -> None:
    levels = [int(x) for x in args.sweep.split(",") if x.strip()]
    stream_count = max(levels) if levels else 1
    streams = load_streams(Path(args.db), stream_count, args.min_duration, args.chunk_ms)
    print(
        f"loaded {len(streams)} streams from {args.db}; "
        f"duration range {min(s.duration_seconds for s in streams):.1f}-"
        f"{max(s.duration_seconds for s in streams):.1f}s; "
        f"levels={levels}",
        flush=True,
    )
    output: dict[str, Any] = {
        "config": {
            "url": args.url,
            "db": args.db,
            "levels": levels,
            "chunk_ms": args.chunk_ms,
            "trailing_silence_ms": TRAILING_SILENCE_MS,
            "min_duration": args.min_duration,
            "vad_lead_ms": args.vad_lead_ms,
            "start_delay_ms": args.start_delay_ms,
            "final_timeout_s": args.final_timeout_s,
        },
        "summaries": {},
        "levels": {},
        "started_epoch": time.time(),
    }
    for n in levels:
        level = await run_level(
            args.url,
            streams,
            n,
            chunk_ms=args.chunk_ms,
            vad_lead_ms=args.vad_lead_ms,
            start_delay_ms=args.start_delay_ms,
            final_timeout_s=args.final_timeout_s,
            include_interims=args.include_interims,
        )
        output["summaries"][str(n)] = level["summary"]
        output["levels"][str(n)] = level["results"]
        s = level["summary"]
        print(
            f"  N={n:<3} ok={s['ok']}/{n} strict={s['strict_keep_up']} "
            f"TTFS95={fmt_ms(s['ttfs_p95_ms'])} lag95={fmt_ms(s['processing_lag_p95_ms'])} "
            f"send_overrun95/max={fmt_ms(s['send_overrun_p95_ms'])}/"
            f"{fmt_ms(s['send_overrun_max_ms'])} "
            f"chunk_late95/max={fmt_ms(s['chunk_late_p95_ms'])}/"
            f"{fmt_ms(s['chunk_late_max_ms'])} violations={s['pacing_violations']}",
            flush=True,
        )
        if args.pause_s > 0:
            await asyncio.sleep(args.pause_s)
    output["ended_epoch"] = time.time()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"wrote {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--sweep", default="60,100,150,180,200")
    ap.add_argument("--chunk-ms", type=int, default=160)
    ap.add_argument("--min-duration", type=float, default=10.0)
    ap.add_argument("--vad-lead-ms", type=int, default=250)
    ap.add_argument("--start-delay-ms", type=int, default=1000)
    ap.add_argument("--final-timeout-s", type=float, default=60.0)
    ap.add_argument("--pause-s", type=float, default=1.0)
    ap.add_argument("--include-interims", action="store_true")
    ap.add_argument(
        "--output",
        default=str(REPO_ROOT / "proj-2026-05-21-0410" / "inphase-loadgen-results.json"),
    )
    args = ap.parse_args()
    if args.chunk_ms <= 0:
        raise SystemExit("--chunk-ms must be > 0")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
