#!/usr/bin/env python3
"""Single-WebSocket-connection demo: end-to-end inference through the LB.

Independent of the bench framework. Opens ONE WS to the LB, streams one
short PCM sample paced at realtime, sends vad_stop, prints the final
transcript + measured server-side TTFS.

Usage:
    PY=stt-benchmark/.venv/bin/python
    $PY proj-2026-05-27-l40s-cluster-deploy/single_ws_demo.py \\
        ws://18.237.212.198:8080 \\
        stt-benchmark/stt_benchmark_data/audio/c84503ca-110a-51d6-0b24-f17deb7a2882.pcm
"""

import asyncio
import json
import sys
import time
import websockets


SAMPLE_RATE = 16000        # server expects 16kHz mono 16-bit signed LE PCM
CHUNK_MS = 20
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2   # 640 bytes / 20ms
TRAILING_SILENCE_MS = 200  # Silero stop_secs window


async def main(url: str, pcm_path: str) -> int:
    pcm = open(pcm_path, "rb").read()
    audio_sec = len(pcm) / (SAMPLE_RATE * 2)
    print(f"  audio: {pcm_path}  {len(pcm)} bytes  {audio_sec:.3f}s")
    print(f"  URL  : {url}")
    print()

    final_event = asyncio.Event()
    final_text_parts: list[str] = []
    partial_count = 0
    t_vad_stop_perf = 0.0
    server_ttfs_ms = None

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        print(f"  [connect] OK  remote={ws.remote_address}")

        async def receiver() -> None:
            nonlocal partial_count, server_ttfs_ms
            async for msg in ws:
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                t = data.get("type")
                if t != "transcript":
                    continue
                if not data.get("is_final"):
                    partial_count += 1
                    if partial_count <= 3 or partial_count % 5 == 0:
                        print(f"  [partial #{partial_count}] {data.get('text','')!r}")
                else:
                    server_ttfs_ms = (time.perf_counter() - t_vad_stop_perf) * 1000.0
                    final_text_parts.append(data.get("text", ""))
                    print(f"  [FINAL    ]  {data.get('text','')!r}")
                    final_event.set()

        recv_task = asyncio.create_task(receiver())
        await ws.send(json.dumps({"type": "vad_start"}))
        print("  [send] vad_start")

        # Realtime-paced PCM stream (640-byte / 20ms chunks)
        t0 = time.monotonic()
        sent = 0
        idx = 0
        while sent < len(pcm):
            chunk = pcm[sent : sent + CHUNK_BYTES]
            await ws.send(chunk)
            sent += len(chunk)
            idx += 1
            dt = (t0 + idx * (CHUNK_MS / 1000.0)) - time.monotonic()
            if dt > 0:
                await asyncio.sleep(dt)
        print(f"  [send] {sent} PCM bytes (realtime paced)")

        # Trailing silence (Silero stop_secs window), realtime paced
        trailing = bytes(CHUNK_BYTES)
        for _ in range(int(TRAILING_SILENCE_MS / CHUNK_MS)):
            await ws.send(trailing)
            idx += 1
            dt = (t0 + idx * (CHUNK_MS / 1000.0)) - time.monotonic()
            if dt > 0:
                await asyncio.sleep(dt)
        print(f"  [send] {TRAILING_SILENCE_MS}ms trailing silence")

        t_vad_stop_perf = time.perf_counter()
        await ws.send(json.dumps({"type": "vad_stop"}))
        await ws.send(json.dumps({"type": "reset", "finalize": True}))
        print("  [send] vad_stop + reset/finalize")

        try:
            await asyncio.wait_for(final_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            print("  [TIMEOUT] no final transcript in 10s")
            recv_task.cancel()
            return 2
        recv_task.cancel()

    print()
    print(f"  ==> partial messages : {partial_count}")
    print(f"  ==> final transcript : {' '.join(final_text_parts).strip()!r}")
    print(f"  ==> client-wall TTFS : {server_ttfs_ms:.1f} ms  (vad_stop -> final received, includes WAN RTT)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(asyncio.run(main(sys.argv[1], sys.argv[2])))
