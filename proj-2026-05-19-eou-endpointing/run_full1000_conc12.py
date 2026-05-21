#!/usr/bin/env python3
"""Full-1000 authoritative benchmark of the production config, run at concurrency 12.

GOAL: the last measurement to close out this phase. Run all 1000 dataset samples
through the production server config (NEMOTRON_FINALIZE_SILENCE_MS=0 +
NEMOTRON_WARMUP_MS=200 + continuous), at a sustained concurrency of 12 live
realtime sessions (the comfortably-in-budget capacity established by
concurrency_test.py). Write transcripts + TTFB to the MAIN results.db (which
holds ground truth for all 1000), so `stt-benchmark wer` (no --test) can then
score semantic WER against ground truth.

WHY concurrency 12: (a) it finishes in ~20 min instead of ~3.3 h sequential;
(b) it measures the production config under the realistic concurrent load we
validated as in-budget. The concurrency test already proved byte-exact
correctness under load, so the transcripts here equal the sequential ones --
this run is both faster AND a realistic-load confirmation.

TTFB recorded = (final transcript received) - (end of real speech audio), i.e.
it INCLUDES the ~200 ms trailing-silence/Silero-detection window + the fork-flush,
matching the benchmark observer's speech-end->final definition (so it's
comparable to the recorded fork=365ms / silence_0=213ms numbers).

Run with the benchmark venv (has websockets):
  stt-benchmark/.venv/bin/python proj-2026-05-19-eou-endpointing/run_full1000_conc12.py --model-tag silence0_warm200_c12

Requires a server already running with:
  NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 NEMOTRON_CONTINUOUS=1
  --right-context 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2
TRAILING_SILENCE_MS = 200
DEFAULT_URL = "ws://127.0.0.1:8080"
MAIN_DB = REPO_ROOT / "stt-benchmark" / "stt_benchmark_data" / "results.db"


def build_connect_url(base_url: str, language: str | None = None, model: str | None = None) -> str:
    if not language and not model:
        return base_url

    split = urlsplit(base_url)
    query = [
        item
        for item in parse_qsl(split.query, keep_blank_values=True)
        if item[0] not in {"language", "model"}
    ]
    if language:
        query.append(("language", language))
    if model:
        query.append(("model", model))
    return urlunsplit(
        (
            split.scheme,
            split.netloc,
            split.path,
            urlencode(query),
            split.fragment,
        )
    )


def load_samples(db_path: Path) -> list[dict]:
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT sample_id, audio_path, duration_seconds FROM samples ORDER BY dataset_index"
    ).fetchall()
    con.close()
    return [{"sample_id": r[0], "audio_path": r[1], "duration_seconds": r[2] or 0.0} for r in rows]


def load_pcm(audio_path: str) -> bytes:
    p = REPO_ROOT / "stt-benchmark" / audio_path
    if not p.exists():
        p = REPO_ROOT / audio_path
    return p.read_bytes()


async def run_session(url: str, audio: dict) -> dict:
    res = {
        "sample_id": audio["sample_id"],
        "audio_duration_seconds": audio["duration_seconds"],
        "transcription": "",
        "ttfb_seconds": None,       # final - real-speech-end (benchmark-comparable)
        "server_ttfs_ms": None,     # final - vad_stop (server-side finalize latency)
        "error": None,
    }
    try:
        pcm = load_pcm(audio["audio_path"])
    except Exception as e:  # noqa: BLE001
        res["error"] = f"load_pcm: {e}"
        return res

    final_parts: list[str] = []
    final_event = asyncio.Event()
    t_real_audio_end = 0.0
    t_vad_stop = 0.0

    try:
        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=30.0)  # ready
            except Exception:  # noqa: BLE001
                pass

            async def receiver():
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        data = json.loads(raw)
                        if data.get("type") != "transcript":
                            continue
                        if data.get("is_final") and data.get("finalize"):
                            txt = data.get("text", "")
                            if txt:
                                final_parts.append(txt)
                            now = time.monotonic()
                            res["ttfb_seconds"] = now - t_real_audio_end
                            res["server_ttfs_ms"] = (now - t_vad_stop) * 1000.0
                            final_event.set()
                except Exception:  # noqa: BLE001
                    pass

            recv_task = asyncio.create_task(receiver())
            await ws.send(json.dumps({"type": "vad_start"}))

            t0 = time.monotonic()
            sent = 0
            idx = 0
            while sent < len(pcm):
                chunk = pcm[sent : sent + CHUNK_BYTES]
                await ws.send(chunk)
                sent += len(chunk)
                idx += 1
                target = t0 + idx * (CHUNK_MS / 1000.0)
                dt = target - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)
            t_real_audio_end = time.monotonic()

            # trailing silence (Silero stop_secs window), still realtime-paced
            trailing = bytes(CHUNK_BYTES)
            sil_chunks = int(TRAILING_SILENCE_MS / CHUNK_MS)
            for _ in range(sil_chunks):
                await ws.send(trailing)
                idx += 1
                target = t0 + idx * (CHUNK_MS / 1000.0)
                dt = target - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)

            t_vad_stop = time.monotonic()
            await ws.send(json.dumps({"type": "vad_stop"}))
            await ws.send(json.dumps({"type": "reset", "finalize": True}))

            try:
                await asyncio.wait_for(final_event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                res["error"] = "timeout"

            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        res["transcription"] = " ".join(final_parts).strip()
    except Exception as e:  # noqa: BLE001
        res["error"] = f"session: {e}"
    return res


async def bounded(sem: asyncio.Semaphore, url: str, audio: dict, progress: dict) -> dict:
    async with sem:
        r = await run_session(url, audio)
        progress["done"] += 1
        if progress["done"] % 50 == 0:
            print(f"  ... {progress['done']}/{progress['total']} done", flush=True)
        return r


def write_results(db_path: Path, service: str, model_tag: str, results: list[dict]) -> int:
    con = sqlite3.connect(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    n = 0
    for r in results:
        con.execute(
            """INSERT OR REPLACE INTO results
               (sample_id, service_name, model_name, ttfb_seconds, transcription,
                audio_duration_seconds, timestamp, error)
               VALUES (?,?,?,?,?,?,?,?)""",
            (r["sample_id"], service, model_tag,
             r["ttfb_seconds"], r["transcription"],
             r["audio_duration_seconds"], ts, r["error"]),
        )
        n += 1
    con.commit()
    con.close()
    return n


async def main_async(args):
    samples = load_samples(MAIN_DB)
    if args.limit:
        samples = samples[: args.limit]
    connect_url = build_connect_url(args.url, args.language, args.model)
    print(f"Running {len(samples)} samples at concurrency {args.concurrency} "
          f"(model_tag={args.model_tag})\n", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0, "total": len(samples)}
    t_start = time.monotonic()
    results = await asyncio.gather(*[bounded(sem, connect_url, s, progress) for s in samples])
    elapsed = time.monotonic() - t_start

    errs = [r for r in results if r["error"]]
    ok = [r for r in results if not r["error"]]
    ttfbs = [r["ttfb_seconds"] * 1000 for r in ok if r["ttfb_seconds"] is not None]
    stts = [r["server_ttfs_ms"] for r in ok if r["server_ttfs_ms"] is not None]

    print(f"\nCompleted in {elapsed:.0f}s. ok={len(ok)} errors={len(errs)}")
    if errs:
        for r in errs[:10]:
            print(f"  ERROR [{r['sample_id'][:8]}]: {r['error']}")
    if ttfbs:
        ttfbs.sort()
        p = lambda q: ttfbs[min(len(ttfbs) - 1, int(q * (len(ttfbs) - 1)))]
        print(f"TTFB (speech-end->final, ms): mean={statistics.mean(ttfbs):.1f} "
              f"p50={p(0.5):.1f} p95={p(0.95):.1f} p99={p(0.99):.1f}")
    if stts:
        stts.sort()
        p = lambda q: stts[min(len(stts) - 1, int(q * (len(stts) - 1)))]
        print(f"server finalize (vad_stop->final, ms): p50={p(0.5):.1f} p95={p(0.95):.1f}")

    written = write_results(MAIN_DB, args.service, args.model_tag, results)
    print(f"\nWrote {written} rows to {MAIN_DB} (service={args.service}, model={args.model_tag})")
    print(f"Next: stt-benchmark wer --services {args.service} --model {args.model_tag}")

    # also dump a JSON artifact
    out = REPO_ROOT / "proj-2026-05-19-eou-endpointing" / f"full1000_{args.model_tag}_results.json"
    out.write_text(json.dumps(
        {"config": {"model_tag": args.model_tag, "concurrency": args.concurrency,
                    "n": len(samples), "elapsed_s": elapsed, "language": args.language,
                    "model": args.model},
         "results": results}, indent=2))
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--service", default="nemotron_local")
    ap.add_argument("--model-tag", default="silence0_warm200_c12")
    ap.add_argument("--language", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
