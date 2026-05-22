"""In-region (us-east-1) co-located load generator for ASR knee sweeps.

This keeps the client in the same Modal region as the ASR app, so WAN latency and
the local laptop are not the bottleneck. It can run the original compact sweep
mode, plus high-N levels from the benchmark DB audio set and an optional cheap
strict-byte smoke.

Run:
  .venv/bin/modal run -m proj-2026-05-20-modal-cost.coloc_loadgen \
      --url wss://daily--nemotron-asr-bench-asr.modal.run \
      --sweep-str 4,8,16,24,40,60 --sample-count 60 \
      --output /tmp/modal-sweep.json
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sqlite3
import statistics
import time
from pathlib import Path

import modal

app = modal.App("asr-loadgen")
AUDIO_DIR = "/audio"
DB_PATH = "/data/test_results.db"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("websockets")
    .add_local_dir("stt-benchmark/stt_benchmark_data/audio", AUDIO_DIR)
    .add_local_file("stt-benchmark/stt_benchmark_data/test_results.db", DB_PATH)
)

SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # 20ms int16
TRAIL_MS = 200
START_JITTER_MS = 400


def _pct(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]


def _strict_record(res: dict) -> dict:
    return {
        "transcript": res["transcript"],
        "interims": res["interims"],
        "final_deltas": res["final_deltas"],
        "no_duplicate_final": len(res["final_deltas"]) <= 1,
    }


def _strict_diff(res: dict, baseline: dict) -> list[str]:
    diffs: list[str] = []
    rec = _strict_record(res)
    if rec["transcript"] != baseline.get("transcript"):
        diffs.append("final")
    if rec["final_deltas"] != baseline.get("final_deltas"):
        diffs.append("final_deltas")
    if rec["interims"] != baseline.get("interims"):
        diffs.append("interims")
    if not rec["no_duplicate_final"]:
        diffs.append("duplicate_final")
    return diffs


def _select_audio_specs(count: int) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT sample_id, audio_path, duration_seconds FROM samples "
        "WHERE duration_seconds IS NOT NULL ORDER BY duration_seconds"
    ).fetchall()
    con.close()
    if count > len(rows):
        raise ValueError(f"need {count} samples, db has {len(rows)}")
    if count <= 1:
        idxs = [0]
    else:
        idxs = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    specs = []
    for i in idxs:
        sid, audio_path, dur = rows[i]
        specs.append(
            {
                "sid": sid,
                "path": os.path.join(AUDIO_DIR, os.path.basename(audio_path)),
                "dur": float(dur),
            }
        )
    return specs


def _load_audios(count: int) -> list[dict]:
    audios = []
    for spec in _select_audio_specs(count):
        with open(spec["path"], "rb") as f:
            audios.append({**spec, "pcm": f.read()})
    return audios


async def _run_session(url: str, audio: dict, delay: float, n_level: int) -> dict:
    import websockets

    res = {
        "sid": audio["sid"],
        "n": n_level,
        "transcript": "",
        "interims": [],
        "final_deltas": [],
        "ttfs": None,
        "proc_lag": None,
        "error": None,
        "interim_count": 0,
        "overrun_ms": None,
        "tag_leak": False,
    }
    await asyncio.sleep(delay)
    try:
        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=60)  # ready
            except Exception:
                pass
            final_ev = asyncio.Event()
            last_sent = [0.0]
            vad_stop_t = [0.0]
            final_parts: list[str] = []

            async def recv():
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        d = json.loads(raw)
                        if d.get("type") != "transcript":
                            continue
                        txt = d.get("text", "")
                        if "<|" in txt or "|>" in txt:
                            res["tag_leak"] = True
                        if d.get("is_final") and d.get("finalize"):
                            now = time.monotonic()
                            res["final_deltas"].append(txt)
                            if txt:
                                final_parts.append(txt)
                            res["ttfs"] = (now - vad_stop_t[0]) * 1000
                            res["proc_lag"] = (now - last_sent[0]) * 1000
                            final_ev.set()
                        else:
                            res["interims"].append(txt)
                            res["interim_count"] += 1
                except Exception:
                    pass

            rt = asyncio.create_task(recv())
            await ws.send(json.dumps({"type": "vad_start"}))
            stream = audio["pcm"] + bytes(int(SAMPLE_RATE * TRAIL_MS / 1000) * 2)
            expected_s = len(stream) / 2 / SAMPLE_RATE
            t0 = time.monotonic()
            sent = 0
            chunk_i = 0
            while sent < len(stream):
                await ws.send(stream[sent:sent + CHUNK_BYTES])
                sent += CHUNK_BYTES
                last_sent[0] = time.monotonic()
                chunk_i += 1
                dt = t0 + chunk_i * (CHUNK_MS / 1000.0) - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)
            res["overrun_ms"] = (time.monotonic() - t0 - expected_s) * 1000
            vad_stop_t[0] = time.monotonic()
            await ws.send(json.dumps({"type": "vad_stop"}))
            await ws.send(json.dumps({"type": "reset", "finalize": True}))
            try:
                await asyncio.wait_for(final_ev.wait(), timeout=30)
            except asyncio.TimeoutError:
                res["error"] = "timeout"
            res["transcript"] = " ".join(final_parts).strip()
            rt.cancel()
    except Exception as e:  # noqa: BLE001
        res["error"] = str(e)[:120]
    return res


def _run_level(url: str, audios: list[dict], n: int) -> list[dict]:
    rnd = random.Random(1234 + n)
    delays = [rnd.uniform(0, START_JITTER_MS / 1000) for _ in range(n)]
    sess = audios[:n]

    async def go():
        return await asyncio.wait_for(
            asyncio.gather(*[_run_session(url, sess[i], delays[i], n) for i in range(n)]),
            timeout=360,
        )

    return asyncio.run(go())


def _summarize_level(results: list[dict], baselines: dict[str, dict] | None) -> dict:
    ok = [r for r in results if not r["error"]]
    lags = [r["proc_lag"] for r in ok if r["proc_lag"] is not None]
    ttfs = [r["ttfs"] for r in ok if r["ttfs"] is not None]
    over = [r["overrun_ms"] for r in results if r["overrun_ms"] is not None]
    strict_exact = None
    strict_diffs: list[dict] = []
    if baselines is not None:
        exact = 0
        for r in ok:
            diffs = _strict_diff(r, baselines.get(r["sid"], {}))
            if not diffs:
                exact += 1
            elif len(strict_diffs) < 10:
                strict_diffs.append({"sid": r["sid"], "diffs": diffs})
        strict_exact = exact
    return {
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "tag_leaks": sum(1 for r in results if r["tag_leak"]),
        "ttfs_p50_ms": _pct(ttfs, 0.50),
        "ttfs_p95_ms": _pct(ttfs, 0.95),
        "proc_lag_p50_ms": _pct(lags, 0.50),
        "proc_lag_p95_ms": _pct(lags, 0.95),
        "overrun_p95_ms": _pct(over, 0.95),
        "overrun_max_ms": max(over) if over else None,
        "strict_exact": strict_exact,
        "strict_diffs": strict_diffs,
        "sample_transcripts": [
            {
                "sid": r["sid"],
                "text": r["transcript"][:180],
                "error": r["error"],
                "tag_leak": r["tag_leak"],
            }
            for r in ok[:5]
        ],
    }


@app.function(image=image, cpu=8.0, region="us-east-1", timeout=2400)
def sweep(url: str, levels: list[int], sample_count: int, strict_byte: bool) -> dict:
    count = max(sample_count, max(levels) if levels else 1, 1)
    audios = _load_audios(count)
    result = {
        "url": url,
        "levels": levels,
        "sample_count": count,
        "strict_byte": strict_byte,
        "started_epoch": time.time(),
        "summaries": {},
        "level_results": {},
    }
    baselines = None
    if strict_byte:
        baselines = {}
        for audio in audios[:count]:
            r = _run_level(url, [audio], 1)[0]
            if not r["error"]:
                baselines[audio["sid"]] = _strict_record(r)
        result["baseline_count"] = len(baselines)
    for n in levels:
        started = time.time()
        rs = _run_level(url, audios, n)
        ended = time.time()
        summary = _summarize_level(rs, baselines)
        summary["started_epoch"] = started
        summary["ended_epoch"] = ended
        result["summaries"][str(n)] = summary
        result["level_results"][str(n)] = rs
    result["ended_epoch"] = time.time()
    return result


def _fmt(v: float | None) -> str:
    return "nan" if v is None else f"{v:.0f}"


@app.local_entrypoint()
def main(
    url: str,
    sweep_str: str = "1,2,4,6,8,12,16",
    sample_count: int = 0,
    strict_byte: bool = False,
    output: str = "",
):
    levels = [int(x) for x in sweep_str.split(",") if x.strip()]
    print(f"co-located (us-east-1) sweep vs {url}")
    print(f"levels={levels} sample_count={sample_count or max(levels)} strict_byte={strict_byte}")
    result = sweep.remote(url, levels, sample_count, strict_byte)
    if output:
        Path(output).write_text(json.dumps(result, indent=2))
        print(f"wrote {output}")
    print("  N   ok  strict  TTFSp50  TTFSp95  lag_p50  lag_p95  over_p95  over_max  errs tags")
    for n in levels:
        s = result["summaries"][str(n)]
        strict = "-" if s["strict_exact"] is None else f"{s['strict_exact']}/{s['ok']}"
        print(
            f"  {n:<3} {s['ok']:<3} {strict:<7} "
            f"{_fmt(s['ttfs_p50_ms']):>7}  {_fmt(s['ttfs_p95_ms']):>7}  "
            f"{_fmt(s['proc_lag_p50_ms']):>7}  {_fmt(s['proc_lag_p95_ms']):>7}  "
            f"{_fmt(s['overrun_p95_ms']):>8}  {_fmt(s['overrun_max_ms']):>8}  "
            f"{s['errors']:<4} {s['tag_leaks']}"
        )
        for t in s["sample_transcripts"][:2]:
            print(f"      {t['sid'][:8]}: {t['text']}")
