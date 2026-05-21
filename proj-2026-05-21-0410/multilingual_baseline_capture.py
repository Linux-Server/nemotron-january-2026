"""Step 0b — capture the byte-exact multilingual prompted baseline artifact.

Streams the same fixed 8-clip set as baseline_capture.py through the CURRENT
multilingual local server (ws://127.0.0.1:8081, silence0_warm200 / rc3 / greedy /
loop_labels=False) for en-US, es-ES, and auto. Records, per (clip, language), the
FULL interim transcript sequence + final text + final delta + effective target_lang
and verifies language tags do not leak into user-visible text.

Usage:
  multilingual_baseline_capture.py [ws-url-base]

Default writes:
  proj-2026-05-21-0410/baseline/multilingual_baseline.json
"""
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sqlite3
import sys
import time
from urllib.parse import urlencode

import websockets

URL_BASE = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8081"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "stt-benchmark/stt_benchmark_data/results.db")
OUTDIR = os.path.join(REPO, "proj-2026-05-21-0410", "baseline")
OUT = os.path.join(OUTDIR, "multilingual_baseline.json")
os.makedirs(OUTDIR, exist_ok=True)

LANGUAGES = ["en-US", "es-ES", "auto"]
TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")

# Fixed clip set: deterministic, spanning durations (incl. a long one for multi-segment behavior).
con = sqlite3.connect(DB)
rows = con.execute(
    "SELECT sample_id, audio_path, duration_seconds FROM samples "
    "WHERE duration_seconds IS NOT NULL ORDER BY duration_seconds"
).fetchall()
n = len(rows)
PICKS = [rows[round(i * (n - 1) / 7)] for i in range(8)]  # 8 clips spanning the range


def git(*args):
    try:
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def url_for_language(language):
    if language == "auto":
        return URL_BASE
    sep = "&" if "?" in URL_BASE else "?"
    return f"{URL_BASE}{sep}{urlencode({'language': language})}"


def has_tag_leak(texts):
    return any(TAG_RE.search(text or "") for text in texts)


async def recv_ready(ws):
    raw = await asyncio.wait_for(ws.recv(), timeout=120)
    if isinstance(raw, bytes):
        raise RuntimeError("expected ready/error text frame, got bytes")
    msg = json.loads(raw)
    if msg.get("type") == "ready":
        return
    if msg.get("type") == "error":
        raise RuntimeError(f"server error before ready: {msg.get('message', '')}")
    raise RuntimeError(f"expected ready, got {msg}")


async def capture_one(sample_id, audio_path, dur, language):
    p = os.path.join(REPO, "stt-benchmark", audio_path)
    with open(p, "rb") as f:
        pcm = f.read()
    rec = {
        "sample_id": sample_id,
        "duration_s": round(dur, 2),
        "language": language,
        "target_lang_effective": language,
        "interims": [],
        "final": "",
        "final_delta": "",
        "error": None,
        "tag_leak": False,
    }
    try:
        async with websockets.connect(
            url_for_language(language), max_size=16 * 1024 * 1024, open_timeout=120
        ) as ws:
            await recv_ready(ws)
            finals = []
            done = asyncio.Event()

            async def recv():
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        d = json.loads(raw)
                        if d.get("type") != "transcript":
                            continue
                        if d.get("is_final") and d.get("finalize"):
                            txt = d.get("text", "")
                            finals.append(txt)
                            done.set()
                        else:
                            rec["interims"].append(d.get("text", ""))
                except Exception:
                    pass

            rt = asyncio.create_task(recv())
            await ws.send(json.dumps({"type": "vad_start"}))
            CH = int(16000 * 2 * 0.02)
            t0 = time.monotonic()
            i = s = 0
            while s < len(pcm):
                await ws.send(pcm[s:s + CH]); s += CH; i += 1
                dt = t0 + i * 0.02 - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)
            for _ in range(10):
                await ws.send(bytes(CH)); i += 1
                dt = t0 + i * 0.02 - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)
            await ws.send(json.dumps({"type": "vad_stop"}))
            await ws.send(json.dumps({"type": "reset", "finalize": True}))
            try:
                await asyncio.wait_for(done.wait(), timeout=20)
            except asyncio.TimeoutError:
                pass
            rt.cancel()
            rec["final_delta"] = finals[-1] if finals else ""
            rec["final"] = " ".join(f for f in finals if f).strip()
    except Exception as e:
        rec["error"] = str(e)[:120]

    rec["tag_leak"] = has_tag_leak(rec["interims"] + [rec["final"]])
    return rec


async def main():
    print(f"multilingual baseline capture vs {URL_BASE}, {len(PICKS)} clips x {len(LANGUAGES)} languages")
    records = []
    for language in LANGUAGES:
        print(f"\nlanguage={language} url={url_for_language(language)}")
        for sid, ap, dur in PICKS:
            r = await capture_one(sid, ap, dur, language)
            records.append(r)
            print(
                f"  {sid[:8]} {r['duration_s']:>5}s  final={r['final'][:60]!r}  "
                f"interims={len(r['interims'])}  tag_leak={r['tag_leak']}  err={r['error']}"
            )

    artifact = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url_base": URL_BASE,
        "config": "multilingual rc3 prompted greedy loop_labels=False silence0_warm200",
        "git_head": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short"),
        "server_py_diff_sha": hashlib.sha256(
            git("diff", "--", "src/nemotron_speech/server.py").encode()).hexdigest()[:16],
        "languages": LANGUAGES,
        "records": records,
    }
    with open(OUT, "w") as f:
        json.dump(artifact, f, indent=2)

    ok = sum(1 for r in records if not r["error"] and r["final"])
    leaks = sum(1 for r in records if r["tag_leak"])
    errors = sum(1 for r in records if r["error"])
    print(
        f"\nwrote {OUT}  ({ok}/{len(records)} rows with final, "
        f"errors={errors}, tag_leaks={leaks}, git={artifact['git_head'][:8]})"
    )
    if len(records) != len(PICKS) * len(LANGUAGES) or leaks:
        raise SystemExit(1)


asyncio.run(main())
