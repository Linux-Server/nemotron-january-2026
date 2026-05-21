"""Step 0 — capture the byte-exact baseline artifact (current B=1 greedy server).

Streams a fixed English clip set through the CURRENT local server (ws://127.0.0.1:8080,
silence0_warm200 / rc1 / greedy / loop_labels=False) and records, per clip, the FULL interim
transcript sequence + final text + final delta, plus git identity (the tree is dirty). Every later
batching/compile step diffs flag-on output against this artifact.

(Multilingual prompted baseline needs the EA-venv server separately — captured later.)

Usage: baseline_capture.py [ws-url]  -> writes proj-2026-05-21-0410/baseline/english_baseline.json
"""
import asyncio
import hashlib
import json
import os
import subprocess
import sqlite3
import sys
import time

import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8080"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "stt-benchmark/stt_benchmark_data/results.db")
OUTDIR = os.path.join(REPO, "proj-2026-05-21-0410", "baseline")
os.makedirs(OUTDIR, exist_ok=True)

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


async def capture_one(sample_id, audio_path, dur):
    p = os.path.join(REPO, "stt-benchmark", audio_path)
    with open(p, "rb") as f:
        pcm = f.read()
    rec = {"sample_id": sample_id, "duration_s": round(dur, 2), "interims": [], "final": "",
           "final_delta": "", "error": None}
    try:
        async with websockets.connect(URL, max_size=16 * 1024 * 1024, open_timeout=120) as ws:
            try:
                await asyncio.wait_for(ws.recv(), timeout=120)  # ready
            except Exception:
                pass
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
    return rec


async def main():
    print(f"baseline capture vs {URL}, {len(PICKS)} clips")
    records = []
    for sid, ap, dur in PICKS:
        r = await capture_one(sid, ap, dur)
        records.append(r)
        print(f"  {sid[:8]} {r['duration_s']:>5}s  final={r['final'][:60]!r}  "
              f"interims={len(r['interims'])}  err={r['error']}")
    artifact = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "url": URL,
        "config": "silence0_warm200 rc1 greedy loop_labels=False (current default)",
        "git_head": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short"),
        "server_py_diff_sha": hashlib.sha256(
            git("diff", "--", "src/nemotron_speech/server.py").encode()).hexdigest()[:16],
        "records": records,
    }
    out = os.path.join(OUTDIR, "english_baseline.json")
    with open(out, "w") as f:
        json.dump(artifact, f, indent=2)
    ok = sum(1 for r in records if not r["error"] and r["final"])
    print(f"\nwrote {out}  ({ok}/{len(records)} clips with final, git={artifact['git_head'][:8]})")

asyncio.run(main())
