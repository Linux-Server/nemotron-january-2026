#!/usr/bin/env python3
"""rt_loadgen.py — realtime MULTIPROCESS pacer for the C++ ws_server clean-knee measurement.

Why multiprocess: a single Python asyncio process pacing 64-128 WS connections at 1x
realtime + JSON parsing becomes a CPU/event-loop bottleneck and inflates the measured
TTFS/lag (the loadgen, not the server, would be the limiter). This shards the N
concurrent sessions across --procs worker processes, each running the *exact* proven
ec2_loadgen.py session logic (vad_start -> stream PCM at 1x -> vad_stop -> finalize,
recording finalize-TTFS + proc-lag), so the pacing semantics are byte-for-byte the
validated ones — only the driver is parallelized.

SUSTAINED steady window: each session slot loops, launching back-to-back utterances,
until --window-s elapses, so N connections stay continuously active and telemetry can
be sampled over a stable steady state (not a one-shot burst).

  python rt_loadgen.py --url ws://127.0.0.1:8080 --audio-dir ~/density/loadgen_audio_smoke \
      --sweep 64,80,96 --procs 32 --window-s 60 --output knee.json

Metrics per level: ok/err counts, finalize-TTFS p50/p95/p99/max, proc-lag (keepup) p50/p95/p99,
maxjit (worst per-chunk realtime-pacing lateness, ms — loadgen-health signal).
"""
from __future__ import annotations
import argparse
import asyncio
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # 20ms int16
TRAIL_MS = 200
START_JITTER_MS = int(os.environ.get("LOADGEN_JITTER_MS", "400"))
KEEPUP_LAG_MS = 500.0  # proc-lag p95 below this == realtime keep-up


def _pct(values, p):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]


def load_audios(audio_dir, count):
    files = sorted(Path(audio_dir).glob("*.pcm"))
    if not files:
        raise SystemExit(f"no .pcm files in {audio_dir}")
    return [{"sid": files[i % len(files)].stem, "pcm": files[i % len(files)].read_bytes()}
            for i in range(count)]


async def _stream_once(ws, audio, res):
    """One utterance over an already-open ws. Records ttfs, proc_lag, maxjit. Returns ok bool."""
    import websockets  # noqa: F401
    st = {"final_ev": asyncio.Event(), "last_sent": 0.0, "vad_stop_t": 0.0, "final_parts": []}

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
                    if txt:
                        st["final_parts"].append(txt)
                    res["ttfs_list"].append((now - st["vad_stop_t"]) * 1000)
                    res["proc_lag_list"].append((now - st["last_sent"]) * 1000)
                    st["final_ev"].set()
                    return
                else:
                    res["interim_count"] += 1
        except Exception:
            pass

    rt = asyncio.create_task(recv())
    stream = audio["pcm"] + bytes(int(SAMPLE_RATE * TRAIL_MS / 1000) * 2)
    await ws.send(json.dumps({"type": "vad_start"}))
    t0 = time.monotonic()
    sent = 0
    chunk_i = 0
    while sent < len(stream):
        await ws.send(stream[sent:sent + CHUNK_BYTES])
        sent += CHUNK_BYTES
        st["last_sent"] = time.monotonic()
        chunk_i += 1
        target = t0 + chunk_i * (CHUNK_MS / 1000.0)
        dt = target - time.monotonic()
        if dt > 0:
            await asyncio.sleep(dt)
        else:
            # late: loadgen could not keep up with realtime pacing (loadgen-health signal)
            lateness_ms = -dt * 1000.0
            if lateness_ms > res["maxjit_ms"]:
                res["maxjit_ms"] = lateness_ms
    st["vad_stop_t"] = time.monotonic()
    await ws.send(json.dumps({"type": "vad_stop"}))
    await ws.send(json.dumps({"type": "reset", "finalize": True}))
    ok = True
    try:
        await asyncio.wait_for(st["final_ev"].wait(), timeout=30)
    except asyncio.TimeoutError:
        res["error"] = "timeout"
        ok = False
    rt.cancel()
    return ok


async def _run_slot(url, audios, slot_idx, delay, deadline, res):
    """One steady session slot: open a fresh ws per utterance, loop until deadline."""
    import websockets
    await asyncio.sleep(delay)
    rotate = slot_idx
    while time.monotonic() < deadline:
        audio = audios[rotate % len(audios)]
        rotate += 1
        try:
            async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
                try:
                    await asyncio.wait_for(ws.recv(), timeout=60)  # ready frame
                except Exception:
                    pass
                ok = await _stream_once(ws, audio, res)
                if not ok:
                    res["errors"] += 1
                    return  # stop this slot on first error (matches strict 0-err gate)
                else:
                    res["completed"] += 1
        except Exception as e:  # noqa: BLE001
            res["error"] = str(e)[:120]
            res["errors"] += 1
            return


def _worker(args):
    (url, audio_dir, n_total, proc_idx, n_procs, window_s, jitter_ms, seed_off, max_clips) = args
    audios = load_audios(audio_dir, max_clips)
    # round-robin slot assignment across procs
    slots = list(range(proc_idx, n_total, n_procs))
    rnd = random.Random(1234 + n_total + seed_off * 997 + proc_idx * 7)
    res = {"ttfs_list": [], "proc_lag_list": [], "maxjit_ms": 0.0,
           "completed": 0, "errors": 0, "error": None, "interim_count": 0, "tag_leak": False}

    async def go():
        t_start = time.monotonic()
        deadline = t_start + window_s
        tasks = []
        for s in slots:
            delay = rnd.uniform(0, jitter_ms / 1000.0)
            tasks.append(asyncio.create_task(_run_slot(url, audios, s, delay, deadline, res)))
        if tasks:
            await asyncio.gather(*tasks)

    asyncio.run(go())
    return res


def run_level(url, audio_dir, n, procs, window_s, max_clips, seed_off=0):
    procs = max(1, min(procs, n))
    jobs = [(url, audio_dir, n, p, procs, window_s, START_JITTER_MS, seed_off, max_clips)
            for p in range(procs)]
    with mp.Pool(procs) as pool:
        parts = pool.map(_worker, jobs)
    # pool
    ttfs, lags = [], []
    completed = errors = interim = 0
    maxjit = 0.0
    err_msg = None
    tag_leaks = 0
    for r in parts:
        ttfs.extend(r["ttfs_list"])
        lags.extend(r["proc_lag_list"])
        completed += r["completed"]
        errors += r["errors"]
        interim += r["interim_count"]
        maxjit = max(maxjit, r["maxjit_ms"])
        if r["error"] and not err_msg:
            err_msg = r["error"]
        if r["tag_leak"]:
            tag_leaks += 1
    lag95 = _pct(lags, 0.95)
    return {
        "n": n, "procs": procs, "window_s": window_s,
        "completed": completed, "errors": errors, "err_msg": err_msg,
        "ttfs_p50": _pct(ttfs, 0.5), "ttfs_p95": _pct(ttfs, 0.95), "ttfs_p99": _pct(ttfs, 0.99),
        "ttfs_max": max(ttfs) if ttfs else None, "ttfs_n": len(ttfs),
        "lag_p50": _pct(lags, 0.5), "lag_p95": lag95, "lag_p99": _pct(lags, 0.99),
        "maxjit_ms": maxjit, "tag_leaks": tag_leaks,
        "keepup": (lag95 is not None and lag95 < KEEPUP_LAG_MS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8080")
    ap.add_argument("--sweep", default="64,80,96")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--procs", type=int, default=32)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    levels = [int(x) for x in args.sweep.split(",") if x.strip()]
    max_clips = max(max(levels), 24)
    print(f"rt_loadgen vs {args.url} | procs={args.procs} window={args.window_s}s | levels={levels}")
    print("  N    procs  completed  errs  TTFSp50 TTFSp95 TTFSp99 TTFSmax  ttfsN  lagp50  lagp95  lagp99  maxjit  keepup")
    out = {"url": args.url, "levels": levels, "procs": args.procs, "window_s": args.window_s, "summaries": {}}
    knee = 0
    for n in levels:
        s = run_level(args.url, args.audio_dir, n, args.procs, args.window_s, max_clips)
        out["summaries"][str(n)] = s
        f = lambda v: "nan" if v is None else f"{v:.0f}"
        print(f"  {n:<4} {s['procs']:<5}  {s['completed']:<9} {s['errors']:<4}  "
              f"{f(s['ttfs_p50']):>7} {f(s['ttfs_p95']):>7} {f(s['ttfs_p99']):>7} {f(s['ttfs_max']):>7} "
              f"{s['ttfs_n']:>6}  {f(s['lag_p50']):>6} {f(s['lag_p95']):>6} {f(s['lag_p99']):>6} "
              f"{f(s['maxjit_ms']):>6}  {'YES' if (s['keepup'] and s['errors']==0) else 'no'}")
        if s["keepup"] and s["errors"] == 0 and (s["ttfs_p95"] is not None and s["ttfs_p95"] < 400):
            knee = n
    print(f"\nKNEE (max N: 0-err AND lag-p95<{KEEPUP_LAG_MS:.0f}ms AND TTFS-p95<400ms): {knee}")
    out["knee"] = knee
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
