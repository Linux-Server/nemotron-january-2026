#!/usr/bin/env python3
"""Standalone in-box ASR knee load-gen — lifted from proj-2026-05-20-modal-cost/coloc_loadgen.py
(Modal wrapper stripped; reads .pcm clips from a dir instead of the benchmark DB).

Streams clips at 1x realtime over N concurrent WS sessions vs a localhost server, measuring
realtime keep-up (processing lag) + finalize TTFS. Run ON the EC2 box vs the local server.

  python ec2_loadgen.py --url ws://127.0.0.1:8080 --sweep 1,4,8,12,16,24,32,40,48,56,64 \
      --audio-dir ~/nemotron/loadgen_audio --output baseline.json
"""
from __future__ import annotations
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # 20ms int16
TRAIL_MS = 200
START_JITTER_MS = 400
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


async def _run_session(url, audio, delay, n_level):
    import websockets
    res = {"sid": audio["sid"], "n": n_level, "transcript": "", "ttfs": None, "proc_lag": None,
           "error": None, "interim_count": 0, "overrun_ms": None, "tag_leak": False}
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
            final_parts = []

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
                                final_parts.append(txt)
                            res["ttfs"] = (now - vad_stop_t[0]) * 1000
                            res["proc_lag"] = (now - last_sent[0]) * 1000
                            final_ev.set()
                        else:
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


def _run_level(url, audios, n):
    rnd = random.Random(1234 + n)
    delays = [rnd.uniform(0, START_JITTER_MS / 1000) for _ in range(n)]
    async def go():
        return await asyncio.wait_for(
            asyncio.gather(*[_run_session(url, audios[i], delays[i], n) for i in range(n)]),
            timeout=480)
    return asyncio.run(go())


def _summ(results):
    ok = [r for r in results if not r["error"]]
    lags = [r["proc_lag"] for r in ok if r["proc_lag"] is not None]
    ttfs = [r["ttfs"] for r in ok if r["ttfs"] is not None]
    over = [r["overrun_ms"] for r in results if r["overrun_ms"] is not None]
    lag95 = _pct(lags, 0.95)
    return {"ok": len(ok), "errors": len(results) - len(ok),
            "tag_leaks": sum(1 for r in results if r["tag_leak"]),
            "ttfs_p50": _pct(ttfs, 0.5), "ttfs_p95": _pct(ttfs, 0.95),
            "lag_p50": _pct(lags, 0.5), "lag_p95": lag95,
            "over_p95": _pct(over, 0.95), "over_max": max(over) if over else None,
            "keepup": (lag95 is not None and lag95 < KEEPUP_LAG_MS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8080")
    ap.add_argument("--sweep", default="1,4,8,12,16,24,32,40,48,56,64")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    levels = [int(x) for x in args.sweep.split(",") if x.strip()]
    audios = load_audios(args.audio_dir, max(levels))
    ndistinct = len({a["sid"] for a in audios})
    print(f"loadgen vs {args.url} | {ndistinct} distinct clips (cycled to N) | levels={levels}")
    print("  N   ok  errs  TTFSp50 TTFSp95  lagp50  lagp95  over95  keepup")
    out = {"url": args.url, "levels": levels, "summaries": {}}
    knee = 0
    for n in levels:
        s = _summ(_run_level(args.url, audios, n))
        out["summaries"][str(n)] = s
        f = lambda v: "nan" if v is None else f"{v:.0f}"
        print(f"  {n:<3} {s['ok']:<3} {s['errors']:<4} {f(s['ttfs_p50']):>7} {f(s['ttfs_p95']):>7} "
              f"{f(s['lag_p50']):>7} {f(s['lag_p95']):>7} {f(s['over_p95']):>6}  {'YES' if s['keepup'] else 'no'}")
        if s["keepup"] and s["errors"] == 0:
            knee = n
    print(f"\nKNEE (max N with proc-lag p95 < {KEEPUP_LAG_MS:.0f}ms and 0 errors): {knee}")
    out["knee"] = knee
    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
