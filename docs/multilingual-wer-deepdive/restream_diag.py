"""H1/H2 resolver for the multilingual front-drop bug.

Re-streams a front-drop clip through the running ML server under the EXACT full1000 conditions
(?language=en-US, 20ms chunks realtime-paced, 200ms trailing silence, vad_stop+reset finalize) and
dumps the FULL message stream (every interim + every finalize delta, with timestamps).

DISCRIMINATOR:
  - If the utterance FRONT appears in the interims (cumulative current_text) but NOT in the final emit
    -> the streaming decode produced it, the FINALIZE/delta code dropped it  => H2 (our code).
  - If the front NEVER appears in any interim (interims start mid/late)
    -> the streaming decode never produced it                                => H1 (model fragility).
  - If the front appears in early interims then DISAPPEARS from later ones (current_text shrinks)
    -> a mid-stream reset/truncation in our code                             => H2 (our code).

Usage: restream_diag.py [ws-base=ws://127.0.0.1:8081] [sample_id ...]
Run with a python that has `websockets` (e.g. stt-benchmark/.venv/bin/python3).
"""
import asyncio, json, os, sqlite3, sys, time
import websockets

WS = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8081"
SIDS = sys.argv[2:] or ["a934808b-1458-4655-5c9c-26655716a079",
                        "5054b614-58eb-54fc-8ac2-0f03677f92eb"]
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(REPO, "stt-benchmark/stt_benchmark_data/results.db")
SR, CHUNK_MS = 16000, 20
CHUNK_BYTES = int(SR * CHUNK_MS / 1000) * 2
TRAILING_SILENCE_MS = 200


def lookup(sid):
    c = sqlite3.connect(DB)
    ap = c.execute("SELECT audio_path FROM samples WHERE sample_id=?", (sid,)).fetchone()
    ref = c.execute("SELECT normalized_reference FROM wer_metrics WHERE sample_id=? AND "
                    "model_name='ml_silence0_warm200_c12'", (sid,)).fetchone()
    return (os.path.join(REPO, "stt-benchmark", ap[0]) if ap else None,
            ref[0] if ref else "")


async def restream(sid):
    path, ref = lookup(sid)
    pcm = open(path, "rb").read()
    url = f"{WS}?language=en-US"
    msgs = []  # (t_rel, is_final, finalize, text)
    t_connect = time.monotonic()
    async with websockets.connect(url, max_size=16 * 1024 * 1024, open_timeout=120) as ws:
        first = await asyncio.wait_for(ws.recv(), timeout=120)
        assert json.loads(first).get("type") in ("ready", "error"), first
        done = asyncio.Event()

        async def rx():
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    d = json.loads(raw)
                    if d.get("type") != "transcript":
                        continue
                    msgs.append((time.monotonic() - t_connect, bool(d.get("is_final")),
                                 bool(d.get("finalize")), d.get("text", "")))
                    if d.get("is_final") and d.get("finalize"):
                        done.set()
            except Exception:
                pass

        rt = asyncio.create_task(rx())
        await ws.send(json.dumps({"type": "vad_start"}))
        t0 = time.monotonic()
        i = s = 0
        while s < len(pcm):
            await ws.send(pcm[s:s + CHUNK_BYTES]); s += CHUNK_BYTES; i += 1
            dt = t0 + i * (CHUNK_MS / 1000.0) - time.monotonic()
            if dt > 0:
                await asyncio.sleep(dt)
        for _ in range(TRAILING_SILENCE_MS // CHUNK_MS):
            await ws.send(bytes(CHUNK_BYTES)); i += 1
            dt = t0 + i * (CHUNK_MS / 1000.0) - time.monotonic()
            if dt > 0:
                await asyncio.sleep(dt)
        await ws.send(json.dumps({"type": "vad_stop"}))
        await ws.send(json.dumps({"type": "reset", "finalize": True}))
        try:
            await asyncio.wait_for(done.wait(), timeout=20)
        except asyncio.TimeoutError:
            pass
        rt.cancel()

    interims = [(t, x) for t, isf, fin, x in msgs if not (isf and fin)]
    finals = [(t, x) for t, isf, fin, x in msgs if isf and fin]
    joined_final = " ".join(x for _, x in finals if x).strip()
    longest_interim = max((x for _, x in interims), key=len, default="")

    # front detection: distinctive 2nd-5th content words of the reference
    rw = ref.split()
    front_probe = " ".join(rw[1:5]).lower() if len(rw) >= 5 else ref[:30].lower()
    def has_front(t): return front_probe and front_probe in (t or "").lower()
    front_in_any_interim = any(has_front(x) for _, x in interims)
    front_in_longest = has_front(longest_interim)
    front_in_final = has_front(joined_final)
    # shrink detection: did cumulative interim length drop after containing the front?
    shrank = False
    seen_front = False
    for _, x in interims:
        if has_front(x):
            seen_front = True
        elif seen_front and not has_front(x):
            shrank = True
            break

    print(f"\n{'='*70}\n{sid[:8]}  ref_front_probe={front_probe!r}")
    print(f"REF           : {ref[:110]}")
    print(f"#interims={len(interims)}  #finalize_msgs={len(finals)}")
    print(f"first 3 interims:")
    for t, x in interims[:3]:
        print(f"   t={t:5.1f}s  {x[:95]!r}")
    print(f"last 3 interims:")
    for t, x in interims[-3:]:
        print(f"   t={t:5.1f}s  {x[:95]!r}")
    print(f"finalize deltas:")
    for t, x in finals:
        print(f"   t={t:5.1f}s  {x[:95]!r}")
    print(f"JOINED FINAL  : {joined_final[:110]!r}")
    print(f"--- front '{front_probe}' : in_any_interim={front_in_any_interim}  "
          f"in_longest_interim={front_in_longest}  in_final={front_in_final}  interim_shrank={shrank}")
    if front_in_any_interim and not front_in_final:
        verdict = "H2 (our code): front WAS decoded in streaming, lost at finalize/delta"
    elif front_in_any_interim and shrank:
        verdict = "H2 (our code): front decoded then current_text truncated mid-stream"
    elif not front_in_any_interim:
        verdict = "H1 (model): streaming decode NEVER produced the front"
    else:
        verdict = "INCONCLUSIVE (front present everywhere — did this clip reproduce the drop?)"
    print(f"=> VERDICT: {verdict}")
    return verdict


async def main():
    print(f"re-stream H1/H2 diag vs {WS}  clips={[s[:8] for s in SIDS]}")
    for sid in SIDS:
        try:
            await restream(sid)
        except Exception as e:
            print(f"  {sid[:8]} ERROR: {e}")

asyncio.run(main())
