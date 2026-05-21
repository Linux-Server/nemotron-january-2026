#!/usr/bin/env python3
"""Concurrency / parallel-inference test for the Nemotron streaming ASR server.

================================================================================
TEST GOALS
================================================================================
The parent project measured latency single-session only (sequential benchmark).
This harness closes the documented "R4" concurrency gap: how does the server
behave when N live sessions stream simultaneously?

The server does NOT batch requests. Every `conformer_stream_step` call is
serialized through one global `inference_lock` (server.py:453), and the model
always runs batch_size=1 (the batch API was found to corrupt the cache-aware
streaming state — server.py:671). So this test measures *concurrent-session
correctness + latency under lock contention*, NOT GPU batching throughput.

Saturation arithmetic: each chunk is ~10-15 ms of GPU work and arrives every
160 ms of real-time audio, so one session occupies ~7-9% of the GPU. Saturation
(GPU busy ~100% of the real-time window) is expected around N ~= 12. The sweep
{1,4,8,12,16,20,24} brackets that knee and extends well past it.

We measure four things per concurrency level N:
  1. byte-accuracy   - each session's transcript vs its own N=1 baseline.
                       Cross-session state leakage / races would show here.
                       By default this preserves the historical edit-distance
                       summary. With --strict-byte it also compares the interim
                       sequence, final transcript, final delta(s), and duplicate
                       final emissions exactly.
  2. finalize TTFS   - (final transcript received) - (vad_stop sent), per session.
                       The headline latency metric. p50/p95 across sessions.
  3. processing lag  - (final transcript received) - (last audio chunk sent).
                       THE production-ceiling metric: at/past saturation the
                       server falls behind realtime chunk arrival and this grows
                       unbounded, which is worse than a one-time TTFS bump.
  4. ready latency   - (server "ready") - (websocket connected), per session.
                       Captures warm-up (warm200) contention at concurrent connect.

================================================================================
CONSTRAINTS (faithful to the production Pipecat use case)
================================================================================
- REALTIME STREAMING: audio is streamed at 1x playback rate (20 ms chunk every
  20 ms wall-clock), simulating a live microphone. This is the core constraint:
  we are reproducing the realtime production scenario.
    * NOTE: warm-up prefix (at connect) and fork-flush tail-silence (at finalize)
      are SERVER-SIDE synthetic audio processed faster-than-realtime. They do not
      violate the realtime-client constraint (the client never sends them) -- but
      they DO consume the shared inference_lock, so they contribute to contention.
      That is intentional and is part of what we measure.
- PRODUCTION SETTINGS: server runs NEMOTRON_FINALIZE_SILENCE_MS=0 (the validated
  finalize-debounce removal) + NEMOTRON_WARMUP_MS=200 (warm200) + continuous mode.
- RANDOMIZED, TIGHTLY-GROUPED START: sessions start at small random offsets
  (default 0-400 ms jitter) -- a realistic but pessimistic arrival pattern (users
  do not synchronize to the millisecond, but a busy moment clusters them). We do
  NOT test the fully-synchronized worst case (separate concern).
- DISTINCT AUDIOS: each session gets its own sample (varied 1-16 s durations),
  so identical-input cannot mask a cross-session state-leak bug. The historical
  default sweep still selects exactly 24 samples. For high-N sweeps, the harness
  selects max(sweep) distinct samples unless --sample-count asks for more.

Run with the benchmark venv (has `websockets`):
  stt-benchmark/.venv/bin/python proj-2026-05-19-eou-endpointing/concurrency_test.py

Requires a server already running with:
  NEMOTRON_FINALIZE_SILENCE_MS=0 NEMOTRON_WARMUP_MS=200 NEMOTRON_CONTINUOUS=1
  python src/nemotron_speech/server.py --host 127.0.0.1 --port 8080 --right-context 1
================================================================================
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import websockets

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_MS / 1000) * 2  # int16 LE
TRAILING_SILENCE_MS = 200  # mimic Silero stop_secs=0.2 detection window before vad_stop
DEFAULT_URL = "ws://127.0.0.1:8080"
DEFAULT_DB = REPO_ROOT / "stt-benchmark" / "stt_benchmark_data" / "test_results.db"
DEFAULT_SWEEP = [1, 4, 8, 12, 16, 20, 24]
START_JITTER_MS = 400  # tight randomized grouping


@dataclass
class SessionResult:
    sample_id: str
    n_level: int
    transcript: str = ""
    interims: list[str] = field(default_factory=list)
    final_deltas: list[str] = field(default_factory=list)
    first_interim_ms: float | None = None  # first interim recv - vad_start sent
    interim_lag_ms: list[float] = field(default_factory=list)  # interim recv - latest audio sent
    ttfs_ms: float | None = None          # final recv - vad_stop sent
    processing_lag_ms: float | None = None  # final recv - last audio sent
    ready_latency_ms: float | None = None   # ready - connected
    audio_duration_s: float = 0.0
    interim_count: int = 0
    error: str | None = None


def select_audios(db_path: Path, count: int) -> list[dict]:
    """Pick `count` distinct samples spanning the duration range (reproducible)."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT sample_id, audio_path, duration_seconds FROM samples "
        "WHERE duration_seconds IS NOT NULL ORDER BY duration_seconds"
    ).fetchall()
    con.close()
    n = len(rows)
    if n < count:
        raise SystemExit(f"need {count} samples, db has {n}")
    idxs = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    return [
        {"sample_id": rows[i][0], "audio_path": rows[i][1], "duration_seconds": rows[i][2]}
        for i in idxs
    ]


def load_pcm(audio_path: str) -> bytes:
    p = REPO_ROOT / "stt-benchmark" / audio_path
    if not p.exists():
        p = REPO_ROOT / audio_path
    return p.read_bytes()


async def run_session(url: str, audio: dict, n_level: int, start_delay_s: float) -> SessionResult:
    """Drive one session: connect -> stream realtime -> vad_stop -> await final."""
    res = SessionResult(
        sample_id=audio["sample_id"],
        n_level=n_level,
        audio_duration_s=audio["duration_seconds"],
    )
    await asyncio.sleep(start_delay_s)  # randomized tight stagger

    try:
        pcm = load_pcm(audio["audio_path"])
    except Exception as e:  # noqa: BLE001
        res.error = f"load_pcm: {e}"
        return res

    final_text_parts: list[str] = []
    final_event = asyncio.Event()
    last_audio_sent_t = 0.0
    vad_stop_sent_t = 0.0
    vad_start_sent_t = 0.0

    try:
        t_connect = time.monotonic()
        async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
            # --- ready handshake ---
            t_ready = None
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                if json.loads(msg).get("type") == "ready":
                    t_ready = time.monotonic()
            except Exception:  # noqa: BLE001
                t_ready = time.monotonic()
            res.ready_latency_ms = (t_ready - t_connect) * 1000.0

            # --- receiver task: collect interim + final transcripts ---
            async def receiver():
                nonlocal final_text_parts
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        data = json.loads(raw)
                        if data.get("type") != "transcript":
                            continue
                        now = time.monotonic()
                        if data.get("is_final") and data.get("finalize"):
                            # continuous mode emits incremental finalize deltas
                            txt = data.get("text", "")
                            res.final_deltas.append(txt)
                            if txt:
                                final_text_parts.append(txt)
                            res.ttfs_ms = (now - vad_stop_sent_t) * 1000.0
                            res.processing_lag_ms = (now - last_audio_sent_t) * 1000.0
                            final_event.set()
                        else:
                            txt = data.get("text", "")
                            res.interims.append(txt)
                            res.interim_count += 1
                            if res.first_interim_ms is None and vad_start_sent_t > 0:
                                res.first_interim_ms = (now - vad_start_sent_t) * 1000.0
                            if last_audio_sent_t > 0:
                                res.interim_lag_ms.append((now - last_audio_sent_t) * 1000.0)
                except Exception:  # noqa: BLE001
                    pass

            recv_task = asyncio.create_task(receiver())

            # --- vad_start ---
            await ws.send(json.dumps({"type": "vad_start"}))
            vad_start_sent_t = time.monotonic()

            # --- stream audio at 1x realtime (drift-corrected pacing) ---
            t_stream_start = time.monotonic()
            sent = 0
            chunk_idx = 0
            total = len(pcm)
            # append trailing silence so Silero-style stop has a window to detect
            trailing = bytes(int(SAMPLE_RATE * TRAILING_SILENCE_MS / 1000) * 2)
            stream = pcm + trailing
            while sent < len(stream):
                chunk = stream[sent : sent + CHUNK_BYTES]
                await ws.send(chunk)
                sent += len(chunk)
                last_audio_sent_t = time.monotonic()
                chunk_idx += 1
                # drift-corrected sleep: target wall-clock = start + chunk_idx*CHUNK_MS
                target = t_stream_start + chunk_idx * (CHUNK_MS / 1000.0)
                dt = target - time.monotonic()
                if dt > 0:
                    await asyncio.sleep(dt)

            # --- vad_stop + finalize reset (continuous-mode client behavior) ---
            vad_stop_sent_t = time.monotonic()
            await ws.send(json.dumps({"type": "vad_stop"}))
            await ws.send(json.dumps({"type": "reset", "finalize": True}))

            # --- await final transcript ---
            try:
                await asyncio.wait_for(final_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                res.error = "timeout waiting for final transcript"

            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        res.transcript = " ".join(final_text_parts).strip()
    except Exception as e:  # noqa: BLE001
        res.error = f"session: {e}"

    return res


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pctl(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, int(round(p * (len(vals) - 1)))))
    return vals[k]


async def run_level(url: str, audios: list[dict], n: int, seed: int) -> list[SessionResult]:
    rnd = random.Random(seed + n)
    sessions = audios[:n]
    delays = [rnd.uniform(0, START_JITTER_MS / 1000.0) for _ in range(n)]
    tasks = [run_session(url, sessions[i], n, delays[i]) for i in range(n)]
    return await asyncio.gather(*tasks)


async def run_all_bounded(
    url: str,
    audios: list[dict],
    concurrency: int,
    seed: int,
) -> list[SessionResult]:
    """Run every selected audio while keeping at most `concurrency` live sessions."""
    sem = asyncio.Semaphore(concurrency)
    rnd = random.Random(seed + concurrency + len(audios))

    async def bounded(audio: dict) -> SessionResult:
        async with sem:
            delay = rnd.uniform(0, START_JITTER_MS / 1000.0)
            return await run_session(url, audio, concurrency, delay)

    return await asyncio.gather(*[bounded(a) for a in audios])


def strict_record(r: SessionResult) -> dict:
    return {
        "transcript": r.transcript,
        "interims": r.interims,
        "final_deltas": r.final_deltas,
        "no_duplicate_final": len(r.final_deltas) <= 1,
        "first_interim_ms": r.first_interim_ms,
        "interim_lag_ms": r.interim_lag_ms,
        "ttfs_ms": r.ttfs_ms,
        "processing_lag_ms": r.processing_lag_ms,
        "ready_latency_ms": r.ready_latency_ms,
    }


def strict_diff(r: SessionResult, baseline: dict) -> list[str]:
    diffs: list[str] = []
    rec = strict_record(r)
    if rec["transcript"] != baseline.get("transcript"):
        diffs.append(
            f"final {baseline.get('transcript', '')!r} -> {rec['transcript']!r}"
        )
    if rec["final_deltas"] != baseline.get("final_deltas"):
        diffs.append(
            f"final_deltas {baseline.get('final_deltas', [])!r} -> {rec['final_deltas']!r}"
        )
    if rec["interims"] != baseline.get("interims"):
        diffs.append(
            f"interims len {len(baseline.get('interims', []))} -> {len(rec['interims'])}"
        )
    if not rec["no_duplicate_final"]:
        diffs.append(f"duplicate final emissions: {rec['final_deltas']!r}")
    return diffs


def latency_summary(level: list[SessionResult]) -> dict:
    first = [r.first_interim_ms for r in level if r.first_interim_ms is not None]
    interim_lag = [v for r in level for v in r.interim_lag_ms]
    final = [r.ttfs_ms for r in level if r.ttfs_ms is not None]
    return {
        "first_interim_ms": {"p50": pctl(first, 0.5), "p95": pctl(first, 0.95), "p99": pctl(first, 0.99)},
        "interim_lag_ms": {"p50": pctl(interim_lag, 0.5), "p95": pctl(interim_lag, 0.95), "p99": pctl(interim_lag, 0.99)},
        "final_ttfs_ms": {"p50": pctl(final, 0.5), "p95": pctl(final, 0.95), "p99": pctl(final, 0.99)},
    }


def summarize_level(
    label: str,
    level: list[SessionResult],
    baselines: dict[str, str] | dict[str, dict],
    strict_byte: bool,
) -> dict:
    errs = [r for r in level if r.error]
    if errs:
        for r in errs:
            print(f"  ERROR [{r.sample_id[:8]}]: {r.error}")

    eds = []
    exact = 0
    strict_exact = 0
    strict_diffs: list[dict] = []
    for r in level:
        if r.error or r.sample_id not in baselines:
            continue
        if strict_byte:
            base = baselines[r.sample_id]  # type: ignore[index]
            baseline_text = base["transcript"]
            diffs = strict_diff(r, base)
            if not diffs:
                strict_exact += 1
            elif len(strict_diffs) < 20:
                strict_diffs.append({"sample_id": r.sample_id, "diffs": diffs})
        else:
            baseline_text = baselines[r.sample_id]  # type: ignore[index]
        ed = levenshtein(r.transcript, baseline_text)
        eds.append(ed)
        if ed == 0:
            exact += 1

    ttfs = [r.ttfs_ms for r in level if r.ttfs_ms is not None]
    lag = [r.processing_lag_ms for r in level if r.processing_lag_ms is not None]
    ready = [r.ready_latency_ms for r in level if r.ready_latency_ms is not None]
    n_ok = len(eds)
    max_ed = max(eds) if eds else 0
    print(f"  byte-accuracy: {exact}/{n_ok} exact, "
          f"median ED={statistics.median(eds) if eds else 'NA'}, max ED={max_ed}")
    if strict_byte:
        print(f"  strict-byte:   {strict_exact}/{n_ok} exact")
        for d in strict_diffs[:5]:
            print(f"    DIFF [{d['sample_id'][:8]}]: {'; '.join(d['diffs'])}")
    if max_ed > 20:
        print(f"  *** ALARM: max edit-distance {max_ed} > 20 — possible word-level "
              f"corruption (real concurrency bug, not CUDA nondeterminism) ***")
    print(f"  TTFS ms:        p50={pctl(ttfs,0.5):.1f} p95={pctl(ttfs,0.95):.1f} max={max(ttfs):.1f}" if ttfs else "  TTFS: none")
    print(f"  proc-lag ms:    p50={pctl(lag,0.5):.1f} p95={pctl(lag,0.95):.1f} max={max(lag):.1f}" if lag else "  proc-lag: none")
    print(f"  ready ms:       p50={pctl(ready,0.5):.1f} p95={pctl(ready,0.95):.1f} max={max(ready):.1f}" if ready else "  ready: none")
    lat = latency_summary(level)
    if any(v["p50"] is not None for v in lat.values()):
        print(
            "  first-interim ms: "
            f"p50={lat['first_interim_ms']['p50'] or 0:.1f} "
            f"p95={lat['first_interim_ms']['p95'] or 0:.1f} "
            f"p99={lat['first_interim_ms']['p99'] or 0:.1f}"
        )
        print(
            "  interim-lag ms:   "
            f"p50={lat['interim_lag_ms']['p50'] or 0:.1f} "
            f"p95={lat['interim_lag_ms']['p95'] or 0:.1f} "
            f"p99={lat['interim_lag_ms']['p99'] or 0:.1f}"
        )
    print()
    return {
        "label": label,
        "ok": n_ok,
        "errors": len(errs),
        "exact": exact,
        "strict_exact": strict_exact if strict_byte else None,
        "strict_diffs": strict_diffs,
        "median_ed": statistics.median(eds) if eds else None,
        "max_ed": max_ed,
        "ttfs_p50_ms": pctl(ttfs, 0.5),
        "ttfs_p95_ms": pctl(ttfs, 0.95),
        "processing_lag_p50_ms": pctl(lag, 0.5),
        "processing_lag_p95_ms": pctl(lag, 0.95),
        "ready_p95_ms": pctl(ready, 0.95),
        "latency": lat,
    }


async def main_async(args):
    max_sweep = max(args.sweep) if args.sweep else 0
    audio_count = max(args.sample_count or 0, max_sweep, 24)
    audios = select_audios(Path(args.db), audio_count)
    print(f"Selected {audio_count} audios, durations "
          f"{min(a['duration_seconds'] for a in audios):.1f}-{max(a['duration_seconds'] for a in audios):.1f}s\n")

    # N=1 baseline first (each audio in its own isolated session, sequential)
    print("=== establishing N=1 baselines (sequential) ===")
    baselines: dict[str, str] | dict[str, dict] = {}
    for a in audios:
        r = (await run_level(args.url, [a], 1, args.seed))[0]
        if r.error:
            print(f"  baseline {a['sample_id'][:8]} ERROR: {r.error}")
        else:
            if args.strict_byte:
                baselines[a["sample_id"]] = strict_record(r)  # type: ignore[index]
            else:
                baselines[a["sample_id"]] = r.transcript  # type: ignore[index]
        await asyncio.sleep(0.1)
    print(f"  captured {len(baselines)}/{audio_count} baselines\n")

    results: dict[int, list[SessionResult]] = {}
    summaries: dict[str, dict] = {}
    for n in ([] if args.skip_sweep else args.sweep):
        print(f"=== N={n} (concurrent) ===")
        level_started_at = time.time()
        level = await run_level(args.url, audios, n, args.seed)
        level_ended_at = time.time()
        results[n] = level
        summaries[str(n)] = summarize_level(str(n), level, baselines, args.strict_byte)
        summaries[str(n)]["started_at_epoch"] = level_started_at
        summaries[str(n)]["ended_at_epoch"] = level_ended_at
        summaries[str(n)]["started_at"] = datetime.fromtimestamp(level_started_at).isoformat()
        summaries[str(n)]["ended_at"] = datetime.fromtimestamp(level_ended_at).isoformat()

    canary_results: list[SessionResult] = []
    if args.run_all_concurrency:
        print(f"=== all {len(audios)} samples at sustained concurrency {args.run_all_concurrency} ===")
        level_started_at = time.time()
        canary_results = await run_all_bounded(
            args.url,
            audios,
            args.run_all_concurrency,
            args.seed,
        )
        level_ended_at = time.time()
        summaries[f"all_c{args.run_all_concurrency}"] = summarize_level(
            f"all_c{args.run_all_concurrency}",
            canary_results,
            baselines,
            args.strict_byte,
        )
        summaries[f"all_c{args.run_all_concurrency}"]["started_at_epoch"] = level_started_at
        summaries[f"all_c{args.run_all_concurrency}"]["ended_at_epoch"] = level_ended_at
        summaries[f"all_c{args.run_all_concurrency}"]["started_at"] = datetime.fromtimestamp(level_started_at).isoformat()
        summaries[f"all_c{args.run_all_concurrency}"]["ended_at"] = datetime.fromtimestamp(level_ended_at).isoformat()

    # JSON dump
    def result_json(r: SessionResult) -> dict:
        d = {
            "sample_id": r.sample_id, "transcript": r.transcript, "ttfs_ms": r.ttfs_ms,
            "processing_lag_ms": r.processing_lag_ms, "ready_latency_ms": r.ready_latency_ms,
            "audio_duration_s": r.audio_duration_s, "interim_count": r.interim_count,
            "error": r.error,
        }
        if args.strict_byte:
            base = baselines.get(r.sample_id)
            d.update({
                "interims": r.interims,
                "final_deltas": r.final_deltas,
                "first_interim_ms": r.first_interim_ms,
                "interim_lag_ms": r.interim_lag_ms,
                "strict_diffs": strict_diff(r, base) if isinstance(base, dict) else None,
            })
        baseline_text = (
            baselines[r.sample_id]["transcript"]  # type: ignore[index]
            if args.strict_byte and r.sample_id in baselines
            else baselines.get(r.sample_id)
        )
        d["edit_distance_vs_baseline"] = (
            levenshtein(r.transcript, baseline_text)
            if (not r.error and isinstance(baseline_text, str)) else None
        )
        return d

    out = {
        "config": {"url": args.url, "sweep": args.sweep, "sample_count": audio_count,
                   "run_all_concurrency": args.run_all_concurrency,
                   "strict_byte": args.strict_byte, "start_jitter_ms": START_JITTER_MS,
                   "trailing_silence_ms": TRAILING_SILENCE_MS, "chunk_ms": CHUNK_MS,
                   "seed": args.seed},
        "baselines": baselines,
        "levels": {
            str(n): [result_json(r) for r in level]
            for n, level in results.items()
        },
        "run_all": [result_json(r) for r in canary_results],
        "summaries": summaries,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'N':>3} {'exact':>8} {'medED':>6} {'maxED':>6} {'TTFS p50':>9} {'TTFS p95':>9} "
          f"{'lag p50':>8} {'lag p95':>8} {'ready p95':>10}")
    for n, level in results.items():
        eds = []
        for r in level:
            if r.error or r.sample_id not in baselines:
                continue
            baseline_text = (
                baselines[r.sample_id]["transcript"]  # type: ignore[index]
                if args.strict_byte else baselines[r.sample_id]  # type: ignore[index]
            )
            eds.append(levenshtein(r.transcript, baseline_text))
        ttfs = [r.ttfs_ms for r in level if r.ttfs_ms is not None]
        lag = [r.processing_lag_ms for r in level if r.processing_lag_ms is not None]
        ready = [r.ready_latency_ms for r in level if r.ready_latency_ms is not None]
        exact = sum(1 for e in eds if e == 0)
        print(f"{n:>3} {f'{exact}/{len(eds)}':>8} "
              f"{statistics.median(eds) if eds else 0:>6.0f} {max(eds) if eds else 0:>6} "
              f"{pctl(ttfs,0.5) or 0:>9.1f} {pctl(ttfs,0.95) or 0:>9.1f} "
              f"{pctl(lag,0.5) or 0:>8.1f} {pctl(lag,0.95) or 0:>8.1f} {pctl(ready,0.95) or 0:>10.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--sweep", type=int, nargs="+", default=DEFAULT_SWEEP)
    ap.add_argument("--sample-count", type=int, default=0,
                    help="Number of distinct DB samples to select; default is max(24, max(sweep)).")
    ap.add_argument("--strict-byte", action="store_true",
                    help="Strictly compare interim sequence, final text, final deltas, and duplicate finals.")
    ap.add_argument("--skip-sweep", action="store_true",
                    help="Only capture baselines and optional --run-all-concurrency canary.")
    ap.add_argument("--run-all-concurrency", type=int, default=0,
                    help="After baselines, run all selected samples with this bounded concurrency.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", default=str(REPO_ROOT / "proj-2026-05-19-eou-endpointing" / "concurrency_test_results.json"))
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
