#!/usr/bin/env python3
"""Analyze a load-test run: drift over time + A/B comparison + admission shed.

Reads:
- stats_<backend>.jsonl   (sliding-window /stats snapshots every 30s)
- lb_showstat.jsonl        (HAProxy `show stat` every 30s)
- phase_summaries.jsonl    (per-phase success/duration)
- bench_<phase>.log        (per-run stdout from run_full1000_conc12.py)

Reports:
- Per-backend p50/p95/p99 of vad_stop_to_sent_ms at 6 checkpoints across the
  run (start, B-mid, B-end, D-mid, D-end, final).
- Drift = abs((D-end p95 / B-end p95) - 1).  Pass if drift < 15% AND p95
  doesn't trend upward across phases.
- Per-backend admission.rejected (cumulative) — was 1013-shedding stable?
- Box4 (stats-off A/B): only direct evidence is HAProxy show-stat traffic
  share + bench-client client-wall latency; we extract those.
- Bench-client client-wall TTFB by phase (grep `TTFB.*p95` from bench logs)
  — drift of client-perceived latency.
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_iso(s):
    return datetime.fromisoformat(s)


def load_stats_jsonl(path: Path) -> list[dict]:
    """One snapshot per line. Skip error/exception lines."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            if "stats" in d:
                out.append(d)
        except json.JSONDecodeError:
            pass
    return out


def pick_at_relative(snaps: list[dict], target_unix: float) -> dict | None:
    """Find the snapshot closest to target unix time."""
    best = None
    best_dt = float("inf")
    for s in snaps:
        ts = parse_iso(s["ts"]).timestamp()
        dt = abs(ts - target_unix)
        if dt < best_dt:
            best_dt = dt
            best = s
    return best


def fmt_p(snap, metric, key):
    if not snap:
        return "—"
    v = snap["stats"]["metrics"].get(metric, {}).get(key)
    return f"{v:.1f}" if v is not None else "—"


def fmt_active(snap, key):
    if not snap:
        return "—"
    v = snap["stats"]["active_sessions_at_emit"].get(key)
    return f"{v:.0f}" if v is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir).resolve()

    manifest = json.loads((d / "manifest.json").read_text())
    backends = list(manifest["backends"].keys())
    started = parse_iso(manifest["started"]).timestamp()

    # --- Per-phase checkpoint times ---
    phase_summaries = []
    for line in (d / "phase_summaries.jsonl").read_text().splitlines():
        if line.strip():
            phase_summaries.append(json.loads(line))
    phase_by_name = {p["phase"]: p for p in phase_summaries}

    # 6 checkpoints
    def end_of(name):
        return parse_iso(phase_by_name[name]["ended"]).timestamp() if name in phase_by_name else None

    checkpoints = [
        ("init",       started + 5),
        ("A_end",      end_of("A_warmup_20")),
        ("B_mid",      (parse_iso(phase_by_name["B_sustained_50_a"]["started"]).timestamp() +
                        parse_iso(phase_by_name["B_sustained_50_a"]["ended"]).timestamp()) / 2,
                       ) if "B_sustained_50_a" in phase_by_name else ("B_mid", None),
        ("B_end",      end_of("B_sustained_50_a")),
        ("D_mid",      (parse_iso(phase_by_name["D_sustained_50_b"]["started"]).timestamp() +
                        parse_iso(phase_by_name["D_sustained_50_b"]["ended"]).timestamp()) / 2,
                       ) if "D_sustained_50_b" in phase_by_name else ("D_mid", None),
        ("D_end",      end_of("D_sustained_50_b")),
        ("F_end",      end_of("F_cooldown")),
    ]
    # Normalize: each checkpoint is (label, unix_ts_or_None)
    checkpoints = [(label_or_t[0] if isinstance(label_or_t, tuple) else cps[0], (label_or_t[1] if isinstance(label_or_t, tuple) else label_or_t))
                   for cps, label_or_t in zip([("init", started + 5)] * 7, checkpoints)]
    # Above is gnarly — simpler explicit list:
    checkpoints = []
    checkpoints.append(("init",  started + 5))
    checkpoints.append(("A_end", end_of("A_warmup_20")))
    if "B_sustained_50_a" in phase_by_name:
        s = parse_iso(phase_by_name["B_sustained_50_a"]["started"]).timestamp()
        e = parse_iso(phase_by_name["B_sustained_50_a"]["ended"]).timestamp()
        checkpoints.append(("B_mid", (s + e) / 2))
        checkpoints.append(("B_end", e))
    if "D_sustained_50_b" in phase_by_name:
        s = parse_iso(phase_by_name["D_sustained_50_b"]["started"]).timestamp()
        e = parse_iso(phase_by_name["D_sustained_50_b"]["ended"]).timestamp()
        checkpoints.append(("D_mid", (s + e) / 2))
        checkpoints.append(("D_end", e))
    checkpoints.append(("F_end", end_of("F_cooldown")))

    # --- Phase summary table ---
    print("\n=== PHASE SUMMARIES ===")
    print(f"  {'phase':<28s} {'runs':>5s} {'all_ok':>7s} {'dur(s)':>8s}")
    total_runs = sum(p["runs"] for p in phase_summaries)
    any_failed = any(not p["all_ok"] for p in phase_summaries)
    for p in phase_summaries:
        print(f"  {p['phase']:<28s} {p['runs']:>5d} {str(p['all_ok']):>7s} {p['actual_dur_s']:>8.1f}")
    print(f"  {'TOTAL':<28s} {total_runs:>5d} {str(not any_failed):>7s}")

    # --- Per-backend stats at checkpoints ---
    print("\n=== /stats vad_stop_to_sent_ms p50 / p95 / p99 AT CHECKPOINTS ===")
    print("  metric: server-side TTFS (ms) — pure processing, no network RTT\n")
    drift = {}  # backend -> phase D-end p95 / phase B-end p95
    for backend in backends:
        path = d / f"stats_{backend}.jsonl"
        if not path.exists():
            print(f"  [{backend}] no stats file (control? skip)"); continue
        snaps = load_stats_jsonl(path)
        if not snaps:
            print(f"  [{backend}] no stats snapshots (control? skip)"); continue
        # Check if disabled
        if snaps[-1]["stats"].get("enabled") is False:
            print(f"  [{backend}] /stats DISABLED (A/B control — no per-server numbers)"); continue
        print(f"  [{backend}]   {'cp':<10s} {'p50':>8s} {'p95':>8s} {'p99':>8s}  {'samples':>7s} {'act_p95':>7s}")
        cp_p95 = {}
        for label, t in checkpoints:
            if t is None:
                continue
            snap = pick_at_relative(snaps, t)
            p50 = fmt_p(snap, "vad_stop_to_sent_ms", "p50")
            p95 = fmt_p(snap, "vad_stop_to_sent_ms", "p95")
            p99 = fmt_p(snap, "vad_stop_to_sent_ms", "p99")
            samples = snap["stats"]["samples"] if snap else "—"
            act_p95 = fmt_active(snap, "p95")
            print(f"               {label:<10s} {p50:>8s} {p95:>8s} {p99:>8s}  {samples:>7}  {act_p95:>7s}")
            if snap and snap["stats"]["metrics"]["vad_stop_to_sent_ms"]["p95"] is not None:
                cp_p95[label] = snap["stats"]["metrics"]["vad_stop_to_sent_ms"]["p95"]
        if "B_end" in cp_p95 and "D_end" in cp_p95:
            drift_pct = (cp_p95["D_end"] - cp_p95["B_end"]) / cp_p95["B_end"] * 100
            drift[backend] = drift_pct
            print(f"               drift D_end vs B_end p95: {drift_pct:+.1f}%")

    # --- Drift verdict ---
    print("\n=== DRIFT VERDICT (phase D-end p95 vs phase B-end p95) ===")
    if not drift:
        print("  no stats-enabled backends had both B_end and D_end snapshots")
    else:
        for b, pct in drift.items():
            ok = abs(pct) < 15
            print(f"  {b}: {pct:+.1f}%   {'PASS (<15%)' if ok else 'FAIL (>=15%)'}")

    # --- Admission rejections cumulative ---
    print("\n=== ADMISSION REJECTIONS (cumulative across the 2h run) ===")
    print("  (1013-style server-side admission shed — protects p95 under overload)")
    for backend in backends:
        path = d / f"stats_{backend}.jsonl"
        if not path.exists():
            continue
        snaps = load_stats_jsonl(path)
        if not snaps:
            continue
        latest = snaps[-1]["stats"]["admission"]
        print(f"  {backend}: attempted={latest['attempted']} admitted={latest['admitted']} rejected={latest['rejected']}  signal_backlog_now={latest['signal']['backlog_count']}")

    # --- LB show stat: per-backend traffic share (validates leastconn) ---
    print("\n=== LB show stat: final per-backend traffic share ===")
    rows = []
    for line in (d / "lb_showstat.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if "rows" in rec:
                rows.append(rec)
        except json.JSONDecodeError:
            pass
    if rows:
        last = rows[-1]
        total_stot = sum(int(r["stot"] or 0) for r in last["rows"] if r["svname"] not in ("BACKEND", "FRONTEND"))
        for r in last["rows"]:
            if r["svname"] in ("BACKEND", "FRONTEND"):
                continue
            stot = int(r["stot"] or 0)
            share = (stot / total_stot * 100) if total_stot else 0
            print(f"  {r['svname']:<22s} status={r['status']:<8s} stot={stot:>7d} ({share:5.1f}%)")
        print(f"  {'TOTAL':<22s} stot={total_stot}")

    # --- Bench-client client-wall TTFB by phase ---
    print("\n=== BENCH CLIENT client-wall TTFB by phase (mean / p50 / p95) ===")
    print("  (server time + ~24ms WAN RTT from California to us-west-2)\n")
    print(f"  {'phase':<28s} {'runs':>4s} {'TTFB_p50':>8s} {'TTFB_p95':>8s} {'srv_p50':>7s} {'srv_p95':>7s}  {'ok':>5s} {'err':>4s}")
    pat = re.compile(
        r"ok=(\d+) errors=(\d+) timed_out/empty=(\d+).*?"
        r"TTFB \(speech-end->final, ms\): mean=([\d\.]+) p50=([\d\.]+) p95=([\d\.]+) p99=([\d\.]+)\s*\n"
        r"server finalize \(vad_stop->final, ms\): p50=([\d\.]+) p95=([\d\.]+)",
        re.DOTALL,
    )
    bench_phase_files = sorted((d).glob("bench_*.log"))
    for f in bench_phase_files:
        text = f.read_text()
        matches = pat.findall(text)
        if not matches:
            print(f"  {f.stem.replace('bench_',''):<28s} (no parseable run summaries)")
            continue
        # Aggregate across runs of this phase
        ok = sum(int(m[0]) for m in matches)
        err = sum(int(m[1]) for m in matches)
        # Each run gives a p50/p95 of its own batch; aggregate by averaging
        ttfb_p50 = statistics.median(float(m[4]) for m in matches)
        ttfb_p95 = statistics.median(float(m[5]) for m in matches)
        srv_p50 = statistics.median(float(m[7]) for m in matches)
        srv_p95 = statistics.median(float(m[8]) for m in matches)
        phase_name = f.stem.replace("bench_", "")
        print(f"  {phase_name:<28s} {len(matches):>4d} {ttfb_p50:>8.1f} {ttfb_p95:>8.1f} {srv_p50:>7.1f} {srv_p95:>7.1f}  {ok:>5d} {err:>4d}")


if __name__ == "__main__":
    main()
