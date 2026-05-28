#!/usr/bin/env python3
"""2-hour load-test orchestrator for the Phase-1 L40S cluster.

Runs the existing `proj-2026-05-19-eou-endpointing/run_full1000_conc12.py`
bench client through the HAProxy LB in a phased schedule, while polling
each backend's `/stats` endpoint and the LB's `show stat` every 30s.

Designed to be repeatable: invoking with the same args reproduces the same
test (provided the cluster topology matches LOAD-TEST-NOTES.md).

Usage from repo root:

    python3 proj-2026-05-27-l40s-cluster-deploy/load_test_orchestrator.py \\
        --lb-ip 16.147.220.55 \\
        --backend-pubs 35.89.23.14,54.185.112.230,54.212.183.230,44.251.189.250 \\
        --backend-names box1,box2,box3,box4 \\
        --ssh-key ec2-bench/nemotron-bench-key.pem \\
        --out-dir proj-2026-05-27-l40s-cluster-deploy/load_test_run \\
        --short    # for testing the harness (~3 min); omit for the full 2h run
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


# ----- Phase definitions ---------------------------------------------------

# Each phase: (name, concurrency, limit_per_run, total_seconds, repeat_runs_until_total)
# limit_per_run=0 means "all 1000 samples"
# total_seconds = how long this phase should run (orchestrator loops the
#   bench client until total_seconds elapses unless repeat_runs_until_total
#   is False, in which case a single run is launched and we wait for it).
FULL_PLAN = [
    # name,                conc, limit, secs,  loop?
    ("A_warmup_5",            5,    20,   60, True),
    ("A_warmup_10",          10,    20,  120, True),
    ("A_warmup_20",          20,    20,  120, True),
    ("B_sustained_50_a",     50,    50, 3000, True),   # 50 min
    ("C_burst_50",           50,    20,  120, True),
    ("C_burst_5",             5,    20,  120, True),
    ("C_burst_50_b",         50,    20,  120, True),
    ("C_burst_5_b",           5,    20,  120, True),
    ("C_burst_50_c",         50,    20,  120, True),
    ("D_sustained_50_b",     50,    50, 2700, True),   # 45 min
    ("E_full_stt",           50,     0,  900, False),  # full 1000-sample bench at conc=50, one shot
    ("F_cooldown",            5,    20,  300, True),   # 5 min
]

# Short version for harness debugging (~3 minutes total).
SHORT_PLAN = [
    ("A_warmup_5",            5,    10,   30, True),
    ("B_sustained_20",       20,    10,   60, True),
    ("D_sustained_20_b",     20,    10,   60, True),
    ("F_cooldown",            5,    10,   30, True),
]


# ----- Polling -------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def poll_backend_stats(
    backend_name: str,
    backend_pub_ip: str,
    ssh_key: str,
    out_path: Path,
    interval: float,
    stop_event: threading.Event,
) -> None:
    """SSH into a backend every `interval` seconds, curl /stats, append to JSONL."""
    cmd_tmpl = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=5",
        f"ubuntu@{backend_pub_ip}",
        "curl -fsS --max-time 5 http://127.0.0.1:8080/stats",
    ]
    with out_path.open("a", buffering=1) as f:
        while not stop_event.is_set():
            t0 = time.time()
            try:
                r = subprocess.run(cmd_tmpl, capture_output=True, timeout=15)
                if r.returncode == 0 and r.stdout:
                    record = {
                        "ts": now_iso(),
                        "backend": backend_name,
                        "stats": json.loads(r.stdout),
                    }
                    f.write(json.dumps(record, default=str) + "\n")
                else:
                    f.write(json.dumps({"ts": now_iso(), "backend": backend_name, "error": r.stderr.decode("utf-8", errors="replace")[:200]}) + "\n")
            except Exception as e:
                f.write(json.dumps({"ts": now_iso(), "backend": backend_name, "exception": str(e)}) + "\n")
            elapsed = time.time() - t0
            stop_event.wait(max(0.0, interval - elapsed))


def poll_lb_show_stat(
    lb_pub_ip: str,
    ssh_key: str,
    out_path: Path,
    interval: float,
    stop_event: threading.Event,
) -> None:
    """SSH into LB, run `show stat` via socat, append parsed rows to JSONL."""
    cmd = [
        "ssh",
        "-i", ssh_key,
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ConnectTimeout=5",
        f"ubuntu@{lb_pub_ip}",
        "echo 'show stat' | sudo socat /run/haproxy/admin.sock stdio",
    ]
    with out_path.open("a", buffering=1) as f:
        while not stop_event.is_set():
            t0 = time.time()
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=15)
                if r.returncode == 0 and r.stdout:
                    raw = r.stdout.decode("utf-8", errors="replace")
                    # Parse the CSV: first line is "# pxname,svname,..."; strip leading "# ".
                    lines = [ln for ln in raw.splitlines() if ln.strip()]
                    if lines and lines[0].startswith("#"):
                        lines[0] = lines[0].lstrip("# ").strip()
                    import csv as _csv
                    rdr = _csv.DictReader(lines)
                    rows = [row for row in rdr if row.get("pxname") == "asr_pool"]
                    snapshot = {
                        "ts": now_iso(),
                        "rows": [
                            {
                                "svname": row.get("svname"),
                                "status": row.get("status"),
                                "scur": row.get("scur"),
                                "smax": row.get("smax"),
                                "stot": row.get("stot"),
                                "bin": row.get("bin"),
                                "bout": row.get("bout"),
                            }
                            for row in rows
                        ],
                    }
                    f.write(json.dumps(snapshot) + "\n")
                else:
                    f.write(json.dumps({"ts": now_iso(), "error": r.stderr.decode("utf-8", errors="replace")[:200]}) + "\n")
            except Exception as e:
                f.write(json.dumps({"ts": now_iso(), "exception": str(e)}) + "\n")
            elapsed = time.time() - t0
            stop_event.wait(max(0.0, interval - elapsed))


# ----- Bench-client phase runner -------------------------------------------


def run_bench(
    lb_url: str,
    concurrency: int,
    limit: int,
    model_tag: str,
    log_path: Path,
    bench_client_path: Path,
    timeout: float,
    bench_python: str,
) -> subprocess.CompletedProcess:
    """Run run_full1000_conc12.py once; capture stdout/stderr to log."""
    cmd = [
        bench_python,
        str(bench_client_path),
        "--url", lb_url,
        "--concurrency", str(concurrency),
        "--limit", str(limit),
        "--model-tag", model_tag,
    ]
    with log_path.open("a") as f:
        f.write(f"\n========= {now_iso()} START {model_tag} conc={concurrency} limit={limit} =========\n")
        f.write("CMD: " + " ".join(cmd) + "\n")
        f.flush()
        try:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, text=True)
        except subprocess.TimeoutExpired as e:
            f.write(f"\n[TIMEOUT after {timeout}s]\n")
            return subprocess.CompletedProcess(cmd, returncode=124, stdout=None, stderr=None)
        f.write(f"========= {now_iso()} END rc={r.returncode} =========\n")
        return r


def run_phase(
    phase: tuple[str, int, int, int, bool],
    lb_url: str,
    bench_client_path: Path,
    out_dir: Path,
    summary_path: Path,
    bench_python: str,
) -> None:
    name, conc, limit, secs, loop = phase
    bench_log = out_dir / f"bench_{name}.log"
    phase_start = time.time()
    phase_start_iso = now_iso()
    runs = 0
    rcs: list[int] = []

    print(f"[{now_iso()}] === PHASE START {name} conc={conc} limit={limit} budget={secs}s loop={loop}", flush=True)

    if loop:
        while time.time() - phase_start < secs:
            tag = f"{name}_run{runs:03d}_{datetime.now().strftime('%H%M%S')}"
            # Cap per-run timeout at remaining budget * 1.5 (give a little extra) or 1800s
            remaining = secs - (time.time() - phase_start)
            r_timeout = min(1800.0, max(60.0, remaining * 1.5 + 60.0))
            r = run_bench(lb_url, conc, limit, tag, bench_log, bench_client_path, r_timeout, bench_python)
            rcs.append(r.returncode)
            runs += 1
            # Brief breath between runs
            time.sleep(1)
    else:
        # Single run; let it finish even if it overruns
        tag = f"{name}_run000_{datetime.now().strftime('%H%M%S')}"
        r = run_bench(lb_url, conc, limit, tag, bench_log, bench_client_path, max(1800.0, secs * 2.0), bench_python)
        rcs.append(r.returncode)
        runs += 1

    phase_dur = time.time() - phase_start
    summary = {
        "phase": name,
        "concurrency": conc,
        "limit_per_run": limit,
        "budget_s": secs,
        "actual_dur_s": round(phase_dur, 1),
        "runs": runs,
        "return_codes": rcs,
        "all_ok": all(c == 0 for c in rcs),
        "started": phase_start_iso,
        "ended": now_iso(),
    }
    with summary_path.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"[{now_iso()}] === PHASE DONE  {name} runs={runs} all_ok={summary['all_ok']} dur={phase_dur:.1f}s", flush=True)


# ----- Main ----------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lb-ip", required=True, help="LB host public IP (workstation routes to LB)")
    ap.add_argument("--lb-port", type=int, default=8080, help="LB front port (default 8080 for plain ws://)")
    ap.add_argument("--backend-pubs", required=True, help="Comma-separated backend public IPs for SSH polling")
    ap.add_argument("--backend-names", required=True, help="Comma-separated friendly names matching backend-pubs order")
    ap.add_argument("--ssh-key", required=True, help="Path to SSH private key for ubuntu user")
    ap.add_argument("--bench-client", default="proj-2026-05-19-eou-endpointing/run_full1000_conc12.py",
                    help="Path to the WebSocket bench client")
    ap.add_argument("--bench-python", default="stt-benchmark/.venv/bin/python",
                    help="Python interpreter for the bench client (must have 'websockets' installed). "
                         "The orchestrator runs in any venv; the BENCH client needs a specific venv.")
    ap.add_argument("--out-dir", required=True, help="Output directory for logs + summaries")
    ap.add_argument("--poll-interval", type=float, default=30.0, help="Polling interval (sec) for /stats + show stat")
    ap.add_argument("--short", action="store_true", help="Use SHORT_PLAN (~3 min) for harness debugging")
    args = ap.parse_args()

    backend_pubs = [s.strip() for s in args.backend_pubs.split(",") if s.strip()]
    backend_names = [s.strip() for s in args.backend_names.split(",") if s.strip()]
    if len(backend_pubs) != len(backend_names):
        sys.exit("backend-pubs and backend-names must have the same length")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = SHORT_PLAN if args.short else FULL_PLAN
    lb_url = f"ws://{args.lb_ip}:{args.lb_port}"

    # Manifest at the top of the output dir documents the run.
    manifest = {
        "started": now_iso(),
        "lb_url": lb_url,
        "backends": dict(zip(backend_names, backend_pubs)),
        "poll_interval_s": args.poll_interval,
        "plan_kind": "SHORT" if args.short else "FULL",
        "plan": [
            {"name": n, "concurrency": c, "limit": l, "budget_s": s, "loop": loop}
            for (n, c, l, s, loop) in plan
        ],
        "bench_client": args.bench_client,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {out_dir / 'manifest.json'}")

    # Polling threads
    stop = threading.Event()
    threads: list[threading.Thread] = []
    for name, pub in zip(backend_names, backend_pubs):
        path = out_dir / f"stats_{name}.jsonl"
        t = threading.Thread(target=poll_backend_stats,
                             args=(name, pub, args.ssh_key, path, args.poll_interval, stop),
                             name=f"stats-{name}", daemon=True)
        t.start()
        threads.append(t)
    lb_path = out_dir / "lb_showstat.jsonl"
    t = threading.Thread(target=poll_lb_show_stat,
                         args=(args.lb_ip, args.ssh_key, lb_path, args.poll_interval, stop),
                         name="lb-showstat", daemon=True)
    t.start()
    threads.append(t)

    # Phase loop
    summary_path = out_dir / "phase_summaries.jsonl"
    overall_start = time.time()
    # Do NOT call .resolve() on bench_python — the venv shim's magic comes from
    # the symlink + adjacent pyvenv.cfg; following the symlink loses sys.path.
    bench_python = str(Path(args.bench_python).absolute())
    if not Path(bench_python).is_file():
        sys.exit(f"--bench-python not found: {bench_python}")
    try:
        for phase in plan:
            run_phase(phase, lb_url, Path(args.bench_client).resolve(), out_dir, summary_path, bench_python)
    except KeyboardInterrupt:
        print("interrupted; stopping pollers and exiting", flush=True)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)

    overall = time.time() - overall_start
    print(f"\nTOTAL wall time: {overall:.1f}s ({overall/60:.1f} min)")
    print(f"Logs + JSONL in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
