#!/usr/bin/env python3
import argparse
import os
import re
import select
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
START = ROOT / "start_ws_server_local.sh"
ALNUM6 = re.compile(r"^[A-Za-z0-9]{6}$")
ACTIVE_SERVERS = []


class RunningServer:
    def __init__(self, label, proc, log_path):
        self.label = label
        self.proc = proc
        self.log_path = log_path
        self.lines = []
        self.port = None
        self.closed = False


def fail(message):
    print(f"TMP_HYGIENE_TEST FAIL {message}", flush=True)
    raise SystemExit(1)


def candidate_roots():
    out = set()
    euid = os.geteuid()
    tmp = Path("/tmp")
    for child in tmp.iterdir():
        if not ALNUM6.match(child.name):
            continue
        try:
            st = child.lstat()
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != euid:
            continue
        if has_aoti_marker(child):
            out.add(str(child))
    return out


def has_aoti_marker(root):
    try:
        for dirpath, dirnames, _ in os.walk(root, followlinks=False):
            if Path(dirpath).name == "data" and "aotinductor" in dirnames:
                return True
    except OSError:
        return False
    return False


def launch_server(label, log_dir, min_age=None):
    env = os.environ.copy()
    direct_ws = env.get("TMP_HYGIENE_TEST_DIRECT_WS") == "1"
    env["NEMOTRON_WS_BACKGROUND_WARMUP"] = "0"
    if min_age is None:
        env.pop("NEMOTRON_WS_TMP_HYGIENE_MIN_AGE_S", None)
    else:
        env["NEMOTRON_WS_TMP_HYGIENE_MIN_AGE_S"] = str(min_age)

    if direct_ws:
        repo = ROOT.parent.parent
        env["HF_HUB_OFFLINE"] = "1"
        env["PYTHONPATH"] = str(repo / "src")
        env["NEMOTRON_CONTINUOUS"] = "1"
        env["NEMOTRON_FINALIZE_SILENCE_MS"] = "0"
        env["NEMOTRON_ARTIFACT_DIR"] = str(ROOT / "artifacts")
        env["NEMOTRON_WS_SCHEDULER"] = "1"
        env["NEMOTRON_DENSITY_BATCH_STEADY"] = "1"
        env["NEMOTRON_DENSITY_BATCH_MAX"] = "4"
        env["NEMOTRON_DENSITY_BATCH_WINDOW_MS"] = "10"
        env["NEMOTRON_DENSITY_BATCH_LONE_TIMEOUT_MS"] = "0"
        env["NEMOTRON_DENSITY_ADMISSION_ACTIVE_CAP"] = "1"
        env["NEMOTRON_WS_LANES"] = "1"
        env["NEMOTRON_WS_FINALIZE_RUNNERS"] = "1"
        env["NEMOTRON_DENSITY_FINALIZE_RUNNERS"] = "1"
        cmd = [
            str(ROOT / "cpp" / "build_step10" / "ws_server"),
            "--port",
            "0",
            "--admission-active-cap",
            "1",
            "--steady-batch-dir",
            str(ROOT / "steady_b_artifacts"),
        ]
    else:
        env["PORT"] = "0"
        env["CAP"] = "1"
        env["LANES"] = "1"
        env["FINALIZE_RUNNERS"] = "1"
        env["SHADOW"] = "0"
        cmd = ["bash", str(START)]

    log_path = log_dir / f"{label}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    proc._tmp_hygiene_log_file = log_file
    server = RunningServer(label, proc, log_path)
    ACTIVE_SERVERS.append(server)
    return server


def read_available(server):
    proc = server.proc
    fd = proc.stdout.fileno()
    lines = []
    while True:
        ready, _, _ = select.select([fd], [], [], 0)
        if not ready:
            break
        line = proc.stdout.readline()
        if not line:
            break
        record_line(server, line)
        lines.append(line)
    return lines


def record_line(server, line):
    server.lines.append(line)
    server.proc._tmp_hygiene_log_file.write(line)
    server.proc._tmp_hygiene_log_file.flush()
    print(f"{server.label}: {line.rstrip()}", flush=True)
    if line.startswith("ws_server listening on 127.0.0.1:"):
        server.port = int(line.rsplit(":", 1)[1])


def wait_for(server, predicate, timeout_s):
    deadline = time.time() + timeout_s
    proc = server.proc
    fd = proc.stdout.fileno()
    while time.time() < deadline:
        if proc.poll() is not None:
            read_available(server)
            fail(f"{server.label} exited before condition rc={proc.returncode} log={server.log_path}")
        remaining = max(0.0, min(1.0, deadline - time.time()))
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            continue
        line = proc.stdout.readline()
        if not line:
            continue
        record_line(server, line)
        if predicate(server):
            return
    fail(f"{server.label} timed out after {timeout_s}s log={server.log_path}")


def wait_ready(server, timeout_s):
    wait_for(server, lambda s: s.port is not None, timeout_s)


def wait_hygiene_summary(server, timeout_s):
    wait_for(server, lambda s: any("TMP_HYGIENE_SUMMARY" in line for line in s.lines), timeout_s)


def kill_server(server):
    if server.closed:
        return
    if server.proc.poll() is None:
        os.killpg(server.proc.pid, signal.SIGKILL)
    server.proc.wait(timeout=60)
    read_available(server)
    server.proc._tmp_hygiene_log_file.close()
    server.closed = True
    if server in ACTIVE_SERVERS:
        ACTIVE_SERVERS.remove(server)


def terminate_server(server, timeout_s=240):
    if server.closed:
        return
    if server.proc.poll() is None:
        os.killpg(server.proc.pid, signal.SIGTERM)
        try:
            server.proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(server.proc.pid, signal.SIGKILL)
            server.proc.wait(timeout=60)
    read_available(server)
    server.proc._tmp_hygiene_log_file.close()
    server.closed = True
    if server in ACTIVE_SERVERS:
        ACTIVE_SERVERS.remove(server)


def assert_paths_exist(paths, label):
    missing = [p for p in sorted(paths) if not Path(p).exists()]
    if missing:
        fail(f"{label} missing paths={missing}")


def assert_no_extra_candidates(baseline, label):
    current = candidate_roots()
    extra = sorted(current - baseline)
    missing = sorted(baseline - current)
    if extra or missing:
        fail(f"{label} tmp mismatch extra={extra} missing_baseline={missing}")


def run(args):
    if not START.exists():
        fail(f"missing launcher {START}")

    ts = time.strftime("%Y%m%dT%H%M%S")
    log_dir = ROOT / "artifacts" / "tmp_hygiene_test_logs" / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"TMP_HYGIENE_TEST logs={log_dir}", flush=True)

    if args.preclean:
        preclean = launch_server("preclean_min_age0", log_dir, min_age=0)
        wait_hygiene_summary(preclean, args.hygiene_timeout_s)
        kill_server(preclean)
        time.sleep(1.0)
        if args.preclean_only:
            assert_no_extra_candidates(set(), "after_preclean")
            print("TMP_HYGIENE_TEST preclean_only=PASS", flush=True)
            return

    baseline = candidate_roots()
    print(f"TMP_HYGIENE_TEST baseline_count={len(baseline)} baseline={sorted(baseline)}", flush=True)

    leak_source = launch_server("leak_source", log_dir)
    wait_ready(leak_source, args.start_timeout_s)
    live_before_kill = candidate_roots() - baseline
    if not live_before_kill:
        fail("leak_source produced no owned /tmp/<6alnum>/data/aotinductor tree")
    print(f"TMP_HYGIENE_TEST leak_source_live={sorted(live_before_kill)}", flush=True)
    kill_server(leak_source)
    time.sleep(1.0)
    leaked = candidate_roots() - baseline
    if not leaked:
        fail("SIGKILL did not leave an observable leaked AOTI tree")
    print(f"TMP_HYGIENE_TEST leaked_after_sigkill={sorted(leaked)}", flush=True)

    fresh = launch_server("fresh_restart", log_dir)
    wait_hygiene_summary(fresh, args.hygiene_timeout_s)
    fresh_text = "".join(fresh.lines)
    if "reason=too_recent" not in fresh_text:
        fail(f"fresh restart did not log reason=too_recent log={fresh.log_path}")
    kill_server(fresh)
    time.sleep(1.0)
    assert_paths_exist(leaked, "fresh_restart_preserved_leak")
    print("TMP_HYGIENE_TEST fresh_leak_skipped_too_recent=PASS", flush=True)

    time.sleep(args.age_wait_s)
    aged = launch_server("aged_restart", log_dir, min_age=args.min_age_s)
    wait_hygiene_summary(aged, args.hygiene_timeout_s)
    aged_text = "".join(aged.lines)
    if "TMP_HYGIENE reclaimed dir=" not in aged_text:
        fail(f"aged restart did not reclaim any tree log={aged.log_path}")
    for path in leaked:
        if path not in aged_text:
            fail(f"aged restart did not log reclaim for leaked path {path} log={aged.log_path}")
    kill_server(aged)
    time.sleep(1.0)
    assert_no_extra_candidates(baseline, "after_aged_reclaim")
    print("TMP_HYGIENE_TEST aged_leak_reclaimed_and_baseline_restored=PASS", flush=True)

    live_primary = launch_server("live_primary", log_dir)
    wait_ready(live_primary, args.start_timeout_s)
    primary_live = candidate_roots() - baseline
    if not primary_live:
        fail("live_primary produced no live AOTI tree")
    print(f"TMP_HYGIENE_TEST primary_live={sorted(primary_live)}", flush=True)

    live_second = launch_server("live_second", log_dir)
    wait_hygiene_summary(live_second, args.hygiene_timeout_s)
    live_text = "".join(live_second.lines)
    if "TMP_HYGIENE skipped_live" not in live_text:
        fail(f"second startup did not report skipped_live log={live_second.log_path}")
    assert_paths_exist(primary_live, "live_second_preserved_primary_tree")
    kill_server(live_second)
    time.sleep(1.0)
    assert_paths_exist(primary_live, "live_second_after_kill_preserved_primary_tree")
    terminate_server(live_primary, args.stop_timeout_s)

    cleanup = launch_server("cleanup_after_live_min_age0", log_dir, min_age=0)
    wait_hygiene_summary(cleanup, args.hygiene_timeout_s)
    kill_server(cleanup)
    time.sleep(1.0)
    assert_no_extra_candidates(baseline, "after_live_safety")
    print("TMP_HYGIENE_TEST live_safety_skipped_live=PASS", flush=True)
    print("TMP_HYGIENE_TEST PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-timeout-s", type=float, default=900.0)
    parser.add_argument("--hygiene-timeout-s", type=float, default=120.0)
    parser.add_argument("--stop-timeout-s", type=float, default=240.0)
    parser.add_argument("--min-age-s", type=int, default=1)
    parser.add_argument("--age-wait-s", type=float, default=2.0)
    parser.add_argument("--skip-preclean", dest="preclean", action="store_false")
    parser.add_argument("--preclean-only", action="store_true")
    parser.set_defaults(preclean=True)
    args = parser.parse_args()
    try:
        run(args)
    finally:
        for server in list(reversed(ACTIVE_SERVERS)):
            try:
                kill_server(server)
            except Exception as exc:
                print(f"TMP_HYGIENE_TEST cleanup_error label={server.label} error={exc}", flush=True)


if __name__ == "__main__":
    main()
