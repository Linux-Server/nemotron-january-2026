#!/usr/bin/env python3
"""Prepare the Step-2 EOU signal-collection subset and runbook.

This script does not start the ASR server and does not run stt-benchmark. It
only writes the documented subset manifest and the shell commands for a human
operator to run.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
SLICE_B_PATH = (
    REPO_ROOT
    / "stt-benchmark"
    / "stt_benchmark_data"
    / "slices"
    / "slice_B_duration_stratified_seed1234.json"
)
FORK_TELEMETRY_PATH = (
    REPO_ROOT
    / "stt-benchmark"
    / "stt_benchmark_data"
    / "client_telemetry"
    / "fork.jsonl"
)
RESULTS_DB_PATH = REPO_ROOT / "stt-benchmark" / "stt_benchmark_data" / "results.db"
DEFAULT_RUN_TAG = "eou_step2_collect"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sample_ids(path: Path) -> list[str]:
    payload = load_json(path)
    sample_ids = payload.get("sample_ids")
    if not isinstance(sample_ids, list) or not all(isinstance(item, str) for item in sample_ids):
        raise ValueError(f"{path} must contain a string list at key 'sample_ids'")
    return sample_ids


def load_augment_ids(path: Path) -> list[str]:
    payload = load_json(path)
    if isinstance(payload, list):
        sample_ids = payload
    elif isinstance(payload, dict):
        sample_ids = payload.get("sample_ids")
    else:
        sample_ids = None
    if not isinstance(sample_ids, list) or not all(isinstance(item, str) for item in sample_ids):
        raise ValueError(f"{path} must contain a string list or a 'sample_ids' string list")
    return sample_ids


def unique_preserving_order(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sample_id in ids:
        if sample_id in seen:
            continue
        seen.add(sample_id)
        ordered.append(sample_id)
    return ordered


def fork_multi_segment_summary(subset_ids: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "telemetry": repo_relative(FORK_TELEMETRY_PATH),
        "results_db": repo_relative(RESULTS_DB_PATH),
        "available": False,
        "subset_multi_segment_count": None,
        "fork_multi_segment_total": None,
        "definition": "final_transcription_frames > 1 in fork.jsonl",
    }
    if not FORK_TELEMETRY_PATH.exists() or not RESULTS_DB_PATH.exists():
        return summary

    with sqlite3.connect(RESULTS_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT sample_id FROM samples ORDER BY dataset_index"
        ).fetchall()
    batch_index_to_sample_id = {index: row[0] for index, row in enumerate(rows)}

    multi_segment_ids: set[str] = set()
    with FORK_TELEMETRY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            frames = int(record.get("final_transcription_frames") or 0)
            if frames <= 1:
                continue
            batch_index = record.get("benchmark_batch_index")
            if not isinstance(batch_index, int):
                continue
            sample_id = batch_index_to_sample_id.get(batch_index)
            if sample_id:
                multi_segment_ids.add(sample_id)

    subset_id_set = set(subset_ids)
    summary.update(
        {
            "available": True,
            "subset_multi_segment_count": len(subset_id_set & multi_segment_ids),
            "fork_multi_segment_total": len(multi_segment_ids),
        }
    )
    return summary


def build_subset_manifest(
    *,
    slice_b_payload: dict[str, Any],
    subset_ids: list[str],
    augment_ids: list[str],
) -> dict[str, Any]:
    explanation = (
        "Step 2 uses the duration-stratified slice-B subset as the documented "
        "offline collection set. Explicit multi-segment augmentation is "
        "supported by this script, but the default is empty because the fork "
        "telemetry cross-reference already gives strong multi-segment coverage "
        "inside slice-B."
    )
    return {
        "manifest": {
            "source": repo_relative(SLICE_B_PATH),
            "source_name": slice_b_payload.get("name"),
            "source_method": slice_b_payload.get("method"),
            "count": len(subset_ids),
            "generated_at": utc_now_iso(),
            "explanation": explanation,
            "augmentation": {
                "explicit_multi_segment_ids": augment_ids,
                "default_augmentation_count": 0,
            },
            "fork_multi_segment_cross_reference": fork_multi_segment_summary(subset_ids),
        },
        "sample_ids": subset_ids,
    }


def build_runbook(
    *,
    run_tag: str,
    snapshot_dir: str,
    telemetry_dir: str,
    server_host: str,
    server_port: int,
    snapshot_every: int,
) -> str:
    benchmark_dir = REPO_ROOT / "stt-benchmark"
    server_url = f"ws://{server_host}:{server_port}"
    telemetry_dir_for_benchmark = (
        str(Path(telemetry_dir))
        if Path(telemetry_dir).is_absolute()
        else str((REPO_ROOT / telemetry_dir).resolve())
    )
    return f"""# Step 2 EOU Signal Collection Runbook

Generated by `proj-2026-05-19-eou-endpointing/collect_signals.py`.

This script does not start the server or run the benchmark. Run these commands manually.

## Subset

The documented analysis subset is `proj-2026-05-19-eou-endpointing/subset.json`.
It contains the 200 IDs from `stt-benchmark/stt_benchmark_data/slices/slice_B_duration_stratified_seed1234.json`.

CLI subsetting verdict: the current `stt-benchmark run` command has `--limit`, but no `--ids`,
`--sample-ids-file`, or equivalent sample-id selector. Therefore the production capture command
below runs the normal 1000-sample benchmark and captures snapshots for all sessions; downstream
Step 3+ analysis filters to the 200 IDs in `subset.json`.

## Start the server

```bash
cd {REPO_ROOT}
mkdir -p {snapshot_dir} {telemetry_dir} ./eou-collect/logs

env \\
  NEMOTRON_EOU_PROBE=1 \\
  NEMOTRON_EOU_SNAPSHOT_DIR={snapshot_dir} \\
  NEMOTRON_EOU_SNAPSHOT_EVERY={snapshot_every} \\
  NEMOTRON_EOU_CLIENT=1 \\
  NEMOTRON_CONTINUOUS=1 \\
  NEMOTRON_FINALIZE_SILENCE_MS=150 \\
  NEMOTRON_FORK_ASSERT=1 \\
  NEMOTRON_RUN_TAG={run_tag} \\
  NEMOTRON_TELEMETRY_DIR={telemetry_dir} \\
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python \\
    src/nemotron_speech/server.py \\
    --host {server_host} \\
    --port {server_port} \\
    --right-context 1 \\
  2>&1 | tee ./eou-collect/logs/{run_tag}.server.log
```

## Run the full capture

Use this for the real Step-2 collection. Do not score this as an authoritative WER gate; it is
an offline signal/snapshot capture run.

```bash
cd {benchmark_dir}

env \\
  NEMOTRON_EOU_PROBE=1 \\
  NEMOTRON_EOU_CLIENT=1 \\
  NEMOTRON_CONTINUOUS=1 \\
  NEMOTRON_FINALIZE_SILENCE_MS=150 \\
  NEMOTRON_RUN_TAG={run_tag} \\
  NEMOTRON_TELEMETRY_DIR={telemetry_dir_for_benchmark} \\
  NEMOTRON_LOCAL_URL={server_url} \\
  .venv/bin/stt-benchmark run \\
    --services nemotron_local \\
    --model {run_tag} \\
    --vad-stop-secs 0.2 \\
    --no-skip-existing
```

## Small-N debug run

Use this only to verify wiring and disk growth before the full capture. `--limit` selects the
first N database samples, not arbitrary IDs, so it is not a substitute for slice-B collection.

```bash
cd {benchmark_dir}

env \\
  NEMOTRON_EOU_PROBE=1 \\
  NEMOTRON_EOU_CLIENT=1 \\
  NEMOTRON_CONTINUOUS=1 \\
  NEMOTRON_FINALIZE_SILENCE_MS=150 \\
  NEMOTRON_RUN_TAG={run_tag}_debug \\
  NEMOTRON_TELEMETRY_DIR={telemetry_dir_for_benchmark} \\
  NEMOTRON_LOCAL_URL={server_url} \\
  .venv/bin/stt-benchmark run \\
    --services nemotron_local \\
    --model {run_tag}_debug \\
    --vad-stop-secs 0.2 \\
    --no-skip-existing \\
    --test \\
    --limit 3
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--augment-id", action="append", default=[])
    parser.add_argument("--augment-ids-file", type=Path)
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--snapshot-dir", default="./eou-collect/snapshots")
    parser.add_argument("--telemetry-dir", default="./eou-collect/telemetry")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8080)
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the would-write summary without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.snapshot_every < 1:
        raise ValueError("--snapshot-every must be >= 1")

    slice_b_payload = load_json(SLICE_B_PATH)
    slice_b_ids = load_sample_ids(SLICE_B_PATH)

    augment_ids = list(args.augment_id)
    if args.augment_ids_file is not None:
        augment_ids.extend(load_augment_ids(args.augment_ids_file))

    subset_ids = unique_preserving_order(slice_b_ids + augment_ids)
    subset_payload = build_subset_manifest(
        slice_b_payload=slice_b_payload,
        subset_ids=subset_ids,
        augment_ids=augment_ids,
    )
    runbook = build_runbook(
        run_tag=args.run_tag,
        snapshot_dir=args.snapshot_dir,
        telemetry_dir=args.telemetry_dir,
        server_host=args.server_host,
        server_port=args.server_port,
        snapshot_every=args.snapshot_every,
    )

    output_dir = args.output_dir
    subset_path = output_dir / "subset.json"
    runbook_path = output_dir / "runbook.md"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "subset_path": str(subset_path),
                    "runbook_path": str(runbook_path),
                    "subset_count": len(subset_ids),
                    "augmentation_count": len(augment_ids),
                    "fork_multi_segment_cross_reference": subset_payload["manifest"][
                        "fork_multi_segment_cross_reference"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path.write_text(json.dumps(subset_payload, indent=2, sort_keys=True) + "\n")
    runbook_path.write_text(runbook, encoding="utf-8")
    print(f"Wrote {subset_path}")
    print(f"Wrote {runbook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
