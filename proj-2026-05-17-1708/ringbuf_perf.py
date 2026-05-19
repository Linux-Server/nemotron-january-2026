#!/usr/bin/env python3
"""Step 6c contained O(N) preprocessing probe for the constant-plan ring.

This does not start the server or run the benchmark. It loads the same
Nemotron model configuration used by the scratch Step 6 harnesses, drives the
server-equivalent preprocessor inputs on synthetic audio, and compares the
current fixed-K ring path against the old growing-reprocess-all path.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from equiv_harness import (  # noqa: E402
    build_fixed_preprocess_audio,
    ring_plan_from_model,
    update_raw_ring,
)
from probe_alias import SAMPLE_RATE, load_model_and_config  # noqa: E402


BENCHMARK_CHUNK_MS = 20
DEFAULT_SECONDS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


@dataclass
class Row:
    seconds: int
    chunks: int
    ring_mean_ms: float
    ring_p95_ms: float
    ring_total_s: float
    old_mean_ms: float
    old_p95_ms: float
    old_total_s: float


def set_determinism() -> None:
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


def synthetic_audio(seconds: int) -> np.ndarray:
    sample_count = seconds * SAMPLE_RATE
    t = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
    rng = np.random.default_rng(20260518)
    audio = (
        0.045 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.025 * np.sin(2.0 * np.pi * 880.0 * t)
        + 0.003 * rng.standard_normal(sample_count).astype(np.float32)
    )
    return np.ascontiguousarray(audio.astype(np.float32))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timed_preprocess(model: Any, cfg: Any, audio: np.ndarray, valid_samples: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).to(cfg.device)
    audio_len = torch.tensor([valid_samples], device=cfg.device, dtype=torch.long)
    with torch.inference_mode():
        mel, mel_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)
    del mel, mel_len, audio_tensor, audio_len
    torch.cuda.synchronize()
    return time.perf_counter() - start


def warmup_preprocessor(model: Any, cfg: Any, plan: Any, audio: np.ndarray) -> None:
    raw_ring = np.zeros(plan.raw_audio_ring_samples, dtype=np.float32)
    fixed_audio, valid_samples = build_fixed_preprocess_audio(
        plan,
        raw_ring,
        audio[: plan.new_audio_samples],
    )
    for _ in range(6):
        _ = timed_preprocess(model, cfg, fixed_audio, valid_samples)
        old_samples = min(len(audio), 8 * plan.new_audio_samples)
        _ = timed_preprocess(model, cfg, audio[:old_samples], old_samples)
    torch.cuda.empty_cache()


def process_schedule(total_samples: int, cfg: Any) -> list[int]:
    audio_chunk_samples = int(SAMPLE_RATE * BENCHMARK_CHUNK_MS / 1000)
    emitted_frames = 0
    received_samples = 0
    schedule: list[int] = []
    for offset in range(0, total_samples, audio_chunk_samples):
        received_samples = min(total_samples, offset + audio_chunk_samples)
        min_audio_for_chunk = (emitted_frames + cfg.shift_frames + 1) * cfg.hop_samples
        while received_samples >= min_audio_for_chunk:
            schedule.append(received_samples)
            emitted_frames += cfg.shift_frames
            min_audio_for_chunk = (emitted_frames + cfg.shift_frames + 1) * cfg.hop_samples
    return schedule


def measure_ring(model: Any, cfg: Any, plan: Any, audio: np.ndarray) -> tuple[list[float], int]:
    raw_ring = np.zeros(plan.raw_audio_ring_samples, dtype=np.float32)
    pending = np.array([], dtype=np.float32)
    total_audio_samples = 0
    emitted_frames = 0
    times: list[float] = []
    audio_chunk_samples = int(SAMPLE_RATE * BENCHMARK_CHUNK_MS / 1000)

    for offset in range(0, len(audio), audio_chunk_samples):
        audio_chunk = audio[offset : offset + audio_chunk_samples]
        pending = np.concatenate([pending, audio_chunk])
        total_audio_samples += len(audio_chunk)
        min_audio_for_chunk = (emitted_frames + cfg.shift_frames + 1) * cfg.hop_samples

        while total_audio_samples >= min_audio_for_chunk:
            if len(pending) < plan.new_audio_samples:
                raise RuntimeError(
                    f"pending underrun: pending={len(pending)}, need={plan.new_audio_samples}"
                )
            new_audio = pending[: plan.new_audio_samples]
            fixed_audio, valid_samples = build_fixed_preprocess_audio(plan, raw_ring, new_audio)
            times.append(timed_preprocess(model, cfg, fixed_audio, valid_samples))

            consumed_samples = cfg.shift_frames * cfg.hop_samples
            consumed_audio = pending[:consumed_samples]
            raw_ring = update_raw_ring(plan, raw_ring, consumed_audio)
            pending = pending[consumed_samples:]
            emitted_frames += cfg.shift_frames
            min_audio_for_chunk = (emitted_frames + cfg.shift_frames + 1) * cfg.hop_samples

    return times, emitted_frames


def measure_old(model: Any, cfg: Any, audio: np.ndarray) -> tuple[list[float], int]:
    times: list[float] = []
    emitted_frames = 0
    for received_samples in process_schedule(len(audio), cfg):
        times.append(timed_preprocess(model, cfg, audio[:received_samples], received_samples))
        emitted_frames += cfg.shift_frames
    return times, emitted_frames


def configured_seconds() -> tuple[int, ...]:
    configured = os.environ.get("RINGBUF_PERF_SECONDS", "").strip()
    if not configured:
        return DEFAULT_SECONDS
    values = tuple(int(item) for item in configured.split(",") if item.strip())
    if len(values) < 4:
        raise ValueError("RINGBUF_PERF_SECONDS must contain at least four comma-separated lengths")
    if sorted(values) != list(values) or len(set(values)) != len(values):
        raise ValueError("RINGBUF_PERF_SECONDS must be unique and increasing")
    return values


def fit_growth(rows: list[Row], attr: str, min_seconds: int) -> tuple[float, float]:
    selected = [row for row in rows if row.seconds >= min_seconds]
    x = np.log(np.asarray([row.seconds for row in selected], dtype=np.float64))
    y = np.log(np.asarray([getattr(row, attr) for row in selected], dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return float(slope), float(r2)


def print_table(rows: list[Row]) -> None:
    print()
    print("| length | chunks | ring mean ms | ring p95 ms | ring total s | old mean ms | old p95 ms | old total s |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.seconds}s | {row.chunks} | "
            f"{row.ring_mean_ms:.3f} | {row.ring_p95_ms:.3f} | {row.ring_total_s:.4f} | "
            f"{row.old_mean_ms:.3f} | {row.old_p95_ms:.3f} | {row.old_total_s:.4f} |"
        )


def summarize_times(seconds: int, ring_times: list[float], old_times: list[float]) -> Row:
    if len(ring_times) != len(old_times):
        raise RuntimeError(
            f"chunk count mismatch for {seconds}s: ring={len(ring_times)} old={len(old_times)}"
        )
    return Row(
        seconds=seconds,
        chunks=len(ring_times),
        ring_mean_ms=statistics.fmean(ring_times) * 1000.0,
        ring_p95_ms=percentile(ring_times, 95) * 1000.0,
        ring_total_s=float(sum(ring_times)),
        old_mean_ms=statistics.fmean(old_times) * 1000.0,
        old_p95_ms=percentile(old_times, 95) * 1000.0,
        old_total_s=float(sum(old_times)),
    )


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe.")
    set_determinism()

    model, cfg = load_model_and_config()
    plan = ring_plan_from_model(model, cfg)
    seconds_values = configured_seconds()
    max_audio = synthetic_audio(max(seconds_values))
    warmup_preprocessor(model, cfg, plan, max_audio)

    print("Step 6c ring-buffer preprocessing perf probe")
    print(
        "Plan: "
        f"K={plan.constant_preprocess_samples} samples "
        f"({plan.constant_preprocess_frames} frames), "
        f"raw_ring={plan.raw_audio_ring_samples}, align={plan.align_pad_samples}, "
        f"new_audio={plan.new_audio_samples}, first_mel={plan.first_mel_frame}"
    )
    print(
        "Timing scope: server-equivalent preprocessor calls only "
        "(fixed-K ring vs old accumulated-audio reprocess-all)."
    )

    rows: list[Row] = []
    for seconds in seconds_values:
        audio = max_audio[: seconds * SAMPLE_RATE]
        gc.collect()
        torch.cuda.empty_cache()
        ring_times, ring_emitted = measure_ring(model, cfg, plan, audio)
        gc.collect()
        torch.cuda.empty_cache()
        old_times, old_emitted = measure_old(model, cfg, audio)
        if ring_emitted != old_emitted:
            raise RuntimeError(
                f"emitted frame mismatch for {seconds}s: ring={ring_emitted} old={old_emitted}"
            )
        rows.append(summarize_times(seconds, ring_times, old_times))

    print_table(rows)

    asymptotic_min_seconds = 64 if max(seconds_values) >= 256 else 4
    ring_slope, ring_r2 = fit_growth(rows, "ring_total_s", asymptotic_min_seconds)
    old_slope, old_r2 = fit_growth(rows, "old_total_s", asymptotic_min_seconds)
    ring_mean_slope, ring_mean_r2 = fit_growth(rows, "ring_mean_ms", asymptotic_min_seconds)
    old_mean_slope, old_mean_r2 = fit_growth(rows, "old_mean_ms", asymptotic_min_seconds)

    print()
    print(f"Fitted growth on {asymptotic_min_seconds}-{max(seconds_values)}s points:")
    print(f"  ring total: seconds^{ring_slope:.2f} (R^2={ring_r2:.3f})")
    print(f"  old total:  seconds^{old_slope:.2f} (R^2={old_r2:.3f})")
    print(f"  ring per-chunk mean: seconds^{ring_mean_slope:.2f} (R^2={ring_mean_r2:.3f})")
    print(f"  old per-chunk mean:  seconds^{old_mean_slope:.2f} (R^2={old_mean_r2:.3f})")

    ring_flat_ratio = rows[-1].ring_mean_ms / rows[0].ring_mean_ms
    old_grow_ratio = rows[-1].old_mean_ms / rows[0].old_mean_ms
    print()
    print(
        "Verdict: "
        f"ring per-chunk mean changes {ring_flat_ratio:.2f}x from {rows[0].seconds}s "
        f"to {rows[-1].seconds}s, "
        f"while old per-chunk mean grows {old_grow_ratio:.2f}x. "
        f"Ring total fits O(N) (slope {ring_slope:.2f}); "
        f"old total fits O(N^2) (slope {old_slope:.2f})."
    )

    if not (0.70 <= ring_slope <= 1.30 and old_slope >= 1.60 and old_grow_ratio >= 4.0):
        print("Probe verdict is not decisive enough; failing for rerun/inspection.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
