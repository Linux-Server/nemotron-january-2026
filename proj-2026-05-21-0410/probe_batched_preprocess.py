"""Probe Step 7a batched preprocessor safety.

Compare the server's fixed-K B=1 preprocessor path with a single batched
`model.preprocessor(input_signal=[B,K], length=[B])` call. The gate is fp32
allclose per row; bit equality is reported but not required by Step 7a.

Run:
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python \
    proj-2026-05-21-0410/probe_batched_preprocess.py
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
DB = REPO / "stt-benchmark" / "stt_benchmark_data" / "results.db"
MODEL_PATH_FILE = Path("/tmp/en-nemo-path")
SR = 16000
BATCHES = (2, 4, 8)
POSITIONS = (0, 1, 3, 7)
ATOL = 1e-4
RTOL = 1e-5


@dataclass(frozen=True)
class Geometry:
    hop_samples: int
    window_size_samples: int
    raw_audio_ring_samples: int
    preprocess_align_pad_samples: int
    preprocess_new_audio_samples: int
    constant_preprocess_frames: int
    constant_preprocess_samples: int
    first_preprocess_mel_frame: int
    shift_frames: int
    pre_encode_cache_size: int
    final_padding_frames: int


@dataclass(frozen=True)
class Clip:
    sample_id: str
    audio_path: str
    duration_seconds: float
    audio: np.ndarray


def log(message: str) -> None:
    print(message, flush=True)


def cfg_get(container: Any, key: str, default: Any = None) -> Any:
    try:
        if hasattr(container, "get"):
            return container.get(key, default)
        return getattr(container, key)
    except Exception:
        return default


def build_geometry(model: Any, right_context: int = 1) -> Geometry:
    scfg = model.encoder.streaming_cfg
    preprocessor_cfg = model.cfg.preprocessor
    hop_length_sec = cfg_get(preprocessor_cfg, "window_stride", 0.01)
    window_size_sec = cfg_get(preprocessor_cfg, "window_size", 0.025)
    featurizer = model.preprocessor.featurizer
    hop_samples = int(getattr(featurizer, "hop_length", int(hop_length_sec * SR)))
    window_size_samples = int(
        getattr(featurizer, "win_length", int(window_size_sec * SR))
    )
    shift_size = scfg.shift_size
    shift_frames = int(shift_size[1] if isinstance(shift_size, list) else shift_size)
    pre_cache = scfg.pre_encode_cache_size
    pre_encode_cache_size = int(pre_cache[1] if isinstance(pre_cache, list) else pre_cache)
    final_padding_frames = int((right_context + 1) * shift_frames)

    raw_audio_ring_samples = window_size_samples - hop_samples
    preprocess_align_pad_samples = (
        hop_samples - (raw_audio_ring_samples % hop_samples)
    ) % hop_samples
    preprocess_new_audio_samples = (shift_frames + 1) * hop_samples
    prefix_samples = preprocess_align_pad_samples + raw_audio_ring_samples
    if prefix_samples % hop_samples != 0:
        raise RuntimeError(
            "prefix is not hop-aligned: "
            f"prefix={prefix_samples} hop={hop_samples}"
        )
    first_preprocess_mel_frame = prefix_samples // hop_samples
    min_plan_frames = (
        first_preprocess_mel_frame
        + pre_encode_cache_size
        + shift_frames
        + final_padding_frames
        + 1
    )
    constant_preprocess_frames = 1 << (min_plan_frames - 1).bit_length()
    constant_preprocess_samples = (constant_preprocess_frames - 1) * hop_samples

    return Geometry(
        hop_samples=hop_samples,
        window_size_samples=window_size_samples,
        raw_audio_ring_samples=raw_audio_ring_samples,
        preprocess_align_pad_samples=preprocess_align_pad_samples,
        preprocess_new_audio_samples=preprocess_new_audio_samples,
        constant_preprocess_frames=constant_preprocess_frames,
        constant_preprocess_samples=constant_preprocess_samples,
        first_preprocess_mel_frame=first_preprocess_mel_frame,
        shift_frames=shift_frames,
        pre_encode_cache_size=pre_encode_cache_size,
        final_padding_frames=final_padding_frames,
    )


def build_fixed_preprocess_audio(
    geom: Geometry,
    raw_audio_ring: np.ndarray,
    new_audio: np.ndarray,
) -> tuple[np.ndarray, int]:
    if len(raw_audio_ring) != geom.raw_audio_ring_samples:
        raise ValueError(
            f"expected raw ring {geom.raw_audio_ring_samples}, got {len(raw_audio_ring)}"
        )
    valid_samples = (
        geom.preprocess_align_pad_samples + geom.raw_audio_ring_samples + len(new_audio)
    )
    if valid_samples > geom.constant_preprocess_samples:
        raise ValueError(
            f"fixed window valid span {valid_samples} exceeds K={geom.constant_preprocess_samples}"
        )
    audio = np.zeros(geom.constant_preprocess_samples, dtype=np.float32)
    cursor = geom.preprocess_align_pad_samples
    audio[cursor : cursor + geom.raw_audio_ring_samples] = raw_audio_ring
    cursor += geom.raw_audio_ring_samples
    audio[cursor : cursor + len(new_audio)] = new_audio
    return audio, valid_samples


def fixed_windows_by_position(
    geom: Geometry,
    audio: np.ndarray,
    positions: tuple[int, ...],
) -> dict[int, tuple[np.ndarray, int]]:
    max_position = max(positions)
    min_required = max_position * geom.shift_frames * geom.hop_samples
    min_required += geom.preprocess_new_audio_samples
    if len(audio) < min_required:
        raise ValueError(
            f"audio too short for position {max_position}: {len(audio)} < {min_required}"
        )

    pending = audio
    raw_ring = np.zeros(geom.raw_audio_ring_samples, dtype=np.float32)
    out: dict[int, tuple[np.ndarray, int]] = {}
    for position in range(max_position + 1):
        new_audio = pending[: geom.preprocess_new_audio_samples]
        if len(new_audio) != geom.preprocess_new_audio_samples:
            raise RuntimeError(
                f"position {position} has short new_audio {len(new_audio)}"
            )
        fixed_audio, valid_samples = build_fixed_preprocess_audio(
            geom,
            raw_ring,
            new_audio,
        )
        if position in positions:
            out[position] = (fixed_audio, valid_samples)

        consumed = pending[: geom.shift_frames * geom.hop_samples]
        if len(consumed) >= geom.raw_audio_ring_samples:
            raw_ring = consumed[-geom.raw_audio_ring_samples :].copy()
        else:
            keep = geom.raw_audio_ring_samples - len(consumed)
            raw_ring = np.concatenate([raw_ring[-keep:], consumed]).astype(
                np.float32,
                copy=False,
            )
        pending = pending[geom.shift_frames * geom.hop_samples :]
    return out


def load_clip(row: tuple[str, str, float]) -> Clip:
    sample_id, audio_path, duration_seconds = row
    path = REPO / "stt-benchmark" / audio_path
    if not path.exists():
        path = REPO / audio_path
    audio = np.frombuffer(path.read_bytes(), dtype=np.int16).astype(np.float32) / 32768.0
    return Clip(
        sample_id=sample_id,
        audio_path=audio_path,
        duration_seconds=float(duration_seconds),
        audio=audio,
    )


def select_clips(geom: Geometry, count: int) -> list[Clip]:
    min_required = max(POSITIONS) * geom.shift_frames * geom.hop_samples
    min_required += geom.preprocess_new_audio_samples
    min_duration = min_required / SR
    con = sqlite3.connect(DB)
    try:
        rows = con.execute(
            "SELECT sample_id, audio_path, duration_seconds FROM samples "
            "WHERE duration_seconds IS NOT NULL AND duration_seconds >= ? "
            "ORDER BY duration_seconds",
            (min_duration,),
        ).fetchall()
    finally:
        con.close()
    if len(rows) < count:
        raise SystemExit(
            f"need {count} clips >= {min_duration:.2f}s, found {len(rows)} in {DB}"
        )
    indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [load_clip(rows[index]) for index in indices]


def preprocess_b1(
    model: Any,
    audio: np.ndarray,
    valid_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).cuda()
    length = torch.tensor([valid_samples], device="cuda", dtype=torch.long)
    return model.preprocessor(input_signal=audio_tensor, length=length)


def preprocess_batch(
    model: Any,
    audios: list[np.ndarray],
    valid_samples: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = np.stack([np.ascontiguousarray(audio) for audio in audios], axis=0)
    audio_tensor = torch.from_numpy(stacked).cuda()
    length = torch.tensor(valid_samples, device="cuda", dtype=torch.long)
    return model.preprocessor(input_signal=audio_tensor, length=length)


def position_label(position: int) -> str:
    return "first" if position == 0 else f"steady{position}"


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not MODEL_PATH_FILE.exists():
        raise SystemExit(f"missing model path file: {MODEL_PATH_FILE}")
    model_path = MODEL_PATH_FILE.read_text().strip()

    import nemo.collections.asr as nemo_asr

    log(f"loading {model_path}")
    model = nemo_asr.models.ASRModel.restore_from(model_path, map_location="cuda")
    model.encoder.set_default_att_context_size([70, 1])
    model.eval()
    model.preprocessor.featurizer.dither = 0.0

    geom = build_geometry(model, right_context=1)
    log(
        "geometry "
        f"K={geom.constant_preprocess_samples} frames={geom.constant_preprocess_frames} "
        f"hop={geom.hop_samples} raw_ring={geom.raw_audio_ring_samples} "
        f"align={geom.preprocess_align_pad_samples} "
        f"new_audio={geom.preprocess_new_audio_samples} "
        f"first_mel={geom.first_preprocess_mel_frame} shift={geom.shift_frames}"
    )
    log("tf32 cuda.matmul=False cudnn=False")

    clips = select_clips(geom, max(BATCHES))
    log(
        "clips "
        + ", ".join(f"{clip.sample_id[:8]}:{clip.duration_seconds:.1f}s" for clip in clips)
    )
    windows = {
        clip.sample_id: fixed_windows_by_position(geom, clip.audio, POSITIONS)
        for clip in clips
    }

    # Warm the exact B=1 and max-B preprocessor shapes before comparing.
    zero = np.zeros(geom.constant_preprocess_samples, dtype=np.float32)
    with torch.inference_mode():
        preprocess_b1(model, zero, len(zero))
        preprocess_batch(
            model,
            [zero for _ in range(max(BATCHES))],
            [len(zero)] * max(BATCHES),
        )
        torch.cuda.synchronize()

    total_rows = 0
    equal_rows = 0
    allclose_rows = 0
    worst_diff = 0.0
    failures: list[str] = []

    with torch.inference_mode():
        for batch_size in BATCHES:
            b_rows = 0
            b_equal = 0
            b_allclose = 0
            b_worst = 0.0
            for position in POSITIONS:
                selected = clips[:batch_size]
                audios = [windows[clip.sample_id][position][0] for clip in selected]
                lengths = [windows[clip.sample_id][position][1] for clip in selected]
                batched_mel, batched_len = preprocess_batch(model, audios, lengths)
                per_position_equal = True
                per_position_allclose = True
                per_position_worst = 0.0
                per_position_len_equal = True
                for row_index, clip in enumerate(selected):
                    single_mel, single_len = preprocess_b1(
                        model,
                        audios[row_index],
                        lengths[row_index],
                    )
                    row_mel = batched_mel[row_index : row_index + 1]
                    row_len = batched_len[row_index : row_index + 1]
                    length_equal = torch.equal(row_len, single_len)
                    equal = torch.equal(row_mel, single_mel) and length_equal
                    close = (
                        torch.allclose(row_mel, single_mel, atol=ATOL, rtol=RTOL)
                        and length_equal
                    )
                    max_diff = float((row_mel - single_mel).abs().max().item())

                    total_rows += 1
                    b_rows += 1
                    if equal:
                        equal_rows += 1
                        b_equal += 1
                    if close:
                        allclose_rows += 1
                        b_allclose += 1
                    worst_diff = max(worst_diff, max_diff)
                    b_worst = max(b_worst, max_diff)
                    per_position_worst = max(per_position_worst, max_diff)
                    per_position_equal = per_position_equal and equal
                    per_position_allclose = per_position_allclose and close
                    per_position_len_equal = per_position_len_equal and length_equal
                    if not close:
                        failures.append(
                            f"B={batch_size} pos={position_label(position)} "
                            f"row={row_index} clip={clip.sample_id} "
                            f"len_equal={length_equal} max_abs={max_diff:.9g}"
                        )

                log(
                    f"B={batch_size} pos={position_label(position):>7} "
                    f"rows={batch_size} "
                    f"torch.equal={per_position_equal} "
                    f"allclose={per_position_allclose} "
                    f"mel_len_equal={per_position_len_equal} "
                    f"max_abs={per_position_worst:.9g}"
                )
            log(
                f"SUMMARY B={batch_size} rows={b_rows} "
                f"torch.equal_rows={b_equal}/{b_rows} "
                f"allclose_rows={b_allclose}/{b_rows} "
                f"max_abs={b_worst:.9g}"
            )

    log(
        f"OVERALL rows={total_rows} torch.equal_rows={equal_rows}/{total_rows} "
        f"allclose_rows={allclose_rows}/{total_rows} max_abs={worst_diff:.9g}"
    )
    if failures:
        log("FAILURES")
        for failure in failures:
            log(f"  {failure}")
        log("VERDICT NO-GO: batched preprocessor diverged beyond fp32 allclose gate")
        return 2

    log("VERDICT GO: batched preprocessor passed fp32 allclose gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
