#!/usr/bin/env python3
"""Step 6a byte-equivalence harness for the current full-re-mel path.

This is a scratch oracle for Step 6b. It intentionally captures the current
server.py behavior instead of proposing an improved preprocessing path:

- load/configure the model exactly like probe_alias.py and server.py
- re-run the NeMo preprocessor over all accumulated audio for every chunk
- feed the same mel slice into conformer_stream_step that server.py uses
- persist the mel slices, chunk boundaries, drop_extra values, and text

The persisted goldens are compared with torch.equal for every mel tensor.
Production code is not imported or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from probe_alias import (
    DEFAULT_AUDIO_DIR,
    MODEL_NAME,
    RIGHT_CONTEXT,
    SAMPLE_RATE,
    ProbeConfig,
    ProbeSession,
    load_model_and_config,
    load_pcm_s16le,
    new_session,
    text_from_hypotheses,
    to_float_audio,
    warmup_like_server,
)


PROJECT_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = PROJECT_DIR / "equiv_golden"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
PARAMS_PATH = GOLDEN_DIR / "PARAMS.md"
SERVER_PATH = PROJECT_DIR.parents[0] / "src/nemotron_speech/server.py"

FIXTURE_ROLES = ("smallest", "mid", "largest")
BENCHMARK_CHUNK_MS = 20
SERVER_REPREPROCESS_SITES = {
    "streaming": "src/nemotron_speech/server.py:403-415",
    "final": "src/nemotron_speech/server.py:615-623",
}
SERVER_CHUNK_SITES = {
    "streaming": "src/nemotron_speech/server.py:421-462",
    "final": "src/nemotron_speech/server.py:625-668",
}


@dataclass(frozen=True)
class FixtureSpec:
    role: str
    filename: str
    path: str
    size_bytes: int
    samples: int
    duration_seconds: float
    sha256: str

    @property
    def artifact_name(self) -> str:
        return f"{self.role}_{Path(self.filename).stem}.pt"


def set_determinism() -> None:
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_fixtures(audio_dir: Path) -> list[FixtureSpec]:
    paths = sorted(audio_dir.glob("*.pcm"), key=lambda p: (p.stat().st_size, p.name))
    if len(paths) < 3:
        raise FileNotFoundError(f"Need at least 3 .pcm fixtures under {audio_dir}, found {len(paths)}")

    selected = [paths[0], paths[(len(paths) - 1) // 2], paths[-1]]
    fixtures: list[FixtureSpec] = []
    for role, path in zip(FIXTURE_ROLES, selected, strict=True):
        size_bytes = path.stat().st_size
        samples = size_bytes // 2
        fixtures.append(
            FixtureSpec(
                role=role,
                filename=path.name,
                path=str(path),
                size_bytes=size_bytes,
                samples=samples,
                duration_seconds=samples / SAMPLE_RATE,
                sha256=sha256_file(path),
            )
        )
    return fixtures


def fixture_from_manifest(item: dict[str, Any]) -> FixtureSpec:
    fixture = item["fixture"]
    path = DEFAULT_AUDIO_DIR / fixture["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Golden fixture missing from audio dir: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != fixture["sha256"]:
        raise RuntimeError(
            f"Fixture sha256 changed for {fixture['filename']}: "
            f"golden={fixture['sha256']} actual={actual_sha}"
        )
    return FixtureSpec(
        role=fixture["role"],
        filename=fixture["filename"],
        path=str(path),
        size_bytes=fixture["size_bytes"],
        samples=fixture["samples"],
        duration_seconds=fixture["duration_seconds"],
        sha256=fixture["sha256"],
    )


def tensor_to_golden(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def params_from_model(model: Any, cfg: ProbeConfig) -> dict[str, Any]:
    preprocessor_cfg = model.cfg.preprocessor
    window_stride = float(preprocessor_cfg.get("window_stride", 0.01))
    window_size = float(preprocessor_cfg.get("window_size", 0.02))
    hop_samples = int(window_stride * SAMPLE_RATE)
    window_size_samples = int(window_size * SAMPLE_RATE)
    overlap_samples = cfg.pre_encode_cache_size * cfg.hop_samples

    return {
        "model_name": MODEL_NAME,
        "right_context": RIGHT_CONTEXT,
        "att_context_size": [70, RIGHT_CONTEXT],
        "sample_rate": SAMPLE_RATE,
        "decoder_class": cfg.decoder_class,
        "decoding": {
            "strategy": "greedy",
            "loop_labels": False,
            "use_cuda_graph_decoder": False,
            "max_symbols": 10,
        },
        "dither": float(model.preprocessor.featurizer.dither),
        "window_stride": window_stride,
        "window_size": window_size,
        "hop_samples": hop_samples,
        "window_size_samples": window_size_samples,
        "shift_frames": cfg.shift_frames,
        "shift_samples": cfg.shift_frames * cfg.hop_samples,
        "shift_ms": cfg.shift_frames * window_stride * 1000,
        "pre_encode_cache_size": cfg.pre_encode_cache_size,
        "drop_extra_pre_encoded": cfg.drop_extra,
        "overlap_samples": overlap_samples,
        "overlap_ms": overlap_samples * 1000 / SAMPLE_RATE,
        "final_padding_frames": cfg.final_padding_frames,
        "final_padding_samples": cfg.final_padding_frames * cfg.hop_samples,
        "final_padding_formula": "(right_context + 1) * shift_frames",
        "benchmark_audio_chunk_ms": BENCHMARK_CHUNK_MS,
        "benchmark_audio_chunk_samples": int(SAMPLE_RATE * BENCHMARK_CHUNK_MS / 1000),
        "server_repreprocess_sites": SERVER_REPREPROCESS_SITES,
        "server_chunk_sites": SERVER_CHUNK_SITES,
    }


def process_one_chunk_capture(
    model: Any,
    cfg: ProbeConfig,
    session: ProbeSession,
    chunk_index: int,
) -> dict[str, Any] | None:
    """Mirror probe_alias.process_one_chunk and capture the exact mel slice."""

    audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).to(cfg.device)
    audio_len = torch.tensor([len(session.accumulated_audio)], device=cfg.device)

    with torch.inference_mode():
        mel, _mel_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)

        available_frames = mel.shape[-1] - 1
        new_frame_count = available_frames - session.emitted_frames
        if new_frame_count < cfg.shift_frames:
            return None

        emitted_before = session.emitted_frames
        current_text_before = session.current_text
        if session.emitted_frames == 0:
            chunk_start = 0
            chunk_end = cfg.shift_frames
            drop_extra = 0
        else:
            chunk_start = session.emitted_frames - cfg.pre_encode_cache_size
            chunk_end = session.emitted_frames + cfg.shift_frames
            drop_extra = cfg.drop_extra

        chunk_mel = mel[:, :, chunk_start:chunk_end]
        captured_mel = tensor_to_golden(chunk_mel)
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=cfg.device)

        (
            session.pred_out_stream,
            transcribed_texts,
            session.cache_last_channel,
            session.cache_last_time,
            session.cache_last_channel_len,
            session.previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=False,
            previous_hypotheses=session.previous_hypotheses,
            previous_pred_out=session.pred_out_stream,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True,
        )

    session.emitted_frames += cfg.shift_frames
    returned_text = text_from_hypotheses(transcribed_texts, session.current_text)
    emitted_text = returned_text if returned_text is not None and returned_text != session.current_text else None
    if emitted_text is not None:
        session.current_text = returned_text

    return {
        "chunk_index": chunk_index,
        "audio_samples_at_process": int(len(session.accumulated_audio)),
        "available_frames": int(available_frames),
        "new_frame_count": int(new_frame_count),
        "emitted_frames_before": int(emitted_before),
        "emitted_frames_after": int(session.emitted_frames),
        "chunk_start": int(chunk_start),
        "chunk_end": int(chunk_end),
        "drop_extra_pre_encoded": int(drop_extra),
        "keep_all_outputs": False,
        "mel": captured_mel,
        "mel_shape": list(captured_mel.shape),
        "mel_dtype": str(captured_mel.dtype),
        "returned_text": returned_text,
        "emitted_text": emitted_text,
        "current_text_before": current_text_before,
        "current_text_after": session.current_text,
    }


def feed_audio_capture(
    model: Any,
    cfg: ProbeConfig,
    session: ProbeSession,
    audio_float: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    audio_chunk_samples = int(SAMPLE_RATE * BENCHMARK_CHUNK_MS / 1000)
    for offset in range(0, len(audio_float), audio_chunk_samples):
        audio_chunk = audio_float[offset : offset + audio_chunk_samples]
        session.accumulated_audio = np.concatenate([session.accumulated_audio, audio_chunk])
        min_audio_for_chunk = (session.emitted_frames + cfg.shift_frames + 1) * cfg.hop_samples

        while len(session.accumulated_audio) >= min_audio_for_chunk:
            record = process_one_chunk_capture(model, cfg, session, len(records))
            if record is not None:
                records.append(record)
            min_audio_for_chunk = (session.emitted_frames + cfg.shift_frames + 1) * cfg.hop_samples
    return records


def flush_final_capture(model: Any, cfg: ProbeConfig, session: ProbeSession) -> dict[str, Any]:
    """Mirror server.py hard-reset padding plus _process_final_chunk capture."""

    original_text = session.current_text
    audio_samples_before_padding = len(session.accumulated_audio)
    padding_samples = 0
    if audio_samples_before_padding > 0:
        padding_samples = cfg.final_padding_frames * cfg.hop_samples
        silence_padding = np.zeros(padding_samples, dtype=np.float32)
        session.accumulated_audio = np.concatenate([session.accumulated_audio, silence_padding])

    if len(session.accumulated_audio) == 0:
        return {
            "audio_samples_before_padding": 0,
            "padding_samples": 0,
            "audio_samples_after_padding": 0,
            "emitted_frames_before": int(session.emitted_frames),
            "chunk_start": 0,
            "chunk_end": 0,
            "drop_extra_pre_encoded": 0,
            "keep_all_outputs": True,
            "total_mel_frames": 0,
            "remaining_frames": 0,
            "mel": torch.empty(0),
            "mel_shape": [0],
            "mel_dtype": str(torch.empty(0).dtype),
            "current_text_before": original_text,
            "final_text": "",
            "delta_text": "",
        }

    audio_tensor = torch.from_numpy(session.accumulated_audio).unsqueeze(0).to(cfg.device)
    audio_len = torch.tensor([len(session.accumulated_audio)], device=cfg.device)

    with torch.inference_mode():
        mel, _mel_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)

        total_mel_frames = mel.shape[-1]
        remaining_frames = total_mel_frames - session.emitted_frames
        if remaining_frames <= 0:
            return {
                "audio_samples_before_padding": int(audio_samples_before_padding),
                "padding_samples": int(padding_samples),
                "audio_samples_after_padding": int(len(session.accumulated_audio)),
                "emitted_frames_before": int(session.emitted_frames),
                "chunk_start": int(session.emitted_frames),
                "chunk_end": int(session.emitted_frames),
                "drop_extra_pre_encoded": 0,
                "keep_all_outputs": True,
                "total_mel_frames": int(total_mel_frames),
                "remaining_frames": int(remaining_frames),
                "mel": torch.empty(0),
                "mel_shape": [0],
                "mel_dtype": str(torch.empty(0).dtype),
                "current_text_before": original_text,
                "final_text": session.current_text,
                "delta_text": "",
            }

        emitted_before = session.emitted_frames
        if session.emitted_frames == 0:
            chunk_start = 0
            drop_extra = 0
        else:
            chunk_start = session.emitted_frames - cfg.pre_encode_cache_size
            drop_extra = cfg.drop_extra

        chunk_mel = mel[:, :, chunk_start:]
        captured_mel = tensor_to_golden(chunk_mel)
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=cfg.device)

        (
            session.pred_out_stream,
            transcribed_texts,
            session.cache_last_channel,
            session.cache_last_time,
            session.cache_last_channel_len,
            session.previous_hypotheses,
        ) = model.conformer_stream_step(
            processed_signal=chunk_mel,
            processed_signal_length=chunk_len,
            cache_last_channel=session.cache_last_channel,
            cache_last_time=session.cache_last_time,
            cache_last_channel_len=session.cache_last_channel_len,
            keep_all_outputs=True,
            previous_hypotheses=session.previous_hypotheses,
            previous_pred_out=session.pred_out_stream,
            drop_extra_pre_encoded=drop_extra,
            return_transcription=True,
        )

    final_text = text_from_hypotheses(transcribed_texts, session.current_text)
    session.current_text = final_text
    if final_text.startswith(original_text):
        delta_text = final_text[len(original_text) :].lstrip()
    else:
        delta_text = final_text

    return {
        "audio_samples_before_padding": int(audio_samples_before_padding),
        "padding_samples": int(padding_samples),
        "audio_samples_after_padding": int(len(session.accumulated_audio)),
        "emitted_frames_before": int(emitted_before),
        "chunk_start": int(chunk_start),
        "chunk_end": int(total_mel_frames),
        "drop_extra_pre_encoded": int(drop_extra),
        "keep_all_outputs": True,
        "total_mel_frames": int(total_mel_frames),
        "remaining_frames": int(remaining_frames),
        "mel": captured_mel,
        "mel_shape": list(captured_mel.shape),
        "mel_dtype": str(captured_mel.dtype),
        "current_text_before": original_text,
        "final_text": final_text,
        "delta_text": delta_text,
    }


def capture_fixture(model: Any, cfg: ProbeConfig, params: dict[str, Any], fixture: FixtureSpec) -> dict[str, Any]:
    audio_i16 = load_pcm_s16le(Path(fixture.path))
    audio_float = to_float_audio(audio_i16)
    session = new_session(model, fixture.filename)
    chunks = feed_audio_capture(model, cfg, session, audio_float)
    final = flush_final_capture(model, cfg, session)

    return {
        "schema_version": 1,
        "fixture": asdict(fixture),
        "params": params,
        "chunk_records": chunks,
        "final_record": final,
    }


def boundary_summary(data: dict[str, Any]) -> dict[str, Any]:
    chunks = data["chunk_records"]
    final = data["final_record"]
    return {
        "chunk_count": len(chunks),
        "chunk_boundaries": [
            {
                "chunk_index": rec["chunk_index"],
                "audio_samples_at_process": rec["audio_samples_at_process"],
                "emitted_frames_before": rec["emitted_frames_before"],
                "emitted_frames_after": rec["emitted_frames_after"],
                "chunk_start": rec["chunk_start"],
                "chunk_end": rec["chunk_end"],
                "drop_extra_pre_encoded": rec["drop_extra_pre_encoded"],
                "keep_all_outputs": rec["keep_all_outputs"],
                "mel_shape": rec["mel_shape"],
                "mel_dtype": rec["mel_dtype"],
            }
            for rec in chunks
        ],
        "texts": [
            {
                "chunk_index": rec["chunk_index"],
                "returned_text": rec["returned_text"],
                "emitted_text": rec["emitted_text"],
                "current_text_after": rec["current_text_after"],
            }
            for rec in chunks
        ],
        "final_boundary": {
            key: final[key]
            for key in (
                "audio_samples_before_padding",
                "padding_samples",
                "audio_samples_after_padding",
                "emitted_frames_before",
                "chunk_start",
                "chunk_end",
                "drop_extra_pre_encoded",
                "keep_all_outputs",
                "total_mel_frames",
                "remaining_frames",
                "mel_shape",
                "mel_dtype",
            )
        },
        "final_text": final["final_text"],
        "final_delta_text": final["delta_text"],
    }


def manifest_for_capture(params: dict[str, Any], captured: list[tuple[FixtureSpec, dict[str, Any]]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_by": "proj-2026-05-17-1708/equiv_harness.py",
        "purpose": "Step 6a golden oracle for current server.py full-re-mel streaming path",
        "determinism": {
            "torch_manual_seed": 0,
            "numpy_seed": 0,
            "torch_inference_mode": True,
            "cudnn_benchmark": False,
            "dither": 0.0,
        },
        "fixture_selection": {
            "audio_dir": str(DEFAULT_AUDIO_DIR),
            "method": "sort *.pcm by (size_bytes, filename); pick first, middle index (n-1)//2, last",
            "roles": list(FIXTURE_ROLES),
        },
        "params": params,
        "server_repreprocess_sites": SERVER_REPREPROCESS_SITES,
        "server_chunk_sites": SERVER_CHUNK_SITES,
        "artifacts": [
            {
                "fixture": asdict(fixture),
                "artifact": str(Path("equiv_golden") / fixture.artifact_name),
                **boundary_summary(data),
            }
            for fixture, data in captured
        ],
    }


def write_params_md(manifest: dict[str, Any]) -> None:
    params = manifest["params"]
    lines = [
        "# Step 6a Equivalence Harness Parameters",
        "",
        "This file is generated by `equiv_harness.py capture`. It documents the live values read from the model for Step 6b.",
        "",
        "## Server Reference Sites",
        "",
        f"- Streaming full re-preprocess: `{SERVER_REPREPROCESS_SITES['streaming']}`",
        f"- Final full re-preprocess: `{SERVER_REPREPROCESS_SITES['final']}`",
        f"- Streaming chunk slice/conformer call: `{SERVER_CHUNK_SITES['streaming']}`",
        f"- Final chunk slice/conformer call: `{SERVER_CHUNK_SITES['final']}`",
        "",
        "## Live Model Parameters",
        "",
        f"- `model_name`: `{params['model_name']}`",
        f"- `att_context_size`: `{params['att_context_size']}`",
        f"- `right_context`: `{params['right_context']}`",
        f"- `sample_rate`: `{params['sample_rate']}`",
        f"- `window_stride`: `{params['window_stride']}`",
        f"- `window_size`: `{params['window_size']}`",
        f"- `hop_samples`: `{params['hop_samples']}`",
        f"- `window_size_samples`: `{params['window_size_samples']}`",
        f"- `shift_frames`: `{params['shift_frames']}`",
        f"- `shift_samples`: `{params['shift_samples']}`",
        f"- `pre_encode_cache_size`: `{params['pre_encode_cache_size']}`",
        f"- `drop_extra_pre_encoded`: `{params['drop_extra_pre_encoded']}`",
        f"- `overlap_samples`: `{params['overlap_samples']}`",
        f"- `final_padding_frames`: `{params['final_padding_frames']}`",
        f"- `final_padding_samples`: `{params['final_padding_samples']}`",
        f"- `final_padding_formula`: `{params['final_padding_formula']}`",
        f"- `decoder_class`: `{params['decoder_class']}`",
        f"- `decoding.strategy`: `{params['decoding']['strategy']}`",
        f"- `decoding.loop_labels`: `{params['decoding']['loop_labels']}`",
        f"- `decoding.use_cuda_graph_decoder`: `{params['decoding']['use_cuda_graph_decoder']}`",
        f"- `dither`: `{params['dither']}`",
        f"- `benchmark_audio_chunk_ms`: `{params['benchmark_audio_chunk_ms']}`",
        "",
        "## Fixtures",
        "",
        "| role | filename | bytes | seconds | sha256 | chunks | final text |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for item in manifest["artifacts"]:
        fixture = item["fixture"]
        final_text = item["final_text"].replace("|", "\\|")
        lines.append(
            f"| {fixture['role']} | `{fixture['filename']}` | {fixture['size_bytes']} | "
            f"{fixture['duration_seconds']:.3f} | `{fixture['sha256']}` | "
            f"{item['chunk_count']} | {final_text!r} |"
        )
    lines.append("")
    PARAMS_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_capture(manifest: dict[str, Any], captured: list[tuple[FixtureSpec, dict[str, Any]]]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for fixture, data in captured:
        torch.save(data, GOLDEN_DIR / fixture.artifact_name)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_params_md(manifest)


def metadata_without_mels(data: dict[str, Any]) -> dict[str, Any]:
    chunks = []
    for rec in data["chunk_records"]:
        chunks.append({key: value for key, value in rec.items() if key != "mel"})
    final = {key: value for key, value in data["final_record"].items() if key != "mel"}
    return {
        "schema_version": data["schema_version"],
        "fixture": data["fixture"],
        "params": data["params"],
        "chunk_records": chunks,
        "final_record": final,
    }


def compare_captures(golden: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if metadata_without_mels(golden) != metadata_without_mels(actual):
        errors.append("metadata/text/boundary mismatch")

    golden_chunks = golden["chunk_records"]
    actual_chunks = actual["chunk_records"]
    if len(golden_chunks) != len(actual_chunks):
        errors.append(f"chunk count mismatch golden={len(golden_chunks)} actual={len(actual_chunks)}")
    else:
        for idx, (g_rec, a_rec) in enumerate(zip(golden_chunks, actual_chunks)):
            if not torch.equal(g_rec["mel"], a_rec["mel"]):
                errors.append(f"chunk {idx} mel tensor mismatch")
                break

    if not torch.equal(golden["final_record"]["mel"], actual["final_record"]["mel"]):
        errors.append("final mel tensor mismatch")

    return errors


def load_golden(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def print_table(rows: list[dict[str, Any]]) -> None:
    print("| fixture | role | chunks | mel tensors | boundaries/text | verdict |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['filename']} | {row['role']} | {row['chunks']} | "
            f"{row['mel_status']} | {row['metadata_status']} | {row['verdict']} |"
        )


def command_capture(args: argparse.Namespace) -> int:
    del args
    set_determinism()
    started = time.perf_counter()
    model, cfg = load_model_and_config()
    warmup_like_server(model, cfg)
    params = params_from_model(model, cfg)
    fixtures = select_fixtures(DEFAULT_AUDIO_DIR)

    captured: list[tuple[FixtureSpec, dict[str, Any]]] = []
    for fixture in fixtures:
        print(f"Capturing {fixture.role}: {fixture.filename} ({fixture.duration_seconds:.2f}s)")
        captured.append((fixture, capture_fixture(model, cfg, params, fixture)))

    manifest = manifest_for_capture(params, captured)
    save_capture(manifest, captured)

    print(f"\nWrote {MANIFEST_PATH}")
    print(f"Wrote {PARAMS_PATH}")
    for fixture, _data in captured:
        print(f"Wrote {GOLDEN_DIR / fixture.artifact_name}")
    print(f"Capture complete in {time.perf_counter() - started:.1f}s")
    return 0


def command_assert(args: argparse.Namespace) -> int:
    del args
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing golden manifest: {MANIFEST_PATH}")

    set_determinism()
    started = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model, cfg = load_model_and_config()
    warmup_like_server(model, cfg)
    params = params_from_model(model, cfg)

    rows: list[dict[str, Any]] = []
    all_ok = True
    for item in manifest["artifacts"]:
        fixture = fixture_from_manifest(item)
        artifact_path = PROJECT_DIR / item["artifact"]
        golden = load_golden(artifact_path)
        actual = capture_fixture(model, cfg, params, fixture)
        errors = compare_captures(golden, actual)
        mel_ok = not any("mel tensor" in error for error in errors)
        metadata_ok = not any("metadata/text/boundary" in error or "chunk count" in error for error in errors)
        ok = not errors
        all_ok = all_ok and ok
        rows.append(
            {
                "filename": fixture.filename,
                "role": fixture.role,
                "chunks": len(actual["chunk_records"]),
                "mel_status": "PASS" if mel_ok else "FAIL",
                "metadata_status": "PASS" if metadata_ok else "FAIL",
                "verdict": "PASS" if ok else "FAIL",
                "errors": errors,
            }
        )

    print("\nByte-identical golden assert:")
    print_table(rows)
    for row in rows:
        if row["errors"]:
            print(f"{row['filename']} errors: {', '.join(row['errors'])}")
    print(f"Overall verdict: {'PASS' if all_ok else 'FAIL'}")
    print(f"Assert complete in {time.perf_counter() - started:.1f}s")
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="capture and persist Step 6a goldens")
    capture.set_defaults(func=command_capture)
    assert_cmd = subparsers.add_parser("assert", help="re-run capture and assert byte-identical vs goldens")
    assert_cmd.set_defaults(func=command_assert)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
