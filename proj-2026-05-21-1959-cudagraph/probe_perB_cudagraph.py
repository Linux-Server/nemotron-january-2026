"""Per-B manual CUDA graph probe for the steady streaming encoder bucket.

Standalone probe. It does not import or edit server.py.

For B=1..MAX_B, this captures encoder.cache_aware_stream_step in one
torch.cuda.CUDAGraph per exact batch size, runs B independent clips in lockstep
with no padding, and compares:

  graph encoder replay + eager RNNT greedy decode
  vs
  fully eager batched model.conformer_stream_step(B)

The hard gate is byte-exact interim/final text plus bit-identical state
snapshots (max_abs=0) for every tested B.

Run:
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python proj-2026-05-21-1959-cudagraph/probe_perB_cudagraph.py
"""

from __future__ import annotations

import dataclasses
import gc
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
TBS_PATH = REPO / "proj-2026-05-21-0410" / "test_batch_state.py"
ARTIFACT_DIR = REPO / "proj-2026-05-21-inference-opt" / "round5-artifacts"

spec = importlib.util.spec_from_file_location("perb_cudagraph_harness", TBS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to import harness: {TBS_PATH}")
tbs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tbs
spec.loader.exec_module(tbs)


MAX_B = int(os.environ.get("PROBE_PERB_MAX_B", "16"))
CAPTURE_WARMUP_ITERS = int(os.environ.get("PROBE_CUDAGRAPH_WARMUPS", "5"))
MIN_STEADY_CHUNKS = int(os.environ.get("PROBE_PERB_MIN_STEADY_CHUNKS", "60"))
FORCED_CHUNK_COUNT = os.environ.get("PROBE_PERB_CHUNK_COUNT")
RESULT_JSON = Path(os.environ.get("PROBE_PERB_RESULT_JSON", ARTIFACT_DIR / "perB-cudagraph-results.json"))


def log(*args: Any) -> None:
    print(*args, flush=True)


@dataclasses.dataclass
class BatchRun:
    label: str
    batch_size: int
    sample_ids: list[str]
    captures: dict[str, list[dict[str, Any]]]
    states: dict[str, Any]
    final_texts: dict[str, str]
    steady_wall_ms: list[float]
    steady_gpu_ms: list[float]
    chunk_count: int = 0
    steady_count: int = 0
    graph_replays: int = 0
    eager_fallbacks: int = 0

    @property
    def avg_wall_ms(self) -> float:
        return mean(self.steady_wall_ms) if self.steady_wall_ms else 0.0

    @property
    def p50_wall_ms(self) -> float:
        return median(self.steady_wall_ms) if self.steady_wall_ms else 0.0

    @property
    def p95_wall_ms(self) -> float:
        return percentile(self.steady_wall_ms, 0.95)

    @property
    def avg_gpu_ms(self) -> float:
        return mean(self.steady_gpu_ms) if self.steady_gpu_ms else 0.0

    @property
    def p50_gpu_ms(self) -> float:
        return median(self.steady_gpu_ms) if self.steady_gpu_ms else 0.0

    @property
    def p95_gpu_ms(self) -> float:
        return percentile(self.steady_gpu_ms, 0.95)


@dataclasses.dataclass
class CompareResult:
    text_byte_exact: bool = True
    final_byte_exact: bool = True
    state_allclose: bool = True
    state_bit_equal: bool = True
    tensor_count: int = 0
    max_abs: float = 0.0
    max_path: str = ""
    diffs: list[str] = dataclasses.field(default_factory=list)

    @property
    def hard_pass(self) -> bool:
        return (
            self.text_byte_exact
            and self.final_byte_exact
            and self.state_allclose
            and self.state_bit_equal
            and self.max_abs == 0.0
        )


@dataclasses.dataclass
class BResult:
    batch_size: int
    sample_ids: list[str]
    capture_ms: float
    steady_chunks: int
    eager_wall_avg_ms: float
    graph_wall_avg_ms: float
    wall_speedup: float
    eager_gpu_avg_ms: float
    graph_gpu_avg_ms: float
    gpu_speedup: float
    gpu_drop_pct: float
    compare: CompareResult
    graph_replays: int
    eager_fallbacks: int


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return ordered[idx]


def enforce_probe_environment() -> None:
    os.environ["NEMOTRON_WARMUP_MS"] = "200"
    torch.backends.cudnn.benchmark = False
    log(
        "environment: "
        f"NEMOTRON_WARMUP_MS={os.environ['NEMOTRON_WARMUP_MS']} "
        f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
        f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} "
        "(TF32 defaults preserved)"
    )


def gpu_compute_apps() -> list[str]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def other_gpu_apps() -> list[str]:
    pid = str(os.getpid())
    return [line for line in gpu_compute_apps() if not line.startswith(pid + ",")]


def load_model() -> Any:
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    enforce_probe_environment()
    log(f"loading model: {tbs.EN_NEMO}")
    model = nemo_asr.models.ASRModel.restore_from(tbs.EN_NEMO, map_location="cuda")
    model.encoder.set_default_att_context_size([70, 1])
    decoding_cfg = OmegaConf.create(
        {
            "strategy": "greedy",
            "greedy": {
                "max_symbols": 10,
                "loop_labels": False,
                "use_cuda_graph_decoder": False,
            },
        }
    )
    try:
        model.change_decoding_strategy(decoding_cfg=decoding_cfg, verbose=False)
    except TypeError:
        model.change_decoding_strategy(decoding_cfg=decoding_cfg)
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    return model


def encoder_stream_step_restoring_drop_extra(model: Any, **kwargs: Any) -> tuple[Any, ...]:
    streaming_cfg = model.encoder.streaming_cfg
    original_drop_extra = streaming_cfg.drop_extra_pre_encoded
    try:
        return model.encoder.cache_aware_stream_step(**kwargs)
    finally:
        streaming_cfg.drop_extra_pre_encoded = original_drop_extra


class ManualCudaGraphEncoder:
    """Static-buffer CUDA graph wrapper for one exact steady batch size."""

    def __init__(self, model: Any, geom: Any, *, batch_size: int, warmup_iters: int) -> None:
        self.model = model
        self.geom = geom
        self.batch_size = int(batch_size)
        self.T = int(geom.pre_encode_cache_size + geom.shift_frames)
        self.drop_extra = int(geom.drop_extra)
        self.replays = 0

        feat = int(model.cfg.preprocessor.features)
        cache = model.encoder.get_initial_cache_state(batch_size=self.batch_size)
        device = cache[0].device
        dtype = cache[0].dtype

        self.static_processed = torch.empty((self.batch_size, feat, self.T), device=device, dtype=dtype)
        self.static_processed.zero_()
        self.static_len = torch.full((self.batch_size,), self.T, device=device, dtype=torch.long)
        self.static_clc = torch.empty_like(cache[0])
        self.static_clt = torch.empty_like(cache[1])
        self.static_clcl = torch.empty_like(cache[2])
        self.static_clc.zero_()
        self.static_clt.zero_()
        self.static_clcl.zero_()

        self.graph = torch.cuda.CUDAGraph()
        self.static_outputs: Optional[tuple[torch.Tensor, ...]] = None

        start = time.perf_counter()
        self._capture(warmup_iters)
        torch.cuda.synchronize()
        self.capture_ms = (time.perf_counter() - start) * 1000.0

    def _call_encoder(self) -> tuple[torch.Tensor, ...]:
        return encoder_stream_step_restoring_drop_extra(
            self.model,
            processed_signal=self.static_processed,
            processed_signal_length=self.static_len,
            cache_last_channel=self.static_clc,
            cache_last_time=self.static_clt,
            cache_last_channel_len=self.static_clcl,
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra,
        )

    def _capture(self, warmup_iters: int) -> None:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(side_stream):
            for _ in range(warmup_iters):
                self._call_encoder()
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.static_outputs = self._call_encoder()

        if self.static_outputs is None or len(self.static_outputs) != 5:
            raise RuntimeError("manual graph capture did not return the 5 encoder tensors")

    def replay(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
        cache_last_channel: torch.Tensor,
        cache_last_time: torch.Tensor,
        cache_last_channel_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if processed_signal.shape != self.static_processed.shape:
            raise ValueError(
                f"manual graph bucket mismatch: got processed {tuple(processed_signal.shape)}, "
                f"expected {tuple(self.static_processed.shape)}"
            )
        self.static_processed.copy_(processed_signal)
        self.static_len.copy_(processed_signal_length)
        self.static_clc.copy_(cache_last_channel)
        self.static_clt.copy_(cache_last_time)
        self.static_clcl.copy_(cache_last_channel_len)
        self.graph.replay()
        self.replays += 1
        assert self.static_outputs is not None
        return self.static_outputs


def decode_rnnt_greedy(
    model: Any,
    encoded: torch.Tensor,
    encoded_len: torch.Tensor,
    previous_hypotheses: Optional[list[Any]],
) -> tuple[list[torch.Tensor], list[Any], list[Any]]:
    best_hyp = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoded,
        encoded_lengths=encoded_len,
        return_hypotheses=True,
        partial_hypotheses=previous_hypotheses,
    )
    greedy_predictions = [hyp.y_sequence for hyp in best_hyp]
    return greedy_predictions, best_hyp, best_hyp


def normal_chunk_count(geom: Any, total_audio_samples: int) -> int:
    emitted_frames = int(geom.warmup_frames or 0)
    synthetic_prefix_samples = emitted_frames * int(geom.hop_samples)
    pending_audio_len = int(total_audio_samples)
    count = 0
    while tbs.ready_predicate(
        synthetic_prefix_samples=synthetic_prefix_samples,
        total_audio_samples=int(total_audio_samples),
        emitted_frames=emitted_frames,
        shift_frames=int(geom.shift_frames),
        hop_samples=int(geom.hop_samples),
        pending_audio_len=pending_audio_len,
        preprocess_new_audio_samples=int(geom.preprocess_new_audio_samples),
    ):
        count += 1
        emitted_frames += int(geom.shift_frames)
        pending_audio_len = max(0, pending_audio_len - int(geom.shift_frames) * int(geom.hop_samples))
    return count


def select_clip_ids(geom: Any, *, max_b: int) -> tuple[list[str], int]:
    con = sqlite3.connect(tbs.DB)
    try:
        rows = con.execute(
            "SELECT sample_id, audio_path, dataset_index FROM samples "
            "WHERE language='eng' ORDER BY dataset_index"
        ).fetchall()
    finally:
        con.close()

    groups: dict[int, list[tuple[int, str]]] = {}
    for sample_id, audio_path, dataset_index in rows:
        audio_file = REPO / "stt-benchmark" / str(audio_path)
        if not audio_file.exists():
            continue
        total_audio_samples = audio_file.stat().st_size // np.dtype(np.int16).itemsize
        chunks = normal_chunk_count(geom, total_audio_samples)
        groups.setdefault(chunks, []).append((int(dataset_index), str(sample_id)))

    if FORCED_CHUNK_COUNT is not None:
        chunk_count = int(FORCED_CHUNK_COUNT)
        selected_group = groups.get(chunk_count, [])
        if len(selected_group) < max_b:
            raise RuntimeError(
                f"forced chunk count {chunk_count} has {len(selected_group)} clips; need {max_b}"
            )
    else:
        candidates = [
            (chunks, clips)
            for chunks, clips in groups.items()
            if chunks >= MIN_STEADY_CHUNKS and len(clips) >= max_b
        ]
        if not candidates:
            candidates = [
                (chunks, clips)
                for chunks, clips in groups.items()
                if len(clips) >= max_b
            ]
        if not candidates:
            best = max((len(clips), chunks) for chunks, clips in groups.items())
            raise RuntimeError(f"no chunk-count group has {max_b} clips; best={best}")
        chunk_count, selected_group = sorted(candidates, key=lambda item: item[0])[0]

    selected = [sample_id for _idx, sample_id in sorted(selected_group)[:max_b]]
    return selected, int(chunk_count)


def load_clips(sample_ids: list[str]) -> dict[str, np.ndarray]:
    return {sample_id: tbs.load_clip(sample_id) for sample_id in sample_ids}


def update_raw_ring(geom: Any, raw_ring: np.ndarray, consumed_audio: np.ndarray) -> np.ndarray:
    if len(consumed_audio) >= geom.raw_audio_ring_samples:
        return consumed_audio[-geom.raw_audio_ring_samples :].copy()
    if len(consumed_audio) == 0:
        return raw_ring
    keep = geom.raw_audio_ring_samples - len(consumed_audio)
    return np.concatenate([raw_ring[-keep:], consumed_audio]).astype(np.float32, copy=False)


def process_batch_step(
    model: Any,
    geom: Any,
    group: list[Any],
    *,
    graph_encoder: Optional[ManualCudaGraphEncoder],
    run: BatchRun,
) -> None:
    prepared = [tbs.prepare_row(model, geom, state) for state in group]
    chunk_mels = [item[0] for item in prepared]
    valid_new_mels = [item[1] for item in prepared]
    drop_values = [int(item[2]) for item in prepared]
    if len(set(drop_values)) != 1:
        raise RuntimeError(f"mixed drop_extra in batch: {drop_values}")

    chunk, chunk_len = tbs.stack_processed(chunk_mels)
    clc, clt, clcl = tbs.stack_caches(
        [(state.cache_last_channel, state.cache_last_time, state.cache_last_channel_len) for state in group]
    )
    per_session_hyps = [tbs.clone_tree(state.previous_hypotheses) for state in group]
    stacked_hyps = tbs.stack_hypotheses(per_session_hyps)
    previous_hypotheses = None if all(h is None for h in stacked_hyps) else stacked_hyps
    per_session_pred = [tbs.clone_tree(state.pred_out_stream) for state in group]
    previous_pred_out = tbs.stack_pred_out(per_session_pred, rnnt=True)

    is_steady = (
        drop_values[0] == int(geom.drop_extra)
        and int(chunk.shape[-1]) == int(geom.pre_encode_cache_size + geom.shift_frames)
    )
    use_graph = graph_encoder is not None and is_steady

    start_event: Optional[torch.cuda.Event] = None
    end_event: Optional[torch.cuda.Event] = None
    if is_steady:
        torch.cuda.synchronize()
        start_wall = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_wall = 0.0

    with torch.inference_mode():
        if use_graph:
            encoded, encoded_len, next_clc_raw, next_clt_raw, next_clcl_raw = graph_encoder.replay(
                chunk,
                chunk_len,
                clc,
                clt,
                clcl,
            )
            pred_out, transcribed_texts, best_hyp = decode_rnnt_greedy(
                model,
                encoded,
                encoded_len,
                previous_hypotheses,
            )
        else:
            (
                pred_out,
                transcribed_texts,
                next_clc_raw,
                next_clt_raw,
                next_clcl_raw,
                best_hyp,
            ) = tbs.conformer_stream_step_restoring_drop_extra(
                model,
                processed_signal=chunk,
                processed_signal_length=chunk_len,
                cache_last_channel=clc,
                cache_last_time=clt,
                cache_last_channel_len=clcl,
                keep_all_outputs=False,
                previous_hypotheses=previous_hypotheses,
                previous_pred_out=previous_pred_out,
                drop_extra_pre_encoded=drop_values[0],
                return_transcription=True,
            )

        row_caches = [
            tbs.scatter_cache_row(next_clc_raw, next_clt_raw, next_clcl_raw, row)
            for row in range(len(group))
        ]

    if is_steady:
        assert start_event is not None and end_event is not None
        end_event.record()
        torch.cuda.synchronize()
        run.steady_wall_ms.append((time.perf_counter() - start_wall) * 1000.0)
        run.steady_gpu_ms.append(float(start_event.elapsed_time(end_event)))
        run.steady_count += 1

    run.chunk_count += 1
    if use_graph:
        run.graph_replays += 1
    elif graph_encoder is not None:
        run.eager_fallbacks += 1

    for row, state in enumerate(group):
        state.cache_last_channel, state.cache_last_time, state.cache_last_channel_len = row_caches[row]
        state.previous_hypotheses = [best_hyp[row]] if best_hyp is not None else None
        state.pred_out_stream = [pred_out[row]] if pred_out is not None else None

        consumed_audio = state.pending_audio[: geom.shift_frames * geom.hop_samples]
        state.raw_audio_ring = update_raw_ring(geom, state.raw_audio_ring, consumed_audio)
        state.pending_audio = state.pending_audio[geom.shift_frames * geom.hop_samples :]
        tbs.update_mel_frame_ring(geom, state, valid_new_mels[row])
        state.emitted_frames += geom.shift_frames
        if transcribed_texts and len(transcribed_texts) > row and transcribed_texts[row] is not None:
            state.current_text = tbs.text_from_hyp(transcribed_texts[row])


def process_final_chunk(model: Any, geom: Any, state: Any) -> str:
    if state.pending_audio is None or len(state.pending_audio) == 0:
        return state.current_text

    padding_samples = 2 * geom.shift_frames * geom.hop_samples
    pending = np.concatenate([state.pending_audio, np.zeros(padding_samples, dtype=np.float32)])
    raw_ring = state.raw_audio_ring.copy()

    padded_total_samples = state.emitted_frames * geom.hop_samples + len(pending)
    total_mel_frames = (padded_total_samples // geom.hop_samples) + 1
    remaining_frames = total_mel_frames - state.emitted_frames
    if remaining_frames <= 0:
        return state.current_text

    new_mels: list[torch.Tensor] = []
    frames_collected = 0
    while frames_collected < remaining_frames:
        frames_this_call = min(geom.shift_frames, remaining_frames - frames_collected)
        needed_new_samples = min(len(pending), geom.preprocess_new_audio_samples)
        new_audio = pending[:needed_new_samples]
        fixed_audio, valid_samples = tbs.build_fixed_preprocess_audio(geom, raw_ring, new_audio)
        mel, _mel_len = tbs.preprocess_fixed_audio(model, fixed_audio, valid_samples)
        start = geom.first_preprocess_mel_frame
        new_mels.append(mel[:, :, start : start + frames_this_call])

        if frames_this_call == geom.shift_frames:
            consumed_samples = min(geom.shift_frames * geom.hop_samples, len(pending))
            consumed_audio = pending[:consumed_samples]
            raw_ring = update_raw_ring(geom, raw_ring, consumed_audio)
            pending = pending[consumed_samples:]
        frames_collected += frames_this_call

    new_mel = torch.cat(new_mels, dim=-1)
    if state.emitted_frames == 0:
        chunk_mel = new_mel
        drop_extra = 0
    else:
        if state.mel_frame_ring is None:
            raise RuntimeError(f"{state.sample_id}: missing mel_frame_ring for final chunk")
        chunk_mel = torch.cat((state.mel_frame_ring, new_mel), dim=-1)
        drop_extra = geom.drop_extra

    chunk_len = torch.tensor([chunk_mel.shape[-1]], device="cuda")
    per_session_hyps = [tbs.clone_tree(state.previous_hypotheses)]
    stacked_hyps = tbs.stack_hypotheses(per_session_hyps)
    previous_hypotheses = None if all(h is None for h in stacked_hyps) else stacked_hyps
    previous_pred_out = tbs.stack_pred_out([tbs.clone_tree(state.pred_out_stream)], rnnt=True)

    (
        pred_out,
        transcribed_texts,
        next_clc,
        next_clt,
        next_clcl,
        best_hyp,
    ) = tbs.conformer_stream_step_restoring_drop_extra(
        model,
        processed_signal=chunk_mel,
        processed_signal_length=chunk_len,
        cache_last_channel=state.cache_last_channel,
        cache_last_time=state.cache_last_time,
        cache_last_channel_len=state.cache_last_channel_len,
        keep_all_outputs=True,
        previous_hypotheses=previous_hypotheses,
        previous_pred_out=previous_pred_out,
        drop_extra_pre_encoded=drop_extra,
        return_transcription=True,
    )

    state.cache_last_channel = next_clc.detach().clone()
    state.cache_last_time = next_clt.detach().clone()
    state.cache_last_channel_len = next_clcl.detach().clone()
    state.previous_hypotheses = [best_hyp[0]] if best_hyp is not None else None
    state.pred_out_stream = [pred_out[0]] if pred_out is not None else None
    state.pending_audio = pending
    state.raw_audio_ring = raw_ring
    tbs.update_mel_frame_ring(geom, state, new_mel[:, :, -geom.shift_frames :])
    state.emitted_frames += remaining_frames
    if transcribed_texts and transcribed_texts[0] is not None:
        state.current_text = tbs.text_from_hyp(transcribed_texts[0])
    return state.current_text


def run_streams(
    model: Any,
    geom: Any,
    clips: dict[str, np.ndarray],
    *,
    sample_ids: list[str],
    label: str,
    graph_encoder: Optional[ManualCudaGraphEncoder],
) -> BatchRun:
    states = [tbs.init_state(model, geom, sample_id, clips[sample_id]) for sample_id in sample_ids]
    run = BatchRun(
        label=label,
        batch_size=len(sample_ids),
        sample_ids=sample_ids,
        captures={sample_id: [] for sample_id in sample_ids},
        states={},
        final_texts={},
        steady_wall_ms=[],
        steady_gpu_ms=[],
    )
    while True:
        ready = [state for state in states if tbs.state_is_ready(geom, state)]
        if not ready:
            break
        if len(ready) != len(states):
            raise RuntimeError(
                f"{label} B={len(states)}: non-lockstep ready set {len(ready)} of {len(states)}; no padding allowed"
            )
        process_batch_step(model, geom, ready, graph_encoder=graph_encoder, run=run)
        for state in ready:
            run.captures[state.sample_id].append(tbs.state_snapshot(state))

    run.states = {state.sample_id: state for state in states}
    run.final_texts = {
        sample_id: process_final_chunk(model, geom, run.states[sample_id])
        for sample_id in sample_ids
    }
    return run


def merge_summary(result: CompareResult, summary: Any) -> None:
    result.tensor_count += summary.tensor_count
    result.state_bit_equal = result.state_bit_equal and summary.bit_equal
    result.state_allclose = result.state_allclose and summary.allclose
    if summary.max_abs > result.max_abs:
        result.max_abs = float(summary.max_abs)
        result.max_path = summary.max_path


def compare_runs(eager: BatchRun, manual: BatchRun) -> CompareResult:
    result = CompareResult()
    for sample_id in eager.sample_ids:
        eager_steps = eager.captures[sample_id]
        manual_steps = manual.captures[sample_id]
        if len(eager_steps) != len(manual_steps):
            result.text_byte_exact = False
            result.diffs.append(
                f"{sample_id}: normal step count mismatch {len(eager_steps)} != {len(manual_steps)}"
            )
            continue
        for step_idx, (exp, got) in enumerate(zip(eager_steps, manual_steps), start=1):
            if str(exp["text"]).encode("utf-8") != str(got["text"]).encode("utf-8"):
                result.text_byte_exact = False
                result.diffs.append(
                    f"{sample_id} step={step_idx}: text mismatch "
                    f"eager={exp['text']!r} manual={got['text']!r}"
                )
            try:
                summary = tbs.compare_snapshot(f"{sample_id[:8]}.step{step_idx}", exp, got)
                merge_summary(result, summary)
            except AssertionError as exc:
                result.state_allclose = False
                result.state_bit_equal = False
                if len(result.diffs) < 12:
                    result.diffs.append(f"{sample_id} step={step_idx}: state diff: {exc}")

        eager_final = eager.final_texts[sample_id]
        manual_final = manual.final_texts[sample_id]
        if eager_final.encode("utf-8") != manual_final.encode("utf-8"):
            result.final_byte_exact = False
            result.diffs.append(
                f"{sample_id}: FINAL_TEXT_DIFF eager={eager_final!r} manual={manual_final!r}"
            )
    return result


def summarize_run(run: BatchRun) -> str:
    return (
        f"{run.label} B={run.batch_size}: chunks={run.chunk_count} steady={run.steady_count} "
        f"graph_replays={run.graph_replays} eager_fallbacks={run.eager_fallbacks} "
        f"wall_avg={run.avg_wall_ms:.3f}ms wall_p50={run.p50_wall_ms:.3f}ms "
        f"wall_p95={run.p95_wall_ms:.3f}ms gpu_avg={run.avg_gpu_ms:.3f}ms "
        f"gpu_p50={run.p50_gpu_ms:.3f}ms gpu_p95={run.p95_gpu_ms:.3f}ms"
    )


def result_to_dict(result: BResult) -> dict[str, Any]:
    return {
        "B": result.batch_size,
        "sample_ids": result.sample_ids,
        "capture_ms": result.capture_ms,
        "steady_chunks": result.steady_chunks,
        "eager_wall_avg_ms": result.eager_wall_avg_ms,
        "graph_wall_avg_ms": result.graph_wall_avg_ms,
        "wall_speedup": result.wall_speedup,
        "eager_gpu_avg_ms": result.eager_gpu_avg_ms,
        "graph_gpu_avg_ms": result.graph_gpu_avg_ms,
        "gpu_speedup": result.gpu_speedup,
        "gpu_drop_pct": result.gpu_drop_pct,
        "graph_replays": result.graph_replays,
        "eager_fallbacks": result.eager_fallbacks,
        "compare": {
            "hard_pass": result.compare.hard_pass,
            "text_byte_exact": result.compare.text_byte_exact,
            "final_byte_exact": result.compare.final_byte_exact,
            "state_allclose": result.compare.state_allclose,
            "state_bit_equal": result.compare.state_bit_equal,
            "tensor_count": result.compare.tensor_count,
            "max_abs": result.compare.max_abs,
            "max_path": result.compare.max_path,
            "diffs": result.compare.diffs,
        },
    }


def recommended_k(results: list[BResult], threshold: float = 1.15) -> dict[str, Any]:
    for result in results:
        if result.wall_speedup < threshold:
            return {
                "threshold": threshold,
                "first_b_below_threshold": result.batch_size,
                "recommended_max_b": max(1, result.batch_size - 1),
            }
    return {
        "threshold": threshold,
        "first_b_below_threshold": None,
        "recommended_max_b": results[-1].batch_size if results else 0,
    }


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    shutil.rmtree(THIS_DIR / "__pycache__", ignore_errors=True)


def main() -> int:
    preexisting_gpu_apps = other_gpu_apps()
    if preexisting_gpu_apps:
        log("gpu contention note: other compute apps before probe:")
        for line in preexisting_gpu_apps:
            log(f"  {line}")
    else:
        log("gpu contention note: no other compute apps visible before probe")

    model = None
    graphs: dict[int, ManualCudaGraphEncoder] = {}
    results: list[BResult] = []
    try:
        model = load_model()
        geom = tbs.build_geometry(model)
        selected_ids, selected_chunk_count = select_clip_ids(geom, max_b=MAX_B)
        clips = load_clips(selected_ids)
        log(
            "geometry: "
            f"shift={geom.shift_frames} pre_cache={geom.pre_encode_cache_size} "
            f"steady_T={geom.pre_encode_cache_size + geom.shift_frames} "
            f"drop_extra={geom.drop_extra} warmup_ms={geom.warmup_ms} "
            f"warmup_frames={geom.warmup_frames} selected_clips={len(selected_ids)} "
            f"normal_chunks_per_clip={selected_chunk_count}"
        )
        log("selected sample_ids:")
        for idx, sample_id in enumerate(selected_ids, start=1):
            log(f"  {idx:02d}: {sample_id}")

        for batch_size in range(1, MAX_B + 1):
            graphs[batch_size] = ManualCudaGraphEncoder(
                model,
                geom,
                batch_size=batch_size,
                warmup_iters=CAPTURE_WARMUP_ITERS,
            )
            log(
                "manual graph captured: "
                f"B={batch_size} steady_T={graphs[batch_size].T} "
                f"drop_extra={graphs[batch_size].drop_extra} "
                f"warmup_iters={CAPTURE_WARMUP_ITERS} "
                f"capture_ms={graphs[batch_size].capture_ms:.1f}"
            )

        for batch_size in range(1, MAX_B + 1):
            sample_ids = selected_ids[:batch_size]
            graph_encoder = graphs[batch_size]

            eager = run_streams(
                model,
                geom,
                clips,
                sample_ids=sample_ids,
                label="eager",
                graph_encoder=None,
            )
            log(summarize_run(eager))

            manual = run_streams(
                model,
                geom,
                clips,
                sample_ids=sample_ids,
                label="manual_graph",
                graph_encoder=graph_encoder,
            )
            log(summarize_run(manual))

            compare = compare_runs(eager, manual)
            wall_speedup = eager.avg_wall_ms / manual.avg_wall_ms if manual.avg_wall_ms else 0.0
            gpu_speedup = eager.avg_gpu_ms / manual.avg_gpu_ms if manual.avg_gpu_ms else 0.0
            gpu_drop_pct = (
                ((eager.avg_gpu_ms - manual.avg_gpu_ms) / eager.avg_gpu_ms) * 100.0
                if eager.avg_gpu_ms
                else 0.0
            )
            b_result = BResult(
                batch_size=batch_size,
                sample_ids=sample_ids,
                capture_ms=graph_encoder.capture_ms,
                steady_chunks=eager.steady_count,
                eager_wall_avg_ms=eager.avg_wall_ms,
                graph_wall_avg_ms=manual.avg_wall_ms,
                wall_speedup=wall_speedup,
                eager_gpu_avg_ms=eager.avg_gpu_ms,
                graph_gpu_avg_ms=manual.avg_gpu_ms,
                gpu_speedup=gpu_speedup,
                gpu_drop_pct=gpu_drop_pct,
                compare=compare,
                graph_replays=manual.graph_replays,
                eager_fallbacks=manual.eager_fallbacks,
            )
            results.append(b_result)
            log(
                "compare: "
                f"B={batch_size} hard_pass={compare.hard_pass} "
                f"text_byte_exact={compare.text_byte_exact} "
                f"final_byte_exact={compare.final_byte_exact} "
                f"state_allclose={compare.state_allclose} "
                f"state_bit_equal={compare.state_bit_equal} "
                f"tensor_count={compare.tensor_count} "
                f"max_abs={compare.max_abs:.3e} path={compare.max_path or 'n/a'}"
            )
            if compare.diffs:
                log("diffs:")
                for diff in compare.diffs[:12]:
                    log(f"  {diff}")
            log(
                "RESULT_B "
                f"B={batch_size} hard_pass={compare.hard_pass} "
                f"capture_ms={graph_encoder.capture_ms:.1f} "
                f"eager_wall_avg_ms={eager.avg_wall_ms:.3f} "
                f"graph_wall_avg_ms={manual.avg_wall_ms:.3f} "
                f"wall_speedup={wall_speedup:.3f} "
                f"eager_gpu_avg_ms={eager.avg_gpu_ms:.3f} "
                f"graph_gpu_avg_ms={manual.avg_gpu_ms:.3f} "
                f"gpu_speedup={gpu_speedup:.3f} "
                f"gpu_drop_pct={gpu_drop_pct:.1f} "
                f"replays={manual.graph_replays} "
                f"fallbacks={manual.eager_fallbacks}"
            )
            if not compare.hard_pass:
                break

        rec = recommended_k(results)
        payload = {
            "max_b": MAX_B,
            "capture_warmup_iters": CAPTURE_WARMUP_ITERS,
            "min_steady_chunks": MIN_STEADY_CHUNKS,
            "forced_chunk_count": FORCED_CHUNK_COUNT,
            "selected_chunk_count": selected_chunk_count,
            "selected_sample_ids": selected_ids,
            "gpu_compute_apps_before": preexisting_gpu_apps,
            "recommendation": rec,
            "results": [result_to_dict(item) for item in results],
        }
        RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
        RESULT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        log(f"wrote results json: {RESULT_JSON}")
        log(
            "RECOMMENDATION "
            f"threshold={rec['threshold']:.2f} "
            f"first_b_below_threshold={rec['first_b_below_threshold']} "
            f"recommended_max_b={rec['recommended_max_b']}"
        )
        all_pass = len(results) == MAX_B and all(item.compare.hard_pass for item in results)
        return 0 if all_pass else 2
    finally:
        graphs.clear()
        del model
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
