"""Manual CUDA graph probe for the Conformer encoder steady streaming bucket.

This is intentionally standalone: it does not import or edit server.py.  It
compares the normal eager streaming path against:

  encoder.cache_aware_stream_step captured once with torch.cuda.CUDAGraph
  + eager RNNT greedy decode

Only the steady B=1 bucket is graphed (T=pre_encode_cache + shift, drop_extra
from the encoder streaming config).  Warmup, first-chunk fallback, and final
chunks remain eager.

Run:
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python proj-2026-05-21-0410/probe_manual_cudagraph.py
"""

from __future__ import annotations

import dataclasses
import gc
import importlib.util
import os
import shutil
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
TBS_PATH = THIS_DIR / "test_batch_state.py"

spec = importlib.util.spec_from_file_location("manual_cudagraph_harness", TBS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to import harness: {TBS_PATH}")
tbs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tbs
spec.loader.exec_module(tbs)


CLIP_IDS = tuple(tbs.CLIP_IDS[:4])
CAPTURE_WARMUP_ITERS = int(os.environ.get("PROBE_CUDAGRAPH_WARMUPS", "5"))


def log(*args: Any) -> None:
    print(*args, flush=True)


@dataclasses.dataclass
class StreamRun:
    label: str
    captures: dict[str, list[dict[str, Any]]]
    states: dict[str, Any]
    final_texts: dict[str, str]
    steady_ms: list[float]
    chunk_count: int = 0
    steady_count: int = 0
    graph_replays: int = 0
    eager_fallbacks: int = 0

    @property
    def avg_steady_ms(self) -> float:
        return mean(self.steady_ms) if self.steady_ms else 0.0

    @property
    def p50_steady_ms(self) -> float:
        return median(self.steady_ms) if self.steady_ms else 0.0

    @property
    def p95_steady_ms(self) -> float:
        if not self.steady_ms:
            return 0.0
        values = sorted(self.steady_ms)
        idx = min(len(values) - 1, int(round(0.95 * (len(values) - 1))))
        return values[idx]


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
    """Static-buffer CUDA graph wrapper for one B=1 steady encoder bucket."""

    def __init__(self, model: Any, geom: Any, *, warmup_iters: int) -> None:
        self.model = model
        self.geom = geom
        self.T = int(geom.pre_encode_cache_size + geom.shift_frames)
        self.drop_extra = int(geom.drop_extra)
        self.replays = 0

        feat = int(model.cfg.preprocessor.features)
        cache = model.encoder.get_initial_cache_state(batch_size=1)
        device = cache[0].device
        dtype = cache[0].dtype

        self.static_processed = torch.empty((1, feat, self.T), device=device, dtype=dtype)
        self.static_processed.zero_()
        self.static_len = torch.full((1,), self.T, device=device, dtype=torch.long)
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


def update_raw_ring(geom: Any, raw_ring: np.ndarray, consumed_audio: np.ndarray) -> np.ndarray:
    if len(consumed_audio) >= geom.raw_audio_ring_samples:
        return consumed_audio[-geom.raw_audio_ring_samples :].copy()
    if len(consumed_audio) == 0:
        return raw_ring
    keep = geom.raw_audio_ring_samples - len(consumed_audio)
    return np.concatenate([raw_ring[-keep:], consumed_audio]).astype(np.float32, copy=False)


def process_one_chunk(
    model: Any,
    geom: Any,
    state: Any,
    *,
    graph_encoder: Optional[ManualCudaGraphEncoder],
    run: StreamRun,
) -> None:
    chunk_mel, valid_new_mel, drop_extra = tbs.prepare_row(model, geom, state)
    chunk, chunk_len = tbs.stack_processed([chunk_mel])
    clc, clt, clcl = tbs.stack_caches(
        [(state.cache_last_channel, state.cache_last_time, state.cache_last_channel_len)]
    )
    per_session_hyps = [tbs.clone_tree(state.previous_hypotheses)]
    stacked_hyps = tbs.stack_hypotheses(per_session_hyps)
    previous_hypotheses = None if all(h is None for h in stacked_hyps) else stacked_hyps
    previous_pred_out = tbs.stack_pred_out([tbs.clone_tree(state.pred_out_stream)], rnnt=True)

    is_steady = (
        int(drop_extra) == int(geom.drop_extra)
        and int(chunk.shape[-1]) == int(geom.pre_encode_cache_size + geom.shift_frames)
    )
    use_graph = graph_encoder is not None and is_steady

    if is_steady:
        torch.cuda.synchronize()
        start = time.perf_counter()

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
            next_clc = next_clc_raw.detach().clone()
            next_clt = next_clt_raw.detach().clone()
            next_clcl = next_clcl_raw.detach().clone()
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
                drop_extra_pre_encoded=drop_extra,
                return_transcription=True,
            )
            next_clc = next_clc_raw.detach().clone()
            next_clt = next_clt_raw.detach().clone()
            next_clcl = next_clcl_raw.detach().clone()

    if is_steady:
        torch.cuda.synchronize()
        run.steady_ms.append((time.perf_counter() - start) * 1000.0)
        run.steady_count += 1

    run.chunk_count += 1
    if use_graph:
        run.graph_replays += 1
    elif graph_encoder is not None:
        run.eager_fallbacks += 1

    state.cache_last_channel = next_clc
    state.cache_last_time = next_clt
    state.cache_last_channel_len = next_clcl
    state.previous_hypotheses = [best_hyp[0]] if best_hyp is not None else None
    state.pred_out_stream = [pred_out[0]] if pred_out is not None else None

    consumed_audio = state.pending_audio[: geom.shift_frames * geom.hop_samples]
    state.raw_audio_ring = update_raw_ring(geom, state.raw_audio_ring, consumed_audio)
    state.pending_audio = state.pending_audio[geom.shift_frames * geom.hop_samples :]
    tbs.update_mel_frame_ring(geom, state, valid_new_mel)
    state.emitted_frames += geom.shift_frames
    if transcribed_texts and transcribed_texts[0] is not None:
        state.current_text = tbs.text_from_hyp(transcribed_texts[0])


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
    label: str,
    graph_encoder: Optional[ManualCudaGraphEncoder],
) -> StreamRun:
    states = [tbs.init_state(model, geom, sample_id, clips[sample_id]) for sample_id in CLIP_IDS]
    run = StreamRun(
        label=label,
        captures={sample_id: [] for sample_id in CLIP_IDS},
        states={},
        final_texts={},
        steady_ms=[],
    )
    while True:
        ready = [state for state in states if tbs.state_is_ready(geom, state)]
        if not ready:
            break
        for state in ready:
            process_one_chunk(model, geom, state, graph_encoder=graph_encoder, run=run)
            run.captures[state.sample_id].append(tbs.state_snapshot(state))

    run.states = {state.sample_id: state for state in states}
    run.final_texts = {
        sample_id: process_final_chunk(model, geom, run.states[sample_id])
        for sample_id in CLIP_IDS
    }
    return run


def merge_summary(result: CompareResult, summary: Any) -> None:
    result.tensor_count += summary.tensor_count
    result.state_bit_equal = result.state_bit_equal and summary.bit_equal
    result.state_allclose = result.state_allclose and summary.allclose
    if summary.max_abs > result.max_abs:
        result.max_abs = summary.max_abs
        result.max_path = summary.max_path


def compare_runs(eager: StreamRun, manual: StreamRun) -> CompareResult:
    result = CompareResult()
    for sample_id in CLIP_IDS:
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
                if len(result.diffs) < 8:
                    result.diffs.append(f"{sample_id} step={step_idx}: state diff: {exc}")

        eager_final = eager.final_texts[sample_id]
        manual_final = manual.final_texts[sample_id]
        if eager_final.encode("utf-8") != manual_final.encode("utf-8"):
            result.final_byte_exact = False
            result.diffs.append(
                f"{sample_id}: FINAL_TEXT_DIFF eager={eager_final!r} manual={manual_final!r}"
            )
    return result


def load_clips() -> dict[str, np.ndarray]:
    return {sample_id: tbs.load_clip(sample_id) for sample_id in CLIP_IDS}


def summarize_run(run: StreamRun) -> str:
    return (
        f"{run.label}: chunks={run.chunk_count} steady={run.steady_count} "
        f"graph_replays={run.graph_replays} eager_fallbacks={run.eager_fallbacks} "
        f"avg={run.avg_steady_ms:.3f}ms p50={run.p50_steady_ms:.3f}ms "
        f"p95={run.p95_steady_ms:.3f}ms"
    )


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
    graph_encoder = None
    try:
        model = load_model()
        geom = tbs.build_geometry(model)
        clips = load_clips()
        log(
            "geometry: "
            f"shift={geom.shift_frames} pre_cache={geom.pre_encode_cache_size} "
            f"steady_T={geom.pre_encode_cache_size + geom.shift_frames} "
            f"drop_extra={geom.drop_extra} warmup_ms={geom.warmup_ms} "
            f"warmup_frames={geom.warmup_frames} clips={len(CLIP_IDS)}"
        )

        graph_encoder = ManualCudaGraphEncoder(
            model,
            geom,
            warmup_iters=CAPTURE_WARMUP_ITERS,
        )
        log(
            "manual graph captured: "
            f"steady_T={graph_encoder.T} drop_extra={graph_encoder.drop_extra} "
            f"warmup_iters={CAPTURE_WARMUP_ITERS} capture_ms={graph_encoder.capture_ms:.1f}"
        )

        eager = run_streams(model, geom, clips, label="eager", graph_encoder=None)
        log(summarize_run(eager))

        manual = run_streams(model, geom, clips, label="manual_graph", graph_encoder=graph_encoder)
        log(summarize_run(manual))

        compare = compare_runs(eager, manual)
        speedup = eager.avg_steady_ms / manual.avg_steady_ms if manual.avg_steady_ms else 0.0
        log(
            "compare: "
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
            "RESULT "
            f"byte_exact={compare.text_byte_exact and compare.final_byte_exact} "
            f"capture_ms={graph_encoder.capture_ms:.1f} "
            f"eager_avg_ms={eager.avg_steady_ms:.3f} "
            f"manual_avg_ms={manual.avg_steady_ms:.3f} "
            f"speedup={speedup:.3f} "
            f"replays={manual.graph_replays} "
            f"fallbacks={manual.eager_fallbacks} "
            f"other_gpu_apps={len(preexisting_gpu_apps)}"
        )
        return 0 if compare.text_byte_exact and compare.final_byte_exact else 2
    finally:
        del graph_encoder
        del model
        cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
