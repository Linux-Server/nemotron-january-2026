"""Probe C3: greedy_batch B>=2 multi-chunk correctness and timing.

This is the blocking round-4 Step-7 probe. It compares the current shippable
`greedy` B=1 stream against `greedy_batch` B=1/2/4 over full multi-chunk
streams, including per-step state, B=4 row permutation, keep_all_outputs final
chunks, and decode/full-step timing at B=1/2/4/8.

Run with:
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python proj-2026-05-21-0410/probe_greedy_batch_multichunk.py
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
from typing import Any, Optional

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parent
TBS_PATH = THIS_DIR / "test_batch_state.py"

spec = importlib.util.spec_from_file_location("probe_b2_harness", TBS_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"failed to import harness: {TBS_PATH}")
tbs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tbs
spec.loader.exec_module(tbs)


BATCHES_TO_VERIFY = (1, 2, 4)
BATCHES_TO_TIME = (1, 2, 4, 8)
ATOL = 1.0e-4
RTOL = 1.0e-5
DEFAULT_TIMING_WARMUPS = 5
DEFAULT_TIMING_REPEATS = 20


def log(*args: Any) -> None:
    print(*args, flush=True)


@dataclasses.dataclass
class CaptureCheck:
    label: str
    text_byte_identical: bool = True
    emitted_exact: bool = True
    state_allclose: bool = True
    tensor_bit_equal: bool = True
    tensor_count: int = 0
    max_abs: float = 0.0
    max_path: str = ""
    diffs: list[str] = dataclasses.field(default_factory=list)
    suppressed_state_diffs: int = 0


@dataclasses.dataclass
class TimingCase:
    batch_size: int
    chunk: torch.Tensor
    chunk_len: torch.Tensor
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor
    previous_hypotheses: Optional[list[Any]]
    previous_pred_out: Optional[list[Any]]
    drop_extra: int
    encoded: torch.Tensor
    encoded_len: torch.Tensor


@dataclasses.dataclass
class TimingResult:
    batch_size: int
    greedy_decode_ms: float
    greedy_batch_decode_ms: float
    greedy_full_ms: float
    greedy_batch_full_ms: float
    encoder_ms: float
    fallback_full_ms: float

    @property
    def greedy_decode_ms_per_stream(self) -> float:
        return self.greedy_decode_ms / self.batch_size

    @property
    def greedy_batch_decode_ms_per_stream(self) -> float:
        return self.greedy_batch_decode_ms / self.batch_size

    @property
    def decode_stream_reduction(self) -> float:
        if self.greedy_decode_ms_per_stream == 0:
            return 0.0
        return 1.0 - (self.greedy_batch_decode_ms_per_stream / self.greedy_decode_ms_per_stream)

    @property
    def greedy_batch_full_speedup(self) -> float:
        return self.greedy_full_ms / self.greedy_batch_full_ms if self.greedy_batch_full_ms else 0.0

    @property
    def fallback_full_speedup(self) -> float:
        return self.greedy_full_ms / self.fallback_full_ms if self.fallback_full_ms else 0.0


def enforce_probe_environment() -> None:
    os.environ["NEMOTRON_WARMUP_MS"] = "200"
    os.environ.pop("NEMOTRON_KEEP_TF32", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def set_strategy(model: Any, strategy: str) -> None:
    from omegaconf import OmegaConf

    if strategy not in ("greedy", "greedy_batch"):
        raise ValueError(strategy)
    cfg = {
        "strategy": strategy,
        "greedy": {
            "max_symbols": 10,
            "loop_labels": strategy == "greedy_batch",
            "use_cuda_graph_decoder": False,
        },
    }
    decoding_cfg = OmegaConf.create(cfg)
    try:
        model.change_decoding_strategy(decoding_cfg=decoding_cfg, verbose=False)
    except TypeError:
        model.change_decoding_strategy(decoding_cfg=decoding_cfg)
    model.eval()


def load_model() -> Any:
    import nemo.collections.asr as nemo_asr

    enforce_probe_environment()
    log("TF32 DISABLED: torch.backends.cuda.matmul.allow_tf32=False cudnn.allow_tf32=False")
    log(f"loading model: {tbs.EN_NEMO}")
    model = nemo_asr.models.ASRModel.restore_from(tbs.EN_NEMO, map_location="cuda")
    model.encoder.set_default_att_context_size([70, 1])
    set_strategy(model, "greedy")
    model.preprocessor.featurizer.dither = 0.0
    return model


def text_bytes(value: Any) -> bytes:
    return ("" if value is None else str(value)).encode("utf-8")


def merge_check_summary(check: CaptureCheck, summary: Any) -> None:
    check.tensor_count += summary.tensor_count
    check.tensor_bit_equal = check.tensor_bit_equal and summary.bit_equal
    check.state_allclose = check.state_allclose and summary.allclose
    if summary.max_abs > check.max_abs:
        check.max_abs = summary.max_abs
        check.max_path = summary.max_path


def compare_state_payload(label: str, expected: dict[str, Any], actual: dict[str, Any]) -> tuple[Any, list[str]]:
    summary = tbs.CompareSummary(label=label)
    failures: list[str] = []
    for key in (
        "cache_last_channel",
        "cache_last_time",
        "cache_last_channel_len",
        "previous_hypotheses",
        "pred_out_stream",
    ):
        try:
            tbs.compare_value(f"{label}.{key}", expected[key], actual[key], summary)
        except AssertionError as exc:
            summary.allclose = False
            failures.append(compact_state_failure(str(exc)))
    return summary, failures


def compact_state_failure(message: str) -> str:
    if ": value mismatch " in message:
        path, _rest = message.split(": value mismatch ", 1)
        return f"{path}: value mismatch (values suppressed; non-tensor state differs)"
    if len(message) > 500:
        return message[:500] + " ... [truncated]"
    return message


def append_state_failures(check: CaptureCheck, failures: list[str]) -> None:
    for failure in failures:
        if sum(1 for diff in check.diffs if diff.startswith("STATE_DIFF ")) < 24:
            check.diffs.append(f"STATE_DIFF {failure}")
        else:
            check.suppressed_state_diffs += 1


def compare_captures_verbose(
    label: str,
    reference: dict[str, list[dict[str, Any]]],
    candidate: dict[str, list[dict[str, Any]]],
) -> CaptureCheck:
    check = CaptureCheck(label=label)
    for sample_id in tbs.CLIP_IDS:
        ref_steps = reference[sample_id]
        cand_steps = candidate[sample_id]
        if len(ref_steps) != len(cand_steps):
            check.text_byte_identical = False
            check.emitted_exact = False
            check.state_allclose = False
            check.diffs.append(
                f"{label} {sample_id}: step-count mismatch ref={len(ref_steps)} cand={len(cand_steps)}"
            )
        for step, (ref_snapshot, cand_snapshot) in enumerate(zip(ref_steps, cand_steps), start=1):
            ref_text = ref_snapshot["text"]
            cand_text = cand_snapshot["text"]
            if text_bytes(ref_text) != text_bytes(cand_text):
                check.text_byte_identical = False
                check.diffs.append(
                    f"{label} {sample_id} step={step} TEXT_DIFF\n"
                    f"  ref={ref_text!r}\n"
                    f"  cand={cand_text!r}"
                )
            if ref_snapshot["emitted_frames"] != cand_snapshot["emitted_frames"]:
                check.emitted_exact = False
                check.diffs.append(
                    f"{label} {sample_id} step={step} EMITTED_DIFF "
                    f"ref={ref_snapshot['emitted_frames']} cand={cand_snapshot['emitted_frames']}"
                )
            summary, failures = compare_state_payload(
                f"{label}.{sample_id[:8]}.step{step}",
                ref_snapshot,
                cand_snapshot,
            )
            merge_check_summary(check, summary)
            if failures:
                check.state_allclose = False
                append_state_failures(check, failures)
    return check


def log_capture_check(check: CaptureCheck) -> None:
    log(
        f"{check.label}: text_byte_identical={check.text_byte_identical} "
        f"emitted_exact={check.emitted_exact} "
        f"torch.equal(all tensors)={check.tensor_bit_equal} "
        f"state_allclose(atol={ATOL},rtol={RTOL})={check.state_allclose} "
        f"tensor_count={check.tensor_count} "
        f"max_abs={check.max_abs:.6g} path={check.max_path or 'n/a'}"
    )
    for diff in check.diffs:
        log(diff)
    if check.suppressed_state_diffs:
        log(f"{check.label}: suppressed_state_diffs={check.suppressed_state_diffs}")


def run_streams_with_states(
    model: Any,
    geom: Any,
    clips: dict[str, np.ndarray],
    *,
    max_batch: int,
    order: Optional[list[int]] = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    sample_ids = list(clips)
    if order is not None:
        sample_ids = [sample_ids[i] for i in order]
    states = [tbs.init_state(model, geom, sample_id, clips[sample_id]) for sample_id in sample_ids]
    captures: dict[str, list[dict[str, Any]]] = {sample_id: [] for sample_id in clips}
    while True:
        ready = [state for state in states if tbs.state_is_ready(geom, state)]
        if not ready:
            break
        for start in range(0, len(ready), max_batch):
            group = ready[start : start + max_batch]
            tbs.process_group(model, geom, group)
            for state in group:
                captures[state.sample_id].append(tbs.state_snapshot(state))
    return captures, {state.sample_id: state for state in states}


def update_raw_ring(geom: Any, raw_ring: np.ndarray, consumed_audio: np.ndarray) -> np.ndarray:
    if len(consumed_audio) >= geom.raw_audio_ring_samples:
        return consumed_audio[-geom.raw_audio_ring_samples :].copy()
    if len(consumed_audio) == 0:
        return raw_ring
    keep = geom.raw_audio_ring_samples - len(consumed_audio)
    return np.concatenate([raw_ring[-keep:], consumed_audio]).astype(np.float32, copy=False)


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


def run_final_chunks(model: Any, geom: Any, states: dict[str, Any]) -> dict[str, str]:
    return {sample_id: process_final_chunk(model, geom, states[sample_id]) for sample_id in tbs.CLIP_IDS}


def compare_final_text(label: str, reference: dict[str, str], candidate: dict[str, str]) -> CaptureCheck:
    check = CaptureCheck(label=label)
    for sample_id in tbs.CLIP_IDS:
        ref_text = reference[sample_id]
        cand_text = candidate[sample_id]
        if text_bytes(ref_text) != text_bytes(cand_text):
            check.text_byte_identical = False
            check.diffs.append(
                f"{label} {sample_id} FINAL_TEXT_DIFF\n"
                f"  ref={ref_text!r}\n"
                f"  cand={cand_text!r}"
            )
    check.state_allclose = True
    check.emitted_exact = True
    return check


def encoder_stream_step_restoring_drop_extra(model: Any, **kwargs: Any) -> tuple[Any, ...]:
    streaming_cfg = model.encoder.streaming_cfg
    original_drop_extra = streaming_cfg.drop_extra_pre_encoded
    try:
        return model.encoder.cache_aware_stream_step(**kwargs)
    finally:
        streaming_cfg.drop_extra_pre_encoded = original_drop_extra


def build_timing_case(model: Any, geom: Any, clips: dict[str, np.ndarray], batch_size: int) -> TimingCase:
    clip_ids = list(tbs.CLIP_IDS)
    states = []
    for row in range(batch_size):
        sample_id = clip_ids[row % len(clip_ids)]
        states.append(tbs.init_state(model, geom, f"{sample_id}#timing{row}", clips[sample_id]))
    prepared = [tbs.prepare_row(model, geom, state) for state in states]
    chunk_mels = [item[0] for item in prepared]
    drop_values = [item[2] for item in prepared]
    if len(set(drop_values)) != 1:
        raise RuntimeError(f"mixed timing drop_extra values: {drop_values}")
    chunk, chunk_len = tbs.stack_processed(chunk_mels)
    clc, clt, clcl = tbs.stack_caches(
        [(state.cache_last_channel, state.cache_last_time, state.cache_last_channel_len) for state in states]
    )
    per_session_hyps = [tbs.clone_tree(state.previous_hypotheses) for state in states]
    flat_hyps = tbs.stack_hypotheses(per_session_hyps)
    previous_hypotheses = None if all(h is None for h in flat_hyps) else flat_hyps
    previous_pred_out = tbs.stack_pred_out([tbs.clone_tree(state.pred_out_stream) for state in states], rnnt=True)

    with torch.inference_mode():
        encoded, encoded_len, _next_clc, _next_clt, _next_clcl = encoder_stream_step_restoring_drop_extra(
            model,
            processed_signal=chunk,
            processed_signal_length=chunk_len,
            cache_last_channel=clc,
            cache_last_time=clt,
            cache_last_channel_len=clcl,
            keep_all_outputs=False,
            drop_extra_pre_encoded=drop_values[0],
        )
    return TimingCase(
        batch_size=batch_size,
        chunk=chunk.detach(),
        chunk_len=chunk_len.detach(),
        cache_last_channel=clc.detach(),
        cache_last_time=clt.detach(),
        cache_last_channel_len=clcl.detach(),
        previous_hypotheses=tbs.clone_tree(previous_hypotheses),
        previous_pred_out=tbs.clone_tree(previous_pred_out),
        drop_extra=drop_values[0],
        encoded=encoded.detach(),
        encoded_len=encoded_len.detach(),
    )


def clone_hyp_repeats(case: TimingCase, total: int, *, serial: bool) -> list[Any]:
    if case.previous_hypotheses is None:
        return [None for _ in range(total)]
    if serial:
        return [
            [[tbs.clone_tree(case.previous_hypotheses[row])] for row in range(case.batch_size)]
            for _ in range(total)
        ]
    return [tbs.clone_tree(case.previous_hypotheses) for _ in range(total)]


def time_cuda(fn: Any, *, warmups: int, repeats: int) -> float:
    for idx in range(warmups):
        fn(idx)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for idx in range(warmups, warmups + repeats):
        fn(idx)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def time_greedy_decode_serial(model: Any, case: TimingCase, warmups: int, repeats: int) -> float:
    set_strategy(model, "greedy")
    hyps = clone_hyp_repeats(case, warmups + repeats, serial=True)

    def run(idx: int) -> None:
        for row in range(case.batch_size):
            partial = None if hyps[idx] is None else hyps[idx][row]
            model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=case.encoded[row : row + 1],
                encoded_lengths=case.encoded_len[row : row + 1],
                return_hypotheses=True,
                partial_hypotheses=partial,
            )

    try:
        return time_cuda(run, warmups=warmups, repeats=repeats)
    finally:
        del hyps


def time_greedy_batch_decode(model: Any, case: TimingCase, warmups: int, repeats: int) -> float:
    set_strategy(model, "greedy_batch")
    hyps = clone_hyp_repeats(case, warmups + repeats, serial=False)

    def run(idx: int) -> None:
        model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=case.encoded,
            encoded_lengths=case.encoded_len,
            return_hypotheses=True,
            partial_hypotheses=hyps[idx],
        )

    try:
        return time_cuda(run, warmups=warmups, repeats=repeats)
    finally:
        del hyps


def serial_pred_for_row(case: TimingCase, row: int) -> Optional[list[Any]]:
    if case.previous_pred_out is None:
        return None
    return [case.previous_pred_out[row]]


def time_greedy_full_serial(model: Any, case: TimingCase, warmups: int, repeats: int) -> float:
    set_strategy(model, "greedy")
    hyps = clone_hyp_repeats(case, warmups + repeats, serial=True)

    def run(idx: int) -> None:
        for row in range(case.batch_size):
            partial = None if hyps[idx] is None else hyps[idx][row]
            tbs.conformer_stream_step_restoring_drop_extra(
                model,
                processed_signal=case.chunk[row : row + 1],
                processed_signal_length=case.chunk_len[row : row + 1],
                cache_last_channel=case.cache_last_channel[:, row : row + 1, ...],
                cache_last_time=case.cache_last_time[:, row : row + 1, ...],
                cache_last_channel_len=case.cache_last_channel_len[row : row + 1],
                keep_all_outputs=False,
                previous_hypotheses=partial,
                previous_pred_out=serial_pred_for_row(case, row),
                drop_extra_pre_encoded=case.drop_extra,
                return_transcription=True,
            )

    try:
        return time_cuda(run, warmups=warmups, repeats=repeats)
    finally:
        del hyps


def time_greedy_batch_full(model: Any, case: TimingCase, warmups: int, repeats: int) -> float:
    set_strategy(model, "greedy_batch")
    hyps = clone_hyp_repeats(case, warmups + repeats, serial=False)

    def run(idx: int) -> None:
        tbs.conformer_stream_step_restoring_drop_extra(
            model,
            processed_signal=case.chunk,
            processed_signal_length=case.chunk_len,
            cache_last_channel=case.cache_last_channel,
            cache_last_time=case.cache_last_time,
            cache_last_channel_len=case.cache_last_channel_len,
            keep_all_outputs=False,
            previous_hypotheses=hyps[idx],
            previous_pred_out=case.previous_pred_out,
            drop_extra_pre_encoded=case.drop_extra,
            return_transcription=True,
        )

    try:
        return time_cuda(run, warmups=warmups, repeats=repeats)
    finally:
        del hyps


def time_encoder_only(model: Any, case: TimingCase, warmups: int, repeats: int) -> float:
    def run(_idx: int) -> None:
        encoder_stream_step_restoring_drop_extra(
            model,
            processed_signal=case.chunk,
            processed_signal_length=case.chunk_len,
            cache_last_channel=case.cache_last_channel,
            cache_last_time=case.cache_last_time,
            cache_last_channel_len=case.cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=case.drop_extra,
        )

    return time_cuda(run, warmups=warmups, repeats=repeats)


def time_fallback_full(model: Any, case: TimingCase, warmups: int, repeats: int) -> float:
    set_strategy(model, "greedy")
    hyps = clone_hyp_repeats(case, warmups + repeats, serial=True)

    def run(idx: int) -> None:
        encoded, encoded_len, _next_clc, _next_clt, _next_clcl = encoder_stream_step_restoring_drop_extra(
            model,
            processed_signal=case.chunk,
            processed_signal_length=case.chunk_len,
            cache_last_channel=case.cache_last_channel,
            cache_last_time=case.cache_last_time,
            cache_last_channel_len=case.cache_last_channel_len,
            keep_all_outputs=False,
            drop_extra_pre_encoded=case.drop_extra,
        )
        for row in range(case.batch_size):
            partial = None if hyps[idx] is None else hyps[idx][row]
            model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded[row : row + 1],
                encoded_lengths=encoded_len[row : row + 1],
                return_hypotheses=True,
                partial_hypotheses=partial,
            )

    try:
        return time_cuda(run, warmups=warmups, repeats=repeats)
    finally:
        del hyps


def run_timing(model: Any, geom: Any, clips: dict[str, np.ndarray]) -> dict[int, TimingResult]:
    warmups = int(os.environ.get("PROBE_TIMING_WARMUPS", str(DEFAULT_TIMING_WARMUPS)))
    repeats = int(os.environ.get("PROBE_TIMING_REPEATS", str(DEFAULT_TIMING_REPEATS)))
    log(f"timing: warmups={warmups} repeats={repeats} batches={BATCHES_TO_TIME}")
    results: dict[int, TimingResult] = {}
    set_strategy(model, "greedy")
    cases = {batch_size: build_timing_case(model, geom, clips, batch_size) for batch_size in BATCHES_TO_TIME}
    for batch_size in BATCHES_TO_TIME:
        case = cases[batch_size]
        greedy_decode = time_greedy_decode_serial(model, case, warmups, repeats)
        greedy_batch_decode = time_greedy_batch_decode(model, case, warmups, repeats)
        greedy_full = time_greedy_full_serial(model, case, warmups, repeats)
        greedy_batch_full = time_greedy_batch_full(model, case, warmups, repeats)
        encoder_ms = time_encoder_only(model, case, warmups, repeats)
        fallback_ms = time_fallback_full(model, case, warmups, repeats)
        result = TimingResult(
            batch_size=batch_size,
            greedy_decode_ms=greedy_decode,
            greedy_batch_decode_ms=greedy_batch_decode,
            greedy_full_ms=greedy_full,
            greedy_batch_full_ms=greedy_batch_full,
            encoder_ms=encoder_ms,
            fallback_full_ms=fallback_ms,
        )
        results[batch_size] = result
        log(
            f"timing B={batch_size}: "
            f"greedy_decode_serial={greedy_decode:.3f}ms "
            f"({result.greedy_decode_ms_per_stream:.3f}ms/stream) "
            f"greedy_batch_decode={greedy_batch_decode:.3f}ms "
            f"({result.greedy_batch_decode_ms_per_stream:.3f}ms/stream, "
            f"reduction={100.0 * result.decode_stream_reduction:.1f}%) "
            f"greedy_full_serial={greedy_full:.3f}ms "
            f"greedy_batch_full={greedy_batch_full:.3f}ms "
            f"full_step_x={result.greedy_batch_full_speedup:.2f} "
            f"encoder_only={encoder_ms:.3f}ms "
            f"fallback_full={fallback_ms:.3f}ms "
            f"fallback_x={result.fallback_full_speedup:.2f}"
        )
        gc.collect()
        torch.cuda.empty_cache()
    return results


def print_gpu_free(prefix: str) -> None:
    if not torch.cuda.is_available():
        log(f"{prefix}: cuda unavailable")
        return
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    log(
        f"{prefix}: torch cuda mem free={free_bytes / (1024 ** 2):.0f}MiB "
        f"total={total_bytes / (1024 ** 2):.0f}MiB"
    )
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        log(f"{prefix}: nvidia-smi memory.used/free MiB: {out}")
    except Exception as exc:
        log(f"{prefix}: nvidia-smi unavailable: {exc}")


def remove_pycache() -> None:
    removed = 0
    for path in THIS_DIR.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    log(f"cleanup: removed __pycache__ dirs under {THIS_DIR}: {removed}")


def decision(correctness: list[CaptureCheck], final_checks: list[CaptureCheck], timings: dict[int, TimingResult]) -> str:
    text_ok = all(check.text_byte_identical for check in correctness + final_checks)
    state_ok = all(check.state_allclose and check.emitted_exact for check in correctness)
    decode_ok = all(timings[b].decode_stream_reduction >= 0.25 for b in (4, 8))
    full_ok = all(timings[b].greedy_batch_full_speedup >= 1.5 for b in (4, 8))
    if text_ok and state_ok and decode_ok and full_ok:
        return (
            "GO greedy_batch: correctness passed; "
            f"decode reductions B4={100.0 * timings[4].decode_stream_reduction:.1f}% "
            f"B8={100.0 * timings[8].decode_stream_reduction:.1f}%; "
            f"full-step x B4={timings[4].greedy_batch_full_speedup:.2f} "
            f"B8={timings[8].greedy_batch_full_speedup:.2f}"
        )

    fallback_ok = timings[4].fallback_full_speedup >= 1.5 and timings[8].fallback_full_speedup >= 1.5
    reason = []
    if not text_ok:
        reason.append("text/final byte divergence")
    if text_ok and not state_ok:
        reason.append("state/emitted mismatch")
    if text_ok and state_ok and not decode_ok:
        reason.append("decode speed below 25% reduction")
    if text_ok and state_ok and decode_ok and not full_ok:
        reason.append("full-step speed below 1.5x")
    reason_text = ", ".join(reason) if reason else "gate failed"
    if fallback_ok:
        return (
            f"NO-GO greedy_batch ({reason_text}) -> fallback viable: "
            f"fallback full-step x B4={timings[4].fallback_full_speedup:.2f} "
            f"B8={timings[8].fallback_full_speedup:.2f}"
        )
    return (
        f"STOP: NO-GO greedy_batch ({reason_text}) and fallback below 1.5x: "
        f"fallback full-step x B4={timings[4].fallback_full_speedup:.2f} "
        f"B8={timings[8].fallback_full_speedup:.2f}"
    )


def main() -> None:
    model = None
    try:
        model = load_model()
        geom = tbs.build_geometry(model)
        clips = {sample_id: tbs.load_clip(sample_id) for sample_id in tbs.CLIP_IDS}
        chunk_counts = {
            sample_id: max(0, (len(audio) - geom.hop_samples) // (geom.shift_frames * geom.hop_samples))
            for sample_id, audio in clips.items()
        }
        log(
            "geometry: "
            f"shift_frames={geom.shift_frames} pre_cache={geom.pre_encode_cache_size} "
            f"drop_extra={geom.drop_extra} hop={geom.hop_samples} "
            f"new_audio={geom.preprocess_new_audio_samples} K={geom.constant_preprocess_samples} "
            f"warmup_ms={geom.warmup_ms} att_context=[70,1]"
        )
        for sample_id in tbs.CLIP_IDS:
            log(f"clip {sample_id}: samples={len(clips[sample_id])} normal_chunks={chunk_counts[sample_id]}")
        if len(set(chunk_counts.values())) != 1:
            raise AssertionError(f"clips do not have equal normal chunk counts: {chunk_counts}")

        correctness_checks: list[CaptureCheck] = []
        final_checks: list[CaptureCheck] = []
        with torch.inference_mode():
            log("REFERENCE: strategy=greedy loop_labels=False B=1 full streams")
            set_strategy(model, "greedy")
            reference, reference_states = run_streams_with_states(model, geom, clips, max_batch=1)
            reference_final = run_final_chunks(model, geom, reference_states)
            log(f"reference chunks_per_clip={len(reference[tbs.CLIP_IDS[0]])}")

            log("CANDIDATE: strategy=greedy_batch loop_labels=True use_cuda_graph_decoder=False")
            set_strategy(model, "greedy_batch")
            for batch_size in BATCHES_TO_VERIFY:
                candidate, candidate_states = run_streams_with_states(model, geom, clips, max_batch=batch_size)
                check = compare_captures_verbose(f"greedy_batch_B{batch_size}", reference, candidate)
                log_capture_check(check)
                correctness_checks.append(check)
                candidate_final = run_final_chunks(model, geom, candidate_states)
                final_check = compare_final_text(f"greedy_batch_B{batch_size}_final", reference_final, candidate_final)
                log_capture_check(final_check)
                final_checks.append(final_check)

            permuted, permuted_states = run_streams_with_states(
                model,
                geom,
                clips,
                max_batch=4,
                order=tbs.PERMUTED_ORDER,
            )
            perm_check = compare_captures_verbose(
                f"greedy_batch_B4_permuted_{tbs.PERMUTED_ORDER}",
                reference,
                permuted,
            )
            log_capture_check(perm_check)
            correctness_checks.append(perm_check)
            perm_final = run_final_chunks(model, geom, permuted_states)
            perm_final_check = compare_final_text("greedy_batch_B4_permuted_final", reference_final, perm_final)
            log_capture_check(perm_final_check)
            final_checks.append(perm_final_check)

        timings = run_timing(model, geom, clips)
        verdict = decision(correctness_checks, final_checks, timings)
        log(f"DECISION: {verdict}")
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_gpu_free("cleanup")
        log("cleanup: probe started no long-running child processes; no PIDs to kill")
        remove_pycache()


if __name__ == "__main__":
    main()
