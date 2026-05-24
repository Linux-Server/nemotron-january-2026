#!/usr/bin/env python3
"""Padded-T finalize eager-vs-eager probe.

This is a local GO/NO-GO probe for replacing exact finalize-T encoder CUDA graph
buckets with one padded T_max bucket. It intentionally does not import or touch
production graph managers; it compares:

  exact eager finalize input at real T
  vs
  zero-padded eager finalize input at T_max with processed_signal_length=real T

The decode path is the real NeMo conformer_stream_step RNNT path used by the
server. Encoder outputs are captured by wrapping encoder.cache_aware_stream_step
inside this process only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent
HARNESS_PATH = REPO / "proj-2026-05-23-1731" / "decoder_graph_harness.py"
DEFAULT_OUT = PROJECT / "padded-t-findings.md"
DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
DEFAULT_T_MIN = 42
DEFAULT_T_MAX = 60
DEFAULT_NORMAL_CHUNKS = 20
RTOL = 1.0e-5
ATOL = 1.0e-4


@dataclasses.dataclass
class ForkState:
    cache_last_channel: Any
    cache_last_time: Any
    cache_last_channel_len: Any
    previous_hypotheses: Any
    pred_out_stream: Any


@dataclasses.dataclass
class DecodeSummary:
    tokens: tuple[int, ...]
    text: str


@dataclasses.dataclass
class RunOutput:
    decode: DecodeSummary
    encoded: torch.Tensor
    encoded_len: torch.Tensor
    cache_last_channel_len: torch.Tensor
    physical_t: int
    encoded_shape: tuple[int, ...]


@dataclasses.dataclass
class ProbeRow:
    real_t: int
    row_t: int
    drop_extra: int
    exact_encoded_shape: tuple[int, ...]
    padded_encoded_shape: tuple[int, ...]
    exact_encoded_len: list[int]
    padded_encoded_len: list[int]
    exact_cache_len: list[int]
    padded_cache_len: list[int]
    tokens_exact: bool
    text_exact: bool
    cropped_tokens_exact: bool
    cropped_text_exact: bool
    encoded_len_exact: bool
    cache_len_exact: bool
    encoded_allclose: bool
    encoded_byte_equal: bool
    encoded_compare_frames: int
    encoded_max_abs: float
    encoded_max_rel: float
    exact_text: str
    padded_text: str
    cropped_text: str
    exact_tokens: tuple[int, ...]
    padded_tokens: tuple[int, ...]
    cropped_tokens: tuple[int, ...]
    error: Optional[str] = None

    @property
    def gate_ok(self) -> bool:
        return (
            self.error is None
            and self.tokens_exact
            and self.text_exact
            and self.encoded_len_exact
            and self.cache_len_exact
            and self.encoded_allclose
        )


@dataclasses.dataclass
class ContinuationEvent:
    event: str
    kind: str
    text: str
    tokens: tuple[int, ...]
    emitted_frames: int
    cache_last_channel_len: list[int]
    continuous_state: str


@dataclasses.dataclass
class FinalizeCallRecord:
    ordinal: int
    mode: str
    real_t: int
    fed_t: int
    input_cache_len: list[int]
    fork_cache_len: list[int]
    text: str
    tokens: tuple[int, ...]


@dataclasses.dataclass
class FinalizeObservation:
    ordinal: int
    reason: str
    session_cache_before: list[int]
    session_cache_after: list[int]
    session_emitted_frames_before: int
    session_emitted_frames_after: int
    continuous_state_after: str
    call: Optional[FinalizeCallRecord]

    @property
    def session_cache_retained(self) -> bool:
        return self.session_cache_before == self.session_cache_after

    @property
    def session_reset(self) -> bool:
        return not self.session_cache_after and self.session_emitted_frames_after == 0

    @property
    def session_adopted_fork_cache_len(self) -> bool:
        if self.call is None:
            return False
        return (
            self.session_cache_after == self.call.fork_cache_len
            and self.session_cache_after != self.session_cache_before
        )


@dataclasses.dataclass
class ContinuationRun:
    mode: str
    events: list[ContinuationEvent]
    finalizes: list[FinalizeObservation]
    websocket_payloads: list[dict[str, Any]]
    clip_path: str
    pre_chunks: int
    post_chunks: int


@dataclasses.dataclass
class ContinuationComparison:
    exact: ContinuationRun
    padded: ContinuationRun
    first_final_exact: bool
    post_finalize_exact: bool
    full_stream_exact: bool
    first_divergence: Optional[str]
    session_retained_own_cache: bool
    session_reset_after_finalize: bool
    session_adopted_fork_cache: bool

    @property
    def go(self) -> bool:
        return (
            self.post_finalize_exact
            and not self.session_adopted_fork_cache
            and (self.session_retained_own_cache or self.session_reset_after_finalize)
        )


def _load_harness() -> Any:
    spec = importlib.util.spec_from_file_location("decoder_graph_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load harness spec from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _configure_probe_env() -> None:
    os.environ.update(
        {
            "NEMOTRON_CONTINUOUS": "1",
            "NEMOTRON_SCHEDULER_B1": "1",
            "NEMOTRON_BATCH_SCHED": "1",
            "NEMOTRON_BATCH_FINALIZE": "1",
            "NEMOTRON_BATCH_MAX_SIZE": "1",
            "NEMOTRON_MODEL_LANES": "1",
            "NEMOTRON_FINALIZE_SILENCE_MS": "0",
            "NEMOTRON_WARMUP_MS": "200",
            "NEMOTRON_DECODING": "greedy",
            "NEMOTRON_TARGET_LANG": "en-US",
            "NEMOTRON_FORK_ASSERT": "1",
        }
    )
    for name in (
        "NEMOTRON_EOU_PROBE",
        "NEMOTRON_ENCODER_COMPILE",
        "NEMOTRON_ENCODER_CUDAGRAPH",
        "NEMOTRON_ENCODER_CUDAGRAPH_FINALIZE",
        "NEMOTRON_BATCH_FINALIZE_PREPROC",
    ):
        os.environ.pop(name, None)
    torch.backends.cudnn.benchmark = False


def _clone_fork_state(fork: Any) -> ForkState:
    from nemotron_speech.server import clone_hypotheses_deep, clone_tree, tensor_clone

    return ForkState(
        cache_last_channel=(
            tensor_clone(fork.cache_last_channel)
            if fork.cache_last_channel is not None
            else None
        ),
        cache_last_time=(
            tensor_clone(fork.cache_last_time)
            if fork.cache_last_time is not None
            else None
        ),
        cache_last_channel_len=(
            tensor_clone(fork.cache_last_channel_len)
            if fork.cache_last_channel_len is not None
            else None
        ),
        previous_hypotheses=clone_hypotheses_deep(fork.previous_hypotheses),
        pred_out_stream=clone_tree(fork.pred_out_stream),
    )


def _clone_state_for_run(state: ForkState) -> ForkState:
    from nemotron_speech.server import clone_hypotheses_deep, clone_tree, tensor_clone

    return ForkState(
        cache_last_channel=(
            tensor_clone(state.cache_last_channel)
            if state.cache_last_channel is not None
            else None
        ),
        cache_last_time=(
            tensor_clone(state.cache_last_time)
            if state.cache_last_time is not None
            else None
        ),
        cache_last_channel_len=(
            tensor_clone(state.cache_last_channel_len)
            if state.cache_last_channel_len is not None
            else None
        ),
        previous_hypotheses=clone_hypotheses_deep(state.previous_hypotheses),
        pred_out_stream=clone_tree(state.pred_out_stream),
    )


def _tensor_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def _tokens_from_hyp(harness: Any, hyp: Any) -> tuple[int, ...]:
    return tuple(harness._token_ids_from_hyp(hyp))


def _decode_summary(server: Any, harness: Any, hyps: Any) -> DecodeSummary:
    hyp = hyps[0] if isinstance(hyps, (list, tuple)) and hyps else hyps
    text = "" if hyp is None else server._extract_hypothesis_text(hyp)
    return DecodeSummary(tokens=_tokens_from_hyp(harness, hyp), text=text)


def _session_hypothesis(session: Any) -> Any:
    hyps = getattr(session, "previous_hypotheses", None)
    if not hyps:
        return None
    return hyps[0]


def _session_tokens(harness: Any, session: Any) -> tuple[int, ...]:
    return _tokens_from_hyp(harness, _session_hypothesis(session))


def _cache_len_list(value: Any) -> list[int]:
    if value is None:
        return []
    if torch.is_tensor(value):
        return _tensor_int_list(value)
    return [int(x) for x in value]


def _session_cache_len(session: Any) -> list[int]:
    return _cache_len_list(getattr(session, "cache_last_channel_len", None))


def _capture_session_event(
    *,
    harness: Any,
    session: Any,
    event: str,
    kind: str,
    text: Optional[str] = None,
) -> ContinuationEvent:
    return ContinuationEvent(
        event=event,
        kind=kind,
        text=session.current_text if text is None else str(text),
        tokens=_session_tokens(harness, session),
        emitted_frames=int(getattr(session, "emitted_frames", 0)),
        cache_last_channel_len=_session_cache_len(session),
        continuous_state=str(getattr(session, "continuous_state", "")),
    )


class _CapturingWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self.payloads: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.payloads.append(json.loads(data))


@contextlib.contextmanager
def _capture_encoder_outputs(model: Any):
    encoder = model.encoder
    attr_name = "cache_aware_stream_step"
    had_instance_attr = attr_name in vars(encoder)
    original_instance_attr = vars(encoder).get(attr_name)
    original_callable = getattr(encoder, attr_name)
    captured: dict[str, torch.Tensor] = {}

    def wrapper(*args: Any, **kwargs: Any):
        outputs = original_callable(*args, **kwargs)
        if len(outputs) < 5:
            raise RuntimeError(f"expected at least 5 encoder outputs, got {len(outputs)}")
        captured["encoded"] = outputs[0].detach().clone()
        captured["encoded_len"] = outputs[1].detach().clone()
        captured["cache_last_channel_len"] = outputs[4].detach().clone()
        return outputs

    object.__setattr__(encoder, attr_name, wrapper)
    try:
        yield captured
    finally:
        if had_instance_attr:
            object.__setattr__(encoder, attr_name, original_instance_attr)
        else:
            object.__delattr__(encoder, attr_name)


@contextlib.contextmanager
def _patch_finalize_encoder_input(
    *,
    server: Any,
    harness: Any,
    mode: str,
    t_max: int,
    calls: list[FinalizeCallRecord],
):
    original = server._conformer_stream_step

    def wrapper(*args: Any, **kwargs: Any):
        is_finalize = bool(kwargs.get("keep_all_outputs", False))
        call_kwargs = kwargs
        real_t = None
        fed_t = None
        input_cache_len: list[int] = []
        if is_finalize and "processed_signal" in kwargs:
            processed_signal = kwargs["processed_signal"]
            processed_signal_length = kwargs.get("processed_signal_length")
            if not torch.is_tensor(processed_signal):
                raise RuntimeError("finalize processed_signal is not a tensor")
            real_t = int(processed_signal.shape[-1])
            if torch.is_tensor(processed_signal_length) and processed_signal_length.numel() > 0:
                real_t = int(processed_signal_length.detach().max().item())
            physical_t = int(processed_signal.shape[-1])
            if real_t > int(t_max):
                raise RuntimeError(f"finalize real T={real_t} exceeds T_max={t_max}")
            fed_signal = processed_signal
            if mode == "padded" and physical_t < int(t_max):
                fed_signal = processed_signal.new_zeros(
                    (*processed_signal.shape[:-1], int(t_max))
                )
                fed_signal[..., :physical_t].copy_(processed_signal)
                call_kwargs = dict(kwargs)
                call_kwargs["processed_signal"] = fed_signal
            fed_t = int(fed_signal.shape[-1])
            input_cache_len = _cache_len_list(kwargs.get("cache_last_channel_len"))

        result = original(*args, **call_kwargs)
        if is_finalize:
            hyps = result[5] if len(result) > 5 else None
            hyp = hyps[0] if isinstance(hyps, (list, tuple)) and hyps else hyps
            calls.append(
                FinalizeCallRecord(
                    ordinal=len(calls),
                    mode=mode,
                    real_t=int(real_t) if real_t is not None else -1,
                    fed_t=int(fed_t) if fed_t is not None else -1,
                    input_cache_len=input_cache_len,
                    fork_cache_len=_cache_len_list(result[4] if len(result) > 4 else None),
                    text="" if hyp is None else server._extract_hypothesis_text(hyp),
                    tokens=_tokens_from_hyp(harness, hyp),
                )
            )
        return result

    object.__setattr__(server, "_conformer_stream_step", wrapper)
    try:
        yield
    finally:
        object.__setattr__(server, "_conformer_stream_step", original)


@contextlib.contextmanager
def _capture_prepared_finalize_events(
    *,
    server: Any,
    harness: Any,
    events: list[ContinuationEvent],
):
    original = server._continuous_emit_prepared_finalize_locked

    async def wrapper(item: Any):
        fork = item.fork
        tokens = _tokens_from_hyp(harness, _session_hypothesis(fork)) if fork is not None else ()
        events.append(
            ContinuationEvent(
                event=f"final:{sum(1 for event in events if event.kind == 'final'):04d}",
                kind="final",
                text=str(item.final_text),
                tokens=tokens,
                emitted_frames=int(getattr(item.session, "emitted_frames", 0)),
                cache_last_channel_len=_session_cache_len(item.session),
                continuous_state=str(getattr(item.session, "continuous_state", "")),
            )
        )
        return await original(item)

    object.__setattr__(server, "_continuous_emit_prepared_finalize_locked", wrapper)
    try:
        yield
    finally:
        object.__setattr__(server, "_continuous_emit_prepared_finalize_locked", original)


async def _feed_one_steady_chunk(
    *,
    server: Any,
    harness: Any,
    runtime: Any,
    events: list[ContinuationEvent],
) -> None:
    session = runtime.session
    await harness.feed_until_ready(server, runtime)
    texts = harness._call_model_path(
        server,
        server._process_ready_batch,
        [session],
        lane_id=0,
    )
    server._scheduler_ready.discard(session.id)
    session.scheduler_ready_since = None
    text = texts.get(session.id)
    if text is not None:
        session.current_text = text
    events.append(
        _capture_session_event(
            harness=harness,
            session=session,
            event=f"chunk:{runtime.next_chunk_index:04d}",
            kind="chunk",
            text=session.current_text,
        )
    )
    runtime.next_chunk_index += 1


async def _trigger_vad_debounce_finalize(
    *,
    server: Any,
    session: Any,
) -> None:
    queue = session.continuous_event_queue
    if queue is None:
        raise RuntimeError("session has no continuous_event_queue")

    await server._scheduler_continuous_handle_vad_stop_locked(session)
    stop_seq = int(session.continuous_stop_seq)

    await server._continuous_cancel_debounce_locked(session, invalidate=False)
    while not queue.empty():
        try:
            _event = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        else:
            queue.task_done()

    await queue.put(("debounce_expired", stop_seq, time.perf_counter()))
    progressed = await server._scheduler_drain_once()
    if not progressed:
        raise RuntimeError("scheduler did not process the debounce finalize event")
    if not queue.empty():
        raise RuntimeError("debounce finalize left unexpected queued events")


def _set_reproducible_state(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


async def _run_continuation_sequence(
    *,
    server: Any,
    harness: Any,
    clip: Any,
    mode: str,
    pre_chunks: int,
    post_chunks: int,
    t_max: int,
    second_finalize: bool,
    seed: int,
) -> ContinuationRun:
    _set_reproducible_state(seed)
    _, ASRSession = harness.import_server_classes()
    websocket = _CapturingWebSocket()
    session = ASRSession(id="continuation", websocket=websocket, target_lang=server.target_lang)
    session.continuous_event_queue = asyncio.Queue()
    server._init_session_without_synthetic_warmup(session)
    runtime = harness.SessionRuntime(session=session, clip=clip)
    server.sessions = {session.id: session}
    server._scheduler_ready.clear()
    server._scheduler_batch_first_ready.clear()
    session.scheduler_ready_since = None

    events: list[ContinuationEvent] = []
    finalizes: list[FinalizeObservation] = []
    calls: list[FinalizeCallRecord] = []

    try:
        with _patch_finalize_encoder_input(
            server=server,
            harness=harness,
            mode=mode,
            t_max=t_max,
            calls=calls,
        ), _capture_prepared_finalize_events(
            server=server,
            harness=harness,
            events=events,
        ):
            for _ in range(int(pre_chunks)):
                await _feed_one_steady_chunk(
                    server=server,
                    harness=harness,
                    runtime=runtime,
                    events=events,
                )

            before_cache = _session_cache_len(session)
            before_frames = int(session.emitted_frames)
            before_calls = len(calls)
            await _trigger_vad_debounce_finalize(server=server, session=session)
            finalizes.append(
                FinalizeObservation(
                    ordinal=len(finalizes),
                    reason="debounce_expired",
                    session_cache_before=before_cache,
                    session_cache_after=_session_cache_len(session),
                    session_emitted_frames_before=before_frames,
                    session_emitted_frames_after=int(session.emitted_frames),
                    continuous_state_after=str(session.continuous_state),
                    call=calls[before_calls] if len(calls) > before_calls else None,
                )
            )

            for _ in range(int(post_chunks)):
                await _feed_one_steady_chunk(
                    server=server,
                    harness=harness,
                    runtime=runtime,
                    events=events,
                )

            if second_finalize:
                before_cache = _session_cache_len(session)
                before_frames = int(session.emitted_frames)
                before_calls = len(calls)
                await _trigger_vad_debounce_finalize(server=server, session=session)
                finalizes.append(
                    FinalizeObservation(
                        ordinal=len(finalizes),
                        reason="debounce_expired",
                        session_cache_before=before_cache,
                        session_cache_after=_session_cache_len(session),
                        session_emitted_frames_before=before_frames,
                        session_emitted_frames_after=int(session.emitted_frames),
                        continuous_state_after=str(session.continuous_state),
                        call=calls[before_calls] if len(calls) > before_calls else None,
                    )
                )
    finally:
        server.sessions = {}
        server._scheduler_ready.clear()
        server._scheduler_batch_first_ready.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return ContinuationRun(
        mode=mode,
        events=events,
        finalizes=finalizes,
        websocket_payloads=websocket.payloads,
        clip_path=str(clip.path),
        pre_chunks=int(pre_chunks),
        post_chunks=int(post_chunks),
    )


def _event_signature(event: ContinuationEvent) -> tuple[str, str, tuple[int, ...]]:
    return (event.event, event.text, event.tokens)


def _first_event_divergence(
    exact_events: list[ContinuationEvent],
    padded_events: list[ContinuationEvent],
    *,
    start_index: int,
) -> Optional[str]:
    exact_tail = exact_events[start_index:]
    padded_tail = padded_events[start_index:]
    if len(exact_tail) != len(padded_tail):
        return (
            f"post-finalize event count {len(exact_tail)} != {len(padded_tail)} "
            f"(exact total={len(exact_events)} padded total={len(padded_events)})"
        )
    for offset, (exact_event, padded_event) in enumerate(zip(exact_tail, padded_tail, strict=True)):
        if _event_signature(exact_event) != _event_signature(padded_event):
            return (
                f"event[{start_index + offset}] {exact_event.event}: "
                f"tokens {list(exact_event.tokens)} != {list(padded_event.tokens)}; "
                f"text {exact_event.text!r} != {padded_event.text!r}"
            )
    return None


def _compare_continuation_runs(
    exact: ContinuationRun,
    padded: ContinuationRun,
) -> ContinuationComparison:
    first_final_index = next(
        (index for index, event in enumerate(exact.events) if event.kind == "final"),
        None,
    )
    padded_first_final_index = next(
        (index for index, event in enumerate(padded.events) if event.kind == "final"),
        None,
    )
    if first_final_index is None or padded_first_final_index is None:
        first_final_exact = False
        post_exact = False
        full_exact = False
        first_divergence = "missing first finalize event"
    else:
        first_final_exact = (
            _event_signature(exact.events[first_final_index])
            == _event_signature(padded.events[padded_first_final_index])
        )
        post_start = first_final_index + 1
        first_divergence = _first_event_divergence(
            exact.events,
            padded.events,
            start_index=post_start,
        )
        post_exact = first_divergence is None
        full_divergence = _first_event_divergence(
            exact.events,
            padded.events,
            start_index=0,
        )
        full_exact = full_divergence is None
        if first_divergence is None and full_divergence is not None:
            first_divergence = full_divergence

    first_observations = [
        run.finalizes[0]
        for run in (exact, padded)
        if run.finalizes
    ]
    session_retained = bool(first_observations) and all(
        obs.session_cache_retained for obs in first_observations
    )
    session_reset = bool(first_observations) and all(
        obs.session_reset for obs in first_observations
    )
    adopted_fork = any(obs.session_adopted_fork_cache_len for obs in first_observations)

    return ContinuationComparison(
        exact=exact,
        padded=padded,
        first_final_exact=first_final_exact,
        post_finalize_exact=post_exact,
        full_stream_exact=full_exact,
        first_divergence=first_divergence if not post_exact else None,
        session_retained_own_cache=session_retained,
        session_reset_after_finalize=session_reset,
        session_adopted_fork_cache=adopted_fork,
    )


def _run_finalize_step(
    *,
    server: Any,
    harness: Any,
    processed_signal: torch.Tensor,
    real_t: int,
    state: ForkState,
    drop_extra: int,
) -> RunOutput:
    run_state = _clone_state_for_run(state)
    processed_signal = processed_signal.detach().clone()
    processed_signal_length = torch.tensor(
        [int(real_t)],
        dtype=torch.long,
        device=processed_signal.device,
    )
    model = server._current_inference_model()
    with torch.inference_mode(), _capture_encoder_outputs(model) as captured:
        result = server._conformer_stream_step(
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length,
            cache_last_channel=run_state.cache_last_channel,
            cache_last_time=run_state.cache_last_time,
            cache_last_channel_len=run_state.cache_last_channel_len,
            keep_all_outputs=True,
            previous_hypotheses=run_state.previous_hypotheses,
            previous_pred_out=run_state.pred_out_stream,
            drop_extra_pre_encoded=int(drop_extra),
            return_transcription=True,
        )
    torch.cuda.synchronize()
    if "encoded" not in captured:
        raise RuntimeError("encoder wrapper did not capture outputs")
    hyps = result[5]
    return RunOutput(
        decode=_decode_summary(server, harness, hyps),
        encoded=captured["encoded"],
        encoded_len=captured["encoded_len"],
        cache_last_channel_len=captured["cache_last_channel_len"],
        physical_t=int(processed_signal.shape[-1]),
        encoded_shape=tuple(int(x) for x in captured["encoded"].shape),
    )


def _run_decoder_only(
    *,
    server: Any,
    harness: Any,
    encoded: torch.Tensor,
    encoded_len: torch.Tensor,
    state: ForkState,
) -> DecodeSummary:
    run_state = _clone_state_for_run(state)
    with torch.inference_mode():
        hyps = server.model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded.detach().clone(),
            encoded_lengths=encoded_len.detach().clone(),
            return_hypotheses=True,
            partial_hypotheses=run_state.previous_hypotheses,
        )
    torch.cuda.synchronize()
    return _decode_summary(server, harness, hyps)


def _max_abs_rel(expected: torch.Tensor, actual: torch.Tensor) -> tuple[float, float]:
    if expected.numel() == 0 and actual.numel() == 0:
        return 0.0, 0.0
    exp = expected.to(torch.float32)
    act = actual.to(torch.float32)
    diff = (exp - act).abs()
    max_abs = float(diff.max().item())
    denom = exp.abs().clamp_min(1.0e-12)
    max_rel = float((diff / denom).max().item())
    return max_abs, max_rel


def _assert_norm_eval(server: Any) -> dict[str, Any]:
    model_eval = not bool(server.model.training)
    encoder_eval = not bool(server.model.encoder.training)
    norm_types = sorted(
        {
            str(getattr(module, "norm_type"))
            for module in server.model.encoder.modules()
            if module.__class__.__name__ == "ConformerConvolution"
            and hasattr(module, "norm_type")
        }
    )
    unsafe_norms = [
        norm_type
        for norm_type in norm_types
        if norm_type == "instance_norm" or norm_type.startswith("group_norm")
    ]
    batch_norm_eval_match = model_eval and encoder_eval and norm_types == ["batch_norm"]
    pad_stat_safe = model_eval and encoder_eval and not unsafe_norms
    info = {
        "model_eval": model_eval,
        "encoder_eval": encoder_eval,
        "conformer_conv_norm_types": norm_types,
        "batch_norm_eval_match": batch_norm_eval_match,
        "pad_stat_safe": pad_stat_safe,
        "unsafe_norms": unsafe_norms,
        "ok": pad_stat_safe,
    }
    if not pad_stat_safe:
        raise RuntimeError(f"norm/eval assertion failed: {info}")
    return info


def _clip_float32_samples(clip: Any, start: int, count: int) -> np.ndarray:
    end = start + count
    if end > len(clip.pcm_i16):
        raise RuntimeError(
            f"clip {clip.path} has only {len(clip.pcm_i16)} samples; need {end}"
        )
    return clip.pcm_i16[start:end].astype(np.float32) / 32768.0


async def _prepare_parent_runtime(
    *,
    server: Any,
    harness: Any,
    normal_chunks: int,
    max_tail_samples: int,
    audio_dir: Path,
) -> Any:
    _, ASRSession = harness.import_server_classes()
    clips = harness.select_audio_clips(
        server,
        audio_dir=audio_dir,
        session_count=1,
        normal_chunks=int(normal_chunks),
        final_tail_samples=int(max_tail_samples),
        allow_reuse=True,
        session_prefix="pad",
    )
    clip = clips[0]
    session = ASRSession(id=clip.session_id, websocket=None, target_lang=server.target_lang)
    session.continuous_event_queue = asyncio.Queue()
    server._init_session_without_synthetic_warmup(session)
    runtime = harness.SessionRuntime(session=session, clip=clip)
    server.sessions = {session.id: session}

    for chunk_index in range(int(normal_chunks)):
        await harness.feed_until_ready(server, runtime)
        texts = harness._call_model_path(
            server,
            server._process_ready_batch,
            [session],
            lane_id=0,
        )
        text = texts.get(session.id)
        if text is not None:
            session.current_text = text
        runtime.next_chunk_index = chunk_index + 1

    torch.cuda.synchronize()
    return runtime


def _make_finalize_row_for_t(
    *,
    server: Any,
    runtime: Any,
    real_t: int,
    t_base: int,
    tail_start: int,
    base_pending: np.ndarray,
) -> Any:
    session = runtime.session
    desired_pending = max(0, int(real_t) - int(t_base)) * int(server.hop_samples)
    if desired_pending <= len(base_pending):
        pending = base_pending[:desired_pending].copy()
    else:
        extra_count = desired_pending - len(base_pending)
        extra = _clip_float32_samples(runtime.clip, tail_start, extra_count)
        pending = np.concatenate([base_pending, extra]).astype(np.float32, copy=False)

    session.pending_audio = np.ascontiguousarray(pending)
    session.accumulated_audio = session.pending_audio
    session.total_audio_samples = int(session.emitted_frames) * int(server.hop_samples) + len(pending)

    item = server._continuous_prepare_finalize_item_locked(
        session,
        reason=f"padded_t_probe:T{real_t}",
        expected_generation=session.scheduler_generation,
    )
    row = server._prepare_final_fork_batch_row(item)
    if row is None:
        raise RuntimeError(f"T={real_t}: finalize row was not built")
    actual_t = int(row.chunk_mel.shape[-1])
    if actual_t != int(real_t):
        raise RuntimeError(
            f"T={real_t}: built row has T={actual_t}; "
            f"pending={desired_pending} base={t_base}"
        )
    if int(row.drop_extra) != int(server.drop_extra):
        raise RuntimeError(
            f"T={real_t}: drop_extra={row.drop_extra}, expected {server.drop_extra}"
        )
    return row


def _probe_one_t(
    *,
    server: Any,
    harness: Any,
    row: Any,
    real_t: int,
    t_max: int,
    atol: float,
) -> ProbeRow:
    try:
        exact_signal = row.chunk_mel.detach().clone()
        padded_signal = exact_signal.new_zeros(
            (exact_signal.shape[0], exact_signal.shape[1], int(t_max))
        )
        padded_signal[..., : int(real_t)].copy_(exact_signal)

        initial_state = _clone_fork_state(row.item.fork)
        exact = _run_finalize_step(
            server=server,
            harness=harness,
            processed_signal=exact_signal,
            real_t=real_t,
            state=initial_state,
            drop_extra=row.drop_extra,
        )
        padded = _run_finalize_step(
            server=server,
            harness=harness,
            processed_signal=padded_signal,
            real_t=real_t,
            state=initial_state,
            drop_extra=row.drop_extra,
        )

        encoded_len_exact = bool(torch.equal(exact.encoded_len, padded.encoded_len))
        cache_len_exact = bool(
            torch.equal(exact.cache_last_channel_len, padded.cache_last_channel_len)
        )
        compare_frames = int(exact.encoded_len.detach().max().item())
        if exact.encoded.shape[-1] < compare_frames or padded.encoded.shape[-1] < compare_frames:
            raise RuntimeError(
                f"encoded output shorter than encoded_len: "
                f"exact_shape={tuple(exact.encoded.shape)} "
                f"padded_shape={tuple(padded.encoded.shape)} "
                f"compare_frames={compare_frames}"
            )
        exact_region = exact.encoded[..., :compare_frames]
        padded_region = padded.encoded[..., :compare_frames]
        encoded_byte_equal = bool(torch.equal(exact_region, padded_region))
        encoded_allclose = bool(torch.allclose(exact_region, padded_region, atol=atol, rtol=RTOL))
        max_abs, max_rel = _max_abs_rel(exact_region, padded_region)

        cropped_encoded = padded.encoded[..., :compare_frames]
        cropped_decode = _run_decoder_only(
            server=server,
            harness=harness,
            encoded=cropped_encoded,
            encoded_len=padded.encoded_len,
            state=initial_state,
        )

        return ProbeRow(
            real_t=int(real_t),
            row_t=int(row.chunk_mel.shape[-1]),
            drop_extra=int(row.drop_extra),
            exact_encoded_shape=exact.encoded_shape,
            padded_encoded_shape=padded.encoded_shape,
            exact_encoded_len=_tensor_int_list(exact.encoded_len),
            padded_encoded_len=_tensor_int_list(padded.encoded_len),
            exact_cache_len=_tensor_int_list(exact.cache_last_channel_len),
            padded_cache_len=_tensor_int_list(padded.cache_last_channel_len),
            tokens_exact=exact.decode.tokens == padded.decode.tokens,
            text_exact=exact.decode.text == padded.decode.text,
            cropped_tokens_exact=exact.decode.tokens == cropped_decode.tokens,
            cropped_text_exact=exact.decode.text == cropped_decode.text,
            encoded_len_exact=encoded_len_exact,
            cache_len_exact=cache_len_exact,
            encoded_allclose=encoded_allclose,
            encoded_byte_equal=encoded_byte_equal,
            encoded_compare_frames=compare_frames,
            encoded_max_abs=max_abs,
            encoded_max_rel=max_rel,
            exact_text=exact.decode.text,
            padded_text=padded.decode.text,
            cropped_text=cropped_decode.text,
            exact_tokens=exact.decode.tokens,
            padded_tokens=padded.decode.tokens,
            cropped_tokens=cropped_decode.tokens,
        )
    except Exception as exc:
        return ProbeRow(
            real_t=int(real_t),
            row_t=-1,
            drop_extra=-1,
            exact_encoded_shape=(),
            padded_encoded_shape=(),
            exact_encoded_len=[],
            padded_encoded_len=[],
            exact_cache_len=[],
            padded_cache_len=[],
            tokens_exact=False,
            text_exact=False,
            cropped_tokens_exact=False,
            cropped_text_exact=False,
            encoded_len_exact=False,
            cache_len_exact=False,
            encoded_allclose=False,
            encoded_byte_equal=False,
            encoded_compare_frames=0,
            encoded_max_abs=float("inf"),
            encoded_max_rel=float("inf"),
            exact_text="",
            padded_text="",
            cropped_text="",
            exact_tokens=(),
            padded_tokens=(),
            cropped_tokens=(),
            error=f"{type(exc).__name__}: {exc}",
        )


def _table_bool(value: bool) -> str:
    return "yes" if value else "NO"


def _tokens_preview(tokens: tuple[int, ...], limit: int = 18) -> str:
    values = list(tokens)
    if len(values) <= limit:
        return str(values)
    return f"{values[:limit]}...(+{len(values) - limit})"


def _write_findings(
    *,
    out_path: Path,
    rows: list[ProbeRow],
    metadata: dict[str, Any],
    norm_info: dict[str, Any],
    elapsed_ms: float,
) -> None:
    gate_ok = all(row.gate_ok for row in rows)
    no_crop_tokens_ok = all(row.tokens_exact and row.text_exact for row in rows)
    cropped_tokens_ok = all(row.cropped_tokens_exact and row.cropped_text_exact for row in rows)
    if no_crop_tokens_ok:
        crop_answer = "Passing the real encoded_len is sufficient; no explicit encoded slice is needed."
    elif cropped_tokens_ok:
        crop_answer = "The no-crop decode diverged, but cropped decode matched; an explicit encoded slice is required."
    else:
        crop_answer = "Cropping does not rescue every divergence; keep exact-T buckets."

    blockers: list[str] = []
    for row in rows:
        if row.gate_ok:
            continue
        failed = []
        if row.error:
            failed.append(row.error)
        if not (row.tokens_exact and row.text_exact):
            failed.append("tokens/text")
        if not row.encoded_len_exact:
            failed.append("encoded_len")
        if not row.cache_len_exact:
            failed.append("cache_last_channel_len")
        if not row.encoded_allclose:
            failed.append(f"encoded_allclose max_abs={row.encoded_max_abs:.6g}")
        blockers.append(f"T={row.real_t}: {', '.join(failed)}")

    lines: list[str] = []
    lines.append("# Padded-T Finalize Probe Findings")
    lines.append("")
    lines.append(f"- Verdict: **{'GO' if gate_ok else 'NO-GO'}**")
    lines.append(f"- Scope: eager exact-T vs eager zero-padded-to-{metadata['t_max']} with real processed_signal_length")
    lines.append(f"- Model: `{metadata['model']}`")
    lines.append(f"- Device: `{metadata['device']}`")
    lines.append(f"- NeMo: `{metadata['nemo_version']}` from `{metadata['nemo_path']}`")
    lines.append(f"- Geometry: shift_frames={metadata['shift_frames']}, pre_encode_cache_size={metadata['pre_encode_cache_size']}, drop_extra={metadata['drop_extra']}, normal_chunks={metadata['normal_chunks']}")
    lines.append(f"- Gate: tokens/text exact + encoded_len exact + cache_last_channel_len exact + encoded real region allclose(atol={metadata['atol']}, rtol={RTOL})")
    lines.append(f"- Runtime: {elapsed_ms:.1f} ms")
    lines.append("")
    lines.append("## Encoded-Length Crop Answer")
    lines.append("")
    lines.append(crop_answer)
    lines.append("")
    lines.append("## Norm / Mode Caveat")
    lines.append("")
    lines.append(
        f"- model_eval={norm_info['model_eval']}, encoder_eval={norm_info['encoder_eval']}, "
        f"ConformerConvolution.norm_type={norm_info['conformer_conv_norm_types']}"
    )
    if norm_info["batch_norm_eval_match"]:
        lines.append("- This satisfies the requested batch-norm eval assertion; instance/group norm is absent.")
    else:
        lines.append(
            "- Caveat: the local model is not batch_norm; it reports the norm type above. "
            "The probe continued because instance/group norm is absent and layer_norm is per-frame over channels, "
            "so padded timesteps are not included in sequence-wide normalization statistics."
        )
    lines.append("")
    lines.append("## Per-T Results")
    lines.append("")
    lines.append(
        "| T | tokens/text exact | encoded_len exact | cache_len exact | encoded allclose | max_abs | encoded_len | cache_len | exact encoded | padded encoded | crop decode exact |"
    )
    lines.append(
        "|---:|---|---|---|---|---:|---|---|---|---|---|"
    )
    for row in rows:
        if row.error:
            lines.append(
                f"| {row.real_t} | NO | NO | NO | NO | inf | - | - | - | - | error: `{row.error}` |"
            )
            continue
        lines.append(
            f"| {row.real_t} | {_table_bool(row.tokens_exact and row.text_exact)} | "
            f"{_table_bool(row.encoded_len_exact)} | {_table_bool(row.cache_len_exact)} | "
            f"{_table_bool(row.encoded_allclose)} | {row.encoded_max_abs:.6g} | "
            f"`{row.exact_encoded_len}`/`{row.padded_encoded_len}` | "
            f"`{row.exact_cache_len}`/`{row.padded_cache_len}` | "
            f"`{row.exact_encoded_shape}` | `{row.padded_encoded_shape}` | "
            f"{_table_bool(row.cropped_tokens_exact and row.cropped_text_exact)} |"
        )
    lines.append("")
    lines.append("## Divergences")
    lines.append("")
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
        lines.append("")
        for row in rows:
            if row.tokens_exact and row.text_exact and not row.error:
                continue
            lines.append(
                f"- T={row.real_t} tokens exact={row.tokens_exact} text exact={row.text_exact}; "
                f"exact_tokens=`{_tokens_preview(row.exact_tokens)}`, "
                f"padded_tokens=`{_tokens_preview(row.padded_tokens)}`, "
                f"exact_text={row.exact_text!r}, padded_text={row.padded_text!r}"
            )
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    if gate_ok:
        lines.append("Proceed to Step 2b: implement the single B=1 padded T_max finalize bucket behind a default-off flag and a fail-closed startup/CI canary.")
    else:
        lines.append("Do not implement the padded T_max bucket. Keep per-T buckets and address K=4 memory by trimming the T range or accepting the K=3 fallback.")
    lines.append("")

    out_path.write_text("\n".join(lines))


def _tokens_preview_any(tokens: tuple[int, ...] | list[int], limit: int = 18) -> str:
    values = list(tokens)
    if len(values) <= limit:
        return str(values)
    return f"{values[:limit]}...(+{len(values) - limit})"


def _find_first_final_index(events: list[ContinuationEvent]) -> int:
    for index, event in enumerate(events):
        if event.kind == "final":
            return index
    return -1


def _write_continuation_findings(
    *,
    out_path: Path,
    comparison: ContinuationComparison,
    metadata: dict[str, Any],
    norm_info: dict[str, Any],
    elapsed_ms: float,
) -> None:
    exact = comparison.exact
    padded = comparison.padded
    first_exact = exact.finalizes[0] if exact.finalizes else None
    first_padded = padded.finalizes[0] if padded.finalizes else None
    exact_first_final_index = _find_first_final_index(exact.events)
    post_events = (
        exact.events[exact_first_final_index + 1 :]
        if exact_first_final_index >= 0
        else []
    )
    post_event_names = [event.event for event in post_events]
    fork_cache_diverged = (
        first_exact is not None
        and first_padded is not None
        and first_exact.call is not None
        and first_padded.call is not None
        and first_exact.call.fork_cache_len != first_padded.call.fork_cache_len
    )

    blockers: list[str] = []
    if not comparison.post_finalize_exact:
        blockers.append(comparison.first_divergence or "post-finalize event stream diverged")
    if comparison.session_adopted_fork_cache:
        blockers.append("session cache_last_channel_len adopted the finalize fork output")
    if not (comparison.session_retained_own_cache or comparison.session_reset_after_finalize):
        blockers.append("session cache behavior after finalize was neither retained nor reset")

    lines: list[str] = []
    lines.append("# Padded-T Finalize Probe Findings")
    lines.append("")
    lines.append(f"- Continuation verdict: **{'GO' if comparison.go else 'NO-GO'}**")
    lines.append("- Scope: realistic continuous VAD-stop/debounce finalize, then continued steady chunks")
    lines.append(f"- Finalize variants: exact-T vs zero-padded-to-{metadata['t_max']} with real processed_signal_length")
    lines.append(f"- Model: `{metadata['model']}`")
    lines.append(f"- Device: `{metadata['device']}`")
    lines.append(f"- NeMo: `{metadata['nemo_version']}` from `{metadata['nemo_path']}`")
    lines.append(
        f"- Geometry: shift_frames={metadata['shift_frames']}, "
        f"pre_encode_cache_size={metadata['pre_encode_cache_size']}, "
        f"drop_extra={metadata['drop_extra']}, pre_chunks={metadata['pre_chunks']}, "
        f"post_chunks={metadata['post_chunks']}"
    )
    lines.append(f"- Runtime: {elapsed_ms:.1f} ms")
    lines.append("")
    lines.append("## Continuation Verdict")
    lines.append("")
    lines.append(
        f"- First finalize tokens/text exact: **{_table_bool(comparison.first_final_exact)}**"
    )
    lines.append(
        f"- Post-finalize continuation tokens/text exact: **{_table_bool(comparison.post_finalize_exact)}**"
    )
    lines.append(
        f"- Full captured stream tokens/text exact: **{_table_bool(comparison.full_stream_exact)}**"
    )
    lines.append(
        f"- Session retained own steady cache after first finalize: **{_table_bool(comparison.session_retained_own_cache)}**"
    )
    lines.append(
        f"- Session cold-reset after first finalize: **{_table_bool(comparison.session_reset_after_finalize)}**"
    )
    lines.append(
        f"- Session adopted fork cache_last_channel_len: **{_table_bool(comparison.session_adopted_fork_cache)}**"
    )
    lines.append(
        f"- Fork cache_last_channel_len diverged as expected: **{_table_bool(fork_cache_diverged)}**"
    )
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append(
        "The finalize path exercised here is the scheduler batched debounce path: "
        "`vad_stop` arms a pending finalize, the probe injects the matching "
        "`debounce_expired` event, `_scheduler_drain_once()` batches it through "
        "`_scheduler_process_finalize_event_batch()`, and the server resumes via "
        "`_continuous_finish_speculative_finalize_locked()`."
    )
    lines.append("")
    lines.append(
        "| run | final# | real_T | fed_T | session cache before | fork cache out | session cache after | after state | retained own | adopted fork |"
    )
    lines.append("|---|---:|---:|---:|---|---|---|---|---|---|")
    for run in (exact, padded):
        for observation in run.finalizes:
            call = observation.call
            lines.append(
                f"| {run.mode} | {observation.ordinal} | "
                f"{call.real_t if call is not None else '-'} | "
                f"{call.fed_t if call is not None else '-'} | "
                f"`{observation.session_cache_before}` | "
                f"`{call.fork_cache_len if call is not None else []}` | "
                f"`{observation.session_cache_after}` | "
                f"{observation.continuous_state_after} | "
                f"{_table_bool(observation.session_cache_retained)} | "
                f"{_table_bool(observation.session_adopted_fork_cache_len)} |"
            )
    lines.append("")
    if post_event_names:
        lines.append(
            f"- Post-first-final events compared: `{post_event_names}`"
        )
    else:
        lines.append("- Post-first-final events compared: none captured.")
    if comparison.first_divergence:
        lines.append(f"- First divergence: {comparison.first_divergence}")
    else:
        lines.append("- First divergence: none.")
    lines.append("")
    lines.append("## Norm / Mode Caveat")
    lines.append("")
    lines.append(
        f"- model_eval={norm_info['model_eval']}, encoder_eval={norm_info['encoder_eval']}, "
        f"ConformerConvolution.norm_type={norm_info['conformer_conv_norm_types']}"
    )
    if norm_info["batch_norm_eval_match"]:
        lines.append("- This satisfies the requested batch-norm eval assertion; instance/group norm is absent.")
    else:
        lines.append(
            "- Caveat: the local model is not batch_norm; it reports the norm type above. "
            "The probe continued because instance/group norm is absent and layer_norm is per-frame over channels."
        )
    lines.append("")
    lines.append("## First Probe Context")
    lines.append("")
    lines.append(
        "The initial exact-T vs padded-T finalize probe was byte-exact on "
        "tokens/text/encoded_len/real encoder frames for T=42..60. Its only "
        "divergence was `cache_last_channel_len` on the disposable finalize fork "
        "(`[46]` or `[47]` exact vs `[48]` padded for T<58)."
    )
    lines.append("")
    lines.append("## Blockers")
    lines.append("")
    if blockers:
        for blocker in blockers:
            lines.append(f"- {blocker}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    if comparison.go:
        lines.append(
            "Proceed to Step 2b: the fork-only cache length divergence is dead for "
            "the continued session, so a single B=1 padded T_max finalize bucket is safe "
            "to implement behind the planned default-off, fail-closed guard."
        )
    else:
        lines.append(
            "Do not implement the padded T_max bucket. Keep per-T buckets until the "
            "continuation divergence or fork-cache adoption is understood."
        )
    lines.append("")
    lines.append("## Captured Events")
    lines.append("")
    lines.append("| run | event | kind | cache_len | tokens | text |")
    lines.append("|---|---|---|---|---|---|")
    for run in (exact, padded):
        for event in run.events:
            lines.append(
                f"| {run.mode} | {event.event} | {event.kind} | "
                f"`{event.cache_last_channel_len}` | "
                f"`{_tokens_preview_any(event.tokens)}` | {event.text!r} |"
            )
    lines.append("")

    out_path.write_text("\n".join(lines))


def _jsonable_row(row: ProbeRow) -> dict[str, Any]:
    out = dataclasses.asdict(row)
    out["gate_ok"] = row.gate_ok
    return out


def _jsonable_continuation(comparison: ContinuationComparison) -> dict[str, Any]:
    out = dataclasses.asdict(comparison)
    out["verdict"] = "GO" if comparison.go else "NO-GO"
    return out


async def _run_continuation(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    _configure_probe_env()
    harness = _load_harness()
    server = harness.build_server(
        model=args.model,
        lanes=1,
        batch_max_size=1,
        right_context=args.right_context,
    )
    norm_info = _assert_norm_eval(server)

    import nemo

    max_tail_samples = (
        int(server.preprocess_new_audio_samples)
        + (int(args.pre_chunks) + int(args.post_chunks) + 2)
        * int(server.shift_frames)
        * int(server.hop_samples)
    )
    clips = harness.select_audio_clips(
        server,
        audio_dir=args.audio_dir,
        session_count=1,
        normal_chunks=int(args.pre_chunks) + int(args.post_chunks) + 2,
        final_tail_samples=max_tail_samples,
        allow_reuse=True,
        session_prefix="cont",
    )
    clip = clips[0]

    print(
        "padded_t_probe: continuation exact-T run "
        f"pre_chunks={args.pre_chunks} post_chunks={args.post_chunks}",
        flush=True,
    )
    exact = await _run_continuation_sequence(
        server=server,
        harness=harness,
        clip=clip,
        mode="exact",
        pre_chunks=args.pre_chunks,
        post_chunks=args.post_chunks,
        t_max=args.t_max,
        second_finalize=not args.no_second_finalize,
        seed=args.seed,
    )
    print(
        "padded_t_probe: continuation padded-T run "
        f"T_max={args.t_max}",
        flush=True,
    )
    padded = await _run_continuation_sequence(
        server=server,
        harness=harness,
        clip=clip,
        mode="padded",
        pre_chunks=args.pre_chunks,
        post_chunks=args.post_chunks,
        t_max=args.t_max,
        second_finalize=not args.no_second_finalize,
        seed=args.seed,
    )
    comparison = _compare_continuation_runs(exact, padded)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metadata = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda unavailable",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nemo_version": str(getattr(nemo, "__version__", "unknown")),
        "nemo_path": str(Path(nemo.__file__).resolve()),
        "t_max": int(args.t_max),
        "shift_frames": int(server.shift_frames),
        "pre_encode_cache_size": int(server.pre_encode_cache_size),
        "drop_extra": int(server.drop_extra),
        "pre_chunks": int(args.pre_chunks),
        "post_chunks": int(args.post_chunks),
        "clip_path": str(clip.path),
        "seed": int(args.seed),
    }
    _write_continuation_findings(
        out_path=args.out,
        comparison=comparison,
        metadata=metadata,
        norm_info=norm_info,
        elapsed_ms=elapsed_ms,
    )
    print(f"padded_t_probe: wrote {args.out}", flush=True)
    print(
        json.dumps(
            _jsonable_continuation(comparison),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if comparison.go else 2


async def _run(args: argparse.Namespace) -> int:
    if args.continuation_only:
        return await _run_continuation(args)

    start = time.perf_counter()
    _configure_probe_env()
    harness = _load_harness()
    server = harness.build_server(
        model=args.model,
        lanes=1,
        batch_max_size=1,
        right_context=args.right_context,
    )
    norm_info = _assert_norm_eval(server)

    import nemo

    t_values = list(range(int(args.t_min), int(args.t_max) + 1))
    t_base = int(server.pre_encode_cache_size) + int(server.final_padding_frames) + 1
    if args.t_min < t_base:
        raise RuntimeError(
            f"cannot synthesize T={args.t_min} with current geometry base T={t_base}"
        )
    max_tail_samples = max(0, int(args.t_max) - t_base) * int(server.hop_samples)
    runtime = await _prepare_parent_runtime(
        server=server,
        harness=harness,
        normal_chunks=args.normal_chunks,
        max_tail_samples=max_tail_samples,
        audio_dir=args.audio_dir,
    )
    tail_start = int(runtime.audio_cursor)
    base_pending = (
        runtime.session.pending_audio.copy()
        if runtime.session.pending_audio is not None
        else np.array([], dtype=np.float32)
    )

    rows: list[ProbeRow] = []
    for real_t in t_values:
        print(f"padded_t_probe: T={real_t} exact eager vs padded eager", flush=True)
        row = _make_finalize_row_for_t(
            server=server,
            runtime=runtime,
            real_t=real_t,
            t_base=t_base,
            tail_start=tail_start,
            base_pending=base_pending,
        )
        rows.append(
            _probe_one_t(
                server=server,
                harness=harness,
                row=row,
                real_t=real_t,
                t_max=args.t_max,
                atol=args.atol,
            )
        )
        latest = rows[-1]
        print(
            "padded_t_probe: "
            f"T={real_t} gate={latest.gate_ok} "
            f"tokens_text={latest.tokens_exact and latest.text_exact} "
            f"encoded_len={latest.encoded_len_exact} "
            f"cache_len={latest.cache_len_exact} "
            f"allclose={latest.encoded_allclose} "
            f"max_abs={latest.encoded_max_abs:.6g}",
            flush=True,
        )

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    metadata = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cuda unavailable",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nemo_version": str(getattr(nemo, "__version__", "unknown")),
        "nemo_path": str(Path(nemo.__file__).resolve()),
        "t_min": int(args.t_min),
        "t_max": int(args.t_max),
        "atol": float(args.atol),
        "shift_frames": int(server.shift_frames),
        "pre_encode_cache_size": int(server.pre_encode_cache_size),
        "drop_extra": int(server.drop_extra),
        "normal_chunks": int(args.normal_chunks),
        "clip_path": str(runtime.clip.path),
        "tail_start": tail_start,
    }
    _write_findings(
        out_path=args.out,
        rows=rows,
        metadata=metadata,
        norm_info=norm_info,
        elapsed_ms=elapsed_ms,
    )
    print(f"padded_t_probe: wrote {args.out}", flush=True)
    print(
        json.dumps(
            {
                "verdict": "GO" if all(row.gate_ok for row in rows) else "NO-GO",
                "encoded_len_suffices": all(row.tokens_exact and row.text_exact for row in rows),
                "rows": [_jsonable_row(row) for row in rows],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if all(row.gate_ok for row in rows) else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio-dir", type=Path, default=harness_default_audio_dir())
    parser.add_argument("--normal-chunks", type=int, default=DEFAULT_NORMAL_CHUNKS)
    parser.add_argument("--t-min", type=int, default=DEFAULT_T_MIN)
    parser.add_argument("--t-max", type=int, default=DEFAULT_T_MAX)
    parser.add_argument("--right-context", type=int, default=1)
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--continuation-only",
        action="store_true",
        help="run only the continuous finalize-and-continue padded-T probe",
    )
    parser.add_argument("--pre-chunks", type=int, default=DEFAULT_NORMAL_CHUNKS)
    parser.add_argument("--post-chunks", type=int, default=8)
    parser.add_argument(
        "--no-second-finalize",
        action="store_true",
        help="skip the optional second debounce finalize after the continuation chunks",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def harness_default_audio_dir() -> Path:
    return REPO / "proj-2026-05-20-modal-cost" / "loadgen_audio"


def main() -> int:
    return asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
