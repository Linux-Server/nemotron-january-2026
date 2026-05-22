"""Bucketed manual CUDA-graph capture for the steady streaming encoder.

This module only graphs ``model.encoder.cache_aware_stream_step``. RNNT/CTC
decode stays eager and remains the caller's responsibility.

The bucket manager captures one graph per exact batch size ``B`` in ``1..K``.
It does not pad smaller batches into larger buckets. Static buffer layouts match
``batch_primitives.stack_caches``:

* ``processed_signal``: ``[B, F, T]``
* ``cache_last_channel``: ``[layers, B, cache_T, d_model]``
* ``cache_last_time``: ``[layers, B, d_model, time_T]``
* ``cache_last_channel_len``: ``[B]``

``replay`` returns tensors owned by the static CUDA-graph output pool. Callers
that need to retain results after another replay must clone/detach them before
the next replay.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import time
from typing import Any, Optional, Sequence

import torch


EncoderGraphOutputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class EncoderGraphInputs:
    """Input tensors for one exact steady encoder bucket."""

    processed_signal: torch.Tensor
    processed_signal_length: torch.Tensor
    cache_last_channel: torch.Tensor
    cache_last_time: torch.Tensor
    cache_last_channel_len: torch.Tensor


def encoder_stream_step_restoring_drop_extra(model: Any, **kwargs: Any) -> tuple[Any, ...]:
    """Call ``encoder.cache_aware_stream_step`` and restore NeMo's global drop setting."""

    streaming_cfg = model.encoder.streaming_cfg
    original_drop_extra = streaming_cfg.drop_extra_pre_encoded
    try:
        return model.encoder.cache_aware_stream_step(**kwargs)
    finally:
        streaming_cfg.drop_extra_pre_encoded = original_drop_extra


def _streaming_cfg_int(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return int(value[1])
    return int(value)


def _coerce_inputs(inputs: EncoderGraphInputs | Sequence[torch.Tensor]) -> Optional[EncoderGraphInputs]:
    if isinstance(inputs, EncoderGraphInputs):
        return inputs
    try:
        values = tuple(inputs)
    except Exception:
        return None
    if len(values) != 5 or not all(torch.is_tensor(item) for item in values):
        return None
    return EncoderGraphInputs(
        processed_signal=values[0],
        processed_signal_length=values[1],
        cache_last_channel=values[2],
        cache_last_time=values[3],
        cache_last_channel_len=values[4],
    )


class _CudaGraphEncoderBucket:
    """One captured graph and its static input/output buffers for a single B."""

    def __init__(
        self,
        model: Any,
        *,
        batch_size: int,
        steady_T: int,
        drop_extra: int,
        warmup_iters: int,
    ) -> None:
        self.model = model
        self.batch_size = int(batch_size)
        self.steady_T = int(steady_T)
        self.drop_extra = int(drop_extra)
        self.replays = 0
        self.capture_ms = 0.0

        cache = model.encoder.get_initial_cache_state(batch_size=self.batch_size)
        self.device = cache[0].device
        if self.device.type != "cuda":
            raise RuntimeError(f"CUDA graph capture requires CUDA tensors, got {self.device}")

        feat = int(model.cfg.preprocessor.features)
        dtype = cache[0].dtype
        self.static_processed = torch.empty(
            (self.batch_size, feat, self.steady_T),
            device=self.device,
            dtype=dtype,
        )
        self.static_processed.zero_()
        self.static_len = torch.full(
            (self.batch_size,),
            self.steady_T,
            device=self.device,
            dtype=torch.long,
        )
        self.static_clc = torch.empty_like(cache[0])
        self.static_clt = torch.empty_like(cache[1])
        self.static_clcl = torch.empty_like(cache[2])
        self.static_clc.zero_()
        self.static_clt.zero_()
        self.static_clcl.zero_()

        self.graph = torch.cuda.CUDAGraph()
        self.static_outputs: Optional[EncoderGraphOutputs] = None

        start = time.perf_counter()
        self._capture(warmup_iters=warmup_iters)
        torch.cuda.synchronize(device=self.device)
        self.capture_ms = (time.perf_counter() - start) * 1000.0

    def _call_encoder(self) -> EncoderGraphOutputs:
        outputs = encoder_stream_step_restoring_drop_extra(
            self.model,
            processed_signal=self.static_processed,
            processed_signal_length=self.static_len,
            cache_last_channel=self.static_clc,
            cache_last_time=self.static_clt,
            cache_last_channel_len=self.static_clcl,
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop_extra,
        )
        if len(outputs) != 5 or not all(torch.is_tensor(item) for item in outputs):
            raise RuntimeError("encoder graph capture expected exactly 5 tensor outputs")
        return outputs  # type: ignore[return-value]

    def _capture(self, *, warmup_iters: int) -> None:
        with torch.cuda.device(self.device):
            side_stream = torch.cuda.Stream(device=self.device)
            side_stream.wait_stream(torch.cuda.current_stream(device=self.device))
            with torch.inference_mode(), torch.cuda.stream(side_stream):
                for _ in range(int(warmup_iters)):
                    self._call_encoder()

            torch.cuda.current_stream(device=self.device).wait_stream(side_stream)
            torch.cuda.synchronize(device=self.device)

            with torch.inference_mode(), torch.cuda.graph(self.graph):
                self.static_outputs = self._call_encoder()

        if self.static_outputs is None:
            raise RuntimeError("encoder graph capture did not produce static outputs")

    def input_shapes_match(self, inputs: EncoderGraphInputs) -> bool:
        return (
            inputs.processed_signal.shape == self.static_processed.shape
            and inputs.processed_signal_length.shape == self.static_len.shape
            and inputs.cache_last_channel.shape == self.static_clc.shape
            and inputs.cache_last_time.shape == self.static_clt.shape
            and inputs.cache_last_channel_len.shape == self.static_clcl.shape
        )

    def replay(self, inputs: EncoderGraphInputs) -> EncoderGraphOutputs:
        self.static_processed.copy_(inputs.processed_signal)
        self.static_len.copy_(inputs.processed_signal_length)
        self.static_clc.copy_(inputs.cache_last_channel)
        self.static_clt.copy_(inputs.cache_last_time)
        self.static_clcl.copy_(inputs.cache_last_channel_len)
        self.graph.replay()
        self.replays += 1
        assert self.static_outputs is not None
        return self.static_outputs


class BucketedCudaGraphEncoder:
    """Fail-closed bucket manager for steady streaming encoder CUDA graphs.

    Use ``BucketedCudaGraphEncoder.warmup(model, K)`` to capture exact buckets
    ``B=1..K``. ``captured(B)`` reports whether a bucket can be replayed.
    ``replay(B, inputs)`` returns static graph outputs for captured, shape-
    matching buckets and returns ``None`` as the use-eager signal for uncaptured
    buckets, ``B > K``, non-positive/invalid B, or mismatched inputs.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_batch_size: int,
        warmup_iters: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.model = model
        self.max_batch_size = max(0, int(max_batch_size))
        self.warmup_iters = max(0, int(warmup_iters))
        self.logger = logger or logging.getLogger(__name__)

        streaming_cfg = model.encoder.streaming_cfg
        self.shift_frames = _streaming_cfg_int(streaming_cfg.shift_size)
        self.pre_encode_cache_size = _streaming_cfg_int(streaming_cfg.pre_encode_cache_size)
        self.steady_T = int(self.pre_encode_cache_size + self.shift_frames)
        self.drop_extra = int(streaming_cfg.drop_extra_pre_encoded)

        self._buckets: dict[int, _CudaGraphEncoderBucket] = {}
        self._capture_errors: dict[int, str] = {}
        self._replay_errors: dict[int, str] = {}
        self._warmed = False

    @classmethod
    def warmup(
        cls,
        model: Any,
        K: int,
        *,
        warmup_iters: int = 5,
        logger: Optional[logging.Logger] = None,
    ) -> "BucketedCudaGraphEncoder":
        """Capture buckets ``B=1..K`` and return the fail-closed manager."""

        manager = cls(
            model,
            max_batch_size=int(K),
            warmup_iters=warmup_iters,
            logger=logger,
        )
        manager.capture()
        return manager

    def capture(self) -> None:
        """Capture all requested buckets, marking failures uncaptured."""

        if self._warmed:
            return
        self._warmed = True

        if not torch.cuda.is_available():
            for batch_size in range(1, self.max_batch_size + 1):
                self._capture_errors[batch_size] = "torch.cuda is unavailable"
            return

        for batch_size in range(1, self.max_batch_size + 1):
            try:
                self._buckets[batch_size] = _CudaGraphEncoderBucket(
                    self.model,
                    batch_size=batch_size,
                    steady_T=self.steady_T,
                    drop_extra=self.drop_extra,
                    warmup_iters=self.warmup_iters,
                )
            except Exception as exc:  # CUDA arch/OOM/capture failures must not escape.
                self._buckets.pop(batch_size, None)
                self._capture_errors[batch_size] = f"{type(exc).__name__}: {exc}"
                self.logger.warning(
                    "encoder_cuda_graph_capture_failed B=%s error=%s",
                    batch_size,
                    self._capture_errors[batch_size],
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def captured(self, batch_size: Any) -> bool:
        """Return ``False`` instead of raising for out-of-range or uncaptured buckets."""

        try:
            batch_size_int = int(batch_size)
        except Exception:
            return False
        if batch_size_int < 1 or batch_size_int > self.max_batch_size:
            return False
        return batch_size_int in self._buckets

    def replay(
        self,
        batch_size: Any,
        inputs: EncoderGraphInputs | Sequence[torch.Tensor],
    ) -> Optional[EncoderGraphOutputs]:
        """Replay a captured bucket or return ``None`` to signal eager fallback.

        The returned tensors are static graph-owned buffers. Clone/detach them
        before another replay if the values must outlive the next graph call.
        """

        try:
            batch_size_int = int(batch_size)
        except Exception:
            return None
        bucket = self._buckets.get(batch_size_int)
        if bucket is None:
            return None

        coerced_inputs = _coerce_inputs(inputs)
        if coerced_inputs is None:
            self._replay_errors[batch_size_int] = "invalid input container"
            return None
        try:
            input_shapes_match = bucket.input_shapes_match(coerced_inputs)
        except Exception as exc:
            self._replay_errors[batch_size_int] = f"invalid input tensors: {type(exc).__name__}: {exc}"
            return None
        if not input_shapes_match:
            self._replay_errors[batch_size_int] = (
                "input shape mismatch: "
                f"processed={tuple(coerced_inputs.processed_signal.shape)} "
                f"expected={tuple(bucket.static_processed.shape)} "
                f"clc={tuple(coerced_inputs.cache_last_channel.shape)} "
                f"expected_clc={tuple(bucket.static_clc.shape)} "
                f"clt={tuple(coerced_inputs.cache_last_time.shape)} "
                f"expected_clt={tuple(bucket.static_clt.shape)} "
                f"clcl={tuple(coerced_inputs.cache_last_channel_len.shape)} "
                f"expected_clcl={tuple(bucket.static_clcl.shape)}"
            )
            return None

        try:
            with torch.inference_mode():
                return bucket.replay(coerced_inputs)
        except Exception as exc:  # Fail closed: future calls to this bucket use eager.
            self._buckets.pop(batch_size_int, None)
            self._replay_errors[batch_size_int] = f"{type(exc).__name__}: {exc}"
            self.logger.warning(
                "encoder_cuda_graph_replay_failed B=%s error=%s",
                batch_size_int,
                self._replay_errors[batch_size_int],
            )
            return None

    def capture_error(self, batch_size: Any) -> Optional[str]:
        try:
            return self._capture_errors.get(int(batch_size))
        except Exception:
            return None

    def replay_error(self, batch_size: Any) -> Optional[str]:
        try:
            return self._replay_errors.get(int(batch_size))
        except Exception:
            return None

    @property
    def captured_batch_sizes(self) -> tuple[int, ...]:
        return tuple(sorted(self._buckets))

    @property
    def uncaptured_batch_sizes(self) -> tuple[int, ...]:
        return tuple(
            batch_size
            for batch_size in range(1, self.max_batch_size + 1)
            if batch_size not in self._buckets
        )

    def capture_ms(self, batch_size: Any) -> Optional[float]:
        try:
            bucket = self._buckets.get(int(batch_size))
        except Exception:
            return None
        return None if bucket is None else float(bucket.capture_ms)

    def replays(self, batch_size: Any) -> int:
        try:
            bucket = self._buckets.get(int(batch_size))
        except Exception:
            return 0
        return 0 if bucket is None else int(bucket.replays)
