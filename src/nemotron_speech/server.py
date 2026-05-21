"""WebSocket ASR server for Nemotron-Speech with true incremental streaming."""

import asyncio
import argparse
import concurrent.futures
import copy
import contextlib
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from aiohttp import ClientConnectionResetError, web, WSMsgType
from loguru import logger

try:
    from nemotron_speech.batch_primitives import (
        batch_group_key,
        ready_predicate,
        scatter_cache_row,
        stack_caches,
        stack_hypotheses,
        stack_pred_out,
        stack_processed,
    )
except ImportError:  # Allows `python src/nemotron_speech/server.py`.
    from batch_primitives import (
        batch_group_key,
        ready_predicate,
        scatter_cache_row,
        stack_caches,
        stack_hypotheses,
        stack_pred_out,
        stack_processed,
    )

# Enable debug logging with DEBUG_ASR=1
DEBUG_ASR = os.environ.get("DEBUG_ASR", "0") == "1"

_DEFAULT_FINALIZE_SILENCE_MS = 150
_MAX_FINALIZE_SILENCE_MS = 10_000

STREAMING = "STREAMING"
PENDING_FINALIZE = "PENDING_FINALIZE"
FINALIZED = "FINALIZED"
_TRUE_BOUNDARY_FINALIZE_REASONS = frozenset({"close", "end"})
_EOU_PROBE_LOCK = threading.Lock()
_EOU_SNAPSHOT_LOCK = threading.Lock()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"{name} must be a float, got {value!r}") from e


def _telemetry_run_tag() -> str | None:
    for env_name in (
        "NEMOTRON_RUN_TAG",
        "NEMOTRON_TELEMETRY_RUN_TAG",
        "STT_BENCHMARK_RUN_TAG",
    ):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _safe_tag_filename(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("._-") or "tag"


def _telemetry_dir() -> Path:
    configured = os.environ.get("NEMOTRON_TELEMETRY_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[2] / "stt-benchmark" / "stt_benchmark_data" / "client_telemetry"


def _hash_audio(audio: np.ndarray) -> str:
    """Get short hash of audio array for debugging."""
    if audio is None or len(audio) == 0:
        return "empty"
    return hashlib.md5(audio.tobytes()).hexdigest()[:8]


def tensor_clone(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()


def tensor_clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def clone_tree(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Tensor-aware deepcopy for disposable ASR fork state."""
    if memo is None:
        memo = {}

    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if torch.is_tensor(obj):
        return tensor_clone(obj)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        cloned_list: list[Any] = []
        memo[oid] = cloned_list
        cloned_list.extend(clone_tree(item, memo) for item in obj)
        return cloned_list
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        cloned_tuple = tuple(clone_tree(item, memo) for item in obj)
        memo[oid] = cloned_tuple
        return cloned_tuple
    if isinstance(obj, dict):
        cloned_dict: dict[Any, Any] = {}
        memo[oid] = cloned_dict
        for key, value in obj.items():
            cloned_dict[clone_tree(key, memo)] = clone_tree(value, memo)
        return cloned_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Covers NeMo LabelLoopingStateItem decoder state under greedy_batch.
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for field in dataclasses.fields(obj):
            setattr(cloned_obj, field.name, clone_tree(getattr(obj, field.name), memo))
        return cloned_obj

    # Some NeMo helper objects may live inside optional fields. Avoid copying
    # modules or other heavy objects; copy plain object state only when present.
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for key, value in vars(obj).items():
            setattr(cloned_obj, key, clone_tree(value, memo))
        return cloned_obj

    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def clone_hypotheses_deep(previous_hypotheses: Any) -> Any:
    """Deep-copy each Hypothesis and recursively clone tensor decoder state."""
    if previous_hypotheses is None:
        return None
    return [clone_tree(hyp) for hyp in previous_hypotheses]


def snapshot_tree_cpu(obj: Any, memo: Optional[dict[int, Any]] = None) -> Any:
    """Deep-copy snapshot state with all tensors detached and moved to CPU."""
    if memo is None:
        memo = {}

    oid = id(obj)
    if oid in memo:
        return memo[oid]

    if torch.is_tensor(obj):
        return tensor_clone_cpu(obj)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj
    if isinstance(obj, list):
        cloned_list: list[Any] = []
        memo[oid] = cloned_list
        cloned_list.extend(snapshot_tree_cpu(item, memo) for item in obj)
        return cloned_list
    if isinstance(obj, tuple):
        placeholder: list[Any] = []
        memo[oid] = placeholder
        cloned_tuple = tuple(snapshot_tree_cpu(item, memo) for item in obj)
        memo[oid] = cloned_tuple
        return cloned_tuple
    if isinstance(obj, dict):
        cloned_dict: dict[Any, Any] = {}
        memo[oid] = cloned_dict
        for key, value in obj.items():
            cloned_dict[snapshot_tree_cpu(key, memo)] = snapshot_tree_cpu(value, memo)
        return cloned_dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Covers NeMo LabelLoopingStateItem decoder state under greedy_batch.
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for field in dataclasses.fields(obj):
            setattr(cloned_obj, field.name, snapshot_tree_cpu(getattr(obj, field.name), memo))
        return cloned_obj

    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
        cloned_obj = copy.copy(obj)
        memo[oid] = cloned_obj
        for key, value in vars(obj).items():
            setattr(cloned_obj, key, snapshot_tree_cpu(value, memo))
        return cloned_obj

    try:
        return copy.deepcopy(obj, memo)
    except Exception:
        return obj


def _tensor_assert_hash(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.md5()
    digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    digest.update(str(cpu_tensor.dtype).encode("utf-8"))
    digest.update(cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:12]


def _array_assert_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.md5()
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()[:12]


def _assert_tree_equal(label: str, before: Any, after: Any) -> None:
    if torch.is_tensor(before) or torch.is_tensor(after):
        if not (torch.is_tensor(before) and torch.is_tensor(after)):
            raise AssertionError(f"{label}: tensor/non-tensor mismatch")
        same_meta = (
            tuple(before.shape) == tuple(after.shape)
            and before.dtype == after.dtype
            and before.device == after.device
        )
        same_bytes = same_meta and torch.equal(before, after)
        if not same_bytes:
            raise AssertionError(
                f"{label}: tensor changed "
                f"before(shape={tuple(before.shape)}, dtype={before.dtype}, "
                f"device={before.device}, hash={_tensor_assert_hash(before)}) "
                f"after(shape={tuple(after.shape)}, dtype={after.dtype}, "
                f"device={after.device}, hash={_tensor_assert_hash(after)})"
            )
        return

    if isinstance(before, np.ndarray) or isinstance(after, np.ndarray):
        if not (isinstance(before, np.ndarray) and isinstance(after, np.ndarray)):
            raise AssertionError(f"{label}: ndarray/non-ndarray mismatch")
        if (
            before.shape != after.shape
            or before.dtype != after.dtype
            or not np.array_equal(before, after)
        ):
            raise AssertionError(
                f"{label}: ndarray changed "
                f"before(shape={before.shape}, dtype={before.dtype}, "
                f"hash={_array_assert_hash(before)}) "
                f"after(shape={after.shape}, dtype={after.dtype}, "
                f"hash={_array_assert_hash(after)})"
            )
        return

    if before is None or after is None or isinstance(before, (str, bytes, int, float, bool)):
        if before != after:
            raise AssertionError(f"{label}: value changed from {before!r} to {after!r}")
        return

    if isinstance(before, list) or isinstance(after, list):
        if not (isinstance(before, list) and isinstance(after, list)):
            raise AssertionError(f"{label}: list/non-list mismatch")
        if len(before) != len(after):
            raise AssertionError(f"{label}: list length changed {len(before)} -> {len(after)}")
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            _assert_tree_equal(f"{label}[{index}]", before_item, after_item)
        return

    if isinstance(before, tuple) or isinstance(after, tuple):
        if not (isinstance(before, tuple) and isinstance(after, tuple)):
            raise AssertionError(f"{label}: tuple/non-tuple mismatch")
        if len(before) != len(after):
            raise AssertionError(f"{label}: tuple length changed {len(before)} -> {len(after)}")
        for index, (before_item, after_item) in enumerate(zip(before, after)):
            _assert_tree_equal(f"{label}[{index}]", before_item, after_item)
        return

    if isinstance(before, dict) or isinstance(after, dict):
        if not (isinstance(before, dict) and isinstance(after, dict)):
            raise AssertionError(f"{label}: dict/non-dict mismatch")
        if before.keys() != after.keys():
            raise AssertionError(f"{label}: dict keys changed")
        for key in before:
            _assert_tree_equal(f"{label}[{key!r}]", before[key], after[key])
        return

    before_is_dataclass = dataclasses.is_dataclass(before) and not isinstance(before, type)
    after_is_dataclass = dataclasses.is_dataclass(after) and not isinstance(after, type)
    if before_is_dataclass or after_is_dataclass:
        if not (before_is_dataclass and after_is_dataclass and before.__class__ is after.__class__):
            raise AssertionError(f"{label}: dataclass type changed")
        # Covers NeMo LabelLoopingStateItem decoder state under greedy_batch.
        for field in dataclasses.fields(before):
            _assert_tree_equal(
                f"{label}.{field.name}",
                getattr(before, field.name),
                getattr(after, field.name),
            )
        return

    before_is_nemo_obj = hasattr(before, "__dict__") and before.__class__.__module__.startswith("nemo.")
    after_is_nemo_obj = hasattr(after, "__dict__") and after.__class__.__module__.startswith("nemo.")
    if before_is_nemo_obj or after_is_nemo_obj:
        if not (before_is_nemo_obj and after_is_nemo_obj and before.__class__ is after.__class__):
            raise AssertionError(f"{label}: NeMo object type changed")
        if vars(before).keys() != vars(after).keys():
            raise AssertionError(f"{label}: NeMo object fields changed")
        for key in vars(before):
            _assert_tree_equal(f"{label}.{key}", getattr(before, key), getattr(after, key))
        return

    try:
        equal = before == after
    except Exception:
        equal = repr(before) == repr(after)
    if not equal:
        raise AssertionError(f"{label}: object changed from {before!r} to {after!r}")


def _continuous_append_only_delta(final_text: str, emitted_text: str) -> str:
    """Return the collector-safe word suffix to append for a cumulative final."""
    final_tokens = final_text.split()
    emitted_tokens = emitted_text.split()

    common = 0
    for emitted_token, final_token in zip(emitted_tokens, final_tokens):
        if emitted_token != final_token:
            break
        common += 1

    if common == len(emitted_tokens):
        delta_tokens = final_tokens[common:]
    elif len(final_tokens) <= len(emitted_tokens):
        delta_tokens = []
    else:
        # A non-prefix correction cannot be sent to the append-only collector.
        # Freeze the already-emitted token count and only append the new tail.
        delta_tokens = final_tokens[len(emitted_tokens):]
        max_overlap = min(len(emitted_tokens), len(delta_tokens))
        for overlap in range(max_overlap, 0, -1):
            if emitted_tokens[-overlap:] == delta_tokens[:overlap]:
                delta_tokens = delta_tokens[overlap:]
                break

    return " ".join(delta_tokens)


# Default model - HuggingFace model name (auto-downloads) or local .nemo path
DEFAULT_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"

# Right context options for att_context_size=[70, X]
RIGHT_CONTEXT_OPTIONS = {
    0: "~80ms ultra-low latency",
    1: "~160ms low latency (recommended)",
    6: "~560ms balanced",
    13: "~1.12s highest accuracy",
}

PROMPTED_FALLBACK_ATT_CONTEXT_SIZES = [[56, 0], [56, 3], [56, 6], [56, 13]]
PROMPTED_DEFAULT_RIGHT_CONTEXT = 3
PROMPTED_DEFAULT_TARGET_LANG = "auto"
LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>")
PARTIAL_LANG_TAG_RE = re.compile(r"\s*<[a-z]{0,2}(?:-[A-Z]{0,2})?$")


@dataclass
class ASRSession:
    """Per-connection session state with caches for true incremental streaming."""

    id: str
    websocket: Any
    target_lang: Optional[str] = None

    # Legacy/debug audio buffer name. Step 6b keeps this bounded to pending
    # audio only; the preprocessor must never see a growing full-stream buffer.
    accumulated_audio: Optional[np.ndarray] = None

    # Raw audio not yet advanced past `emitted_frames * hop_samples`.
    pending_audio: Optional[np.ndarray] = None

    # Total real audio samples received in this session, excluding synthetic
    # finalization padding.
    total_audio_samples: int = 0

    # Synthetic warm-up samples already emitted to the model. This is only a
    # timeline cursor offset; it is not real audio and is never accumulated.
    synthetic_prefix_samples: int = 0

    # STFT boundary state: trailing window_size - hop samples before pending.
    raw_audio_ring: Optional[np.ndarray] = None

    # Cache-aware chunker state: trailing pre_encode_cache_size mel frames.
    mel_frame_ring: Optional[torch.Tensor] = None

    # Number of mel frames already emitted to encoder
    emitted_frames: int = 0

    # Encoder cache state
    cache_last_channel: Optional[torch.Tensor] = None
    cache_last_time: Optional[torch.Tensor] = None
    cache_last_channel_len: Optional[torch.Tensor] = None

    # Decoder state
    previous_hypotheses: Any = None
    pred_out_stream: Any = None

    # Current transcription (model's cumulative output)
    current_text: str = ""

    # Last text emitted to client on hard reset (for server-side deduplication)
    # We only send the delta (new portion) to avoid downstream duplication
    last_emitted_text: str = ""

    # Continuous-context mode state. These fields are only used when
    # NEMOTRON_CONTINUOUS=1; the default hard-reset path ignores them.
    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    continuous_state: str = STREAMING
    committed_text: str = ""
    continuous_emitted_text: str = ""
    continuous_event_queue: Optional[asyncio.Queue] = None
    continuous_worker_task: Optional[asyncio.Task] = None
    continuous_debounce_task: Optional[asyncio.Task] = None
    continuous_stop_seq: int = 0
    continuous_reset_seen: bool = False
    continuous_post_stop_audio: bytearray = field(default_factory=bytearray)
    continuous_vad_stop_ts: Optional[float] = None
    continuous_debounce_expiry_ts: Optional[float] = None
    eou_probe_chunk_index: int = 0
    eou_snapshot_audio: bytearray = field(default_factory=bytearray)
    scheduler_generation: int = 0
    scheduler_inflight_generation: Optional[int] = None
    scheduler_ready_since: Optional[float] = None
    scheduler_closed: bool = False
    scheduler_last_audio_monotonic: Optional[float] = None

    # Audio overlap buffer for mid-utterance reset continuity
    # This preserves the last N ms of audio to provide encoder left-context
    # when a new segment starts after a reset
    overlap_buffer: Optional[np.ndarray] = None


@dataclass
class SchedulerBatchRow:
    session: ASRSession
    generation: int
    chunk_mel: torch.Tensor
    valid_new_mel: torch.Tensor
    drop_extra: int
    eou_probe_snapshot: Optional[dict[str, Any]]


class ASRServer:
    """WebSocket server for streaming ASR with true incremental processing."""

    def __init__(
        self,
        model: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        right_context: Optional[int] = None,
    ):
        self.model_name_or_path = model
        self.host = host
        self.port = port
        self.right_context = right_context
        self.model = None
        self.prompted_model = False
        self.prompt_dictionary: dict[str, Any] = {}
        self.target_lang = os.environ.get("NEMOTRON_TARGET_LANG", "en-US").strip() or "en-US"
        self.sample_rate = 16000
        # ASR benchmark server REQUIRES CUDA — fail fast, never silently fall
        # back to CPU (a CPU/wrong-device run yields invalid benchmark numbers
        # while looking 'fine'). Baseline hardening 2026-05-18 (dual review).
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for the Nemotron ASR benchmark server "
                "(torch.cuda.is_available() is False); refusing to run on CPU."
            )

        # Inference lock
        self.inference_lock = asyncio.Lock()

        self.continuous_context = os.environ.get("NEMOTRON_CONTINUOUS", "") == "1"
        self.scheduler_b1_requested = os.environ.get("NEMOTRON_SCHEDULER_B1", "") == "1"
        self.scheduler_enabled = self.continuous_context and self.scheduler_b1_requested
        self.batch_requested = os.environ.get("NEMOTRON_BATCH_SCHED", "") == "1"
        self.batch_enabled = self.batch_requested
        self.batch_fallback_reason: Optional[str] = None
        self.requested_decoder_strategy = (
            os.environ.get("NEMOTRON_DECODING", "greedy").strip().lower() or "greedy"
        )
        self.fork_assert_enabled = os.environ.get("NEMOTRON_FORK_ASSERT", "") == "1"
        self.eou_probe_enabled = os.environ.get("NEMOTRON_EOU_PROBE", "") == "1"
        self._scheduler_batch_fallback_counts: dict[str, int] = {}
        if self.batch_enabled and not self.scheduler_b1_requested:
            raise ValueError("NEMOTRON_BATCH_SCHED=1 requires NEMOTRON_SCHEDULER_B1=1")
        if self.batch_enabled and not self.scheduler_enabled:
            raise ValueError(
                "NEMOTRON_BATCH_SCHED=1 requires the continuous scheduler "
                "(NEMOTRON_CONTINUOUS=1 and NEMOTRON_SCHEDULER_B1=1)"
            )
        if self.batch_requested and self.requested_decoder_strategy not in ("", "greedy"):
            raise ValueError(
                "NEMOTRON_BATCH_SCHED=1 requires NEMOTRON_DECODING=greedy "
                f"(got {self.requested_decoder_strategy!r})"
            )
        if self.batch_enabled and self.eou_probe_enabled:
            self._disable_batching("eou_probe_preserve_alignments_unprobed")
        if self.batch_enabled:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logger.info(
                "batch_sched_tf32_disabled "
                "cuda.matmul.allow_tf32=False cudnn.allow_tf32=False"
            )
        self.scheduler_queue_maxsize = _env_int("NEMOTRON_SCHEDULER_QUEUE_MAXSIZE", 256)
        if self.scheduler_queue_maxsize <= 0:
            raise ValueError("NEMOTRON_SCHEDULER_QUEUE_MAXSIZE must be > 0")
        # Defaults from the max-parallelism sweep (proj-2026-05-21-0410/max-parallelism-sweep.md):
        # MAX_SIZE=32 + MAX_WAIT=8ms raised the local realtime knee 40->56 with N=1 latency unchanged
        # (~17ms p95). The Step-8 device-aware startup cap clamps MAX_SIZE down on smaller GPUs, so 32 is
        # safe as a ceiling. Only active when NEMOTRON_BATCH_SCHED=1 (batching is off by default).
        self.batch_max_wait_ms = _env_int("NEMOTRON_BATCH_MAX_WAIT_MS", 8)
        self.batch_max_size = _env_int("NEMOTRON_BATCH_MAX_SIZE", 32)
        if self.batch_max_wait_ms < 0:
            raise ValueError("NEMOTRON_BATCH_MAX_WAIT_MS must be >= 0")
        if self.batch_max_size <= 0:
            raise ValueError("NEMOTRON_BATCH_MAX_SIZE must be > 0")
        if self.batch_requested and not self.batch_enabled:
            self.batch_max_size = 1
        self.batch_memory_headroom_fraction = _env_float(
            "NEMOTRON_BATCH_MEMORY_HEADROOM_FRACTION",
            0.80,
        )
        if not (0.0 < self.batch_memory_headroom_fraction <= 1.0):
            raise ValueError(
                "NEMOTRON_BATCH_MEMORY_HEADROOM_FRACTION must be > 0 and <= 1"
            )
        self.batch_memory_row_floor_bytes = (
            _env_int("NEMOTRON_BATCH_MEMORY_ROW_FLOOR_MB", 512) * 1024 * 1024
        )
        if self.batch_memory_row_floor_bytes <= 0:
            raise ValueError("NEMOTRON_BATCH_MEMORY_ROW_FLOOR_MB must be > 0")
        self.batch_memory_telemetry_every = _env_int(
            "NEMOTRON_BATCH_MEMORY_TELEMETRY_EVERY",
            1,
        )
        if self.batch_memory_telemetry_every <= 0:
            raise ValueError("NEMOTRON_BATCH_MEMORY_TELEMETRY_EVERY must be > 0")
        self.scheduler_task: Optional[asyncio.Task] = None
        self._scheduler_wakeup: Optional[asyncio.Event] = None
        self._scheduler_ready: set[str] = set()
        self._scheduler_batch_first_ready: dict[tuple, float] = {}
        self._scheduler_batch_size_hist: dict[int, int] = {}
        self._scheduler_batch_queue_wait_ms_total = 0.0
        self._scheduler_batch_queue_wait_ms_max = 0.0
        self._scheduler_batch_queue_wait_count = 0
        self._scheduler_batches = 0
        self._scheduler_chunks = 0
        self._scheduler_lane_wait_ms_total = 0.0
        self._scheduler_lane_wait_ms_max = 0.0
        self.decoder_strategy = (
            "greedy_batch" if self.batch_enabled else self.requested_decoder_strategy
        )
        # Per-chunk profiling (additive, flag-gated): time preprocess vs conformer_stream_step
        # to locate the single-thread bottleneck. Adds cuda.synchronize() so it perturbs
        # timing slightly — only enabled under NEMOTRON_PROFILE_CHUNK=1.
        self.profile_chunk = os.environ.get("NEMOTRON_PROFILE_CHUNK", "") == "1"
        self._prof_n = 0
        self._prof_pre_ms = 0.0
        self._prof_step_ms = 0.0
        self._prof_enc_ms = 0.0
        self.encoder_compile_requested = os.environ.get("NEMOTRON_ENCODER_COMPILE", "") == "1"
        self.encoder_compile_enabled = False
        self._encoder_compile_startup_logged = False
        self._encoder_compiled_cache_aware_stream_step: Any = None
        self._encoder_compile_warmup_done = False
        self._encoder_compile_warmed_buckets: set[tuple[int, int]] = set()
        self._encoder_compile_calls = 0
        self._encoder_compile_recapture_events = 0
        self._encoder_compile_last_graph_count = 0
        self._encoder_compile_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._encoder_compile_thread_id: Optional[int] = None
        self.eou_probe_tag = _telemetry_run_tag() or "eou_probe"
        self.eou_probe_path = (
            _telemetry_dir() / f"{_safe_tag_filename(self.eou_probe_tag)}.eou_probe.jsonl"
            if self.eou_probe_enabled
            else None
        )
        snapshot_dir = os.environ.get("NEMOTRON_EOU_SNAPSHOT_DIR", "")
        self.eou_snapshot_dir = (
            Path(snapshot_dir).expanduser()
            if self.eou_probe_enabled and snapshot_dir
            else None
        )
        self.eou_snapshot_every = 1
        if self.eou_snapshot_dir is not None:
            self.eou_snapshot_every = _env_int("NEMOTRON_EOU_SNAPSHOT_EVERY", 1)
            if self.eou_snapshot_every < 1:
                raise ValueError("NEMOTRON_EOU_SNAPSHOT_EVERY must be >= 1")
        self.session_warmup_ms = _env_int("NEMOTRON_WARMUP_MS", 0)
        if self.session_warmup_ms < 0:
            raise ValueError("NEMOTRON_WARMUP_MS must be >= 0")
        self.finalize_silence_ms = _DEFAULT_FINALIZE_SILENCE_MS
        if self.continuous_context:
            self.finalize_silence_ms = _env_int(
                "NEMOTRON_FINALIZE_SILENCE_MS", _DEFAULT_FINALIZE_SILENCE_MS
            )
            if not (0 <= self.finalize_silence_ms < _MAX_FINALIZE_SILENCE_MS):
                raise ValueError(
                    "NEMOTRON_FINALIZE_SILENCE_MS must be >= 0 and < "
                    f"{_MAX_FINALIZE_SILENCE_MS}"
                )
        self.finalize_silence_seconds = self.finalize_silence_ms / 1000

        # Active sessions
        self.sessions: dict[str, ASRSession] = {}

        # Model loaded flag for health check
        self.model_loaded = False

        # Streaming parameters (calculated from model config)
        self.shift_frames = None
        self.pre_encode_cache_size = None
        self.hop_samples = None
        self.window_size_samples = None
        self.raw_audio_ring_samples = None
        self.preprocess_align_pad_samples = None
        self.preprocess_new_audio_samples = None
        self.constant_preprocess_frames = None
        self.constant_preprocess_samples = None
        self.stream_preprocess_valid_samples = None
        self.first_preprocess_mel_frame = None
        self.att_context_size = None
        self.drop_extra = None
        self.final_padding_frames = None

        # Audio overlap for mid-utterance reset continuity (calculated in load_model)
        self.overlap_samples = None

    @staticmethod
    def _to_plain(value: Any) -> Any:
        if value.__class__.__module__.startswith("omegaconf."):
            try:
                from omegaconf import OmegaConf

                return OmegaConf.to_container(value, resolve=True)
            except Exception:
                pass
        if hasattr(value, "to_container"):
            try:
                return value.to_container(resolve=True)
            except TypeError:
                return value.to_container()
        if isinstance(value, dict):
            return {key: ASRServer._to_plain(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [ASRServer._to_plain(item) for item in value]
        return value

    @staticmethod
    def _cfg_get(container: Any, *keys: str) -> Any:
        current = container
        for key in keys:
            if current is None:
                return None
            try:
                if hasattr(current, "get"):
                    current = current.get(key)
                else:
                    current = getattr(current, key)
            except Exception:
                return None
        return current

    def _record_batch_fallback(self, reason: str) -> None:
        self._scheduler_batch_fallback_counts[reason] = (
            self._scheduler_batch_fallback_counts.get(reason, 0) + 1
        )

    def _disable_batching(self, reason: str) -> None:
        if not self.batch_enabled and self.batch_fallback_reason == reason:
            return
        self.batch_enabled = False
        self.batch_fallback_reason = reason
        self.decoder_strategy = getattr(self, "requested_decoder_strategy", "greedy") or "greedy"
        if hasattr(self, "batch_max_size"):
            self.batch_max_size = 1
        self._record_batch_fallback(reason)
        logger.warning(
            "batch_sched_disabled "
            f"reason={reason} requested={self.batch_requested} fallback_to_B=1"
        )

    def _batch_model_rnnt_pure_status(self) -> tuple[bool, str]:
        model = self.model
        if model is None:
            return False, "model_not_loaded"
        if not hasattr(model, "conformer_stream_step"):
            return False, "model_missing_conformer_stream_step"
        if not hasattr(model, "joint"):
            return False, "model_missing_rnnt_joint"
        if not hasattr(model, "decoder"):
            return False, "model_missing_rnnt_decoder"
        if hasattr(model, "ctc_decoder"):
            return False, "hybrid_ctc_decoder_present"
        if hasattr(model, "ctc_loss"):
            return False, "ctc_loss_present"

        class_label = f"{model.__class__.__module__}.{model.__class__.__name__}".lower()
        if "ctc" in class_label and "rnnt" not in class_label:
            return False, "ctc_model_class"
        if "hybrid" in class_label and "ctc" in class_label:
            return False, "hybrid_ctc_model_class"

        cfg = self._to_plain(getattr(model, "cfg", None))
        if isinstance(cfg, dict):
            for key in ("ctc", "ctc_decoder", "ctc_loss"):
                if key in cfg and cfg.get(key) not in (None, False):
                    return False, f"cfg_{key}_present"
            decoder_cfg = cfg.get("decoder")
            if isinstance(decoder_cfg, dict):
                decoder_name = str(decoder_cfg.get("_target_", "")).lower()
                if "ctc" in decoder_name and "rnnt" not in decoder_name:
                    return False, "cfg_decoder_ctc_target"
            decoding_cfg = cfg.get("decoding")
            if isinstance(decoding_cfg, dict):
                strategy = str(decoding_cfg.get("strategy", "")).lower()
                if "ctc" in strategy and "rnnt" not in strategy:
                    return False, "cfg_decoding_ctc_strategy"
        return True, "rnnt_pure"

    def _assert_batch_decoder_blackwell_safe(self) -> None:
        if not self.batch_enabled:
            return
        decoding = getattr(self.model, "decoding", None)
        uses_cuda_graph = False
        for owner in (
            decoding,
            getattr(decoding, "decoding", None),
            getattr(decoding, "greedy", None),
        ):
            value = getattr(owner, "use_cuda_graph_decoder", None)
            if value is not None:
                uses_cuda_graph = uses_cuda_graph or bool(value)
        cfg = self._to_plain(getattr(self.model, "cfg", None))
        cfg_value = self._cfg_get(cfg, "decoding", "greedy", "use_cuda_graph_decoder")
        if cfg_value is not None:
            uses_cuda_graph = uses_cuda_graph or bool(cfg_value)
        if uses_cuda_graph:
            self._disable_batching("cuda_graph_decoder_enabled")
        else:
            logger.info(
                "batch_sched_decoder_assert "
                "use_cuda_graph_decoder=False status=ok"
            )

    @staticmethod
    def _tensor_storage_nbytes(tensor: torch.Tensor, seen: set[tuple]) -> int:
        if tensor.device.type != "cuda":
            return 0
        try:
            storage = tensor.untyped_storage()
            key = (tensor.device.type, tensor.device.index, int(storage.data_ptr()))
            if key in seen:
                return 0
            seen.add(key)
            return int(storage.nbytes())
        except Exception:
            key = (tensor.device.type, tensor.device.index, int(tensor.data_ptr()))
            if key in seen:
                return 0
            seen.add(key)
            return int(tensor.nelement() * tensor.element_size())

    @classmethod
    def _tensor_tree_storage_nbytes(
        cls,
        obj: Any,
        *,
        seen_tensors: Optional[set[int]] = None,
        seen_storages: Optional[set[tuple]] = None,
        seen_objects: Optional[set[int]] = None,
    ) -> int:
        if seen_tensors is None:
            seen_tensors = set()
        if seen_storages is None:
            seen_storages = set()
        if seen_objects is None:
            seen_objects = set()

        if torch.is_tensor(obj):
            oid = id(obj)
            if oid in seen_tensors:
                return 0
            seen_tensors.add(oid)
            return cls._tensor_storage_nbytes(obj, seen_storages)
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return 0

        oid = id(obj)
        if oid in seen_objects:
            return 0
        seen_objects.add(oid)

        if isinstance(obj, dict):
            return sum(
                cls._tensor_tree_storage_nbytes(
                    value,
                    seen_tensors=seen_tensors,
                    seen_storages=seen_storages,
                    seen_objects=seen_objects,
                )
                for value in obj.values()
            )
        if isinstance(obj, (list, tuple, set)):
            return sum(
                cls._tensor_tree_storage_nbytes(
                    value,
                    seen_tensors=seen_tensors,
                    seen_storages=seen_storages,
                    seen_objects=seen_objects,
                )
                for value in obj
            )
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return sum(
                cls._tensor_tree_storage_nbytes(
                    getattr(obj, field.name),
                    seen_tensors=seen_tensors,
                    seen_storages=seen_storages,
                    seen_objects=seen_objects,
                )
                for field in dataclasses.fields(obj)
            )
        if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("nemo."):
            return sum(
                cls._tensor_tree_storage_nbytes(
                    value,
                    seen_tensors=seen_tensors,
                    seen_storages=seen_storages,
                    seen_objects=seen_objects,
                )
                for value in vars(obj).values()
            )
        return 0

    def _session_cache_storage_bytes(self, session: ASRSession) -> int:
        seen_tensors: set[int] = set()
        seen_storages: set[tuple] = set()
        seen_objects: set[int] = set()
        total = 0
        for value in (
            session.cache_last_channel,
            session.cache_last_time,
            session.cache_last_channel_len,
            session.mel_frame_ring,
            session.previous_hypotheses,
            session.pred_out_stream,
        ):
            total += self._tensor_tree_storage_nbytes(
                value,
                seen_tensors=seen_tensors,
                seen_storages=seen_storages,
                seen_objects=seen_objects,
            )
        return total

    def _retained_session_cache_bytes(self) -> int:
        return sum(
            self._session_cache_storage_bytes(session)
            for session in list(self.sessions.values())
            if not session.scheduler_closed
        )

    def _cuda_memory_snapshot(self) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {
                "active_bytes": 0,
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "max_reserved_bytes": 0,
                "retained_session_cache_bytes": self._retained_session_cache_bytes(),
            }
        stats = torch.cuda.memory_stats()
        active = int(stats.get("active_bytes.all.current", torch.cuda.memory_allocated()))
        return {
            "active_bytes": active,
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "retained_session_cache_bytes": self._retained_session_cache_bytes(),
        }

    def _log_retained_cache_telemetry(self, reason: str) -> None:
        if not self.batch_requested:
            return
        mem = self._cuda_memory_snapshot()
        logger.info(
            "scheduler_batch_retained_memory "
            f"reason={reason} "
            f"active_sessions={len(self.sessions)} "
            f"retained_session_cache_bytes={mem['retained_session_cache_bytes']} "
            f"cuda_active_bytes={mem['active_bytes']} "
            f"cuda_allocated_bytes={mem['allocated_bytes']} "
            f"cuda_reserved_bytes={mem['reserved_bytes']} "
            f"cuda_max_reserved_bytes={mem['max_reserved_bytes']}"
        )

    def _estimate_batch_extra_row_bytes(self) -> int:
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        cache_bytes = sum(
            self._tensor_tree_storage_nbytes(tensor)
            for tensor in cache
        )
        mel_bytes = 0
        if self.constant_preprocess_frames is not None:
            mel_bytes = int(self.constant_preprocess_frames) * 128 * 4
        return max(
            int(self.batch_memory_row_floor_bytes),
            int(cache_bytes * 4),
            int(mel_bytes * 16),
        )

    def _configure_batch_memory_cap(self) -> None:
        if not self.batch_enabled:
            if self.batch_requested:
                logger.info(
                    "batch_memory_startup_cap "
                    f"requested_max={self.batch_max_size} effective_max=1 "
                    f"reason={self.batch_fallback_reason or 'batch_disabled'}"
                )
            return
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        reserved_bytes = int(torch.cuda.memory_reserved())
        allocated_bytes = int(torch.cuda.memory_allocated())
        max_reserved_bytes = int(torch.cuda.max_memory_reserved())
        headroom_bytes = int(total_bytes * self.batch_memory_headroom_fraction)
        extra_row_bytes = self._estimate_batch_extra_row_bytes()
        if reserved_bytes >= headroom_bytes:
            device_cap = 1
        else:
            device_cap = 1 + ((headroom_bytes - reserved_bytes) // extra_row_bytes)
            device_cap = max(1, int(device_cap))
        requested_max = int(self.batch_max_size)
        effective_max = max(1, min(requested_max, device_cap))
        if effective_max < requested_max:
            self.batch_max_size = effective_max
            self._record_batch_fallback("memory_cap_clamped")
        logger.info(
            "batch_memory_startup_cap "
            f"requested_max={requested_max} effective_max={self.batch_max_size} "
            f"device_cap={device_cap} "
            f"headroom_fraction={self.batch_memory_headroom_fraction:.2f} "
            f"total_bytes={int(total_bytes)} free_bytes={int(free_bytes)} "
            f"reserved_bytes={reserved_bytes} allocated_bytes={allocated_bytes} "
            f"max_reserved_bytes={max_reserved_bytes} "
            f"estimated_extra_row_bytes={extra_row_bytes} "
            f"fallback_counts={dict(sorted(self._scheduler_batch_fallback_counts.items()))}"
        )

    @staticmethod
    def _is_int_like(value: Any) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return False
        return isinstance(value, (int, np.integer))

    @classmethod
    def _normalize_att_context_sizes(cls, value: Any) -> list[list[int]]:
        value = cls._to_plain(value)
        if value is None:
            return []
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or not value:
            return []
        if all(cls._is_int_like(item) for item in value):
            return [[int(item) for item in value]]

        sizes: list[list[int]] = []
        for item in value:
            normalized = cls._normalize_att_context_sizes(item)
            sizes.extend(normalized)
        return sizes

    def _supported_att_context_sizes(self) -> list[list[int]]:
        cfg_contexts = self._normalize_att_context_sizes(
            self._cfg_get(getattr(self.model, "cfg", None), "encoder", "att_context_size")
        )
        if cfg_contexts:
            return cfg_contexts

        encoder_contexts = self._normalize_att_context_sizes(
            getattr(getattr(self.model, "encoder", None), "att_context_size_all", None)
        )
        if encoder_contexts:
            return encoder_contexts

        if self.prompted_model:
            return [context.copy() for context in PROMPTED_FALLBACK_ATT_CONTEXT_SIZES]
        return [[70, right_context] for right_context in RIGHT_CONTEXT_OPTIONS]

    def _select_att_context_size(self) -> list[int]:
        if not self.prompted_model:
            if self.right_context is None:
                self.right_context = 1
            if self.right_context not in RIGHT_CONTEXT_OPTIONS:
                raise ValueError(
                    "English model right context must be one of "
                    f"{sorted(RIGHT_CONTEXT_OPTIONS)}, got {self.right_context}"
                )
            return [70, self.right_context]

        supported_contexts = self._supported_att_context_sizes()
        requested_right_context = (
            PROMPTED_DEFAULT_RIGHT_CONTEXT
            if self.right_context is None
            else self.right_context
        )
        for context in supported_contexts:
            if len(context) >= 2 and context[-1] == requested_right_context:
                self.right_context = requested_right_context
                return context.copy()

        supported_right_contexts = sorted(
            {context[-1] for context in supported_contexts if len(context) >= 2}
        )
        raise ValueError(
            "Prompted model right context must be one of "
            f"{supported_right_contexts}, got {requested_right_context}"
        )

    def _ensure_session_target_lang(self, session: ASRSession) -> str:
        if session.target_lang is None:
            session.target_lang = self.target_lang
        return session.target_lang

    def _apply_inference_prompt(self, session: ASRSession) -> None:
        if self.prompted_model:
            self.model.set_inference_prompt(self._ensure_session_target_lang(session))

    def _read_prompt_dictionary(self) -> dict[str, Any]:
        prompt_dictionary_paths = (
            ("model_defaults", "prompt_dictionary"),
            ("train_ds", "prompt_dictionary"),
            ("validation_ds", "prompt_dictionary"),
            ("test_ds", "prompt_dictionary"),
        )
        for cfg in (getattr(self.model, "cfg", None), getattr(self.model, "_cfg", None)):
            for path in prompt_dictionary_paths:
                prompt_dict = self._to_plain(self._cfg_get(cfg, *path))
                if isinstance(prompt_dict, dict):
                    return {str(key): value for key, value in prompt_dict.items()}
        return {}

    @staticmethod
    def _format_supported_languages(prompt_dictionary: dict[str, Any]) -> str:
        return ", ".join(sorted(prompt_dictionary))

    def _model_identity_aliases(self) -> set[str]:
        aliases: set[str] = set()
        configured = os.environ.get("NEMOTRON_MODEL_NAME", "").strip()
        if configured:
            aliases.add(configured)

        if self.model_name_or_path:
            aliases.add(self.model_name_or_path)
            model_path = Path(self.model_name_or_path)
            aliases.add(model_path.name)
            aliases.add(model_path.stem)

        if self.prompted_model:
            aliases.update({"multilingual", "ml"})
        else:
            aliases.update({"english", "en"})

        return {alias for alias in aliases if alias}

    def _validate_model_query_param(self, requested_model: Optional[str]) -> None:
        if not requested_model:
            return

        aliases = self._model_identity_aliases()
        if not aliases:
            logger.info(
                "Client requested model={} but no server model identity is configured; accepting",
                requested_model,
            )
            return

        normalized_aliases = {alias.lower() for alias in aliases}
        if requested_model.lower() not in normalized_aliases:
            raise ValueError(
                "model mismatch: requested "
                f"{requested_model}; server accepts: {', '.join(sorted(aliases))}"
            )

    def _validate_session_target_lang(self, requested_language: Optional[str]) -> str:
        if requested_language:
            if not self.prompted_model:
                raise ValueError("this model does not accept a language argument")

            if requested_language not in self.prompt_dictionary:
                raise ValueError(
                    f"unsupported language {requested_language}; supported: "
                    f"{self._format_supported_languages(self.prompt_dictionary)}"
                )
            return requested_language

        if self.prompted_model:
            if PROMPTED_DEFAULT_TARGET_LANG not in self.prompt_dictionary:
                raise ValueError(
                    f"unsupported language {PROMPTED_DEFAULT_TARGET_LANG}; supported: "
                    f"{self._format_supported_languages(self.prompt_dictionary)}"
                )
            return PROMPTED_DEFAULT_TARGET_LANG

        return self.target_lang

    def _validate_connection_query(self, query: Any) -> str:
        requested_language = (query.get("language") or "").strip()
        requested_model = (query.get("model") or "").strip()

        self._validate_model_query_param(requested_model or None)
        return self._validate_session_target_lang(requested_language or None)

    def _strip_lang_tags(self, text: Any) -> str:
        stripped = LANG_TAG_RE.sub(" ", str(text))
        # A cumulative streaming hypothesis may end with the beginning of the
        # next language tag token. Keep that fragment out of emitted state; the
        # complete tag is stripped when it appears in a later cumulative hyp.
        stripped = PARTIAL_LANG_TAG_RE.sub("", stripped)
        return re.sub(r"\s+", " ", stripped).strip()

    def _extract_hypothesis_text(self, hyp: Any) -> str:
        if hasattr(hyp, 'text'):
            text = hyp.text
        elif isinstance(hyp, str):
            text = hyp
        else:
            text = str(hyp)

        if self.prompted_model:
            text = self._strip_lang_tags(text)
        return text

    def load_model(self):
        """Load the NeMo ASR model with streaming configuration."""
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf

        # Detect if model is a local .nemo file or HuggingFace model name
        is_local_file = (
            self.model_name_or_path.endswith('.nemo') or
            os.path.exists(self.model_name_or_path)
        )

        if is_local_file:
            logger.info(f"Loading model from local file: {self.model_name_or_path}")
            self.model = nemo_asr.models.ASRModel.restore_from(
                self.model_name_or_path, map_location='cpu'
            )
        else:
            logger.info(f"Loading model from HuggingFace: {self.model_name_or_path}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                self.model_name_or_path, map_location='cpu'
            )
        self.model = self.model.cuda()
        logger.info("ASR model loaded on CUDA")
        self.prompted_model = hasattr(self.model, "set_inference_prompt")
        self.prompt_dictionary = (
            self._read_prompt_dictionary() if self.prompted_model else {}
        )
        if self.batch_enabled:
            rnnt_ok, rnnt_reason = self._batch_model_rnnt_pure_status()
            if not rnnt_ok:
                self._disable_batching(rnnt_reason)
            else:
                logger.info(
                    "batch_sched_model_assert "
                    f"rnnt_pure=True reason={rnnt_reason}"
                )

        # Configure attention context for streaming
        self.att_context_size = self._select_att_context_size()
        if self.prompted_model:
            logger.info(
                "Prompted model detected; setting "
                f"att_context_size={self.att_context_size} "
                f"(NEMOTRON_TARGET_LANG={self.target_lang!r})"
            )
        else:
            logger.info(f"Setting att_context_size=[70, {self.right_context}] ({RIGHT_CONTEXT_OPTIONS.get(self.right_context, 'custom')})")
        self.model.encoder.set_default_att_context_size(self.att_context_size)

        # Decoding strategy: greedy (default, Blackwell-safe) or beam via
        # NEMOTRON_DECODING=beam (experimental; may be slower / unsupported on
        # some GPUs).
        _decoding = self.decoder_strategy
        if _decoding == "beam":
            logger.info("Configuring BEAM (maes) decoding...")
            _decoding_cfg = OmegaConf.create({
                'strategy': 'beam',
                'beam': {
                    'beam_size': 4,
                    'search_type': 'maes',
                    'maes_num_steps': 2,
                    'return_best_hypothesis': True,
                }
            })
        elif _decoding == "greedy_batch":
            logger.info(
                "Configuring greedy_batch decoding "
                "(loop_labels=True, use_cuda_graph_decoder=False)..."
            )
            _decoding_cfg = OmegaConf.create({
                'strategy': 'greedy_batch',
                'greedy': {
                    'max_symbols': 10,
                    'loop_labels': True,
                    'use_cuda_graph_decoder': False,
                }
            })
            if self.eou_probe_enabled:
                _decoding_cfg.greedy.preserve_alignments = True
                _decoding_cfg.greedy.preserve_frame_confidence = True
                _decoding_cfg.greedy.confidence_method_cfg = {
                    'name': 'entropy',
                    'entropy_type': 'tsallis',
                    'alpha': 0.5,
                    'entropy_norm': 'exp',
                }
        else:
            logger.info("Configuring greedy decoding for Blackwell compatibility...")
            _decoding_cfg = OmegaConf.create({
                'strategy': 'greedy',
                'greedy': {
                    'max_symbols': 10,
                    'loop_labels': False,
                    'use_cuda_graph_decoder': False,
                }
            })
            if self.eou_probe_enabled:
                _decoding_cfg.greedy.preserve_alignments = True
                _decoding_cfg.greedy.preserve_frame_confidence = True
                _decoding_cfg.greedy.confidence_method_cfg = {
                    'name': 'entropy',
                    'entropy_type': 'tsallis',
                    'alpha': 0.5,
                    'entropy_norm': 'exp',
                }
        self.model.change_decoding_strategy(decoding_cfg=_decoding_cfg)
        self.model.eval()
        self._assert_batch_decoder_blackwell_safe()

        # Disable dither for deterministic preprocessing
        self.model.preprocessor.featurizer.dither = 0.0

        # Get streaming config
        scfg = self.model.encoder.streaming_cfg
        logger.info(f"Streaming config: chunk_size={scfg.chunk_size}, shift_size={scfg.shift_size}")

        # Calculate parameters
        preprocessor_cfg = self.model.cfg.preprocessor
        hop_length_sec = preprocessor_cfg.get('window_stride', 0.01)
        window_size_sec = preprocessor_cfg.get('window_size', 0.025)
        featurizer = self.model.preprocessor.featurizer
        self.hop_samples = int(getattr(featurizer, "hop_length", int(hop_length_sec * self.sample_rate)))
        self.window_size_samples = int(
            getattr(featurizer, "win_length", int(window_size_sec * self.sample_rate))
        )

        # shift_size[1] = 16 frames for 160ms chunks
        self.shift_frames = scfg.shift_size[1] if isinstance(scfg.shift_size, list) else scfg.shift_size

        # pre_encode_cache_size[1] = 9 frames
        pre_cache = scfg.pre_encode_cache_size
        self.pre_encode_cache_size = pre_cache[1] if isinstance(pre_cache, list) else pre_cache

        # drop_extra_pre_encoded for non-first chunks
        self.drop_extra = scfg.drop_extra_pre_encoded

        # Calculate silence padding for final chunk:
        # - right_context chunks for encoder lookahead
        # - 1 additional chunk for decoder finalization
        # With right_context=1, this is (1+1)*160ms = 320ms
        self.final_padding_frames = (self.right_context + 1) * self.shift_frames
        padding_ms = self.final_padding_frames * hop_length_sec * 1000

        # Constant-plan incremental preprocessor:
        # - raw ring is only STFT boundary context (window - hop)
        # - mel ring below is the cache-aware pre-encode context
        # - one hop of right-edge guard matches the current streaming gate
        self.raw_audio_ring_samples = self.window_size_samples - self.hop_samples
        self.preprocess_align_pad_samples = (
            self.hop_samples - (self.raw_audio_ring_samples % self.hop_samples)
        ) % self.hop_samples
        self.preprocess_new_audio_samples = (self.shift_frames + 1) * self.hop_samples
        self.stream_preprocess_valid_samples = (
            self.preprocess_align_pad_samples
            + self.raw_audio_ring_samples
            + self.preprocess_new_audio_samples
        )
        prefix_samples = self.preprocess_align_pad_samples + self.raw_audio_ring_samples
        if prefix_samples % self.hop_samples != 0:
            raise RuntimeError(
                "Constant preprocessor prefix must align to mel frame hops: "
                f"prefix={prefix_samples}, hop={self.hop_samples}"
            )
        self.first_preprocess_mel_frame = prefix_samples // self.hop_samples
        min_plan_frames = (
            self.first_preprocess_mel_frame
            + self.pre_encode_cache_size
            + self.shift_frames
            + self.final_padding_frames
            + 1
        )
        self.constant_preprocess_frames = 1 << (min_plan_frames - 1).bit_length()
        self.constant_preprocess_samples = (self.constant_preprocess_frames - 1) * self.hop_samples

        # Calculate audio overlap for mid-utterance reset continuity
        # Use pre_encode_cache_size frames = 90ms of left-context
        # This allows the encoder to have proper context when starting a new segment
        self.overlap_samples = self.pre_encode_cache_size * self.hop_samples
        overlap_ms = self.overlap_samples * 1000 / self.sample_rate

        shift_ms = self.shift_frames * hop_length_sec * 1000
        logger.info(f"Model loaded: {type(self.model).__name__}")
        logger.info(f"Shift size: {shift_ms:.0f}ms ({self.shift_frames} frames)")
        logger.info(f"Pre-encode cache: {self.pre_encode_cache_size} frames")
        logger.info(
            "Constant preprocessor plan: "
            f"K={self.constant_preprocess_samples} samples "
            f"({self.constant_preprocess_frames} STFT frames, min={min_plan_frames}) "
            f"(align={self.preprocess_align_pad_samples}, "
            f"raw_ring={self.raw_audio_ring_samples}, "
            f"new_audio={self.preprocess_new_audio_samples})"
        )
        logger.info(f"Final chunk padding: {padding_ms:.0f}ms ({self.final_padding_frames} frames)")
        logger.info(f"Audio overlap for resets: {overlap_ms:.0f}ms ({self.overlap_samples} samples)")

        self._configure_encoder_compile()
        logger.info(
            "startup_flags "
            f"scheduler_enabled={self.scheduler_enabled} "
            f"batch_requested={self.batch_requested} "
            f"batch_enabled={self.batch_enabled} "
            f"batch_fallback_reason={self.batch_fallback_reason or 'none'} "
            f"decoder_strategy={self.decoder_strategy} "
            f"encoder_compile_enabled={self.encoder_compile_enabled} "
            f"batch_max_size={self.batch_max_size}"
        )

        # Warmup inference to ensure model is fully loaded on GPU
        # This prevents GPU memory issues when LLM starts later
        self._warmup()
        self._configure_batch_memory_cap()

    def _configure_encoder_compile(self) -> None:
        """Configure optional B=1 static-shape encoder compilation."""
        if self._encoder_compile_startup_logged:
            return

        if not self.encoder_compile_requested:
            logger.info("encoder_compile_enabled=False requested=False")
            self._encoder_compile_startup_logged = True
            return

        if self.prompted_model:
            logger.warning(
                "encoder_compile_enabled=False requested=True "
                "reason=prompted_model_static_shapes_unvalidated"
            )
            self._encoder_compile_startup_logged = True
            return

        if not hasattr(torch, "compile"):
            raise RuntimeError("NEMOTRON_ENCODER_COMPILE=1 requires torch.compile")

        self._encoder_compiled_cache_aware_stream_step = torch.compile(
            self.model.encoder.cache_aware_stream_step,
            mode="reduce-overhead",
        )
        self._encoder_compile_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nemotron-encoder-compile",
        )
        self.encoder_compile_enabled = True
        self._encoder_compile_last_graph_count = self._encoder_compile_counter_snapshot()
        logger.info(
            "encoder_compile_enabled=True requested=True mode=reduce-overhead "
            "scope=B1_static_buckets warmup=enabled first=enabled steady=enabled "
            "final_uncompiled=True"
        )
        self._encoder_compile_startup_logged = True

    def _session_warmup_frames(self) -> Optional[int]:
        if self.session_warmup_ms <= 0:
            return None
        target_samples = int(round(self.sample_rate * self.session_warmup_ms / 1000))
        return max(
            self.pre_encode_cache_size,
            int(round(target_samples / self.hop_samples)),
        )

    @staticmethod
    def _encoder_compile_counter_snapshot() -> int:
        try:
            from torch._dynamo.utils import counters
        except Exception:
            return 0
        try:
            return int(counters.get("stats", {}).get("unique_graphs", 0))
        except Exception:
            return 0

    def _encoder_compile_bucket_for_call(self, kwargs: dict[str, Any]) -> Optional[tuple[int, int]]:
        if not self.encoder_compile_enabled:
            return None
        if kwargs.get("keep_all_outputs", True):
            return None
        if kwargs.get("bypass_pre_encode", False):
            return None

        processed_signal = kwargs.get("processed_signal")
        if not torch.is_tensor(processed_signal) or processed_signal.ndim != 3:
            return None
        if int(processed_signal.shape[0]) != 1:
            return None

        try:
            chunk_frames = int(processed_signal.shape[-1])
            drop_extra = int(kwargs.get("drop_extra_pre_encoded"))
        except Exception:
            return None

        static_buckets: set[tuple[int, int]] = {
            (int(self.shift_frames), 0),
            (int(self.pre_encode_cache_size + self.shift_frames), int(self.drop_extra)),
        }
        warmup_frames = self._session_warmup_frames()
        if warmup_frames is not None:
            static_buckets.add((int(warmup_frames), 0))

        bucket = (chunk_frames, drop_extra)
        if bucket in static_buckets:
            return bucket
        return None

    @contextlib.contextmanager
    def _compiled_encoder_cache_step_installed(self):
        encoder = self.model.encoder
        attr_name = "cache_aware_stream_step"
        had_instance_attr = attr_name in vars(encoder)
        original_instance_attr = vars(encoder).get(attr_name)
        object.__setattr__(
            encoder,
            attr_name,
            self._encoder_compiled_cache_aware_stream_step,
        )
        try:
            yield
        finally:
            if had_instance_attr:
                object.__setattr__(encoder, attr_name, original_instance_attr)
            else:
                object.__delattr__(encoder, attr_name)

    def _conformer_stream_step(self, **kwargs):
        """Call NeMo's stream step with drop-extra restoration and optional compiled encoder."""
        streaming_cfg = self.model.encoder.streaming_cfg
        original_drop_extra = streaming_cfg.drop_extra_pre_encoded
        bucket = self._encoder_compile_bucket_for_call(kwargs)
        saw_unwarmed_bucket = False

        try:
            if bucket is None:
                return self.model.conformer_stream_step(**kwargs)

            self._encoder_compile_calls += 1
            if (
                self._encoder_compile_thread_id is not None
                and threading.get_ident() != self._encoder_compile_thread_id
            ):
                logger.warning(
                    "encoder_compile_call_on_different_thread "
                    f"warm_thread={self._encoder_compile_thread_id} "
                    f"call_thread={threading.get_ident()}"
                )
            saw_unwarmed_bucket = (
                self._encoder_compile_warmup_done
                and bucket not in self._encoder_compile_warmed_buckets
            )
            if saw_unwarmed_bucket:
                logger.warning(
                    "encoder_compile_unwarmed_static_bucket_after_warmup "
                    f"bucket_T={bucket[0]} drop_extra={bucket[1]}"
                )

            mark_step_begin = getattr(
                getattr(torch, "compiler", None),
                "cudagraph_mark_step_begin",
                None,
            )
            if mark_step_begin is not None:
                mark_step_begin()
            with self._compiled_encoder_cache_step_installed():
                result = self.model.conformer_stream_step(**kwargs)
            result_list = list(result)
            for result_index in (2, 3, 4):
                if result_index < len(result_list) and torch.is_tensor(result_list[result_index]):
                    result_list[result_index] = result_list[result_index].detach().clone()
            return tuple(result_list)
        finally:
            streaming_cfg.drop_extra_pre_encoded = original_drop_extra

            if bucket is not None:
                graph_count = self._encoder_compile_counter_snapshot()
                if self._encoder_compile_warmup_done:
                    if graph_count > self._encoder_compile_last_graph_count:
                        delta = graph_count - self._encoder_compile_last_graph_count
                        self._encoder_compile_recapture_events += delta
                        logger.warning(
                            "encoder_compile_recapture_after_warmup "
                            f"delta={delta} total={self._encoder_compile_recapture_events} "
                            f"bucket_T={bucket[0]} drop_extra={bucket[1]}"
                        )
                    elif saw_unwarmed_bucket:
                        self._encoder_compile_recapture_events += 1
                        logger.warning(
                            "encoder_compile_recapture_after_warmup "
                            f"delta=unknown total={self._encoder_compile_recapture_events} "
                            f"bucket_T={bucket[0]} drop_extra={bucket[1]}"
                        )
                self._encoder_compile_last_graph_count = max(
                    self._encoder_compile_last_graph_count,
                    graph_count,
                )
                if self._encoder_compile_calls % 50 == 0:
                    logger.info(
                        "encoder_compile_status "
                        f"compiled_calls={self._encoder_compile_calls} "
                        f"recapture_counter={self._encoder_compile_recapture_events} "
                        f"warmed_buckets={sorted(self._encoder_compile_warmed_buckets)}"
                    )

    async def _run_inference_call(self, fn, *args):
        executor = self._encoder_compile_executor if self.encoder_compile_enabled else None
        return await asyncio.get_event_loop().run_in_executor(executor, fn, *args)

    def _warm_encoder_compile_static_buckets(
        self,
        _base_mel: torch.Tensor,
        on_compile_executor: bool = False,
    ) -> None:
        if not self.encoder_compile_enabled:
            return
        if self._encoder_compile_executor is not None and not on_compile_executor:
            self._encoder_compile_executor.submit(
                self._warm_encoder_compile_static_buckets,
                _base_mel,
                True,
            ).result()
            return

        self._encoder_compile_thread_id = threading.get_ident()

        warmup_frames = self._session_warmup_frames()
        first_bucket = (int(self.shift_frames), 0)
        steady_bucket = (
            int(self.pre_encode_cache_size + self.shift_frames),
            int(self.drop_extra),
        )
        warm_repeats = 2
        profile_chunk = self.profile_chunk
        self.profile_chunk = False
        try:
            for repeat in range(warm_repeats):
                first_session = ASRSession(
                    id=f"encoder_compile_first_{repeat}",
                    websocket=None,
                    target_lang=self.target_lang,
                )
                self._init_session_without_synthetic_warmup(first_session)
                self._queue_silent_compile_chunk(first_session)
                if self._process_chunk(first_session) is None:
                    raise RuntimeError("encoder compile first-bucket warmup failed")

                steady_session = ASRSession(
                    id=f"encoder_compile_steady_{repeat}",
                    websocket=None,
                    target_lang=self.target_lang,
                )
                if warmup_frames is not None:
                    self._init_session(steady_session)
                else:
                    self._init_session_without_synthetic_warmup(steady_session)
                    self._queue_silent_compile_chunk(steady_session)
                    if self._process_chunk(steady_session) is None:
                        raise RuntimeError("encoder compile pre-steady warmup failed")
                self._queue_silent_compile_chunk(steady_session)
                if self._process_chunk(steady_session) is None:
                    raise RuntimeError("encoder compile steady-bucket warmup failed")
        finally:
            self.profile_chunk = profile_chunk

        warmed_labels = [
            f"first:T={first_bucket[0]}:drop={first_bucket[1]}:repeats={warm_repeats}",
            f"steady:T={steady_bucket[0]}:drop={steady_bucket[1]}:repeats={warm_repeats}",
        ]
        self._encoder_compile_warmed_buckets.add(first_bucket)
        self._encoder_compile_warmed_buckets.add(steady_bucket)
        if warmup_frames is not None:
            warmup_bucket = (int(warmup_frames), 0)
            self._encoder_compile_warmed_buckets.add(warmup_bucket)
            warmed_labels.insert(
                0,
                f"warmup:T={warmup_bucket[0]}:drop={warmup_bucket[1]}:repeats={warm_repeats}",
            )

        self._encoder_compile_warmup_done = True
        self._encoder_compile_last_graph_count = self._encoder_compile_counter_snapshot()
        logger.info(
            "encoder_compile_warmup_complete "
            f"buckets={warmed_labels} "
            f"unique_graphs={self._encoder_compile_last_graph_count} "
            f"recapture_counter={self._encoder_compile_recapture_events}"
        )

    def _init_session_without_synthetic_warmup(self, session: ASRSession) -> None:
        self._ensure_session_target_lang(session)
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        session.cache_last_channel = cache[0]
        session.cache_last_time = cache[1]
        session.cache_last_channel_len = cache[2]
        session.pending_audio = np.array([], dtype=np.float32)
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples = 0
        session.raw_audio_ring = np.zeros(self.raw_audio_ring_samples, dtype=np.float32)
        session.mel_frame_ring = None
        session.emitted_frames = 0
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.current_text = ""
        session.eou_probe_chunk_index = 0
        session.synthetic_prefix_samples = 0

    def _queue_silent_compile_chunk(self, session: ASRSession) -> None:
        session.pending_audio = np.zeros(self.preprocess_new_audio_samples, dtype=np.float32)
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples += len(session.pending_audio)

    def _warmup(self):
        """Run warmup inference using streaming API to claim GPU memory.

        IMPORTANT: We use the streaming API (conformer_stream_step) for warmup,
        NOT the batch API (model.transcribe). The batch API corrupts internal
        model state and causes subsequent streaming inference to become
        non-deterministic. See docs/asr-determinism-investigation.md.
        """
        import time

        logger.info("Running warmup inference (streaming API) to claim GPU memory...")
        start = time.perf_counter()

        # Keep warmup on the same fixed preprocessor plan as live streaming.
        warmup_audio = np.zeros(self.constant_preprocess_samples, dtype=np.float32)

        # Run streaming inference to force all CUDA kernels to compile
        with torch.inference_mode():
            mel, mel_len = self._preprocess_fixed_audio(warmup_audio, len(warmup_audio))

            # Get initial cache
            cache = self.model.encoder.get_initial_cache_state(batch_size=1)
            warmup_session = ASRSession(
                id="warmup",
                websocket=None,
                target_lang=self.target_lang,
            )

            # Run streaming step (processes entire mel as one chunk)
            if self.prompted_model:
                self._apply_inference_prompt(warmup_session)
            _ = self._conformer_stream_step(
                processed_signal=mel,
                processed_signal_length=mel_len,
                cache_last_channel=cache[0],
                cache_last_time=cache[1],
                cache_last_channel_len=cache[2],
                keep_all_outputs=True,
                previous_hypotheses=None,
                previous_pred_out=None,
                drop_extra_pre_encoded=0,
                return_transcription=True,
            )
            self._warm_encoder_compile_static_buckets(mel)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Warmup complete in {elapsed:.0f}ms - GPU memory claimed")

    def _init_session(self, session: ASRSession):
        """Initialize a fresh session.

        If an overlap_buffer is present from a previous segment, it will be
        prepended to the accumulated audio to provide encoder left-context.
        This enables seamless transcription across mid-utterance resets.
        """
        self._ensure_session_target_lang(session)

        # Initialize encoder cache
        cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        session.cache_last_channel = cache[0]
        session.cache_last_time = cache[1]
        session.cache_last_channel_len = cache[2]

        # Reset audio buffer and frame counter
        # If overlap buffer exists, use it as the starting audio
        if session.overlap_buffer is not None and len(session.overlap_buffer) > 0:
            session.pending_audio = session.overlap_buffer.copy()
            session.accumulated_audio = session.pending_audio
            session.total_audio_samples = len(session.pending_audio)
            overlap_ms = len(session.overlap_buffer) * 1000 / self.sample_rate
            logger.debug(
                f"Session {session.id}: prepending {len(session.overlap_buffer)} samples "
                f"({overlap_ms:.0f}ms) of overlap audio"
            )
            session.overlap_buffer = None  # Clear after use
        else:
            session.pending_audio = np.array([], dtype=np.float32)
            session.accumulated_audio = session.pending_audio
            session.total_audio_samples = 0

        session.raw_audio_ring = np.zeros(self.raw_audio_ring_samples, dtype=np.float32)
        session.mel_frame_ring = None

        # (Removed 2026-05-18 baseline hardening: the NEMOTRON_ONSET_WARMUP_MS
        # buffer-prepend was an ineffective + buggy onset warm-up. PLAN Step 8
        # implements the correct conformer_stream_step warm-up from scratch.)
        session.emitted_frames = 0

        # Reset decoder state
        session.previous_hypotheses = None
        session.pred_out_stream = None
        session.current_text = ""
        session.eou_probe_chunk_index = 0

        session.synthetic_prefix_samples = 0
        if self.session_warmup_ms > 0:
            self._run_session_warmup(session)

    def _run_session_warmup(self, session: ASRSession) -> None:
        """Prime one fresh session with synthetic silence without seeding text."""
        warmup_frames = self._session_warmup_frames()
        if warmup_frames is None:
            return
        warmup_samples = warmup_frames * self.hop_samples
        preprocess_samples = warmup_samples + self.hop_samples

        warmup_audio = np.zeros(preprocess_samples, dtype=np.float32)
        fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
            session.raw_audio_ring,
            warmup_audio,
        )

        with torch.inference_mode():
            mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
            warmup_mel = mel[
                :,
                :,
                self.first_preprocess_mel_frame : self.first_preprocess_mel_frame + warmup_frames,
            ]
            chunk_len = torch.tensor([warmup_mel.shape[-1]], device='cuda')

            if self.prompted_model:
                self._apply_inference_prompt(session)
            (
                session.pred_out_stream,
                discarded_transcribed_texts,
                session.cache_last_channel,
                session.cache_last_time,
                session.cache_last_channel_len,
                session.previous_hypotheses,
            ) = self._conformer_stream_step(
                processed_signal=warmup_mel,
                processed_signal_length=chunk_len,
                cache_last_channel=session.cache_last_channel,
                cache_last_time=session.cache_last_time,
                cache_last_channel_len=session.cache_last_channel_len,
                keep_all_outputs=False,
                previous_hypotheses=None,
                previous_pred_out=None,
                drop_extra_pre_encoded=0,
                return_transcription=True,
            )

        consumed_audio = warmup_audio[:warmup_samples]
        if len(consumed_audio) >= self.raw_audio_ring_samples:
            session.raw_audio_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
        else:
            keep = self.raw_audio_ring_samples - len(consumed_audio)
            session.raw_audio_ring = np.concatenate(
                [session.raw_audio_ring[-keep:], consumed_audio]
            ).astype(np.float32, copy=False)
        self._update_mel_frame_ring(session, warmup_mel)
        session.emitted_frames = warmup_frames
        session.synthetic_prefix_samples = warmup_samples

        discarded_text_present = bool(discarded_transcribed_texts and discarded_transcribed_texts[0])
        logger.info(
            f"Session {session.id}: per-session warm-up ran once at init "
            f"(NEMOTRON_WARMUP_MS={self.session_warmup_ms}, "
            f"frames={warmup_frames}, samples={warmup_samples}, "
            f"discarded_returned_text={discarded_text_present}, "
            f"current_text_chars={len(session.current_text)}, "
            f"last_emitted_text_chars={len(session.last_emitted_text)}, "
            f"mel_ring_frames={int(session.mel_frame_ring.shape[-1]) if session.mel_frame_ring is not None else 0}, "
            f"emitted_frames={session.emitted_frames})"
        )

    def _session_timeline_samples(self, session: ASRSession) -> int:
        return session.synthetic_prefix_samples + session.total_audio_samples

    def _preprocess_fixed_audio(
        self,
        audio: np.ndarray,
        valid_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the preprocessor with an invariant input shape.

        `valid_samples` may be shorter for final partial chunks; the tensor
        shape stays fixed so CUDA uses the same STFT/cuFFT plan every call.
        """
        if len(audio) != self.constant_preprocess_samples:
            raise ValueError(
                f"Expected fixed preprocessor input of {self.constant_preprocess_samples} samples, "
                f"got {len(audio)}"
            )
        audio_tensor = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0).cuda()
        audio_len = torch.tensor([valid_samples], device='cuda', dtype=torch.long)
        return self.model.preprocessor(input_signal=audio_tensor, length=audio_len)

    def _build_fixed_preprocess_audio(
        self,
        raw_audio_ring: np.ndarray,
        new_audio: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Assemble align-pad + raw ring + new audio and zero-pad to K."""
        if len(raw_audio_ring) != self.raw_audio_ring_samples:
            raise ValueError(
                f"Expected raw ring of {self.raw_audio_ring_samples} samples, "
                f"got {len(raw_audio_ring)}"
            )
        prefix_len = self.preprocess_align_pad_samples + self.raw_audio_ring_samples
        valid_samples = prefix_len + len(new_audio)
        if valid_samples > self.constant_preprocess_samples:
            raise ValueError(
                f"Fixed preprocessor valid span {valid_samples} exceeds K={self.constant_preprocess_samples}"
            )

        audio = np.zeros(self.constant_preprocess_samples, dtype=np.float32)
        cursor = self.preprocess_align_pad_samples
        audio[cursor : cursor + self.raw_audio_ring_samples] = raw_audio_ring
        cursor += self.raw_audio_ring_samples
        audio[cursor : cursor + len(new_audio)] = new_audio
        return audio, valid_samples

    def _update_mel_frame_ring(self, session: ASRSession, new_mel: torch.Tensor) -> None:
        """Retain the mel pre-encode cache separately from raw STFT context."""
        if session.mel_frame_ring is None:
            combined = new_mel.detach()
        else:
            combined = torch.cat((session.mel_frame_ring, new_mel.detach()), dim=-1)
        session.mel_frame_ring = combined[:, :, -self.pre_encode_cache_size :].detach()

    @staticmethod
    def _probe_scalar(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            value = value.detach().cpu().reshape(-1)[0].item()
        if isinstance(value, np.generic):
            value = value.item()
        return value

    @classmethod
    def _probe_int(cls, value: Any) -> Optional[int]:
        value = cls._probe_scalar(value)
        if value is None:
            return None
        return int(value)

    @classmethod
    def _probe_float(cls, value: Any) -> Optional[float]:
        value = cls._probe_scalar(value)
        if value is None:
            return None
        return float(value)

    @classmethod
    def _probe_int_list(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if torch.is_tensor(value):
            return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
        return [int(cls._probe_scalar(item)) for item in value]

    def _probe_blank_id(self) -> int:
        decoding = getattr(self.model, "decoding", None)
        blank_id = getattr(decoding, "blank_id", None)
        if blank_id is not None:
            return int(self._probe_scalar(blank_id))
        joint = getattr(self.model, "joint", None)
        return int(getattr(joint, "num_classes_with_blank") - 1)

    def _probe_token_strings(self, token_ids: list[int]) -> list[str]:
        if not token_ids:
            return []
        decoding = getattr(self.model, "decoding", None)
        if decoding is not None and hasattr(decoding, "decode_ids_to_tokens"):
            try:
                token_strings = [str(token) for token in decoding.decode_ids_to_tokens(token_ids)]
                if len(token_strings) == len(token_ids):
                    return token_strings
            except Exception:
                pass
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "ids_to_tokens"):
            try:
                token_strings = [str(token) for token in tokenizer.ids_to_tokens(token_ids)]
                if len(token_strings) == len(token_ids):
                    return token_strings
            except Exception:
                pass
        return [str(token_id) for token_id in token_ids]

    @staticmethod
    def _probe_token_starts_word(token: str) -> bool:
        return token.startswith(("\u2581", "\u0120")) or token[:1].isspace()

    @classmethod
    def _probe_word_boundary_state(cls, tokens: list[str], index: int) -> dict[str, Any]:
        token = tokens[index] if index < len(tokens) else ""
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        stripped = token.replace("\u2581", "").replace("\u0120", "").strip()
        starts_word = index == 0 or cls._probe_token_starts_word(token)
        next_starts_word = next_token is not None and cls._probe_token_starts_word(next_token)
        punctuation_boundary = stripped in {".", ",", "?", "!", ";", ":"}
        return {
            "starts_word": starts_word,
            "extends_word": not starts_word,
            "completes_word": bool(next_starts_word or punctuation_boundary),
            "completion_observed": next_token is not None or punctuation_boundary,
        }

    @classmethod
    def _probe_hyp_timestamps(cls, hyp: Any) -> list[int]:
        timestamp = getattr(hyp, "timestamp", None)
        if isinstance(timestamp, dict):
            timestamp = timestamp.get("timestep", [])
        return cls._probe_int_list(timestamp)

    @classmethod
    def _probe_alignment_label(cls, item: Any) -> Optional[int]:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            item = item[-1]
        return cls._probe_int(item)

    @classmethod
    def _probe_alignment_labels_by_frame(cls, alignments: Any) -> list[list[int]]:
        if not alignments:
            return []
        labels_by_frame: list[list[int]] = []
        for frame in alignments:
            if frame is None or (isinstance(frame, (list, tuple)) and len(frame) == 0):
                labels_by_frame.append([])
                continue
            if isinstance(frame, (list, tuple)) and len(frame) > 0 and isinstance(frame[0], (list, tuple)):
                items = frame
            else:
                items = [frame]
            labels = []
            for item in items:
                label = cls._probe_alignment_label(item)
                if label is not None:
                    labels.append(label)
            labels_by_frame.append(labels)
        return labels_by_frame

    def _probe_frame_alignment(
        self,
        labels_by_frame: list[list[int]],
        *,
        blank_id: int,
        chunk_model_frame_start: int,
    ) -> list[dict[str, Any]]:
        frame_alignment = []
        for frame_offset, labels in enumerate(labels_by_frame):
            frame_alignment.append(
                {
                    "frame_offset": frame_offset,
                    "model_frame_index": chunk_model_frame_start + frame_offset,
                    "labels": labels,
                    "has_non_blank": any(label != blank_id for label in labels),
                    "all_blank": bool(labels) and all(label == blank_id for label in labels),
                }
            )
        return frame_alignment

    @classmethod
    def _probe_frame_confidence(cls, frame_confidence: Any) -> list[list[float]]:
        if not frame_confidence:
            return []
        confidence_rows: list[list[float]] = []
        for frame in frame_confidence:
            if isinstance(frame, (list, tuple)):
                confidence_rows.append(
                    [
                        confidence
                        for confidence in (cls._probe_float(item) for item in frame)
                        if confidence is not None
                    ]
                )
            else:
                confidence = cls._probe_float(frame)
                confidence_rows.append([] if confidence is None else [confidence])
        return confidence_rows

    @staticmethod
    def _probe_changed_positions(prev_y: list[int], y_sequence: list[int]) -> list[int]:
        changed_positions = []
        for index in range(max(len(prev_y), len(y_sequence))):
            prev_token = prev_y[index] if index < len(prev_y) else None
            token = y_sequence[index] if index < len(y_sequence) else None
            if prev_token != token:
                changed_positions.append(index)
        return changed_positions

    @staticmethod
    def _probe_token_frame_from_alignments(
        labels_by_frame: list[list[int]],
        *,
        blank_id: int,
        token_offset: int,
    ) -> Optional[int]:
        non_blank_frames = []
        for frame_offset, labels in enumerate(labels_by_frame):
            non_blank_frames.extend(frame_offset for label in labels if label != blank_id)
        if token_offset < len(non_blank_frames):
            return non_blank_frames[token_offset]
        return None

    @staticmethod
    def _probe_token_frame_events_from_alignments(
        labels_by_frame: list[list[int]],
        *,
        blank_id: int,
    ) -> list[tuple[int, int]]:
        token_events = []
        for frame_offset, labels in enumerate(labels_by_frame):
            frame_token_offset = 0
            for label in labels:
                if label == blank_id:
                    continue
                token_events.append((frame_offset, frame_token_offset))
                frame_token_offset += 1
        return token_events

    def _eou_probe_snapshot(self, session: ASRSession) -> Optional[dict[str, Any]]:
        if not self.eou_probe_enabled:
            return None
        prev_hyp = session.previous_hypotheses[0] if session.previous_hypotheses else None
        prev_y = self._probe_int_list(getattr(prev_hyp, "y_sequence", [])) if prev_hyp is not None else []
        return {
            "chunk_index": session.eou_probe_chunk_index,
            "chunk_model_frame_start": session.emitted_frames,
            "prev_y": prev_y,
            "prev_y_len": len(prev_y),
            "monotonic_start": time.monotonic(),
            "wall_time_start": time.time(),
        }

    def _write_eou_probe_chunk(
        self,
        session: ASRSession,
        snapshot: Optional[dict[str, Any]],
    ) -> None:
        if not self.eou_probe_enabled or snapshot is None or self.eou_probe_path is None:
            return
        try:
            hyp = session.previous_hypotheses[0] if session.previous_hypotheses else None
            chunk_index = session.eou_probe_chunk_index
            session.eou_probe_chunk_index += 1
            chunk_model_frame_start = int(snapshot["chunk_model_frame_start"])
            prev_y = snapshot["prev_y"]
            blank_id = self._probe_blank_id()
            y_sequence = self._probe_int_list(getattr(hyp, "y_sequence", [])) if hyp is not None else []
            token_strings = self._probe_token_strings(y_sequence)
            timestamps = self._probe_hyp_timestamps(hyp) if hyp is not None else []
            alignments = getattr(hyp, "alignments", None) if hyp is not None else None
            labels_by_frame = self._probe_alignment_labels_by_frame(alignments)
            token_frame_events = self._probe_token_frame_events_from_alignments(
                labels_by_frame,
                blank_id=blank_id,
            )
            frame_confidence = self._probe_frame_confidence(
                getattr(hyp, "frame_confidence", None) if hyp is not None else None
            )

            new_tokens = []
            prev_y_len = len(prev_y)
            frame_subindex_counts: dict[int, int] = {}
            for token_index in range(prev_y_len, len(y_sequence)):
                token_offset = token_index - prev_y_len
                alignment_frame_index = None
                alignment_subindex = None
                if token_offset < len(token_frame_events):
                    alignment_frame_index, alignment_subindex = token_frame_events[token_offset]
                chunk_frame_index = (
                    timestamps[token_offset]
                    if token_offset < len(timestamps)
                    else alignment_frame_index
                )
                if chunk_frame_index is None:
                    chunk_frame_index = self._probe_token_frame_from_alignments(
                        labels_by_frame,
                        blank_id=blank_id,
                        token_offset=token_offset,
                    )
                model_frame_index = (
                    chunk_model_frame_start + chunk_frame_index
                    if chunk_frame_index is not None
                    else None
                )
                model_frame_subindex = None
                model_frame_event_index = None
                if chunk_frame_index is not None:
                    fallback_subindex = frame_subindex_counts.get(chunk_frame_index, 0)
                    model_frame_subindex = (
                        alignment_subindex
                        if alignment_subindex is not None and alignment_frame_index == chunk_frame_index
                        else fallback_subindex
                    )
                    frame_subindex_counts[chunk_frame_index] = max(
                        frame_subindex_counts.get(chunk_frame_index, 0),
                        model_frame_subindex + 1,
                    )
                if model_frame_index is not None and model_frame_subindex is not None:
                    model_frame_event_index = model_frame_index * 1024 + model_frame_subindex
                token_string = token_strings[token_index] if token_index < len(token_strings) else str(y_sequence[token_index])
                new_tokens.append(
                    {
                        "token_index": token_index,
                        "token_id": y_sequence[token_index],
                        "token": token_string,
                        "chunk_frame_index": chunk_frame_index,
                        "model_frame_index": model_frame_index,
                        "model_frame_subindex": model_frame_subindex,
                        "model_frame_event_index": model_frame_event_index,
                        "word_boundary": self._probe_word_boundary_state(token_strings, token_index),
                    }
                )

            payload = {
                "type": "eou_probe_chunk",
                "run_tag": self.eou_probe_tag,
                "session_id": session.id,
                "chunk_index": chunk_index,
                "chunk_model_frame_start": chunk_model_frame_start,
                "prev_y_len": snapshot["prev_y_len"],
                "emitted_frames_after": session.emitted_frames,
                "shift_frames": self.shift_frames,
                "right_context_chunks": self.right_context,
                "right_context_frames": self.right_context * self.shift_frames,
                "R": self.right_context * self.shift_frames,
                "blank_id": blank_id,
                "monotonic_start": snapshot["monotonic_start"],
                "monotonic_done": time.monotonic(),
                "wall_time_start": snapshot["wall_time_start"],
                "wall_time_done": time.time(),
                "real_audio_cursor_samples": session.total_audio_samples,
                "real_audio_cursor_seconds": session.total_audio_samples / self.sample_rate,
                "timeline_cursor_samples": self._session_timeline_samples(session),
                "hyp_score": self._probe_float(getattr(hyp, "score", None)) if hyp is not None else None,
                "y_sequence": y_sequence,
                "changed_positions": self._probe_changed_positions(prev_y, y_sequence),
                "new_tokens": new_tokens,
                "frame_alignment": self._probe_frame_alignment(
                    labels_by_frame,
                    blank_id=blank_id,
                    chunk_model_frame_start=chunk_model_frame_start,
                ),
                "frame_confidence": frame_confidence,
            }
            self.eou_probe_path.parent.mkdir(parents=True, exist_ok=True)
            with _EOU_PROBE_LOCK:
                with self.eou_probe_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as e:
            logger.error(f"Session {session.id} EOU probe write error: {e}")

    def _eou_snapshot_file_stem(self, session: ASRSession) -> str:
        return f"{_safe_tag_filename(self.eou_probe_tag)}_{_safe_tag_filename(session.id)}"

    def _capture_eou_snapshot_audio(self, session: ASRSession, audio_bytes: bytes) -> None:
        if self.eou_snapshot_dir is None or not audio_bytes:
            return
        session.eou_snapshot_audio.extend(audio_bytes)

    def _flush_eou_snapshot_audio(self, session: ASRSession) -> None:
        if self.eou_snapshot_dir is None or not session.eou_snapshot_audio:
            return
        try:
            audio_path = self.eou_snapshot_dir / f"{self._eou_snapshot_file_stem(session)}_audio.bin"
            with _EOU_SNAPSHOT_LOCK:
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                with audio_path.open("wb") as f:
                    f.write(session.eou_snapshot_audio)
        except Exception as e:
            logger.error(f"Session {session.id} EOU snapshot audio write error: {e}")

    def _write_eou_snapshot_chunk(
        self,
        session: ASRSession,
        snapshot: Optional[dict[str, Any]],
    ) -> None:
        if self.eou_snapshot_dir is None or snapshot is None:
            return

        chunk_index = int(snapshot["chunk_index"])
        if chunk_index % self.eou_snapshot_every != 0:
            return

        try:
            previous_hypotheses = clone_hypotheses_deep(session.previous_hypotheses)
            payload = {
                "cache_last_channel": snapshot_tree_cpu(session.cache_last_channel),
                "cache_last_time": snapshot_tree_cpu(session.cache_last_time),
                "cache_last_channel_len": snapshot_tree_cpu(session.cache_last_channel_len),
                "previous_hypotheses": snapshot_tree_cpu(previous_hypotheses),
                "pred_out_stream": snapshot_tree_cpu(session.pred_out_stream),
                "pending_audio": (
                    session.pending_audio.copy()
                    if session.pending_audio is not None
                    else np.array([], dtype=np.float32)
                ),
                "raw_audio_ring": (
                    session.raw_audio_ring.copy()
                    if session.raw_audio_ring is not None
                    else np.array([], dtype=np.float32)
                ),
                "mel_frame_ring": snapshot_tree_cpu(session.mel_frame_ring),
                "emitted_frames": int(session.emitted_frames),
                "synthetic_prefix_samples": int(session.synthetic_prefix_samples),
                "total_audio_samples": int(session.total_audio_samples),
                "chunk_index": chunk_index,
                "monotonic_time": time.monotonic(),
                "run_tag": self.eou_probe_tag,
                "session_id": session.id,
                "real_audio_cursor_samples": int(session.total_audio_samples),
                "timeline_cursor_samples": int(self._session_timeline_samples(session)),
            }
            snapshot_path = (
                self.eou_snapshot_dir
                / f"{self._eou_snapshot_file_stem(session)}_chunk{chunk_index:06d}.pt"
            )
            with _EOU_SNAPSHOT_LOCK:
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, snapshot_path)
        except Exception as e:
            logger.error(f"Session {session.id} EOU snapshot chunk write error: {e}")

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket client connection."""
        import uuid

        ws = web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)
        await ws.prepare(request)

        try:
            session_target_lang = self._validate_connection_query(request.query)
        except ValueError as e:
            logger.warning(f"Rejecting WebSocket connection: {e}")
            await ws.send_str(json.dumps({"type": "error", "message": str(e)}))
            await ws.close()
            return ws

        session_id = str(uuid.uuid4())[:8]
        session = ASRSession(
            id=session_id,
            websocket=ws,
            target_lang=session_target_lang,
        )
        self.sessions[session_id] = session

        logger.info(f"Client {session_id} connected")

        try:
            async with self.inference_lock:
                await self._run_inference_call(self._init_session, session)

            if self.continuous_context:
                self._start_continuous_session(session)

            await ws.send_str(json.dumps({"type": "ready"}))
            logger.debug(f"Client {session_id}: sent ready")

            async for msg in ws:
                if self.continuous_context:
                    if self.scheduler_enabled:
                        await self._queue_scheduler_ws_message(session, msg)
                    else:
                        await self._queue_continuous_ws_message(session, msg)
                    continue

                if msg.type == WSMsgType.BINARY:
                    await self._handle_audio(session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "reset" or msg_type == "end":
                            # finalize=True (default): hard reset with padding + keep_all_outputs
                            # finalize=False: soft reset, just return current text
                            finalize = data.get("finalize", True)
                            await self._reset_session(session, finalize=finalize)
                        elif msg_type == "vad_start" or msg_type == "vad_stop":
                            logger.debug(
                                f"Client {session_id}: received {msg_type} (no-op)"
                            )
                        else:
                            logger.warning(f"Client {session_id}: unknown message type: {msg_type}")

                    except json.JSONDecodeError:
                        logger.warning(f"Client {session_id}: invalid JSON")
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"Client {session_id} WebSocket error: {ws.exception()}")
                    break

            logger.info(f"Client {session_id} disconnected")

        except Exception as e:
            logger.error(f"Client {session_id} error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await ws.send_str(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except:
                pass
        finally:
            if self.continuous_context:
                await self._close_continuous_session(session)
            self._flush_eou_snapshot_audio(session)
            if session_id in self.sessions:
                del self.sessions[session_id]
            self._log_retained_cache_telemetry("session_removed")

        return ws

    def _start_continuous_session(self, session: ASRSession) -> None:
        """Start the ordered per-session event worker for continuous mode."""
        if self.scheduler_enabled:
            self._start_scheduler_continuous_session(session)
            return

        session.continuous_event_queue = asyncio.Queue()
        session.continuous_worker_task = asyncio.create_task(
            self._continuous_session_worker(session),
            name=f"nemotron-continuous-session-{session.id}",
        )
        logger.info(
            f"Session {session.id}: continuous context enabled "
            f"(debounce={self.finalize_silence_ms}ms)"
        )

    async def _queue_continuous_ws_message(self, session: ASRSession, msg) -> None:
        """Queue raw WS events so continuous-mode control/audio is ordered."""
        queue = session.continuous_event_queue
        if queue is None:
            logger.warning(f"Session {session.id}: continuous event queue missing")
            return

        if msg.type == WSMsgType.BINARY:
            await queue.put(("audio", msg.data))
        elif msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning(f"Client {session.id}: invalid JSON")
                return

            msg_type = data.get("type")
            if msg_type == "reset" or msg_type == "end":
                finalize = data.get("finalize", True)
                await queue.put(("reset", finalize, msg_type))
            elif msg_type == "vad_start" or msg_type == "vad_stop":
                await queue.put((msg_type,))
            else:
                logger.warning(f"Client {session.id}: unknown message type: {msg_type}")
        elif msg.type == WSMsgType.ERROR:
            logger.error(f"Client {session.id} WebSocket error: {session.websocket.exception()}")
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            await queue.put(("close",))

    def _ensure_scheduler_task(self) -> None:
        if not self.scheduler_enabled:
            return
        if self._scheduler_wakeup is None:
            self._scheduler_wakeup = asyncio.Event()
        if self.scheduler_task is None or self.scheduler_task.done():
            self.scheduler_task = asyncio.create_task(
                self._scheduler_loop(),
                name="nemotron-scheduler-b1",
            )
            logger.info(
                "scheduler_b1_started "
                f"queue_maxsize={self.scheduler_queue_maxsize} "
                f"batch_enabled={self.batch_enabled} "
                f"batch_max_wait_ms={self.batch_max_wait_ms} "
                f"batch_max_size={self.batch_max_size}"
            )

    def _wake_scheduler(self) -> None:
        if self._scheduler_wakeup is not None:
            self._scheduler_wakeup.set()

    def _start_scheduler_continuous_session(self, session: ASRSession) -> None:
        """Start scheduler-owned continuous mode without a per-session worker."""
        self._ensure_scheduler_task()
        session.continuous_event_queue = asyncio.Queue(
            maxsize=self.scheduler_queue_maxsize
        )
        session.continuous_worker_task = None
        session.scheduler_closed = False
        logger.info(
            f"Session {session.id}: continuous context enabled via scheduler_b1 "
            f"(debounce={self.finalize_silence_ms}ms)"
        )

    async def _scheduler_queue_event(self, session: ASRSession, event: tuple) -> None:
        queue = session.continuous_event_queue
        if queue is None:
            logger.warning(f"Session {session.id}: scheduler event queue missing")
            return
        await queue.put(event)
        self._wake_scheduler()

    async def _queue_scheduler_ws_message(self, session: ASRSession, msg) -> None:
        """Queue raw WS events for the central B=1 scheduler."""
        if msg.type == WSMsgType.BINARY:
            await self._scheduler_queue_event(session, ("audio", msg.data))
        elif msg.type == WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning(f"Client {session.id}: invalid JSON")
                return

            msg_type = data.get("type")
            if msg_type == "reset" or msg_type == "end":
                finalize = data.get("finalize", True)
                await self._scheduler_queue_event(session, ("reset", finalize, msg_type))
            elif msg_type == "vad_start" or msg_type == "vad_stop":
                await self._scheduler_queue_event(session, (msg_type,))
            else:
                logger.warning(f"Client {session.id}: unknown message type: {msg_type}")
        elif msg.type == WSMsgType.ERROR:
            logger.error(f"Client {session.id} WebSocket error: {session.websocket.exception()}")
        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            await self._scheduler_queue_event(session, ("close",))

    def _scheduler_next_batch_deadline(self) -> Optional[float]:
        if not self.batch_enabled or not self._scheduler_ready:
            return None
        if not self._scheduler_batch_first_ready:
            return None
        return min(self._scheduler_batch_first_ready.values())

    def _scheduler_wait_timeout(self) -> Optional[float]:
        deadline = self._scheduler_next_batch_deadline()
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _scheduler_has_queued_events(self) -> bool:
        for session in list(self.sessions.values()):
            queue = session.continuous_event_queue
            if queue is not None and not queue.empty():
                return True
        return False

    def _scheduler_has_work_or_due_timer(self) -> bool:
        if self._scheduler_has_queued_events():
            return True
        if self._scheduler_ready:
            if not self.batch_enabled:
                return True
            deadline = self._scheduler_next_batch_deadline()
            if deadline is None or time.monotonic() >= deadline:
                return True
        return False

    async def _scheduler_loop(self) -> None:
        logger.info(
            "scheduler_b1_loop_running "
            f"batch_enabled={self.batch_enabled} "
            f"batch_max_wait_ms={self.batch_max_wait_ms} "
            f"batch_max_size={self.batch_max_size}"
        )
        while True:
            try:
                progressed = await self._scheduler_drain_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"scheduler_b1_loop_error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                progressed = False

            if progressed:
                continue

            if self._scheduler_wakeup is None:
                self._scheduler_wakeup = asyncio.Event()
            self._scheduler_wakeup.clear()
            if self._scheduler_has_work_or_due_timer():
                continue
            timeout = self._scheduler_wait_timeout()
            if timeout is None:
                await self._scheduler_wakeup.wait()
            else:
                try:
                    await asyncio.wait_for(self._scheduler_wakeup.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass

    async def _scheduler_drain_once(self) -> bool:
        progressed = False
        for session in list(self.sessions.values()):
            queue = session.continuous_event_queue
            if queue is None:
                continue
            try:
                event = queue.get_nowait()
            except asyncio.QueueEmpty:
                continue

            try:
                await self._scheduler_process_event(session, event)
            finally:
                queue.task_done()
            progressed = True

        if self._scheduler_ready:
            progressed = await self._scheduler_process_ready_pass() or progressed

        return progressed

    def _scheduler_session_ready(self, session: ASRSession) -> bool:
        if session.scheduler_closed or session.continuous_event_queue is None:
            return False
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        return ready_predicate(
            synthetic_prefix_samples=session.synthetic_prefix_samples,
            total_audio_samples=session.total_audio_samples,
            emitted_frames=session.emitted_frames,
            shift_frames=self.shift_frames,
            hop_samples=self.hop_samples,
            pending_audio_len=pending_len,
            preprocess_new_audio_samples=self.preprocess_new_audio_samples,
        )

    def _scheduler_mark_ready_if_ready_locked(self, session: ASRSession) -> None:
        if self._scheduler_session_ready(session):
            if session.id not in self._scheduler_ready:
                session.scheduler_ready_since = time.monotonic()
            self._scheduler_ready.add(session.id)
        else:
            self._scheduler_ready.discard(session.id)
            session.scheduler_ready_since = None

    async def _scheduler_process_ready_pass(self) -> bool:
        if self.batch_enabled:
            return await self._scheduler_process_batched_ready_pass()

        progressed = False
        ready_ids = list(self._scheduler_ready)
        self._scheduler_ready.clear()
        for session_id in ready_ids:
            session = self.sessions.get(session_id)
            if session is None:
                continue
            async with session.state_lock:
                if not self._scheduler_session_ready(session):
                    session.scheduler_ready_since = None
                    continue
                processed = await self._scheduler_process_one_ready_chunk_locked(
                    session,
                    reason="ready",
                    requeue=True,
                )
                progressed = processed or progressed
        return progressed

    def _scheduler_batch_group_key_for_session(self, session: ASRSession) -> tuple:
        if session.emitted_frames == 0:
            chunk_t = self.shift_frames
            drop_extra = 0
        else:
            chunk_t = self.pre_encode_cache_size + self.shift_frames
            drop_extra = self.drop_extra
        target_lang = session.target_lang if session.target_lang is not None else self.target_lang
        base_key = batch_group_key(
            target_lang,
            False,
            drop_extra,
            chunk_t,
            self.decoder_strategy,
        )
        # RNNT batched state is validated only when decoder histories are
        # uniformly fresh or uniformly established. Include that in the live
        # grouping key so a newly joined stream cannot be coerced into an
        # established batch.
        return (
            *base_key,
            session.previous_hypotheses is None,
            session.pred_out_stream is None,
        )

    def _scheduler_batch_session_eligible_for_key(self, session: ASRSession, key: tuple) -> bool:
        if session.scheduler_closed or session.continuous_event_queue is None:
            return False
        if self._scheduler_batch_group_key_for_session(session) != key:
            return False
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        queue = session.continuous_event_queue
        queue_has_events = queue is not None and not queue.empty()
        return self._scheduler_session_ready(session) or pending_len > 0 or queue_has_events

    def _scheduler_batch_eligible_count(self, key: tuple) -> int:
        count = 0
        for session in list(self.sessions.values()):
            if self._scheduler_batch_session_eligible_for_key(session, key):
                count += 1
        return count

    async def _scheduler_collect_ready_groups(self) -> dict[tuple, list[ASRSession]]:
        ready_groups: dict[tuple, list[ASRSession]] = {}
        now = time.monotonic()
        for session_id in list(self._scheduler_ready):
            session = self.sessions.get(session_id)
            if session is None:
                self._scheduler_ready.discard(session_id)
                continue
            async with session.state_lock:
                if not self._scheduler_session_ready(session):
                    self._scheduler_ready.discard(session_id)
                    session.scheduler_ready_since = None
                    continue
                if session.scheduler_ready_since is None:
                    session.scheduler_ready_since = now
                key = self._scheduler_batch_group_key_for_session(session)
                ready_groups.setdefault(key, []).append(session)

        for key in list(self._scheduler_batch_first_ready):
            if key not in ready_groups:
                self._scheduler_batch_first_ready.pop(key, None)
        return ready_groups

    async def _scheduler_process_batched_ready_pass(self) -> bool:
        ready_groups = await self._scheduler_collect_ready_groups()
        if not ready_groups:
            return False

        now = time.monotonic()
        candidates: list[tuple[int, float, str, tuple, list[ASRSession], str, int, int]] = []
        for key, sessions in ready_groups.items():
            sessions.sort(key=lambda s: (s.scheduler_ready_since or now, s.id))
            ready_count = len(sessions)
            active_count = self._scheduler_batch_eligible_count(key)
            first_ready = min((s.scheduler_ready_since or now) for s in sessions)
            deadline = self._scheduler_batch_first_ready.get(key)
            if deadline is None:
                deadline = first_ready + (self.batch_max_wait_ms / 1000.0)
                self._scheduler_batch_first_ready[key] = deadline

            reason = ""
            if active_count <= 1:
                reason = "solo"
            elif ready_count >= self.batch_max_size:
                reason = "max_size"
            elif now >= deadline:
                reason = "timer"
            else:
                continue

            safe_size = min(ready_count, self.batch_max_size)
            candidates.append(
                (
                    -safe_size,
                    deadline,
                    str(key),
                    key,
                    sessions[:safe_size],
                    reason,
                    ready_count,
                    active_count,
                )
            )

        if not candidates:
            return False

        candidates.sort()
        (
            _neg_size,
            _deadline,
            _key_label,
            key,
            sessions,
            reason,
            ready_count,
            active_count,
        ) = candidates[0]
        return await self._scheduler_process_ready_batch_locked_sessions(
            key,
            sessions,
            reason=reason,
            ready_count=ready_count,
            eligible_count=active_count,
        )

    async def _scheduler_process_ready_batch_locked_sessions(
        self,
        key: tuple,
        sessions: list[ASRSession],
        *,
        reason: str,
        ready_count: int,
        eligible_count: int,
    ) -> bool:
        if not sessions:
            return False

        async with contextlib.AsyncExitStack() as stack:
            for session in sorted(sessions, key=lambda s: s.id):
                await stack.enter_async_context(session.state_lock)

            valid_sessions: list[ASRSession] = []
            for session in sessions:
                if (
                    not session.scheduler_closed
                    and self._scheduler_session_ready(session)
                    and self._scheduler_batch_group_key_for_session(session) == key
                ):
                    valid_sessions.append(session)
                else:
                    self._scheduler_ready.discard(session.id)
                    session.scheduler_ready_since = None

            if not valid_sessions:
                return False
            if len(valid_sessions) > self.batch_max_size:
                valid_sessions = valid_sessions[: self.batch_max_size]

            for session in valid_sessions:
                self._scheduler_ready.discard(session.id)
                session.scheduler_inflight_generation = session.scheduler_generation

            generations = {
                session.id: session.scheduler_generation for session in valid_sessions
            }
            dispatch_start = time.monotonic()
            lane_wait_start = time.perf_counter()
            lane_wait_ms = 0.0
            texts: dict[str, Optional[str]] = {}
            try:
                async with self.inference_lock:
                    lane_wait_ms = (time.perf_counter() - lane_wait_start) * 1000.0
                    live_sessions = [
                        session
                        for session in valid_sessions
                        if (
                            generations[session.id] == session.scheduler_generation
                            and not session.scheduler_closed
                        )
                    ]
                    if not live_sessions:
                        return False
                    texts = await self._run_inference_call(
                        self._process_ready_batch,
                        live_sessions,
                    )
            finally:
                for session in valid_sessions:
                    session.scheduler_inflight_generation = None

            progressed = False
            sent_count = 0
            for session in valid_sessions:
                generation = generations[session.id]
                if generation != session.scheduler_generation or session.scheduler_closed:
                    logger.debug(
                        f"Session {session.id}: suppressed stale scheduler batch output "
                        f"reason={reason} gen={generation} current={session.scheduler_generation}"
                    )
                    continue

                text = texts.get(session.id)
                if text is not None and text != session.current_text:
                    session.current_text = text
                    logger.debug(
                        f"Session {session.id} interim: "
                        f"{text[-50:] if len(text) > 50 else text}"
                    )
                    await self._send_json_locked(
                        session,
                        {
                            "type": "transcript",
                            "text": text,
                            "is_final": False,
                        },
                        tolerate_closed=True,
                        description="scheduler batch interim transcript",
                    )
                    sent_count += 1

                queue_wait_ms = 0.0
                if session.scheduler_ready_since is not None:
                    queue_wait_ms = (
                        dispatch_start - session.scheduler_ready_since
                    ) * 1000.0
                session.scheduler_ready_since = None
                self._scheduler_record_batch_row_telemetry(
                    session,
                    batch_size=len(valid_sessions),
                    lane_wait_ms=lane_wait_ms,
                    queue_wait_ms=queue_wait_ms,
                    reason=reason,
                )
                self._scheduler_mark_ready_if_ready_locked(session)
                progressed = True

            self._scheduler_record_batch_telemetry(
                batch_size=len(valid_sessions),
                reason=reason,
                sent_count=sent_count,
                key=key,
                ready_count=ready_count,
                eligible_count=eligible_count,
            )
            return progressed

    async def _scheduler_process_one_ready_chunk_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        requeue: bool,
    ) -> bool:
        if not self._scheduler_session_ready(session):
            self._scheduler_ready.discard(session.id)
            session.scheduler_ready_since = None
            return False

        self._scheduler_ready.discard(session.id)
        session.scheduler_ready_since = None
        generation = session.scheduler_generation
        session.scheduler_inflight_generation = generation
        lane_wait_start = time.perf_counter()
        lane_wait_ms = 0.0
        text: Optional[str] = None
        try:
            async with self.inference_lock:
                lane_wait_ms = (time.perf_counter() - lane_wait_start) * 1000.0
                if generation != session.scheduler_generation or session.scheduler_closed:
                    logger.debug(
                        f"Session {session.id}: skipped stale scheduler chunk "
                        f"reason={reason} gen={generation} current={session.scheduler_generation}"
                    )
                    return False
                text = await self._run_inference_call(self._process_chunk, session)
        finally:
            session.scheduler_inflight_generation = None

        if generation != session.scheduler_generation or session.scheduler_closed:
            logger.debug(
                f"Session {session.id}: suppressed stale scheduler chunk output "
                f"reason={reason} gen={generation} current={session.scheduler_generation}"
            )
            return False

        if text is not None and text != session.current_text:
            session.current_text = text
            logger.debug(
                f"Session {session.id} interim: "
                f"{text[-50:] if len(text) > 50 else text}"
            )
            if generation == session.scheduler_generation and not session.scheduler_closed:
                await self._send_json_locked(
                    session,
                    {
                        "type": "transcript",
                        "text": text,
                        "is_final": False,
                    },
                    tolerate_closed=True,
                    description="scheduler interim transcript",
                )

        self._scheduler_record_chunk_telemetry(session, lane_wait_ms, reason)
        if requeue:
            self._scheduler_mark_ready_if_ready_locked(session)
        return True

    def _scheduler_record_batch_row_telemetry(
        self,
        session: ASRSession,
        *,
        batch_size: int,
        lane_wait_ms: float,
        queue_wait_ms: float,
        reason: str,
    ) -> None:
        self._scheduler_record_chunk_telemetry(
            session,
            lane_wait_ms,
            f"batch:{reason}:B{batch_size}",
        )
        self._scheduler_batch_queue_wait_ms_total += queue_wait_ms
        self._scheduler_batch_queue_wait_ms_max = max(
            self._scheduler_batch_queue_wait_ms_max,
            queue_wait_ms,
        )
        self._scheduler_batch_queue_wait_count += 1

    def _scheduler_record_batch_telemetry(
        self,
        *,
        batch_size: int,
        reason: str,
        sent_count: int,
        key: Optional[tuple] = None,
        ready_count: Optional[int] = None,
        eligible_count: Optional[int] = None,
    ) -> None:
        self._scheduler_batches += 1
        self._scheduler_batch_size_hist[batch_size] = (
            self._scheduler_batch_size_hist.get(batch_size, 0) + 1
        )
        if self._scheduler_batches % 25 == 0 or batch_size > 1:
            wait_avg = 0.0
            if self._scheduler_batch_queue_wait_count:
                wait_avg = (
                    self._scheduler_batch_queue_wait_ms_total
                    / self._scheduler_batch_queue_wait_count
                )
            logger.info(
                "scheduler_batch_telemetry "
                f"batches={self._scheduler_batches} "
                f"last_batch_size={batch_size} "
                f"last_reason={reason} "
                f"group_key={key} "
                f"prompt_lang={(key[0] if key else 'n/a')} "
                f"ready_count={(ready_count if ready_count is not None else 'n/a')} "
                f"eligible_ready_count={(eligible_count if eligible_count is not None else 'n/a')} "
                f"sent_count={sent_count} "
                f"effective_batch_hist={dict(sorted(self._scheduler_batch_size_hist.items()))} "
                f"fallback_counts={dict(sorted(self._scheduler_batch_fallback_counts.items()))} "
                f"last_fallback_reason={self.batch_fallback_reason or 'none'} "
                f"queue_wait_avg_ms={wait_avg:.2f} "
                f"queue_wait_max_ms={self._scheduler_batch_queue_wait_ms_max:.2f}"
            )

    def _scheduler_record_chunk_telemetry(
        self,
        session: ASRSession,
        lane_wait_ms: float,
        reason: str,
    ) -> None:
        self._scheduler_chunks += 1
        self._scheduler_lane_wait_ms_total += lane_wait_ms
        self._scheduler_lane_wait_ms_max = max(
            self._scheduler_lane_wait_ms_max,
            lane_wait_ms,
        )
        queue = session.continuous_event_queue
        queue_depth = queue.qsize() if queue is not None else 0
        lag_ms = None
        if session.scheduler_last_audio_monotonic is not None:
            lag_ms = (time.monotonic() - session.scheduler_last_audio_monotonic) * 1000.0
        lag_label = f"{lag_ms:.2f}" if lag_ms is not None else "n/a"
        if self._scheduler_chunks % 50 == 0:
            avg_wait = self._scheduler_lane_wait_ms_total / self._scheduler_chunks
            logger.info(
                "scheduler_b1_telemetry "
                f"chunks={self._scheduler_chunks} "
                f"model_lane_wait_avg_ms={avg_wait:.2f} "
                f"model_lane_wait_max_ms={self._scheduler_lane_wait_ms_max:.2f} "
                f"ready_set_size={len(self._scheduler_ready)} "
                f"queue_depth={queue_depth} "
                f"last_session={session.id} "
                f"last_session_lag_ms={lag_label}"
            )

    async def _scheduler_drain_ready_barrier_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        self._scheduler_ready.discard(session.id)
        if session.scheduler_inflight_generation is not None:
            logger.debug(
                f"Session {session.id}: scheduler barrier waiting for in-flight "
                f"gen={session.scheduler_inflight_generation} reason={reason}"
            )
        drained = 0
        while self._scheduler_session_ready(session):
            processed = await self._scheduler_process_one_ready_chunk_locked(
                session,
                reason=f"barrier:{reason}",
                requeue=False,
            )
            self._scheduler_ready.discard(session.id)
            if not processed:
                break
            drained += 1
        if drained:
            logger.debug(
                f"Session {session.id}: scheduler barrier drained {drained} "
                f"ready chunks before {reason}"
            )

    def _scheduler_invalidate_session_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> int:
        self._scheduler_ready.discard(session.id)
        session.scheduler_ready_since = None
        session.scheduler_generation += 1
        logger.debug(
            f"Session {session.id}: scheduler generation={session.scheduler_generation} "
            f"reason={reason}"
        )
        return session.scheduler_generation

    async def _scheduler_process_event(self, session: ASRSession, event: tuple) -> None:
        event_type = event[0]
        close_future = event[1] if event_type == "close" and len(event) > 1 else None
        try:
            async with session.state_lock:
                if event_type != "audio":
                    self._scheduler_ready.discard(session.id)
                    await self._scheduler_drain_ready_barrier_locked(
                        session,
                        reason=event_type,
                    )

                if event_type == "close":
                    await self._scheduler_continuous_handle_close_locked(session)
                elif event_type == "audio":
                    await self._scheduler_continuous_handle_audio_locked(session, event[1])
                elif event_type == "vad_start":
                    await self._scheduler_continuous_handle_vad_start_locked(session)
                elif event_type == "vad_stop":
                    await self._scheduler_continuous_handle_vad_stop_locked(session)
                elif event_type == "reset":
                    await self._scheduler_continuous_handle_reset_locked(
                        session,
                        finalize=event[1],
                        msg_type=event[2],
                    )
                elif event_type == "debounce_expired":
                    await self._scheduler_continuous_handle_debounce_expired_locked(
                        session,
                        stop_seq=event[1],
                    )
                else:
                    logger.warning(
                        f"Session {session.id}: unknown scheduler event {event_type}"
                    )
        except Exception as e:
            logger.error(f"Session {session.id} scheduler worker error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await session.websocket.send_str(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
            except Exception:
                pass
        finally:
            if close_future is not None and not close_future.done():
                close_future.set_result(True)

    async def _scheduler_continuous_handle_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
    ) -> None:
        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_post_stop_audio.extend(audio_bytes)
            if DEBUG_ASR:
                samples = len(session.continuous_post_stop_audio) // 2
                logger.debug(
                    f"Session {session.id}: held {len(audio_bytes)}B post-vad_stop "
                    f"audio ({samples} total samples) while pending finalize"
                )
            return

        if session.continuous_post_stop_audio:
            await self._scheduler_flush_post_stop_audio_locked(
                session,
                reason="streaming_resume",
            )

        await self._scheduler_append_audio_locked(session, audio_bytes)

    async def _scheduler_append_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
    ) -> None:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if DEBUG_ASR:
            chunk_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
            logger.debug(
                f"Session {session.id}: recv chunk {len(audio_bytes)}B hash={chunk_hash}"
            )

        self._capture_eou_snapshot_audio(session, audio_bytes)

        session.pending_audio = np.concatenate([session.pending_audio, audio_np])
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples += len(audio_np)
        session.scheduler_last_audio_monotonic = time.monotonic()
        self._scheduler_mark_ready_if_ready_locked(session)

    async def _scheduler_flush_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        audio_bytes = bytes(session.continuous_post_stop_audio)
        session.continuous_post_stop_audio.clear()
        samples = len(audio_bytes) // 2
        logger.debug(
            f"Session {session.id}: flushing {samples} post-vad_stop samples "
            f"for {reason}"
        )
        await self._scheduler_append_audio_locked(session, audio_bytes)

    def _scheduler_discard_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        samples = len(session.continuous_post_stop_audio) // 2
        session.continuous_post_stop_audio.clear()
        logger.debug(
            f"Session {session.id}: discarded {samples} post-vad_stop samples "
            f"at true boundary ({reason})"
        )

    async def _scheduler_continuous_force_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        include_post_stop_audio: bool,
    ) -> None:
        if reason not in _TRUE_BOUNDARY_FINALIZE_REASONS:
            raise RuntimeError(
                "continuous cold reset requested for non-boundary reason "
                f"{reason!r}"
            )

        await self._continuous_cancel_debounce_locked(session, invalidate=True)
        if include_post_stop_audio:
            await self._scheduler_flush_post_stop_audio_locked(session, reason=reason)
            await self._scheduler_drain_ready_barrier_locked(
                session,
                reason=f"{reason}:post_stop",
            )
        else:
            self._scheduler_discard_post_stop_audio_locked(session, reason=reason)

        if not self._continuous_has_audio_or_text(session):
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            session.continuous_vad_stop_ts = None
            session.continuous_debounce_expiry_ts = None
            logger.debug(
                f"Session {session.id}: ignored empty forced continuous finalize "
                f"for {reason}"
            )
            return

        session.continuous_state = FINALIZED
        if session.continuous_debounce_expiry_ts is None:
            session.continuous_debounce_expiry_ts = time.time()
        self._scheduler_invalidate_session_locked(session, reason=reason)
        logger.debug(f"Session {session.id}: forced continuous finalize for {reason}")
        await self._scheduler_continuous_finalize_and_reset_locked(
            session,
            reason=reason,
        )

    async def _scheduler_continuous_handle_close_locked(self, session: ASRSession) -> None:
        if (
            session.continuous_state == PENDING_FINALIZE
            or self._continuous_has_audio_or_text(session)
            or session.continuous_post_stop_audio
        ):
            await self._scheduler_continuous_force_finalize_locked(
                session,
                reason="close",
                include_post_stop_audio=True,
            )
        else:
            logger.debug(
                f"Session {session.id}: continuous close with no pending final"
            )
        session.scheduler_closed = True
        self._scheduler_ready.discard(session.id)
        self._scheduler_invalidate_session_locked(session, reason="close")

    async def _scheduler_continuous_handle_vad_start_locked(
        self,
        session: ASRSession,
    ) -> None:
        if session.continuous_state == PENDING_FINALIZE:
            await self._continuous_cancel_debounce_locked(session, invalidate=True)
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            session.continuous_vad_stop_ts = None
            session.continuous_debounce_expiry_ts = None
            logger.debug(
                f"Session {session.id}: vad_start canceled pending finalize; "
                "discarded speculative fork and continuing same ASR context"
            )
            await self._scheduler_flush_post_stop_audio_locked(
                session,
                reason="vad_start",
            )
        else:
            if session.continuous_post_stop_audio:
                await self._scheduler_flush_post_stop_audio_locked(
                    session,
                    reason="vad_start_after_speculative_finalize",
                )
            logger.debug(
                f"Session {session.id}: vad_start in state={session.continuous_state}"
            )

    async def _scheduler_continuous_handle_vad_stop_locked(
        self,
        session: ASRSession,
    ) -> None:
        self._scheduler_invalidate_session_locked(session, reason="vad_stop")
        await self._continuous_cancel_debounce_locked(session, invalidate=False)
        session.continuous_stop_seq += 1
        stop_seq = session.continuous_stop_seq
        session.continuous_state = PENDING_FINALIZE
        session.continuous_reset_seen = False
        session.continuous_vad_stop_ts = time.time()
        session.continuous_debounce_expiry_ts = None
        session.continuous_debounce_task = asyncio.create_task(
            self._continuous_debounce_timer(session.id, stop_seq),
            name=f"nemotron-continuous-debounce-{session.id}-{stop_seq}",
        )
        logger.debug(
            f"Session {session.id}: vad_stop armed pending finalize seq={stop_seq} "
            f"({self.finalize_silence_ms}ms)"
        )

    async def _scheduler_continuous_handle_reset_locked(
        self,
        session: ASRSession,
        *,
        finalize: bool,
        msg_type: str,
    ) -> None:
        if not finalize:
            text = session.current_text
            await self._send_json_locked(
                session,
                {
                    "type": "transcript",
                    "text": text,
                    "is_final": True,
                    "finalize": False,
                },
                tolerate_closed=False,
                description="continuous soft reset",
            )
            logger.debug(
                f"Session {session.id}: continuous soft reset: "
                f"'{text[-50:] if len(text) > 50 else text}'"
            )
            return

        if msg_type == "end":
            if session.continuous_state == PENDING_FINALIZE:
                session.continuous_reset_seen = True
            if (
                session.continuous_state == PENDING_FINALIZE
                or self._continuous_has_audio_or_text(session)
                or session.continuous_post_stop_audio
            ):
                await self._scheduler_continuous_force_finalize_locked(
                    session,
                    reason=msg_type,
                    include_post_stop_audio=True,
                )
                return

            logger.debug(
                f"Session {session.id}: ignored empty continuous {msg_type} in "
                f"state={session.continuous_state}"
            )
            return

        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_reset_seen = True
            logger.debug(
                f"Session {session.id}: delayed client {msg_type} while "
                "server debounce is pending"
            )
            return

        if self._continuous_has_audio_or_text(session):
            logger.debug(
                f"Session {session.id}: immediate continuous {msg_type} "
                "without pending VAD stop; speculative finalizing with "
                "context retained"
            )
            session.continuous_state = FINALIZED
            self._scheduler_invalidate_session_locked(session, reason=msg_type)
            await self._scheduler_continuous_finalize_emit_locked(
                session,
                reason=msg_type,
            )
            self._continuous_finish_speculative_finalize_locked(
                session,
                reason=msg_type,
            )
            return

        logger.debug(
            f"Session {session.id}: ignored empty continuous {msg_type} in "
            f"state={session.continuous_state}"
        )

    async def _scheduler_continuous_handle_debounce_expired_locked(
        self,
        session: ASRSession,
        *,
        stop_seq: int,
    ) -> None:
        if (
            session.continuous_state != PENDING_FINALIZE
            or stop_seq != session.continuous_stop_seq
        ):
            logger.debug(
                f"Session {session.id}: ignored stale debounce expiry "
                f"seq={stop_seq} current={session.continuous_stop_seq} "
                f"state={session.continuous_state}"
            )
            return

        session.continuous_debounce_task = None
        reset_seen = session.continuous_reset_seen
        session.continuous_reset_seen = False
        session.continuous_state = FINALIZED
        session.continuous_debounce_expiry_ts = time.time()
        logger.debug(
            f"Session {session.id}: debounce expired seq={stop_seq}; "
            f"finalizing (reset_seen={reset_seen})"
        )
        reason = "reset_then_debounce" if reset_seen else "debounce_expired"
        self._scheduler_invalidate_session_locked(session, reason=reason)
        await self._scheduler_continuous_finalize_emit_locked(session, reason=reason)
        self._continuous_finish_speculative_finalize_locked(
            session,
            reason=reason,
        )

    async def _scheduler_continuous_finalize_emit_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        await self._continuous_finalize_emit_locked(
            session,
            reason=reason,
            expected_generation=session.scheduler_generation,
        )

    async def _scheduler_continuous_finalize_and_reset_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        await self._scheduler_continuous_finalize_emit_locked(session, reason=reason)
        await self._continuous_cold_reset_after_finalize_locked(session, reason=reason)
        self._scheduler_invalidate_session_locked(
            session,
            reason=f"{reason}:cold_reset_complete",
        )

    async def _send_json_locked(
        self,
        session: ASRSession,
        payload: dict[str, Any],
        *,
        tolerate_closed: bool,
        description: str,
    ) -> bool:
        websocket = session.websocket
        if websocket is None or getattr(websocket, "closed", False):
            if tolerate_closed:
                logger.debug(
                    f"Session {session.id}: skipped {description} send; "
                    "websocket already closed"
                )
                return False
            raise ConnectionResetError(
                f"Session {session.id}: websocket closed before {description} send"
            )

        try:
            await websocket.send_str(json.dumps(payload))
            return True
        except (ClientConnectionResetError, ConnectionResetError) as e:
            if tolerate_closed:
                logger.debug(
                    f"Session {session.id}: skipped {description} send after "
                    f"connection close: {e}"
                )
                return False
            raise

    async def _continuous_session_worker(self, session: ASRSession) -> None:
        """Process continuous-mode events in arrival order."""
        queue = session.continuous_event_queue
        if queue is None:
            return

        while True:
            event = await queue.get()
            should_stop = event[0] == "close"
            try:
                event_type = event[0]

                async with session.state_lock:
                    if event_type == "close":
                        await self._continuous_handle_close_locked(session)
                    elif event_type == "audio":
                        await self._continuous_handle_audio_locked(session, event[1])
                    elif event_type == "vad_start":
                        await self._continuous_handle_vad_start_locked(session)
                    elif event_type == "vad_stop":
                        await self._continuous_handle_vad_stop_locked(session)
                    elif event_type == "reset":
                        await self._continuous_handle_reset_locked(
                            session,
                            finalize=event[1],
                            msg_type=event[2],
                        )
                    elif event_type == "debounce_expired":
                        await self._continuous_handle_debounce_expired_locked(
                            session,
                            stop_seq=event[1],
                        )
                    else:
                        logger.warning(
                            f"Session {session.id}: unknown continuous event {event_type}"
                        )
            except Exception as e:
                logger.error(f"Session {session.id} continuous worker error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                try:
                    await session.websocket.send_str(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                except Exception:
                    pass
            finally:
                queue.task_done()
                if should_stop:
                    return

    async def _close_continuous_session(self, session: ASRSession) -> None:
        """Drain pending finalization through the worker and stop continuous mode."""
        if self.scheduler_enabled:
            await self._close_scheduler_continuous_session(session)
            return

        queue = session.continuous_event_queue
        worker = session.continuous_worker_task
        if queue is not None and worker is not None and not worker.done():
            await queue.put(("close",))
            with contextlib.suppress(asyncio.CancelledError):
                await worker

        task = session.continuous_debounce_task
        session.continuous_debounce_task = None
        session.continuous_stop_seq += 1
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        session.continuous_event_queue = None
        session.continuous_worker_task = None
        session.continuous_post_stop_audio.clear()

    async def _close_scheduler_continuous_session(self, session: ASRSession) -> None:
        queue = session.continuous_event_queue
        if queue is not None:
            close_future = asyncio.get_running_loop().create_future()
            await queue.put(("close", close_future))
            self._wake_scheduler()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await close_future

        task = session.continuous_debounce_task
        session.continuous_debounce_task = None
        session.continuous_stop_seq += 1
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._scheduler_ready.discard(session.id)
        session.continuous_event_queue = None
        session.continuous_worker_task = None
        session.continuous_post_stop_audio.clear()
        session.scheduler_closed = True
        self._log_retained_cache_telemetry("scheduler_session_closed")

    async def _continuous_debounce_timer(self, session_id: str, stop_seq: int) -> None:
        """Wake after server-side silence and enqueue a finalize decision."""
        try:
            await asyncio.sleep(self.finalize_silence_seconds)
            session = self.sessions.get(session_id)
            if session is None or session.continuous_event_queue is None:
                return
            await session.continuous_event_queue.put(("debounce_expired", stop_seq))
            self._wake_scheduler()
        except asyncio.CancelledError:
            pass

    async def _continuous_cancel_debounce_locked(
        self,
        session: ASRSession,
        *,
        invalidate: bool,
    ) -> None:
        task = session.continuous_debounce_task
        session.continuous_debounce_task = None
        if invalidate:
            session.continuous_stop_seq += 1

        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _continuous_handle_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
    ) -> None:
        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_post_stop_audio.extend(audio_bytes)
            if DEBUG_ASR:
                samples = len(session.continuous_post_stop_audio) // 2
                logger.debug(
                    f"Session {session.id}: held {len(audio_bytes)}B post-vad_stop "
                    f"audio ({samples} total samples) while pending finalize"
                )
            return

        if session.continuous_post_stop_audio:
            await self._continuous_flush_post_stop_audio_locked(
                session,
                reason="streaming_resume",
            )

        await self._handle_audio_locked(
            session,
            audio_bytes,
            tolerate_closed_send=True,
        )

    async def _continuous_flush_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        audio_bytes = bytes(session.continuous_post_stop_audio)
        session.continuous_post_stop_audio.clear()
        samples = len(audio_bytes) // 2
        logger.debug(
            f"Session {session.id}: flushing {samples} post-vad_stop samples "
            f"for {reason}"
        )
        await self._handle_audio_locked(
            session,
            audio_bytes,
            tolerate_closed_send=(reason == "close"),
        )

    def _continuous_discard_post_stop_audio_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if not session.continuous_post_stop_audio:
            return

        samples = len(session.continuous_post_stop_audio) // 2
        session.continuous_post_stop_audio.clear()
        logger.debug(
            f"Session {session.id}: discarded {samples} post-vad_stop samples "
            f"at true boundary ({reason})"
        )

    async def _continuous_force_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        include_post_stop_audio: bool,
    ) -> None:
        if reason not in _TRUE_BOUNDARY_FINALIZE_REASONS:
            raise RuntimeError(
                "continuous cold reset requested for non-boundary reason "
                f"{reason!r}"
            )

        await self._continuous_cancel_debounce_locked(session, invalidate=True)
        if include_post_stop_audio:
            await self._continuous_flush_post_stop_audio_locked(session, reason=reason)
        else:
            self._continuous_discard_post_stop_audio_locked(session, reason=reason)

        if not self._continuous_has_audio_or_text(session):
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            session.continuous_vad_stop_ts = None
            session.continuous_debounce_expiry_ts = None
            logger.debug(
                f"Session {session.id}: ignored empty forced continuous finalize "
                f"for {reason}"
            )
            return

        session.continuous_state = FINALIZED
        if session.continuous_debounce_expiry_ts is None:
            session.continuous_debounce_expiry_ts = time.time()
        logger.debug(f"Session {session.id}: forced continuous finalize for {reason}")
        await self._continuous_finalize_and_reset_locked(session, reason=reason)

    async def _continuous_handle_close_locked(self, session: ASRSession) -> None:
        if (
            session.continuous_state == PENDING_FINALIZE
            or self._continuous_has_audio_or_text(session)
            or session.continuous_post_stop_audio
        ):
            await self._continuous_force_finalize_locked(
                session,
                reason="close",
                include_post_stop_audio=True,
            )
            return

        logger.debug(
            f"Session {session.id}: continuous close with no pending final"
        )

    async def _continuous_handle_vad_start_locked(self, session: ASRSession) -> None:
        if session.continuous_state == PENDING_FINALIZE:
            await self._continuous_cancel_debounce_locked(session, invalidate=True)
            session.continuous_state = STREAMING
            session.continuous_reset_seen = False
            session.continuous_vad_stop_ts = None
            session.continuous_debounce_expiry_ts = None
            logger.debug(
                f"Session {session.id}: vad_start canceled pending finalize; "
                "discarded speculative fork and continuing same ASR context"
            )
            await self._continuous_flush_post_stop_audio_locked(
                session,
                reason="vad_start",
            )
        else:
            if session.continuous_post_stop_audio:
                await self._continuous_flush_post_stop_audio_locked(
                    session,
                    reason="vad_start_after_speculative_finalize",
                )
            logger.debug(
                f"Session {session.id}: vad_start in state={session.continuous_state}"
            )

    async def _continuous_handle_vad_stop_locked(self, session: ASRSession) -> None:
        await self._continuous_cancel_debounce_locked(session, invalidate=False)
        session.continuous_stop_seq += 1
        stop_seq = session.continuous_stop_seq
        session.continuous_state = PENDING_FINALIZE
        session.continuous_reset_seen = False
        session.continuous_vad_stop_ts = time.time()
        session.continuous_debounce_expiry_ts = None
        session.continuous_debounce_task = asyncio.create_task(
            self._continuous_debounce_timer(session.id, stop_seq),
            name=f"nemotron-continuous-debounce-{session.id}-{stop_seq}",
        )
        logger.debug(
            f"Session {session.id}: vad_stop armed pending finalize seq={stop_seq} "
            f"({self.finalize_silence_ms}ms)"
        )

    async def _continuous_handle_reset_locked(
        self,
        session: ASRSession,
        *,
        finalize: bool,
        msg_type: str,
    ) -> None:
        if not finalize:
            text = session.current_text
            await session.websocket.send_str(json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": True,
                "finalize": False
            }))
            logger.debug(
                f"Session {session.id}: continuous soft reset: "
                f"'{text[-50:] if len(text) > 50 else text}'"
            )
            return

        if msg_type == "end":
            if session.continuous_state == PENDING_FINALIZE:
                session.continuous_reset_seen = True
            if (
                session.continuous_state == PENDING_FINALIZE
                or self._continuous_has_audio_or_text(session)
                or session.continuous_post_stop_audio
            ):
                await self._continuous_force_finalize_locked(
                    session,
                    reason=msg_type,
                    include_post_stop_audio=True,
                )
                return

            logger.debug(
                f"Session {session.id}: ignored empty continuous {msg_type} in "
                f"state={session.continuous_state}"
            )
            return

        if session.continuous_state == PENDING_FINALIZE:
            session.continuous_reset_seen = True
            logger.debug(
                f"Session {session.id}: delayed client {msg_type} while "
                "server debounce is pending"
            )
            return

        if self._continuous_has_audio_or_text(session):
            logger.debug(
                f"Session {session.id}: immediate continuous {msg_type} "
                "without pending VAD stop; speculative finalizing with "
                "context retained"
            )
            session.continuous_state = FINALIZED
            await self._continuous_finalize_emit_locked(session, reason=msg_type)
            self._continuous_finish_speculative_finalize_locked(
                session,
                reason=msg_type,
            )
            return

        logger.debug(
            f"Session {session.id}: ignored empty continuous {msg_type} in "
            f"state={session.continuous_state}"
        )

    async def _continuous_handle_debounce_expired_locked(
        self,
        session: ASRSession,
        *,
        stop_seq: int,
    ) -> None:
        if (
            session.continuous_state != PENDING_FINALIZE
            or stop_seq != session.continuous_stop_seq
        ):
            logger.debug(
                f"Session {session.id}: ignored stale debounce expiry "
                f"seq={stop_seq} current={session.continuous_stop_seq} "
                f"state={session.continuous_state}"
            )
            return

        session.continuous_debounce_task = None
        reset_seen = session.continuous_reset_seen
        session.continuous_reset_seen = False
        session.continuous_state = FINALIZED
        session.continuous_debounce_expiry_ts = time.time()
        logger.debug(
            f"Session {session.id}: debounce expired seq={stop_seq}; "
            f"finalizing (reset_seen={reset_seen})"
        )
        reason = "reset_then_debounce" if reset_seen else "debounce_expired"
        await self._continuous_finalize_emit_locked(session, reason=reason)
        self._continuous_finish_speculative_finalize_locked(
            session,
            reason=reason,
        )

    def _continuous_has_audio_or_text(self, session: ASRSession) -> bool:
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        return bool(session.current_text) or session.total_audio_samples > 0 or pending_len > 0

    def _build_continuous_finalize_fork(self, session: ASRSession) -> ASRSession:
        """Create a disposable fork for final padding without touching parent state."""
        pending_audio = (
            session.pending_audio.copy()
            if session.pending_audio is not None
            else np.array([], dtype=np.float32)
        )
        padding_samples = 0
        if session.total_audio_samples > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            pending_audio = np.concatenate([pending_audio, silence_padding])

        fork = ASRSession(
            id=f"{session.id}:fork",
            websocket=None,
            target_lang=session.target_lang,
        )
        fork.pending_audio = pending_audio
        fork.accumulated_audio = fork.pending_audio
        fork.total_audio_samples = session.total_audio_samples + padding_samples
        fork.synthetic_prefix_samples = session.synthetic_prefix_samples
        fork.raw_audio_ring = (
            session.raw_audio_ring.copy()
            if session.raw_audio_ring is not None
            else np.zeros(self.raw_audio_ring_samples, dtype=np.float32)
        )
        fork.mel_frame_ring = clone_tree(session.mel_frame_ring)
        fork.emitted_frames = session.emitted_frames
        fork.cache_last_channel = (
            tensor_clone(session.cache_last_channel)
            if session.cache_last_channel is not None
            else None
        )
        fork.cache_last_time = (
            tensor_clone(session.cache_last_time)
            if session.cache_last_time is not None
            else None
        )
        fork.cache_last_channel_len = (
            tensor_clone(session.cache_last_channel_len)
            if session.cache_last_channel_len is not None
            else None
        )
        fork.previous_hypotheses = clone_hypotheses_deep(session.previous_hypotheses)
        fork.pred_out_stream = clone_tree(session.pred_out_stream)
        fork.current_text = session.current_text
        fork.last_emitted_text = session.last_emitted_text
        fork.committed_text = session.committed_text
        fork.continuous_emitted_text = session.continuous_emitted_text
        return fork

    def _snapshot_fork_assert_parent(self, session: ASRSession) -> dict[str, Any]:
        return {
            "cache_last_channel": (
                tensor_clone(session.cache_last_channel)
                if session.cache_last_channel is not None
                else None
            ),
            "cache_last_time": (
                tensor_clone(session.cache_last_time)
                if session.cache_last_time is not None
                else None
            ),
            "cache_last_channel_len": (
                tensor_clone(session.cache_last_channel_len)
                if session.cache_last_channel_len is not None
                else None
            ),
            "previous_hypotheses": clone_hypotheses_deep(session.previous_hypotheses),
            "pred_out_stream": clone_tree(session.pred_out_stream),
        }

    def _assert_fork_flush_parent_unchanged(
        self,
        session: ASRSession,
        snapshot: dict[str, Any],
    ) -> None:
        try:
            _assert_tree_equal(
                "cache_last_channel",
                snapshot["cache_last_channel"],
                session.cache_last_channel,
            )
            _assert_tree_equal(
                "cache_last_time",
                snapshot["cache_last_time"],
                session.cache_last_time,
            )
            _assert_tree_equal(
                "cache_last_channel_len",
                snapshot["cache_last_channel_len"],
                session.cache_last_channel_len,
            )
            _assert_tree_equal(
                "previous_hypotheses",
                snapshot["previous_hypotheses"],
                session.previous_hypotheses,
            )
            _assert_tree_equal(
                "pred_out_stream",
                snapshot["pred_out_stream"],
                session.pred_out_stream,
            )
        except AssertionError as e:
            logger.error(
                f"Session {session.id}: fork alias assertion FAILED: {e}"
            )
            raise

        logger.info(
            f"Session {session.id}: fork alias assertion PASSED "
            "(parent cache tensors + previous_hypotheses + pred_out_stream byte-identical)"
        )

    def _continuous_context_retention_summary(self, session: ASRSession) -> str:
        raw_ring_samples = (
            len(session.raw_audio_ring) if session.raw_audio_ring is not None else 0
        )
        mel_ring_frames = (
            int(session.mel_frame_ring.shape[-1])
            if session.mel_frame_ring is not None
            else 0
        )
        pending_samples = (
            len(session.pending_audio) if session.pending_audio is not None else 0
        )
        return (
            f"cache_last_channel={'set' if session.cache_last_channel is not None else 'None'}, "
            f"cache_last_time={'set' if session.cache_last_time is not None else 'None'}, "
            f"cache_last_channel_len={'set' if session.cache_last_channel_len is not None else 'None'}, "
            f"previous_hypotheses={'set' if session.previous_hypotheses is not None else 'None'}, "
            f"pred_out_stream={'set' if session.pred_out_stream is not None else 'None'}, "
            f"raw_audio_ring_samples={raw_ring_samples}, "
            f"mel_frame_ring_frames={mel_ring_frames}, "
            f"current_text_chars={len(session.current_text)}, "
            f"continuous_emitted_chars={len(session.continuous_emitted_text)}, "
            f"emitted_frames={session.emitted_frames}, "
            f"pending_samples={pending_samples}, "
            f"total_audio_samples={session.total_audio_samples}"
        )

    async def _continuous_finalize_emit_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
        expected_generation: Optional[int] = None,
    ) -> None:
        """Finalize once on a disposable fork and emit one incremental delta."""
        if (
            expected_generation is not None
            and expected_generation != session.scheduler_generation
        ):
            logger.debug(
                f"Session {session.id}: skipped stale continuous finalize "
                f"reason={reason} expected_gen={expected_generation} "
                f"current_gen={session.scheduler_generation}"
            )
            return

        audio_samples = session.total_audio_samples
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        pending_len = len(session.pending_audio) if session.pending_audio is not None else 0
        held_len = len(session.continuous_post_stop_audio) // 2
        timing: dict[str, Any] = {
            "reason": reason,
            "vad_stop": session.continuous_vad_stop_ts,
            "debounce_expiry": session.continuous_debounce_expiry_ts,
            "fork_flush_start": None,
            "fork_flush_done": None,
            "final_sent": None,
            "inference_lock_acquire_wait_ms": None,
        }
        logger.debug(
            f"Session {session.id} continuous finalize ({reason}): "
            f"audio={audio_samples} samples ({audio_duration_ms}ms), "
            f"pending={pending_len} samples, held_post_stop={held_len} samples, "
            f"emitted={session.emitted_frames} frames"
        )

        final_text = session.current_text
        should_flush = (
            session.total_audio_samples > 0
            or (session.pending_audio is not None and len(session.pending_audio) > 0)
        )
        if should_flush:
            timing["fork_flush_start"] = time.time()
            start_perf = time.perf_counter()
            parent_snapshot = (
                self._snapshot_fork_assert_parent(session)
                if self.fork_assert_enabled
                else None
            )
            fork_clone_start = time.perf_counter()
            fork = self._build_continuous_finalize_fork(session)
            fork_clone_ms = (time.perf_counter() - fork_clone_start) * 1000
            if self.scheduler_enabled:
                logger.info(
                    f"Session {session.id}: scheduler_b1 fork_clone_ms="
                    f"{fork_clone_ms:.2f} reason={reason}"
                )
            if fork.pending_audio is not None and len(fork.pending_audio) > 0:
                lock_wait_start = time.perf_counter()
                async with self.inference_lock:
                    if (
                        expected_generation is not None
                        and expected_generation != session.scheduler_generation
                    ):
                        logger.debug(
                            f"Session {session.id}: skipped stale fork final chunk "
                            f"reason={reason} expected_gen={expected_generation} "
                            f"current_gen={session.scheduler_generation}"
                        )
                        return
                    timing["inference_lock_acquire_wait_ms"] = (
                        time.perf_counter() - lock_wait_start
                    ) * 1000
                    text = await self._run_inference_call(self._process_final_chunk, fork)
                    if text is not None:
                        final_text = text
            if parent_snapshot is not None:
                self._assert_fork_flush_parent_unchanged(session, parent_snapshot)
            timing["fork_flush_done"] = time.time()
            elapsed_ms = (time.perf_counter() - start_perf) * 1000
            logger.debug(
                f"Session {session.id} continuous fork final chunk processed in "
                f"{elapsed_ms:.1f}ms: "
                f"'{final_text[-50:] if len(final_text) > 50 else final_text}'"
            )

        if not final_text.startswith(session.committed_text):
            logger.debug(
                f"Session {session.id}: continuous ASR correction detected, "
                f"committed='{session.committed_text[-30:]}', "
                f"new='{final_text[-30:]}'"
            )
        delta_text = _continuous_append_only_delta(
            final_text,
            session.continuous_emitted_text,
        )

        if (
            expected_generation is not None
            and expected_generation != session.scheduler_generation
        ):
            logger.debug(
                f"Session {session.id}: suppressed stale continuous final "
                f"reason={reason} expected_gen={expected_generation} "
                f"current_gen={session.scheduler_generation}"
            )
            return

        session.committed_text = final_text
        session.last_emitted_text = final_text

        if delta_text:
            timing["final_sent"] = time.time()
            sent = await self._send_json_locked(
                session,
                {
                    "type": "transcript",
                    "text": delta_text,
                    "is_final": True,
                    "finalize": True,
                    "finalize_timing": timing,
                },
                tolerate_closed=(
                    reason == "close" or getattr(session.websocket, "closed", False)
                ),
                description="continuous final transcript",
            )
            if sent:
                session.continuous_emitted_text = (
                    session.continuous_emitted_text + " " + delta_text
                ).strip()
                logger.debug(
                    f"Session {session.id} continuous final: delta='{delta_text}' "
                    f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}', "
                    f"collector='{session.continuous_emitted_text[-50:]}')"
                )
        else:
            logger.debug(
                f"Session {session.id}: suppressed empty/duplicate continuous final "
                f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}', "
                f"collector='{session.continuous_emitted_text[-50:]}')"
            )

    def _continuous_finish_speculative_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        """Clear debounce bookkeeping after a speculative emit; keep ASR state."""
        held_len = len(session.continuous_post_stop_audio) // 2
        session.continuous_state = STREAMING
        session.continuous_vad_stop_ts = None
        session.continuous_debounce_expiry_ts = None
        session.continuous_debounce_task = None
        session.continuous_reset_seen = False

        logger.info(
            f"Session {session.id}: continuous speculative finalize complete "
            f"(context retained; no cold reset): reason={reason}, "
            f"committed_chars={len(session.committed_text)}, "
            f"retained_post_stop_samples={held_len}, "
            f"{self._continuous_context_retention_summary(session)}"
        )
        self._log_retained_cache_telemetry(f"speculative_finalize:{reason}")

    async def _continuous_cold_reset_after_finalize_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        if reason not in _TRUE_BOUNDARY_FINALIZE_REASONS:
            raise RuntimeError(
                "continuous cold reset requested for non-boundary reason "
                f"{reason!r}"
            )

        # True utterance boundary: now it is safe to cold-reset the ASR state.
        session.committed_text = ""
        session.last_emitted_text = ""
        session.continuous_emitted_text = ""
        session.overlap_buffer = None
        session.continuous_post_stop_audio.clear()
        session.continuous_reset_seen = False
        session.continuous_vad_stop_ts = None
        session.continuous_debounce_expiry_ts = None
        session.continuous_stop_seq += 1
        if self.session_warmup_ms > 0:
            async with self.inference_lock:
                await self._run_inference_call(self._init_session, session)
        else:
            self._init_session(session)
        session.continuous_state = STREAMING

        logger.info(
            f"Session {session.id}: continuous true-boundary cold reset complete "
            f"(reason={reason})"
        )
        self._log_retained_cache_telemetry(f"cold_reset:{reason}")

    async def _continuous_finalize_and_reset_locked(
        self,
        session: ASRSession,
        *,
        reason: str,
    ) -> None:
        """Finalize on the shared fork path, then cold-reset at a true boundary."""
        await self._continuous_finalize_emit_locked(session, reason=reason)
        await self._continuous_cold_reset_after_finalize_locked(session, reason=reason)

    async def _handle_audio(self, session: ASRSession, audio_bytes: bytes):
        """Accumulate audio and process when enough frames available."""
        await self._handle_audio_locked(session, audio_bytes)

    async def _handle_audio_locked(
        self,
        session: ASRSession,
        audio_bytes: bytes,
        *,
        tolerate_closed_send: bool = False,
    ):
        """Accumulate audio and process when enough frames available."""
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if DEBUG_ASR:
            chunk_hash = hashlib.md5(audio_bytes).hexdigest()[:8]
            logger.debug(f"Session {session.id}: recv chunk {len(audio_bytes)}B hash={chunk_hash}")

        self._capture_eou_snapshot_audio(session, audio_bytes)

        session.pending_audio = np.concatenate([session.pending_audio, audio_np])
        session.accumulated_audio = session.pending_audio
        session.total_audio_samples += len(audio_np)

        # Process if we have enough audio for new frames
        # We need shift_frames worth of new mel frames (after skipping edge frame)
        min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples

        while self._session_timeline_samples(session) >= min_audio_for_chunk:
            async with self.inference_lock:
                text = await self._run_inference_call(self._process_chunk, session)

            if text is not None and text != session.current_text:
                session.current_text = text
                logger.debug(f"Session {session.id} interim: {text[-50:] if len(text) > 50 else text}")
                await self._send_json_locked(
                    session,
                    {
                        "type": "transcript",
                        "text": text,
                        "is_final": False,
                    },
                    tolerate_closed=tolerate_closed_send,
                    description="interim transcript",
                )

            # Update minimum for next iteration
            min_audio_for_chunk = (session.emitted_frames + self.shift_frames + 1) * self.hop_samples

    def _prepare_scheduler_fixed_preprocess_audio(
        self,
        session: ASRSession,
    ) -> Optional[tuple[np.ndarray, int]]:
        if len(session.pending_audio) < self.preprocess_new_audio_samples:
            return None

        new_audio = session.pending_audio[: self.preprocess_new_audio_samples]
        return self._build_fixed_preprocess_audio(
            session.raw_audio_ring,
            new_audio,
        )

    def _preprocess_scheduler_fixed_audio_batch(
        self,
        fixed_audios: list[np.ndarray],
        valid_samples: list[int],
    ) -> Optional[list[torch.Tensor]]:
        if len(fixed_audios) < 2:
            return None
        if len(fixed_audios) != len(valid_samples):
            raise ValueError(
                f"fixed audio count {len(fixed_audios)} != length count {len(valid_samples)}"
            )

        audio_rows: list[np.ndarray] = []
        for audio, samples in zip(fixed_audios, valid_samples):
            if len(audio) != self.constant_preprocess_samples:
                return None
            if audio.dtype != np.float32:
                return None
            if samples < 0 or samples > self.constant_preprocess_samples:
                return None
            audio_rows.append(np.ascontiguousarray(audio))

        audio_batch = np.stack(audio_rows, axis=0)
        if audio_batch.shape != (len(audio_rows), self.constant_preprocess_samples):
            return None

        audio_tensor = torch.from_numpy(audio_batch).cuda()
        audio_len = torch.tensor(valid_samples, device='cuda', dtype=torch.long)
        mel, _mel_len = self.model.preprocessor(
            input_signal=audio_tensor,
            length=audio_len,
        )
        start = self.first_preprocess_mel_frame
        end = start + self.shift_frames
        return [
            mel[index : index + 1, :, start:end].detach().clone()
            for index in range(len(audio_rows))
        ]

    def _prepare_scheduler_batch_row(
        self,
        session: ASRSession,
        valid_new_mel: Optional[torch.Tensor] = None,
    ) -> Optional[SchedulerBatchRow]:
        if len(session.pending_audio) < self.preprocess_new_audio_samples:
            return None

        if valid_new_mel is None:
            fixed_input = self._prepare_scheduler_fixed_preprocess_audio(session)
            if fixed_input is None:
                return None
            fixed_audio, valid_samples = fixed_input
            mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
            start = self.first_preprocess_mel_frame
            valid_new_mel = mel[:, :, start : start + self.shift_frames]

        if session.emitted_frames == 0:
            chunk_mel = valid_new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
            drop_extra = self.drop_extra

        return SchedulerBatchRow(
            session=session,
            generation=session.scheduler_generation,
            chunk_mel=chunk_mel,
            valid_new_mel=valid_new_mel,
            drop_extra=int(drop_extra),
            eou_probe_snapshot=self._eou_probe_snapshot(session),
        )

    def _advance_session_after_normal_chunk(
        self,
        session: ASRSession,
        valid_new_mel: torch.Tensor,
    ) -> None:
        consumed_audio = session.pending_audio[: self.shift_frames * self.hop_samples]
        if len(consumed_audio) >= self.raw_audio_ring_samples:
            session.raw_audio_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
        else:
            keep = self.raw_audio_ring_samples - len(consumed_audio)
            session.raw_audio_ring = np.concatenate(
                [session.raw_audio_ring[-keep:], consumed_audio]
            ).astype(np.float32, copy=False)
        session.pending_audio = session.pending_audio[self.shift_frames * self.hop_samples :]
        session.accumulated_audio = session.pending_audio
        self._update_mel_frame_ring(session, valid_new_mel)
        session.emitted_frames += self.shift_frames

    @staticmethod
    def _scatter_batch_list_item(value: Any, index: int, batch_size: int) -> Any:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != batch_size:
                raise RuntimeError(
                    f"batched decoder returned {len(value)} rows for B={batch_size}"
                )
            return [value[index]]
        if batch_size == 1:
            return [value]
        raise RuntimeError(
            f"batched decoder returned non-list {type(value).__name__} for B={batch_size}"
        )

    def _process_ready_batch_solo_fallback(
        self,
        sessions: list[ASRSession],
        *,
        reason: str,
        error: Exception,
    ) -> dict[str, Optional[str]]:
        self._record_batch_fallback(reason)
        logger.warning(
            "scheduler_batch_fallback "
            f"reason={reason} sessions={','.join(session.id for session in sessions)} "
            f"error={type(error).__name__}: {error}"
        )
        texts: dict[str, Optional[str]] = {}
        for session in sessions:
            if session.scheduler_closed:
                texts[session.id] = None
                continue
            texts[session.id] = self._process_chunk(session)
        return texts

    def _log_scheduler_batch_memory(
        self,
        *,
        rows: list[SchedulerBatchRow],
        preprocessor_ms: float,
        model_ms: float,
        scatter_ms: float,
        mem_before: dict[str, int],
        mem_after: dict[str, int],
    ) -> None:
        batch_size = len(rows)
        if batch_size <= 0:
            return
        if (
            batch_size == 1
            and self._scheduler_batches % self.batch_memory_telemetry_every != 0
        ):
            return
        prompt_lang = rows[0].session.target_lang or self.target_lang
        logger.info(
            "scheduler_batch_memory "
            f"batch_size={batch_size} "
            f"prompt_lang={prompt_lang} "
            f"drop_extra={rows[0].drop_extra} "
            f"preprocessor_batch_ms={preprocessor_ms:.2f} "
            f"model_batch_ms={model_ms:.2f} "
            f"scatter_postprocess_ms={scatter_ms:.2f} "
            f"cuda_active_before_bytes={mem_before['active_bytes']} "
            f"cuda_active_after_bytes={mem_after['active_bytes']} "
            f"cuda_allocated_before_bytes={mem_before['allocated_bytes']} "
            f"cuda_allocated_after_bytes={mem_after['allocated_bytes']} "
            f"cuda_reserved_before_bytes={mem_before['reserved_bytes']} "
            f"cuda_reserved_after_bytes={mem_after['reserved_bytes']} "
            f"cuda_max_reserved_bytes={mem_after['max_reserved_bytes']} "
            f"retained_session_cache_bytes={mem_after['retained_session_cache_bytes']}"
        )

    def _process_ready_batch(self, sessions: list[ASRSession]) -> dict[str, Optional[str]]:
        """Process one scheduler batch of same-group ready normal chunks."""
        try:
            if not sessions:
                return {}

            with torch.inference_mode():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                mem_before = self._cuda_memory_snapshot()
                pre_start = time.perf_counter()
                preprocess_inputs: list[tuple[ASRSession, np.ndarray, int]] = []
                for session in sessions:
                    fixed_input = self._prepare_scheduler_fixed_preprocess_audio(session)
                    if fixed_input is None:
                        continue
                    fixed_audio, valid_samples = fixed_input
                    preprocess_inputs.append((session, fixed_audio, valid_samples))

                if not preprocess_inputs:
                    return {session.id: session.current_text for session in sessions}

                fixed_audios = [item[1] for item in preprocess_inputs]
                valid_samples = [item[2] for item in preprocess_inputs]
                batched_valid_new_mels = self._preprocess_scheduler_fixed_audio_batch(
                    fixed_audios,
                    valid_samples,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                preprocessor_ms = (time.perf_counter() - pre_start) * 1000.0

                rows: list[SchedulerBatchRow] = []
                if batched_valid_new_mels is None:
                    for session, _fixed_audio, _valid_samples in preprocess_inputs:
                        row = self._prepare_scheduler_batch_row(session)
                        if row is None:
                            continue
                        rows.append(row)
                else:
                    for index, (session, _fixed_audio, _valid_samples) in enumerate(
                        preprocess_inputs
                    ):
                        row = self._prepare_scheduler_batch_row(
                            session,
                            batched_valid_new_mels[index],
                        )
                        if row is None:
                            continue
                        rows.append(row)

                if not rows:
                    return {session.id: session.current_text for session in sessions}

                try:
                    drop_extras = {row.drop_extra for row in rows}
                    if len(drop_extras) != 1:
                        raise RuntimeError(
                            f"mixed drop_extra in scheduler batch: {sorted(drop_extras)}"
                        )

                    chunk_mels = [row.chunk_mel for row in rows]
                    processed_signal, processed_signal_length = stack_processed(chunk_mels)
                    cache_last_channel, cache_last_time, cache_last_channel_len = stack_caches(
                        [
                            (
                                row.session.cache_last_channel,
                                row.session.cache_last_time,
                                row.session.cache_last_channel_len,
                            )
                            for row in rows
                        ]
                    )
                    previous_hypotheses = [
                        clone_hypotheses_deep(row.session.previous_hypotheses)
                        for row in rows
                    ]
                    previous_pred_out = [
                        clone_tree(row.session.pred_out_stream)
                        for row in rows
                    ]
                    flat_hypotheses = stack_hypotheses(previous_hypotheses)
                    flat_pred_out = stack_pred_out(previous_pred_out, rnnt=True)
                except Exception as e:
                    if len(rows) > 1:
                        return self._process_ready_batch_solo_fallback(
                            [row.session for row in rows],
                            reason="unsafe_stack",
                            error=e,
                        )
                    raise

                if self.prompted_model:
                    self._apply_inference_prompt(rows[0].session)

                model_start = time.perf_counter()
                (
                    pred_out_stream,
                    transcribed_texts,
                    batch_cache_last_channel,
                    batch_cache_last_time,
                    batch_cache_last_channel_len,
                    batch_previous_hypotheses,
                ) = self._conformer_stream_step(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    keep_all_outputs=False,
                    previous_hypotheses=flat_hypotheses,
                    previous_pred_out=flat_pred_out,
                    drop_extra_pre_encoded=rows[0].drop_extra,
                    return_transcription=True,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model_ms = (time.perf_counter() - model_start) * 1000.0

                batch_size = len(rows)
                scatter_start = time.perf_counter()
                scattered: list[tuple[SchedulerBatchRow, Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, Optional[str]]] = []
                try:
                    for index, row in enumerate(rows):
                        row_cache = scatter_cache_row(
                            batch_cache_last_channel,
                            batch_cache_last_time,
                            batch_cache_last_channel_len,
                            index,
                        )
                        row_pred_out = self._scatter_batch_list_item(
                            pred_out_stream,
                            index,
                            batch_size,
                        )
                        row_hypotheses = self._scatter_batch_list_item(
                            batch_previous_hypotheses,
                            index,
                            batch_size,
                        )
                        text = row.session.current_text
                        if transcribed_texts and len(transcribed_texts) > index and transcribed_texts[index]:
                            text = self._extract_hypothesis_text(transcribed_texts[index])
                        scattered.append(
                            (
                                row,
                                row_pred_out,
                                row_hypotheses,
                                row_cache[0],
                                row_cache[1],
                                row_cache[2],
                                text,
                            )
                        )
                except Exception as e:
                    if len(rows) > 1:
                        return self._process_ready_batch_solo_fallback(
                            [row.session for row in rows],
                            reason="unsafe_scatter",
                            error=e,
                        )
                    raise

                texts: dict[str, Optional[str]] = {}
                for (
                    row,
                    row_pred_out,
                    row_hypotheses,
                    row_cache_last_channel,
                    row_cache_last_time,
                    row_cache_last_channel_len,
                    text,
                ) in scattered:
                    session = row.session
                    if row.generation != session.scheduler_generation or session.scheduler_closed:
                        continue
                    session.pred_out_stream = row_pred_out
                    session.previous_hypotheses = row_hypotheses
                    session.cache_last_channel = row_cache_last_channel
                    session.cache_last_time = row_cache_last_time
                    session.cache_last_channel_len = row_cache_last_channel_len
                    self._advance_session_after_normal_chunk(session, row.valid_new_mel)
                    self._write_eou_probe_chunk(session, row.eou_probe_snapshot)
                    self._write_eou_snapshot_chunk(session, row.eou_probe_snapshot)
                    texts[session.id] = text

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                scatter_ms = (time.perf_counter() - scatter_start) * 1000.0
                mem_after = self._cuda_memory_snapshot()
                self._log_scheduler_batch_memory(
                    rows=rows,
                    preprocessor_ms=preprocessor_ms,
                    model_ms=model_ms,
                    scatter_ms=scatter_ms,
                    mem_before=mem_before,
                    mem_after=mem_after,
                )
                return texts

        except Exception as e:
            session_ids = ",".join(session.id for session in sessions)
            oom = "out of memory" in str(e).lower()
            if oom and self.batch_max_size > 1:
                self.batch_max_size = 1
                self._record_batch_fallback("cuda_oom_clamped_to_B1")
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                logger.error(
                    "scheduler batch processing CUDA OOM; clamped batch_max_size=1 "
                    f"sessions={session_ids}: {e}"
                )
            else:
                logger.error(f"scheduler batch processing error sessions={session_ids}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {session.id: None for session in sessions}

    def _process_chunk(self, session: ASRSession) -> Optional[str]:
        """Process one fixed-plan audio window and run streaming inference."""
        try:
            if len(session.pending_audio) < self.preprocess_new_audio_samples:
                return session.current_text

            if DEBUG_ASR:
                audio_hash = _hash_audio(session.pending_audio)
                logger.debug(
                    f"Session {session.id}: process pending={len(session.pending_audio)} "
                    f"total={session.total_audio_samples} hash={audio_hash}"
                )

            new_audio = session.pending_audio[: self.preprocess_new_audio_samples]
            fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
                session.raw_audio_ring,
                new_audio,
            )

            with torch.inference_mode():
                if self.profile_chunk:
                    torch.cuda.synchronize()
                    _prof_t0 = time.perf_counter()
                mel, mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
                if self.profile_chunk:
                    torch.cuda.synchronize()
                    self._prof_pre_ms += (time.perf_counter() - _prof_t0) * 1000.0

                if DEBUG_ASR:
                    mel_hash = hashlib.md5(mel.cpu().numpy().tobytes()).hexdigest()[:8]
                    logger.debug(f"Session {session.id}: mel shape={mel.shape[-1]} hash={mel_hash}")

                valid_new_mel = mel[
                    :,
                    :,
                    self.first_preprocess_mel_frame : self.first_preprocess_mel_frame + self.shift_frames,
                ]

                # Extract chunk with pre-encode cache
                if session.emitted_frames == 0:
                    # First chunk: just shift_frames, no mel cache
                    chunk_mel = valid_new_mel
                    drop_extra = 0
                else:
                    # Subsequent chunks: prepend retained mel pre-encode cache
                    chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
                    drop_extra = self.drop_extra

                chunk_len = torch.tensor([chunk_mel.shape[-1]], device='cuda')
                eou_probe_snapshot = self._eou_probe_snapshot(session)

                # Run streaming inference
                if self.prompted_model:
                    self._apply_inference_prompt(session)
                if self.profile_chunk:
                    torch.cuda.synchronize()
                    _prof_t1 = time.perf_counter()
                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self._conformer_stream_step(
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
                if self.profile_chunk:
                    torch.cuda.synchronize()
                    self._prof_step_ms += (time.perf_counter() - _prof_t1) * 1000.0
                    self._prof_n += 1
                    if self._prof_n % 25 == 0:
                        n = self._prof_n
                        logger.info(
                            f"[PROFILE] chunks={n} "
                            f"preprocess={self._prof_pre_ms / n:.2f}ms/chunk "
                            f"step(enc+dec)={self._prof_step_ms / n:.2f}ms/chunk "
                            f"total={(self._prof_pre_ms + self._prof_step_ms) / n:.2f}ms/chunk"
                        )

                # Update emitted frame count
                consumed_audio = session.pending_audio[: self.shift_frames * self.hop_samples]
                if len(consumed_audio) >= self.raw_audio_ring_samples:
                    session.raw_audio_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
                else:
                    keep = self.raw_audio_ring_samples - len(consumed_audio)
                    session.raw_audio_ring = np.concatenate(
                        [session.raw_audio_ring[-keep:], consumed_audio]
                    ).astype(np.float32, copy=False)
                session.pending_audio = session.pending_audio[self.shift_frames * self.hop_samples :]
                session.accumulated_audio = session.pending_audio
                self._update_mel_frame_ring(session, valid_new_mel)
                session.emitted_frames += self.shift_frames
                self._write_eou_probe_chunk(session, eou_probe_snapshot)
                self._write_eou_snapshot_chunk(session, eou_probe_snapshot)

                # Extract text
                if transcribed_texts and transcribed_texts[0]:
                    return self._extract_hypothesis_text(transcribed_texts[0])

                return session.current_text

        except Exception as e:
            logger.error(f"Session {session.id} chunk processing error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def _reset_session(self, session: ASRSession, finalize: bool = True):
        """Handle reset with soft or hard finalization.

        Args:
            finalize: If True (hard reset), add padding and use keep_all_outputs=True
                      to capture trailing words, then reset decoder state.
                      If False (soft reset), just return current cumulative text
                      without forcing decoder output.

        Soft reset (finalize=False):
        - Returns current_text as is_final (model's streaming output)
        - No audio processing, no decoder finalization
        - Decoder state preserved (no corruption)
        - Used on VADUserStoppedSpeakingFrame for fast response

        Hard reset (finalize=True):
        - Adds padding and processes with keep_all_outputs=True
        - Captures trailing words at segment boundaries
        - Resets decoder state to prevent corruption from multiple hard resets
        - Preserves encoder cache for acoustic context
        - Used on UserStoppedSpeakingFrame for complete transcription
        """
        import time

        # Log audio state at reset for diagnostics
        audio_samples = session.total_audio_samples
        audio_duration_ms = (audio_samples * 1000) // self.sample_rate
        logger.debug(
            f"Session {session.id} {'hard' if finalize else 'soft'} reset: "
            f"audio={audio_samples} samples ({audio_duration_ms}ms), "
            f"pending={len(session.pending_audio)} samples, "
            f"emitted={session.emitted_frames} frames"
        )

        if not finalize:
            # SOFT RESET: Return current text without processing
            # This is fast (~0ms) and doesn't corrupt decoder state.
            # The model's current_text is already cumulative (contains all text
            # from session start), so we just return it directly.
            # We don't concatenate with cumulative_text to avoid duplication.
            text = session.current_text

            await session.websocket.send_str(json.dumps({
                "type": "transcript",
                "text": text,
                "is_final": True,
                "finalize": False  # Tell client this was soft reset
            }))

            logger.debug(f"Session {session.id} soft reset: '{text[-50:] if len(text) > 50 else text}'")
            # Keep all state intact - decoder, encoder, audio buffer
            return

        # HARD RESET: Full finalization with padding
        # Save original audio length before adding padding
        original_audio_length = session.total_audio_samples

        # Pad with silence to ensure the model has enough trailing context
        # to finalize the last word. Padding = (right_context + 1) * shift_frames.
        if original_audio_length > 0:
            padding_samples = self.final_padding_frames * self.hop_samples
            silence_padding = np.zeros(padding_samples, dtype=np.float32)
            session.pending_audio = np.concatenate([session.pending_audio, silence_padding])
            session.accumulated_audio = session.pending_audio

        # Process all remaining audio with keep_all_outputs=True
        final_text = session.current_text
        if session.pending_audio is not None and len(session.pending_audio) > 0:
            start_time = time.perf_counter()
            async with self.inference_lock:
                text = await self._run_inference_call(self._process_final_chunk, session)
                if text is not None:
                    final_text = text
                    session.current_text = text  # Update current_text for next soft reset
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Session {session.id} final chunk processed in {elapsed_ms:.1f}ms: '{final_text[-50:] if len(final_text) > 50 else final_text}'")

        # Server-side deduplication: only send the delta (new portion)
        # This avoids downstream duplication when aggregators concatenate transcripts
        if final_text.startswith(session.last_emitted_text):
            delta_text = final_text[len(session.last_emitted_text):].lstrip()
        else:
            # ASR corrected earlier text - send full text
            # (This is rare but can happen with model corrections)
            delta_text = final_text
            logger.debug(
                f"Session {session.id}: ASR correction detected, "
                f"last='{session.last_emitted_text[-30:]}', new='{final_text[-30:]}'"
            )

        # Update tracking state before sending
        session.last_emitted_text = final_text

        # Send only the delta to client
        await session.websocket.send_str(json.dumps({
            "type": "transcript",
            "text": delta_text,
            "is_final": True,
            "finalize": True  # Tell client this was hard reset
        }))

        logger.debug(
            f"Session {session.id} hard reset: delta='{delta_text}' "
            f"(cumulative='{final_text[-50:] if len(final_text) > 50 else final_text}')"
        )

        # MEMORY BOUNDING: Clear all state after hard reset
        # This prevents unbounded memory growth by resetting completely each turn:
        # - Audio buffer: cleared (no carryover between turns)
        # - Decoder state: reset fresh (no hypothesis accumulation)
        # - Encoder cache: re-initialized
        #
        # We considered keeping audio overlap for encoder context continuity,
        # but since we reset the encoder cache, overlap audio would just be
        # re-transcribed, causing duplicates. Clean reset avoids this.

        session.last_emitted_text = ""
        session.overlap_buffer = None
        if self.session_warmup_ms > 0:
            async with self.inference_lock:
                await self._run_inference_call(self._init_session, session)
        else:
            self._init_session(session)

        logger.debug(
            f"Session {session.id} hard reset complete, state fully reset for next turn"
        )
        self._log_retained_cache_telemetry("hard_reset")

    def _process_final_chunk(self, session: ASRSession) -> Optional[str]:
        """Process remaining pending audio with fixed-plan preprocessing."""
        try:
            if len(session.pending_audio) == 0:
                return session.current_text

            with torch.inference_mode():
                # For final chunk, use ALL remaining frames (including edge)
                padded_total_samples = (
                    session.emitted_frames * self.hop_samples + len(session.pending_audio)
                )
                total_mel_frames = (padded_total_samples // self.hop_samples) + 1
                remaining_frames = total_mel_frames - session.emitted_frames

                logger.debug(
                    f"Session {session.id} final chunk: "
                    f"total_mel={total_mel_frames}, emitted={session.emitted_frames}, "
                    f"remaining={remaining_frames}"
                )

                if remaining_frames <= 0:
                    logger.warning(f"Session {session.id}: No remaining frames to process!")
                    return session.current_text

                pending = session.pending_audio
                raw_ring = session.raw_audio_ring
                new_mels: list[torch.Tensor] = []
                frames_collected = 0
                while frames_collected < remaining_frames:
                    frames_this_call = min(self.shift_frames, remaining_frames - frames_collected)
                    needed_new_samples = min(
                        len(pending),
                        self.preprocess_new_audio_samples,
                    )
                    new_audio = pending[:needed_new_samples]
                    fixed_audio, valid_samples = self._build_fixed_preprocess_audio(
                        raw_ring,
                        new_audio,
                    )
                    mel, _mel_len = self._preprocess_fixed_audio(fixed_audio, valid_samples)
                    start = self.first_preprocess_mel_frame
                    new_mels.append(mel[:, :, start : start + frames_this_call])

                    if frames_this_call == self.shift_frames:
                        consumed_samples = min(self.shift_frames * self.hop_samples, len(pending))
                        consumed_audio = pending[:consumed_samples]
                        if len(consumed_audio) >= self.raw_audio_ring_samples:
                            raw_ring = consumed_audio[-self.raw_audio_ring_samples :].copy()
                        elif len(consumed_audio) > 0:
                            keep = self.raw_audio_ring_samples - len(consumed_audio)
                            raw_ring = np.concatenate([raw_ring[-keep:], consumed_audio]).astype(
                                np.float32,
                                copy=False,
                            )
                        pending = pending[consumed_samples:]
                    frames_collected += frames_this_call

                new_mel = torch.cat(new_mels, dim=-1)

                # Extract final chunk with pre-encode cache
                if session.emitted_frames == 0:
                    chunk_mel = new_mel
                    drop_extra = 0
                else:
                    chunk_mel = torch.cat((session.mel_frame_ring, new_mel), dim=-1)
                    drop_extra = self.drop_extra

                chunk_len = torch.tensor([chunk_mel.shape[-1]], device='cuda')

                if self.prompted_model:
                    self._apply_inference_prompt(session)
                (
                    session.pred_out_stream,
                    transcribed_texts,
                    session.cache_last_channel,
                    session.cache_last_time,
                    session.cache_last_channel_len,
                    session.previous_hypotheses,
                ) = self._conformer_stream_step(
                    processed_signal=chunk_mel,
                    processed_signal_length=chunk_len,
                    cache_last_channel=session.cache_last_channel,
                    cache_last_time=session.cache_last_time,
                    cache_last_channel_len=session.cache_last_channel_len,
                    keep_all_outputs=True,  # Final chunk - output all remaining
                    previous_hypotheses=session.previous_hypotheses,
                    previous_pred_out=session.pred_out_stream,
                    drop_extra_pre_encoded=drop_extra,
                    return_transcription=True,
                )

                if transcribed_texts and transcribed_texts[0]:
                    final_text = self._extract_hypothesis_text(transcribed_texts[0])
                    logger.debug(
                        f"Session {session.id} final chunk output: '{final_text[-50:] if len(final_text) > 50 else final_text}' "
                        f"(was: '{session.current_text[-30:] if len(session.current_text) > 30 else session.current_text}')"
                    )
                    return final_text

                logger.debug(f"Session {session.id} final chunk: no new text from model")
                return session.current_text

        except Exception as e:
            logger.error(f"Session {session.id} final chunk error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    async def health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy" if self.model_loaded else "loading",
            "model_loaded": self.model_loaded,
        })

    async def start(self):
        """Start the HTTP + WebSocket server."""
        self.load_model()
        self.model_loaded = True
        if self.scheduler_enabled:
            self._ensure_scheduler_task()

        logger.info(f"Starting streaming ASR server on ws://{self.host}:{self.port}")

        app = web.Application()
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/", self.websocket_handler)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"ASR server listening on ws://{self.host}:{self.port}")
        logger.info(f"Health check available at http://{self.host}:{self.port}/health")
        await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(description="Nemotron Streaming ASR WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model name or path to local .nemo file"
    )
    parser.add_argument(
        "--right-context",
        type=int,
        default=None,
        choices=[0, 1, 3, 6, 13],
        help=(
            "Right context frames. Omit for model default "
            "(English=1, prompted multilingual=3)."
        )
    )
    args = parser.parse_args()

    server = ASRServer(
        model=args.model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
    )

    asyncio.run(server.start())


if __name__ == "__main__":
    main()
