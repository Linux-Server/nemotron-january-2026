"""Step 8 gate: variable-B churn, retained-cache memory, and fail-closed guards.

Run with:
  /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python \
    proj-2026-05-21-0410/test_batch_memory_churn.py
"""

from __future__ import annotations

import contextlib
import asyncio
import dataclasses
import gc
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Iterator, Optional

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nemotron_speech.server import ASRServer, ASRSession, _continuous_append_only_delta  # noqa: E402


DB = REPO / "stt-benchmark/stt_benchmark_data/results.db"
SR = 16000
BATCHES = (1, 2, 4, 8, 16)
INITIAL_IDS = [
    "b980f45a-7289-f63f-0923-2fe102deb8c2",
    "f1069e64-44c8-9fd5-d54e-f03567582f63",
    "b35e06ae-28f5-8498-8d1e-244dc02d1671",
    "8facde28-4202-7470-0814-fa34f0ca84a5",
    "fbe7161c-ac49-4909-1608-3a32179341f3",
    "937c1bf8-4857-71dc-c05f-dfe0a21b65ca",
    "ca46319c-e790-a8f6-4793-7ab542a6d95d",
    "da66feba-9052-77a3-3334-8cc99533b59a",
    "80644538-fa3c-4f76-3f90-eff8ce0695d2",
    "934c2ffb-d39d-2163-5fd3-4348360314ef",
    "02ac751f-65cf-aa0e-1797-3f240fab9759",
    "9565dd38-8116-54a7-e667-144c1a09d504",
    "95c52928-0ca8-f732-e0f6-76ca2a96af35",
    "0ff9664b-2d12-7d3d-eb35-10a4b57061ee",
    "209eda2a-35fa-868a-1ac6-d74fa208bbcf",
    "052ba5bd-48a3-2c25-883d-e2f25a953d7e",
]
JOIN_IDS = [
    "f2df60f7-c3ae-3603-910b-1092bd05aed5",
    "06c41bf6-3d0e-be8a-37da-e8e76c13a25c",
]
LEAVER_LABELS = {"churn_00", "churn_01"}
LEAVE_AFTER_NORMAL_CHUNKS = 4
JOIN_TICK = 5


@dataclasses.dataclass
class Capture:
    interims: list[str] = dataclasses.field(default_factory=list)
    final: str = ""
    delta: str = ""
    normal_chunks: int = 0


@dataclasses.dataclass
class ManagedStream:
    label: str
    sample_id: str
    session: ASRSession
    capture: Capture
    leave_after_chunks: Optional[int] = None


@dataclasses.dataclass
class MemoryRecord:
    batch_size: int
    active_before: int
    active_after: int
    reserved_before: int
    reserved_after: int
    max_reserved: int
    retained_after_step: int
    retained_after_remove: int


def log(message: str) -> None:
    print(message, flush=True)


@contextlib.contextmanager
def patched_env(updates: dict[str, str], remove: tuple[str, ...] = ()) -> Iterator[None]:
    old: dict[str, Optional[str]] = {}
    for key in set(updates) | set(remove):
        old[key] = os.environ.get(key)
    try:
        for key in remove:
            os.environ.pop(key, None)
        os.environ.update(updates)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def model_path() -> str:
    env_path = os.environ.get("NEMOTRON_TEST_MODEL")
    if env_path:
        return env_path
    return Path("/tmp/en-nemo-path").read_text().strip()


def enforce_env() -> None:
    os.environ["NEMOTRON_CONTINUOUS"] = "1"
    os.environ["NEMOTRON_SCHEDULER_B1"] = "1"
    os.environ["NEMOTRON_BATCH_SCHED"] = "1"
    os.environ["NEMOTRON_BATCH_MAX_SIZE"] = "16"
    os.environ["NEMOTRON_BATCH_MAX_WAIT_MS"] = "5"
    os.environ["NEMOTRON_FINALIZE_SILENCE_MS"] = "0"
    os.environ["NEMOTRON_WARMUP_MS"] = "200"
    os.environ["NEMOTRON_FORK_ASSERT"] = "1"
    os.environ.pop("NEMOTRON_EOU_PROBE", None)
    os.environ.pop("NEMOTRON_DECODING", None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_clip(sample_id: str) -> np.ndarray:
    con = sqlite3.connect(DB)
    try:
        row = con.execute(
            "SELECT audio_path FROM samples WHERE sample_id=?",
            (sample_id,),
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise RuntimeError(f"sample_id not found in {DB}: {sample_id}")
    path = REPO / "stt-benchmark" / row[0]
    pcm = path.read_bytes()
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def append_audio(server: ASRServer, session: ASRSession, audio: np.ndarray) -> None:
    pending = session.pending_audio
    if pending is None:
        pending = np.array([], dtype=np.float32)
    session.pending_audio = np.concatenate([pending, audio]).astype(np.float32, copy=False)
    session.accumulated_audio = session.pending_audio
    session.total_audio_samples += len(audio)
    session.scheduler_last_audio_monotonic = time.monotonic()
    session.scheduler_ready_since = time.monotonic()


def make_session(server: ASRServer, label: str, sample_id: str, audio: np.ndarray) -> ASRSession:
    session = ASRSession(id=label, websocket=None, target_lang=server.target_lang)
    server._init_session(session)
    session.continuous_event_queue = asyncio.Queue()
    append_audio(server, session, audio)
    server.sessions[label] = session
    return session


def cleanup_session(server: ASRServer, session: ASRSession) -> None:
    session.scheduler_closed = True
    server.sessions.pop(session.id, None)


def update_capture(
    server: ASRServer,
    managed: ManagedStream,
    text: Optional[str],
) -> None:
    session = managed.session
    if text is None:
        return
    if text != session.current_text:
        session.current_text = text
        if not managed.capture.interims or managed.capture.interims[-1] != text:
            managed.capture.interims.append(text)


def process_group(server: ASRServer, group: list[ManagedStream]) -> None:
    key = server._scheduler_batch_group_key_for_session(group[0].session)
    assert all(server._scheduler_batch_group_key_for_session(item.session) == key for item in group)
    processed: set[str] = set()
    before_frames = {item.label: item.session.emitted_frames for item in group}
    texts = server._process_ready_batch([item.session for item in group])
    for item in group:
        assert item.label not in processed, f"{item.label} processed twice in one tick"
        processed.add(item.label)
        after = item.session.emitted_frames
        expected = before_frames[item.label] + server.shift_frames
        assert after == expected, f"{item.label}: fairness shift mismatch {after} != {expected}"
        item.capture.normal_chunks += 1
        update_capture(server, item, texts.get(item.session.id))


def finalize_stream(server: ASRServer, managed: ManagedStream) -> None:
    session = managed.session
    if session.total_audio_samples > 0:
        padding_samples = server.final_padding_frames * server.hop_samples
        silence = np.zeros(padding_samples, dtype=np.float32)
        session.pending_audio = np.concatenate([session.pending_audio, silence])
        session.accumulated_audio = session.pending_audio
    text = server._process_final_chunk(session)
    if text is not None:
        session.current_text = text
    managed.capture.final = session.current_text
    managed.capture.delta = _continuous_append_only_delta(session.current_text, "")


def run_reference(
    server: ASRServer,
    label: str,
    sample_id: str,
    audio: np.ndarray,
    *,
    leave_after_chunks: Optional[int] = None,
) -> Capture:
    session = make_session(server, f"solo_{label}", sample_id, audio)
    managed = ManagedStream(label=label, sample_id=sample_id, session=session, capture=Capture())
    while server._scheduler_session_ready(session):
        process_group(server, [managed])
        if leave_after_chunks is not None and managed.capture.normal_chunks >= leave_after_chunks:
            break
    if leave_after_chunks is not None:
        assert managed.capture.normal_chunks == leave_after_chunks
    finalize_stream(server, managed)
    cleanup_session(server, session)
    del session
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return managed.capture


def run_memory_sweep(server: ASRServer, clips: dict[str, np.ndarray]) -> dict[int, MemoryRecord]:
    records: dict[int, MemoryRecord] = {}
    ids = list(clips)
    for batch_size in BATCHES:
        sessions = [
            make_session(server, f"mem_B{batch_size}_{idx}", ids[idx], clips[ids[idx]])
            for idx in range(batch_size)
        ]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        before = server._cuda_memory_snapshot()
        texts = server._process_ready_batch(sessions)
        for session in sessions:
            text = texts.get(session.id)
            if text is not None:
                session.current_text = text
        torch.cuda.synchronize()
        after = server._cuda_memory_snapshot()
        retained_after_step = server._retained_session_cache_bytes()
        for session in sessions:
            cleanup_session(server, session)
        del sessions
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        retained_after_remove = server._retained_session_cache_bytes()
        record = MemoryRecord(
            batch_size=batch_size,
            active_before=before["active_bytes"],
            active_after=after["active_bytes"],
            reserved_before=before["reserved_bytes"],
            reserved_after=after["reserved_bytes"],
            max_reserved=int(torch.cuda.max_memory_reserved()),
            retained_after_step=retained_after_step,
            retained_after_remove=retained_after_remove,
        )
        records[batch_size] = record
        log(
            "MEMORY "
            f"B={batch_size} active_before={record.active_before} active_after={record.active_after} "
            f"reserved_before={record.reserved_before} reserved_after={record.reserved_after} "
            f"max_reserved={record.max_reserved} retained_after_step={record.retained_after_step} "
            f"retained_after_remove={record.retained_after_remove}"
        )
        assert retained_after_remove == 0, f"B={batch_size}: retained cache after remove leaked"
    return records


def permute_ready(tick: int, ready: list[ManagedStream]) -> list[ManagedStream]:
    ready = list(ready)
    if tick % 2:
        ready.reverse()
    if ready and tick % 3 == 2:
        ready = ready[1:] + ready[:1]
    return ready


def run_churn(
    server: ASRServer,
    clips: dict[str, np.ndarray],
    references: dict[str, Capture],
) -> tuple[dict[str, Capture], list[tuple[str, int, int]], int]:
    active: dict[str, ManagedStream] = {}
    completed: dict[str, Capture] = {}
    retained_events: list[tuple[str, int, int]] = []
    backlogged_requeues = 0

    def add(label: str, sample_id: str, leave_after: Optional[int] = None) -> None:
        session = make_session(server, label, sample_id, clips[sample_id])
        active[label] = ManagedStream(
            label=label,
            sample_id=sample_id,
            session=session,
            capture=Capture(),
            leave_after_chunks=leave_after,
        )

    for idx, sample_id in enumerate(INITIAL_IDS):
        label = f"churn_{idx:02d}"
        leave_after = LEAVE_AFTER_NORMAL_CHUNKS if label in LEAVER_LABELS else None
        add(label, sample_id, leave_after)

    tick = 0
    joined = False
    while active:
        if tick == JOIN_TICK and not joined:
            for idx, sample_id in enumerate(JOIN_IDS):
                add(f"join_{idx:02d}", sample_id)
            joined = True

        ready = [
            item
            for item in active.values()
            if server._scheduler_session_ready(item.session)
        ]
        ready = permute_ready(tick, ready)
        groups: dict[tuple, list[ManagedStream]] = {}
        for item in ready:
            groups.setdefault(server._scheduler_batch_group_key_for_session(item.session), []).append(item)

        for group in groups.values():
            for start in range(0, len(group), server.batch_max_size):
                sub_group = group[start : start + server.batch_max_size]
                process_group(server, sub_group)
                for item in sub_group:
                    if server._scheduler_session_ready(item.session):
                        backlogged_requeues += 1

        for label, item in list(active.items()):
            should_leave = (
                item.leave_after_chunks is not None
                and item.capture.normal_chunks >= item.leave_after_chunks
            )
            normal_done = (
                item.leave_after_chunks is None
                and not server._scheduler_session_ready(item.session)
            )
            if not should_leave and not normal_done:
                continue
            finalize_stream(server, item)
            retained_before = server._retained_session_cache_bytes()
            completed[label] = item.capture
            cleanup_session(server, item.session)
            active.pop(label)
            del item
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            retained_after = server._retained_session_cache_bytes()
            retained_events.append((label, retained_before, retained_after))
            assert retained_after <= retained_before, (
                f"{label}: retained cache grew after remove/reset "
                f"{retained_before} -> {retained_after}"
            )

        tick += 1
        if tick > 200:
            raise RuntimeError("churn loop did not drain")

    assert joined, "fresh streams did not join mid-run"
    assert backlogged_requeues > 0, "fairness gate did not observe ready requeues"
    assert server._retained_session_cache_bytes() == 0
    for label, expected in references.items():
        actual = completed[label]
        assert actual.interims == expected.interims, f"{label}: interim sequence mismatch"
        assert actual.final == expected.final, f"{label}: final mismatch"
        assert actual.delta == expected.delta, f"{label}: final delta mismatch"
    return completed, retained_events, backlogged_requeues


def test_fail_closed_guards() -> None:
    base = {
        "NEMOTRON_CONTINUOUS": "1",
        "NEMOTRON_SCHEDULER_B1": "1",
        "NEMOTRON_BATCH_SCHED": "1",
        "NEMOTRON_BATCH_MAX_SIZE": "16",
    }
    with patched_env({**base, "NEMOTRON_DECODING": "beam"}):
        try:
            ASRServer(model=model_path())
            raise AssertionError("beam+batch should refuse cleanly")
        except ValueError as exc:
            assert "NEMOTRON_DECODING=greedy" in str(exc)

    with patched_env({**base, "NEMOTRON_EOU_PROBE": "1"}, remove=("NEMOTRON_DECODING",)):
        server = ASRServer(model=model_path())
        assert not server.batch_enabled
        assert server.batch_max_size == 1
        assert server.batch_fallback_reason == "eou_probe_preserve_alignments_unprobed"

    with patched_env(base, remove=("NEMOTRON_DECODING", "NEMOTRON_EOU_PROBE")):
        server = ASRServer(model=model_path())

        class FakeCTCModel:
            cfg = {"aux_ctc": {"enabled": True}}
            joint = object()
            decoder = object()
            ctc_decoder = object()

            def conformer_stream_step(self) -> None:
                pass

        server.model = FakeCTCModel()
        ok, reason = server._batch_model_rnnt_pure_status()
        assert not ok and "ctc" in reason
        server._disable_batching(reason)
        assert not server.batch_enabled
        assert server.batch_max_size == 1

        calls: list[str] = []

        def fake_process_chunk(session: ASRSession) -> str:
            calls.append(session.id)
            return f"solo:{session.id}"

        server._process_chunk = fake_process_chunk  # type: ignore[method-assign]
        sessions = [
            ASRSession(id="unsafe_a", websocket=None),
            ASRSession(id="unsafe_b", websocket=None),
        ]
        texts = server._process_ready_batch_solo_fallback(
            sessions,
            reason="unsafe_stack",
            error=RuntimeError("mixed state"),
        )
        assert calls == ["unsafe_a", "unsafe_b"]
        assert texts == {"unsafe_a": "solo:unsafe_a", "unsafe_b": "solo:unsafe_b"}
        assert server._scheduler_batch_fallback_counts["unsafe_stack"] >= 1

    log("FAIL_CLOSED guards: beam refuses; EOU/CTC/unsafe fallback to B=1")


def test_batch_memory_churn() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Step 8 memory/churn gate")
    enforce_env()
    test_fail_closed_guards()

    clips = {sample_id: load_clip(sample_id) for sample_id in [*INITIAL_IDS, *JOIN_IDS]}
    server = ASRServer(model=model_path(), host="127.0.0.1", port=0, right_context=1)
    server.load_model()
    try:
        assert server.batch_enabled, f"batch disabled unexpectedly: {server.batch_fallback_reason}"
        assert server.batch_max_size >= 16, f"startup cap below B=16: {server.batch_max_size}"
        log(f"STARTUP_CAP effective_max={server.batch_max_size}")
        records = run_memory_sweep(server, clips)

        references: dict[str, Capture] = {}
        for idx, sample_id in enumerate(INITIAL_IDS):
            label = f"churn_{idx:02d}"
            leave_after = LEAVE_AFTER_NORMAL_CHUNKS if label in LEAVER_LABELS else None
            references[label] = run_reference(
                server,
                label,
                sample_id,
                clips[sample_id],
                leave_after_chunks=leave_after,
            )
        for idx, sample_id in enumerate(JOIN_IDS):
            label = f"join_{idx:02d}"
            references[label] = run_reference(server, label, sample_id, clips[sample_id])

        completed, retained_events, backlogged_requeues = run_churn(server, clips, references)
        retained_after_churn = server._retained_session_cache_bytes()
        max_reserved_by_b = {
            batch_size: record.max_reserved for batch_size, record in records.items()
        }
        log(f"MEMORY_MAX_RESERVED_BY_B {max_reserved_by_b}")
        log(f"RETAINED_AFTER_CHURN {retained_after_churn}")
        log(f"CHURN_RETAINED_EVENTS {retained_events}")
        log(
            "CHURN_BYTE_EXACT "
            f"streams={len(completed)} backlogged_requeues={backlogged_requeues} "
            "interims_final_delta=True"
        )
    finally:
        server.sessions.clear()
        del server
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    test_batch_memory_churn()
    log("STEP8 MEMORY/CHURN GATE PASS")
