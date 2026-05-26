#!/usr/bin/env python3
"""Export a deterministic MEL-level continuous-session replay bundle.

The C++ session harness consumes this bundle to replay the verified
``finalize_ref.ContinuousFinalizeRef`` state machine without needing NeMo or an
audio frontend in the C++ process.  Each utterance contains:

* ordered steady ``new_mel`` chunks plus geometry flags;
* the single finalize remainder ``chunk_mel`` plus ``drop_extra`` and ``T``;
* gold cumulative token ids from ``ContinuousFinalizeRef.debounce_expire``.

Run from runtime/:
  HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python export_session_bundle.py --n 20
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from finalize_ref import (
    BLANK,
    MAX_SYMBOLS,
    RIGHT_CONTEXT,
    ContinuousFinalizeRef,
    ContinuousSession,
    load_benchmark_dataset,
    load_model,
    load_wav,
)


ART = os.path.join(os.path.dirname(__file__), "artifacts")


def _as_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value).__name__}")
    return value.detach().cpu().clone()


def _scalar(value: int | bool) -> torch.Tensor:
    return torch.tensor([int(value)], dtype=torch.int64)


def _decoder_state_hc(state: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(state, (tuple, list)) and len(state) == 2:
        h, c = state
        if torch.is_tensor(h) and torch.is_tensor(c):
            return h, c
    raise TypeError(f"unsupported decoder_state shape for export: {type(state).__name__}")


class RecordingContinuousFinalizeRef(ContinuousFinalizeRef):
    """Reference runtime with non-invasive steady chunk capture."""

    def begin_recording(self) -> None:
        self.recorded_steady_chunks: list[dict[str, Any]] = []

    @torch.inference_mode()
    def _process_one_steady_chunk(self, session: ContinuousSession) -> None:
        g = self.geometry
        new_audio = session.pending_audio[: g.preprocess_new_audio_samples]
        mel = self._fixed_mel_from_new_audio(session.raw_audio_ring, new_audio)
        valid_new_mel = mel[
            :,
            :,
            g.first_preprocess_mel_frame : g.first_preprocess_mel_frame
            + g.shift_frames,
        ]

        is_first = session.emitted_frames == 0
        chunk_T = (
            int(valid_new_mel.shape[-1])
            if is_first
            else int(session.mel_frame_ring.shape[-1] + valid_new_mel.shape[-1])
        )
        self.recorded_steady_chunks.append(
            {
                "new_mel": _as_cpu_tensor(valid_new_mel),
                "is_first": is_first,
                "drop_extra": 0 if is_first else int(g.drop_extra),
                "chunk_T": chunk_T,
                "emitted_before": int(session.emitted_frames),
            }
        )

        super()._process_one_steady_chunk(session)


def _build_row(
    rt: RecordingContinuousFinalizeRef,
    wav,
    sample_index: int,
) -> dict[str, Any]:
    session = rt.new_session(f"session-bundle-{sample_index}")
    rt.begin_recording()
    rt.append_audio(session, wav)
    steady_chunks = list(rt.recorded_steady_chunks)

    rt.vad_stop(session)
    fork = rt.build_continuous_finalize_fork(session)
    inputs = rt.prepare_finalize_inputs(fork)

    if inputs is None:
        mel_dim = (
            int(steady_chunks[0]["new_mel"].shape[1])
            if steady_chunks
            else 128
        )
        final_chunk_mel = torch.empty((1, mel_dim, 0), dtype=torch.float32)
        final_new_mel = torch.empty((1, mel_dim, 0), dtype=torch.float32)
        final_drop_extra = -1
        final_T = 0
        remaining_frames = 0
        padded_total_samples = int(session.emitted_frames * rt.geometry.hop_samples)
    else:
        final_chunk_mel = _as_cpu_tensor(inputs.chunk_mel)
        final_new_mel = _as_cpu_tensor(inputs.new_mel)
        final_drop_extra = int(inputs.drop_extra)
        final_T = int(inputs.chunk_mel.shape[-1])
        remaining_frames = int(inputs.remaining_frames)
        padded_total_samples = int(inputs.padded_total_samples)

    result = rt.debounce_expire(session)
    if not result.fork_assert_passed:
        raise AssertionError(f"finalize_ref FORK_ASSERT failed for sample {sample_index}")

    steady_tokens = torch.tensor(result.steady_tokens, dtype=torch.int64)
    gold_tokens = torch.tensor(result.final_tokens, dtype=torch.int64)
    finalize_new_tokens = torch.tensor(
        result.final_tokens[len(result.steady_tokens) :],
        dtype=torch.int64,
    )

    return {
        "sample_index": int(sample_index),
        "audio_samples": int(len(wav)),
        "steady_chunks": steady_chunks,
        "steady_tokens": steady_tokens,
        "gold_tokens": gold_tokens,
        "finalize_new_tokens": finalize_new_tokens,
        "final_chunk_mel": final_chunk_mel,
        "final_new_mel": final_new_mel,
        "final_drop_extra": final_drop_extra,
        "final_T": final_T,
        "final_remaining_frames": remaining_frames,
        "final_padded_total_samples": padded_total_samples,
        "finalize_ref_meta": dict(result.meta),
    }


class SessionBundle(torch.nn.Module):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        init_session: ContinuousSession,
        geometry,
    ):
        super().__init__()
        init_h, init_c = _decoder_state_hc(init_session.decoder_state)
        self.register_buffer("num_utts", torch.tensor([len(rows)], dtype=torch.int64))
        self.register_buffer(
            "meta",
            torch.tensor(
                [
                    len(rows),
                    BLANK,
                    MAX_SYMBOLS,
                    int(geometry.shift_frames),
                    int(geometry.pre_encode_cache_size),
                    int(geometry.drop_extra),
                    int(geometry.final_padding_frames),
                    RIGHT_CONTEXT,
                    int(geometry.first_preprocess_mel_frame),
                    int(geometry.hop_samples),
                ],
                dtype=torch.int64,
            ),
        )
        self.register_buffer("init_clc", _as_cpu_tensor(init_session.cache_last_channel))
        self.register_buffer("init_clt", _as_cpu_tensor(init_session.cache_last_time))
        self.register_buffer("init_clcl", _as_cpu_tensor(init_session.cache_last_channel_len))
        self.register_buffer("init_g", _as_cpu_tensor(init_session.pred_out_stream))
        self.register_buffer("init_h", _as_cpu_tensor(init_h))
        self.register_buffer("init_c", _as_cpu_tensor(init_c))

        for i, row in enumerate(rows):
            prefix = f"utt{i}"
            steady_chunks = row["steady_chunks"]
            self.register_buffer(f"{prefix}_sample_index", _scalar(row["sample_index"]))
            self.register_buffer(f"{prefix}_audio_samples", _scalar(row["audio_samples"]))
            self.register_buffer(f"{prefix}_num_steady", _scalar(len(steady_chunks)))
            self.register_buffer(f"{prefix}_steady_tokens", row["steady_tokens"].cpu().to(torch.int64))
            self.register_buffer(f"{prefix}_gold_tokens", row["gold_tokens"].cpu().to(torch.int64))
            self.register_buffer(
                f"{prefix}_finalize_new_tokens",
                row["finalize_new_tokens"].cpu().to(torch.int64),
            )
            self.register_buffer(f"{prefix}_final_chunk_mel", row["final_chunk_mel"].cpu())
            self.register_buffer(f"{prefix}_final_new_mel", row["final_new_mel"].cpu())
            self.register_buffer(f"{prefix}_final_drop_extra", _scalar(row["final_drop_extra"]))
            self.register_buffer(f"{prefix}_final_T", _scalar(row["final_T"]))
            self.register_buffer(
                f"{prefix}_final_remaining_frames",
                _scalar(row["final_remaining_frames"]),
            )
            self.register_buffer(
                f"{prefix}_final_padded_total_samples",
                _scalar(row["final_padded_total_samples"]),
            )

            for j, chunk in enumerate(steady_chunks):
                cprefix = f"{prefix}_chunk{j}"
                self.register_buffer(f"{cprefix}_new_mel", chunk["new_mel"].cpu())
                self.register_buffer(f"{cprefix}_is_first", _scalar(chunk["is_first"]))
                self.register_buffer(f"{cprefix}_drop_extra", _scalar(chunk["drop_extra"]))
                self.register_buffer(f"{cprefix}_chunk_T", _scalar(chunk["chunk_T"]))
                self.register_buffer(
                    f"{cprefix}_emitted_before",
                    _scalar(chunk["emitted_before"]),
                )

    def forward(self):
        return self.num_utts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of stt-benchmark utterances")
    parser.add_argument("--start", type=int, default=0, help="dataset start index")
    parser.add_argument("--out", default=os.path.join(ART, "session_bundle.ts"))
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = load_model()
    rt = RecordingContinuousFinalizeRef(model)
    dataset = load_benchmark_dataset()

    rows: list[dict[str, Any]] = []
    for offset in range(args.n):
        sample_index = args.start + offset
        wav = load_wav(dataset[sample_index])
        row = _build_row(rt, wav, sample_index)
        rows.append(row)
        print(
            f"row{offset} sample={sample_index} audio={row['audio_samples']} "
            f"steady_chunks={len(row['steady_chunks'])} "
            f"steady_tok={row['steady_tokens'].numel()} gold_tok={row['gold_tokens'].numel()} "
            f"final_drop={row['final_drop_extra']} final_T={row['final_T']}"
        )

    init_session = rt.new_session("session-bundle-init")
    bundle = torch.jit.script(SessionBundle(rows, init_session, rt.geometry))
    bundle.save(args.out)
    print(f"wrote {args.out} ({len(rows)} utterances)")
    print(
        "schema: meta, init_*; utt{i}_{num_steady,steady_tokens,gold_tokens,"
        "final_chunk_mel,final_drop_extra,final_T}; utt{i}_chunk{j}_*"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
