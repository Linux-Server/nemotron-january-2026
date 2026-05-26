#!/usr/bin/env python3
"""Export a deterministic MEL-level continuous-session replay bundle.

The C++ session harness consumes this bundle to replay the verified
``finalize_ref.ContinuousFinalizeRef`` state machine without needing NeMo or an
audio frontend in the C++ process.  Each utterance contains:

* ordered steady ``new_mel`` chunks plus geometry flags;
* the single finalize remainder ``chunk_mel`` plus ``drop_extra`` and ``T``;
* gold cumulative token ids from ``ContinuousFinalizeRef.debounce_expire``.
* gold emitted-event stream from the finalize_ref/server WORD/TEXT semantics.
  Text is packed as UTF-8 bytes and C++ compares it directly.  Interim events
  are generated from the same AOTI steady encoder used by the C++ session, so
  this gate checks event logic rather than eager-vs-AOTI timing drift.
* the tokenizer id->piece table plus Python ``ids_to_text`` self-test sequences
  so C++ can prove its detokenizer matches the reference at load.

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
    _continuous_append_only_delta,
    load_benchmark_dataset,
    load_model,
    load_wav,
)
from ref_decode import ref_greedy_range


ART = os.path.join(os.path.dirname(__file__), "artifacts")
EVENT_INTERIM = 0
EVENT_FINAL = 1
EVENT_SUPPRESSED = 2


def _as_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"expected tensor, got {type(value).__name__}")
    return value.detach().cpu().clone()


def _scalar(value: int | bool) -> torch.Tensor:
    return torch.tensor([int(value)], dtype=torch.int64)


def _pack_i64_lists(values: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    flat: list[int] = []
    for item in values:
        flat.extend(int(v) for v in item)
        offsets.append(len(flat))
    return (
        torch.tensor(flat, dtype=torch.int64),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _pack_utf8(strings: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    flat = bytearray()
    for text in strings:
        encoded = text.encode("utf-8")
        flat.extend(encoded)
        offsets.append(len(flat))
    return (
        torch.tensor(list(flat), dtype=torch.uint8),
        torch.tensor(offsets, dtype=torch.int64),
    )


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    for attr in ("original_vocab_size", "vocab_size"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    sp = getattr(tokenizer, "tokenizer", None)
    if sp is not None and hasattr(sp, "get_piece_size"):
        return int(sp.get_piece_size())
    raise TypeError("cannot determine tokenizer vocabulary size")


def _tokenizer_pieces(tokenizer: Any) -> list[str]:
    vocab = _tokenizer_vocab_size(tokenizer)
    return [str(tokenizer.ids_to_tokens([idx])[0]) for idx in range(vocab)]


def _build_detok_selftest(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[list[list[int]], list[str]]:
    vocab = _tokenizer_vocab_size(tokenizer)
    sequences: list[list[int]] = [[]]
    sequences.extend([[idx] for idx in range(vocab)])
    for row in rows:
        sequences.append(row["steady_tokens"].cpu().to(torch.int64).tolist())
        sequences.append(row["gold_tokens"].cpu().to(torch.int64).tolist())
        for event in row["events"]:
            sequences.append(list(event["tokens"]))
            sequences.append(list(event["collector_tokens"]))

    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for seq in sequences:
        key = tuple(int(v) for v in seq)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(key))

    texts = [tokenizer.ids_to_text(seq) if seq else "" for seq in unique]
    return unique, texts


def _append_only_delta_tokens(final_tokens: list[int], emitted_tokens: list[int]) -> list[int]:
    """Legacy token payload helper; text fields are the authoritative oracle."""
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
        delta_tokens = final_tokens[len(emitted_tokens) :]
        max_overlap = min(len(emitted_tokens), len(delta_tokens))
        for overlap in range(max_overlap, 0, -1):
            if emitted_tokens[-overlap:] == delta_tokens[:overlap]:
                delta_tokens = delta_tokens[overlap:]
                break

    return list(delta_tokens)


def _decoder_state_hc(state: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(state, (tuple, list)) and len(state) == 2:
        h, c = state
        if torch.is_tensor(h) and torch.is_tensor(c):
            return h, c
    raise TypeError(f"unsupported decoder_state shape for export: {type(state).__name__}")


class RecordingContinuousFinalizeRef(ContinuousFinalizeRef):
    """Reference runtime with non-invasive steady chunk capture."""

    def __init__(self, model, *, artifacts_dir: str = ART):
        super().__init__(model)
        self.enc_first = torch.jit.load(os.path.join(artifacts_dir, "enc_first.ts")).to(self.device)
        self.enc_first.eval()
        self.enc_steady_aoti = torch._inductor.aoti_load_package(
            os.path.join(artifacts_dir, "enc_steady_aoti.pt2")
        )

    def begin_recording(self) -> None:
        self.recorded_steady_chunks: list[dict[str, Any]] = []
        self.recorded_events: list[dict[str, Any]] = []
        self.recording_continuous_emitted_tokens: list[int] = []

    def _record_event(
        self,
        *,
        kind: int,
        text: str,
        tokens: list[int],
        collector_text: str,
        collector_tokens: list[int],
    ) -> None:
        self.recorded_events.append(
            {
                "kind": int(kind),
                "text": text,
                "tokens": list(tokens),
                "collector_text": collector_text,
                "collector_tokens": list(collector_tokens),
            }
        )

    @torch.inference_mode()
    def _run_aoti_consistent_steady_encoder(
        self,
        session: ContinuousSession,
        chunk_mel: torch.Tensor,
        drop_extra: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_len = torch.tensor([chunk_mel.shape[-1]], device=self.device)
        if drop_extra == 0:
            out = self.enc_first(
                chunk_mel.contiguous(),
                chunk_len.contiguous(),
                session.cache_last_channel.contiguous(),
                session.cache_last_time.contiguous(),
                session.cache_last_channel_len.contiguous(),
            )
            return tuple(out)  # type: ignore[return-value]
        out = self.enc_steady_aoti(
            chunk_mel.contiguous(),
            chunk_len.contiguous(),
            session.cache_last_channel.contiguous(),
            session.cache_last_time.contiguous(),
            session.cache_last_channel_len.contiguous(),
        )
        return tuple(out)  # type: ignore[return-value]

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

        old_text = session.current_text
        if is_first:
            chunk_mel = valid_new_mel
            drop_extra = 0
        else:
            chunk_mel = torch.cat((session.mel_frame_ring, valid_new_mel), dim=-1)
            drop_extra = g.drop_extra

        enc_out, enc_len, clc, clt, clcl = self._run_aoti_consistent_steady_encoder(
            session,
            chunk_mel,
            drop_extra,
        )
        tokens, decoder_state, pred_out = ref_greedy_range(
            self.decoder,
            self.joint,
            enc_out.transpose(1, 2).contiguous(),
            0,
            int(enc_len[0]),
            session.decoder_state,
            session.pred_out_stream,
        )
        session.hyp_tokens.extend(tokens)
        session.decoder_state = decoder_state
        session.pred_out_stream = pred_out
        session.cache_last_channel = clc
        session.cache_last_time = clt
        session.cache_last_channel_len = clcl

        consumed_audio = session.pending_audio[: g.shift_frames * g.hop_samples]
        session.raw_audio_ring = self._advance_raw_ring(
            session.raw_audio_ring,
            consumed_audio,
        )
        session.pending_audio = session.pending_audio[g.shift_frames * g.hop_samples :]
        self._update_mel_frame_ring(session, valid_new_mel)
        session.emitted_frames += g.shift_frames
        session.current_text = self.tokenizer.ids_to_text(session.hyp_tokens)

        text_changed = session.current_text != old_text
        if text_changed:
            self._record_event(
                kind=EVENT_INTERIM,
                text=session.current_text,
                tokens=list(session.hyp_tokens),
                collector_text=session.continuous_emitted_text,
                collector_tokens=list(self.recording_continuous_emitted_tokens),
            )

    def _finalize_and_emit(
        self,
        session: ContinuousSession,
        *,
        reason: str,
    ):
        emitted_before = list(self.recording_continuous_emitted_tokens)
        emitted_text_before = session.continuous_emitted_text
        result = super()._finalize_and_emit(session, reason=reason)
        token_delta = _append_only_delta_tokens(result.final_tokens, emitted_before)
        text_delta = _continuous_append_only_delta(result.final_text, emitted_text_before)
        if text_delta != result.delta_text:
            raise AssertionError(
                "finalize_ref delta oracle mismatch during export: "
                f"computed={text_delta!r} result={result.delta_text!r}"
            )
        if result.delta_text:
            collector_tokens = emitted_before + token_delta
            kind = EVENT_FINAL
            text = result.delta_text
        else:
            collector_tokens = emitted_before
            kind = EVENT_SUPPRESSED
            text = ""
        self._record_event(
            kind=kind,
            text=text,
            tokens=token_delta,
            collector_text=session.continuous_emitted_text,
            collector_tokens=collector_tokens,
        )
        self.recording_continuous_emitted_tokens = list(collector_tokens)
        return result


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
    events = list(rt.recorded_events)
    if not events:
        raise AssertionError(f"no emitted events recorded for sample {sample_index}")
    if events[-1]["kind"] not in (EVENT_FINAL, EVENT_SUPPRESSED):
        raise AssertionError(
            f"sample {sample_index} event stream did not end in final/suppressed"
        )

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
        "events": events,
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
        tokenizer_pieces: list[str],
        detok_sequences: list[list[int]],
        detok_texts: list[str],
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
        piece_bytes, piece_offsets = _pack_utf8(tokenizer_pieces)
        self.register_buffer("token_piece_bytes", piece_bytes)
        self.register_buffer("token_piece_offsets", piece_offsets)
        detok_tokens, detok_token_offsets = _pack_i64_lists(detok_sequences)
        detok_text_bytes, detok_text_offsets = _pack_utf8(detok_texts)
        self.register_buffer("detok_selftest_tokens", detok_tokens)
        self.register_buffer("detok_selftest_token_offsets", detok_token_offsets)
        self.register_buffer("detok_selftest_text_bytes", detok_text_bytes)
        self.register_buffer("detok_selftest_text_offsets", detok_text_offsets)

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
            events = row["events"]
            self.register_buffer(
                f"{prefix}_event_kinds",
                torch.tensor([event["kind"] for event in events], dtype=torch.int64),
            )
            event_tokens, event_token_offsets = _pack_i64_lists(
                [event["tokens"] for event in events]
            )
            collector_tokens, collector_token_offsets = _pack_i64_lists(
                [event["collector_tokens"] for event in events]
            )
            event_text_bytes, event_text_offsets = _pack_utf8(
                [event["text"] for event in events]
            )
            collector_text_bytes, collector_text_offsets = _pack_utf8(
                [event["collector_text"] for event in events]
            )
            self.register_buffer(f"{prefix}_event_tokens", event_tokens)
            self.register_buffer(f"{prefix}_event_token_offsets", event_token_offsets)
            self.register_buffer(f"{prefix}_event_collector_tokens", collector_tokens)
            self.register_buffer(
                f"{prefix}_event_collector_token_offsets",
                collector_token_offsets,
            )
            self.register_buffer(f"{prefix}_event_text_bytes", event_text_bytes)
            self.register_buffer(f"{prefix}_event_text_offsets", event_text_offsets)
            self.register_buffer(
                f"{prefix}_event_collector_text_bytes",
                collector_text_bytes,
            )
            self.register_buffer(
                f"{prefix}_event_collector_text_offsets",
                collector_text_offsets,
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
            f"events={len(row['events'])} "
            f"final_drop={row['final_drop_extra']} final_T={row['final_T']}"
        )

    init_session = rt.new_session("session-bundle-init")
    detok_sequences, detok_texts = _build_detok_selftest(rows, rt.tokenizer)
    bundle = torch.jit.script(
        SessionBundle(
            rows,
            init_session,
            rt.geometry,
            _tokenizer_pieces(rt.tokenizer),
            detok_sequences,
            detok_texts,
        )
    )
    bundle.save(args.out)
    print(
        f"wrote {args.out} ({len(rows)} utterances, "
        f"detok_selftests={len(detok_sequences)})"
    )
    print(
        "schema: meta, init_*; utt{i}_{num_steady,steady_tokens,gold_tokens,"
        "event_kinds,event_tokens,event_token_offsets,event_collector_tokens,"
        "event_collector_token_offsets,event_text_bytes,event_text_offsets,"
        "event_collector_text_bytes,event_collector_text_offsets,"
        "final_chunk_mel,final_drop_extra,final_T}; tokenizer token_piece_* "
        "and detok_selftest_*; utt{i}_chunk{j}_*"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
