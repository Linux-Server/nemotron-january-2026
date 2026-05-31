#!/usr/bin/env python3
"""Read-only Step 1(b) probe: AOTI first-chunk parity vs shipped TorchScript first chunk.

Only chunk 0 differs between the two arms:
  baseline  = artifacts/enc_first.ts
  candidate = artifacts/enc_first_aoti.pt2 bound to finalize_shared_weights.pt

All non-first chunks use the same eager encoder path in both arms.

Run from runtime/:
  HF_HUB_OFFLINE=1 ./.venv/bin/python enc_first_parity_probe.py --n 200
"""
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from ref_decode import ref_greedy_range


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
ART = ROOT / "artifacts"
DEFAULT_BENCH_ROOT = REPO_ROOT / "stt-benchmark"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    dataset_index: int
    duration_seconds: float
    reference: str
    audio_path: Path | None = None
    audio_bytes: bytes | None = None


@dataclass
class StreamResult:
    tokens: list[int]
    text: str
    events: list[tuple[int, tuple[int, ...]]]
    first_output_diffs: dict[str, float]


def resolve_shared(weights: dict[str, torch.Tensor], fqn: str) -> str | None:
    if fqn in weights:
        return fqn
    if fqn.startswith("encoder."):
        alt = "e." + fqn[len("encoder.") :]
        if alt in weights:
            return alt
    elif fqn.startswith("e."):
        alt = "encoder." + fqn[len("e.") :]
        if alt in weights:
            return alt
    return None


def stream_cfg_int(value: Any) -> int:
    return int(value[1]) if isinstance(value, (list, tuple)) else int(value)


def read_pcm16(path: Path) -> tuple[np.ndarray, int]:
    raw = np.fromfile(path, dtype="<i2")
    return (raw.astype(np.float32) / 32768.0).copy(), 16000


def read_sample_audio(sample: Sample) -> np.ndarray:
    if sample.audio_bytes is not None:
        wav, sr = sf.read(io.BytesIO(sample.audio_bytes), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
    elif sample.audio_path is not None:
        wav, sr = read_pcm16(sample.audio_path)
    else:
        raise ValueError(f"sample {sample.sample_id} has no audio payload")
    if sr != 16000:
        n = int(len(wav) * 16000 / sr)
        wav = np.interp(
            np.linspace(0, len(wav), n, endpoint=False),
            np.arange(len(wav)),
            wav,
        ).astype(np.float32)
    return wav.astype(np.float32, copy=False)


def load_local_db_samples(bench_root: Path, n: int, start: int) -> list[Sample]:
    db_path = bench_root / "stt_benchmark_data" / "results.db"
    if not db_path.exists():
        raise FileNotFoundError(f"benchmark DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT
          s.sample_id,
          s.dataset_index,
          s.duration_seconds,
          s.audio_path,
          gt.text AS reference
        FROM samples s
        JOIN ground_truth gt ON gt.sample_id = s.sample_id
        ORDER BY s.dataset_index
        LIMIT ? OFFSET ?
        """,
        (n, start),
    ).fetchall()
    out: list[Sample] = []
    for row in rows:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = bench_root / audio_path
        out.append(
            Sample(
                sample_id=str(row["sample_id"]),
                dataset_index=int(row["dataset_index"]),
                duration_seconds=float(row["duration_seconds"]),
                reference=str(row["reference"]),
                audio_path=audio_path,
            )
        )
    return out


def load_hf_samples(n: int, start: int) -> list[Sample]:
    import datasets

    ds = datasets.load_dataset("pipecat-ai/stt-benchmark-data", split="train").cast_column(
        "audio", datasets.Audio(decode=False)
    )
    end = min(start + n, len(ds))
    out: list[Sample] = []
    for idx in range(start, end):
        ex = ds[idx]
        # Duration is only informational here; decoding uses the audio bytes.
        audio_bytes = ex["audio"]["bytes"]
        try:
            info = sf.info(io.BytesIO(audio_bytes))
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            duration = 0.0
        out.append(
            Sample(
                sample_id=str(ex["sample_id"]),
                dataset_index=idx,
                duration_seconds=duration,
                reference=str(ex["transcription"]),
                audio_bytes=audio_bytes,
            )
        )
    return out


def load_aoti_first(shared_path: Path, package_path: Path, device: torch.device) -> tuple[Any, dict[str, int]]:
    weights_obj = torch.load(shared_path, map_location="cpu", weights_only=False)
    if not isinstance(weights_obj, dict):
        raise TypeError(f"{shared_path} did not contain a dict")
    weights = {
        str(key): value.detach().cpu().contiguous()
        for key, value in weights_obj.items()
        if torch.is_tensor(value)
    }
    runner = torch._inductor.aoti_load_package(str(package_path))
    fqns = [str(x) for x in runner.loader.get_constant_fqns()]
    cuda_owner: dict[str, torch.Tensor] = {}
    cmap: dict[str, torch.Tensor] = {}
    direct = 0
    alias = 0
    missing: list[str] = []
    for fqn in fqns:
        shared_key = resolve_shared(weights, fqn)
        if shared_key is None:
            missing.append(fqn)
            continue
        if shared_key == fqn:
            direct += 1
        else:
            alias += 1
        if shared_key not in cuda_owner:
            cuda_owner[shared_key] = weights[shared_key].to(device=device, non_blocking=False).contiguous()
        cmap[fqn] = cuda_owner[shared_key]
    if missing:
        raise RuntimeError(f"missing {len(missing)} AOTI first constants; first={missing[:5]}")
    runner.loader.load_constants(cmap, False, False, True)
    # Keep the user-managed tensors alive for the lifetime of the runner.
    runner._probe_cuda_constants_owner = cuda_owner  # type: ignore[attr-defined]
    return runner, {"fqns": len(fqns), "direct": direct, "alias": alias, "matched": len(cmap)}


def first_diff(lhs: list[Any], rhs: list[Any]) -> int | None:
    n = min(len(lhs), len(rhs))
    for i in range(n):
        if lhs[i] != rhs[i]:
            return i
    if len(lhs) != len(rhs):
        return n
    return None


class FirstChunkParity:
    def __init__(self, model: Any, ts_first: Any, aoti_first: Any):
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.joint = model.joint
        self.tokenizer = model.tokenizer
        self.ts_first = ts_first
        self.aoti_first = aoti_first
        self.device = next(model.parameters()).device
        cfg = self.encoder.streaming_cfg
        self.shift = stream_cfg_int(cfg.shift_size)
        self.pre = stream_cfg_int(cfg.pre_encode_cache_size)
        self.drop = int(cfg.drop_extra_pre_encoded)

    def init_cache(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self.encoder.get_initial_cache_state(batch_size=1)
        return cache[0].clone(), cache[1].clone(), cache[2].clone()

    def init_decoder(self) -> tuple[Any, torch.Tensor]:
        state = self.decoder.initialize_state(
            torch.zeros(1, 1, dtype=torch.float32, device=self.device)
        )
        g, state = self.decoder.predict(None, state, add_sos=False, batch_size=1)
        return state, g

    @torch.inference_mode()
    def first_ts(
        self,
        chunk: torch.Tensor,
        length: torch.Tensor,
        clc: torch.Tensor,
        clt: torch.Tensor,
        clcl: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out = self.ts_first(
            chunk.contiguous(),
            length.contiguous(),
            clc.contiguous(),
            clt.contiguous(),
            clcl.contiguous(),
        )
        return tuple(out)

    @torch.inference_mode()
    def first_aoti(
        self,
        chunk: torch.Tensor,
        length: torch.Tensor,
        clc: torch.Tensor,
        clt: torch.Tensor,
        clcl: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out = self.aoti_first(
            chunk.contiguous(),
            length.contiguous(),
            clc.contiguous(),
            clt.contiguous(),
            clcl.contiguous(),
        )
        return tuple(out)

    @torch.inference_mode()
    def steady_eager(
        self,
        chunk: torch.Tensor,
        length: torch.Tensor,
        clc: torch.Tensor,
        clt: torch.Tensor,
        clcl: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        out = self.encoder.cache_aware_stream_step(
            processed_signal=chunk.contiguous(),
            processed_signal_length=length.contiguous(),
            cache_last_channel=clc.contiguous(),
            cache_last_time=clt.contiguous(),
            cache_last_channel_len=clcl.contiguous(),
            keep_all_outputs=False,
            drop_extra_pre_encoded=self.drop,
        )
        return tuple(out)

    @staticmethod
    def output_diffs(lhs: tuple[torch.Tensor, ...], rhs: tuple[torch.Tensor, ...]) -> dict[str, float]:
        names = ["enc_out", "enc_len", "cache_ch", "cache_t", "cache_ch_len"]
        diffs: dict[str, float] = {}
        for name, a, b in zip(names, lhs, rhs):
            if not torch.is_tensor(a) or not torch.is_tensor(b) or a.numel() == 0:
                diffs[name] = 0.0
                continue
            if a.shape != b.shape:
                diffs[name] = float("inf")
            elif a.is_floating_point() or b.is_floating_point():
                diffs[name] = float((a.float() - b.float()).abs().max().item())
            else:
                diffs[name] = float((a.to(torch.int64) - b.to(torch.int64)).abs().max().item())
        return diffs

    @torch.inference_mode()
    def stream(self, mel: torch.Tensor, *, use_aoti_first: bool) -> StreamResult:
        t_mel = mel.shape[-1]
        clc, clt, clcl = self.init_cache()
        dec_state, g = self.init_decoder()
        ring: torch.Tensor | None = None
        tokens: list[int] = []
        events: list[tuple[int, tuple[int, ...]]] = []
        emitted = 0
        pos = 0
        chunk_idx = 0
        first_output_diffs: dict[str, float] = {}

        while pos < t_mel:
            new_mel = mel[:, :, pos : pos + self.shift]
            first = emitted == 0
            chunk = new_mel if first else torch.cat((ring, new_mel), dim=-1)
            length = torch.full((1,), chunk.shape[-1], device=self.device, dtype=torch.long)
            if first:
                if use_aoti_first:
                    # Also compute the TS output for diagnostics, but carry the AOTI output forward.
                    ts_out = self.first_ts(chunk, length, clc, clt, clcl)
                    out = self.first_aoti(chunk, length, clc, clt, clcl)
                    first_output_diffs = self.output_diffs(ts_out, out)
                else:
                    out = self.first_ts(chunk, length, clc, clt, clcl)
            else:
                out = self.steady_eager(chunk, length, clc, clt, clcl)

            enc_out, enc_len, clc, clt, clcl = out
            f = enc_out.transpose(1, 2).contiguous()
            new_tokens, dec_state, g = ref_greedy_range(
                self.decoder,
                self.joint,
                f,
                0,
                int(enc_len[0]),
                dec_state,
                g,
            )
            if new_tokens:
                tokens.extend(new_tokens)
                events.append((chunk_idx, tuple(tokens)))

            ring = (torch.cat((ring, new_mel), dim=-1) if ring is not None else new_mel)[
                :, :, -self.pre :
            ]
            emitted += new_mel.shape[-1]
            pos += self.shift
            chunk_idx += 1

        return StreamResult(
            tokens=tokens,
            text=self.tokenizer.ids_to_text(tokens),
            events=events,
            first_output_diffs=first_output_diffs,
        )


def normalize_for_wer(texts: list[str]) -> list[str]:
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    return [norm(t) for t in texts]


def compute_wer(refs: list[str], hyps: list[str]) -> tuple[float, int]:
    import jiwer

    nr = normalize_for_wer(refs)
    nh = normalize_for_wer(hyps)
    keep = [i for i, ref in enumerate(nr) if ref.strip()]
    if not keep:
        return 0.0, 0
    return float(jiwer.wer([nr[i] for i in keep], [nh[i] for i in keep])), len(keep)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--b2-count", type=int, default=4)
    parser.add_argument("--source", choices=["hf", "local-db"], default="hf")
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--include-event-details", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the enc_first AOTI parity probe")
    device = torch.device("cuda")

    if args.source == "hf":
        samples = load_hf_samples(args.n, args.start)
        sample_source = "pipecat-ai/stt-benchmark-data"
    else:
        samples = load_local_db_samples(args.benchmark_root, args.n, args.start)
        sample_source = str(args.benchmark_root)
    if not samples:
        raise RuntimeError("no benchmark samples loaded")

    print(f"loaded {len(samples)} samples from {sample_source}", flush=True)
    started = time.time()

    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b",
        map_location="cpu",
    ).to(device).eval()
    try:
        model.preprocessor.featurizer.dither = 0.0
    except Exception:
        pass
    model.encoder.set_default_att_context_size([70, 1])
    model.change_decoding_strategy(
        decoding_cfg=OmegaConf.create(
            {
                "strategy": "greedy_batch",
                "greedy": {
                    "max_symbols": 10,
                    "loop_labels": True,
                    "use_cuda_graph_decoder": False,
                },
            }
        )
    )

    print("loading baseline enc_first.ts", flush=True)
    ts_first = torch.jit.load(str(ART / "enc_first.ts"), map_location=device).to(device).eval()
    print("loading candidate enc_first_aoti.pt2 and binding shared constants", flush=True)
    aoti_first, aoti_stats = load_aoti_first(
        ART / "finalize_shared_weights.pt",
        ART / "enc_first_aoti.pt2",
        device,
    )
    probe = FirstChunkParity(model, ts_first, aoti_first)

    rows: list[dict[str, Any]] = []
    first_diff_max: dict[str, float] = {}
    token_divergences: list[dict[str, Any]] = []
    event_divergences: list[dict[str, Any]] = []
    baseline_texts: list[str] = []
    candidate_texts: list[str] = []
    refs: list[str] = []

    for idx, sample in enumerate(samples):
        wav = read_sample_audio(sample)
        audio = torch.tensor(wav, device=device).unsqueeze(0)
        audio_len = torch.tensor([wav.shape[0]], device=device, dtype=torch.long)
        with torch.inference_mode():
            mel, _ = model.preprocessor(input_signal=audio, length=audio_len)

        baseline = probe.stream(mel, use_aoti_first=False)
        candidate = probe.stream(mel, use_aoti_first=True)

        token_diff_idx = first_diff(baseline.tokens, candidate.tokens)
        event_diff_idx = first_diff(baseline.events, candidate.events)
        if token_diff_idx is not None:
            token_divergences.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset_index": sample.dataset_index,
                    "first_diff": token_diff_idx,
                    "baseline_len": len(baseline.tokens),
                    "candidate_len": len(candidate.tokens),
                    "baseline_token": (
                        None if token_diff_idx >= len(baseline.tokens) else baseline.tokens[token_diff_idx]
                    ),
                    "candidate_token": (
                        None
                        if token_diff_idx >= len(candidate.tokens)
                        else candidate.tokens[token_diff_idx]
                    ),
                }
            )
        if event_diff_idx is not None:
            event_divergences.append(
                {
                    "sample_id": sample.sample_id,
                    "dataset_index": sample.dataset_index,
                    "first_diff": event_diff_idx,
                    "baseline_len": len(baseline.events),
                    "candidate_len": len(candidate.events),
                }
            )

        for key, value in candidate.first_output_diffs.items():
            first_diff_max[key] = max(first_diff_max.get(key, 0.0), float(value))

        baseline_texts.append(baseline.text)
        candidate_texts.append(candidate.text)
        refs.append(sample.reference)
        row = {
            "sample_id": sample.sample_id,
            "dataset_index": sample.dataset_index,
            "duration_seconds": sample.duration_seconds,
            "token_equal": token_diff_idx is None,
            "event_equal": event_diff_idx is None,
            "baseline_tokens": len(baseline.tokens),
            "candidate_tokens": len(candidate.tokens),
            "baseline_events": len(baseline.events),
            "candidate_events": len(candidate.events),
            "baseline_text": baseline.text,
            "candidate_text": candidate.text,
            "reference": sample.reference,
        }
        if args.include_event_details and event_diff_idx is not None:
            row["baseline_event_details"] = [
                [chunk_idx, list(tokens)] for chunk_idx, tokens in baseline.events
            ]
            row["candidate_event_details"] = [
                [chunk_idx, list(tokens)] for chunk_idx, tokens in candidate.events
            ]
        rows.append(row)

        if args.progress_every and ((idx + 1) % args.progress_every == 0 or idx + 1 == len(samples)):
            elapsed = time.time() - started
            print(
                f"{idx + 1}/{len(samples)} "
                f"token_div={len(token_divergences)} event_div={len(event_divergences)} "
                f"{elapsed / (idx + 1):.2f}s/utt",
                flush=True,
            )

    wer_baseline, wer_count = compute_wer(refs, baseline_texts)
    wer_candidate, _ = compute_wer(refs, candidate_texts)
    b2_rows = rows[: min(args.b2_count, len(rows))]
    b2_token_divergences = sum(1 for row in b2_rows if not row["token_equal"])
    b2_event_divergences = sum(1 for row in b2_rows if not row["event_equal"])

    result = {
        "elapsed_seconds": time.time() - started,
        "torch": str(torch.__version__),
        "device": torch.cuda.get_device_name(0),
        "samples": len(samples),
        "start": args.start,
        "source": args.source,
        "sample_source": sample_source,
        "benchmark_root": str(args.benchmark_root) if args.source == "local-db" else None,
        "aoti_first_constants": aoti_stats,
        "first_output_max_abs_ts_vs_aoti": first_diff_max,
        "token_divergences": len(token_divergences),
        "event_divergences": len(event_divergences),
        "divergent_sample_ids": [d["sample_id"] for d in token_divergences],
        "event_divergent_sample_ids": [d["sample_id"] for d in event_divergences],
        "token_divergence_details_first20": token_divergences[:20],
        "event_divergence_details_first20": event_divergences[:20],
        "b2_subset": {
            "rows": len(b2_rows),
            "token_divergences": b2_token_divergences,
            "event_divergences": b2_event_divergences,
            "sample_ids": [row["sample_id"] for row in b2_rows],
        },
        "wer": {
            "normalizer": "whisper_normalizer.english.EnglishTextNormalizer",
            "non_empty_refs": wer_count,
            "baseline": wer_baseline,
            "candidate": wer_candidate,
            "delta_candidate_minus_baseline": wer_candidate - wer_baseline,
        },
        "go_enc_first_unify": len(token_divergences) == 0 and len(event_divergences) == 0,
    }
    if args.json_out:
        detail = dict(result)
        detail["rows"] = rows
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")

    print("\n=== AOTI first-chunk parity probe ===")
    print(
        f"samples={len(samples)} b2_subset={len(b2_rows)} "
        f"constants={aoti_stats['matched']}/{aoti_stats['fqns']} "
        f"direct={aoti_stats['direct']} alias={aoti_stats['alias']}"
    )
    print(
        f"token_divergences={len(token_divergences)} "
        f"event_divergences={len(event_divergences)} "
        f"b2_token_divergences={b2_token_divergences} "
        f"b2_event_divergences={b2_event_divergences}"
    )
    print(
        "first_output_max_abs_ts_vs_aoti="
        + json.dumps(first_diff_max, sort_keys=True)
    )
    print(
        "traditional WER whisper-normalized: "
        f"baseline={wer_baseline * 100:.3f}% "
        f"candidate={wer_candidate * 100:.3f}% "
        f"delta={(wer_candidate - wer_baseline) * 100:+.4f} pp "
        f"non_empty_refs={wer_count}"
    )
    if token_divergences:
        print(f"divergent sample_ids={result['divergent_sample_ids'][:20]}")
    print(f"go_enc_first_unify={result['go_enc_first_unify']}")
    if args.json_out:
        print(f"json: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
