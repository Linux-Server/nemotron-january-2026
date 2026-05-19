#!/usr/bin/env python3
"""Feasibility probe for bit-exact incremental mel preprocessing.

Scratch diagnostic for Step 6b only. It loads the Nemotron ASR model the same
way probe_alias.py does, then exercises model.preprocessor directly. It does
not start the server, run the benchmark, or touch the Step 6 implementation.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
SAMPLE_RATE = 16000
RIGHT_CONTEXT = 1
REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = REPO_ROOT / "stt-benchmark/stt_benchmark_data/audio"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    prefix_sec: float
    extra_sec: float
    prefer_total_sec: float


@dataclass
class ProbeCase:
    spec: CaseSpec
    path: Path
    total_samples: int
    prefix_samples: int
    long_samples: int
    short_audio: np.ndarray
    long_audio: np.ndarray

    @property
    def total_sec(self) -> float:
        return self.total_samples / SAMPLE_RATE

    @property
    def prefix_sec(self) -> float:
        return self.prefix_samples / SAMPLE_RATE

    @property
    def extra_sec(self) -> float:
        return (self.long_samples - self.prefix_samples) / SAMPLE_RATE

    @property
    def file_id(self) -> str:
        return self.path.stem[:8]


@dataclass
class MelOut:
    mel: torch.Tensor
    valid_frames: int
    shape_frames: int


@dataclass
class CompareMetrics:
    equal: bool
    max_abs: float
    max_rel: float
    max_ulp: int
    p99_ulp: float
    changed_frac: float
    first_div_frame: int


@dataclass
class CompareRow:
    label: str
    case: ProbeCase
    k_frames: int
    short_valid_frames: int
    long_valid_frames: int
    short_shape_frames: int
    long_shape_frames: int
    metrics: CompareMetrics


CASE_SPECS = [
    CaseSpec("1s_vs_1.75s", 1.0, 0.75, 2.0),
    CaseSpec("2s_vs_3s", 2.0, 1.0, 3.0),
    CaseSpec("2.5s_vs_4s", 2.5, 1.5, 4.0),
    CaseSpec("4s_vs_5s", 4.0, 1.0, 5.0),
    CaseSpec("4s_vs_7s", 4.0, 3.0, 8.0),
    CaseSpec("6s_vs_8s", 6.0, 2.0, 8.0),
    CaseSpec("8s_vs_9s", 8.0, 1.0, 10.0),
    CaseSpec("8s_vs_12s", 8.0, 4.0, 12.0),
    CaseSpec("12s_vs_13s", 12.0, 1.0, 13.5),
    CaseSpec("12s_vs_15.5s", 12.0, 3.5, 16.0),
]


def load_model() -> Any:
    import nemo.collections.asr as nemo_asr

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this probe is about CUDA preprocessor behavior.")

    device = torch.device(os.environ.get("NEMOTRON_SPEECH_DEVICE", "cuda"))
    if device.type != "cuda":
        raise RuntimeError(f"Expected CUDA device, got NEMOTRON_SPEECH_DEVICE={device}")

    print(f"Loading {MODEL_NAME} on {device}")
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu")
    model = model.to(device)
    model.encoder.set_default_att_context_size([70, RIGHT_CONTEXT])
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    return model


def read_audio_files() -> list[tuple[Path, int]]:
    files = []
    for path in sorted(AUDIO_DIR.glob("*.pcm")):
        files.append((path, path.stat().st_size // 2))
    if not files:
        raise FileNotFoundError(f"No .pcm fixtures found under {AUDIO_DIR}")
    return files


def load_pcm_float(path: Path, samples: int) -> np.ndarray:
    audio_i16 = np.fromfile(path, dtype=np.int16, count=samples)
    if audio_i16.shape[0] < samples:
        raise RuntimeError(f"Short read from {path}: wanted {samples}, got {audio_i16.shape[0]}")
    return np.ascontiguousarray(audio_i16.astype(np.float32) / 32768.0)


def pick_cases() -> list[ProbeCase]:
    files = read_audio_files()
    used: set[Path] = set()
    cases: list[ProbeCase] = []

    for spec in CASE_SPECS:
        prefix_samples = int(round(spec.prefix_sec * SAMPLE_RATE))
        long_samples = int(round((spec.prefix_sec + spec.extra_sec) * SAMPLE_RATE))
        eligible = [(path, samples) for path, samples in files if samples >= long_samples and path not in used]
        if not eligible:
            eligible = [(path, samples) for path, samples in files if samples >= long_samples]
        if not eligible:
            raise RuntimeError(f"No fixture can cover case {spec.name} ({long_samples} samples)")

        def score(item: tuple[Path, int]) -> tuple[float, int, str]:
            path, samples = item
            return (abs(samples / SAMPLE_RATE - spec.prefer_total_sec), samples, path.name)

        path, total_samples = min(eligible, key=score)
        used.add(path)
        long_audio = load_pcm_float(path, long_samples)
        cases.append(
            ProbeCase(
                spec=spec,
                path=path,
                total_samples=total_samples,
                prefix_samples=prefix_samples,
                long_samples=long_samples,
                short_audio=long_audio[:prefix_samples].copy(),
                long_audio=long_audio,
            )
        )
    return cases


def get_preproc_params(model: Any) -> dict[str, int | str | float]:
    featurizer = model.preprocessor.featurizer
    return {
        "n_fft": int(featurizer.n_fft),
        "win_length": int(featurizer.win_length),
        "hop_length": int(featurizer.hop_length),
        "normalize": str(featurizer.normalize),
        "pad_to": int(featurizer.pad_to),
        "nfilt": int(featurizer.nfilt),
        "dither": float(featurizer.dither),
    }


def safe_leading_frames(prefix_samples: int, win_length: int, hop_length: int) -> int:
    # With torch.stft(center=True), the non-zero Hann window spans roughly
    # frame_center +/- win_length/2. Exclude the trailing boundary frames so
    # the compared frames cannot depend on samples after prefix_samples.
    right_dependency = win_length // 2
    if prefix_samples <= right_dependency:
        return 0
    return ((prefix_samples - right_dependency) // hop_length) + 1


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_preprocessor(preprocessor: Any, audio: np.ndarray, device: torch.device) -> MelOut:
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
    audio_len = torch.tensor([len(audio)], device=device, dtype=torch.long)
    with torch.inference_mode():
        mel, mel_len = preprocessor(input_signal=audio_tensor, length=audio_len)
    sync_if_cuda(device)
    return MelOut(mel=mel.detach(), valid_frames=int(mel_len.item()), shape_frames=int(mel.shape[-1]))


def ordered_float32_ints(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32, copy=False).view(np.uint32).astype(np.int64)
    return np.where((bits & 0x80000000) != 0, 0x80000000 - bits, bits)


def compare_mels(left: torch.Tensor, right: torch.Tensor, k_frames: int) -> CompareMetrics:
    left_k = left[:, :, :k_frames].contiguous()
    right_k = right[:, :, :k_frames].contiguous()
    if left_k.shape != right_k.shape:
        raise RuntimeError(f"Shape mismatch: {tuple(left_k.shape)} vs {tuple(right_k.shape)}")
    if left_k.numel() == 0:
        return CompareMetrics(True, 0.0, 0.0, 0, 0.0, 0.0, -1)

    equal = bool(torch.equal(left_k, right_k))
    abs_diff = (left_k - right_k).abs()
    max_abs = float(abs_diff.max().item())
    denom = torch.maximum(torch.maximum(left_k.abs(), right_k.abs()), torch.tensor(1e-12, device=left_k.device))
    max_rel = float((abs_diff / denom).max().item())
    neq_by_frame = (left_k != right_k).any(dim=(0, 1))
    if bool(neq_by_frame.any().item()):
        first_div_frame = int(torch.nonzero(neq_by_frame, as_tuple=False)[0].item())
    else:
        first_div_frame = -1

    left_np = left_k.detach().cpu().numpy()
    right_np = right_k.detach().cpu().numpy()
    ulp = np.abs(ordered_float32_ints(left_np) - ordered_float32_ints(right_np))
    changed_frac = float(np.count_nonzero(ulp) / ulp.size)
    max_ulp = int(ulp.max()) if ulp.size else 0
    p99_ulp = float(np.percentile(ulp, 99)) if ulp.size else 0.0
    return CompareMetrics(equal, max_abs, max_rel, max_ulp, p99_ulp, changed_frac, first_div_frame)


def compare_case_with_outputs(
    label: str,
    preprocessor: Any,
    case: ProbeCase,
    device: torch.device,
    win_length: int,
    hop_length: int,
    short_audio: np.ndarray,
    long_audio: np.ndarray,
) -> CompareRow:
    short = run_preprocessor(preprocessor, short_audio, device)
    long = run_preprocessor(preprocessor, long_audio, device)
    k_frames = min(
        safe_leading_frames(case.prefix_samples, win_length, hop_length),
        short.valid_frames,
        long.valid_frames,
        short.shape_frames,
        long.shape_frames,
    )
    metrics = compare_mels(short.mel, long.mel, k_frames)
    row = CompareRow(
        label=label,
        case=case,
        k_frames=k_frames,
        short_valid_frames=short.valid_frames,
        long_valid_frames=long.valid_frames,
        short_shape_frames=short.shape_frames,
        long_shape_frames=long.shape_frames,
        metrics=metrics,
    )
    del short, long
    return row


def run_rows(
    label: str,
    preprocessor: Any,
    cases: list[ProbeCase],
    device: torch.device,
    win_length: int,
    hop_length: int,
    transform_short: Callable[[np.ndarray], np.ndarray] | None = None,
    transform_long: Callable[[np.ndarray], np.ndarray] | None = None,
    clear_cache_each_pair: bool = False,
) -> list[CompareRow]:
    rows = []
    for case in cases:
        if clear_cache_each_pair:
            clear_cufft_plan_cache()
        short_audio = transform_short(case.short_audio) if transform_short else case.short_audio
        long_audio = transform_long(case.long_audio) if transform_long else case.long_audio
        rows.append(
            compare_case_with_outputs(label, preprocessor, case, device, win_length, hop_length, short_audio, long_audio)
        )
    return rows


def pad_to_plan_frames(audio: np.ndarray, plan_frames: int, hop_length: int) -> np.ndarray:
    target_samples = (plan_frames - 1) * hop_length
    if len(audio) > target_samples:
        raise ValueError(f"Audio has {len(audio)} samples, exceeds fixed plan target {target_samples}")
    if len(audio) == target_samples:
        return audio
    return np.pad(audio, (0, target_samples - len(audio)), mode="constant").astype(np.float32, copy=False)


def clear_cufft_plan_cache() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.cufft_plan_cache.clear()
    except Exception:
        try:
            torch.backends.cuda.cufft_plan_cache[torch.cuda.current_device()].clear()
        except Exception:
            pass


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * width for width in widths]))
    for row in rows:
        print(fmt.format(*row))


def fmt_bool(value: bool) -> str:
    return "T" if value else "F"


def fmt_sci(value: float) -> str:
    return f"{value:.3e}"


def row_to_table(row: CompareRow) -> list[str]:
    m = row.metrics
    return [
        row.case.spec.name,
        row.case.file_id,
        f"{row.case.total_sec:.2f}",
        f"{row.case.prefix_sec:.2f}",
        f"{row.case.extra_sec:.2f}",
        str(row.k_frames),
        f"{row.short_valid_frames}/{row.long_valid_frames}",
        f"{row.short_shape_frames}/{row.long_shape_frames}",
        fmt_bool(m.equal),
        fmt_sci(m.max_abs),
        fmt_sci(m.max_rel),
        str(m.max_ulp),
        f"{m.p99_ulp:.1f}",
        f"{100.0 * m.changed_frac:.1f}%",
        str(m.first_div_frame),
    ]


def summarize_rows(rows: list[CompareRow]) -> dict[str, Any]:
    bad = [row for row in rows if not row.metrics.equal]
    return {
        "all_equal": not bad,
        "bad_count": len(bad),
        "total": len(rows),
        "first_bad": bad[0].case.spec.name if bad else "-",
        "max_abs": max((row.metrics.max_abs for row in rows), default=0.0),
        "max_rel": max((row.metrics.max_rel for row in rows), default=0.0),
        "max_ulp": max((row.metrics.max_ulp for row in rows), default=0),
        "max_changed_frac": max((row.metrics.changed_frac for row in rows), default=0.0),
        "min_first_div": min(
            (row.metrics.first_div_frame for row in rows if row.metrics.first_div_frame >= 0),
            default=-1,
        ),
    }


def summary_table_row(name: str, rows: list[CompareRow], note: str) -> list[str]:
    s = summarize_rows(rows)
    return [
        name,
        f"{s['total'] - s['bad_count']}/{s['total']}",
        fmt_bool(bool(s["all_equal"])),
        str(s["first_bad"]),
        fmt_sci(float(s["max_abs"])),
        fmt_sci(float(s["max_rel"])),
        str(s["max_ulp"]),
        f"{100.0 * float(s['max_changed_frac']):.1f}%",
        str(s["min_first_div"]),
        note,
    ]


def time_preprocessor(
    preprocessor: Any,
    audio: np.ndarray,
    device: torch.device,
    repeats: int = 7,
    warmups: int = 2,
) -> dict[str, float]:
    times_ms: list[float] = []
    for idx in range(warmups + repeats):
        sync_if_cuda(device)
        t0 = time.perf_counter()
        out = run_preprocessor(preprocessor, audio, device)
        sync_if_cuda(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if idx >= warmups:
            times_ms.append(elapsed_ms)
        del out
    return {
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def set_determinism_flags() -> dict[str, Any]:
    old = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    clear_cufft_plan_cache()
    return old


def restore_determinism_flags(old: dict[str, Any]) -> None:
    torch.use_deterministic_algorithms(bool(old["deterministic"]))
    torch.backends.cudnn.benchmark = bool(old["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(old["cudnn_deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = bool(old["cuda_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(old["cudnn_tf32"])
    torch.set_float32_matmul_precision(str(old["matmul_precision"]))


def run() -> int:
    torch.set_grad_enabled(False)
    cases = pick_cases()
    model = load_model()
    device = next(model.preprocessor.parameters(), model.preprocessor.featurizer.window).device
    params = get_preproc_params(model)
    win_length = int(params["win_length"])
    hop_length = int(params["hop_length"])

    print()
    print("Preprocessor config")
    for key in ["n_fft", "win_length", "hop_length", "normalize", "pad_to", "nfilt", "dither"]:
        print(f"  {key}: {params[key]}")
    print(f"  compare margin: safe leading frames exclude right boundary dependency of {win_length // 2} samples")

    print()
    print("Selected real PCM fixtures")
    print_table(
        ["case", "file", "fixture_s", "prefix_L_s", "extra_s", "long_s"],
        [
            [
                case.spec.name,
                case.path.name,
                f"{case.total_sec:.2f}",
                f"{case.prefix_sec:.2f}",
                f"{case.extra_sec:.2f}",
                f"{case.long_samples / SAMPLE_RATE:.2f}",
            ]
            for case in cases
        ],
    )

    print()
    print("Q1: CUDA growing-plan SHORT prefix vs LONG prefix with identical leading samples")
    q1_rows = run_rows("cuda_growing", model.preprocessor, cases, device, win_length, hop_length)
    print_table(
        [
            "case",
            "file",
            "fixture_s",
            "L_s",
            "extra_s",
            "k",
            "valid_fr",
            "shape_fr",
            "equal",
            "max_abs",
            "max_rel",
            "max_ulp",
            "p99_ulp",
            "changed",
            "first_div_fr",
        ],
        [row_to_table(row) for row in q1_rows],
    )
    q1_summary = summarize_rows(q1_rows)
    print(
        "Q1 root-cause conclusion: "
        f"GPU preprocessor leading frames are plan-size invariant? {fmt_bool(bool(q1_summary['all_equal']))} "
        f"({q1_summary['total'] - q1_summary['bad_count']}/{q1_summary['total']} cases bit-exact)."
    )

    max_long_shape = max(row.long_shape_frames for row in q1_rows)
    fixed_plan_frames = int(os.environ.get("STFT_FIXED_PLAN_FRAMES", str(max(2048, max_long_shape))))
    print()
    print(f"Q2a: Fixed CUDA frame plan K={fixed_plan_frames}")
    fixed_rows = run_rows(
        "cuda_fixedK",
        model.preprocessor,
        cases,
        device,
        win_length,
        hop_length,
        transform_short=lambda audio: pad_to_plan_frames(audio, fixed_plan_frames, hop_length),
        transform_long=lambda audio: pad_to_plan_frames(audio, fixed_plan_frames, hop_length),
    )

    fixed_vs_growing_rows: list[CompareRow] = []
    for case in cases:
        fixed_long = pad_to_plan_frames(case.long_audio, fixed_plan_frames, hop_length)
        fixed_vs_growing_rows.append(
            compare_case_with_outputs(
                "fixed_vs_growing_same_long",
                model.preprocessor,
                case,
                device,
                win_length,
                hop_length,
                fixed_long,
                case.long_audio,
            )
        )

    print_table(
        [
            "regime",
            "equal_cases",
            "all_equal",
            "first_bad",
            "max_abs",
            "max_rel",
            "max_ulp",
            "max_changed",
            "min_first_div",
            "note",
        ],
        [
            summary_table_row(
                f"cuda_fixed_K{fixed_plan_frames}",
                fixed_rows,
                "same bounded shape for every call",
            ),
            summary_table_row(
                "fixedK_vs_current_growing",
                fixed_vs_growing_rows,
                "cost: current goldens change",
            ),
        ],
    )

    print()
    print("Q2c: CUDA determinism flags / TF32 off / cuFFT cache clear against growing plans")
    old_flags: dict[str, Any] | None = None
    try:
        old_flags = set_determinism_flags()
        det_rows = run_rows(
            "cuda_deterministic_growing",
            model.preprocessor,
            cases,
            device,
            win_length,
            hop_length,
            clear_cache_each_pair=True,
        )
    finally:
        if old_flags is not None:
            restore_determinism_flags(old_flags)
    print_table(
        [
            "regime",
            "equal_cases",
            "all_equal",
            "first_bad",
            "max_abs",
            "max_rel",
            "max_ulp",
            "max_changed",
            "min_first_div",
            "note",
        ],
        [
            summary_table_row(
                "cuda_deterministic_growing",
                det_rows,
                "torch deterministic + TF32 off",
            )
        ],
    )

    ten_sec_case = next((case for case in cases if len(case.long_audio) >= 10 * SAMPLE_RATE), cases[-1])
    ten_sec_audio = ten_sec_case.long_audio[: 10 * SAMPLE_RATE]
    fixed_10s_audio = pad_to_plan_frames(ten_sec_audio, fixed_plan_frames, hop_length)
    gpu_10s_time = time_preprocessor(model.preprocessor, ten_sec_audio, device)
    gpu_fixed_time = time_preprocessor(model.preprocessor, fixed_10s_audio, device)

    print()
    print("Q2 timing: CUDA preprocessor")
    print_table(
        ["input", "samples", "frames_shape", "median_ms", "min_ms", "max_ms"],
        [
            [
                "10s_growing",
                str(len(ten_sec_audio)),
                str(run_preprocessor(model.preprocessor, ten_sec_audio, device).shape_frames),
                f"{gpu_10s_time['median_ms']:.2f}",
                f"{gpu_10s_time['min_ms']:.2f}",
                f"{gpu_10s_time['max_ms']:.2f}",
            ],
            [
                f"10s_padded_K{fixed_plan_frames}",
                str(len(fixed_10s_audio)),
                str(run_preprocessor(model.preprocessor, fixed_10s_audio, device).shape_frames),
                f"{gpu_fixed_time['median_ms']:.2f}",
                f"{gpu_fixed_time['min_ms']:.2f}",
                f"{gpu_fixed_time['max_ms']:.2f}",
            ],
        ],
    )

    print()
    print("Q2b: CPU preprocessor growing-plan length-invariance")
    model.preprocessor = model.preprocessor.cpu()
    gc.collect()
    cpu_device = torch.device("cpu")
    cpu_rows = run_rows("cpu_growing", model.preprocessor, cases, cpu_device, win_length, hop_length)
    cpu_10s_time = time_preprocessor(model.preprocessor, ten_sec_audio, cpu_device, repeats=5, warmups=1)
    print_table(
        [
            "regime",
            "equal_cases",
            "all_equal",
            "first_bad",
            "max_abs",
            "max_rel",
            "max_ulp",
            "max_changed",
            "min_first_div",
            "note",
        ],
        [
            summary_table_row(
                "cpu_growing",
                cpu_rows,
                f"10s median {cpu_10s_time['median_ms']:.2f} ms",
            )
        ],
    )

    q3 = summarize_rows(q1_rows)
    print()
    print("Q3: Magnitude summary for CUDA growing-plan divergence")
    print_table(
        ["cases_changed", "max_abs", "max_rel", "max_ulp", "max_changed_values", "earliest_first_div"],
        [
            [
                f"{q3['bad_count']}/{q3['total']}",
                fmt_sci(float(q3["max_abs"])),
                fmt_sci(float(q3["max_rel"])),
                str(q3["max_ulp"]),
                f"{100.0 * float(q3['max_changed_frac']):.1f}%",
                str(q3["min_first_div"]),
            ]
        ],
    )

    fixed_summary = summarize_rows(fixed_rows)
    cpu_summary = summarize_rows(cpu_rows)
    det_summary = summarize_rows(det_rows)
    print()
    print("Decisive verdict")
    if not q1_summary["all_equal"]:
        print(
            "  (i) NO: bit-exact incremental == current growing full re-preprocess is not generally "
            "achievable with a bounded length-independent CUDA preprocessor plan."
        )
    else:
        print(
            "  (i) YES: this run did not expose CUDA length sensitivity; byte-equivalence to growing full "
            "would remain feasible under the tested cases."
        )
    if fixed_summary["all_equal"]:
        print(
            f"  (ii.a) YES: a fixed CUDA plan K={fixed_plan_frames} is length-independent and bit-exact "
            "across all Q1 cases, but it is a different golden regime."
        )
    else:
        print(f"  (ii.a) NO: fixed CUDA plan K={fixed_plan_frames} still diverged.")
    if cpu_summary["all_equal"]:
        print(
            f"  (ii.b) YES: CPU growing-plan preprocessing is bit-exact across all Q1 cases; "
            f"10s median {cpu_10s_time['median_ms']:.2f} ms."
        )
    else:
        print(
            f"  (ii.b) NO: CPU growing-plan preprocessing is not bit-exact across all Q1 cases; "
            f"10s median {cpu_10s_time['median_ms']:.2f} ms."
        )
    if det_summary["all_equal"]:
        print("  (ii.c) YES: determinism flags removed CUDA growing-plan sensitivity in this run.")
    else:
        print("  (ii.c) NO: determinism flags did not remove CUDA growing-plan sensitivity.")
    print(
        "  (iii) The observed mel differences are small numerical perturbations, not semantic-scale "
        "feature changes, but byte equality is invalid as a success rule for the current growing-plan golden."
    )
    if fixed_summary["all_equal"]:
        print(
            "Recommended Step-6 criterion: either recapture the oracle under a fixed-plan regime and keep "
            "byte-equivalence within that new regime, or relax the current growing-full golden to paired "
            "full-1000 WER-within-CI. Do not keep byte-equivalence against the current growing-plan goldens."
        )
    else:
        print(
            "Recommended Step-6 criterion: relax the current growing-full golden to paired full-1000 "
            "WER-within-CI; no tested length-independent byte-exact regime survived."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
