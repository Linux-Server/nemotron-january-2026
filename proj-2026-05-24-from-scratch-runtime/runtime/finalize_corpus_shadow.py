#!/usr/bin/env python3
"""Corpus shadow for exact-T AOTI finalize buckets vs eager finalize_ref.

Run from runtime/:
  HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python finalize_corpus_shadow.py 200
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from finalize_ref import (
    ContinuousFinalizeRef,
    FinalizeInputs,
    clone_tree,
    load_benchmark_dataset,
    load_model,
    load_wav,
    tensor_clone,
)
from ref_decode import ref_greedy_range


BUCKET_RE = re.compile(r"^enc_finalize_d(?P<drop>\d+)_T(?P<T>\d+)\.pt2$")


@dataclass
class SampleResult:
    index: int
    sample_id: str
    drop: int | None
    T: int | None
    covered: bool
    token_exact: bool | None
    eager_tokens: list[int]
    bucket_tokens: list[int] | None
    detail: str = ""


def discover_buckets(path: str) -> dict[tuple[int, int], str]:
    buckets: dict[tuple[int, int], str] = {}
    if not os.path.isdir(path):
        return buckets
    for name in os.listdir(path):
        match = BUCKET_RE.match(name)
        if match:
            key = (int(match.group("drop")), int(match.group("T")))
            buckets[key] = os.path.join(path, name)
    return buckets


def resolve_shared_weight(weights: dict[str, Any], fqn: str) -> torch.Tensor | None:
    if fqn in weights:
        return weights[fqn]
    if fqn.startswith("encoder."):
        return weights.get("e." + fqn[len("encoder.") :])
    if fqn.startswith("e."):
        return weights.get("encoder." + fqn[len("e.") :])
    return None


def gb(value: int | float) -> float:
    return float(value) / 1_000_000_000.0


def cuda_mem() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "driver_used_gb": 0.0}
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gb": gb(torch.cuda.memory_allocated()),
        "reserved_gb": gb(torch.cuda.memory_reserved()),
        "driver_used_gb": gb(total - free),
    }


def format_mem(mem: dict[str, float]) -> str:
    return (
        f"alloc={mem['allocated_gb']:.3f}GB "
        f"reserved={mem['reserved_gb']:.3f}GB "
        f"driver_used={mem['driver_used_gb']:.3f}GB"
    )


def nvidia_smi_used() -> str | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    return out or None


class BucketManager:
    def __init__(self, buckets_dir: str, shared_weights_path: str):
        self.buckets = discover_buckets(buckets_dir)
        self.shared_weights_path = shared_weights_path
        self.shared_weights_cpu: dict[str, Any] | None = None
        self.shared_weights_cuda: dict[str, torch.Tensor] = {}
        self.runners: dict[tuple[int, int], Any] = {}
        self.load_events: list[dict[str, Any]] = []

    def has_bucket(self, drop: int, T: int) -> bool:
        return (drop, T) in self.buckets

    def _load_shared_weights_cpu(self) -> dict[str, Any]:
        if self.shared_weights_cpu is None:
            self.shared_weights_cpu = torch.load(
                self.shared_weights_path,
                map_location="cpu",
                weights_only=False,
            )
        return self.shared_weights_cpu

    def get(self, drop: int, T: int):
        key = (drop, T)
        if key in self.runners:
            return self.runners[key]
        path = self.buckets.get(key)
        if path is None:
            return None

        before = cuda_mem()
        runner = torch._inductor.aoti_load_package(path)
        fqns = list(runner.loader.get_constant_fqns())
        weights = self._load_shared_weights_cpu()

        cmap: dict[str, torch.Tensor] = {}
        missing: list[str] = []
        new_cuda_tensors = 0
        for fqn in fqns:
            tensor = resolve_shared_weight(weights, fqn)
            if tensor is None:
                missing.append(fqn)
                continue
            if fqn not in self.shared_weights_cuda:
                self.shared_weights_cuda[fqn] = tensor.cuda()
                new_cuda_tensors += 1
            cmap[fqn] = self.shared_weights_cuda[fqn]
        if missing:
            raise RuntimeError(
                f"bucket drop={drop} T={T} missing {len(missing)} constants; "
                f"first={missing[:5]}"
            )

        runner.loader.load_constants(cmap, False, False, True)
        self.runners[key] = runner
        after = cuda_mem()
        self.load_events.append(
            {
                "drop": drop,
                "T": T,
                "path": path,
                "fqns": len(fqns),
                "matched": len(cmap),
                "new_cuda_tensors": new_cuda_tensors,
                "before": before,
                "after": after,
            }
        )
        return runner


def sample_id_for(example: dict[str, Any], index: int) -> str:
    value = example.get("sample_id", index)
    return str(value)


def first_token_diff(lhs: list[int], rhs: list[int]) -> str:
    for i, (a, b) in enumerate(zip(lhs, rhs)):
        if a != b:
            return f"first_diff={i} eager={a} bucket={b}"
    if len(lhs) != len(rhs):
        return f"prefix_equal eager_len={len(lhs)} bucket_len={len(rhs)}"
    return "identical"


def decode_from_prefinal(
    rt: ContinuousFinalizeRef,
    enc_out: torch.Tensor,
    enc_len: torch.Tensor,
    pre_state: Any,
    pre_pred_out: torch.Tensor,
    pre_tokens: list[int],
) -> list[int]:
    new_tokens, _state, _pred_out = ref_greedy_range(
        rt.decoder,
        rt.joint,
        enc_out.transpose(1, 2).contiguous(),
        0,
        int(enc_len.detach().reshape(-1)[0].item()),
        clone_tree(pre_state),
        tensor_clone(pre_pred_out),
    )
    return list(pre_tokens) + list(new_tokens)


def eager_finalize_tokens(
    rt: ContinuousFinalizeRef,
    inputs: FinalizeInputs,
    pre_state: Any,
    pre_pred_out: torch.Tensor,
    pre_tokens: list[int],
) -> tuple[list[int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    enc_out, enc_len, clc, clt, clcl = rt.encoder.cache_aware_stream_step(
        processed_signal=inputs.chunk_mel,
        processed_signal_length=inputs.chunk_len,
        cache_last_channel=inputs.cache_last_channel,
        cache_last_time=inputs.cache_last_time,
        cache_last_channel_len=inputs.cache_last_channel_len,
        keep_all_outputs=True,
        drop_extra_pre_encoded=inputs.drop_extra,
    )
    tokens = decode_from_prefinal(rt, enc_out, enc_len, pre_state, pre_pred_out, pre_tokens)
    return tokens, (enc_out, enc_len, clc, clt, clcl)


def bucket_finalize_tokens(
    rt: ContinuousFinalizeRef,
    runner: Any,
    inputs: FinalizeInputs,
    pre_state: Any,
    pre_pred_out: torch.Tensor,
    pre_tokens: list[int],
) -> tuple[list[int], tuple[torch.Tensor, ...]]:
    outputs = runner(
        inputs.chunk_mel.contiguous(),
        inputs.cache_last_channel.contiguous(),
        inputs.cache_last_time.contiguous(),
        inputs.cache_last_channel_len.contiguous(),
    )
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    if len(outputs) < 2:
        raise RuntimeError(f"AOTI bucket returned {len(outputs)} outputs, expected at least 2")
    enc_out = outputs[0]
    enc_len = outputs[1]
    tokens = decode_from_prefinal(rt, enc_out, enc_len, pre_state, pre_pred_out, pre_tokens)
    return tokens, tuple(outputs)


def run_sample(
    rt: ContinuousFinalizeRef,
    manager: BucketManager,
    example: dict[str, Any],
    index: int,
) -> SampleResult:
    wav = load_wav(example)
    session = rt.new_session(f"shadow-{index}")
    rt.append_audio(session, wav)
    rt.vad_stop(session)
    fork = rt.build_continuous_finalize_fork(session)
    inputs = rt.prepare_finalize_inputs(fork)

    sid = sample_id_for(example, index)
    pre_tokens = list(fork.hyp_tokens)
    pre_state = clone_tree(fork.decoder_state)
    pre_pred_out = tensor_clone(fork.pred_out_stream)

    if inputs is None:
        return SampleResult(
            index=index,
            sample_id=sid,
            drop=None,
            T=None,
            covered=False,
            token_exact=None,
            eager_tokens=pre_tokens,
            bucket_tokens=None,
            detail="no finalize inputs",
        )

    eager_tokens, _eager_outputs = eager_finalize_tokens(
        rt,
        inputs,
        pre_state,
        pre_pred_out,
        pre_tokens,
    )

    drop = int(inputs.drop_extra)
    T = int(inputs.chunk_mel.shape[-1])
    runner = manager.get(drop, T)
    if runner is None:
        return SampleResult(
            index=index,
            sample_id=sid,
            drop=drop,
            T=T,
            covered=False,
            token_exact=None,
            eager_tokens=eager_tokens,
            bucket_tokens=None,
            detail="missing bucket",
        )

    bucket_tokens, _bucket_outputs = bucket_finalize_tokens(
        rt,
        runner,
        inputs,
        pre_state,
        pre_pred_out,
        pre_tokens,
    )
    exact = eager_tokens == bucket_tokens
    return SampleResult(
        index=index,
        sample_id=sid,
        drop=drop,
        T=T,
        covered=True,
        token_exact=exact,
        eager_tokens=eager_tokens,
        bucket_tokens=bucket_tokens,
        detail="" if exact else first_token_diff(eager_tokens, bucket_tokens),
    )


def print_coverage(
    counts: Counter[tuple[int, int]],
    exact_counts: Counter[tuple[int, int]],
    divergent_counts: Counter[tuple[int, int]],
    bucket_keys: set[tuple[int, int]],
) -> None:
    print("\nper-(drop,T) coverage:")
    if not counts:
        print("  none")
        return
    for drop, T in sorted(counts):
        covered = (drop, T) in bucket_keys
        exact = exact_counts[(drop, T)]
        divergent = divergent_counts[(drop, T)]
        print(
            f"  drop={drop} T={T}: occurred={counts[(drop, T)]} "
            f"bucket={'yes' if covered else 'no'} exact={exact} divergent={divergent}"
        )


def print_memory_report(manager: BucketManager, mem_start: dict[str, float], mem_model: dict[str, float]) -> None:
    mem_end = cuda_mem()
    print("\nGPU memory:")
    print(f"  start:       {format_mem(mem_start)}")
    print(f"  after model: {format_mem(mem_model)}")
    if manager.load_events:
        first = manager.load_events[0]["after"]
        last = manager.load_events[-1]["after"]
        print(f"  after first bucket constants: {format_mem(first)}")
        print(f"  after all buckets:            {format_mem(last)}")
        print(
            "  bucket-load delta first->last: "
            f"alloc={last['allocated_gb'] - first['allocated_gb']:+.3f}GB "
            f"reserved={last['reserved_gb'] - first['reserved_gb']:+.3f}GB "
            f"driver_used={last['driver_used_gb'] - first['driver_used_gb']:+.3f}GB"
        )
        print("  loaded buckets:")
        for event in manager.load_events:
            before = event["before"]
            after = event["after"]
            print(
                f"    drop={event['drop']} T={event['T']}: fqns={event['fqns']} "
                f"new_cuda_tensors={event['new_cuda_tensors']} "
                f"alloc_delta={after['allocated_gb'] - before['allocated_gb']:+.3f}GB "
                f"driver_delta={after['driver_used_gb'] - before['driver_used_gb']:+.3f}GB"
            )
    print(f"  end:         {format_mem(mem_end)}")
    smi = nvidia_smi_used()
    if smi:
        print(f"  nvidia-smi:  {smi}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="?", type=int, default=200, help="number of stt-benchmark samples")
    parser.add_argument("--start", type=int, default=0, help="dataset start index")
    parser.add_argument("--buckets-dir", default="artifacts/stripped_finalize_buckets")
    parser.add_argument("--shared-weights", default="artifacts/finalize_shared_weights.pt")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-divergent-print", type=int, default=10)
    args = parser.parse_args()

    faulthandler.enable()
    torch.set_grad_enabled(False)

    t0 = time.time()
    print("loading model + stt-benchmark...")
    mem_start = cuda_mem()
    model = load_model()
    rt = ContinuousFinalizeRef(model)
    mem_model = cuda_mem()
    ds = load_benchmark_dataset()
    manager = BucketManager(args.buckets_dir, args.shared_weights)
    print(f"discovered {len(manager.buckets)} stripped buckets in {args.buckets_dir}")

    end = min(args.start + args.n, len(ds))
    if args.start < 0 or args.start >= len(ds) or end <= args.start:
        raise SystemExit(f"empty sample range start={args.start} n={args.n} dataset_len={len(ds)}")

    results: list[SampleResult] = []
    counts: Counter[tuple[int, int]] = Counter()
    exact_counts: Counter[tuple[int, int]] = Counter()
    divergent_counts: Counter[tuple[int, int]] = Counter()
    uncovered: list[SampleResult] = []
    divergent: list[SampleResult] = []

    with torch.inference_mode():
        for offset, index in enumerate(range(args.start, end), 1):
            result = run_sample(rt, manager, ds[index], index)
            results.append(result)
            if result.drop is not None and result.T is not None:
                key = (result.drop, result.T)
                counts[key] += 1
                if result.covered and result.token_exact:
                    exact_counts[key] += 1
                elif result.covered and result.token_exact is False:
                    divergent_counts[key] += 1
            if not result.covered:
                uncovered.append(result)
            elif result.token_exact is False:
                divergent.append(result)

            if args.progress_every > 0 and offset % args.progress_every == 0:
                covered = sum(1 for r in results if r.covered)
                exact = sum(1 for r in results if r.covered and r.token_exact)
                print(
                    f"  {offset}/{end - args.start} samples "
                    f"covered={covered} exact={exact} divergent={len(divergent)} "
                    f"uncovered={len(uncovered)} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    covered = sum(1 for r in results if r.covered)
    token_exact = sum(1 for r in results if r.covered and r.token_exact)
    no_inputs = [r for r in uncovered if r.drop is None]
    missing_bucket = [r for r in uncovered if r.drop is not None]
    # FAIL-CLOSED (enc-scale review B2): a finalize with no matching bucket emits NO final tokens -> a dropped
    # end-of-turn transcript in production. Until a NAMED, VALIDATED eager fallback exists, missing buckets are a FAILURE,
    # not a footnote. (no_inputs = sample had no finalize remainder at all; benign, not counted as a miss.)
    pass_ok = len(divergent) == 0 and len(missing_bucket) == 0

    print(f"\n=== finalize corpus shadow: samples {args.start}..{end - 1} (n={len(results)}) ===")
    print(f"samples={len(results)} covered={covered} token_exact={token_exact} divergent={len(divergent)} uncovered={len(uncovered)}")
    if no_inputs:
        ids = [r.sample_id for r in no_inputs[: args.max_divergent_print]]
        print(f"no finalize inputs: {len(no_inputs)} ids={ids}")
    if missing_bucket:
        by_missing: dict[tuple[int, int], list[str]] = defaultdict(list)
        for r in missing_bucket:
            by_missing[(int(r.drop), int(r.T))].append(r.sample_id)
        print("uncovered missing buckets:")
        for key in sorted(by_missing):
            ids = by_missing[key][: args.max_divergent_print]
            print(f"  drop={key[0]} T={key[1]} count={len(by_missing[key])} sample_ids={ids}")

    if divergent:
        print("first divergent covered samples:")
        for r in divergent[: args.max_divergent_print]:
            print(
                f"  index={r.index} sample_id={r.sample_id} drop={r.drop} T={r.T} "
                f"eager_len={len(r.eager_tokens)} bucket_len={len(r.bucket_tokens or [])} {r.detail}"
            )

    print_coverage(counts, exact_counts, divergent_counts, set(manager.buckets))
    print_memory_report(manager, mem_start, mem_model)
    print(f"\nPASS/FAIL: {'PASS' if pass_ok else 'FAIL'}")
    print(
        "PASS criterion: every covered sample token-exact AND zero missing-bucket samples "
        "(fail-closed; a missing bucket = a dropped final transcript until a validated fallback exists)."
    )
    return 0 if pass_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
