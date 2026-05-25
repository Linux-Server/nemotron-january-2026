#!/usr/bin/env python3
"""1.3b-enc-scale (host): collect a REAL finalize example per missing (drop=2, T) from the corpus and torch.export each
fixed-T finalize bucket (byte-exact fixed-shape). The corpus T-distribution is drop=2, T=43..58 (finalize_t_distribution.py);
we already have 44/45/58, so export the rest. aot_compile_buckets.py then AOTI-compiles them (container), strip_bucket_weights
strips the dead in-package weights. Real (not dummy) examples avoid any export-fidelity question.

Run: HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python build_range_examples.py --out ./artifacts/finalize_buckets
"""
from __future__ import annotations
import argparse, glob, json, os, re, torch
from finalize_ref import ContinuousFinalizeRef, load_model, load_benchmark_dataset, load_wav
from export_finalize_t2a import FinalizeStep

T_LO, T_HI, DROP = 43, 58, 2


def existing_T(out_dir: str) -> set[int]:
    """Count T already built — in finalize_buckets/ AND the stripped output dir (so we skip already-stripped ones)."""
    have = set()
    dirs = [out_dir, os.path.join(os.path.dirname(out_dir.rstrip("/")), "stripped_finalize_buckets")]
    for d in dirs:
        for p in glob.glob(os.path.join(d, f"enc_finalize_d{DROP}_T*.pt2")):
            m = re.search(rf"_d{DROP}_T(\d+)(?:_ep|_stripped)?\.pt2$", os.path.basename(p))
            if m: have.add(int(m.group(1)))
    return have


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./artifacts/finalize_buckets")
    ap.add_argument("--scan", type=int, default=400); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model = load_model(); rt = ContinuousFinalizeRef(model)
    ds = load_benchmark_dataset()
    have = existing_T(a.out)
    need = [T for T in range(T_LO, T_HI + 1) if T not in have]
    print(f"have T={sorted(have)}; need T={need}")

    examples: dict[int, tuple] = {}
    for i in range(min(a.scan, len(ds))):
        if all(T in examples or T in have for T in range(T_LO, T_HI + 1)): break
        s = rt.new_session(f"s{i}"); rt.append_audio(s, load_wav(ds[i])); rt.vad_stop(s)
        fork = rt.build_continuous_finalize_fork(s); fi = rt.prepare_finalize_inputs(fork)
        if fi is None or int(fi.drop_extra) != DROP: continue
        T = int(fi.chunk_mel.shape[-1])
        if T in need and T not in examples:
            examples[T] = (fi.chunk_mel, fi.cache_last_channel, fi.cache_last_time, fi.cache_last_channel_len)

    manifest = []
    for T in sorted(examples):
        chunk, clc, clt, clcl = examples[T]
        step = FinalizeStep(model.encoder, drop_extra=DROP).cuda().eval()
        with torch.inference_mode():
            ep = torch.export.export(step, (chunk.cuda(), clc.cuda(), clt.cuda(), clcl.cuda()))
            eager = step(chunk.cuda(), clc.cuda(), clt.cuda(), clcl.cuda()); exp = ep.module()(chunk.cuda(), clc.cuda(), clt.cuda(), clcl.cuda())
            be = all(torch.equal(e, x) for e, x in zip(eager, exp) if torch.is_tensor(e))
        name = f"enc_finalize_d{DROP}_T{T}_ep.pt2"
        torch.export.save(ep, os.path.join(a.out, name))
        manifest.append({"drop": DROP, "T": T, "ep": name})
        print(f"  exported drop={DROP} T={T} byte_exact={be} -> {name}")
    missing = [T for T in need if T not in examples]
    json.dump(manifest, open(os.path.join(a.out, "buckets_manifest.json"), "w"), indent=2)
    print(f"=== exported {len(manifest)} new buckets; still-missing T (not found in {a.scan} samples)={missing} ===")


if __name__ == "__main__":
    main()
