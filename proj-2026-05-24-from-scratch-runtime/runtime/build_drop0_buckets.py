#!/usr/bin/env python3
"""1.3b-enc-prod: build the drop=0 (first-finalize) buckets. drop_extra=0 fires when a turn is finalized BEFORE any steady
chunk drains (emitted_frames==0) — i.e. a very short utterance. The corpus never hits this (utterances are long), so we
FORCE it by truncating a clip to short lengths and sweeping. Per the server math the drop=0 finalize T spans ~33..49
(final_padding=32 frames + the short remainder). Exports a fixed-shape torch.export per distinct (drop=0,T); aot_compile +
strip via the usual pipeline. (rc=1 EN v1: this completes the bounded finalize T range 33..58, so full bucket coverage +
fail-closed = no eager path needed.)

Run: HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python build_drop0_buckets.py --out ./artifacts/finalize_buckets
"""
from __future__ import annotations
import argparse, glob, json, os, re, torch
from finalize_ref import ContinuousFinalizeRef, load_model, load_benchmark_dataset, load_wav
from export_finalize_t2a import FinalizeStep

DROP = 0


def existing_T(out_dir: str) -> set[int]:
    have = set()
    stripped = os.path.join(os.path.dirname(out_dir.rstrip("/")), "stripped_finalize_buckets")
    for d in (out_dir, stripped):
        for p in glob.glob(os.path.join(d, f"enc_finalize_d{DROP}_T*.pt2")):
            m = re.search(rf"_d{DROP}_T(\d+)(?:_ep|_stripped)?\.pt2$", os.path.basename(p))
            if m: have.add(int(m.group(1)))
    return have


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./artifacts/finalize_buckets")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model = load_model(); rt = ContinuousFinalizeRef(model)
    g = rt.geometry
    ds = load_benchmark_dataset()
    base = load_wav(ds[0])
    have = existing_T(a.out)

    # sweep short truncation lengths to force drop=0 and span the T range; one example per distinct (drop=0,T)
    examples: dict[int, tuple] = {}
    seen_T = {}
    # lengths from ~1 frame up to just under the first-steady-ready threshold (~ (shift+1)*hop)
    max_len = (g.shift_frames + 1) * g.hop_samples
    for L in range(g.hop_samples, max_len + g.hop_samples, g.hop_samples):
        wav = base[:L]
        if wav.shape[0] < g.hop_samples: continue
        s = rt.new_session("d0"); rt.append_audio(s, wav)
        if s.emitted_frames != 0:  # a steady chunk drained -> not a first-finalize
            seen_T.setdefault("steady_drained", 0); seen_T["steady_drained"] += 1; continue
        rt.vad_stop(s)
        fork = rt.build_continuous_finalize_fork(s); fi = rt.prepare_finalize_inputs(fork)
        if fi is None or int(fi.drop_extra) != DROP: continue
        T = int(fi.chunk_mel.shape[-1]); seen_T[T] = seen_T.get(T, 0) + 1
        if T not in examples and T not in have:
            examples[T] = (fi.chunk_mel, fi.cache_last_channel, fi.cache_last_time, fi.cache_last_channel_len)
    print(f"drop0 T seen (len-sweep): { {k: seen_T[k] for k in sorted(seen_T, key=lambda x: (isinstance(x,str), x))} }")
    print(f"already have drop0 T={sorted(have)}; new to export T={sorted(examples)}")

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
        print(f"  exported drop=0 T={T} byte_exact={be} -> {name}")
    json.dump(manifest, open(os.path.join(a.out, "buckets_manifest.json"), "w"), indent=2)
    print(f"=== exported {len(manifest)} drop=0 buckets; T range covered = {sorted(set(examples) | have)} ===")


if __name__ == "__main__":
    main()
