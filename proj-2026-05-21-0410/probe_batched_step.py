"""Probe B — batched conformer_stream_step correctness (the load-bearing batching de-risk).

Question: does running 2 streams BATCHED (B=2) through one conformer_stream_step per tick give
byte-identical PER-STREAM output vs running them SEPARATELY at B=1? This tests the cache-aware-state
hazard (no cross-talk) + the flat per-row hypothesis handling.

Key: this tests batched==separate (per-stream CONSISTENCY), NOT absolute transcript correctness — so a
simple consistent chunk feed (same for both arms) is valid. Uses the model's streaming API directly.

Two sub-tests:
  1. batched-from-start: both streams as a B=2 batch the whole way (get_initial_cache_state(2)).
  2. mid-stream stack: run 2 streams separately for k chunks (each B=1), then STACK their independent
     caches on the correct axes (channel/time dim 1, len dim 0) into a B=2 batch and continue —
     this is the REAL scheduler scenario (the documented dim-1 hazard).

Run under .venv-asr.  Usage: probe_batched_step.py
"""
import os
import sqlite3
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(REPO, "src") if (REPO := os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) else "src")
from nemotron_speech.batch_primitives import (  # noqa: E402
    stack_processed, stack_caches, scatter_cache_row, stack_hypotheses, stack_pred_out,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "stt-benchmark/stt_benchmark_data/results.db")
EN_NEMO = open("/tmp/en-nemo-path").read().strip()
SR = 16000


def log(*a):
    print(*a, flush=True)


def load_clip(sample_id):
    con = sqlite3.connect(DB)
    ap = con.execute("SELECT audio_path FROM samples WHERE sample_id=?", (sample_id,)).fetchone()[0]
    with open(os.path.join(REPO, "stt-benchmark", ap), "rb") as f:
        pcm = f.read()
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def main():
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    log(f"loading {EN_NEMO} ...")
    m = nemo_asr.models.ASRModel.restore_from(EN_NEMO, map_location="cuda")
    m.encoder.set_default_att_context_size([70, 1])
    m.change_decoding_strategy(decoding_cfg=OmegaConf.create(
        {"strategy": "greedy", "greedy": {"max_symbols": 10, "loop_labels": False,
                                           "use_cuda_graph_decoder": False}}))
    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    scfg = m.encoder.streaming_cfg
    log(f"streaming_cfg: chunk_size={scfg.chunk_size} shift_size={getattr(scfg,'shift_size',None)} "
        f"pre_encode_cache_size={getattr(scfg,'pre_encode_cache_size',None)} "
        f"drop_extra_pre_encoded={getattr(scfg,'drop_extra_pre_encoded',None)}")

    # chunk_size may be a [left,cur] list — use the current (feed) size.
    cs = scfg.chunk_size[1] if isinstance(scfg.chunk_size, (list, tuple)) else int(scfg.chunk_size)
    feat = m.cfg.preprocessor.features

    def _t(h):
        """Extract a comparable+displayable string from a Hypothesis (or '')."""
        if h is None or (isinstance(h, str) and h == ""):
            return ""
        if isinstance(h, str):
            return h
        txt = getattr(h, "text", None)
        if isinstance(txt, str) and txt:
            return txt
        ys = getattr(h, "y_sequence", None)
        if ys is not None:
            seq = ys.tolist() if hasattr(ys, "tolist") else list(ys)
            return "tok:" + ",".join(map(str, seq))
        return str(h)

    def to_mel(clip):
        sig = torch.tensor(clip, device="cuda").unsqueeze(0)
        ln = torch.tensor([sig.shape[1]], device="cuda")
        with torch.inference_mode():
            mel, mel_len = m.preprocessor(input_signal=sig, length=ln)  # [1,F,T]
        return mel

    def run_separate(mel, n_chunks):
        """Stream one clip at B=1; return list of cumulative transcripts per chunk."""
        cache = m.encoder.get_initial_cache_state(batch_size=1)
        clc, clt, clcl = cache[0], cache[1], cache[2]
        prev_hyp, prev_pred = None, None
        texts = []
        with torch.inference_mode():
            for i in range(n_chunks):
                chunk = mel[:, :, i * cs:(i + 1) * cs]
                if chunk.shape[-1] == 0:
                    break
                ln = torch.tensor([chunk.shape[-1]], device="cuda")
                pred_out, txt, clc, clt, clcl, prev_hyp = m.conformer_stream_step(
                    processed_signal=chunk, processed_signal_length=ln,
                    cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                    keep_all_outputs=False, previous_hypotheses=prev_hyp, previous_pred_out=prev_pred,
                    drop_extra_pre_encoded=0, return_transcription=True)
                prev_pred = pred_out
                texts.append(_t(txt[0]) if txt else "")
        return texts, (clc, clt, clcl, prev_hyp, prev_pred)

    def run_batched(mels, n_chunks, start=0, init=None):
        """Stream B=len(mels) clips as one batch; return per-row cumulative transcripts per chunk.
        Optionally start at chunk `start` from a provided init state (clc,clt,clcl,prev_hyp,prev_pred)
        — used to test MID-STREAM stacking of independent per-stream caches (the dim-1 hazard)."""
        B = len(mels)
        if init is None:
            cache = m.encoder.get_initial_cache_state(batch_size=B)
            clc, clt, clcl = cache[0], cache[1], cache[2]
            prev_hyp, prev_pred = None, None
        else:
            clc, clt, clcl, prev_hyp, prev_pred = init
        per_row = [[] for _ in range(B)]
        with torch.inference_mode():
            for i in range(start, start + n_chunks):
                slices = [mels[b][:, :, i * cs:(i + 1) * cs] for b in range(B)]
                if slices[0].shape[-1] == 0:
                    break
                chunk, ln = stack_processed(slices)  # batch_primitives
                pred_out, txt, clc, clt, clcl, prev_hyp = m.conformer_stream_step(
                    processed_signal=chunk, processed_signal_length=ln,
                    cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                    keep_all_outputs=False, previous_hypotheses=prev_hyp, previous_pred_out=prev_pred,
                    drop_extra_pre_encoded=0, return_transcription=True)
                prev_pred = pred_out
                for b in range(B):
                    per_row[b].append(_t(txt[b]) if txt and len(txt) > b else "")
        return per_row

    # --- pick 2 clips, equalize chunk count (lockstep, in-phase) ---
    ids = ["ae87f7c2-9b1e-...", "f90c183c-..."]  # resolved below by prefix
    con = sqlite3.connect(DB)
    def resolve(pfx):
        return con.execute("SELECT sample_id FROM samples WHERE sample_id LIKE ?", (pfx + "%",)).fetchone()[0]
    sid0 = resolve("ae87f7c2"); sid1 = resolve("f90c183c")
    log(f"clips: {sid0[:8]} (A), {sid1[:8]} (B); chunk_size(feed)={cs}, feat={feat}")
    mel0, mel1 = to_mel(load_clip(sid0)), to_mel(load_clip(sid1))
    nch = min(mel0.shape[-1], mel1.shape[-1]) // cs
    log(f"mel T: A={mel0.shape[-1]} B={mel1.shape[-1]} -> lockstep n_chunks={nch}")

    sepA, _ = run_separate(mel0, nch)
    sepB, _ = run_separate(mel1, nch)
    batched = run_batched([mel0, mel1], nch)

    finalA_sep, finalA_bat = sepA[-1] if sepA else "", batched[0][-1] if batched[0] else ""
    finalB_sep, finalB_bat = sepB[-1] if sepB else "", batched[1][-1] if batched[1] else ""
    matchA = sepA == batched[0]
    matchB = sepB == batched[1]
    log("\n=== SUB-TEST 1: batched-from-start (B=2) vs separate (B=1) ===")
    log(f"  A: per-chunk match={matchA}  final_sep={finalA_sep[:55]!r}  final_bat={finalA_bat[:55]!r}")
    log(f"  B: per-chunk match={matchB}  final_sep={finalB_sep[:55]!r}  final_bat={finalB_bat[:55]!r}")
    # row-order permutation: [B,A]
    batched_perm = run_batched([mel1, mel0], nch)
    permA = sepA == batched_perm[1]
    permB = sepB == batched_perm[0]
    log(f"  row-order permute [B,A]: A match={permA}  B match={permB}")

    # --- SUB-TEST 2: mid-stream dim-1 cache stacking (the REAL scheduler scenario + the hazard) ---
    k = nch // 2
    _, stA = run_separate(mel0, k)
    _, stB = run_separate(mel1, k)
    clcA, cltA, clclA, hypA_s, predA_s = stA
    clcB, cltB, clclB, hypB_s, predB_s = stB
    # use batch_primitives for the mid-stream stack (the real scheduler recipe)
    clc, clt, clcl = stack_caches([(clcA, cltA, clclA), (clcB, cltB, clclB)])
    hyp_stacked = stack_hypotheses([hypA_s, hypB_s])
    pred_stacked = stack_pred_out([predA_s, predB_s])
    cont = run_batched([mel0, mel1], nch - k, start=k,
                       init=(clc, clt, clcl, hyp_stacked, pred_stacked))
    contA = cont[0] == sepA[k:]
    contB = cont[1] == sepB[k:]
    log("\n=== SUB-TEST 2: mid-stream dim-1 cache stack (scheduler scenario) ===")
    log(f"  ran A,B separately {k} chunks, STACKED independent caches (dim1/dim0), continued batched {nch-k} chunks")
    log(f"  A continuation match={contA}  B continuation match={contB}")

    verdict = matchA and matchB and permA and permB and contA and contB
    log(f"\n=== PROBE B VERDICT: {'GO — byte-identical batched==separate (incl. mid-stream dim-1 stack)' if verdict else 'DIVERGENCE — investigate'} ===")
    if not verdict:
        # show first divergence chunk
        for arm, sep, bat in [("A", sepA, batched[0]), ("B", sepB, batched[1])]:
            for i, (s, b) in enumerate(zip(sep, bat)):
                if s != b:
                    log(f"  {arm} first diverges at chunk {i}: sep={s[:40]!r} bat={b[:40]!r}")
                    break


if __name__ == "__main__":
    main()
