"""Probe C (focused) — is `strategy=greedy_batch` byte-exact vs the current `strategy=greedy` at B=1?

This gates whether the DECODE can be switched to the batched label-looping path (the bigger throughput
win). If greedy_batch == greedy byte-exact → decode batching is safe; else the plan falls back to
batching the encoder only + per-row greedy decode (lower ceiling). Same chunk feed both arms (consistency).

Run under .venv-asr.
"""
import os
import sqlite3

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(REPO, "stt-benchmark/stt_benchmark_data/results.db")
EN_NEMO = open("/tmp/en-nemo-path").read().strip()


def log(*a):
    print(*a, flush=True)


def main():
    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf

    m = nemo_asr.models.ASRModel.restore_from(EN_NEMO, map_location="cuda")
    m.encoder.set_default_att_context_size([70, 1])
    m.eval()
    m.preprocessor.featurizer.dither = 0.0
    scfg = m.encoder.streaming_cfg
    cs = scfg.chunk_size[1] if isinstance(scfg.chunk_size, (list, tuple)) else int(scfg.chunk_size)

    con = sqlite3.connect(DB)
    def clip(pfx):
        sid, ap = con.execute("SELECT sample_id,audio_path FROM samples WHERE sample_id LIKE ?", (pfx + "%",)).fetchone()
        with open(os.path.join(REPO, "stt-benchmark", ap), "rb") as f:
            pcm = f.read()
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    def to_mel(c):
        sig = torch.tensor(c, device="cuda").unsqueeze(0)
        with torch.inference_mode():
            mel, _ = m.preprocessor(input_signal=sig, length=torch.tensor([sig.shape[1]], device="cuda"))
        return mel

    def _t(h):
        if h is None or h == "":
            return ""
        if isinstance(h, str):
            return h
        t = getattr(h, "text", None)
        if isinstance(t, str) and t:
            return t
        ys = getattr(h, "y_sequence", None)
        return "tok:" + ",".join(map(str, ys.tolist())) if ys is not None else str(h)

    def stream(mel):
        cache = m.encoder.get_initial_cache_state(batch_size=1)
        clc, clt, clcl = cache
        ph, pp = None, None
        out = []
        nch = mel.shape[-1] // cs
        with torch.inference_mode():
            for i in range(nch):
                ch = mel[:, :, i * cs:(i + 1) * cs]
                ln = torch.tensor([ch.shape[-1]], device="cuda")
                pp, txt, clc, clt, clcl, ph = m.conformer_stream_step(
                    processed_signal=ch, processed_signal_length=ln,
                    cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                    keep_all_outputs=False, previous_hypotheses=ph, previous_pred_out=pp,
                    drop_extra_pre_encoded=0, return_transcription=True)
                out.append(_t(txt[0]) if txt else "")
        return out

    mels = [to_mel(clip(p)) for p in ("ae87f7c2", "f90c183c", "ab3a98fa")]

    def set_strategy(strat, loop_labels=False):
        cfg = {"strategy": strat, strat: {"max_symbols": 10}}
        if strat == "greedy":
            cfg[strat].update({"loop_labels": loop_labels, "use_cuda_graph_decoder": False})
        else:  # greedy_batch
            cfg[strat].update({"loop_labels": True, "use_cuda_graph_decoder": False})
        m.change_decoding_strategy(decoding_cfg=OmegaConf.create(cfg))
        m.eval()

    set_strategy("greedy")
    greedy_out = [stream(mel) for mel in mels]
    try:
        set_strategy("greedy_batch")
        gb_out = [stream(mel) for mel in mels]
    except Exception as e:
        log(f"greedy_batch setup/run FAILED: {type(e).__name__}: {str(e)[:160]}")
        log("=== PROBE C VERDICT: greedy_batch unavailable -> use encoder-only batching + per-row greedy fallback ===")
        return

    allmatch = True
    for i, (g, gb) in enumerate(zip(greedy_out, gb_out)):
        match = g == gb
        allmatch = allmatch and match
        log(f"  clip{i}: greedy==greedy_batch byte-exact={match}  "
            f"greedy_final={(g[-1] if g else '')[:45]!r}  gb_final={(gb[-1] if gb else '')[:45]!r}")
    log(f"\n=== PROBE C VERDICT: {'GO — greedy_batch byte-identical -> decode CAN be batched' if allmatch else 'NO-GO — greedy_batch differs -> encoder-only batching + per-row greedy fallback (lower ceiling)'} ===")


if __name__ == "__main__":
    main()
