#!/usr/bin/env python3
"""Action E, step 1 — AOTI recurrent-drift characterization (review #3/#4). The action-D smoke probe was ONE 3s clip,
single token list. The dangerous failure mode is: the recurrent cache_t drift (1.66e-2 per steady step) COMPOUNDS
silently across many chunks until a near-tie argmax flips/deletes/inserts a token, with no byte-exact tripwire.

This streams a LONG continuous signal (N concatenated corpus clips) through TWO independent passes that each thread their
OWN recurrent caches forward — eager vs AOTI-steady — and logs PER CHUNK: cache_t / cache_ch / enc_out max-abs-diff
between the paths (does drift grow or plateau?), token-sequence divergence (first differing chunk), and the minimum
greedy argmax margin (top1-top2 logit) along each path (how close to a flip we ever got).

Run: HF_HUB_OFFLINE=1 /home/khkramer/src/parakeet/venv/bin/python aoti_drift_probe.py [N_CLIPS]
"""
from __future__ import annotations
import io, os, sys, numpy as np, torch, soundfile as sf
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr

ART = os.path.join(os.path.dirname(__file__), "artifacts")
BLANK, MAX_SYMBOLS = 1024, 10

@torch.inference_mode()
def decode_range_m(decoder, joint, f, t0, t1, state, g):
    """ref_greedy_range + min argmax margin (top1-top2) tracking. Algorithmically identical to the verified decode."""
    hyp = []; min_margin = float("inf")
    for t in range(t0, t1):
        f_t = f[:, t:t+1, :]; n_sym = 0
        while n_sym < MAX_SYMBOLS:
            logits = joint.joint(f_t, g).reshape(-1)
            top2 = torch.topk(logits, 2).values
            margin = (top2[0] - top2[1]).item()
            if margin < min_margin: min_margin = margin
            k = int(logits.argmax().item())
            if k == BLANK: break
            hyp.append(k)
            y = torch.full((1, 1), k, dtype=torch.long, device=f.device)
            g, state = decoder.predict(y, state, add_sos=False, batch_size=1)
            n_sym += 1
    return hyp, state, g, min_margin

def main():
    n_clips = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu").cuda().eval()
    try: m.preprocessor.featurizer.dither = 0.0
    except Exception: pass
    m.encoder.set_default_att_context_size([70, 1])
    m.change_decoding_strategy(decoding_cfg=OmegaConf.create(
        {"strategy": "greedy_batch", "greedy": {"max_symbols": 10, "loop_labels": True, "use_cuda_graph_decoder": False}}))
    e, dec, joint = m.encoder, m.decoder, m.joint
    sc = e.streaming_cfg; _int = lambda v: int(v[1]) if isinstance(v, (list, tuple)) else int(v)
    shift, pre, drop = _int(sc.shift_size), _int(sc.pre_encode_cache_size), int(sc.drop_extra_pre_encoded)
    dev = next(m.parameters()).device

    import datasets
    ds = datasets.load_dataset("pipecat-ai/stt-benchmark-data", split="train").cast_column("audio", datasets.Audio(decode=False))
    wavs = []
    for i in range(n_clips):
        w, sr = sf.read(io.BytesIO(ds[i]["audio"]["bytes"]), dtype="float32")
        if w.ndim > 1: w = w.mean(1)
        if sr != 16000:
            nn = int(len(w)*16000/sr); w = np.interp(np.linspace(0,len(w),nn,endpoint=False),np.arange(len(w)),w).astype(np.float32)
        wavs.append(w)
    wav = np.concatenate(wavs)
    audio = torch.tensor(wav, device=dev).unsqueeze(0); alen = torch.tensor([wav.shape[0]], device=dev)
    with torch.inference_mode():
        mel, _ = m.preprocessor(input_signal=audio, length=alen)
    Tm = mel.shape[-1]

    runner = torch._inductor.aoti_load_package(os.path.join(ART, "enc_steady_aoti.pt2"))
    def aoti_steady(chunk, L, clc, clt, clcl):
        out = runner(chunk, L, clc, clt, clcl); return out[0], out[1], out[2], out[3], out[4]

    # two independent passes, each threading its OWN caches + decode state
    def fresh():
        c = e.get_initial_cache_state(batch_size=1)
        st = dec.initialize_state(torch.zeros(1,1,dtype=torch.float32,device=dev))
        g, st = dec.predict(None, st, add_sos=False, batch_size=1)
        return [c[0].clone(), c[1].clone(), c[2].clone()], st, g
    E = {"cache": None, "st": None, "g": None, "ring": None, "tok": []}
    A = {"cache": None, "st": None, "g": None, "ring": None, "tok": []}
    E["cache"], E["st"], E["g"] = fresh()
    A["cache"], A["st"], A["g"] = fresh()
    minmarg_e = minmarg_a = float("inf")

    rows = []  # (chunk, cache_t_diff, cache_ch_diff, enc_out_diff, n_aoti_so_far)
    emitted = 0; pos = 0; ci = 0; n_aoti = 0
    with torch.inference_mode():
        while pos < Tm:
            nm = mel[:, :, pos:pos+shift]
            first = (emitted == 0)
            # ---- eager path ----
            ec = nm if first else torch.cat((E["ring"], nm), dim=-1)
            Le = torch.full((1,), ec.shape[-1], device=dev, dtype=torch.long); de = 0 if first else drop
            eo_e, el_e, E["cache"][0], E["cache"][1], E["cache"][2] = e.cache_aware_stream_step(
                processed_signal=ec, processed_signal_length=Le, cache_last_channel=E["cache"][0],
                cache_last_time=E["cache"][1], cache_last_channel_len=E["cache"][2], keep_all_outputs=False, drop_extra_pre_encoded=de)
            # ---- aoti path (AOTI for full 25-frame steady; eager for first + tail) ----
            ac = nm if first else torch.cat((A["ring"], nm), dim=-1)
            La = torch.full((1,), ac.shape[-1], device=dev, dtype=torch.long); da = 0 if first else drop
            use_aoti = (not first) and ac.shape[-1] == (pre + shift)
            if use_aoti:
                eo_a, el_a, A["cache"][0], A["cache"][1], A["cache"][2] = aoti_steady(ac, La, A["cache"][0], A["cache"][1], A["cache"][2]); n_aoti += 1
            else:
                eo_a, el_a, A["cache"][0], A["cache"][1], A["cache"][2] = e.cache_aware_stream_step(
                    processed_signal=ac, processed_signal_length=La, cache_last_channel=A["cache"][0],
                    cache_last_time=A["cache"][1], cache_last_channel_len=A["cache"][2], keep_all_outputs=False, drop_extra_pre_encoded=da)
            # per-chunk drift between the independently-threaded caches
            ct = (E["cache"][1].float() - A["cache"][1].float()).abs().max().item()
            cc = (E["cache"][0].float() - A["cache"][0].float()).abs().max().item()
            eod = (eo_e.float() - eo_a.float()).abs().max().item() if eo_e.shape == eo_a.shape else float("nan")
            rows.append((ci, ct, cc, eod, n_aoti))
            # decode each path over its own enc_out frames
            Toe = int(el_e[0]); Toa = int(el_a[0])
            fe = eo_e.transpose(1,2).contiguous(); fa = eo_a.transpose(1,2).contiguous()
            te, E["st"], E["g"], mm_e = decode_range_m(dec, joint, fe, 0, Toe, E["st"], E["g"]); E["tok"] += te; minmarg_e = min(minmarg_e, mm_e)
            ta, A["st"], A["g"], mm_a = decode_range_m(dec, joint, fa, 0, Toa, A["st"], A["g"]); A["tok"] += ta; minmarg_a = min(minmarg_a, mm_a)
            E["ring"] = (torch.cat((E["ring"], nm), dim=-1) if E["ring"] is not None else nm)[:, :, -pre:]
            A["ring"] = (torch.cat((A["ring"], nm), dim=-1) if A["ring"] is not None else nm)[:, :, -pre:]
            emitted += nm.shape[-1]; pos += shift; ci += 1

    cts = [r[1] for r in rows]; nchunk = len(rows)
    half = nchunk // 2
    first_half = sum(cts[:half])/max(half,1); second_half = sum(cts[half:])/max(nchunk-half,1)
    print(f"stream: {n_clips} clips concat -> {wav.shape[0]/16000:.1f}s, mel {Tm} frames, {nchunk} chunks ({n_aoti} AOTI-steady)")
    print(f"cache_t drift: max={max(cts):.3e} mean={sum(cts)/nchunk:.3e} | first-half mean={first_half:.3e} second-half mean={second_half:.3e} (grow ratio {second_half/max(first_half,1e-12):.2f}x)")
    print("per-chunk cache_t diff (every ~10th chunk):")
    for r in rows[::max(nchunk//25,1)]:
        print(f"  chunk {r[0]:4d}: cache_t={r[1]:.3e} cache_ch={r[2]:.3e} enc_out={r[3]:.3e}")
    # token divergence
    te2, ta2 = E["tok"], A["tok"]
    same = (te2 == ta2)
    first_div = next((i for i,(x,y) in enumerate(zip(te2,ta2)) if x!=y), None)
    print(f"tokens: eager {len(te2)} | aoti {len(ta2)} | identical={same}" + ("" if same else f" | first diff at token #{first_div} (len-eq={len(te2)==len(ta2)})"))
    print(f"min argmax margin (top1-top2 logit): eager={minmarg_e:.3f} aoti={minmarg_a:.3f}  (smaller = closer to a flip)")
    print(f"eager text: {m.tokenizer.ids_to_text(te2)!r}")
    print(f"aoti  text: {m.tokenizer.ids_to_text(ta2)!r}")
    print(f"=== DRIFT VERDICT: cache_t {'GROWS' if second_half>2*first_half else 'bounded/plateau'}; tokens {'MATCH' if same else 'DIVERGE'} over {nchunk} chunks ===")

if __name__ == "__main__":
    main()
