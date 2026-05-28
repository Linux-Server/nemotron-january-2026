#!/usr/bin/env python3
"""Action E.2 — the full-1000 T1 shadow (the real ship gate for the AOTI encoder). Streams every stt-benchmark
utterance through TWO pipelines that differ ONLY in the steady encoder step — eager vs AOTI-steady — and measures
whether AOTI introduces any transcription regression.

The production 1.95% baseline is SEMANTIC WER judged by Claude (ignores punctuation/plurals/contractions; needs an API
key) over the SERVER path (finalize/VAD); it is not reproducible here and not the question. The T1-encoder question is
the eager-vs-AOTI DELTA on the SAME pipeline under identical normalization. We report:
  - exact token-sequence divergence count (eager vs AOTI) — the cleanest signal,
  - traditional WER (jiwer + whisper EnglishTextNormalizer) for eager-vs-ref and AOTI-vs-ref + the delta (pipeline
    sanity vs ~2%, and the WER cost of any divergence),
  - all hypotheses saved to JSON so the few divergent utterances (if any) can get semantic scoring later.

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python aoti_full1000_shadow.py [N]
"""
from __future__ import annotations
import io, os, sys, json, time, numpy as np, torch, soundfile as sf
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from ref_decode import ref_greedy_range

ART = os.path.join(os.path.dirname(__file__), "artifacts")
OUT = os.path.join(ART, "full1000_shadow_results.json")

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
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

    runner = torch._inductor.aoti_load_package(os.path.join(ART, "enc_steady_aoti.pt2"))
    def aoti_steady(c, L, a, b, d): o = runner(c, L, a, b, d); return o[0], o[1], o[2], o[3], o[4]

    @torch.inference_mode()
    def stream(mel, use_aoti):
        Tm = mel.shape[-1]
        cs = e.get_initial_cache_state(batch_size=1); clc, clt, clcl = cs[0].clone(), cs[1].clone(), cs[2].clone()
        st = dec.initialize_state(torch.zeros(1,1,dtype=torch.float32,device=dev)); g, st = dec.predict(None, st, add_sos=False, batch_size=1)
        ring = None; toks = []; emitted = 0; pos = 0
        while pos < Tm:
            nm = mel[:, :, pos:pos+shift]; first = (emitted == 0)
            chunk = nm if first else torch.cat((ring, nm), dim=-1)
            L = torch.full((1,), chunk.shape[-1], device=dev, dtype=torch.long); dd = 0 if first else drop
            if use_aoti and (not first) and chunk.shape[-1] == (pre + shift):
                eo, el, clc, clt, clcl = aoti_steady(chunk, L, clc, clt, clcl)
            else:
                eo, el, clc, clt, clcl = e.cache_aware_stream_step(processed_signal=chunk, processed_signal_length=L,
                    cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl, keep_all_outputs=False, drop_extra_pre_encoded=dd)
            f = eo.transpose(1, 2).contiguous()
            t, st, g = ref_greedy_range(dec, joint, f, 0, int(el[0]), st, g); toks += t
            ring = (torch.cat((ring, nm), dim=-1) if ring is not None else nm)[:, :, -pre:]; emitted += nm.shape[-1]; pos += shift
        return toks

    import datasets
    ds = datasets.load_dataset("pipecat-ai/stt-benchmark-data", split="train").cast_column("audio", datasets.Audio(decode=False))
    N = min(N, len(ds))
    results = []; t0 = time.time()
    for i in range(N):
        ex = ds[i]; wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != 16000:
            nn = int(len(wav)*16000/sr); wav = np.interp(np.linspace(0,len(wav),nn,endpoint=False),np.arange(len(wav)),wav).astype(np.float32)
        audio = torch.tensor(wav, device=dev).unsqueeze(0); alen = torch.tensor([wav.shape[0]], device=dev)
        with torch.inference_mode(): mel, _ = m.preprocessor(input_signal=audio, length=alen)
        te = stream(mel, use_aoti=False); ta = stream(mel, use_aoti=True)
        results.append({"sample_id": ex["sample_id"], "ref": ex["transcription"],
                        "eager": m.tokenizer.ids_to_text(te), "aoti": m.tokenizer.ids_to_text(ta),
                        "tok_eq": te == ta})
        if (i+1) % 50 == 0:
            nd = sum(1 for r in results if not r["tok_eq"])
            print(f"  {i+1}/{N}  divergent={nd}  ({(time.time()-t0)/(i+1):.2f}s/utt)", flush=True)
            json.dump(results, open(OUT, "w"))
    json.dump(results, open(OUT, "w"))

    # metrics
    from whisper_normalizer.english import EnglishTextNormalizer
    import jiwer
    norm = EnglishTextNormalizer()
    refs = [norm(r["ref"]) for r in results]
    he = [norm(r["eager"]) for r in results]; ha = [norm(r["aoti"]) for r in results]
    keep = [i for i in range(len(refs)) if refs[i].strip()]  # jiwer needs non-empty refs
    R = [refs[i] for i in keep]; HE = [he[i] for i in keep]; HA = [ha[i] for i in keep]
    wer_e = jiwer.wer(R, HE); wer_a = jiwer.wer(R, HA)
    ndiv = sum(1 for r in results if not r["tok_eq"])
    div_ids = [r["sample_id"] for r in results if not r["tok_eq"]]
    print(f"\n=== FULL-{N} T1 SHADOW (eager vs AOTI steady encoder) ===")
    print(f"exact token divergences: {ndiv}/{N}" + (f"  ids={div_ids[:10]}{'...' if ndiv>10 else ''}" if ndiv else ""))
    print(f"traditional WER (whisper-normalized, {len(keep)} non-empty refs):")
    print(f"  eager: {wer_e*100:.3f}%")
    print(f"  aoti : {wer_a*100:.3f}%")
    print(f"  delta (aoti - eager): {(wer_a-wer_e)*100:+.4f} pp")
    print(f"results saved: {OUT}")
    print(f"=== AOTI encoder WER {'NEUTRAL (delta==0)' if wer_a==wer_e else f'delta {(wer_a-wer_e)*100:+.4f}pp'} | tokens {'IDENTICAL' if ndiv==0 else f'{ndiv} diverge'} ===")

if __name__ == "__main__":
    main()
