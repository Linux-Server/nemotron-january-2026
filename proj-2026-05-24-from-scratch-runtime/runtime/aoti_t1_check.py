#!/usr/bin/env python3
"""Action D follow-up — AOTI is NOT byte-exact (aot_compile.py: cache_t drifts 1.66e-2). The decision-relevant question
is whether that drift survives to TOKENS: is the AOTI-compiled steady encoder T1 (token-exact) even though it fails T2a
(byte-exact)? T1 is the actual ship gate. Run the streaming loop with the AOTI runner on the steady chunks (first chunk
+ any non-25-frame tail stay eager — the export was specialized to the 9+16=25 steady geometry), decode with the
verified ref decode, and compare the emitted token sequence vs the full-eager NeMo streaming reference.

The AOTI .so was compiled in the container (sm_120); this loads it on the host (same torch 2.8.0+cu128, same 5090) — no
nvcc needed at load. Run: HF_HUB_OFFLINE=1 ./.venv/bin/python aoti_t1_check.py
"""
from __future__ import annotations
import io, os, numpy as np, torch, soundfile as sf
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from ref_decode import ref_greedy_range

ART = os.path.join(os.path.dirname(__file__), "artifacts")

def run_stream(m, e, dec, joint, mel, shift, pre, drop, dev, steady_fn, label):
    """Stream mel through `steady_fn` for the 25-frame steady chunks; eager for first + tail. Return token list."""
    Tm = mel.shape[-1]
    cache = e.get_initial_cache_state(batch_size=1)
    clc, clt, clcl = cache[0].clone(), cache[1].clone(), cache[2].clone()
    ring = None
    my_state = dec.initialize_state(torch.zeros(1, 1, dtype=torch.float32, device=dev))
    g, my_state = dec.predict(None, my_state, add_sos=False, batch_size=1)
    toks_all = []; n_aoti = 0; n_eager = 0
    emitted = 0; pos = 0
    with torch.inference_mode():
        while pos < Tm:
            new_mel = mel[:, :, pos:pos+shift]
            if emitted == 0:
                chunk = new_mel; d = 0; use = "eager-first"
            else:
                chunk = torch.cat((ring, new_mel), dim=-1); d = drop
                use = "steady"
            L = torch.full((1,), chunk.shape[-1], device=dev, dtype=torch.long)
            if use == "steady" and steady_fn is not None and chunk.shape[-1] == (pre + shift):
                enc_out, enc_len_out, clc, clt, clcl = steady_fn(chunk, L, clc, clt, clcl); n_aoti += 1
            else:
                enc_out, enc_len_out, clc, clt, clcl = e.cache_aware_stream_step(
                    processed_signal=chunk, processed_signal_length=L, cache_last_channel=clc,
                    cache_last_time=clt, cache_last_channel_len=clcl, keep_all_outputs=False, drop_extra_pre_encoded=d)
                n_eager += 1
            To = int(enc_len_out[0])
            f = enc_out.transpose(1, 2).contiguous()
            t, my_state, g = ref_greedy_range(dec, joint, f, 0, To, my_state, g)
            toks_all += t
            ring = (torch.cat((ring, new_mel), dim=-1) if ring is not None else new_mel)[:, :, -pre:]
            emitted += new_mel.shape[-1]; pos += shift
    print(f"[{label}] chunks: {n_aoti} via steady_fn + {n_eager} eager -> {len(toks_all)} tok")
    return toks_all

def main():
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
    wav, sr = sf.read(io.BytesIO(ds[1]["audio"]["bytes"]), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if sr != 16000:
        n = int(len(wav)*16000/sr); wav = np.interp(np.linspace(0,len(wav),n,endpoint=False),np.arange(len(wav)),wav).astype(np.float32)
    audio = torch.tensor(wav, device=dev).unsqueeze(0); alen = torch.tensor([wav.shape[0]], device=dev)
    with torch.inference_mode():
        mel, _ = m.preprocessor(input_signal=audio, length=alen)

    # AOTI runner (the container-compiled .so), loaded on the host
    # aot_compile.py builds the .so with -Wl,-z,noexecstack, so PT_GNU_STACK is non-exec and the hardened host kernel
    # loads it directly (no binary patching).
    runner = torch._inductor.aoti_load_package(os.path.join(ART, "enc_steady_aoti.pt2"))
    def aoti_steady(chunk, L, clc, clt, clcl):
        out = runner(chunk, L, clc, clt, clcl)
        return out[0], out[1], out[2], out[3], out[4]

    eager_tokens = run_stream(m, e, dec, joint, mel, shift, pre, drop, dev, None, "all-eager")
    aoti_tokens  = run_stream(m, e, dec, joint, mel, shift, pre, drop, dev, aoti_steady, "aoti-steady")
    ok = (eager_tokens == aoti_tokens)
    print(f"all-eager : {m.tokenizer.ids_to_text(eager_tokens)!r}")
    print(f"aoti-steady: {m.tokenizer.ids_to_text(aoti_tokens)!r}")
    print(f"=== AOTI steady-encoder T1 (token-exact vs all-eager): {ok} ===")
    if not ok:
        print("  eager:", eager_tokens[:40]); print("  aoti :", aoti_tokens[:40])

if __name__ == "__main__":
    main()
