#!/usr/bin/env python3
"""Round-2b probe: does bf16/fp16 cut the encoder's GPU-ACTIVE time (the ceiling lever once lanes fill the GPU),
and is the transcript stable (WER proxy)? Resolves the round-1 fp16 disagreement. Run ON the EC2 box.

  env -u LD_LIBRARY_PATH HF_HOME=$HOME/hf ~/nemo-venv/bin/python fp16_probe.py
"""
import contextlib
import glob
import os
import statistics
import time

import numpy as np
import torch

MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
AUDIO_DIR = os.path.expanduser("~/nemotron/loadgen_audio")
T_STEADY = 25  # rc1 steady encoder bucket (pre_encode_cache 9 + shift 16), per manual-cudagraph-probe
ITERS = 200


def log(*a):
    print(*a, flush=True)


import nemo.collections.asr as nemo_asr  # noqa: E402

torch.backends.cudnn.benchmark = True
log(f"loading {MODEL} ...")
try:
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL, map_location="cuda")
except Exception:
    cands = glob.glob(os.path.expanduser("~/hf/**/*.nemo"), recursive=True)
    model = nemo_asr.models.ASRModel.restore_from(cands[0], map_location="cuda")
model.encoder.set_default_att_context_size([70, 1])
model.eval()
model.preprocessor.featurizer.dither = 0.0
feat = int(model.cfg.preprocessor.features)
log(f"GPU: {torch.cuda.get_device_name(0)}  features={feat}  steady_T={T_STEADY}")


def time_encoder(autocast_dtype):
    cache = model.encoder.get_initial_cache_state(batch_size=1)
    processed = torch.randn(1, feat, T_STEADY, device="cuda", dtype=cache[0].dtype if autocast_dtype is None else torch.float32)
    plen = torch.full((1,), T_STEADY, device="cuda", dtype=torch.long)
    cm = (lambda: torch.autocast("cuda", dtype=autocast_dtype)) if autocast_dtype else contextlib.nullcontext

    def one():
        clc, clt, clcl = cache[0].clone(), cache[1].clone(), cache[2].clone()
        with torch.inference_mode(), cm():
            model.encoder.cache_aware_stream_step(
                processed_signal=processed, processed_signal_length=plen,
                cache_last_channel=clc, cache_last_time=clt, cache_last_channel_len=clcl,
                keep_all_outputs=False, drop_extra_pre_encoded=2)

    for _ in range(20):
        one()
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(ITERS):
        st.record(); one(); en.record(); torch.cuda.synchronize()
        times.append(st.elapsed_time(en))
    return statistics.mean(times), statistics.median(times)


def load_pcm_to_wav(pcm_path, wav_path):
    import soundfile as sf
    pcm = np.frombuffer(open(pcm_path, "rb").read(), dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(wav_path, pcm, 16000)
    return wav_path


def transcribe(paths, autocast_dtype):
    cm = (lambda: torch.autocast("cuda", dtype=autocast_dtype)) if autocast_dtype else contextlib.nullcontext
    with torch.inference_mode(), cm():
        out = model.transcribe(paths, batch_size=len(paths), verbose=False)
    return [getattr(h, "text", h) for h in out]


log("\n=== encoder cache_aware_stream_step GPU-active time (steady B=1) ===")
fp32_avg, fp32_p50 = time_encoder(None)
log(f"fp32       : avg {fp32_avg:.3f} ms  p50 {fp32_p50:.3f} ms")
for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    try:
        a, p = time_encoder(dt)
        log(f"{name}-autocast: avg {a:.3f} ms  p50 {p:.3f} ms   speedup {fp32_avg / a:.2f}x")
    except Exception as e:
        log(f"{name}-autocast: FAILED {type(e).__name__}: {e}")

log("\n=== transcript stability (offline) fp32 vs bf16 ===")
clips = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.pcm")))[:6]
wavs = [load_pcm_to_wav(c, f"/tmp/fp16probe_{i}.wav") for i, c in enumerate(clips)]
try:
    base = transcribe(wavs, None)
    bf = transcribe(wavs, torch.bfloat16)
    exact = sum(1 for a, b in zip(base, bf) if a == b)
    log(f"exact-match {exact}/{len(base)} clips")
    for i, (a, b) in enumerate(zip(base, bf)):
        if a != b:
            log(f"  clip{i} fp32: {a}")
            log(f"  clip{i} bf16: {b}")
except Exception as e:
    log(f"transcribe compare FAILED: {type(e).__name__}: {e}")

log("\nVERDICT: bf16 worthwhile iff encoder speedup is meaningful (~>1.3x) AND transcripts stay stable.")
