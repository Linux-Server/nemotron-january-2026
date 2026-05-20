"""Probe A — NeMo decoding-config placement validation for the EOU plan.

Mirrors server.py:425-490 setup exactly (set_default_att_context_size →
change_decoding_strategy → streaming_cfg-driven chunk iteration). Tests three
decoding-cfg variants and asserts:
  - all three accept (change_decoding_strategy doesn't raise)
  - transcript identical to baseline (argmax-invariant to log_normalize)
  - both candidate placements populate Hypothesis.alignments + .frame_confidence

Usage: /home/khkramer/src/nemotron-nano-omni/.venv-asr/bin/python proj-2026-05-19-eou-endpointing/probe_a_decoding_cfg.py
"""

import sys, wave
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.dont_write_bytecode = True

import nemo.collections.asr as nemo_asr

MODEL_NAME = "nvidia/nemotron-speech-streaming-en-0.6b"
FIXTURE = Path("tests/fixtures/harvard_16k.wav")
RIGHT_CTX = 1


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_cfg(variant: str):
    if variant == "baseline":
        return OmegaConf.create({
            "strategy": "greedy",
            "greedy": {"max_symbols": 10, "loop_labels": False, "use_cuda_graph_decoder": False},
        })
    if variant == "flat_confidence_cfg":
        return OmegaConf.create({
            "strategy": "greedy",
            "preserve_alignments": True,
            "confidence_cfg": {
                "preserve_frame_confidence": True,
                "method_cfg": {"name": "entropy", "entropy_type": "tsallis",
                               "alpha": 0.5, "entropy_norm": "exp"},
            },
            "greedy": {"max_symbols": 10, "loop_labels": False, "use_cuda_graph_decoder": False},
        })
    if variant == "nested_greedy":
        return OmegaConf.create({
            "strategy": "greedy",
            "greedy": {
                "max_symbols": 10, "loop_labels": False, "use_cuda_graph_decoder": False,
                "preserve_alignments": True, "preserve_frame_confidence": True,
                "confidence_method_cfg": {"name": "entropy", "entropy_type": "tsallis",
                                          "alpha": 0.5, "entropy_norm": "exp"},
            },
        })
    raise ValueError(variant)


def run_stream(model, audio: np.ndarray, shift: int, drop_extra: int):
    """Single end-to-end stream over the clip; return (final_text, last_hyp)."""
    cache = model.encoder.get_initial_cache_state(batch_size=1)
    c_lc, c_lt, c_lcl = cache[0], cache[1], cache[2]
    prev_hyps = None
    prev_pred = None

    sig = torch.tensor(audio, dtype=torch.float32, device="cuda").unsqueeze(0)
    sig_len = torch.tensor([sig.shape[1]], device="cuda")
    proc, proc_len = model.preprocessor(input_signal=sig, length=sig_len)
    total = int(proc_len.item())

    text = ""
    last_hyp = None
    pos = 0
    is_first = True
    with torch.inference_mode():
        while pos < total:
            end = min(pos + shift, total)
            chunk = proc[:, :, pos:end]
            chunk_len = torch.tensor([chunk.shape[-1]], device="cuda")
            (prev_pred, texts, c_lc, c_lt, c_lcl, prev_hyps) = model.conformer_stream_step(
                processed_signal=chunk,
                processed_signal_length=chunk_len,
                cache_last_channel=c_lc,
                cache_last_time=c_lt,
                cache_last_channel_len=c_lcl,
                keep_all_outputs=False,
                previous_hypotheses=prev_hyps,
                previous_pred_out=prev_pred,
                drop_extra_pre_encoded=(0 if is_first else drop_extra),
                return_transcription=True,
            )
            if texts and texts[0]:
                t = texts[0]
                # NeMo's stream_step may return a Hypothesis or a str depending on version
                text = t.text if hasattr(t, "text") else str(t)
            if prev_hyps:
                last_hyp = prev_hyps[0]
            pos = end
            is_first = False
    return text, last_hyp


def main():
    audio = load_audio(FIXTURE)
    print(f"[loaded] {FIXTURE} {len(audio)} samples ({len(audio)/16000:.1f}s)")

    print("[loading] model ...", flush=True)
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME, map_location="cpu")
    model = model.cuda()
    model.encoder.set_default_att_context_size([70, RIGHT_CTX])
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    scfg = model.encoder.streaming_cfg
    shift = scfg.shift_size[1] if isinstance(scfg.shift_size, list) else scfg.shift_size
    drop_extra = scfg.drop_extra_pre_encoded
    if isinstance(drop_extra, list):
        drop_extra = drop_extra[1]
    print(f"[ready] shift_size={shift} drop_extra_pre_encoded={drop_extra}")

    results = {}
    for variant in ("baseline", "flat_confidence_cfg", "nested_greedy"):
        print(f"\n=== variant: {variant} ===")
        cfg = build_cfg(variant)
        try:
            model.change_decoding_strategy(decoding_cfg=cfg)
        except Exception as e:
            print(f"  change_decoding_strategy FAILED: {type(e).__name__}: {e}")
            results[variant] = {"ok": False, "stage": "cfg", "err": str(e)}
            continue
        try:
            text, hyp = run_stream(model, audio, shift, drop_extra)
        except Exception as e:
            print(f"  stream FAILED: {type(e).__name__}: {e}")
            results[variant] = {"ok": False, "stage": "stream", "err": str(e)}
            continue
        align_pop = hyp is not None and getattr(hyp, "alignments", None) is not None
        conf_pop = hyp is not None and getattr(hyp, "frame_confidence", None) is not None
        def _flat_len(x):
            if x is None: return 0
            while x and isinstance(x[0], list):
                x = [v for sub in x for v in sub]
                break
            return len(x) if isinstance(x, list) else "?"
        align_n = _flat_len(getattr(hyp, "alignments", None)) if align_pop else 0
        conf_n = _flat_len(getattr(hyp, "frame_confidence", None)) if conf_pop else 0
        results[variant] = {"ok": True, "text": text,
                            "align_populated": align_pop, "align_len": align_n,
                            "conf_populated": conf_pop, "conf_len": conf_n}
        print(f"  text ({len(text)} chars): '{text[:160]}'")
        print(f"  hyp.alignments populated: {align_pop}  (≈{align_n} entries)")
        print(f"  hyp.frame_confidence populated: {conf_pop}  (≈{conf_n} entries)")
        if conf_pop:
            flat = hyp.frame_confidence
            while flat and isinstance(flat[0], list):
                flat = [v for sub in flat for v in sub]
            if flat:
                fl = [float(v) for v in flat]
                print(f"  frame_confidence sample (first 5): {fl[:5]}  "
                      f"(min={min(fl):.4f} max={max(fl):.4f}  ⇒ entropy mode returns normalized confidence ∈[0,1] when in [0,1])")

    print("\n=== VERDICT ===")
    base = results.get("baseline", {})
    if not base.get("ok"):
        print(f"  baseline FAILED ({base.get('err','?')}) — cannot compare")
        return
    base_text = base["text"]
    print(f"  baseline transcript: '{base_text[:200]}'")
    for k in ("flat_confidence_cfg", "nested_greedy"):
        r = results.get(k, {})
        if not r.get("ok"):
            print(f"  {k}: FAILED at {r.get('stage')} ({r.get('err','?')})")
            continue
        tm = r["text"] == base_text
        ap = r["align_populated"]
        cp = r["conf_populated"]
        status = "OK" if (tm and ap and cp) else "PARTIAL"
        print(f"  {k}: {status}  transcript_matches_baseline={tm}  "
              f"alignments_populated={ap}  frame_confidence_populated={cp}")


if __name__ == "__main__":
    main()
