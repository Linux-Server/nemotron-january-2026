#!/usr/bin/env python3
"""1.1a — VERIFIED Python reference of the RNNT greedy decode (the executable spec for the C++ port).

Reimplements classic RNNT greedy (frame-loop + inner symbol-loop, max_symbols, blank advance) using ONLY
model.decoder (prediction net) + model.joint (joint net) — NOT NeMo's decode. Validates it produces BYTE-IDENTICAL
y_sequence to NeMo's deployed greedy_batch label-looping decode on every golden fixture. If it matches, we've proven
the algorithm exactly → the spec for B1b's C++ decode. (Greedy label-looping and frame-looping yield the same greedy
argmax sequence; this is the simplest correct reference.)

Run: /home/khkramer/src/parakeet/venv/bin/python ref_decode.py
"""
from __future__ import annotations
import glob, os, torch
import nemo.collections.asr as nemo_asr

BLANK = 1024
MAX_SYMBOLS = 10

@torch.inference_mode()
def ref_greedy(decoder, joint, enc, enc_len):
    """enc: [1,1024,T] (channel-first). Returns list[int] token ids (greedy, no blanks)."""
    f = enc.transpose(1, 2).contiguous()           # [1, T, 1024]
    T = int(enc_len[0])
    state = decoder.initialize_state(torch.zeros(1, 1, dtype=torch.float32, device=enc.device))
    g, state = decoder.predict(None, state, add_sos=False, batch_size=1)   # SOS pred output [1,1,640]
    hyp = []
    for t in range(T):
        f_t = f[:, t:t+1, :]                        # [1,1,1024]
        n_sym = 0
        while n_sym < MAX_SYMBOLS:
            logits = joint.joint(f_t, g)            # [1,1,1,1025]
            k = int(logits.reshape(-1).argmax().item())
            if k == BLANK:
                break
            hyp.append(k)
            y = torch.full((1, 1), k, dtype=torch.long, device=enc.device)
            g, state = decoder.predict(y, state, add_sos=False, batch_size=1)
            n_sym += 1
    return hyp

def main():
    model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b", map_location="cpu").cuda().eval()
    decoder, joint = model.decoder, model.joint
    fixtures = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "fixtures", "decode_*.pt")))
    n_pass = n_total = 0
    for fp in fixtures:
        d = torch.load(fp, weights_only=False)
        enc = d["enc"].cuda(); enc_len = d["enc_len"].cuda()
        gold = d["y_sequence"]; gold = gold.tolist() if torch.is_tensor(gold) else list(gold)
        got = ref_greedy(decoder, joint, enc, enc_len)
        ok = (got == gold)
        n_total += 1; n_pass += int(ok)
        name = os.path.basename(fp)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: ref {len(got)} tok vs gold {len(gold)} tok"
              + ("" if ok else f"\n   gold[:20]={gold[:20]}\n   ref [:20]={got[:20]}"))
    print(f"\n=== {n_pass}/{n_total} fixtures byte-exact ===")
    print("VERIFIED REFERENCE" if n_pass == n_total else "DIVERGENCE — investigate before C++ port")

if __name__ == "__main__":
    main()
