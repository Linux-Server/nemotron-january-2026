#!/usr/bin/env python3
"""1.3b-enc-prod: token-exactness shadow for the drop=0 (first-finalize) buckets — the analog of finalize_corpus_shadow
for the short-utterance case the corpus never hits. For a sweep of short truncated audio (forcing emitted_frames==0 ->
drop_extra=0), compare the BUCKETED drop=0 finalize tokens (route to stripped enc_finalize_d0_T{T}.pt2 + load_constants
shared + run + decode-continue) against the eager finalize_ref tokens. Closes the drop=0 T1 gap (strip-validate used
basis=unstripped_package = stripped==unstripped; the compile self-check was encoder-level within atol; this is the
token-level AOTI-vs-finalize_ref check).

Run: HF_HUB_OFFLINE=1 ./.venv/bin/python finalize_drop0_shadow.py
"""
from __future__ import annotations
import glob, os, re, torch
from finalize_ref import ContinuousFinalizeRef, load_model, load_benchmark_dataset, load_wav
from ref_decode import ref_greedy_range

ART = "artifacts"; BD = os.path.join(ART, "stripped_finalize_buckets")


def load_buckets():
    W = torch.load(os.path.join(ART, "finalize_shared_weights.pt"), weights_only=False)
    cuda = {}
    runners = {}
    for p in glob.glob(os.path.join(BD, "enc_finalize_d0_T*.pt2")):
        if "_stripped" in p: continue
        T = int(re.search(r"_T(\d+)\.pt2$", p).group(1))
        r = torch._inductor.aoti_load_package(p)
        fqns = r.loader.get_constant_fqns()
        cmap = {}
        for f in fqns:
            key = f if f in W else ("e." + f[len("encoder."):] if f.startswith("encoder.") and ("e." + f[len("encoder."):]) in W else f)
            if key not in W: raise RuntimeError(f"missing shared weight for {f}")
            if key not in cuda: cuda[key] = W[key].cuda()
            cmap[f] = cuda[key]
        r.loader.load_constants(cmap, False, False, True)
        runners[T] = r
    return runners


@torch.inference_mode()
def main():
    rt = ContinuousFinalizeRef(load_model()); g = rt.geometry
    enc, dec, joint = rt.encoder, rt.decoder, rt.joint
    runners = load_buckets()
    print(f"loaded {len(runners)} drop0 bucket runners: T={sorted(runners)}")
    ds = load_benchmark_dataset(); base = load_wav(ds[0])
    max_len = (g.shift_frames + 1) * g.hop_samples
    n = exact = 0; miss = []; div = []
    for L in range(g.hop_samples, max_len + g.hop_samples, g.hop_samples):
        s = rt.new_session("d0"); rt.append_audio(s, base[:L])
        if s.emitted_frames != 0: continue
        rt.vad_stop(s)
        fork = rt.build_continuous_finalize_fork(s); fi = rt.prepare_finalize_inputs(fork)
        if fi is None or int(fi.drop_extra) != 0: continue
        T = int(fi.chunk_mel.shape[-1]); n += 1
        if T not in runners: miss.append(T); continue
        # eager finalize_ref tokens
        eo, el, *_ = enc.cache_aware_stream_step(processed_signal=fi.chunk_mel, processed_signal_length=fi.chunk_len,
            cache_last_channel=fi.cache_last_channel.clone(), cache_last_time=fi.cache_last_time.clone(),
            cache_last_channel_len=fi.cache_last_channel_len.clone(), keep_all_outputs=True, drop_extra_pre_encoded=0)
        import copy
        te,_,_ = ref_greedy_range(dec, joint, eo.transpose(1,2).contiguous(), 0, int(el[0]),
                                  copy.deepcopy(fork.decoder_state), fork.pred_out_stream.clone())
        # bucketed tokens
        out = runners[T](fi.chunk_mel.contiguous(), fi.cache_last_channel.contiguous(),
                         fi.cache_last_time.contiguous(), fi.cache_last_channel_len.contiguous())
        bo, bl = out[0], out[1]
        tb,_,_ = ref_greedy_range(dec, joint, bo.transpose(1,2).contiguous(), 0, int(bl[0]),
                                  copy.deepcopy(fork.decoder_state), fork.pred_out_stream.clone())
        ok = (list(fork.hyp_tokens)+te) == (list(fork.hyp_tokens)+tb)
        exact += int(ok)
        if not ok: div.append((T, len(te), len(tb)))
        print(f"  drop0 T={T}: eager_tok={len(te)} bucket_tok={len(tb)} TOKEN_EXACT={ok}")
    print(f"=== drop0 shadow: {exact}/{n} token-exact; missing-bucket T={miss}; divergent={div} -> {'PASS' if exact==n and not miss else 'FAIL'} ===")


if __name__ == "__main__":
    main()
