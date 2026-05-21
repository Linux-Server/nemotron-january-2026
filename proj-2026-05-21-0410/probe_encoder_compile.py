"""Probe A (focused) — can the encoder's cache-aware streaming step be torch.compile'd?

Phase 1 (single-stream) lever: collapse the per-call dispatch/launch overhead of the encoder's
`cache_aware_stream_step` via torch.compile(mode="reduce-overhead"). Tests feasibility (does it compile
without error?), correctness (compiled output == eager, allclose), and speedup at the steady-state
streaming-chunk shape. Secondary to batching; informational.

Run under .venv-asr.
"""
import os
import time

import torch

EN_NEMO = open("/tmp/en-nemo-path").read().strip()


def log(*a):
    print(*a, flush=True)


def main():
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.restore_from(EN_NEMO, map_location="cuda")
    m.encoder.set_default_att_context_size([70, 1])
    m.eval()
    scfg = m.encoder.streaming_cfg
    cs = scfg.chunk_size[1] if isinstance(scfg.chunk_size, (list, tuple)) else int(scfg.chunk_size)
    feat = m.cfg.preprocessor.features
    dev = "cuda"

    def fresh_inputs():
        cache = m.encoder.get_initial_cache_state(batch_size=1)
        x = torch.randn(1, feat, cs, device=dev)
        ln = torch.tensor([cs], device=dev)
        return x, ln, cache

    step = m.encoder.cache_aware_stream_step

    def call(fn, x, ln, cache):
        return fn(processed_signal=x, processed_signal_length=ln,
                  cache_last_channel=cache[0], cache_last_time=cache[1],
                  cache_last_channel_len=cache[2], keep_all_outputs=False, drop_extra_pre_encoded=0)

    x, ln, cache = fresh_inputs()
    with torch.inference_mode():
        eager_out = call(step, x, ln, cache)
    log(f"streaming-step shape: chunk[1,{feat},{cs}]; eager OK, returned {len(eager_out)} tensors")

    # try to compile
    try:
        cstep = torch.compile(step, mode="reduce-overhead")
        with torch.inference_mode():
            for _ in range(5):  # warmup / capture
                call(cstep, x, ln, cache)
            torch.cuda.synchronize()
            comp_out = call(cstep, x, ln, cache)
        # correctness: encoded (out[0]) allclose
        ok = torch.allclose(eager_out[0], comp_out[0], atol=1e-3, rtol=1e-3)
        maxdiff = (eager_out[0] - comp_out[0]).abs().max().item()
        log(f"compile: OK; encoded allclose={ok} (max|Δ|={maxdiff:.2e})")
    except Exception as e:
        import traceback
        log(f"compile FAILED: {type(e).__name__}: {str(e)[:200]}")
        log("  (Phase 1 NO-GO via torch.compile on this NeMo encoder step; manual CUDA-graph capture is the alternative.)")
        traceback.print_exc()
        return

    # timing (re-feed same inputs; op is deterministic)
    def bench(fn, iters=100):
        with torch.inference_mode():
            for _ in range(10):
                call(fn, x, ln, cache)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                call(fn, x, ln, cache)
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / iters * 1000.0

    eager_ms = bench(step)
    comp_ms = bench(cstep)
    speedup = eager_ms / comp_ms if comp_ms else 0
    log(f"\nstreaming-step latency: eager={eager_ms:.3f}ms  compiled={comp_ms:.3f}ms  speedup={speedup:.2f}×")
    go = ok and speedup >= 1.2
    log(f"=== PROBE A VERDICT: {'GO (compiles, correct, ≥1.2× faster)' if go else ('compiles+correct but <1.2× — marginal' if ok else 'NO-GO')} ===")


if __name__ == "__main__":
    main()
