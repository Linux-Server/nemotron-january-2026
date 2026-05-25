# T2a — byte-exact streaming encoder (ACHIEVED via torch.export)

## Result
`export_t2a.py`: **`torch.export` of the steady `cache_aware_stream_step` is BYTE-EXACT vs eager across all 19 steady
chunks (max_abs_diff = 0.000e+00)** — including the varying `cache_channel_len` (0→70) that made the `torch.jit.trace`
version drift ~1e-5. Saved `enc_steady_t2a.pt2`.

## Root cause (confirmed)
`torch.jit.trace` records ops with concrete values and **bakes cache_len-dependent ops** (the attention mask keyed to
the valid cache length) at the trace-time cache_len → ~1e-5 drift at other cache_lens. `torch.export` (dynamo graph
capture) represents those as **symbolic tensor ops** → correct for any cache_len → byte-exact. (And it exported cleanly,
so `cache_aware_stream_step`'s cache_len usage is pure tensor ops, no data-dependent Python control flow needing
torch.cond.)

## Effect on the correctness bars
- **T2a (encoder byte-exact across the stream): ACHIEVED** at the model-export level (0.000e+00).
- This **eliminates the near-tie argmax-flip risk** the paired review flagged (BLOCKER@byte-exact) — the encoder is now
  exactly eager, so the greedy decode is exactly NeMo's.
- The first-chunk geometry should get the same treatment (export it too); steady is the one that varies with cache_len.

## C++ integration path (the remaining wiring) + its blocker
`torch.export` produces an **`ExportedProgram` (.pt2)**, which **`torch::jit::load` (TorchScript) cannot load**. To run
it byte-exact in the C++ runtime: **AOTInductor**-compile the ExportedProgram to a `.so` (`torch._inductor.aot_compile`)
and load via the AOTI C++ runtime.
- **Blocker on THIS box:** AOTInductor generates + compiles CUDA code → needs a working `nvcc`, which **glibc 2.41
  breaks** (the same wall as Wave-2 kernels). → the byte-exact C++ encoder must be built in a **CUDA devel container /
  Ubuntu-22.04 build env** (consistent with the earlier finding). On the L40S box (Ubuntu 22.04) AOTI would work.

## Net
- **T1 ships now**: the C++ streaming runtime (hardened) uses the token-exact traces — correct transcripts, ships.
- **T2a is proven**: a byte-exact streaming encoder export exists (0.000e+00); wiring it into C++ = AOTInductor in a
  glibc-compatible build env (container/L40S). A clean, well-scoped integration step, not open research.
