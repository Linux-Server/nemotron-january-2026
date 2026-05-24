# Spike 0.3 — Free-threaded CPython 3.13t probe (SKELETON; run BLOCKED on a py3.13t env)

**Goal (PLAN §6 / 0.3):** the cheapest thesis probe and the **B4 branch decider**. B4 (free-threaded py3.13t) keeps
NeMo's Python decode/encoder and just removes the GIL + rethreads the scheduler — **if it closes the post-Python
residual, the entire native C++ rewrite is avoided.**

Two stages:
1. **Feasibility:** does PyTorch + NeMo even import & run a streaming chunk under free-threaded CPython? C-extensions must
   opt into `Py_mod_gil`; this is the maturity risk.
2. **Thesis test (the real one):** stand up a **real off-event-loop dispatcher** (not the single asyncio loop —
   `server.py:4456-4491`) on py3.13t with the actual stack, and measure **end-to-end scheduler tail under load** vs the
   post-Python baseline.

## Go / No-go
- **Go (→ choose B4):** the residual tail/density gap closes end-to-end AND free-threaded PyTorch/NeMo is
  production-stable.
- **No-go:** free-threaded wheels immature OR the dispatcher rewrite doesn't close the gap → B4 off the table; B1 (native)
  is the only remaining build path (if 0.0/0.1 still say "worth it").

## Run prerequisites — **BLOCKED**
- A CPython **3.13t** (free-threaded) environment with PyTorch + NeMo built/installed for it (likely a custom build;
  this is itself part of the feasibility finding).
- The post-Python baseline for the tail comparison.

`probe.py` is the stage-1 feasibility skeleton (import + single-chunk + a thread-fanout sanity check). Stage 2 (the
dispatcher rewrite + tail measurement) is described, not scaffolded, since it needs the real stack.
