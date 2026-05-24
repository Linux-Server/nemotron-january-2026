# Spike 0.5 — Trace-driven batching simulator + graph-capacity model (SKELETON)

**Goal (PLAN §6 / 0.5):** validate or KILL the **3–5× throughput** claim and the steady-graph density claim, *before*
funding Phase 2. Today 86% of steady decodes are B=1 — "fill B" is a hypothesis, not a checkbox.

## What it does
Replays **real per-tick readiness traces** through a pure-Python scheduler model and reports the achievable **B
distribution** without adding latency, PLUS a **graph-bucket capacity model**: per-lane graph memory per B, expected
exact-B replay hit-rate from the B histogram, eager-fallback %, and L4/L40S memory headroom.

## Go / No-go (against PRE-REGISTERED thresholds in ../decision-template.md)
- **Go:** median/p95 B ≫ 1 (numeric) AND the graph pool fits L4/L40S at target K×lanes.
- B stays ~1 → **drop the 3–5× target**, re-run 0.0.
- B>1 but poor hit-rate / high eager-fallback / no memory headroom → **drop the steady-graph density claim**.

## Trace schema (one record per scheduler tick per session)
```
{ "t_ms": float,                 # tick time
  "session_id": str,
  "ready": bool,                 # batch_primitives.py:34-56 ready predicate satisfied
  "target_lang": str|null,       # batch key component (batch_primitives.py:24-31)
  "keep_all_outputs": bool,
  "drop_extra": int,
  "chunk_T": int,
  "decoder_state_fresh": bool,    # previous_hypotheses is None / pred_out_stream is None (server.py:4789-4812)
  "finalize_state": str,          # STREAMING | PENDING_FINALIZE | FINALIZED
  "lane_affinity": int|null,      # server.py:3295-3308
  "deadline_ms": float }          # max-wait (NEMOTRON_BATCH_MAX_WAIT_MS, server.py:665-670)
```

## Source of traces — **instrumentation SPEC only (NOT applied to server.py)**
Capturing real traces requires emitting the above per tick from the scheduler. The mapping (read-only — do not edit
server.py in this Wave):
- ready predicate: `batch_primitives.py:34-56`; batch key: `:24-31`; fresh/established flag: `server.py:4789-4812`
- dispatch sort / candidate selection: `server.py:4864-4918`; max-wait/size: `:665-670`; lane affinity: `:3295-3308`
- The metric/log schema parity required by 0.10 should carry these fields so capture is a logging add, not new plumbing.

**Until traces exist**, `simulate.py` runs on **synthetic arrival traces** (Poisson / bursty / phase-aligned) to
exercise the model and let us pre-register thresholds against plausible distributions.

## Run prerequisites
- Synthetic mode: runnable now (no GPU). `python simulate.py --synthetic`
- Real mode: **BLOCKED** on the post-Python server emitting the trace schema above.
