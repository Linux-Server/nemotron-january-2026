# Conc-10 Pivot Findings

Step 2 did not produce a usable FULL_GRAPH small-B timing projection because max-B FULL_GRAPH replay failed at `B=1` after warm. The conc-10 UPSIDE gate is therefore a miss, and the FLOOR gate is also blocked until the replay failure is resolved or an exact-B capture design proves safe.

Observed eager-only decode sizing:

- steady weighted p50 at the conc-10 B distribution: `0.829 ms`
- finalize B=1 p50: `1.703 ms`
- records finalize decode wall p50/p95: `3.696 / 6.100 ms`

Alternative conc-10 p50/spread levers to pursue:

- Reduce finalize fork/clone double-clone cost at `server.py:6370`, `server.py:6480`, and `server.py:7371`.
- Add or promote one-shot finalize preprocessor work around `server.py:6927` and `server.py:7087`.
- Fix or retune reset-while-`PENDING_FINALIZE` debounce delay around `server.py:5823`.
- Add a global active-session/inflight admission cap around `server.py:4163` and `server.py:4326`.
