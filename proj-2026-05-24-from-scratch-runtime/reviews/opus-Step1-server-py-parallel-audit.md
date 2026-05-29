# Step 1 — independent Opus parallel audit of `src/nemotron_speech/server.py`

Written in parallel with the Codex audit (job `bkddcxjjd`) per PLAN_RULES.md
"PAIRED adversarial review" for Step 1 (decision-critical). This doc captures
the Opus pass; once Codex lands, both fold into `reviews/server-py-protocol-audit.md`.

server.py: 10,296 lines. Framework: **aiohttp** (`web.WebSocketResponse`,
`app.router.add_get`, `web.json_response`). Not FastAPI/Flask as some docs
might imply elsewhere.

## Headline findings (deltas vs v5 architecture)

1. **/health enum has TWO values, not four.** v5 §VII claimed
   `loading|healthy|draining|degraded`. Python (`server.py:10204-10210`)
   actually returns `"healthy" if self.model_loaded else "loading"`. No
   draining state, no degraded state. The C++ port adding extra enum values
   would extend (not match) Python — needs an explicit decision.

2. **`finalize_silence_ms` default in code = 150, NOT 0.** `server.py:87`
   (`_DEFAULT_FINALIZE_SILENCE_MS = 150`). The 0 is a deploy override (the
   `silence0_warm200` config). The architecture v5 said default 0; that's the
   production-deploy value, but the code default is 150ms debounce.

3. **`NEMOTRON_WARMUP_MS` ≠ a VAD cancellation hold.** v5 §VI conflated this
   with a debounce-window cancellation. The actual semantics
   (`server.py:918, 2320-2327`):
   - `NEMOTRON_WARMUP_MS` (`session_warmup_ms`) = synthetic PRE-ROLL audio in
     milliseconds, fed into the encoder at session start to warm the model.
     Computed as `pre_encode_cache_size` worth of mel frames. Completely
     unrelated to VAD.
   - `NEMOTRON_FINALIZE_SILENCE_MS` (`finalize_silence_ms`) = THE debounce
     window after `vad_stop` before invoking finalize. **A subsequent
     `vad_start` during this same window cancels the debounce**, not a
     separate "warmup hold."
   - So "silence0_warm200" = `FINALIZE_SILENCE_MS=0 + WARMUP_MS=200`:
     - 0ms debounce = finalize immediately on `vad_stop` (no wait, no
       cancellation possible).
     - 200ms session pre-roll = inject 200ms of synthetic frames before
       real audio (model-warmup-only, not VAD-related).
   - **v5 architecture §VI's "VAD_WARMUP_MS = cancellation hold" semantic
     is wrong**. The cancellation happens DURING the `FINALIZE_SILENCE_MS`
     window itself.

4. **Silero: ZERO HITS confirmed.** `grep -i silero|vad_model|VADInfer|webrtcvad
   src/nemotron_speech/server.py` → no matches. Client (Pipecat) runs VAD
   externally; server consumes the `vad_start`/`vad_stop` control messages
   and applies (or skips, at default 0) the debounce.

## HTTP routes (DEFINITIVE)

### GET /health
Handler: `server.py:10202-10211`
Response shape:
```json
{
  "status": "healthy" | "loading",   // enum: 2 values only
  "model_loaded": <bool>,
  "admission": { ... }               // OPTIONAL; only if self.admission_enabled
}
```
`status` is `"healthy"` iff `model_loaded == True` (no separate loading state
during model swap; just true/false). NO `pid`, NO `process_label`, NO
`draining`, NO `degraded` enum values.

### GET /stats[?last=N]
Handler: `server.py:5183-5193` → `_stats_snapshot` at `server.py:5141-5177`.
Response shape:
```json
{
  "enabled": <bool>,
  "window_size": <int>,                              // default 2048
  "samples": <int>,                                  // = len(deque) within window
  "since_unix": <float|null>,                        // timestamp of oldest sample (null if empty)
  "until_unix": <float|null>,                        // timestamp of newest sample
  "emitted_in_window": <int>,                        // count where emitted_to_client==True
  "suppressed_in_window": <int>,                     // = samples - emitted_in_window
  "lifetime_emitted": <int>,                         // process-lifetime counter
  "lifetime_suppressed": <int>,                      // process-lifetime counter
  "metrics": {
    "vad_stop_to_sent_ms":          {p50, p90, p95, p99, max, count},
    "fork_flush_wall_ms":           {p50, p90, p95, p99, max, count},
    "vad_stop_recv_to_process_ms":  {p50, p90, p95, p99, max, count},
    "lock_wait_ms":                 {p50, p90, p95, p99, max, count},
    "vad_stop_to_finalize_start_ms":{p50, p90, p95, p99, max, count}
  },
  "active_sessions_at_emit": {p50, p90, p95, p99, max, count},
  "admission": { ... }                               // same shape as in /health
}
```

Each `{p50..max, count}` block per `_compute_quantile_summary`
(`server.py:97-118`). Empty list returns
`{"p50":null,"p90":null,"p95":null,"p99":null,"max":null,"count":0}`.

Query param: `?last=<positive int>`. Invalid (`?last=abc` or `?last=0`)
returns **HTTP 400** with `{"error": "invalid 'last': '<raw>'"}`
(`server.py:5183-5193`).

### Admission sub-object (DEFINITIVE)
Implementation: `_admission_status_snapshot` at `server.py:5075-5084`.
Shape:
```json
{
  "enabled": <bool>,
  "attempted": <int>,           // total connections offered
  "admitted": <int>,            // total accepted
  "rejected": <int>,            // total shed
  "max_backlog": <int>,         // peak backlog observed
  "max_ready_age_ms": <float>,  // peak time a ready session has waited
  "signal": {                   // _admission_backlog_signal (server.py:5032+)
     "queued_events": <int>,
     "oldest_ready_age_ms": <float>,
     "oldest_ready_session_id": <str|null>
     // ... possibly more; full read of _admission_backlog_signal needed
  }
}
```

This is the exact Python wire shape the C++ DensityAdmission must serialize
INTO at `/stats` serialization time. v5 §IV "Python-exact" requirement met
by writing this shape from DensityAdmission's native counters.

## WS endpoint

Path: `GET /` (with WS upgrade headers).
Handler: `websocket_handler` at `server.py:5195`.
Frame size limit: `web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)` =
**10 MiB exactly** (`server.py:5199`).

### Connection setup
1. `await ws.prepare(request)` (aiohttp WS handshake).
2. Validate query params via `_validate_connection_query(request.query)`
   (`server.py:1493-1498`): validates `?model=` and `?language=`. On invalid:
   - Caught by the outer `try/except` (`server.py:5202-5207`).
   - Logs warning.
   - Sends `{"type": "error", "message": str(e)}` via `ws.send_str`.
   - Calls `ws.close()` (default code = 1000).
   - **Note**: error+close happens POST-handshake (WS already prepared);
     it's not an HTTP 400 pre-upgrade. This means the connection upgrades
     successfully first, THEN gets the error frame + clean close.

### Admission decision
At `server.py:5229`: `await ws.close(code=1013, message=b"admission_backpressure")`
when admission shed. **Post-handshake close with code 1013.**
**No HTTP 503 pre-upgrade path in Python** — Python does the full WS
handshake first, then closes with 1013 if shed. This contradicts v5 §VII's
"HTTP 503 pre-handshake" plan. **Decision needed**: the C++ port can EITHER:
(a) match Python (handshake-then-1013); (b) deviate (HTTP 503 pre-upgrade
saves the handshake cost + lets HTTP load balancers see the shed).
v5 §VII picked (b) without flagging this as a deviation. **Open question.**

### Control messages
Three handler paths in server.py (basic / continuous / scheduler) — see
`server.py:5275-5282`, `5351-5356`, `5425-5430`. The handlers differ:

**Basic mode** (no `continuous_context`, no `scheduler_enabled`):
- `reset`/`end` → `_reset_session(session, finalize=data.get("finalize", True))`.
  - `finalize=True` (default): hard reset with padding + keep_all_outputs.
  - `finalize=False`: soft reset, return current text.
- `vad_start`/`vad_stop` → log `received {type} (no-op)` and ignore.
- Unknown type → log warning, ignore (forward-compat).
- Invalid JSON → log warning, ignore.

**Continuous mode** (`continuous_context=True`, no scheduler):
- Enqueues events onto `session.continuous_event_queue` for a per-session
  worker to process.
- `vad_stop` records `session.continuous_vad_stop_recv_ts` IF
  `finalize_profile_enabled` (for the I/O-gap probe metric).

**Scheduler mode** (`scheduler_enabled=True`, production):
- Enqueues events onto the central scheduler via
  `_scheduler_queue_event(session, event)`.
- The scheduler loop applies the `finalize_silence_ms` debounce + handles
  cancellation on subsequent `vad_start`.

### The debounce + cancellation state machine (production path)

In the scheduler loop (around `server.py:5925-6800`):
- `vad_stop` event arrives → if `finalize_silence_seconds > 0`, set
  `session.continuous_debounce_expiry_ts = now + finalize_silence_seconds`,
  enqueue a `("debounce_expired",)` event scheduled for the expiry time.
- `vad_start` event arrives → if a debounce is pending, cancel it
  (`_continuous_cancel_debounce_locked`).
- `debounce_expired` event fires (timer match) → check the debounce wasn't
  cancelled (stale-seq check); if still valid, invoke finalize.
- If `finalize_silence_seconds == 0` (production silence0_warm200 config):
  finalize is invoked immediately on `vad_stop` (no debounce, no cancellation
  possible).

### Wire format (server → client)

All text JSON frames via `ws.send_str(json.dumps(...))`. Discovered shapes:

1. `{"type": "ready"}` — sent ONCE after handshake completes
   (`server.py:5259`). No other fields.
2. `{"type": "error", "message": "<error text>"}` — on connection error or
   query validation failure (`server.py:5206, 5300-5310`).
3. `{"type": "transcript", "text": "<text>", "is_final": <bool>}` — interim
   transcript, no extra fields (`server.py:6293-6304`).
4. `{"type": "transcript", "text": "<text>", "is_final": true, "finalize_timing": {...}}` —
   FINAL transcript with timing block (`server.py:6502+, 6624+`). Other
   optional fields depending on path (need full read of those 2 paths).

### Wire format (client → server)

- `WSMsgType.BINARY`: PCM audio (`_handle_audio(session, msg.data)`). int16
  LE 16kHz mono per the documented contract (need to confirm in code —
  search for `dtype=np.int16` / `frombuffer`).
- `WSMsgType.TEXT`: JSON control messages (`reset`/`end`/`vad_start`/`vad_stop`).
- `WSMsgType.ERROR`: logged + connection break.
- `WSMsgType.CLOSE/CLOSING/CLOSED`: enqueues `("close",)` event.

### finalize_timing keys

The 5 SLO metrics referenced by `/stats` are confirmed. There are ALSO
additional internal fields in the per-finalize profile dict (`server.py:2929-2940`):
- `debounce_wait_ms` — vad_stop → debounce_expiry duration (in-window wait).
- `debounce_to_finalize_start_ms` — debounce_expiry → fork_flush_start (pickup delay).
- `finalize_done_to_sent_ms` — fork_flush_done → final_sent (emit lag).

**Open question**: are these fields ALSO sent on the wire in
`finalize_timing`, or are they only in the internal profile dict that feeds
`/stats`? Need to inspect the actual `ws.send_str` call for the FINAL
transcript (around `server.py:6502` and `6624`).

## Close codes used in Python (DEFINITIVE)

From `grep -nE "ws\.close|WSCloseCode" src/nemotron_speech/server.py`:
- `await ws.close()` (default = 1000) at `server.py:5207` — connection error
  / validation failure path.
- `await ws.close(code=1013, message=b"admission_backpressure")` at
  `server.py:5229` — admission shed.

That's it. **Python only uses codes 1000 and 1013**.
- v5 §VIII's expansive table (1000/1001/1003/1008/1009/1011/1013) is
  ASPIRATIONAL for the C++ port, not a Python-parity requirement. Most of
  those codes (1003 unsupported-data, 1009 message-too-big, 1011 internal
  error, 1001 going-away) would be additions, not parity.
- For Python parity, the C++ port MUST use 1000 + 1013. Other codes are
  best-practice additions that improve operator observability but don't
  match Python's behavior.

## Quantile formula (CONFIRMED)

`_compute_quantile_summary` at `server.py:97-118`:
```python
def percentile(p: float) -> float:
    idx = max(0, min(n - 1, int(round(p * (n - 1)))))
    return float(sorted_values[idx])
```
Nearest-rank with `round(p * (n-1))`, clamped to `[0, n-1]`. Single value →
all quantiles = that value. Matches v5 §IV exactly.

## Stats sliding window deque tuple shape

From `_record_stats_sample` (`server.py:5086-5138`) and `_stats_snapshot`
(`server.py:5141-5177`):
- Each sample = a tuple of 8 elements:
  - `[0]` = timestamp (unix seconds).
  - `[1]` = vad_stop_to_sent_ms.
  - `[2]` = fork_flush_wall_ms.
  - `[3]` = vad_stop_recv_to_process_ms.
  - `[4]` = lock_wait_ms.
  - `[5]` = vad_stop_to_finalize_start_ms.
  - `[6]` = active_sessions_at_emit.
  - `[7]` = emitted_to_client (bool).
- Append is atomic (CPython deque.append).
- Window size from `self.stats_window_size` (env: `NEMOTRON_STATS_WINDOW`).
- Enable from `self.stats_enabled` (env: `NEMOTRON_STATS_ENABLED`).

## Open questions / verification gaps

1. **Pre-handshake admission shed vs post-handshake 1013**: v5 §VII picked
   pre-handshake HTTP 503, Python uses post-handshake 1013. **Decision
   needed** for the C++ port: match Python, or deviate with documented
   rationale.
2. **`finalize_timing` keys actually sent on the wire**: need to read the
   exact `ws.send_str` for final-transcript-with-timing to enumerate which
   of the 8+ internal keys make it to the client.
3. **Full admission `signal` sub-object**: need to read all of
   `_admission_backlog_signal` to enumerate fields.
4. **PCM frame dtype**: need to confirm `_handle_audio` uses int16 LE 16kHz
   mono (likely from the `np.frombuffer(..., dtype=np.int16)` pattern).
5. **Odd-length binary payload**: v5 §VIII says WS-1003 close; Python's
   behavior unverified — does it reject, or silently truncate to int16 boundary?

## Net

The v5 architecture is mostly correct on shape but has 3 concrete deltas
(/health enum, WARMUP_MS semantics, pre-handshake-503-vs-post-handshake-1013)
that need fold-back into the architecture OR explicit "C++ extends Python"
decisions. The C++ port choices ARE under our control — the audit's job is
to make them with eyes open, not to match Python blindly.

When Codex's audit lands, this doc's findings get folded into
`reviews/server-py-protocol-audit.md` (the canonical deliverable).
