# server.py Protocol Audit

Audit target: `src/nemotron_speech/server.py` at 10,296 lines. This document is the
Python-compatibility contract for the Step 3b C++ WS/HTTP port.

Scope: HTTP routes, WebSocket routing and handshake delegation, query validation, WS frame
conventions, control messages, close-code behavior, `vad_stop` debounce, server-to-client JSON,
`finalize_timing`, admission counters, and `/stats` quantiles.

Important audit result: several v5 architecture expectations are not what the shipped Python
server currently does. Those divergences are called out explicitly under "Compatibility decisions
and ambiguities" and should be resolved before C++ intentionally deviates from Python.

## Source Anchors

| Area | server.py lines |
|---|---:|
| Imports use `aiohttp.web`, `WSMsgType` | 21-24 |
| Constants: finalize silence, admission defaults, stats window | 84-94 |
| Quantile helper | 97-122 |
| Session continuous/debounce fields | 519-544 |
| Constructor env/config | 602-1005 |
| Query validation | 1433-1498 |
| Admission signal/status | 5032-5084 |
| Stats sample/snapshot/handler | 5086-5193 |
| WebSocket handler | 5195-5317 |
| Continuous queueing/scheduler queueing | 5335-5438 |
| Scheduler continuous event handling | 6771-7200 |
| Non-scheduler continuous worker/debounce/control | 7232-7638 |
| Continuous finalize wire send | 7864-8128 and 8875-9141 |
| Interim transcript wire send | 6283-6299, 6492-6508, 6620-6630, 9332-9344 |
| Reset/hard reset wire send | 9898-10025 |
| HTTP route registration and `/health` | 10202-10244 |

## HTTP Routes

Routes are registered only after `load_model()` completes and `self.model_loaded = True` is set
in `start()` (`server.py:10212-10215`). The `aiohttp` app registers exactly three GET routes:
`/health`, `/stats`, and `/` (`server.py:10241-10244`). No POST, wildcard, static, CORS, auth,
cookie, or admin-only checks are implemented in `server.py`.

| Route | Handler | Request params | Success JSON | Invalid-value behavior |
|---|---|---|---|---|
| `GET /health` | `health_handler` (`server.py:10202-10210`) | Query ignored by handler | `{"status":"healthy"|"loading","model_loaded":bool}` plus optional `"admission"` only when `self.admission_enabled` is true (`server.py:10204-10209`) | No handler-level 400 path. Non-GET/path behavior is `aiohttp` router behavior, not application code. |
| `GET /stats` | `stats_handler` (`server.py:5178-5193`) | Optional `last=N`; empty/missing means full window (`server.py:5184-5186`) | Full shape below from `_stats_snapshot()` (`server.py:5140-5176`) | If `last` is present and truthy but not an int greater than 0, returns HTTP 400 JSON `{"error":"invalid 'last': <repr>"}` (`server.py:5186-5192`). |
| `GET /` | `websocket_handler` (`server.py:5195-5317`) | Optional `model`, `language`; extra params ignored (`server.py:1493-1498`) | Valid WS handshake, session init, then `{"type":"ready"}` (`server.py:5199-5200`, `server.py:5259`) | Invalid `model`/`language` is after WS upgrade: server sends `{"type":"error","message":...}` then closes with no explicit code (`server.py:5199-5208`). Admission rejection is also after upgrade: close code 1013, message `admission_backpressure` (`server.py:5211-5229`). |

### `/health` Exact Shape

The handler builds this object in insertion order (`server.py:10204-10209`):

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

The full contract is:

```jsonc
{
  "status": "healthy" | "loading",
  "model_loaded": boolean,
  "admission"?: {
    "enabled": boolean,
    "attempted": integer,
    "admitted": integer,
    "rejected": integer,
    "max_backlog": integer,
    "max_ready_age_ms": number,
    "signal": {
      "queued_events": integer,
      "ready_count": integer,
      "backlog_count": integer,
      "oldest_ready_age_ms": number,
      "oldest_ready_session_id": string | null
    }
  }
}
```

Notes:

- The only status values in Python are `"healthy"` and `"loading"` (`server.py:10205`). There is
  no Python `draining` or `degraded` status.
- In normal `start()` flow, external clients only see `"healthy"` because the app is created after
  `self.model_loaded = True` (`server.py:10212-10215`). The handler still supports `"loading"` if
  invoked while `model_loaded` is false.
- `/health` includes `"admission"` only when admission is enabled (`server.py:10208-10209`).
  `/stats` always includes an `"admission"` object (`server.py:5174-5175`).

### `/stats` Exact Shape

`/stats` returns `web.json_response(self._stats_snapshot(last_n=last_n))` (`server.py:5193`).
The full response object is assembled at `server.py:5157-5176`:

```jsonc
{
  "enabled": boolean,
  "window_size": integer,
  "samples": integer,
  "since_unix": number | null,
  "until_unix": number | null,
  "emitted_in_window": integer,
  "suppressed_in_window": integer,
  "lifetime_emitted": integer,
  "lifetime_suppressed": integer,
  "metrics": {
    "vad_stop_to_sent_ms": {
      "p50": number | null,
      "p90": number | null,
      "p95": number | null,
      "p99": number | null,
      "max": number | null,
      "count": integer
    },
    "fork_flush_wall_ms": {
      "p50": number | null,
      "p90": number | null,
      "p95": number | null,
      "p99": number | null,
      "max": number | null,
      "count": integer
    },
    "vad_stop_recv_to_process_ms": {
      "p50": number | null,
      "p90": number | null,
      "p95": number | null,
      "p99": number | null,
      "max": number | null,
      "count": integer
    },
    "lock_wait_ms": {
      "p50": number | null,
      "p90": number | null,
      "p95": number | null,
      "p99": number | null,
      "max": number | null,
      "count": integer
    },
    "vad_stop_to_finalize_start_ms": {
      "p50": number | null,
      "p90": number | null,
      "p95": number | null,
      "p99": number | null,
      "max": number | null,
      "count": integer
    }
  },
  "active_sessions_at_emit": {
    "p50": number | null,
    "p90": number | null,
    "p95": number | null,
    "p99": number | null,
    "max": number | null,
    "count": integer
  },
  "admission": {
    "enabled": boolean,
    "attempted": integer,
    "admitted": integer,
    "rejected": integer,
    "max_backlog": integer,
    "max_ready_age_ms": number,
    "signal": {
      "queued_events": integer,
      "ready_count": integer,
      "backlog_count": integer,
      "oldest_ready_age_ms": number,
      "oldest_ready_session_id": string | null
    }
  }
}
```

Stats semantics:

- Stats are enabled unless `NEMOTRON_STATS_ENABLED=0` (`server.py:959-965`). The endpoint still
  returns the same object shape when disabled; samples remain empty because `_record_stats_sample`
  returns immediately (`server.py:5099-5100`).
- `window_size` defaults to 2048 (`server.py:91-94`, `server.py:965-968`) and must be greater
  than 0 (`server.py:969-970`).
- `last=N` narrows the already-snapshotted deque to the last N samples only when N is greater than
  0 (`server.py:5146-5148`, `server.py:5184-5192`). `?last=` is ignored because the empty string
  is falsey at `server.py:5186`.
- The stats sample tuple is appended only when `timing` has both `vad_stop` and `final_sent`
  (`server.py:5102-5105`, `server.py:5121-5131`). Incomplete finalizes increment only lifetime
  counters, not the window (`server.py:5104-5110`).
- `emitted_in_window` counts tuple field `emitted`; `suppressed_in_window` is `len(samples) -
  emitted_in_window` (`server.py:5151-5152`). Lifetime counters are `_stats_emitted_count` and
  `_stats_suppressed_count` (`server.py:5131-5135`, `server.py:5165-5166`).

Quantile formula:

- Empty input returns all percentile/max fields as `null` and `count: 0` (`server.py:105-106`).
- Non-empty input sorts values, then uses `idx = max(0, min(n - 1, int(round(p * (n - 1)))))`
  (`server.py:107-113`). Percentiles are p50, p90, p95, p99, plus max and count
  (`server.py:115-122`).

## Admission Counters

Admission defaults are effectively disabled unless the operator changes at least one threshold:
`_ADMISSION_MAX_BACKLOG_DEFAULT = 1_000_000_000`,
`_ADMISSION_MAX_READY_AGE_MS_DEFAULT = 1_000_000_000_000.0` (`server.py:89-90`), and
`self.admission_enabled` is true only when a configured value differs from those defaults
(`server.py:933-948`).

The Python-compatible admission shape is `_admission_status_snapshot()` (`server.py:5075-5084`):

```jsonc
{
  "enabled": boolean,
  "attempted": integer,
  "admitted": integer,
  "rejected": integer,
  "max_backlog": integer,
  "max_ready_age_ms": number,
  "signal": {
    "queued_events": integer,
    "ready_count": integer,
    "backlog_count": integer,
    "oldest_ready_age_ms": number,
    "oldest_ready_session_id": string | null
  }
}
```

Signal computation:

- `queued_events` is the sum of all per-session continuous queue sizes (`server.py:5032-5038`).
- `ready_count` is `len(self._scheduler_ready)` (`server.py:5056`).
- `backlog_count` is `queued_events + ready_count` (`server.py:5057-5060`).
- `oldest_ready_age_ms` is the maximum age of sessions in `_scheduler_ready`, based on
  `time.monotonic()` and `session.scheduler_ready_since` (`server.py:5039-5055`).
- `oldest_ready_session_id` is a session id string or `None` (`server.py:5041-5062`).

Rejection conditions:

- Reject if `backlog_count > admission_max_backlog` (`server.py:5065-5070`).
- Reject if `oldest_ready_age_ms > admission_max_ready_age_ms` (`server.py:5071-5072`).
- Rejection occurs after `ws.prepare(request)` and closes the WebSocket with code 1013 and close
  message `admission_backpressure`; it is not an HTTP 503 pre-upgrade response in Python
  (`server.py:5199-5200`, `server.py:5211-5229`).

## WebSocket Handshake And Routing

Python's application-level code does not manually parse or validate RFC6455 headers. It delegates
all WebSocket handshake validation to `aiohttp.web.WebSocketResponse.prepare()`:

- `aiohttp` is imported at `server.py:23`.
- The only WS route registered by the application is `GET /` (`server.py:10241-10244`).
- The handler constructs `web.WebSocketResponse(max_msg_size=10 * 1024 * 1024)` and immediately
  calls `await ws.prepare(request)` (`server.py:5199-5200`).
- Query validation and admission checks happen after `prepare()` (`server.py:5202-5229`).

Application-level handshake contract:

| Item | Python behavior | Lines |
|---|---|---:|
| Allowed WS path | Only `GET /` reaches `websocket_handler`; `/health` and `/stats` are normal JSON routes. Other paths/methods are `aiohttp` router behavior. | 10241-10244 |
| `Sec-WebSocket-Key`, `Sec-WebSocket-Version`, `Upgrade`, `Connection` | No manual checks in `server.py`; `aiohttp` enforces the handshake during `ws.prepare()`. | 5199-5200 |
| Subprotocols | None requested or validated. | 5199-5200 |
| Origin/auth/CORS/cookies | No application checks. | 5195-5317 |
| Max message size | 10 MiB via `max_msg_size=10 * 1024 * 1024`. | 5199 |
| First server frame after accepted session init | `{"type": "ready"}`. | 5259 |

Query validation:

- `model` and `language` are read with `(query.get(...) or "").strip()`; extra query keys are
  ignored (`server.py:1493-1498`).
- `model` is optional. If present, it is compared case-insensitively against aliases from
  `NEMOTRON_MODEL_NAME`, the configured model path/name/stem, and either `english`/`en` or
  `multilingual`/`ml` depending on model type (`server.py:1433-1450`, `server.py:1452-1469`).
- If `model` mismatches, `_validate_model_query_param()` raises `ValueError` with
  `model mismatch: requested ...; server accepts: ...` (`server.py:1464-1469`).
- `language` is optional. If present on a non-prompted model, it raises
  `this model does not accept a language argument` (`server.py:1471-1474`).
- If `language` is present on a prompted model, it must be an exact key in `prompt_dictionary`;
  there is no case normalization (`server.py:1476-1481`).
- If `language` is absent on a prompted model, Python defaults to `PROMPTED_DEFAULT_TARGET_LANG =
  "auto"` and requires it to exist in `prompt_dictionary` (`server.py:466`, `server.py:1483-1489`).
- If `language` is absent on a non-prompted model, Python uses `self.target_lang`, initialized from
  `NEMOTRON_TARGET_LANG` with default `"en-US"` (`server.py:617`, `server.py:1491`).
- Invalid query behavior is not HTTP 400: Python has already upgraded, sends an error frame, calls
  `ws.close()` with no explicit code, and returns (`server.py:5199-5208`).

## WS Frame Conventions

| Direction | Frame type | Payload | Python behavior |
|---|---|---|---|
| Client -> server | Binary | Raw PCM samples. Python interprets bytes with `np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0`. | Non-scheduler path at `server.py:9311-9323`; scheduler path at `server.py:6878-6897`. |
| Client -> server | Text | JSON object control message. | `json.loads`, then `data.get("type")` (`server.py:5272-5288`, `server.py:5344-5360`, `server.py:5418-5434`). |
| Server -> client | Text | JSON via `json.dumps(...)`. | Ready, error, transcript sends at `server.py:5206`, `server.py:5259`, `server.py:7221`, `server.py:9940-9997`. |

PCM details:

- Sample rate is hard-coded in the server instance as 16,000 Hz (`server.py:618`).
- There is no channel-count negotiation or header; by convention the stream is 16 kHz mono.
- `np.int16` is native-endian. On the shipped Linux target this is little-endian, but
  `server.py` does not contain an explicit endianness check (`server.py:6883`, `server.py:9312`).
- Odd-length binary payloads are not prechecked. `np.frombuffer(..., dtype=np.int16)` will raise if
  the buffer length is not a multiple of two (`server.py:6883`, `server.py:9312`). In
  non-continuous mode the outer WS handler catches the exception, sends `{"type":"error",
  "message": str(e)}`, then exits without an explicit close code (`server.py:5297-5308`). In
  continuous worker paths, worker-level exception handlers send the same error frame and continue
  unless the exception occurs before queueing (`server.py:6813-6823`, `server.py:6840-6850`,
  `server.py:7271-7281`).

Text JSON details:

- Invalid JSON is logged and ignored, with no error frame and no close (`server.py:5289-5290`,
  `server.py:5345-5349`, `server.py:5419-5423`).
- Unknown `"type"` values are logged and ignored, with no error frame and no close
  (`server.py:5286-5287`, `server.py:5359-5360`, `server.py:5433-5434`).
- JSON that parses to a non-object is not guarded before `data.get(...)`; it raises an exception
  and follows the generic error-frame path (`server.py:5273-5280`, `server.py:5297-5308`).
- The `finalize` control field is read with `data.get("finalize", True)` and is not type-checked
  (`server.py:5277-5281`, `server.py:5351-5354`, `server.py:5425-5428`).

## WS Control Messages

Python recognizes `reset`, `end`, `vad_start`, and `vad_stop`.

| Message | `finalize` default | Non-continuous mode (`NEMOTRON_CONTINUOUS` off) | Continuous mode (`NEMOTRON_CONTINUOUS=1`) | Closes socket? |
|---|---:|---|---|---|
| `{"type":"reset"}` | `true` | Calls `_reset_session(session, finalize=True)`: pad pending audio, process final chunk, send final transcript with `"finalize": true`, then reinitialize session state (`server.py:5277-5281`, `server.py:9951-10025`). | If pending debounce, sets `continuous_reset_seen=True` and waits for debounce expiry (`server.py:7571-7577`, scheduler equivalent `server.py:7093-7099`). If not pending and audio/text exists, sends a speculative final with context retained (`server.py:7579-7590`, scheduler equivalent `server.py:7101-7117`). Empty reset is ignored (`server.py:7593-7596`, scheduler equivalent `server.py:7119-7122`). | No. |
| `{"type":"reset","finalize":false}` | n/a | Soft reset: sends current cumulative text as final with `"finalize": false`, preserves decoder/audio state (`server.py:9932-9949`). | Soft reset: sends current cumulative text as final with `"finalize": false`, preserves state (`server.py:7536-7548`, scheduler equivalent `server.py:7053-7070`). | No. |
| `{"type":"end"}` | `true` | Treated identically to `reset`; calls `_reset_session(..., finalize=True)` (`server.py:5277-5281`). It does not close. | True boundary. If pending/audio/text/post-stop audio exists, force-finalizes with `reason="end"`, includes post-stop audio, sends final if there is a delta, then cold-resets ASR state (`server.py:7550-7563`, `server.py:7434-7468`, `server.py:9290-9298`; scheduler equivalent `server.py:7072-7085`, `server.py:6933-6975`, `server.py:7177-7198`). Empty end is ignored (`server.py:7565-7569`, scheduler equivalent `server.py:7087-7091`). | No. |
| `{"type":"end","finalize":false}` | n/a | Same soft-reset behavior as `reset` with `finalize:false` because msg type is not distinguished before `_reset_session` (`server.py:5277-5281`, `server.py:9932-9949`). | Same soft-reset behavior because the `not finalize` branch runs before `msg_type == "end"` (`server.py:7536-7550`, scheduler equivalent `server.py:7053-7072`). | No. |
| `{"type":"vad_start"}` | no field used | No-op except debug log (`server.py:5282-5285`). | If state is `PENDING_FINALIZE`, cancels the debounce task, returns to `STREAMING`, clears `vad_stop`/expiry bookkeeping, and flushes held post-stop audio (`server.py:7487-7501`, scheduler equivalent `server.py:6997-7015`). If not pending, flushes any retained post-stop audio and logs (`server.py:7502-7510`, scheduler equivalent `server.py:7015-7023`). | No. |
| `{"type":"vad_stop"}` | no field used | No-op except debug log (`server.py:5282-5285`). | Arms pending finalization: cancels any existing debounce without invalidating the current sequence, increments `continuous_stop_seq`, sets state `PENDING_FINALIZE`, records `continuous_vad_stop_ts = time.time()`, clears expiry, and creates a debounce timer (`server.py:7512-7527`, scheduler equivalent with generation invalidation at `server.py:7025-7044`). | No. |

`end` never calls `ws.close()` in Python. It is an ASR lifecycle/finalization control, not a
transport close command.

## `vad_stop` Debounce Semantics

No server-side Silero VAD exists in `server.py`. `grep -ni silero src/nemotron_speech/server.py`
returns zero lines. The only VAD-related server inputs are client text control messages
`vad_start` and `vad_stop` (`server.py:5282-5285`, `server.py:5355-5358`, `server.py:5429-5432`).

Actual shipped debounce behavior:

1. Debounce is meaningful only in continuous mode. With `NEMOTRON_CONTINUOUS` off, both
   `vad_start` and `vad_stop` are logged as no-ops (`server.py:5270-5285`).
2. The code default is `_DEFAULT_FINALIZE_SILENCE_MS = 150`, not 0 (`server.py:87`). The env var
   `NEMOTRON_FINALIZE_SILENCE_MS` is read only when continuous mode is enabled, defaults to 150,
   and must satisfy `0 <= value < 10000` (`server.py:921-931`).
3. There is no `NEMOTRON_VAD_WARMUP_MS` or `VAD_WARMUP` setting in `server.py`. The only warmup env
   var in this area is `NEMOTRON_WARMUP_MS`, which is session/model warmup, not VAD cancellation
   hold (`server.py:918-920`).
4. On `vad_stop`, Python enters `PENDING_FINALIZE`, records `continuous_vad_stop_ts`, and starts
   `_continuous_debounce_timer(session.id, stop_seq)` (`server.py:7512-7527`; scheduler equivalent
   `server.py:7025-7044`).
5. The debounce timer sleeps exactly `self.finalize_silence_seconds`, then enqueues
   `("debounce_expired", stop_seq)` (plus a perf timestamp when finalize profiling is enabled)
   (`server.py:7336-7349`).
6. Binary audio received while `PENDING_FINALIZE` is held in `continuous_post_stop_audio` and is not
   appended to ASR state until the pending finalize is canceled or a true-boundary finalize includes
   it (`server.py:7373-7382`, scheduler equivalent `server.py:6855-6868`).
7. A `vad_start` while still `PENDING_FINALIZE` cancels the debounce task, invalidates the stop
   sequence, returns the session to `STREAMING`, clears `continuous_vad_stop_ts` and
   `continuous_debounce_expiry_ts`, and flushes held audio into the ASR state (`server.py:7353-7367`,
   `server.py:7487-7501`; scheduler equivalent `server.py:6997-7015`).
8. On debounce expiry, Python checks both state and stop sequence to ignore stale timers
   (`server.py:7598-7614`; scheduler equivalent `server.py:7124-7140`).
9. A live expiry sets state `FINALIZED`, records `continuous_debounce_expiry_ts`, chooses reason
   `"reset_then_debounce"` if a reset arrived during the debounce window or `"debounce_expired"`
   otherwise, emits a speculative final, and returns to streaming with context retained
   (`server.py:7616-7634`, `server.py:9181-9202`; scheduler equivalent `server.py:7142-7161`).
10. There is no separate 200 ms cancellation hold after debounce expiry. Cancellation is possible
    only while the session is still `PENDING_FINALIZE` and before the matching `debounce_expired`
    event is processed.

Implication for `NEMOTRON_FINALIZE_SILENCE_MS=0`: the timer does `await asyncio.sleep(0)`, so
finalization is scheduled immediately on the event loop (`server.py:7336-7349`). A subsequent
`vad_start` cancels only if it is processed before the matching expiry event wins the state/sequence
check (`server.py:7487-7501`, `server.py:7598-7614`).

## Server-To-Client Wire JSON

All server-to-client application messages are text frames produced by `json.dumps()` without
`sort_keys` on the WS path (`server.py:5206`, `server.py:5259`, `server.py:7221`,
`server.py:9940-9997`). Field order follows Python dict insertion order at the send site, but
contract consumers should treat these as JSON objects, not byte-stable strings.

| Event | JSON shape | When emitted | Lines |
|---|---|---|---:|
| Ready | `{"type":"ready"}` | After query validation, admission acceptance, session creation, session init, and continuous worker start if enabled. | 5232-5259 |
| Error | `{"type":"error","message":string}` | Invalid query after upgrade, top-level WS handler exception, scheduler/continuous worker exception. | 5202-5208, 5297-5305, 6813-6821, 6840-6848, 7271-7279 |
| Interim transcript | `{"type":"transcript","text":string,"is_final":false}` | When model text changes during chunk processing. | 6283-6299, 6492-6508, 6620-6630, 9332-9344 |
| Soft reset transcript | `{"type":"transcript","text":string,"is_final":true,"finalize":false}` | `reset` or `end` with `finalize:false`. | 7536-7543, 9932-9945 |
| Non-continuous hard reset transcript | `{"type":"transcript","text":string,"is_final":true,"finalize":true}` | `reset` or `end` with `finalize:true` when continuous mode is off. | 9991-9997 |
| Continuous final transcript | `{"type":"transcript","text":string,"is_final":true,"finalize":true,"finalize_timing":object}` | Continuous-mode finalization when `delta_text` is non-empty. Empty/duplicate finals are suppressed and send no transcript. | 8118-8128, 8152-8166, 9131-9141, 9165-9179 |

There are no top-level timestamps, sequence numbers, session ids, model names, language fields,
or close-code fields in server-to-client JSON. Timing timestamps appear only inside
`finalize_timing`.

## `finalize_timing`

Wire `finalize_timing` is not the five derived SLO metrics. Python sends the raw timing dictionary.
The five SLO metrics are derived later for `/stats`.

The timing dict is initialized in `_continuous_finalize_timing()` (`server.py:7864-7880`) and
duplicated in the serial finalize path (`server.py:8902-8912`). The exact wire key set is:

```jsonc
{
  "reason": string,
  "vad_stop": number | null,
  "vad_stop_recv": number | null,
  "debounce_expiry": number | null,
  "fork_flush_start": number | null,
  "fork_flush_done": number | null,
  "final_sent": number | null,
  "inference_lock_acquire_wait_ms": number | null,
  "gil_attrib_enabled": boolean
}
```

Key semantics:

- `reason`: string such as `"debounce_expired"`, `"reset_then_debounce"`, `"reset"`, `"end"`, or
  `"close"` depending on finalization path (`server.py:7616-7630`, `server.py:7550-7590`,
  `server.py:7434-7468`).
- `vad_stop`: Unix timestamp from `time.time()` when `vad_stop` is processed, or null for
  non-`vad_stop` finalizations (`server.py:7518`, `server.py:7872`, `server.py:8904`).
- `vad_stop_recv`: Unix timestamp from the socket receive side only when
  `NEMOTRON_FINALIZE_PROFILE=1`; otherwise it remains null (`server.py:5355-5358`,
  `server.py:5429-5432`, `server.py:7873`, `server.py:8905`).
- `debounce_expiry`: Unix timestamp when debounce expiry is processed, or a forced-finalize
  timestamp for true-boundary `end`/`close` paths if unset (`server.py:7464-7467`,
  `server.py:7620`, `server.py:7874`, `server.py:8906`; scheduler equivalent
  `server.py:6967-6970`, `server.py:7145-7147`).
- `fork_flush_start`: Unix timestamp just before building the finalize fork when there is audio to
  flush (`server.py:7918-7920`, `server.py:8925-8927`).
- `fork_flush_done`: Unix timestamp after final fork processing completes (`server.py:8069-8070`,
  `server.py:9087-9090`).
- `final_sent`: Unix timestamp set immediately before the final transcript send (`server.py:8118-8128`,
  `server.py:9131-9141`).
- `inference_lock_acquire_wait_ms`: milliseconds spent waiting for the inference/model lane lock,
  or null if no fork flush/model call occurred (`server.py:8025-8053`, `server.py:8950-9070`).
- `gil_attrib_enabled`: boolean copy of `self.gil_attrib_enabled` (`server.py:7878-7880`,
  `server.py:8910-8911`).

Presence rules:

- `finalize_timing` appears only on continuous final transcript frames and only when a non-empty
  `delta_text` is sent (`server.py:8118-8128`, `server.py:9131-9141`).
- Non-continuous hard resets do not include `finalize_timing` (`server.py:9991-9997`).
- Soft resets do not include `finalize_timing` (`server.py:7536-7543`, `server.py:9932-9945`).
- Empty/duplicate continuous finals are suppressed; no final transcript and no wire
  `finalize_timing` is sent (`server.py:8152-8166`, `server.py:9165-9179`).

The `/stats` five SLO metric keys are derived from the raw timing dict:

| `/stats` metric | Formula | Lines |
|---|---|---:|
| `vad_stop_to_sent_ms` | `final_sent - vad_stop` in ms | 5121-5124, 5168 |
| `fork_flush_wall_ms` | `fork_flush_done - fork_flush_start` in ms | 5124, 5169 |
| `vad_stop_recv_to_process_ms` | `vad_stop - vad_stop_recv` in ms | 5125, 5170 |
| `lock_wait_ms` | raw `inference_lock_acquire_wait_ms` | 5114, 5126, 5171 |
| `vad_stop_to_finalize_start_ms` | `fork_flush_start - vad_stop` in ms | 5127, 5172 |

## Error Frame Format

Recoverable/application error frames are:

```json
{
  "type": "error",
  "message": "..."
}
```

No other fields are used. There is no `code`, `close_code`, `details`, `fatal`, `timestamp`, or
sequence number. Send sites:

- Invalid `model`/`language` query after upgrade (`server.py:5202-5208`).
- Generic WS handler exception (`server.py:5297-5305`).
- Scheduler and continuous worker exceptions (`server.py:6813-6821`, `server.py:6840-6848`,
  `server.py:7271-7279`).

Invalid JSON and unknown control message types do not produce error frames (`server.py:5286-5290`,
`server.py:5345-5360`, `server.py:5419-5434`).

## Close Codes

`server.py` explicitly calls `ws.close()` in only two application scenarios: invalid query with no
explicit code, and admission backpressure with code 1013 (`server.py:5202-5208`,
`server.py:5229`). Other close behavior is `aiohttp` protocol behavior or handler return behavior.

| Code | Python trigger | Audit finding |
|---:|---|---|
| 1000 | Invalid `model`/`language` query calls `await ws.close()` without a code after sending an error frame (`server.py:5202-5208`). Aiohttp's default close code is normal closure. Normal peer disconnects are also handled by the `async for msg in ws` loop ending (`server.py:5262-5295`). | No explicit `code=1000` in `server.py`; default/delegated behavior. |
| 1001 | None in application code. | No explicit going-away close. Shutdown code cancels local tasks/logs telemetry but does not send WS-1001 (`server.py:10253-10262`). |
| 1003 | None in application code. | Invalid JSON/unknown type are ignored, and odd binary payloads are not prechecked for 1003 (`server.py:5286-5290`, `server.py:6883`, `server.py:9312`). |
| 1008 | None in application code. | Query policy failures use error frame plus default close, not 1008 (`server.py:5202-5208`). No subprotocol/origin policy checks exist. |
| 1009 | Aiohttp may close oversized messages because `max_msg_size=10 * 1024 * 1024` is configured (`server.py:5199`). | No explicit `code=1009` in `server.py`; delegated to aiohttp frame handling. |
| 1011 | None in application code. | Generic exceptions send an error frame but do not call `ws.close(code=1011)` (`server.py:5297-5308`, `server.py:6813-6823`, `server.py:6840-6850`, `server.py:7271-7281`). |
| 1013 | Admission backpressure after upgrade: `await ws.close(code=1013, message=b"admission_backpressure")` (`server.py:5229`). | Explicit Python behavior. |

## Compatibility Decisions And Ambiguities

These items need an explicit Step 3b decision if the C++ port is expected to be Python-exact:

1. v5 architecture says `NEMOTRON_FINALIZE_SILENCE_MS` default 0 and
   `NEMOTRON_VAD_WARMUP_MS` default 200. Shipped `server.py` defaults finalize silence to 150 ms
   and has no `NEMOTRON_VAD_WARMUP_MS` setting or post-expiry hold (`server.py:87`,
   `server.py:921-931`; grep for `NEMOTRON_VAD_WARMUP_MS` has zero hits).
2. v5 architecture says invalid `model`/`language` should be HTTP 400 pre-upgrade. Python upgrades
   first, then sends an error frame and closes with no explicit code (`server.py:5199-5208`).
3. v5 architecture says admission shed is HTTP 503 pre-handshake. Python upgrades first, then
   closes with WS-1013 and message `admission_backpressure` (`server.py:5199-5200`,
   `server.py:5211-5229`).
4. v5 close-code table expects explicit 1003/1008/1011 scenarios. Python does not explicitly emit
   those close codes; it ignores invalid JSON/unknown controls, lets odd PCM raise, sends error
   frames for exceptions, and relies on aiohttp for max-message handling.
5. v5 discussion treats `finalize_timing` as likely the five SLO metrics. Python wire
   `finalize_timing` is a raw timing object with nine keys; the five SLO metrics are derived only
   in `/stats` (`server.py:7864-7880`, `server.py:8118-8128`, `server.py:8902-8912`,
   `server.py:9131-9141`, `server.py:5121-5172`).
6. Exact HTTP status/body for malformed non-WS `GET /` and bad RFC6455 headers is owned by
   `aiohttp`, not visible in `server.py`. The C++ implementation needs an explicit choice if it
   cannot delegate to an equivalent HTTP/WS library.
