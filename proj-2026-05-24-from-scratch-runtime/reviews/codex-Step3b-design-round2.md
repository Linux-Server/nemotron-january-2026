# Codex Step 3b Design Review - Round 2

Verdict: **GO-with-substantive-revisions-to-v3. Not converged.** The architecture is still the right
shape, but v2 is not minor-only. It folds Round 1 directionally, then reopens risk by deferring the
server.py audit while Part A still freezes protocol/API skeleton choices.

## Must Fold To v3

1. **Part A should be hard-aborted/relaunched on v3, not diff-audited after landing.**

   The mitigation in XIII.1 ("audit the diff against v2; redo if substantive") is not practical for
   this change class. The local in-flight Part A already touched the large CMake/session carve-out
   path, and the kinds of mistakes v2 is trying to prevent are public-header and route-skeleton
   decisions, not small patch hunks. If the boundary, stats shape, handshake parser, or stubs are
   wrong, reviewing after landing means reverse-engineering a 4k-line move plus API choices.

   Fold action: say explicitly: stop the v1-driven Part A, relaunch after v3, and only salvage
   mechanical moves after checking they match v3. Do not let a v1/v2 hybrid become the baseline.

2. **The Python protocol audit is still inconsistently placed.**

   XII puts the full `server.py` audit in Part B, but XIII.2 says it was "moved forward into Part A."
   That contradiction matters because Part A still creates `lib/ws/handshake`, route stubs, ready
   behavior, `/stats` behavior, query parsing, and close-code choices.

   Specific fallout:
   - Python validates WS query params `model` and `language` before admitting the session; v2 only
     mentions these as possible future audit findings.
   - Python `/health` returns `{"status":"healthy"|"loading","model_loaded":...}` and optionally
     admission; v2's Part A stub says `{"status":"ok"}`.
   - VIII requires `/stats` smoke behavior, but XII says `/stats` is a placeholder until Part B.
   - Invalid JSON is log+ignore in Python; v2 specifies unknown-type ignore but not invalid JSON.

   Fold action: move the protocol compatibility table into Part A before any WS skeleton is accepted,
   or reduce Part A to a pure CMake/library extraction with no protocol-visible behavior.

3. **StatsCollector is not yet a Python-compatible contract.**

   v2 fixes the percentile formula, but the window/counter semantics are internally inconsistent.
   It says suppressed records are not appended to the metric window, while `suppressed_in_window` and
   `?last=N` are defined as if suppressed events have window membership. Python only appends complete
   samples with `vad_stop` and `final_sent`; incomplete suppressed/stale finalizes increment lifetime
   counters only. Complete send-failed samples can still enter the deque with `emitted=false`.

   The admission sub-object is also not Python-compatible: Python returns
   `enabled/attempted/admitted/rejected/max_backlog/max_ready_age_ms/signal{...}`, while v2 documents
   native `active_cap/backlog_cap/offered/active_count/...` fields. That can be a native extension,
   but it cannot be labeled "matching Python" without a compatibility wrapper.

   Fold action: define one of two models:
   - Python-exact: main deque contains only complete samples, each sample has `emitted`; incomplete
     suppressed finalizes affect lifetime counters only. `suppressed_in_window` is computed from
     deque samples, not from all suppressions.
   - Native-extended: keep a second suppression deque for windowed suppressed counts. If so, say it
     intentionally differs from Python and update the oracle.

4. **The library boundary is explicit but still not implementable as written.**

   The table names production wrappers, but their signatures are still `...`, and the public owner of
   device, TorchScript modules, AOTI loaders, tokenizer, audio frontend, scheduler, and runtime config
   is not defined. `WorkerContext` and `AOTIArtifacts` are listed but not specified. `DensityAdmission`
   and scheduler types should not be exposed via `lib/session/session.h`; they belong to their own
   headers/modules. The current runtime code needs a real `SessionRuntime`/`SessionCore`-like owner,
   not just `SessionState& + PCMFrame& + ...`.

   Fold action: add concrete public signatures before Part A. Minimum: define `PCMFrame`, `WireEvent`,
   `SessionRuntime` ownership, lifecycle methods, and which headers own admission/scheduler. Keep
   `EmittedEvent`, gold fixture helpers, `emit_event`, equality checks, bundle accessors, and harness
   modes out of the public header.

5. **Graceful shutdown currently closes before it can drain.**

   VII says to send WS-1001 to existing connections, then await natural VAD-stop/finalize. A close
   frame starts the close handshake; clients should not be expected to keep streaming or send
   `vad_stop` after that. The dispatcher shutdown path is also missing.

   Fold action: on SIGTERM, set draining, stop new admission, stop accepting, then for existing
   sessions either wait for natural completion without sending 1001 yet or enqueue a server-side
   finalize. Send 1001 after final send or at the drain deadline. Specify what happens when a GPU
   finalize exceeds 30s: close socket, mark forced, join/close dispatcher, exit nonzero if work could
   not be cleanly joined.

6. **Single-listener parse failure/slowloris behavior is missing.**

   The accept/router thread reads request line + headers before routing. v2 does not define header
   timeout, max header bytes, malformed first-line behavior, or non-HTTP bytes.

   Fold action: set e.g. 2s header read timeout, 8 KiB max headers, HTTP 400 on malformed HTTP,
   HTTP 431 on oversized headers, and close-without-WS for bytes that are not parseable as HTTP. Do
   not let a slow client pin the accept thread.

## Test Oracle

The XIII.2 oracle is not concrete enough yet, and "byte-for-byte JSON" is probably the wrong default:
Python emits volatile timing values and `json.dumps` insertion order. Use canonicalized assertions
unless v3 also mandates deterministic field order and strips volatile fields.

Fold action:
- Audio: `runtime/artifacts/session_audio_bundle.ts`, rows `utt0..utt7` for smoke; later full bundle.
- Wire: 16 kHz mono signed int16 little-endian PCM, 640-byte/20ms chunks, `vad_start` before audio,
  `vad_stop` after audio, `NEMOTRON_CONTINUOUS=1`, `NEMOTRON_FINALIZE_SILENCE_MS=0`.
- Assertion: ready frame is exactly `{"type":"ready"}`; transcript sequence matches normalized
  Python/bundle events on `type`, `text`, `is_final`, `finalize` flag, final collector text, and event
  count. `finalize_timing` is checked for required keys and numeric/non-null constraints, not exact
  values. Invalid-query and invalid-`last` cases get explicit status/error assertions.

There is already a nearby starting point in `runtime/step6_server_oracle.py`; v3 should either reuse
that contract or supersede it explicitly.

## Minor / Deferrable

- OpenSSL Option A is safe as a dependency choice, but only if the runtime image is actually rebuilt.
  Adding `libssl-dev` to `runtime/container/Dockerfile` will not help CI or developers using an
  existing `nemotron-aoti:cu128` image unless the build flow invalidates/rebuilds that image. No new
  design dependency gap is obvious beyond OpenSSL for SHA1/base64.
- Admin handler pool: choose one model. If accept-thread synchronous, `/stats` must be tiny and
  header reads must time out. Better: accept thread dispatches HTTP work to a fixed pool of 2 with a
  bounded queue; `/stats` can queue briefly but must not block WS accepts indefinitely.
- Backpressure belongs in Part B, but the close table should define invalid query/model/language
  rejection. Python currently sends an error frame after accepting and then closes; native can choose
  HTTP 400 before upgrade, but then it is not byte-for-byte Python.
- Auth can remain deferred with the trusted-network assumption. TLS/proxy assumptions still need one
  sentence: cleartext WS behind TLS-terminating LB/API gateway; server ignores forwarded identity
  headers unless auth is later added.
- Correlation/session IDs, process label in `/health` and `/stats`, and admin endpoint exposure are
  production-ops items, not Part A blockers.
- Fragmentation, ping/pong, close-frame echoing, odd-length binary frames, and oversized text frames
  are Part B blockers before production WS lifecycle, not CMake-extraction blockers.

## Net

Do not call this converged. Fold the Part A sequencing, Python protocol placement, StatsCollector
semantics, concrete session API, graceful shutdown ordering, and malformed-request behavior into v3.
Then Part A can proceed if it is either pure extraction or has protocol stubs constrained by the
completed audit.
