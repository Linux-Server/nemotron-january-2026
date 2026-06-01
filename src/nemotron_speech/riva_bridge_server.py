"""Legacy WebSocket protocol bridge backed by Riva realtime ASR.

This keeps the existing Nemotron client contract on `ws://<host>:8080/`:
- audio: binary PCM16 (16 kHz mono)
- control: {"type":"reset","finalize":true|false} / {"type":"end"}
- output: {"type":"transcript","text":"...","is_final":bool,"finalize":bool}

Internally it forwards audio/control to Riva realtime WebSocket:
`ws://<riva-host>:9000/v1/realtime?intent=transcription`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional
from urllib.parse import urlparse, urlunparse

import aiohttp
from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType, web
from loguru import logger


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex[:8]}"


def _merge_completed_text(existing: str, new_text: str) -> str:
    """Merge incremental completed segments into one stable transcript.

    Riva can emit either segment-only or cumulative text on completed events.
    Build a best-effort assembled string that works for both patterns.
    """
    existing = (existing or "").strip()
    new_text = (new_text or "").strip()

    if not new_text:
        return existing
    if not existing:
        return new_text
    if new_text in existing:
        return existing
    if existing in new_text:
        return new_text

    max_overlap = min(len(existing), len(new_text))
    for overlap in range(max_overlap, 0, -1):
        if existing[-overlap:] == new_text[:overlap]:
            return f"{existing}{new_text[overlap:]}".strip()

    joiner = "" if existing.endswith((" ", "\n")) or new_text.startswith((" ", "\n")) else " "
    return f"{existing}{joiner}{new_text}".strip()


def _select_commit_final_text(assembled: str, terminal: str) -> str:
    """Pick the best final text for a completed commit window.

    In this Riva path, `is_last_result=true` can carry either:
    - a full-window transcript (preferred), or
    - only a trailing fragment.

    We prefer the terminal text when it is close in length to the assembled
    partials; otherwise keep the assembled variant.
    """
    assembled = (assembled or "").strip()
    terminal = (terminal or "").strip()

    if not assembled:
        return terminal
    if not terminal:
        return assembled
    if assembled.endswith(terminal) and len(terminal) >= int(0.4 * len(assembled)):
        # Common case: assembled text is duplicated/expanded history and the
        # terminal chunk is the clean full-window transcript.
        return terminal
    if assembled in terminal:
        return terminal
    if terminal in assembled and len(terminal) < int(0.8 * len(assembled)):
        return assembled
    if len(terminal) >= int(0.8 * len(assembled)):
        return terminal
    return assembled


@dataclass
class PendingCommit:
    finalize: bool
    send_done: bool
    assembled_text: str = ""


@dataclass
class BridgeSession:
    id: str
    client_ws: web.WebSocketResponse
    riva_ws: ClientWebSocketResponse
    # Queue of pending commit metadata and assembled completed text.
    pending_commits: Deque[PendingCommit] = field(default_factory=deque)
    samples_since_last_commit: int = 0
    last_interim_text: str = ""
    last_completed_text: str = ""
    last_sent_interim: str = ""


class RivaCompatBridgeServer:
    """Legacy websocket server that bridges to Riva realtime events."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        riva_ws_url: str = "ws://localhost:9000/v1/realtime?intent=transcription",
        model: str = "conformer",
        language: str = "en-US",
        sample_rate_hz: int = 16000,
        num_channels: int = 1,
        endpoint_start_history: int | None = None,
        endpoint_start_threshold: float | None = None,
        endpoint_stop_history: int | None = None,
        endpoint_stop_threshold: float | None = None,
        endpoint_stop_history_eou: int | None = None,
        endpoint_stop_threshold_eou: float | None = None,
        connect_timeout_s: float = 15.0,
    ):
        self.host = host
        self.port = port
        self.riva_ws_url = riva_ws_url
        self.model = model
        self.language = language
        self.sample_rate_hz = sample_rate_hz
        self.num_channels = num_channels
        self.connect_timeout_s = connect_timeout_s
        self.endpoint_start_history = endpoint_start_history
        self.endpoint_start_threshold = endpoint_start_threshold
        self.endpoint_stop_history = endpoint_stop_history
        self.endpoint_stop_threshold = endpoint_stop_threshold
        self.endpoint_stop_history_eou = endpoint_stop_history_eou
        self.endpoint_stop_threshold_eou = endpoint_stop_threshold_eou

        self._http_session: Optional[ClientSession] = None
        self.sessions: dict[str, BridgeSession] = {}

    async def start(self):
        """Start bridge HTTP/WebSocket server."""
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self.connect_timeout_s)
        self._http_session = ClientSession(timeout=timeout)

        app = web.Application()
        app.router.add_get("/health", self.health_handler)
        app.router.add_get("/", self.websocket_handler)

        logger.info(
            "Starting Riva bridge on ws://{}:{} (riva_ws_url={})",
            self.host,
            self.port,
            self.riva_ws_url,
        )

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info("Bridge ready on ws://{}:{}/", self.host, self.port)
        await asyncio.Future()

    async def health_handler(self, request: web.Request) -> web.Response:
        """Health endpoint for bridge and upstream realtime server."""
        if self._http_session is None:
            return web.json_response({"status": "starting"}, status=503)

        upstream_ok = False
        upstream_status = "unreachable"
        health_url = self._riva_health_url()
        try:
            async with self._http_session.get(health_url) as resp:
                upstream_ok = resp.status == 200
                upstream_status = "ok" if upstream_ok else f"http_{resp.status}"
        except Exception as e:
            upstream_status = str(e)

        payload = {
            "status": "healthy" if upstream_ok else "degraded",
            "upstream_realtime_health": upstream_status,
            "riva_ws_url": self.riva_ws_url,
            "sessions": len(self.sessions),
        }
        return web.json_response(payload, status=200 if upstream_ok else 503)

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Accept legacy client websocket and bridge to Riva websocket."""
        ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await ws.prepare(request)

        session_id = uuid.uuid4().hex[:8]
        logger.info("Client {} connected", session_id)

        if self._http_session is None:
            await self._send_client_error(ws, "Bridge HTTP session is not initialized")
            await ws.close()
            return ws

        riva_ws: Optional[ClientWebSocketResponse] = None
        upstream_task: Optional[asyncio.Task] = None
        session: Optional[BridgeSession] = None
        try:
            riva_ws = await self._http_session.ws_connect(
                self.riva_ws_url,
                heartbeat=30.0,
                max_msg_size=16 * 1024 * 1024,
            )
            session = BridgeSession(id=session_id, client_ws=ws, riva_ws=riva_ws)
            self.sessions[session_id] = session

            await self._send_riva_session_update(session)

            upstream_task = asyncio.create_task(self._pump_riva_to_client(session))
            await ws.send_str(json.dumps({"type": "ready"}))

            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    await self._handle_client_audio(session, msg.data)
                elif msg.type == WSMsgType.TEXT:
                    await self._handle_client_text(session, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.error("Client {} websocket error: {}", session_id, ws.exception())
                    break

        except Exception as e:
            logger.error("Client {} bridge error: {}", session_id, e)
            await self._send_client_error(ws, str(e))
        finally:
            if upstream_task is not None:
                upstream_task.cancel()
                try:
                    await upstream_task
                except asyncio.CancelledError:
                    pass

            if riva_ws is not None and not riva_ws.closed:
                await riva_ws.close()

            self.sessions.pop(session_id, None)
            logger.info("Client {} disconnected", session_id)

        return ws

    async def _handle_client_audio(self, session: BridgeSession, audio_bytes: bytes):
        if not audio_bytes:
            return
        session.samples_since_last_commit += len(audio_bytes) // 2
        payload = {
            "event_id": _event_id(),
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio_bytes).decode("ascii"),
        }
        await session.riva_ws.send_str(json.dumps(payload))

    async def _handle_client_text(self, session: BridgeSession, raw_text: str):
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            await self._send_client_error(session.client_ws, "Invalid JSON control message")
            return

        msg_type = data.get("type")
        if msg_type not in {"reset", "end"}:
            logger.warning("Client {} unknown control message: {}", session.id, msg_type)
            return

        finalize = True if msg_type == "end" else bool(data.get("finalize", True))
        # For this Riva path, finalize semantics require done to guarantee a
        # completed final result. `end` always finalizes; `reset` finalizes
        # only when requested.
        send_done = finalize
        await self._commit_current_buffer(session, finalize=finalize, send_done=send_done)

    async def _commit_current_buffer(self, session: BridgeSession, finalize: bool, send_done: bool = False):
        # No new audio since last commit: emit an empty final immediately so
        # downstream clients can release pending turn state.
        if session.samples_since_last_commit <= 0:
            await self._emit_final_transcript(session, text="", finalize=finalize)
            if finalize:
                if send_done:
                    await self._send_riva_done(session)
                else:
                    await self._send_riva_clear(session)
            return

        session.pending_commits.append(PendingCommit(finalize=finalize, send_done=send_done))
        session.samples_since_last_commit = 0
        payload = {"event_id": _event_id(), "type": "input_audio_buffer.commit"}
        await session.riva_ws.send_str(json.dumps(payload))
        if send_done:
            await self._send_riva_done(session)

    async def _pump_riva_to_client(self, session: BridgeSession):
        async for msg in session.riva_ws:
            if msg.type != WSMsgType.TEXT:
                if msg.type == WSMsgType.ERROR:
                    logger.error("Riva websocket error for {}: {}", session.id, session.riva_ws.exception())
                    await self._send_client_error(session.client_ws, "Upstream Riva websocket error")
                break

            try:
                event = json.loads(msg.data)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from Riva for session {}", session.id)
                continue

            await self._handle_riva_event(session, event)

    async def _handle_riva_event(self, session: BridgeSession, event: dict):
        event_type = event.get("type")

        if event_type == "error":
            message = (event.get("error") or {}).get("message", "Upstream Riva error")
            await self._send_client_error(session.client_ws, message)
            return

        if event_type == "input_audio_buffer.committed":
            # Keep commit metadata queued and consume only when upstream emits
            # completed with is_last_result=true. Item IDs are not reliable for
            # correlation in this API path.
            return

        if event_type == "conversation.item.input_audio_transcription.delta":
            delta = event.get("delta") or ""
            if delta and delta != session.last_sent_interim:
                session.last_interim_text = delta
                session.last_sent_interim = delta
                await self._emit_interim_transcript(session, delta)
            return

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript") or ""
            if transcript:
                session.last_completed_text = transcript

            pending_commit = session.pending_commits[0] if session.pending_commits else None
            if pending_commit and transcript:
                pending_commit.assembled_text = _merge_completed_text(
                    pending_commit.assembled_text, transcript
                )

            if event.get("is_last_result"):
                if pending_commit is None:
                    # No local reset pending for this completion; treat as interim.
                    if transcript and transcript != session.last_sent_interim:
                        session.last_interim_text = transcript
                        session.last_sent_interim = transcript
                        await self._emit_interim_transcript(session, transcript)
                    return

                completed_commit = session.pending_commits.popleft()
                final_text = _select_commit_final_text(
                    assembled=completed_commit.assembled_text, terminal=transcript
                )
                await self._emit_final_transcript(
                    session, text=final_text, finalize=completed_commit.finalize
                )
                if completed_commit.finalize and not completed_commit.send_done:
                    await self._send_riva_clear(session)
                return

            # Non-terminal completion events are surfaced as interim updates.
            if transcript and transcript != session.last_sent_interim:
                session.last_interim_text = transcript
                session.last_sent_interim = transcript
                await self._emit_interim_transcript(session, transcript)
            return

    async def _send_riva_session_update(self, session: BridgeSession):
        payload = {
            "event_id": _event_id(),
            "type": "transcription_session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "pcm16",
                "input_audio_transcription": {
                    "language": self.language,
                    "model": self.model,
                    "prompt": "",
                },
                "input_audio_params": {
                    "sample_rate_hz": self.sample_rate_hz,
                    "num_channels": self.num_channels,
                },
                "recognition_config": {
                    "max_alternatives": 1,
                    "enable_automatic_punctuation": False,
                    "enable_word_time_offsets": False,
                    "enable_profanity_filter": False,
                    "enable_verbatim_transcripts": False,
                },
            },
        }
        endpointing_config: dict[str, int | float] = {}
        if self.endpoint_start_history is not None:
            endpointing_config["start_history"] = self.endpoint_start_history
        if self.endpoint_start_threshold is not None:
            endpointing_config["start_threshold"] = self.endpoint_start_threshold
        if self.endpoint_stop_history is not None:
            endpointing_config["stop_history"] = self.endpoint_stop_history
        if self.endpoint_stop_threshold is not None:
            endpointing_config["stop_threshold"] = self.endpoint_stop_threshold
        if self.endpoint_stop_history_eou is not None:
            endpointing_config["stop_history_eou"] = self.endpoint_stop_history_eou
        if self.endpoint_stop_threshold_eou is not None:
            endpointing_config["stop_threshold_eou"] = self.endpoint_stop_threshold_eou
        if endpointing_config:
            payload["session"]["endpointing_config"] = endpointing_config
        await session.riva_ws.send_str(json.dumps(payload))

    async def _send_riva_clear(self, session: BridgeSession):
        payload = {"event_id": _event_id(), "type": "input_audio_buffer.clear"}
        await session.riva_ws.send_str(json.dumps(payload))

    async def _send_riva_done(self, session: BridgeSession):
        payload = {"event_id": _event_id(), "type": "input_audio_buffer.done"}
        await session.riva_ws.send_str(json.dumps(payload))

    async def _emit_interim_transcript(self, session: BridgeSession, text: str):
        payload = {"type": "transcript", "text": text, "is_final": False}
        await session.client_ws.send_str(json.dumps(payload))

    async def _emit_final_transcript(self, session: BridgeSession, text: str, finalize: bool):
        payload = {"type": "transcript", "text": text, "is_final": True, "finalize": finalize}
        await session.client_ws.send_str(json.dumps(payload))

    async def _send_client_error(self, client_ws: web.WebSocketResponse, message: str):
        try:
            await client_ws.send_str(json.dumps({"type": "error", "message": message}))
        except Exception:
            pass

    def _riva_health_url(self) -> str:
        parsed = urlparse(self.riva_ws_url)
        scheme = "https" if parsed.scheme == "wss" else "http"
        return urlunparse((scheme, parsed.netloc, "/v1/realtime/health", "", "", ""))


def main():
    parser = argparse.ArgumentParser(description="Nemotron legacy WebSocket bridge to Riva realtime ASR")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind bridge server")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind bridge server")
    parser.add_argument(
        "--riva-ws-url",
        default="ws://localhost:9000/v1/realtime?intent=transcription",
        help="Riva realtime websocket URL",
    )
    parser.add_argument("--model", default="conformer", help="Riva ASR model name")
    parser.add_argument("--language", default="en-US", help="Riva language code")
    parser.add_argument("--sample-rate-hz", type=int, default=16000, help="Input sample rate")
    parser.add_argument("--num-channels", type=int, default=1, help="Input channel count")
    parser.add_argument("--endpoint-start-history", type=int, default=None)
    parser.add_argument("--endpoint-start-threshold", type=float, default=None)
    parser.add_argument("--endpoint-stop-history", type=int, default=None)
    parser.add_argument("--endpoint-stop-threshold", type=float, default=None)
    parser.add_argument("--endpoint-stop-history-eou", type=int, default=None)
    parser.add_argument("--endpoint-stop-threshold-eou", type=float, default=None)
    args = parser.parse_args()

    server = RivaCompatBridgeServer(
        host=args.host,
        port=args.port,
        riva_ws_url=args.riva_ws_url,
        model=args.model,
        language=args.language,
        sample_rate_hz=args.sample_rate_hz,
        num_channels=args.num_channels,
        endpoint_start_history=args.endpoint_start_history,
        endpoint_start_threshold=args.endpoint_start_threshold,
        endpoint_stop_history=args.endpoint_stop_history,
        endpoint_stop_threshold=args.endpoint_stop_threshold,
        endpoint_stop_history_eou=args.endpoint_stop_history_eou,
        endpoint_stop_threshold_eou=args.endpoint_stop_threshold_eou,
    )
    asyncio.run(server.start())


if __name__ == "__main__":
    main()
