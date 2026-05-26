"""
ConnectionManager — unified WebSocket + SSE connection management for API Server.

Architecture:
    Browser ↔ WebSocket (/ws/stream/{session_id}) ↔ ConnectionManager
                                                  ↕
                                            Agent Run (async task)

Responsibilities:
  - One agent run per session at a time (exclusive execution)
  - Event buffer (max 500 events, TTL 10 min) for replay on reconnect
  - Multiple browser WebSocket connections per session (multiplexing)
  - Diff sync via ``since`` parameter (event offset)
  - Browser disconnect does NOT tear down the agent run
  - Agent run completion / error notifies all connected browsers
"""

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

# Forward reference for type hints when aiohttp is unavailable
if TYPE_CHECKING or AIOHTTP_AVAILABLE:
    WebSocketResponseType = web.WebSocketResponse
else:
    WebSocketResponseType = Any

logger = logging.getLogger(__name__)

_MAX_BUFFER_EVENTS = 500
_BUFFER_TTL_SECONDS = 600  # 10 minutes
_AGENT_RUN_TIMEOUT_SECONDS = 600  # 10 minutes max per run


class _BufferedEvent:
    """Single event with monotonic offset for diff sync."""

    __slots__ = ("offset", "timestamp", "event", "data")

    def __init__(self, offset: int, event: str, data: Dict[str, Any]) -> None:
        self.offset = offset
        self.timestamp = time.monotonic()
        self.event = event
        self.data = data

    def to_json(self) -> str:
        return json.dumps({"event": self.event, "data": self.data, "offset": self.offset}, ensure_ascii=False)


class SessionConnection:
    """
    Per-session state: one active agent run + N downstream WebSockets.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._lock = asyncio.Lock()

        # --- Active agent run ---
        self._agent_task: Optional[asyncio.Task] = None
        self._agent_run_id: Optional[str] = None
        self._agent_busy = False

        # --- Downstream WebSockets ---
        self._browsers: Set["WebSocketResponseType"] = set()

        # --- Event buffer ---
        self._buffer: deque[_BufferedEvent] = deque(maxlen=_MAX_BUFFER_EVENTS)
        self._next_offset = 0
        self._last_activity = time.monotonic()

    # ------------------------------------------------------------------
    # Buffer
    # ------------------------------------------------------------------

    def _append_event(self, event: str, data: Dict[str, Any]) -> _BufferedEvent:
        bev = _BufferedEvent(self._next_offset, event, data)
        self._next_offset += 1
        self._buffer.append(bev)
        self._last_activity = time.monotonic()
        # TTL eviction — drop events older than _BUFFER_TTL_SECONDS
        cutoff = time.monotonic() - _BUFFER_TTL_SECONDS
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()
        return bev

    def get_events_since(self, since: int) -> List[_BufferedEvent]:
        """Return buffered events with offset >= since."""
        return [ev for ev in self._buffer if ev.offset >= since]

    def get_last_offset(self) -> int:
        """Return the next offset (i.e. last_offset + 1)."""
        return self._next_offset

    # ------------------------------------------------------------------
    # Browser WebSocket management
    # ------------------------------------------------------------------

    async def add_browser(self, ws: "WebSocketResponseType") -> None:
        async with self._lock:
            self._browsers.add(ws)
        logger.debug("[SessionConnection:%s] Browser added (total=%d)", self.session_id, len(self._browsers))

    async def remove_browser(self, ws: "WebSocketResponseType") -> None:
        async with self._lock:
            self._browsers.discard(ws)
        logger.debug("[SessionConnection:%s] Browser removed (total=%d)", self.session_id, len(self._browsers))

    async def broadcast(self, payload: str) -> None:
        """Send JSON payload to all connected browsers."""
        dead: List["WebSocketResponseType"] = []
        async with self._lock:
            browsers = list(self._browsers)
        for ws in browsers:
            try:
                await ws.send_str(payload)
            except (ConnectionResetError, BrokenPipeError, OSError):
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._browsers.discard(ws)

    async def broadcast_event(self, event: str, data: Dict[str, Any]) -> None:
        bev = self._append_event(event, data)
        await self.broadcast(bev.to_json())

    # ------------------------------------------------------------------
    # Agent run lifecycle
    # ------------------------------------------------------------------

    @property
    def is_agent_busy(self) -> bool:
        return self._agent_busy

    async def start_agent_run(
        self,
        message: str,
        attachments: Optional[List[Dict[str, Any]]],
        system_message: Optional[str],
        create_agent_fn,
        get_session_db_fn,
        build_user_content_fn,
    ) -> None:
        """
        Start an agent chat turn and stream events to all browsers.
        Exclusive per session — raises RuntimeError if already running.
        """
        if self._agent_busy:
            raise RuntimeError("Agent run already in progress for this session")

        self._agent_busy = True
        self._agent_run_id = f"run_{uuid.uuid4().hex}"

        try:
            await self._run_agent(
                message, attachments, system_message,
                create_agent_fn, get_session_db_fn, build_user_content_fn,
            )
        finally:
            self._agent_busy = False
            self._agent_run_id = None

    async def _run_agent(
        self,
        message: str,
        attachments: Optional[List[Dict[str, Any]]],
        system_message: Optional[str],
        create_agent_fn,
        get_session_db_fn,
        build_user_content_fn,
    ) -> None:
        """Execute agent run and broadcast events."""
        import queue as _q
        import uuid

        session_id = self.session_id
        run_id = self._agent_run_id
        db = get_session_db_fn()
        history = db.get_messages_as_conversation(session_id)
        user_content, persist_text = build_user_content_fn(message, attachments)

        stream_q: _q.Queue = _q.Queue()
        assistant_message_id = f"msg_asst_{uuid.uuid4().hex}"
        user_message_id = f"msg_user_{uuid.uuid4().hex}"

        def _encode_sse(event_name: str, payload: Dict[str, Any]) -> bytes:
            return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        def _queue_event(event_name: str, payload: Dict[str, Any]) -> None:
            stream_q.put(_encode_sse(event_name, payload))

        def _on_delta(delta):
            if delta:
                _queue_event("assistant.delta", {
                    "session_id": session_id, "run_id": run_id,
                    "message_id": assistant_message_id, "delta": delta,
                })

        def _on_tool_start(tool_call_id, function_name, function_args):
            _queue_event("tool.started", {
                "session_id": session_id, "run_id": run_id,
                "tool_call_id": tool_call_id, "tool_name": function_name, "args": function_args,
            })

        def _on_tool_progress(event_type, name, preview, args, **kwargs):
            if name == "_thinking":
                _queue_event("tool.progress", {
                    "session_id": session_id, "run_id": run_id,
                    "message_id": assistant_message_id, "delta": preview,
                })
                return
            tool_call_id = kwargs.get("tool_call_id", "")
            payload = {
                "session_id": session_id, "run_id": run_id,
                "tool_name": name, "preview": preview, "args": args,
                "tool_call_id": tool_call_id,
            }
            _queue_event("tool.started", payload)
            _queue_event("tool.progress", payload)

        loop = asyncio.get_event_loop()

        def _make_tool_complete_callback(run_id, loop):
            def _callback(tool_call_id, tool_name, args, function_result):
                try:
                    result_preview = ""
                    is_error = False
                    if isinstance(function_result, dict):
                        result_preview = function_result.get("preview", "") or function_result.get("output_preview", "")
                        is_error = function_result.get("is_error", False) or function_result.get("error", False)
                    elif isinstance(function_result, str):
                        result_preview = function_result[:200]
                    loop.call_soon_threadsafe(
                        _queue_event, "tool.completed",
                        {
                            "run_id": run_id, "timestamp": time.time(),
                            "tool": tool_name, "tool_name": tool_name,
                            "tool_call_id": tool_call_id,
                            "result_preview": result_preview, "is_error": is_error,
                        }
                    )
                except Exception:
                    pass
            return _callback

        async def _agent_task():
            def _run():
                agent = create_agent_fn(
                    ephemeral_system_prompt=system_message,
                    session_id=session_id,
                    stream_delta_callback=_on_delta,
                    tool_progress_callback=_on_tool_progress,
                    tool_start_callback=_on_tool_start,
                    tool_complete_callback=_make_tool_complete_callback(run_id, loop),
                )
                agent._session_db = db
                return agent.run_conversation(
                    user_content,
                    conversation_history=history,
                    persist_user_message=persist_text,
                )
            return await loop.run_in_executor(None, _run)

        # Emit run.started
        await self.broadcast_event("run.started", {
            "session_id": session_id, "run_id": run_id,
            "user_message": {"id": user_message_id, "role": "user", "content": message},
        })
        await self.broadcast_event("message.started", {
            "session_id": session_id, "run_id": run_id,
            "message": {"id": assistant_message_id, "role": "assistant"},
        })

        agent_task = asyncio.ensure_future(_agent_task())
        self._agent_task = agent_task

        # Drain queue and broadcast to all browsers
        try:
            while True:
                try:
                    frame = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Flush remaining queue
                        while True:
                            try:
                                frame = stream_q.get_nowait()
                                if frame is None:
                                    break
                                await self._broadcast_frame(frame)
                            except _q.Empty:
                                break
                        break
                    continue

                if frame is None:
                    break
                await self._broadcast_frame(frame)

            # Get result and emit completion events
            try:
                result = await agent_task
            except Exception as exc:
                logger.error("[SessionConnection:%s] Agent run error: %s", session_id, exc)
                result = {
                    "messages": [],
                    "final_response": "",
                    "completed": False,
                    "failed": True,
                    "error": str(exc),
                }

            final_content = result.get("final_response") or ""
            completed = bool(result.get("completed", False))
            partial = bool(result.get("partial", False))
            interrupted = bool(result.get("interrupted", False))
            failed = bool(result.get("failed", False))
            error_message = str(result.get("error") or final_content or "Agent run failed")

            await self.broadcast_event("assistant.completed", {
                "session_id": session_id, "run_id": run_id,
                "message_id": assistant_message_id,
                "content": final_content,
                "completed": completed,
                "partial": partial,
                "interrupted": interrupted,
                "failed": failed,
                "error": error_message if failed else None,
            })
            if failed and not interrupted:
                await self.broadcast_event("run.failed", {
                    "session_id": session_id, "run_id": run_id,
                    "message_id": assistant_message_id,
                    "completed": completed,
                    "partial": partial,
                    "interrupted": interrupted,
                    "failed": True,
                    "error": error_message,
                    "api_calls": result.get("api_calls"),
                })
            else:
                await self.broadcast_event("run.completed", {
                    "session_id": session_id, "run_id": run_id,
                    "message_id": assistant_message_id,
                    "completed": completed,
                    "partial": partial,
                    "interrupted": interrupted,
                    "api_calls": result.get("api_calls"),
                })

        except Exception as exc:
            logger.error("[SessionConnection:%s] Error in agent run handler: %s", session_id, exc)
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except asyncio.CancelledError:
                    pass
            await self.broadcast_event("run.error", {
                "session_id": session_id, "run_id": run_id,
                "error": str(exc),
            })
        finally:
            self._agent_task = None

    async def _broadcast_frame(self, frame: bytes) -> None:
        """Decode an SSE frame and broadcast as a structured event."""
        try:
            text = frame.decode("utf-8")
            event_name = "data"
            event_data = {}
            for line in text.strip().split("\n"):
                if line.startswith("event: "):
                    event_name = line[7:]
                elif line.startswith("data: "):
                    try:
                        event_data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        event_data = {"raw": line[6:]}
            await self.broadcast_event(event_name, event_data)
        except Exception:
            pass

    async def interrupt_run(self) -> bool:
        """Interrupt the active agent run. Returns True if interrupted."""
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
            try:
                await self._agent_task
            except asyncio.CancelledError:
                pass
            self._agent_task = None
            self._agent_busy = False
            await self.broadcast_event("run.interrupted", {"session_id": self.session_id})
            return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_idle(self) -> bool:
        """True if no browsers, no active run, and no recent activity."""
        no_browsers = len(self._browsers) == 0
        no_run = not self._agent_busy
        stale = (time.monotonic() - self._last_activity) > _BUFFER_TTL_SECONDS
        return no_browsers and no_run and stale

    @property
    def browser_count(self) -> int:
        return len(self._browsers)


class ConnectionManager:
    """
    Global manager: session_id → SessionConnection.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionConnection] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def get_or_create(self, session_id: str) -> SessionConnection:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionConnection(session_id)
            return self._sessions[session_id]

    async def remove_session(self, session_id: str) -> None:
        async with self._lock:
            conn = self._sessions.pop(session_id, None)
        if conn:
            await conn.interrupt_run()
            # Close all browser connections
            for ws in list(conn._browsers):
                try:
                    await ws.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Browser WebSocket entry
    # ------------------------------------------------------------------

    async def handle_browser_ws(
        self,
        request: "Any",
        session_id: str,
        *,
        create_agent_fn,
        get_session_db_fn,
        build_user_content_fn,
        cors_headers_for_origin_fn,
    ) -> "Any":
        """
        Handle a browser WebSocket connection for a session.

        Protocol:
          - Client connects to /ws/stream/{session_id}?since={offset}
          - Server replays buffered events from offset, then streams live
          - Client sends JSON: {"type":"chat","message":"...","attachments":[]}
          - Server starts an agent run and proxies events via WebSocket
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        conn = await self.get_or_create(session_id)
        await conn.add_browser(ws)

        # Diff sync: replay missed events
        since_str = request.query.get("since", "0")
        try:
            since = int(since_str)
        except ValueError:
            since = 0

        missed = conn.get_events_since(since)
        for ev in missed:
            try:
                await ws.send_str(ev.to_json())
            except Exception:
                break

        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")

                if msg_type == "chat":
                    message = data.get("message", "")
                    attachments = data.get("attachments")
                    system_message = data.get("system_message")

                    if conn.is_agent_busy:
                        await ws.send_str(json.dumps({
                            "event": "run.busy",
                            "data": {"session_id": session_id, "error": "Agent run already in progress"},
                            "offset": conn.get_last_offset(),
                        }, ensure_ascii=False))
                        continue

                    try:
                        await conn.start_agent_run(
                            message, attachments, system_message,
                            create_agent_fn, get_session_db_fn, build_user_content_fn,
                        )
                    except Exception as exc:
                        logger.error("[ConnectionManager:%s] Chat handler error: %s", session_id, exc)
                        await conn.broadcast_event("run.error", {
                            "session_id": session_id, "error": str(exc),
                        })

                elif msg_type == "ping":
                    await ws.send_str(json.dumps({
                        "event": "pong",
                        "data": {"timestamp": time.time()},
                        "offset": conn.get_last_offset(),
                    }, ensure_ascii=False))

                elif msg_type == "interrupt":
                    interrupted = await conn.interrupt_run()
                    await ws.send_str(json.dumps({
                        "event": "run.interrupt_ack",
                        "data": {"session_id": session_id, "interrupted": interrupted},
                        "offset": conn.get_last_offset(),
                    }, ensure_ascii=False))

        except Exception as exc:
            logger.warning("[ConnectionManager:%s] WebSocket error: %s", session_id, exc)
        finally:
            await conn.remove_browser(ws)
            if not ws.closed:
                await ws.close()

            if conn.browser_count == 0:
                logger.info("[ConnectionManager:%s] Last browser disconnected; run continues if active", session_id)

        return ws

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def start_cleanup(self) -> None:
        """Start periodic cleanup of idle sessions."""
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            dead_sessions: List[str] = []
            async with self._lock:
                for sid, conn in list(self._sessions.items()):
                    if conn.is_idle:
                        dead_sessions.append(sid)
                for sid in dead_sessions:
                    del self._sessions[sid]
            for sid in dead_sessions:
                logger.info("[ConnectionManager] Cleaned up idle session %s", sid)


# Global singleton
CONNECTION_MANAGER = ConnectionManager()
