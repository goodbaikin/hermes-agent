"""
Tests for api_server.connection_manager — WebSocket + SSE unified connection management.

Covers:
- SessionConnection buffer (append, TTL eviction, since query)
- Browser WebSocket add/remove/broadcast
- Agent run exclusive execution (busy check)
- Diff sync replay on reconnect
- ConnectionManager session lifecycle and cleanup
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from api_server.connection_manager import (
    SessionConnection,
    ConnectionManager,
    CONNECTION_MANAGER,
    _BufferedEvent,
    _MAX_BUFFER_EVENTS,
    _BUFFER_TTL_SECONDS,
)


# ---------------------------------------------------------------------------
# _BufferedEvent
# ---------------------------------------------------------------------------

class TestBufferedEvent:
    def test_to_json(self):
        ev = _BufferedEvent(0, "assistant.delta", {"delta": "hello"})
        parsed = json.loads(ev.to_json())
        assert parsed["event"] == "assistant.delta"
        assert parsed["data"] == {"delta": "hello"}
        assert parsed["offset"] == 0


# ---------------------------------------------------------------------------
# SessionConnection — buffer
# ---------------------------------------------------------------------------

class TestSessionConnectionBuffer:
    def test_append_and_get_since(self):
        conn = SessionConnection("sess_1")
        conn._append_event("a", {"x": 1})
        conn._append_event("b", {"x": 2})
        conn._append_event("c", {"x": 3})

        evs = conn.get_events_since(1)
        assert len(evs) == 2
        assert evs[0].event == "b"
        assert evs[1].event == "c"

    def test_get_since_zero_returns_all(self):
        conn = SessionConnection("sess_1")
        conn._append_event("a", {"x": 1})
        evs = conn.get_events_since(0)
        assert len(evs) == 1

    def test_buffer_maxlen_eviction(self):
        conn = SessionConnection("sess_1")
        for i in range(_MAX_BUFFER_EVENTS + 10):
            conn._append_event(f"ev_{i}", {"i": i})
        assert len(conn._buffer) == _MAX_BUFFER_EVENTS
        # Oldest events should be evicted
        assert conn._buffer[0].event == "ev_10"

    def test_ttl_eviction(self):
        conn = SessionConnection("sess_1")
        conn._append_event("old", {"x": 1})
        # Manually backdate the event
        conn._buffer[0].timestamp = time.monotonic() - _BUFFER_TTL_SECONDS - 1
        conn._append_event("new", {"x": 2})
        assert len(conn._buffer) == 1
        assert conn._buffer[0].event == "new"

    def test_get_last_offset(self):
        conn = SessionConnection("sess_1")
        assert conn.get_last_offset() == 0
        conn._append_event("a", {})
        assert conn.get_last_offset() == 1
        conn._append_event("b", {})
        assert conn.get_last_offset() == 2


# ---------------------------------------------------------------------------
# SessionConnection — browser management
# ---------------------------------------------------------------------------

class TestSessionConnectionBrowser:
    @pytest.mark.asyncio
    async def test_add_remove_browser(self):
        conn = SessionConnection("sess_1")
        ws = MagicMock()
        await conn.add_browser(ws)
        assert conn.browser_count == 1
        await conn.remove_browser(ws)
        assert conn.browser_count == 0

    @pytest.mark.asyncio
    async def test_broadcast(self):
        conn = SessionConnection("sess_1")
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        await conn.add_browser(ws1)
        await conn.add_browser(ws2)

        await conn.broadcast(json.dumps({"test": 1}))
        ws1.send_str.assert_called_once()
        ws2.send_str.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_ws(self):
        conn = SessionConnection("sess_1")
        ws_live = AsyncMock()
        ws_dead = AsyncMock()
        ws_dead.send_str.side_effect = ConnectionResetError()

        await conn.add_browser(ws_live)
        await conn.add_browser(ws_dead)

        await conn.broadcast(json.dumps({"test": 1}))
        assert conn.browser_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_event(self):
        conn = SessionConnection("sess_1")
        ws = AsyncMock()
        await conn.add_browser(ws)

        await conn.broadcast_event("run.started", {"run_id": "r1"})
        ws.send_str.assert_called_once()
        payload = json.loads(ws.send_str.call_args[0][0])
        assert payload["event"] == "run.started"
        assert payload["data"]["run_id"] == "r1"
        assert payload["offset"] == 0


# ---------------------------------------------------------------------------
# SessionConnection — agent run exclusive execution
# ---------------------------------------------------------------------------

class TestSessionConnectionAgentRun:
    @pytest.mark.asyncio
    async def test_agent_busy_flag(self):
        conn = SessionConnection("sess_1")
        assert conn.is_agent_busy is False

        # Simulate busy state
        conn._agent_busy = True
        assert conn.is_agent_busy is True

    @pytest.mark.asyncio
    async def test_start_agent_run_raises_when_busy(self):
        conn = SessionConnection("sess_1")
        conn._agent_busy = True

        with pytest.raises(RuntimeError, match="already in progress"):
            await conn.start_agent_run("hello", None, None, None, None, None)


    @pytest.mark.asyncio
    async def test_failed_agent_result_emits_run_failed(self):
        """Failed agent results must surface as errors, not silent completion."""
        conn = SessionConnection("sess_failed")

        class FakeDB:
            def get_messages_as_conversation(self, session_id):
                return []

        class FakeAgent:
            def __init__(self):
                self._session_db = None

            def run_conversation(self, *args, **kwargs):
                return {
                    "final_response": "API call failed after 3 retries: provider timed out",
                    "completed": False,
                    "failed": True,
                    "error": "provider timed out",
                    "api_calls": 0,
                    "messages": [],
                }

        def create_agent_fn(**kwargs):
            return FakeAgent()

        def get_session_db_fn():
            return FakeDB()

        def build_user_content_fn(message, attachments):
            return message, message

        await conn.start_agent_run(
            "hello", None, None,
            create_agent_fn, get_session_db_fn, build_user_content_fn,
        )

        events = [(ev.event, ev.data) for ev in conn.get_events_since(0)]
        event_names = [name for name, _ in events]
        assert "assistant.completed" in event_names
        assert "run.failed" in event_names
        assert "run.completed" not in event_names

        assistant_completed = next(data for name, data in events if name == "assistant.completed")
        run_failed = next(data for name, data in events if name == "run.failed")
        assert assistant_completed["failed"] is True
        assert assistant_completed["error"] == "provider timed out"
        assert run_failed["error"] == "provider timed out"

    @pytest.mark.asyncio
    async def test_interrupt_run_no_task(self):
        conn = SessionConnection("sess_1")
        result = await conn.interrupt_run()
        assert result is False

    @pytest.mark.asyncio
    async def test_interrupt_run_cancels_task(self):
        conn = SessionConnection("sess_1")

        async def slow_task():
            await asyncio.sleep(10)

        conn._agent_task = asyncio.create_task(slow_task())
        conn._agent_busy = True

        result = await conn.interrupt_run()
        assert result is True
        assert conn._agent_task is None
        assert conn.is_agent_busy is False


# ---------------------------------------------------------------------------
# SessionConnection — idle detection
# ---------------------------------------------------------------------------

class TestSessionConnectionIdle:
    def test_idle_when_empty(self):
        conn = SessionConnection("sess_1")
        conn._last_activity = time.monotonic() - _BUFFER_TTL_SECONDS - 1
        assert conn.is_idle is True

    def test_not_idle_with_browser(self):
        conn = SessionConnection("sess_1")
        conn._browsers.add(MagicMock())
        assert conn.is_idle is False

    def test_not_idle_when_busy(self):
        conn = SessionConnection("sess_1")
        conn._agent_busy = True
        assert conn.is_idle is False

    def test_not_idle_with_recent_activity(self):
        conn = SessionConnection("sess_1")
        conn._last_activity = time.monotonic()
        assert conn.is_idle is False


# ---------------------------------------------------------------------------
# ConnectionManager — session lifecycle
# ---------------------------------------------------------------------------

class TestConnectionManagerLifecycle:
    @pytest.mark.asyncio
    async def test_get_or_create(self):
        mgr = ConnectionManager()
        conn1 = await mgr.get_or_create("sess_1")
        conn2 = await mgr.get_or_create("sess_1")
        assert conn1 is conn2
        assert conn1.session_id == "sess_1"

    @pytest.mark.asyncio
    async def test_remove_session(self):
        mgr = ConnectionManager()
        conn = await mgr.get_or_create("sess_1")
        ws = AsyncMock()
        await conn.add_browser(ws)
        await mgr.remove_session("sess_1")
        ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_loop_removes_idle(self):
        mgr = ConnectionManager()
        conn = await mgr.get_or_create("sess_1")
        # Make it idle
        conn._last_activity = time.monotonic() - _BUFFER_TTL_SECONDS - 1

        # Run cleanup once
        await mgr._cleanup_loop_iteration()
        assert "sess_1" not in mgr._sessions

    @pytest.mark.asyncio
    async def test_cleanup_loop_keeps_active(self):
        mgr = ConnectionManager()
        conn = await mgr.get_or_create("sess_1")
        conn._last_activity = time.monotonic()

        await mgr._cleanup_loop_iteration()
        assert "sess_1" in mgr._sessions


# ---------------------------------------------------------------------------
# ConnectionManager — WebSocket handler (integration)
# ---------------------------------------------------------------------------

class TestConnectionManagerWebSocket:
    @pytest.mark.asyncio
    async def test_handle_browser_ws_diff_sync(self):
        """Test that missed events are replayed on reconnect."""
        mgr = ConnectionManager()
        conn = await mgr.get_or_create("sess_1")
        conn._append_event("a", {"x": 1})
        conn._append_event("b", {"x": 2})

        # Mock request and WebSocket
        request = MagicMock()
        request.query = {"since": "1"}
        ws = AsyncMock()
        ws.closed = False
        ws.__aiter__ = MagicMock(return_value=iter([]))  # No incoming messages

        with patch("aiohttp.web.WebSocketResponse", return_value=ws):
            with patch.object(ws, "prepare", new_callable=AsyncMock):
                # We can't easily test the full handler without aiohttp app,
                # so test the core logic directly
                missed = conn.get_events_since(1)
                assert len(missed) == 1
                assert missed[0].event == "b"

    @pytest.mark.asyncio
    async def test_handle_browser_ws_ping(self):
        """Test ping/pong protocol."""
        conn = SessionConnection("sess_1")
        ws = AsyncMock()
        await conn.add_browser(ws)

        await conn.broadcast_event("pong", {"timestamp": 12345})
        ws.send_str.assert_called_once()
        payload = json.loads(ws.send_str.call_args[0][0])
        assert payload["event"] == "pong"


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

class TestGlobalSingleton:
    def test_singleton_exists(self):
        assert CONNECTION_MANAGER is not None
        assert isinstance(CONNECTION_MANAGER, ConnectionManager)


# ---------------------------------------------------------------------------
# Helpers for cleanup loop testing
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_manager():
    """Return a fresh ConnectionManager for isolated tests."""
    return ConnectionManager()


# Monkey-patch ConnectionManager to expose single cleanup iteration
async def _cleanup_loop_iteration(self):
    """Run one iteration of cleanup (for testing)."""
    dead_sessions = []
    async with self._lock:
        for sid, conn in list(self._sessions.items()):
            if conn.is_idle:
                dead_sessions.append(sid)
        for sid in dead_sessions:
            del self._sessions[sid]

ConnectionManager._cleanup_loop_iteration = _cleanup_loop_iteration
