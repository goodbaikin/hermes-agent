"""
SSE stream writer — unified send / keepalive / drain / disconnect handling.

Consolidates duplicated logic from _write_sse_chat_completion and
_write_sse_responses into a single reusable layer.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0


class SSEWriter:
    """Wraps aiohttp StreamResponse for consistent SSE output.

    Handles:
      - CORS header injection (up-front, before prepare())
      - Keepalive ping on idle
      - Graceful drain on queue exhaustion
      - Connection-reset cleanup
    """

    def __init__(
        self,
        request: "web.Request",
        cors_headers: Optional[Dict[str, str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        keepalive_interval: float = CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS,
    ):
        self._request = request
        self._keepalive_interval = keepalive_interval
        self._last_activity = time.monotonic()
        self._closed = False

        headers: Dict[str, str] = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        if cors_headers:
            headers.update(cors_headers)
        if extra_headers:
            headers.update(extra_headers)
        self._headers = headers
        self._response: Optional["web.StreamResponse"] = None

    async def prepare(self) -> "web.StreamResponse":
        """Prepare the StreamResponse and return it."""
        if self._response is not None:
            return self._response
        self._response = web.StreamResponse(status=200, headers=self._headers)
        await self._response.prepare(self._request)
        self._last_activity = time.monotonic()
        return self._response

    async def write_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write a named SSE event."""
        if self._closed or self._response is None:
            return
        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        await self._response.write(payload.encode())
        self._last_activity = time.monotonic()

    async def write_data(self, data: Dict[str, Any]) -> None:
        """Write a default ``data:`` SSE chunk."""
        if self._closed or self._response is None:
            return
        payload = f"data: {json.dumps(data)}\n\n"
        await self._response.write(payload.encode())
        self._last_activity = time.monotonic()

    async def write_done(self) -> None:
        """Write the OpenAI-style [DONE] sentinel."""
        if self._closed or self._response is None:
            return
        await self._response.write(b"data: [DONE]\n\n")
        self._last_activity = time.monotonic()

    async def keepalive(self) -> None:
        """Send a keepalive comment if the interval has elapsed."""
        if self._closed or self._response is None:
            return
        if time.monotonic() - self._last_activity >= self._keepalive_interval:
            await self._response.write(b": keepalive\n\n")
            self._last_activity = time.monotonic()

    async def drain_queue(self, stream_q, agent_task, emit_fn) -> None:
        """Drain a queue until the sentinel ``None`` arrives or the task ends.

        Args:
            stream_q: queue.Queue (or asyncio.Queue) yielding items.
            agent_task: asyncio.Task that produces items.
            emit_fn: async callable(item) invoked for each non-None item.
        """
        import queue as _q

        loop = asyncio.get_running_loop()
        while True:
            try:
                delta = await loop.run_in_executor(
                    None, lambda: stream_q.get(timeout=0.5)
                )
            except _q.Empty:
                if agent_task.done():
                    # Flush remainder
                    while True:
                        try:
                            delta = stream_q.get_nowait()
                            if delta is None:
                                break
                            await emit_fn(delta)
                        except _q.Empty:
                            break
                    break
                await self.keepalive()
                continue

            if delta is None:
                break
            await emit_fn(delta)

    async def write_raw(self, data: bytes) -> None:
        """Write raw bytes directly to the underlying response."""
        if self._closed or self._response is None:
            return
        await self._response.write(data)
        self._last_activity = time.monotonic()

    async def close(self) -> None:
        """Mark writer closed; idempotent."""
        self._closed = True

    async def __aenter__(self):
        await self.prepare()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


def build_cors_headers(origin: str, allowed_origins: tuple[str, ...]) -> Optional[Dict[str, str]]:
    """Return CORS headers dict for an allowed origin, or None."""
    if not origin or not allowed_origins:
        return None

    _CORS_HEADERS = {
        "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Authorization, Content-Type, X-Hermes-Session-Id, X-Idempotency-Key",
        "Access-Control-Allow-Credentials": "true",
    }

    if "*" in allowed_origins:
        headers = dict(_CORS_HEADERS)
        headers["Access-Control-Allow-Origin"] = "*"
        headers["Access-Control-Max-Age"] = "600"
        return headers

    if origin not in allowed_origins:
        return None

    headers = dict(_CORS_HEADERS)
    headers["Access-Control-Allow-Origin"] = origin
    headers["Vary"] = "Origin"
    headers["Access-Control-Max-Age"] = "600"
    return headers
