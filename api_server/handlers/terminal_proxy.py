"""WebSocket terminal proxy: browser <-> API Server <-> node_client."""

import asyncio
import base64
import hmac
import json
import logging
import uuid
from typing import Any, Dict, Optional

from aiohttp import web

from api_server.node_registry import NODE_REGISTRY

logger = logging.getLogger(__name__)


class TerminalProxy:
    """Manages a single browser <-> node terminal proxy connection."""

    def __init__(self, proxy_id: str, node_id: str, browser_ws: web.WebSocketResponse):
        self.proxy_id = proxy_id
        self.node_id = node_id
        self.browser_ws = browser_ws
        self.node_session = None
        self._closed = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the proxy — begin reading from browser."""
        self.node_session = NODE_REGISTRY.get(self.node_id)
        if not self.node_session:
            await self._send_browser_error(f"Node {self.node_id} not connected")
            return

        # Send open command to node
        self._send_to_node({
            "proxyId": self.proxy_id,
            "action": "open",
            "cols": 80,
            "rows": 24,
        })

        # Read from browser and forward to node
        async for msg in self.browser_ws:
            if self._closed:
                break
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    await self._handle_browser_message(data)
                except json.JSONDecodeError:
                    pass
            elif msg.type == web.WSMsgType.BINARY:
                # Accept binary data as raw terminal input
                self._send_to_node({
                    "proxyId": self.proxy_id,
                    "action": "write",
                    "data": base64.b64encode(msg.data).decode("ascii"),
                })
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
                break

    async def _handle_browser_message(self, data: Dict[str, Any]) -> None:
        """Handle message from browser."""
        msg_type = data.get("type")

        if msg_type == "input":
            self._send_to_node({
                "proxyId": self.proxy_id,
                "action": "write",
                "data": data.get("data", ""),
            })
        elif msg_type == "resize":
            self._send_to_node({
                "proxyId": self.proxy_id,
                "action": "resize",
                "cols": data.get("cols", 80),
                "rows": data.get("rows", 24),
            })
        elif msg_type == "close":
            self._send_to_node({
                "proxyId": self.proxy_id,
                "action": "close",
            })
            await self.close()
        elif msg_type == "ping":
            await self._send_browser({"type": "pong"})

    def _send_to_node(self, payload: Dict[str, Any]) -> None:
        """Send data to node via NODE_REGISTRY."""
        if self._closed:
            return
        session = NODE_REGISTRY.get(self.node_id)
        if session is None:
            logger.warning("[TerminalProxy %s] Node %s disconnected", self.proxy_id, self.node_id)
            asyncio.create_task(self._send_browser_error("Node disconnected"))
            asyncio.create_task(self.close())
            return
        try:
            session.send({
                "type": "event",
                "event": "node.terminal.data",
                "payload": payload,
            })
        except Exception as exc:
            logger.warning("[TerminalProxy %s] Failed to send to node: %s", self.proxy_id, exc)

    async def send_output_to_browser(self, data_b64: str) -> None:
        """Send base64-encoded output from node to browser."""
        if self._closed or self.browser_ws.closed:
            return
        await self._send_browser({"type": "output", "data": data_b64})

    async def send_close_to_browser(self) -> None:
        """Notify browser that terminal session closed."""
        if self._closed or self.browser_ws.closed:
            return
        await self._send_browser({"type": "close"})

    async def _send_browser(self, msg: Dict[str, Any]) -> None:
        """Send JSON message to browser."""
        async with self._lock:
            if not self.browser_ws.closed:
                try:
                    await self.browser_ws.send_str(json.dumps(msg))
                except Exception as exc:
                    logger.debug("[TerminalProxy %s] Browser send failed: %s", self.proxy_id, exc)

    async def _send_browser_error(self, message: str) -> None:
        """Send error message to browser."""
        await self._send_browser({"type": "error", "error": message})

    async def close(self) -> None:
        """Close the proxy connection."""
        if self._closed:
            return
        self._closed = True

        # Notify node to close terminal
        session = NODE_REGISTRY.get(self.node_id)
        if session is not None:
            try:
                session.send({
                    "type": "event",
                    "event": "node.terminal.data",
                    "payload": {
                        "proxyId": self.proxy_id,
                        "action": "close",
                    },
                })
            except Exception:
                pass

        if not self.browser_ws.closed:
            try:
                await self.browser_ws.close()
            except Exception:
                pass


class TerminalProxyManager:
    """Manages all active terminal proxy connections."""

    def __init__(self):
        self._proxies: Dict[str, TerminalProxy] = {}
        self._lock = asyncio.Lock()

    def register(self, proxy: TerminalProxy) -> None:
        self._proxies[proxy.proxy_id] = proxy

    def unregister(self, proxy_id: str) -> None:
        self._proxies.pop(proxy_id, None)

    def get(self, proxy_id: str) -> Optional[TerminalProxy]:
        return self._proxies.get(proxy_id)

    async def handle_node_output(self, proxy_id: str, data_b64: str) -> None:
        """Handle output from node — forward to browser."""
        proxy = self._proxies.get(proxy_id)
        if proxy:
            await proxy.send_output_to_browser(data_b64)

    async def handle_node_close(self, proxy_id: str) -> None:
        """Handle terminal close from node."""
        proxy = self._proxies.pop(proxy_id, None)
        if proxy:
            await proxy.send_close_to_browser()
            await proxy.close()

    async def handle_node_error(self, proxy_id: str, error: str) -> None:
        """Handle error from node."""
        proxy = self._proxies.pop(proxy_id, None)
        if proxy:
            await proxy._send_browser_error(error)
            await proxy.close()


# Global singleton
TERMINAL_PROXY_MANAGER = TerminalProxyManager()


async def handle_terminal_ws(
    request: web.Request,
    *,
    check_auth,
    api_key: str = "",
) -> web.WebSocketResponse:
    """GET /ws/nodes/{node_id}/terminal — WebSocket terminal proxy.

    Browser WebSocket cannot set custom headers, so we accept the token
    via query parameter ``?token=...`` as a fallback.
    """
    # Auth: try Authorization header first, then ?token= query param
    auth_err = check_auth(request)
    if auth_err:
        token = request.query.get("token", "")
        if not token or not api_key or not hmac.compare_digest(token, api_key):
            # Return a plain HTTP error before WS upgrade
            return auth_err

    node_id = request.match_info.get("node_id", "")
    if not node_id:
        return web.json_response({"error": "node_id required"}, status=400)

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    proxy_id = str(uuid.uuid4())
    proxy = TerminalProxy(proxy_id, node_id, ws)
    TERMINAL_PROXY_MANAGER.register(proxy)

    logger.info("[TerminalProxy %s] Browser connected for node %s", proxy_id, node_id)

    try:
        await proxy.start()
    except Exception as exc:
        logger.warning("[TerminalProxy %s] Error: %s", proxy_id, exc)
    finally:
        logger.info("[TerminalProxy %s] Disconnecting", proxy_id)
        TERMINAL_PROXY_MANAGER.unregister(proxy_id)
        await proxy.close()

    return ws
