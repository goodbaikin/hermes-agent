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


def _get_profile_config_sync(profile_name: str) -> Optional[Dict[str, Any]]:
    """Synchronous helper to get profile config (used in async context via thread pool if needed)."""
    if not profile_name or profile_name == 'default':
        return None
    try:
        from hermes_cli.profiles import get_profile_dir
        import yaml

        profile_dir = get_profile_dir(profile_name)
        config_path = profile_dir / "config.yaml"
        if not config_path.exists():
            return None

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        active_workspace = config.get("active_workspace")
        workspaces = config.get("workspaces", {})
        ws_config = workspaces.get(active_workspace, {}) if active_workspace else {}

        return {
            "name": profile_name,
            "active_workspace": active_workspace,
            "workspace_mode": config.get("workspace_mode"),
            "node_id": ws_config.get("node_id", "local"),
            "path_prefixes": ws_config.get("path_prefixes", []),
            "tools": ws_config.get("tools", []),
        }
    except Exception:
        return None


class TerminalProxy:
    """Manages a single browser <-> node terminal proxy connection."""

    def __init__(self, proxy_id: str, node_id: str, browser_ws: web.WebSocketResponse, cwd: str = None):
        self.proxy_id = proxy_id
        self.node_id = node_id
        self.browser_ws = browser_ws
        self.cwd = cwd
        self.node_session = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._open_event = asyncio.Event()

    async def start(self) -> None:
        """Start the proxy — open terminal on node then relay browser messages."""
        logger.info("[TerminalProxy %s] Starting proxy for node %s", self.proxy_id, self.node_id)
        self.node_session = NODE_REGISTRY.get(self.node_id)
        if not self.node_session:
            logger.error("[TerminalProxy %s] Node %s not found", self.proxy_id, self.node_id)
            await self._send_browser_error(f"Node {self.node_id} not connected")
            return

        # Open terminal session via node.invoke.request (waits for response)
        open_result = await NODE_REGISTRY.invoke(
            node_id=self.node_id,
            command="terminal.stream",
            params={
                "proxyId": self.proxy_id,
                "action": "open",
                "cols": 80,
                "rows": 24,
                "cwd": self.cwd,
            },
            timeout_ms=10000,
        )
        logger.info("[TerminalProxy %s] Open result: %s", self.proxy_id, open_result)
        if not open_result.get("ok"):
            error = open_result.get("error", {}).get("message", "Failed to open terminal")
            logger.error("[TerminalProxy %s] Open failed: %s", self.proxy_id, error)
            await self._send_browser_error(error)
            return

        logger.info("[TerminalProxy %s] Terminal opened on node", self.proxy_id)

        # Relay browser <-> node
        try:
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
                    self._send_to_node({
                        "proxyId": self.proxy_id,
                        "action": "write",
                        "data": base64.b64encode(msg.data).decode("ascii"),
                    })
                elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.CLOSING, web.WSMsgType.CLOSED):
                    break
        except Exception as exc:
            logger.debug("[TerminalProxy %s] Browser read error: %s", self.proxy_id, exc)

    async def _handle_browser_message(self, data: Dict[str, Any]) -> None:
        """Handle message from browser."""
        msg_type = data.get("type")
        logger.info("[TerminalProxy %s] Browser msg: type=%s", self.proxy_id, msg_type)

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
        logger.info("[TerminalProxy %s] Sending to node: action=%s session=%s", self.proxy_id, payload.get("action"), session is not None)
        if session is None:
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
            logger.warning("[TerminalProxy %s] Cannot send output: closed=%s ws.closed=%s", self.proxy_id, self._closed, self.browser_ws.closed)
            return
        logger.info("[TerminalProxy %s] Sending output to browser: %d bytes", self.proxy_id, len(data_b64))
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
                    data = json.dumps(msg)
                    logger.info("[TerminalProxy %s] Sending to browser: %s", self.proxy_id, data[:100])
                    await self.browser_ws.send_str(data)
                    logger.info("[TerminalProxy %s] Sent to browser OK", self.proxy_id)
                except Exception as exc:
                    logger.warning("[TerminalProxy %s] Browser send failed: %s", self.proxy_id, exc)

    async def _send_browser_error(self, message: str) -> None:
        """Send error message to browser."""
        await self._send_browser({"type": "error", "error": message})

    async def close(self) -> None:
        """Close the proxy connection."""
        if self._closed:
            return
        # Send close to node FIRST, before marking _closed
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
        self._closed = True
        if not self.browser_ws.closed:
            try:
                await self.browser_ws.close()
            except Exception:
                pass


class TerminalProxyManager:
    """Manages all active terminal proxy connections."""

    def __init__(self):
        self._proxies: Dict[str, TerminalProxy] = {}

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

# Track active terminal per node_id — only one terminal per node
_node_terminal_locks: Dict[str, asyncio.Lock] = {}

async def _acquire_node_terminal(node_id: str) -> asyncio.Lock:
    """Get or create a lock for the given node_id."""
    if node_id not in _node_terminal_locks:
        _node_terminal_locks[node_id] = asyncio.Lock()
    return _node_terminal_locks[node_id]

async def _close_existing_node_terminal(node_id: str) -> None:
    """Close any existing terminal proxy for the given node_id synchronously."""
    for proxy_id, proxy in list(TERMINAL_PROXY_MANAGER._proxies.items()):
        if proxy.node_id == node_id:
            logger.info("[TerminalProxy %s] Closing existing terminal for node %s", proxy_id, node_id)
            # Synchronously close via invoke so node_client cleanup completes before new open
            try:
                await NODE_REGISTRY.invoke(
                    node_id=node_id,
                    command="terminal.stream",
                    params={"proxyId": proxy_id, "action": "close"},
                    timeout_ms=5000,
                )
            except Exception as exc:
                logger.warning("[TerminalProxy %s] Sync close failed: %s", proxy_id, exc)
            # Also close browser side and unregister
            await proxy.close()
            TERMINAL_PROXY_MANAGER.unregister(proxy_id)
    # Extra safety: send a node-wide close-all to ensure no orphaned PTY
    try:
        await NODE_REGISTRY.invoke(
            node_id=node_id,
            command="terminal.stream",
            params={"proxyId": "__all__", "action": "close"},
            timeout_ms=3000,
        )
    except Exception:
        pass


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
            return auth_err

    node_id = request.match_info.get("node_id", "")
    if not node_id:
        return web.json_response({"error": "node_id required"}, status=400)

    # Resolve profile config to get cwd for the terminal
    profile_name = request.query.get("profile", "")
    cwd = None
    if profile_name:
        profile_config = _get_profile_config_sync(profile_name)
        if profile_config:
            path_prefixes = profile_config.get("path_prefixes", [])
            if path_prefixes:
                cwd = path_prefixes[0]

    # Close any existing terminal for this node before opening new one
    await _close_existing_node_terminal(node_id)

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    proxy_id = str(uuid.uuid4())
    proxy = TerminalProxy(proxy_id, node_id, ws, cwd=cwd)
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
