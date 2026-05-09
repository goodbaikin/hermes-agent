"""
Node Registry — Remote tool execution nodes for Hermes Agent API Server.

A lightweight WebSocket control plane that lets remote machines connect
to the API Server and expose local tool execution.

Usage:
    from api_server.node_registry import NODE_REGISTRY
    result = await NODE_REGISTRY.invoke("dev-win01", "terminal.exec", {"cmd": "dir"})
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeSession:
    """Represents a connected remote node."""

    def __init__(
        self,
        node_id: str,
        send_fn: Callable[[Dict[str, Any]], None],
        caps: List[str],
        commands: List[str],
        platform: str = "unknown",
        version: str = "unknown",
    ):
        self.node_id = node_id
        self.send = send_fn
        self.caps = caps
        self.commands = commands
        self.platform = platform
        self.version = version
        self.connected_at = asyncio.get_event_loop().time()

    def __repr__(self) -> str:
        return f"<NodeSession {self.node_id} ({self.platform})>"


class NodeRegistry:
    """Manages connected remote nodes and dispatches invoke requests."""

    def __init__(self):
        self._nodes: Dict[str, NodeSession] = {}
        # request_id -> (asyncio.Event, result_dict)
        self._pending: Dict[str, tuple[asyncio.Event, Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def register(self, session: NodeSession) -> None:
        async with self._lock:
            old = self._nodes.get(session.node_id)
            if old:
                logger.warning("Node %s reconnected; dropping old session", session.node_id)
            self._nodes[session.node_id] = session
            logger.info("Node registered: %s (platform=%s, commands=%d)",
                        session.node_id, session.platform, len(session.commands))

    async def unregister(self, node_id: str) -> None:
        async with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info("Node unregistered: %s", node_id)
            # Reject any pending invocations for this node
            for req_id, (event, result) in list(self._pending.items()):
                if req_id.startswith(f"{node_id}:"):
                    result["ok"] = False
                    result["error"] = {"code": "DISCONNECTED", "message": f"Node {node_id} disconnected"}
                    event.set()

    def get(self, node_id: str) -> Optional[NodeSession]:
        return self._nodes.get(node_id)

    def list_nodes(self) -> List[Dict[str, Any]]:
        return [
            {
                "nodeId": n.node_id,
                "platform": n.platform,
                "version": n.version,
                "caps": n.caps,
                "commands": n.commands,
                "connectedAt": n.connected_at,
            }
            for n in self._nodes.values()
        ]

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def invoke(
        self,
        node_id: str,
        command: str,
        params: Dict[str, Any],
        timeout_ms: int = 30000,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invoke a command on a remote node and wait for the response."""
        node = self._nodes.get(node_id)
        if not node:
            return {"ok": False, "error": {"code": "NOT_FOUND", "message": f"Node {node_id} not connected"}}

        if command not in node.commands:
            return {"ok": False, "error": {"code": "NOT_SUPPORTED", "message": f"Command {command} not supported by node {node_id}"}}

        req_id = f"{node_id}:{asyncio.get_event_loop().time()}:{id(node)}"
        event = asyncio.Event()
        result: Dict[str, Any] = {}

        async with self._lock:
            self._pending[req_id] = (event, result)

        try:
            node.send({
                "type": "event",
                "event": "node.invoke.request",
                "payload": {
                    "id": req_id,
                    "nodeId": node_id,
                    "command": command,
                    "paramsJSON": json.dumps(params) if params else None,
                    "timeoutMs": timeout_ms,
                    "idempotencyKey": idempotency_key,
                },
            })

            await asyncio.wait_for(event.wait(), timeout=timeout_ms / 1000)
            return result
        except asyncio.TimeoutError:
            return {"ok": False, "error": {"code": "TIMEOUT", "message": f"Node {node_id} did not respond within {timeout_ms}ms"}}
        finally:
            async with self._lock:
                self._pending.pop(req_id, None)

    def handle_response(self, req_id: str, payload: Dict[str, Any]) -> None:
        """Called when a node sends back an invoke response."""
        pending = self._pending.get(req_id)
        if pending:
            event, result = pending
            result.update(payload)
            event.set()
        else:
            logger.warning("Received response for unknown/expired request: %s", req_id)

    def handle_result(self, request_id: str, ok: bool, payload: Any, error: Optional[Dict[str, str]]) -> bool:
        """Called by the WebSocket handler when a node returns a result."""
        pending = self._pending.get(request_id)
        if not pending:
            logger.debug("Received result for unknown/expired request: %s", request_id)
            return False

        event, result_container = pending
        result_container["ok"] = ok
        result_container["payload"] = payload
        result_container["error"] = error
        event.set()
        return True


# Global singleton
NODE_REGISTRY = NodeRegistry()
