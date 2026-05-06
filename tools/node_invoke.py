"""
Node Invoke Tool

Execute commands on remote Hermes nodes via the HTTP gateway API.
Eliminates SSH escaping/encoding issues by running commands locally on the
remote machine through a persistent WebSocket connection.

Uses HTTP API (127.0.0.1:8642) instead of importing NODE_REGISTRY directly,
because the registry is a gateway-process singleton and tools run in a
separate interpreter.

Requires:
    - api_server enabled (API_SERVER_ENABLED=true)
    - A remote node client connected (e.g., hermes_node_client.py on Windows)

Example:
    node_invoke(node_id="dev-win01", command="terminal.exec",
                params={"cmd": "msbuild src\\miniport.vcxproj"})
"""

import json
import logging
import urllib.request
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:8642"


def _check_node_registry() -> bool:
    """Verify that the node HTTP API is reachable."""
    try:
        with urllib.request.urlopen(f"{API_BASE}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _api_get(path: str) -> dict:
    """Make a GET request to the node API."""
    with urllib.request.urlopen(f"{API_BASE}{path}", timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_post(path: str, body: dict) -> dict:
    """Make a POST request to the node API."""
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def node_invoke(
    node_id: str,
    command: str,
    params: Optional[Dict[str, Any]] = None,
    timeout_ms: int = 30000,
    idempotency_key: Optional[str] = None,
) -> str:
    """
    Invoke a command on a remote Hermes node.

    Args:
        node_id: Unique identifier of the target node (e.g., "dev-win01")
        command: Command to execute (e.g., "terminal.exec", "file.read", "msbuild")
        params: Command-specific parameters as a dict
        timeout_ms: Maximum time to wait for the node to respond (default 30s)
        idempotency_key: Optional key to deduplicate requests

    Returns:
        JSON string with keys: ok (bool), payload (dict), error (dict|null)

    Example:
        node_invoke("dev-win01", "terminal.exec", {"cmd": "dir", "cwd": "C:\\\\workspace"})
        node_invoke("dev-win01", "file.read", {"path": "C:\\\\workspace\\\\ile.txt", "encoding": "utf-8"})
        node_invoke("dev-win01", "msbuild", {"project": "src\\miniport\\miniport.vcxproj", "configuration": "Release"})
    """
    body = {
        "command": command,
        "params": params or {},
        "timeoutMs": timeout_ms,
    }
    if idempotency_key:
        body["idempotencyKey"] = idempotency_key

    result = _api_post(f"/v1/nodes/{node_id}/invoke", body)
    return json.dumps(result, ensure_ascii=False, default=str)


def node_list() -> str:
    """List all connected remote nodes."""
    result = _api_get("/v1/nodes")
    return json.dumps(result, ensure_ascii=False, default=str)


def node_describe(node_id: str) -> str:
    """Describe a specific connected node."""
    result = _api_get("/v1/nodes")
    if not result.get("ok"):
        return json.dumps(result, ensure_ascii=False)

    nodes = result.get("nodes", [])
    node = next((n for n in nodes if n.get("nodeId") == node_id), None)
    if not node:
        return json.dumps(
            {"ok": False, "error": {"code": "NOT_FOUND", "message": f"Node '{node_id}' not connected"}},
            ensure_ascii=False,
        )

    return json.dumps({"ok": True, "node": node}, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

registry.register(
    name="node_invoke",
    toolset="node",
    schema={
        "name": "node_invoke",
        "description": (
            "Invoke a command on a remote Hermes node connected via WebSocket. "
            "Use this to execute commands on remote machines (e.g., Windows build servers) "
            "without SSH escaping or encoding issues. Common commands: terminal.exec, "
            "file.read, file.write, msbuild, signtool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Unique identifier of the target node (e.g., dev-win01)",
                },
                "command": {
                    "type": "string",
                    "description": "Command to execute on the node (e.g., terminal.exec, file.read, msbuild)",
                },
                "params": {
                    "type": "object",
                    "description": "Command-specific parameters as a JSON object",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "Maximum time to wait for response in milliseconds (default 30000)",
                    "default": 30000,
                },
                "idempotency_key": {
                    "type": "string",
                    "description": "Optional key to deduplicate/retry requests safely",
                },
            },
            "required": ["node_id", "command"],
        },
    },
    handler=lambda args, **kw: node_invoke(
        node_id=args.get("node_id", ""),
        command=args.get("command", ""),
        params=args.get("params"),
        timeout_ms=args.get("timeout_ms", 30000),
        idempotency_key=args.get("idempotency_key"),
    ),
    check_fn=_check_node_registry,
)

registry.register(
    name="node_list",
    toolset="node",
    schema={
        "name": "node_list",
        "description": "List all remote nodes currently connected to the Hermes gateway.",
        "parameters": {"type": "object", "properties": {}},
    },
    handler=lambda args, **kw: node_list(),
    check_fn=_check_node_registry,
)

registry.register(
    name="node_describe",
    toolset="node",
    schema={
        "name": "node_describe",
        "description": "Get detailed information about a specific connected remote node.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Unique identifier of the node to describe",
                },
            },
            "required": ["node_id"],
        },
    },
    handler=lambda args, **kw: node_describe(args.get("node_id", "")),
    check_fn=_check_node_registry,
)
