"""
Node Invoke Tool — HTTP API wrapper for remote node execution.

Calls the API Server's /v1/nodes endpoints to invoke commands on remote nodes.
This avoids the process-isolation issue where Agent and Gateway run in separate processes.
"""

import json
import urllib.request
from typing import Any, Dict, Optional

API_BASE = "http://127.0.0.1:8642"
API_KEY = "aZQU8VBHKYr!PF"


def _api_request(path: str, method: str = "GET", body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make an HTTP request to the API Server."""
    url = f"{API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": {"code": "HTTP_ERROR", "message": str(e), "status": e.code}}
    except Exception as e:
        return {"ok": False, "error": {"code": "REQUEST_FAILED", "message": str(e)}}


def node_list() -> str:
    """List all connected remote nodes."""
    result = _api_request("/v1/nodes")
    return json.dumps(result, ensure_ascii=False, default=str)


def node_describe(node_id: str) -> str:
    """Describe a specific node by ID."""
    result = _api_request("/v1/nodes")
    nodes = result.get("nodes", [])
    node = next((n for n in nodes if n.get("nodeId") == node_id), None)
    if not node:
        return json.dumps({"ok": False, "error": {"code": "NOT_FOUND", "message": f"Node '{node_id}' not connected"}}, ensure_ascii=False)
    return json.dumps({"ok": True, "node": node}, ensure_ascii=False, default=str)


def node_invoke(node_id: str, command: str, params: Optional[Dict[str, Any]] = None, timeout_ms: int = 30000) -> str:
    """Invoke a command on a remote node."""
    body = {
        "command": command,
        "params": params or {},
        "timeoutMs": timeout_ms,
    }
    result = _api_request(f"/v1/nodes/{node_id}/invoke", method="POST", body=body)
    return json.dumps(result, ensure_ascii=False, default=str)


# Tool registration
if __name__ != "__main__":
    try:
        from tools.registry import register
        
        register(
            name="node_list",
            fn=node_list,
            description="List all connected remote nodes",
            parameters={},
        )
        register(
            name="node_describe",
            fn=node_describe,
            description="Describe a specific remote node",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID to describe"},
                },
                "required": ["node_id"],
            },
        )
        register(
            name="node_invoke",
            fn=node_invoke,
            description="Invoke a command on a remote node",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID to invoke"},
                    "command": {"type": "string", "description": "Command to execute"},
                    "params": {"type": "object", "description": "Command parameters"},
                    "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds", "default": 30000},
                },
                "required": ["node_id", "command"],
            },
        )
    except ImportError:
        pass
