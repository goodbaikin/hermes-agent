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
    
    # ローカルノードの場合は直接実行
    if node_id == "local":
        return _execute_local(command, params or {})
    
    body = {
        "command": command,
        "params": params or {},
        "timeoutMs": timeout_ms,
    }
    result = _api_request(f"/v1/nodes/{node_id}/invoke", method="POST", body=body)
    return json.dumps(result, ensure_ascii=False, default=str)


def _execute_local(command: str, params: Dict[str, Any]) -> str:
    """ローカルで直接ツール実行"""
    try:
        from model_tools import handle_function_call
        result = handle_function_call(command, params)
        return json.dumps({"ok": True, "payload": result}, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"ok": False, "error": {"code": "LOCAL_EXEC_FAILED", "message": str(e)}}, ensure_ascii=False)


# Tool registration
try:
    from tools.registry import registry
    
    registry.register(
        name="node_list",
        toolset="node",
        schema={
            "name": "node_list",
            "description": "List all connected remote nodes",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        handler=lambda args, **kw: node_list(),
    )
    registry.register(
        name="node_describe",
        toolset="node",
        schema={
            "name": "node_describe",
            "description": "Describe a specific remote node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID to describe"},
                },
                "required": ["node_id"],
            },
        },
        handler=lambda args, **kw: node_describe(
            node_id=args.get("node_id", ""),
        ),
    )
    registry.register(
        name="node_invoke",
        toolset="node",
        schema={
            "name": "node_invoke",
            "description": "Invoke a command on a remote node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID to invoke"},
                    "command": {"type": "string", "description": "Command to execute"},
                    "params": {"type": "object", "description": "Command parameters"},
                    "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds", "default": 30000},
                },
                "required": ["node_id", "command"],
            },
        },
        handler=lambda args, **kw: node_invoke(
            node_id=args.get("node_id", ""),
            command=args.get("command", ""),
            params=args.get("params"),
            timeout_ms=args.get("timeout_ms", 30000),
        ),
    )
except ImportError:
    pass
