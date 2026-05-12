"""
Workspace Router — Routes tool calls to remote nodes based on active workspace.

In "replace" mode: ALL target tools are routed to the active workspace's node.
The active workspace is the single source of truth — no path-based fallback.

Target tools: read_file, write_file, patch, search_files, terminal, execute_code
"""

import json
import logging
from typing import Any, Dict, Optional

from agent.workspace_manager import get_workspace_manager

logger = logging.getLogger(__name__)

# Tools that support workspace-based routing
_WORKSPACE_TOOLS = {
    "read_file", "write_file", "patch", "search_files",
    "terminal", "execute_code",
}


def _get_workspace_mode() -> str:
    """Read workspace_mode from config. Defaults to 'extend'."""
    try:
        from hermes_constants import get_hermes_home
        import yaml
        import os

        config_path = os.path.join(get_hermes_home(), "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            return config.get("workspace_mode", "extend")
    except Exception:
        pass
    return "extend"


def should_route_to_node(tool_name: str, _params: Optional[Dict] = None) -> Optional[str]:
    """
    In replace mode: if active workspace is non-default, route ALL workspace
    tools to that workspace's node. No path checks, no tool checks.
    """
    if tool_name not in _WORKSPACE_TOOLS:
        return None

    if _get_workspace_mode() != "replace":
        return None

    active = get_workspace_manager().get_active()
    if active.name == "default":
        return None

    return active.node_id


def route_tool_call(tool_name: str, params: Dict[str, Any], **_kwargs) -> Optional[str]:
    """
    Route a tool call to the active workspace's node.
    Returns JSON result string if routed, None for local execution.
    """
    node_id = should_route_to_node(tool_name)
    if node_id is None:
        return None

    from tools import node_lib

    try:
        if tool_name == "read_file":
            text = node_lib.node_read(node_id, params.get("path", ""))
            # Apply pagination
            offset = params.get("offset", 1)
            limit = params.get("limit", 500)
            from tools.file_operations import normalize_read_pagination
            offset, limit = normalize_read_pagination(offset, limit)
            lines = text.split("\n")
            start = max(0, offset - 1)
            end = min(len(lines), start + limit)
            selected = lines[start:end]
            result = []
            for i, line in enumerate(selected, start=offset):
                result.append(f"{i:>5}|{line}")
            return "\n".join(result)

        elif tool_name == "write_file":
            node_lib.node_write(node_id, params["path"], params["content"])
            return json.dumps({"ok": True, "message": f"Written to {params['path']} on {node_id}"})

        elif tool_name == "patch":
            if params.get("mode") == "replace":
                node_lib.node_patch(node_id, params["path"], [
                    {"old": params["old_string"], "new": params.get("new_string", "")}
                ])
                return json.dumps({"success": True, "message": f"Patched {params['path']} on {node_id}"})
            else:
                return json.dumps({"error": "V4A patch not supported on remote"})

        elif tool_name == "search_files":
            entries = node_lib.node_search(
                node_id, pattern=params.get("pattern", ""),
                path=params.get("path", "."),
                file_glob=params.get("file_glob"),
                target=params.get("target", "content"),
                limit=params.get("limit", 50)
            )
            return json.dumps({
                "total_count": len(entries),
                "matches": [
                    {"path": e.get("file", ""), "line": e.get("line", 0), "content": e.get("content", "")}
                    for e in entries
                ],
            }, ensure_ascii=False)

        elif tool_name == "terminal":
            cmd = params.get("command", "")
            workdir = params.get("workdir")
            if workdir:
                cmd = f"cd {workdir}; {cmd}"
            result = node_lib.node_exec(node_id, cmd, timeout=params.get("timeout", 180))
            payload = result.get("payload", {})
            return json.dumps({
                "output": payload.get("output", ""),
                "stderr": payload.get("stderr", ""),
                "exit_code": payload.get("exit_code", 0),
            }, ensure_ascii=False)

        elif tool_name == "execute_code":
            return _route_execute_code(node_id, params)

    except Exception as e:
        logger.exception("Workspace routing failed for %s on %s: %s", tool_name, node_id, e)
        return json.dumps({"error": f"Remote execution failed on {node_id}: {e}"}, ensure_ascii=False)

    return None


def _route_execute_code(node_id: str, params: Dict[str, Any]) -> str:
    """Route execute_code to a remote node."""
    import uuid

    code = params.get("code", "")
    if not code:
        return json.dumps({"error": "No code provided"}, ensure_ascii=False)

    # Detect if target node is Windows by node_id naming convention
    is_windows = node_id.startswith("dev-win") or node_id.startswith("win-")
    if is_windows:
        temp_name = f"C:/Users/goodb/workspace/.hermes_tmp_{uuid.uuid4().hex[:8]}.py"
        py_cmd = f"python {temp_name}"
        rm_cmd = f"Remove-Item -Path '{temp_name}' -Force -ErrorAction SilentlyContinue"
    else:
        temp_name = f"/tmp/hermes_workspace_{uuid.uuid4().hex[:8]}.py"
        py_cmd = f"python3 {temp_name}"
        rm_cmd = f"rm -f {temp_name}"

    try:
        from tools import node_lib
        node_lib.node_write(node_id, temp_name, code)
        timeout = params.get("timeout", 300)
        result = node_lib.node_exec(node_id, py_cmd, timeout=timeout)
        try:
            node_lib.node_exec(node_id, rm_cmd, timeout=10)
        except Exception:
            pass

        payload = result.get("payload", {})
        return json.dumps({
            "output": payload.get("output", ""),
            "exit_code": payload.get("exit_code", 0),
        }, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": f"Remote execute_code failed on {node_id}: {e}"}, ensure_ascii=False)


def get_routing_info(tool_name: str, _params: Optional[Dict] = None) -> Dict[str, Any]:
    """Return routing information for debugging."""
    active = get_workspace_manager().get_active()
    node_id = should_route_to_node(tool_name)

    return {
        "tool": tool_name,
        "mode": _get_workspace_mode(),
        "active_workspace": active.name,
        "active_node": active.node_id,
        "will_route": node_id is not None,
        "target_node": node_id,
    }
