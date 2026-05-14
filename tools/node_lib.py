"""
Node Library — High-level wrappers for remote/local node operations.

Provides read/write/patch/exec/list functions that work with both remote nodes
(via node_invoke HTTP API) and local node (direct execution).
"""

import base64
import json
from typing import Any, Dict, List, Optional

from tools.node_invoke import node_invoke


def _parse_result(result_str: str) -> Dict[str, Any]:
    """Parse node_invoke JSON result and check for errors."""
    result = json.loads(result_str)
    if not result.get("ok"):
        error = result.get("error", {})
        raise RuntimeError(f"Node operation failed: {error.get('message', 'Unknown error')}")
    
    # payload may be double-encoded JSON string
    payload = result.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
            result["payload"] = payload
        except json.JSONDecodeError:
            pass  # keep as string if not valid JSON
    
    return result


def node_read(node_id: str, path: str, offset: int = 1, limit: int = None) -> str:
    """Read a file from a node (text). Supports pagination via offset/limit."""
    # Expand ~ to home directory
    if path.startswith("~"):
        import os
        path = os.path.expanduser(path)
    elif not path.startswith("/") and not path.startswith("C:"):
        # Resolve relative path using workspace config
        if node_id == "local":
            import os
            path = os.path.abspath(path)
        else:
            from agent.workspace_manager import get_node_workdir
            workdir = get_node_workdir(node_id)
            if workdir:
                # Ensure trailing slash for concatenation
                prefix = workdir if workdir.endswith("/") else workdir + "/"
                path = prefix + path
            else:
                # Fallback for remote Windows nodes
                if node_id.startswith("dev-win") or node_id.startswith("win-"):
                    path = f"C:/Users/goodb/workspace/{path}"
                else:
                    path = f"/home/node/workspace/{path}"
    
    params = {"path": path, "encoding": "utf-8"}
    if offset is not None:
        params["offset"] = offset
    if limit is not None:
        params["limit"] = limit
    
    result_str = node_invoke(node_id, "file.read", params)
    result = _parse_result(result_str)
    
    payload = result["payload"]
    if payload.get("binary") or payload.get("encoding") == "base64":
        raw = base64.b64decode(payload["content"])
        return raw.decode("utf-8-sig")  # BOM auto-remove
    return payload.get("content", "")


def node_write(node_id: str, path: str, content: str) -> Dict[str, Any]:
    """Write a file to a node (text)."""
    # Expand ~ to home directory
    if path.startswith("~"):
        import os
        path = os.path.expanduser(path)
    elif not path.startswith("/") and not path.startswith("C:"):
        # Resolve relative path using workspace config
        if node_id == "local":
            import os
            path = os.path.abspath(path)
        else:
            from agent.workspace_manager import get_node_workdir
            workdir = get_node_workdir(node_id)
            if workdir:
                prefix = workdir if workdir.endswith("/") else workdir + "/"
                path = prefix + path
            else:
                if node_id.startswith("dev-win") or node_id.startswith("win-"):
                    path = f"C:/Users/goodb/workspace/{path}"
                else:
                    path = f"/home/node/workspace/{path}"
    
    content_b64 = base64.b64encode(content.encode("utf-8")).decode()
    result_str = node_invoke(node_id, "file.write", {
        "path": path,
        "content": content_b64,
    })
    return _parse_result(result_str)


def node_patch(node_id: str, path: str, changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply content-based patches to a file on a node."""
    text = node_read(node_id, path)
    
    for change in changes:
        old = change["old"]
        new = change["new"]
        
        if isinstance(old, list):
            old = "\n".join(old)
        if isinstance(new, list):
            new = "\n".join(new)
        
        if old not in text:
            raise ValueError(f"Patch target not found: {old[:100]}...")
        
        text = text.replace(old, new, 1)
    
    return node_write(node_id, path, text)


def node_exec(node_id: str, cmd: str, timeout: int = 30, cwd: str = None) -> Dict[str, Any]:
    """Execute a command on a node."""
    params = {
        "cmd": cmd,
        "timeout": timeout,
    }
    if cwd:
        params["cwd"] = cwd
    
    result_str = node_invoke(node_id, "terminal.exec", params)
    result = _parse_result(result_str)
    
    # Normalize payload: terminal.exec returns stdout/stderr/exitCode
    payload = result.get("payload", {})
    if "stdout" in payload:
        result["payload"] = {
            "output": payload.get("stdout", ""),
            "stderr": payload.get("stderr", ""),
            "exit_code": payload.get("exitCode", 0),
        }
    
    return result


def node_list_dir(node_id: str, path: str) -> List[Dict[str, Any]]:
    """List directory contents on a node."""
    result_str = node_invoke(node_id, "file.list", {"path": path})
    result = _parse_result(result_str)
    return result["payload"].get("entries", [])


def node_search(node_id: str, pattern: str, path: str = ".", file_glob: str = None, 
                target: str = "content", limit: int = 50) -> List[Dict[str, Any]]:
    """Search files on a node using native node commands (search.content/search.files)."""
    
    # Resolve relative path to absolute for local node
    if not path.startswith("/") and not path.startswith("~") and not path.startswith("C:"):
        if node_id == "local":
            import os
            path = os.path.abspath(path)
        else:
            from agent.workspace_manager import get_node_workdir
            workdir = get_node_workdir(node_id)
            if workdir:
                prefix = workdir if workdir.endswith("/") else workdir + "/"
                path = prefix + path
    
    if target == "files":
        result_str = node_invoke(node_id, "search.files", {
            "pattern": file_glob or pattern,
            "path": path,
            "limit": limit,
        })
    else:
        result_str = node_invoke(node_id, "search.content", {
            "pattern": pattern,
            "path": path,
            "file_glob": file_glob,
            "limit": limit,
        })
    
    result = _parse_result(result_str)
    payload = result["payload"]
    
    if target == "files":
        return [
            {
                "file": m.get("path", ""),
                "line": 0,
                "content": "",
            }
            for m in payload.get("matches", [])
        ]
    else:
        return [
            {
                "file": m.get("path", ""),
                "line": m.get("line", 0),
                "content": m.get("content", ""),
            }
            for m in payload.get("matches", [])
        ]


def node_find_files(node_id: str, pattern: str, path: str = ".", limit: int = 50) -> List[str]:
    """Find files by name pattern on a node using native search.files command."""
    
    # Resolve relative path to absolute for local node
    if not path.startswith("/") and not path.startswith("~") and not path.startswith("C:"):
        if node_id == "local":
            import os
            path = os.path.abspath(path)
        else:
            from agent.workspace_manager import get_node_workdir
            workdir = get_node_workdir(node_id)
            if workdir:
                prefix = workdir if workdir.endswith("/") else workdir + "/"
                path = prefix + path

    result_str = node_invoke(node_id, "search.files", {
        "pattern": pattern,
        "path": path,
        "limit": limit,
    })
    
    result = _parse_result(result_str)
    payload = result["payload"]
    
    return [m.get("path", "") for m in payload.get("matches", [])]


# ---------------------------------------------------------------------------
# Computer Use wrappers
# ---------------------------------------------------------------------------

def node_screenshot(node_id: str, region: list = None, redact_regions: list = None) -> dict:
    """Take a screenshot on a remote node. Returns dict with screenshot_b64."""
    params = {"action": "screenshot"}
    if region:
        params["region"] = region
    if redact_regions:
        params["redact_regions"] = redact_regions
    result_str = node_invoke(node_id, "computer.use", params)
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_click(node_id: str, x: int, y: int, button: str = "left") -> dict:
    """Click at coordinates on a remote node."""
    action_map = {
        "left": "left_click",
        "right": "right_click",
        "middle": "middle_click",
    }
    result_str = node_invoke(node_id, "computer.use", {
        "action": action_map.get(button, "left_click"),
        "x": x, "y": y,
    })
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_double_click(node_id: str, x: int, y: int) -> dict:
    """Double-click at coordinates on a remote node."""
    result_str = node_invoke(node_id, "computer.use", {
        "action": "double_click", "x": x, "y": y,
    })
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_mouse_move(node_id: str, x: int, y: int) -> dict:
    """Move mouse to coordinates on a remote node."""
    result_str = node_invoke(node_id, "computer.use", {
        "action": "mouse_move", "x": x, "y": y,
    })
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_drag(node_id: str, x1: int, y1: int, x2: int, y2: int) -> dict:
    """Drag from (x1,y1) to (x2,y2) on a remote node."""
    result_str = node_invoke(node_id, "computer.use", {
        "action": "mouse_drag",
        "x": x1, "y": y1, "x2": x2, "y2": y2,
    })
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_type(node_id: str, text: str) -> dict:
    """Type text on a remote node."""
    result_str = node_invoke(node_id, "computer.use", {
        "action": "type", "text": text,
    })
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_key(node_id: str, keys: str) -> dict:
    """Send key combination on a remote node (e.g. 'Win+R', 'Ctrl+C')."""
    result_str = node_invoke(node_id, "computer.use", {
        "action": "key", "keys": keys,
    })
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_scroll(node_id: str, direction: str = "down", amount: int = 3) -> dict:
    """Scroll on a remote node."""
    result_str = node_invoke(node_id, "computer.use", {
        "action": "scroll", "direction": direction, "amount": amount,
    })
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_cursor_position(node_id: str) -> dict:
    """Get cursor position on a remote node."""
    result_str = node_invoke(node_id, "computer.use", {"action": "cursor_position"})
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_screen_size(node_id: str) -> dict:
    """Get screen size on a remote node."""
    result_str = node_invoke(node_id, "computer.use", {"action": "screen_size"})
    result = _parse_result(result_str)
    return result.get("payload", {})

def node_get_active_window(node_id: str) -> dict:
    """Get active window on a remote node."""
    result_str = node_invoke(node_id, "computer.use", {"action": "get_active_window"})
    result = _parse_result(result_str)
    return result.get("payload", {})


# Tool registration
try:
    from tools.registry import registry
    
    registry.register(
        name="node_read",
        toolset="node",
        schema={
            "name": "node_read",
            "description": "Read a file from a local or remote node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Line number to start from (1-based)", "default": 1},
                    "limit": {"type": "integer", "description": "Maximum lines to return", "default": 500},
                },
                "required": ["node_id", "path"],
            },
        },
        handler=lambda args, **kw: node_read(
            node_id=args.get("node_id", ""),
            path=args.get("path", ""),
            offset=args.get("offset", 1),
            limit=args.get("limit"),
        ),
    )
    registry.register(
        name="node_write",
        toolset="node",
        schema={
            "name": "node_write",
            "description": "Write a file to a local or remote node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Text content to write"},
                },
                "required": ["node_id", "path", "content"],
            },
        },
        handler=lambda args, **kw: node_write(
            node_id=args.get("node_id", ""),
            path=args.get("path", ""),
            content=args.get("content", ""),
        ),
    )
    registry.register(
        name="node_patch",
        toolset="node",
        schema={
            "name": "node_patch",
            "description": "Apply content-based patches to a file on a node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "path": {"type": "string", "description": "File path to patch"},
                    "changes": {
                        "type": "array",
                        "description": "List of {old, new} changes",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old": {"type": "string", "description": "Text to replace"},
                                "new": {"type": "string", "description": "Replacement text"},
                            },
                            "required": ["old", "new"],
                        },
                    },
                },
                "required": ["node_id", "path", "changes"],
            },
        },
        handler=lambda args, **kw: node_patch(
            node_id=args.get("node_id", ""),
            path=args.get("path", ""),
            changes=args.get("changes", []),
        ),
    )
    registry.register(
        name="node_exec",
        toolset="node",
        schema={
            "name": "node_exec",
            "description": "Execute a command on a local or remote node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "cmd": {"type": "string", "description": "Command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "required": ["node_id", "cmd"],
            },
        },
        handler=lambda args, **kw: node_exec(
            node_id=args.get("node_id", ""),
            cmd=args.get("cmd", ""),
            timeout=args.get("timeout", 30),
        ),
    )
    registry.register(
        name="node_list_dir",
        toolset="node",
        schema={
            "name": "node_list_dir",
            "description": "List directory contents on a local or remote node",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "path": {"type": "string", "description": "Directory path to list"},
                },
                "required": ["node_id", "path"],
            },
        },
        handler=lambda args, **kw: node_list_dir(
            node_id=args.get("node_id", ""),
            path=args.get("path", ""),
        ),
    )
    registry.register(
        name="node_search",
        toolset="node",
        schema={
            "name": "node_search",
            "description": "Search file contents on a node (OS-agnostic: uses ripgrep on Linux, PowerShell on Windows)",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "pattern": {"type": "string", "description": "Search pattern (regex supported on Linux)"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."},
                    "file_glob": {"type": "string", "description": "File pattern filter (e.g. '*.py')"},
                    "target": {"type": "string", "enum": ["content", "files"], "description": "Search target", "default": "content"},
                    "limit": {"type": "integer", "description": "Max results", "default": 50},
                },
                "required": ["node_id", "pattern"],
            },
        },
        handler=lambda args, **kw: node_search(
            node_id=args.get("node_id", ""),
            pattern=args.get("pattern", ""),
            path=args.get("path", "."),
            file_glob=args.get("file_glob"),
            target=args.get("target", "content"),
            limit=args.get("limit", 50),
        ),
    )
    registry.register(
        name="node_find_files",
        toolset="node",
        schema={
            "name": "node_find_files",
            "description": "Find files by name pattern on a node (OS-agnostic)",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "pattern": {"type": "string", "description": "File name pattern (e.g. '*.py')"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."},
                    "limit": {"type": "integer", "description": "Max results", "default": 50},
                },
                "required": ["node_id", "pattern"],
            },
        },
        handler=lambda args, **kw: node_find_files(
            node_id=args.get("node_id", ""),
            pattern=args.get("pattern", ""),
            path=args.get("path", "."),
            limit=args.get("limit", 50),
        ),
    )
except ImportError:
    pass
