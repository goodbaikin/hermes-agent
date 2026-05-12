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


def node_read(node_id: str, path: str) -> str:
    """Read a file from a node (text)."""
    # Expand ~ to home directory
    if path.startswith("~"):
        import os
        path = os.path.expanduser(path)
    
    result_str = node_invoke(node_id, "file.read", {"path": path})
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


def node_exec(node_id: str, cmd: str, timeout: int = 30) -> Dict[str, Any]:
    """Execute a command on a node."""
    result_str = node_invoke(node_id, "terminal.exec", {
        "cmd": cmd,
        "timeout": timeout,
    })
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
    """Search files on a node (OS-agnostic wrapper for ripgrep/findstr/grep)."""
    
    # Resolve relative path to absolute
    if not path.startswith("/") and not path.startswith("~"):
        # For local node, resolve relative to current working directory
        if node_id == "local":
            import os
            path = os.path.abspath(path)
        # For remote nodes, assume path is relative to node's home or workspace
        # (remote nodes should handle path resolution)
    
    # OS detection
    os_check = node_exec(node_id, "uname -s", timeout=5)
    output = os_check.get("payload", {}).get("output", "")
    is_windows = "Linux" not in output and "Darwin" not in output
    
    if is_windows:
        # Windows: use PowerShell Select-String or findstr
        if target == "files":
            # Find files by name
            glob_filter = f"-Filter '{file_glob}'" if file_glob else ""
            cmd = f'powershell -Command "Get-ChildItem -Path \'{path}\' {glob_filter} -Recurse -Name"'
        else:
            # Search content
            include_opt = f"-Include '{file_glob}'" if file_glob else ""
            cmd = f'powershell -Command "Get-ChildItem -Path \'{path}\' {include_opt} -Recurse | Select-String -Pattern \'{pattern}\' | Select-Object -First {limit} | ForEach-Object {{ \"$($_.Filename):$($_.LineNumber):$($_.Line)\" }}"'
    else:
        # Linux/macOS: use ripgrep (rg) with fallback to grep
        rg_check = node_exec(node_id, "which rg", timeout=5)
        has_rg = rg_check.get("payload", {}).get("exit_code", 1) == 0
        
        if has_rg:
            glob_opt = f"-g '{file_glob}'" if file_glob else ""
            if target == "files":
                cmd = f"rg {glob_opt} -l '{pattern}' {path} | head -{limit}"
            else:
                cmd = f"rg {glob_opt} -n -H '{pattern}' {path} | head -{limit}"
        else:
            # Fallback to grep
            name_opt = f"--include='{file_glob}'" if file_glob else ""
            if target == "files":
                cmd = f"grep -rl {name_opt} '{pattern}' {path} | head -{limit}"
            else:
                cmd = f"grep -rnH {name_opt} '{pattern}' {path} | head -{limit}"
    
    result = node_exec(node_id, cmd, timeout=30)
    payload = result.get("payload", {})
    stdout = payload.get("output", "")
    
    # Parse results
    entries = []
    for line in stdout.strip().split("\n"):
        if not line:
            continue
        
        # Format: filename:line_number:content (rg/grep) or filename:line (Select-String)
        parts = line.split(":", 2)
        if len(parts) >= 2:
            entries.append({
                "file": parts[0],
                "line": int(parts[1]) if parts[1].isdigit() else 0,
                "content": parts[2] if len(parts) > 2 else "",
            })
        else:
            entries.append({"file": line, "line": 0, "content": ""})
    
    return entries[:limit]


def node_find_files(node_id: str, pattern: str, path: str = ".", limit: int = 50) -> List[str]:
    """Find files by name pattern on a node (OS-agnostic)."""
    
    # Resolve relative path to absolute
    if not path.startswith("/") and not path.startswith("~"):
        if node_id == "local":
            import os
            path = os.path.abspath(path)
    
    os_check = node_exec(node_id, "uname -s", timeout=5)
    output = os_check.get("payload", {}).get("output", "")
    is_windows = "Linux" not in output and "Darwin" not in output
    
    if is_windows:
        cmd = f'powershell -Command "Get-ChildItem -Path \'{path}\' -Filter \'{pattern}\' -Recurse -Name | Select-Object -First {limit}"'
    else:
        cmd = f"find {path} -name '{pattern}' -type f 2>/dev/null | head -{limit}"
    
    result = node_exec(node_id, cmd, timeout=30)
    stdout = result.get("payload", {}).get("output", "")
    
    return [line.strip() for line in stdout.strip().split("\n") if line.strip()]


# Tool registration
if __name__ != "__main__":
    try:
        from tools.registry import register
        
        register(
            name="node_read",
            fn=node_read,
            description="Read a file from a local or remote node",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["node_id", "path"],
            },
        )
        register(
            name="node_write",
            fn=node_write,
            description="Write a file to a local or remote node",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Text content to write"},
                },
                "required": ["node_id", "path", "content"],
            },
        )
        register(
            name="node_patch",
            fn=node_patch,
            description="Apply content-based patches to a file on a node",
            parameters={
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
        )
        register(
            name="node_exec",
            fn=node_exec,
            description="Execute a command on a local or remote node",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "cmd": {"type": "string", "description": "Command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "required": ["node_id", "cmd"],
            },
        )
        register(
            name="node_list_dir",
            fn=node_list_dir,
            description="List directory contents on a local or remote node",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "path": {"type": "string", "description": "Directory path to list"},
                },
                "required": ["node_id", "path"],
            },
        )
        register(
            name="node_search",
            fn=node_search,
            description="Search file contents on a node (OS-agnostic: uses ripgrep on Linux, PowerShell on Windows)",
            parameters={
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
        )
        register(
            name="node_find_files",
            fn=node_find_files,
            description="Find files by name pattern on a node (OS-agnostic)",
            parameters={
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "Node ID ('local' for this machine)"},
                    "pattern": {"type": "string", "description": "File name pattern (e.g. '*.py')"},
                    "path": {"type": "string", "description": "Directory to search in", "default": "."},
                    "limit": {"type": "integer", "description": "Max results", "default": 50},
                },
                "required": ["node_id", "pattern"],
            },
        )
    except ImportError:
        pass
