"""
Background Exec Tool

Execute shell commands with optional background continuation.
Long-running commands can be backgrounded and polled later via the process tool.

Key parameters:
- command (required): shell command to execute
- yield_ms (default 10000): auto-background after this delay (milliseconds)
- background (bool): background immediately
- timeout (seconds, default 1800): kill after this timeout; 0 to disable
- workdir: working directory
- env: environment variables dict

Behavior:
- Foreground runs return output directly.
- When backgrounded, returns status: "running" + session_id + short tail.
- Output is kept in memory until polled or cleared.
- Sessions are lost on process restart (no disk persistence).

Requires: subprocess, threading, time
"""

import json
import logging
import os
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

# In-memory session store
_sessions: Dict[str, Dict[str, Any]] = {}
_sessions_lock = threading.Lock()

DEFAULT_YIELD_MS = 10000
DEFAULT_TIMEOUT_SEC = 1800
MAX_OUTPUT_CHARS = 100000  # Cap output to prevent memory bloat


def _make_session_id() -> str:
    return f"exec-{uuid.uuid4().hex[:8]}"


def _store_session(session_id: str, proc: subprocess.Popen, command: str, workdir: Optional[str]) -> None:
    with _sessions_lock:
        _sessions[session_id] = {
            "proc": proc,
            "command": command,
            "workdir": workdir,
            "started_at": time.time(),
            "stdout_chunks": [],
            "stderr_chunks": [],
            "exit_code": None,
            "finished_at": None,
            "status": "running",
        }


def _update_session(session_id: str, stdout: str, stderr: str, exit_code: Optional[int] = None) -> None:
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            return
        sess["stdout_chunks"].append(stdout)
        sess["stderr_chunks"].append(stderr)
        # Trim if too large
        total = sum(len(c) for c in sess["stdout_chunks"])
        if total > MAX_OUTPUT_CHARS:
            sess["stdout_chunks"] = ["...output truncated..."]
        if exit_code is not None:
            sess["exit_code"] = exit_code
            sess["finished_at"] = time.time()
            sess["status"] = "finished"


def _read_session_output(session_id: str, offset: int = 0, limit: Optional[int] = None) -> Dict[str, Any]:
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            return {"error": f"Session {session_id} not found"}
        stdout = "".join(sess["stdout_chunks"])
        stderr = "".join(sess["stderr_chunks"])
        lines = (stdout + stderr).splitlines()
        if limit is None:
            selected = lines[offset:]
        else:
            selected = lines[offset:offset + limit]
        return {
            "session_id": session_id,
            "status": sess["status"],
            "exit_code": sess["exit_code"],
            "command": sess["command"],
            "output": "\n".join(selected),
            "total_lines": len(lines),
        }


def _kill_session(session_id: str) -> Dict[str, Any]:
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if not sess:
            return {"error": f"Session {session_id} not found"}
        proc = sess.get("proc")
        if proc and proc.poll() is None:
            proc.kill()
            sess["status"] = "killed"
            sess["exit_code"] = -9
            return {"ok": True, "session_id": session_id, "status": "killed"}
        return {"ok": False, "session_id": session_id, "status": sess["status"], "error": "Already finished"}


def _list_sessions() -> List[Dict[str, Any]]:
    with _sessions_lock:
        result = []
        for sid, sess in _sessions.items():
            result.append({
                "session_id": sid,
                "status": sess["status"],
                "command": sess["command"],
                "exit_code": sess["exit_code"],
                "started_at": sess["started_at"],
                "finished_at": sess["finished_at"],
            })
        return result


def _run_background(proc: subprocess.Popen, session_id: str) -> None:
    """Thread worker that drains stdout/stderr and updates session."""
    try:
        stdout_data, stderr_data = proc.communicate(timeout=DEFAULT_TIMEOUT_SEC)
        _update_session(session_id, stdout_data or "", stderr_data or "", proc.returncode)
    except subprocess.TimeoutExpired:
        proc.kill()
        _update_session(session_id, "", "", -9)
    except Exception as e:
        logger.exception("Background exec error for session %s", session_id)
        _update_session(session_id, "", str(e), -1)


def background_exec(
    command: str,
    yield_ms: int = DEFAULT_YIELD_MS,
    background: bool = False,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    workdir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """
    Execute a shell command with optional background continuation.

    Args:
        command: shell command to execute
        yield_ms: auto-background after this delay in milliseconds (default 10000)
        background: background immediately (default False)
        timeout: kill after this timeout in seconds; 0 to disable (default 1800)
        workdir: working directory
        env: environment variables dict

    Returns:
        JSON string. Foreground: {stdout, stderr, exitCode}. Background: {status, session_id, tail}.
    """
    import shlex

    # Build subprocess
    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "shell": True,
    }
    if workdir:
        popen_kwargs["cwd"] = workdir
    if env:
        merged_env = os.environ.copy()
        merged_env.update(env)
        popen_kwargs["env"] = merged_env

    proc = subprocess.Popen(command, **popen_kwargs)

    if background:
        # Immediate background
        session_id = _make_session_id()
        _store_session(session_id, proc, command, workdir)
        thread = threading.Thread(target=_run_background, args=(proc, session_id), daemon=True)
        thread.start()
        # Read a small tail for immediate feedback
        time.sleep(0.5)
        tail = _read_session_output(session_id, limit=10)
        return json.dumps({
            "status": "running",
            "session_id": session_id,
            "command": command,
            "tail": tail.get("output", ""),
        }, ensure_ascii=False)

    # Foreground with yield_ms auto-background
    try:
        stdout_data, stderr_data = proc.communicate(timeout=yield_ms / 1000)
        # Completed within yield_ms
        return json.dumps({
            "stdout": stdout_data or "",
            "stderr": stderr_data or "",
            "exitCode": proc.returncode,
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        # Auto-background after yield_ms
        session_id = _make_session_id()
        _store_session(session_id, proc, command, workdir)
        thread = threading.Thread(target=_run_background, args=(proc, session_id), daemon=True)
        thread.start()
        # Read initial output
        time.sleep(0.5)
        tail = _read_session_output(session_id, limit=10)
        return json.dumps({
            "status": "running",
            "session_id": session_id,
            "command": command,
            "tail": tail.get("output", ""),
            "note": f"Command exceeded {yield_ms}ms and was backgrounded. Use process_poll to check progress.",
        }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Process tool
# ---------------------------------------------------------------------------

def process_list() -> str:
    """List running and finished background exec sessions."""
    sessions = _list_sessions()
    return json.dumps({"ok": True, "sessions": sessions}, ensure_ascii=False, default=str)


def process_poll(session_id: str) -> str:
    """Poll a background session for new output and exit status."""
    result = _read_session_output(session_id)
    return json.dumps(result, ensure_ascii=False, default=str)


def process_log(session_id: str, offset: int = 0, limit: Optional[int] = None) -> str:
    """Read aggregated output for a session with pagination."""
    result = _read_session_output(session_id, offset, limit)
    return json.dumps(result, ensure_ascii=False, default=str)


def process_kill(session_id: str) -> str:
    """Kill a running background session."""
    result = _kill_session(session_id)
    return json.dumps(result, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

registry.register(
    name="background_exec",
    toolset="hermes-cli",
    schema={
        "name": "background_exec",
        "description": (
            "Execute shell commands with optional background continuation. "
            "Long-running commands can be backgrounded and polled later via process_poll. "
            "Use background=True for immediate backgrounding, or yield_ms for auto-background "
            "after a delay."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "yield_ms": {"type": "integer", "description": "Auto-background after this delay in ms (default 10000)", "default": 10000},
                "background": {"type": "boolean", "description": "Background immediately (default False)", "default": False},
                "timeout": {"type": "integer", "description": "Kill after this timeout in seconds; 0 to disable (default 1800)", "default": 1800},
                "workdir": {"type": "string", "description": "Working directory"},
                "env": {"type": "object", "description": "Environment variables dict"},
            },
            "required": ["command"],
        },
    },
    handler=lambda args, **kw: background_exec(
        command=args.get("command", ""),
        yield_ms=args.get("yield_ms", DEFAULT_YIELD_MS),
        background=args.get("background", False),
        timeout=args.get("timeout", DEFAULT_TIMEOUT_SEC),
        workdir=args.get("workdir"),
        env=args.get("env"),
    ),
)

registry.register(
    name="process_list",
    toolset="hermes-cli",
    schema={
        "name": "process_list",
        "description": "List running and finished background exec sessions.",
        "parameters": {"type": "object", "properties": {}},
    },
    handler=lambda args, **kw: process_list(),
)

registry.register(
    name="process_poll",
    toolset="hermes-cli",
    schema={
        "name": "process_poll",
        "description": "Poll a background session for new output and exit status.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID returned by background_exec"},
            },
            "required": ["session_id"],
        },
    },
    handler=lambda args, **kw: process_poll(args.get("session_id", "")),
)

registry.register(
    name="process_log",
    toolset="hermes-cli",
    schema={
        "name": "process_log",
        "description": "Read aggregated output for a session with pagination (offset/limit).",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID"},
                "offset": {"type": "integer", "description": "Line offset (default 0)", "default": 0},
                "limit": {"type": "integer", "description": "Max lines to return"},
            },
            "required": ["session_id"],
        },
    },
    handler=lambda args, **kw: process_log(
        session_id=args.get("session_id", ""),
        offset=args.get("offset", 0),
        limit=args.get("limit"),
    ),
)

registry.register(
    name="process_kill",
    toolset="hermes-cli",
    schema={
        "name": "process_kill",
        "description": "Kill a running background session.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID"},
            },
            "required": ["session_id"],
        },
    },
    handler=lambda args, **kw: process_kill(args.get("session_id", "")),
)
