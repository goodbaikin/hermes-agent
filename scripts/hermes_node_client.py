#!/usr/bin/env python3
"""
Hermes Remote Node Client for Windows
=====================================
OpenClaw-style WebSocket node that runs on a Windows development machine,
exposing local tool execution (PowerShell, MSBuild, signtool, file I/O)
to a remote Hermes Gateway without SSH escaping/encoding issues.

Features:
    - Auto-reconnect with exponential backoff
    - Persistent background execution (Windows Service compatible)
    - BOM/CRLF aware file I/O for Japanese text
    - Heartbeat/ping support

Usage (foreground):
    python hermes_node_client.py --gateway ws://hermes-gateway:8642/ws \
        --token your-secret-token --node-id dev-win01

Usage (install as Windows Service via NSSM):
    nssm install HermesNode "C:\Path\To\python.exe" \
        "C:\Path\To\hermes_node_client.py --gateway ws://... --token ... --node-id ..."

Dependencies:
    pip install websockets

Security:
    - Use wss:// (TLS) in production
    - Keep NODE_TOKEN in environment variables, never hardcode
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import websockets

# ---------------------------------------------------------------------------
# Logging setup (Windows Event Log friendly)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (override via env vars or CLI)
# ---------------------------------------------------------------------------
NODE_ID = os.environ.get("HERMES_NODE_ID", "dev-win01")
GATEWAY_URL = os.environ.get("HERMES_GATEWAY_URL", "ws://localhost:8642/ws")
NODE_TOKEN = os.environ.get("HERMES_NODE_TOKEN", "dev-token-change-me")

RECONNECT_MIN_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0
RECONNECT_BACKOFF_MULTIPLIER = 2.0

COMMANDS = {
    "terminal.exec",
    "file.read",
    "file.write",
    "file.delete",
    "file.list",
    "msbuild",
    "signtool",
}

DEFAULT_TIMEOUT_SEC = 30
MAX_TIMEOUT_SEC = 600


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def handle_terminal_exec(params: dict[str, Any]) -> dict[str, Any]:
    """Execute PowerShell locally. No SSH escaping hell."""
    cmd = params["cmd"]
    cwd = params.get("cwd")
    timeout_sec = min(
        params.get("timeoutMs", DEFAULT_TIMEOUT_SEC * 1000) / 1000, MAX_TIMEOUT_SEC
    )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # On Windows: use PowerShell with UTF-8 encoding, bypassing cmd.exe AutoRun
    if sys.platform == "win32":
        # Build PowerShell command as argument list to avoid shell escaping issues
        # Use -EncodedCommand for complex commands to avoid quoting hell
        import base64
        ps_script = (
            '[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; '
            '$OutputEncoding = [System.Text.Encoding]::UTF8; '
            f'{cmd}'
        )
        encoded = base64.b64encode(ps_script.encode('utf-16le')).decode()
        args = [
            "powershell", "-NoProfile", "-NonInteractive",
            "-ExecutionPolicy", "Bypass",
            "-EncodedCommand", encoded,
        ]
    else:
        args = ["bash", "-c", cmd]

    proc = await asyncio.create_subprocess_exec(
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout_sec}s",
            "exitCode": -1,
        }

    return {
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace").replace("#< CLIXML", "").strip(),
        "exitCode": proc.returncode,
    }


async def handle_file_read(params: dict[str, Any]) -> dict[str, Any]:
    """Read a file locally. Supports binary (base64) or text (UTF-8)."""
    path = Path(params["path"])
    encoding = params.get("encoding")

    if encoding:
        with open(path, "r", encoding=encoding) as f:
            return {"content": f.read(), "encoding": encoding}
    else:
        with open(path, "rb") as f:
            return {"content": base64.b64encode(f.read()).decode(), "binary": True}


async def handle_file_write(params: dict[str, Any]) -> dict[str, Any]:
    """Write a file locally. Handles BOM+CRLF for Japanese source files."""
    path = Path(params["path"])
    content_b64 = params.get("content")
    encoding = params.get("encoding", "utf-8-sig")  # BOM-aware UTF-8
    newline = params.get("newline", "\r\n")  # Windows default

    if content_b64:
        raw = base64.b64decode(content_b64)
        with open(path, "wb") as f:
            f.write(raw)
    else:
        text = params.get("text", "")
        with open(path, "w", encoding=encoding, newline=newline) as f:
            f.write(text)

    return {"path": str(path), "bytesWritten": path.stat().st_size}


async def handle_file_delete(params: dict[str, Any]) -> dict[str, Any]:
    path = Path(params["path"])
    path.unlink(missing_ok=True)
    return {"deleted": True, "path": str(path)}


async def handle_file_list(params: dict[str, Any]) -> dict[str, Any]:
    path = Path(params["path"])
    entries = []
    for entry in path.iterdir():
        stat = entry.stat()
        entries.append({
            "name": entry.name,
            "isFile": entry.is_file(),
            "isDir": entry.is_dir(),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        })
    return {"entries": entries, "path": str(path)}


async def handle_msbuild(params: dict[str, Any]) -> dict[str, Any]:
    """MSBuild wrapper with common defaults."""
    project = params["project"]
    configuration = params.get("configuration", "Release")
    platform = params.get("platform", "x64")
    targets = params.get("targets", "Build")
    cwd = params.get("cwd")

    msbuild_path = params.get("msbuildPath", "MSBuild.exe")
    cmd = (
        f'"{msbuild_path}" "{project}" '
        f"/p:Configuration={configuration} "
        f"/p:Platform={platform} "
        f"/t:{targets} "
        f"/m"
    )

    return await handle_terminal_exec({"cmd": cmd, "cwd": cwd, "timeoutMs": 300000})


async def handle_signtool(params: dict[str, Any]) -> dict[str, Any]:
    """SignTool wrapper. Uses Set-AuthenticodeSignature fallback if signtool path not given."""
    file_path = params["file"]
    thumbprint = params.get("thumbprint")
    hash_alg = params.get("hashAlgorithm", "SHA256")
    subject = params.get("subject")
    signtool_path = params.get("signtoolPath")

    if signtool_path and thumbprint:
        cmd = (
            f'"{signtool_path}" sign '
            f"/sha1 {thumbprint} "
            f"/fd {hash_alg} "
            f'"{file_path}"'
        )
    elif subject:
        cmd = (
            f'powershell.exe -Command "'
            f"$cert = Get-ChildItem Cert:\\LocalMachine\\My | "
            f"Where-Object {{ $_.Subject -eq 'CN={subject}' }} | Select-Object -First 1; "
            f"Set-AuthenticodeSignature -FilePath '{file_path}' -Certificate $cert -HashAlgorithm {hash_alg}"
            f'"'
        )
    else:
        raise ValueError("thumbprint+signtoolPath or subject required")

    return await handle_terminal_exec({"cmd": cmd, "timeoutMs": 60000})


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

HANDLERS = {
    "terminal.exec": handle_terminal_exec,
    "file.read": handle_file_read,
    "file.write": handle_file_write,
    "file.delete": handle_file_delete,
    "file.list": handle_file_list,
    "msbuild": handle_msbuild,
    "signtool": handle_signtool,
}


async def handle_invoke(ws: websockets.WebSocketClientProtocol, payload: dict[str, Any]) -> None:
    request_id = payload["id"]
    command = payload["command"]
    params = json.loads(payload.get("paramsJSON") or "{}")

    logger.info("[%s] Invoking %s", NODE_ID, command)
    try:
        handler = HANDLERS.get(command)
        if not handler:
            raise ValueError(f"Unknown command: {command}")

        result = await handler(params)
        await send_result(ws, request_id, True, result)
    except Exception as e:
        logger.exception("[%s] Command %s failed", NODE_ID, command)
        await send_result(ws, request_id, False, None, str(e))


async def send_result(
    ws: websockets.WebSocketClientProtocol,
    request_id: str,
    ok: bool,
    payload: Any = None,
    error: str | None = None,
) -> None:
    await ws.send(json.dumps({
        "type": "event",
        "event": "node.invoke.result",
        "payload": {
            "id": request_id,
            "nodeId": NODE_ID,
            "ok": ok,
            "payload": payload,
            "error": {"message": error} if error else None,
        }
    }))


# ---------------------------------------------------------------------------
# Connection logic with auto-reconnect
# ---------------------------------------------------------------------------

async def connect_and_serve(gateway_url: str, token: str, node_id: str) -> None:
    """Connect to gateway with exponential backoff reconnect."""
    delay = RECONNECT_MIN_DELAY

    while True:
        try:
            logger.info("[%s] Connecting to %s ...", node_id, gateway_url)
            async with websockets.connect(gateway_url) as ws:
                delay = RECONNECT_MIN_DELAY  # Reset on successful connect

                # --- Handshake ---
                connect_req = {
                    "type": "req",
                    "id": str(uuid.uuid4()),
                    "method": "connect",
                    "params": {
                        "minProtocol": 1,
                        "maxProtocol": 1,
                        "client": {
                            "id": node_id,
                            "version": "1.0.0",
                            "platform": "windows",
                            "mode": "node",
                        },
                        "role": "node",
                        "scopes": [],
                        "caps": ["terminal", "file", "build"],
                        "commands": sorted(COMMANDS),
                        "auth": {"token": token},
                    }
                }
                await ws.send(json.dumps(connect_req))

                response = await ws.recv()
                data = json.loads(response)
                if data.get("type") == "res" and data.get("ok"):
                    logger.info("[%s] Connected. Waiting for commands...", node_id)
                else:
                    logger.error("[%s] Handshake failed: %s", node_id, data)
                    await asyncio.sleep(delay)
                    delay = min(delay * RECONNECT_BACKOFF_MULTIPLIER, RECONNECT_MAX_DELAY)
                    continue

                # --- Main loop ---
                while True:
                    try:
                        msg = await ws.recv()
                        data = json.loads(msg)
                    except websockets.ConnectionClosed:
                        logger.warning("[%s] Connection closed by gateway.", node_id)
                        break

                    if data.get("type") == "event" and data.get("event") == "node.invoke.request":
                        asyncio.create_task(handle_invoke(ws, data["payload"]))
                    elif data.get("type") == "event" and data.get("event") == "ping":
                        # Keepalive ping
                        pass

        except (websockets.ConnectionClosed, websockets.ConnectionClosedOK, websockets.ConnectionClosedError) as exc:
            logger.warning("[%s] Connection closed: %s", node_id, exc)
        except OSError as exc:
            logger.error("[%s] Connection error: %s", node_id, exc)
        except Exception as exc:
            logger.exception("[%s] Unexpected error in connection loop", node_id)

        logger.info("[%s] Reconnecting in %.1f seconds...", node_id, delay)
        await asyncio.sleep(delay)
        delay = min(delay * RECONNECT_BACKOFF_MULTIPLIER, RECONNECT_MAX_DELAY)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Hermes Remote Node Client")
    parser.add_argument("--gateway", default=GATEWAY_URL, help="Gateway WebSocket URL")
    parser.add_argument("--token", default=NODE_TOKEN, help="Node authentication token")
    parser.add_argument("--node-id", default=NODE_ID, help="Unique node identifier")
    args = parser.parse_args()

    try:
        asyncio.run(connect_and_serve(args.gateway, args.token, args.node_id))
    except KeyboardInterrupt:
        logger.info("[%s] Shutting down.", args.node_id)


if __name__ == "__main__":
    main()
