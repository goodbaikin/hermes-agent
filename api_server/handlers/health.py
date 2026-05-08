import os
from aiohttp import web

from gateway.status import read_runtime_status


async def handle_health(request: web.Request) -> web.Response:
    """GET /health -- simple health check."""
    return web.json_response({"status": "ok", "platform": "hermes-agent"})


async def handle_health_detailed(request: web.Request) -> web.Response:
    """GET /health/detailed -- rich status for cross-container dashboard probing.

    Returns gateway state, connected platforms, PID, and uptime so the
    dashboard can display full status without needing a shared PID file or
    /proc access.  No authentication required.
    """
    runtime = read_runtime_status() or {}
    return web.json_response({
        "status": "ok",
        "platform": "hermes-agent",
        "gateway_state": runtime.get("gateway_state"),
        "platforms": runtime.get("platforms", {}),
        "active_agents": runtime.get("active_agents", 0),
        "exit_reason": runtime.get("exit_reason"),
        "updated_at": runtime.get("updated_at"),
        "pid": os.getpid(),
    })
