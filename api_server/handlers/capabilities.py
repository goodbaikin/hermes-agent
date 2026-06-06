from aiohttp import web

from hermes_cli.tools_config import _get_platform_tools
from hermes_cli.config import load_config


async def handle_capabilities(request: web.Request, *, check_auth, api_key: str = "") -> web.Response:
    """GET /v1/capabilities -- list available toolsets and reasoning modes."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    config = load_config()
    toolsets = sorted(_get_platform_tools(config, "api_server"))

    reasoning_modes = ["disabled", "enabled", "auto"]

    return web.json_response({
        "object": "hermes.api_server.capabilities",
        "platform": "hermes-agent",
        "model": "hermes-agent",
        "auth": {"type": "bearer", "required": bool(api_key)},
        "runtime": {
            "mode": "server_agent",
            "version": "0.13.0",
            "tool_execution": "server",
            "split_runtime": False,
            "description": "API-server host for Hermes Agent",
        },
        "features": {
            "chat_completions": True,
            "run_status": True,
            "run_events_sse": True,
            "session_continuity_header": "X-Hermes-Session-Id",
            "session_key_header": "X-Hermes-Session-Key",
        },
        "endpoints": {
            "run_status": {"method": "GET", "path": "/v1/runs/{run_id}"},
            "run_events": {"method": "GET", "path": "/v1/runs/{run_id}/events"},
            "run_stop": {"method": "POST", "path": "/v1/runs/{run_id}/stop"},
            "skills": {"method": "GET", "path": "/v1/skills"},
            "toolsets": {"method": "GET", "path": "/v1/toolsets"},
        },
        "toolsets": toolsets,
        "reasoning_modes": reasoning_modes,
        "supports_streaming": True,
        "supports_multimodal": True,
        "supports_responses_api": True,
        "supports_runs_api": True,
    })
