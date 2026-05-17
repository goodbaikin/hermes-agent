"""Profile handlers for the API Server."""

import logging
from typing import Any, Dict, List

from aiohttp import web

logger = logging.getLogger(__name__)


def _profile_to_dict(p: Any) -> Dict[str, Any]:
    """Serialize a ProfileInfo object to a JSON-friendly dict."""
    return {
        "id": p.name,
        "name": p.name,
        "model": p.model or "",
        "provider": p.provider or "",
        "gateway_running": p.gateway_running,
        "skill_count": p.skill_count,
    }


async def handle_list_profiles(
    request: web.Request,
    *,
    check_auth,
) -> web.Response:
    """GET /api/profiles -- list all profiles."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    try:
        from hermes_cli.profiles import list_profiles

        profiles = list_profiles()
        return web.json_response({
            "profiles": [_profile_to_dict(p) for p in profiles],
        })
    except Exception as e:
        logger.exception("Error listing profiles")
        return web.json_response({"error": str(e)}, status=500)


async def handle_get_profile_config(
    request: web.Request,
    *,
    check_auth,
) -> web.Response:
    """GET /api/profiles/{name} -- get profile config (workspace, node_id)."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    name = request.match_info.get("name", "")
    if not name:
        return web.json_response({"error": "Profile name required"}, status=400)

    try:
        from hermes_cli.profiles import get_profile_dir
        import yaml

        profile_dir = get_profile_dir(name)
        config_path = profile_dir / "config.yaml"
        if not config_path.exists():
            return web.json_response({"error": "Profile not found"}, status=404)

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        active_workspace = config.get("active_workspace")
        workspaces = config.get("workspaces", {})
        ws_config = workspaces.get(active_workspace, {}) if active_workspace else {}

        return web.json_response({
            "name": name,
            "active_workspace": active_workspace,
            "workspace_mode": config.get("workspace_mode"),
            "node_id": ws_config.get("node_id", "local"),
            "path_prefixes": ws_config.get("path_prefixes", []),
            "tools": ws_config.get("tools", []),
        })
    except Exception as e:
        logger.exception("Error getting profile config")
        return web.json_response({"error": str(e)}, status=500)
