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
