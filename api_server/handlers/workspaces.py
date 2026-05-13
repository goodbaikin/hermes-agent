"""Workspace handlers for the API Server."""

import logging
from typing import Any, Dict, List

from aiohttp import web

logger = logging.getLogger(__name__)


def _workspace_to_dict(ws: Any) -> Dict[str, Any]:
    """Serialize a Workspace object to a JSON-friendly dict."""
    return {
        "id": ws.name,
        "name": ws.name,
        "node_id": ws.node_id,
        "description": ws.description or "",
        "path_prefixes": ws.path_prefixes or [],
        "tools": list(ws.tools) if ws.tools else ["all"],
    }


async def handle_list_workspaces(
    request: web.Request,
    *,
    check_auth,
) -> web.Response:
    """GET /api/workspaces -- list configured workspaces."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    try:
        from agent.workspace_manager import get_workspace_manager

        manager = get_workspace_manager()
        workspaces: List[Dict[str, Any]] = []
        for name in manager.list_workspaces():
            ws = manager.get_workspace(name)
            if ws:
                workspaces.append(_workspace_to_dict(ws))

        return web.json_response({
            "workspaces": workspaces,
            "active": manager.active_name,
        })
    except Exception as e:
        logger.exception("Error listing workspaces")
        return web.json_response({"error": str(e)}, status=500)
