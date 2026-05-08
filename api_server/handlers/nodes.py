import json
from aiohttp import web

from api_server.node_registry import NODE_REGISTRY


async def handle_list_nodes(request: web.Request, *, check_auth) -> web.Response:
    """GET /v1/nodes -- list registered nodes."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    nodes = NODE_REGISTRY.list_nodes()
    return web.json_response({"ok": True, "nodes": nodes})


async def handle_node_invoke(request: web.Request, *, check_auth) -> web.Response:
    """POST /v1/nodes/{node_id}/invoke -- invoke a command on a remote node."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    node_id = request.match_info.get("node_id", "")
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"ok": False, "error": {"message": "Invalid JSON body"}}, status=400)

    command = body.get("command", "")
    params = body.get("params", {})
    timeout_ms = body.get("timeoutMs", 30000)
    idempotency_key = body.get("idempotencyKey")

    result = await NODE_REGISTRY.invoke(
        node_id=node_id,
        command=command,
        params=params,
        timeout_ms=timeout_ms,
        idempotency_key=idempotency_key,
    )
    return web.json_response(result)
