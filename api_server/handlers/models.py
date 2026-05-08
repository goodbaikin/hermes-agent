import time
from aiohttp import web


async def handle_models(request: web.Request, *, model_name: str, check_auth) -> web.Response:
    """GET /v1/models -- return hermes-agent as an available model."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    return web.json_response({
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "hermes",
                "permission": [],
                "root": model_name,
                "parent": None,
            }
        ],
    })
