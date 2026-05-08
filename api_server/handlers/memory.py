import json
from aiohttp import web


async def handle_get_memory(request: web.Request, *, check_auth, get_memory_store) -> web.Response:
    """GET /api/memory -- read current memory state."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    target = (request.query.get("target") or "all").strip().lower()
    if target not in {"all", "memory", "user"}:
        return web.json_response({"error": "target must be one of: all, memory, user"}, status=400)

    store = get_memory_store()
    store.load_from_disk()
    targets = []
    if target in {"all", "memory"}:
        targets.append({
            "target": "memory",
            "entries": store.memory_entries,
            "entry_count": len(store.memory_entries),
        })
    if target in {"all", "user"}:
        targets.append({
            "target": "user",
            "entries": store.user_entries,
            "entry_count": len(store.user_entries),
        })
    return web.json_response({"targets": targets})


async def handle_add_memory(request: web.Request, *, check_auth, get_memory_store) -> web.Response:
    """POST /api/memory -- add a memory entry."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON in request body"}, status=400)

    target = str(body.get("target") or "").strip().lower()
    content = str(body.get("content") or "")
    if target not in {"memory", "user"}:
        return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
    result = get_memory_store().add(target, content)
    status = 200 if result.get("success") else 400
    return web.json_response(result, status=status)


async def handle_replace_memory(request: web.Request, *, check_auth, get_memory_store) -> web.Response:
    """PATCH /api/memory -- replace a memory entry."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON in request body"}, status=400)

    target = str(body.get("target") or "").strip().lower()
    old_text = str(body.get("old_text") or "")
    content = str(body.get("content") or "")
    if target not in {"memory", "user"}:
        return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
    result = get_memory_store().replace(target, old_text, content)
    status = 200 if result.get("success") else 400
    return web.json_response(result, status=status)


async def handle_delete_memory(request: web.Request, *, check_auth, get_memory_store) -> web.Response:
    """DELETE /api/memory -- delete a memory entry."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON in request body"}, status=400)

    target = str(body.get("target") or "").strip().lower()
    old_text = str(body.get("old_text") or "")
    if target not in {"memory", "user"}:
        return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
    result = get_memory_store().remove(target, old_text)
    status = 200 if result.get("success") else 400
    return web.json_response(result, status=status)
