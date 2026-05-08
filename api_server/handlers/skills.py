import json
from aiohttp import web

from tools.skills_tool import skills_list, skill_view


async def handle_list_skills(request: web.Request, *, check_auth) -> web.Response:
    """GET /api/skills -- list skills."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    category = (request.query.get("category") or "").strip() or None
    return web.json_response(json.loads(skills_list(category=category)))


async def handle_view_skill(request: web.Request, *, check_auth) -> web.Response:
    """GET /api/skills/{name} -- fetch skill details."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    name = request.match_info["name"]
    file_path = (request.query.get("file_path") or "").strip() or None
    return web.json_response(json.loads(skill_view(name, file_path=file_path)))
