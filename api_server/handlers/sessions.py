import logging
import time
import uuid
from typing import Any, Dict, Optional
from aiohttp import web

logger = logging.getLogger(__name__)


def _normalize_session_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "user_id": record.get("user_id"),
        "model": record.get("model"),
        "title": record.get("title"),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "end_reason": record.get("end_reason"),
        "message_count": record.get("message_count") or 0,
        "tool_call_count": record.get("tool_call_count") or 0,
        "input_tokens": record.get("input_tokens") or 0,
        "output_tokens": record.get("output_tokens") or 0,
        "last_active": record.get("last_active"),
        "parent_session_id": record.get("parent_session_id"),
    }


async def handle_list_sessions(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        db = ensure_session_db()
        if not db:
            return web.json_response({"items": [], "total": 0})
        limit = max(1, min(500, int(request.query.get("limit", "50"))))
        offset = max(0, int(request.query.get("offset", "0")))
        items = db.list_sessions_rich(limit=limit, offset=offset)
        total = db.session_count()
        return web.json_response({
            "items": [_normalize_session_record(item) for item in items],
            "total": total,
        })
    except Exception as e:
        logger.exception("Error listing sessions")
        return web.json_response({"error": str(e)}, status=500)


async def handle_get_session(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        db = ensure_session_db()
        if not db:
            return web.json_response({"error": "Session DB unavailable"}, status=503)
        session_id = request.match_info.get("session_id", "")
        resolved = db.resolve_session_id(session_id) or session_id
        item = db.get_session(resolved)
        if not item:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"session": _normalize_session_record(item)})
    except Exception as e:
        logger.exception("Error getting session")
        return web.json_response({"error": str(e)}, status=500)


async def handle_create_session(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        body = await request.json() if request.can_read_body else {}
        requested_id = str(body.get("id") or "").strip()
        session_id = requested_id or str(uuid.uuid4())
        title = str(body.get("title") or "").strip() or None
        model = str(body.get("model") or "").strip() or None
        db = ensure_session_db()
        if not db:
            return web.json_response({"error": "Session DB unavailable"}, status=503)
        created_id = db.create_session(session_id=session_id, source="api_server", model=model)
        if title:
            try:
                db.set_session_title(created_id, title)
            except Exception:
                pass
        item = db.get_session(created_id) or {"id": created_id, "model": model, "title": title, "started_at": time.time()}
        return web.json_response({"session": _normalize_session_record(item)})
    except Exception as e:
        logger.exception("Error creating session")
        return web.json_response({"error": str(e)}, status=500)


async def handle_update_session(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        db = ensure_session_db()
        if not db:
            return web.json_response({"error": "Session DB unavailable"}, status=503)
        session_id = request.match_info.get("session_id", "")
        resolved = db.resolve_session_id(session_id) or session_id
        body = await request.json() if request.can_read_body else {}
        title = str(body.get("title") or "").strip()
        if not title:
            return web.json_response({"error": "title required"}, status=400)
        ok = db.set_session_title(resolved, title)
        if not ok:
            return web.json_response({"error": "Session not found"}, status=404)
        item = db.get_session(resolved)
        return web.json_response({"session": _normalize_session_record(item or {"id": resolved, "title": title})})
    except Exception as e:
        logger.exception("Error updating session")
        return web.json_response({"error": str(e)}, status=500)


async def handle_delete_session(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        db = ensure_session_db()
        if not db:
            return web.json_response({"error": "Session DB unavailable"}, status=503)
        session_id = request.match_info.get("session_id", "")
        resolved = db.resolve_session_id(session_id) or session_id
        ok = db.delete_session(resolved)
        if not ok:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"ok": True, "session_id": resolved})
    except Exception as e:
        logger.exception("Error deleting session")
        return web.json_response({"error": str(e)}, status=500)


async def handle_get_session_messages(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        db = ensure_session_db()
        if not db:
            return web.json_response({"items": [], "total": 0})
        session_id = request.match_info.get("session_id", "")
        resolved = db.resolve_session_id(session_id) or session_id
        items = db.get_messages(resolved)
        return web.json_response({"items": items, "total": len(items)})
    except Exception as e:
        logger.exception("Error getting session messages")
        return web.json_response({"error": str(e)}, status=500)


async def handle_fork_session(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    session_id = request.match_info.get("session_id", "")
    db = ensure_session_db()
    original = db.get_session(session_id)
    if original is None:
        return web.json_response({"error": "Session not found"}, status=404)

    forked_id = f"sess_{uuid.uuid4().hex}"
    try:
        db.create_session(
            session_id=forked_id,
            source=original.get("source") or "api_server",
            model=original.get("model"),
            system_prompt=original.get("system_prompt"),
            user_id=original.get("user_id"),
            parent_session_id=session_id,
        )
        messages = db.get_messages(session_id)
        for message in messages:
            db.append_message(
                session_id=forked_id,
                role=message.get("role"),
                content=message.get("content"),
                tool_name=message.get("tool_name"),
                tool_calls=message.get("tool_calls"),
                tool_call_id=message.get("tool_call_id"),
                token_count=message.get("token_count"),
                finish_reason=message.get("finish_reason"),
                reasoning=message.get("reasoning"),
            )
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

    session = _normalize_session_record(db.get_session(forked_id))
    return web.json_response({"session": session, "forked_from": session_id})
