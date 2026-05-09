import json
import logging
import time
import uuid
from typing import Any, Dict, Optional
from aiohttp import web

logger = logging.getLogger(__name__)


def _normalize_session_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    normalized = dict(record)
    model_config = normalized.get("model_config")
    if model_config:
        try:
            import json
            normalized["model_config"] = json.loads(model_config)
        except (TypeError, json.JSONDecodeError):
            pass
    return {
        "id": normalized.get("id") or normalized.get("session_id"),
        "session_id": normalized.get("session_id") or normalized.get("id"),
        "source": normalized.get("source"),
        "user_id": normalized.get("user_id"),
        "model": normalized.get("model"),
        "title": normalized.get("title"),
        "started_at": normalized.get("started_at"),
        "ended_at": normalized.get("ended_at"),
        "end_reason": normalized.get("end_reason"),
        "message_count": normalized.get("message_count") or 0,
        "tool_call_count": normalized.get("tool_call_count") or 0,
        "input_tokens": normalized.get("input_tokens") or 0,
        "output_tokens": normalized.get("output_tokens") or 0,
        "last_active": normalized.get("last_active"),
        "parent_session_id": normalized.get("parent_session_id"),
        "model_config": normalized.get("model_config"),
    }


async def handle_list_sessions(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        db = ensure_session_db()
        if not db:
            return web.json_response({"items": [], "total": 0})
        from api_server.utils import _parse_int
        limit = _parse_int(request.query.get("limit"), 50)
        offset = _parse_int(request.query.get("offset"), 0)
        source = (request.query.get("source") or "").strip() or None
        items = db.list_sessions_rich(source=source, limit=limit, offset=offset)
        total = db.session_count(source=source)
        return web.json_response({
            "items": [_normalize_session_record(item) for item in items],
            "total": total,
        })
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
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
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON in request body"}, status=400)
    try:
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
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
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
        deleted = db.delete_session(session_id)
        if not deleted:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"ok": True, "session_id": session_id})
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
        if db.get_session(resolved) is None:
            db.ensure_session(resolved, source="web")
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
