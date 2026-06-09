import json
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple
from aiohttp import web

from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)


def _parse_model_config(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_session_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    normalized = dict(record)
    model_config = _parse_model_config(normalized.get("model_config"))
    if model_config:
        normalized["model_config"] = model_config
    return {
        "id": normalized.get("id") or normalized.get("session_id"),
        "session_id": normalized.get("session_id") or normalized.get("id"),
        "source": normalized.get("source"),
        "user_id": normalized.get("user_id"),
        "model": normalized.get("model"),
        "title": normalized.get("title"),
        "profile": normalized.get("profile"),
        "started_at": normalized.get("started_at"),
        "ended_at": normalized.get("ended_at"),
        "end_reason": normalized.get("end_reason"),
        "message_count": normalized.get("message_count") or 0,
        "tool_call_count": normalized.get("tool_call_count") or 0,
        "input_tokens": normalized.get("input_tokens") or 0,
        "output_tokens": normalized.get("output_tokens") or 0,
        "current_prompt_tokens": normalized.get("current_prompt_tokens") or 0,
        "last_active": normalized.get("last_active"),
        "parent_session_id": normalized.get("parent_session_id"),
        "model_config": normalized.get("model_config"),
    }


def _clean_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    allowed = ("model", "provider", "base_url", "api_mode", "provider_label", "resolved_via_alias")
    return {
        key: str(config.get(key) or "").strip()
        for key in allowed
        if str(config.get(key) or "").strip()
    }


def _query_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _lineage_ids(db: Any, session_id: str) -> list[str]:
    if not session_id:
        return []
    if hasattr(db, "_session_lineage_root_to_tip"):
        try:
            ids = db._session_lineage_root_to_tip(session_id)
            if ids:
                return list(ids)
        except Exception:
            logger.debug("Failed to resolve session lineage for %s", session_id, exc_info=True)
    return [session_id]


def _messages_for_lineage(db: Any, session_id: str, order: str = "asc") -> tuple[list[Dict[str, Any]], list[str]]:
    ids = _lineage_ids(db, session_id)
    items: list[Dict[str, Any]] = []
    for sid in ids:
        try:
            items.extend(db.get_messages(sid, order="asc"))
        except Exception:
            logger.debug("Failed to load lineage messages for %s", sid, exc_info=True)
    items.sort(key=lambda msg: (msg.get("timestamp") or 0, msg.get("id") or 0))
    if order == "desc":
        items.reverse()
    return items, ids


def _resolve_session_model_config(body: Dict[str, Any], current_session: Optional[Dict[str, Any]] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """Resolve a PATCH/POST body into a safe session model override.

    Returns (model, model_config, error). api_key is intentionally never stored.
    """
    raw_cfg = _parse_model_config(body.get("model_config"))
    if body.get("model_config") is None and "model_config" in body:
        return None, None, None

    model = str(body.get("model") or raw_cfg.get("model") or "").strip()
    provider = str(body.get("provider") or raw_cfg.get("provider") or "").strip()
    base_url = str(body.get("base_url") or raw_cfg.get("base_url") or "").strip()
    api_mode = str(body.get("api_mode") or raw_cfg.get("api_mode") or "").strip()

    if not (model or provider or base_url or api_mode):
        return None, None, "model or provider required"

    try:
        from api_server.handlers.config import _current_model_settings
        from hermes_cli.config import load_config
        from hermes_cli.model_switch import switch_model

        config = load_config()
        current = _current_model_settings(config)
        current_model = str(current.get("model") or "").strip()
        current_provider = str(current.get("provider") or "").strip()
        current_base_url = str(current.get("base_url") or "").strip()
        session_cfg = _parse_model_config((current_session or {}).get("model_config"))
        if session_cfg.get("model"):
            current_model = str(session_cfg.get("model") or "").strip()
        if session_cfg.get("provider"):
            current_provider = str(session_cfg.get("provider") or "").strip()
        if session_cfg.get("base_url"):
            current_base_url = str(session_cfg.get("base_url") or "").strip()

        result = switch_model(
            model or current_model,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            is_global=False,
            explicit_provider=provider,
            user_providers=config.get("providers") or {},
            custom_providers=config.get("custom_providers") or [],
        )
        if result.success:
            resolved = _clean_model_config({
                "model": result.new_model,
                "provider": result.target_provider,
                "base_url": result.base_url,
                "api_mode": result.api_mode,
                "provider_label": result.provider_label,
                "resolved_via_alias": result.resolved_via_alias,
            })
            if base_url:
                resolved["base_url"] = base_url
            if api_mode:
                resolved["api_mode"] = api_mode
            return resolved.get("model"), resolved, None
    except Exception as exc:
        logger.debug("Session model switch resolution failed; falling back to runtime provider: %s", exc, exc_info=True)

    # Deliberate fallback: allow provider-backed models that are not yet in the
    # catalog (subscription endpoints often expose new names before models.dev).
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from hermes_cli.providers import determine_api_mode, get_label
        from hermes_cli.model_normalize import normalize_model_for_provider

        requested_provider = provider or raw_cfg.get("provider") or None
        runtime = resolve_runtime_provider(
            requested=requested_provider,
            explicit_base_url=base_url or None,
            target_model=model or None,
        )
        resolved_provider = str(runtime.get("provider") or requested_provider or "").strip()
        resolved_model = normalize_model_for_provider(model, resolved_provider) if model else model
        resolved_base_url = base_url or str(runtime.get("base_url") or "").strip()
        resolved_api_mode = api_mode or str(runtime.get("api_mode") or "").strip() or determine_api_mode(resolved_provider, resolved_base_url)
        resolved = _clean_model_config({
            "model": resolved_model,
            "provider": resolved_provider,
            "base_url": resolved_base_url,
            "api_mode": resolved_api_mode,
            "provider_label": get_label(resolved_provider),
        })
        if not resolved.get("model"):
            return None, None, "model required"
        return resolved.get("model"), resolved, None
    except Exception as exc:
        return None, None, f"Could not resolve session model: {exc}"


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
        normalized = _normalize_session_record(item)
        if normalized and not normalized.get("current_prompt_tokens"):
            try:
                messages = db.get_messages(resolved, order="asc")
                conversation = [
                    {"role": msg.get("role"), "content": msg.get("content") or ""}
                    for msg in messages
                    if msg.get("role") in {"system", "user", "assistant", "tool"}
                ]
                estimated = estimate_messages_tokens_rough(conversation)
                if estimated > 0:
                    normalized["estimated_prompt_tokens"] = estimated
                    normalized["current_prompt_tokens"] = estimated
            except Exception:
                logger.debug("Failed to estimate prompt tokens for %s", resolved, exc_info=True)
        return web.json_response({"session": normalized})
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
        session_id = requested_id or f"sess_{uuid.uuid4().hex}"
        title = str(body.get("title") or "").strip() or None
        source = str(body.get("source") or "api_server").strip() or "api_server"
        model = str(body.get("model") or "").strip() or None
        model_config = None
        has_model_override = bool(model) or any(
            key in body for key in ("model_config", "provider", "base_url", "api_mode")
        )
        if has_model_override:
            resolved_model, resolved_config, err = _resolve_session_model_config(body)
            if err:
                return web.json_response({"error": err}, status=400)
            model = resolved_model
            model_config = resolved_config
        system_prompt = str(body.get("system_prompt") or "").strip() or None
        profile = str(body.get("profile") or "").strip() or None
        db = ensure_session_db()
        if not db:
            return web.json_response({"error": "Session DB unavailable"}, status=503)
        created_id = db.create_session(
            session_id=session_id,
            source=source,
            model=model,
            model_config=model_config,
            system_prompt=system_prompt,
            profile=profile,
        )
        if title:
            try:
                db.set_session_title(created_id, title)
            except Exception:
                pass
        item = db.get_session(created_id) or {"id": created_id, "model": model, "title": title, "source": source, "started_at": time.time()}
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
        if "title" in body:
            title = str(body.get("title") or "").strip()
            if not title:
                return web.json_response({"error": "title required"}, status=400)
            ok = db.set_session_title(resolved, title)
            if not ok:
                return web.json_response({"error": "Session not found"}, status=404)
        if "system_prompt" in body:
            db.update_system_prompt(resolved, str(body.get("system_prompt") or "").strip())
        if "end_reason" in body:
            db.end_session(resolved, str(body.get("end_reason") or "updated"))
        if "profile" in body:
            current = db.get_session(resolved)
            if current is None:
                return web.json_response({"error": "Session not found"}, status=404)
            profile = str(body.get("profile") or "").strip() or None

            def _do_profile(conn):
                cur = conn.execute(
                    "UPDATE sessions SET profile = ? WHERE id = ?",
                    (profile, resolved),
                )
                return cur.rowcount

            ok = (db._execute_write(_do_profile) or 0) > 0
            if not ok:
                return web.json_response({"error": "Session not found"}, status=404)
        if any(key in body for key in ("model", "model_config", "provider", "base_url", "api_mode")):
            current = db.get_session(resolved)
            if current is None:
                return web.json_response({"error": "Session not found"}, status=404)
            resolved_model, resolved_config, err = _resolve_session_model_config(body, current_session=current)
            if err:
                return web.json_response({"error": err}, status=400)
            if hasattr(db, "update_session_model"):
                ok = db.update_session_model(resolved, resolved_model, resolved_config)
            else:
                def _do(conn):
                    cur = conn.execute(
                        "UPDATE sessions SET model = ?, model_config = ? WHERE id = ?",
                        (resolved_model, json.dumps(resolved_config) if resolved_config else None, resolved),
                    )
                    return cur.rowcount
                ok = (db._execute_write(_do) or 0) > 0
            if not ok:
                return web.json_response({"error": "Session not found"}, status=404)
        item = db.get_session(resolved)
        if item is None:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"session": _normalize_session_record(item)})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
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
    t_start = time.time()
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
        from api_server.utils import _parse_int
        limit = _parse_int(request.query.get("limit"), None)
        offset = _parse_int(request.query.get("offset"), 0)
        order = request.query.get("order", "asc")  # "asc" | "desc"
        include_lineage = _query_bool(request.query.get("include_lineage"))
        t_db = time.time()
        if include_lineage:
            all_items, lineage = _messages_for_lineage(db, resolved, order=order)
            total = len(all_items)
            items = all_items[offset:]
            if limit is not None:
                items = items[:limit]
        else:
            lineage = None
            items = db.get_messages(resolved, limit=limit, offset=offset, order=order)
            total = db.message_count(resolved) if hasattr(db, 'message_count') else len(items)
        t_items = time.time()
        t_total = time.time()
        elapsed = (t_total - t_start) * 1000
        db_elapsed = (t_items - t_db) * 1000
        count_elapsed = (t_total - t_items) * 1000
        if elapsed > 50:
            logger.warning(f"[SLOW MSG] session={resolved} total={elapsed:.0f}ms db={db_elapsed:.0f}ms count={count_elapsed:.0f}ms items={len(items)} limit={limit}")
        payload = {"items": items, "total": total, "limit": limit, "offset": offset}
        if include_lineage:
            payload["lineage"] = lineage
        return web.json_response(payload)
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
            profile=original.get("profile"),
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


async def handle_search_sessions(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    """GET /api/sessions/search -- search messages across sessions."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    query = (request.query.get("q") or "").strip()
    if not query:
        return web.json_response({"error": "Missing query parameter: q"}, status=400)
    try:
        db = ensure_session_db()
        if not db:
            return web.json_response({"items": [], "total": 0})
        from api_server.utils import _parse_int
        limit = _parse_int(request.query.get("limit"), 20)
        offset = _parse_int(request.query.get("offset"), 0)
        results = db.search_messages(query=query, limit=limit, offset=offset)
        return web.json_response({"query": query, "count": len(results), "results": results})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        logger.exception("Error searching sessions")
        return web.json_response({"error": str(e)}, status=500)


# ============================================================================
# Discord Alternative — Phase 2: Session Status & Message Diff Sync
# ============================================================================

async def handle_session_status(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    """GET /api/sessions/{session_id}/status — observe session state for cross-device sync.

    Returns: last_message_at, message_count, is_streaming, last_message_id
    """
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    try:
        db = ensure_session_db()
        if not db:
            return web.json_response({"error": "Session DB unavailable"}, status=503)
        session_id = request.match_info.get("session_id", "")
        resolved = db.resolve_session_id(session_id) or session_id
        session = db.get_session(resolved)
        if not session:
            return web.json_response({"error": "Session not found"}, status=404)

        messages = db.get_messages(resolved)
        last_message = messages[-1] if messages else None
        last_message_id = last_message.get("id") if last_message else None
        last_message_at = last_message.get("created_at") if last_message else None

        return web.json_response({
            "session_id": resolved,
            "last_message_at": last_message_at,
            "message_count": len(messages),
            "last_message_id": last_message_id,
            "last_message_role": last_message.get("role") if last_message else None,
        })
    except Exception as e:
        logger.exception("Error getting session status")
        return web.json_response({"error": str(e)}, status=500)


async def handle_session_messages_diff(request: web.Request, *, check_auth, ensure_session_db) -> web.Response:
    """GET /api/sessions/{session_id}/messages?since=message_id — differential sync.

    Returns only messages after the given message_id (exclusive).
    If since is not provided, returns all messages (fallback).
    """
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

        since_id = (request.query.get("since") or "").strip()
        items = db.get_messages(resolved)

        if since_id:
            # Find index of since_id, return messages after it
            found_idx = -1
            for i, msg in enumerate(items):
                if str(msg.get("id", "")) == since_id:
                    found_idx = i
                    break
            if found_idx >= 0:
                items = items[found_idx + 1:]
            # If not found, return all (client may have stale since_id)

        return web.json_response({
            "items": items,
            "total": len(items),
            "since": since_id or None,
            "session_id": resolved,
        })
    except Exception as e:
        logger.exception("Error getting session messages diff")
        return web.json_response({"error": str(e)}, status=500)
