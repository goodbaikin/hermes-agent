"""Session chat handlers for the API Server."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 15


async def handle_session_chat(
    request: web.Request,
    *,
    check_auth,
    get_session_db,
    normalize_session_record,
    build_user_content,
    create_agent,
) -> web.Response:
    """POST /api/sessions/{session_id}/chat -- run a session-aware chat turn."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    session_id = request.match_info["session_id"]
    db = get_session_db()
    session = normalize_session_record(db.get_session(session_id))
    if session is None:
        db.ensure_session(session_id, source="web")
        session = normalize_session_record(db.get_session(session_id))
        if session is None:
            session = {"id": session_id, "title": None}

    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON in request body"}, status=400)

    message = body.get("message")
    if not isinstance(message, str):
        return web.json_response({"error": "Missing or invalid 'message' field"}, status=400)

    raw_attachments_sync = body.get("attachments")
    if raw_attachments_sync:
        logger.debug("[chat] Received %d attachment(s): %s",
                     len(raw_attachments_sync),
                     [(a.get("name"), a.get("contentType"), len(a.get("content", "") or a.get("base64", "") or "")) for a in raw_attachments_sync if isinstance(a, dict)])
    user_content, persist_text = build_user_content(message, raw_attachments_sync)
    if isinstance(user_content, list):
        logger.debug("[chat] Built multimodal content with %d parts", len(user_content))

    model = body.get("model") or session.get("model") or "hermes-agent"
    system_message = body.get("system_message")
    history = db.get_messages_as_conversation(session_id)
    loop = asyncio.get_event_loop()

    def _run():
        agent = create_agent(
            ephemeral_system_prompt=system_message,
            session_id=session_id,
        )
        agent._session_db = db  # Enable session persistence
        result = agent.run_conversation(
            user_content,
            conversation_history=history,
            persist_user_message=persist_text,
        )
        usage = {
            "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
            "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
            "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
        }
        return result, usage

    try:
        result, usage = await loop.run_in_executor(None, _run)
    except Exception as e:
        logger.error("Error running session chat for %s: %s", session_id, e, exc_info=True)
        return web.json_response({"error": str(e)}, status=500)

    return web.json_response({
        "session_id": session_id,
        "run_id": f"run_{uuid.uuid4().hex}",
        "model": model,
        "final_response": result.get("final_response"),
        "completed": result.get("completed", False),
        "partial": result.get("partial", False),
        "interrupted": result.get("interrupted", False),
        "api_calls": result.get("api_calls", 0),
        "messages": result.get("messages", []),
        "last_reasoning": result.get("last_reasoning"),
        "response_previewed": result.get("response_previewed", False),
        "usage": usage,
    })


async def handle_session_chat_stream(
    request: web.Request,
    *,
    check_auth,
    get_session_db,
    normalize_session_record,
    build_user_content,
    create_agent,
    cors_headers_for_origin,
) -> web.StreamResponse:
    """POST /api/sessions/{session_id}/chat/stream -- stream a session chat turn over SSE."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    session_id = request.match_info["session_id"]
    db = get_session_db()
    session = normalize_session_record(db.get_session(session_id))
    if session is None:
        db.ensure_session(session_id, source="web")
        session = normalize_session_record(db.get_session(session_id))
        if session is None:
            session = {"id": session_id, "title": None}

    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response({"error": "Invalid JSON in request body"}, status=400)

    message = body.get("message")
    if not isinstance(message, str):
        return web.json_response({"error": "Missing or invalid 'message' field"}, status=400)

    # Build multimodal content if image attachments are present
    raw_attachments = body.get("attachments")
    if raw_attachments:
        logger.debug("[chat/stream] Received %d attachment(s): %s",
                     len(raw_attachments),
                     [(a.get("name"), a.get("contentType"), len(a.get("content", "") or a.get("base64", "") or "")) for a in raw_attachments if isinstance(a, dict)])
    user_content, persist_text = build_user_content(message, raw_attachments)
    if isinstance(user_content, list):
        logger.debug("[chat/stream] Built multimodal content with %d parts", len(user_content))

    system_message = body.get("system_message")
    history = db.get_messages_as_conversation(session_id)
    assistant_message_id = f"msg_asst_{uuid.uuid4().hex}"

    # Note: user message persistence is handled by AIAgent._flush_messages_to_session_db
    # Don't double-persist here or messages will appear twice

    import queue as _q
    stream_q: _q.Queue = _q.Queue()

    def _encode_sse(event_name: str, payload: Dict[str, Any]) -> bytes:
        return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

    def _queue_event(event_name: str, payload: Dict[str, Any]) -> None:
        stream_q.put(_encode_sse(event_name, payload))

    def _tool_map(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapping: Dict[str, Dict[str, Any]] = {}
        for item in messages:
            if item.get("role") != "assistant":
                continue
            for index, tool_call in enumerate(item.get("tool_calls") or []):
                tool_id = tool_call.get("id")
                if not tool_id:
                    continue
                fn = tool_call.get("function") or {}
                raw_args = fn.get("arguments")
                try:
                    parsed_args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
                except json.JSONDecodeError:
                    parsed_args = raw_args
                mapping[tool_id] = {
                    "tool_name": fn.get("name") or item.get("tool_name") or f"tool_{index + 1}",
                    "args": parsed_args,
                }
        return mapping

    def _result_preview(content: Any, limit: int = 4000) -> str:
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return text[:limit] + ("..." if len(text) > limit else "")

    run_id = f"run_{uuid.uuid4().hex}"

    def _on_delta(delta):
        if delta:
            _queue_event(
                "assistant.delta",
                {"session_id": session_id, "run_id": run_id, "message_id": assistant_message_id, "delta": delta},
            )

    def _on_tool_start(tool_call_id, function_name, function_args):
        _queue_event("tool.started", {
            "session_id": session_id,
            "run_id": run_id,
            "tool_call_id": tool_call_id,
            "tool_name": function_name,
            "args": function_args,
        })

    def _on_tool_progress(event_type, name, preview, args, **kwargs):
        if name == "_thinking":
            _queue_event(
                "tool.progress",
                {"session_id": session_id, "run_id": run_id, "message_id": assistant_message_id, "delta": preview},
            )
            return
        tool_call_id = kwargs.get("tool_call_id", "")
        logger.info(f"[_on_tool_progress] tool.started name={name} tool_call_id={tool_call_id}")
        payload = {
            "session_id": session_id,
            "run_id": run_id,
            "tool_name": name,
            "preview": preview,
            "args": args,
            "tool_call_id": tool_call_id,
        }
        _queue_event("tool.started", payload)
        # Also send tool.progress for progress updates
        _queue_event("tool.progress", payload)

    agent_ref = [None]
    loop = asyncio.get_event_loop()

    def _make_tool_complete_callback(run_id, loop):
        """Return a tool_complete_callback that pushes tool.completed events to the SSE queue."""
        def _callback(tool_call_id, tool_name, args, function_result):
            logger.info(f"[TOOL COMPLETE CALLBACK] tool_call_id={tool_call_id} tool_name={tool_name}")
            try:
                result_preview = ""
                is_error = False
                if isinstance(function_result, dict):
                    result_preview = function_result.get("preview", "") or function_result.get("output_preview", "")
                    is_error = function_result.get("is_error", False) or function_result.get("error", False)
                elif isinstance(function_result, str):
                    result_preview = function_result[:200]
                loop.call_soon_threadsafe(
                    _queue_event,
                    "tool.completed",
                    {
                        "run_id": run_id,
                        "timestamp": time.time(),
                        "tool": tool_name,
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "result_preview": result_preview,
                        "is_error": is_error,
                    }
                )
            except Exception:
                pass
        return _callback

    async def _run_agent_task():
        def _run():
            agent = create_agent(
                ephemeral_system_prompt=system_message,
                session_id=session_id,
                stream_delta_callback=_on_delta,
                tool_progress_callback=_on_tool_progress,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_make_tool_complete_callback(run_id, loop),
            )
            agent._session_db = db  # Enable session persistence
            agent_ref[0] = agent
            return agent.run_conversation(
                user_content,
                conversation_history=history,
                persist_user_message=persist_text,
            )

        return await loop.run_in_executor(None, _run)

    agent_task = asyncio.ensure_future(_run_agent_task())

    sse_headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    origin = request.headers.get("Origin", "")
    cors = cors_headers_for_origin(origin) if origin else None
    if cors:
        sse_headers.update(cors)

    response = web.StreamResponse(status=200, headers=sse_headers)
    await response.prepare(request)

    try:
        user_message_id = f"msg_user_{uuid.uuid4().hex}"
        await response.write(_encode_sse("session.created", {
            "session_id": session_id,
            "run_id": run_id,
            "title": session.get("title") or "New Chat",
        }))
        await response.write(_encode_sse("run.started", {
            "session_id": session_id,
            "run_id": run_id,
            "user_message": {
                "id": user_message_id,
                "role": "user",
                "content": message,
            },
        }))
        await response.write(_encode_sse("message.started", {
            "session_id": session_id,
            "run_id": run_id,
            "message": {"id": assistant_message_id, "role": "assistant"},
        }))

        last_activity = time.monotonic()
        while True:
            try:
                frame = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
            except _q.Empty:
                if agent_task.done():
                    while True:
                        try:
                            frame = stream_q.get_nowait()
                            if frame is None:
                                break
                            await response.write(frame)
                        except _q.Empty:
                            break
                    break
                # Send periodic keepalive to prevent client/proxy
                # timeouts during agent init and long LLM API calls.
                if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                    await response.write(b": keepalive\n\n")
                    last_activity = time.monotonic()
                continue

            if frame is None:
                break

            await response.write(frame)
            last_activity = time.monotonic()

        try:
            result = await agent_task
        except Exception:
            result = {"messages": [], "final_response": "", "completed": False}

        # Auto-generate session title after first exchange (synchronous — HTTP request/response)
        try:
            from agent.title_generator import auto_title_session
            conversation_history = (history or []) + (result.get("messages") or [])
            user_msg = ""
            assistant_response = result.get("final_response") or ""
            for m in conversation_history:
                if m.get("role") == "user" and not user_msg:
                    user_msg = m.get("content", "")
                    if isinstance(user_msg, list):
                        text_parts = [p.get("text", "") for p in user_msg if isinstance(p, dict) and p.get("type") == "text"]
                        user_msg = " ".join(text_parts)
                    user_msg = str(user_msg).strip()
            auto_title_session(
                session_db=db,
                session_id=session_id,
                user_message=user_msg,
                assistant_response=assistant_response,
            )
        except Exception:
            logger.debug("Auto-title failed for session %s", session_id, exc_info=True)

        # Build usage from agent result
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        try:
            agent = agent_ref[0]
            if agent is not None:
                usage = {
                    "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                    "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                    "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                }
                logger.info("[session_chat] Agent usage for %s: input=%s output=%s total=%s",
                           session_id, usage["input_tokens"], usage["output_tokens"], usage["total_tokens"])
                # Save current prompt tokens to session DB for context gauge
                try:
                    db._execute_write(lambda conn: conn.execute(
                        "UPDATE sessions SET current_prompt_tokens = ? WHERE id = ?",
                        (usage["input_tokens"], session_id)
                    ))
                    logger.info("[session_chat] Saved current_prompt_tokens=%s for session %s",
                               usage["input_tokens"], session_id)
                except Exception as e:
                    logger.error("[session_chat] Failed to save current_prompt_tokens: %s", e)
        except Exception as e:
            logger.error("[session_chat] Error building usage: %s", e)

        await response.write(_encode_sse("assistant.completed", {
            "session_id": session_id,
            "run_id": run_id,
            "message_id": assistant_message_id,
            "content": result.get("final_response") or "",
            "completed": result.get("completed", False),
            "partial": result.get("partial", False),
            "interrupted": result.get("interrupted", False),
        }))
        await response.write(_encode_sse("run.completed", {
            "session_id": session_id,
            "run_id": run_id,
            "message_id": assistant_message_id,
            "completed": result.get("completed", False),
            "partial": result.get("partial", False),
            "interrupted": result.get("interrupted", False),
            "api_calls": result.get("api_calls"),
            "usage": usage,
        }))
        await response.write(_encode_sse("done", {"session_id": session_id, "run_id": run_id, "state": "final"}))
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
        agent = agent_ref[0]
        if agent is not None:
            try:
                agent.interrupt("SSE client disconnected")
            except Exception:
                pass
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
        logger.info("Session SSE client disconnected; interrupted session %s", session_id)

    return response
