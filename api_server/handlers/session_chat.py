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


def _effective_session_model(session: Dict[str, Any], body: Optional[Dict[str, Any]] = None) -> str:
    body = body or {}
    cfg = session.get("model_config") if isinstance(session, dict) else None
    if isinstance(cfg, str) and cfg.strip():
        try:
            cfg = json.loads(cfg)
        except json.JSONDecodeError:
            cfg = None
    if not isinstance(cfg, dict):
        cfg = {}
    return str(body.get("model") or cfg.get("model") or session.get("model") or "hermes-agent")


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce usage counters to JSON-safe ints; MagicMock/test doubles become 0."""
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.strip():
            return int(float(value))
    except (TypeError, ValueError, OverflowError):
        pass
    return default


def _agent_session_id(agent: Any, fallback: str) -> str:
    """Return the live agent session id after compression rotation, if known."""
    try:
        current = getattr(agent, "session_id", None)
    except Exception:
        current = None
    return current if isinstance(current, str) and current else fallback


async def handle_session_chat(
    request: web.Request,
    *,
    check_auth,
    get_session_db,
    normalize_session_record,
    build_user_content,
    create_agent,
    register_active_session_task=None,
    register_active_session_agent=None,
    unregister_active_session=None,
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

    model = _effective_session_model(session, body)
    system_message = body.get("system_message")
    history = db.get_messages_as_conversation(session_id, include_ancestors=True)
    loop = asyncio.get_event_loop()

    def _run():
        agent = create_agent(
            ephemeral_system_prompt=system_message,
            session_id=session_id,
            profile=session.get("profile"),
        )
        agent._session_db = db  # Enable session persistence
        if register_active_session_agent is not None:
            try:
                register_active_session_agent(session_id, agent)
            except Exception:
                logger.debug("Failed to register active session agent %s", session_id, exc_info=True)
        result = agent.run_conversation(
            user_content,
            conversation_history=history,
            persist_user_message=persist_text,
        )
        actual_session_id = _agent_session_id(agent, session_id)
        usage = {
            "input_tokens": _safe_int(getattr(agent, "session_prompt_tokens", 0)),
            "output_tokens": _safe_int(getattr(agent, "session_completion_tokens", 0)),
            "total_tokens": _safe_int(getattr(agent, "session_total_tokens", 0)),
        }
        return result, usage, actual_session_id

    run_task = None
    try:
        import contextvars
        ctx = contextvars.copy_context()
        run_task = asyncio.ensure_future(loop.run_in_executor(None, ctx.run, _run))
        if register_active_session_task is not None:
            try:
                register_active_session_task(session_id, run_task)
            except Exception:
                logger.debug("Failed to register active session task %s", session_id, exc_info=True)
        result, usage, actual_session_id = await run_task
    except Exception as e:
        logger.error("Error running session chat for %s: %s", session_id, e, exc_info=True)
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if unregister_active_session is not None:
            try:
                unregister_active_session(session_id, run_task)
            except Exception:
                logger.debug("Failed to unregister active session %s", session_id, exc_info=True)

    return web.json_response({
        "session_id": actual_session_id,
        "continued_from": session_id if actual_session_id != session_id else None,
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
    register_active_session_task=None,
    register_active_session_agent=None,
    unregister_active_session=None,
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
    history = db.get_messages_as_conversation(session_id, include_ancestors=True)
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
        if delta and delta != "":
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
        if event_type == "tool.started":
            logger.info("[_on_tool_progress] tool.started name=%s tool_call_id=%s", name, tool_call_id)
            from agent.display import build_tool_preview
            preview = build_tool_preview(name, args) or name
            payload = {
                "session_id": session_id,
                "run_id": run_id,
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "preview": preview,
                "args": args,
            }
            _queue_event("tool.started", payload)
        elif event_type == "tool.completed":
            # tool.completed is handled by _make_tool_complete_callback for consistent
            # tool_call_id and result_preview. Skip here to avoid double-fire.
            pass
        elif event_type == "tool.progress":
            payload = {
                "session_id": session_id,
                "run_id": run_id,
                "tool_call_id": tool_call_id,
                "tool_name": name,
                "preview": preview,
                "args": args,
            }
            _queue_event("tool.progress", payload)

    agent_ref = [None]
    completed_tool_call_ids = set()
    loop = asyncio.get_event_loop()

    def _make_tool_complete_callback(run_id, loop):
        """Return a tool_complete_callback that pushes tool.completed events to the SSE queue."""
        def _callback(tool_call_id, tool_name, args, function_result):
            logger.info("[TOOL COMPLETE CALLBACK] tool_call_id=%s tool_name=%s", tool_call_id, tool_name)
            try:
                if tool_call_id:
                    completed_tool_call_ids.add(tool_call_id)
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
                tool_start_callback=None,  # tool.started is handled by tool_progress_callback to avoid double-fire
                tool_complete_callback=_make_tool_complete_callback(run_id, loop),
                profile=session.get("profile"),
            )
            agent._session_db = db  # Enable session persistence
            agent_ref[0] = agent
            if register_active_session_agent is not None:
                try:
                    register_active_session_agent(session_id, agent)
                except Exception:
                    logger.debug("Failed to register active session agent %s", session_id, exc_info=True)
            return agent.run_conversation(
                user_content,
                conversation_history=history,
                persist_user_message=persist_text,
            )

        import contextvars
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(None, ctx.run, _run)

    agent_task = asyncio.ensure_future(_run_agent_task())
    if register_active_session_task is not None:
        try:
            register_active_session_task(session_id, agent_task)
        except Exception:
            logger.debug("Failed to register active session task %s", session_id, exc_info=True)

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
        except Exception as e:
            logger.error("[session_chat] Agent task failed: %s", e)
            result = {
                "messages": [],
                "final_response": "",
                "completed": False,
                "failed": True,
                "error": str(e),
            }
            # Signal SSE loop to terminate
            try:
                stream_q.put(None)
            except Exception:
                pass

        agent = agent_ref[0]
        actual_session_id = _agent_session_id(agent, session_id)
        if actual_session_id != session_id:
            try:
                rotated_session = normalize_session_record(db.get_session(actual_session_id)) or {
                    "id": actual_session_id,
                    "session_id": actual_session_id,
                    "title": session.get("title") or "New Chat",
                    "parent_session_id": session_id,
                }
            except Exception:
                rotated_session = {
                    "id": actual_session_id,
                    "session_id": actual_session_id,
                    "title": session.get("title") or "New Chat",
                    "parent_session_id": session_id,
                }
            await response.write(_encode_sse("session.created", {
                "session_id": actual_session_id,
                "run_id": run_id,
                "title": rotated_session.get("title") or "New Chat",
                "parent_session_id": rotated_session.get("parent_session_id") or session_id,
                "session": rotated_session,
            }))

        # Some test doubles and older agent paths return tool-call transcripts
        # without invoking the live tool_complete_callback. Backfill missing
        # tool.completed events from the final message list, but do not double-fire
        # calls already seen through the callback path.
        try:
            tool_lookup = _tool_map(result.get("messages") or [])
            for item in result.get("messages") or []:
                if item.get("role") != "tool":
                    continue
                tool_call_id = item.get("tool_call_id") or ""
                if tool_call_id and tool_call_id in completed_tool_call_ids:
                    continue
                meta = tool_lookup.get(tool_call_id, {})
                await response.write(_encode_sse("tool.completed", {
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "tool": item.get("tool_name") or meta.get("tool_name") or "tool",
                    "tool_name": item.get("tool_name") or meta.get("tool_name") or "tool",
                    "tool_call_id": tool_call_id,
                    "result_preview": _result_preview(item.get("content", ""), limit=4000),
                    "is_error": False,
                }))
                if tool_call_id:
                    completed_tool_call_ids.add(tool_call_id)
        except Exception:
            logger.debug("Failed to backfill tool.completed SSE events", exc_info=True)

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
                session_id=actual_session_id,
                user_message=user_msg,
                assistant_response=assistant_response,
            )
        except Exception:
            logger.debug("Auto-title failed for session %s", actual_session_id, exc_info=True)

        # Build usage from agent result
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        try:
            agent = agent_ref[0]
            if agent is not None:
                # Use last_prompt_tokens (current turn) instead of session_prompt_tokens (cumulative)
                last_prompt = 0
                try:
                    compressor = getattr(agent, "context_compressor", None)
                    last_prompt = _safe_int(getattr(compressor, "last_prompt_tokens", 0))
                except Exception:
                    last_prompt = _safe_int(getattr(agent, "session_prompt_tokens", 0))
                if not last_prompt:
                    last_prompt = _safe_int(getattr(agent, "session_prompt_tokens", 0))
                usage = {
                    "input_tokens": last_prompt,
                    "output_tokens": _safe_int(getattr(agent, "session_completion_tokens", 0)),
                    "total_tokens": _safe_int(getattr(agent, "session_total_tokens", 0)),
                }
                logger.info("[session_chat] Agent usage for %s: input=%s output=%s total=%s",
                           actual_session_id, usage["input_tokens"], usage["output_tokens"], usage["total_tokens"])
                # Save current prompt tokens to session DB for context gauge.
                # Also persist cumulative token counters here: the agent loop may
                # only have canonical input_tokens (excluding cache reads/writes),
                # while the context gauge needs prompt_tokens as the actual context
                # size. Keeping both in the session row lets the Web UI recover
                # from /api/sessions/{id} alone.
                try:
                    if hasattr(db, "update_token_counts"):
                        db.update_token_counts(
                            actual_session_id,
                            input_tokens=_safe_int(getattr(agent, "session_input_tokens", 0)) or usage["input_tokens"],
                            output_tokens=_safe_int(getattr(agent, "session_output_tokens", 0)) or usage["output_tokens"],
                            cache_read_tokens=_safe_int(getattr(agent, "session_cache_read_tokens", 0)),
                            cache_write_tokens=_safe_int(getattr(agent, "session_cache_write_tokens", 0)),
                            reasoning_tokens=_safe_int(getattr(agent, "session_reasoning_tokens", 0)),
                            model=getattr(agent, "model", None),
                            api_call_count=_safe_int(getattr(agent, "session_api_calls", 0)),
                            absolute=True,
                        )
                    db.update_current_prompt_tokens(actual_session_id, last_prompt)
                    logger.info("[session_chat] Saved current_prompt_tokens=%s for session %s",
                               last_prompt, actual_session_id)
                except Exception as e:
                    logger.error("[session_chat] Failed to save token usage/current_prompt_tokens: %s", e)
        except Exception as e:
            logger.error("[session_chat] Error building usage: %s", e)

        final_content = result.get("final_response") or ""
        completed = bool(result.get("completed", False))
        partial = bool(result.get("partial", False))
        interrupted = bool(result.get("interrupted", False))
        failed = bool(result.get("failed", False))
        error_message = str(result.get("error") or final_content or "Agent run failed")

        await response.write(_encode_sse("assistant.completed", {
            "session_id": actual_session_id,
            "run_id": run_id,
            "message_id": assistant_message_id,
            "content": final_content,
            "completed": completed,
            "partial": partial,
            "interrupted": interrupted,
            "failed": failed,
            "error": error_message if failed else None,
        }))
        if failed and not interrupted:
            await response.write(_encode_sse("run.failed", {
                "session_id": actual_session_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "completed": completed,
                "partial": partial,
                "interrupted": interrupted,
                "failed": True,
                "error": error_message,
                "api_calls": result.get("api_calls"),
                "usage": usage,
            }))
        else:
            await response.write(_encode_sse("run.completed", {
                "session_id": actual_session_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "completed": completed,
                "partial": partial,
                "interrupted": interrupted,
                "api_calls": result.get("api_calls"),
                "usage": usage,
            }))
        await response.write(_encode_sse("done", {"session_id": actual_session_id, "run_id": run_id, "state": "final"}))
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

    # Ensure response is fully closed for clean SSE termination
    try:
        await response.write_eof()
    except Exception:
        pass
    finally:
        if unregister_active_session is not None:
            try:
                unregister_active_session(session_id, agent_task)
            except Exception:
                logger.debug("Failed to unregister active session %s", session_id, exc_info=True)

    return response
