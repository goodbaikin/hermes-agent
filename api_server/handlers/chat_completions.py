import asyncio
import json
import logging
import queue as _q
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import web

from api_server.utils import (
    _openai_error,
    _normalize_chat_content,
    _normalize_multimodal_content,
    _content_has_visible_payload,
    _multimodal_validation_error,
    _derive_chat_session_id,
    _make_request_fingerprint,
)


logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0


def _parse_stream_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


async def handle_chat_completions(
    request: web.Request,
    *,
    adapter: Any,
    idem_cache: Any,
) -> web.Response:
    """POST /v1/chat/completions -- OpenAI Chat Completions format."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err
    session_key, session_key_err = adapter._parse_session_key_header(request)
    if session_key_err:
        return session_key_err

    # Parse request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return web.json_response(
            {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
            status=400,
        )

    # Fast-path for capability probes (max_tokens=1)
    max_tokens = body.get("max_tokens")
    if max_tokens == 1:
        return web.json_response({
            "id": f"chatcmpl-probe-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "") or "hermes-agent",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
        })

    stream = _parse_stream_flag(body.get("stream", False))

    # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
    system_prompt = None
    conversation_messages: List[Dict[str, str]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role", "")
        raw_content = msg.get("content", "")
        if role == "system":
            content = _normalize_chat_content(raw_content)
            if system_prompt is None:
                system_prompt = content
            else:
                system_prompt = system_prompt + "\n" + content
        elif role in ("user", "assistant"):
            try:
                content = _normalize_multimodal_content(raw_content)
            except ValueError as exc:
                return _multimodal_validation_error(exc, param=f"messages[{idx}].content")
            conversation_messages.append({"role": role, "content": content})

    # Extract the last user message as the primary input
    user_message: Any = ""
    history = []
    if conversation_messages:
        user_message = conversation_messages[-1].get("content", "")
        history = conversation_messages[:-1]

    if not _content_has_visible_payload(user_message):
        return web.json_response(
            {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
            status=400,
        )

    # Allow caller to continue an existing session by passing X-Hermes-Session-Id.
    provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
    if provided_session_id:
        if not adapter._api_key:
            logger.warning(
                "Session continuation via X-Hermes-Session-Id rejected: "
                "no API key configured.  Set API_SERVER_KEY to enable "
                "session continuity."
            )
            return web.json_response(
                _openai_error(
                    "Session continuation requires API key authentication. "
                    "Configure API_SERVER_KEY to enable this feature."
                ),
                status=403,
            )
        # Sanitize: reject control characters that could enable header injection.
        if re.search(r'[\r\n\x00]', provided_session_id):
            return web.json_response(
                {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                status=400,
            )
        session_id = provided_session_id
        try:
            db = adapter._ensure_session_db()
            if db is not None:
                history = db.get_messages_as_conversation(session_id)
        except Exception as e:
            logger.warning("Failed to load session history for %s: %s", session_id, e)
            history = []
    else:
        first_user = ""
        for cm in conversation_messages:
            if cm.get("role") == "user":
                first_user = cm.get("content", "")
                break
        session_id = _derive_chat_session_id(system_prompt, first_user)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
    model_name = body.get("model", adapter._model_name)
    created = int(time.time())

    if stream:
        _stream_q: _q.Queue = _q.Queue()

        def _on_delta(delta):
            if delta is not None and delta != "":
                _stream_q.put(delta)

        _started_tool_call_ids: set[str] = set()

        def _on_tool_progress(event_type, name, preview, args, **kwargs):
            if event_type == "tool.started":
                tool_call_id = kwargs.get("tool_call_id", "")
                if not tool_call_id or name.startswith("_"):
                    return
                _started_tool_call_ids.add(tool_call_id)
                from agent.display import build_tool_preview, get_tool_emoji
                label = build_tool_preview(name, args) or name
                _stream_q.put(("__tool_progress__", {
                    "tool": name,
                    "emoji": get_tool_emoji(name),
                    "label": label,
                    "toolCallId": tool_call_id,
                    "status": "running",
                }))
            elif event_type == "tool.completed":
                tool_call_id = kwargs.get("tool_call_id", "")
                if not tool_call_id or tool_call_id not in _started_tool_call_ids:
                    return
                _started_tool_call_ids.discard(tool_call_id)
                _stream_q.put(("__tool_progress__", {
                    "tool": name,
                    "toolCallId": tool_call_id,
                    "status": "completed",
                }))

        def _on_tool_start(tool_call_id, function_name, function_args):
            if not tool_call_id or str(function_name).startswith("_"):
                return
            _started_tool_call_ids.add(tool_call_id)
            from agent.display import build_tool_preview, get_tool_emoji
            label = build_tool_preview(function_name, function_args) or function_name
            _stream_q.put(("__tool_progress__", {
                "tool": function_name,
                "emoji": get_tool_emoji(function_name),
                "label": label,
                "toolCallId": tool_call_id,
                "status": "running",
            }))

        def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
            if not tool_call_id or tool_call_id not in _started_tool_call_ids:
                return
            _started_tool_call_ids.discard(tool_call_id)
            _stream_q.put(("__tool_progress__", {
                "tool": function_name,
                "toolCallId": tool_call_id,
                "status": "completed",
            }))

        agent_ref = [None]
        agent_task = asyncio.ensure_future(adapter._run_agent(
            user_message=user_message,
            conversation_history=history,
            ephemeral_system_prompt=system_prompt,
            session_id=session_id,
            gateway_session_key=session_key,
            stream_delta_callback=_on_delta,
            tool_progress_callback=_on_tool_progress,
            tool_start_callback=_on_tool_start,
            tool_complete_callback=_on_tool_complete,
            agent_ref=agent_ref,
        ))
        agent_task.add_done_callback(lambda _task: _stream_q.put(None))

        return await write_sse_chat_completion(
            request, completion_id, model_name, created, _stream_q,
            agent_task, agent_ref, session_id=session_id,
            adapter=adapter,
            session_key=session_key,
        )

    # Non-streaming: run the agent (with optional Idempotency-Key)
    async def _compute_completion():
        return await adapter._run_agent(
            user_message=user_message,
            conversation_history=history,
            ephemeral_system_prompt=system_prompt,
            session_id=session_id,
            gateway_session_key=session_key,
        )

    idempotency_key = request.headers.get("Idempotency-Key")
    if idempotency_key:
        fp = _make_request_fingerprint(body, keys=["model", "messages", "tools", "tool_choice", "stream"])
        try:
            result, usage = await idem_cache.get_or_set(idempotency_key, fp, _compute_completion)
        except Exception as e:
            logger.error("Error running agent for chat completions: %s", e, exc_info=True)
            return web.json_response(
                _openai_error(f"Internal server error: {e}", err_type="server_error"),
                status=500,
            )
    else:
        try:
            result, usage = await _compute_completion()
        except Exception as e:
            logger.error("Error running agent for chat completions: %s", e, exc_info=True)
            return web.json_response(
                _openai_error(f"Internal server error: {e}", err_type="server_error"),
                status=500,
            )

    final_response = result.get("final_response", "")
    has_text = isinstance(final_response, str) and bool(final_response)
    completed = result.get("completed", True)
    partial = bool(result.get("partial", False))
    failed = bool(result.get("failed", False))
    error_text = str(result.get("error") or "")

    headers = {"X-Hermes-Session-Id": session_id}
    if session_key:
        headers["X-Hermes-Session-Key"] = session_key

    if not has_text and (failed or completed is False or partial):
        headers["X-Hermes-Completed"] = "false"
        if partial:
            headers["X-Hermes-Partial"] = "true"
        error_payload = _openai_error(
            error_text or "Agent did not complete and produced no assistant text",
            err_type="server_error",
            code="agent_incomplete",
        )
        error_payload["error"]["hermes"] = {"partial": partial, "completed": completed, "failed": failed}
        return web.json_response(
            error_payload,
            status=502,
            headers=headers,
        )

    if not has_text:
        final_response = error_text or "(No response generated)"

    finish_reason = "length" if (partial or completed is False) else "stop"
    response_data = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_response,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }

    if partial or completed is False or failed:
        response_data["hermes"] = {
            "partial": partial,
            "completed": completed,
            "failed": failed,
            "error_code": "output_truncated" if (partial or completed is False) else "agent_failed",
        }
        if error_text:
            response_data["hermes"]["error"] = error_text
        headers["X-Hermes-Completed"] = "true" if completed is True else "false"
        headers["X-Hermes-Partial"] = "true" if partial else "false"

    return web.json_response(response_data, headers=headers)


async def write_sse_chat_completion(
    request: web.Request,
    completion_id: str,
    model: str,
    created: int,
    stream_q,
    agent_task,
    agent_ref=None,
    session_id: str = None,
    session_key: str = None,
    *,
    adapter: Any,
) -> web.StreamResponse:
    """Write real streaming SSE from agent's stream_delta_callback queue.

    If the client disconnects mid-stream (network drop, browser tab close),
    the agent is interrupted via ``agent.interrupt()`` so it stops making
    LLM API calls, and the asyncio task wrapper is cancelled.
    """
    sse_headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    # CORS middleware can't inject headers into StreamResponse after
    # prepare() flushes them, so resolve CORS headers up front.
    origin = request.headers.get("Origin", "")
    cors = adapter._cors_headers_for_origin(origin) if origin else None
    if cors:
        sse_headers.update(cors)
    if session_id:
        sse_headers["X-Hermes-Session-Id"] = session_id
    if session_key:
        sse_headers["X-Hermes-Session-Key"] = session_key
    response = web.StreamResponse(status=200, headers=sse_headers)
    await response.prepare(request)

    try:
        last_activity = time.monotonic()

        # Role chunk
        role_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
        last_activity = time.monotonic()

        # Helper -- route a queue item to the correct SSE event.
        async def _emit(item):
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "__tool_progress__":
                event_data = json.dumps(item[1])
                await response.write(
                    f"event: hermes.tool.progress\ndata: {event_data}\n\n".encode()
                )
            else:
                content_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                }
                await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
            return time.monotonic()

        # Stream content chunks as they arrive from the agent
        loop = asyncio.get_running_loop()
        while True:
            try:
                delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
            except _q.Empty:
                if agent_task.done():
                    # Drain any remaining items
                    while True:
                        try:
                            delta = stream_q.get_nowait()
                            if delta is None:
                                break
                            last_activity = await _emit(delta)
                        except _q.Empty:
                            break
                    break
                if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                    await response.write(b": keepalive\n\n")
                    last_activity = time.monotonic()
                continue

            if delta is None:  # End of stream sentinel
                break

            last_activity = await _emit(delta)

        # Get usage from completed agent
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        try:
            result, agent_usage = await agent_task
            usage = agent_usage or usage
        except Exception:
            pass

        # Finish chunk
        finish_chunk = {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }
        await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
        # Client disconnected mid-stream.  Interrupt the agent so it
        # stops making LLM API calls at the next loop iteration, then
        # cancel the asyncio task wrapper.
        agent = agent_ref[0] if agent_ref else None
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
        logger.info("SSE client disconnected; interrupted agent task %s", completion_id)

    return response
