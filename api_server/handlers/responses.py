import asyncio
import json
import logging
import queue as _q
import time
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import web

from api_server.utils import (
    _openai_error,
    _normalize_multimodal_content,
    _content_has_visible_payload,
    _multimodal_validation_error,
    _make_request_fingerprint,
)


logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS = 30.0


def _parse_bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


async def handle_responses(
    request: web.Request,
    *,
    adapter: Any,
    idem_cache: Any,
) -> web.Response:
    """POST /v1/responses -- OpenAI Responses API format."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    session_key, err = adapter._parse_session_key_header(request)
    if err:
        return err

    # Parse request body
    try:
        body = await request.json()
    except (json.JSONDecodeError, Exception):
        return web.json_response(
            {"error": {"message": "Invalid JSON in request body", "type": "invalid_request_error"}},
            status=400,
        )

    raw_input = body.get("input")
    if raw_input is None:
        return web.json_response(_openai_error("Missing 'input' field"), status=400)

    instructions = body.get("instructions")
    previous_response_id = body.get("previous_response_id")
    conversation = body.get("conversation")
    store = _parse_bool_flag(body.get("store"), default=True)

    # conversation and previous_response_id are mutually exclusive
    if conversation and previous_response_id:
        return web.json_response(_openai_error("Cannot use both 'conversation' and 'previous_response_id'"), status=400)

    # Resolve conversation name to latest response_id
    if conversation:
        previous_response_id = adapter._response_store.get_conversation(conversation)

    # Normalize input to message list
    input_messages: List[Dict[str, Any]] = []
    if isinstance(raw_input, str):
        input_messages = [{"role": "user", "content": raw_input}]
    elif isinstance(raw_input, list):
        for idx, item in enumerate(raw_input):
            if isinstance(item, str):
                input_messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                try:
                    content = _normalize_multimodal_content(item.get("content", ""))
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"input[{idx}].content")
                input_messages.append({"role": role, "content": content})
    else:
        return web.json_response(_openai_error("'input' must be a string or array"), status=400)

    # Accept explicit conversation_history from the request body.
    conversation_history: List[Dict[str, Any]] = []
    raw_history = body.get("conversation_history")
    if raw_history:
        if not isinstance(raw_history, list):
            return web.json_response(
                _openai_error("'conversation_history' must be an array of message objects"),
                status=400,
            )
        for i, entry in enumerate(raw_history):
            if not isinstance(entry, dict) or "role" not in entry or "content" not in entry:
                return web.json_response(
                    _openai_error(f"conversation_history[{i}] must have 'role' and 'content' fields"),
                    status=400,
                )
            try:
                entry_content = _normalize_multimodal_content(entry["content"])
            except ValueError as exc:
                return _multimodal_validation_error(exc, param=f"conversation_history[{i}].content")
            conversation_history.append({"role": str(entry["role"]), "content": entry_content})
        if previous_response_id:
            logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

    stored = None
    stored_session_id = None
    if not conversation_history and previous_response_id:
        stored = adapter._response_store.get(previous_response_id)
        if stored is None:
            return web.json_response(_openai_error(f"Previous response not found: {previous_response_id}"), status=404)
        conversation_history = list(stored.get("conversation_history", []))
        stored_session_id = stored.get("session_id")
        if instructions is None:
            instructions = stored.get("instructions")

    # Append new input messages to history (all but the last become history)
    for msg in input_messages[:-1]:
        conversation_history.append(msg)

    # Last input message is the user_message
    user_message: Any = input_messages[-1].get("content", "") if input_messages else ""
    if not _content_has_visible_payload(user_message):
        return web.json_response(_openai_error("No user message found in input"), status=400)

    # Truncation support
    if body.get("truncation") == "auto" and len(conversation_history) > 100:
        conversation_history = conversation_history[-100:]

    # Reuse session from previous_response_id chain
    session_id = stored_session_id or str(uuid.uuid4())

    stream = _parse_bool_flag(body.get("stream"), default=False)
    if stream:
        _stream_q: _q.Queue = _q.Queue()

        def _on_delta(delta):
            if delta is not None and delta != "":
                _stream_q.put(delta)

        def _on_tool_progress(event_type, name, preview, args, **kwargs):
            if event_type == "tool.started":
                _stream_q.put(("__tool_started__", {
                    "tool_call_id": kwargs.get("tool_call_id", ""),
                    "name": name,
                    "arguments": args or {},
                }))
            elif event_type == "tool.completed":
                _stream_q.put(("__tool_completed__", {
                    "tool_call_id": kwargs.get("tool_call_id", ""),
                    "name": name,
                    "arguments": args or {},
                    "result_preview": preview,
                }))

        def _on_tool_start(tool_call_id, function_name, function_args):
            _stream_q.put(("__tool_started__", {
                "tool_call_id": tool_call_id,
                "name": function_name,
                "arguments": function_args or {},
            }))

        def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
            result_preview = function_result[:4000] if isinstance(function_result, str) else json.dumps(function_result, ensure_ascii=False)[:4000]
            _stream_q.put(("__tool_completed__", {
                "tool_call_id": tool_call_id,
                "name": function_name,
                "arguments": function_args or {},
                "result": function_result,
                "result_preview": result_preview,
            }))

        agent_ref = [None]
        agent_task = asyncio.ensure_future(adapter._run_agent(
            user_message=user_message,
            conversation_history=conversation_history,
            ephemeral_system_prompt=instructions,
            session_id=session_id,
            gateway_session_key=session_key,
            stream_delta_callback=_on_delta,
            tool_progress_callback=_on_tool_progress,
            tool_start_callback=_on_tool_start,
            tool_complete_callback=_on_tool_complete,
            agent_ref=agent_ref,
        ))
        agent_task.add_done_callback(lambda _task: _stream_q.put(None))

        response_id = f"resp_{uuid.uuid4().hex[:28]}"
        model_name = body.get("model", adapter._model_name)
        created_at = int(time.time())

        sse_response = await write_sse_responses(
            request=request,
            response_id=response_id,
            model=model_name,
            created_at=created_at,
            stream_q=_stream_q,
            agent_task=agent_task,
            agent_ref=agent_ref,
            conversation_history=conversation_history,
            user_message=user_message,
            instructions=instructions,
            conversation=conversation,
            store=store,
            session_id=session_id,
            adapter=adapter,
            session_key=session_key,
            previous_response_id=previous_response_id,
        )
        return sse_response

    async def _compute_response():
        return await adapter._run_agent(
            user_message=user_message,
            conversation_history=conversation_history,
            ephemeral_system_prompt=instructions,
            session_id=session_id,
            gateway_session_key=session_key,
        )

    idempotency_key = request.headers.get("Idempotency-Key")
    if idempotency_key:
        fp = _make_request_fingerprint(
            body,
            keys=["input", "instructions", "previous_response_id", "conversation", "model", "tools"],
        )
        try:
            result, usage = await idem_cache.get_or_set(idempotency_key, fp, _compute_response)
        except Exception as e:
            logger.error("Error running agent for responses: %s", e, exc_info=True)
            return web.json_response(
                _openai_error(f"Internal server error: {e}", err_type="server_error"),
                status=500,
            )
    else:
        try:
            result, usage = await _compute_response()
        except Exception as e:
            logger.error("Error running agent for responses: %s", e, exc_info=True)
            return web.json_response(
                _openai_error(f"Internal server error: {e}", err_type="server_error"),
                status=500,
            )

    final_response = result.get("final_response", "")
    if not final_response:
        final_response = result.get("error", "(No response generated)")

    response_id = f"resp_{uuid.uuid4().hex[:28]}"
    created_at = int(time.time())

    # Build the full conversation history for storage
    # If the agent returned messages, use those directly (they contain the full history)
    agent_messages = result.get("messages", [])
    if agent_messages:
        full_history = list(agent_messages)
    else:
        full_history = list(conversation_history)
        full_history.append({"role": "user", "content": user_message})
        full_history.append({"role": "assistant", "content": final_response})

    # Build output items (includes tool calls + final message)
    # When using previous_response_id, only include messages from the current turn
    if previous_response_id and conversation_history:
        # Find where the new turn starts (after the previous conversation history)
        prev_len = len(stored.get("conversation_history", [])) if stored else 0
        current_messages = result.get("messages", [])[prev_len:]
        # Create a temporary result with only current turn messages
        temp_result = dict(result)
        temp_result["messages"] = current_messages
        output_items = adapter._extract_output_items(temp_result)
    else:
        output_items = adapter._extract_output_items(result)

    response_data = {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "created_at": created_at,
        "model": body.get("model", adapter._model_name),
        "output": output_items,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }

    # Store the complete response object for future chaining / GET retrieval
    if store:
        adapter._response_store.put(response_id, {
            "response": response_data,
            "conversation_history": full_history,
            "instructions": instructions,
            "session_id": session_id,
        })
        if conversation:
            adapter._response_store.set_conversation(conversation, response_id)

    headers = {}
    if session_key:
        headers["X-Hermes-Session-Key"] = session_key
    return web.json_response(response_data, headers=headers)


async def write_sse_responses(
    request: web.Request,
    response_id: str,
    model: str,
    created_at: int,
    stream_q,
    agent_task,
    agent_ref,
    conversation_history: List[Dict[str, str]],
    user_message: str,
    instructions: Optional[str],
    conversation: Optional[str],
    store: bool,
    session_id: str,
    *,
    adapter: Any,
    session_key: Optional[str] = None,
    previous_response_id: Optional[str] = None,
) -> web.StreamResponse:
    """Write an SSE stream for POST /v1/responses (OpenAI Responses API)."""
    sse_headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    origin = request.headers.get("Origin", "")
    cors = adapter._cors_headers_for_origin(origin) if origin else None
    if cors:
        sse_headers.update(cors)
    if session_id:
        sse_headers["X-Hermes-Session-Id"] = session_id
    response = web.StreamResponse(status=200, headers=sse_headers)
    await response.prepare(request)

    # State accumulated during the stream
    final_text_parts: List[str] = []
    pending_tool_calls: List[Dict[str, Any]] = []
    emitted_items: List[Dict[str, Any]] = []
    output_index = 0
    call_counter = 0
    sequence_number = 0
    message_item_id = f"msg_{uuid.uuid4().hex[:24]}"
    message_output_index: Optional[int] = None
    message_opened = False

    async def _write_event(event_type: str, data: Dict[str, Any]) -> None:
        nonlocal sequence_number
        if "sequence_number" not in data:
            data["sequence_number"] = sequence_number
        sequence_number += 1
        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        await response.write(payload.encode())

    def _envelope(status: str) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "id": response_id,
            "object": "response",
            "status": status,
            "created_at": created_at,
            "model": model,
        }
        return env

    final_response_text = ""
    agent_error: Optional[str] = None
    usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    terminal_snapshot_persisted = False

    def _persist_response_snapshot(
        response_env: Dict[str, Any],
        *,
        conversation_history_snapshot: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not store:
            return
        if conversation_history_snapshot is None:
            conversation_history_snapshot = list(conversation_history)
            conversation_history_snapshot.append({"role": "user", "content": user_message})
        adapter._response_store.put(response_id, {
            "response": response_env,
            "conversation_history": conversation_history_snapshot,
            "instructions": instructions,
            "session_id": session_id,
        })
        if conversation:
            adapter._response_store.set_conversation(conversation, response_id)

    def _persist_incomplete_if_needed() -> None:
        if not store or terminal_snapshot_persisted:
            return
        incomplete_text = "".join(final_text_parts) or final_response_text
        incomplete_items: List[Dict[str, Any]] = list(emitted_items)
        if incomplete_text:
            incomplete_items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": incomplete_text}],
            })
        incomplete_env = _envelope("incomplete")
        incomplete_env["output"] = incomplete_items
        incomplete_env["usage"] = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        incomplete_history = list(conversation_history)
        incomplete_history.append({"role": "user", "content": user_message})
        if incomplete_text:
            incomplete_history.append({"role": "assistant", "content": incomplete_text})
        _persist_response_snapshot(
            incomplete_env,
            conversation_history_snapshot=incomplete_history,
        )

    try:
        # response.created -- initial envelope, status=in_progress
        created_env = _envelope("in_progress")
        created_env["output"] = []
        await _write_event("response.created", {
            "type": "response.created",
            "response": created_env,
        })
        _persist_response_snapshot(created_env)
        last_activity = time.monotonic()

        async def _open_message_item() -> None:
            nonlocal message_opened, message_output_index, output_index
            if message_opened:
                return
            message_opened = True
            message_output_index = output_index
            output_index += 1
            item = {
                "id": message_item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            await _write_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": message_output_index,
                "item": item,
            })

        async def _emit_text_delta(delta_text: str) -> None:
            await _open_message_item()
            final_text_parts.append(delta_text)
            await _write_event("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": message_item_id,
                "output_index": message_output_index,
                "content_index": 0,
                "delta": delta_text,
                "logprobs": [],
            })

        async def _emit_tool_started(payload: Dict[str, Any]) -> str:
            nonlocal output_index, call_counter
            call_counter += 1
            call_id = payload.get("tool_call_id") or f"call_{response_id[5:]}_{call_counter}"
            args = payload.get("arguments", {})
            if isinstance(args, dict):
                arguments_str = json.dumps(args)
            else:
                arguments_str = str(args)
            item = {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "status": "in_progress",
                "name": payload.get("name", ""),
                "call_id": call_id,
                "arguments": arguments_str,
            }
            idx = output_index
            output_index += 1
            pending_tool_calls.append({
                "call_id": call_id,
                "name": payload.get("name", ""),
                "arguments": arguments_str,
                "item_id": item["id"],
                "output_index": idx,
            })
            emitted_items.append({
                "type": "function_call",
                "name": payload.get("name", ""),
                "arguments": arguments_str,
                "call_id": call_id,
            })
            await _write_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": item,
            })
            return call_id

        async def _emit_tool_completed(payload: Dict[str, Any]) -> None:
            nonlocal output_index
            call_id = payload.get("tool_call_id")
            result = payload.get("result", "")
            pending = None
            if call_id:
                for i, p in enumerate(pending_tool_calls):
                    if p["call_id"] == call_id:
                        pending = pending_tool_calls.pop(i)
                        break
            if pending is None:
                return

            # function_call done
            done_item = {
                "id": pending["item_id"],
                "type": "function_call",
                "status": "completed",
                "name": pending["name"],
                "call_id": pending["call_id"],
                "arguments": pending["arguments"],
            }
            await _write_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": pending["output_index"],
                "item": done_item,
            })

            # function_call_output added (result)
            result_str = result if isinstance(result, str) else json.dumps(result)
            output_parts = [{"type": "input_text", "text": result_str}]
            output_item = {
                "id": f"fco_{uuid.uuid4().hex[:24]}",
                "type": "function_call_output",
                "call_id": pending["call_id"],
                "output": output_parts,
                "status": "completed",
            }
            idx = output_index
            output_index += 1
            emitted_items.append({
                "type": "function_call_output",
                "call_id": pending["call_id"],
                "output": output_parts,
            })
            await _write_event("response.output_item.added", {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": output_item,
            })
            await _write_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": idx,
                "item": output_item,
            })

        # Main drain loop -- thread-safe queue fed by agent callbacks.
        async def _dispatch(it) -> None:
            if isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str):
                tag, payload = it
                if tag == "__tool_started__":
                    await _emit_tool_started(payload)
                elif tag == "__tool_completed__":
                    await _emit_tool_completed(payload)
            elif isinstance(it, str):
                await _emit_text_delta(it)

        loop = asyncio.get_running_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
            except _q.Empty:
                if agent_task.done():
                    # Drain remaining
                    while True:
                        try:
                            item = stream_q.get_nowait()
                            if item is None:
                                break
                            await _dispatch(item)
                            last_activity = time.monotonic()
                        except _q.Empty:
                            break
                    break
                if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                    await response.write(b": keepalive\n\n")
                    last_activity = time.monotonic()
                continue

            if item is None:  # EOS sentinel
                break

            await _dispatch(item)
            last_activity = time.monotonic()

        # Pick up agent result + usage from the completed task
        try:
            result, agent_usage = await agent_task
            usage = agent_usage or usage
            agent_final = result.get("final_response", "") if isinstance(result, dict) else ""
            if agent_final and not final_text_parts:
                await _emit_text_delta(agent_final)
            if agent_final and not final_response_text:
                final_response_text = agent_final
            if isinstance(result, dict) and result.get("error") and not final_response_text:
                agent_error = result["error"]
        except Exception as e:  # noqa: BLE001
            logger.error("Error running agent for streaming responses: %s", e, exc_info=True)
            agent_error = str(e)

        # Close the message item if it was opened
        final_response_text = "".join(final_text_parts) or final_response_text
        if message_opened:
            await _write_event("response.output_text.done", {
                "type": "response.output_text.done",
                "item_id": message_item_id,
                "output_index": message_output_index,
                "content_index": 0,
                "text": final_response_text,
                "logprobs": [],
            })
            msg_done_item = {
                "id": message_item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": final_response_text}
                ],
            }
            await _write_event("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": message_output_index,
                "item": msg_done_item,
            })

        # Always append a final message item in the completed response envelope
        final_items: List[Dict[str, Any]] = list(emitted_items)
        final_items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": final_response_text or (agent_error or "")}
            ],
        })

        if agent_error:
            failed_env = _envelope("failed")
            failed_env["output"] = final_items
            failed_env["error"] = {"message": agent_error, "type": "server_error"}
            failed_env["usage"] = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            _failed_history = list(conversation_history)
            _failed_history.append({"role": "user", "content": user_message})
            if final_response_text or agent_error:
                _failed_history.append({
                    "role": "assistant",
                    "content": final_response_text or agent_error,
                })
            _persist_response_snapshot(
                failed_env,
                conversation_history_snapshot=_failed_history,
            )
            terminal_snapshot_persisted = True
            await _write_event("response.failed", {
                "type": "response.failed",
                "response": failed_env,
            })
        else:
            completed_env = _envelope("completed")
            completed_env["output"] = final_items
            completed_env["usage"] = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            full_history = list(conversation_history)
            full_history.append({"role": "user", "content": user_message})
            if isinstance(result, dict) and result.get("messages"):
                # When using previous_response_id, the agent returns full history
                # We should use the agent's messages directly as they contain the complete transcript
                if previous_response_id:
                    # Replace with the agent's full messages (they contain the complete history)
                    full_history = list(result["messages"])
                else:
                    full_history.extend(result["messages"])
            else:
                full_history.append({"role": "assistant", "content": final_response_text})
            _persist_response_snapshot(
                completed_env,
                conversation_history_snapshot=full_history,
            )
            terminal_snapshot_persisted = True
            await _write_event("response.completed", {
                "type": "response.completed",
                "response": completed_env,
            })

    except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
        _persist_incomplete_if_needed()
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
        logger.info("SSE client disconnected; interrupted agent task %s", response_id)
    except asyncio.CancelledError:
        _persist_incomplete_if_needed()
        agent = agent_ref[0] if agent_ref else None
        if agent is not None:
            try:
                agent.interrupt("SSE task cancelled")
            except Exception:
                pass
        if not agent_task.done():
            agent_task.cancel()
        logger.info("SSE task cancelled; persisted incomplete snapshot for %s", response_id)
        raise

    # Echo X-Hermes-Session-Key header if present
    if session_key:
        response.headers["X-Hermes-Session-Key"] = session_key
    return response


async def handle_get_response(request: web.Request, *, check_auth, response_store) -> web.Response:
    """GET /v1/responses/{response_id} -- retrieve a stored response."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    response_id = request.match_info["response_id"]
    stored = response_store.get(response_id)
    if stored is None:
        return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

    return web.json_response(stored["response"])


async def handle_delete_response(request: web.Request, *, check_auth, response_store) -> web.Response:
    """DELETE /v1/responses/{response_id} -- delete a stored response."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err

    response_id = request.match_info["response_id"]
    deleted = response_store.delete(response_id)
    if not deleted:
        return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

    return web.json_response({
        "id": response_id,
        "object": "response",
        "deleted": True,
    })
