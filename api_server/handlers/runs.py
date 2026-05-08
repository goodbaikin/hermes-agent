import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from aiohttp import web

from api_server.utils import _openai_error


logger = logging.getLogger(__name__)

_MAX_CONCURRENT_RUNS = 10
_RUN_STREAM_TTL = 300
_RUN_STATUS_TTL = 3600


async def handle_runs(
    request: web.Request,
    *,
    adapter: Any,
) -> web.Response:
    """POST /v1/runs -- start an agent run, return run_id immediately."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    # Enforce concurrency limit
    if len(adapter._run_streams) >= _MAX_CONCURRENT_RUNS:
        return web.json_response(
            _openai_error(f"Too many concurrent runs (max {_MAX_CONCURRENT_RUNS})", code="rate_limit_exceeded"),
            status=429,
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response(_openai_error("Invalid JSON"), status=400)

    raw_input = body.get("input")
    if not raw_input:
        return web.json_response(_openai_error("Missing 'input' field"), status=400)

    user_message = raw_input if isinstance(raw_input, str) else (raw_input[-1].get("content", "") if isinstance(raw_input, list) else "")
    if not user_message:
        return web.json_response(_openai_error("No user message found in input"), status=400)

    instructions = body.get("instructions")
    previous_response_id = body.get("previous_response_id")

    # Accept explicit conversation_history from the request body.
    conversation_history: List[Dict[str, str]] = []
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
            conversation_history.append({"role": str(entry["role"]), "content": str(entry["content"])})
        if previous_response_id:
            logger.debug("Both conversation_history and previous_response_id provided; using conversation_history")

    stored_session_id = None
    if not conversation_history and previous_response_id:
        stored = adapter._response_store.get(previous_response_id)
        if stored:
            conversation_history = list(stored.get("conversation_history", []))
            stored_session_id = stored.get("session_id")
            if instructions is None:
                instructions = stored.get("instructions")

    # When input is a multi-message array, extract all but the last
    # message as conversation history (the last becomes user_message).
    # Only fires when no explicit history was provided.
    if not conversation_history and isinstance(raw_input, list) and len(raw_input) > 1:
        for msg in raw_input[:-1]:
            if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                content = msg["content"]
                if isinstance(content, list):
                    # Flatten multi-part content blocks to text
                    content = " ".join(
                        part.get("text", "") for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                conversation_history.append({"role": msg["role"], "content": str(content)})

    run_id = f"run_{uuid.uuid4().hex}"
    session_id = body.get("session_id") or stored_session_id or run_id
    ephemeral_system_prompt = instructions
    loop = asyncio.get_running_loop()
    q: "asyncio.Queue[Optional[Dict]]" = asyncio.Queue()
    created_at = time.time()
    adapter._run_streams[run_id] = q
    adapter._run_streams_created[run_id] = created_at

    event_cb = adapter._make_run_event_callback(run_id, loop)

    # Also wire stream_delta_callback so message.delta events flow through.
    def _text_cb(delta: Optional[str]) -> None:
        if delta is None:
            return
        try:
            loop.call_soon_threadsafe(q.put_nowait, {
                "event": "message.delta",
                "run_id": run_id,
                "timestamp": time.time(),
                "delta": delta,
            })
        except Exception:
            pass

    _set_run_status(
        adapter,
        run_id,
        "queued",
        created_at=created_at,
        session_id=session_id,
        model=body.get("model", adapter._model_name),
    )

    async def _run_and_close():
        try:
            _set_run_status(adapter, run_id, "running")
            agent = adapter._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                stream_delta_callback=_text_cb,
                tool_progress_callback=event_cb,
            )
            adapter._active_run_agents[run_id] = agent
            def _run_sync():
                effective_task_id = session_id or run_id
                r = agent.run_conversation(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    task_id=effective_task_id,
                )
                u = {
                    "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                    "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                    "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
                }
                return r, u

            result, usage = await asyncio.get_running_loop().run_in_executor(None, _run_sync)
            # Check for structured failure (non-retryable client errors like
            # 401/400 return failed=True instead of raising, so the except
            # block below never fires -- issue #15561).
            if isinstance(result, dict) and result.get("failed"):
                error_msg = result.get("error") or "agent run failed"
                q.put_nowait({
                    "event": "run.failed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "error": error_msg,
                })
                _set_run_status(
                    adapter,
                    run_id,
                    "failed",
                    error=error_msg,
                    last_event="run.failed",
                )
            else:
                final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                q.put_nowait({
                    "event": "run.completed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "output": final_response,
                    "usage": usage,
                })
                _set_run_status(
                    adapter,
                    run_id,
                    "completed",
                    output=final_response,
                    usage=usage,
                    last_event="run.completed",
                )
        except asyncio.CancelledError:
            _set_run_status(
                adapter,
                run_id,
                "cancelled",
                last_event="run.cancelled",
            )
            try:
                q.put_nowait({
                    "event": "run.cancelled",
                    "run_id": run_id,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
            raise
        except Exception as exc:
            logger.exception("[api_server] run %s failed", run_id)
            _set_run_status(
                adapter,
                run_id,
                "failed",
                error=str(exc),
                last_event="run.failed",
            )
            try:
                q.put_nowait({
                    "event": "run.failed",
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "error": str(exc),
                })
            except Exception:
                pass
        finally:
            # Sentinel: signal SSE stream to close
            try:
                q.put_nowait(None)
            except Exception:
                pass
            adapter._active_run_agents.pop(run_id, None)
            adapter._active_run_tasks.pop(run_id, None)

    task = asyncio.create_task(_run_and_close())
    adapter._active_run_tasks[run_id] = task
    try:
        adapter._background_tasks.add(task)
    except TypeError:
        pass
    if hasattr(task, "add_done_callback"):
        task.add_done_callback(adapter._background_tasks.discard)

    return web.json_response({"run_id": run_id, "status": "started"}, status=202)


async def handle_get_run(
    request: web.Request,
    *,
    adapter: Any,
) -> web.Response:
    """GET /v1/runs/{run_id} -- return pollable run status for external UIs."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    run_id = request.match_info["run_id"]
    status = adapter._run_statuses.get(run_id)
    if status is None:
        return web.json_response(
            _openai_error(f"Run not found: {run_id}", code="run_not_found"),
            status=404,
        )
    return web.json_response(status)


async def handle_run_events(
    request: web.Request,
    *,
    adapter: Any,
) -> web.StreamResponse:
    """GET /v1/runs/{run_id}/events -- SSE stream of structured agent lifecycle events."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    run_id = request.match_info["run_id"]

    # Allow subscribing slightly before the run is registered (race condition window)
    for _ in range(20):
        if run_id in adapter._run_streams:
            break
        await asyncio.sleep(0.05)
    else:
        return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

    q = adapter._run_streams[run_id]

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")
                continue
            if event is None:
                # Run finished -- send final SSE comment and close
                await response.write(b": stream closed\n\n")
                break
            payload = f"data: {json.dumps(event)}\n\n"
            await response.write(payload.encode())
    except Exception as exc:
        logger.debug("[api_server] SSE stream error for run %s: %s", run_id, exc)
    finally:
        adapter._run_streams.pop(run_id, None)
        adapter._run_streams_created.pop(run_id, None)

    return response


async def handle_stop_run(
    request: web.Request,
    *,
    adapter: Any,
) -> web.Response:
    """POST /v1/runs/{run_id}/stop -- interrupt a running agent."""
    auth_err = adapter._check_auth(request)
    if auth_err:
        return auth_err

    run_id = request.match_info["run_id"]
    agent = adapter._active_run_agents.get(run_id)
    task = adapter._active_run_tasks.get(run_id)

    if agent is None and task is None:
        return web.json_response(_openai_error(f"Run not found: {run_id}", code="run_not_found"), status=404)

    _set_run_status(adapter, run_id, "stopping", last_event="run.stopping")

    if agent is not None:
        try:
            agent.interrupt("Stop requested via API")
        except Exception:
            pass

    if task is not None and not task.done():
        task.cancel()
        # Bounded wait: run_conversation() executes in the default
        # executor thread which task.cancel() cannot preempt -- we rely on
        # agent.interrupt() above to break the loop. Cap the wait so a
        # slow/unresponsive interrupt can't hang this handler.
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                "[api_server] stop for run %s timed out after 5s; "
                "agent may still be finishing the current step",
                run_id,
            )
        except (asyncio.CancelledError, Exception):
            pass

    return web.json_response({"run_id": run_id, "status": "stopping"})


async def sweep_orphaned_runs(
    adapter: Any,
) -> None:
    """Periodically clean up run streams that were never consumed."""
    while True:
        await asyncio.sleep(60)
        now = time.time()
        stale = [
            run_id
            for run_id, created_at in list(adapter._run_streams_created.items())
            if now - created_at > _RUN_STREAM_TTL
        ]
        for run_id in stale:
            logger.debug("[api_server] sweeping orphaned run %s", run_id)
            adapter._run_streams.pop(run_id, None)
            adapter._run_streams_created.pop(run_id, None)
            adapter._active_run_agents.pop(run_id, None)
            adapter._active_run_tasks.pop(run_id, None)

        stale_statuses = [
            run_id
            for run_id, status in list(adapter._run_statuses.items())
            if status.get("status") in {"completed", "failed", "cancelled"}
            and now - float(status.get("updated_at", 0) or 0) > _RUN_STATUS_TTL
        ]
        for run_id in stale_statuses:
            adapter._run_statuses.pop(run_id, None)


def _set_run_status(
    adapter: Any,
    run_id: str,
    status: str,
    **fields: Any,
) -> Dict[str, Any]:
    """Update pollable run status without exposing private agent objects."""
    now = time.time()
    current = adapter._run_statuses.get(run_id, {})
    current.update({
        "object": "hermes.run",
        "run_id": run_id,
        "status": status,
        "updated_at": now,
    })
    current.setdefault("created_at", fields.pop("created_at", now))
    current.update(fields)
    adapter._run_statuses[run_id] = current
    return current
