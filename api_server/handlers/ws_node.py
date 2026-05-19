"""WebSocket handler for node_client registration at /ws/node."""

import asyncio
import hmac
import json
import logging
from typing import Any, Dict

from aiohttp import web

from api_server.node_registry import NODE_REGISTRY, NodeSession

logger = logging.getLogger(__name__)


async def handle_ws_node(request: web.Request, api_key: str = "") -> web.WebSocketResponse:
    """
    GET /ws/node -- WebSocket endpoint for node_client registration.

    Protocol (same as /ws):
    1. Node sends {type:"req", method:"connect", params:{role:"node", ...}}
    2. Server responds {type:"res", ok:true, payload:{type:"hello-ok", ...}}
    3. Server sends {type:"event", event:"node.invoke.request", payload:{...}}
    4. Node responds {type:"event", event:"node.invoke.result", payload:{...}}

    Authentication:
    - Query param: ?token=xxx
    - Or Authorization header: Bearer xxx
    - Or connect message auth.token (legacy fallback)
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    node_session = None
    node_id = None

    # --- Auth check ---
    auth_ok = False
    if api_key:
        # Check query param ?token=xxx
        query_token = request.query.get("token", "")
        if query_token and hmac.compare_digest(query_token, api_key):
            auth_ok = True
        else:
            # Check Authorization header
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                header_token = auth_header[7:].strip()
                if hmac.compare_digest(header_token, api_key):
                    auth_ok = True
    else:
        # No API key configured -- allow all (local development only)
        auth_ok = True

    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue

            try:
                data = json.loads(msg.data)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type")
            event = data.get("event")

            # --- Handshake ---
            if msg_type == "req" and data.get("method") == "connect":
                params = data.get("params", {})
                role = params.get("role", "")
                if role != "node":
                    await ws.send_str(json.dumps({
                        "type": "res",
                        "id": data.get("id"),
                        "ok": False,
                        "error": {"message": "Only 'node' role is supported on /ws/node"},
                    }))
                    await ws.close()
                    return ws

                # If not yet authenticated, check connect message auth.token
                if not auth_ok and api_key:
                    connect_token = params.get("auth", {}).get("token", "")
                    if connect_token and hmac.compare_digest(connect_token, api_key):
                        auth_ok = True

                if not auth_ok:
                    await ws.send_str(json.dumps({
                        "type": "res",
                        "id": data.get("id"),
                        "ok": False,
                        "error": {"message": "Authentication failed", "code": "auth_failed"},
                    }))
                    await ws.close()
                    return ws

                node_id = params.get("client", {}).get("id", "unknown")
                caps = params.get("caps", [])
                commands = params.get("commands", [])
                platform = params.get("client", {}).get("platform", "unknown")
                version = params.get("client", {}).get("version", "unknown")

                def send_fn(payload: Dict[str, Any]) -> None:
                    asyncio.create_task(ws.send_str(json.dumps(payload)))

                node_session = NodeSession(
                    node_id=node_id,
                    send_fn=send_fn,
                    caps=caps,
                    commands=commands,
                    platform=platform,
                    version=version,
                )
                await NODE_REGISTRY.register(node_session)

                await ws.send_str(json.dumps({
                    "type": "res",
                    "id": data.get("id"),
                    "ok": True,
                    "payload": {
                        "type": "hello-ok",
                        "protocol": 1,
                        "policy": {
                            "maxPayload": 26214400,
                            "tickIntervalMs": 15000,
                        },
                    },
                }))
                continue

            # --- Invoke result from node ---
            if msg_type == "event" and event == "node.invoke.result":
                payload = data.get("payload", {})
                request_id = payload.get("id")
                ok = payload.get("ok", False)
                result_payload = payload.get("payload")
                error = payload.get("error")
                NODE_REGISTRY.handle_result(request_id, ok, result_payload, error)
                continue

            # --- Terminal output from node ---
            if msg_type == "event" and event == "node.terminal.output":
                payload = data.get("payload", {})
                proxy_id = payload.get("proxyId")
                data_b64 = payload.get("data")
                if proxy_id and data_b64:
                    from api_server.handlers.terminal_proxy import TERMINAL_PROXY_MANAGER
                    await TERMINAL_PROXY_MANAGER.handle_node_output(proxy_id, data_b64)
                continue

            if msg_type == "event" and event == "node.terminal.close":
                payload = data.get("payload", {})
                proxy_id = payload.get("proxyId")
                if proxy_id:
                    from api_server.handlers.terminal_proxy import TERMINAL_PROXY_MANAGER
                    await TERMINAL_PROXY_MANAGER.handle_node_close(proxy_id)
                continue

            if msg_type == "event" and event == "node.terminal.error":
                payload = data.get("payload", {})
                proxy_id = payload.get("proxyId")
                error = payload.get("error", "Unknown terminal error")
                if proxy_id:
                    from api_server.handlers.terminal_proxy import TERMINAL_PROXY_MANAGER
                    await TERMINAL_PROXY_MANAGER.handle_node_error(proxy_id, error)
                continue

    except Exception as exc:
        logger.warning("[Node WS /ws/node] Connection error for %s: %s", node_id, exc)
    finally:
        if node_id:
            await NODE_REGISTRY.unregister(node_id)
        if not ws.closed:
            await ws.close()

    return ws
