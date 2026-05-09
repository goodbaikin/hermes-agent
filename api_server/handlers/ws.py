"""WebSocket handlers for the API Server."""

import asyncio
import json
import logging
from typing import Any, Dict

from aiohttp import web

from api_server.node_registry import NODE_REGISTRY, NodeSession

logger = logging.getLogger(__name__)


async def handle_ws_real(request: web.Request) -> web.WebSocketResponse:
    """
    WebSocket endpoint for remote node connections.

    OpenClaw-style protocol:
    1. Node sends {type:"req", method:"connect", params:{role:"node", ...}}
    2. Gateway responds {type:"res", ok:true, payload:{type:"hello-ok", ...}}
    3. Gateway sends {type:"event", event:"node.invoke.request", payload:{...}}
    4. Node responds {type:"event", event:"node.invoke.result", payload:{...}}
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    node_session = None
    node_id = None

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
                        "error": {"message": "Only 'node' role is supported on /ws"},
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

    except Exception as exc:
        logger.warning("[Node WS] Connection error for %s: %s", node_id, exc)
    finally:
        if node_id:
            await NODE_REGISTRY.unregister(node_id)
        if not ws.closed:
            await ws.close()

    return ws
