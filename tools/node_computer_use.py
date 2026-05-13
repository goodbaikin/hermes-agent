"""
Node Computer Use Tool — Remote desktop control via node operations.

Provides computer_use functionality for remote nodes (Windows/Linux/macOS)
through the node_invoke API. In workspace replace mode, calls are automatically
routed to the active workspace's node.

Schema compatible with PR #20660 (computer_use_common).
"""

import json
from typing import Any, Dict

from tools.node_invoke import node_invoke


# Valid actions matching PR #20660 schema
VALID_ACTIONS = {
    "screenshot", "left_click", "right_click", "middle_click", "double_click",
    "mouse_move", "mouse_drag", "scroll", "type", "key",
    "cursor_position", "screen_size", "get_active_window", "wait",
}


def computer_use(
    action: str,
    x: int = None,
    y: int = None,
    x2: int = None,
    y2: int = None,
    text: str = None,
    keys: str = None,
    direction: str = None,
    amount: int = None,
    ms: int = None,
    region: list = None,
    redact_regions: list = None,
    node_id: str = None,
    **kwargs,
) -> str:
    """Execute a computer use action on a local or remote node.

    If node_id is provided (or resolved via workspace routing), the action
    is executed on that remote node via the node_invoke API.
    """
    if action not in VALID_ACTIONS:
        return json.dumps({"error": f"Unknown action: {action}"}, ensure_ascii=False)

    # Build params dict, omitting None values
    params: Dict[str, Any] = {"action": action}
    for key, val in [
        ("x", x), ("y", y), ("x2", x2), ("y2", y2),
        ("text", text), ("keys", keys),
        ("direction", direction), ("amount", amount), ("ms", ms),
        ("region", region), ("redact_regions", redact_regions),
    ]:
        if val is not None:
            params[key] = val

    # Remote execution via node
    if node_id:
        result_str = node_invoke(node_id, "computer.use", params)
        try:
            result = json.loads(result_str)
            if result.get("ok"):
                payload = result.get("payload", {})
                return json.dumps(payload, ensure_ascii=False)
            else:
                return json.dumps({
                    "success": False,
                    "error": result.get("error", {}).get("message", "Unknown error"),
                }, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

    # Local execution — not implemented yet (requires local computer_use backend)
    return json.dumps({
        "success": False,
        "error": "Local computer_use not available. Use workspace replace mode or specify node_id.",
    }, ensure_ascii=False)


# Tool registration
try:
    from tools.registry import registry

    registry.register(
        name="computer_use",
        toolset="node",
        schema={
            "name": "computer_use",
            "description": "Control the desktop (screenshot, click, type, keys) on the active workspace node. In replace mode, automatically routes to the workspace's node.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": list(VALID_ACTIONS),
                        "description": "Action to perform",
                    },
                    "x": {"type": "integer", "description": "X coordinate (absolute screen pixels)"},
                    "y": {"type": "integer", "description": "Y coordinate (absolute screen pixels)"},
                    "x2": {"type": "integer", "description": "Drag destination X"},
                    "y2": {"type": "integer", "description": "Drag destination Y"},
                    "text": {"type": "string", "description": "Text to type"},
                    "keys": {"type": "string", "description": "Key combination (e.g. 'Win+R', 'Ctrl+C')"},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "Scroll direction"},
                    "amount": {"type": "integer", "description": "Scroll amount (ticks)", "default": 3},
                    "ms": {"type": "integer", "description": "Wait duration in milliseconds"},
                    "region": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Screenshot crop region [x1, y1, x2, y2]",
                    },
                    "redact_regions": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "integer"}},
                        "description": "Regions to black out [[x1,y1,x2,y2], ...]",
                    },
                    "node_id": {"type": "string", "description": "Target node ID (auto-set in replace mode)"},
                },
                "required": ["action"],
            },
        },
        handler=lambda args, **kw: computer_use(**args),
    )
except ImportError:
    pass
