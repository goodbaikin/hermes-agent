import json
from unittest.mock import patch

from agent import workspace_router


def test_route_tool_call_patch_v4a_uses_node_file_patch():
    params = {
        "mode": "patch",
        "patch": "*** Begin Patch\n*** Add File: foo.txt\n+hello\n*** End Patch",
    }

    with patch("agent.workspace_router.should_route_to_node", return_value="node-1"), patch(
        "tools.node_lib.node_patch_v4a", return_value={"payload": {"success": True}}
    ) as mock_node_patch_v4a:
        result_json = workspace_router.route_tool_call("patch", params)

    assert result_json is not None
    payload = json.loads(result_json)
    assert payload == {"success": True}
    mock_node_patch_v4a.assert_called_once_with("node-1", params["patch"], base_dir=None)


def test_route_tool_call_patch_v4a_forwards_workspace_base_dir():
    params = {
        "mode": "patch",
        "patch": "*** Begin Patch\n*** Add File: foo.txt\n+hello\n*** End Patch",
    }

    with patch("agent.workspace_router.should_route_to_node", return_value="node-1"), patch(
        "agent.workspace_router._get_workspace_workdir", return_value="/tmp/workspace"
    ), patch(
        "tools.node_lib.node_patch_v4a", return_value={"payload": {"success": True}}
    ) as mock_node_patch_v4a:
        workspace_router.route_tool_call("patch", params)

    mock_node_patch_v4a.assert_called_once_with(
        "node-1",
        params["patch"],
        base_dir="/tmp/workspace",
    )


def test_route_tool_call_patch_v4a_missing_payload_returns_error():
    with patch("agent.workspace_router.should_route_to_node", return_value="node-1"):
        result_json = workspace_router.route_tool_call("patch", {
            "mode": "patch",
        })

    assert result_json is not None
    payload = json.loads(result_json)
    assert payload == {"error": "Patch content must be a string"}


def test_route_tool_call_patch_unsupported_mode_returns_error():
    with patch("agent.workspace_router.should_route_to_node", return_value="node-1"):
        result_json = workspace_router.route_tool_call("patch", {
            "mode": "other",
            "patch": "*** Begin Patch\n*** End Patch",
        })

    assert result_json is not None
    result = json.loads(result_json)
    assert "error" in result
    assert result["error"] == "Unsupported patch mode: other"
