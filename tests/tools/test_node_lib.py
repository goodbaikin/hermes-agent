import json
from unittest.mock import patch

from tools import node_lib


def test_node_patch_v4a_invokes_node_file_patch_command():
    patch_text = "*** Begin Patch\n*** End Patch"
    response = {"ok": True, "payload": {"success": True, "files_modified": []}}

    with patch("tools.node_lib.node_invoke") as mock_invoke:
        mock_invoke.return_value = json.dumps(response)
        result = node_lib.node_patch_v4a("node-1", patch_text)

    mock_invoke.assert_called_once_with("node-1", "file.patch", {"patch": patch_text})
    assert result["payload"] == response["payload"]


def test_node_patch_v4a_includes_base_dir_when_passed():
    patch_text = "*** Begin Patch\n*** End Patch"
    response = {"ok": True, "payload": {"success": True}}
    with patch("tools.node_lib.node_invoke") as mock_invoke:
        mock_invoke.return_value = json.dumps(response)
        node_lib.node_patch_v4a("node-1", patch_text, base_dir="/tmp/workspace")

    mock_invoke.assert_called_once_with(
        "node-1",
        "file.patch",
        {"patch": patch_text, "base_dir": "/tmp/workspace"},
    )


def test_node_exec_converts_public_seconds_timeout_to_node_milliseconds():
    response = {
        "ok": True,
        "payload": {"stdout": "done", "stderr": "", "exitCode": 0},
    }

    with patch("tools.node_lib.node_invoke") as mock_invoke:
        mock_invoke.return_value = json.dumps(response)
        result = node_lib.node_exec("node-1", "sleep 1", timeout=30)

    mock_invoke.assert_called_once_with(
        "node-1",
        "terminal.exec",
        {"cmd": "sleep 1", "timeoutMs": 30000},
        timeout_ms=35000,
    )
    assert result["payload"]["output"] == "done"


def test_node_exec_default_timeout_is_five_minutes():
    response = {
        "ok": True,
        "payload": {"stdout": "done", "stderr": "", "exitCode": 0},
    }

    with patch("tools.node_lib.node_invoke") as mock_invoke:
        mock_invoke.return_value = json.dumps(response)
        result = node_lib.node_exec("node-1", "long-running-command")

    mock_invoke.assert_called_once_with(
        "node-1",
        "terminal.exec",
        {"cmd": "long-running-command", "timeoutMs": 300000},
        timeout_ms=305000,
    )
    assert result["payload"]["output"] == "done"



