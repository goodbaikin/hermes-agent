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
