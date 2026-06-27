import json

import tools.browser_tool as browser_tool


def _reset_browser_state(monkeypatch):
    monkeypatch.setattr(browser_tool, "_active_sessions", {})
    monkeypatch.setattr(browser_tool, "_session_last_activity", {})
    monkeypatch.setattr(browser_tool, "_recording_sessions", set())
    monkeypatch.setattr(browser_tool, "_last_active_session_key", {})
    monkeypatch.setattr(browser_tool, "_task_browser_nodes", {})
    monkeypatch.setattr(browser_tool, "_start_browser_cleanup_thread", lambda: None)
    monkeypatch.setattr(browser_tool, "_update_session_activity", lambda task_id: None)
    monkeypatch.setattr(browser_tool, "_ensure_cdp_supervisor", lambda task_id: None)


def test_get_session_info_uses_remote_node_browser_status(monkeypatch):
    _reset_browser_state(monkeypatch)
    monkeypatch.setattr(browser_tool, "_get_cdp_override", lambda: "")
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)
    browser_tool._task_browser_nodes["task-remote"] = "main-pc"

    monkeypatch.setattr(
        "tools.node_lib.node_browser_debug_status",
        lambda node_id: {
            "listening": True,
            "suggested_connect_url": "ws://192.168.1.50:9222/devtools/browser/abc123",
            "websocket_debugger_url": "ws://127.0.0.1:9222/devtools/browser/abc123",
        },
        raising=False,
    )

    session = browser_tool._get_session_info("task-remote")

    assert session["node_id"] == "main-pc"
    assert session["cdp_url"] == "ws://192.168.1.50:9222/devtools/browser/abc123"
    assert session["features"]["remote_node_browser"] is True


def test_browser_navigate_with_node_id_bypasses_private_url_block_and_hybrid_sidecar(monkeypatch):
    _reset_browser_state(monkeypatch)
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: object())
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_always_blocked_url", lambda url: False)
    monkeypatch.setattr(browser_tool, "_allow_private_urls", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_safe_url", lambda url: False, raising=False)
    monkeypatch.setattr(browser_tool, "_normalize_url_for_request", lambda url: url, raising=False)
    monkeypatch.setattr(browser_tool, "check_website_access", lambda url: None)

    seen = {}

    def fake_get_session_info(task_id, node_id=None):
        seen["session_task_id"] = task_id
        return {"_first_nav": True, "features": {"remote_node_browser": True}}

    def fake_run_browser_command(task_id, command, args, timeout=None, _engine_override=None):
        seen["run_task_id"] = task_id
        assert command == "open"
        return {"success": True, "data": {"title": "Remote App", "url": "http://192.168.1.10:3000"}}

    monkeypatch.setattr(browser_tool, "_get_session_info", fake_get_session_info)
    monkeypatch.setattr(browser_tool, "_run_browser_command", fake_run_browser_command)

    result = json.loads(
        browser_tool.browser_navigate(
            "http://192.168.1.10:3000",
            task_id="verify-main-browser",
            node_id="main-pc",
        )
    )

    assert result["success"] is True
    assert seen["session_task_id"] == "verify-main-browser"
    assert seen["run_task_id"] == "verify-main-browser"
    assert browser_tool._task_browser_nodes["verify-main-browser"] == "main-pc"
    assert browser_tool._last_active_session_key["verify-main-browser"] == "verify-main-browser"


def test_browser_schema_exposes_optional_node_id_parameter():
    schema_map = {schema["name"]: schema for schema in browser_tool.BROWSER_TOOL_SCHEMAS}

    assert "node_id" in schema_map["browser_navigate"]["parameters"]["properties"]
    assert "node_id" in schema_map["browser_snapshot"]["parameters"]["properties"]
