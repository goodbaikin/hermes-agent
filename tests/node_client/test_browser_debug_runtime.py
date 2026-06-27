import types

import pytest

from node_client import hermes_node_client as mod


@pytest.fixture(autouse=True)
def _reset_runtime_features():
    mod.configure_runtime_features(False)
    yield
    mod.configure_runtime_features(False)


def test_default_runtime_commands_include_browser_but_not_computer_use():
    mod.configure_runtime_features(False)

    assert "browser.debug_status" in mod.COMMANDS
    assert "browser.debug_launch" in mod.COMMANDS
    assert "computer.use" not in mod.COMMANDS
    assert mod.NODE_METADATA["browser_scope"] == "full_browser"
    assert mod.NODE_METADATA["computer_use_enabled"] is False


def test_configure_runtime_features_enables_computer_use_when_handler_available(monkeypatch):
    fake_module = types.SimpleNamespace()

    async def fake_handle_computer_use(params):
        return {"ok": True, "params": params}

    fake_module.handle_computer_use = fake_handle_computer_use
    real_import_module = mod.importlib.import_module

    def fake_import_module(name, package=None):
        if name == ".computer_use_handler":
            raise ImportError("no package-local handler in test")
        if name == "computer_use_handler":
            return fake_module
        return real_import_module(name, package)

    monkeypatch.setattr(mod.importlib, "import_module", fake_import_module)

    mod.configure_runtime_features(True)

    assert "computer.use" in mod.COMMANDS
    assert mod.HANDLERS["computer.use"] is fake_handle_computer_use
    assert mod.NODE_METADATA["computer_use_enabled"] is True


def test_browser_debug_status_payload_rewrites_websocket_host(monkeypatch):
    monkeypatch.setattr(
        mod,
        "_read_browser_debug_payload",
        lambda host, port: {
            "Browser": "Chrome/137.0",
            "Protocol-Version": "1.3",
            "User-Agent": "Chrome",
            "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/abc123",
            "_target_count": 4,
        },
    )

    payload = mod._browser_debug_status_payload(
        port=9222,
        discovery_host="127.0.0.1",
        public_host="192.168.1.50",
    )

    assert payload["listening"] is True
    assert payload["target_count"] == 4
    assert payload["websocket_debugger_url"] == "ws://127.0.0.1:9222/devtools/browser/abc123"
    assert payload["suggested_connect_url"] == "ws://192.168.1.50:9222/devtools/browser/abc123"


def test_build_arg_parser_registers_allow_computer_use_flag():
    parser = mod.build_arg_parser()
    args = parser.parse_args(["--allow-computer-use"])

    assert args.allow_computer_use is True
