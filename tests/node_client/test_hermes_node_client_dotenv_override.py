import importlib.util
import sys
from pathlib import Path


def test_node_client_dotenv_overrides_stale_env(monkeypatch, tmp_path):
    source = (
        Path("/home/goodbaikin/.hermes/hermes-agent/node_client/hermes_node_client.py")
        .read_text(encoding="utf-8")
    )
    script_path = tmp_path / "hermes_node_client.py"
    script_path.write_text(source, encoding="utf-8")
    (tmp_path / ".env").write_text("HERMES_NODE_ID=main\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_NODE_ID", "dev-win01")

    spec = importlib.util.spec_from_file_location("node_client_dotenv_override_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.NODE_ID == "main"
        assert module.os.environ["HERMES_NODE_ID"] == "main"
    finally:
        sys.modules.pop(spec.name, None)


def test_node_client_dotenv_with_bom_loads_node_id(monkeypatch, tmp_path):
    source = (
        Path("/home/goodbaikin/.hermes/hermes-agent/node_client/hermes_node_client.py")
        .read_text(encoding="utf-8")
    )
    script_path = tmp_path / "hermes_node_client.py"
    script_path.write_text(source, encoding="utf-8")
    (tmp_path / ".env").write_text("HERMES_NODE_ID=main\n", encoding="utf-8-sig")

    monkeypatch.delenv("HERMES_NODE_ID", raising=False)

    spec = importlib.util.spec_from_file_location("node_client_dotenv_bom_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.NODE_ID == "main"
    finally:
        sys.modules.pop(spec.name, None)


def test_node_client_defaults_node_id_to_hostname(monkeypatch, tmp_path):
    source = (
        Path("/home/goodbaikin/.hermes/hermes-agent/node_client/hermes_node_client.py")
        .read_text(encoding="utf-8")
    )
    script_path = tmp_path / "hermes_node_client.py"
    script_path.write_text(source, encoding="utf-8")

    monkeypatch.delenv("HERMES_NODE_ID", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "DESKTOP-OCEVV34.example.local")

    spec = importlib.util.spec_from_file_location("node_client_hostname_default_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.NODE_ID == "DESKTOP-OCEVV34"
    finally:
        sys.modules.pop(spec.name, None)
