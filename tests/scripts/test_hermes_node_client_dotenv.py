import importlib.util
from pathlib import Path


def test_scripts_hermes_node_client_loads_node_id_from_adjacent_dotenv(monkeypatch, tmp_path):
    src = Path(__file__).resolve().parents[2] / "scripts" / "hermes_node_client.py"
    script_copy = tmp_path / "hermes_node_client.py"
    script_copy.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / ".env").write_text(
        "HERMES_NODE_ID=main\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_NODE_ID", "dev-win01")

    spec = importlib.util.spec_from_file_location("_scripts_node_client_under_test", script_copy)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NODE_ID == "main"
    assert module.os.environ["HERMES_NODE_ID"] == "main"


def test_scripts_hermes_node_client_defaults_node_id_to_hostname(monkeypatch, tmp_path):
    src = Path(__file__).resolve().parents[2] / "scripts" / "hermes_node_client.py"
    script_copy = tmp_path / "hermes_node_client.py"
    script_copy.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.delenv("HERMES_NODE_ID", raising=False)
    monkeypatch.setattr("socket.gethostname", lambda: "DESKTOP-OCEVV34.example.local")

    spec = importlib.util.spec_from_file_location("_scripts_node_client_hostname_under_test", script_copy)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NODE_ID == "DESKTOP-OCEVV34"
