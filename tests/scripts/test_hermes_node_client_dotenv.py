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

    monkeypatch.delenv("HERMES_NODE_ID", raising=False)

    spec = importlib.util.spec_from_file_location("_scripts_node_client_under_test", script_copy)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.NODE_ID == "main"
