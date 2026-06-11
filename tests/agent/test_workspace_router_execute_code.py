import json


def test_remote_execute_code_injects_explicit_env_passthrough(monkeypatch):
    from agent import workspace_router
    from tools import node_lib

    writes = []
    execs = []

    monkeypatch.setenv("BACKLOG_API_KEY", "test-backlog-key")
    monkeypatch.setattr(workspace_router, "_load_parent_dotenv_for_remote_execute_code", lambda: None)
    monkeypatch.setattr(workspace_router, "_get_workspace_workdir", lambda: "C:/Users/goodb/workspace/COCONV.Deploy/")

    import tools.env_passthrough as env_passthrough
    monkeypatch.setattr(env_passthrough, "get_all_passthrough", lambda: frozenset({"BACKLOG_API_KEY", "MISSING_KEY"}))

    def fake_write(node_id, path, content):
        writes.append((node_id, path, content))

    def fake_exec(node_id, cmd, timeout=300, cwd=None):
        execs.append((node_id, cmd, timeout, cwd))
        return {"payload": {"output": "ok", "exit_code": 0}}

    monkeypatch.setattr(node_lib, "node_write", fake_write)
    monkeypatch.setattr(node_lib, "node_exec", fake_exec)

    result = json.loads(workspace_router._route_execute_code(
        "dev-win01",
        {"code": "import os\nprint(os.environ.get('BACKLOG_API_KEY'))", "timeout": 12},
    ))

    assert result == {"output": "ok", "exit_code": 0}
    assert writes
    _node_id, _path, content = writes[0]
    assert "explicit allowlist only" in content
    assert "BACKLOG_API_KEY" in content
    assert "test-backlog-key" in content
    assert "MISSING_KEY" not in content
    assert content.rstrip().endswith("print(os.environ.get('BACKLOG_API_KEY'))")
    assert execs[0][0] == "dev-win01"
    assert execs[0][2] == 12
    assert execs[0][3] == "C:/Users/goodb/workspace/COCONV.Deploy/"


def test_remote_terminal_injects_explicit_env_passthrough(monkeypatch):
    from agent import workspace_router
    from tools import node_lib

    execs = []

    monkeypatch.setenv("BACKLOG_API_KEY", "test-backlog-key")
    monkeypatch.setattr(workspace_router, "_load_parent_dotenv_for_remote_execute_code", lambda: None)
    monkeypatch.setattr(workspace_router, "_get_workspace_mode", lambda: "replace")
    monkeypatch.setattr(workspace_router, "_resolve_workspace", lambda: "webui")
    monkeypatch.setattr(workspace_router, "_get_node_for_workspace", lambda name: "lxc-207")
    monkeypatch.setattr(workspace_router, "_get_workspace_workdir", lambda: "/opt/hermes-webui-lite/")

    import tools.env_passthrough as env_passthrough
    monkeypatch.setattr(env_passthrough, "get_all_passthrough", lambda: frozenset({"BACKLOG_API_KEY"}))

    def fake_exec(node_id, cmd, timeout=300, cwd=None):
        execs.append((node_id, cmd, timeout, cwd))
        return {"payload": {"output": "present", "exit_code": 0}}

    monkeypatch.setattr(node_lib, "node_exec", fake_exec)

    routed = workspace_router.route_tool_call(
        "terminal",
        {"command": "python3 -c \"import os; print(bool(os.environ.get('BACKLOG_API_KEY')))\"", "timeout": 9},
    )
    assert routed is not None
    result = json.loads(routed)

    assert result == {"output": "present", "stderr": "", "exit_code": 0}
    assert execs[0][0] == "lxc-207"
    assert execs[0][2] == 9
    assert execs[0][3] == "/opt/hermes-webui-lite/"
    assert execs[0][1].startswith("export BACKLOG_API_KEY=test-backlog-key; ")
    assert "python3 -c" in execs[0][1]


def test_remote_execute_code_has_no_env_prelude_without_allowlist(monkeypatch):
    from agent import workspace_router

    monkeypatch.setenv("BACKLOG_API_KEY", "test-backlog-key")
    monkeypatch.setattr(workspace_router, "_load_parent_dotenv_for_remote_execute_code", lambda: None)

    import tools.env_passthrough as env_passthrough
    monkeypatch.setattr(env_passthrough, "get_all_passthrough", lambda: frozenset())

    assert workspace_router._remote_execute_code_env_prelude() == ""
