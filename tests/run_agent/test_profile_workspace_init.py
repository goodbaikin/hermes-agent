from types import SimpleNamespace

import run_agent


def test_aiagent_accepts_profile_workspace_and_forwards(monkeypatch):
    captured = {}

    import agent.agent_init as agent_init

    def fake_init(agent, **kwargs):
        captured.update(kwargs)
        setattr(agent, "profile", kwargs.get("profile"))
        setattr(agent, "_workspace", kwargs.get("workspace"))

    monkeypatch.setattr(agent_init, "init_agent", fake_init)

    agent = run_agent.AIAgent(
        model="dummy-model",
        profile="webui-lite-eng",
        workspace="dev",
    )

    assert captured["profile"] == "webui-lite-eng"
    assert captured["workspace"] == "dev"
    assert getattr(agent, "profile") == "webui-lite-eng"
    assert getattr(agent, "_workspace") == "dev"


def test_ensure_db_session_persists_profile_metadata():
    class FakeSessionDB:
        def __init__(self):
            self.created = {}

        def create_session(self, **kwargs):
            self.created = kwargs
            return kwargs["session_id"]

    db = FakeSessionDB()
    agent = SimpleNamespace(
        _session_db_created=False,
        _session_db=db,
        session_id="session-1",
        platform="api_server",
        model="dummy-model",
        _session_init_model_config={},
        _cached_system_prompt="system",
        _parent_session_id=None,
        profile="webui-lite-eng",
    )

    run_agent.AIAgent._ensure_db_session(agent)  # type: ignore[arg-type]

    assert agent._session_db_created is True
    assert db.created["profile"] == "webui-lite-eng"


def test_run_conversation_binds_agent_workspace_context(monkeypatch):
    captured = {}

    import agent.agent_init as agent_init
    import agent.conversation_loop as conversation_loop
    from agent.workspace_context import get_workspace

    def fake_init(agent, **kwargs):
        setattr(agent, "profile", kwargs.get("profile"))
        setattr(agent, "_workspace", kwargs.get("workspace"))

    def fake_run_conversation(agent, *args, **kwargs):
        captured["workspace_during_turn"] = get_workspace()
        return {"final_response": "ok"}

    monkeypatch.setattr(agent_init, "init_agent", fake_init)
    monkeypatch.setattr(conversation_loop, "run_conversation", fake_run_conversation)

    agent = run_agent.AIAgent(
        model="dummy-model",
        profile="csharp-eng",
        workspace="dev",
    )

    result = agent.run_conversation("hello")

    assert result["final_response"] == "ok"
    assert captured["workspace_during_turn"] == "dev"
    assert get_workspace() is None
