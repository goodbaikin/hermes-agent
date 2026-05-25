from pathlib import Path
from types import SimpleNamespace

from agent.conversation_compression import compress_context
from run_agent import AIAgent


class _FakeSessionDB:
    def __init__(self):
        self.created = None
        self.ended = None

    def get_session_title(self, session_id):
        return None

    def end_session(self, session_id, reason):
        self.ended = (session_id, reason)

    def create_session(self, **kwargs):
        self.created = kwargs

    def update_system_prompt(self, session_id, system_prompt):
        self.system_prompt = (session_id, system_prompt)


class _FakeCompressor:
    compression_count = 1
    _last_compress_aborted = False
    _last_summary_error = None
    threshold_tokens = 100000
    last_prompt_tokens = 0
    last_completion_tokens = 0

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        return list(messages[:1]) + [{"role": "user", "content": "summary"}] + list(messages[-1:])


class _FakeTodoStore:
    def format_for_injection(self):
        return ""


def test_compression_child_session_preserves_profile(tmp_path):
    db = _FakeSessionDB()
    agent = SimpleNamespace(
        session_id="parent-session",
        model="test-model",
        platform="api_server",
        profile="csharp-eng",
        _workspace="dev",
        _session_db=db,
        _session_db_created=True,
        _session_init_model_config={"provider": "test"},
        _memory_manager=None,
        _todo_store=_FakeTodoStore(),
        context_compressor=_FakeCompressor(),
        tools=[],
        logs_dir=Path(tmp_path),
        _cached_system_prompt=None,
        session_log_file=None,
        _last_flushed_db_idx=99,
        _emit_status=lambda *args, **kwargs: None,
        _emit_warning=lambda *args, **kwargs: None,
        _vprint=lambda *args, **kwargs: None,
        _invalidate_system_prompt=lambda: None,
        _build_system_prompt=lambda system_message=None: "new system prompt",
        commit_memory_session=lambda messages: None,
        log_prefix="",
    )

    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    compressed, new_prompt = compress_context(agent, messages, "system", approx_tokens=123, task_id="test")

    assert new_prompt == "new system prompt"
    assert compressed[1]["content"] == "summary"
    assert db.ended == ("parent-session", "compression")
    assert db.created["parent_session_id"] == "parent-session"
    assert db.created["profile"] == "csharp-eng"
    assert "workspace" not in db.created
    assert agent._last_flushed_db_idx == 0


def test_initial_db_session_preserves_profile():
    db = _FakeSessionDB()
    agent = SimpleNamespace(
        _session_db_created=False,
        _session_db=db,
        session_id="initial-session",
        platform="api_server",
        model="test-model",
        _session_init_model_config={"provider": "test"},
        _cached_system_prompt="system prompt",
        _parent_session_id="parent-session",
        profile="csharp-eng",
        _workspace="dev",
    )

    AIAgent._ensure_db_session(agent)

    assert db.created["session_id"] == "initial-session"
    assert db.created["parent_session_id"] == "parent-session"
    assert db.created["profile"] == "csharp-eng"
    assert "workspace" not in db.created
    assert agent._session_db_created is True
