"""
Tests for the Session API endpoints on the API server gateway adapter.

Tests cover:
- Session CRUD (create, list, get, update, delete, fork, search, messages)
- Memory CRUD (get, add, replace, delete)
- Skills (list, categories, view)
- Config (get, update, available-models)
- Auth enforcement on all session/memory/skills/config endpoints
- Chat endpoints (sync and streaming SSE)
- Capability probe fast-path
"""

import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from api_server.server import StandaloneAPIServer
from api_server.middleware import _CORS_HEADERS, cors_middleware, security_headers_middleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", cors_origins=None) -> StandaloneAPIServer:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    if cors_origins is not None:
        extra["cors_origins"] = cors_origins
    config = PlatformConfig(enabled=True, extra=extra)
    return StandaloneAPIServer(config)


def _create_session_app(adapter: StandaloneAPIServer) -> web.Application:
    """Create the aiohttp app with ALL routes (existing + session API)."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    # Existing routes
    app.router.add_get("/health", adapter._handle_health)
    app.router.add_get("/v1/models", adapter._handle_models)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    # Session routes
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/search", adapter._handle_search_sessions)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_get_session_messages)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_update_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    # Memory
    app.router.add_get("/api/memory", adapter._handle_get_memory)
    app.router.add_post("/api/memory", adapter._handle_add_memory)
    app.router.add_patch("/api/memory", adapter._handle_replace_memory)
    app.router.add_delete("/api/memory", adapter._handle_delete_memory)
    # Skills
    app.router.add_get("/api/skills", adapter._handle_list_skills)
    # app.router.add_get("/api/skills/categories", adapter._handle_skill_categories)  # not implemented
    app.router.add_get("/api/skills/{name}", adapter._handle_view_skill)
    # Config
    app.router.add_get("/api/config", adapter._handle_get_config)
    app.router.add_patch("/api/config", adapter._handle_update_config)
    app.router.add_get("/api/available-models", adapter._handle_available_models)
    return app


def _mock_session_db():
    """Create a mock SessionDB with all needed methods."""
    db = MagicMock()
    db.list_sessions_rich.return_value = []
    db.session_count.return_value = 0
    db.get_session.return_value = None
    db.resolve_session_id.side_effect = lambda session_id: session_id
    db.get_messages.return_value = []
    db.message_count.side_effect = lambda *args, **kwargs: len(db.get_messages.return_value)
    db.get_messages_as_conversation.return_value = []
    db.search_messages.return_value = []
    db.create_session.return_value = None
    db.set_session_title.return_value = True
    db.update_system_prompt.return_value = None
    db.update_session_model.return_value = True
    db.end_session.return_value = None
    db.delete_session.return_value = True
    db.ensure_session.return_value = None
    db.append_message.return_value = None
    return db


def _mock_memory_store():
    """Create a mock MemoryStore with all needed methods."""
    store = MagicMock()
    store.memory_entries = ["Remember: user likes Python"]
    store.user_entries = ["Name: Alice"]
    store.load_from_disk.return_value = None
    store.add.return_value = {"success": True, "message": "Added"}
    store.replace.return_value = {"success": True, "message": "Replaced"}
    store.remove.return_value = {"success": True, "message": "Removed"}
    return store


@pytest.fixture
def adapter():
    a = _make_adapter()
    a._session_db = _mock_session_db()
    a._memory_store = _mock_memory_store()
    return a


@pytest.fixture
def auth_adapter():
    a = _make_adapter(api_key="sk-secret")
    a._session_db = _mock_session_db()
    a._memory_store = _mock_memory_store()
    return a


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_empty(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions")
            assert resp.status == 200
            data = await resp.json()
            assert data["items"] == []
            assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_list_sessions_with_items(self, adapter):
        adapter._session_db.list_sessions_rich.return_value = [
            {"session_id": "sess_1", "title": "First", "source": "api_server"},
            {"session_id": "sess_2", "title": "Second", "source": "cli"},
        ]
        adapter._session_db.session_count.return_value = 2
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions")
            assert resp.status == 200
            data = await resp.json()
            assert len(data["items"]) == 2
            assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions?limit=10&offset=5")
            assert resp.status == 200
            call_kwargs = adapter._session_db.list_sessions_rich.call_args
            assert call_kwargs.kwargs.get("limit") == 10 or call_kwargs[1].get("limit") == 10

    @pytest.mark.asyncio
    async def test_list_sessions_source_filter(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions?source=telegram")
            assert resp.status == 200
            call_kwargs = adapter._session_db.list_sessions_rich.call_args
            assert call_kwargs.kwargs.get("source") == "telegram" or call_kwargs[1].get("source") == "telegram"

    @pytest.mark.asyncio
    async def test_list_sessions_invalid_limit(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions?limit=notanumber")
            assert resp.status == 400


class TestCreateSession:
    @pytest.mark.asyncio
    async def test_create_session_minimal(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_new",
            "title": None,
            "source": "api_server",
            "model": None,
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions", json={})
            assert resp.status == 200
            data = await resp.json()
            assert "session" in data
            adapter._session_db.create_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_session_with_fields(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_new",
            "title": "My Chat",
            "source": "web",
            "model": "claude-opus-4-0-20250514",
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions",
                json={"title": "My Chat", "source": "web", "model": "claude-opus-4-0-20250514"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["session"]["title"] == "My Chat"

    @pytest.mark.asyncio
    async def test_create_session_with_model_resolves_session_override(self, adapter):
        """POST /api/sessions with a model must persist a usable session-scoped override."""
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_new",
            "source": "webui",
            "model": "gpt-5.3-codex-spark",
            "model_config": json.dumps({
                "model": "gpt-5.3-codex-spark",
                "provider": "openai-codex",
                "api_mode": "codex_responses",
            }),
        }
        switch_result = SimpleNamespace(
            success=True,
            new_model="gpt-5.3-codex-spark",
            target_provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
            provider_label="OpenAI Codex",
            resolved_via_alias="",
        )
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_cli.config.load_config", return_value={"model": {"default": "kimi-k2.6", "provider": "openrouter"}}), \
                 patch("hermes_cli.model_switch.switch_model", return_value=switch_result):
                resp = await cli.post(
                    "/api/sessions",
                    json={"source": "webui", "model": "gpt-5.3-codex-spark", "provider": "openai-codex"},
                )
        assert resp.status == 200
        kwargs = adapter._session_db.create_session.call_args.kwargs
        assert kwargs["model"] == "gpt-5.3-codex-spark"
        assert kwargs["model_config"]["provider"] == "openai-codex"
        assert kwargs["model_config"]["api_mode"] == "codex_responses"
        assert "api_key" not in kwargs["model_config"]

    @pytest.mark.asyncio
    async def test_create_session_with_model_only_resolves_against_current_provider(self, adapter):
        """A bare model value should still become a real override, not a display-only DB field."""
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_new",
            "source": "webui",
            "model": "moonshotai/kimi-k2.6",
            "model_config": json.dumps({"model": "moonshotai/kimi-k2.6", "provider": "openrouter"}),
        }
        switch_result = SimpleNamespace(
            success=True,
            new_model="moonshotai/kimi-k2.6",
            target_provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            provider_label="OpenRouter",
            resolved_via_alias="",
        )
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_cli.config.load_config", return_value={"model": {"default": "gpt-5.3-codex-spark", "provider": "openai-codex"}}), \
                 patch("hermes_cli.model_switch.switch_model", return_value=switch_result):
                resp = await cli.post("/api/sessions", json={"source": "webui", "model": "moonshotai/kimi-k2.6"})
        assert resp.status == 200
        kwargs = adapter._session_db.create_session.call_args.kwargs
        assert kwargs["model"] == "moonshotai/kimi-k2.6"
        assert kwargs["model_config"]["provider"] == "openrouter"

    @pytest.mark.asyncio
    async def test_create_session_invalid_json(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400
            data = await resp.json()
            assert "Invalid JSON" in data["error"]

    @pytest.mark.asyncio
    async def test_create_session_db_error(self, adapter):
        adapter._session_db.create_session.side_effect = ValueError("bad input")
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions", json={"title": "Test"})
            assert resp.status == 400


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_session_found(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_abc",
            "title": "Found",
            "source": "api_server",
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/sess_abc")
            assert resp.status == 200
            data = await resp.json()
            assert data["session"]["session_id"] == "sess_abc"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, adapter):
        adapter._session_db.get_session.return_value = None
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/sess_nonexistent")
            assert resp.status == 404
            data = await resp.json()
            assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_get_session_normalizes_model_config(self, adapter):
        """model_config JSON string is parsed into an object."""
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_mc",
            "title": "Model Config Test",
            "model_config": '{"temperature": 0.7}',
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/sess_mc")
            assert resp.status == 200
            data = await resp.json()
            assert data["session"]["model_config"] == {"temperature": 0.7}


class TestGetSessionMessages:
    @pytest.mark.asyncio
    async def test_get_messages(self, adapter):
        adapter._session_db.get_session.return_value = {"session_id": "sess_m"}
        adapter._session_db.get_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/sess_m/messages")
            assert resp.status == 200
            data = await resp.json()
            assert data["total"] == 2
            assert len(data["items"]) == 2
            assert data["items"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_messages_auto_creates_session(self, adapter):
        """If session doesn't exist, ensure_session is called."""
        adapter._session_db.get_session.return_value = None
        adapter._session_db.get_messages.return_value = []
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/sess_new/messages")
            assert resp.status == 200
            adapter._session_db.ensure_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_messages_include_lineage_returns_parent_and_child_messages(self, adapter):
        """Compression continuations must render as one logical transcript."""
        adapter._session_db.get_session.return_value = {"session_id": "child", "parent_session_id": "parent"}
        adapter._session_db._session_lineage_root_to_tip.return_value = ["parent", "child"]

        def _messages(session_id, limit=None, offset=0, order="asc"):
            return {
                "parent": [{"id": 1, "session_id": "parent", "role": "user", "content": "before compression", "timestamp": 1}],
                "child": [{"id": 2, "session_id": "child", "role": "assistant", "content": "after compression", "timestamp": 2}],
            }.get(session_id, [])

        adapter._session_db.get_messages.side_effect = _messages
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/child/messages?include_lineage=true")
            assert resp.status == 200
            data = await resp.json()
            assert data["total"] == 2
            assert [m["content"] for m in data["items"]] == ["before compression", "after compression"]
            assert data["lineage"] == ["parent", "child"]


class TestUpdateSession:
    @pytest.mark.asyncio
    async def test_update_title(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_upd",
            "title": "Updated Title",
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch("/api/sessions/sess_upd", json={"title": "Updated Title"})
            assert resp.status == 200
            data = await resp.json()
            assert "session" in data
            adapter._session_db.set_session_title.assert_called()

    @pytest.mark.asyncio
    async def test_update_system_prompt(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_upd",
            "system_prompt": "You are a pirate.",
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch(
                "/api/sessions/sess_upd",
                json={"system_prompt": "You are a pirate."},
            )
            assert resp.status == 200
            adapter._session_db.update_system_prompt.assert_called()

    @pytest.mark.asyncio
    async def test_update_end_reason(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_upd",
            "end_reason": "completed",
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch("/api/sessions/sess_upd", json={"end_reason": "completed"})
            assert resp.status == 200
            adapter._session_db.end_session.assert_called()

    @pytest.mark.asyncio
    async def test_update_not_found(self, adapter):
        adapter._session_db.get_session.return_value = None
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch("/api/sessions/sess_missing", json={"title": "X"})
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_update_invalid_json(self, adapter):
        adapter._session_db.get_session.return_value = {"session_id": "sess_upd"}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch(
                "/api/sessions/sess_upd",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_update_model_override(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_upd",
            "model": "gpt-5.3-codex-spark",
            "model_config": json.dumps({
                "model": "gpt-5.3-codex-spark",
                "provider": "openai-codex",
                "api_mode": "codex_responses",
            }),
        }
        switch_result = SimpleNamespace(
            success=True,
            new_model="gpt-5.3-codex-spark",
            target_provider="openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
            provider_label="OpenAI Codex",
            resolved_via_alias="",
        )
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("hermes_cli.config.load_config", return_value={"model": {"default": "kimi-k2.6", "provider": "openrouter"}}), \
                 patch("hermes_cli.model_switch.switch_model", return_value=switch_result):
                resp = await cli.patch(
                    "/api/sessions/sess_upd",
                    json={"model": "gpt-5.3-codex-spark", "provider": "openai-codex"},
                )
        assert resp.status == 200
        model, model_config = adapter._session_db.update_session_model.call_args.args[1:3]
        assert model == "gpt-5.3-codex-spark"
        assert model_config["provider"] == "openai-codex"
        assert model_config["api_mode"] == "codex_responses"
        assert "api_key" not in model_config

    @pytest.mark.asyncio
    async def test_clear_model_override(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_upd",
            "model": None,
            "model_config": None,
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch("/api/sessions/sess_upd", json={"model_config": None})
        assert resp.status == 200
        adapter._session_db.update_session_model.assert_called_with("sess_upd", None, None)


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_session(self, adapter):
        adapter._session_db.delete_session.return_value = True
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/api/sessions/sess_del")
            assert resp.status == 200
            data = await resp.json()
            assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self, adapter):
        adapter._session_db.delete_session.return_value = False
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/api/sessions/sess_missing")
            assert resp.status == 404


class TestForkSession:
    @pytest.mark.asyncio
    async def test_fork_session(self, adapter):
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_orig",
            "title": "Original",
            "source": "api_server",
            "model": "gpt-4",
            "system_prompt": "You are helpful.",
        }
        adapter._session_db.get_messages.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/sess_orig/fork")
            assert resp.status == 200
            data = await resp.json()
            assert "session" in data
            assert data["forked_from"] == "sess_orig"
            # create_session should have been called for the new session
            assert adapter._session_db.create_session.call_count >= 1
            # Messages should have been copied
            assert adapter._session_db.append_message.call_count == 2

    @pytest.mark.asyncio
    async def test_fork_session_not_found(self, adapter):
        adapter._session_db.get_session.return_value = None
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/sess_ghost/fork")
            assert resp.status == 404


class TestSearchSessions:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, adapter):
        adapter._session_db.search_messages.return_value = [
            {"session_id": "sess_1", "content": "Hello world", "role": "user"},
        ]
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/search?q=hello")
            assert resp.status == 200
            data = await resp.json()
            assert data["query"] == "hello"
            assert data["count"] == 1
            assert len(data["results"]) == 1

    @pytest.mark.asyncio
    async def test_search_empty_query(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/search?q=")
            assert resp.status == 400
            data = await resp.json()
            assert "Missing" in data["error"]

    @pytest.mark.asyncio
    async def test_search_no_query_param(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/search")
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_search_pagination(self, adapter):
        adapter._session_db.search_messages.return_value = []
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/search?q=test&limit=5&offset=10")
            assert resp.status == 200
            call_kwargs = adapter._session_db.search_messages.call_args
            assert call_kwargs.kwargs.get("limit") == 5 or call_kwargs[1].get("limit") == 5


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------


class TestGetMemory:
    @pytest.mark.asyncio
    async def test_get_memory_all(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/memory")
            assert resp.status == 200
            data = await resp.json()
            assert "targets" in data
            # Default target is "all" which returns both memory and user
            assert len(data["targets"]) == 2

    @pytest.mark.asyncio
    async def test_get_memory_target_memory(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/memory?target=memory")
            assert resp.status == 200
            data = await resp.json()
            assert len(data["targets"]) == 1
            assert data["targets"][0]["target"] == "memory"
            assert data["targets"][0]["entries"] == ["Remember: user likes Python"]

    @pytest.mark.asyncio
    async def test_get_memory_target_user(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/memory?target=user")
            assert resp.status == 200
            data = await resp.json()
            assert len(data["targets"]) == 1
            assert data["targets"][0]["target"] == "user"
            assert data["targets"][0]["entries"] == ["Name: Alice"]

    @pytest.mark.asyncio
    async def test_get_memory_invalid_target(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/memory?target=invalid")
            assert resp.status == 400
            data = await resp.json()
            assert "target" in data["error"]


class TestAddMemory:
    @pytest.mark.asyncio
    async def test_add_memory(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/memory",
                json={"target": "memory", "content": "User prefers dark mode"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["success"] is True
            adapter._memory_store.add.assert_called_once_with("memory", "User prefers dark mode")

    @pytest.mark.asyncio
    async def test_add_memory_user_target(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/memory",
                json={"target": "user", "content": "Favorite color: blue"},
            )
            assert resp.status == 200
            adapter._memory_store.add.assert_called_once_with("user", "Favorite color: blue")

    @pytest.mark.asyncio
    async def test_add_memory_invalid_target(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/memory",
                json={"target": "invalid", "content": "something"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_add_memory_invalid_json(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/memory",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_add_memory_failure(self, adapter):
        adapter._memory_store.add.return_value = {"success": False, "error": "Limit exceeded"}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/memory",
                json={"target": "memory", "content": "Too much data"},
            )
            assert resp.status == 400


class TestReplaceMemory:
    @pytest.mark.asyncio
    async def test_replace_memory(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch(
                "/api/memory",
                json={"target": "memory", "old_text": "old entry", "content": "new entry"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["success"] is True
            adapter._memory_store.replace.assert_called_once_with("memory", "old entry", "new entry")

    @pytest.mark.asyncio
    async def test_replace_memory_invalid_target(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch(
                "/api/memory",
                json={"target": "bad", "old_text": "x", "content": "y"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_replace_memory_invalid_json(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch(
                "/api/memory",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400


class TestDeleteMemory:
    @pytest.mark.asyncio
    async def test_delete_memory(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete(
                "/api/memory",
                json={"target": "memory", "old_text": "Remember: user likes Python"},
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["success"] is True
            adapter._memory_store.remove.assert_called_once_with("memory", "Remember: user likes Python")

    @pytest.mark.asyncio
    async def test_delete_memory_invalid_target(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete(
                "/api/memory",
                json={"target": "nope", "old_text": "x"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_delete_memory_invalid_json(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete(
                "/api/memory",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_delete_memory_failure(self, adapter):
        adapter._memory_store.remove.return_value = {"success": False, "error": "Not found"}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete(
                "/api/memory",
                json={"target": "memory", "old_text": "nonexistent"},
            )
            assert resp.status == 400


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


class TestListSkills:
    @pytest.mark.asyncio
    async def test_list_skills(self, adapter):
        mock_result = json.dumps({"skills": [{"name": "git", "description": "Git commands"}]})
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("api_server.handlers.skills.skills_list", return_value=mock_result):
                resp = await cli.get("/api/skills")
                assert resp.status == 200
                data = await resp.json()
                assert "skills" in data

    @pytest.mark.asyncio
    async def test_list_skills_with_category(self, adapter):
        mock_result = json.dumps({"skills": []})
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("api_server.handlers.skills.skills_list", return_value=mock_result) as mock_fn:
                resp = await cli.get("/api/skills?category=development")
                assert resp.status == 200
                mock_fn.assert_called_once_with(category="development")


class TestSkillCategories:
    @pytest.mark.skip(reason="_handle_skill_categories not implemented")
    @pytest.mark.asyncio
    async def test_skill_categories(self, adapter):
        mock_result = json.dumps({"categories": ["development", "productivity"]})
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("api_server.server.skills_categories", return_value=mock_result):
                resp = await cli.get("/api/skills/categories")
                assert resp.status == 200
                data = await resp.json()
                assert "categories" in data
                assert len(data["categories"]) == 2


class TestViewSkill:
    @pytest.mark.asyncio
    async def test_view_skill(self, adapter):
        mock_result = json.dumps({
            "name": "git",
            "description": "Git commands",
            "content": "# Git Skill\nUse git for version control.",
        })
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("api_server.handlers.skills.skill_view", return_value=mock_result) as mock_fn:
                resp = await cli.get("/api/skills/git")
                assert resp.status == 200
                data = await resp.json()
                assert data["name"] == "git"
                mock_fn.assert_called_once_with("git", file_path=None)

    @pytest.mark.asyncio
    async def test_view_skill_with_file_path(self, adapter):
        mock_result = json.dumps({"name": "git", "content": "# Git"})
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("api_server.handlers.skills.skill_view", return_value=mock_result) as mock_fn:
                resp = await cli.get("/api/skills/git?file_path=README.md")
                assert resp.status == 200
                mock_fn.assert_called_once_with("git", file_path="README.md")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestGetConfig:
    @pytest.mark.asyncio
    async def test_get_config(self, adapter):
        mock_config = {
            "model": {"default": "claude-opus-4-0-20250514", "provider": "anthropic", "base_url": ""},
        }
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch("api_server.handlers.config.load_config", return_value=mock_config):
                resp = await cli.get("/api/config")
                assert resp.status == 200
                data = await resp.json()
                assert "model" in data
                assert "provider" in data
                assert "config" in data
                assert data["model"] == "claude-opus-4-0-20250514"
                assert data["provider"] == "anthropic"


class TestUpdateConfig:
    @pytest.mark.asyncio
    async def test_update_config_model(self, adapter):
        mock_config = {"model": {"default": "old-model", "provider": "openai"}}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch("api_server.handlers.config.load_config", return_value=mock_config),
                patch("api_server.handlers.config.save_config") as mock_save,
            ):
                resp = await cli.patch("/api/config", json={"model": "gpt-4o"})
                assert resp.status == 200
                data = await resp.json()
                assert data["ok"] is True
                mock_save.assert_called_once()
                saved_config = mock_save.call_args[0][0]
                assert saved_config["model"]["default"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_update_config_provider(self, adapter):
        mock_config = {"model": {"default": "gpt-4", "provider": "openai"}}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch("api_server.handlers.config.load_config", return_value=mock_config),
                patch("api_server.handlers.config.save_config") as mock_save,
            ):
                resp = await cli.patch("/api/config", json={"provider": "anthropic"})
                assert resp.status == 200
                data = await resp.json()
                assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_update_config_invalid_json(self, adapter):
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch(
                "/api/config",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_update_config_save_error(self, adapter):
        mock_config = {"model": {"default": "gpt-4"}}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch("api_server.handlers.config.load_config", return_value=mock_config),
                patch("api_server.handlers.config.save_config", side_effect=IOError("Permission denied")),
            ):
                resp = await cli.patch("/api/config", json={"model": "gpt-4o"})
                assert resp.status == 500


class TestAvailableModels:
    @pytest.mark.asyncio
    async def test_available_models(self, adapter):
        mock_config = {"model": {"default": "gpt-4", "provider": "anthropic"}}
        mock_curated = [("claude-opus-4-0-20250514", "Powerful model"), ("claude-sonnet-4-20250514", "Fast model")]
        mock_providers = ["anthropic", "openai", "openrouter"]
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch("api_server.handlers.config.load_config", return_value=mock_config),
                patch("api_server.handlers.config.curated_models_for_provider", return_value=mock_curated),
                patch("api_server.handlers.config.list_available_providers", return_value=mock_providers),
            ):
                resp = await cli.get("/api/available-models?provider=anthropic")
                assert resp.status == 200
                data = await resp.json()
                assert data["provider"] == "anthropic"
                assert len(data["models"]) == 2
                assert data["models"][0]["id"] == "claude-opus-4-0-20250514"
                assert "providers" in data
                assert len(data["providers"]) == 3

    @pytest.mark.asyncio
    async def test_available_models_default_provider(self, adapter):
        """When no provider query param, uses config provider or defaults to openrouter."""
        mock_config = {"model": {"default": "gpt-4", "provider": "openai"}}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch("api_server.handlers.config.load_config", return_value=mock_config),
                patch("api_server.handlers.config.curated_models_for_provider", return_value=[]) as mock_curated,
                patch("api_server.handlers.config.list_available_providers", return_value=[]),
            ):
                resp = await cli.get("/api/available-models")
                assert resp.status == 200
                # Should use config provider "openai"
                mock_curated.assert_called_once_with("openai")

    def test_context_windows_are_provider_aware(self):
        """API model-picker context display should share agent metadata logic.

        The old local static table reported GPT-5.5 as 128k for every provider,
        while the agent resolver knows direct OpenAI/OpenRouter is 1.05M and
        Codex OAuth is capped lower.
        """
        from api_server.handlers.config import _get_model_context_window

        assert _get_model_context_window("openai/gpt-5.5", provider="openrouter") == 1_050_000
        assert _get_model_context_window("gpt-5.5", provider="openai") == 1_050_000
        assert _get_model_context_window("gpt-5.5", provider="openai-codex") == 272_000

    @pytest.mark.asyncio
    async def test_available_models_passes_provider_to_context_resolver(self, adapter):
        """Endpoint output must not fall back to a provider-agnostic 128k table."""
        from api_server.handlers.config import invalidate_models_cache

        invalidate_models_cache()
        mock_config = {"model": {"default": "openai/gpt-5.5", "provider": "openrouter"}}
        mock_curated = [("openai/gpt-5.5", "GPT-5.5")]
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch("api_server.handlers.config.load_config", return_value=mock_config),
                patch("api_server.handlers.config.curated_models_for_provider", return_value=mock_curated),
                patch("api_server.handlers.config.list_available_providers", return_value=[]),
            ):
                resp = await cli.get("/api/available-models?provider=openrouter")
                assert resp.status == 200
                data = await resp.json()
                assert data["models"][0]["context_window"] == 1_050_000


# ---------------------------------------------------------------------------
# Auth enforcement on session/memory/skills/config endpoints
# ---------------------------------------------------------------------------


class TestSessionApiAuth:
    """When an API key is configured, all endpoints should require auth."""

    @pytest.mark.asyncio
    async def test_list_sessions_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_create_session_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions", json={})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_get_session_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/sess_123")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_get_session_messages_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/sess_123/messages")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_update_session_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch("/api/sessions/sess_123", json={"title": "X"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_delete_session_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/api/sessions/sess_123")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_fork_session_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/sess_123/fork")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_search_sessions_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions/search?q=hello")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_session_chat_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/sess_123/chat", json={"message": "hi"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_session_chat_stream_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/sess_123/chat/stream", json={"message": "hi"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_get_memory_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/memory")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_add_memory_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/memory", json={"target": "memory", "content": "x"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_replace_memory_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch("/api/memory", json={"target": "memory", "old_text": "x", "content": "y"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_delete_memory_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/api/memory", json={"target": "memory", "old_text": "x"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_list_skills_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/skills")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_skill_categories_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/skills/categories")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_view_skill_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/skills/git")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_get_config_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/config")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_update_config_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.patch("/api/config", json={"model": "gpt-4"})
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_available_models_requires_auth(self, auth_adapter):
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/available-models")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_valid_auth_passes(self, auth_adapter):
        """With a valid Bearer token, requests should succeed."""
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(
                "/api/sessions",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_does_not_require_auth(self, auth_adapter):
        """Health check is always open regardless of API key."""
        app = _create_session_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------


class TestSessionChat:
    @pytest.mark.asyncio
    async def test_session_chat_success(self, adapter):
        """Sync chat returns the agent result."""
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_chat",
            "title": "Chat Test",
            "source": "api_server",
            "model": "hermes-agent",
        }
        adapter._session_db.get_messages_as_conversation.return_value = []

        mock_result = {
            "final_response": "Hello! I'm here to help.",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
            "last_reasoning": None,
            "response_previewed": False,
        }

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = mock_result
        mock_agent.session_prompt_tokens = 100
        mock_agent.session_completion_tokens = 20
        mock_agent.session_total_tokens = 120

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post(
                    "/api/sessions/sess_chat/chat",
                    json={"message": "Hello"},
                )
                assert resp.status == 200
                data = await resp.json()
                assert data["session_id"] == "sess_chat"
                assert data["final_response"] == "Hello! I'm here to help."
                assert data["completed"] is True
                assert "run_id" in data
                assert "usage" in data
                assert data["usage"]["input_tokens"] == 100
                adapter._session_db.get_messages_as_conversation.assert_called_with(
                    "sess_chat", include_ancestors=True
                )

    @pytest.mark.asyncio
    async def test_session_chat_returns_rotated_compression_session_id(self, adapter):
        """Sync chat must follow the agent's live session after compression rotation."""
        adapter._session_db.get_session.return_value = {
            "session_id": "parent",
            "title": "Compressed Chat",
            "source": "api_server",
            "model": "hermes-agent",
        }
        adapter._session_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "before compression"},
        ]

        mock_agent = MagicMock()
        mock_agent.session_id = "child"
        mock_agent.run_conversation.return_value = {
            "final_response": "continued",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }
        mock_agent.session_prompt_tokens = 10
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 12

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post("/api/sessions/parent/chat", json={"message": "続き"})
                assert resp.status == 200
                data = await resp.json()

        assert data["session_id"] == "child"
        assert data["continued_from"] == "parent"
        adapter._session_db.get_messages_as_conversation.assert_called_with(
            "parent", include_ancestors=True
        )

    @pytest.mark.asyncio
    async def test_session_chat_missing_message(self, adapter):
        adapter._session_db.get_session.return_value = {"session_id": "sess_chat"}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/sess_chat/chat", json={})
            assert resp.status == 400
            data = await resp.json()
            assert "message" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_session_chat_invalid_json(self, adapter):
        adapter._session_db.get_session.return_value = {"session_id": "sess_chat"}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions/sess_chat/chat",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_session_chat_auto_creates_session(self, adapter):
        """If session doesn't exist, ensure_session is called."""
        adapter._session_db.get_session.return_value = None
        adapter._session_db.get_messages_as_conversation.return_value = []

        mock_result = {
            "final_response": "Hi!",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = mock_result
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post(
                    "/api/sessions/sess_new_chat/chat",
                    json={"message": "Hi"},
                )
                assert resp.status == 200
                adapter._session_db.ensure_session.assert_called()

    @pytest.mark.asyncio
    async def test_session_chat_agent_error(self, adapter):
        adapter._session_db.get_session.return_value = {"session_id": "sess_err"}
        adapter._session_db.get_messages_as_conversation.return_value = []

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=RuntimeError("Agent init failed")):
                resp = await cli.post(
                    "/api/sessions/sess_err/chat",
                    json={"message": "Hello"},
                )
                assert resp.status == 500
                data = await resp.json()
                assert "error" in data


class TestSessionChatStream:
    @pytest.mark.asyncio
    async def test_stream_returns_sse(self, adapter):
        """Streaming chat returns SSE events with correct lifecycle."""
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_stream",
            "title": "Stream Test",
            "source": "api_server",
            "model": "hermes-agent",
        }
        adapter._session_db.get_messages_as_conversation.return_value = []

        mock_result = {
            "final_response": "Streamed response!",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = mock_result
        mock_agent.session_prompt_tokens = 50
        mock_agent.session_completion_tokens = 10
        mock_agent.session_total_tokens = 60

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post(
                    "/api/sessions/sess_stream/chat/stream",
                    json={"message": "Hello"},
                )
                assert resp.status == 200
                assert "text/event-stream" in resp.headers.get("Content-Type", "")
                assert resp.headers.get("X-Accel-Buffering") == "no"

                body = await resp.text()
                # Verify lifecycle events
                assert "event: session.created" in body
                assert "event: run.started" in body
                assert "event: message.started" in body
                assert "event: assistant.completed" in body
                assert "event: run.completed" in body
                assert "event: done" in body
                # Verify session_id appears in events
                assert "sess_stream" in body
                # Verify final response content
                assert "Streamed response!" in body
                adapter._session_db.get_messages_as_conversation.assert_called_with(
                    "sess_stream", include_ancestors=True
                )

    @pytest.mark.asyncio
    async def test_stream_emits_continuation_session_created_after_compression_rotation(self, adapter):
        """Streaming chat must tell WebUI to switch to #2 after compression."""
        def _get_session(session_id):
            if session_id == "child":
                return {
                    "session_id": "child",
                    "id": "child",
                    "title": "Compressed Chat #2",
                    "parent_session_id": "parent",
                }
            return {
                "session_id": "parent",
                "id": "parent",
                "title": "Compressed Chat",
                "source": "api_server",
                "model": "hermes-agent",
            }

        adapter._session_db.get_session.side_effect = _get_session
        adapter._session_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "before compression"},
        ]

        mock_agent = MagicMock()
        mock_agent.session_id = "child"
        mock_agent.run_conversation.return_value = {
            "final_response": "continued",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }
        mock_agent.session_prompt_tokens = 10
        mock_agent.session_completion_tokens = 2
        mock_agent.session_total_tokens = 12

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post("/api/sessions/parent/chat/stream", json={"message": "続き"})
                assert resp.status == 200
                body = await resp.text()

        assert body.count("event: session.created") >= 2
        assert '"session_id": "child"' in body
        assert '"parent_session_id": "parent"' in body
        assert '"session_id": "child", "run_id"' in body
        adapter._session_db.get_messages_as_conversation.assert_called_with(
            "parent", include_ancestors=True
        )

    @pytest.mark.asyncio
    async def test_stream_missing_message(self, adapter):
        adapter._session_db.get_session.return_value = {"session_id": "sess_stream"}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/sessions/sess_stream/chat/stream", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_stream_invalid_json(self, adapter):
        adapter._session_db.get_session.return_value = {"session_id": "sess_stream"}
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/sessions/sess_stream/chat/stream",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_stream_with_deltas(self, adapter):
        """Verify that stream_delta_callback produces assistant.delta events."""
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_delta",
            "title": "Delta Test",
            "source": "api_server",
        }
        adapter._session_db.get_messages_as_conversation.return_value = []

        mock_result = {
            "final_response": "Hello world",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

        def _make_agent(**kwargs):
            stream_cb = kwargs.get("stream_delta_callback")
            mock_agent = MagicMock()

            def _run_conv(user_content, **kw):
                if stream_cb:
                    stream_cb("Hello ")
                    stream_cb("world")
                return mock_result

            mock_agent.run_conversation.side_effect = _run_conv
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=_make_agent):
                resp = await cli.post(
                    "/api/sessions/sess_delta/chat/stream",
                    json={"message": "Say hello"},
                )
                assert resp.status == 200
                body = await resp.text()
                assert "event: assistant.delta" in body
                assert "Hello " in body
                assert "event: done" in body

    @pytest.mark.asyncio
    async def test_stream_with_tool_results(self, adapter):
        """Tool call results appear as tool.completed events."""
        adapter._session_db.get_session.return_value = {
            "session_id": "sess_tools",
            "title": "Tool Test",
            "source": "api_server",
        }
        adapter._session_db.get_messages_as_conversation.return_value = []

        mock_result = {
            "final_response": "Here are the files.",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "api_calls": 2,
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "function": {"name": "terminal", "arguments": '{"command": "ls"}'},
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_abc123",
                    "content": "file1.txt\nfile2.txt",
                    "tool_name": "terminal",
                },
            ],
        }

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = mock_result
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post(
                    "/api/sessions/sess_tools/chat/stream",
                    json={"message": "List files"},
                )
                assert resp.status == 200
                body = await resp.text()
                assert "event: tool.completed" in body
                assert "terminal" in body
                assert "file1.txt" in body
                assert "event: done" in body

    @pytest.mark.asyncio
    async def test_stream_auto_creates_session(self, adapter):
        """If session doesn't exist, ensure_session is called."""
        adapter._session_db.get_session.return_value = None
        adapter._session_db.get_messages_as_conversation.return_value = []

        mock_result = {
            "final_response": "Hi!",
            "completed": True,
            "partial": False,
            "interrupted": False,
            "api_calls": 1,
            "messages": [],
        }

        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = mock_result
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post(
                    "/api/sessions/sess_auto/chat/stream",
                    json={"message": "Hi"},
                )
                assert resp.status == 200
                adapter._session_db.ensure_session.assert_called()


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


class TestCapabilityProbe:
    @pytest.mark.asyncio
    async def test_probe_max_tokens_1(self, adapter):
        """max_tokens=1 returns a fast minimal response without running the agent."""
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "probe"}],
                    "max_tokens": 1,
                },
            )
            assert resp.status == 200
            data = await resp.json()
            assert data["object"] == "chat.completion"
            assert data["id"].startswith("chatcmpl-probe-")
            assert data["choices"][0]["message"]["content"] == "ok"
            assert data["choices"][0]["finish_reason"] == "stop"
            assert data["usage"]["total_tokens"] == 1

    @pytest.mark.asyncio
    async def test_probe_does_not_call_agent(self, adapter):
        """Probe requests must not invoke the agent at all."""
        app = _create_session_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
                resp = await cli.post(
                    "/v1/chat/completions",
                    json={
                        "model": "hermes-agent",
                        "messages": [{"role": "user", "content": "probe"}],
                        "max_tokens": 1,
                    },
                )
                assert resp.status == 200
                mock_run.assert_not_called()
