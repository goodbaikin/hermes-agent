import asyncio
import hmac
import json
import logging
import os
import socket as _socket
import sqlite3
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from api_server.node_registry import NODE_REGISTRY
from gateway.platforms.base import SendResult
from api_server.middleware import cors_middleware, body_limit_middleware, security_headers_middleware, _openai_error, MAX_REQUEST_BYTES, _CORS_HEADERS, _IdempotencyCache, _idem_cache, _CRON_AVAILABLE, _cron_list, _cron_get, _cron_create, _cron_update, _cron_remove, _cron_pause, _cron_resume, _cron_trigger
from api_server.sse import CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS
from api_server.utils import is_network_accessible
from api_server.utils import _make_request_fingerprint, _derive_chat_session_id, _normalize_chat_content, _normalize_multimodal_content, _content_has_visible_payload, _multimodal_validation_error

from hermes_cli.config import get_hermes_home
from hermes_cli.models import curated_models_for_provider, list_available_providers
from hermes_state import SessionDB
from tools.memory_tool import MemoryStore, get_memory_dir

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642
MAX_STORED_RESPONSES = 10000


def _coerce_port(value: Any, default: int = DEFAULT_PORT) -> int:
    """Parse a listen port without letting malformed env/config values crash startup."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def check_api_server_requirements() -> bool:
    """Check if API server dependencies are available."""
    return AIOHTTP_AVAILABLE


class ResponseStore:
    """
    SQLite-backed LRU store for Responses API state.

    Each stored response includes the full internal conversation history
    (with tool calls and results) so it can be reconstructed on subsequent
    requests via previous_response_id.

    Persists across gateway restarts.  Falls back to in-memory SQLite
    if the on-disk path is unavailable.
    """

    def __init__(self, max_size: int = MAX_STORED_RESPONSES, db_path: str = None):
        self._max_size = max_size
        if db_path is None:
            try:
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
            except Exception:
                db_path = ":memory:"
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                name TEXT PRIMARY KEY,
                response_id TEXT NOT NULL
            )"""
        )
        self._conn.commit()

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", (response_id,)
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id),
        )
        self._conn.commit()
        return json.loads(row[0])

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()),
        )
        # Evict oldest entries beyond max_size
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            self._conn.execute(
                "DELETE FROM responses WHERE response_id IN "
                "(SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?)",
                (count - self._max_size,),
            )
        self._conn.commit()

    def delete(self, response_id: str) -> bool:
        """Remove a response from the store. Returns True if found and deleted."""
        cursor = self._conn.execute(
            "DELETE FROM responses WHERE response_id = ?", (response_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        row = self._conn.execute(
            "SELECT response_id FROM conversations WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else None

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        self._conn.execute(
            "INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)",
            (name, response_id),
        )
        self._conn.commit()

    def __len__(self) -> int:
        """Return the number of stored responses."""
        return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]


class StandaloneAPIServer:
    """
    Standalone OpenAI-compatible HTTP API server.

    Independent from Gateway — runs as a pure aiohttp application that
    accepts OpenAI-format requests and routes them through hermes-agent's
    AIAgent.  Can be started directly via ``python -m api_server`` or
    managed by systemd.
    """

    platform = Platform.API_SERVER

    def __init__(self, config: Optional[PlatformConfig] = None):
        extra = (config.extra or {}) if config else {}
        self._host: str = extra.get("host", os.getenv("API_SERVER_HOST", DEFAULT_HOST))
        raw_port = extra.get("port")
        if raw_port is None:
            raw_port = os.getenv("API_SERVER_PORT", str(DEFAULT_PORT))
        self._port: int = _coerce_port(raw_port, DEFAULT_PORT)
        self._api_key: str = extra.get("key", os.getenv("API_SERVER_KEY", ""))
        self._cors_origins: tuple[str, ...] = self._parse_cors_origins(
            extra.get("cors_origins", os.getenv("API_SERVER_CORS_ORIGINS", "")),
        )
        self._model_name: str = self._resolve_model_name(
            extra.get("model_name", os.getenv("API_SERVER_MODEL_NAME", "")),
        )
        self._app: Optional["web.Application"] = None
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._response_store = ResponseStore()
        # Active run streams: run_id -> asyncio.Queue of SSE event dicts
        self._run_streams: Dict[str, "asyncio.Queue[Optional[Dict]]"] = {}
        # Creation timestamps for orphaned-run TTL sweep
        self._run_streams_created: Dict[str, float] = {}
        # Pollable run statuses
        self._run_statuses: Dict[str, Dict[str, Any]] = {}
        # Active agent instances for interrupt support
        self._active_run_agents: Dict[str, Any] = {}
        # Active asyncio tasks for cancellation
        self._active_run_tasks: Dict[str, asyncio.Task] = {}
        # Background tasks (replaces BasePlatformAdapter._background_tasks)
        self._background_tasks: set[asyncio.Task] = set()

        self._session_db: Optional[SessionDB] = None
        self._memory_store: Optional[MemoryStore] = None
        self._running = False
        # Session-scoped AIAgent cache — reuse agents across turns for the same
        # session to avoid rebuilding tool schemas, memory providers, and LLM
        # clients on every request (fixes unclosed aiohttp session leaks).
        self._session_agent_cache: Dict[str, Any] = {}
        self._session_agent_cache_lock = asyncio.Lock()


    @staticmethod
    def _parse_cors_origins(value: Any) -> tuple[str, ...]:
        """Normalize configured CORS origins into a stable tuple."""
        if not value:
            return ()

        if isinstance(value, str):
            items = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            items = value
        else:
            items = [str(value)]

        return tuple(str(item).strip() for item in items if str(item).strip())

    @staticmethod
    def _resolve_model_name(explicit: str) -> str:
        """Derive the advertised model name for /v1/models.

        Priority:
        1. Explicit override (config extra or API_SERVER_MODEL_NAME env var)
        2. Active profile name (so each profile advertises a distinct model)
        3. Fallback: "hermes-agent"
        """
        if explicit and explicit.strip():
            return explicit.strip()
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile and profile not in ("default", "custom"):
                return profile
        except Exception:
            pass
        return "hermes-agent"

    def _cors_headers_for_origin(self, origin: str) -> Optional[Dict[str, str]]:
        """Return CORS headers for an allowed browser origin."""
        if not origin or not self._cors_origins:
            return None

        if "*" in self._cors_origins:
            headers = dict(_CORS_HEADERS)
            headers["Access-Control-Allow-Origin"] = "*"
            headers["Access-Control-Max-Age"] = "600"
            return headers

        if origin not in self._cors_origins:
            return None

        headers = dict(_CORS_HEADERS)
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"
        headers["Access-Control-Max-Age"] = "600"
        return headers

    def _origin_allowed(self, origin: str) -> bool:
        """Allow non-browser clients and explicitly configured browser origins."""
        if not origin:
            return True

        if not self._cors_origins:
            return False

        return "*" in self._cors_origins or origin in self._cors_origins

    # ------------------------------------------------------------------
    # Auth helper
    # ------------------------------------------------------------------

    def _check_auth(self, request: "web.Request") -> Optional["web.Response"]:
        """
        Validate Bearer token from Authorization header.

        Returns None if auth is OK, or a 401 web.Response on failure.
        If no API key is configured, all requests are allowed (only when API
        server is local).
        """
        if not self._api_key:
            # No key configured -- allow all, but reject X-Hermes-Session-Key
            # because accepting caller-supplied memory scopes without auth is unsafe
            session_key = request.headers.get("X-Hermes-Session-Key")
            if session_key:
                return web.json_response(
                    {"error": {"message": "X-Hermes-Session-Key requires API key authentication", "type": "invalid_request_error", "code": "session_key_requires_auth"}},
                    status=403,
                )
            return None

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if hmac.compare_digest(token, self._api_key):
                return None  # Auth OK

        return web.json_response(
            {"error": {"message": "Invalid API key", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status=401,
        )

    def _parse_session_key_header(self, request) -> tuple:
        """Parse and validate X-Hermes-Session-Key header.

        Returns (session_key, error_response) where error_response is a
        web.Response if validation fails, otherwise None.
        """
        session_key = request.headers.get("X-Hermes-Session-Key")
        if not session_key:
            return None, None
        if len(session_key) > 256:
            return None, web.json_response(
                {"error": {"message": "X-Hermes-Session-Key too long", "type": "invalid_request_error", "code": "session_key_too_long"}},
                status=400,
            )
        if any(ord(c) < 32 for c in session_key):
            return None, web.json_response(
                {"error": {"message": "X-Hermes-Session-Key contains control characters", "type": "invalid_request_error", "code": "session_key_invalid"}},
                status=400,
            )
        return session_key, None

    def _get_session_db(self) -> SessionDB:
        """Create the session DB lazily."""
        if self._session_db is None:
            self._session_db = SessionDB()
        return self._session_db

    def _get_memory_store(self) -> MemoryStore:
        """Create the memory store lazily."""
        if self._memory_store is None:
            self._memory_store = MemoryStore()
            self._memory_store.load_from_disk()
        return self._memory_store

    @staticmethod
    def _current_model_settings(config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model/provider/base_url/api_mode from config.yaml."""
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            return {
                "model": str(model_cfg.get("default") or model_cfg.get("model") or "").strip(),
                "provider": str(model_cfg.get("provider") or "").strip(),
                "api_mode": str(model_cfg.get("api_mode") or "").strip(),
                "base_url": str(model_cfg.get("base_url") or "").strip(),
            }
        if isinstance(model_cfg, str):
            return {
                "model": model_cfg.strip(),
                "provider": "",
                "api_mode": "",
                "base_url": "",
            }
        return {"model": "", "provider": "", "api_mode": "", "base_url": ""}

    @staticmethod
    def _parse_int(value: Any, default: int, minimum: int = 0) -> int:
        """Parse an integer query parameter with bounds."""
        if value in (None, ""):
            return default
        parsed = int(value)
        if parsed < minimum:
            raise ValueError(f"Value must be >= {minimum}")
        return parsed

    # ------------------------------------------------------------------
    # Session DB helper
    # ------------------------------------------------------------------

    def _ensure_session_db(self):
        """Lazily initialise and return the shared SessionDB instance.

        Sessions are persisted to ``state.db`` so that ``hermes sessions list``
        shows API-server conversations alongside CLI and gateway ones.
        """
        if self._session_db is None:
            try:
                from hermes_state import SessionDB
                self._session_db = SessionDB()
            except Exception as e:
                logger.debug("SessionDB unavailable for API server: %s", e)
        return self._session_db

    # ------------------------------------------------------------------
    # Agent creation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_content(
        text: str, attachments: Optional[List[Dict[str, Any]]] = None
    ) -> tuple:
        """Build multimodal content from text + image attachments."""
        from api_server.utils import _build_user_content as _build
        return _build(text, attachments)

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        gateway_session_key: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        profile: Optional[str] = None,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.

        When *session_id* is provided, the agent is cached and reused for
        subsequent requests to the same session. This avoids rebuilding tool
        schemas, memory providers, and LLM clients on every request.
        """
        # Return cached agent for this session if available and config matches
        if session_id:
            cached = self._session_agent_cache.get(session_id)
            if cached is not None:
                # Update callbacks (they may differ per request)
                if stream_delta_callback:
                    cached.stream_delta_callback = stream_delta_callback
                if tool_progress_callback:
                    cached.tool_progress_callback = tool_progress_callback
                if tool_start_callback:
                    cached.tool_start_callback = tool_start_callback
                if tool_complete_callback:
                    cached.tool_complete_callback = tool_complete_callback
                # Update ephemeral system prompt if provided
                if ephemeral_system_prompt is not None:
                    cached.ephemeral_system_prompt = ephemeral_system_prompt
                return cached

        from run_agent import AIAgent
        from gateway.run import _resolve_runtime_agent_kwargs, _resolve_gateway_model, _load_gateway_config, GatewayRunner
        from hermes_cli.tools_config import _get_platform_tools

        runtime_kwargs = _resolve_runtime_agent_kwargs()
        reasoning_config = GatewayRunner._load_reasoning_config()
        model = _resolve_gateway_model()

        user_config = _load_gateway_config()
        enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))

        max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

        # Load fallback provider chain so the API server platform has the
        # same fallback behaviour as Telegram/Discord/Slack (fixes #4954).
        fallback_model = GatewayRunner._load_fallback_model()

        agent = AIAgent(
            model=model,
            **runtime_kwargs,
            max_iterations=max_iterations,
            quiet_mode=True,
            verbose_logging=False,
            ephemeral_system_prompt=ephemeral_system_prompt or None,
            enabled_toolsets=enabled_toolsets,
            session_id=session_id,
            gateway_session_key=gateway_session_key,
            platform="api_server",
            stream_delta_callback=stream_delta_callback,
            tool_progress_callback=tool_progress_callback,
            tool_start_callback=tool_start_callback,
            tool_complete_callback=tool_complete_callback,
            session_db=self._ensure_session_db(),
            fallback_model=fallback_model,
            reasoning_config=reasoning_config,
            profile=profile,
        )

        if session_id:
            self._session_agent_cache[session_id] = agent

        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_list_profiles(self, request: "web.Request") -> "web.Response":
        """GET /api/profiles -- list all profiles."""
        from api_server.handlers.profiles import handle_list_profiles
        return await handle_list_profiles(request, check_auth=self._check_auth)

    async def _handle_get_profile_config(self, request: "web.Request") -> "web.Response":
        """GET /api/profiles/{name} -- get profile config."""
        from api_server.handlers.profiles import handle_get_profile_config
        return await handle_get_profile_config(request, check_auth=self._check_auth)

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health -- simple health check."""
        from api_server.handlers.health import handle_health
        return await handle_health(request)

    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed -- rich status for cross-container dashboard probing."""
        from api_server.handlers.health import handle_health_detailed
        return await handle_health_detailed(request)

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models -- return hermes-agent as an available model."""
        from api_server.handlers.models import handle_models
        return await handle_models(request, check_auth=self._check_auth, model_name=self._model_name)


    async def _handle_list_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions -- list sessions."""
        from api_server.handlers.sessions import handle_list_sessions
        return await handle_list_sessions(
            request,
            check_auth=self._check_auth,
            ensure_session_db=self._ensure_session_db,
        )

    async def _handle_create_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions -- create a new session."""
        from api_server.handlers.sessions import handle_create_session
        return await handle_create_session(
            request,
            check_auth=self._check_auth,
            ensure_session_db=self._ensure_session_db,
        )

    async def _handle_search_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/search -- search messages across sessions."""
        from api_server.handlers.sessions import handle_search_sessions
        return await handle_search_sessions(
            request,
            check_auth=self._check_auth,
            ensure_session_db=self._ensure_session_db,
        )

    async def _handle_get_session(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id} -- fetch one session."""
        from api_server.handlers.sessions import handle_get_session
        return await handle_get_session(
            request,
            check_auth=self._check_auth,
            ensure_session_db=self._ensure_session_db,
        )

    async def _handle_get_session_messages(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}/messages -- fetch session messages."""
        from api_server.handlers.sessions import handle_get_session_messages
        return await handle_get_session_messages(
            request,
            check_auth=self._check_auth,
            ensure_session_db=self._ensure_session_db,
        )

    async def _handle_update_session(self, request: "web.Request") -> "web.Response":
        """PATCH /api/sessions/{session_id} -- update a session."""
        from api_server.handlers.sessions import handle_update_session
        return await handle_update_session(
            request,
            check_auth=self._check_auth,
            ensure_session_db=self._ensure_session_db,
        )

    async def _handle_delete_session(self, request: "web.Request") -> "web.Response":
        """DELETE /api/sessions/{session_id} -- delete a session."""
        from api_server.handlers.sessions import handle_delete_session
        session_id = request.match_info.get("session_id", "")
        response = await handle_delete_session(
            request,
            check_auth=self._check_auth,
            ensure_session_db=self._ensure_session_db,
        )
        # Evict cached agent for this session so we don't leak memory or
        # keep stale providers (e.g. unclosed aiohttp sessions) alive.
        if response.status == 200 and session_id in self._session_agent_cache:
            agent = self._session_agent_cache.pop(session_id, None)
            if agent is not None and hasattr(agent, 'memory_manager') and agent.memory_manager is not None:
                try:
                    agent.memory_manager.shutdown_all()
                except Exception:
                    pass
        return response

    async def _handle_fork_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/fork -- clone a session and its messages."""
        from api_server.handlers.sessions import handle_fork_session
        return await handle_fork_session(
            request,
            check_auth=self._check_auth,
            ensure_session_db=self._ensure_session_db,
        )

    async def _handle_session_chat(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/chat -- run a session-aware chat turn."""
        from api_server.handlers.session_chat import handle_session_chat
        return await handle_session_chat(
            request,
            check_auth=self._check_auth,
            get_session_db=self._get_session_db,
            normalize_session_record=self._normalize_session_record,
            build_user_content=self._build_user_content,
            create_agent=self._create_agent,
        )

    async def _handle_session_chat_stream(self, request: "web.Request") -> "web.StreamResponse":
        """POST /api/sessions/{session_id}/chat/stream -- stream a session chat turn over SSE."""
        from api_server.handlers.session_chat import handle_session_chat_stream
        return await handle_session_chat_stream(
            request,
            check_auth=self._check_auth,
            get_session_db=self._get_session_db,
            normalize_session_record=self._normalize_session_record,
            build_user_content=self._build_user_content,
            create_agent=self._create_agent,
            cors_headers_for_origin=self._cors_headers_for_origin,
        )

    async def _handle_get_memory(self, request: "web.Request") -> "web.Response":
        """GET /api/memory -- read current memory state."""
        from api_server.handlers.memory import handle_get_memory
        return await handle_get_memory(
            request,
            check_auth=self._check_auth,
            get_memory_store=self._get_memory_store,
        )

    async def _handle_add_memory(self, request: "web.Request") -> "web.Response":
        """POST /api/memory -- add a memory entry."""
        from api_server.handlers.memory import handle_add_memory
        return await handle_add_memory(
            request,
            check_auth=self._check_auth,
            get_memory_store=self._get_memory_store,
        )

    async def _handle_replace_memory(self, request: "web.Request") -> "web.Response":
        """PATCH /api/memory -- replace a memory entry."""
        from api_server.handlers.memory import handle_replace_memory
        return await handle_replace_memory(
            request,
            check_auth=self._check_auth,
            get_memory_store=self._get_memory_store,
        )

    async def _handle_delete_memory(self, request: "web.Request") -> "web.Response":
        """DELETE /api/memory -- delete a memory entry."""
        from api_server.handlers.memory import handle_delete_memory
        return await handle_delete_memory(
            request,
            check_auth=self._check_auth,
            get_memory_store=self._get_memory_store,
        )

    async def _handle_list_skills(self, request: "web.Request") -> "web.Response":
        """GET /api/skills -- list skills."""
        from api_server.handlers.skills import handle_list_skills
        return await handle_list_skills(request, check_auth=self._check_auth)

    async def _handle_view_skill(self, request: "web.Request") -> "web.Response":
        """GET /api/skills/{name} -- fetch skill details."""
        from api_server.handlers.skills import handle_view_skill
        return await handle_view_skill(request, check_auth=self._check_auth)

    async def _handle_get_config(self, request: "web.Request") -> "web.Response":
        """GET /api/config -- fetch the current config."""
        from api_server.handlers.config import handle_get_config
        return await handle_get_config(request, check_auth=self._check_auth)

    async def _handle_update_config(self, request: "web.Request") -> "web.Response":
        """PATCH /api/config -- update model/provider/base_url settings."""
        from api_server.handlers.config import handle_update_config
        return await handle_update_config(request, check_auth=self._check_auth)

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """GET /v1/capabilities -- list available toolsets and reasoning modes."""
        from api_server.handlers.capabilities import handle_capabilities
        return await handle_capabilities(request, check_auth=self._check_auth, api_key=self._api_key)
    async def _handle_available_models(self, request: "web.Request") -> "web.Response":
        """GET /api/available-models -- list provider models and available providers."""
        from api_server.handlers.config import handle_available_models
        return await handle_available_models(
            request,
            check_auth=self._check_auth,
            current_model_settings=self._current_model_settings,
        )


    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions -- OpenAI Chat Completions format."""
        from api_server.handlers.chat_completions import handle_chat_completions
        return await handle_chat_completions(
            request,
            adapter=self,
            idem_cache=_idem_cache,
        )

    async def _write_sse_chat_completion(
        self, request: "web.Request", completion_id: str, model: str,
        created: int, stream_q, agent_task, agent_ref=None, session_id: str = None,
    ) -> "web.StreamResponse":
        """Write real streaming SSE from agent's stream_delta_callback queue.

        If the client disconnects mid-stream (network drop, browser tab close),
        the agent is interrupted via ``agent.interrupt()`` so it stops making
        LLM API calls, and the asyncio task wrapper is cancelled.
        """
        import queue as _q

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        # CORS middleware can't inject headers into StreamResponse after
        # prepare() flushes them, so resolve CORS headers up front.
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)
        if session_id:
            sse_headers["X-Hermes-Session-Id"] = session_id
        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        try:
            last_activity = time.monotonic()

            # Role chunk
            role_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            await response.write(f"data: {json.dumps(role_chunk)}\n\n".encode())
            last_activity = time.monotonic()

            # Helper -- route a queue item to the correct SSE event.
            async def _emit(item):
                """Write a single queue item to the SSE stream.

                Plain strings are sent as normal ``delta.content`` chunks.
                Tagged tuples ``("__tool_progress__", payload)`` are sent
                as a custom ``event: hermes.tool.progress`` SSE event so
                frontends can display them without storing the markers in
                conversation history.  See #6972 for the original event,
                #16588 for the ``toolCallId``/``status`` lifecycle fields.
                """
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "__tool_progress__":
                    event_data = json.dumps(item[1])
                    await response.write(
                        f"event: hermes.tool.progress\ndata: {event_data}\n\n".encode()
                    )
                else:
                    content_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}],
                    }
                    await response.write(f"data: {json.dumps(content_chunk)}\n\n".encode())
                return time.monotonic()

            # Stream content chunks as they arrive from the agent
            loop = asyncio.get_running_loop()
            while True:
                try:
                    delta = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        # Drain any remaining items
                        while True:
                            try:
                                delta = stream_q.get_nowait()
                                if delta is None:
                                    break
                                last_activity = await _emit(delta)
                            except _q.Empty:
                                break
                        break
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if delta is None:  # End of stream sentinel
                    break

                last_activity = await _emit(delta)

            # Get usage from completed agent
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            try:
                result, agent_usage = await agent_task
                usage = agent_usage or usage
            except Exception:
                pass

            # Finish chunk
            finish_chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }
            await response.write(f"data: {json.dumps(finish_chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            # Client disconnected mid-stream.  Interrupt the agent so it
            # stops making LLM API calls at the next loop iteration, then
            # cancel the asyncio task wrapper.
            agent = agent_ref[0] if agent_ref else None
            if agent is not None:
                try:
                    agent.interrupt("SSE client disconnected")
                except Exception:
                    pass
            if not agent_task.done():
                agent_task.cancel()
                try:
                    await agent_task
                except (asyncio.CancelledError, Exception):
                    pass
            logger.info("SSE client disconnected; interrupted agent task %s", completion_id)

        return response

    async def _handle_responses(self, request: "web.Request") -> "web.Response":
        """POST /v1/responses -- OpenAI Responses API format."""
        from api_server.handlers.responses import handle_responses
        return await handle_responses(request, adapter=self, idem_cache=_idem_cache)

    async def _handle_get_response(self, request: "web.Request") -> "web.Response":
        """GET /v1/responses/{response_id} -- retrieve a stored response."""
        from api_server.handlers.responses import handle_get_response
        return await handle_get_response(
            request,
            check_auth=self._check_auth,
            response_store=self._response_store,
        )

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} -- delete a stored response."""
        from api_server.handlers.responses import handle_delete_response
        return await handle_delete_response(
            request,
            check_auth=self._check_auth,
            response_store=self._response_store,
        )

    # ------------------------------------------------------------------
    # Cron jobs API
    # ------------------------------------------------------------------

    async def _handle_list_jobs(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs -- list all cron jobs."""
        from api_server.handlers.jobs import handle_list_jobs
        return await handle_list_jobs(request, check_auth=self._check_auth)

    async def _handle_create_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs -- create a new cron job."""
        from api_server.handlers.jobs import handle_create_job
        return await handle_create_job(request, check_auth=self._check_auth)

    async def _handle_get_job(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs/{job_id} -- get a single cron job."""
        from api_server.handlers.jobs import handle_get_job
        return await handle_get_job(request, check_auth=self._check_auth)

    async def _handle_update_job(self, request: "web.Request") -> "web.Response":
        """PATCH /api/jobs/{job_id} -- update a cron job."""
        from api_server.handlers.jobs import handle_update_job
        return await handle_update_job(request, check_auth=self._check_auth)

    async def _handle_delete_job(self, request: "web.Request") -> "web.Response":
        """DELETE /api/jobs/{job_id} -- delete a cron job."""
        from api_server.handlers.jobs import handle_delete_job
        return await handle_delete_job(request, check_auth=self._check_auth)

    async def _handle_pause_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/pause -- pause a cron job."""
        from api_server.handlers.jobs import handle_pause_job
        return await handle_pause_job(request, check_auth=self._check_auth)

    async def _handle_resume_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/resume -- resume a paused cron job."""
        from api_server.handlers.jobs import handle_resume_job
        return await handle_resume_job(request, check_auth=self._check_auth)

    async def _handle_run_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/run -- trigger immediate execution."""
        from api_server.handlers.jobs import handle_run_job
        return await handle_run_job(request, check_auth=self._check_auth)

    # ------------------------------------------------------------------
    # Read-only / metadata APIs for web UIs
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_session_record(record: Dict[str, Any]) -> Dict[str, Any]:
        from api_server.handlers.sessions import _normalize_session_record as _norm
        return _norm(record)

    # ------------------------------------------------------------------
    # Output extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_output_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the full output item array from the agent's messages.

        Walks *result["messages"]* and emits:
        - ``function_call`` items for each tool_call on assistant messages
        - ``function_call_output`` items for each tool-role message
        - a final ``message`` item with the assistant's text reply
        """
        items: List[Dict[str, Any]] = []
        messages = result.get("messages", [])

        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    items.append({
                        "type": "function_call",
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", ""),
                        "call_id": tc.get("id", ""),
                    })
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        # Final assistant message
        final = result.get("final_response", "")
        if not final:
            final = result.get("error", "(No response generated)")

        items.append({
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": final,
                }
            ],
        })
        return items

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        gateway_session_key: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
        agent_ref: Optional[list] = None,
    ) -> tuple:
        """
        Create an agent and run a conversation in a thread executor.

        Returns ``(result_dict, usage_dict)`` where *usage_dict* contains
        ``input_tokens``, ``output_tokens`` and ``total_tokens``.

        If *agent_ref* is a one-element list, the AIAgent instance is stored
        at ``agent_ref[0]`` before ``run_conversation`` begins.  This allows
        callers (e.g. the SSE writer) to call ``agent.interrupt()`` from
        another thread to stop in-progress LLM calls.
        """
        loop = asyncio.get_running_loop()

        def _run():
            agent = self._create_agent(
                ephemeral_system_prompt=ephemeral_system_prompt,
                session_id=session_id,
                gateway_session_key=gateway_session_key,
                stream_delta_callback=stream_delta_callback,
                tool_progress_callback=tool_progress_callback,
                tool_start_callback=tool_start_callback,
                tool_complete_callback=tool_complete_callback,
            )
            if agent_ref is not None:
                agent_ref[0] = agent
            effective_task_id = session_id or str(uuid.uuid4())
            result = agent.run_conversation(
                user_message=user_message,
                conversation_history=conversation_history,
                task_id=effective_task_id,
            )
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return result, usage

        return await loop.run_in_executor(None, _run)

    # ------------------------------------------------------------------
    # /v1/runs -- structured event streaming
    # ------------------------------------------------------------------

    _MAX_CONCURRENT_RUNS = 10  # Prevent unbounded resource allocation
    _RUN_STREAM_TTL = 300  # seconds before orphaned runs are swept
    _RUN_STATUS_TTL = 3600  # seconds to retain terminal run status for polling

    def _set_run_status(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        """Update pollable run status without exposing private agent objects."""
        now = time.time()
        current = self._run_statuses.get(run_id, {})
        current.update({
            "object": "hermes.run",
            "run_id": run_id,
            "status": status,
            "updated_at": now,
        })
        current.setdefault("created_at", fields.pop("created_at", now))
        current.update(fields)
        self._run_statuses[run_id] = current
        return current

    def _make_run_event_callback(self, run_id: str, loop: "asyncio.AbstractEventLoop"):
        """Return a tool_progress_callback that pushes structured events to the run's SSE queue."""
        def _push(event: Dict[str, Any]) -> None:
            self._set_run_status(
                run_id,
                self._run_statuses.get(run_id, {}).get("status", "running"),
                last_event=event.get("event"),
            )
            q = self._run_streams.get(run_id)
            if q is None:
                return
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception:
                pass

        def _callback(event_type: str, tool_name: str = None, preview: str = None, args=None, **kwargs):
            ts = time.time()
            if event_type == "tool.started":
                _push({
                    "event": "tool.started",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "tool_name": tool_name,
                    "tool_call_id": kwargs.get("tool_call_id", ""),
                    "preview": preview,
                    "args": args,
                })
            elif event_type == "tool.completed":
                _push({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "tool_name": tool_name,
                    "tool_call_id": kwargs.get("tool_call_id", ""),
                    "duration": round(kwargs.get("duration", 0), 3),
                    "error": kwargs.get("is_error", False),
                    "result_preview": kwargs.get("result_preview", ""),
                })
            elif event_type == "reasoning.available":
                _push({
                    "event": "reasoning.available",
                    "run_id": run_id,
                    "timestamp": ts,
                    "text": preview or "",
                })
            # _thinking and subagent_progress are intentionally not forwarded

        return _callback

    async def _handle_runs(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs -- start an agent run, return run_id immediately."""
        from api_server.handlers.runs import handle_runs
        return await handle_runs(request, adapter=self)

    async def _handle_get_run(self, request: "web.Request") -> "web.Response":
        """GET /v1/runs/{run_id} -- return pollable run status for external UIs."""
        from api_server.handlers.runs import handle_get_run
        return await handle_get_run(request, adapter=self)

    async def _handle_run_events(self, request: "web.Request") -> "web.StreamResponse":
        """GET /v1/runs/{run_id}/events -- SSE stream of structured agent lifecycle events."""
        from api_server.handlers.runs import handle_run_events
        return await handle_run_events(request, adapter=self)

    async def _handle_stop_run(self, request: "web.Request") -> "web.Response":
        """POST /v1/runs/{run_id}/stop -- interrupt a running agent."""
        from api_server.handlers.runs import handle_stop_run
        return await handle_stop_run(request, adapter=self)

    async def _sweep_orphaned_runs(self) -> None:
        """Periodically clean up run streams that were never consumed."""
        from api_server.handlers.runs import sweep_orphaned_runs
        await sweep_orphaned_runs(self)

    async def connect(self) -> bool:
        """Start the aiohttp web server."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("[%s] aiohttp not installed", "api_server")
            return False

        try:
            mws = [mw for mw in (cors_middleware, body_limit_middleware, security_headers_middleware) if mw is not None]
            self._app = web.Application(middlewares=mws)
            self._app["api_server_adapter"] = self
            self._app.router.add_get("/health", self._handle_health)
            self._app.router.add_get("/health/detailed", self._handle_health_detailed)
            self._app.router.add_get("/v1/health", self._handle_health)
            self._app.router.add_get("/v1/models", self._handle_models)
            self._app.router.add_get("/v1/capabilities", self._handle_capabilities)
            self._app.router.add_post("/v1/chat/completions", self._handle_chat_completions)
            self._app.router.add_post("/v1/responses", self._handle_responses)
            self._app.router.add_get("/v1/responses/{response_id}", self._handle_get_response)
            self._app.router.add_delete("/v1/responses/{response_id}", self._handle_delete_response)
            # Cron jobs management API
            self._app.router.add_get("/api/jobs", self._handle_list_jobs)
            self._app.router.add_post("/api/jobs", self._handle_create_job)
            self._app.router.add_get("/api/jobs/{job_id}", self._handle_get_job)
            self._app.router.add_patch("/api/jobs/{job_id}", self._handle_update_job)
            self._app.router.add_delete("/api/jobs/{job_id}", self._handle_delete_job)
            self._app.router.add_post("/api/jobs/{job_id}/pause", self._handle_pause_job)
            self._app.router.add_post("/api/jobs/{job_id}/resume", self._handle_resume_job)
            self._app.router.add_post("/api/jobs/{job_id}/run", self._handle_run_job)
            # Remote node HTTP API
            self._app.router.add_get("/v1/nodes", self._handle_list_nodes)
            self._app.router.add_post("/v1/nodes/{node_id}/invoke", self._handle_node_invoke)
            # Remote node WebSocket (OpenClaw-style gateway-node protocol)
            self._app.router.add_get("/ws", self._handle_ws)
            # Structured event streaming
            self._app.router.add_post("/v1/runs", self._handle_runs)
            self._app.router.add_get("/v1/runs/{run_id}", self._handle_get_run)
            self._app.router.add_get("/v1/runs/{run_id}/events", self._handle_run_events)
            self._app.router.add_post("/v1/runs/{run_id}/stop", self._handle_stop_run)
            # Start background sweep to clean up orphaned (unconsumed) run streams
            sweep_task = asyncio.create_task(self._sweep_orphaned_runs())
            try:
                self._background_tasks.add(sweep_task)
            except TypeError:
                pass
            if hasattr(sweep_task, "add_done_callback"):
                sweep_task.add_done_callback(self._background_tasks.discard)
            self._app.router.add_get("/api/sessions", self._handle_list_sessions)
            self._app.router.add_post("/api/sessions", self._handle_create_session)
            self._app.router.add_get("/api/sessions/search", self._handle_search_sessions)
            self._app.router.add_get("/api/sessions/{session_id}", self._handle_get_session)
            self._app.router.add_get("/api/sessions/{session_id}/messages", self._handle_get_session_messages)
            self._app.router.add_patch("/api/sessions/{session_id}", self._handle_update_session)
            self._app.router.add_delete("/api/sessions/{session_id}", self._handle_delete_session)
            self._app.router.add_post("/api/sessions/{session_id}/fork", self._handle_fork_session)
            self._app.router.add_post("/api/sessions/{session_id}/chat", self._handle_session_chat)
            self._app.router.add_post("/api/sessions/{session_id}/chat/stream", self._handle_session_chat_stream)
            self._app.router.add_get("/api/memory", self._handle_get_memory)
            self._app.router.add_post("/api/memory", self._handle_add_memory)
            self._app.router.add_patch("/api/memory", self._handle_replace_memory)
            self._app.router.add_delete("/api/memory", self._handle_delete_memory)
            self._app.router.add_get("/api/skills", self._handle_list_skills)
            self._app.router.add_get("/api/skills/{name}", self._handle_view_skill)
            self._app.router.add_get("/api/config", self._handle_get_config)
            self._app.router.add_patch("/api/config", self._handle_update_config)
            self._app.router.add_get("/api/available-models", self._handle_available_models)
            self._app.router.add_get("/api/profiles", self._handle_list_profiles)
            self._app.router.add_get("/api/profiles/{name}", self._handle_get_profile_config)

            # Refuse to start network-accessible without authentication
            if is_network_accessible(self._host) and not self._api_key:
                logger.error(
                    "[%s] Refusing to start: binding to %s requires API_SERVER_KEY. "
                    "Set API_SERVER_KEY or use the default 127.0.0.1.",
                    "api_server", self._host,
                )
                return False

            # Refuse to start network-accessible with a placeholder key.
            # Ported from openclaw/openclaw#64586.
            if is_network_accessible(self._host) and self._api_key:
                try:
                    from hermes_cli.auth import has_usable_secret
                    if not has_usable_secret(self._api_key, min_length=8):
                        logger.error(
                        "[%s] Refusing to start: API_SERVER_KEY is set to a "
                        "placeholder value. Generate a real secret "
                        "(e.g. `openssl rand -hex 32`) and set API_SERVER_KEY "
                        "before exposing the API server on %s.",
                        "api_server", self._host,
                    )
                        return False
                except ImportError:
                    pass

            # Port conflict detection -- fail fast if port is already in use
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _s:
                    _s.settimeout(1)
                    _s.connect(('127.0.0.1', self._port))
                logger.error('[%s] Port %d already in use. Set a different port in config.yaml: platforms.api_server.port', "api_server", self._port)
                return False
            except (ConnectionRefusedError, OSError):
                pass  # port is free

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()

            self._running = True
            if not self._api_key:
                logger.warning(
                    "[%s] ⚠️  No API key configured (API_SERVER_KEY / platforms.api_server.key). "
                    "All requests will be accepted without authentication. "
                    "Set an API key for production deployments to prevent "
                    "unauthorized access to sessions, responses, and cron jobs.",
                    "api_server",
                )
            logger.info(
                "[%s] API server listening on http://%s:%d (model: %s)",
                "api_server", self._host, self._port, self._model_name,
            )
            return True

        except Exception as e:
            logger.error("[%s] Failed to start API server: %s", "api_server", e)
            return False

    async def disconnect(self) -> None:
        """Stop the aiohttp web server."""
        self._running = False
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        if self._session_db is not None:
            self._session_db.close()
            self._session_db = None
        self._memory_store = None
        logger.info("[%s] API server stopped", "api_server")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Not used -- HTTP request/response cycle handles delivery directly.
        """
        return SendResult(success=False, error="API server uses HTTP request/response, not send()")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about the API server."""
        return {
            "name": "API Server",
            "type": "api",
            "host": self._host,
            "port": self._port,
        }

    # ------------------------------------------------------------------
    # Remote Node WebSocket handler
    # ------------------------------------------------------------------

    async def _handle_ws(self, request: "web.Request") -> "web.WebSocketResponse":
        """GET /ws -- WebSocket endpoint for remote node protocol (OpenClaw-style)."""
        from api_server.handlers.ws import handle_ws_real
        return await handle_ws_real(request)

    # ------------------------------------------------------------------
    # Node HTTP API handlers
    # ------------------------------------------------------------------

    async def _handle_list_nodes(self, request: "web.Request") -> "web.Response":
        """GET /v1/nodes -- list all connected remote nodes."""
        from api_server.handlers.nodes import handle_list_nodes
        return await handle_list_nodes(request, check_auth=self._check_auth)

    async def _handle_node_invoke(self, request: "web.Request") -> "web.Response":
        """POST /v1/nodes/{node_id}/invoke -- invoke a command on a remote node."""
        from api_server.handlers.nodes import handle_node_invoke
        return await handle_node_invoke(request, check_auth=self._check_auth)
