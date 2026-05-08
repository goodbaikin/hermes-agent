import asyncio
import hashlib
import hmac
import json
import logging
import os
import socket as _socket
import re
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
from api_server.middleware import cors_middleware, body_limit_middleware, security_headers_middleware, _openai_error, MAX_REQUEST_BYTES, _CORS_HEADERS, _IdempotencyCache, _idem_cache
from api_server.sse import CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS
from api_server.utils import is_network_accessible
from api_server.utils import _make_request_fingerprint, _derive_chat_session_id, _normalize_chat_content, _normalize_multimodal_content, _content_has_visible_payload

from hermes_cli.config import load_config, save_config
from hermes_cli.tools_config import _get_platform_tools
from hermes_cli.auth import has_usable_secret
from hermes_state import SessionDB
from tools.memory_tool import MemoryStore

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
    def _normalize_session_record(session: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Parse serialized session fields into API-friendly JSON."""
        if session is None:
            return None
        normalized = dict(session)
        model_config = normalized.get("model_config")
        if model_config:
            try:
                normalized["model_config"] = json.loads(model_config)
            except (TypeError, json.JSONDecodeError):
                pass
        return normalized

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
        """Build multimodal content from text + image attachments.

        Returns (user_content, persist_text) where user_content is either
        a plain string or a list of content parts for multimodal input.
        """
        if not attachments:
            return text, text

        image_parts: List[Dict[str, Any]] = []
        for att in attachments:
            if not isinstance(att, dict):
                continue
            mime = ""
            for key in ("contentType", "mimeType", "mediaType"):
                val = att.get(key)
                if isinstance(val, str) and val.strip():
                    mime = val.strip()
                    break
            if not mime.startswith("image/"):
                continue
            content = ""
            for key in ("content", "base64", "data"):
                val = att.get(key)
                if isinstance(val, str) and val.strip():
                    content = val.strip()
                    break
            if not content:
                # Try dataUrl format: data:image/png;base64,...
                data_url = att.get("dataUrl", "")
                if isinstance(data_url, str) and data_url.startswith("data:"):
                    content = data_url.split(",", 1)[-1] if "," in data_url else ""
            if not content:
                continue
            image_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{content}"},
            })

        if not image_parts:
            return text, text

        content_parts: List[Dict[str, Any]] = []
        if text.strip():
            content_parts.append({"type": "text", "text": text})
        content_parts.extend(image_parts)
        return content_parts, text

    def _create_agent(
        self,
        ephemeral_system_prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        gateway_session_key: Optional[str] = None,
        stream_delta_callback=None,
        tool_progress_callback=None,
        tool_start_callback=None,
        tool_complete_callback=None,
    ) -> Any:
        """
        Create an AIAgent instance using the gateway's runtime config.

        Uses _resolve_runtime_agent_kwargs() to pick up model, api_key,
        base_url, etc. from config.yaml / env vars.  Toolsets are resolved
        from config.yaml platform_toolsets.api_server (same as all other
        gateway platforms), falling back to the hermes-api-server default.
        """
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
        )
        return agent

    # ------------------------------------------------------------------
    # HTTP Handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        """GET /health -- simple health check."""
        return web.json_response({"status": "ok", "platform": "hermes-agent"})

    async def _handle_health_detailed(self, request: "web.Request") -> "web.Response":
        """GET /health/detailed -- rich status for cross-container dashboard probing.

        Returns gateway state, connected platforms, PID, and uptime so the
        dashboard can display full status without needing a shared PID file or
        /proc access.  No authentication required.
        """
        from gateway.status import read_runtime_status

        runtime = read_runtime_status() or {}
        return web.json_response({
            "status": "ok",
            "platform": "hermes-agent",
            "gateway_state": runtime.get("gateway_state"),
            "platforms": runtime.get("platforms", {}),
            "active_agents": runtime.get("active_agents", 0),
            "exit_reason": runtime.get("exit_reason"),
            "updated_at": runtime.get("updated_at"),
            "pid": os.getpid(),
        })

    async def _handle_models(self, request: "web.Request") -> "web.Response":
        """GET /v1/models -- return hermes-agent as an available model."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": self._model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "hermes",
                    "permission": [],
                    "root": self._model_name,
                    "parent": None,
                }
            ],
        })


    async def _handle_list_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions -- list sessions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            limit = self._parse_int(request.query.get("limit"), 50)
            offset = self._parse_int(request.query.get("offset"), 0)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        source = (request.query.get("source") or "").strip() or None
        db = self._get_session_db()
        items = [
            self._normalize_session_record(item)
            for item in db.list_sessions_rich(source=source, limit=limit, offset=offset)
        ]
        total = db.session_count(source=source)
        return web.json_response({"items": items, "total": total})

    async def _handle_create_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions -- create a new session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        title = body.get("title")
        source = str(body.get("source") or "api_server").strip() or "api_server"
        model = body.get("model")
        system_prompt = body.get("system_prompt")
        session_id = f"sess_{uuid.uuid4().hex}"
        db = self._get_session_db()

        try:
            db.create_session(
                session_id=session_id,
                source=source,
                model=model,
                system_prompt=system_prompt,
            )
            if title is not None:
                db.set_session_title(session_id, str(title))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        session = self._normalize_session_record(db.get_session(session_id))
        return web.json_response({"session": session})

    async def _handle_search_sessions(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/search -- search messages across sessions."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        query = (request.query.get("q") or "").strip()
        if not query:
            return web.json_response({"error": "Missing query parameter: q"}, status=400)
        try:
            limit = self._parse_int(request.query.get("limit"), 20)
            offset = self._parse_int(request.query.get("offset"), 0)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)

        results = self._get_session_db().search_messages(query=query, limit=limit, offset=offset)
        return web.json_response({"query": query, "count": len(results), "results": results})

    async def _handle_get_session(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id} -- fetch one session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        session = self._normalize_session_record(self._get_session_db().get_session(session_id))
        if session is None:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"session": session})

    async def _handle_get_session_messages(self, request: "web.Request") -> "web.Response":
        """GET /api/sessions/{session_id}/messages -- fetch session messages."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        if db.get_session(session_id) is None:
            db.ensure_session(session_id, source="web")
        items = db.get_messages(session_id)
        return web.json_response({"items": items, "total": len(items)})

    async def _handle_update_session(self, request: "web.Request") -> "web.Response":
        """PATCH /api/sessions/{session_id} -- update a session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        if db.get_session(session_id) is None:
            return web.json_response({"error": "Session not found"}, status=404)
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        try:
            if "title" in body:
                db.set_session_title(session_id, body.get("title"))
            if "system_prompt" in body:
                db.update_system_prompt(session_id, body.get("system_prompt"))
            if "end_reason" in body:
                db.end_session(session_id, str(body.get("end_reason") or "updated"))
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        session = self._normalize_session_record(db.get_session(session_id))
        return web.json_response({"session": session})

    async def _handle_delete_session(self, request: "web.Request") -> "web.Response":
        """DELETE /api/sessions/{session_id} -- delete a session."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        deleted = self._get_session_db().delete_session(session_id)
        if not deleted:
            return web.json_response({"error": "Session not found"}, status=404)
        return web.json_response({"ok": True})

    async def _handle_fork_session(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/fork -- clone a session and its messages."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        original = db.get_session(session_id)
        if original is None:
            return web.json_response({"error": "Session not found"}, status=404)

        forked_id = f"sess_{uuid.uuid4().hex}"
        try:
            db.create_session(
                session_id=forked_id,
                source=original.get("source") or "api_server",
                model=original.get("model"),
                system_prompt=original.get("system_prompt"),
                user_id=original.get("user_id"),
                parent_session_id=session_id,
            )
            messages = db.get_messages(session_id)
            for message in messages:
                db.append_message(
                    session_id=forked_id,
                    role=message.get("role"),
                    content=message.get("content"),
                    tool_name=message.get("tool_name"),
                    tool_calls=message.get("tool_calls"),
                    tool_call_id=message.get("tool_call_id"),
                    token_count=message.get("token_count"),
                    finish_reason=message.get("finish_reason"),
                    reasoning=message.get("reasoning"),
                )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        session = self._normalize_session_record(db.get_session(forked_id))
        return web.json_response({"session": session, "forked_from": session_id})

    async def _handle_session_chat(self, request: "web.Request") -> "web.Response":
        """POST /api/sessions/{session_id}/chat -- run a session-aware chat turn."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        session = self._normalize_session_record(db.get_session(session_id))
        if session is None:
            db.ensure_session(session_id, source="web")
            session = self._normalize_session_record(db.get_session(session_id))
            if session is None:
                session = {"id": session_id, "title": None}

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        message = body.get("message")
        if not isinstance(message, str):
            return web.json_response({"error": "Missing or invalid 'message' field"}, status=400)

        raw_attachments_sync = body.get("attachments")
        if raw_attachments_sync:
            logger.debug("[chat] Received %d attachment(s): %s",
                         len(raw_attachments_sync),
                         [(a.get("name"), a.get("contentType"), len(a.get("content", "") or a.get("base64", "") or "")) for a in raw_attachments_sync if isinstance(a, dict)])
        user_content, persist_text = self._build_user_content(message, raw_attachments_sync)
        if isinstance(user_content, list):
            logger.debug("[chat] Built multimodal content with %d parts", len(user_content))

        model = body.get("model") or session.get("model") or "hermes-agent"
        system_message = body.get("system_message")
        history = db.get_messages_as_conversation(session_id)
        loop = asyncio.get_event_loop()

        def _run():
            agent = self._create_agent(
                ephemeral_system_prompt=system_message,
                session_id=session_id,
            )
            agent._session_db = db  # Enable session persistence
            result = agent.run_conversation(
                user_content,
                conversation_history=history,
                persist_user_message=persist_text,
            )
            usage = {
                "input_tokens": getattr(agent, "session_prompt_tokens", 0) or 0,
                "output_tokens": getattr(agent, "session_completion_tokens", 0) or 0,
                "total_tokens": getattr(agent, "session_total_tokens", 0) or 0,
            }
            return result, usage

        try:
            result, usage = await loop.run_in_executor(None, _run)
        except Exception as e:
            logger.error("Error running session chat for %s: %s", session_id, e, exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

        return web.json_response({
            "session_id": session_id,
            "run_id": f"run_{uuid.uuid4().hex}",
            "model": model,
            "final_response": result.get("final_response"),
            "completed": result.get("completed", False),
            "partial": result.get("partial", False),
            "interrupted": result.get("interrupted", False),
            "api_calls": result.get("api_calls", 0),
            "messages": result.get("messages", []),
            "last_reasoning": result.get("last_reasoning"),
            "response_previewed": result.get("response_previewed", False),
            "usage": usage,
        })

    async def _handle_session_chat_stream(self, request: "web.Request") -> "web.StreamResponse":
        """POST /api/sessions/{session_id}/chat/stream -- stream a session chat turn over SSE."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_id = request.match_info["session_id"]
        db = self._get_session_db()
        session = self._normalize_session_record(db.get_session(session_id))
        if session is None:
            db.ensure_session(session_id, source="web")
            session = self._normalize_session_record(db.get_session(session_id))
            if session is None:
                session = {"id": session_id, "title": None}

        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        message = body.get("message")
        if not isinstance(message, str):
            return web.json_response({"error": "Missing or invalid 'message' field"}, status=400)

        # Build multimodal content if image attachments are present
        raw_attachments = body.get("attachments")
        if raw_attachments:
            logger.debug("[chat/stream] Received %d attachment(s): %s",
                         len(raw_attachments),
                         [(a.get("name"), a.get("contentType"), len(a.get("content", "") or a.get("base64", "") or "")) for a in raw_attachments if isinstance(a, dict)])
        user_content, persist_text = self._build_user_content(message, raw_attachments)
        if isinstance(user_content, list):
            logger.debug("[chat/stream] Built multimodal content with %d parts", len(user_content))

        system_message = body.get("system_message")
        history = db.get_messages_as_conversation(session_id)
        assistant_message_id = f"msg_asst_{uuid.uuid4().hex}"

        # Note: user message persistence is handled by AIAgent._flush_messages_to_session_db
        # Don't double-persist here or messages will appear twice

        import queue as _q
        stream_q: _q.Queue = _q.Queue()

        def _encode_sse(event_name: str, payload: Dict[str, Any]) -> bytes:
            return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        def _queue_event(event_name: str, payload: Dict[str, Any]) -> None:
            stream_q.put(_encode_sse(event_name, payload))

        def _tool_map(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
            mapping: Dict[str, Dict[str, Any]] = {}
            for item in messages:
                if item.get("role") != "assistant":
                    continue
                for index, tool_call in enumerate(item.get("tool_calls") or []):
                    tool_id = tool_call.get("id")
                    if not tool_id:
                        continue
                    fn = tool_call.get("function") or {}
                    raw_args = fn.get("arguments")
                    try:
                        parsed_args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else {}
                    except json.JSONDecodeError:
                        parsed_args = raw_args
                    mapping[tool_id] = {
                        "tool_name": fn.get("name") or item.get("tool_name") or f"tool_{index + 1}",
                        "args": parsed_args,
                    }
            return mapping

        def _result_preview(content: Any, limit: int = 4000) -> str:
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            return text[:limit] + ("..." if len(text) > limit else "")

        run_id = f"run_{uuid.uuid4().hex}"

        def _on_delta(delta):
            if delta:
                _queue_event(
                    "assistant.delta",
                    {"session_id": session_id, "run_id": run_id, "message_id": assistant_message_id, "delta": delta},
                )

        def _on_tool_progress(event_type, name, preview, args):
            if name == "_thinking":
                _queue_event(
                    "tool.progress",
                    {"session_id": session_id, "run_id": run_id, "message_id": assistant_message_id, "delta": preview},
                )
                return
            payload = {
                "session_id": session_id,
                "run_id": run_id,
                "tool_name": name,
                "preview": preview,
                "args": args,
            }
            _queue_event("tool.started", payload)
            # Also send tool.progress for progress updates
            _queue_event("tool.progress", payload)

        agent_ref = [None]
        loop = asyncio.get_event_loop()

        async def _run_agent_task():
            def _run():
                agent = self._create_agent(
                    ephemeral_system_prompt=system_message,
                    session_id=session_id,
                    stream_delta_callback=_on_delta,
                    tool_progress_callback=_on_tool_progress,
                )
                agent._session_db = db  # Enable session persistence
                agent_ref[0] = agent
                return agent.run_conversation(
                    user_content,
                    conversation_history=history,
                    persist_user_message=persist_text,
                )

            return await loop.run_in_executor(None, _run)

        agent_task = asyncio.ensure_future(_run_agent_task())

        sse_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        origin = request.headers.get("Origin", "")
        cors = self._cors_headers_for_origin(origin) if origin else None
        if cors:
            sse_headers.update(cors)

        response = web.StreamResponse(status=200, headers=sse_headers)
        await response.prepare(request)

        try:
            user_message_id = f"msg_user_{uuid.uuid4().hex}"
            await response.write(_encode_sse("session.created", {
                "session_id": session_id,
                "run_id": run_id,
                "title": session.get("title") or "New Chat",
            }))
            await response.write(_encode_sse("run.started", {
                "session_id": session_id,
                "run_id": run_id,
                "user_message": {
                    "id": user_message_id,
                    "role": "user",
                    "content": message,
                },
            }))
            await response.write(_encode_sse("message.started", {
                "session_id": session_id,
                "run_id": run_id,
                "message": {"id": assistant_message_id, "role": "assistant"},
            }))

            last_activity = time.monotonic()
            while True:
                try:
                    frame = await loop.run_in_executor(None, lambda: stream_q.get(timeout=0.5))
                except _q.Empty:
                    if agent_task.done():
                        while True:
                            try:
                                frame = stream_q.get_nowait()
                                if frame is None:
                                    break
                                await response.write(frame)
                            except _q.Empty:
                                break
                        break
                    # Send periodic keepalive to prevent client/proxy
                    # timeouts during agent init and long LLM API calls.
                    if time.monotonic() - last_activity >= CHAT_COMPLETIONS_SSE_KEEPALIVE_SECONDS:
                        await response.write(b": keepalive\n\n")
                        last_activity = time.monotonic()
                    continue

                if frame is None:
                    break

                await response.write(frame)
                last_activity = time.monotonic()

            try:
                result = await agent_task
            except Exception:
                result = {"messages": [], "final_response": "", "completed": False}
            tools = _tool_map(result.get("messages") or [])
            for item in result.get("messages") or []:
                if item.get("role") != "tool":
                    continue
                tool_id = item.get("tool_call_id")
                tool_meta = tools.get(tool_id, {})
                await response.write(_encode_sse("tool.completed", {
                    "session_id": session_id,
                    "run_id": run_id,
                    "tool_call_id": tool_id,
                    "tool_name": tool_meta.get("tool_name") or item.get("tool_name") or "unknown",
                    "args": tool_meta.get("args"),
                    "result_preview": _result_preview(item.get("content")),
                }))

            await response.write(_encode_sse("assistant.completed", {
                "session_id": session_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "content": result.get("final_response") or "",
                "completed": result.get("completed", False),
                "partial": result.get("partial", False),
                "interrupted": result.get("interrupted", False),
            }))
            await response.write(_encode_sse("run.completed", {
                "session_id": session_id,
                "run_id": run_id,
                "message_id": assistant_message_id,
                "completed": result.get("completed", False),
                "partial": result.get("partial", False),
                "interrupted": result.get("interrupted", False),
                "api_calls": result.get("api_calls"),
            }))
            await response.write(_encode_sse("done", {"session_id": session_id, "run_id": run_id, "state": "final"}))
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, OSError):
            agent = agent_ref[0]
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
            logger.info("Session SSE client disconnected; interrupted session %s", session_id)

        return response

    async def _handle_get_memory(self, request: "web.Request") -> "web.Response":
        """GET /api/memory -- read current memory state."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        target = (request.query.get("target") or "all").strip().lower()
        if target not in {"all", "memory", "user"}:
            return web.json_response({"error": "target must be one of: all, memory, user"}, status=400)

        store = self._get_memory_store()
        store.load_from_disk()
        targets = []
        if target in {"all", "memory"}:
            targets.append({
                "target": "memory",
                "entries": store.memory_entries,
                "entry_count": len(store.memory_entries),
            })
        if target in {"all", "user"}:
            targets.append({
                "target": "user",
                "entries": store.user_entries,
                "entry_count": len(store.user_entries),
            })
        return web.json_response({"targets": targets})

    async def _handle_add_memory(self, request: "web.Request") -> "web.Response":
        """POST /api/memory -- add a memory entry."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        target = str(body.get("target") or "").strip().lower()
        content = str(body.get("content") or "")
        if target not in {"memory", "user"}:
            return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
        result = self._get_memory_store().add(target, content)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)

    async def _handle_replace_memory(self, request: "web.Request") -> "web.Response":
        """PATCH /api/memory -- replace a memory entry."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        target = str(body.get("target") or "").strip().lower()
        old_text = str(body.get("old_text") or "")
        content = str(body.get("content") or "")
        if target not in {"memory", "user"}:
            return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
        result = self._get_memory_store().replace(target, old_text, content)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)

    async def _handle_delete_memory(self, request: "web.Request") -> "web.Response":
        """DELETE /api/memory -- delete a memory entry."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        target = str(body.get("target") or "").strip().lower()
        old_text = str(body.get("old_text") or "")
        if target not in {"memory", "user"}:
            return web.json_response({"error": "target must be 'memory' or 'user'"}, status=400)
        result = self._get_memory_store().remove(target, old_text)
        status = 200 if result.get("success") else 400
        return web.json_response(result, status=status)

    async def _handle_list_skills(self, request: "web.Request") -> "web.Response":
        """GET /api/skills -- list skills."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        category = (request.query.get("category") or "").strip() or None
        return web.json_response(json.loads(skills_list(category=category)))

    async def _handle_view_skill(self, request: "web.Request") -> "web.Response":
        """GET /api/skills/{name} -- fetch skill details."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        file_path = (request.query.get("file_path") or "").strip() or None
        return web.json_response(json.loads(skill_view(name, file_path=file_path)))

    async def _handle_get_config(self, request: "web.Request") -> "web.Response":
        """GET /api/config -- fetch the current config."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        config = load_config()
        current = self._current_model_settings(config)
        return web.json_response({
            "model": current["model"],
            "provider": current["provider"],
            "api_mode": current["api_mode"],
            "base_url": current["base_url"],
            "config": config,
        })

    async def _handle_update_config(self, request: "web.Request") -> "web.Response":
        """PATCH /api/config -- update model/provider/base_url settings."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response({"error": "Invalid JSON in request body"}, status=400)

        config = load_config()
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            updated_model_cfg = dict(model_cfg)
        elif isinstance(model_cfg, str) and model_cfg.strip():
            updated_model_cfg = {"default": model_cfg.strip()}
        else:
            updated_model_cfg = {}

        if "model" in body:
            updated_model_cfg["default"] = str(body.get("model") or "").strip()
        if "provider" in body:
            updated_model_cfg["provider"] = str(body.get("provider") or "").strip()
        if "base_url" in body:
            updated_model_cfg["base_url"] = str(body.get("base_url") or "").strip()

        config["model"] = updated_model_cfg
        try:
            save_config(config)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

        current = self._current_model_settings(config)
        return web.json_response({
            "ok": True,
            "model": current["model"],
            "provider": current["provider"],
            "base_url": current["base_url"],
        })

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        """GET /v1/capabilities -- list available toolsets and reasoning modes."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        # Collect available toolsets from config.yaml
        from hermes_cli.tools_config import _get_platform_tools
        from hermes_cli.config import load_config

        config = load_config()
        toolsets = sorted(_get_platform_tools(config, "api_server"))

        # Reasoning modes
        reasoning_modes = ["disabled", "enabled", "auto"]

        return web.json_response({
            "object": "hermes.api_server.capabilities",
            "platform": "hermes-agent",
            "model": "hermes-agent",
            "auth": {"type": "bearer", "required": bool(self._api_key)},
            "runtime": {
                "mode": "server_agent",
                "version": "0.13.0",
                "tool_execution": "server",
                "split_runtime": False,
                "description": "API-server host for Hermes Agent",
            },
            "features": {
                "chat_completions": True,
                "run_status": True,
                "run_events_sse": True,
                "session_continuity_header": "X-Hermes-Session-Id",
                "session_key_header": "X-Hermes-Session-Key",
            },
            "endpoints": {
                "run_status": {"path": "/v1/runs/{run_id}"},
            },
            "toolsets": toolsets,
            "reasoning_modes": reasoning_modes,
            "supports_streaming": True,
            "supports_multimodal": True,
            "supports_responses_api": True,
            "supports_runs_api": True,
        })

    async def _handle_available_models(self, request: "web.Request") -> "web.Response":
        """GET /api/available-models -- list provider models and available providers."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        config = load_config()
        current = self._current_model_settings(config)
        provider = (request.query.get("provider") or current["provider"] or "openrouter").strip()
        models = [
            {"id": model_id, "description": description}
            for model_id, description in curated_models_for_provider(provider)
        ]
        providers = list_available_providers()
        return web.json_response({"provider": provider, "models": models, "providers": providers})


    async def _handle_chat_completions(self, request: "web.Request") -> "web.Response":
        """POST /v1/chat/completions -- OpenAI Chat Completions format."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        session_key, err = self._parse_session_key_header(request)
        if err:
            return err

        # Parse request body
        try:
            body = await request.json()
        except (json.JSONDecodeError, Exception):
            return web.json_response(_openai_error("Invalid JSON in request body"), status=400)

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return web.json_response(
                {"error": {"message": "Missing or invalid 'messages' field", "type": "invalid_request_error"}},
                status=400,
            )

        # Fast-path for capability probes (max_tokens=1)
        # Return a minimal valid response so frontends detect the endpoint
        # without spinning up a full agent.
        max_tokens = body.get("max_tokens")
        if max_tokens == 1:
            return web.json_response({
                "id": f"chatcmpl-probe-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "") or "hermes-agent",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
            })

        stream = body.get("stream", False)

        # Extract system message (becomes ephemeral system prompt layered ON TOP of core)
        system_prompt = None
        conversation_messages: List[Dict[str, str]] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            if role == "system":
                # System messages don't support images (Anthropic rejects, OpenAI
                # text-model systems don't render them).  Flatten to text.
                content = _normalize_chat_content(raw_content)
                if system_prompt is None:
                    system_prompt = content
                else:
                    system_prompt = system_prompt + "\n" + content
            elif role in ("user", "assistant"):
                try:
                    content = _normalize_multimodal_content(raw_content)
                except ValueError as exc:
                    return _multimodal_validation_error(exc, param=f"messages[{idx}].content")
                conversation_messages.append({"role": role, "content": content})

        # Extract the last user message as the primary input
        user_message: Any = ""
        history = []
        if conversation_messages:
            user_message = conversation_messages[-1].get("content", "")
            history = conversation_messages[:-1]

        if not _content_has_visible_payload(user_message):
            return web.json_response(
                {"error": {"message": "No user message found in messages", "type": "invalid_request_error"}},
                status=400,
            )

        # Allow caller to continue an existing session by passing X-Hermes-Session-Id.
        # When provided, history is loaded from state.db instead of from the request body.
        #
        # Security: session continuation exposes conversation history, so it is
        # only allowed when the API key is configured and the request is
        # authenticated.  Without this gate, any unauthenticated client could
        # read arbitrary session history by guessing/enumerating session IDs.
        provided_session_id = request.headers.get("X-Hermes-Session-Id", "").strip()
        if provided_session_id:
            if not self._api_key:
                logger.warning(
                    "Session continuation via X-Hermes-Session-Id rejected: "
                    "no API key configured.  Set API_SERVER_KEY to enable "
                    "session continuity."
                )
                return web.json_response(
                    _openai_error(
                        "Session continuation requires API key authentication. "
                        "Configure API_SERVER_KEY to enable this feature."
                    ),
                    status=403,
                )
            # Sanitize: reject control characters that could enable header injection.
            if re.search(r'[\r\n\x00]', provided_session_id):
                return web.json_response(
                    {"error": {"message": "Invalid session ID", "type": "invalid_request_error"}},
                    status=400,
                )
            session_id = provided_session_id
            try:
                db = self._ensure_session_db()
                if db is not None:
                    history = db.get_messages_as_conversation(session_id)
            except Exception as e:
                logger.warning("Failed to load session history for %s: %s", session_id, e)
                history = []
        else:
            # Derive a stable session ID from the conversation fingerprint so
            # that consecutive messages from the same Open WebUI (or similar)
            # conversation map to the same Hermes session.  The first user
            # message + system prompt are constant across all turns.
            first_user = ""
            for cm in conversation_messages:
                if cm.get("role") == "user":
                    first_user = cm.get("content", "")
                    break
            session_id = _derive_chat_session_id(system_prompt, first_user)
            # history already set from request body above

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        model_name = body.get("model", self._model_name)
        created = int(time.time())

        if stream:
            import queue as _q
            _stream_q: _q.Queue = _q.Queue()

            def _on_delta(delta):
                # Filter out None -- the agent fires stream_delta_callback(None)
                # to signal the CLI display to close its response box before
                # tool execution, but the SSE writer uses None as end-of-stream
                # sentinel.  Forwarding it would prematurely close the HTTP
                # response, causing Open WebUI (and similar frontends) to miss
                # the final answer after tool calls.  The SSE loop detects
                # completion via agent_task.done() instead.
                if delta is not None:
                    _stream_q.put(delta)

            # Track which tool_call_ids we've emitted a "running" lifecycle
            # event for, so a "completed" event without a matching "running"
            # (e.g. internal/filtered tools) is silently dropped instead of
            # producing an orphaned event clients can't correlate.
            _started_tool_call_ids: set[str] = set()

            def _on_tool_start(tool_call_id, function_name, function_args):
                """Emit ``hermes.tool.progress`` with ``status: running``.

                Replaces the old ``tool_progress_callback("tool.started",
                ...)`` emit so SSE consumers receive a single event per
                tool start, carrying both the legacy ``tool``/``emoji``/
                ``label`` payload (for #6972 frontends) and the new
                ``toolCallId``/``status`` correlation fields (#16588).

                Skips tools whose names start with ``_`` so internal
                events (``_thinking``, …) stay off the wire -- matching
                the prior ``_on_tool_progress`` filter exactly.
                """
                if not tool_call_id or function_name.startswith("_"):
                    return
                _started_tool_call_ids.add(tool_call_id)
                from agent.display import build_tool_preview, get_tool_emoji
                label = build_tool_preview(function_name, function_args) or function_name
                _stream_q.put(("__tool_progress__", {
                    "tool": function_name,
                    "emoji": get_tool_emoji(function_name),
                    "label": label,
                    "toolCallId": tool_call_id,
                    "status": "running",
                }))

            def _on_tool_complete(tool_call_id, function_name, function_args, function_result):
                """Emit the matching ``status: completed`` event.

                Dropped if the start was filtered (internal tool, missing
                id, or never seen) so clients never get an orphaned
                ``completed`` they can't correlate to a prior ``running``.
                """
                if not tool_call_id or tool_call_id not in _started_tool_call_ids:
                    return
                _started_tool_call_ids.discard(tool_call_id)
                _stream_q.put(("__tool_progress__", {
                    "tool": function_name,
                    "toolCallId": tool_call_id,
                    "status": "completed",
                }))

            # Start agent in background.  agent_ref is a mutable container
            # so the SSE writer can interrupt the agent on client disconnect.
            #
            # ``tool_progress_callback`` is intentionally not wired here:
            # it would duplicate every emit because ``run_agent`` fires it
            # side-by-side with ``tool_start_callback``/``tool_complete_callback``.
            # The structured callbacks are strictly richer (they carry the
            # tool_call id), so they own the chat-completions SSE channel.
            agent_ref = [None]
            agent_task = asyncio.ensure_future(self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                gateway_session_key=request.headers.get("X-Hermes-Session-Key"),
                stream_delta_callback=_on_delta,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                agent_ref=agent_ref,
            ))

            return await self._write_sse_chat_completion(
                request, completion_id, model_name, created, _stream_q,
                agent_task, agent_ref, session_id=session_id,
            )

        # Non-streaming: run the agent (with optional Idempotency-Key)
        async def _compute_completion():
            return await self._run_agent(
                user_message=user_message,
                conversation_history=history,
                ephemeral_system_prompt=system_prompt,
                session_id=session_id,
                gateway_session_key=request.headers.get("X-Hermes-Session-Key"),
            )

        idempotency_key = request.headers.get("Idempotency-Key")
        if idempotency_key:
            fp = _make_request_fingerprint(body, keys=["model", "messages", "tools", "tool_choice", "stream"])
            try:
                result, usage = await _idem_cache.get_or_set(idempotency_key, fp, _compute_completion)
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )
        else:
            try:
                result, usage = await _compute_completion()
            except Exception as e:
                logger.error("Error running agent for chat completions: %s", e, exc_info=True)
                return web.json_response(
                    _openai_error(f"Internal server error: {e}", err_type="server_error"),
                    status=500,
                )

        final_response = result.get("final_response", "")
        if not final_response:
            final_response = result.get("error", "(No response generated)")

        response_data = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

        headers = {"X-Hermes-Session-Id": session_id}
        session_key = request.headers.get("X-Hermes-Session-Key")
        if session_key:
            headers["X-Hermes-Session-Key"] = session_key
        return web.json_response(response_data, headers=headers)

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
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        stored = self._response_store.get(response_id)
        if stored is None:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response(stored["response"])

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} -- delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    async def _handle_delete_response(self, request: "web.Request") -> "web.Response":
        """DELETE /v1/responses/{response_id} -- delete a stored response."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err

        response_id = request.match_info["response_id"]
        deleted = self._response_store.delete(response_id)
        if not deleted:
            return web.json_response(_openai_error(f"Response not found: {response_id}"), status=404)

        return web.json_response({
            "id": response_id,
            "object": "response",
            "deleted": True,
        })

    # ------------------------------------------------------------------
    # Cron jobs API
    # ------------------------------------------------------------------

    _JOB_ID_RE = __import__("re").compile(r"[a-f0-9]{12}")
    # Allowed fields for update -- prevents clients injecting arbitrary keys
    _UPDATE_ALLOWED_FIELDS = {"name", "schedule", "prompt", "deliver", "skills", "skill", "repeat", "enabled"}
    _MAX_NAME_LENGTH = 200
    _MAX_PROMPT_LENGTH = 5000

    @staticmethod
    def _check_jobs_available() -> Optional["web.Response"]:
        """Return error response if cron module isn't available."""
        if not _CRON_AVAILABLE:
            return web.json_response(
                {"error": "Cron module not available"}, status=501,
            )
        return None

    def _check_job_id(self, request: "web.Request") -> tuple:
        """Validate and extract job_id. Returns (job_id, error_response)."""
        job_id = request.match_info["job_id"]
        if not self._JOB_ID_RE.fullmatch(job_id):
            return job_id, web.json_response(
                {"error": "Invalid job ID format"}, status=400,
            )
        return job_id, None

    async def _handle_list_jobs(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs -- list all cron jobs."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            include_disabled = request.query.get("include_disabled", "").lower() in ("true", "1")
            jobs = _cron_list(include_disabled=include_disabled)
            return web.json_response({"jobs": jobs})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_create_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs -- create a new cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        try:
            body = await request.json()
            name = (body.get("name") or "").strip()
            schedule = (body.get("schedule") or "").strip()
            prompt = body.get("prompt", "")
            deliver = body.get("deliver", "local")
            skills = body.get("skills")
            repeat = body.get("repeat")

            if not name:
                return web.json_response({"error": "Name is required"}, status=400)
            if len(name) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if not schedule:
                return web.json_response({"error": "Schedule is required"}, status=400)
            if len(prompt) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            if repeat is not None and (not isinstance(repeat, int) or repeat < 1):
                return web.json_response({"error": "Repeat must be a positive integer"}, status=400)

            kwargs = {
                "prompt": prompt,
                "schedule": schedule,
                "name": name,
                "deliver": deliver,
            }
            if skills:
                kwargs["skills"] = skills
            if repeat is not None:
                kwargs["repeat"] = repeat

            job = _cron_create(**kwargs)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_job(self, request: "web.Request") -> "web.Response":
        """GET /api/jobs/{job_id} -- get a single cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_get(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_update_job(self, request: "web.Request") -> "web.Response":
        """PATCH /api/jobs/{job_id} -- update a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            body = await request.json()
            # Whitelist allowed fields to prevent arbitrary key injection
            sanitized = {k: v for k, v in body.items() if k in self._UPDATE_ALLOWED_FIELDS}
            if not sanitized:
                return web.json_response({"error": "No valid fields to update"}, status=400)
            # Validate lengths if present
            if "name" in sanitized and len(sanitized["name"]) > self._MAX_NAME_LENGTH:
                return web.json_response(
                    {"error": f"Name must be ≤ {self._MAX_NAME_LENGTH} characters"}, status=400,
                )
            if "prompt" in sanitized and len(sanitized["prompt"]) > self._MAX_PROMPT_LENGTH:
                return web.json_response(
                    {"error": f"Prompt must be ≤ {self._MAX_PROMPT_LENGTH} characters"}, status=400,
                )
            job = _cron_update(job_id, sanitized)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_delete_job(self, request: "web.Request") -> "web.Response":
        """DELETE /api/jobs/{job_id} -- delete a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            success = _cron_remove(job_id)
            if not success:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"ok": True})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_pause_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/pause -- pause a cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_pause(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_resume_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/resume -- resume a paused cron job."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_resume(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_run_job(self, request: "web.Request") -> "web.Response":
        """POST /api/jobs/{job_id}/run -- trigger immediate execution."""
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        cron_err = self._check_jobs_available()
        if cron_err:
            return cron_err
        job_id, id_err = self._check_job_id(request)
        if id_err:
            return id_err
        try:
            job = _cron_trigger(job_id)
            if not job:
                return web.json_response({"error": "Job not found"}, status=404)
            return web.json_response({"job": job})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # ------------------------------------------------------------------
    # Read-only / metadata APIs for web UIs
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_session_record(record: Dict[str, Any]) -> Dict[str, Any]:
        if record is None:
            return None
        return {
            "id": record.get("id"),
            "source": record.get("source"),
            "user_id": record.get("user_id"),
            "model": record.get("model"),
            "title": record.get("title"),
            "started_at": record.get("started_at"),
            "ended_at": record.get("ended_at"),
            "end_reason": record.get("end_reason"),
            "message_count": record.get("message_count") or 0,
            "tool_call_count": record.get("tool_call_count") or 0,
            "input_tokens": record.get("input_tokens") or 0,
            "output_tokens": record.get("output_tokens") or 0,
            "last_active": record.get("last_active"),
            "parent_session_id": record.get("parent_session_id"),
        }

    async def _handle_list_sessions(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            db = self._ensure_session_db()
            if not db:
                return web.json_response({"items": [], "total": 0})
            limit = max(1, min(500, int(request.query.get("limit", "50"))))
            offset = max(0, int(request.query.get("offset", "0")))
            items = db.list_sessions_rich(limit=limit, offset=offset)
            total = db.session_count()
            return web.json_response({
                "items": [self._normalize_session_record(item) for item in items],
                "total": total,
            })
        except Exception as e:
            logger.exception("Error listing sessions")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_session(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            db = self._ensure_session_db()
            if not db:
                return web.json_response({"error": "Session DB unavailable"}, status=503)
            session_id = request.match_info.get("session_id", "")
            resolved = db.resolve_session_id(session_id) or session_id
            item = db.get_session(resolved)
            if not item:
                return web.json_response({"error": "Session not found"}, status=404)
            return web.json_response({"session": self._normalize_session_record(item)})
        except Exception as e:
            logger.exception("Error getting session")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_create_session(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            body = await request.json() if request.can_read_body else {}
            requested_id = str(body.get("id") or "").strip()
            session_id = requested_id or str(uuid.uuid4())
            title = str(body.get("title") or "").strip() or None
            model = str(body.get("model") or "").strip() or None
            db = self._ensure_session_db()
            if not db:
                return web.json_response({"error": "Session DB unavailable"}, status=503)
            created_id = db.create_session(session_id=session_id, source="api_server", model=model)
            if title:
                try:
                    db.set_session_title(created_id, title)
                except Exception:
                    pass
            item = db.get_session(created_id) or {"id": created_id, "model": model, "title": title, "started_at": time.time()}
            return web.json_response({"session": self._normalize_session_record(item)})
        except Exception as e:
            logger.exception("Error creating session")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_update_session(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            db = self._ensure_session_db()
            if not db:
                return web.json_response({"error": "Session DB unavailable"}, status=503)
            session_id = request.match_info.get("session_id", "")
            resolved = db.resolve_session_id(session_id) or session_id
            body = await request.json() if request.can_read_body else {}
            title = str(body.get("title") or "").strip()
            if not title:
                return web.json_response({"error": "title required"}, status=400)
            ok = db.set_session_title(resolved, title)
            if not ok:
                return web.json_response({"error": "Session not found"}, status=404)
            item = db.get_session(resolved)
            return web.json_response({"session": self._normalize_session_record(item or {"id": resolved, "title": title})})
        except Exception as e:
            logger.exception("Error updating session")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_delete_session(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            db = self._ensure_session_db()
            if not db:
                return web.json_response({"error": "Session DB unavailable"}, status=503)
            session_id = request.match_info.get("session_id", "")
            resolved = db.resolve_session_id(session_id) or session_id
            ok = db.delete_session(resolved)
            if not ok:
                return web.json_response({"error": "Session not found"}, status=404)
            return web.json_response({"ok": True, "session_id": resolved})
        except Exception as e:
            logger.exception("Error deleting session")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_session_messages(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            db = self._ensure_session_db()
            if not db:
                return web.json_response({"items": [], "total": 0})
            session_id = request.match_info.get("session_id", "")
            resolved = db.resolve_session_id(session_id) or session_id
            items = db.get_messages(resolved)
            return web.json_response({"items": items, "total": len(items)})
        except Exception as e:
            logger.exception("Error getting session messages")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_memory(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            from tools.memory_tool import get_memory_dir
            mem_dir = get_memory_dir()
            result = {}
            for name in ("MEMORY.md", "USER.md"):
                path = mem_dir / name
                if path.exists():
                    result[name] = path.read_text(encoding="utf-8")
            return web.json_response(result)
        except Exception as e:
            logger.exception("Error reading memory")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_list_skills(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            import yaml
            from agent.skill_utils import get_all_skills_dirs, iter_skill_index_files

            skills = []
            for skills_dir in get_all_skills_dirs():
                if not skills_dir.is_dir():
                    continue
                for skill_path in iter_skill_index_files(skills_dir, "SKILL.md"):
                    try:
                        raw = skill_path.read_text(encoding="utf-8")
                        frontmatter = {}
                        if raw.startswith("---"):
                            parts = raw.split("---", 2)
                            if len(parts) >= 3:
                                frontmatter = yaml.safe_load(parts[1]) or {}
                        skill_dir = skill_path.parent
                        name = frontmatter.get("name") or skill_dir.name
                        skills.append({
                            "id": name,
                            "name": name,
                            "description": frontmatter.get("description", ""),
                            "author": frontmatter.get("author", ""),
                            "tags": frontmatter.get("tags", []),
                            "triggers": frontmatter.get("triggers", []),
                            "category": frontmatter.get("category", ""),
                            "installed": True,
                            "enabled": True,
                            "sourcePath": str(skill_path),
                            "content": raw,
                        })
                    except Exception as skill_err:
                        logger.debug("Error reading skill %s: %s", skill_path, skill_err)
            return web.json_response({"items": skills, "total": len(skills)})
        except Exception as e:
            logger.exception("Error listing skills")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_get_config(self, request: "web.Request") -> "web.Response":
        auth_err = self._check_auth(request)
        if auth_err:
            return auth_err
        try:
            import copy
            import yaml

            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                return web.json_response({})
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            safe_keys = [
                "model", "provider", "display", "memory", "timezone",
                "skills", "toolsets", "agent", "tts", "stt",
                "smart_model_routing", "custom_providers",
            ]
            safe_config = {k: copy.deepcopy(config[k]) for k in safe_keys if k in config}
            if "custom_providers" in safe_config and isinstance(safe_config["custom_providers"], list):
                for provider in safe_config["custom_providers"]:
                    if isinstance(provider, dict):
                        provider.pop("api_key", None)
            return web.json_response(safe_config)
        except Exception as e:
            logger.exception("Error reading config")
            return web.json_response({"error": str(e)}, status=500)

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
                    "preview": preview,
                })
            elif event_type == "tool.completed":
                _push({
                    "event": "tool.completed",
                    "run_id": run_id,
                    "timestamp": ts,
                    "tool": tool_name,
                    "duration": round(kwargs.get("duration", 0), 3),
                    "error": kwargs.get("is_error", False),
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
        return await self._handle_ws_real(request)

    async def _handle_ws_real(self, request: "web.Request") -> "web.WebSocketResponse":
        """
        WebSocket endpoint for remote node connections.

        OpenClaw-style protocol:
        1. Node sends {type:"req", method:"connect", params:{role:"node", ...}}
        2. Gateway responds {type:"res", ok:true, payload:{type:"hello-ok", ...}}
        3. Gateway sends {type:"event", event:"node.invoke.request", payload:{...}}
        4. Node responds {type:"event", event:"node.invoke.result", payload:{...}}
        """
        if not AIOHTTP_AVAILABLE or web is None:
            return web.Response(status=503, text="aiohttp not available")

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        node_session = None
        node_id = None

        try:
            async for msg in ws:
                if msg.type != web.WSMsgType.TEXT:
                    continue

                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                event = data.get("event")

                # --- Handshake ---
                if msg_type == "req" and data.get("method") == "connect":
                    params = data.get("params", {})
                    role = params.get("role", "")
                    if role != "node":
                        await ws.send_str(json.dumps({
                            "type": "res",
                            "id": data.get("id"),
                            "ok": False,
                            "error": {"message": "Only 'node' role is supported on /ws"},
                        }))
                        await ws.close()
                        return ws

                    node_id = params.get("client", {}).get("id", "unknown")
                    caps = params.get("caps", [])
                    commands = params.get("commands", [])
                    platform = params.get("client", {}).get("platform", "unknown")
                    version = params.get("client", {}).get("version", "unknown")

                    from api_server.node_registry import NodeSession

                    def send_fn(payload: Dict[str, Any]) -> None:
                        asyncio.create_task(ws.send_str(json.dumps(payload)))

                    node_session = NodeSession(
                        node_id=node_id,
                        send_fn=send_fn,
                        caps=caps,
                        commands=commands,
                        platform=platform,
                        version=version,
                    )
                    await NODE_REGISTRY.register(node_session)

                    await ws.send_str(json.dumps({
                        "type": "res",
                        "id": data.get("id"),
                        "ok": True,
                        "payload": {
                            "type": "hello-ok",
                            "protocol": 1,
                            "policy": {
                                "maxPayload": 26214400,
                                "tickIntervalMs": 15000,
                            },
                        },
                    }))
                    continue

                # --- Invoke result from node ---
                if msg_type == "event" and event == "node.invoke.result":
                    payload = data.get("payload", {})
                    request_id = payload.get("id")
                    ok = payload.get("ok", False)
                    result_payload = payload.get("payload")
                    error = payload.get("error")
                    NODE_REGISTRY.handle_result(request_id, ok, result_payload, error)
                    continue

        except Exception as exc:
            logger.warning("[Node WS] Connection error for %s: %s", node_id, exc)
        finally:
            if node_id:
                await NODE_REGISTRY.unregister(node_id)
            if not ws.closed:
                await ws.close()

        return ws

    # ------------------------------------------------------------------
    # Node HTTP API handlers
    # ------------------------------------------------------------------

    async def _handle_list_nodes(self, request: "web.Request") -> "web.Response":
        """GET /v1/nodes -- list all connected remote nodes."""
        nodes = NODE_REGISTRY.list_nodes()
        return web.json_response({"ok": True, "nodes": nodes})

    async def _handle_node_invoke(self, request: "web.Request") -> "web.Response":
        """POST /v1/nodes/{node_id}/invoke -- invoke a command on a remote node."""
        node_id = request.match_info.get("node_id", "")
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"ok": False, "error": {"message": "Invalid JSON body"}}, status=400)

        command = body.get("command", "")
        params = body.get("params", {})
        timeout_ms = body.get("timeoutMs", 30000)
        idempotency_key = body.get("idempotencyKey")

        result = await NODE_REGISTRY.invoke(
            node_id=node_id,
            command=command,
            params=params,
            timeout_ms=timeout_ms,
            idempotency_key=idempotency_key,
        )
        return web.json_response(result)
