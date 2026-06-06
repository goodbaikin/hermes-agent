"""Mem0 memory plugin — MemoryProvider interface.

Supports two modes:
  - cloud: Mem0 Platform API with server-side LLM fact extraction
  - local: Self-hosted with Ollama embeddings + ChromaDB + configurable LLM

Config via $HERMES_HOME/mem0.json:
  mode: "cloud" | "local"
  api_key: Mem0 Platform API key (cloud mode)
  user_id: User identifier
  agent_id: Agent identifier
  rerank: Enable reranking (cloud mode)
  ollama_base_url: Ollama server URL (local mode, default http://localhost:11434)
  embedding_model: Embedding model name (local mode, default nomic-embed-text:latest)
  llm_model: LLM model name for fact extraction (local mode)
  llm_base_url: OpenAI-compatible API base URL (local mode)
  llm_api_key_env: Env var name for LLM API key (local mode)
  db_path: ChromaDB storage path (local mode, default $HERMES_HOME/mem0-local-db)
  collection_name: ChromaDB collection name (local mode, default hermes-memory)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides."""
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "mode": "cloud",
        "rerank": True,
        "keyword_search": False,
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# Local mode helpers
# ---------------------------------------------------------------------------

def _build_local_config(cfg: dict) -> dict:
    """Build mem0 config dict for local Memory.from_config()."""
    from hermes_constants import get_hermes_home

    db_path = cfg.get("db_path", str(get_hermes_home() / "mem0-local-db"))
    collection = cfg.get("collection_name", "hermes-memory")
    ollama_url = cfg.get("ollama_base_url", "http://localhost:11434")
    embed_model = cfg.get("embedding_model", "nomic-embed-text:latest")
    llm_model = cfg.get("llm_model", "glm-5.1")
    llm_base_url = cfg.get("llm_base_url", "https://api.z.ai/api/coding/paas/v4")

    # Resolve API key: env var name or direct value
    api_key_env = cfg.get("llm_api_key_env", "GLM_API_KEY")
    api_key = os.environ.get(api_key_env, cfg.get("llm_api_key", ""))

    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": collection,
                "path": db_path,
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "temperature": 0,
                "max_tokens": 1500,
                "openai_base_url": llm_base_url,
                "api_key": api_key,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": embed_model,
                "ollama_base_url": ollama_url,
            },
        },
    }


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with cloud and local mode support."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._mode = "cloud"
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        mode = cfg.get("mode", "cloud")
        if mode == "local":
            # Local mode: just need Ollama running
            return True
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "mode", "description": "Memory mode: cloud or local", "default": "cloud", "choices": ["cloud", "local"]},
            {"key": "api_key", "description": "Mem0 Platform API key (cloud mode)", "secret": True, "required": False, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall (cloud mode)", "default": "true", "choices": ["true", "false"]},
            {"key": "llm_model", "description": "LLM model for fact extraction (local mode)", "default": "glm-5.1"},
            {"key": "llm_base_url", "description": "LLM API base URL (local mode)", "default": "https://api.z.ai/api/coding/paas/v4"},
            {"key": "llm_api_key_env", "description": "Env var name for LLM API key (local mode)", "default": "GLM_API_KEY"},
            {"key": "embedding_model", "description": "Embedding model (local mode)", "default": "nomic-embed-text:latest"},
            {"key": "ollama_base_url", "description": "Ollama server URL (local mode)", "default": "http://localhost:11434"},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                if self._mode == "local":
                    from mem0 import Memory
                    local_cfg = _build_local_config(self._config)
                    # mem0's OpenAILLM hijacks OPENROUTER_API_KEY if present,
                    # ignoring the configured api_key/base_url.  Temporarily
                    # hide it so mem0 uses the correct LLM endpoint.
                    _or_key = os.environ.pop("OPENROUTER_API_KEY", None)
                    try:
                        self._client = Memory.from_config(local_cfg)
                    finally:
                        if _or_key is not None:
                            os.environ["OPENROUTER_API_KEY"] = _or_key
                    logger.info("Mem0 local mode initialized. LLM=%s, embed=%s",
                                self._config.get("llm_model", "glm-5.1"),
                                self._config.get("embedding_model", "nomic-embed-text:latest"))
                else:
                    from mem0 import MemoryClient
                    self._client = MemoryClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = self._config.get("mode", "cloud")
        self._api_key = self._config.get("api_key", "")
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)

    def _read_kwargs(self) -> Dict[str, Any]:
        """Kwargs for search/get_all — user-scoped."""
        if self._mode == "local":
            return {"user_id": self._user_id}
        return {"filters": {"user_id": self._user_id}}

    def _write_kwargs(self) -> Dict[str, Any]:
        """Kwargs for add — user + agent scoped."""
        if self._mode == "local":
            return {"user_id": self._user_id, "agent_id": self._agent_id}
        return {"filters": {"user_id": self._user_id, "agent_id": self._agent_id}}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — both cloud and local wrap in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        mode_str = "Local" if self._mode == "local" else "Cloud"
        return (
            "# Mem0 Memory\n"
            f"Active ({mode_str}). User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                kwargs = self._read_kwargs()
                if self._mode == "local":
                    results = self._unwrap_results(client.search(
                        query=query, limit=5, **kwargs,
                    ))
                else:
                    results = self._unwrap_results(client.search(
                        query=query, rerank=self._rerank, top_k=5, **kwargs,
                    ))
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                write_kwargs = self._write_kwargs()
                client.add(messages, **write_kwargs)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                read_kwargs = self._read_kwargs()
                if self._mode == "local":
                    memories = self._unwrap_results(client.get_all(**read_kwargs))
                else:
                    memories = self._unwrap_results(client.get_all(**read_kwargs))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                read_kwargs = self._read_kwargs()
                if self._mode == "local":
                    results = self._unwrap_results(client.search(
                        query=query, limit=top_k, **read_kwargs,
                    ))
                else:
                    results = self._unwrap_results(client.search(
                        query=query, rerank=rerank, top_k=top_k, **read_kwargs,
                    ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                write_kwargs = self._write_kwargs()
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **write_kwargs,
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
