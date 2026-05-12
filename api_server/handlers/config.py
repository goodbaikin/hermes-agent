import json
from typing import Any, Dict
from aiohttp import web

from hermes_cli.config import load_config, save_config


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


async def handle_get_config(request: web.Request, *, check_auth) -> web.Response:
    """GET /api/config -- fetch the current config."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    config = load_config()
    current = _current_model_settings(config)
    return web.json_response({
        "model": current["model"],
        "provider": current["provider"],
        "api_mode": current["api_mode"],
        "base_url": current["base_url"],
        "config": config,
    })


async def handle_update_config(request: web.Request, *, check_auth) -> web.Response:
    """PATCH /api/config -- update model/provider/base_url settings."""
    auth_err = check_auth(request)
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

    current = _current_model_settings(config)
    return web.json_response({
        "ok": True,
        "model": current["model"],
        "provider": current["provider"],
        "base_url": current["base_url"],
    })


from hermes_cli.models import curated_models_for_provider, list_available_providers
import time
import asyncio

# Cache for available-models to avoid repeated slow API calls
_models_cache: dict[str, Any] | None = None
_models_cache_at: float = 0.0
_MODELS_CACHE_TTL: float = 86400.0  # 24 hours


async def _get_cached_models(provider: str) -> dict[str, Any]:
    """Return cached model list, refreshing in background if stale."""
    global _models_cache, _models_cache_at
    now = time.monotonic()
    if _models_cache is not None and (now - _models_cache_at) < _MODELS_CACHE_TTL:
        return _models_cache
    # Refresh cache
    models = [
        {"id": model_id, "description": description, "context_window": _get_model_context_window(model_id)}
        for model_id, description in curated_models_for_provider(provider)
    ]
    providers = list_available_providers()
    _models_cache = {"provider": provider, "models": models, "providers": providers}
    _models_cache_at = now
    return _models_cache


# Model ID → context window (approximate, in tokens)
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Anthropic
    "claude-opus-4-7": 200000,
    "claude-opus-4-6": 200000,
    "claude-opus-4-5-20251101": 200000,
    "claude-opus-4-20250514": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-5-20250929": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    # OpenAI
    "gpt-5-5": 128000,
    "gpt-5-4": 128000,
    "gpt-5-4-mini": 128000,
    "gpt-5-3-codex": 200000,
    "gpt-5-2-codex": 200000,
    "gpt-5-mini": 128000,
    "gpt-5-nano": 128000,
    "gpt-4-1": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    # Kimi
    "kimi-k2-6": 256000,
    "kimi-k2-5": 256000,
    "kimi-for-coding": 256000,
    "kimi-k2-thinking": 256000,
    "kimi-k2-thinking-turbo": 256000,
    "kimi-k2-turbo-preview": 256000,
    "kimi-k2-0905-preview": 256000,
    # DeepSeek
    "deepseek-v4-pro": 64000,
    "deepseek-v4-flash": 64000,
    "deepseek-chat": 64000,
    "deepseek-reasoner": 64000,
    # Google
    "gemini-3-pro-preview": 1000000,
    "gemini-3-flash-preview": 1000000,
    "gemini-3-1-pro-preview": 1000000,
    "gemini-3-1-flash-lite-preview": 1000000,
    # Qwen
    "qwen3-6-plus": 128000,
    "qwen3-5-plus-02-15": 128000,
    "qwen3-5-35b-a3b": 128000,
    # MiniMax
    "minimax-m2-7": 256000,
    "minimax-m2-5": 256000,
    # Zhipu
    "glm-5-1": 128000,
    "glm-5": 128000,
    "glm-5v-turbo": 128000,
    "glm-5-turbo": 128000,
    # xAI
    "grok-4-20": 128000,
    "grok-4-3": 128000,
    # Xiaomi
    "mimo-v2-5-pro": 128000,
    "mimo-v2-5": 128000,
}


def _get_model_context_window(model_id: str) -> int:
    """Return approximate context window for a model ID, or 128000 as fallback."""
    if not model_id:
        return 128000
    raw = model_id.strip().lower().replace("_", "-").replace(".", "-")
    # Exact match
    if raw in _MODEL_CONTEXT_WINDOWS:
        return _MODEL_CONTEXT_WINDOWS[raw]
    # Prefix match (e.g. "anthropic/claude-sonnet-4-6" → "claude-sonnet-4-6")
    for key, val in _MODEL_CONTEXT_WINDOWS.items():
        if raw.endswith(key) or key in raw:
            return val
    # Provider-specific fallbacks
    if "claude" in raw:
        return 200000
    if "kimi" in raw:
        return 256000
    if "deepseek" in raw:
        return 64000
    if "gemini" in raw:
        return 1000000
    if "grok" in raw:
        return 128000
    if "glm" in raw:
        return 128000
    return 128000


async def handle_available_models(request: web.Request, *, check_auth, current_model_settings=None) -> web.Response:
    """GET /api/available-models -- list provider models and available providers."""
    auth_err = check_auth(request)
    if auth_err:
        return auth_err
    config = load_config()
    if current_model_settings:
        current = current_model_settings(config)
    else:
        current = _current_model_settings(config)
    provider = (request.query.get("provider") or current["provider"] or "openrouter").strip()
    data = await _get_cached_models(provider)
    return web.json_response(data)
