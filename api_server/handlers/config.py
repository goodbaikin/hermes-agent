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

    # Invalidate models cache so next fetch reflects new provider
    invalidate_models_cache()

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
# Keyed by provider name so switching providers doesn't serve stale data
_models_cache: dict[str, dict[str, Any]] = {}
_models_cache_at: dict[str, float] = {}
_MODELS_CACHE_TTL: float = 86400.0  # 24 hours

# Separate cache for providers list (it's the slow part — 6s for auth checks)
_providers_cache: list[dict[str, str]] | None = None
_providers_cache_at: float = 0.0
_PROVIDERS_CACHE_TTL: float = 86400.0  # 24 hours — auth status rarely changes


def _get_cached_providers() -> list[dict[str, str]]:
    """Return cached provider list (expensive auth check, 24h TTL)."""
    global _providers_cache, _providers_cache_at
    now = time.monotonic()
    if _providers_cache is not None and (now - _providers_cache_at) < _PROVIDERS_CACHE_TTL:
        return _providers_cache
    _providers_cache = list_available_providers()
    _providers_cache_at = now
    return _providers_cache


async def _get_cached_models(provider: str) -> dict[str, Any]:
    """Return cached model list for a provider, refreshing if stale."""
    global _models_cache, _models_cache_at
    now = time.monotonic()
    cache_key = (provider or "").strip() or "default"
    if cache_key in _models_cache and (now - _models_cache_at.get(cache_key, 0)) < _MODELS_CACHE_TTL:
        # Return cached models + fresh provider list (providers are cached separately)
        result = dict(_models_cache[cache_key])
        result["providers"] = _get_cached_providers()
        return result
    # Refresh cache for this provider
    models = [
        {"id": model_id, "description": description, "context_window": _get_model_context_window(model_id)}
        for model_id, description in curated_models_for_provider(provider)
    ]
    providers = _get_cached_providers()
    result = {"provider": provider, "models": models, "providers": providers}
    _models_cache[cache_key] = {"provider": provider, "models": models, "providers": providers}
    _models_cache_at[cache_key] = now
    return result


def invalidate_models_cache(provider: str = None):
    """Invalidate models cache (called when provider/model config changes)."""
    global _models_cache, _models_cache_at, _providers_cache, _providers_cache_at
    if provider:
        cache_key = (provider or "").strip() or "default"
        _models_cache.pop(cache_key, None)
        _models_cache_at.pop(cache_key, None)
    else:
        _models_cache.clear()
        _models_cache_at.clear()
    # Also invalidate providers cache (auth may have changed)
    _providers_cache = None
    _providers_cache_at = 0.0


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
    "kimi-k2-6": 262144,
    "kimi-k2-5": 262144,
    "kimi-for-coding": 262144,
    "kimi-k2-thinking": 262144,
    "kimi-k2-thinking-turbo": 262144,
    "kimi-k2-turbo-preview": 262144,
    "kimi-k2-0905-preview": 262144,
    # DeepSeek
    "deepseek-v4-pro": 1000000,
    "deepseek-v4-flash": 1000000,
    "deepseek-chat": 1000000,
    "deepseek-reasoner": 1000000,
    # Google
    "gemini-3-pro-preview": 1048576,
    "gemini-3-flash-preview": 1048576,
    "gemini-3-1-pro-preview": 1048576,
    "gemini-3-1-flash-lite-preview": 1048576,
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
        return 262144
    if "deepseek" in raw:
        return 1000000
    if "gemini" in raw:
        return 1048576
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
