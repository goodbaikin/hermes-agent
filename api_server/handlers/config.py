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
    models = [
        {"id": model_id, "description": description}
        for model_id, description in curated_models_for_provider(provider)
    ]
    providers = list_available_providers()
    return web.json_response({"provider": provider, "models": models, "providers": providers})
