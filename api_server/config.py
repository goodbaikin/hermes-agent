"""
API Server configuration — Gateway-independent config resolution.

Replaces imports from gateway.run (_load_gateway_config, _resolve_gateway_model,
GatewayRunner._load_reasoning_config, etc.) with local resolution via hermes_cli.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_cli.config import read_raw_config
from hermes_constants import get_hermes_home, parse_reasoning_effort

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8642


@dataclass
class APIServerConfig:
    """Resolved API Server runtime configuration."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    api_key: str = ""
    cors_origins: Tuple[str, ...] = ()
    model_name: str = "hermes-agent"

    # Provider / model settings (resolved from config.yaml / env / runtime_provider)
    model: str = ""
    provider: str = ""
    api_mode: str = ""
    base_url: Optional[str] = None
    api_key_provider: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    credential_pool: Any = None

    # Agent behaviour
    max_iterations: int = 90
    enabled_toolsets: List[str] = field(default_factory=list)
    reasoning_config: Optional[Dict[str, Any]] = None
    fallback_model: Any = None

    # Derived runtime kwargs for AIAgent
    runtime_kwargs: Dict[str, Any] = field(default_factory=dict)


def _coerce_port(value: Any, default: int = DEFAULT_PORT) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_cors_origins(value: Any) -> Tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [str(value)]
    return tuple(str(item).strip() for item in items if str(item).strip())


def _resolve_model_name(explicit: str, config: Dict[str, Any]) -> str:
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


def _resolve_model(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    if isinstance(model_cfg, str):
        return model_cfg.strip()
    elif isinstance(model_cfg, dict):
        return model_cfg.get("default") or model_cfg.get("model") or ""
    return ""


def _load_reasoning_config(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    effort = ""
    try:
        from hermes_cli.config import cfg_get
        effort = str(cfg_get(config, "agent", "reasoning_effort", default="") or "").strip()
    except Exception:
        pass
    result = parse_reasoning_effort(effort)
    if effort and effort.strip() and result is None:
        logger.warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def _load_fallback_model(config: Dict[str, Any]) -> Any:
    fb = config.get("fallback_providers") or config.get("fallback_model") or None
    return fb


def _try_resolve_fallback_provider() -> Optional[Dict[str, Any]]:
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        cfg = read_raw_config()
        fb = cfg.get("fallback_providers") or cfg.get("fallback_model")
        if not fb:
            return None
        fb_list = fb if isinstance(fb, list) else [fb]
        for entry in fb_list:
            if not isinstance(entry, dict):
                continue
            try:
                runtime = resolve_runtime_provider(
                    requested=entry.get("provider"),
                    explicit_base_url=entry.get("base_url"),
                    explicit_api_key=entry.get("api_key"),
                )
                logger.info("Fallback provider resolved: %s", runtime.get("provider"))
                return {
                    "api_key": runtime.get("api_key"),
                    "base_url": runtime.get("base_url"),
                    "provider": runtime.get("provider"),
                    "api_mode": runtime.get("api_mode"),
                    "command": runtime.get("command"),
                    "args": list(runtime.get("args") or []),
                    "credential_pool": runtime.get("credential_pool"),
                }
            except Exception as fb_exc:
                logger.debug("Fallback entry %s failed: %s", entry.get("provider"), fb_exc)
                continue
    except Exception:
        pass
    return None


def _resolve_runtime_provider() -> Dict[str, Any]:
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider as _resolve,
        format_runtime_provider_error,
    )
    from hermes_cli.auth import AuthError

    try:
        runtime = _resolve(requested=os.getenv("HERMES_INFERENCE_PROVIDER"))
    except AuthError as auth_exc:
        logger.warning("Primary provider auth failed: %s — trying fallback", auth_exc)
        fb_config = _try_resolve_fallback_provider()
        if fb_config is not None:
            return fb_config
        raise RuntimeError(format_runtime_provider_error(auth_exc)) from auth_exc
    except Exception as exc:
        raise RuntimeError(format_runtime_provider_error(exc)) from exc

    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
    }


def load_api_server_config(
    extra: Optional[Dict[str, Any]] = None,
    env_prefix: str = "API_SERVER_",
) -> APIServerConfig:
    """Resolve API Server configuration without importing gateway.run.

    Reads:
      1. config.yaml (model, provider, reasoning, fallback, toolsets)
      2. Environment variables (API_SERVER_HOST, API_SERVER_PORT, ...)
      3. Platform-specific ``extra`` dict (from gateway config)
    """
    extra = extra or {}
    user_config = read_raw_config()

    # Host / port / auth
    host = extra.get("host", os.getenv(f"{env_prefix}HOST", DEFAULT_HOST))
    raw_port = extra.get("port", os.getenv(f"{env_prefix}PORT", str(DEFAULT_PORT)))
    port = _coerce_port(raw_port, DEFAULT_PORT)
    api_key = extra.get("key", os.getenv(f"{env_prefix}KEY", ""))
    cors_origins = _parse_cors_origins(
        extra.get("cors_origins", os.getenv(f"{env_prefix}CORS_ORIGINS", ""))
    )
    model_name = _resolve_model_name(
        extra.get("model_name", os.getenv(f"{env_prefix}MODEL_NAME", "")),
        user_config,
    )

    # Provider settings
    runtime = _resolve_runtime_provider()
    model = _resolve_model(user_config)

    # Toolsets
    enabled_toolsets: List[str] = []
    try:
        from hermes_cli.tools_config import _get_platform_tools
        enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))
    except Exception:
        pass

    max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", "90"))

    return APIServerConfig(
        host=host,
        port=port,
        api_key=api_key,
        cors_origins=cors_origins,
        model_name=model_name,
        model=model or runtime.get("provider", ""),
        provider=runtime.get("provider", ""),
        api_mode=runtime.get("api_mode", ""),
        base_url=runtime.get("base_url"),
        api_key_provider=runtime.get("api_key"),
        command=runtime.get("command"),
        args=runtime.get("args", []),
        credential_pool=runtime.get("credential_pool"),
        max_iterations=max_iterations,
        enabled_toolsets=enabled_toolsets,
        reasoning_config=_load_reasoning_config(user_config),
        fallback_model=_load_fallback_model(user_config),
        runtime_kwargs=runtime,
    )
