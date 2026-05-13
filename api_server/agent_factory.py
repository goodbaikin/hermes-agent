"""
Agent factory — Gateway-independent AIAgent creation for the API server.

Replaces the inline _create_agent() in gateway/platforms/api_server.py
with a typed, injectable factory.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from api_server.config import APIServerConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentCallbacks:
    """Typed callback bundle for AIAgent."""

    stream_delta_callback: Optional[Callable[[str], None]] = None
    tool_progress_callback: Optional[Callable[[str, str, Optional[str], Any], None]] = None
    tool_start_callback: Optional[Callable[[str, Any], None]] = None
    tool_complete_callback: Optional[Callable[[str, Any, Optional[str], bool], None]] = None


def create_agent(
    config: APIServerConfig,
    *,
    callbacks: AgentCallbacks = None,
    ephemeral_system_prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    session_db: Any = None,
    workspace: Optional[str] = None,
) -> Any:
    """Create an AIAgent using API Server configuration only.

    Args:
        config: Resolved APIServerConfig (from load_api_server_config).
        callbacks: Optional AgentCallbacks for streaming / tool events.
        ephemeral_system_prompt: One-shot system prompt override.
        session_id: Session ID for persistence.
        session_db: Optional SessionDB instance.
        workspace: Workspace directory or identifier for the agent session.

    Returns:
        An AIAgent instance.
    """
    from run_agent import AIAgent

    callbacks = callbacks or AgentCallbacks()

    kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": config.api_key_provider,
        "base_url": config.base_url,
        "provider": config.provider,
        "api_mode": config.api_mode,
        "command": config.command,
        "args": config.args,
        "credential_pool": config.credential_pool,
        "max_iterations": config.max_iterations,
        "quiet_mode": True,
        "verbose_logging": False,
        "ephemeral_system_prompt": ephemeral_system_prompt or None,
        "enabled_toolsets": config.enabled_toolsets,
        "session_id": session_id,
        "platform": "api_server",
        "session_db": session_db,
        "fallback_model": config.fallback_model,
        "reasoning_config": config.reasoning_config,
        "workspace": workspace,
    }

    if callbacks.stream_delta_callback:
        kwargs["stream_delta_callback"] = callbacks.stream_delta_callback
    if callbacks.tool_progress_callback:
        kwargs["tool_progress_callback"] = callbacks.tool_progress_callback
    if callbacks.tool_start_callback:
        kwargs["tool_start_callback"] = callbacks.tool_start_callback
    if callbacks.tool_complete_callback:
        kwargs["tool_complete_callback"] = callbacks.tool_complete_callback

    return AIAgent(**kwargs)
