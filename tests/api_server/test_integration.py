"""Integration tests for API Server modularisation."""

import os
from unittest.mock import MagicMock, patch

from api_server.config import load_api_server_config, APIServerConfig
from api_server.agent_factory import create_agent, AgentCallbacks
from api_server.sse import build_cors_headers, SSEWriter
from api_server.events import EventBus, ToolStartedEvent


def test_load_api_server_config_returns_dataclass():
    """Config loads without gateway.run imports."""
    with patch("api_server.config._resolve_runtime_provider", return_value={
        "api_key": "test-key", "base_url": "http://test", "provider": "openrouter",
        "api_mode": "", "command": None, "args": [], "credential_pool": None,
    }):
        config = load_api_server_config()
    assert isinstance(config, APIServerConfig)
    assert config.host is not None
    assert config.port is not None


def test_build_cors_headers_integration():
    """CORS headers resolve for allowed origins."""
    assert build_cors_headers("http://localhost:8080", ("http://localhost:8080",)) is not None
    assert build_cors_headers("http://evil.com", ("http://localhost:8080",)) is None


def test_event_bus_to_sse_bridge():
    """EventBus can be wired into SSE emission logic."""
    bus = EventBus()
    sse_events = []

    def capture(event):
        sse_events.append({
            "event": event.event,
            "tool": getattr(event, "tool", None),
        })

    bus.on_tool_started(capture)
    bus.emit(ToolStartedEvent(event="tool.started", timestamp=0.0, tool="bash", preview="ls"))

    assert len(sse_events) == 1
    assert sse_events[0]["event"] == "tool.started"
    assert sse_events[0]["tool"] == "bash"
