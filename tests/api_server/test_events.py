"""Tests for api_server.events type-safe event bus."""

import time
from api_server.events import (
    EventBus,
    ToolStartedEvent,
    ToolCompletedEvent,
    ReasoningAvailableEvent,
    TextDeltaEvent,
)


def test_event_bus_emit_tool_started():
    bus = EventBus()
    received = []
    bus.on_tool_started(lambda ev: received.append(ev))

    ev = ToolStartedEvent(
        event="tool.started", timestamp=time.time(), tool="web_search", preview="searching..."
    )
    bus.emit(ev)

    assert len(received) == 1
    assert received[0].tool == "web_search"
    assert received[0].preview == "searching..."


def test_event_bus_emit_tool_completed():
    bus = EventBus()
    received = []
    bus.on_tool_completed(lambda ev: received.append(ev))

    ev = ToolCompletedEvent(
        event="tool.completed", timestamp=time.time(), tool="web_search",
        duration=1.23, is_error=False,
    )
    bus.emit(ev)

    assert len(received) == 1
    assert received[0].tool == "web_search"
    assert received[0].duration == 1.23
    assert received[0].is_error is False


def test_event_bus_emit_reasoning():
    bus = EventBus()
    received = []
    bus.on_reasoning_available(lambda ev: received.append(ev))

    ev = ReasoningAvailableEvent(
        event="reasoning.available", timestamp=time.time(), text="let me think..."
    )
    bus.emit(ev)

    assert len(received) == 1
    assert received[0].text == "let me think..."


def test_event_bus_no_handlers():
    bus = EventBus()
    # Should not raise even with no handlers registered
    ev = TextDeltaEvent(event="text.delta", timestamp=time.time(), delta="hello")
    bus.emit(ev)


def test_event_bus_multiple_handlers():
    bus = EventBus()
    received1 = []
    received2 = []
    bus.on_tool_started(lambda ev: received1.append(ev))
    bus.on_tool_started(lambda ev: received2.append(ev))

    ev = ToolStartedEvent(
        event="tool.started", timestamp=time.time(), tool="bash", preview="ls"
    )
    bus.emit(ev)

    assert len(received1) == 1
    assert len(received2) == 1
