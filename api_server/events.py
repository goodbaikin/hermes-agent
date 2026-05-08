"""
Type-safe event bus for API server streaming and tool lifecycle.

Replaces ad-hoc dict-based callbacks with dataclass events and a
lightweight pub/sub bus.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StreamEvent:
    """Base for all API server events."""
    event: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextDeltaEvent(StreamEvent):
    """Emitted for each streamed text delta."""
    delta: str = ""


@dataclass
class ToolStartedEvent(StreamEvent):
    """Emitted when a tool call starts."""
    tool: str = ""
    preview: str = ""
    args: Optional[Dict[str, Any]] = None


@dataclass
class ToolCompletedEvent(StreamEvent):
    """Emitted when a tool call completes."""
    tool: str = ""
    duration: float = 0.0
    is_error: bool = False
    output_preview: str = ""


@dataclass
class ReasoningAvailableEvent(StreamEvent):
    """Emitted when reasoning text is available."""
    text: str = ""


class EventBus:
    """Lightweight typed pub/sub bus for API server events.

    Example::

        bus = EventBus()
        bus.on_text_delta(lambda ev: print(ev.delta))
        bus.emit(TextDeltaEvent(event="text.delta", timestamp=time.time(), delta="hello"))
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[[StreamEvent], None]]] = {
            "text_delta": [],
            "tool_started": [],
            "tool_completed": [],
            "reasoning_available": [],
        }

    def on_text_delta(self, callback: Callable[[TextDeltaEvent], None]) -> None:
        self._handlers["text_delta"].append(callback)  # type: ignore[arg-type]

    def on_tool_started(self, callback: Callable[[ToolStartedEvent], None]) -> None:
        self._handlers["tool_started"].append(callback)  # type: ignore[arg-type]

    def on_tool_completed(self, callback: Callable[[ToolCompletedEvent], None]) -> None:
        self._handlers["tool_completed"].append(callback)  # type: ignore[arg-type]

    def on_reasoning_available(self, callback: Callable[[ReasoningAvailableEvent], None]) -> None:
        self._handlers["reasoning_available"].append(callback)  # type: ignore[arg-type]

    def emit(self, event: StreamEvent) -> None:
        key = event.event.replace(".", "_")
        for handler in self._handlers.get(key, []):
            try:
                handler(event)
            except Exception:
                pass
