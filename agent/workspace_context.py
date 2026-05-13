"""
Workspace Context — Per-session workspace routing via contextvars.

Provides thread-safe and async-safe workspace context propagation
without global state mutation.
"""

import contextvars
from typing import Optional

_workspace_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "workspace", default=None
)


def set_workspace(workspace: Optional[str]):
    """Set the current workspace context. Returns a token for reset."""
    return _workspace_ctx.set(workspace)


def get_workspace() -> Optional[str]:
    """Get the current workspace from context, or None."""
    return _workspace_ctx.get()


def reset_workspace(token):
    """Reset the workspace context using the token from set_workspace."""
    _workspace_ctx.reset(token)
