"""
Workspace Manager — Manages active workspace and resolves nodes for tool execution.

Loaded from config.yaml workspaces section. Provides process-level workspace
switching via CLI flag or slash command.
"""

import threading
from typing import Dict, List, Optional

from agent.workspace import Workspace


class WorkspaceManager:
    """Singleton workspace manager."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._workspaces: Dict[str, Workspace] = {}
        self._active: str = "default"
        self._load_from_config()

    def _load_from_config(self):
        """Load workspaces from config.yaml."""
        try:
            from hermes_constants import get_hermes_home
            import yaml
            import os

            config_path = os.path.join(get_hermes_home(), "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}

                workspaces = config.get("workspaces", {})
                for name, data in workspaces.items():
                    self._workspaces[name] = Workspace.from_dict(name, data)

                # Ensure default exists
                if "default" not in self._workspaces:
                    self._workspaces["default"] = Workspace(
                        name="default",
                        node_id="local",
                        path_prefixes=["~/", "/tmp/"],
                        description="ローカルマシン",
                    )
                
                # Set active workspace from config or default to "default"
                self._active = config.get("active_workspace", "default")
                if self._active not in self._workspaces:
                    self._active = "default"
        except Exception:
            # Fallback if config loading fails
            self._workspaces["default"] = Workspace(
                name="default",
                node_id="local",
                path_prefixes=["~/", "/tmp/"],
                description="ローカルマシン",
            )
            self._active = "default"

    def set_active(self, name: str) -> bool:
        """Switch to the named workspace. Returns True if successful."""
        if name in self._workspaces:
            self._active = name
            return True
        return False

    def get_active(self) -> Workspace:
        """Get the currently active workspace."""
        return self._workspaces.get(self._active, self._workspaces.get("default"))

    def resolve_node(self, tool_name: str, params: Optional[Dict] = None) -> str:
        """
        Resolve which node should execute the given tool call.
        Always returns the active workspace's node_id.
        """
        workspace = self.get_active()
        return workspace.node_id

    def list_workspaces(self) -> List[str]:
        """List all available workspace names."""
        return list(self._workspaces.keys())

    def get_workspace(self, name: str) -> Optional[Workspace]:
        """Get a workspace by name."""
        return self._workspaces.get(name)

    @property
    def active_name(self) -> str:
        return self._active


# Global accessor
def get_workspace_manager() -> WorkspaceManager:
    return WorkspaceManager()


def get_active_workspace() -> Workspace:
    return get_workspace_manager().get_active()


def resolve_node(tool_name: str, params: Optional[Dict] = None) -> str:
    """Convenience function: resolve node for a tool call."""
    return get_workspace_manager().resolve_node(tool_name, params)
