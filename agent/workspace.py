"""
Workspace — Execution context that maps tools/paths to remote/local nodes.

A workspace defines which node (local or remote) should handle tool executions
based on path prefixes and tool capabilities.
"""

from typing import Dict, List, Optional


class Workspace:
    """Single workspace definition."""

    def __init__(
        self,
        name: str,
        node_id: str,
        path_prefixes: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        description: str = "",
    ):
        self.name = name
        self.node_id = node_id
        self.path_prefixes = path_prefixes or []
        self.tools = set(tools or ["all"])
        self.description = description

    def matches_path(self, path: str) -> bool:
        """Check if the given path falls under this workspace."""
        for prefix in self.path_prefixes:
            # Normalize ~ to home for comparison
            normalized_path = path
            if path.startswith("~"):
                import os
                normalized_path = os.path.expanduser(path)
            if normalized_path.startswith(prefix):
                return True
        return False

    def supports_tool(self, tool_name: str) -> bool:
        """Check if this workspace supports the given tool."""
        if "all" in self.tools:
            return True
        return tool_name in self.tools

    def resolve_node(self, tool_name: str, params: Optional[Dict] = None) -> Optional[str]:
        """
        Determine if this workspace should handle the tool call.
        Returns node_id if matched, None otherwise.
        """
        # Check path-based routing first
        if params and "path" in params:
            if self.matches_path(params["path"]):
                return self.node_id

        # Check tool-based routing
        if self.supports_tool(tool_name):
            return self.node_id

        return None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "node_id": self.node_id,
            "path_prefixes": self.path_prefixes,
            "tools": list(self.tools),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, name: str, data: Dict) -> "Workspace":
        return cls(
            name=name,
            node_id=data.get("node_id", "local"),
            path_prefixes=data.get("path_prefixes", []),
            tools=data.get("tools", ["all"]),
            description=data.get("description", ""),
        )
