"""Remote LSP client — delegates diagnostics to a node_client LSP RPC server.

This module provides a bridge between Hermes's local LSP infrastructure
and language servers running on remote nodes (e.g. dev-win01 for C#).

Usage:
    client = RemoteLSPClient("dev-win01")
    diagnostics = await client.lint_after_write(
        file_path="C:/Users/goodb/workspace/COCONV.Deploy/Program.cs",
        content="...",
        language="csharp",
        workspace_root="C:/Users/goodb/workspace/COCONV.Deploy",
    )
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RemoteLSPClient:
    """LSP client that delegates to a node_client's LSP RPC server."""

    def __init__(self, node_id: str):
        self.node_id = node_id

    async def lint_after_write(
        self,
        file_path: str,
        content: str,
        language: str,
        workspace_root: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run LSP diagnostics after a file write.

        Returns a list of diagnostic dicts with keys:
            range: {start: {line, character}, end: {line, character}}
            severity: 1=Error, 2=Warning, 3=Info, 4=Hint
            message: str
            source: str (optional)
            code: str (optional)
        """
        if workspace_root is None:
            workspace_root = str(Path(file_path).parent)

        result = await self._invoke_node({
            "tool": "lsp",
            "action": "lint_after_write",
            "language": language,
            "workspace_root": workspace_root,
            "file_path": file_path,
            "content": content,
        })

        if result is None:
            logger.warning("Remote LSP returned None for %s", file_path)
            return []

        if isinstance(result, dict) and "error" in result:
            logger.warning("Remote LSP error: %s", result["error"])
            return []

        # result may be the raw node response or already parsed
        if isinstance(result, dict) and "diagnostics" in result:
            return result["diagnostics"]

        # Fallback: result might be double-wrapped
        if isinstance(result, dict) and "result" in result:
            inner = result["result"]
            if isinstance(inner, dict) and "diagnostics" in inner:
                return inner["diagnostics"]

        logger.debug("Remote LSP unexpected result shape: %s", type(result))
        return []

    async def get_diagnostics(
        self,
        file_path: str,
        language: str,
        workspace_root: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get current diagnostics for a file (no didOpen/didChange)."""
        if workspace_root is None:
            workspace_root = str(Path(file_path).parent)

        result = await self._invoke_node({
            "tool": "lsp",
            "action": "get_diagnostics",
            "language": language,
            "workspace_root": workspace_root,
            "file_path": file_path,
        })

        if result is None:
            return []
        if isinstance(result, dict) and "error" in result:
            logger.warning("Remote LSP error: %s", result["error"])
            return []
        if isinstance(result, dict) and "diagnostics" in result:
            return result["diagnostics"]
        return []

    async def shutdown(self) -> None:
        """Shutdown all remote LSP servers."""
        await self._invoke_node({
            "tool": "lsp",
            "action": "shutdown",
        })

    async def _invoke_node(self, params: Dict[str, Any]) -> Any:
        """Invoke the node_client via the existing node_invoke tool."""
        # Import here to avoid circular imports at module load time
        from tools.node_invoke import node_invoke

        try:
            result = node_invoke(self.node_id, "lsp", params)
            return result
        except Exception as exc:
            logger.warning("Remote LSP node invoke failed: %s", exc)
            return None


def format_diagnostics(diagnostics: List[Dict[str, Any]], file_path: str) -> str:
    """Format LSP diagnostics into a human-readable string for injection."""
    if not diagnostics:
        return ""

    severity_names = {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}
    lines = [f"⚠️  {len(diagnostics)} diagnostic(s) in {Path(file_path).name}:"]

    for diag in diagnostics:
        severity = diag.get("severity", 1)
        sev_name = severity_names.get(severity, "Unknown")
        msg = diag.get("message", "")
        rng = diag.get("range", {})
        start = rng.get("start", {})
        line = start.get("line", 0) + 1  # 0-indexed to 1-indexed
        char = start.get("character", 0)
        source = diag.get("source", "")
        code = diag.get("code", "")

        prefix = f"  [{sev_name}] Line {line}:{char}"
        if source:
            prefix += f" ({source})"
        if code:
            prefix += f" [{code}]"
        lines.append(f"{prefix}: {msg}")

    return "\n".join(lines)
