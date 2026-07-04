"""tools.py — sandboxed tool registry for milly-agent.

All file tools operate strictly inside the workspace directory
(data/workspace by default). Paths are resolved and verified to stay
within the sandbox; absolute paths and traversal (`..`, symlink escapes)
are rejected before any filesystem access.
"""

import os
from pathlib import Path

_MAX_READ_CHARS = 20_000
_MAX_WRITE_CHARS = 100_000


class ToolError(Exception):
    """Raised when a tool call is invalid or violates the sandbox."""


class ToolRegistry:
    def __init__(self, workspace_dir: str | os.PathLike):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Sandbox
    # ------------------------------------------------------------------

    def _resolve(self, relpath: str) -> Path:
        """Resolve a user-supplied path inside the workspace, or raise."""
        if not isinstance(relpath, str) or not relpath.strip():
            raise ToolError("path must be a non-empty string")
        candidate = Path(relpath)
        if candidate.is_absolute():
            raise ToolError(f"absolute paths are not allowed: {relpath}")
        resolved = (self.workspace / candidate).resolve()
        try:
            resolved.relative_to(self.workspace.resolve())
        except ValueError:
            raise ToolError(f"path escapes the workspace sandbox: {relpath}")
        return resolved

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _list_files(self) -> str:
        root = self.workspace.resolve()
        entries = sorted(
            str(p.relative_to(root))
            for p in root.rglob("*")
            if p.is_file() and p.name != ".gitkeep"
        )
        if not entries:
            return "(workspace is empty)"
        return "\n".join(entries)

    def _read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"no such file in workspace: {path}")
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > _MAX_READ_CHARS:
            text = text[:_MAX_READ_CHARS] + "\n[... truncated ...]"
        return text

    def _write_file(self, path: str, content: str) -> str:
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if len(content) > _MAX_WRITE_CHARS:
            raise ToolError(
                f"content too large ({len(content)} chars > {_MAX_WRITE_CHARS} limit)"
            )
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def specs(self) -> list[dict]:
        """Tool descriptions injected into the system prompt."""
        return [
            {
                "name": "list_files",
                "description": "List all files in the agent workspace.",
                "args": {},
            },
            {
                "name": "read_file",
                "description": "Read a text file from the workspace.",
                "args": {"path": "relative path inside the workspace"},
            },
            {
                "name": "write_file",
                "description": "Write a text file into the workspace.",
                "args": {
                    "path": "relative path inside the workspace",
                    "content": "full text content to write",
                },
            },
        ]

    def names(self) -> set[str]:
        return {spec["name"] for spec in self.specs()}

    def execute(self, name: str, args: dict) -> str:
        """Run a tool by name. Raises ToolError on any invalid call."""
        args = args or {}
        if not isinstance(args, dict):
            raise ToolError("tool args must be an object")
        if name == "list_files":
            return self._list_files()
        if name == "read_file":
            return self._read_file(args.get("path", ""))
        if name == "write_file":
            return self._write_file(args.get("path", ""), args.get("content", ""))
        raise ToolError(f"unknown tool: {name}")
