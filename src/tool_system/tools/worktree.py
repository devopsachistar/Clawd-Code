from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permissions import ToolPermissionContext
from ..protocol import ToolResult
from ..registry import ToolSpec


_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")


def _allow_extra_root(context: ToolContext, root: Path) -> None:
    pc = context.permission_context
    context.permission_context = ToolPermissionContext.from_iterables(
        pc.deny_names, pc.deny_prefixes, workspace_root=pc.workspace_root,
        additional_working_directories=[*pc.additional_working_directories, root], allow_docs=pc.allow_docs,
    )


class EnterWorktreeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="EnterWorktree",
            description=(
                "Create an isolated git worktree on its own branch (increscent/session-<id>/<name>) and switch the "
                "session into it. Nothing is merged or pushed. In a non-git directory a plain scratch directory is used."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {"name": {"type": "string"}},
            },
            is_destructive=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.worktree_root is not None:
            raise ToolPermissionError("already in a worktree session")
        name = tool_input.get("name")
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise ToolInputError("name must be a non-empty string when provided")
            if len(name) > 64 or not _NAME_RE.match(name):
                raise ToolInputError("invalid worktree name")
            slug = name
        else:
            slug = "worktree"

        from ...gitops.worktrees import GitError, WorktreeManager

        try:
            manager = WorktreeManager(context.workspace_root)
        except GitError:
            manager = None

        if manager is not None and manager.has_commits():
            info = manager.create(f"session-{uuid.uuid4().hex[:6]}", slug.replace("/", "-"))
            root = Path(info.path)
            _allow_extra_root(context, root)
            context.worktree_root = root
            context.cwd = root
            return ToolResult(
                name="EnterWorktree",
                output={
                    "worktreePath": str(root),
                    "worktreeBranch": info.branch,
                    "isGitWorktree": True,
                    "message": f"Created git worktree at {root} on branch {info.branch}. "
                               "The session is now working in the worktree.",
                },
            )

        # Not a git repository (or no commits yet): isolated scratch directory.
        root = context.workspace_root / ".clawd" / "worktrees" / slug
        root.mkdir(parents=True, exist_ok=True)
        context.worktree_root = root
        context.cwd = root
        return ToolResult(
            name="EnterWorktree",
            output={
                "worktreePath": str(root),
                "worktreeBranch": None,
                "isGitWorktree": False,
                "message": f"Created worktree at {root} (scratch directory: this workspace is not a git repository "
                           "with commits). The session is now working in the worktree.",
            },
        )


class ExitWorktreeTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ExitWorktree",
            description="Exit the current worktree session and return to the original workspace (the worktree and its branch are kept).",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_destructive=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.worktree_root is None:
            raise ToolPermissionError("not in a worktree session")
        old = str(context.worktree_root)
        context.worktree_root = None
        context.cwd = context.workspace_root
        return ToolResult(
            name="ExitWorktree",
            output={"message": f"Exited worktree session ({old}). Returned to {context.workspace_root}. "
                               "The worktree was preserved; remove it with 'increscent worktree remove'."},
        )
