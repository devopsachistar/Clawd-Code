from __future__ import annotations

import subprocess
from pathlib import Path

from .models import GitContext

_MAX_STATUS_CHARS = 2_000


def collect_git_context(workspace_root: str | Path) -> GitContext:
    cwd = Path(workspace_root).expanduser().resolve()

    top = _run_git(cwd, "rev-parse", "--show-toplevel")
    if top is None or top.returncode != 0:
        return GitContext(available=False, error="not a git repository")

    repo_root_text = (top.stdout or "").strip()
    if not repo_root_text:
        return GitContext(available=False, error="unable to resolve git root")
    repo_root = Path(repo_root_text).resolve()

    branch = _read_git_text(repo_root, "symbolic-ref", "--short", "HEAD")
    if not branch:
        branch = _read_git_text(repo_root, "rev-parse", "--short", "HEAD")

    status = _read_git_text(repo_root, "status", "--short", "--branch")
    if status and len(status) > _MAX_STATUS_CHARS:
        status = status[: _MAX_STATUS_CHARS - 32].rstrip() + "\n...[truncated]"

    recent_commit = _read_git_text(repo_root, "log", "-1", "--oneline", "--decorate")

    return GitContext(
        available=True,
        repo_root=repo_root,
        branch=branch or None,
        status=status or None,
        recent_commit=recent_commit or None,
    )


def _read_git_text(cwd: Path, *args: str) -> str | None:
    result = _run_git(cwd, *args)
    if result is None or result.returncode != 0:
        return None
    text = (result.stdout or "").strip()
    return text or None


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
