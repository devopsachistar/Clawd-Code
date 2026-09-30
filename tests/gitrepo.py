"""Disposable git repositories for worktree / orchestration tests."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout.strip()


class TempRepo:
    """``<tmp>/repo`` with one commit on ``main``; worktrees go to ``<tmp>/worktrees``."""

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.base = Path(self._tmp.name).resolve()
        self.path = self.base / "repo"
        self.worktrees = self.base / "worktrees"
        self.path.mkdir()
        git(self.path, "init", "-q", "-b", "main")
        git(self.path, "config", "user.name", "Test")
        git(self.path, "config", "user.email", "test@example.com")
        git(self.path, "config", "core.autocrlf", "false")
        for name, content in (files or {"README.md": "# test\n", ".gitignore": ".env\n"}).items():
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "-m", "init")
        self.initial = git(self.path, "rev-parse", "HEAD")

    def head(self, ref: str = "main") -> str:
        return git(self.path, "rev-parse", ref)

    def cleanup(self) -> None:
        subprocess.run(["git", "worktree", "prune"], cwd=str(self.path), capture_output=True)
        self._tmp.cleanup()
