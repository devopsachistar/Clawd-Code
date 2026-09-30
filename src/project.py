"""Project / repository root detection.

The agent must always know which project it is operating on. Detection walks
up from the starting directory:

1. The nearest directory containing ``.git`` (a directory, or a file for
   worktrees/submodules) is the **git root**, and the project root.
2. Without git, the nearest directory containing a project marker
   (``pyproject.toml``, ``package.json``, ...) is the project root.
3. Otherwise the starting directory itself is the project root.

The user's home directory and filesystem roots are never *auto-detected* as a
project root while walking up from a subdirectory. A dotfiles repository in
``$HOME`` or a stray ``package.json`` there must not turn the whole home
directory into the agent's workspace. If the start directory itself is home or
a filesystem root, detection reports it with ``is_unsafe_root=True`` so callers
can refuse or warn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_MARKERS: tuple[str, ...] = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "composer.json",
    "Gemfile",
    "pubspec.yaml",
    "CMakeLists.txt",
    ".clawd",
    "CLAUDE.md",
)


@dataclass(frozen=True)
class ProjectInfo:
    """Where the agent is operating."""

    project_root: Path
    cwd: Path
    git_root: Path | None = None
    markers: tuple[str, ...] = field(default_factory=tuple)
    detection: str = "cwd"  # "git" | "marker" | "cwd"
    is_unsafe_root: bool = False

    @property
    def is_git_repo(self) -> bool:
        return self.git_root is not None

    @property
    def name(self) -> str:
        return self.project_root.name or str(self.project_root)

    def describe(self) -> list[str]:
        lines = [f"- Project root: {self.project_root}"]
        lines.append(f"- Git root: {self.git_root}" if self.git_root else "- Git root: (not a git repository)")
        if self.cwd != self.project_root:
            lines.append(f"- Working directory: {self.cwd}")
        return lines

    def to_dict(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "git_root": str(self.git_root) if self.git_root else None,
            "cwd": str(self.cwd),
            "detection": self.detection,
            "markers": list(self.markers),
            "is_unsafe_root": self.is_unsafe_root,
        }


def _home() -> Path | None:
    try:
        return Path.home().resolve()
    except Exception:
        return None


def is_filesystem_root(path: Path) -> bool:
    return path.parent == path


def is_unsafe_project_root(path: Path) -> bool:
    """Home directory or a filesystem/drive root is never a safe workspace."""
    resolved = path.resolve()
    home = _home()
    return is_filesystem_root(resolved) or (home is not None and resolved == home)


def _has_git(path: Path) -> bool:
    return (path / ".git").exists()


def _markers_in(path: Path) -> tuple[str, ...]:
    return tuple(m for m in PROJECT_MARKERS if (path / m).exists())


def _ancestors_below_boundary(start: Path) -> list[Path]:
    """``start`` and its parents, stopping before home or the filesystem root."""
    home = _home()
    chain: list[Path] = []
    node = start
    while True:
        chain.append(node)
        if is_filesystem_root(node) or (home is not None and node == home):
            break
        parent = node.parent
        if is_filesystem_root(parent) or (home is not None and parent == home):
            break
        node = parent
    return chain


def detect_project(start: str | Path | None = None) -> ProjectInfo:
    """Detect the project and git root for ``start`` (defaults to the cwd)."""
    start_path = Path(start if start is not None else os.getcwd()).expanduser().resolve()
    if start_path.is_file():
        start_path = start_path.parent

    chain = _ancestors_below_boundary(start_path)

    for node in chain:
        if _has_git(node):
            return ProjectInfo(
                project_root=node,
                cwd=start_path,
                git_root=node,
                markers=_markers_in(node),
                detection="git",
                is_unsafe_root=is_unsafe_project_root(node),
            )

    for node in chain:
        markers = _markers_in(node)
        if markers:
            return ProjectInfo(
                project_root=node,
                cwd=start_path,
                markers=markers,
                detection="marker",
                is_unsafe_root=is_unsafe_project_root(node),
            )

    return ProjectInfo(
        project_root=start_path,
        cwd=start_path,
        detection="cwd",
        is_unsafe_root=is_unsafe_project_root(start_path),
    )
