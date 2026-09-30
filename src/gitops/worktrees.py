"""Real git worktrees for isolated agent work.

Layout (example)::

    C:\\Projects\\myapp                                   main repository
    C:\\Projects\\.increscent-worktrees\\myapp-T-1a2b-frontend  worktree
        branch: increscent/T-1a2b/frontend

Guarantees:

* every agent worktree is a real ``git worktree`` on its own branch
* a worktree is used by at most one agent at a time (lock files in the
  shared state directory, stale locks from dead processes are reclaimed)
* nothing is ever merged into the main branch or pushed. Integration happens
  on a separate ``increscent/<graph>/integration`` branch after human approval.
* commits refuse to include secret files that are not git-ignored
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from ..branding import BRANCH_PREFIX, WORKTREES_DIR_NAME
from ..orchestration.state import FileLock, atomic_write_json, now_iso, read_json, resolve_state_dir
from ..security.secrets import redact_secrets


class GitError(RuntimeError):
    pass


class WorktreeBusy(GitError):
    pass


class MergeConflict(GitError):
    def __init__(self, branch: str, files: list[str]):
        super().__init__(f"merge conflict merging {branch}: {', '.join(files) or 'unknown files'}")
        self.branch = branch
        self.files = files


def run_git(args: list[str], cwd: str | Path, *, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL, timeout=timeout, env=env,
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {(proc.stderr or proc.stdout).strip()}")
    return proc


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-._").lower()
    return (s or "work")[:limit]


def _branch_part(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip()).strip("-.")
    return s or "work"


@dataclass
class WorktreeInfo:
    name: str
    path: str
    branch: Optional[str]
    head: Optional[str] = None
    task_id: Optional[str] = None
    role: Optional[str] = None
    base_ref: Optional[str] = None
    created_at: Optional[str] = None
    managed: bool = False
    locked_by: Optional[str] = None
    exists: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WorktreeManager:
    def __init__(self, repo_root: str | Path, *, base_dir: str | Path | None = None,
                 state_dir: str | Path | None = None, events: Any = None) -> None:
        top = run_git(["rev-parse", "--show-toplevel"], repo_root, check=False)
        if top.returncode != 0:
            raise GitError(f"not a git repository: {repo_root}")
        self.repo_root = Path(top.stdout.strip()).resolve()
        self.base_dir = Path(base_dir).resolve() if base_dir else (self.repo_root.parent / WORKTREES_DIR_NAME)
        self.state_dir = Path(state_dir) if state_dir else resolve_state_dir(self.repo_root)
        self.registry_path = self.state_dir / "worktrees.json"
        self.locks_dir = self.state_dir / "locks"
        self.events = events
        self._lock = threading.RLock()

    # -- registry -------------------------------------------------------------
    def _registry(self) -> dict[str, dict[str, Any]]:
        return dict(read_json(self.registry_path, {}).get("worktrees", {}))

    def _write_registry(self, data: dict[str, dict[str, Any]]) -> None:
        atomic_write_json(self.registry_path, {"schema_version": 1, "worktrees": data})

    def _emit(self, event: str, **fields: Any) -> None:
        if self.events is not None:
            self.events.emit(event, **fields)

    # -- queries ----------------------------------------------------------------
    def has_commits(self) -> bool:
        return run_git(["rev-parse", "--verify", "HEAD"], self.repo_root, check=False).returncode == 0

    def current_branch(self, cwd: str | Path | None = None) -> Optional[str]:
        proc = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd or self.repo_root, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else None

    def list(self, *, managed_only: bool = False) -> list[WorktreeInfo]:
        registry = self._registry()
        by_path = {str(Path(v["path"]).resolve()).lower(): (k, v) for k, v in registry.items()}
        out: list[WorktreeInfo] = []
        seen: set[str] = set()
        porcelain = run_git(["worktree", "list", "--porcelain"], self.repo_root).stdout
        for block in porcelain.strip().split("\n\n"):
            fields: dict[str, str] = {}
            for line in block.splitlines():
                key, _, value = line.partition(" ")
                fields[key] = value
            if "worktree" not in fields:
                continue
            path = str(Path(fields["worktree"]).resolve())
            branch = fields.get("branch", "").removeprefix("refs/heads/") or None
            entry = by_path.get(path.lower())
            name, meta = entry if entry else (Path(path).name, {})
            seen.add(path.lower())
            info = WorktreeInfo(
                name=name, path=path, branch=branch, head=fields.get("HEAD"),
                task_id=meta.get("task_id"), role=meta.get("role"), base_ref=meta.get("base_ref"),
                created_at=meta.get("created_at"), managed=bool(entry),
                locked_by=self._lock_owner(name) if entry else None,
            )
            if managed_only and not info.managed:
                continue
            out.append(info)
        for key, (name, meta) in by_path.items():  # registered but missing on disk
            if key not in seen:
                out.append(WorktreeInfo(name=name, path=meta["path"], branch=meta.get("branch"),
                                        task_id=meta.get("task_id"), role=meta.get("role"), managed=True, exists=False))
        return out

    def get(self, name_or_path: str) -> WorktreeInfo:
        target = name_or_path.lower()
        for info in self.list():
            if info.name.lower() == target or info.path.lower() == str(Path(name_or_path).resolve()).lower():
                return info
        raise GitError(f"unknown worktree: {name_or_path}")

    # -- lifecycle ---------------------------------------------------------------
    def create(self, task_id: str, role: str, *, base_ref: str = "HEAD", name: str | None = None) -> WorktreeInfo:
        with self._lock:
            if not self.has_commits():
                raise GitError("the repository has no commits yet; create an initial commit first")
            name = name or f"{_slug(self.repo_root.name, 30)}-{_slug(task_id, 20)}-{_slug(role, 20)}"
            branch = f"{BRANCH_PREFIX}/{_branch_part(task_id)}/{_branch_part(role)}"
            path = (self.base_dir / name).resolve()
            registry = self._registry()
            if name in registry and Path(registry[name]["path"]).exists():
                return self.get(name)  # idempotent: retries reuse the preserved worktree
            self.base_dir.mkdir(parents=True, exist_ok=True)
            branch_exists = run_git(["rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"], self.repo_root,
                                    check=False).returncode == 0
            if branch_exists:
                run_git(["worktree", "add", str(path), branch], self.repo_root)
            else:
                run_git(["worktree", "add", "-b", branch, str(path), base_ref], self.repo_root)
            base_sha = run_git(["rev-parse", base_ref], self.repo_root).stdout.strip() if not branch_exists else None
            registry[name] = {
                "path": str(path), "branch": branch, "task_id": task_id, "role": role,
                "base_ref": base_sha or registry.get(name, {}).get("base_ref"), "created_at": now_iso(),
            }
            self._write_registry(registry)
        self._emit("worktree.created", name=name, path=str(path), branch=branch, task_id=task_id, role=role,
                   base_ref=registry[name]["base_ref"])
        return self.get(name)

    def remove(self, name: str, *, force: bool = False, delete_branch: bool = False) -> None:
        with self._lock:
            info = self.get(name)
            if not info.managed:
                raise GitError(f"refusing to remove a worktree not created by Increscent: {info.path}")
            owner = self._lock_owner(info.name)
            if owner:
                raise WorktreeBusy(f"worktree {info.name} is in use by {owner}")
            if info.exists:
                if not force and self.changed_files(info.name):
                    raise GitError(f"worktree {info.name} has uncommitted changes (use force to discard)")
                run_git(["worktree", "remove", *(["--force"] if force else []), info.path], self.repo_root)
            else:
                run_git(["worktree", "prune"], self.repo_root, check=False)
            if delete_branch and info.branch:
                current = self.current_branch()
                if current == info.branch:
                    raise GitError("refusing to delete the currently checked-out branch")
                run_git(["branch", "-D", info.branch], self.repo_root)
            registry = self._registry()
            registry.pop(info.name, None)
            self._write_registry(registry)
        self._emit("worktree.removed", name=info.name, path=info.path, branch=info.branch,
                   branch_deleted=bool(delete_branch))

    # -- locking --------------------------------------------------------------------
    def _lock_file(self, name: str, owner: str = "") -> FileLock:
        return FileLock(self.locks_dir / f"{name}.lock", owner)

    def _lock_owner(self, name: str) -> Optional[str]:
        lock = self._lock_file(name)
        info = lock.read()
        if not info or lock.is_stale():
            return None
        return str(info.get("owner"))

    def acquire(self, name: str, owner: str) -> FileLock:
        lock = FileLock(self.locks_dir / f"{name}.lock", owner, metadata={"worktree": name})
        if not lock.acquire():
            raise WorktreeBusy(f"worktree {name} is already in use by {self._lock_owner(name)}")
        return lock

    # -- working with changes ------------------------------------------------------------
    def status(self, name: str) -> dict[str, Any]:
        info = self.get(name)
        files = self._status_entries(info.path)
        return {"name": info.name, "branch": info.branch, "head": info.head, "clean": not files,
                "files": [{"status": s, "path": p} for s, p in files]}

    def _status_entries(self, path: str) -> list[tuple[str, str]]:
        out = run_git(["status", "--porcelain", "--untracked-files=all"], path).stdout
        entries = []
        for line in out.splitlines():
            if not line.strip():
                continue
            status, file = line[:2].strip(), line[3:].strip().strip('"')
            if " -> " in file:
                file = file.split(" -> ", 1)[1]
            entries.append((status, file))
        return entries

    def changed_files(self, name: str) -> list[str]:
        return [p for _, p in self._status_entries(self.get(name).path)]

    def commit(self, name: str, message: str, *, author: str = "Increscent Agent") -> Optional[str]:
        """Commit all changes in the worktree. Returns the sha, or None when clean."""
        from ..tool_system.safety import is_secret_file

        info = self.get(name)
        files = self.changed_files(name)
        if not files:
            return None
        secret = [f for f in files if is_secret_file(f)]
        if secret:
            raise GitError(f"refusing to commit secret files (add them to .gitignore): {', '.join(secret)}")
        role = _slug(info.role or "agent", 20)
        run_git(["add", "-A"], info.path)
        run_git(["-c", f"user.name={author} ({role})", "-c", f"user.email=agent+{role}@increscent.local",
                 "commit", "-q", "-m", message], info.path)
        sha = run_git(["rev-parse", "HEAD"], info.path).stdout.strip()
        self._emit("commit.created", worktree=info.name, branch=info.branch, sha=sha, task_id=info.task_id,
                   files=files, message=message.splitlines()[0][:200])
        return sha

    def diff(self, name: str, *, base_ref: str | None = None, stat: bool = False) -> str:
        """Committed branch changes since its base plus uncommitted work (secrets redacted)."""
        info = self.get(name)
        base = base_ref or info.base_ref or "HEAD"
        args = ["diff", "--stat"] if stat else ["diff"]
        committed = run_git([*args, f"{base}...HEAD"], info.path, check=False).stdout
        working = run_git([*args, "HEAD"], info.path, check=False).stdout
        untracked = [p for s, p in self._status_entries(info.path) if s == "??"]
        parts = [committed.strip(), working.strip()]
        if untracked:
            parts.append("Untracked files:\n" + "\n".join(f"  {p}" for p in untracked))
        return redact_secrets("\n".join(p for p in parts if p))

    def merge_branch(self, name: str, branch: str, *, message: str | None = None) -> str:
        """Merge ``branch`` into the worktree's branch (never main). Aborts on conflict."""
        info = self.get(name)
        if info.branch in (None, "main", "master"):
            raise GitError("refusing to merge into a main branch")
        proc = run_git(["-c", "user.name=Increscent Orchestrator", "-c", "user.email=orchestrator@increscent.local",
                        "merge", "--no-ff", "--no-edit", *(["-m", message] if message else []), branch],
                       info.path, check=False)
        if proc.returncode != 0:
            conflicts = run_git(["diff", "--name-only", "--diff-filter=U"], info.path, check=False).stdout.split()
            run_git(["merge", "--abort"], info.path, check=False)
            raise MergeConflict(branch, conflicts)
        sha = run_git(["rev-parse", "HEAD"], info.path).stdout.strip()
        self._emit("merge.completed", worktree=info.name, target_branch=info.branch, source_branch=branch, sha=sha)
        return sha
