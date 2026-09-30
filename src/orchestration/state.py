"""Durable local state for multi-agent runs.

State lives in ``<git-common-dir>/increscent/``, inside ``.git``. It is shared
by the main checkout and every worktree, survives restarts, and can never be
committed by accident. Projects without git fall back to
``<project>/.clawd/state/``.

Files are plain JSON / JSON-lines written atomically. That is enough for
local, single-machine orchestration, so no database is used.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from ..branding import PROJECT_DIR_NAME, STATE_DIR_NAME
from ..observability import log_event, redact


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def git_common_dir(project_root: str | Path) -> Path | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"], cwd=str(project_root), capture_output=True, text=True,
            timeout=15, stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
        )
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    path = Path(proc.stdout.strip())
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def resolve_state_dir(project_root: str | Path, *, create: bool = True) -> Path:
    common = git_common_dir(project_root)
    state = (common / STATE_DIR_NAME) if common else (Path(project_root).resolve() / PROJECT_DIR_NAME / "state")
    if create:
        state.mkdir(parents=True, exist_ok=True)
    return state


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        for attempt in range(20):  # Windows: a concurrent reader can briefly hold the target
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path, default: Any) -> Any:
    for attempt in range(20):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default
        except (PermissionError, json.JSONDecodeError):
            if attempt == 19:
                raise
            time.sleep(0.05)
    return default


def pid_alive(pid: int) -> bool:
    """Whether a process exists (never signals it; os.kill(pid, 0) would kill on Windows)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FileLock:
    """Cross-process exclusive lock file (``O_CREAT | O_EXCL``) with stale detection."""

    def __init__(self, path: Path, owner: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.path = path
        self.owner = owner
        self.metadata = metadata or {}
        self.acquired = False

    def read(self) -> dict[str, Any] | None:
        return read_json(self.path, None)

    def is_stale(self) -> bool:
        info = self.read()
        if not info:
            return True
        return not pid_alive(int(info.get("pid", -1)))

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = self.read()
                if info and info.get("owner") == self.owner and int(info.get("pid", -1)) == os.getpid():
                    self.acquired = True  # re-entrant for the same owner in this process
                    return True
                if self.is_stale():
                    try:
                        self.path.unlink()
                    except OSError:
                        return False
                    continue
                return False
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"owner": self.owner, "pid": os.getpid(), "acquired_at": now_iso(), **self.metadata}, f)
            self.acquired = True
            return True
        return False

    def release(self) -> None:
        info = self.read()
        if info and info.get("owner") == self.owner:
            try:
                self.path.unlink()
            except OSError:
                pass
        self.acquired = False


class EventRecorder:
    """Persist structured events (JSON lines) and mirror them to the logger."""

    def __init__(self, state_dir: Path | None) -> None:
        self.path = (state_dir / "events.jsonl") if state_dir is not None else None
        self._lock = threading.Lock()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> None:
        self._listeners.append(listener)

    def emit(self, event: str, *, mirror_to_log: bool = True, **fields: Any) -> dict[str, Any]:
        record = {"ts": now_iso(), "event": event, **redact(fields)}
        if mirror_to_log:
            log_event(event, **fields)
        if self.path is not None:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception:
                pass
        return record

    def read(self, **filters: Any) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if all(record.get(k) == v for k, v in filters.items()):
                out.append(record)
        return out

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.read())
