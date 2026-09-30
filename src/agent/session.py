"""Session management with persistence.

A session file stores everything needed to continue a conversation after the
CLI restarts:

* the full neutral conversation: user and assistant messages, tool calls,
  tool results, and per-message timestamps
* provider/model and creation/update timestamps
* ``metadata``: project root, git root, working directory, app version,
  platform. The system prompt itself is *not* stored; it is rebuilt from the
  live project on every request so git status and instructions stay current.
* ``state``: restorable tool-context state (working directory, todos, tasks)

Files live in ``<user dir>/sessions/<session_id>.json`` (``~/.clawd/sessions``
by default) and are written atomically as UTF-8.
"""

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .conversation import Conversation

SESSION_SCHEMA_VERSION = 2
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,127}$")


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def sessions_dir(directory: str | Path | None = None) -> Path:
    if directory is not None:
        return Path(directory)
    from src.config import get_user_dir

    return get_user_dir() / "sessions"


def validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID_RE.match(session_id) or ".." in session_id:
        raise ValueError(f"Invalid session id: {session_id!r}")
    return session_id


def new_session_id() -> str:
    """Sortable, collision-resistant id, e.g. ``20260930_193512_a1b2c3``."""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def capture_tool_state(tool_context: Any) -> dict[str, Any]:
    """Extract the JSON-serialisable parts of a ToolContext worth restoring."""
    state: dict[str, Any] = {}
    cwd = getattr(tool_context, "cwd", None)
    if cwd is not None:
        state["cwd"] = str(cwd)
    for name in ("todos", "tasks"):
        value = getattr(tool_context, name, None)
        if value:
            try:
                state[name] = json.loads(json.dumps(value, default=str))
            except Exception:
                pass
    if getattr(tool_context, "plan_mode", False):
        state["plan_mode"] = True
    return state


def restore_tool_state(tool_context: Any, state: dict[str, Any] | None) -> list[str]:
    """Apply saved state to a ToolContext. Returns human-readable warnings."""
    warnings: list[str] = []
    if not state:
        return warnings
    cwd = state.get("cwd")
    if isinstance(cwd, str):
        try:
            resolved = tool_context.ensure_allowed_path(cwd)
            if resolved.is_dir():
                tool_context.cwd = resolved
            else:
                warnings.append(f"Saved working directory no longer exists: {cwd}")
        except Exception:
            warnings.append(f"Saved working directory is outside this workspace, ignored: {cwd}")
    if isinstance(state.get("todos"), list):
        tool_context.todos = state["todos"]
    if isinstance(state.get("tasks"), dict):
        tool_context.tasks = state["tasks"]
    if state.get("plan_mode"):
        tool_context.plan_mode = True
    return warnings


@dataclass
class Session:
    """Session manager with persistence."""
    session_id: str
    provider: str
    model: str
    conversation: Conversation = field(default_factory=Conversation)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form. Secret values never reach disk (see src/security/secrets.py)."""
        from src.security.secrets import redact_structure

        return {
            "schema_version": SESSION_SCHEMA_VERSION,
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": redact_structure(self.metadata),
            "state": redact_structure(self.state),
            "conversation": redact_structure(self.conversation.to_dict()),
        }

    def save(self, directory: str | Path | None = None) -> Path:
        """Save session to disk atomically. Returns the file path."""
        validate_session_id(self.session_id)
        target_dir = sessions_dir(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        session_file = target_dir / f"{self.session_id}.json"

        self.updated_at = _now()
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.session_id}.", suffix=".tmp", dir=str(target_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_name, session_file)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        if os.name != "nt":
            try:
                os.chmod(session_file, 0o600)
            except OSError:
                pass

        from src.observability import log_event

        log_event("session.saved", session_id=self.session_id, path=str(session_file),
                  messages=len(self.conversation.messages))
        return session_file

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        return cls(
            session_id=data["session_id"],
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            conversation=Conversation.from_dict(data.get("conversation") or {}),
            created_at=data.get("created_at") or _now(),
            updated_at=data.get("updated_at") or data.get("created_at") or _now(),
            metadata=dict(data.get("metadata") or {}),
            state=dict(data.get("state") or {}),
        )

    @classmethod
    def load(cls, session_id: str, directory: str | Path | None = None) -> Optional["Session"]:
        """Load session from disk (None if it does not exist or is invalid)."""
        try:
            validate_session_id(session_id)
        except ValueError:
            return None
        session_file = sessions_dir(directory) / f"{session_id}.json"
        if not session_file.exists():
            return None

        with open(session_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        session = cls.from_dict(data)

        from src.observability import log_event

        log_event("session.loaded", session_id=session.session_id, path=str(session_file),
                  messages=len(session.conversation.messages),
                  schema_version=data.get("schema_version", 1))
        return session

    @classmethod
    def create(cls, provider: str, model: str, metadata: dict[str, Any] | None = None) -> "Session":
        """Create a new session."""
        base_meta: dict[str, Any] = {"platform": platform.system(), "python": platform.python_version()}
        try:
            from src import __version__

            base_meta["app_version"] = __version__
        except Exception:
            pass
        base_meta.update(metadata or {})
        return cls(session_id=new_session_id(), provider=provider, model=model, metadata=base_meta)


def list_sessions(directory: str | Path | None = None, project_root: str | Path | None = None) -> list[dict[str, Any]]:
    """Summaries of saved sessions, newest first (optionally for one project)."""
    target = sessions_dir(directory)
    if not target.is_dir():
        return []
    want_root = str(Path(project_root).resolve()) if project_root is not None else None
    out: list[dict[str, Any]] = []
    for path in target.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = data.get("metadata") or {}
        if want_root is not None and meta.get("project_root") != want_root:
            continue
        out.append({
            "session_id": data.get("session_id", path.stem),
            "updated_at": data.get("updated_at", ""),
            "provider": data.get("provider"),
            "model": data.get("model"),
            "project_root": meta.get("project_root"),
            "messages": len((data.get("conversation") or {}).get("messages", [])),
            "_mtime": path.stat().st_mtime,
        })
    out.sort(key=lambda d: (d["updated_at"], d["_mtime"]), reverse=True)
    for d in out:
        d.pop("_mtime", None)
    return out


def latest_session_id(directory: str | Path | None = None, project_root: str | Path | None = None) -> Optional[str]:
    sessions = list_sessions(directory, project_root)
    return sessions[0]["session_id"] if sessions else None
