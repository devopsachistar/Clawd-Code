"""Task graph data model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from ..agents.models import new_id
from ..orchestration.state import now_iso


class TaskStatus(str, Enum):
    PENDING = "PENDING"      # waiting for dependencies
    READY = "READY"          # dependencies complete; can be assigned
    RUNNING = "RUNNING"      # an agent is working on it
    BLOCKED = "BLOCKED"      # dependency failed, paused, merge conflict, or agent reported blocked
    REVIEW = "REVIEW"        # awaiting human review/approval (project task)
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"        # out of retries
    CANCELLED = "CANCELLED"


TERMINAL_TASK_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED})
PAUSED_PREFIX = "paused"


@dataclass
class Task:
    title: str
    description: str
    graph_id: str
    profile: Optional[str] = None                 # agent profile that should execute it
    kind: str = "task"                            # "project" (root) | "task"
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0                             # higher runs first among READY tasks
    owner: str = "orchestrator"
    agent: Optional[str] = None                   # current/last agent id
    dependencies: list[str] = field(default_factory=list)
    worktree: Optional[str] = None
    branch: Optional[str] = None
    files_changed: list[str] = field(default_factory=list)
    tests: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    max_attempts: int = 3
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    blocked_reason: Optional[str] = None
    history: list[dict[str, Any]] = field(default_factory=list)  # one entry per attempt
    id: str = field(default_factory=lambda: new_id("T"))
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES

    @property
    def is_paused(self) -> bool:
        return self.status is TaskStatus.BLOCKED and (self.blocked_reason or "").startswith(PAUSED_PREFIX)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        data = dict(data)
        data["status"] = TaskStatus(data.get("status", "PENDING"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TaskGraph:
    goal: str
    base_ref: Optional[str] = None                 # commit the graph started from
    root_task_id: Optional[str] = None
    status: str = "PLANNED"                        # PLANNED | RUNNING | REVIEW | MERGED | FAILED | CANCELLED
    integration_branch: Optional[str] = None
    settings: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("G"))
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskGraph":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
