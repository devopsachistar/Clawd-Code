"""Agent data model: identity, configuration, lifecycle and structured results."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

from ..orchestration.state import now_iso


class AgentStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    WAITING = "WAITING"      # waiting for an approval decision
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


TERMINAL_AGENT_STATUSES = frozenset({AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED})

# Allowed lifecycle transitions.
_TRANSITIONS: dict[AgentStatus, frozenset[AgentStatus]] = {
    AgentStatus.CREATED: frozenset({AgentStatus.RUNNING, AgentStatus.CANCELLED, AgentStatus.FAILED}),
    AgentStatus.RUNNING: frozenset({AgentStatus.WAITING, AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED}),
    AgentStatus.WAITING: frozenset({AgentStatus.RUNNING, AgentStatus.FAILED, AgentStatus.CANCELLED}),
    AgentStatus.COMPLETED: frozenset(),
    AgentStatus.FAILED: frozenset(),
    AgentStatus.CANCELLED: frozenset(),
}


class InvalidTransition(RuntimeError):
    pass


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class AgentBudget:
    """Hard limits that stop runaway agents."""

    max_iterations: int = 30          # model turns
    max_tool_calls: int = 100
    max_seconds: int = 1800           # wall clock
    max_tokens: Optional[int] = None  # input + output, when the provider reports usage

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AgentBudget":
        data = data or {}
        return cls(**{k: data[k] for k in ("max_iterations", "max_tool_calls", "max_seconds", "max_tokens") if k in data})


@dataclass
class Agent:
    """One agent instance working on one task."""

    name: str
    role: str
    profile: str
    system_prompt: str
    provider: str
    model: str
    allowed_tools: list[str]
    permissions: dict[str, Any]          # PermissionPolicy.to_dict()
    working_directory: str
    task_id: Optional[str] = None
    budget: AgentBudget = field(default_factory=AgentBudget)
    timeout_s: int = 1800
    status: AgentStatus = AgentStatus.CREATED
    id: str = field(default_factory=lambda: new_id("A"))
    created_at: str = field(default_factory=now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    attempt: int = 1
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    usage: Optional[dict[str, Any]] = None
    session_id: Optional[str] = None

    def transition(self, new_status: AgentStatus) -> None:
        if new_status == self.status:
            return
        if new_status not in _TRANSITIONS[self.status]:
            raise InvalidTransition(f"agent {self.id}: {self.status.value} -> {new_status.value} is not allowed")
        self.status = new_status
        if new_status is AgentStatus.RUNNING and self.started_at is None:
            self.started_at = now_iso()
        if new_status in TERMINAL_AGENT_STATUSES:
            self.finished_at = now_iso()

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_AGENT_STATUSES

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["budget"] = self.budget.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        data = dict(data)
        data["status"] = AgentStatus(data.get("status", "CREATED"))
        data["budget"] = AgentBudget.from_dict(data.get("budget"))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


RESULT_STATUSES = ("completed", "failed", "blocked")


@dataclass
class AgentResult:
    """Structured outcome every agent run produces (never just free text)."""

    task_id: Optional[str]
    agent_id: str
    status: str                      # completed | failed | blocked
    summary: str
    files_changed: list[str] = field(default_factory=list)
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    risks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    commit: Optional[str] = None
    next_action: str = ""
    # Provenance / diagnostics
    reported_by_agent: bool = False  # False when synthesised because the agent never called ReportResult
    stop_reason: Optional[str] = None
    review_verdict: Optional[str] = None  # approve | request_changes (reviewers)
    error: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise ValueError(f"invalid result status {self.status!r}; expected one of {RESULT_STATUSES}")
        for name in ("tests_run", "tests_passed", "tests_failed"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("files_changed", "risks", "warnings"):
            value = getattr(self, name)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"{name} must be a list of strings")
        if self.review_verdict not in (None, "approve", "request_changes"):
            raise ValueError("review_verdict must be 'approve' or 'request_changes'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentResult":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
