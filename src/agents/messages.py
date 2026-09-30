"""Structured, auditable agent communication.

Agents never talk to each other or reach each other's shells. Every message
goes through the :class:`MessageBus`, is validated, redacted and appended to
``messages.jsonl`` in the project state directory. Agents may only address
the orchestrator; the orchestrator relays assignments and results.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..orchestration.state import now_iso
from ..security.secrets import redact_structure
from .models import new_id

ORCHESTRATOR = "orchestrator"
HUMAN = "human"


class MessageType(str, Enum):
    TASK_ASSIGNMENT = "TASK_ASSIGNMENT"
    TASK_RESULT = "TASK_RESULT"
    BLOCKED = "BLOCKED"
    QUESTION = "QUESTION"
    ERROR = "ERROR"
    REVIEW_REQUEST = "REVIEW_REQUEST"
    REVIEW_RESULT = "REVIEW_RESULT"


# Which message types each kind of sender may emit.
AGENT_SENDABLE = frozenset({MessageType.TASK_RESULT, MessageType.BLOCKED, MessageType.QUESTION,
                            MessageType.ERROR, MessageType.REVIEW_RESULT})


@dataclass(frozen=True)
class AgentMessage:
    type: MessageType
    sender: str
    recipient: str
    content: str
    task_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("M"))
    created_at: str = field(default_factory=now_iso)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["type"] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        data = dict(data)
        data["type"] = MessageType(data["type"])
        return cls(**data)


class MessageRejected(ValueError):
    pass


class MessageBus:
    def __init__(self, state_dir: Path | None = None, *, events: Any = None) -> None:
        self.path = (state_dir / "messages.jsonl") if state_dir is not None else None
        self._memory: list[AgentMessage] = []
        self._lock = threading.Lock()
        self._events = events

    def send(self, message: AgentMessage, *, sender_is_agent: bool = False) -> AgentMessage:
        if sender_is_agent:
            if message.type not in AGENT_SENDABLE:
                raise MessageRejected(f"agents may not send {message.type.value} messages")
            if message.recipient != ORCHESTRATOR:
                raise MessageRejected("agents may only message the orchestrator (no direct agent-to-agent channel)")
        clean = AgentMessage.from_dict(redact_structure(message.to_dict()))
        with self._lock:
            self._memory.append(clean)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(clean.to_dict(), ensure_ascii=False) + "\n")
        if self._events is not None:
            self._events.emit("message.sent", message_id=clean.id, type=clean.type.value, sender=clean.sender,
                              recipient=clean.recipient, task_id=clean.task_id)
        return clean

    def all(self) -> list[AgentMessage]:
        if self.path is not None and self.path.exists():
            return [AgentMessage.from_dict(json.loads(line))
                    for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return list(self._memory)

    def query(self, *, task_id: str | None = None, type: MessageType | None = None,
              recipient: str | None = None) -> list[AgentMessage]:
        return [m for m in self.all()
                if (task_id is None or m.task_id == task_id)
                and (type is None or m.type == type)
                and (recipient is None or m.recipient == recipient)]
