"""Durable agent registry (``agents.json`` in the project state directory)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from ..orchestration.state import atomic_write_json, read_json
from .models import Agent

SCHEMA_VERSION = 1


class AgentStore:
    def __init__(self, state_dir: Path) -> None:
        self.path = Path(state_dir) / "agents.json"
        self._lock = threading.RLock()

    def _load(self) -> dict[str, dict]:
        data = read_json(self.path, {"schema_version": SCHEMA_VERSION, "agents": {}})
        return dict(data.get("agents", {}))

    def save(self, agent: Agent) -> None:
        with self._lock:
            agents = self._load()
            agents[agent.id] = agent.to_dict()
            atomic_write_json(self.path, {"schema_version": SCHEMA_VERSION, "agents": agents})

    def get(self, agent_id: str) -> Optional[Agent]:
        with self._lock:
            data = self._load().get(agent_id)
        return Agent.from_dict(data) if data else None

    def list(self, *, task_id: str | None = None) -> list[Agent]:
        with self._lock:
            agents = [Agent.from_dict(a) for a in self._load().values()]
        if task_id is not None:
            agents = [a for a in agents if a.task_id == task_id]
        return sorted(agents, key=lambda a: a.created_at)
