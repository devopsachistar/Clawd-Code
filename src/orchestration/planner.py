"""Turn a goal into a task graph specification.

Phase 2 ships a deterministic template planner. It covers the standard
delivery chain; custom plans can be supplied as JSON. LLM-generated plans
(the Architect agent proposing the graph) are a Phase 3 item.

Standard chain::

    Architecture -> Data model -> Backend -> Frontend -> Integration tests -> Code review
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    description: str
    profile: str
    depends_on: tuple[str, ...] = ()
    priority: int = 0
    acceptance: tuple[str, ...] = field(default_factory=tuple)

    def full_description(self) -> str:
        text = self.description.strip()
        if self.acceptance:
            text += "\n\nAcceptance criteria:\n" + "\n".join(f"- {a}" for a in self.acceptance)
        return text


def standard_plan(goal: str) -> list[TaskSpec]:
    return [
        TaskSpec("architecture", "Design the architecture",
                 f"Design the solution for: {goal}. Define modules, their responsibilities and interfaces.",
                 "Architect", (), 50,
                 ("docs/ARCHITECTURE.md describes modules, interfaces and data model",)),
        TaskSpec("database", "Implement the data model and persistence",
                 f"Implement the data model and local persistence layer for: {goal}. No external database servers.",
                 "DatabaseEngineer", ("architecture",), 40,
                 ("Persistence module with load/save and a documented schema",)),
        TaskSpec("backend", "Implement the core service / business logic",
                 f"Implement the domain/service layer for: {goal}, on top of the persistence layer.",
                 "BackendDeveloper", ("database",), 30,
                 ("Service API covers create/list/update/delete", "Input validation with clear errors")),
        TaskSpec("frontend", "Implement the user interface",
                 f"Implement the user-facing interface for: {goal}, using only the service API.",
                 "FrontendDeveloper", ("backend",), 20,
                 ("Every service operation is reachable from the interface",)),
        TaskSpec("integration-tests", "Write and run integration tests",
                 f"Write integration tests for: {goal} covering the interface through to persistence, and run them.",
                 "QAEngineer", ("frontend",), 10,
                 ("Automated tests for the main user flows", "All tests pass")),
        TaskSpec("review", "Review the integrated changes",
                 f"Review all changes made for: {goal}. Check correctness, tests and risks; give a verdict.",
                 "CodeReviewer", ("integration-tests",), 0,
                 ("Clear approve/request_changes verdict with findings",)),
    ]


def load_plan_file(path: str | Path) -> list[TaskSpec]:
    """JSON: ``[{"key", "title", "description", "profile", "depends_on": [...], "priority", "acceptance": [...]}]``."""
    raw: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("tasks") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise ValueError("plan file must contain a non-empty list of tasks")
    specs: list[TaskSpec] = []
    keys: set[str] = set()
    for item in items:
        for required in ("key", "title", "profile"):
            if not item.get(required):
                raise ValueError(f"plan task missing '{required}': {item}")
        if item["key"] in keys:
            raise ValueError(f"duplicate plan key {item['key']}")
        keys.add(item["key"])
        specs.append(TaskSpec(
            key=str(item["key"]), title=str(item["title"]), description=str(item.get("description", "")),
            profile=str(item["profile"]), depends_on=tuple(item.get("depends_on", [])),
            priority=int(item.get("priority", 0)), acceptance=tuple(item.get("acceptance", [])),
        ))
    for spec in specs:
        for dep in spec.depends_on:
            if dep not in keys:
                raise ValueError(f"task {spec.key} depends on unknown key {dep}")
    return specs
