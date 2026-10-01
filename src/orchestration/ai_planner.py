"""AI planning: natural-language requirement -> validated plan -> existing Task Graph.

    Requirement
        |  build_provider_request() (provider-neutral; same path as the agent loop)
        v
    LLM (any configured BaseProvider)
        |  JSON response
        v
    parse_plan() + validate  -> ImplementationPlan (typed)
        |  to_task_specs()
        v
    Orchestrator.create_graph()  -> TaskStore (Phase 2 graph; its own validation stays authoritative)

The planner never fabricates output. Provider errors, invalid JSON and plans
that fail validation raise :class:`PlanningError` with a clear message.
Nothing is retried, and nothing is written to the task graph unless the
whole plan is valid.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from ..agent.message_format import build_provider_request, resolve_wire_format
from ..observability import log_event
from ..providers.base import BaseProvider
from ..security.secrets import redact_structure
from .planner import TaskSpec

ARCHITECTURE_KEYS = ("frontend", "backend", "database", "infrastructure")
_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class PlanningError(Exception):
    """The plan could not be produced. ``issues`` lists every validation problem found."""

    def __init__(self, message: str, issues: Iterable[str] = ()) -> None:
        self.issues = list(issues)
        detail = ("\n  - " + "\n  - ".join(self.issues)) if self.issues else ""
        super().__init__(message + detail)


@dataclass(frozen=True)
class PlannedModule:
    name: str
    description: str


@dataclass(frozen=True)
class PlannedTask:
    id: str
    title: str
    description: str
    role: str
    dependencies: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImplementationPlan:
    project_name: str
    summary: str
    requirements: tuple[str, ...]
    assumptions: tuple[str, ...]
    architecture: dict[str, str]
    modules: tuple[PlannedModule, ...]
    tasks: tuple[PlannedTask, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": {"name": self.project_name, "summary": self.summary},
            "requirements": list(self.requirements),
            "assumptions": list(self.assumptions),
            "architecture": dict(self.architecture),
            "modules": [asdict(m) for m in self.modules],
            "tasks": [{**asdict(t), "dependencies": list(t.dependencies),
                       "acceptance_criteria": list(t.acceptance_criteria)} for t in self.tasks],
        }

    def to_task_specs(self, role_to_profile: dict[str, str]) -> list[TaskSpec]:
        """Convert to the existing TaskSpec type consumed by Orchestrator.create_graph()."""
        count = len(self.tasks)
        return [
            TaskSpec(
                key=t.id,
                title=t.title,
                description=t.description,
                profile=role_to_profile[t.role],
                depends_on=t.dependencies,
                priority=count - i,  # earlier planned tasks first among READY ones
                acceptance=t.acceptance_criteria,
            )
            for i, t in enumerate(self.tasks)
        ]


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

def extract_json(text: str) -> Any:
    """The JSON value in a model response: the whole body, a fenced ```json block, or one {...} object."""
    body = (text or "").strip()
    if not body:
        raise PlanningError("the model returned an empty response")
    candidates = [body]
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)\n?```", body, re.S | re.I)
    if len(fenced) == 1:
        candidates.insert(0, fenced[0])
    if "{" in body and "}" in body:
        candidates.append(body[body.index("{"): body.rindex("}") + 1])
    last_error = ""
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_error = str(e)
    raise PlanningError(f"the model response is not valid JSON ({last_error})")


def _str(value: Any, path: str, issues: list[str], *, required: bool = True) -> str:
    if not isinstance(value, str):
        issues.append(f"{path} must be a string")
        return ""
    if required and not value.strip():
        issues.append(f"{path} must not be empty")
    return value.strip()


def _str_list(value: Any, path: str, issues: list[str]) -> tuple[str, ...]:
    if not isinstance(value, list):
        issues.append(f"{path} must be a list of strings")
        return ()
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            issues.append(f"{path}[{i}] must be a non-empty string")
        else:
            out.append(item.strip())
    return tuple(out)


def _find_cycle(tasks: dict[str, tuple[str, ...]]) -> Optional[list[str]]:
    state: dict[str, int] = {}

    def visit(node: str, path: list[str]) -> Optional[list[str]]:
        if state.get(node) == 1:
            return path[path.index(node):] + [node]
        if state.get(node) == 2:
            return None
        state[node] = 1
        for dep in tasks.get(node, ()):
            if dep in tasks:
                found = visit(dep, path + [node])
                if found:
                    return found
        state[node] = 2
        return None

    for node in tasks:
        found = visit(node, [])
        if found:
            return found
    return None


def parse_plan(data: Any, valid_roles: Iterable[str]) -> ImplementationPlan:
    """Validate raw plan data. Raises PlanningError listing *all* problems."""
    roles = {r.lower() for r in valid_roles}
    issues: list[str] = []
    if not isinstance(data, dict):
        raise PlanningError("invalid plan", ["top level must be a JSON object"])
    for key in ("project", "requirements", "assumptions", "architecture", "modules", "tasks"):
        if key not in data:
            issues.append(f"missing required field '{key}'")
    if issues:
        raise PlanningError("invalid plan", issues)

    project = data["project"]
    if not isinstance(project, dict):
        issues.append("project must be an object")
        project = {}
    name = _str(project.get("name"), "project.name", issues)
    summary = _str(project.get("summary"), "project.summary", issues)
    requirements = _str_list(data["requirements"], "requirements", issues)
    if isinstance(data["requirements"], list) and not data["requirements"]:
        issues.append("requirements must not be empty")
    assumptions = _str_list(data["assumptions"], "assumptions", issues)

    architecture: dict[str, str] = {}
    if not isinstance(data["architecture"], dict):
        issues.append("architecture must be an object")
    else:
        for key in ARCHITECTURE_KEYS:
            if key not in data["architecture"]:
                issues.append(f"missing required field 'architecture.{key}'")
            else:
                architecture[key] = _str(data["architecture"][key], f"architecture.{key}", issues, required=False)

    modules: list[PlannedModule] = []
    if not isinstance(data["modules"], list):
        issues.append("modules must be a list")
    else:
        for i, m in enumerate(data["modules"]):
            if not isinstance(m, dict):
                issues.append(f"modules[{i}] must be an object")
                continue
            modules.append(PlannedModule(_str(m.get("name"), f"modules[{i}].name", issues),
                                         _str(m.get("description"), f"modules[{i}].description", issues, required=False)))

    tasks: list[PlannedTask] = []
    raw_tasks = data["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        issues.append("tasks must be a non-empty list")
        raw_tasks = []
    seen: set[str] = set()
    for i, t in enumerate(raw_tasks):
        where = f"tasks[{i}]"
        if not isinstance(t, dict):
            issues.append(f"{where} must be an object")
            continue
        for key in ("id", "title", "description", "dependencies", "role", "acceptance_criteria"):
            if key not in t:
                issues.append(f"missing required field '{where}.{key}'")
        task_id = t.get("id")
        if not isinstance(task_id, str) or not _TASK_ID.match(task_id):
            issues.append(f"{where}.id must be a lowercase identifier (a-z, 0-9, '-', '_'), got {task_id!r}")
            task_id = str(task_id)
        elif task_id in seen:
            issues.append(f"duplicate task id '{task_id}'")
        seen.add(task_id)
        role = t.get("role")
        if not isinstance(role, str) or role.strip().lower() not in roles:
            issues.append(f"{where}.role {role!r} is not a valid agent role (valid: {', '.join(sorted(roles))})")
            role = ""
        deps = t.get("dependencies", [])
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            issues.append(f"{where}.dependencies must be a list of task ids")
            deps = []
        tasks.append(PlannedTask(
            id=task_id,
            title=_str(t.get("title"), f"{where}.title", issues),
            description=_str(t.get("description"), f"{where}.description", issues, required=False),
            role=role.strip().lower(),
            dependencies=tuple(deps),
            acceptance_criteria=_str_list(t.get("acceptance_criteria", []), f"{where}.acceptance_criteria", issues),
        ))

    ids = {t.id for t in tasks}
    for t in tasks:
        for dep in t.dependencies:
            if dep == t.id:
                issues.append(f"task '{t.id}' depends on itself")
            elif dep not in ids:
                issues.append(f"task '{t.id}' depends on unknown task '{dep}'")
    if not any("depends on" in i or "duplicate" in i for i in issues):
        cycle = _find_cycle({t.id: t.dependencies for t in tasks})
        if cycle:
            issues.append("dependency cycle: " + " -> ".join(cycle))

    if issues:
        raise PlanningError("invalid plan", issues)
    return ImplementationPlan(name, summary, requirements, assumptions, architecture, tuple(modules), tuple(tasks))


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the planning engine of an AI software factory.
Analyse the user's software requirement and produce an implementation plan.

Respond with ONE JSON object and nothing else (no prose, no markdown). Schema:
{{
  "project": {{"name": string, "summary": string}},
  "requirements": [string, ...],          // concrete functional requirements (non-empty)
  "assumptions": [string, ...],           // assumptions you made (may be empty)
  "architecture": {{"frontend": string, "backend": string, "database": string, "infrastructure": string}},
  "modules": [{{"name": string, "description": string}}, ...],
  "tasks": [
    {{
      "id": string,                       // unique lowercase id, e.g. "architecture", "backend-api"
      "title": string,
      "description": string,
      "dependencies": [task id, ...],     // ids of tasks that must finish first; no cycles
      "role": string,                     // one of: {roles}
      "acceptance_criteria": [string, ...]
    }}
  ]
}}

Rules:
- Order tasks so dependencies come first. Start with an architecture task (role "architect").
- Include implementation, test (role "qa") and review (role "reviewer") tasks.
- Keep the plan small and implementable: typically 4-10 tasks.
- Do not invent technologies the requirement rules out; state choices in "assumptions".
"""


class Planner:
    """Natural-language requirement -> validated :class:`ImplementationPlan`."""

    def __init__(self, provider: BaseProvider, profiles) -> None:
        """``profiles``: the configurable ProfileRegistry; its roles are the valid task roles."""
        self.provider = provider
        self.prompt_roles = sorted({p.role.lower() for p in profiles.all()})
        self.roles = sorted(role_to_profile_map(profiles))  # roles + profile names are accepted

    def build_request(self, requirement: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        system = _SYSTEM_PROMPT.format(roles=", ".join(self.prompt_roles))
        messages = [{"role": "user", "content": f"Requirement:\n{requirement.strip()}"}]
        api_messages, kwargs = build_provider_request(messages, system_prompt=system,
                                                      wire_format=resolve_wire_format(self.provider))
        # Same outgoing secret boundary as the agent loop.
        return redact_structure(api_messages), redact_structure(kwargs)

    def create_plan(self, requirement: str) -> ImplementationPlan:
        if not isinstance(requirement, str) or not requirement.strip():
            raise PlanningError("the requirement must be a non-empty string")
        api_messages, kwargs = self.build_request(requirement)
        log_event("plan.requested", provider=type(self.provider).__name__, model=getattr(self.provider, "model", None))
        try:
            response = self.provider.chat(api_messages, **kwargs)
        except Exception as e:  # provider failure, timeout, auth error, ...
            log_event("plan.failed", stage="provider", error_type=type(e).__name__, error=str(e))
            raise PlanningError(f"the model provider failed: {type(e).__name__}: {e}") from e
        finish = str(getattr(response, "finish_reason", "") or "").lower()
        try:
            data = extract_json(getattr(response, "content", "") or "")
            plan = parse_plan(data, self.roles)
        except PlanningError as e:
            if finish in ("length", "max_tokens"):
                e = PlanningError(f"{e} (the response was truncated: finish_reason={finish})", e.issues)
            log_event("plan.failed", stage="validation", error=str(e))
            raise e
        log_event("plan.created", project=plan.project_name, tasks=len(plan.tasks))
        return plan


def role_to_profile_map(profiles) -> dict[str, str]:
    """Agent roles (and profile names) from the configurable ProfileRegistry -> profile name."""
    mapping: dict[str, str] = {}
    for profile in profiles.all():
        mapping.setdefault(profile.role.lower(), profile.name)
        mapping.setdefault(profile.name.lower(), profile.name)
    return mapping


def plan_to_task_graph(plan: ImplementationPlan, orchestrator, requirement: str):
    """Create the plan as an existing Phase 2 Task Graph via Orchestrator.create_graph()."""
    specs = plan.to_task_specs(role_to_profile_map(orchestrator.profiles))
    graph = orchestrator.create_graph(requirement.strip(), specs)  # TaskStore validation is authoritative
    graph.settings["plan"] = plan.to_dict()
    orchestrator.tasks.update_graph(graph)
    return graph


__all__ = [
    "ImplementationPlan", "PlannedModule", "PlannedTask", "Planner", "PlanningError", "extract_json", "parse_plan",
    "plan_to_task_graph", "role_to_profile_map",
]
