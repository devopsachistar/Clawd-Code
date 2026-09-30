"""Configurable agent profiles.

Profiles are Markdown files with frontmatter (the same format as skills). The
body is the profile's system instructions. They are *data*: the orchestration
engine only reads the fields below and never hard-codes roles.

Search order, later overriding earlier by name (case-insensitive):

1. built-in   ``src/agents/profiles/*.md``
2. user       ``~/.clawd/agents/*.md``
3. project    ``<project>/.clawd/agents/*.md``

Frontmatter fields::

    name, role, description
    responsibilities: [..]
    allowed_tools: [Read, Grep, ...]
    allow: [READ, WRITE, ...]   ask: [...]   deny: [...]   (capabilities; unlisted = DENY)
    max_iterations, max_tool_calls, max_seconds, max_tokens
    needs_worktree: true|false   # gets its own git worktree/branch
    commits: true|false          # orchestrator commits its changes on its branch

Safety clamps, which apply whatever a file says:

* no profile can pre-approve (ALLOW) GIT_PUSH, DATABASE_WRITE, DEPLOY_STAGING,
  DEPLOY_PRODUCTION or SECRET_USE. Those become ASK.
* project-level profiles (shipped inside a repository) may not add tools
  beyond the built-in tool set, and get the same clamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..branding import PROJECT_DIR_NAME
from ..security.permissions import NEVER_AUTO_APPROVE, Capability, Decision, PermissionPolicy, parse_capabilities
from ..skills.frontmatter import parse_frontmatter
from .models import AgentBudget

BUILTIN_PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
AGENT_ONLY_TOOLS = ("ReportResult", "MessageOrchestrator")


@dataclass(frozen=True)
class AgentProfile:
    name: str
    role: str
    description: str
    responsibilities: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    allow: frozenset[Capability]
    ask: frozenset[Capability]
    deny: frozenset[Capability]
    instructions: str
    budget: AgentBudget = field(default_factory=AgentBudget)
    needs_worktree: bool = True
    commits: bool = False
    source: str = "builtin"
    warnings: tuple[str, ...] = ()

    def policy(self) -> PermissionPolicy:
        return PermissionPolicy.from_lists(
            f"profile:{self.name}", allow=[c.value for c in self.allow], ask=[c.value for c in self.ask],
            deny=[c.value for c in self.deny],
        )

    def system_prompt(self) -> str:
        lines = [f"# Role: {self.name}", "", self.description.strip(), ""]
        if self.responsibilities:
            lines.append("## Responsibilities")
            lines.extend(f"- {r}" for r in self.responsibilities)
            lines.append("")
        lines.append(self.instructions.strip())
        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "responsibilities": list(self.responsibilities),
            "allowed_tools": list(self.allowed_tools),
            "allow": sorted(c.value for c in self.allow),
            "ask": sorted(c.value for c in self.ask),
            "deny": sorted(c.value for c in self.deny),
            "budget": self.budget.to_dict(),
            "needs_worktree": self.needs_worktree,
            "commits": self.commits,
            "source": self.source,
        }


def _as_text(value: Any) -> str:
    """Free-text field (the shared frontmatter parser splits values on commas)."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return "" if value is None else str(value)


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def parse_profile(text: str, *, source: str = "builtin", known_tools: Iterable[str] | None = None,
                  trusted: bool = True) -> AgentProfile:
    parsed = parse_frontmatter(text)
    fm = parsed.frontmatter
    name = _as_text(fm.get("name")).strip()
    if not name:
        raise ValueError(f"profile from {source} has no name")
    warnings: list[str] = []

    allow = parse_capabilities(_as_list(fm.get("allow")))
    ask = parse_capabilities(_as_list(fm.get("ask")))
    deny = parse_capabilities(_as_list(fm.get("deny")))
    clamped = allow & NEVER_AUTO_APPROVE
    if clamped:
        allow -= clamped
        ask |= clamped
        warnings.append(f"{name}: {', '.join(sorted(c.value for c in clamped))} cannot be pre-approved; changed to ASK")
    ask -= deny
    allow -= ask | deny

    tools = _as_list(fm.get("allowed_tools"))
    if known_tools is not None:
        known = {t.lower() for t in known_tools}
        unknown = [t for t in tools if t.lower() not in known]
        if unknown:
            if trusted:
                raise ValueError(f"profile {name}: unknown tools {unknown}")
            tools = [t for t in tools if t.lower() in known]
            warnings.append(f"{name}: ignored unknown tools {unknown}")

    defaults = AgentBudget()
    budget = AgentBudget(
        max_iterations=int(fm.get("max_iterations", defaults.max_iterations)),
        max_tool_calls=int(fm.get("max_tool_calls", defaults.max_tool_calls)),
        max_seconds=int(fm.get("max_seconds", defaults.max_seconds)),
        max_tokens=int(fm["max_tokens"]) if fm.get("max_tokens") not in (None, "", "none") else None,
    )
    return AgentProfile(
        name=name,
        role=_as_text(fm.get("role")) or name.lower(),
        description=_as_text(fm.get("description")),
        responsibilities=tuple(_as_list(fm.get("responsibilities"))),
        allowed_tools=tuple(tools),
        allow=frozenset(allow),
        ask=frozenset(ask),
        deny=frozenset(deny),
        instructions=parsed.body,
        budget=budget,
        needs_worktree=bool(fm.get("needs_worktree", True)),
        commits=bool(fm.get("commits", False)),
        source=source,
        warnings=tuple(warnings),
    )


class ProfileRegistry:
    def __init__(self, profiles: Iterable[AgentProfile] = ()) -> None:
        self._profiles: dict[str, AgentProfile] = {}
        self.warnings: list[str] = []
        for p in profiles:
            self.add(p)

    def add(self, profile: AgentProfile) -> None:
        self._profiles[profile.name.lower()] = profile
        self.warnings.extend(profile.warnings)

    def get(self, name: str) -> AgentProfile:
        try:
            return self._profiles[name.lower()]
        except KeyError:
            raise KeyError(f"unknown agent profile: {name} (available: {', '.join(self.names())})") from None

    def names(self) -> list[str]:
        return sorted(p.name for p in self._profiles.values())

    def all(self) -> list[AgentProfile]:
        return [self._profiles[k] for k in sorted(self._profiles)]

    @classmethod
    def load(cls, project_root: str | Path | None = None, *, user_dir: Path | None = None,
             known_tools: Iterable[str] | None = None) -> "ProfileRegistry":
        if known_tools is None:
            from ..tool_system.defaults import build_default_registry

            known_tools = [s.name for s in build_default_registry(include_user_tools=False).list_specs()]
        known = list(known_tools)
        registry = cls()
        sources: list[tuple[Path, str, bool]] = [(BUILTIN_PROFILE_DIR, "builtin", True)]
        if user_dir is None:
            from ..config import get_user_dir

            user_dir = get_user_dir() / "agents"
        sources.append((user_dir, "user", True))
        if project_root is not None:
            sources.append((Path(project_root) / PROJECT_DIR_NAME / "agents", "project", False))
        for directory, label, trusted in sources:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                try:
                    profile = parse_profile(path.read_text(encoding="utf-8"), source=f"{label}:{path}",
                                            known_tools=known, trusted=trusted)
                except ValueError as e:
                    if label == "builtin":
                        raise
                    registry.warnings.append(f"skipped {path}: {e}")
                    continue
                registry.add(profile)
        return registry


__all__ = ["AGENT_ONLY_TOOLS", "AgentProfile", "BUILTIN_PROFILE_DIR", "Decision", "ProfileRegistry", "parse_profile"]
