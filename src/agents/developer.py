"""Developer agent: execute ONE TaskSpec in ONE workspace and report verified results.

    TaskSpec (description + acceptance criteria)
        ↓  assignment prompt
    AgentRunner (existing): Developer profile tools/policy, ReportResult, budgets
        ↓  Read/Glob/Grep → Write/Edit → Bash (tests) → fix → ReportResult
    DeveloperAgent verification (not the model's word):
        files_changed  = filesystem snapshot diff of the workspace
        tests          = orchestration.testing.run_tests() in the workspace
        ↓
    DeveloperResult(status=completed|failed, ...)

``status`` is ``completed`` only when **all** of these hold:

* the agent reported ``completed`` through ReportResult
* the measured tests passed, or no tests were detected
* the workspace actually changed, or tests ran and passed

Otherwise the result is ``failed`` and carries the real failure information.
Nothing is committed or pushed.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..orchestration.planner import TaskSpec
from ..orchestration.state import EventRecorder
from ..orchestration.testing import detect_test_command, run_tests
from .profiles import ProfileRegistry
from .runner import AgentRunner, ApprovalHandler, ProviderFactory

# Heading that starts the repair context appended to a TaskSpec by the orchestrator's repair loop.
REPAIR_MARKER = "## REPAIR ATTEMPT"

_SNAPSHOT_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".venv",
                       "venv", ".tox"}


@dataclass
class DeveloperResult:
    status: str                      # "completed" | "failed"
    task_id: str
    summary: str
    files_changed: list[str] = field(default_factory=list)
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    error: Optional[str] = None
    workspace: str = ""
    test_command: Optional[str] = None
    test_output: str = ""            # tail of the measured test output (secrets redacted)
    agent_id: Optional[str] = None
    agent_reported_status: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def snapshot(workspace: Path) -> dict[str, str]:
    """Relative path -> content hash for every file in the workspace (tooling caches excluded)."""
    files: dict[str, str] = {}
    for root, dirs, names in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _SNAPSHOT_SKIP_DIRS]
        for name in names:
            path = Path(root) / name
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
            files[path.relative_to(workspace).as_posix()] = digest
    return files


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


def build_assignment(task: TaskSpec, workspace: Path, test_command: Optional[str]) -> str:
    lines = [
        f"# Assignment: {task.title}",
        f"Task id: {task.key}",
        f"Workspace (your working directory; stay inside it): {workspace}",
        "",
        "## Task",
        task.description.strip() or task.title,
    ]
    if task.acceptance:
        lines += ["", "## Acceptance criteria (the task is complete only when ALL are met)"]
        lines += [f"- {c}" for c in task.acceptance]
    lines += [
        "",
        "## How to work",
        "1. Inspect the relevant files first (Glob, Grep, Read).",
        "2. Make the changes with Write/Edit (absolute paths inside the workspace).",
        "3. Add or update tests that check the acceptance criteria.",
        f"4. Run the tests: {test_command}" if test_command else
        "4. Run the project's tests (e.g. `python -m pytest -q`) with Bash.",
        "5. If tests fail, read the failure, fix the cause and run them again.",
        "6. Finish with ReportResult: status, summary, files_changed, tests_run/passed/failed. "
        "Report 'failed' honestly if you could not meet the criteria.",
        "Do not commit, push or deploy.",
    ]
    return "\n".join(lines)


class DeveloperAgent:
    """Runs the configurable ``Developer`` profile on one task via the existing AgentRunner."""

    def __init__(
        self,
        provider_factory: ProviderFactory,
        *,
        profiles: Optional[ProfileRegistry] = None,
        profile: str = "Developer",
        provider: str = "configured",
        model: str = "configured",
        events: Optional[EventRecorder] = None,
        store=None,
        sessions_dir: Optional[Path] = None,
        approval_handler: Optional[ApprovalHandler] = None,
        test_command: Optional[str] = None,
        test_timeout_s: int = 600,
        runner: Optional[AgentRunner] = None,
    ) -> None:
        self.profiles = profiles or ProfileRegistry.load()
        self.profile = self.profiles.get(profile)
        self.provider_name, self.model = provider, model
        self.events = events or (runner.events if runner is not None else EventRecorder(None))
        self.test_command = test_command
        self.test_timeout_s = test_timeout_s
        # Reuse an existing AgentRunner (e.g. the Orchestrator's: same store, bus, approvals) when given.
        self.runner = runner or AgentRunner(provider_factory=provider_factory, events=self.events, store=store,
                                            approval_handler=approval_handler, sessions_dir=sessions_dir)

    def execute(self, task: TaskSpec, workspace: str | Path,
                should_cancel: Optional[Callable[[], bool]] = None, *, profile: Optional[str] = None) -> DeveloperResult:
        """Run ``task`` in ``workspace``. ``profile`` overrides the agent profile (default: Developer)."""
        agent_profile = self.profiles.get(profile) if profile else self.profile
        root = Path(workspace).resolve()
        if not root.is_dir():
            return DeveloperResult(status="failed", task_id=task.key, summary="", workspace=str(root),
                                   error=f"workspace does not exist: {root}")
        hinted_command = self.test_command or detect_test_command(root)
        before = snapshot(root)
        agent = self.runner.create_agent(agent_profile, task_id=task.key, working_directory=root,
                                         provider=self.provider_name, model=self.model)
        self.events.emit("developer.started", task_id=task.key, agent_id=agent.id, workspace=str(root))
        agent_result = self.runner.run(agent, build_assignment(task, root, hinted_command), should_cancel=should_cancel)

        files = changed_files(before, snapshot(root))
        result = DeveloperResult(
            status="failed", task_id=task.key, summary=agent_result.summary, files_changed=files,
            workspace=str(root), agent_id=agent.id, agent_reported_status=agent_result.status,
            warnings=list(agent_result.warnings),
        )
        claimed_missing = sorted(set(agent_result.files_changed) - set(files))
        if claimed_missing:
            result.warnings.append(f"agent reported changes that do not exist: {claimed_missing}")

        # Measure the tests ourselves (agent-reported counts are never trusted).
        run = run_tests(root, self.test_command, timeout_s=self.test_timeout_s)
        if run is not None:
            result.test_command, result.test_output = run.command, run.output_tail
            result.tests_run, result.tests_passed, result.tests_failed = run.run, run.passed, run.failed

        if agent_result.status != "completed":
            result.error = agent_result.error or f"agent reported status '{agent_result.status}'"
        elif run is not None and not run.ok:
            result.error = (f"{run.failed} test(s) failed (exit {run.exit_code})" if not run.timed_out
                            else "tests timed out")
        elif not files and not (run is not None and run.ok and run.run > 0):
            result.error = "no changes were made to the workspace and no tests verify the task"
        else:
            result.status = "completed"

        self.events.emit("developer.completed" if result.status == "completed" else "developer.failed",
                         task_id=task.key, agent_id=agent.id, files_changed=files, tests_run=result.tests_run,
                         tests_failed=result.tests_failed, error=result.error)
        return result
