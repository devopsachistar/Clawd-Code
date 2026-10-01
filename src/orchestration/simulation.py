"""Deterministic offline "LLM" for the multi-agent demo and tests.

``increscent team --simulate`` uses this provider instead of a real model, so
the whole pipeline can run safely and offline: planning, worktrees, agents,
tool calls through the permission engine, commits, tests, review. Each role
follows a fixed script. It issues *real* tool calls (Write, Bash, ReportResult)
that go through the same registry, policies and budgets as a real model.

This is a demonstration harness, not an AI. It only knows how to build the
small "task manager" example application.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from ..providers.base import BaseProvider, ChatResponse

PY = f'"{sys.executable}"'

_ARCHITECTURE = """# Architecture: task manager

## Modules
- `taskapp/storage.py`: JSON file persistence (`TaskStorage.load() / save()`).
- `taskapp/service.py`: business logic (`TaskService.add / list / complete / delete`), validation.
- `taskapp/cli.py`: command-line interface; talks only to `TaskService`.
- `tests/`: integration tests through the CLI and service down to storage.

## Data model
Task = {id: int, title: str (1..200 chars), done: bool}

## Interfaces
- `TaskService(storage)`: `add(title) -> dict`, `list(include_done=True) -> list[dict]`,
  `complete(task_id) -> dict`, `delete(task_id) -> None`. Errors raise `ValueError` / `KeyError`.
"""

_STORAGE = '''"""JSON file persistence for tasks."""
import json
from pathlib import Path


class TaskStorage:
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, tasks):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        tmp.replace(self.path)
'''

_SERVICE = '''"""Task business logic."""


class TaskService:
    def __init__(self, storage):
        self.storage = storage

    def add(self, title):
        title = (title or "").strip()
        if not title or len(title) > 200:
            raise ValueError("title must be 1..200 characters")
        tasks = self.storage.load()
        task = {"id": max((t["id"] for t in tasks), default=0) + 1, "title": title, "done": False}
        tasks.append(task)
        self.storage.save(tasks)
        return task

    def list(self, include_done=True):
        return [t for t in self.storage.load() if include_done or not t["done"]]

    def complete(self, task_id):
        tasks = self.storage.load()
        for task in tasks:
            if task["id"] == task_id:
                task["done"] = True
                self.storage.save(tasks)
                return task
        raise KeyError(task_id)

    def delete(self, task_id):
        tasks = self.storage.load()
        remaining = [t for t in tasks if t["id"] != task_id]
        if len(remaining) == len(tasks):
            raise KeyError(task_id)
        self.storage.save(remaining)
'''

_CLI = '''"""Command-line interface for the task manager."""
import argparse
import sys

from taskapp.service import TaskService
from taskapp.storage import TaskStorage


def main(argv=None, out=sys.stdout):
    parser = argparse.ArgumentParser(prog="taskapp")
    parser.add_argument("--db", default="tasks.json")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add").add_argument("title")
    sub.add_parser("list").add_argument("--open", action="store_true")
    sub.add_parser("done").add_argument("id", type=int)
    sub.add_parser("rm").add_argument("id", type=int)
    args = parser.parse_args(argv)
    service = TaskService(TaskStorage(args.db))
    try:
        if args.cmd == "add":
            task = service.add(args.title)
            print(f"added #{task['id']}: {task['title']}", file=out)
        elif args.cmd == "list":
            for t in service.list(include_done=not args.open):
                print(f"[{'x' if t['done'] else ' '}] #{t['id']} {t['title']}", file=out)
        elif args.cmd == "done":
            service.complete(args.id)
            print(f"completed #{args.id}", file=out)
        elif args.cmd == "rm":
            service.delete(args.id)
            print(f"deleted #{args.id}", file=out)
    except (ValueError, KeyError) as e:
        print(f"error: {e}", file=out)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_TESTS = '''import io
import tempfile
import unittest
from pathlib import Path

from taskapp.cli import main
from taskapp.service import TaskService
from taskapp.storage import TaskStorage


class TaskAppIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "tasks.json")

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        out = io.StringIO()
        code = main(["--db", self.db, *args], out=out)
        return code, out.getvalue()

    def test_add_list_complete_delete_flow(self):
        self.assertEqual(self.run_cli("add", "write report")[0], 0)
        self.run_cli("add", "review code")
        self.run_cli("done", "1")
        code, out = self.run_cli("list")
        self.assertIn("[x] #1 write report", out)
        self.assertIn("[ ] #2 review code", out)
        self.assertEqual(self.run_cli("list", "--open")[1].strip(), "[ ] #2 review code")
        self.run_cli("rm", "2")
        self.assertNotIn("review code", self.run_cli("list")[1])

    def test_validation_errors(self):
        code, out = self.run_cli("add", "   ")
        self.assertEqual(code, 1)
        self.assertIn("error", out)
        self.assertEqual(self.run_cli("done", "99")[0], 1)

    def test_persistence_survives_new_service(self):
        TaskService(TaskStorage(self.db)).add("persist me")
        self.assertEqual(TaskService(TaskStorage(self.db)).list()[0]["title"], "persist me")


if __name__ == "__main__":
    unittest.main()
'''


def _write(path: str, content: str) -> dict[str, Any]:
    return {"name": "Write", "input": {"file_path": path, "content": content}}


def _report(**fields: Any) -> dict[str, Any]:
    return {"name": "ReportResult", "input": fields}


def _script(role: str, cwd: str) -> list[Callable[[list[dict]], list[dict[str, Any]]]]:
    """Each step receives the conversation so far and returns tool calls."""
    j = lambda *parts: "/".join([cwd.replace("\\", "/"), *parts])  # noqa: E731
    if role == "architect":
        return [
            lambda m: [_write(j("docs", "ARCHITECTURE.md"), _ARCHITECTURE)],
            lambda m: [_report(status="completed", summary="Defined storage/service/cli modules and the task data model.",
                               files_changed=["docs/ARCHITECTURE.md"], next_action="Implement persistence (DatabaseEngineer).")],
        ]
    if role == "database":
        return [
            lambda m: [_write(j("taskapp", "__init__.py"), '"""Task manager package."""\n'),
                       _write(j("taskapp", "storage.py"), _STORAGE)],
            lambda m: [_report(status="completed", summary="JSON persistence layer with atomic saves.",
                               files_changed=["taskapp/__init__.py", "taskapp/storage.py"],
                               risks=["Single-process file storage; not safe for concurrent writers."],
                               next_action="Implement TaskService (BackendDeveloper).")],
        ]
    if role == "backend":
        return [
            lambda m: [_write(j("taskapp", "service.py"), _SERVICE)],
            lambda m: [_report(status="completed", summary="TaskService with add/list/complete/delete and validation.",
                               files_changed=["taskapp/service.py"], next_action="Build the CLI (FrontendDeveloper).")],
        ]
    if role == "frontend":
        return [
            lambda m: [_write(j("taskapp", "cli.py"), _CLI)],
            lambda m: [_report(status="completed", summary="argparse CLI exposing every service operation.",
                               files_changed=["taskapp/cli.py"], next_action="Integration tests (QAEngineer).")],
        ]
    if role == "qa":
        return [
            lambda m: [_write(j("tests", "__init__.py"), ""), _write(j("tests", "test_taskapp.py"), _TESTS)],
            lambda m: [{"name": "Bash", "input": {"command": f"{PY} -m unittest discover -s tests -t . -v", "timeout_s": 300}}],
            lambda m: [_report(status="completed" if _last_tests(m)[2] == 0 and _last_tests(m)[0] > 0 else "failed",
                               summary="Integration tests for CLI -> service -> storage.",
                               files_changed=["tests/__init__.py", "tests/test_taskapp.py"],
                               tests_run=_last_tests(m)[0], tests_passed=_last_tests(m)[1], tests_failed=_last_tests(m)[2],
                               next_action="Code review.")],
        ]
    if role in ("reviewer", "security"):
        return [
            lambda m: [{"name": "Bash", "input": {"command": "git log --oneline -n 10"}}],
            lambda m: [{"name": "Bash", "input": {"command": f"{PY} -m unittest discover -s tests -t ."}}],
            lambda m: [_report(status="completed", summary="Reviewed architecture, persistence, service, CLI and tests.",
                               review_verdict="approve" if _last_tests(m)[2] == 0 else "request_changes",
                               risks=["JSON file storage is single-user; add locking before multi-user use."],
                               tests_run=_last_tests(m)[0], tests_passed=_last_tests(m)[1], tests_failed=_last_tests(m)[2],
                               next_action="Human review, then approve merge into the integration branch.")],
        ]
    return [lambda m: [_report(status="completed", summary=f"No simulated work defined for role {role}.")]]


def _last_tests(messages: list[dict]) -> tuple[int, int, int]:
    from .testing import parse_test_counts

    for msg in reversed(messages):
        if msg.get("role") == "tool":
            text = msg.get("content") or ""
            try:
                data = json.loads(text)
                text = f"{data.get('stdout', '')}\n{data.get('stderr', '')}"
            except Exception:
                pass
            if re.search(r"Ran \d+ test|passed|failed", text):
                return parse_test_counts(text)
    return (0, 0, 0)


class SimulatedAgentProvider(BaseProvider):
    """Scripted provider for one agent. Speaks the OpenAI wire format."""

    message_format = "openai"

    def __init__(self, role: str, working_directory: str, model: str = "simulated"):
        super().__init__(api_key="simulated", base_url=None, model=model)
        self.role = role
        self.steps = _script(role, working_directory)
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        messages = list(messages)
        if self.calls < len(self.steps):
            uses = self.steps[self.calls](messages)
            self.calls += 1
            return ChatResponse(
                content="", model=self.model or "simulated", usage={"input_tokens": 0, "output_tokens": 0},
                finish_reason="tool_calls",
                tool_uses=[{"id": f"sim_{self.role}_{self.calls}_{i}", **u} for i, u in enumerate(uses)],
            )
        self.calls += 1
        return ChatResponse(content="Done.", model=self.model or "simulated", usage={}, finish_reason="stop")

    def chat_stream(self, messages, tools=None, **kwargs):
        yield self.chat(messages, tools=tools, **kwargs).content

    def get_available_models(self) -> list[str]:
        return ["simulated"]


def simulated_provider_factory(agent) -> SimulatedAgentProvider:
    return SimulatedAgentProvider(agent.role, agent.working_directory)


# ---------------------------------------------------------------------------
# Developer agent (Phase 3.2.1): deterministic offline developer for `increscent develop --simulate`
# ---------------------------------------------------------------------------

_CREATE_FILE_TASK = re.compile(
    r"create\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:named\s+|called\s+)?[`'\"]?(?P<path>[\w.\-/\\]+\.\w+)[`'\"]?\s+"
    r"(?:containing|with)\s+(?:the\s+)?(?:text|content|contents)?\s*"
    r"(?:[`'\"\u201c](?P<quoted>.+?)[`'\"\u201d]|(?P<bare>[^\n]+?))\s*\.?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# "Create a file named X containing the contents of Y followed by the text 'Z'."  (depends on Y existing)
_DERIVED_FILE_TASK = re.compile(
    r"create\s+(?:a\s+)?(?:file\s+)?(?:named\s+)?[`'\"]?(?P<path>[\w.\-/\\]+\.\w+)[`'\"]?\s+containing\s+the\s+contents\s+of\s+"
    r"[`'\"]?(?P<source>[\w.\-/\\]+\.\w+)[`'\"]?\s+followed\s+by\s+(?:the\s+)?(?:text\s+)?[`'\"\u201c](?P<extra>.+?)[`'\"\u201d]",
    re.IGNORECASE,
)
# "Verify that X contains the text 'Y'."  (read-only; runs the existing tests)
_VERIFY_TASK = re.compile(
    r"verify\s+that\s+[`'\"]?(?P<path>[\w.\-/\\]+\.\w+)[`'\"]?\s+contains\s+(?:the\s+)?(?:text\s+)?"
    r"[`'\"\u201c](?P<text>.+?)[`'\"\u201d]",
    re.IGNORECASE,
)


def parse_simulated_task(text: str) -> tuple[str, dict[str, str]] | None:
    """The task forms the simulated developer can execute (checked in this order)."""
    m = _DERIVED_FILE_TASK.search(text)
    if m:
        return "derive", {"path": m.group("path").replace("\\", "/"), "source": m.group("source").replace("\\", "/"),
                          "extra": m.group("extra")}
    m = _VERIFY_TASK.search(text)
    if m:
        return "verify", {"path": m.group("path").replace("\\", "/"), "text": m.group("text")}
    m = _CREATE_FILE_TASK.search(text)
    if m:
        value = m.group("quoted") if m.group("quoted") is not None else m.group("bare").strip()
        return "create", {"path": m.group("path").replace("\\", "/"), "text": value}
    return None


def _read_tool_text(message: dict | None) -> Optional[str]:
    """File text from a Read tool result (line numbers stripped), or None if the read failed."""
    if not message:
        return None
    try:
        data = json.loads(message.get("content") or "")
    except Exception:
        return None
    file = data.get("file") if isinstance(data, dict) else None
    if not isinstance(file, dict) or "content" not in file:
        return None
    return "\n".join(re.sub(r"^\d+\t", "", line) for line in str(file["content"]).splitlines())


class SimulatedDeveloperProvider(BaseProvider):
    """Scripted developer for three deterministic task forms:

    * ``Create a file named X containing 'Y'``
    * ``Create a file named X containing the contents of Y followed by the text 'Z'``. It reads Y,
      which a previous task must have produced, and fails honestly if Y is missing.
    * ``Verify that X contains the text 'Y'``. Read-only; runs the existing tests.

    It issues real tool calls (Glob, Read, Write, Bash, ReportResult) through the normal agent loop
    and reacts to their actual results. It also respects the tools its profile offers: no Bash means
    no local test run, and no Write means it reports failure instead of writing.
    ``inject_bug``/``unfixable`` apply to the first form and exercise the fix and failure paths.
    It is not an AI and cannot do any other task.
    """

    message_format = "openai"

    def __init__(self, working_directory: str, *, inject_bug: bool = False, unfixable: bool = False,
                 model: str = "simulated-developer"):
        super().__init__(api_key="simulated", base_url=None, model=model)
        self.cwd = Path(working_directory)
        self.inject_bug = inject_bug or unfixable
        self.unfixable = unfixable
        self.step = 0
        self.fixed = False
        self.verified: Optional[bool] = None

    def _call(self, name: str, tool_input: dict[str, Any]) -> ChatResponse:
        self.step += 1
        return ChatResponse(content="", model=self.model or "simulated-developer", usage={}, finish_reason="tool_calls",
                            tool_uses=[{"id": f"sim_dev_{self.step}", "name": name, "input": tool_input}])

    def _report(self, **fields: Any) -> ChatResponse:
        return self._call("ReportResult", fields)

    @staticmethod
    def _test_rel(rel: str) -> str:
        return f"tests/test_{re.sub(r'[^A-Za-z0-9]+', '_', Path(rel).stem).strip('_').lower() or 'file'}_content.py"

    @staticmethod
    def _run_cmd(test_rel: str | None = None) -> str:
        return f'"{sys.executable}" -m pytest -q -p no:cacheprovider' + (f" {test_rel}" if test_rel else "")

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        messages = list(messages)
        offered = {t.get("name") for t in (tools or []) if isinstance(t, dict)}
        self.can_run = not offered or "Bash" in offered
        self.can_write = not offered or "Write" in offered
        assignment = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
        task_text = assignment.split("## Task", 1)[-1].split("##", 1)[0]
        last_tool = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
        if last_tool and last_tool.get("tool_call_id", "").startswith("sim_dev_") \
                and '"recorded": true' in (last_tool.get("content") or ""):
            return ChatResponse(content="Done.", model=self.model or "", usage={}, finish_reason="stop")
        parsed = parse_simulated_task(task_text)
        if parsed is None:
            if self.step:
                return ChatResponse(content="Done.", model=self.model or "", usage={}, finish_reason="stop")
            return self._report(status="failed", summary="The simulated developer only supports: \"Create a file "
                                "named X containing 'Y'\", \"... containing the contents of Y followed by the text 'Z'\" "
                                "and \"Verify that X contains the text 'Y'\".", next_action="Use a real model provider.")
        kind, p = parsed
        if kind != "verify" and not self.can_write:
            return self._report(status="failed", summary=f"This profile cannot write files (task needs {p['path']}).")
        return getattr(self, f"_{kind}")(p, messages, last_tool)

    # -- "create X containing 'Y'" -------------------------------------------------------
    def _create(self, p, messages, last_tool) -> ChatResponse:
        rel, text = p["path"], p["text"]
        target, test_rel = self.cwd / rel, self._test_rel(rel)
        if self.step == 0:  # inspect the workspace
            return self._call("Glob", {"pattern": "**/*", "path": str(self.cwd)})
        if self.step == 1:  # implement (optionally with an injected bug)
            return self._call("Write", {"file_path": str(target),
                                        "content": (text.upper() if self.inject_bug else text) + "\n"})
        if self.step == 2:  # a test for the acceptance criterion
            return self._call("Write", {"file_path": str(self.cwd / test_rel), "content": (
                "from pathlib import Path\n\n\n"
                f"def test_{Path(test_rel).stem[5:]}():\n"
                "    root = Path(__file__).resolve().parents[1]\n"
                f"    assert (root / {rel!r}).read_text(encoding=\"utf-8\").strip() == {text!r}\n")})
        if not self.can_run:
            return self._finish(rel, test_rel, ran=False)
        if self.step == 3:
            return self._call("Bash", {"command": self._run_cmd(test_rel), "timeout_s": 300})
        counts = _last_tests(messages)
        if counts[2] > 0 and not self.fixed and not self.unfixable:  # inspect the failure, fix it
            self.fixed = True
            return self._call("Write", {"file_path": str(target), "content": text + "\n"})
        if self.fixed and self.step == 5:  # re-run after the fix
            return self._call("Bash", {"command": self._run_cmd(test_rel), "timeout_s": 300})
        return self._finish(rel, test_rel, ran=True, messages=messages)

    # -- "create X containing the contents of Y followed by 'Z'" ---------------------------------
    def _derive(self, p, messages, last_tool) -> ChatResponse:
        rel, source, extra = p["path"], p["source"], p["extra"]
        test_rel = self._test_rel(rel)
        if self.step == 0:  # inspect the previous task's output
            return self._call("Read", {"file_path": str(self.cwd / source)})
        if self.step == 1:
            content = _read_tool_text(last_tool)
            if content is None:
                return self._report(status="failed", summary=f"{source} does not exist: the dependency's output is "
                                    "missing, so the dependent change cannot be made.",
                                    next_action="Run the task that produces it first.")
            return self._call("Write", {"file_path": str(self.cwd / rel),
                                        "content": content.rstrip("\n") + "\n" + extra + "\n"})
        if self.step == 2:
            return self._call("Write", {"file_path": str(self.cwd / test_rel), "content": (
                "from pathlib import Path\n\n\n"
                f"def test_{Path(test_rel).stem[5:]}():\n"
                "    root = Path(__file__).resolve().parents[1]\n"
                f"    expected = (root / {source!r}).read_text(encoding=\"utf-8\").rstrip(\"\\n\") + \"\\n\" + {extra!r}\n"
                f"    assert (root / {rel!r}).read_text(encoding=\"utf-8\").strip() == expected.strip()\n")})
        if not self.can_run:
            return self._finish(rel, test_rel, ran=False)
        if self.step == 3:
            return self._call("Bash", {"command": self._run_cmd(test_rel), "timeout_s": 300})
        return self._finish(rel, test_rel, ran=True, messages=messages)

    # -- "verify that X contains 'Y'" (read-only) ---------------------------------------------
    def _verify(self, p, messages, last_tool) -> ChatResponse:
        rel, text = p["path"], p["text"]
        if self.step == 0:
            return self._call("Read", {"file_path": str(self.cwd / rel)})
        if self.step == 1:
            content = _read_tool_text(last_tool)
            self.verified = content is not None and text in content
            if self.can_run:
                return self._call("Bash", {"command": self._run_cmd(), "timeout_s": 300})
        counts = _last_tests(messages) if self.can_run else (0, 0, 0)
        ok = bool(self.verified) and counts[2] == 0
        return self._report(
            status="completed" if ok else "failed",
            summary=(f"Verified that {rel} contains the expected text; existing tests pass." if ok
                     else f"Verification of {rel} failed (text present: {self.verified}, failing tests: {counts[2]})."),
            tests_run=counts[0], tests_passed=counts[1], tests_failed=counts[2],
            review_verdict="approve" if ok else "request_changes",
            next_action="Human review." if ok else "Fix the upstream task.",
        )

    def _finish(self, rel: str, test_rel: str, *, ran: bool, messages=None) -> ChatResponse:
        counts = _last_tests(messages) if ran else (0, 0, 0)
        ok = not ran or (counts[2] == 0 and counts[0] > 0)
        return self._report(
            status="completed" if ok else "failed",
            summary=(f"Created {rel} and a test that verifies it." if ok else f"Could not make the test for {rel} pass."),
            files_changed=[rel, test_rel], tests_run=counts[0], tests_passed=counts[1], tests_failed=counts[2],
            next_action="Review the change." if ok else "Investigate the failing test.",
        )

    def chat_stream(self, messages, tools=None, **kwargs):
        yield self.chat(messages, tools=tools, **kwargs).content

    def get_available_models(self) -> list[str]:
        return ["simulated-developer"]


def simulated_developer_factory(agent) -> SimulatedDeveloperProvider:
    return SimulatedDeveloperProvider(agent.working_directory)


# ---------------------------------------------------------------------------
# Planning (Phase 3.1): a deterministic plan for offline use of `increscent plan --simulate`
# ---------------------------------------------------------------------------

SIMULATED_PLAN: dict[str, Any] = {
    "project": {
        "name": "Task Manager",
        "summary": "A small web application for managing users, projects and prioritised tasks with a dashboard.",
    },
    "requirements": [
        "Users can register and sign in",
        "Users can create projects and add tasks to them",
        "Tasks have a title, status and priority (low/medium/high)",
        "A dashboard shows open tasks per project and by priority",
    ],
    "assumptions": [
        "Single organisation; no multi-tenancy",
        "SQLite is sufficient for the first version",
    ],
    "architecture": {
        "frontend": "Server-rendered HTML templates",
        "backend": "Python web API with a service layer",
        "database": "SQLite via a repository layer",
        "infrastructure": "Single container; no deployment in this phase",
    },
    "modules": [
        {"name": "auth", "description": "User registration and sessions"},
        {"name": "projects", "description": "Project CRUD"},
        {"name": "tasks", "description": "Task CRUD, status and priority"},
        {"name": "dashboard", "description": "Aggregated views of open work"},
    ],
    "tasks": [
        {"id": "architecture", "title": "Design the architecture", "role": "architect", "dependencies": [],
         "description": "Create a file named docs/ARCHITECTURE.md containing the text 'Task Manager: users, projects, tasks with priorities, and a dashboard'.",
         "acceptance_criteria": ["docs/ARCHITECTURE.md covers modules, data model and API"]},
        {"id": "database", "title": "Implement the data model", "role": "database", "dependencies": ["architecture"],
         "description": "Create a file named app/schema.txt containing the contents of docs/ARCHITECTURE.md followed by the text 'Tables: users, projects, tasks (title, status, priority)'.",
         "acceptance_criteria": ["Schema covers users, projects, tasks and priority"]},
        {"id": "backend", "title": "Implement the backend API", "role": "backend", "dependencies": ["database"],
         "description": "Create a file named app/api.txt containing the contents of app/schema.txt followed by the text 'Endpoints: /users /projects /tasks'.",
         "acceptance_criteria": ["CRUD endpoints for projects and tasks", "Invalid input is rejected"]},
        {"id": "frontend", "title": "Implement the frontend and dashboard", "role": "frontend",
         "dependencies": ["backend"], "description": "Create a file named app/dashboard.txt containing the contents of app/api.txt followed by the text 'Pages: projects, tasks, dashboard'.",
         "acceptance_criteria": ["Dashboard shows open tasks per project and priority"]},
        {"id": "tests", "title": "Write integration tests", "role": "qa", "dependencies": ["frontend"],
         "description": "Create a file named app/test-report.txt containing the contents of app/dashboard.txt followed by the text 'Verified: main flows covered'.",
         "acceptance_criteria": ["Main flows are covered and pass"]},
        {"id": "review", "title": "Review the changes", "role": "reviewer", "dependencies": ["tests"],
         "description": "Verify that app/test-report.txt contains the text 'Verified: main flows covered'.",
         "acceptance_criteria": ["Approve or request changes with findings"]},
    ],
}


class SimulatedPlannerProvider(BaseProvider):
    """Deterministic offline planning model: always answers with :data:`SIMULATED_PLAN` as JSON.

    It records the requests it receives so tests can check the prompt.
    """

    message_format = "openai"

    def __init__(self, plan: dict[str, Any] | None = None, model: str = "simulated-planner"):
        super().__init__(api_key="simulated", base_url=None, model=model)
        self.plan = plan if plan is not None else SIMULATED_PLAN
        self.requests: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        self.requests.append(list(messages))
        return ChatResponse(content=json.dumps(self.plan, indent=2), model=self.model or "simulated-planner",
                            usage={"input_tokens": 0, "output_tokens": 0}, finish_reason="stop")

    def chat_stream(self, messages, tools=None, **kwargs):
        yield self.chat(messages, tools=tools, **kwargs).content

    def get_available_models(self) -> list[str]:
        return ["simulated-planner"]
