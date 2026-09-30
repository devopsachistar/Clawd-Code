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
from typing import Any, Callable

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
