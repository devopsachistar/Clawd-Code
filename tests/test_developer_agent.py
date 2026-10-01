"""Phase 3.2.1: one real developer-agent execution path (Task -> Agent -> Tools -> Repository)."""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from fakes import FakeOpenAIProvider, reply, tool_call
from gitrepo import TempRepo, git
from src.agents.developer import DeveloperAgent
from src.cli import main
from src.orchestration.planner import TaskSpec
from src.orchestration.simulation import SimulatedDeveloperProvider

HELLO = "Hello from Increscent Factory"
HELLO_TASK = TaskSpec(
    key="hello", title="Create hello.txt", profile="Developer",
    description=f"Create a file named hello.txt containing the text '{HELLO}'.",
    acceptance=("hello.txt contains exactly the requested text",),
)
PY = f'"{sys.executable}"'


class _Workspace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        base = Path(self._tmp.name).resolve()
        self.ws = base / "workspace"
        self.ws.mkdir()
        (self.ws / "README.md").write_text("# Sample project\nGreeting lives in hello.txt\n", encoding="utf-8")
        self.outside = base / "outside"
        self.outside.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def simulated(self, **kw) -> DeveloperAgent:
        return DeveloperAgent(lambda agent: SimulatedDeveloperProvider(agent.working_directory, **kw))

    def scripted(self, steps) -> tuple[DeveloperAgent, list]:
        """A DeveloperAgent whose model is a scripted fake; returns (agent, provider holder)."""
        holder: list[FakeOpenAIProvider] = []

        def factory(agent):
            provider = FakeOpenAIProvider(steps(Path(agent.working_directory)))
            holder.append(provider)
            return provider

        return DeveloperAgent(factory), holder


class DeveloperAgentTests(_Workspace):
    def test_receives_task_criteria_and_workspace(self):
        agent, holder = self.scripted(lambda ws: [reply("", [tool_call("r", "ReportResult", {
            "status": "failed", "summary": "just looking"})]), reply("ok")])
        agent.execute(HELLO_TASK, self.ws)
        first = holder[0].requests[0]["messages"]
        assignment = first[1]["content"]
        self.assertIn(HELLO_TASK.description, assignment)
        self.assertIn("hello.txt contains exactly the requested text", assignment)
        self.assertIn(f"Workspace (your working directory; stay inside it): {self.ws}", assignment)
        self.assertIn("# Role: Developer", first[0]["content"])  # existing Developer profile
        offered = {t["name"] for t in holder[0].requests[0]["tools"]}
        self.assertTrue({"Read", "Glob", "Grep", "Write", "Edit", "Bash", "ReportResult"} <= offered)

    def test_inspects_files_with_existing_tools(self):
        agent, holder = self.scripted(lambda ws: [
            reply("", [tool_call("g", "Glob", {"pattern": "*", "path": str(ws)}),
                       tool_call("r", "Read", {"file_path": str(ws / "README.md")})]),
            reply("", [tool_call("rep", "ReportResult", {"status": "failed", "summary": "inspected"})]),
            reply("ok"),
        ])
        agent.execute(HELLO_TASK, self.ws)
        tool_results = [m["content"] for m in holder[0].requests[1]["messages"] if m["role"] == "tool"]
        self.assertTrue(any("README.md" in r for r in tool_results))
        self.assertTrue(any("Greeting lives in hello.txt" in r for r in tool_results))

    def test_modifies_repository_and_completes(self):
        result = self.simulated().execute(HELLO_TASK, self.ws)
        self.assertEqual((self.ws / "hello.txt").read_text(encoding="utf-8").strip(), HELLO)
        self.assertEqual(result.status, "completed", result)
        self.assertEqual(result.task_id, "hello")
        self.assertEqual(result.files_changed, ["hello.txt", "tests/test_hello_content.py"])
        self.assertEqual((result.tests_run, result.tests_passed, result.tests_failed), (1, 1, 0))
        self.assertIsNone(result.error)
        self.assertIn("pytest", result.test_command)

    def test_runs_commands_and_fixes_failures(self):
        result = self.simulated(inject_bug=True).execute(HELLO_TASK, self.ws)
        self.assertEqual(result.status, "completed", result)
        self.assertEqual((self.ws / "hello.txt").read_text(encoding="utf-8").strip(), HELLO)

    def test_failed_tests_return_failed_with_real_failure(self):
        result = self.simulated(unfixable=True).execute(HELLO_TASK, self.ws)
        self.assertEqual(result.status, "failed")
        self.assertEqual((result.tests_run, result.tests_failed), (1, 1))
        self.assertIn("AssertionError", result.test_output)
        self.assertIn("hello.txt", result.files_changed)

    def test_model_claiming_success_is_not_trusted(self):
        # The agent writes the wrong content and a test, then *claims* success.
        def steps(ws):
            return [
                reply("", [tool_call("w1", "Write", {"file_path": str(ws / "hello.txt"), "content": "wrong\n"}),
                           tool_call("w2", "Write", {"file_path": str(ws / "tests" / "test_hello.py"), "content":
                               "from pathlib import Path\n\ndef test_hello():\n    assert (Path(__file__).resolve()"
                               f".parents[1] / 'hello.txt').read_text().strip() == {HELLO!r}\n"})]),
                reply("", [tool_call("r", "ReportResult", {"status": "completed", "summary": "All done, tests pass!",
                                                           "tests_run": 1, "tests_passed": 1, "tests_failed": 0})]),
                reply("ok"),
            ]

        agent, _ = self.scripted(steps)
        result = agent.execute(HELLO_TASK, self.ws)
        self.assertEqual(result.agent_reported_status, "completed")
        self.assertEqual(result.status, "failed")
        self.assertEqual((result.tests_run, result.tests_failed), (1, 1))
        self.assertIn("1 test(s) failed", result.error)

    def test_no_report_or_no_change_is_failed(self):
        agent, _ = self.scripted(lambda ws: [reply("I finished the task.")])
        result = agent.execute(HELLO_TASK, self.ws)
        self.assertEqual(result.status, "failed")
        self.assertIn("ReportResult", result.error)
        agent, _ = self.scripted(lambda ws: [reply("", [tool_call("r", "ReportResult", {
            "status": "completed", "summary": "nothing to do"})]), reply("ok")])
        result = agent.execute(HELLO_TASK, self.ws)
        self.assertEqual(result.status, "failed")
        self.assertIn("no changes", result.error)

    def test_cannot_operate_outside_workspace(self):
        target = self.outside / "escape.txt"
        (self.outside / "secret.txt").write_text("outside data", encoding="utf-8")

        def steps(ws):
            return [
                reply("", [tool_call("w", "Write", {"file_path": str(target), "content": "x"}),
                           tool_call("w2", "Write", {"file_path": str(ws / ".." / "outside" / "escape2.txt"), "content": "x"}),
                           tool_call("r", "Read", {"file_path": str(self.outside / "secret.txt")}),
                           tool_call("b", "Bash", {"command": "echo hi", "cwd": str(self.outside)})]),
                reply("", [tool_call("rep", "ReportResult", {"status": "failed", "summary": "blocked"})]),
                reply("ok"),
            ]

        agent, holder = self.scripted(steps)
        result = agent.execute(HELLO_TASK, self.ws)
        self.assertFalse(target.exists())
        self.assertFalse((self.outside / "escape2.txt").exists())
        tool_results = [m["content"] for m in holder[0].requests[1]["messages"] if m["role"] == "tool"]
        self.assertEqual(len(tool_results), 4)
        self.assertTrue(all("outside allowed working directories" in r for r in tool_results), tool_results)
        self.assertNotIn("outside data", " ".join(tool_results))
        self.assertEqual(result.files_changed, [])
        self.assertEqual(result.status, "failed")

    def test_missing_workspace(self):
        result = self.simulated().execute(HELLO_TASK, self.ws / "nope")
        self.assertEqual(result.status, "failed")
        self.assertIn("workspace does not exist", result.error)

    def test_unsupported_simulated_task_fails_honestly(self):
        task = TaskSpec(key="x", title="Refactor", description="Refactor the payment module", profile="Developer")
        result = self.simulated().execute(task, self.ws)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.files_changed, [])


class DevelopCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = TempRepo()

    def tearDown(self) -> None:
        self.repo.cleanup()

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_develop_runs_in_isolated_worktree(self):
        code, out, err = self.cli("develop", "--simulate", "--cwd", str(self.repo.path), HELLO_TASK.description)
        self.assertEqual(code, 0, out + err)
        for fragment in ("DEVELOPER AGENT", "Files changed:", "  hello.txt", "Tests:", "passed", "RESULT:", "COMPLETED"):
            self.assertIn(fragment, out)
        workspace = Path(out.split("Workspace:\n  ", 1)[1].split("  (branch", 1)[0])
        self.assertEqual((workspace / "hello.txt").read_text(encoding="utf-8").strip(), HELLO)
        self.assertFalse((self.repo.path / "hello.txt").exists())          # main checkout untouched
        self.assertEqual(self.repo.head(), self.repo.initial)              # nothing committed on main
        branch = out.split("(branch ", 1)[1].split(";", 1)[0]
        self.assertEqual(git(self.repo.path, "rev-parse", branch), self.repo.initial)  # not committed at all
        shutil.rmtree(workspace, ignore_errors=True)

    def test_develop_json_with_explicit_workspace(self):
        ws = self.repo.base / "ws"
        ws.mkdir()
        code, out, _ = self.cli("develop", "--simulate", "--json", "--workspace", str(ws), "--cwd", str(self.repo.path),
                                HELLO_TASK.description)
        data = json.loads(out)
        self.assertEqual((code, data["status"], data["files_changed"][0]), (0, "completed", "hello.txt"))

    def test_develop_errors(self):
        self.assertEqual(self.cli("develop", "--simulate", "--cwd", str(self.repo.path))[0], 2)
        code, _, err = self.cli("develop", "--provider", "openai", "--cwd", str(self.repo.path), "do it")
        self.assertEqual(code, 2)
        self.assertIn("OPENAI_API_KEY", err)
        code, out, _ = self.cli("develop", "--simulate", "--workspace", str(self.repo.base / "ws2"),
                                "--cwd", str(self.repo.path), "Refactor everything")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
