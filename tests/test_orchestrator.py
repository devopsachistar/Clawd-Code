"""Orchestrator: graph execution, results, retries, approvals, pause/resume, parallelism, merge."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Callable

from fakes import FakeOpenAIProvider, reply, tool_call
from gitrepo import TempRepo, git
from src.agents.messages import MessageType
from src.agents.models import AgentStatus
from src.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from src.orchestration.planner import TaskSpec, standard_plan
from src.orchestration.simulation import simulated_provider_factory
from src.taskgraph.models import TaskStatus

REQUIRED_RESULT_FIELDS = {"task_id", "agent_id", "status", "summary", "files_changed", "tests_run", "tests_passed",
                          "tests_failed", "risks", "warnings", "commit", "next_action"}


def write_and_report(filename: str, content: str = "x = 1\n") -> Callable:
    def script(agent) -> list:
        return [
            reply("", [tool_call("w", "Write", {"file_path": str(Path(agent.working_directory) / filename), "content": content})]),
            reply("", [tool_call("r", "ReportResult", {"status": "completed", "summary": f"wrote {filename}",
                                                       "files_changed": [filename]})]),
            reply("done"),
        ]
    return script


def silent_failure(agent) -> list:
    return [reply("I think it is fine.")]  # never calls ReportResult


class ScriptedTeam:
    """provider_factory returning per-task scripts; records every assignment prompt."""

    def __init__(self, scripts: dict[str, list[Callable]]):
        self.scripts = scripts  # task key (title) -> script per attempt
        self.assignments: dict[str, list[str]] = {}
        self.orch: Orchestrator | None = None

    def __call__(self, agent):
        task = self.orch.tasks.require(agent.task_id)
        attempts = self.scripts[task.title]
        script = attempts[min(task.attempts, len(attempts)) - 1]

        class Recording(FakeOpenAIProvider):
            def chat(inner, messages, tools=None, **kw):
                if not inner.requests:
                    self.assignments.setdefault(task.title, []).append(messages[1]["content"])
                return super().chat(messages, tools, **kw)

        return Recording(script(agent))


class OrchestratorCase(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = TempRepo()
        self.approvals = []

    def tearDown(self) -> None:
        self.repo.cleanup()

    def orchestrator(self, factory, *, approve=None, **config) -> Orchestrator:
        def handler(request):
            self.approvals.append(request)
            return bool(approve)

        orch = Orchestrator(self.repo.path, provider_factory=factory, approval_handler=handler,
                            config=OrchestratorConfig(**{"provider": "fake", "model": "fake", **config}),
                            worktree_base=self.repo.worktrees)
        if isinstance(factory, ScriptedTeam):
            factory.orch = orch
        return orch


class FullGraphTests(OrchestratorCase):
    def test_standard_plan_runs_to_review_without_touching_main(self):
        orch = self.orchestrator(simulated_provider_factory, provider="simulated", model="simulated")
        goal = "Create a simple task management application"
        graph = orch.create_graph(goal, standard_plan(goal))
        summary = orch.run(graph.id)

        self.assertEqual(summary.status, "REVIEW", summary)
        tasks = orch.tasks.topological_order(graph.id)
        work = [t for t in tasks if t.kind != "project"]
        self.assertEqual(len(work), 6)
        self.assertTrue(all(t.status is TaskStatus.COMPLETED for t in work))
        root = orch.tasks.require(graph.root_task_id)
        self.assertEqual(root.status, TaskStatus.REVIEW)

        # structured results with every required field, verified against git
        for t in work:
            self.assertTrue(REQUIRED_RESULT_FIELDS <= set(t.result), t.title)
            self.assertTrue(t.branch.startswith(f"increscent/{t.id}/"))
            if t.profile in ("CodeReviewer",):
                self.assertIsNone(t.result["commit"])
            else:
                self.assertEqual(git(self.repo.path, "rev-parse", t.branch), t.result["commit"])
        qa = next(t for t in work if t.profile == "QAEngineer")
        self.assertEqual((qa.tests["run"], qa.tests["failed"]), (3, 0))
        self.assertEqual(summary.verification["failed"], 0)

        # each task was built on top of its dependency's branch
        frontend = next(t for t in work if t.profile == "FrontendDeveloper")
        files = git(self.repo.path, "ls-tree", "-r", "--name-only", frontend.branch).splitlines()
        self.assertTrue({"taskapp/storage.py", "taskapp/service.py", "taskapp/cli.py", "docs/ARCHITECTURE.md"} <= set(files))

        # main / the user's checkout untouched; nothing merged
        self.assertEqual(self.repo.head(), self.repo.initial)
        self.assertFalse((self.repo.path / "taskapp").exists())

        # review request + report + agents + events
        review = orch.bus.query(type=MessageType.REVIEW_REQUEST)
        self.assertEqual(review[0].recipient, "human")
        report = Path(summary.report_path).read_text(encoding="utf-8")
        self.assertIn("status **REVIEW**", report)
        self.assertEqual(len(orch.agents.list()), 6)
        self.assertTrue(all(a.status is AgentStatus.COMPLETED for a in orch.agents.list()))
        names = {e["event"] for e in orch.events.read()}
        for expected in ("agent.created", "agent.started", "agent.tool_started", "agent.tool_completed", "agent.completed",
                         "task.created", "task.started", "task.completed", "worktree.created", "commit.created",
                         "review.requested"):
            self.assertIn(expected, names)
        assignments = orch.bus.query(type=MessageType.TASK_ASSIGNMENT)
        self.assertEqual(len(assignments), 6)

    def test_state_survives_restart(self):
        orch = self.orchestrator(simulated_provider_factory)
        graph = orch.create_graph("demo", standard_plan("demo")[:2])
        orch.run(graph.id)
        fresh = self.orchestrator(simulated_provider_factory)
        self.assertEqual(len(fresh.tasks.list(graph_id=graph.id)), 3)
        self.assertEqual(fresh.tasks.get_graph(graph.id).status, "REVIEW")
        self.assertEqual(len(fresh.agents.list()), 2)


class FailureRecoveryTests(OrchestratorCase):
    PLAN = [
        TaskSpec("a", "Backend", "build backend", "BackendDeveloper"),
        TaskSpec("b", "Frontend", "build frontend", "FrontendDeveloper", ("a",)),
    ]

    def test_retry_receives_failure_context_and_reuses_worktree(self):
        team = ScriptedTeam({"Backend": [silent_failure, write_and_report("api.py")],
                             "Frontend": [write_and_report("ui.py")]})
        orch = self.orchestrator(team, max_attempts=3)
        graph = orch.create_graph("g", self.PLAN)
        summary = orch.run(graph.id)
        self.assertEqual(summary.status, "REVIEW")
        backend = next(t for t in orch.tasks.list(graph_id=graph.id) if t.title == "Backend")
        self.assertEqual(backend.attempts, 2)
        self.assertEqual([h["status"] for h in backend.history], ["failed", "completed"])
        self.assertIn("ReportResult", backend.history[0]["error"])
        second_prompt = team.assignments["Backend"][1]
        self.assertIn("Previous attempts failed", second_prompt)
        self.assertIn("ReportResult", second_prompt)
        self.assertIn("different approach", second_prompt)
        self.assertNotIn("Previous attempts failed", team.assignments["Backend"][0])
        self.assertIn("task.retry", {e["event"] for e in orch.events.read()})

    def test_exhausted_retries_fail_task_block_dependents_and_preserve_worktree(self):
        team = ScriptedTeam({"Backend": [silent_failure], "Frontend": [write_and_report("ui.py")]})
        orch = self.orchestrator(team, max_attempts=2)
        graph = orch.create_graph("g", self.PLAN)
        summary = orch.run(graph.id)
        self.assertEqual(summary.status, "FAILED")
        tasks = {t.title: t for t in orch.tasks.list(graph_id=graph.id)}
        self.assertEqual(tasks["Backend"].status, TaskStatus.FAILED)
        self.assertEqual(tasks["Backend"].attempts, 2)
        self.assertEqual(tasks["Frontend"].status, TaskStatus.BLOCKED)
        self.assertTrue(Path(orch.worktrees.get(tasks["Backend"].worktree).path).exists())
        self.assertIn("task.failed", {e["event"] for e in orch.events.read()})
        self.assertTrue(orch.bus.query(type=MessageType.ERROR))

    def test_manual_retry_after_failure(self):
        team = ScriptedTeam({"Backend": [silent_failure, write_and_report("api.py")],
                             "Frontend": [write_and_report("ui.py")]})
        orch = self.orchestrator(team, max_attempts=1)
        graph = orch.create_graph("g", self.PLAN)
        self.assertEqual(orch.run(graph.id).status, "FAILED")
        backend = next(t for t in orch.tasks.list(graph_id=graph.id) if t.title == "Backend")
        orch.retry_task(backend.id)
        self.assertEqual(orch.run(graph.id).status, "REVIEW")

    def test_interrupted_run_is_recovered(self):
        team = ScriptedTeam({"Backend": [write_and_report("api.py")], "Frontend": [write_and_report("ui.py")]})
        orch = self.orchestrator(team)
        graph = orch.create_graph("g", self.PLAN)
        backend = next(t for t in orch.tasks.list(graph_id=graph.id) if t.title == "Backend")
        orch.tasks.mutate(backend.id, lambda t: (setattr(t, "status", TaskStatus.RUNNING), setattr(t, "attempts", 1)))
        summary = orch.run(graph.id)  # "--resume" after a crash
        self.assertEqual(summary.status, "REVIEW")
        self.assertIn("interrupted", orch.tasks.require(backend.id).history[0]["error"])


class ControlAndApprovalTests(OrchestratorCase):
    def test_paused_task_is_not_executed_until_resumed(self):
        team = ScriptedTeam({"Backend": [write_and_report("api.py")], "Frontend": [write_and_report("ui.py")]})
        orch = self.orchestrator(team)
        graph = orch.create_graph("g", FailureRecoveryTests.PLAN)
        frontend = next(t for t in orch.tasks.list(graph_id=graph.id) if t.title == "Frontend")
        orch.pause_task(frontend.id, "waiting for UX")
        self.assertEqual(orch.run(graph.id).status, "BLOCKED")
        self.assertNotIn("Frontend", team.assignments)
        orch.resume_task(frontend.id)
        self.assertEqual(orch.run(graph.id).status, "REVIEW")

    def test_agent_approval_requests_reach_the_human_handler(self):
        def fetcher(agent):
            return [reply("", [tool_call("n", "Bash", {"command": "curl https://example.invalid/spec"})]),
                    reply("", [tool_call("r", "ReportResult", {"status": "completed", "summary": "no network"})]),
                    reply("ok")]

        team = ScriptedTeam({"Backend": [fetcher]})
        orch = self.orchestrator(team, approve=False)
        graph = orch.create_graph("g", FailureRecoveryTests.PLAN[:1])
        orch.run(graph.id)
        self.assertEqual(len(self.approvals), 1)
        request = self.approvals[0]
        self.assertIn("NETWORK", request.capabilities)
        self.assertTrue(request.agent_id.startswith("A-"))
        backend = orch.tasks.list(graph_id=graph.id)[1]
        self.assertTrue(any("NETWORK" in w for w in backend.result["warnings"]))

    def test_parallel_independent_tasks_use_distinct_worktrees(self):
        plan = [TaskSpec("x", "X", "", "Developer"), TaskSpec("y", "Y", "", "Developer"),
                TaskSpec("z", "Z", "", "CodeReviewer", ("x", "y"))]

        def reviewer(agent):
            return [reply("", [tool_call("r", "ReportResult", {"status": "completed", "summary": "ok",
                                                               "review_verdict": "approve"})]), reply("ok")]

        team = ScriptedTeam({"X": [write_and_report("x.py")], "Y": [write_and_report("y.py")], "Z": [reviewer]})
        orch = self.orchestrator(team, max_parallel=2)
        graph = orch.create_graph("g", plan)
        self.assertEqual(orch.run(graph.id).status, "REVIEW")
        tasks = {t.title: t for t in orch.tasks.list(graph_id=graph.id)}
        self.assertNotEqual(tasks["X"].worktree, tasks["Y"].worktree)
        # the reviewer's worktree merged both independent branches
        review_path = Path(orch.worktrees.get(tasks["Z"].worktree).path)
        self.assertTrue((review_path / "x.py").exists() and (review_path / "y.py").exists())
        self.assertTrue(all(w.locked_by is None for w in orch.worktrees.list(managed_only=True)))

    def test_merge_requires_explicit_approval_and_never_touches_main(self):
        team = ScriptedTeam({"Backend": [write_and_report("api.py")], "Frontend": [write_and_report("ui.py")]})
        orch = self.orchestrator(team)
        graph = orch.create_graph("g", FailureRecoveryTests.PLAN)
        orch.run(graph.id)
        with self.assertRaises(PermissionError):
            orch.prepare_merge(graph.id, approved=False)
        outcome = orch.prepare_merge(graph.id, approved=True)
        self.assertEqual(outcome["integration_branch"], f"increscent/{graph.id}/integration")
        files = git(self.repo.path, "ls-tree", "-r", "--name-only", outcome["integration_branch"]).splitlines()
        self.assertTrue({"api.py", "ui.py"} <= set(files))
        self.assertEqual(self.repo.head(), self.repo.initial)
        self.assertEqual(orch.tasks.get_graph(graph.id).status, "INTEGRATED")
        events = {e["event"] for e in orch.events.read()}
        self.assertTrue({"approval.denied", "approval.approved", "merge.prepared"} <= events)

    def test_qa_counts_are_measured_not_trusted(self):
        tests_file = ("import unittest\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(True)\n"
                      "    def test_b(self):\n        self.assertEqual(1, 2)\n")

        def liar(agent):
            path = Path(agent.working_directory) / "tests"
            return [
                reply("", [tool_call("w0", "Write", {"file_path": str(path / "__init__.py"), "content": ""}),
                           tool_call("w1", "Write", {"file_path": str(path / "test_x.py"), "content": tests_file})]),
                reply("", [tool_call("r", "ReportResult", {"status": "completed", "summary": "all green",
                                                           "tests_run": 2, "tests_passed": 2, "tests_failed": 0})]),
                reply("ok"),
            ]

        team = ScriptedTeam({"QA": [liar]})
        orch = self.orchestrator(team, max_attempts=1)
        graph = orch.create_graph("g", [TaskSpec("qa", "QA", "", "QAEngineer")])
        orch.run(graph.id)
        qa = orch.tasks.list(graph_id=graph.id)[1]
        self.assertEqual(qa.status, TaskStatus.FAILED)
        self.assertEqual((qa.result["tests_run"], qa.result["tests_failed"]), (2, 1))
        self.assertTrue(any("differ from measured" in w for w in qa.result["warnings"]))


if __name__ == "__main__":
    unittest.main()
