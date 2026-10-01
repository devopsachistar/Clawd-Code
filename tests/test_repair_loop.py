"""Phase 3.3.2: controlled self-repair (developer -> measured FAIL -> QA diagnosis -> repair -> measured)."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from fakes import FakeOpenAIProvider, reply
from gitrepo import TempRepo, git
from src.agents.developer import REPAIR_MARKER, snapshot
from src.cli import main
from src.orchestration.ai_planner import Planner, plan_to_task_graph
from src.orchestration.orchestrator import MAX_REPAIRS, Orchestrator, OrchestratorConfig
from src.orchestration.simulation import (SIMULATED_CALC_PLAN, SimulatedDeveloperProvider, SimulatedPlannerProvider,
                                          SimulatedQAProvider)
from src.taskgraph.models import TaskStatus

GOAL = "Implement a calculation module"


class SpyQAProvider(SimulatedQAProvider):
    """SimulatedQAProvider that records the tools it was offered and whether the workspace changed."""

    def __init__(self, workspace: str, calls: list) -> None:
        super().__init__()
        self.workspace, self.calls = Path(workspace), calls

    def chat(self, messages, tools=None, **kwargs):
        before = snapshot(self.workspace)
        response = super().chat(messages, tools=tools, **kwargs)
        self.calls.append({"tools": tools, "kwargs": kwargs, "unchanged": snapshot(self.workspace) == before,
                           "prompt": messages[-1]["content"]})
        return response


class RepairCase(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = TempRepo()
        self.addCleanup(self.repo.cleanup)
        self.developers: list[SimulatedDeveloperProvider] = []
        self.qa_calls: list[dict] = []

    def run_graph(self, *, unfixable=False, max_repairs=2, qa_factory=None):
        def developer_factory(agent):
            provider = SimulatedDeveloperProvider(agent.working_directory, unfixable=unfixable)
            self.developers.append(provider)
            return provider

        self.orch = Orchestrator(
            self.repo.path, provider_factory=developer_factory, approval_handler=lambda r: False,
            config=OrchestratorConfig(provider="simulated", model="simulated-developer", executor="developer",
                                      max_attempts=1, max_repairs=max_repairs),
            worktree_base=self.repo.worktrees,
            qa_provider_factory=qa_factory or (lambda agent: SpyQAProvider(agent.working_directory, self.qa_calls)))
        plan = Planner(SimulatedPlannerProvider(), self.orch.profiles).create_plan(GOAL)
        self.graph = plan_to_task_graph(plan, self.orch, GOAL)
        self.summary = self.orch.run(self.graph.id)
        tasks = {t.title: t for t in self.orch.tasks.list(graph_id=self.graph.id) if t.kind != "project"}
        return tasks["Implement calculation"], tasks["Review the calculation"]

    def agents_of(self, task, profile=None):
        return [a for a in self.orch.agents.list() if a.task_id == task.id and (profile is None or a.profile == profile)]

    def worktree(self, task) -> Path:
        return Path(self.orch.worktrees.get(task.worktree).path)


class SuccessfulRepairTests(RepairCase):
    def setUp(self) -> None:
        super().setUp()
        self.task, self.review = self.run_graph()

    def test_developer_failure_triggers_qa(self):
        log = self.task.result["attempt_log"]
        self.assertEqual([s["step"] for s in log], ["developer", "qa", "developer", "final"])
        self.assertEqual((log[0]["status"], log[0]["agent_claimed"]), ("failed", "completed"))  # claim not trusted
        self.assertEqual(log[0]["tests"], {"run": 1, "passed": 0, "failed": 1})
        self.assertEqual(len(self.qa_calls), 1)
        self.assertIn("assert -1 == 5", self.qa_calls[0]["prompt"])  # QA saw the measured failure

    def test_qa_diagnosis_is_passed_to_the_developer(self):
        self.assertEqual(len(self.developers), 3)  # attempt 1, repair, reviewer
        first, repair = self.developers[0].assignment, self.developers[1].assignment
        self.assertNotIn(REPAIR_MARKER, first)
        self.assertIn(f"{REPAIR_MARKER} 1 of 2", repair)
        self.assertIn("This is a REPAIR attempt", repair)
        before_repair = repair.split(REPAIR_MARKER, 1)[0]
        self.assertIn("Create a function add(a, b) in app/calc.py that returns a + b.", before_repair)  # original
        self.assertIn("add(2, 3) returns 5", before_repair)                                              # criteria
        context = repair.split(REPAIR_MARKER, 1)[1]
        for fact in ("Measured tests: run 1, passed 0, failed 1", "expected 5, received -1",
                     "the implementation returns an incorrect result",
                     "Correct the implementation of add() so add(2, 3) returns 5 instead of -1.",
                     "### Relevant test output", "assert -1 == 5"):
            self.assertIn(fact, context)

    def test_repair_changes_the_same_worktree(self):
        developers = self.agents_of(self.task, "BackendDeveloper")
        self.assertEqual(len(developers), 2)
        self.assertEqual({a.working_directory for a in developers}, {str(self.worktree(self.task).resolve())})
        own = [w for w in self.orch.worktrees.list(managed_only=True) if self.task.id.lower() in w.name.lower()]
        self.assertEqual(len(own), 1)  # no extra worktree for the repair
        self.assertEqual(self.task.branch, own[0].branch)
        self.assertEqual(self.repo.head("main"), self.repo.initial)

    def test_successful_repair_is_measured_completed_and_committed(self):
        self.assertEqual(self.task.status, TaskStatus.COMPLETED)
        self.assertEqual((self.task.tests["run"], self.task.tests["passed"], self.task.tests["failed"]), (1, 1, 0))
        self.assertEqual(sorted(self.task.files_changed), ["app/calc.py", "tests/test_add_function.py"])
        self.assertEqual(git(self.repo.path, "show", f"{self.task.branch}:app/calc.py").strip(),
                         "def add(a, b):\n    return a + b")
        self.assertEqual(git(self.repo.path, "rev-list", "--count", f"main..{self.task.branch}"), "1")  # one final commit
        self.assertEqual(self.review.status, TaskStatus.COMPLETED)
        self.assertEqual(self.summary.status, "REVIEW")  # not merged; review stays a human step

    def test_repair_attempts_are_recorded(self):
        log = self.task.result["attempt_log"]
        self.assertTrue(log[2]["repair"])
        self.assertEqual((log[2]["attempt"], log[2]["status"], log[2]["tests"]["passed"]), (2, "completed", 1))
        self.assertEqual((log[1]["status"], log[1]["confidence"]), ("diagnosed", "high"))
        self.assertEqual(log[1]["failures"][0]["evidence"], "assert -1 == 5")
        self.assertEqual(log[-1], {"step": "final", "status": "COMPLETED", "repairs": 1, "commit": log[-1]["commit"]})
        self.assertTrue(log[-1]["commit"])
        self.assertEqual(self.task.history[-1]["attempt_log"], log)

    def test_qa_cannot_modify_files(self):
        call = self.qa_calls[0]
        self.assertFalse(call["tools"])
        self.assertNotIn("tools", call["kwargs"])
        self.assertTrue(call["unchanged"])
        qa = self.agents_of(self.task, "QAEngineer")
        self.assertEqual(len(qa), 1)
        self.assertEqual(qa[0].allowed_tools, [])
        caps = qa[0].permissions["capabilities"]
        self.assertEqual({c for c, d in caps.items() if d != "DENY"}, {"READ"})
        self.assertEqual(qa[0].result["status"], "diagnosed")


class FailedRepairTests(RepairCase):
    def test_maximum_repairs_are_enforced_and_the_task_stays_failed(self):
        task, review = self.run_graph(unfixable=True)
        log = task.result["attempt_log"]
        self.assertEqual([s["step"] for s in log], ["developer", "qa", "developer", "qa", "developer", "final"])
        self.assertEqual([s["status"] for s in log if s["step"] == "developer"], ["failed"] * 3)
        self.assertEqual(log[-1]["status"], "FAILED")
        self.assertEqual(log[-1]["repairs"], MAX_REPAIRS)
        self.assertEqual(len(self.agents_of(task, "BackendDeveloper")), 1 + MAX_REPAIRS)
        self.assertEqual(len(self.qa_calls), MAX_REPAIRS)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertEqual(task.result["status"], "failed")
        self.assertTrue(task.error)
        self.assertEqual(task.tests["failed"], 1)
        self.assertEqual(git(self.repo.path, "rev-list", "--count", f"main..{task.branch}"), "0")  # nothing committed
        # the dependent never runs
        self.assertEqual(review.status, TaskStatus.BLOCKED)
        self.assertEqual(self.agents_of(review), [])
        self.assertNotEqual(self.summary.status, "REVIEW")
        self.assertEqual(self.repo.head("main"), self.repo.initial)

    def test_configured_repairs_are_capped(self):
        task, _ = self.run_graph(unfixable=True, max_repairs=10)
        self.assertEqual(task.result["attempt_log"][-1]["repairs"], MAX_REPAIRS)
        self.assertEqual(len(self.agents_of(task, "BackendDeveloper")), 1 + MAX_REPAIRS)

    def test_inconclusive_diagnosis_stops_without_a_blind_repair(self):
        inconclusive = '{"status": "inconclusive", "summary": "unclear", "failures": [], ' \
                       '"recommended_fixes": [], "confidence": "low"}'
        task, review = self.run_graph(qa_factory=lambda agent: FakeOpenAIProvider([reply(inconclusive)]))
        self.assertEqual([s["step"] for s in task.result["attempt_log"]], ["developer", "qa", "final"])
        self.assertEqual(len(self.agents_of(task, "BackendDeveloper")), 1)
        self.assertEqual((task.status, review.status), (TaskStatus.FAILED, TaskStatus.BLOCKED))

    def test_repair_loop_is_off_by_default(self):
        task, review = self.run_graph(max_repairs=0)
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertNotIn("attempt_log", task.result)
        self.assertEqual(self.qa_calls, [])
        self.assertEqual(len(self.agents_of(task)), 1)


class RepairCommandTests(unittest.TestCase):
    def cli(self, *argv):
        repo = TempRepo()
        self.addCleanup(repo.cleanup)
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["team", *argv, "--cwd", str(repo.path)])
        return code, out.getvalue() + err.getvalue(), repo

    def test_team_ai_plan_repairs_a_failing_task(self):
        code, output, repo = self.cli("--simulate", "--ai-plan", "--max-repairs", "2", GOAL)
        self.assertEqual(code, 0, output)
        for line in ("Implement calculation  [COMPLETED]", "Attempt 1: FAILED", "QA: diagnosed (high)",
                     "Repair: attempt 2", "Attempt 2: COMPLETED  (tests run 1, passed 1, failed 0",
                     "RESULT: COMPLETED after 1 repair(s)", "qa.diagnosed", "task.repair"):
            self.assertIn(line, output)
        self.assertEqual(repo.head("main"), repo.initial)

    def test_repair_limit_and_mode_are_validated(self):
        for argv in (("--simulate", "--ai-plan", "--max-repairs", "3", GOAL), ("--simulate", "--max-repairs", "1", GOAL)):
            with self.subTest(argv=argv):
                code, output, _ = self.cli(*argv)
                self.assertEqual(code, 2, output)


if __name__ == "__main__":
    unittest.main()
