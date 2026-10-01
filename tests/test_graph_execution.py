"""Phase 3.2.2: the AI-planned task graph executes through the DeveloperAgent path."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest import mock

from gitrepo import TempRepo, git
from src.agents.developer import DeveloperAgent
from src.orchestration.ai_planner import Planner, plan_to_task_graph
from src.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from src.orchestration.planner import TaskSpec
from src.orchestration.simulation import (SIMULATED_PLAN, SimulatedPlannerProvider,
                                          simulated_developer_factory)
from src.taskgraph.models import TaskStatus

GOAL = "Create a simple task management application"


def plan_with(tasks: list[dict]) -> dict:
    plan = copy.deepcopy(SIMULATED_PLAN)
    plan["tasks"] = tasks
    return plan


TWO_STEP = [
    {"id": "notes", "title": "Write notes", "role": "backend", "dependencies": [],
     "description": "Create a file named app/notes.txt containing the text 'first step'.",
     "acceptance_criteria": ["app/notes.txt exists"]},
    {"id": "summary", "title": "Extend notes", "role": "backend", "dependencies": ["notes"],
     "description": "Create a file named app/summary.txt containing the contents of app/notes.txt "
                    "followed by the text 'second step'.",
     "acceptance_criteria": ["app/summary.txt extends app/notes.txt"]},
]


class GraphExecutionCase(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = TempRepo()
        self.created: list[str] = []  # task id of every developer agent the factory built

    def tearDown(self) -> None:
        self.repo.cleanup()

    def factory(self, agent):
        self.created.append(agent.task_id)
        return simulated_developer_factory(agent)

    def orchestrator(self, **config) -> Orchestrator:
        return Orchestrator(self.repo.path, provider_factory=self.factory, approval_handler=lambda r: False,
                            config=OrchestratorConfig(**{"provider": "simulated", "model": "simulated-developer",
                                                         "executor": "developer", "max_attempts": 1, **config}),
                            worktree_base=self.repo.worktrees)

    def planned_graph(self, orch: Orchestrator, plan: dict | None = None):
        ai_plan = Planner(SimulatedPlannerProvider(plan), orch.profiles).create_plan(GOAL)
        return ai_plan, plan_to_task_graph(ai_plan, orch, GOAL)

    def by_title(self, orch, graph):
        return {t.title: t for t in orch.tasks.topological_order(graph.id) if t.kind != "project"}


class PlannedGraphTests(GraphExecutionCase):
    def test_planner_output_becomes_an_executable_graph(self):
        orch = self.orchestrator()
        ai_plan, graph = self.planned_graph(orch)
        tasks = self.by_title(orch, graph)
        self.assertEqual([t.title for t in tasks.values()], [t.title for t in ai_plan.tasks])
        ids = {p.id: tasks[p.title].id for p in ai_plan.tasks}
        for p in ai_plan.tasks:
            self.assertEqual(sorted(tasks[p.title].dependencies), sorted(ids[d] for d in p.dependencies))
        self.assertEqual(orch.tasks.ready_tasks(graph.id)[0].title, ai_plan.tasks[0].title)

        summary = orch.run(graph.id)

        self.assertEqual(summary.status, "REVIEW", summary)
        final = self.by_title(orch, graph)
        self.assertTrue(all(t.status is TaskStatus.COMPLETED for t in final.values()), final)
        self.assertEqual(git(self.repo.path, "rev-parse", "main"), self.repo.initial)  # main untouched

    def test_dependency_order_is_respected(self):
        orch = self.orchestrator()
        _, graph = self.planned_graph(orch)
        orch.run(graph.id)
        order = self.created
        tasks = orch.tasks.topological_order(graph.id)
        position = {task_id: i for i, task_id in enumerate(order)}
        for t in tasks:
            if t.kind == "project":
                continue
            for dep in t.dependencies:
                self.assertLess(position[dep], position[t.id], f"{t.title} ran before its dependency")
        self.assertEqual(len(order), 6)  # one developer agent per task, no retries

    def test_developer_agent_receives_a_task_spec(self):
        received: list[tuple[TaskSpec, str]] = []
        original = DeveloperAgent.execute

        def spy(agent_self, task, workspace, *args, **kwargs):
            received.append((task, str(workspace)))
            return original(agent_self, task, workspace, *args, **kwargs)

        orch = self.orchestrator()
        _, graph = self.planned_graph(orch, plan_with(TWO_STEP))
        with mock.patch.object(DeveloperAgent, "execute", spy):
            orch.run(graph.id)

        self.assertEqual(len(received), 2)
        tasks = self.by_title(orch, graph)
        first, second = received
        self.assertIsInstance(first[0], TaskSpec)
        self.assertEqual(first[0].key, tasks["Write notes"].id)
        self.assertIn("app/notes.txt", first[0].description)
        self.assertEqual(Path(first[1]), Path(orch.worktrees.get(tasks["Write notes"].worktree).path))
        # the dependent spec names the completed dependency and its files
        self.assertIn("Completed dependencies", second[0].description)
        self.assertIn("app/notes.txt", second[0].description.split("Completed dependencies", 1)[1])

    def test_dependent_task_sees_the_previous_task_workspace(self):
        orch = self.orchestrator()
        _, graph = self.planned_graph(orch, plan_with(TWO_STEP))
        orch.run(graph.id)
        tasks = self.by_title(orch, graph)
        first, second = tasks["Write notes"], tasks["Extend notes"]

        self.assertNotEqual(first.worktree, second.worktree)
        summary = (Path(orch.worktrees.get(second.worktree).path) / "app" / "summary.txt").read_text(encoding="utf-8")
        self.assertEqual(summary.splitlines(), ["first step", "second step"])
        # task 1's commit is an ancestor of task 2's branch: the dependent built on it
        git(self.repo.path, "merge-base", "--is-ancestor", first.branch, second.branch)
        self.assertEqual(git(self.repo.path, "show", f"{second.branch}:app/notes.txt").strip(), "first step")

    def test_successful_task_is_measured_and_recorded(self):
        orch = self.orchestrator()
        _, graph = self.planned_graph(orch, plan_with(TWO_STEP[:1]))
        orch.run(graph.id)
        task = self.by_title(orch, graph)["Write notes"]

        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(sorted(task.files_changed), ["app/notes.txt", "tests/test_notes_content.py"])
        self.assertEqual((task.tests["run"], task.tests["passed"], task.tests["failed"]), (1, 1, 0))
        self.assertTrue(task.agent and task.worktree and task.branch)
        self.assertIsNone(task.error)
        self.assertEqual(task.result["status"], "completed")
        self.assertEqual(orch.agents.get(task.agent).task_id, task.id)
        self.assertEqual(git(self.repo.path, "show", f"{task.branch}:app/notes.txt").strip(), "first step")

    def test_files_changed_comes_from_the_filesystem_not_the_model(self):
        orch = self.orchestrator()
        _, graph = self.planned_graph(orch, plan_with(TWO_STEP[:1]))
        real_report = __import__("src.orchestration.simulation", fromlist=["x"]).SimulatedDeveloperProvider._finish

        def lying_finish(self_, rel, test_rel, **kwargs):  # claims a file it never wrote
            response = real_report(self_, rel, test_rel, **kwargs)
            response.tool_uses[0]["input"]["files_changed"] = [rel, "app/invented.py"]
            return response

        with mock.patch("src.orchestration.simulation.SimulatedDeveloperProvider._finish", lying_finish):
            orch.run(graph.id)
        task = self.by_title(orch, graph)["Write notes"]
        self.assertNotIn("app/invented.py", task.files_changed)
        self.assertIn("app/notes.txt", task.files_changed)


class FailureTests(GraphExecutionCase):
    FAILING = [
        {"id": "broken", "title": "Broken step", "role": "backend", "dependencies": [],
         "description": "Create a file named app/out.txt containing the contents of missing/input.txt "
                        "followed by the text 'never'.",
         "acceptance_criteria": ["app/out.txt exists"]},
        {"id": "after", "title": "Dependent step", "role": "backend", "dependencies": ["broken"],
         "description": "Create a file named app/next.txt containing the contents of app/out.txt "
                        "followed by the text 'never either'.",
         "acceptance_criteria": ["app/next.txt exists"]},
    ]

    def test_failed_task_blocks_dependents_without_running_them(self):
        orch = self.orchestrator()
        _, graph = self.planned_graph(orch, plan_with(self.FAILING))
        summary = orch.run(graph.id)
        tasks = self.by_title(orch, graph)
        broken, after = tasks["Broken step"], tasks["Dependent step"]

        self.assertNotEqual(summary.status, "REVIEW")
        self.assertEqual(broken.status, TaskStatus.FAILED)
        self.assertTrue(broken.error)
        self.assertIn("missing/input.txt", broken.result["summary"])
        self.assertEqual(broken.attempts, 1)  # no automatic repair loop
        self.assertEqual(after.status, TaskStatus.BLOCKED)
        self.assertIsNone(after.agent)
        self.assertEqual(self.created, [broken.id])  # the dependent never got an agent
        # the failure is preserved for inspection
        self.assertEqual(broken.result["status"], "failed")
        self.assertTrue(broken.history and broken.history[-1].get("error"))
        self.assertFalse((Path(orch.worktrees.get(broken.worktree).path) / "app" / "out.txt").exists())
        self.assertEqual(git(self.repo.path, "rev-parse", "main"), self.repo.initial)


if __name__ == "__main__":
    unittest.main()
