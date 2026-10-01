"""Phase 3.1: requirement -> LLM -> validated plan -> existing Task Graph."""

from __future__ import annotations

import copy
import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout

from fakes import FailingProvider, FakeAnthropicProvider, FakeOpenAIProvider, reply
from gitrepo import TempRepo
from src.agents.profiles import ProfileRegistry
from src.cli import main
from src.orchestration.ai_planner import Planner, PlanningError, plan_to_task_graph
from src.orchestration.orchestrator import Orchestrator
from src.orchestration.simulation import SIMULATED_PLAN, SimulatedPlannerProvider
from src.taskgraph.models import TaskStatus

REQUIREMENT = "Build a simple task management application with users, projects, tasks, priorities and a dashboard."
PROFILES = ProfileRegistry.load()


def planner_returning(content: str, provider_cls=FakeOpenAIProvider) -> Planner:
    return Planner(provider_cls([reply(content)]), PROFILES)


def plan_json(mutate=None) -> str:
    data = copy.deepcopy(SIMULATED_PLAN)
    if mutate:
        mutate(data)
    return json.dumps(data)


class PlannerTests(unittest.TestCase):
    def test_valid_requirement_produces_valid_plan(self):
        provider = SimulatedPlannerProvider()
        plan = Planner(provider, PROFILES).create_plan(REQUIREMENT)
        self.assertEqual(plan.project_name, "Task Manager")
        self.assertEqual([t.id for t in plan.tasks], ["architecture", "database", "backend", "frontend", "tests", "review"])
        self.assertEqual(plan.tasks[1].dependencies, ("architecture",))
        self.assertEqual(set(plan.architecture), {"frontend", "backend", "database", "infrastructure"})
        self.assertTrue(plan.tasks[0].acceptance_criteria)
        # the prompt carries the requirement, the schema and the configurable roles
        system, user = provider.requests[0][0]["content"], provider.requests[0][1]["content"]
        self.assertIn(REQUIREMENT, user)
        self.assertIn('"tasks"', system)
        for role in ("architect", "backend", "qa", "reviewer"):
            self.assertIn(role, system)

    def test_provider_neutral_request(self):
        provider = FakeAnthropicProvider([reply(plan_json())])
        Planner(provider, PROFILES).create_plan(REQUIREMENT)
        request = provider.requests[0]
        self.assertIn("JSON", request["kwargs"]["system"])          # anthropic wire format: system kwarg
        self.assertEqual([m["role"] for m in request["messages"]], ["user"])

    def test_fenced_json_is_accepted(self):
        plan = planner_returning("Here is the plan:\n```json\n" + plan_json() + "\n```").create_plan(REQUIREMENT)
        self.assertEqual(len(plan.tasks), 6)

    def test_malformed_json(self):
        with self.assertRaisesRegex(PlanningError, "not valid JSON"):
            planner_returning('{"project": {"name": "x",').create_plan(REQUIREMENT)
        with self.assertRaisesRegex(PlanningError, "empty response"):
            planner_returning("").create_plan(REQUIREMENT)

    def test_missing_required_field(self):
        with self.assertRaises(PlanningError) as ctx:
            planner_returning(plan_json(lambda d: d.pop("architecture"))).create_plan(REQUIREMENT)
        self.assertIn("missing required field 'architecture'", ctx.exception.issues)
        with self.assertRaises(PlanningError) as ctx:
            planner_returning(plan_json(lambda d: d["tasks"][2].pop("role"))).create_plan(REQUIREMENT)
        self.assertTrue(any("tasks[2].role" in i for i in ctx.exception.issues))

    def test_wrong_types_and_empty_title(self):
        def mutate(d):
            d["requirements"] = "not a list"
            d["tasks"][0]["title"] = "  "
        with self.assertRaises(PlanningError) as ctx:
            planner_returning(plan_json(mutate)).create_plan(REQUIREMENT)
        issues = " | ".join(ctx.exception.issues)
        self.assertIn("requirements must be a list", issues)
        self.assertIn("tasks[0].title must not be empty", issues)

    def test_duplicate_task_id(self):
        with self.assertRaises(PlanningError) as ctx:
            planner_returning(plan_json(lambda d: d["tasks"][3].update(id="backend"))).create_plan(REQUIREMENT)
        self.assertIn("duplicate task id 'backend'", ctx.exception.issues)

    def test_invalid_dependency(self):
        with self.assertRaises(PlanningError) as ctx:
            planner_returning(plan_json(lambda d: d["tasks"][2].update(dependencies=["auth-service"]))).create_plan(REQUIREMENT)
        self.assertIn("task 'backend' depends on unknown task 'auth-service'", ctx.exception.issues)

    def test_dependency_cycle(self):
        with self.assertRaises(PlanningError) as ctx:
            planner_returning(plan_json(lambda d: d["tasks"][0].update(dependencies=["review"]))).create_plan(REQUIREMENT)
        self.assertTrue(any(i.startswith("dependency cycle:") for i in ctx.exception.issues), ctx.exception.issues)

    def test_invalid_role(self):
        with self.assertRaises(PlanningError) as ctx:
            planner_returning(plan_json(lambda d: d["tasks"][0].update(role="wizard"))).create_plan(REQUIREMENT)
        self.assertTrue(any("not a valid agent role" in i for i in ctx.exception.issues))

    def test_provider_failure(self):
        planner = Planner(FailingProvider(TimeoutError("request timed out")), PROFILES)
        with self.assertRaisesRegex(PlanningError, "provider failed: TimeoutError: request timed out"):
            planner.create_plan(REQUIREMENT)

    def test_truncated_output_is_reported(self):
        response = reply(plan_json()[:200])
        response.finish_reason = "length"
        with self.assertRaisesRegex(PlanningError, "truncated"):
            Planner(FakeOpenAIProvider([response]), PROFILES).create_plan(REQUIREMENT)


class TaskGraphIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = TempRepo()

    def tearDown(self) -> None:
        self.repo.cleanup()

    def orchestrator(self) -> Orchestrator:
        return Orchestrator(self.repo.path, provider_factory=lambda a: None, worktree_base=self.repo.worktrees)

    def test_valid_plan_becomes_existing_task_graph(self):
        orch = self.orchestrator()
        plan = Planner(SimulatedPlannerProvider(), orch.profiles).create_plan(REQUIREMENT)
        graph = plan_to_task_graph(plan, orch, REQUIREMENT)

        fresh = self.orchestrator()  # persisted in the existing TaskStore
        tasks = {t.title: t for t in fresh.tasks.list(graph_id=graph.id) if t.kind != "project"}
        self.assertEqual(len(tasks), 6)
        arch, db = tasks["Design the architecture"], tasks["Implement the data model"]
        review = tasks["Review the changes"]
        self.assertEqual((arch.profile, db.profile, review.profile), ("Architect", "DatabaseEngineer", "CodeReviewer"))
        self.assertEqual(db.dependencies, [arch.id])
        self.assertIn("Acceptance criteria:", arch.description)
        self.assertEqual([t.title for t in fresh.tasks.ready_tasks(graph.id)], ["Design the architecture"])
        self.assertEqual(review.status, TaskStatus.PENDING)
        order = [t.title for t in fresh.tasks.topological_order(graph.id) if t.kind != "project"]
        self.assertEqual(order, ["Design the architecture", "Implement the data model", "Implement the backend API",
                                 "Implement the frontend and dashboard", "Write integration tests", "Review the changes"])
        stored = fresh.tasks.get_graph(graph.id)
        self.assertEqual(stored.goal, REQUIREMENT)
        self.assertEqual(stored.settings["plan"]["project"]["name"], "Task Manager")

    def test_invalid_plan_never_reaches_the_graph(self):
        orch = self.orchestrator()
        with self.assertRaises(PlanningError):
            Planner(FakeOpenAIProvider([reply(plan_json(lambda d: d["tasks"][0].update(dependencies=["review"])))]),
                    orch.profiles).create_plan(REQUIREMENT)
        self.assertEqual(orch.tasks.list_graphs(), [])


class PlanCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = TempRepo()

    def tearDown(self) -> None:
        self.repo.cleanup()

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def test_plan_command_prints_plan_and_creates_graph(self):
        code, out, err = self.cli("plan", "--simulate", "--cwd", str(self.repo.path), REQUIREMENT)
        self.assertEqual(code, 0, err)
        for heading in ("PROJECT", "SUMMARY", "ARCHITECTURE", "TASKS"):
            self.assertIn(heading, out)
        self.assertIn("1. Design the architecture", out)
        self.assertIn("6. Review the changes", out)
        graph_id = re.search(r"Task graph (G-[0-9a-f]{8}) created", out).group(1)
        code, tasks_out, _ = self.cli("tasks", "--graph", graph_id, "--cwd", str(self.repo.path))
        self.assertIn("Implement the backend API", tasks_out)
        self.assertEqual(self.repo.head(), self.repo.initial)  # planning never touches the code

    def test_plan_json_and_no_save(self):
        code, out, _ = self.cli("plan", "--simulate", "--no-save", "--json", "--cwd", str(self.repo.path), REQUIREMENT)
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertIsNone(data["graph_id"])
        self.assertEqual(len(data["plan"]["tasks"]), 6)

    def test_plan_errors(self):
        self.assertEqual(self.cli("plan", "--simulate", "--cwd", str(self.repo.path))[0], 2)  # no requirement
        code, _, err = self.cli("plan", "--provider", "openai", "--cwd", str(self.repo.path), REQUIREMENT)
        self.assertEqual(code, 2)  # no API key configured
        self.assertIn("OPENAI_API_KEY", err)


if __name__ == "__main__":
    unittest.main()
