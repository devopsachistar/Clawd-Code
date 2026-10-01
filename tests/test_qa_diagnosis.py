"""Phase 3.3.1: QA diagnosis of a failed developer task (diagnosis only; no repair)."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from fakes import FakeOpenAIProvider, reply, tool_call
from gitrepo import TempRepo
from src.agents.developer import DeveloperAgent, DeveloperResult, snapshot
from src.agents.qa import DiagnosisError, QADiagnoser, QADiagnosis, failed_tests, parse_diagnosis
from src.cli import main
from src.orchestration.ai_planner import Planner, plan_to_task_graph
from src.orchestration.orchestrator import Orchestrator, OrchestratorConfig
from src.orchestration.planner import TaskSpec
from src.orchestration.simulation import (SIMULATED_PLAN, SimulatedPlannerProvider, SimulatedQAProvider,
                                          simulated_developer_factory)
from src.taskgraph.models import TaskStatus

ADD_TASK = TaskSpec(key="add", title="Add two numbers", profile="Developer",
                    description="Create a function add(a, b) in calc.py that adds two numbers.",
                    acceptance=("add(2, 3) == 5",))
BUGGY_ADD = "def add(a, b):\n    return a + b - 1\n"
ADD_TEST = "from calc import add\n\n\ndef test_addition():\n    assert add(2, 3) == 5\n"
BUGGY_MUL = "def mul(a, b):\n    return a + b\n"
MUL_TEST = "from calc import mul\n\n\ndef test_multiplication():\n    assert mul(3, 4) == 12\n"
TWO_BUGS = {"calc.py": BUGGY_ADD + "\n\n" + BUGGY_MUL, "test_calc.py": ADD_TEST, "test_mul.py": MUL_TEST}


def diagnosis_json(**overrides) -> str:
    data = {"status": "diagnosed", "summary": "add() is wrong", "confidence": "high",
            "failures": [{"test": "test_addition", "error": "expected 5, received 4",
                          "likely_cause": "the implementation returns an incorrect result",
                          "evidence": "assert 4 == 5"}],
            "recommended_fixes": ["Correct add()."]}
    data.update(overrides)
    return json.dumps(data)


class _Workspace(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.ws = Path(self._tmp.name).resolve() / "workspace"
        self.ws.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def failed_developer_result(self, files: dict[str, str], task: TaskSpec = ADD_TASK) -> DeveloperResult:
        """A real DeveloperAgent run whose (scripted) model writes buggy code and claims success."""
        writes = [tool_call(f"w{i}", "Write", {"file_path": str(self.ws / name), "content": content})
                  for i, (name, content) in enumerate(files.items())]
        script = [reply("", writes),
                  reply("", [tool_call("r", "ReportResult", {"status": "completed", "summary": "Implemented."})]),
                  reply("Done.")]
        result = DeveloperAgent(lambda agent: FakeOpenAIProvider(list(script))).execute(task, self.ws)
        self.assertEqual(result.status, "failed")  # measured: the tests fail despite the claim
        self.assertGreater(result.tests_failed, 0)
        return result


class DiagnosisFromDeveloperTests(_Workspace):
    def test_failed_developer_result_is_diagnosed(self):
        result = self.failed_developer_result({"calc.py": BUGGY_ADD, "test_calc.py": ADD_TEST})
        diagnosis = QADiagnoser(SimulatedQAProvider()).diagnose(ADD_TASK, result)

        self.assertIsInstance(diagnosis, QADiagnosis)
        self.assertEqual((diagnosis.status, diagnosis.confidence, diagnosis.task_id), ("diagnosed", "high", "add"))
        self.assertEqual(len(diagnosis.failures), 1)
        failure = diagnosis.failures[0]
        self.assertEqual(failure.test, "test_calc.py::test_addition")
        self.assertEqual(failure.error, "expected 5, received 4")
        self.assertEqual(failure.likely_cause, "the implementation returns an incorrect result")
        self.assertEqual(failure.evidence, "assert 4 == 5")
        self.assertEqual(diagnosis.recommended_fixes,
                         ["Correct the implementation of add() so add(2, 3) returns 5 instead of 4."])
        self.assertEqual(diagnosis.warnings, [])

    def test_diagnosis_uses_actual_failure_information(self):
        result = self.failed_developer_result({"calc.py": BUGGY_ADD, "test_calc.py": ADD_TEST})
        provider = SimulatedQAProvider()
        diagnosis = QADiagnoser(provider).diagnose(ADD_TASK, result)

        prompt = provider.requests[0][-1]["content"]
        for fact in (ADD_TASK.description, "add(2, 3) == 5", str(self.ws), "calc.py", "test_calc.py",
                     result.test_command, "run 1, passed 0, failed 1", "assert 4 == 5"):
            self.assertIn(fact, prompt)
        for f in diagnosis.failures:  # every quote is really in the measured output
            self.assertIn(f.evidence, result.test_output)
            self.assertIn(f.test, result.test_output)

    def test_multiple_test_failures(self):
        result = self.failed_developer_result(TWO_BUGS)
        self.assertEqual(result.tests_failed, 2)
        diagnosis = QADiagnoser(SimulatedQAProvider()).diagnose(ADD_TASK, result)

        self.assertEqual(diagnosis.status, "diagnosed")
        by_test = {f.test.split("::")[-1]: f for f in diagnosis.failures}
        self.assertEqual(set(by_test), {"test_addition", "test_multiplication"})
        self.assertEqual(by_test["test_addition"].error, "expected 5, received 4")
        self.assertEqual(by_test["test_multiplication"].error, "expected 12, received 7")
        self.assertEqual(len(diagnosis.recommended_fixes), 2)
        self.assertIn("mul(3, 4) returns 12", diagnosis.recommended_fixes[1])

    def test_simulated_qa_does_not_modify_files(self):
        result = self.failed_developer_result({"calc.py": BUGGY_ADD, "test_calc.py": ADD_TEST})
        before = snapshot(self.ws)
        provider = SimulatedQAProvider()
        QADiagnoser(provider).diagnose(ADD_TASK, result)
        self.assertEqual(snapshot(self.ws), before)
        self.assertEqual((self.ws / "calc.py").read_text(encoding="utf-8"), BUGGY_ADD)  # no repair


class GroundingTests(_Workspace):
    def setUp(self) -> None:
        super().setUp()
        self.result = self.failed_developer_result({"calc.py": BUGGY_ADD, "test_calc.py": ADD_TEST})

    def test_invented_evidence_is_rejected(self):
        invented = diagnosis_json(failures=[{"test": "test_addition", "error": "expected 5, received 3",
                                             "likely_cause": "off by two", "evidence": "assert 3 == 5"}])
        diagnosis = QADiagnoser(FakeOpenAIProvider([reply(invented)])).diagnose(ADD_TASK, self.result)
        self.assertEqual((diagnosis.status, diagnosis.confidence), ("inconclusive", "low"))
        self.assertEqual(diagnosis.failures, [])
        self.assertEqual(diagnosis.recommended_fixes, [])
        self.assertIn("evidence not found", diagnosis.warnings[0])

    def test_invented_test_and_mixed_failures(self):
        mixed = diagnosis_json(failures=[
            json.loads(diagnosis_json())["failures"][0],
            {"test": "test_subtraction", "error": "x", "likely_cause": "y", "evidence": "assert 4 == 5"},
        ])
        diagnosis = QADiagnoser(FakeOpenAIProvider([reply(mixed)])).diagnose(ADD_TASK, self.result)
        self.assertEqual(diagnosis.status, "diagnosed")
        self.assertEqual([f.test for f in diagnosis.failures], ["test_addition"])
        self.assertEqual(diagnosis.confidence, "low")  # something had to be dropped
        self.assertTrue(any("test_subtraction" in w and "test not found" in w for w in diagnosis.warnings))

    def test_model_gets_no_tools(self):
        provider = FakeOpenAIProvider([reply(diagnosis_json())])
        QADiagnoser(provider).diagnose(ADD_TASK, self.result)
        self.assertFalse(provider.requests[0]["tools"])
        self.assertNotIn("tools", provider.requests[0]["kwargs"])


class UncoveredFailureTests(_Workspace):
    def test_uncovered_failure_lowers_confidence(self):
        result = self.failed_developer_result(TWO_BUGS)
        diagnosis = QADiagnoser(FakeOpenAIProvider([reply(diagnosis_json())])).diagnose(ADD_TASK, result)
        self.assertEqual((diagnosis.status, diagnosis.confidence), ("diagnosed", "medium"))
        self.assertTrue(any("test_multiplication" in w for w in diagnosis.warnings))


class InsufficientInformationTests(unittest.TestCase):
    def result(self, **kw) -> DeveloperResult:
        return DeveloperResult(**{"status": "failed", "task_id": "add", "summary": "", **kw})

    def test_no_output_and_no_error_is_inconclusive_without_a_model_call(self):
        provider = FakeOpenAIProvider([])
        diagnosis = QADiagnoser(provider).diagnose(ADD_TASK, self.result())
        self.assertEqual((diagnosis.status, diagnosis.confidence, diagnosis.failures), ("inconclusive", "low", []))
        self.assertEqual(provider.requests, [])

    def test_completed_task_is_not_diagnosed(self):
        diagnosis = QADiagnoser(FakeOpenAIProvider([])).diagnose(ADD_TASK, self.result(status="completed"))
        self.assertEqual(diagnosis.status, "inconclusive")

    def test_unrecognised_failure_is_inconclusive(self):
        diagnosis = QADiagnoser(SimulatedQAProvider()).diagnose(
            ADD_TASK, self.result(error="tests timed out", test_output="collected 3 items\n(timeout)"))
        self.assertEqual((diagnosis.status, diagnosis.failures), ("inconclusive", []))

    def test_model_may_answer_inconclusive(self):
        answer = diagnosis_json(status="inconclusive", failures=[], recommended_fixes=[], confidence="high")
        diagnosis = QADiagnoser(FakeOpenAIProvider([reply(answer)])).diagnose(
            ADD_TASK, self.result(error="1 test(s) failed"))
        self.assertEqual((diagnosis.status, diagnosis.confidence), ("inconclusive", "low"))


class StructureTests(unittest.TestCase):
    def test_valid_diagnosis_parses(self):
        d = parse_diagnosis(json.loads(diagnosis_json()))
        self.assertEqual((d.status, d.confidence, d.failures[0].test), ("diagnosed", "high", "test_addition"))
        self.assertEqual(set(d.to_dict()), {"status", "summary", "failures", "recommended_fixes", "confidence",
                                            "task_id", "warnings"})
        self.assertEqual(set(d.to_dict()["failures"][0]), {"test", "error", "likely_cause", "evidence"})

    def test_invalid_structures_are_rejected(self):
        cases = {
            "status": {"status": "fixed"},
            "confidence": {"confidence": "certain"},
            "summary": {"summary": "  "},
            "failures": {"failures": "test_addition"},
            "evidence": {"failures": [{"test": "t", "error": "e", "likely_cause": "c"}]},
            "at least one failure": {"failures": []},
            "recommended_fixes": {"recommended_fixes": "fix it"},
        }
        for needle, override in cases.items():
            with self.subTest(needle), self.assertRaises(DiagnosisError) as ctx:
                parse_diagnosis(json.loads(diagnosis_json(**override)))
            self.assertIn(needle, str(ctx.exception))
        with self.assertRaises(DiagnosisError):
            parse_diagnosis(["not", "an", "object"])

    def test_invalid_model_output_and_provider_errors_are_inconclusive(self):
        result = DeveloperResult(status="failed", task_id="add", summary="", error="boom",
                                 test_output="FAILED test_calc.py::test_addition - assert 4 == 5")

        class Broken(FakeOpenAIProvider):
            def chat(self, messages, tools=None, **kwargs):
                raise TimeoutError("model timed out")

        for provider, warning in ((FakeOpenAIProvider([reply("I think add is wrong")]), "not valid JSON"),
                                  (FakeOpenAIProvider([reply(diagnosis_json(status="maybe"))]), "status must be"),
                                  (Broken([]), "TimeoutError")):
            with self.subTest(warning):
                diagnosis = QADiagnoser(provider).diagnose(ADD_TASK, result)
                self.assertEqual((diagnosis.status, diagnosis.failures), ("inconclusive", []))
                self.assertIn(warning, diagnosis.warnings[0])

    def test_failed_tests_parser(self):
        output = ("FAILED tests/test_a.py::test_x - assert 1 == 2\r\n"
                  "ERROR tests/test_b.py::test_y\n"
                  "FAIL: test_z (pkg.tests.TestZ)\n")
        self.assertEqual(failed_tests(output), [("tests/test_a.py::test_x", "assert 1 == 2"),
                                                ("tests/test_b.py::test_y", ""), ("pkg.tests.TestZ.test_z", "")])


class QACommandTests(unittest.TestCase):
    """A failed task of an AI-planned graph, diagnosed through `increscent qa --simulate`."""

    def test_qa_cli_diagnoses_a_failed_graph_task(self):
        repo = TempRepo({"README.md": "# demo\n", "calc.py": BUGGY_ADD, "test_calc.py": ADD_TEST})
        self.addCleanup(repo.cleanup)
        orch = Orchestrator(repo.path, provider_factory=simulated_developer_factory, approval_handler=lambda r: False,
                            config=OrchestratorConfig(provider="simulated", model="simulated-developer",
                                                      executor="developer", max_attempts=1),
                            worktree_base=repo.worktrees)
        plan = Planner(SimulatedPlannerProvider(SIMULATED_PLAN), orch.profiles).create_plan("Task manager")
        graph = plan_to_task_graph(plan, orch, "Task manager")
        orch.run(graph.id)
        failed = [t for t in orch.tasks.list(graph_id=graph.id) if t.status is TaskStatus.FAILED]
        self.assertEqual(len(failed), 1)
        worktree = Path(orch.worktrees.get(failed[0].worktree).path)
        before = (snapshot(worktree), snapshot(repo.path))

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(["qa", failed[0].id, "--simulate", "--json", "--cwd", str(repo.path)])

        self.assertEqual(code, 0, err.getvalue())
        data = json.loads(out.getvalue())
        self.assertEqual((data["status"], data["task_id"]), ("diagnosed", failed[0].id))
        self.assertEqual(data["failures"][0]["error"], "expected 5, received 4")
        self.assertEqual((snapshot(worktree), snapshot(repo.path)), before)
        self.assertEqual(orch.tasks.require(failed[0].id).status, TaskStatus.FAILED)  # diagnosis only

    def test_qa_cli_unknown_task(self):
        repo = TempRepo()
        self.addCleanup(repo.cleanup)
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), redirect_stderr(err):
            code = main(["qa", "T-missing", "--simulate", "--cwd", str(repo.path)])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
