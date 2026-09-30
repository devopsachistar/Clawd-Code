"""CLI: team / agents / agent show / tasks / task show|pause|resume / worktrees / review."""

from __future__ import annotations

import io
import json
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout

from gitrepo import TempRepo, git
from src.cli import main


class CliPhase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = TempRepo()
        cls.code, cls.out, cls.err = cls.cli("team", "--simulate", "--cwd", str(cls.repo.path),
                                             "Create a simple task management application")
        cls.graph_id = re.search(r"graph: (G-[0-9a-f]{8})", cls.err).group(1)

    @classmethod
    def tearDownClass(cls) -> None:
        from src.gitops.worktrees import WorktreeManager

        manager = WorktreeManager(cls.repo.path)
        for w in manager.list(managed_only=True):
            manager.remove(w.name, force=True)
        cls.repo.cleanup()

    @staticmethod
    def cli(*args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(args))
        return code, out.getvalue(), err.getvalue()

    def run_cli(self, *args: str) -> tuple[int, str, str]:
        return self.cli(*args, "--cwd", str(self.repo.path))

    def test_team_demo_runs_to_review_and_stops_before_merging(self):
        self.assertEqual(self.code, 0, self.err[-2000:])
        self.assertIn("SIMULATION", self.err)
        self.assertIn(f"Graph {self.graph_id}: REVIEW", self.out)
        self.assertIn("Nothing was merged", self.out)
        for event in ("task.started", "worktree.created", "commit.created", "tests.run", "review.requested"):
            self.assertIn(event, self.err)
        self.assertEqual(self.repo.head(), self.repo.initial)

    def test_tasks_and_task_show(self):
        code, out, _ = self.run_cli("tasks")
        self.assertEqual(code, 0)
        for title in ("Design the architecture", "Implement the user interface", "Review the integrated changes"):
            self.assertIn(title, out)
        task_id = re.findall(r"T-[0-9a-f]{8}", out)[1]
        code, out, _ = self.run_cli("task", "show", task_id)
        data = json.loads(out)
        self.assertEqual(data["id"], task_id)
        self.assertEqual(data["status"], "COMPLETED")
        self.assertIn("summary", data["result"])

    def test_agents_and_agent_show(self):
        code, out, _ = self.run_cli("agents")
        self.assertEqual(code, 0)
        agent_ids = re.findall(r"A-[0-9a-f]{8}", out)
        self.assertEqual(len(agent_ids), 6)
        code, out, _ = self.run_cli("agent", "show", agent_ids[0])
        data = json.loads(out)
        self.assertEqual(data["status"], "COMPLETED")
        self.assertIn("permissions", data)
        self.assertEqual(self.run_cli("agent", "show", "A-00000000")[0], 2)

    def test_worktrees(self):
        code, out, _ = self.run_cli("worktrees")
        self.assertEqual(code, 0)
        self.assertEqual(len(re.findall(r"increscent/T-[0-9a-f]{8}/", out)), 6)

    def test_review_shows_diffstats_without_merging(self):
        code, out, _ = self.run_cli("review", self.graph_id)
        self.assertEqual(code, 0)
        self.assertIn("taskapp/cli.py", out)
        self.assertIn("verdict: approve", out)
        self.assertEqual(self.repo.head(), self.repo.initial)

    def test_task_pause_resume_errors(self):
        code, _, _ = self.run_cli("task", "pause", "T-00000000")
        self.assertEqual(code, 2)
        code, out, _ = self.run_cli("tasks")
        completed = re.findall(r"T-[0-9a-f]{8}", out)[1]
        self.assertEqual(self.run_cli("task", "retry", completed)[0], 2)  # only FAILED tasks can be retried

    def test_zz_merge_requires_approval_then_integrates(self):
        code, out, _ = self.run_cli("review", self.graph_id, "--merge")
        self.assertEqual(code, 3)  # no approval (non-interactive) -> permission rejection
        self.assertIn("Nothing was merged", out)
        code, out, _ = self.run_cli("review", self.graph_id, "--merge", "--approve")
        self.assertEqual(code, 0, out)
        self.assertIn(f"increscent/{self.graph_id}/integration", out)
        files = git(self.repo.path, "ls-tree", "-r", "--name-only", f"increscent/{self.graph_id}/integration").splitlines()
        self.assertIn("taskapp/service.py", files)
        self.assertEqual(self.repo.head(), self.repo.initial)


class CliErrorTests(unittest.TestCase):
    def test_team_requires_git_repository(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                code = main(["team", "--simulate", "--cwd", d, "build something"])
        self.assertEqual(code, 2)
        self.assertIn("git repository", out.getvalue())

    def test_team_without_key_is_config_error(self):
        repo = TempRepo()
        try:
            with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
                code = main(["team", "--provider", "openai", "--cwd", str(repo.path), "build"])
            self.assertEqual(code, 2)
            self.assertIn("OPENAI_API_KEY", out.getvalue())
        finally:
            repo.cleanup()

    def test_empty_state_listings(self):
        repo = TempRepo()
        try:
            for command in (["tasks"], ["agents"], ["worktrees"]):
                with redirect_stdout(io.StringIO()) as out:
                    code = main([*command, "--cwd", str(repo.path)])
                self.assertEqual(code, 0)
                self.assertIn("No ", out.getvalue())
        finally:
            repo.cleanup()


if __name__ == "__main__":
    unittest.main()
