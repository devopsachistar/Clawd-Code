"""Real git worktrees: create -> modify -> commit -> inspect -> remove, isolation and locking."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from gitrepo import TempRepo, git
from src.gitops.worktrees import GitError, MergeConflict, WorktreeBusy, WorktreeManager
from src.orchestration.state import EventRecorder, FileLock
from src.tool_system.context import ToolContext
from src.tool_system.tools.worktree import EnterWorktreeTool, ExitWorktreeTool


class WorktreeLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = TempRepo()
        self.events = EventRecorder(None)
        self.recorded: list[dict] = []
        self.events.subscribe(self.recorded.append)
        self.m = WorktreeManager(self.repo.path, base_dir=self.repo.worktrees, events=self.events)

    def tearDown(self) -> None:
        self.repo.cleanup()

    def test_create_modify_commit_inspect_remove(self):
        wt = self.m.create("task-123", "frontend")
        path = Path(wt.path)
        self.assertEqual(wt.branch, "increscent/task-123/frontend")
        self.assertEqual(path.parent, self.repo.worktrees)
        self.assertTrue(path.name.endswith("task-123-frontend"))
        self.assertIn(path.as_posix().lower(), git(self.repo.path, "worktree", "list").replace("\\", "/").lower())
        self.assertEqual(git(path, "rev-parse", "--abbrev-ref", "HEAD"), "increscent/task-123/frontend")

        (path / "ui.py").write_text("print('ui')\n", encoding="utf-8")
        status = self.m.status(wt.name)
        self.assertFalse(status["clean"])
        self.assertEqual(status["files"], [{"status": "??", "path": "ui.py"}])

        sha = self.m.commit(wt.name, "feat: ui")
        self.assertEqual(git(path, "rev-parse", "HEAD"), sha)
        self.assertTrue(self.m.status(wt.name)["clean"])
        self.assertIn("agent+frontend@increscent.local", git(path, "log", "-1", "--format=%ae"))
        self.assertIn("ui.py", self.m.diff(wt.name, stat=True))
        self.assertIn("+print('ui')", self.m.diff(wt.name))
        self.assertIsNone(self.m.commit(wt.name, "nothing"))  # clean -> no commit

        self.m.remove(wt.name, delete_branch=True)
        self.assertFalse(path.exists())
        self.assertNotIn("increscent/task-123/frontend", git(self.repo.path, "branch"))
        self.assertEqual(self.repo.head(), self.repo.initial)  # main untouched
        events = [e["event"] for e in self.recorded]
        for expected in ("worktree.created", "commit.created", "worktree.removed"):
            self.assertIn(expected, events)

    def test_worktrees_are_isolated(self):
        a = self.m.create("t1", "backend")
        b = self.m.create("t1", "frontend")
        (Path(a.path) / "only_in_a.txt").write_text("a", encoding="utf-8")
        self.m.commit(a.name, "a")
        self.assertFalse((Path(b.path) / "only_in_a.txt").exists())
        self.assertFalse((self.repo.path / "only_in_a.txt").exists())
        self.assertNotEqual(a.branch, b.branch)

    def test_list_marks_managed_and_persists_across_instances(self):
        wt = self.m.create("t9", "qa")
        fresh = WorktreeManager(self.repo.path, base_dir=self.repo.worktrees)
        managed = fresh.list(managed_only=True)
        self.assertEqual([(w.name, w.task_id, w.role) for w in managed], [(wt.name, "t9", "qa")])
        self.assertEqual(len(fresh.list()), 2)  # main checkout + agent worktree

    def test_create_is_idempotent_for_retries(self):
        first = self.m.create("t2", "dev")
        (Path(first.path) / "partial.txt").write_text("work in progress", encoding="utf-8")
        again = self.m.create("t2", "dev")
        self.assertEqual(first.path, again.path)
        self.assertTrue((Path(again.path) / "partial.txt").exists())  # preserved

    def test_base_ref(self):
        a = self.m.create("t3", "backend")
        (Path(a.path) / "api.py").write_text("x = 1\n", encoding="utf-8")
        self.m.commit(a.name, "api")
        b = self.m.create("t4", "frontend", base_ref=a.branch)
        self.assertTrue((Path(b.path) / "api.py").exists())

    def test_lock_prevents_concurrent_use(self):
        wt = self.m.create("t5", "dev")
        lock = self.m.acquire(wt.name, "A-1")
        with self.assertRaises(WorktreeBusy):
            self.m.acquire(wt.name, "A-2")
        self.assertEqual(self.m.get(wt.name).locked_by, "A-1")
        with self.assertRaises(WorktreeBusy):
            self.m.remove(wt.name, force=True)
        lock.release()
        self.m.acquire(wt.name, "A-2").release()

    def test_stale_lock_from_dead_process_is_reclaimed(self):
        wt = self.m.create("t6", "dev")
        lock_path = self.m.locks_dir / f"{wt.name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({"owner": "ghost", "pid": 999_999_999}), encoding="utf-8")
        self.assertIsNone(self.m.get(wt.name).locked_by)
        self.m.acquire(wt.name, "A-3").release()

    def test_remove_refuses_dirty_unless_forced(self):
        wt = self.m.create("t7", "dev")
        (Path(wt.path) / "wip.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(GitError):
            self.m.remove(wt.name)
        self.m.remove(wt.name, force=True)
        self.assertIn("increscent/t7/dev", git(self.repo.path, "branch"))  # branch kept by default

    def test_refuses_to_remove_unmanaged_worktree(self):
        with self.assertRaises(GitError):
            self.m.remove("repo", force=True)

    def test_commit_refuses_unignored_secret_files(self):
        wt = self.m.create("t8", "dev")
        (Path(wt.path) / "config.env").write_text("API_KEY=abcdefghijklmnop", encoding="utf-8")
        with self.assertRaises(GitError):
            self.m.commit(wt.name, "oops")
        (Path(wt.path) / "config.env").unlink()
        (Path(wt.path) / ".env").write_text("API_KEY=abcdefghijklmnop", encoding="utf-8")  # git-ignored
        (Path(wt.path) / "app.py").write_text("x = 1\n", encoding="utf-8")
        self.m.commit(wt.name, "ok")
        self.assertNotIn(".env", git(wt.path, "show", "--name-only", "--format="))

    def test_merge_branch_and_conflicts(self):
        a = self.m.create("m1", "a")
        b = self.m.create("m1", "b")
        (Path(a.path) / "README.md").write_text("# from a\n", encoding="utf-8")
        self.m.commit(a.name, "a")
        (Path(b.path) / "README.md").write_text("# from b\n", encoding="utf-8")
        self.m.commit(b.name, "b")
        integration = self.m.create("m1", "integration")
        self.m.merge_branch(integration.name, a.branch)
        with self.assertRaises(MergeConflict) as ctx:
            self.m.merge_branch(integration.name, b.branch)
        self.assertIn("README.md", ctx.exception.files)
        self.assertTrue(self.m.status(integration.name)["clean"])  # merge aborted cleanly
        self.assertEqual(self.repo.head(), self.repo.initial)

    def test_requires_git_repo_with_commits(self):
        with self.assertRaises(GitError):
            WorktreeManager(self.repo.base)  # not a repository
        empty = self.repo.base / "empty"
        empty.mkdir()
        git(empty, "init", "-q")
        with self.assertRaises(GitError):
            WorktreeManager(empty, base_dir=self.repo.worktrees).create("t", "r")


class FileLockTests(unittest.TestCase):
    def test_exclusive_and_reentrant(self):
        repo = TempRepo()
        try:
            path = repo.base / "x.lock"
            a = FileLock(path, "owner-a")
            self.assertTrue(a.acquire())
            self.assertTrue(FileLock(path, "owner-a").acquire())  # same owner, same process
            self.assertFalse(FileLock(path, "owner-b").acquire())
            a.release()
            self.assertTrue(FileLock(path, "owner-b").acquire())
            self.assertEqual(json.loads(path.read_text())["pid"], os.getpid())
        finally:
            repo.cleanup()


class WorktreeToolTests(unittest.TestCase):
    def test_enter_worktree_creates_real_git_worktree(self):
        repo = TempRepo()
        try:
            ctx = ToolContext(workspace_root=repo.path)
            out = EnterWorktreeTool().run({"name": "spike"}, ctx).output
            self.assertTrue(out["isGitWorktree"])
            self.assertTrue(out["worktreeBranch"].startswith("increscent/session-"))
            self.assertEqual(ctx.cwd, Path(out["worktreePath"]))
            self.assertEqual(git(out["worktreePath"], "rev-parse", "--abbrev-ref", "HEAD"), out["worktreeBranch"])
            ctx.ensure_allowed_path(Path(out["worktreePath"]) / "file.txt")  # sandbox includes the worktree
            ExitWorktreeTool().run({}, ctx)
            self.assertEqual(ctx.cwd, repo.path)
            self.assertTrue(Path(out["worktreePath"]).exists())  # preserved
            WorktreeManager(repo.path).remove(Path(out["worktreePath"]).name, force=True, delete_branch=True)
        finally:
            repo.cleanup()


if __name__ == "__main__":
    unittest.main()
