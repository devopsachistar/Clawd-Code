"""Project / git root detection."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.project import detect_project, is_unsafe_project_root


class ProjectDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_git_repository_root(self):
        repo = self.base / "shishu-mandir"
        (repo / "src" / "pages").mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        info = detect_project(repo / "src" / "pages")
        self.assertEqual(info.project_root, repo)
        self.assertEqual(info.git_root, repo)
        self.assertEqual(info.cwd, repo / "src" / "pages")
        self.assertEqual(info.detection, "git")
        self.assertTrue(info.is_git_repo)
        self.assertFalse(info.is_unsafe_root)
        self.assertIn(f"Project root: {repo}", "\n".join(info.describe()))
        self.assertIn(f"Git root: {repo}", "\n".join(info.describe()))

    def test_git_worktree_file_is_recognised(self):
        repo = self.base / "wt"
        repo.mkdir()
        (repo / ".git").write_text("gitdir: /somewhere/.git/worktrees/wt\n", encoding="utf-8")
        self.assertEqual(detect_project(repo).git_root, repo)

    def test_nearest_git_root_wins(self):
        outer = self.base / "outer"
        inner = outer / "packages" / "inner"
        inner.mkdir(parents=True)
        (outer / ".git").mkdir()
        (inner / ".git").mkdir()
        self.assertEqual(detect_project(inner).git_root, inner)

    def test_marker_based_project_without_git(self):
        proj = self.base / "app"
        (proj / "lib").mkdir(parents=True)
        (proj / "package.json").write_text("{}", encoding="utf-8")
        info = detect_project(proj / "lib")
        self.assertEqual(info.project_root, proj)
        self.assertIsNone(info.git_root)
        self.assertEqual(info.detection, "marker")
        self.assertIn("package.json", info.markers)

    def test_plain_directory_falls_back_to_cwd(self):
        plain = self.base / "plain" / "dir"
        plain.mkdir(parents=True)
        # Pin the home boundary: the real temp dir may live under a home that
        # contains markers such as ~/.clawd, which must never be picked up.
        with patch("src.project.Path.home", return_value=self.base):
            info = detect_project(plain)
        self.assertEqual(info.project_root, plain)
        self.assertEqual(info.detection, "cwd")

    def test_file_path_uses_parent(self):
        (self.base / "x").mkdir()
        f = self.base / "x" / "a.txt"
        f.write_text("", encoding="utf-8")
        self.assertEqual(detect_project(f).cwd, self.base / "x")

    def test_home_directory_is_never_auto_detected_from_below(self):
        home = self.base / "home"
        work = home / "Downloads" / "loose-files"
        work.mkdir(parents=True)
        (home / ".git").mkdir()          # dotfiles repo in $HOME
        (home / "package.json").write_text("{}", encoding="utf-8")
        with patch("src.project.Path.home", return_value=home):
            info = detect_project(work)
        self.assertEqual(info.project_root, work)
        self.assertIsNone(info.git_root)
        self.assertFalse(info.is_unsafe_root)

    def test_starting_in_home_is_flagged_unsafe(self):
        home = self.base / "home"
        home.mkdir()
        with patch("src.project.Path.home", return_value=home):
            info = detect_project(home)
            self.assertTrue(info.is_unsafe_root)
            self.assertTrue(is_unsafe_project_root(home))

    def test_filesystem_root_is_unsafe(self):
        root = Path(self.base.anchor)
        self.assertTrue(is_unsafe_project_root(root))

    def test_this_repository(self):
        repo_root = Path(__file__).resolve().parent.parent
        info = detect_project(repo_root / "src" / "agent")
        self.assertEqual(info.project_root, repo_root)
        self.assertEqual(info.git_root, repo_root)


if __name__ == "__main__":
    unittest.main()
