"""Guards against tests that silently never run.

A ``async def test_*`` method on a plain ``unittest.TestCase`` is never
awaited: it creates a coroutine, discards it, and counts as a pass. This test
fails if any test module reintroduces that pattern.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ASYNC_CAPABLE_BASES = {"IsolatedAsyncioTestCase"}


def _base_names(node: ast.ClassDef) -> set[str]:
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


class TestSuiteHygiene(unittest.TestCase):
    def test_async_tests_run_on_async_capable_test_cases(self):
        offenders: list[str] = []
        for path in sorted(TESTS_DIR.glob("test_*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if not (_base_names(node) & {"TestCase", "IsolatedAsyncioTestCase"}):
                    continue
                if _base_names(node) & ASYNC_CAPABLE_BASES:
                    continue
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef) and item.name.startswith("test"):
                        offenders.append(f"{path.name}::{node.name}::{item.name}")
        self.assertEqual(offenders, [], "async test methods on unittest.TestCase are never awaited")


if __name__ == "__main__":
    unittest.main()
