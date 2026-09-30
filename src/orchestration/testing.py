"""Run a project's tests and parse the counts (orchestrator-measured, not agent-claimed)."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from ..security.secrets import redact_secrets
from ..tool_system.shell import run_shell_command


@dataclass
class TestRun:
    command: str
    exit_code: int
    run: int
    passed: int
    failed: int
    output_tail: str
    timed_out: bool = False
    worktree: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.failed == 0 and not self.timed_out

    def to_dict(self) -> dict:
        return asdict(self)


def detect_test_command(root: Path) -> Optional[str]:
    root = Path(root)
    py = f'"{sys.executable}"'
    has_py_tests = any(root.glob("tests/test_*.py")) or any(root.glob("test_*.py")) or any(root.glob("tests/**/test_*.py"))
    if has_py_tests:
        if importlib.util.find_spec("pytest") is not None:
            return f"{py} -m pytest -q -p no:cacheprovider"
        return f"{py} -m unittest discover -s tests -t . -v" if (root / "tests").is_dir() else f"{py} -m unittest discover -v"
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
        except Exception:
            scripts = {}
        test = scripts.get("test", "")
        if test and "no test specified" not in test:
            return "npm test --silent"
    return None


_PYTEST = re.compile(r"(\d+) (passed|failed|error|errors)")
_UNITTEST_RAN = re.compile(r"Ran (\d+) tests?")
_UNITTEST_FAIL = re.compile(r"FAILED \((.*?)\)")
_JEST = re.compile(r"Tests:\s+(?:(\d+) failed, )?(?:(\d+) passed, )?(\d+) total")


def parse_test_counts(output: str) -> tuple[int, int, int]:
    """Return (run, passed, failed) from pytest / unittest / jest output."""
    m = _JEST.search(output)
    if m:
        failed, passed, total = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3))
        return total, passed, failed
    ran = _UNITTEST_RAN.search(output)
    if ran:
        total = int(ran.group(1))
        failed = 0
        fm = _UNITTEST_FAIL.search(output)
        if fm:
            failed = sum(int(n) for n in re.findall(r"=(\d+)", fm.group(1)))
        return total, total - failed, failed
    counts = {"passed": 0, "failed": 0}
    for n, kind in _PYTEST.findall(output):
        counts["passed" if kind == "passed" else "failed"] += int(n)
    return counts["passed"] + counts["failed"], counts["passed"], counts["failed"]


def run_tests(root: Path, command: Optional[str] = None, *, timeout_s: int = 900) -> Optional[TestRun]:
    command = command or detect_test_command(root)
    if not command:
        return None
    result = run_shell_command(command, cwd=root, timeout_s=timeout_s)
    output = result.stdout + "\n" + result.stderr
    run, passed, failed = parse_test_counts(output)
    if result.exit_code != 0 and failed == 0:
        failed = max(1, run - passed)  # e.g. collection/import errors
    return TestRun(command=command, exit_code=result.exit_code, run=run, passed=passed, failed=failed,
                   output_tail=redact_secrets(output[-3000:]), timed_out=result.timed_out, worktree=str(root))
