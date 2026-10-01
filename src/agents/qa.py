"""QA diagnosis: failed developer task -> structured, evidence-grounded diagnosis (no repair).

    TaskSpec + DeveloperResult (measured files, test command, test output, error, workspace)
        |  build_provider_request() (provider-neutral; same path as the planner and agent loop)
        v
    LLM (any configured BaseProvider), one request, NO tools: it cannot touch the workspace
        |  JSON response
        v
    parse_diagnosis() (structure)  +  grounding (evidence must be quoted from the real failure output)
        v
    QADiagnosis(status=diagnosed|inconclusive, failures, recommended_fixes, confidence)

The diagnoser never fabricates. Insufficient input, provider errors, invalid
responses and evidence that cannot be found in the measured output all lead
to ``inconclusive``, or to the ungrounded failure being dropped, with the
reason in ``warnings``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from ..agent.message_format import build_provider_request, resolve_wire_format
from ..observability import log_event
from ..orchestration.ai_planner import PlanningError, extract_json
from ..orchestration.planner import TaskSpec
from ..providers.base import BaseProvider
from ..security.secrets import redact_structure
from .developer import DeveloperResult

STATUSES = ("diagnosed", "inconclusive")
CONFIDENCE = ("low", "medium", "high")
_FAILURE_KEYS = ("test", "error", "likely_cause", "evidence")


class DiagnosisError(Exception):
    """The model's diagnosis is structurally invalid. ``issues`` lists every problem found."""

    def __init__(self, message: str, issues: tuple[str, ...] = ()) -> None:
        super().__init__(message + ("".join(f"\n  - {i}" for i in issues) if issues else ""))
        self.issues = issues


@dataclass(frozen=True)
class QAFailure:
    test: str
    error: str
    likely_cause: str
    evidence: str  # quoted verbatim from the test output / error


@dataclass
class QADiagnosis:
    status: str                                   # "diagnosed" | "inconclusive"
    summary: str
    failures: list[QAFailure] = field(default_factory=list)
    recommended_fixes: list[str] = field(default_factory=list)
    confidence: str = "low"                       # "low" | "medium" | "high"
    task_id: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- measured failure facts --------------------------------------------------------------------

_PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) (?P<test>\S+?)(?: - (?P<error>.+))?$", re.M)
_UNITTEST_FAILED = re.compile(r"^(?:FAIL|ERROR): (?P<test>\S+)(?: \((?P<where>[^)]+)\))?", re.M)


def failed_tests(output: str) -> list[tuple[str, str]]:
    """(test id, error) for every failing test named in pytest / unittest output."""
    found: dict[str, str] = {}
    output = (output or "").replace("\r\n", "\n")
    for m in _PYTEST_FAILED.finditer(output or ""):
        found.setdefault(m.group("test"), (m.group("error") or "").strip())
    for m in _UNITTEST_FAILED.finditer(output or ""):
        name = f"{m.group('where')}.{m.group('test')}" if m.group("where") else m.group("test")
        found.setdefault(name, "")
    return list(found.items())


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _grounded(quote: str, sources: str) -> bool:
    quote = _normalize(quote)
    return bool(quote) and quote in sources


# -- validation ----------------------------------------------------------------------------------

def parse_diagnosis(data: Any) -> QADiagnosis:
    """Validate the model's JSON against the diagnosis schema (structure only; grounding is separate)."""
    issues: list[str] = []
    if not isinstance(data, dict):
        raise DiagnosisError("the diagnosis must be a JSON object")
    status = data.get("status")
    if status not in STATUSES:
        issues.append(f"status must be one of {', '.join(STATUSES)}")
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        issues.append("summary must be a non-empty string")
    confidence = data.get("confidence")
    if confidence not in CONFIDENCE:
        issues.append(f"confidence must be one of {', '.join(CONFIDENCE)}")
    failures: list[QAFailure] = []
    raw_failures = data.get("failures", [])
    if not isinstance(raw_failures, list):
        issues.append("failures must be a list")
        raw_failures = []
    for i, item in enumerate(raw_failures):
        if not isinstance(item, dict):
            issues.append(f"failures[{i}] must be an object")
            continue
        bad = [k for k in _FAILURE_KEYS if not isinstance(item.get(k), str) or not item[k].strip()]
        if bad:
            issues.append(f"failures[{i}] needs non-empty strings: {', '.join(bad)}")
            continue
        failures.append(QAFailure(**{k: item[k].strip() for k in _FAILURE_KEYS}))
    fixes = data.get("recommended_fixes", [])
    if not isinstance(fixes, list) or not all(isinstance(f, str) and f.strip() for f in fixes):
        issues.append("recommended_fixes must be a list of non-empty strings")
        fixes = []
    if status == "diagnosed" and not failures:
        issues.append("a 'diagnosed' result needs at least one failure")
    if issues:
        raise DiagnosisError("the diagnosis is invalid", tuple(issues))
    return QADiagnosis(status=status, summary=summary.strip(), failures=failures,
                       recommended_fixes=[f.strip() for f in fixes], confidence=confidence)


# -- diagnoser -------------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are the QA agent of a software factory. A developer task failed verification.
Diagnose WHY it failed using ONLY the information given. You cannot run tools and must not propose that
you already changed anything. Do not repair anything.

Respond with ONE JSON object and nothing else:
{
  "status": "diagnosed" | "inconclusive",
  "summary": string,
  "failures": [{"test": string, "error": string, "likely_cause": string, "evidence": string}],
  "recommended_fixes": [string, ...],
  "confidence": "low" | "medium" | "high"
}

Rules:
- "evidence" MUST be an exact quote copied from the test output or error below. Never invent evidence.
- "test" MUST be the failing test's name as it appears in the output.
- If the information is insufficient to explain the failure, answer "inconclusive" with no failures.
"""


class QADiagnoser:
    """Failed TaskSpec + DeveloperResult -> :class:`QADiagnosis`. Read-only: the model gets no tools."""

    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider

    def build_request(self, task: TaskSpec, result: DeveloperResult) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        tests = (f"command: {result.test_command}\nrun {result.tests_run}, passed {result.tests_passed}, "
                 f"failed {result.tests_failed}" if result.test_command else "no tests were run")
        content = "\n".join([
            f"# Failed task: {task.title}",
            f"Workspace: {result.workspace}",
            "",
            "## Task",
            task.full_description(),
            "",
            "## Files changed",
            "\n".join(f"- {f}" for f in result.files_changed) or "(none)",
            "",
            "## Test results",
            tests,
            "",
            "## Test output",
            "```",
            result.test_output.strip() or "(none)",
            "```",
            "",
            "## Error",
            result.error or "(none)",
        ])
        api_messages, kwargs = build_provider_request([{"role": "user", "content": content}],
                                                      system_prompt=_SYSTEM_PROMPT,
                                                      wire_format=resolve_wire_format(self.provider))
        return redact_structure(api_messages), redact_structure(kwargs)  # same secret boundary as agents

    def diagnose(self, task: TaskSpec, result: DeveloperResult) -> QADiagnosis:
        def inconclusive(summary: str, *warnings: str) -> QADiagnosis:
            log_event("qa.inconclusive", task_id=task.key, reason=summary)
            return QADiagnosis(status="inconclusive", summary=summary, confidence="low", task_id=task.key,
                               warnings=list(warnings))

        if result.status == "completed":
            return inconclusive("The task did not fail; there is nothing to diagnose.")
        if not (result.test_output or "").strip() and not (result.error or "").strip():
            return inconclusive("No test output or error was recorded, so the failure cannot be diagnosed.")

        api_messages, kwargs = self.build_request(task, result)
        log_event("qa.requested", task_id=task.key, provider=type(self.provider).__name__)
        try:
            response = self.provider.chat(api_messages, **kwargs)
        except Exception as e:
            return inconclusive("The QA model provider failed.", f"provider error: {type(e).__name__}: {e}")
        try:
            diagnosis = parse_diagnosis(extract_json(getattr(response, "content", "") or ""))
        except (PlanningError, DiagnosisError) as e:
            return inconclusive("The QA model returned an invalid diagnosis.", str(e))
        diagnosis.task_id = task.key
        return self._ground(diagnosis, result)

    @staticmethod
    def _ground(diagnosis: QADiagnosis, result: DeveloperResult) -> QADiagnosis:
        """Keep only failures whose test name and evidence appear in the measured output."""
        sources = _normalize(f"{result.test_output}\n{result.error or ''}")
        kept: list[QAFailure] = []
        for f in diagnosis.failures:
            if not _grounded(f.evidence, sources):
                diagnosis.warnings.append(f"dropped failure {f.test!r}: evidence not found in the test output")
            elif not _grounded(f.test.split("::")[-1], sources):
                diagnosis.warnings.append(f"dropped failure {f.test!r}: test not found in the test output")
            else:
                kept.append(f)
        dropped = len(diagnosis.failures) - len(kept)
        diagnosis.failures = kept
        measured = failed_tests(result.test_output)
        missing = [t for t, _ in measured if not any(f.test.split("::")[-1] in t for f in kept)]
        if missing and kept:
            diagnosis.warnings.append("not diagnosed: " + ", ".join(missing))
        if diagnosis.status == "diagnosed" and not kept:
            diagnosis.status, diagnosis.recommended_fixes = "inconclusive", []
            diagnosis.summary = "No diagnosed failure is supported by the measured output."
        if dropped or (missing and kept):
            diagnosis.confidence = "low" if dropped else min(diagnosis.confidence, "medium", key=CONFIDENCE.index)
        if diagnosis.status == "inconclusive":
            diagnosis.confidence = "low"
        log_event("qa.diagnosed", task_id=diagnosis.task_id, status=diagnosis.status, failures=len(kept),
                  dropped=dropped)
        return diagnosis


__all__ = ["DiagnosisError", "QADiagnoser", "QADiagnosis", "QAFailure", "failed_tests", "parse_diagnosis"]
