"""Tools that exist only inside agent runs."""

from __future__ import annotations

from typing import Any

from ..tool_system.context import ToolContext
from ..tool_system.errors import ToolInputError
from ..tool_system.protocol import ToolResult
from ..tool_system.registry import ToolSpec
from .models import RESULT_STATUSES, AgentResult

_STR_LIST = {"type": "array", "items": {"type": "string"}}


class ReportResultTool:
    """Structured end-of-task report. The orchestrator relies on this, not prose."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ReportResult",
            description=(
                "Report the structured result of your assignment. Call exactly once, as your final action. "
                "Be precise: list every file you changed and the exact test counts you observed."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {"type": "string", "enum": list(RESULT_STATUSES)},
                    "summary": {"type": "string"},
                    "files_changed": _STR_LIST,
                    "tests_run": {"type": "integer"},
                    "tests_passed": {"type": "integer"},
                    "tests_failed": {"type": "integer"},
                    "risks": _STR_LIST,
                    "warnings": _STR_LIST,
                    "next_action": {"type": "string"},
                    "review_verdict": {"type": "string", "enum": ["approve", "request_changes"]},
                },
                "required": ["status", "summary"],
            },
            is_read_only=True,
            max_result_size_chars=2_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        data = {k: v for k, v in tool_input.items()}
        try:
            result = AgentResult(task_id=context.task_id, agent_id=context.agent_id or "", reported_by_agent=True, **data)
        except (TypeError, ValueError) as e:
            raise ToolInputError(f"invalid result: {e}") from e
        context.agent_result = result.to_dict()
        return ToolResult(name="ReportResult", output={"recorded": True, "status": result.status,
                                                       "message": "Result recorded. Stop now."})


class MessageOrchestratorTool:
    """The only communication channel an agent has (auditable, orchestrator-only)."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="MessageOrchestrator",
            description=(
                "Send a structured message to the orchestrator: BLOCKED when you cannot proceed, QUESTION when a "
                "decision is needed, ERROR for unexpected failures. You cannot message other agents."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["BLOCKED", "QUESTION", "ERROR"]},
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
            },
            is_read_only=True,
            max_result_size_chars=2_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        kind = tool_input["type"]
        content = tool_input["content"]
        if not isinstance(content, str) or not content.strip():
            raise ToolInputError("content must be a non-empty string")
        if context.message_sink is None:
            return ToolResult(name="MessageOrchestrator", output={"error": "no orchestrator attached"}, is_error=True,
                              error_kind="execution")
        context.message_sink(kind, content)
        reply = "Recorded. " + ("Stop and report status 'blocked'." if kind == "BLOCKED" else
                                "Continue with your best judgement; the question is logged for the human reviewer.")
        return ToolResult(name="MessageOrchestrator", output={"delivered": True, "message": reply})
