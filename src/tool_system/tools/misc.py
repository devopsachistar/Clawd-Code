from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec
from .bash import ShellCommandTool


class SendMessageTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="SendMessage",
            description="Send a message to another recipient (best-effort, local only).",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "to": {"type": "string"},
                    "summary": {"type": "string"},
                    "message": {},
                },
                "required": ["to", "message"],
            },
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        to = tool_input.get("to")
        message = tool_input.get("message")
        summary = tool_input.get("summary")
        if not isinstance(to, str) or not to.strip():
            raise ToolInputError("to must be a non-empty string")
        if summary is not None and not isinstance(summary, str):
            raise ToolInputError("summary must be a string when provided")
        context.outbox.append({"tool": "SendMessage", "to": to, "summary": summary, "message": message})
        return ToolResult(name="SendMessage", output={"success": True, "message": f"Message queued for {to}"})


class RemoteTriggerTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="RemoteTrigger",
            description="Trigger a remote action (not implemented).",
            input_schema={"type": "object", "additionalProperties": True},
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(name="RemoteTrigger", output={"error": "RemoteTrigger is not implemented"}, is_error=True)


class PowerShellTool(ShellCommandTool):
    """Run PowerShell (pwsh or Windows PowerShell) with the shared shell safety checks."""

    tool_name = "PowerShell"
    preferred_shell = "powershell"

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="PowerShell",
            description=(
                "Run a PowerShell command (pwsh or Windows PowerShell) in the project. "
                "Destructive commands require user approval; catastrophic ones are refused."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                },
                "required": ["command"],
            },
            is_destructive=True,
            max_result_size_chars=200_000,
            strict=True,
        )


class NotebookEditTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="NotebookEdit",
            description="Edit a Jupyter notebook (not implemented).",
            input_schema={"type": "object", "additionalProperties": True},
            is_destructive=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(name="NotebookEdit", output={"error": "NotebookEdit is not implemented"}, is_error=True)


class REPLTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="REPL",
            description="Interact with the REPL UI (not implemented).",
            input_schema={"type": "object", "additionalProperties": True},
            is_read_only=True,
            max_result_size_chars=100_000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(name="REPL", output={"error": "REPL tool is not implemented"}, is_error=True)


class TestingPermissionTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TestingPermission",
            description="Test-only tool (always succeeds).",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
            max_result_size_chars=100_000,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        return ToolResult(name="TestingPermission", output="TestingPermission executed successfully", content_type="text")

