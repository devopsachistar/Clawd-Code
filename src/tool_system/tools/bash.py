from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..registry import ToolSpec
from ..safety import classify_command
from ..shell import IS_WINDOWS, ShellSpec, detect_shell, run_shell_command


def _truncate(s: str, limit: int = 20000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n\n... [truncated] ..."


_SHELL_OPERATORS = ("&&", "||", ";", "|", "&", ">", "<", "`", "$(", "\n")


def _try_extract_cd(command: str) -> Path | None:
    """Return the target of a *bare* ``cd <dir>`` command, else None.

    Commands such as ``cd sub && make`` are not treated as a directory change;
    they run in full in the shell. The old behaviour swallowed everything after
    the ``cd``.
    """
    stripped = command.strip()
    if not stripped.startswith("cd ") or any(op in stripped for op in _SHELL_OPERATORS):
        return None
    try:
        parts = shlex.split(stripped, posix=not IS_WINDOWS)
    except ValueError:
        return None
    if len(parts) == 2 and parts[0] == "cd":
        return Path(parts[1].strip("\"'"))
    return None


def assess_command_permission(command: Any, context: ToolContext) -> PermissionResult:
    """Shared permission check for shell tools (Bash, PowerShell)."""
    if not isinstance(command, str) or not command.strip():
        return PermissionResult.allow()  # input validation happens in run()
    assessment = classify_command(command, cwd=context.cwd or context.workspace_root)
    if assessment.is_blocked:
        return PermissionResult.deny(f"Refusing to run blocked command: {assessment.message()}")
    if assessment.needs_approval:
        return PermissionResult.ask(
            message=f"Command requires approval ({assessment.message()}): {command.strip()[:300]}",
        )
    return PermissionResult.allow()


class ShellCommandTool:
    """Shared implementation for the Bash and PowerShell tools."""

    tool_name = "Bash"
    preferred_shell: str | None = None

    def _shell(self) -> ShellSpec:
        return detect_shell(self.preferred_shell)  # type: ignore[arg-type]

    def check_permissions(self, tool_input: dict[str, Any], context: ToolContext) -> PermissionResult:
        return assess_command_permission(tool_input.get("command"), context)

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        command = tool_input["command"]
        if not isinstance(command, str) or not command.strip():
            raise ToolInputError("command must be a non-empty string")
        if "\x00" in command:
            raise ToolInputError("command contains NUL byte")

        # Defence in depth: blocked commands are refused even if a caller skips
        # the registry's permission check.
        assessment = classify_command(command, cwd=context.cwd or context.workspace_root)
        if assessment.is_blocked:
            raise ToolPermissionError(f"refusing to run potentially dangerous command: {assessment.message()}")

        explicit_cwd = tool_input.get("cwd")
        if explicit_cwd is not None:
            if not isinstance(explicit_cwd, str) or not Path(explicit_cwd).expanduser().is_absolute():
                raise ToolInputError("cwd must be an absolute path when provided")
            cwd = context.ensure_allowed_path(explicit_cwd)
            if not cwd.is_dir():
                raise ToolInputError(f"cwd is not a directory: {cwd}")
        else:
            cwd = context.cwd or context.workspace_root

        cd_target = _try_extract_cd(command)
        if cd_target is not None:
            target = cd_target.expanduser()
            next_dir = target.resolve() if target.is_absolute() else (cwd / target).resolve()
            next_dir = context.ensure_allowed_path(next_dir)
            if not next_dir.exists() or not next_dir.is_dir():
                return ToolResult(
                    name=self.tool_name,
                    output={"error": f"directory does not exist: {next_dir}"},
                    is_error=True,
                    error_kind="input",
                )
            context.cwd = next_dir
            return ToolResult(name=self.tool_name, output={"cwd": str(context.cwd), "stdout": "", "stderr": "", "exit_code": 0})

        timeout_s = tool_input.get("timeout_s", 60)
        if not isinstance(timeout_s, int) or timeout_s < 1 or timeout_s > 600:
            raise ToolInputError("timeout_s must be an integer between 1 and 600")

        try:
            shell = self._shell()
        except RuntimeError as e:
            return ToolResult(name=self.tool_name, output={"error": str(e)}, is_error=True, error_kind="execution")

        completed = run_shell_command(command, cwd=cwd, timeout_s=timeout_s, shell=shell)
        output: dict[str, Any] = {
            "cwd": str(cwd),
            "shell": completed.shell,
            "exit_code": completed.exit_code,
            "stdout": _truncate(completed.stdout),
            "stderr": _truncate(completed.stderr),
        }
        if completed.timed_out:
            output["timed_out"] = True
            output["error"] = f"command timed out after {timeout_s}s (process tree terminated)"
        return ToolResult(
            name=self.tool_name,
            output=output,
            is_error=completed.exit_code != 0 or completed.timed_out,
            error_kind="execution" if (completed.exit_code != 0 or completed.timed_out) else None,
        )


class BashTool(ShellCommandTool):
    tool_name = "Bash"
    preferred_shell = None  # bash on POSIX; Git Bash, else PowerShell, else cmd on Windows

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Bash",
            description=(
                "Execute a shell command in the project. Uses bash on Linux/macOS and Git Bash on "
                "Windows (falls back to PowerShell/cmd if Git Bash is unavailable; the result's "
                "'shell' field says which ran). Destructive or outward-facing commands (git push, "
                "git commit, recursive deletes, ...) require user approval; catastrophic commands "
                "(sudo, disk formatting, deleting / or home) are always refused."
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
            max_result_size_chars=50_000,
        )
