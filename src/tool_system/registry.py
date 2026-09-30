from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from .context import ToolContext
from .permission_handler import PermissionResult
from .protocol import ToolCall, ToolResult
from .schema_validation import validate_json_schema
from ..observability import log_event
from ..security.permissions import ApprovalRequest, Decision, PermissionEngine

_ENGINE = PermissionEngine()


def _request_approval(request: ApprovalRequest, context: ToolContext, tool_check: PermissionResult) -> bool:
    """Ask the caller's approval handler. No handler means no approval."""
    if context.approval_handler is not None:
        return bool(context.approval_handler(request))
    if context.permission_handler is not None:
        message = tool_check.message if tool_check.behavior.value == "ask" and tool_check.message else request.render()
        allowed, _ = context.permission_handler(request.tool, message, tool_check.suggestion)
        return bool(allowed)
    return False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    aliases: tuple[str, ...] = ()
    is_read_only: bool = False
    is_destructive: bool = False
    strict: bool = False
    max_result_size_chars: int = 20_000


class Tool(Protocol):
    def spec(self) -> ToolSpec: ...

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult: ...

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Check if this tool has permission to run.

        Args:
            tool_input: The input arguments for the tool.
            context: The tool execution context.

        Returns:
            PermissionResult indicating allow, deny, or ask.
        """
        return PermissionResult.allow()


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: list[Tool] = []
        self._by_name: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = tool.spec()
        key = spec.name.lower()
        if key in self._by_name:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools.append(tool)
        self._by_name[key] = tool
        for alias in spec.aliases:
            alias_key = alias.lower()
            if alias_key in self._by_name:
                raise ValueError(f"duplicate tool alias: {alias}")
            self._by_name[alias_key] = tool

    def subset(self, names: Iterable[str], *, extra: Iterable[Tool] = ()) -> "ToolRegistry":
        """A registry exposing only ``names`` (plus ``extra`` tools), e.g. per agent profile."""
        selected: list[Tool] = []
        for name in names:
            tool = self.get(name)
            if tool is None:
                raise ValueError(f"unknown tool: {name}")
            if tool not in selected:
                selected.append(tool)
        registry = ToolRegistry(selected)
        for tool in extra:
            registry.register(tool)
        return registry

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools]

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name.lower())

    def dispatch(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                name=call.name,
                output={"error": f"unknown tool: {call.name}"},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )
        spec = tool.spec()
        context.ensure_tool_allowed(spec.name)
        validate_json_schema(call.input, spec.input_schema, root_name=spec.name)

        # Every invocation is evaluated by the permission engine: capabilities
        # and risk of the call, the caller's policy (session or agent profile),
        # explicit rules, and the tool's own check_permissions().
        tool_check = tool.check_permissions(call.input, context) if hasattr(tool, 'check_permissions') else PermissionResult.allow()
        decision = _ENGINE.evaluate(
            spec.name, call.input, context.permission_policy, spec=spec, context=context, tool_check=tool_check,
        )
        if decision.decision is Decision.DENY:
            log_event("permission.denied", tool=spec.name, agent_id=context.agent_id, task_id=context.task_id,
                      capabilities=sorted(c.value for c in decision.assessment.capabilities),
                      risk=decision.assessment.risk.name, reason=decision.message)
            return ToolResult(
                name=spec.name,
                output={"error": f"permission denied: {decision.message}"},
                is_error=True,
                tool_use_id=call.tool_use_id,
                error_kind="permission",
            )
        if decision.decision is Decision.ASK:
            request = decision.approval_request(spec.name, agent_id=context.agent_id, task_id=context.task_id)
            log_event("approval.requested", **request.to_dict())
            approved = _request_approval(request, context, tool_check)
            log_event("approval.approved" if approved else "approval.denied",
                      approval_id=request.id, tool=spec.name, agent_id=context.agent_id, task_id=context.task_id,
                      risk=request.risk, capabilities=list(request.capabilities))
            if not approved:
                return ToolResult(
                    name=spec.name,
                    output={
                        "error": f"permission denied (not approved): {decision.message}",
                        "approval": request.to_dict(),
                    },
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                    error_kind="permission",
                )
            # Approved - proceed with potentially updated input
            if tool_check.updated_input:
                call = ToolCall(
                    name=call.name,
                    input=tool_check.updated_input,
                    tool_use_id=call.tool_use_id,
                )

        result = tool.run(call.input, context)
        if result.tool_use_id is None and call.tool_use_id is not None:
            return ToolResult(
                name=result.name,
                output=result.output,
                is_error=result.is_error,
                tool_use_id=call.tool_use_id,
                content_type=result.content_type,
                error_kind=result.error_kind,
            )
        return result

