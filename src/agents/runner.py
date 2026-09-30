"""Agent execution lifecycle.

``CREATED -> RUNNING (<-> WAITING during approvals) -> COMPLETED | FAILED | CANCELLED``

An agent run is a real, isolated LLM session:

* its own conversation and system prompt (profile instructions + worktree context)
* its own provider/model
* only the tools its profile allows (plus ReportResult / MessageOrchestrator)
* the profile's permission policy, evaluated on every tool call
* its own working directory (a git worktree), which is its path sandbox
* hard budgets: iterations, tool calls, wall clock, tokens

It always ends with a structured :class:`AgentResult`. An agent that stops
without calling ``ReportResult`` has failed. Its prose is not trusted as a
result.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Optional

from ..agent.conversation import Conversation
from ..agent.session import Session
from ..context_system import build_context_prompt
from ..providers.base import BaseProvider
from ..security.permissions import ApprovalRequest, PermissionPolicy
from ..tool_system.agent_loop import (
    STOP_CANCELLED,
    STOP_COMPLETED,
    LoopBudget,
    ToolEvent,
    run_agent_loop,
    summarize_tool_use,
)
from ..tool_system.context import ToolContext
from ..tool_system.defaults import build_default_registry
from ..tool_system.permissions import ToolPermissionContext
from ..tool_system.registry import ToolRegistry
from .messages import ORCHESTRATOR, AgentMessage, MessageBus, MessageType
from .models import Agent, AgentResult, AgentStatus
from .profiles import AgentProfile
from .store import AgentStore
from .tools import MessageOrchestratorTool, ReportResultTool

ProviderFactory = Callable[[Agent], BaseProvider]
ApprovalHandler = Callable[[ApprovalRequest], bool]

AGENT_RULES = """
## Execution contract
- Working directory: {cwd}
- Task id: {task_id}
- You have a budget of {iterations} model turns and {tool_calls} tool calls. Be efficient.
- Your final action must be a single ReportResult call.
"""


class AgentRunner:
    def __init__(
        self,
        *,
        provider_factory: ProviderFactory,
        events: Any,
        store: AgentStore | None = None,
        bus: MessageBus | None = None,
        approval_handler: ApprovalHandler | None = None,
        base_registry: ToolRegistry | None = None,
        sessions_dir: Path | None = None,
    ) -> None:
        self.provider_factory = provider_factory
        self.events = events
        self.store = store
        self.bus = bus
        self.approval_handler = approval_handler
        self.base_registry = base_registry or build_default_registry(include_user_tools=False)
        self.sessions_dir = sessions_dir
        self._lock = threading.Lock()

    # -- lifecycle helpers -------------------------------------------------
    def _save(self, agent: Agent) -> None:
        if self.store is not None:
            with self._lock:
                self.store.save(agent)

    def create_agent(self, profile: AgentProfile, *, task_id: str | None, working_directory: str | Path,
                     provider: str, model: str, attempt: int = 1) -> Agent:
        agent = Agent(
            name=f"{profile.name}#{attempt}" if attempt > 1 else profile.name,
            role=profile.role,
            profile=profile.name,
            system_prompt=profile.system_prompt(),
            provider=provider,
            model=model,
            allowed_tools=list(profile.allowed_tools),
            permissions=profile.policy().to_dict(),
            working_directory=str(Path(working_directory).resolve()),
            task_id=task_id,
            budget=profile.budget,
            timeout_s=profile.budget.max_seconds,
            attempt=attempt,
        )
        self._save(agent)
        self.events.emit("agent.created", agent_id=agent.id, profile=profile.name, task_id=task_id,
                         provider=provider, model=model, working_directory=agent.working_directory,
                         allowed_tools=agent.allowed_tools, permissions=agent.permissions["capabilities"],
                         budget=agent.budget.to_dict(), attempt=attempt)
        return agent

    def _set_status(self, agent: Agent, status: AgentStatus) -> None:
        agent.transition(status)
        self._save(agent)

    # -- execution -----------------------------------------------------------
    def run(self, agent: Agent, assignment: str, *, should_cancel: Callable[[], bool] | None = None) -> AgentResult:
        if should_cancel is not None and should_cancel():
            self._set_status(agent, AgentStatus.CANCELLED)
            result = AgentResult(task_id=agent.task_id, agent_id=agent.id, status="failed",
                                 summary="Cancelled before start", stop_reason=STOP_CANCELLED, error="cancelled")
            agent.result = result.to_dict()
            self._save(agent)
            self.events.emit("agent.cancelled", agent_id=agent.id, task_id=agent.task_id)
            return result

        registry = self.base_registry.subset(agent.allowed_tools, extra=[ReportResultTool(), MessageOrchestratorTool()])
        workdir = Path(agent.working_directory)
        # Agents work in isolated, reviewable worktrees: documentation writes are
        # governed by the profile's WRITE capability, not the interactive allow_docs prompt.
        context = ToolContext(
            workspace_root=workdir,
            permission_context=ToolPermissionContext(allow_docs=True),
            permission_policy=PermissionPolicy.from_dict(agent.permissions),
        )
        context.agent_id = agent.id
        context.task_id = agent.task_id

        def approval(request: ApprovalRequest) -> bool:
            self._set_status(agent, AgentStatus.WAITING)
            self.events.emit("agent.waiting", agent_id=agent.id, task_id=agent.task_id, approval_id=request.id,
                             tool=request.tool, risk=request.risk, capabilities=list(request.capabilities))
            # dispatch() already logs approval events; persist them in the project event log too.
            self.events.emit("approval.requested", mirror_to_log=False, **request.to_dict())
            approved = False
            try:
                approved = bool(self.approval_handler(request)) if self.approval_handler else False
                return approved
            finally:
                self.events.emit("approval.approved" if approved else "approval.denied", mirror_to_log=False,
                                 approval_id=request.id, agent_id=agent.id, task_id=agent.task_id, tool=request.tool,
                                 risk=request.risk, capabilities=list(request.capabilities))
                self._set_status(agent, AgentStatus.RUNNING)

        context.approval_handler = approval

        def message_sink(kind: str, content: str) -> None:
            if self.bus is None:
                return
            self.bus.send(AgentMessage(type=MessageType(kind), sender=agent.id, recipient=ORCHESTRATOR,
                                       content=content, task_id=agent.task_id), sender_is_agent=True)

        context.message_sink = message_sink

        def on_event(ev: ToolEvent) -> None:
            if ev.kind == "tool_use":
                self.events.emit("agent.tool_started", agent_id=agent.id, task_id=agent.task_id, tool=ev.tool_name,
                                 summary=summarize_tool_use(ev.tool_name, ev.tool_input or {}))
            elif ev.kind in ("tool_result", "tool_error"):
                self.events.emit("agent.tool_completed", agent_id=agent.id, task_id=agent.task_id, tool=ev.tool_name,
                                 is_error=bool(ev.is_error))

        system_prompt = "\n\n".join(part for part in [
            agent.system_prompt,
            AGENT_RULES.format(cwd=workdir, task_id=agent.task_id, iterations=agent.budget.max_iterations,
                               tool_calls=agent.budget.max_tool_calls).strip(),
            _safe_context(workdir),
        ] if part)

        conversation = Conversation(max_history=0)  # agents keep full history within their budget
        conversation.add_user_message(assignment)
        self._set_status(agent, AgentStatus.RUNNING)
        self.events.emit("agent.started", agent_id=agent.id, task_id=agent.task_id, profile=agent.profile,
                         attempt=agent.attempt)

        loop = None
        error: Optional[str] = None
        try:
            provider = self.provider_factory(agent)
            budget = LoopBudget.from_seconds(
                max_tool_calls=agent.budget.max_tool_calls,
                max_tokens=agent.budget.max_tokens,
                max_seconds=min(agent.budget.max_seconds, agent.timeout_s),
            )
            loop = run_agent_loop(
                conversation=conversation,
                provider=provider,
                tool_registry=registry,
                tool_context=context,
                max_turns=agent.budget.max_iterations,
                system_prompt=system_prompt,
                budget=budget,
                should_cancel=should_cancel,
                on_event=on_event,
            )
        except Exception as e:  # provider/runtime failure
            error = f"{type(e).__name__}: {e}"

        result = self._build_result(agent, context, loop, error)
        agent.result = result.to_dict()
        agent.error = result.error
        agent.usage = loop.usage if loop is not None else None
        agent.session_id = self._persist_session(agent, conversation)

        if loop is not None and loop.stop_reason == STOP_CANCELLED:
            final = AgentStatus.CANCELLED
        elif result.status == "completed":
            final = AgentStatus.COMPLETED
        else:
            final = AgentStatus.FAILED
        self._set_status(agent, final)
        self.events.emit(
            {AgentStatus.COMPLETED: "agent.completed", AgentStatus.FAILED: "agent.failed",
             AgentStatus.CANCELLED: "agent.cancelled"}[final],
            agent_id=agent.id, task_id=agent.task_id, status=result.status, stop_reason=result.stop_reason,
            error=result.error, summary=result.summary, turns=loop.num_turns if loop else 0,
            tool_calls=loop.tool_calls if loop else 0, usage=agent.usage,
        )
        return result

    def _build_result(self, agent: Agent, context: ToolContext, loop: Any, error: Optional[str]) -> AgentResult:
        stop_reason = loop.stop_reason if loop is not None else "error"
        denials = [f"permission denied: {d.get('tool')}: {d.get('message')}" for d in (loop.permission_denials if loop else ())]
        if context.agent_result is not None:
            data = dict(context.agent_result)
            data["agent_id"] = agent.id
            data["task_id"] = agent.task_id
            data["stop_reason"] = stop_reason
            data["warnings"] = list(data.get("warnings") or []) + denials
            result = AgentResult.from_dict(data)
            broke_down = error is not None or stop_reason != STOP_COMPLETED
            if result.status == "completed" and broke_down:
                # Reported success, but the run then broke down (error/budget/timeout): not trustworthy.
                result.status = "failed"
                result.error = error or f"run ended with {stop_reason} after reporting"
            elif result.status != "completed" and not result.error:
                result.error = f"agent reported status '{result.status}'"
            return result
        if error is None:
            if loop is None:
                error = "agent did not run"
            elif loop.stop_reason == STOP_COMPLETED:
                error = "agent finished without calling ReportResult (no structured result)"
            else:
                error = f"agent stopped: {loop.stop_reason}"
        return AgentResult(
            task_id=agent.task_id, agent_id=agent.id, status="failed",
            summary=(loop.response_text[:500] if loop is not None and loop.response_text else "No structured result."),
            warnings=denials, stop_reason=stop_reason, error=error, reported_by_agent=False,
        )

    def _persist_session(self, agent: Agent, conversation: Conversation) -> Optional[str]:
        if self.sessions_dir is None:
            return None
        session = Session.create(agent.provider, agent.model, metadata={
            "agent_id": agent.id, "task_id": agent.task_id, "profile": agent.profile,
            "working_directory": agent.working_directory, "attempt": agent.attempt,
        })
        session.session_id = f"{session.session_id}_{agent.id}"
        session.conversation = conversation
        try:
            session.save(self.sessions_dir)
        except Exception:
            return None
        return session.session_id


def _safe_context(workdir: Path) -> str:
    try:
        return build_context_prompt(workdir)
    except Exception:
        return ""
