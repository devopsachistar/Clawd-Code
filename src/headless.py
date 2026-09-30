"""Non-interactive (scriptable / CI) agent runs.

``clawd run "Find and fix the failing tests"`` runs one task with the same
agent loop, tools, config and sessions as the REPL, but without a TTY. It
returns a meaningful process exit code:

    0  success          the agent finished the task
    1  task failure     provider/runtime error, or the turn limit was reached
    2  config error     bad/missing configuration, unsafe project root, bad args
    3  permission       a tool call was refused by the permission/safety policy

Permission prompts cannot be answered in headless mode. Actions that need
approval are therefore denied unless ``--yes`` was given. Blocked commands are
refused even with ``--yes``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Optional, TextIO

from .agent.session import Session, latest_session_id, restore_tool_state, capture_tool_state
from .config import ConfigError, ResolvedProviderConfig, create_provider, missing_api_key_message, resolve_provider_config
from .observability import log_event, redact_text
from .project import ProjectInfo, detect_project
from .tool_system.agent_loop import STOP_MAX_TURNS, ToolEvent, run_agent_loop, summarize_tool_result, summarize_tool_use
from .tool_system.context import ToolContext
from .tool_system.defaults import build_default_registry


class ExitCode(IntEnum):
    SUCCESS = 0
    TASK_FAILURE = 1
    CONFIG_ERROR = 2
    PERMISSION_DENIED = 3


_STATUS = {
    ExitCode.SUCCESS: "success",
    ExitCode.TASK_FAILURE: "failure",
    ExitCode.CONFIG_ERROR: "config_error",
    ExitCode.PERMISSION_DENIED: "permission_denied",
}


@dataclass
class RunOptions:
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    cwd: Optional[str] = None
    max_turns: int = 30
    output: str = "text"  # "text" | "json"
    stream: bool = False
    auto_approve: bool = False
    resume: Optional[str] = None  # session id or "latest"
    save_session: bool = True
    allow_unsafe_root: bool = False
    quiet: bool = False


@dataclass
class RunOutcome:
    exit_code: int
    status: str
    response: str = ""
    error: Optional[str] = None
    stop_reason: Optional[str] = None
    session_id: Optional[str] = None
    session_path: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    project: dict[str, Any] = field(default_factory=dict)
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    permission_denials: list[dict[str, Any]] = field(default_factory=list)
    usage: Optional[dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, default=str)


ProviderFactory = Callable[[ResolvedProviderConfig], Any]


def _is_auth_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name in {"AuthenticationError", "PermissionDeniedError"} or " 401" in str(exc)


def run_headless(
    options: RunOptions,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    provider_factory: ProviderFactory | None = None,
) -> RunOutcome:
    """Run one task non-interactively and return its outcome (never raises for task errors)."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    json_mode = options.output == "json"

    def note(message: str) -> None:
        # Progress goes to stderr; keep separators ASCII so CI logs and
        # legacy Windows code pages render cleanly.
        if not options.quiet:
            print(message.replace(" · ", " - "), file=err, flush=True)

    def finish(outcome: RunOutcome) -> RunOutcome:
        log_event("run.completed", exit_code=outcome.exit_code, status=outcome.status,
                  session_id=outcome.session_id, turns=outcome.turns, tool_calls=outcome.tool_calls,
                  permission_denials=len(outcome.permission_denials), error=outcome.error)
        if json_mode:
            print(outcome.to_json(), file=out, flush=True)
        elif outcome.error:
            print(f"Error: {outcome.error}", file=err, flush=True)
        return outcome

    def config_failure(message: str, **extra: Any) -> RunOutcome:
        return finish(RunOutcome(exit_code=ExitCode.CONFIG_ERROR, status=_STATUS[ExitCode.CONFIG_ERROR], error=message, **extra))

    prompt = (options.prompt or "").strip()
    if not prompt:
        return config_failure("No task given. Usage: clawd run \"<task>\" (or pipe the task on stdin)")
    if options.max_turns < 1:
        return config_failure("--max-turns must be at least 1")

    # 1. Project
    start = Path(options.cwd).expanduser() if options.cwd else Path(os.getcwd())
    if not start.is_dir():
        return config_failure(f"Working directory does not exist: {start}")
    project: ProjectInfo = detect_project(start)
    if project.is_unsafe_root and not options.allow_unsafe_root:
        return config_failure(
            f"Refusing to run with project root {project.project_root}: it is your home directory or a "
            "filesystem root. Run inside a project directory, pass --cwd <project>, or --allow-unsafe-root.",
            project=project.to_dict(),
        )

    # 2. Configuration + provider
    try:
        resolved = resolve_provider_config(
            cli_provider=options.provider,
            cli_model=options.model,
            cli_base_url=options.base_url,
            project_root=project.project_root,
        )
    except ConfigError as e:
        return config_failure(str(e), project=project.to_dict())
    for warning in resolved.warnings:
        note(f"warning: {warning}")
    if not resolved.has_api_key:
        return config_failure(missing_api_key_message(resolved), project=project.to_dict(),
                              provider=resolved.provider, model=resolved.model)
    try:
        provider = (provider_factory or create_provider)(resolved)
    except ConfigError as e:
        return config_failure(str(e), project=project.to_dict())
    except (ImportError, ModuleNotFoundError) as e:
        return config_failure(f"Provider '{resolved.provider}' is unavailable: {e}", project=project.to_dict())

    # 3. Session
    if options.resume:
        session_id = options.resume
        if session_id == "latest":
            session_id = latest_session_id(project_root=project.project_root) or ""
            if not session_id:
                return config_failure(f"No saved session found for project {project.project_root}",
                                      project=project.to_dict())
        try:
            loaded = Session.load(session_id)
        except Exception as e:  # corrupt file
            return config_failure(f"Could not load session {session_id}: {e}", project=project.to_dict())
        if loaded is None:
            return config_failure(f"Session not found: {session_id}", project=project.to_dict())
        session = loaded
        session.provider, session.model = resolved.provider, resolved.model
        session.metadata.update(project.to_dict())
    else:
        session = Session.create(resolved.provider, resolved.model, metadata=project.to_dict())

    # 4. Tools
    from .config import load_session_policy

    policy, policy_warnings = load_session_policy(project.project_root)
    for warning in policy_warnings:
        note(f"warning: {warning}")
    context = ToolContext(workspace_root=project.project_root, cwd=project.cwd, permission_policy=policy)
    for warning in restore_tool_state(context, session.state):
        note(f"warning: {warning}")

    def approval_handler(request: Any) -> bool:
        # --yes only covers low/medium-risk requests. Production deploys,
        # database writes, secret use, pushes and anything HIGH/CRITICAL
        # always need an explicit human decision, which headless mode can't get.
        if options.auto_approve and request.auto_approvable:
            note(f"[auto-approved] {request.summary} (risk {request.risk}, {', '.join(request.capabilities)})")
            return True
        why = "re-run with --yes to approve" if request.auto_approvable else             f"risk {request.risk} / {', '.join(request.capabilities)} requires explicit approval; --yes does not cover it"
        note(f"[denied] {request.summary}: {redact_text('; '.join(request.reasons))} ({why})")
        return False

    context.approval_handler = approval_handler
    registry = build_default_registry()

    def on_event(ev: ToolEvent) -> None:
        if options.quiet:
            return
        if ev.kind == "tool_use":
            summary = summarize_tool_use(ev.tool_name, ev.tool_input or {})
            note(f"* {ev.tool_name}" + (f" ({redact_text(summary)})" if summary else ""))
        elif ev.kind == "tool_result":
            if ev.is_error:
                msg = ev.tool_output.get("error") if isinstance(ev.tool_output, dict) else None
                note(f"  -> error: {redact_text(str(msg or 'failed'))}")
            else:
                note(f"  -> {redact_text(summarize_tool_result(ev.tool_name, ev.tool_output))}")
        elif ev.kind == "tool_error":
            note(f"  -> error: {redact_text(ev.error or 'failed')}")

    streamed = {"any": False}

    def on_text_chunk(chunk: str) -> None:
        if chunk:
            streamed["any"] = True
            out.write(chunk)
            out.flush()

    note(f"project: {project.project_root}" + (f" (git: {project.git_root})" if project.git_root else " (no git)"))
    note(f"provider: {resolved.provider} | model: {resolved.model} | session: {session.session_id}")
    log_event("run.started", session_id=session.session_id, provider=resolved.provider, model=resolved.model,
              project_root=str(project.project_root), git_root=str(project.git_root) if project.git_root else None,
              resumed=bool(options.resume), auto_approve=options.auto_approve, max_turns=options.max_turns)

    # 5. Run
    session.conversation.add_user_message(prompt)
    outcome = RunOutcome(
        exit_code=ExitCode.SUCCESS,
        status=_STATUS[ExitCode.SUCCESS],
        session_id=session.session_id,
        provider=resolved.provider,
        model=resolved.model,
        project=project.to_dict(),
    )
    try:
        result = run_agent_loop(
            conversation=session.conversation,
            provider=provider,
            tool_registry=registry,
            tool_context=context,
            max_turns=options.max_turns,
            stream=options.stream and not json_mode,
            on_event=on_event,
            on_text_chunk=on_text_chunk if (options.stream and not json_mode) else None,
        )
    except KeyboardInterrupt:
        outcome.exit_code, outcome.status, outcome.error = ExitCode.TASK_FAILURE, "interrupted", "Interrupted"
        result = None
    except Exception as e:
        code = ExitCode.CONFIG_ERROR if _is_auth_error(e) else ExitCode.TASK_FAILURE
        outcome.exit_code, outcome.status = code, _STATUS[code]
        outcome.error = f"{type(e).__name__}: {redact_text(str(e))}"
        result = None

    if result is not None:
        outcome.response = result.response_text
        outcome.stop_reason = result.stop_reason
        outcome.turns = result.num_turns
        outcome.tool_calls = result.tool_calls
        outcome.tool_errors = result.tool_errors
        outcome.permission_denials = [dict(d) for d in result.permission_denials]
        outcome.usage = result.usage
        if result.stop_reason == STOP_MAX_TURNS:
            outcome.exit_code, outcome.status = ExitCode.TASK_FAILURE, _STATUS[ExitCode.TASK_FAILURE]
            outcome.error = f"Reached the turn limit ({options.max_turns}) before the task finished"
        elif result.permission_denials:
            outcome.exit_code, outcome.status = ExitCode.PERMISSION_DENIED, _STATUS[ExitCode.PERMISSION_DENIED]
            outcome.error = f"{len(result.permission_denials)} tool call(s) were refused by the permission policy"

    # 6. Persist
    if options.save_session:
        session.state = capture_tool_state(context)
        try:
            outcome.session_path = str(session.save())
        except Exception as e:
            note(f"warning: could not save session: {e}")

    if not json_mode:
        if outcome.response and not streamed["any"]:
            print(outcome.response, file=out, flush=True)
        elif streamed["any"]:
            print(file=out, flush=True)
        note(
            f"[{outcome.status}] exit={int(outcome.exit_code)} turns={outcome.turns} "
            f"tools={outcome.tool_calls} session={outcome.session_id}"
        )
    return finish(outcome)
