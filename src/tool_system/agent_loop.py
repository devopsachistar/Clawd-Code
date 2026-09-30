"""Agent loop for multi-turn tool calling."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .registry import ToolRegistry
from .context import ToolContext
from .errors import ToolInputError, ToolPermissionError
from .protocol import ToolCall
from ..agent.conversation import Conversation, TextContentBlock, ToolResultContentBlock, ToolUseContentBlock
from ..agent.message_format import build_provider_request, resolve_wire_format, serialize_tool_result_content
from ..context_system import build_context_prompt
from ..observability import log_event
from ..outputStyles import resolve_output_style
from ..providers.base import BaseProvider, ChatResponse


def bound_tool_result_content(output: Any, limit: int | None) -> Any:
    """Serialise a tool output for the conversation, enforcing the tool's size limit."""
    content = serialize_tool_result_content(output)
    if not isinstance(content, str) or not limit or limit <= 0 or len(content) <= limit:
        return content
    omitted = len(content) - limit
    return content[:limit] + f"\n\n... [tool output truncated: {omitted} characters omitted]"

def summarize_tool_result(name: str, output: Any) -> str:
    """Create a concise, single-line summary for tool result output."""
    if not isinstance(output, dict):
        return str(output)
    if name.lower() == "write":
        path = output.get("filePath") or output.get("file_path")
        op = output.get("type")
        return f"{name} · {path} · {op}"
    if name.lower() == "edit":
        path = output.get("filePath") or output.get("file_path")
        replace_all = output.get("replaceAll")
        return f"{name} · {path} · replaceAll={replace_all}"
    if name.lower() == "read":
        if output.get("type") == "text" and isinstance(output.get("file"), dict):
            f = output["file"]
            path = f.get("filePath")
            num = f.get("numLines")
            total = f.get("totalLines")
            start = f.get("startLine")
            return f"{name} · {path} · lines={start}-{(start or 1) + (num or 0) - 1}/{total}"
        if output.get("type") == "file_unchanged" and isinstance(output.get("file"), dict):
            return f"{name} · {output['file'].get('filePath')} · unchanged"
        if output.get("type") in {"image", "pdf", "notebook"} and isinstance(output.get("file"), dict):
            return f"{name} · {output['file'].get('filePath')} · {output.get('type')}"
        return f"{name}"
    if name.lower() == "glob":
        n = output.get("numFiles")
        return f"{name} · matches={n}"
    if name.lower() == "grep":
        n = output.get("numFiles")
        mode = output.get("mode")
        return f"{name} · mode={mode} · files={n}"
    if name.lower() == "bash":
        code = output.get("exit_code")
        return f"{name} · exit={code}"
    if name.lower() == "webfetch":
        url = output.get("url")
        ct = output.get("content_type")
        return f"{name} · {url} · {ct}"
    if name.lower() == "websearch":
        q = output.get("query")
        results = output.get("results")
        n = len(results) if isinstance(results, list) else None
        return f"{name} · \"{q}\" · results={n}"
    if name.lower() == "config":
        op = output.get("operation")
        setting = output.get("setting")
        return f"{name} · {op} · {setting}"
    if name.lower() == "taskstop":
        tid = output.get("task_id")
        stopped = output.get("stopped")
        return f"{name} · {tid} · stopped={stopped}"
    if name.lower() == "sendusermessage":
        n = 0
        atts = output.get("attachments")
        if isinstance(atts, list):
            n = len(atts)
        return f"{name} · attachments={n}"
    # default: truncate dict keys for brevity
    keys = ", ".join(list(output.keys())[:3])
    return f"{name} · {keys}"


@dataclass(frozen=True)
class ToolEvent:
    kind: str
    tool_name: str
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    tool_use_id: str | None = None
    is_error: bool = False
    error: str | None = None


STOP_COMPLETED = "completed"
STOP_MAX_TURNS = "max_turns"


@dataclass(frozen=True)
class AgentLoopResult:
    """Result of running the agent loop."""
    response_text: str
    usage: dict[str, Any] | None = None  # {"input_tokens": int, "output_tokens": int}
    num_turns: int = 0
    stop_reason: str = STOP_COMPLETED  # "completed" | "max_turns"
    tool_calls: int = 0
    tool_errors: int = 0
    # One entry per tool call blocked by permission/safety policy or denied by the user.
    permission_denials: tuple[dict[str, Any], ...] = field(default_factory=tuple)


ToolEventHandler = Callable[[ToolEvent], None]
TextChunkHandler = Callable[[str], None]


def _safe_call_handler(handler: ToolEventHandler | None, event: ToolEvent) -> None:
    if handler is None:
        return
    try:
        handler(event)
    except Exception:
        return


def _emit_text_chunks(handler: TextChunkHandler | None, text: str, *, chunk_size: int = 12) -> None:
    """Emit text in small chunks for user-visible streaming without changing loop semantics."""
    if handler is None or not text:
        return
    if chunk_size <= 0:
        chunk_size = len(text)
    for idx in range(0, len(text), chunk_size):
        try:
            handler(text[idx: idx + chunk_size])
        except Exception:
            return


def _call_provider_for_turn(
    *,
    provider: BaseProvider,
    api_messages: list[dict[str, Any]],
    call_kwargs: dict[str, Any],
    stream: bool,
    on_text_chunk: TextChunkHandler | None,
) -> tuple[Any, bool]:
    """Call the provider, preferring structured streaming when available.

    Returns (response, streamed_live_text).
    """
    if stream:
        try:
            response = provider.chat_stream_response(
                api_messages,
                on_text_chunk=on_text_chunk,
                **call_kwargs,
            )
            if not isinstance(response, ChatResponse):
                raise TypeError("Structured streaming must return ChatResponse")
            return response, True
        except NotImplementedError:
            pass
        except Exception:
            # Preserve existing stable behavior if streaming is unsupported or fails.
            pass

    response = provider.chat(api_messages, **call_kwargs)
    return response, False


def _build_effective_system_prompt(style_prompt: str, tool_context: ToolContext) -> str:
    try:
        context_prompt = build_context_prompt(
            tool_context.workspace_root,
            cwd=tool_context.cwd,
        )
    except Exception:
        context_prompt = ""
    if not context_prompt.strip():
        return style_prompt
    return f"{style_prompt}\n\n{context_prompt}"


def summarize_tool_use(name: str, tool_input: dict[str, Any]) -> str:
    lowered = name.lower()
    if lowered == "bash":
        cmd = tool_input.get("command")
        if isinstance(cmd, str):
            s = cmd.strip().replace("\n", " ")
            return s if len(s) <= 80 else s[:77] + "..."
        return ""
    if lowered in {"read", "write", "edit"}:
        p = tool_input.get("file_path") or tool_input.get("filePath") or tool_input.get("path")
        if isinstance(p, str):
            extra = ""
            if lowered == "read":
                off = tool_input.get("offset")
                lim = tool_input.get("limit")
                if isinstance(off, int) or isinstance(lim, int):
                    start = off if isinstance(off, int) else 1
                    if isinstance(lim, int):
                        extra = f" · lines {start}-{start + lim - 1}"
            return f"{p}{extra}"
        return ""
    if lowered == "glob":
        pat = tool_input.get("pattern")
        base = tool_input.get("path")
        if isinstance(pat, str) and isinstance(base, str):
            return f"{pat} · {base}"
        if isinstance(pat, str):
            return pat
        return ""
    if lowered == "grep":
        pat = tool_input.get("pattern")
        base = tool_input.get("path")
        if isinstance(pat, str) and isinstance(base, str):
            return f"{pat} · {base}"
        if isinstance(pat, str):
            return pat
        return ""
    if lowered == "webfetch":
        url = tool_input.get("url")
        return url if isinstance(url, str) else ""
    if lowered == "websearch":
        q = tool_input.get("query")
        return q if isinstance(q, str) else ""
    if lowered == "toolsearch":
        q = tool_input.get("query")
        return q if isinstance(q, str) else ""
    if lowered == "askuserquestion":
        qs = tool_input.get("questions")
        if isinstance(qs, list):
            return f"{len(qs)} question(s)"
        return ""
    if lowered == "sendusermessage":
        status = tool_input.get("status")
        return status if isinstance(status, str) else ""
    return ""



def _execute_tool_call(
    tool_use: dict[str, Any],
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    on_event: ToolEventHandler | None,
    verbose: bool,
) -> tuple[ToolResultContentBlock, Any, str | None]:
    """Run one tool call. Returns (result block, raw output, error_kind)."""
    tool_id = str(tool_use.get("id", ""))
    tool_name = str(tool_use.get("name", ""))
    tool_input = tool_use.get("input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    _safe_call_handler(on_event, ToolEvent(kind="tool_use", tool_name=tool_name, tool_input=tool_input, tool_use_id=tool_id))
    log_event("tool.started", tool=tool_name, tool_use_id=tool_id, input=tool_input)
    started = time.monotonic()

    tool = tool_registry.get(tool_name)
    limit = tool.spec().max_result_size_chars if tool is not None else None
    try:
        result = tool_registry.dispatch(ToolCall(name=tool_name, input=tool_input, tool_use_id=tool_id), tool_context)
    except Exception as e:  # the tool raised instead of returning an error result
        if isinstance(e, ToolPermissionError):
            error_kind = "permission"
        elif isinstance(e, ToolInputError):
            error_kind = "input"
        else:
            error_kind = "execution"
        error_str = f"Error: {e}"
        if verbose:
            print(f"[Tool Error] {error_str}")
        _safe_call_handler(on_event, ToolEvent(
            kind="tool_error", tool_name=tool_name, tool_input=tool_input,
            tool_use_id=tool_id, is_error=True, error=error_str,
        ))
        log_event(
            "permission.denied" if error_kind == "permission" else "tool.failed",
            tool=tool_name, tool_use_id=tool_id, error_kind=error_kind, error=str(e),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        block = ToolResultContentBlock(tool_use_id=tool_id, content=error_str, is_error=True)
        return block, {"error": error_str}, error_kind

    output = result.output
    error_kind = result.error_kind if result.is_error else None
    if result.is_error and error_kind is None:
        error_kind = "execution"
    if verbose:
        use_summary = summarize_tool_use(tool_name, tool_input)
        if use_summary:
            print(f"{tool_name} · {use_summary}")
        print(summarize_tool_result(tool_name, output))
    _safe_call_handler(on_event, ToolEvent(
        kind="tool_result", tool_name=tool_name, tool_output=output,
        tool_use_id=tool_id, is_error=result.is_error,
    ))
    duration_ms = int((time.monotonic() - started) * 1000)
    if error_kind == "permission":
        log_event("permission.denied", tool=tool_name, tool_use_id=tool_id, output=output, duration_ms=duration_ms)
    elif result.is_error:
        log_event("tool.failed", tool=tool_name, tool_use_id=tool_id, error_kind=error_kind, output=output, duration_ms=duration_ms)
    else:
        log_event("tool.completed", tool=tool_name, tool_use_id=tool_id, duration_ms=duration_ms)

    block = ToolResultContentBlock(
        tool_use_id=tool_id,
        content=bound_tool_result_content(output, limit),
        is_error=result.is_error,
    )
    return block, output, error_kind


def run_agent_loop(
    conversation: Conversation,
    provider: BaseProvider,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    max_turns: int = 20,
    stream: bool = False,
    verbose: bool = False,
    on_event: ToolEventHandler | None = None,
    on_text_chunk: TextChunkHandler | None = None,
    system_prompt: str | None = None,
) -> AgentLoopResult:
    """Run agent loop: LLM -> tools -> LLM until no more tools or max turns.

    The conversation is the single source of truth. Every assistant turn
    (including tool calls) and every tool result is persisted to it, and each
    provider request is rebuilt from it via :mod:`src.agent.message_format`, so
    the system prompt and full tool history are present on every request for
    every provider.

    Args:
        conversation: Conversation with initial user message
        provider: LLM provider
        tool_registry: Tool registry to use
        tool_context: Tool context
        max_turns: Maximum model turns before stopping
        stream: Whether to stream responses
        verbose: Whether to print tool calls/results
        on_event: Optional callback for tool events
        on_text_chunk: Optional callback for incremental user-visible text chunks
        system_prompt: Optional explicit system prompt (defaults to output style + project context)

    Returns:
        AgentLoopResult with final text response, usage info, turn count and stop reason
    """
    tool_schemas = [
        {"name": spec.name, "description": spec.description, "input_schema": spec.input_schema}
        for spec in tool_registry.list_specs()
    ]

    if system_prompt is None:
        style_name = getattr(tool_context, "output_style_name", None)
        style_dir = getattr(tool_context, "output_style_dir", None)
        style_prompt = resolve_output_style(style_name, style_dir).prompt
        system_prompt = _build_effective_system_prompt(style_prompt, tool_context)

    wire_format = resolve_wire_format(provider)
    model_name = getattr(provider, "model", None)
    provider_name = type(provider).__name__
    last_user_visible_message: str | None = None
    total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    turn_count = 0
    tool_call_count = 0
    tool_error_count = 0
    denials: list[dict[str, Any]] = []

    def _usage() -> dict[str, int] | None:
        return total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None

    def _result(text: str, stop_reason: str) -> AgentLoopResult:
        log_event(
            "agent.completed" if stop_reason == STOP_COMPLETED else "agent.max_turns",
            turns=turn_count, tool_calls=tool_call_count, tool_errors=tool_error_count,
            permission_denials=len(denials), usage=_usage(), stop_reason=stop_reason,
        )
        return AgentLoopResult(
            response_text=text,
            usage=_usage(),
            num_turns=turn_count,
            stop_reason=stop_reason,
            tool_calls=tool_call_count,
            tool_errors=tool_error_count,
            permission_denials=tuple(denials),
        )

    log_event("agent.started", provider=provider_name, model=model_name, wire_format=wire_format,
              max_turns=max_turns, tools=len(tool_schemas), messages=len(conversation.messages))

    for _turn in range(max_turns):
        api_messages, format_kwargs = build_provider_request(
            conversation.messages, system_prompt=system_prompt, wire_format=wire_format,
        )
        call_kwargs: dict[str, Any] = {"tools": tool_schemas, **format_kwargs}

        log_event("provider.request", provider=provider_name, model=model_name,
                  turn=turn_count + 1, messages=len(api_messages), stream=stream)
        started = time.monotonic()
        try:
            response, streamed_live_text = _call_provider_for_turn(
                provider=provider,
                api_messages=api_messages,
                call_kwargs=call_kwargs,
                stream=stream,
                on_text_chunk=on_text_chunk,
            )
        except Exception as e:
            log_event("provider.error", provider=provider_name, model=model_name,
                      turn=turn_count + 1, error_type=type(e).__name__, error=str(e))
            raise
        turn_count += 1

        if response.usage:
            total_usage["input_tokens"] += response.usage.get("input_tokens", 0) or 0
            total_usage["output_tokens"] += response.usage.get("output_tokens", 0) or 0

        final_assistant_content = response.content or ""
        tool_uses = [tu for tu in (response.tool_uses or []) if isinstance(tu, dict)]
        log_event("provider.response", model=getattr(response, "model", model_name), turn=turn_count,
                  finish_reason=response.finish_reason, tool_calls=len(tool_uses), usage=response.usage,
                  duration_ms=int((time.monotonic() - started) * 1000))

        # Persist the assistant turn (text and tool calls) to the neutral conversation.
        if tool_uses:
            assistant_blocks: list = []
            if final_assistant_content:
                assistant_blocks.append(TextContentBlock(type="text", text=final_assistant_content))
            for tool_use in tool_uses:
                assistant_blocks.append(ToolUseContentBlock(
                    type="tool_use",
                    id=str(tool_use.get("id", "")),
                    name=str(tool_use.get("name", "")),
                    input=dict(tool_use.get("input") or {}),
                ))
            conversation.add_assistant_message(assistant_blocks)
        else:
            conversation.add_assistant_message(final_assistant_content)

        if not tool_uses:
            if stream and final_assistant_content and not streamed_live_text:
                _emit_text_chunks(on_text_chunk, final_assistant_content)
            if final_assistant_content.strip() == "" and last_user_visible_message is not None:
                return _result(last_user_visible_message, STOP_COMPLETED)
            return _result(final_assistant_content, STOP_COMPLETED)

        # Execute every tool call, then record all results as one user turn.
        result_blocks: list[ToolResultContentBlock] = []
        for tool_use in tool_uses:
            block, output, error_kind = _execute_tool_call(tool_use, tool_registry, tool_context, on_event, verbose)
            result_blocks.append(block)
            tool_call_count += 1
            tool_name = str(tool_use.get("name", ""))
            if block.is_error:
                tool_error_count += 1
            if error_kind == "permission":
                message = output.get("error") if isinstance(output, dict) else str(output)
                denials.append({"tool": tool_name, "tool_use_id": block.tool_use_id, "message": message})
            if not block.is_error and isinstance(output, dict):
                if tool_name.lower() == "sendusermessage" and isinstance(output.get("message"), str):
                    last_user_visible_message = output["message"]
                if tool_name.lower() == "structuredoutput":
                    payload = output.get("structured_output")
                    try:
                        last_user_visible_message = json.dumps(payload, ensure_ascii=False, indent=2)
                    except Exception:
                        last_user_visible_message = str(payload)
        conversation.add_tool_results(result_blocks)

    return _result("[Max tool turns reached]", STOP_MAX_TURNS)
