"""Provider-neutral conversation -> provider wire format adapters.

The :class:`~src.agent.conversation.Conversation` is the single source of truth
for a session. It stores messages as neutral content blocks (text, tool_use,
tool_result). Each provider declares which wire format it speaks via its
``message_format`` attribute, and this module performs the conversion.

All providers share one normalisation pass so that every wire format gets the
same guarantees:

* internal messages (e.g. compact boundaries) are never sent
* empty messages are dropped and consecutive same-role messages are merged
* every ``tool_use`` is answered by exactly one ``tool_result`` in the next user
  message (missing results are synthesised, orphaned results are dropped)
* tool result payloads are serialised to text the provider accepts
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Literal

WireFormat = Literal["anthropic", "openai"]

ANTHROPIC_FORMAT: WireFormat = "anthropic"
OPENAI_FORMAT: WireFormat = "openai"
SUPPORTED_WIRE_FORMATS: tuple[str, ...] = (ANTHROPIC_FORMAT, OPENAI_FORMAT)

INTERRUPTED_TOOL_RESULT = "Tool execution was interrupted before a result was recorded."


def resolve_wire_format(provider: Any) -> WireFormat:
    """Return the wire format a provider speaks (defaults to OpenAI-style)."""
    fmt = getattr(type(provider), "message_format", None)
    if not isinstance(fmt, str):
        fmt = getattr(provider, "message_format", None)
    if isinstance(fmt, str) and fmt in SUPPORTED_WIRE_FORMATS:
        return fmt  # type: ignore[return-value]
    return OPENAI_FORMAT


def serialize_tool_result_content(content: Any) -> str | list[dict[str, Any]]:
    """Convert a tool output into content every provider accepts.

    Strings pass through. Lists of Anthropic-style content blocks (text/image)
    pass through. Anything else (dicts, numbers, mixed lists) becomes JSON text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(
        isinstance(b, dict) and b.get("type") in {"text", "image"} for b in content
    ):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content)


def tool_result_as_text(content: Any) -> str:
    """Flatten tool result content into plain text (OpenAI ``tool`` messages)."""
    serialized = serialize_tool_result_content(content)
    if isinstance(serialized, str):
        return serialized
    parts: list[str] = []
    for block in serialized:
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        else:
            parts.append(f"[{block.get('type')} content omitted]")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _block_to_dict(block: Any) -> dict[str, Any] | None:
    if isinstance(block, dict):
        btype = block.get("type")
        data = dict(block)
    else:
        btype = getattr(block, "type", None)
        data = {k: getattr(block, k) for k in ("text", "id", "name", "input", "tool_use_id", "content", "is_error") if hasattr(block, k)}
    if btype == "text":
        text = data.get("text") or ""
        return {"type": "text", "text": str(text)} if str(text).strip() else None
    if btype == "tool_use":
        return {
            "type": "tool_use",
            "id": str(data.get("id", "")),
            "name": str(data.get("name", "")),
            "input": dict(data.get("input") or {}),
        }
    if btype == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": str(data.get("tool_use_id", "")),
            "content": data.get("content", ""),
            "is_error": bool(data.get("is_error", False)),
        }
    if btype == "image":
        return data
    return None


def _message_blocks(message: Any) -> tuple[str, list[dict[str, Any]]] | None:
    if getattr(message, "_is_internal", False):
        return None
    if isinstance(message, dict):
        role = message.get("role")
        content = message.get("content")
    else:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
    if role not in ("user", "assistant"):
        return None
    if isinstance(content, str):
        blocks = [{"type": "text", "text": content}] if content.strip() else []
    elif isinstance(content, list):
        blocks = [b for b in (_block_to_dict(x) for x in content) if b is not None]
    else:
        blocks = []
    return role, blocks


def _merge_same_role(entries: list[tuple[str, list[dict[str, Any]]]]) -> list[tuple[str, list[dict[str, Any]]]]:
    merged: list[tuple[str, list[dict[str, Any]]]] = []
    for role, blocks in entries:
        if not blocks:
            continue
        if merged and merged[-1][0] == role:
            merged[-1] = (role, merged[-1][1] + blocks)
        else:
            merged.append((role, list(blocks)))
    return merged


def normalize_messages(messages: Iterable[Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Return ``(role, blocks)`` pairs with tool_use/tool_result pairing repaired."""
    entries = [e for e in (_message_blocks(m) for m in messages) if e is not None]
    entries = _merge_same_role(entries)

    repaired: list[tuple[str, list[dict[str, Any]]]] = []
    pending: list[str] = []  # tool_use ids awaiting results, in order
    for role, blocks in entries:
        if role == "assistant":
            if pending:
                # Previous assistant turn never got results (e.g. interrupted run).
                repaired.append(("user", [_interrupted_result(tid) for tid in pending]))
            pending = [b["id"] for b in blocks if b["type"] == "tool_use"]
            repaired.append((role, blocks))
            continue

        # user message
        results = [b for b in blocks if b["type"] == "tool_result"]
        others = [b for b in blocks if b["type"] != "tool_result"]
        kept: list[dict[str, Any]] = []
        answered: set[str] = set()
        for r in results:
            if r["tool_use_id"] in pending and r["tool_use_id"] not in answered:
                kept.append(r)
                answered.add(r["tool_use_id"])
        for tid in pending:
            if tid not in answered:
                kept.append(_interrupted_result(tid))
        pending = []
        # tool_result blocks must precede any other content in the user turn
        new_blocks = kept + others
        if new_blocks:
            repaired.append((role, new_blocks))

    if pending:
        repaired.append(("user", [_interrupted_result(tid) for tid in pending]))

    return _merge_same_role(repaired)


def _interrupted_result(tool_use_id: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": INTERRUPTED_TOOL_RESULT,
        "is_error": True,
    }


# ---------------------------------------------------------------------------
# Wire formats
# ---------------------------------------------------------------------------

def to_anthropic_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert neutral messages into Anthropic Messages API format."""
    api_messages: list[dict[str, Any]] = []
    for role, blocks in normalize_messages(messages):
        if len(blocks) == 1 and blocks[0]["type"] == "text":
            api_messages.append({"role": role, "content": blocks[0]["text"]})
            continue
        content: list[dict[str, Any]] = []
        for b in blocks:
            if b["type"] == "tool_result":
                item: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": b["tool_use_id"],
                    "content": serialize_tool_result_content(b["content"]),
                }
                if b.get("is_error"):
                    item["is_error"] = True
                content.append(item)
            else:
                content.append(b)
        api_messages.append({"role": role, "content": content})
    return api_messages


def to_openai_messages(messages: Iterable[Any], system_prompt: str | None = None) -> list[dict[str, Any]]:
    """Convert neutral messages into OpenAI chat-completions format."""
    api_messages: list[dict[str, Any]] = []
    if system_prompt and system_prompt.strip():
        api_messages.append({"role": "system", "content": system_prompt})

    for role, blocks in normalize_messages(messages):
        texts = [b["text"] for b in blocks if b["type"] == "text"]
        if role == "assistant":
            tool_calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"], ensure_ascii=False)},
                }
                for b in blocks
                if b["type"] == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts else ("" if not tool_calls else None)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            api_messages.append(msg)
            continue

        for b in blocks:
            if b["type"] == "tool_result":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": b["tool_use_id"],
                    "content": tool_result_as_text(b["content"]),
                })
        if texts:
            api_messages.append({"role": "user", "content": "\n\n".join(texts)})
    return api_messages


def build_provider_request(
    messages: Iterable[Any],
    *,
    system_prompt: str | None,
    wire_format: WireFormat,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ``(api_messages, extra_call_kwargs)`` for a provider call.

    The system prompt is included on *every* request: as the ``system`` kwarg for
    Anthropic-format providers and as the leading system message otherwise.
    """
    if wire_format == ANTHROPIC_FORMAT:
        kwargs: dict[str, Any] = {}
        if system_prompt and system_prompt.strip():
            kwargs["system"] = system_prompt
        return to_anthropic_messages(messages), kwargs
    return to_openai_messages(messages, system_prompt), {}
