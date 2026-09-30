"""Conversation management for Clawd Codex."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union
from datetime import datetime


@dataclass
class TextContentBlock:
    """A text content block."""
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseContentBlock:
    """A tool use content block."""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultContentBlock:
    """A tool result content block."""
    type: str = "tool_result"
    tool_use_id: str = ""
    content: Union[str, list[dict[str, Any]]] = ""
    is_error: bool = False


ContentBlock = Union[TextContentBlock, ToolUseContentBlock, ToolResultContentBlock]


@dataclass
class Message:
    """Conversation message."""
    role: str  # "user", "assistant", "system"
    content: Union[str, list[ContentBlock]]
    timestamp: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    # Internal marker for messages that should be filtered from API (e.g., compact boundary)
    _is_internal: bool = field(default=False, repr=False)


def _is_turn_start(message: Message) -> bool:
    """True for a user message that starts a new exchange (not a tool result)."""
    if message.role != "user" or getattr(message, "_is_internal", False):
        return False
    if isinstance(message.content, str):
        return True
    return not any(isinstance(b, ToolResultContentBlock) for b in message.content)


@dataclass
class Conversation:
    """Conversation manager.

    This is the provider-neutral source of truth for a session. Provider wire
    formats are produced by :mod:`src.agent.message_format`.
    """
    messages: list[Message] = field(default_factory=list)
    max_history: int = 100

    def add_message(self, role: str, content: Union[str, list[ContentBlock]]):
        """Add a message to conversation, trimming history safely."""
        self.messages.append(Message(role=role, content=content))
        self._trim_history()

    def _trim_history(self) -> None:
        """Drop the oldest messages beyond ``max_history``.

        Trimming only ever cuts at the start of a user turn, so a tool call is
        never separated from its tool result. If no safe cut point exists (one
        very long tool loop), the history is allowed to exceed the soft limit
        rather than corrupting it.
        """
        if self.max_history <= 0 or len(self.messages) <= self.max_history:
            return
        overflow = len(self.messages) - self.max_history
        for idx in range(overflow, len(self.messages)):
            if _is_turn_start(self.messages[idx]):
                del self.messages[:idx]
                return

    def add_user_message(self, text: str):
        """Add a plain user text message."""
        self.add_message("user", text)

    def add_assistant_message(self, content: Union[str, list[ContentBlock]]):
        """Add an assistant message (text or tool use)."""
        self.add_message("assistant", content)

    def add_tool_result_message(self, tool_use_id: str, content: Any, is_error: bool = False):
        """Add a single tool result message."""
        self.add_tool_results([
            ToolResultContentBlock(type="tool_result", tool_use_id=tool_use_id, content=content, is_error=is_error)
        ])

    def add_tool_results(self, results: list[ToolResultContentBlock]):
        """Add all tool results of one assistant turn as a single user message."""
        if results:
            self.add_message("user", list(results))

    def has_tool_blocks(self) -> bool:
        """Whether any message contains tool_use or tool_result blocks."""
        return any(
            isinstance(b, (ToolUseContentBlock, ToolResultContentBlock))
            for m in self.messages
            if not isinstance(m.content, str)
            for b in m.content
        )

    def get_messages(self) -> list[dict]:
        """Get messages in API format (Anthropic style), with tool pairs repaired."""
        from .message_format import to_anthropic_messages

        return to_anthropic_messages(self.messages)

    def clear(self):
        """Clear conversation."""
        self.messages.clear()

    def to_dict(self) -> dict:
        """Serialize conversation."""
        messages_data = []
        for msg in self.messages:
            if isinstance(msg.content, str):
                content_data = msg.content
            else:
                content_data = []
                for block in msg.content:
                    if isinstance(block, TextContentBlock):
                        content_data.append({"type": "text", "text": block.text})
                    elif isinstance(block, ToolUseContentBlock):
                        content_data.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input
                        })
                    elif isinstance(block, ToolResultContentBlock):
                        content_data.append({
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            "is_error": block.is_error
                        })
            messages_data.append({
                "role": msg.role,
                "content": content_data,
                "timestamp": msg.timestamp,
                "_is_internal": getattr(msg, "_is_internal", False),
            })
        return {
            "messages": messages_data,
            "max_history": self.max_history
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Conversation':
        """Deserialize conversation."""
        conv = cls(max_history=data.get("max_history", 100))
        for msg_data in data.get("messages", []):
            content = msg_data["content"]
            if isinstance(content, str):
                msg_content = content
            else:
                msg_content = []
                for block_data in content:
                    block_type = block_data.get("type")
                    if block_type == "text":
                        msg_content.append(TextContentBlock(type="text", text=block_data.get("text", "")))
                    elif block_type == "tool_use":
                        msg_content.append(ToolUseContentBlock(
                            type="tool_use",
                            id=block_data.get("id", ""),
                            name=block_data.get("name", ""),
                            input=block_data.get("input", {})
                        ))
                    elif block_type == "tool_result":
                        msg_content.append(ToolResultContentBlock(
                            type="tool_result",
                            tool_use_id=block_data.get("tool_use_id", ""),
                            content=block_data.get("content", ""),
                            is_error=block_data.get("is_error", False)
                        ))
            conv.messages.append(Message(
                role=msg_data["role"],
                content=msg_content,
                timestamp=msg_data.get("timestamp", ""),
                _is_internal=msg_data.get("_is_internal", False),
            ))
        return conv
