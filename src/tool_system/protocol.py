from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any]
    tool_use_id: Optional[str] = None


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: Any
    is_error: bool = False
    tool_use_id: Optional[str] = None
    content_type: Literal["text", "json"] = "json"
    # Why the call failed, when is_error is True:
    #   "permission" - blocked by permission/safety policy or denied by the user
    #   "input"      - invalid tool input
    #   "execution"  - the tool ran and failed
    error_kind: Optional[Literal["permission", "input", "execution"]] = None

