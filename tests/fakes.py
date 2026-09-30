"""Scripted fake providers for agent-loop / session / headless tests.

Unlike MagicMock, these are real ``BaseProvider`` subclasses with a declared
wire format, and they record a deep copy of every request so tests can assert
exactly what would have been sent to the API.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

from src.providers.base import BaseProvider, ChatResponse


def tool_call(tool_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"id": tool_id, "name": name, "input": tool_input}


def reply(text: str = "", tool_uses: list[dict[str, Any]] | None = None) -> ChatResponse:
    return ChatResponse(
        content=text,
        model="fake-model",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="tool_use" if tool_uses else "stop",
        tool_uses=tool_uses,
    )


class _ScriptedProvider(BaseProvider):
    def __init__(self, responses: Iterable[ChatResponse] = (), model: str = "fake-model"):
        super().__init__(api_key="test-key", base_url=None, model=model)
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def queue(self, *responses: ChatResponse) -> None:
        self.responses.extend(responses)

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        self.requests.append({
            "messages": copy.deepcopy(list(messages)),
            "tools": tools,
            "kwargs": copy.deepcopy(kwargs),
        })
        if not self.responses:
            raise AssertionError("FakeProvider ran out of scripted responses")
        return self.responses.pop(0)

    def chat_stream(self, messages, tools=None, **kwargs):
        yield self.chat(messages, tools=tools, **kwargs).content

    def get_available_models(self) -> list[str]:
        return ["fake-model"]


class FakeOpenAIProvider(_ScriptedProvider):
    message_format = "openai"


class FakeAnthropicProvider(_ScriptedProvider):
    message_format = "anthropic"


class FailingProvider(_ScriptedProvider):
    message_format = "openai"

    def __init__(self, exc: BaseException):
        super().__init__()
        self.exc = exc

    def chat(self, messages, tools=None, **kwargs):
        raise self.exc
