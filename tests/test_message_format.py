"""Provider-specific message conversion from the neutral conversation."""

from __future__ import annotations

import json
import unittest

from src.agent.conversation import (
    Conversation,
    Message,
    TextContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from src.agent.message_format import (
    INTERRUPTED_TOOL_RESULT,
    build_provider_request,
    resolve_wire_format,
    to_anthropic_messages,
    to_openai_messages,
)
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.glm_provider import GLMProvider
from src.providers.minimax_provider import MinimaxProvider
from src.providers.openai_provider import OpenAIProvider


def _tool_conversation() -> Conversation:
    conv = Conversation()
    conv.add_user_message("List files and read README")
    conv.add_assistant_message([
        TextContentBlock(text="Let me look."),
        ToolUseContentBlock(id="call_1", name="Glob", input={"pattern": "*"}),
        ToolUseContentBlock(id="call_2", name="Read", input={"file_path": "README.md"}),
    ])
    conv.add_tool_results([
        ToolResultContentBlock(tool_use_id="call_1", content={"numFiles": 2, "filenames": ["a", "b"]}),
        ToolResultContentBlock(tool_use_id="call_2", content="hello", is_error=False),
    ])
    conv.add_assistant_message("Done.")
    return conv


class WireFormatResolutionTests(unittest.TestCase):
    def test_real_providers_declare_their_format(self):
        self.assertEqual(AnthropicProvider.message_format, "anthropic")
        self.assertEqual(MinimaxProvider.message_format, "anthropic")
        self.assertEqual(OpenAIProvider.message_format, "openai")
        self.assertEqual(GLMProvider.message_format, "openai")

    def test_unknown_objects_default_to_openai(self):
        self.assertEqual(resolve_wire_format(object()), "openai")


class AnthropicConversionTests(unittest.TestCase):
    def test_tool_result_dict_is_serialised_to_text(self):
        messages = to_anthropic_messages(_tool_conversation().messages)
        tool_results = messages[2]["content"]
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(tool_results[0]["type"], "tool_result")
        self.assertIsInstance(tool_results[0]["content"], str)
        self.assertEqual(json.loads(tool_results[0]["content"]), {"numFiles": 2, "filenames": ["a", "b"]})
        self.assertEqual(tool_results[1]["content"], "hello")

    def test_assistant_tool_use_blocks_are_preserved(self):
        messages = to_anthropic_messages(_tool_conversation().messages)
        blocks = messages[1]["content"]
        self.assertEqual([b["type"] for b in blocks], ["text", "tool_use", "tool_use"])
        self.assertEqual(blocks[1], {"type": "tool_use", "id": "call_1", "name": "Glob", "input": {"pattern": "*"}})

    def test_separate_tool_result_messages_are_merged_with_results_first(self):
        conv = Conversation()
        conv.add_user_message("go")
        conv.add_assistant_message([
            ToolUseContentBlock(id="a", name="Read", input={}),
            ToolUseContentBlock(id="b", name="Read", input={}),
        ])
        conv.add_tool_result_message("a", "one")
        conv.add_message("user", "extra instruction")
        conv.add_tool_result_message("b", "two")
        messages = to_anthropic_messages(conv.messages)
        self.assertEqual(len(messages), 3)
        types = [b["type"] for b in messages[2]["content"]]
        self.assertEqual(types, ["tool_result", "tool_result", "text"])

    def test_error_flag_only_sent_for_errors(self):
        conv = Conversation()
        conv.add_user_message("go")
        conv.add_assistant_message([ToolUseContentBlock(id="a", name="Bash", input={})])
        conv.add_tool_result_message("a", "Error: boom", is_error=True)
        result = to_anthropic_messages(conv.messages)[2]["content"][0]
        self.assertTrue(result["is_error"])

    def test_internal_and_empty_messages_are_not_sent(self):
        conv = Conversation()
        conv.messages.append(Message(role="user", content="hi"))
        conv.messages.append(Message(role="assistant", content=""))
        conv.messages.append(Message(role="user", content="internal", _is_internal=True))
        conv.messages.append(Message(role="user", content="again"))
        messages = to_anthropic_messages(conv.messages)
        self.assertEqual(messages, [{"role": "user", "content": [
            {"type": "text", "text": "hi"}, {"type": "text", "text": "again"},
        ]}])

    def test_image_blocks_pass_through(self):
        image = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "xx"}}]
        conv = Conversation()
        conv.add_user_message("see")
        conv.add_assistant_message([ToolUseContentBlock(id="a", name="Read", input={})])
        conv.add_tool_result_message("a", image)
        self.assertEqual(to_anthropic_messages(conv.messages)[2]["content"][0]["content"], image)


class OpenAIConversionTests(unittest.TestCase):
    def test_system_prompt_tool_calls_and_tool_messages(self):
        messages = to_openai_messages(_tool_conversation().messages, "SYSTEM")
        self.assertEqual(messages[0], {"role": "system", "content": "SYSTEM"})
        self.assertEqual(messages[1], {"role": "user", "content": "List files and read README"})
        assistant = messages[2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"], "Let me look.")
        self.assertEqual([c["id"] for c in assistant["tool_calls"]], ["call_1", "call_2"])
        self.assertEqual(json.loads(assistant["tool_calls"][1]["function"]["arguments"]), {"file_path": "README.md"})
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call_1")
        self.assertEqual(json.loads(messages[3]["content"])["numFiles"], 2)
        self.assertEqual(messages[4], {"role": "tool", "tool_call_id": "call_2", "content": "hello"})
        self.assertEqual(messages[5], {"role": "assistant", "content": "Done."})

    def test_assistant_with_only_tool_calls_has_null_content(self):
        conv = Conversation()
        conv.add_user_message("go")
        conv.add_assistant_message([ToolUseContentBlock(id="x", name="Glob", input={})])
        conv.add_tool_result_message("x", "ok")
        messages = to_openai_messages(conv.messages)
        self.assertIsNone(messages[1]["content"])

    def test_no_system_message_when_prompt_empty(self):
        conv = Conversation()
        conv.add_user_message("hi")
        self.assertEqual(to_openai_messages(conv.messages, "  "), [{"role": "user", "content": "hi"}])


class ToolPairRepairTests(unittest.TestCase):
    def test_orphan_tool_result_is_dropped(self):
        conv = Conversation()
        conv.add_tool_result_message("ghost", "stale")
        conv.add_user_message("hello")
        for messages in (to_anthropic_messages(conv.messages), to_openai_messages(conv.messages)):
            self.assertEqual(len(messages), 1)
            self.assertNotIn("ghost", json.dumps(messages))

    def test_missing_tool_result_is_synthesised(self):
        conv = Conversation()
        conv.add_user_message("go")
        conv.add_assistant_message([ToolUseContentBlock(id="lost", name="Bash", input={"command": "ls"})])
        # run interrupted: no result recorded
        anthropic = to_anthropic_messages(conv.messages)
        self.assertEqual(anthropic[-1]["role"], "user")
        self.assertEqual(anthropic[-1]["content"][0]["tool_use_id"], "lost")
        self.assertEqual(anthropic[-1]["content"][0]["content"], INTERRUPTED_TOOL_RESULT)
        openai = to_openai_messages(conv.messages)
        self.assertEqual(openai[-1], {"role": "tool", "tool_call_id": "lost", "content": INTERRUPTED_TOOL_RESULT})

    def test_missing_result_synthesised_before_next_user_text(self):
        conv = Conversation()
        conv.add_user_message("go")
        conv.add_assistant_message([ToolUseContentBlock(id="t1", name="Bash", input={})])
        conv.add_user_message("never mind")
        messages = to_anthropic_messages(conv.messages)
        self.assertEqual([b["type"] for b in messages[2]["content"]], ["tool_result", "text"])


class OpenAIToolSchemaTests(unittest.TestCase):
    def test_every_default_tool_converts_for_openai(self):
        from src.providers.openai_compatible import _convert_to_openai_tool_schema
        from src.tool_system.defaults import build_default_registry

        for spec in build_default_registry(include_user_tools=False).list_specs():
            with self.subTest(tool=spec.name):
                converted = _convert_to_openai_tool_schema(
                    {"name": spec.name, "description": spec.description, "input_schema": dict(spec.input_schema)}
                )
                self.assertIsNotNone(converted, f"{spec.name} would be silently dropped for OpenAI/GLM")
                self.assertEqual(converted["function"]["parameters"]["type"], "object")

    def test_union_schema_is_merged(self):
        from src.providers.openai_compatible import _convert_to_openai_tool_schema

        tool = {"name": "Skill", "input_schema": {"anyOf": [
            {"type": "object", "properties": {"skill": {"type": "string"}, "args": {"type": "string"}}, "required": ["skill"]},
            {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        ]}}
        params = _convert_to_openai_tool_schema(tool)["function"]["parameters"]
        self.assertEqual(set(params["properties"]), {"skill", "args", "name"})
        self.assertNotIn("required", params)


class BuildProviderRequestTests(unittest.TestCase):
    def test_anthropic_uses_system_kwarg(self):
        msgs, kwargs = build_provider_request(_tool_conversation().messages, system_prompt="SYS", wire_format="anthropic")
        self.assertEqual(kwargs, {"system": "SYS"})
        self.assertTrue(all(m["role"] != "system" for m in msgs))

    def test_openai_puts_system_first(self):
        msgs, kwargs = build_provider_request(_tool_conversation().messages, system_prompt="SYS", wire_format="openai")
        self.assertEqual(kwargs, {})
        self.assertEqual(msgs[0], {"role": "system", "content": "SYS"})


if __name__ == "__main__":
    unittest.main()
