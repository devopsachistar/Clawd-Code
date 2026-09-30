"""Regression tests for the agent loop's conversation handling (Phase 1).

Covers the bugs found in the audit:
  A. OpenAI/GLM dropped the system prompt after the first turn
  B. OpenAI/GLM tool calls/results were not persisted to the conversation
  C. Anthropic tool results were sent as raw dicts
  D. history truncation could split a tool call from its result
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fakes import FakeAnthropicProvider, FakeOpenAIProvider, reply, tool_call
from src.agent.conversation import Conversation, ToolResultContentBlock, ToolUseContentBlock
from src.tool_system.agent_loop import STOP_COMPLETED, STOP_MAX_TURNS, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry

SYSTEM = "SYSTEM PROMPT MARKER"


class _LoopTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / "a.txt").write_text("alpha", encoding="utf-8")
        (self.root / "b.txt").write_text("beta", encoding="utf-8")
        self.registry = build_default_registry(include_user_tools=False)
        self.context = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_loop(self, conversation, provider, **kwargs):
        kwargs.setdefault("system_prompt", SYSTEM)
        return run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            **kwargs,
        )


class OpenAIConversationTests(_LoopTestCase):
    def test_system_prompt_present_on_every_request(self):
        provider = FakeOpenAIProvider([
            reply("reading", [tool_call("c1", "Read", {"file_path": str(self.root / "a.txt")})]),
            reply("first answer"),
            reply("second answer"),
        ])
        conv = Conversation()
        conv.add_user_message("read a.txt")
        self.run_loop(conv, provider)
        conv.add_user_message("and again")
        self.run_loop(conv, provider)

        self.assertEqual(len(provider.requests), 3)
        for request in provider.requests:
            self.assertEqual(request["messages"][0], {"role": "system", "content": SYSTEM})
            self.assertEqual(sum(1 for m in request["messages"] if m["role"] == "system"), 1)

    def test_tool_calls_and_results_persist_in_conversation(self):
        provider = FakeOpenAIProvider([
            reply("reading", [tool_call("c1", "Read", {"file_path": str(self.root / "a.txt")})]),
            reply("It says alpha."),
        ])
        conv = Conversation()
        conv.add_user_message("read a.txt")
        result = self.run_loop(conv, provider)

        self.assertEqual(result.response_text, "It says alpha.")
        roles = [m.role for m in conv.messages]
        self.assertEqual(roles, ["user", "assistant", "user", "assistant"])
        assistant_blocks = conv.messages[1].content
        self.assertIsInstance(assistant_blocks[1], ToolUseContentBlock)
        self.assertEqual(assistant_blocks[1].id, "c1")
        result_block = conv.messages[2].content[0]
        self.assertIsInstance(result_block, ToolResultContentBlock)
        self.assertEqual(result_block.tool_use_id, "c1")
        self.assertIn("alpha", result_block.content)

    def test_next_user_turn_request_contains_prior_tool_history(self):
        provider = FakeOpenAIProvider([
            reply("", [tool_call("c1", "Read", {"file_path": str(self.root / "a.txt")})]),
            reply("alpha"),
            reply("ok"),
        ])
        conv = Conversation()
        conv.add_user_message("read a.txt")
        self.run_loop(conv, provider)
        conv.add_user_message("thanks")
        self.run_loop(conv, provider)

        last = provider.requests[-1]["messages"]
        roles = [m["role"] for m in last]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(last[2]["tool_calls"][0]["id"], "c1")
        self.assertEqual(last[3]["tool_call_id"], "c1")
        self.assertIn("alpha", last[3]["content"])

    def test_multiple_tool_calls_in_one_turn(self):
        provider = FakeOpenAIProvider([
            reply("", [
                tool_call("c1", "Read", {"file_path": str(self.root / "a.txt")}),
                tool_call("c2", "Read", {"file_path": str(self.root / "b.txt")}),
            ]),
            reply("both read"),
        ])
        conv = Conversation()
        conv.add_user_message("read both")
        result = self.run_loop(conv, provider)
        self.assertEqual(result.tool_calls, 2)

        # Both results recorded in ONE user message, in call order.
        results = conv.messages[2].content
        self.assertEqual([b.tool_use_id for b in results], ["c1", "c2"])
        second_request = provider.requests[1]["messages"]
        self.assertEqual([m.get("tool_call_id") for m in second_request if m["role"] == "tool"], ["c1", "c2"])

    def test_multi_turn_tool_calls(self):
        provider = FakeOpenAIProvider([
            reply("", [tool_call("c1", "Read", {"file_path": str(self.root / "a.txt")})]),
            reply("", [tool_call("c2", "Write", {"file_path": str(self.root / "c.txt"), "content": "gamma"})]),
            reply("", [tool_call("c3", "Read", {"file_path": str(self.root / "c.txt")})]),
            reply("done"),
        ])
        conv = Conversation()
        conv.add_user_message("do three things")
        result = self.run_loop(conv, provider)
        self.assertEqual(result.stop_reason, STOP_COMPLETED)
        self.assertEqual(result.num_turns, 4)
        self.assertEqual(result.tool_calls, 3)
        self.assertEqual((self.root / "c.txt").read_text(encoding="utf-8"), "gamma")
        final_request = provider.requests[-1]["messages"]
        self.assertEqual([m.get("tool_call_id") for m in final_request if m["role"] == "tool"], ["c1", "c2", "c3"])


class AnthropicConversationTests(_LoopTestCase):
    def test_system_kwarg_on_every_request_and_string_tool_results(self):
        provider = FakeAnthropicProvider([
            reply("globbing", [tool_call("t1", "Glob", {"pattern": "*.txt"})]),
            reply("found them"),
        ])
        conv = Conversation()
        conv.add_user_message("find txt files")
        self.run_loop(conv, provider)

        for request in provider.requests:
            self.assertEqual(request["kwargs"].get("system"), SYSTEM)
            self.assertTrue(all(m["role"] in ("user", "assistant") for m in request["messages"]))

        second = provider.requests[1]["messages"]
        self.assertEqual(second[1]["content"][1]["type"], "tool_use")
        tool_result = second[2]["content"][0]
        self.assertEqual(tool_result["type"], "tool_result")
        self.assertEqual(tool_result["tool_use_id"], "t1")
        self.assertIsInstance(tool_result["content"], str)  # was a raw dict before the fix
        self.assertEqual(json.loads(tool_result["content"])["numFiles"], 2)

    def test_multiple_tool_results_share_one_user_message(self):
        provider = FakeAnthropicProvider([
            reply("", [
                tool_call("t1", "Read", {"file_path": str(self.root / "a.txt")}),
                tool_call("t2", "Read", {"file_path": str(self.root / "b.txt")}),
            ]),
            reply("done"),
        ])
        conv = Conversation()
        conv.add_user_message("read both")
        self.run_loop(conv, provider)
        second = provider.requests[1]["messages"]
        self.assertEqual(len(second), 3)
        self.assertEqual([b["tool_use_id"] for b in second[2]["content"]], ["t1", "t2"])

    def test_tool_error_is_marked(self):
        provider = FakeAnthropicProvider([
            reply("", [tool_call("t1", "Read", {"file_path": str(self.root / "missing.txt")})]),
            reply("not found"),
        ])
        conv = Conversation()
        conv.add_user_message("read missing")
        result = self.run_loop(conv, provider)
        self.assertEqual(result.tool_errors, 1)
        self.assertTrue(provider.requests[1]["messages"][2]["content"][0]["is_error"])


class LoopBehaviourTests(_LoopTestCase):
    def test_history_truncation_during_loop_keeps_tool_pairs(self):
        provider = FakeAnthropicProvider([
            reply("", [tool_call(f"t{i}", "Read", {"file_path": str(self.root / "a.txt")})]) for i in range(4)
        ] + [reply("done")])
        conv = Conversation(max_history=4)
        conv.add_user_message("old question")
        conv.add_assistant_message("old answer")
        conv.add_user_message("loop please")
        self.run_loop(conv, provider)
        for request in provider.requests:
            messages = request["messages"]
            self.assertEqual(messages[0]["role"], "user")
            pending: set[str] = set()
            for m in messages:
                blocks = m["content"] if isinstance(m["content"], list) else []
                if m["role"] == "user":
                    for b in blocks:
                        if b["type"] == "tool_result":
                            self.assertIn(b["tool_use_id"], pending)
                    pending = set()
                else:
                    pending = {b["id"] for b in blocks if b["type"] == "tool_use"}

    def test_tool_output_is_truncated_to_tool_limit(self):
        big = self.root / "big.txt"
        big.write_text("x" * 5000, encoding="utf-8")
        provider = FakeOpenAIProvider([
            reply("", [tool_call("b1", "Bash", {"command": "cat big.txt"})]),
            reply("ok"),
        ])
        bash = self.registry.get("Bash")
        original_spec = bash.spec

        from dataclasses import replace

        bash.spec = lambda: replace(original_spec(), max_result_size_chars=500)  # type: ignore[method-assign]
        try:
            conv = Conversation()
            conv.add_user_message("cat")
            self.run_loop(conv, provider)
        finally:
            del bash.spec
        stored = conv.messages[2].content[0].content
        self.assertLessEqual(len(stored), 600)
        self.assertIn("tool output truncated", stored)

    def test_max_turns_reports_stop_reason(self):
        provider = FakeOpenAIProvider([
            reply("", [tool_call(f"c{i}", "Glob", {"pattern": "*"})]) for i in range(3)
        ])
        conv = Conversation()
        conv.add_user_message("loop forever")
        result = self.run_loop(conv, provider, max_turns=3)
        self.assertEqual(result.stop_reason, STOP_MAX_TURNS)
        self.assertEqual(result.num_turns, 3)
        # the final tool results are still recorded (no dangling tool call)
        self.assertEqual(conv.messages[-1].role, "user")

    def test_permission_denials_are_reported(self):
        provider = FakeOpenAIProvider([
            reply("", [
                tool_call("p1", "Bash", {"command": "sudo rm -rf /tmp/x"}),
                tool_call("p2", "Write", {"file_path": str(self.root / ".env"), "content": "A=1"}),
            ]),
            reply("could not"),
        ])
        conv = Conversation()
        conv.add_user_message("do risky things")
        result = self.run_loop(conv, provider)
        self.assertEqual({d["tool"] for d in result.permission_denials}, {"Bash", "Write"})
        self.assertFalse((self.root / ".env").exists())

    def test_explicit_system_prompt_defaults_to_project_context(self):
        provider = FakeOpenAIProvider([reply("hi")])
        conv = Conversation()
        conv.add_user_message("hello")
        run_agent_loop(conv, provider, self.registry, self.context)
        system = provider.requests[0]["messages"][0]["content"]
        self.assertIn("## Runtime Context", system)
        self.assertIn(f"Project root: {self.root}", system)


if __name__ == "__main__":
    unittest.main()
