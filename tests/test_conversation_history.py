"""History truncation must never separate a tool call from its result."""

from __future__ import annotations

import unittest

from src.agent.conversation import Conversation, TextContentBlock, ToolResultContentBlock, ToolUseContentBlock


def _add_tool_exchange(conv: Conversation, tool_id: str) -> None:
    conv.add_assistant_message([ToolUseContentBlock(id=tool_id, name="Read", input={})])
    conv.add_tool_results([ToolResultContentBlock(tool_use_id=tool_id, content="ok")])


class HistoryTruncationTests(unittest.TestCase):
    def test_plain_messages_keep_last_n(self):
        conv = Conversation(max_history=3)
        for i in range(5):
            conv.add_message("user", f"Message {i}")
        self.assertEqual([m.content for m in conv.messages], ["Message 2", "Message 3", "Message 4"])

    def test_trim_cuts_only_at_user_turn_start(self):
        conv = Conversation(max_history=5)
        conv.add_user_message("q1")
        _add_tool_exchange(conv, "a")
        conv.add_assistant_message("a1")
        conv.add_user_message("q2")
        _add_tool_exchange(conv, "b")
        # 7 messages > 5: the only safe cut is at "q2"
        self.assertEqual(conv.messages[0].content, "q2")
        self.assertIsInstance(conv.messages[1].content[0], ToolUseContentBlock)
        self.assertIsInstance(conv.messages[2].content[0], ToolResultContentBlock)

    def test_never_starts_with_orphan_tool_result(self):
        conv = Conversation(max_history=4)
        conv.add_user_message("q")
        for i in range(6):
            _add_tool_exchange(conv, f"t{i}")
            first = conv.messages[0]
            self.assertEqual(first.role, "user")
            self.assertFalse(
                not isinstance(first.content, str)
                and any(isinstance(b, ToolResultContentBlock) for b in first.content)
            )

    def test_soft_limit_when_no_safe_cut_point(self):
        conv = Conversation(max_history=3)
        conv.add_user_message("single long task")
        for i in range(4):
            _add_tool_exchange(conv, f"t{i}")
        # No later user turn exists, so nothing is dropped (never corrupt pairs).
        self.assertEqual(len(conv.messages), 9)
        self.assertEqual(conv.messages[0].content, "single long task")

    def test_api_messages_are_valid_after_trimming(self):
        conv = Conversation(max_history=6)
        for i in range(5):
            conv.add_user_message(f"q{i}")
            _add_tool_exchange(conv, f"t{i}")
            conv.add_assistant_message([TextContentBlock(text=f"a{i}")])
        api = conv.get_messages()
        self.assertEqual(api[0]["role"], "user")
        seen_tool_ids = {b["id"] for m in api if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_use"}
        result_ids = {b["tool_use_id"] for m in api if isinstance(m["content"], list) for b in m["content"] if b["type"] == "tool_result"}
        self.assertEqual(seen_tool_ids, result_ids)

    def test_has_tool_blocks(self):
        conv = Conversation()
        conv.add_user_message("hi")
        self.assertFalse(conv.has_tool_blocks())
        _add_tool_exchange(conv, "x")
        self.assertTrue(conv.has_tool_blocks())


if __name__ == "__main__":
    unittest.main()
