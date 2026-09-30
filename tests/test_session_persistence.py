"""Session persistence: save -> restart -> reload -> continue correctly."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fakes import FakeAnthropicProvider, FakeOpenAIProvider, reply, tool_call
from src.agent.conversation import TextContentBlock, ToolResultContentBlock, ToolUseContentBlock
from src.agent.session import (
    SESSION_SCHEMA_VERSION,
    Session,
    capture_tool_state,
    latest_session_id,
    list_sessions,
    restore_tool_state,
)
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


class SessionRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name).resolve()
        self.project = base / "project"
        self.project.mkdir()
        (self.project / "notes.txt").write_text("the answer is 42", encoding="utf-8")
        self.sessions = base / "sessions"
        self.registry = build_default_registry(include_user_tools=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, session, provider, context):
        return run_agent_loop(session.conversation, provider, self.registry, context, system_prompt="SYS")

    def _full_cycle(self, provider_cls):
        # 1. create session, send message, execute a tool
        session = Session.create("fake", "fake-model", metadata={"project_root": str(self.project)})
        context = ToolContext(workspace_root=self.project)
        provider = provider_cls([
            reply("Reading notes.", [tool_call("call_1", "Read", {"file_path": str(self.project / "notes.txt")})]),
            reply("The notes say 42."),
        ])
        session.conversation.add_user_message("What do the notes say?")
        self._run(session, provider, context)
        session.state = capture_tool_state(context)
        path = session.save(self.sessions)

        # 2. "restart": load from disk into fresh objects
        restored = Session.load(session.session_id, self.sessions)
        self.assertIsNotNone(restored)
        new_context = ToolContext(workspace_root=self.project)
        restore_tool_state(new_context, restored.state)

        # 3. continue the conversation
        provider2 = provider_cls([reply("You asked about the notes.")])
        restored.conversation.add_user_message("What did I ask before?")
        result = self._run(restored, provider2, new_context)
        return session, restored, path, provider2, result

    def test_openai_continuation_sees_full_tool_history(self):
        _, restored, path, provider, result = self._full_cycle(FakeOpenAIProvider)
        self.assertEqual(result.response_text, "You asked about the notes.")
        messages = provider.requests[0]["messages"]
        self.assertEqual([m["role"] for m in messages],
                         ["system", "user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "Read")
        self.assertIn("42", messages[3]["content"])
        self.assertEqual(messages[-1]["content"], "What did I ask before?")

    def test_anthropic_continuation_sees_full_tool_history(self):
        _, restored, path, provider, _ = self._full_cycle(FakeAnthropicProvider)
        messages = provider.requests[0]["messages"]
        self.assertEqual(provider.requests[0]["kwargs"]["system"], "SYS")
        self.assertEqual(messages[1]["content"][1]["type"], "tool_use")
        self.assertEqual(messages[2]["content"][0]["type"], "tool_result")
        self.assertEqual(messages[2]["content"][0]["tool_use_id"], "call_1")

    def test_saved_file_contains_all_parts(self):
        session, restored, path, _, _ = self._full_cycle(FakeOpenAIProvider)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], SESSION_SCHEMA_VERSION)
        self.assertEqual(data["metadata"]["project_root"], str(self.project))
        self.assertIn("platform", data["metadata"])
        self.assertEqual(data["state"]["cwd"], str(self.project))
        msgs = data["conversation"]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])
        self.assertTrue(all(m["timestamp"] for m in msgs))
        self.assertEqual(msgs[1]["content"][1]["type"], "tool_use")
        self.assertEqual(msgs[2]["content"][0]["type"], "tool_result")

        # restored objects have the proper block types
        blocks = restored.conversation.messages[1].content
        self.assertIsInstance(blocks[0], TextContentBlock)
        self.assertIsInstance(blocks[1], ToolUseContentBlock)
        self.assertIsInstance(restored.conversation.messages[2].content[0], ToolResultContentBlock)
        self.assertEqual(restored.created_at, session.created_at)

    def test_tool_state_is_restored(self):
        sub = self.project / "sub"
        sub.mkdir()
        context = ToolContext(workspace_root=self.project)
        context.cwd = sub
        context.todos = [{"content": "write tests", "status": "pending"}]
        state = capture_tool_state(context)
        fresh = ToolContext(workspace_root=self.project)
        warnings = restore_tool_state(fresh, json.loads(json.dumps(state)))
        self.assertEqual(warnings, [])
        self.assertEqual(fresh.cwd, sub)
        self.assertEqual(fresh.todos, context.todos)

    def test_restore_ignores_cwd_outside_workspace(self):
        other = Path(self.tmp.name).resolve() / "elsewhere"
        other.mkdir()
        fresh = ToolContext(workspace_root=self.project)
        warnings = restore_tool_state(fresh, {"cwd": str(other)})
        self.assertEqual(fresh.cwd, self.project)
        self.assertTrue(warnings)


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_session_ids_are_unique(self):
        ids = {Session.create("p", "m").session_id for _ in range(50)}
        self.assertEqual(len(ids), 50)

    def test_invalid_session_id_is_rejected(self):
        self.assertIsNone(Session.load("../../etc/passwd", self.dir))
        s = Session.create("p", "m")
        s.session_id = "../escape"
        with self.assertRaises(ValueError):
            s.save(self.dir)

    def test_missing_session_returns_none(self):
        self.assertIsNone(Session.load("does_not_exist", self.dir))

    def test_unicode_and_non_json_values_survive(self):
        s = Session.create("p", "m")
        s.conversation.add_user_message("héllo 你好")
        s.conversation.add_tool_result_message("x", {"path": Path("a/b")})
        s.save(self.dir)
        loaded = Session.load(s.session_id, self.dir)
        self.assertEqual(loaded.conversation.messages[0].content, "héllo 你好")

    def test_legacy_v1_file_loads(self):
        legacy = {
            "session_id": "20250101_120000",
            "provider": "glm",
            "model": "glm-4.5",
            "conversation": {"messages": [{"role": "user", "content": "hi", "timestamp": "t"}], "max_history": 100},
            "created_at": "2025-01-01T12:00:00",
            "updated_at": "2025-01-01T12:00:00",
        }
        (self.dir / "20250101_120000.json").write_text(json.dumps(legacy), encoding="utf-8")
        loaded = Session.load("20250101_120000", self.dir)
        self.assertEqual(loaded.conversation.messages[0].content, "hi")
        self.assertEqual(loaded.metadata, {})
        self.assertEqual(loaded.state, {})

    def test_list_and_latest_filtered_by_project(self):
        a = Session.create("p", "m", metadata={"project_root": str(Path("/proj/a").resolve())})
        a.save(self.dir)
        b = Session.create("p", "m", metadata={"project_root": str(Path("/proj/b").resolve())})
        b.save(self.dir)
        self.assertEqual(len(list_sessions(self.dir)), 2)
        self.assertEqual(latest_session_id(self.dir, project_root="/proj/a"), a.session_id)
        self.assertIsNone(latest_session_id(self.dir, project_root="/proj/none"))

    def test_default_location_is_under_user_dir(self):
        s = Session.create("p", "m")
        path = s.save()
        self.assertEqual(path.parent, Path.home() / ".clawd" / "sessions")  # Path.home is sandboxed by conftest

    def test_no_temp_files_left_behind(self):
        s = Session.create("p", "m")
        s.save(self.dir)
        s.save(self.dir)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), [f"{s.session_id}.json"])


if __name__ == "__main__":
    unittest.main()
