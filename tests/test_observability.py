"""Structured logging and secret redaction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fakes import FakeOpenAIProvider, reply, tool_call
from src.agent.conversation import Conversation
from src.agent.session import Session
from src.observability import REDACTED, configure_logging, log_event, redact, redact_text
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry

SECRET = "sk-ant-api03-" + "Z" * 40


class RedactionTests(unittest.TestCase):
    def test_sensitive_keys_are_masked(self):
        data = redact({"api_key": "abc", "Authorization": "Bearer x", "password": "p", "nested": {"client_secret": "s"},
                       "input_tokens": 12, "model": "m"})
        self.assertEqual(data["api_key"], REDACTED)
        self.assertEqual(data["Authorization"], REDACTED)
        self.assertEqual(data["password"], REDACTED)
        self.assertEqual(data["nested"]["client_secret"], REDACTED)
        self.assertEqual(data["input_tokens"], 12)
        self.assertEqual(data["model"], "m")

    def test_secret_values_in_text_are_masked(self):
        for text in [
            f"curl -H 'x-api-key: {SECRET}' https://api",
            "export OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz0123",
            "Authorization: Bearer abcdefghijklmnop123",
            "token ghp_" + "a" * 36,
        ]:
            with self.subTest(text=text):
                masked = redact_text(text)
                self.assertIn(REDACTED, masked)
                self.assertNotIn(SECRET, masked)
                self.assertNotIn("abcdefghijklmnop123", masked)

    def test_long_values_truncated(self):
        self.assertLess(len(redact("x" * 10_000)), 2_200)


class StructuredLoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.log = self.root / "logs" / "run.jsonl"
        configure_logging(log_file=self.log)

    def tearDown(self) -> None:
        configure_logging()  # detach file handler before cleanup
        self.tmp.cleanup()

    def events(self) -> list[dict]:
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_log_event_writes_json_lines_without_secrets(self):
        log_event("provider.request", model="m", api_key=SECRET, headers={"x-api-key": SECRET})
        record = self.events()[0]
        self.assertEqual(record["event"], "provider.request")
        self.assertIn("ts", record)
        self.assertNotIn(SECRET, self.log.read_text(encoding="utf-8"))

    def test_agent_run_emits_lifecycle_events(self):
        (self.root / "f.txt").write_text("data", encoding="utf-8")
        provider = FakeOpenAIProvider([
            reply("", [
                tool_call("c1", "Read", {"file_path": str(self.root / "f.txt")}),
                tool_call("c2", "Read", {"file_path": str(self.root / "missing.txt")}),
                tool_call("c3", "Bash", {"command": f"echo {SECRET}"}),
            ]),
            reply("done"),
        ])
        conv = Conversation()
        conv.add_user_message("go")
        run_agent_loop(conv, provider, build_default_registry(include_user_tools=False),
                       ToolContext(workspace_root=self.root), system_prompt="S")
        session = Session.create("fake", "m")
        session.save(self.root / "sessions")
        Session.load(session.session_id, self.root / "sessions")

        names = [e["event"] for e in self.events()]
        for expected in ["agent.started", "provider.request", "provider.response", "tool.started",
                         "tool.completed", "tool.failed", "agent.completed", "session.saved", "session.loaded"]:
            self.assertIn(expected, names)
        self.assertNotIn(SECRET, self.log.read_text(encoding="utf-8"))

    def test_provider_error_is_logged(self):
        from fakes import FailingProvider

        conv = Conversation()
        conv.add_user_message("go")
        with self.assertRaises(RuntimeError):
            run_agent_loop(conv, FailingProvider(RuntimeError("boom")), build_default_registry(include_user_tools=False),
                           ToolContext(workspace_root=self.root), system_prompt="S")
        errors = [e for e in self.events() if e["event"] == "provider.error"]
        self.assertEqual(errors[0]["error"], "boom")


if __name__ == "__main__":
    unittest.main()
