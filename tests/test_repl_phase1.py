"""REPL behaviour added/fixed in Phase 1."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fakes import FakeAnthropicProvider, reply
from src.agent.conversation import ToolResultContentBlock, ToolUseContentBlock
from src.agent.session import Session
from src.config import resolve_provider_config
from src.project import detect_project
from src.repl import ClawdREPL


class ReplPhase1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.tmp.name).resolve() / "repo"
        (self.project_dir / "pkg").mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=self.project_dir, check=True)
        self.project = detect_project(self.project_dir / "pkg")
        self.resolved = resolve_provider_config(
            cli_provider="anthropic", cli_model="test-model", project_root=self.project.project_root,
            env={"ANTHROPIC_API_KEY": "test-key-000000"},
        )
        self.fake = FakeAnthropicProvider([])
        patcher = patch("src.repl.core.get_provider_class", return_value=lambda **kw: self.fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def make_repl(self, **kwargs) -> ClawdREPL:
        repl = ClawdREPL(resolved=self.resolved, project=self.project, **kwargs)
        repl.console.print = Mock()
        return repl

    def test_construction_does_not_require_a_terminal(self):
        with patch("src.repl.core.PromptSession") as prompt_session:
            repl = self.make_repl()
            repl._refresh_completer()
            prompt_session.assert_not_called()
            _ = repl.prompt_session
            prompt_session.assert_called_once()

    def test_uses_resolved_config_and_project_root(self):
        repl = self.make_repl()
        self.assertEqual(repl.provider_name, "anthropic")
        self.assertEqual(repl.tool_context.workspace_root, self.project_dir)
        self.assertEqual(repl.tool_context.cwd, self.project_dir / "pkg")
        self.assertEqual(repl.session.metadata["git_root"], str(self.project_dir))

    def test_chat_autosaves_session_with_tool_history(self):
        repl = self.make_repl(autosave=True)
        (self.project_dir / "a.txt").write_text("alpha", encoding="utf-8")
        self.fake.queue(
            reply("", [{"id": "t1", "name": "Read", "input": {"file_path": str(self.project_dir / "a.txt")}}]),
            reply("It says alpha."),
        )
        repl.chat("read a.txt")
        saved = Session.load(repl.session.session_id)
        self.assertIsNotNone(saved)
        blocks = saved.conversation.messages[1].content
        self.assertIsInstance(blocks[0], ToolUseContentBlock)
        self.assertIsInstance(saved.conversation.messages[2].content[0], ToolResultContentBlock)
        self.assertEqual(saved.state["cwd"], str(self.project_dir / "pkg"))

    def test_autosave_can_be_disabled(self):
        repl = self.make_repl(autosave=False)
        self.fake.queue(reply("hi"))
        repl.chat("hello")
        self.assertIsNone(Session.load(repl.session.session_id))

    def test_load_session_rebinds_command_context_and_restores_state(self):
        repl = self.make_repl(autosave=False)
        other = Session.create("anthropic", "test-model", metadata=self.project.to_dict())
        other.conversation.add_user_message("earlier question")
        other.state = {"cwd": str(self.project_dir)}
        other.save()
        repl.load_session(other.session_id)
        self.assertIs(repl.command_context.conversation, repl.session.conversation)
        self.assertEqual(repl.tool_context.cwd, self.project_dir)

    def test_resume_loads_latest_session_for_project(self):
        previous = Session.create("anthropic", "test-model", metadata=self.project.to_dict())
        previous.conversation.add_user_message("from yesterday")
        previous.save()
        repl = self.make_repl(autosave=False)
        repl.handle_command("/resume")
        self.assertEqual(repl.session.session_id, previous.session_id)

    def test_direct_stream_skipped_when_history_has_tool_blocks(self):
        repl = self.make_repl(stream=True)
        self.assertTrue(repl._should_try_direct_stream("hello there"))
        repl.session.conversation.add_user_message("q")
        repl.session.conversation.add_assistant_message([ToolUseContentBlock(id="x", name="Glob", input={})])
        repl.session.conversation.add_tool_result_message("x", "ok")
        self.assertFalse(repl._should_try_direct_stream("hello there"))

    def test_direct_stream_payload_uses_wire_format(self):
        repl = self.make_repl()
        repl.session.conversation.add_user_message("hi")
        messages, kwargs = repl._build_direct_stream_payload()
        self.assertEqual(messages, [{"role": "user", "content": "hi"}])
        self.assertIn("system", kwargs)  # anthropic-format: system kwarg, not a message


class InteractiveStartupTests(unittest.TestCase):
    """Drive the real `clawd` entry point through prompt_toolkit's pipe input."""

    def test_cli_starts_repl_runs_commands_and_exits(self):
        import io
        import os

        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        from src.cli import main

        with create_pipe_input() as pipe, \
                patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-not-real", "CLAWD_PROVIDER": "anthropic"}), \
                patch("sys.stdout", new_callable=io.StringIO) as out:
            pipe.send_text("/tools\r/exit\r")
            with create_app_session(input=pipe, output=DummyOutput()):
                with self.assertRaises(SystemExit) as exit_info:
                    main([])
        self.assertEqual(exit_info.exception.code, 0)
        rendered = out.getvalue()
        self.assertIn("CLAWD CODE", rendered)
        self.assertIn("Project", rendered)
        self.assertIn("PowerShell", rendered)  # /tools listing
        self.assertIn("Goodbye", rendered)

    def test_repl_without_api_key_exits_with_config_error(self):
        import io

        from src.cli import main

        with patch("sys.stdout", new_callable=io.StringIO) as out:
            code = main(["--provider", "openai"])
        self.assertEqual(code, 2)
        self.assertIn("OPENAI_API_KEY", out.getvalue())


if __name__ == "__main__":
    unittest.main()
