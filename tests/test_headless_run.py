"""Non-interactive `run` mode: behaviour and exit codes."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fakes import FailingProvider, FakeOpenAIProvider, reply, tool_call
from src.agent.session import Session
from src.cli import main as cli_main
from src.headless import ExitCode, RunOptions, run_headless

REPO_ROOT = Path(__file__).resolve().parent.parent


class _AuthError(Exception):
    pass


_AuthError.__name__ = "AuthenticationError"


class HeadlessRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name).resolve() / "proj"
        self.project.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.project, check=True)
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-not-real-0000", "CLAWD_PROVIDER": "openai"})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def run_task(self, fake, prompt="do the task", **kwargs):
        out, err = io.StringIO(), io.StringIO()
        kwargs.setdefault("cwd", str(self.project))
        outcome = run_headless(
            RunOptions(prompt=prompt, **kwargs),
            stdout=out, stderr=err,
            provider_factory=(lambda resolved: fake) if fake is not None else None,
        )
        return outcome, out.getvalue(), err.getvalue()

    def test_success_exit_code_and_output(self):
        outcome, out, err = self.run_task(FakeOpenAIProvider([reply("All done.")]))
        self.assertEqual(outcome.exit_code, ExitCode.SUCCESS)
        self.assertIn("All done.", out)
        self.assertIn(f"project: {self.project}", err)
        self.assertEqual(outcome.project["git_root"], str(self.project))

    def test_tools_execute_and_session_is_saved(self):
        provider = FakeOpenAIProvider([
            reply("", [tool_call("c1", "Write", {"file_path": str(self.project / "fix.py"), "content": "x = 1\n"})]),
            reply("Created fix.py"),
        ])
        outcome, _, _ = self.run_task(provider)
        self.assertEqual(outcome.exit_code, ExitCode.SUCCESS)
        self.assertEqual((self.project / "fix.py").read_text(encoding="utf-8"), "x = 1\n")
        self.assertEqual(outcome.tool_calls, 1)
        saved = Session.load(outcome.session_id)
        self.assertEqual(saved.metadata["project_root"], str(self.project))
        self.assertEqual([m.role for m in saved.conversation.messages], ["user", "assistant", "user", "assistant"])

    def test_system_prompt_mentions_project_root(self):
        provider = FakeOpenAIProvider([reply("ok")])
        self.run_task(provider)
        system = provider.requests[0]["messages"][0]["content"]
        self.assertIn(f"Project root: {self.project}", system)
        self.assertIn(f"Git root: {self.project}", system)

    def test_permission_rejection_exit_code(self):
        (self.project / "build").mkdir()
        provider = FakeOpenAIProvider([
            reply("", [tool_call("c1", "Bash", {"command": "rm -r build"})]),
            reply("I was not allowed."),
        ])
        outcome, _, err = self.run_task(provider)
        self.assertEqual(outcome.exit_code, ExitCode.PERMISSION_DENIED)
        self.assertTrue((self.project / "build").exists())
        self.assertIn("[denied]", err)
        self.assertEqual(outcome.permission_denials[0]["tool"], "Bash")

    def test_yes_auto_approves_dangerous_but_not_blocked(self):
        (self.project / "build").mkdir()
        provider = FakeOpenAIProvider([
            reply("", [tool_call("c1", "Bash", {"command": "rm -r build"})]),
            reply("", [tool_call("c2", "Bash", {"command": "sudo whoami"})]),
            reply("done"),
        ])
        outcome, _, err = self.run_task(provider, auto_approve=True)
        self.assertFalse((self.project / "build").exists())
        self.assertIn("[auto-approved]", err)
        self.assertEqual(outcome.exit_code, ExitCode.PERMISSION_DENIED)  # sudo is still refused
        self.assertEqual([d["tool"] for d in outcome.permission_denials], ["Bash"])

    def test_max_turns_is_task_failure(self):
        provider = FakeOpenAIProvider([reply("", [tool_call(f"c{i}", "Glob", {"pattern": "*"})]) for i in range(2)])
        outcome, _, _ = self.run_task(provider, max_turns=2)
        self.assertEqual(outcome.exit_code, ExitCode.TASK_FAILURE)
        self.assertEqual(outcome.stop_reason, "max_turns")

    def test_provider_error_is_task_failure(self):
        outcome, _, err = self.run_task(FailingProvider(RuntimeError("upstream exploded")))
        self.assertEqual(outcome.exit_code, ExitCode.TASK_FAILURE)
        self.assertIn("upstream exploded", err)

    def test_auth_error_is_config_error(self):
        outcome, _, _ = self.run_task(FailingProvider(_AuthError("invalid x-api-key")))
        self.assertEqual(outcome.exit_code, ExitCode.CONFIG_ERROR)

    def test_missing_api_key_is_config_error(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            outcome, _, _ = self.run_task(None)
        self.assertEqual(outcome.exit_code, ExitCode.CONFIG_ERROR)
        self.assertIn("OPENAI_API_KEY", outcome.error)

    def test_unknown_provider_is_config_error(self):
        outcome, _, _ = self.run_task(None, provider="nope")
        self.assertEqual(outcome.exit_code, ExitCode.CONFIG_ERROR)

    def test_empty_prompt_is_config_error(self):
        outcome, _, _ = self.run_task(FakeOpenAIProvider([]), prompt="  ")
        self.assertEqual(outcome.exit_code, ExitCode.CONFIG_ERROR)

    def test_refuses_home_directory_as_project(self):
        outcome, _, _ = self.run_task(FakeOpenAIProvider([reply("x")]), cwd=str(Path.home()))
        self.assertEqual(outcome.exit_code, ExitCode.CONFIG_ERROR)
        self.assertIn("home directory", outcome.error)

    def test_json_output(self):
        outcome, out, _ = self.run_task(FakeOpenAIProvider([reply("json answer")]), output="json")
        data = json.loads(out)
        self.assertEqual(data["exit_code"], 0)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["response"], "json answer")
        self.assertEqual(data["project"]["project_root"], str(self.project))
        self.assertNotIn("sk-test-not-real", out)

    def test_resume_latest_continues_conversation(self):
        first, _, _ = self.run_task(FakeOpenAIProvider([reply("first answer")]), prompt="first question")
        provider = FakeOpenAIProvider([reply("second answer")])
        second, _, _ = self.run_task(provider, prompt="second question", resume="latest")
        self.assertEqual(second.session_id, first.session_id)
        contents = [m["content"] for m in provider.requests[0]["messages"]]
        self.assertEqual(contents[1:], ["first question", "first answer", "second question"])

    def test_resume_unknown_session_is_config_error(self):
        outcome, _, _ = self.run_task(FakeOpenAIProvider([]), resume="20990101_000000_ffffff")
        self.assertEqual(outcome.exit_code, ExitCode.CONFIG_ERROR)

    def test_no_save(self):
        outcome, _, _ = self.run_task(FakeOpenAIProvider([reply("x")]), save_session=False)
        self.assertIsNone(outcome.session_path)
        self.assertIsNone(Session.load(outcome.session_id))

    def test_stream_writes_text_to_stdout(self):
        outcome, out, _ = self.run_task(FakeOpenAIProvider([reply("streamed text")]), stream=True)
        self.assertIn("streamed text", out)


class CliRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name).resolve()
        (self.project / "pyproject.toml").write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_cli_run_success_via_main(self):
        provider = FakeOpenAIProvider([reply("cli ok")])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-0000"}), \
                patch("src.headless.create_provider", return_value=provider), \
                patch("sys.stdout", new_callable=io.StringIO) as out, \
                patch("sys.stderr", new_callable=io.StringIO):
            code = cli_main(["run", "--provider", "openai", "--cwd", str(self.project), "say", "hi"])
        self.assertEqual(code, 0)
        self.assertIn("cli ok", out.getvalue())
        self.assertEqual(provider.requests[0]["messages"][-1]["content"], "say hi")

    def test_cli_run_missing_key_exit_2(self):
        with patch("sys.stdout", new_callable=io.StringIO), patch("sys.stderr", new_callable=io.StringIO) as err:
            code = cli_main(["run", "--provider", "openai", "--cwd", str(self.project), "hi"])
        self.assertEqual(code, 2)
        self.assertIn("OPENAI_API_KEY", err.getvalue())

    def test_cli_subprocess_exit_code_and_help(self):
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        env["CLAWD_HOME"] = str(self.project / "home")
        help_proc = subprocess.run([sys.executable, "-m", "src.cli", "run", "--help"], cwd=REPO_ROOT,
                                   capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(help_proc.returncode, 0)
        self.assertIn("Exit codes", help_proc.stdout)
        proc = subprocess.run(
            [sys.executable, "-m", "src.cli", "run", "--provider", "openai", "--cwd", str(self.project), "hello"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=env, timeout=120,
        )
        self.assertEqual(proc.returncode, 2, proc.stderr)


if __name__ == "__main__":
    unittest.main()
