"""Secret values must not enter model messages, logs, sessions or reports."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from fakes import FakeAnthropicProvider, FakeOpenAIProvider, reply, tool_call
from src.agent.conversation import Conversation
from src.agent.session import Session
from src.context_system import build_context_prompt
from src.observability import configure_logging, log_event
from src.security.secrets import (
    SECRET_PLACEHOLDER,
    get_secret_registry,
    redact_secret_file,
    redact_secrets,
    register_project_secrets,
)
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall

DB_PASSWORD = "Sup3r-S3cret-DB-pw-9981"
DATABASE_URL = f"postgres://app:{DB_PASSWORD}@db.internal:5432/prod"
# Synthetic Stripe-shaped fixtures, assembled at runtime so no complete
# credential-like literal is ever committed (GitHub push protection scans for
# these prefixes). At runtime they have exactly the shape the production
# detector in src/security/secrets.py matches, so the tests exercise it for real.
STRIPE_KEY = "sk_" + "live_" + "FAKE0TEST0ONLY0NOT0A0REAL0KEY"
WEBHOOK = "whsec_" + "FAKE0TEST0ONLY0WEBHOOK0SECRET"
ALL_SECRETS = (DB_PASSWORD, DATABASE_URL, STRIPE_KEY, WEBHOOK)
PY = f'"{sys.executable}"'


def assert_no_secrets(test: unittest.TestCase, text: str, where: str) -> None:
    for secret in ALL_SECRETS:
        test.assertNotIn(secret, text, f"secret leaked into {where}")


class SecretProjectCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        (self.root / ".env").write_text(
            f"# app config\nDATABASE_URL={DATABASE_URL}\nSTRIPE_SECRET_KEY={STRIPE_KEY}\n"
            f"STRIPE_WEBHOOK_SECRET=\"{WEBHOOK}\"\nDEBUG=true\nAPP_NAME=tasks\n",
            encoding="utf-8",
        )
        (self.root / "settings.py").write_text(
            "import os\nDATABASE_URL = os.environ['DATABASE_URL']\nPAGE_SIZE = 20\n", encoding="utf-8"
        )
        self.registry = build_default_registry(include_user_tools=False)
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def dispatch(self, name, tool_input, approve=None):
        if approve is not None:
            self.ctx.approval_handler = lambda request: approve
        return self.registry.dispatch(ToolCall(name, tool_input, "t1"), self.ctx)


class RedactionUnitTests(unittest.TestCase):
    def test_model_visible_form(self):
        text = f"DATABASE_URL={DATABASE_URL}\nSTRIPE_SECRET_KEY={STRIPE_KEY}"
        self.assertEqual(redact_secrets(text), "DATABASE_URL=<SECRET>\nSTRIPE_SECRET_KEY=<SECRET>")

    def test_patterns(self):
        for text in [f"key={STRIPE_KEY}", f"url {DATABASE_URL}", "Authorization: Bearer abcdef0123456789abcdef",
                     "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----",
                     "token ghp_" + "b" * 36, "redis://:p4ssw0rdXYZ@localhost:6379"]:
            with self.subTest(text=text[:30]):
                self.assertIn(SECRET_PLACEHOLDER, redact_secrets(text))

    def test_ordinary_code_is_not_redacted(self):
        for text in ["API_KEY = settings.api_key", "password = get_password()", 'token_type = "Bearer"',
                     "const cfg = { apiKey: process.env.KEY }", "DEBUG=true", "PAGE_SIZE=20"]:
            with self.subTest(text=text):
                self.assertEqual(redact_secrets(text), text)

    def test_secret_file_masks_every_value(self):
        masked = redact_secret_file("A=1\nHOST=localhost\n# comment\nEMPTY=\njson_key: abc")
        self.assertEqual(masked, "A=<SECRET>\nHOST=<SECRET>\n# comment\nEMPTY=\njson_key: \"<SECRET>\"")

    def test_registry_masks_known_values_anywhere(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".env.local").write_text("SESSION_SECRET=opaque-value-without-pattern-77\nPORT=3000\n", encoding="utf-8")
            register_project_secrets(d)
        self.assertEqual(redact_secrets("the value is opaque-value-without-pattern-77!"), "the value is <SECRET>!")
        self.assertEqual(redact_secrets("PORT 3000"), "PORT 3000")  # non-sensitive values are not registered
        self.assertIn(".env.local:SESSION_SECRET", get_secret_registry().names())


class ToolSurfaceTests(SecretProjectCase):
    def test_read_env_file_shows_keys_not_values(self):
        out = self.dispatch("Read", {"file_path": str(self.root / ".env")}).output
        content = out["file"]["content"]
        assert_no_secrets(self, content, "Read(.env)")
        self.assertIn("DATABASE_URL=<SECRET>", content)
        self.assertIn("STRIPE_SECRET_KEY=<SECRET>", content)
        self.assertTrue(out["file"]["secretsRedacted"])

    def test_grep_does_not_leak_env_values(self):
        out = self.dispatch("Grep", {"pattern": "STRIPE", "path": str(self.root), "output_mode": "content"}).output
        assert_no_secrets(self, out["content"], "Grep")
        self.assertIn("STRIPE_SECRET_KEY=<SECRET>", out["content"])

    def test_bash_cat_env_requires_approval_and_output_is_masked(self):
        denied = self.dispatch("Bash", {"command": "cat .env"}, approve=False)
        self.assertEqual(denied.error_kind, "permission")
        approved = self.dispatch("Bash", {"command": "cat .env"}, approve=True)
        assert_no_secrets(self, json.dumps(approved.output), "Bash(cat .env)")

    def test_controlled_secret_use_without_disclosure(self):
        script = self.root / "check.py"
        script.write_text(
            "import os\nv=os.environ['STRIPE_SECRET_KEY']\nprint('len', len(v))\nprint('echo', v)\n", encoding="utf-8"
        )
        denied = self.dispatch("Bash", {"command": f"{PY} check.py", "secret_env": ["STRIPE_SECRET_KEY"]}, approve=False)
        self.assertEqual(denied.error_kind, "permission")
        result = self.dispatch("Bash", {"command": f"{PY} check.py", "secret_env": ["STRIPE_SECRET_KEY"]}, approve=True)
        self.assertFalse(result.is_error, result.output)
        self.assertIn(f"len {len(STRIPE_KEY)}", result.output["stdout"])  # the process really received it
        self.assertIn("echo <SECRET>", result.output["stdout"])           # but it never comes back
        missing = self.dispatch("Bash", {"command": "echo x", "secret_env": ["NOPE_NOT_SET"]}, approve=True)
        self.assertEqual(missing.error_kind, "input")

    def test_config_tool_never_returns_keys(self):
        from src.config import save_config

        save_config({"providers": {"openai": {"api_key": STRIPE_KEY}}})
        out = self.dispatch("Config", {"setting": "providers"}).output
        assert_no_secrets(self, json.dumps(out), "Config")

    def test_context_builder_redacts_instruction_files(self):
        (self.root / "CLAUDE.md").write_text(f"Use STRIPE_SECRET_KEY={STRIPE_KEY} for tests", encoding="utf-8")
        prompt = build_context_prompt(self.root)
        assert_no_secrets(self, prompt, "context prompt")
        self.assertIn("STRIPE_SECRET_KEY=<SECRET>", prompt)


class ModelSessionLogReportTests(SecretProjectCase):
    def _run(self, provider):
        conv = Conversation()
        conv.add_user_message("inspect config")
        run_agent_loop(conv, provider, self.registry, self.ctx, system_prompt="S")
        return conv

    def _leaky_script(self):
        return [
            reply("", [
                tool_call("c1", "Read", {"file_path": str(self.root / ".env")}),
                tool_call("c2", "Bash", {"command": f"{PY} -c \"print('{DATABASE_URL}')\""}),
                tool_call("c3", "Grep", {"pattern": "sk_live", "path": str(self.root), "output_mode": "content"}),
            ]),
            reply("done"),
        ]

    def test_secrets_never_reach_model_messages(self):
        for provider_cls in (FakeOpenAIProvider, FakeAnthropicProvider):
            with self.subTest(provider=provider_cls.__name__):
                provider = provider_cls(self._leaky_script())
                self._run(provider)
                assert_no_secrets(self, json.dumps(provider.requests), "provider request")

    def test_secrets_never_reach_session_files(self):
        provider = FakeOpenAIProvider(self._leaky_script())
        conv = self._run(provider)
        conv.add_user_message(f"my stripe key is {STRIPE_KEY}")  # even user-pasted secrets
        session = Session.create("fake", "m")
        session.conversation = conv
        session.state = {"todos": [{"content": f"rotate {DATABASE_URL}"}]}
        path = session.save(self.root / "sessions")
        assert_no_secrets(self, path.read_text(encoding="utf-8"), "session file")

    def test_secrets_never_reach_logs(self):
        log = self.root / "log.jsonl"
        configure_logging(log_file=log)
        try:
            self._run(FakeOpenAIProvider(self._leaky_script()))
            log_event("custom", note=f"connecting to {DATABASE_URL}", header=f"x-key {STRIPE_KEY}")
        finally:
            configure_logging()
        assert_no_secrets(self, log.read_text(encoding="utf-8"), "log file")

    def test_secrets_never_reach_delivery_reports(self):
        from src.orchestration.reports import render_delivery_report

        report = render_delivery_report({
            "goal": "demo", "tasks": [{"id": "T-1", "title": "x", "status": "COMPLETED",
                                       "result": {"summary": f"used {DATABASE_URL}", "warnings": [STRIPE_KEY]}}],
        })
        assert_no_secrets(self, report, "delivery report")


if __name__ == "__main__":
    unittest.main()
