"""Phase 2 capability-based permission engine and approval flow."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fakes import FakeOpenAIProvider, reply, tool_call
from src.headless import ExitCode, RunOptions, run_headless
from src.security.permissions import (
    CAPABILITY_RISK,
    ApprovalRequest,
    Capability,
    Decision,
    PermissionEngine,
    PermissionPolicy,
    PermissionRule,
    RiskLevel,
    analyze_command,
    assess_tool_call,
)
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall

C = Capability


class CapabilityAssessmentTests(unittest.TestCase):
    COMMANDS = {
        "ls -la": ({C.EXECUTE}, RiskLevel.MEDIUM),
        "git status": ({C.EXECUTE}, RiskLevel.MEDIUM),
        "git checkout -b feature/x": ({C.EXECUTE, C.GIT_BRANCH}, RiskLevel.MEDIUM),
        "git commit -m wip": ({C.EXECUTE, C.GIT_COMMIT}, RiskLevel.MEDIUM),
        "git push origin main": ({C.EXECUTE, C.GIT_PUSH}, RiskLevel.HIGH),
        "git reset --hard HEAD~2": ({C.EXECUTE, C.GIT_COMMIT}, RiskLevel.HIGH),
        "curl https://example.com": ({C.EXECUTE, C.NETWORK}, RiskLevel.MEDIUM),
        "pip install requests": ({C.EXECUTE, C.NETWORK}, RiskLevel.MEDIUM),
        "psql -c 'SELECT * FROM users'": ({C.EXECUTE, C.DATABASE_READ}, RiskLevel.MEDIUM),
        "psql -c 'DELETE FROM users'": ({C.EXECUTE, C.DATABASE_READ, C.DATABASE_WRITE}, RiskLevel.HIGH),
        "psql -c 'DROP TABLE users'": ({C.EXECUTE, C.DATABASE_READ, C.DATABASE_WRITE}, RiskLevel.CRITICAL),
        "npx prisma migrate reset --force": ({C.EXECUTE, C.DATABASE_WRITE}, RiskLevel.CRITICAL),
        "vercel deploy": ({C.EXECUTE, C.DEPLOY_STAGING}, RiskLevel.HIGH),
        "vercel --prod": ({C.EXECUTE, C.DEPLOY_PRODUCTION}, RiskLevel.CRITICAL),
        "az webapp deploy --name app --src-path app.zip": ({C.EXECUTE, C.DEPLOY_PRODUCTION}, RiskLevel.CRITICAL),
        "npm run deploy": ({C.EXECUTE, C.DEPLOY_PRODUCTION}, RiskLevel.CRITICAL),
        "npm publish": ({C.EXECUTE, C.DEPLOY_PRODUCTION}, RiskLevel.CRITICAL),
        "cat .env": ({C.EXECUTE, C.SECRET_USE}, RiskLevel.HIGH),
        "Get-Content .env.production": ({C.EXECUTE, C.SECRET_USE}, RiskLevel.HIGH),
        "echo $STRIPE_SECRET_KEY": ({C.EXECUTE, C.SECRET_USE}, RiskLevel.HIGH),
        "printenv": ({C.EXECUTE, C.SECRET_USE}, RiskLevel.HIGH),
        "echo hi > out.txt": ({C.EXECUTE, C.WRITE}, RiskLevel.MEDIUM),
    }

    def test_command_capabilities_and_risk(self):
        for command, (caps, risk) in self.COMMANDS.items():
            with self.subTest(command=command):
                got_caps, got_risk, _ = analyze_command(command)
                self.assertTrue(caps <= got_caps, f"{command}: {sorted(c.value for c in got_caps)}")
                self.assertEqual(got_risk, risk)

    def test_tool_capabilities(self):
        cases = {
            ("Read", ()): {C.READ},
            ("Grep", ()): {C.READ},
            ("Write", (("file_path", "a.py"), ("content", "x"))): {C.WRITE},
            ("Write", (("file_path", ".env"), ("content", "X=1"))): {C.WRITE, C.SECRET_USE},
            ("WebFetch", (("url", "https://x"),)): {C.READ, C.NETWORK},
            ("Config", (("setting", "default_provider"),)): {C.READ},
            ("Config", (("setting", "providers.openai.base_url"), ("value", "x"))): {C.WRITE, C.SECRET_USE},
            ("EnterWorktree", ()): {C.GIT_BRANCH},
            ("SomeUserTool", ()): {C.EXECUTE},
        }
        for (tool, items), expected in cases.items():
            with self.subTest(tool=tool, items=items):
                self.assertEqual(set(assess_tool_call(tool, dict(items)).capabilities), expected)

    def test_blocked_commands_are_critical(self):
        a = assess_tool_call("Bash", {"command": "sudo rm -rf /"})
        self.assertTrue(a.blocked)
        self.assertEqual(a.risk, RiskLevel.CRITICAL)

    def test_secret_env_requires_secret_use(self):
        a = assess_tool_call("Bash", {"command": "python migrate.py", "secret_env": ["DATABASE_URL"]})
        self.assertIn(C.SECRET_USE, a.capabilities)
        self.assertGreaterEqual(a.risk, RiskLevel.HIGH)

    def test_every_capability_has_a_risk(self):
        self.assertEqual(set(CAPABILITY_RISK), set(Capability))


class PolicyDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PermissionEngine()

    def decide(self, tool, tool_input, policy):
        return self.engine.evaluate(tool, tool_input, policy)

    def test_session_policy_preserves_phase1_behaviour(self):
        s = PermissionPolicy.session_default()
        self.assertEqual(self.decide("Read", {"file_path": "a"}, s).decision, Decision.ALLOW)
        self.assertEqual(self.decide("Bash", {"command": "pytest -q"}, s).decision, Decision.ALLOW)
        self.assertEqual(self.decide("Bash", {"command": "rm -r build"}, s).decision, Decision.ASK)
        self.assertEqual(self.decide("Bash", {"command": "git push"}, s).decision, Decision.ASK)
        self.assertEqual(self.decide("Bash", {"command": "sudo ls"}, s).decision, Decision.DENY)

    def test_profile_policy_denies_ungranted_capabilities(self):
        reviewer = PermissionPolicy.from_lists("reviewer", allow=["READ"])
        self.assertEqual(self.decide("Read", {"file_path": "a"}, reviewer).decision, Decision.ALLOW)
        denied = self.decide("Write", {"file_path": "a.py", "content": "x"}, reviewer)
        self.assertEqual(denied.decision, Decision.DENY)
        self.assertIn("WRITE", denied.message)
        self.assertEqual(self.decide("Bash", {"command": "ls"}, reviewer).decision, Decision.DENY)

    def test_ask_and_deny_lists(self):
        dev = PermissionPolicy.from_lists("dev", allow=["READ", "WRITE", "EXECUTE"], ask=["GIT_COMMIT"], deny=["GIT_PUSH"])
        self.assertEqual(self.decide("Bash", {"command": "git commit -m x"}, dev).decision, Decision.ASK)
        self.assertEqual(self.decide("Bash", {"command": "git push"}, dev).decision, Decision.DENY)

    def test_rules(self):
        policy = PermissionPolicy.session_default().with_rules([
            PermissionRule.parse("Bash(git push*)", "deny"),
            PermissionRule.parse("Bash(rm -r build)", "allow"),
            PermissionRule.parse("Write(*.lock)", "ask"),
            PermissionRule.parse("Bash(cat .env*)", "allow"),
        ])
        self.assertEqual(self.decide("Bash", {"command": "git push origin x"}, policy).decision, Decision.DENY)
        self.assertEqual(self.decide("Bash", {"command": "rm -r build"}, policy).decision, Decision.ALLOW)
        self.assertEqual(self.decide("Write", {"file_path": "poetry.lock", "content": ""}, policy).decision, Decision.ASK)
        # allow rules can never waive SECRET_USE / critical risk
        self.assertEqual(self.decide("Bash", {"command": "cat .env"}, policy).decision, Decision.ASK)

    def test_invalid_rule_and_capability(self):
        with self.assertRaises(ValueError):
            PermissionRule.parse("", "allow")
        with self.assertRaises(ValueError):
            PermissionPolicy.from_lists("x", allow=["TELEPORT"])

    def test_auto_approvable_limits(self):
        s = PermissionPolicy.session_default()
        self.assertTrue(self.decide("Bash", {"command": "rm -r build"}, s).auto_approvable)
        self.assertTrue(self.decide("Bash", {"command": "git commit -m x"}, s).auto_approvable)
        for command in ["git push", "vercel --prod", "psql -c 'DROP TABLE t'", "cat .env", "vercel deploy",
                        "psql -c 'UPDATE t SET a=1'"]:
            with self.subTest(command=command):
                self.assertFalse(self.decide("Bash", {"command": command}, s).auto_approvable)
        self.assertFalse(self.decide("Bash", {"command": "ls", "secret_env": ["X"]}, s).auto_approvable)

    def test_structured_approval_request(self):
        d = self.decide("Bash", {"command": "git push origin main"}, PermissionPolicy.session_default())
        req = d.approval_request("Bash", agent_id="A-1", task_id="T-1")
        self.assertIsInstance(req, ApprovalRequest)
        self.assertEqual(req.decision, "ASK")
        self.assertIn("GIT_PUSH", req.capabilities)
        self.assertEqual(req.risk, "HIGH")
        rendered = req.render()
        for fragment in ["Tool: Bash: git push origin main", "Capability:", "GIT_PUSH", "Risk: HIGH", "Decision: ASK",
                         "explicit approval", "A-1"]:
            self.assertIn(fragment, rendered)
        self.assertTrue(req.id.startswith("AP-"))


class DispatchApprovalFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.registry = build_default_registry(include_user_tools=False)
        self.requests: list[ApprovalRequest] = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def ctx(self, policy=None, approve=None) -> ToolContext:
        ctx = ToolContext(workspace_root=self.root, permission_policy=policy)
        if approve is not None:
            def handler(request):
                self.requests.append(request)
                return approve
            ctx.approval_handler = handler
        return ctx

    def test_every_invocation_is_evaluated(self):
        reviewer = PermissionPolicy.from_lists("reviewer", allow=["READ"])
        ctx = self.ctx(reviewer)
        (self.root / "a.txt").write_text("x", encoding="utf-8")
        self.assertFalse(self.registry.dispatch(ToolCall("Read", {"file_path": str(self.root / "a.txt")}), ctx).is_error)
        result = self.registry.dispatch(ToolCall("Write", {"file_path": str(self.root / "b.txt"), "content": "x"}), ctx)
        self.assertEqual(result.error_kind, "permission")
        self.assertFalse((self.root / "b.txt").exists())
        # nested calls through the batch Agent tool are evaluated too
        result = self.registry.dispatch(ToolCall("Agent", {"calls": [{"name": "Read", "input": {"file_path": "a.txt"}}]}), ctx)
        self.assertEqual(result.error_kind, "permission")

    def test_ask_invokes_structured_handler(self):
        ctx = self.ctx(approve=False)
        result = self.registry.dispatch(ToolCall("Bash", {"command": "git push origin main"}), ctx)
        self.assertEqual(result.error_kind, "permission")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(result.output["approval"]["risk"], "HIGH")

    def test_approved_request_runs(self):
        (self.root / "build").mkdir()
        ctx = self.ctx(approve=True)
        result = self.registry.dispatch(ToolCall("Bash", {"command": "rm -r build"}), ctx)
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(self.requests[0].capabilities, ("EXECUTE", "WRITE"))

    def test_no_handler_means_denied(self):
        result = self.registry.dispatch(ToolCall("Bash", {"command": "git commit -m x"}), self.ctx())
        self.assertEqual(result.error_kind, "permission")


class ConfiguredRulesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_user_rules_and_tighten_only_project_rules(self):
        from src.config import load_session_policy, save_config

        save_config({"permissions": {"allow": ["Bash(git push origin feature*)"], "deny": ["Bash(npm publish*)"]}})
        settings = self.root / ".clawd" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text('{"provider": "openai", "permissions": {"allow": ["Bash(git push*)"], '
                            '"ask": ["Write(*.lock)"], "deny": ["Bash(rm *)"]}}', encoding="utf-8")
        policy, warnings = load_session_policy(self.root)
        engine = PermissionEngine()
        d = lambda cmd: engine.evaluate("Bash", {"command": cmd}, policy).decision  # noqa: E731
        self.assertEqual(d("git push origin feature/x"), Decision.ALLOW)   # user allow rule
        self.assertEqual(d("git push origin main"), Decision.ASK)          # project allow ignored
        self.assertEqual(d("rm tmp.txt"), Decision.DENY)                   # project deny applies
        self.assertEqual(d("npm publish"), Decision.DENY)                  # user deny
        self.assertEqual(engine.evaluate("Write", {"file_path": "poetry.lock", "content": ""}, policy).decision, Decision.ASK)
        self.assertTrue(any("may only add ask/deny" in w for w in warnings))

    def test_headless_uses_configured_rules(self):
        from src.config import save_config

        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        save_config({"permissions": {"deny": ["Bash(python*)"]}})
        provider = FakeOpenAIProvider([reply("", [tool_call("c", "Bash", {"command": "python -V"})]), reply("x")])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-0000", "CLAWD_PROVIDER": "openai"}):
            outcome = run_headless(RunOptions(prompt="x", cwd=str(self.root)), stdout=io.StringIO(), stderr=io.StringIO(),
                                   provider_factory=lambda r: provider)
        self.assertEqual(outcome.exit_code, ExitCode.PERMISSION_DENIED)
        self.assertIn("denied by rule", outcome.permission_denials[0]["message"])


class HeadlessAutoApproveTests(unittest.TestCase):
    """--yes must never approve production deploys, secrets, DB writes or pushes."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.project = Path(self.tmp.name).resolve() / "p"
        self.project.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.project, check=True)
        self.env = patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-0000", "CLAWD_PROVIDER": "openai"})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tmp.cleanup()

    def run_with(self, command):
        provider = FakeOpenAIProvider([reply("", [tool_call("c1", "Bash", {"command": command})]), reply("done")])
        err = io.StringIO()
        outcome = run_headless(RunOptions(prompt="x", cwd=str(self.project), auto_approve=True),
                               stdout=io.StringIO(), stderr=err, provider_factory=lambda r: provider)
        return outcome, err.getvalue()

    def test_yes_does_not_approve_high_risk(self):
        for command in ["git push origin main", "vercel --prod", "npm run deploy", "cat .env",
                        "psql -c 'DROP TABLE users'"]:
            with self.subTest(command=command):
                outcome, err = self.run_with(command)
                self.assertEqual(outcome.exit_code, ExitCode.PERMISSION_DENIED)
                self.assertIn("--yes does not cover it", err)

    def test_yes_approves_medium_risk(self):
        outcome, err = self.run_with("git commit --allow-empty -m test")
        self.assertIn("[auto-approved]", err)
        self.assertNotEqual(outcome.exit_code, ExitCode.PERMISSION_DENIED)


if __name__ == "__main__":
    unittest.main()
