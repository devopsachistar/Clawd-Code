"""Agent abstraction: profiles, lifecycle, permissions, budgets, structured results, communication."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from fakes import FakeOpenAIProvider, reply, tool_call
from src.agents.messages import ORCHESTRATOR, AgentMessage, MessageBus, MessageRejected, MessageType
from src.agents.models import Agent, AgentBudget, AgentResult, AgentStatus, InvalidTransition
from src.agents.profiles import ProfileRegistry, parse_profile
from src.agents.runner import AgentRunner
from src.agents.store import AgentStore
from src.orchestration.state import EventRecorder
from src.security.permissions import Capability

EXPECTED_PROFILES = ["Architect", "BackendDeveloper", "CodeReviewer", "DatabaseEngineer", "DevOpsEngineer",
                     "Developer", "FrontendDeveloper", "QAEngineer", "SecurityReviewer"]


def report(**fields):
    fields.setdefault("status", "completed")
    fields.setdefault("summary", "done")
    return tool_call("rep", "ReportResult", fields)


class ProfileTests(unittest.TestCase):
    def test_builtin_profiles(self):
        registry = ProfileRegistry.load()
        self.assertEqual(registry.names(), EXPECTED_PROFILES)
        self.assertEqual(registry.warnings, [])
        for profile in registry.all():
            with self.subTest(profile=profile.name):
                self.assertTrue(profile.responsibilities)
                self.assertTrue(profile.allowed_tools)
                self.assertTrue(profile.instructions.strip())
                self.assertIn("ReportResult", profile.system_prompt())
                self.assertIn(Capability.READ, profile.allow)
                # nothing high-risk is ever pre-approved
                self.assertFalse(profile.allow & {Capability.GIT_PUSH, Capability.DEPLOY_PRODUCTION,
                                                  Capability.DATABASE_WRITE, Capability.SECRET_USE})
        dev, qa = registry.get("developer"), registry.get("QAEngineer")
        self.assertEqual((dev.budget.max_iterations, qa.budget.max_iterations), (40, 20))
        for reviewer in ("CodeReviewer", "SecurityReviewer"):
            self.assertIn(Capability.WRITE, registry.get(reviewer).deny)
            self.assertNotIn("Write", registry.get(reviewer).allowed_tools)
        self.assertIn(Capability.DEPLOY_PRODUCTION, registry.get("DevOpsEngineer").deny)
        self.assertIn("structure, modules, interfaces", registry.get("Architect").description)

    def test_high_risk_capabilities_are_clamped_to_ask(self):
        profile = parse_profile("---\nname: Yolo\nallowed_tools: [Read]\nallow: [READ, DEPLOY_PRODUCTION, SECRET_USE]\n---\nbody")
        self.assertNotIn(Capability.DEPLOY_PRODUCTION, profile.allow)
        self.assertIn(Capability.DEPLOY_PRODUCTION, profile.ask)
        self.assertTrue(profile.warnings)

    def test_project_and_user_profiles_override(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".clawd" / "agents").mkdir(parents=True)
            (root / ".clawd" / "agents" / "dev.md").write_text(
                "---\nname: Developer\nallowed_tools: [Read, Grep, EvilTool]\nallow: [READ]\nmax_iterations: 7\n---\nProject dev",
                encoding="utf-8")
            user = root / "user-agents"
            user.mkdir()
            (user / "extra.md").write_text("---\nname: DocsWriter\nallowed_tools: [Read, Write]\nallow: [READ, WRITE]\n---\nDocs",
                                           encoding="utf-8")
            registry = ProfileRegistry.load(root, user_dir=user)
        dev = registry.get("Developer")
        self.assertEqual(dev.budget.max_iterations, 7)
        self.assertEqual(dev.allowed_tools, ("Read", "Grep"))  # unknown tool from untrusted project file dropped
        self.assertTrue(any("EvilTool" in w for w in registry.warnings))
        self.assertIn("DocsWriter", registry.names())

    def test_unknown_profile(self):
        with self.assertRaises(KeyError):
            ProfileRegistry.load().get("Wizard")


class AgentModelTests(unittest.TestCase):
    def make(self) -> Agent:
        return Agent(name="Dev", role="developer", profile="Developer", system_prompt="s", provider="p", model="m",
                     allowed_tools=["Read"], permissions={}, working_directory=".")

    def test_lifecycle_transitions(self):
        a = self.make()
        self.assertEqual(a.status, AgentStatus.CREATED)
        a.transition(AgentStatus.RUNNING)
        self.assertIsNotNone(a.started_at)
        a.transition(AgentStatus.WAITING)
        a.transition(AgentStatus.RUNNING)
        a.transition(AgentStatus.COMPLETED)
        self.assertTrue(a.is_terminal)
        self.assertIsNotNone(a.finished_at)
        with self.assertRaises(InvalidTransition):
            a.transition(AgentStatus.RUNNING)
        with self.assertRaises(InvalidTransition):
            self.make().transition(AgentStatus.WAITING)  # CREATED -> WAITING not allowed

    def test_round_trip(self):
        a = self.make()
        a.budget = AgentBudget(max_iterations=5, max_tool_calls=9, max_seconds=60, max_tokens=1000)
        b = Agent.from_dict(a.to_dict())
        self.assertEqual(b.to_dict(), a.to_dict())

    def test_structured_result_validation(self):
        r = AgentResult(task_id="T", agent_id="A", status="completed", summary="s", files_changed=["a.py"], tests_run=2,
                        tests_passed=2)
        self.assertEqual(AgentResult.from_dict(r.to_dict()), r)
        for bad in ({"status": "done"}, {"tests_run": -1}, {"files_changed": "a.py"}, {"review_verdict": "meh"}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    AgentResult(**{"task_id": "T", "agent_id": "A", "status": "completed", "summary": "s", **bad})


class AgentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name).resolve()
        self.work = base / "work"
        self.work.mkdir()
        self.state = base / "state"
        self.events = EventRecorder(self.state)
        self.store = AgentStore(self.state)
        self.bus = MessageBus(self.state, events=self.events)
        self.profiles = ProfileRegistry.load()
        self.providers: dict[str, FakeOpenAIProvider] = {}
        self.approvals = []

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def runner(self, script, approve=None):
        provider = FakeOpenAIProvider(script)

        def handler(request):
            agent = self.store.get(request.agent_id)
            self.approvals.append((request, agent.status))
            return bool(approve)

        return AgentRunner(provider_factory=lambda agent: provider, events=self.events, store=self.store, bus=self.bus,
                           approval_handler=handler, sessions_dir=self.state / "sessions"), provider

    def agent(self, runner, profile="Developer", **budget):
        p = self.profiles.get(profile)
        if budget:
            from dataclasses import replace

            p = replace(p, budget=AgentBudget(**{**p.budget.to_dict(), **budget}))
        return runner.create_agent(p, task_id="T-1", working_directory=self.work, provider="fake", model="fake-model")

    def test_successful_run_produces_structured_result(self):
        runner, provider = self.runner([
            reply("", [tool_call("w", "Write", {"file_path": str(self.work / "app.py"), "content": "x = 1\n"})]),
            reply("", [report(files_changed=["app.py"], tests_run=0, next_action="review")]),
            reply("finished"),
        ])
        agent = self.agent(runner)
        self.assertEqual(agent.status, AgentStatus.CREATED)
        result = runner.run(agent, "Create app.py")
        self.assertEqual(result.status, "completed")
        self.assertTrue(result.reported_by_agent)
        self.assertEqual((result.task_id, result.agent_id, result.files_changed), ("T-1", agent.id, ["app.py"]))
        stored = self.store.get(agent.id)
        self.assertEqual(stored.status, AgentStatus.COMPLETED)
        self.assertEqual(stored.result["next_action"], "review")
        self.assertIsNotNone(stored.session_id)
        self.assertTrue((self.state / "sessions" / f"{stored.session_id}.json").exists())
        # system prompt = profile instructions + execution contract + worktree context
        system = provider.requests[0]["messages"][0]["content"]
        self.assertIn("# Role: Developer", system)
        self.assertIn(f"Working directory: {self.work}", system)
        # only the profile's tools + agent tools are offered
        offered = {t["name"] for t in provider.requests[0]["tools"]}
        self.assertEqual(offered, set(agent.allowed_tools) | {"ReportResult", "MessageOrchestrator"})
        names = [e["event"] for e in self.events.read(agent_id=agent.id)]
        for expected in ("agent.created", "agent.started", "agent.tool_started", "agent.tool_completed", "agent.completed"):
            self.assertIn(expected, names)

    def test_missing_structured_result_is_a_failure(self):
        runner, _ = self.runner([reply("I did it, trust me.")])
        agent = self.agent(runner)
        result = runner.run(agent, "task")
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.reported_by_agent)
        self.assertIn("ReportResult", result.error)
        self.assertEqual(self.store.get(agent.id).status, AgentStatus.FAILED)

    def test_profile_permissions_are_enforced(self):
        runner, _ = self.runner([
            reply("", [tool_call("p", "Bash", {"command": "git push origin main"}),
                       tool_call("c", "Bash", {"command": "git commit -m x"})]),
            reply("", [report(status="completed")]),
            reply("ok"),
        ])
        agent = self.agent(runner, "Developer")
        result = runner.run(agent, "try to push")
        self.assertTrue(any("GIT_PUSH" in w for w in result.warnings))
        self.assertTrue(any("GIT_COMMIT" in w for w in result.warnings))
        self.assertEqual(self.approvals, [])  # denied by policy, never asked

    def test_reviewer_cannot_write(self):
        runner, _ = self.runner([
            reply("", [tool_call("w", "Write", {"file_path": str(self.work / "x.py"), "content": "x"})]),
            reply("", [report(status="completed", review_verdict="approve")]),
            reply("ok"),
        ])
        result = runner.run(self.agent(runner, "CodeReviewer"), "review")
        self.assertFalse((self.work / "x.py").exists())
        self.assertEqual(result.review_verdict, "approve")

    def test_waiting_status_during_approval(self):
        runner, _ = self.runner([
            reply("", [tool_call("n", "Bash", {"command": "curl https://example.invalid"})]),  # NETWORK = ask
            reply("", [report(status="completed")]),
            reply("ok"),
        ], approve=False)
        agent = self.agent(runner, "Developer")
        runner.run(agent, "fetch")
        self.assertEqual(len(self.approvals), 1)
        request, status_during = self.approvals[0]
        self.assertEqual(status_during, AgentStatus.WAITING)
        self.assertEqual((request.agent_id, request.task_id), (agent.id, "T-1"))
        self.assertIn("NETWORK", request.capabilities)
        self.assertEqual(self.store.get(agent.id).status, AgentStatus.COMPLETED)
        self.assertTrue(self.events.read(event="agent.waiting"))
        # the approval lifecycle is persisted in the project event log
        requested = self.events.read(event="approval.requested")
        denied = self.events.read(event="approval.denied")
        self.assertEqual(requested[0]["id"], request.id)
        self.assertEqual((denied[0]["approval_id"], denied[0]["agent_id"]), (request.id, agent.id))

    def test_iteration_budget(self):
        runner, _ = self.runner([reply("", [tool_call(f"g{i}", "Glob", {"pattern": "*"})]) for i in range(10)])
        result = runner.run(self.agent(runner, max_iterations=3), "loop")
        self.assertEqual((result.status, result.stop_reason), ("failed", "max_turns"))

    def test_tool_call_budget(self):
        runner, _ = self.runner([reply("", [tool_call(f"g{i}", "Glob", {"pattern": "*"}) for i in range(5)])])
        result = runner.run(self.agent(runner, max_tool_calls=2), "loop")
        self.assertEqual((result.status, result.stop_reason), ("failed", "budget_tool_calls"))

    def test_time_budget(self):
        class Slow(FakeOpenAIProvider):
            def chat(self, messages, tools=None, **kw):
                time.sleep(0.3)
                return super().chat(messages, tools, **kw)

        provider = Slow([reply("", [tool_call(f"g{i}", "Glob", {"pattern": "*"})]) for i in range(20)])
        runner = AgentRunner(provider_factory=lambda a: provider, events=self.events, store=self.store)
        started = time.monotonic()
        result = runner.run(self.agent(runner, max_seconds=1), "loop")
        self.assertEqual(result.stop_reason, "timeout")
        self.assertLess(time.monotonic() - started, 5)

    def test_token_budget(self):
        runner, _ = self.runner([reply("", [tool_call(f"g{i}", "Glob", {"pattern": "*"})]) for i in range(10)])
        result = runner.run(self.agent(runner, max_tokens=30), "loop")  # each reply reports 15 tokens
        self.assertEqual(result.stop_reason, "budget_tokens")

    def test_cancellation(self):
        runner, _ = self.runner([reply("", [report()])])
        agent = self.agent(runner)
        result = runner.run(agent, "x", should_cancel=lambda: True)
        self.assertEqual(self.store.get(agent.id).status, AgentStatus.CANCELLED)
        self.assertEqual(result.stop_reason, "cancelled")

    def test_provider_error_fails_agent(self):
        from fakes import FailingProvider

        runner = AgentRunner(provider_factory=lambda a: FailingProvider(RuntimeError("upstream down")),
                             events=self.events, store=self.store)
        agent = self.agent(runner)
        result = runner.run(agent, "x")
        self.assertIn("upstream down", result.error)
        self.assertEqual(self.store.get(agent.id).status, AgentStatus.FAILED)


class CommunicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_agent_messages_go_only_to_orchestrator(self):
        bus = MessageBus(self.state)
        bus.send(AgentMessage(MessageType.QUESTION, "A-1", ORCHESTRATOR, "which db?", "T-1"), sender_is_agent=True)
        with self.assertRaises(MessageRejected):
            bus.send(AgentMessage(MessageType.QUESTION, "A-1", "A-2", "psst"), sender_is_agent=True)
        with self.assertRaises(MessageRejected):
            bus.send(AgentMessage(MessageType.TASK_ASSIGNMENT, "A-1", ORCHESTRATOR, "do it"), sender_is_agent=True)
        bus.send(AgentMessage(MessageType.TASK_ASSIGNMENT, ORCHESTRATOR, "A-1", "build", "T-1"))
        reloaded = MessageBus(self.state).all()  # persisted, auditable
        self.assertEqual([m.type for m in reloaded], [MessageType.QUESTION, MessageType.TASK_ASSIGNMENT])
        self.assertEqual(len(MessageBus(self.state).query(task_id="T-1", type=MessageType.QUESTION)), 1)

    def test_all_message_types(self):
        self.assertEqual({t.value for t in MessageType}, {"TASK_ASSIGNMENT", "TASK_RESULT", "BLOCKED", "QUESTION",
                                                          "ERROR", "REVIEW_REQUEST", "REVIEW_RESULT"})

    def test_messages_are_redacted(self):
        # Synthetic Stripe-shaped key, assembled at runtime so the complete
        # credential-like literal never appears in the repository.
        stripe_like = "sk_" + "live_" + "FAKE0TEST0ONLY0NOT0A0REAL0KEY"
        bus = MessageBus(self.state)
        bus.send(AgentMessage(MessageType.ERROR, "A-1", ORCHESTRATOR, f"failed with {stripe_like}"),
                 sender_is_agent=True)
        persisted = (self.state / "messages.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(stripe_like, persisted)
        self.assertIn("failed with <SECRET>", persisted)

    def test_message_orchestrator_tool(self):
        tmp_work = self.state / "w"
        tmp_work.mkdir()
        bus = MessageBus(self.state)
        provider = FakeOpenAIProvider([
            reply("", [tool_call("q", "MessageOrchestrator", {"type": "BLOCKED", "content": "need API spec"})]),
            reply("", [report(status="blocked", summary="waiting for API spec")]),
            reply("ok"),
        ])
        runner = AgentRunner(provider_factory=lambda a: provider, events=EventRecorder(None), bus=bus)
        agent = runner.create_agent(ProfileRegistry.load().get("Developer"), task_id="T-9", working_directory=tmp_work,
                                    provider="fake", model="m")
        result = runner.run(agent, "x")
        self.assertEqual(result.status, "blocked")
        blocked = bus.query(type=MessageType.BLOCKED)
        self.assertEqual((blocked[0].sender, blocked[0].recipient, blocked[0].task_id), (agent.id, ORCHESTRATOR, "T-9"))


if __name__ == "__main__":
    unittest.main()
