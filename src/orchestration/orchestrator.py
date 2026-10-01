"""Structured multi-agent orchestration.

Flow for one graph::

    plan -> tasks (PENDING) -> dependencies complete -> READY
      -> worktree on increscent/<task>/<role> (based on the dependency branches)
      -> agent (profile policy, budgets) -> structured result
      -> orchestrator verifies via git (files, commit) and the test runner (counts)
      -> COMPLETED | retry with failure context | FAILED (dependents BLOCKED) | BLOCKED
    all tasks COMPLETED -> verification tests -> REVIEW_REQUEST -> graph REVIEW
    human approval -> merge task branches into increscent/<graph>/integration   (never main)

Nothing is pushed, deployed or merged into the main branch.
"""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..agents.messages import HUMAN, ORCHESTRATOR, AgentMessage, MessageBus, MessageType
from ..agents.models import Agent, AgentResult, AgentStatus
from ..agents.profiles import ProfileRegistry
from ..agents.runner import AgentRunner, ApprovalHandler, ProviderFactory
from ..agents.store import AgentStore
from ..gitops.worktrees import GitError, MergeConflict, WorktreeBusy, WorktreeManager, run_git
from ..security.secrets import redact_secrets
from ..taskgraph.models import Task, TaskGraph, TaskStatus
from ..taskgraph.store import TaskStore
from .planner import TaskSpec
from .reports import render_delivery_report
from .state import EventRecorder, atomic_write_json, now_iso, resolve_state_dir
from .testing import TestRun, run_tests

INTEGRATION_ROLE = "integration"
MAX_REPAIRS = 2  # hard cap on QA-guided repair attempts per task (Phase 3.3.2)


@dataclass
class OrchestratorConfig:
    provider: str = "simulated"
    model: str = "simulated"
    max_parallel: int = 1
    max_attempts: int = 3            # first attempt + retries
    test_command: Optional[str] = None
    run_verification: bool = True
    # "agent": Phase 2 path (runner + _verify_and_commit).
    # "developer": Phase 3.2 path: each task runs through DeveloperAgent.execute() (measured files/tests).
    executor: str = "agent"
    # Developer path only: after a measured failure, QA diagnoses it and the developer repairs it
    # in the same worktree, up to this many times (capped at MAX_REPAIRS). 0 = no repair loop.
    max_repairs: int = 0


@dataclass
class RunSummary:
    graph_id: str
    status: str
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    verification: Optional[dict[str, Any]] = None
    report_path: Optional[str] = None
    review_message_id: Optional[str] = None


class Orchestrator:
    def __init__(
        self,
        project_root: str | Path,
        *,
        provider_factory: ProviderFactory,
        profiles: ProfileRegistry | None = None,
        approval_handler: ApprovalHandler | None = None,
        config: OrchestratorConfig | None = None,
        worktree_base: str | Path | None = None,
        qa_provider_factory: ProviderFactory | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.qa_provider_factory = qa_provider_factory or provider_factory  # QA diagnosis model (no tools)
        self.state_dir = resolve_state_dir(project_root)
        self.events = EventRecorder(self.state_dir)
        self.tasks = TaskStore(self.state_dir, events=self.events)
        self.agents = AgentStore(self.state_dir)
        self.bus = MessageBus(self.state_dir, events=self.events)
        self.worktrees = WorktreeManager(project_root, state_dir=self.state_dir, base_dir=worktree_base, events=self.events)
        self.project_root = self.worktrees.repo_root
        self.profiles = profiles or ProfileRegistry.load(self.project_root)
        self._approval_handler = approval_handler
        self.runner = AgentRunner(
            provider_factory=provider_factory, events=self.events, store=self.agents, bus=self.bus,
            approval_handler=self._approve, sessions_dir=self.state_dir / "sessions",
        )
        self._cancel = threading.Event()
        self._graph_lock = threading.Lock()

    # -- approvals -------------------------------------------------------------------
    def _approve(self, request) -> bool:
        if self._approval_handler is None:
            return False
        return bool(self._approval_handler(request))

    # -- planning ------------------------------------------------------------------------
    def create_graph(self, goal: str, plan: list[TaskSpec]) -> TaskGraph:
        for spec in plan:
            self.profiles.get(spec.profile)  # fail fast on unknown profiles
        base = run_git(["rev-parse", "HEAD"], self.project_root).stdout.strip()
        graph = self.tasks.create_graph(goal, base_ref=base, settings={
            "provider": self.config.provider, "model": self.config.model,
            "max_attempts": self.config.max_attempts, "base_branch": self.worktrees.current_branch(),
        })
        root = self.tasks.add_task(Task(title=f"Project: {goal}", description=goal, graph_id=graph.id,
                                        kind="project", owner=HUMAN, priority=100))
        ids: dict[str, str] = {}
        for spec in plan:
            task = Task(
                title=spec.title, description=spec.full_description(), graph_id=graph.id, profile=spec.profile,
                priority=spec.priority, dependencies=[ids[k] for k in spec.depends_on],
                max_attempts=self.config.max_attempts,
            )
            ids[spec.key] = self.tasks.add_task(task).id
        self.tasks.set_dependencies(root.id, list(ids.values()))
        graph.root_task_id = root.id
        self.tasks.update_graph(graph)
        self.tasks.refresh(graph.id)
        self.events.emit("graph.planned", graph_id=graph.id, goal=goal, tasks=len(plan), root_task_id=root.id)
        return graph

    # -- control -----------------------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()

    def pause_task(self, task_id: str, reason: str = "") -> Task:
        return self.tasks.pause(task_id, reason)

    def resume_task(self, task_id: str) -> Task:
        return self.tasks.resume(task_id)

    def retry_task(self, task_id: str) -> Task:
        return self.tasks.retry(task_id)

    def cancel_task(self, task_id: str) -> Task:
        return self.tasks.cancel(task_id)

    # -- execution -------------------------------------------------------------------------
    def run(self, graph_id: str) -> RunSummary:
        graph = self._require_graph(graph_id)
        graph.status = "RUNNING"
        self.tasks.update_graph(graph)
        self._recover_interrupted(graph_id)
        self.events.emit("graph.started", graph_id=graph_id)

        running: dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_parallel), thread_name_prefix="agent") as pool:
            while not self._cancel.is_set():
                for task in self.tasks.ready_tasks(graph_id):
                    if len(running) >= max(1, self.config.max_parallel):
                        break
                    if task.id in running.values():
                        continue
                    self._mark_running(task.id)
                    running[pool.submit(self._execute_task, task.id)] = task.id
                if not running:
                    break
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for fut in done:
                    task_id = running.pop(fut)
                    try:
                        fut.result()
                    except Exception as e:  # defensive: orchestration bug must not lose the task
                        self._record_failure(task_id, None, None, f"orchestrator error: {type(e).__name__}: {e}")
            for fut in running:
                fut.result()
        return self._finalize(graph_id)

    def _recover_interrupted(self, graph_id: str) -> None:
        """Tasks left RUNNING by a crashed/killed run become retryable (worktree preserved)."""
        for task in self.tasks.list(graph_id=graph_id, status=TaskStatus.RUNNING):
            if task.agent:
                agent = self.agents.get(task.agent)
                if agent is not None and not agent.is_terminal:
                    agent.transition(AgentStatus.CANCELLED)
                    agent.error = "orchestrator interrupted"
                    self.agents.save(agent)
            self._record_failure(task.id, None, None, "interrupted: the previous orchestrator run stopped mid-task")

    def _mark_running(self, task_id: str) -> None:
        def apply(t: Task) -> None:
            t.status = TaskStatus.RUNNING
            t.attempts += 1
            t.error = None

        task = self.tasks.mutate(task_id, apply)
        self.events.emit("task.started", task_id=task.id, graph_id=task.graph_id, attempt=task.attempts,
                         profile=task.profile)

    def _execute_task(self, task_id: str) -> None:
        task = self.tasks.require(task_id)
        graph = self._require_graph(task.graph_id)
        profile = self.profiles.get(task.profile or "Developer")
        lock = None
        agent: Agent | None = None
        try:
            worktree = self._prepare_worktree(task, graph)
            lock = self.worktrees.acquire(worktree.name, owner=task.id)
            if self.config.executor == "developer":
                self._execute_with_developer(self.tasks.require(task.id), graph, profile, worktree)
                return
            agent = self.runner.create_agent(profile, task_id=task.id, working_directory=worktree.path,
                                             provider=self.config.provider, model=self.config.model,
                                             attempt=task.attempts)
            self.tasks.mutate(task.id, lambda t: (setattr(t, "agent", agent.id), setattr(t, "worktree", worktree.name),
                                                  setattr(t, "branch", worktree.branch)))
            assignment = self._assignment(self.tasks.require(task.id), graph, worktree)
            self.bus.send(AgentMessage(type=MessageType.TASK_ASSIGNMENT, sender=ORCHESTRATOR, recipient=agent.id,
                                       content=f"{task.title}", task_id=task.id,
                                       payload={"worktree": worktree.path, "branch": worktree.branch,
                                                "attempt": task.attempts, "profile": profile.name}))
            result = self.runner.run(agent, assignment, should_cancel=self._cancel.is_set)
            result = self._verify_and_commit(task, profile, worktree, agent, result)
            self._record_result(task.id, agent, result)
        except MergeConflict as e:
            self._block(task.id, f"merge conflict preparing worktree: {e}")
        except WorktreeBusy as e:
            self._record_failure(task.id, agent, None, f"worktree busy: {e}")
        except GitError as e:
            self._record_failure(task.id, agent, None, f"git error: {e}")
        finally:
            if lock is not None:
                lock.release()

    def _execute_with_developer(self, task: Task, graph: TaskGraph, profile, worktree) -> None:
        """Run one graph task through DeveloperAgent (measured result), commit it, record it."""
        from ..agents.developer import DeveloperAgent

        self.tasks.mutate(task.id, lambda t: (setattr(t, "worktree", worktree.name), setattr(t, "branch", worktree.branch)))
        description = task.description
        deps = [self.tasks.require(d) for d in task.dependencies]
        if deps:
            description += "\n\nCompleted dependencies (their committed changes are in your workspace):\n" + "\n".join(
                f"- {d.title}: {(d.result or {}).get('summary', '')} Files: {', '.join(d.files_changed) or '-'}"
                for d in deps)
        spec = TaskSpec(key=task.id, title=task.title, description=description, profile=profile.name)
        self.bus.send(AgentMessage(type=MessageType.TASK_ASSIGNMENT, sender=ORCHESTRATOR, recipient=f"agent:{task.id}",
                                   content=task.title, task_id=task.id,
                                   payload={"worktree": worktree.path, "branch": worktree.branch,
                                            "attempt": task.attempts, "profile": profile.name, "executor": "developer"}))
        developer = DeveloperAgent(None, runner=self.runner, profiles=self.profiles, profile=profile.name,
                                   provider=self.config.provider, model=self.config.model,
                                   test_command=self.config.test_command)
        measured = developer.execute(spec, worktree.path, should_cancel=self._cancel.is_set)
        attempt_log = [_developer_entry(1, measured)]
        files = list(measured.files_changed)
        limit = max(0, min(self.config.max_repairs, MAX_REPAIRS))
        repairs = 0
        while measured.status != "completed" and repairs < limit and measured.agent_id and not self._cancel.is_set():
            diagnosis, qa_agent = self._diagnose(task, spec, measured, worktree)
            attempt_log.append(_qa_entry(repairs + 1, qa_agent, diagnosis))
            if diagnosis.status != "diagnosed":
                break  # never repair blind
            repairs += 1
            self.events.emit("task.repair", task_id=task.id, graph_id=graph.id, attempt=repairs + 1,
                             reason=diagnosis.summary[:200])
            measured = developer.execute(_repair_spec(spec, measured, diagnosis, repairs, limit), worktree.path,
                                         should_cancel=self._cancel.is_set)  # same worktree and branch
            files = sorted(set(files) | set(measured.files_changed))
            attempt_log.append(_developer_entry(repairs + 1, measured))
        measured.files_changed = files
        tests = ({"command": measured.test_command, "run": measured.tests_run, "passed": measured.tests_passed,
                  "failed": measured.tests_failed, "output_tail": measured.test_output}
                 if measured.test_command else {})
        self.tasks.mutate(task.id, lambda t: (setattr(t, "agent", measured.agent_id), setattr(t, "tests", tests)))

        commit = None
        if measured.status == "completed" and profile.commits and measured.files_changed:
            try:  # dependents' worktrees are created from this branch
                commit = self.worktrees.commit(worktree.name, f"{task.title}\n\nTask: {task.id}\n"
                                                              f"Agent: {measured.agent_id} ({profile.name})")
            except GitError as e:
                measured.status, measured.error = "failed", str(e)
        result = AgentResult(
            task_id=task.id, agent_id=measured.agent_id or "", status=measured.status, summary=measured.summary,
            files_changed=list(measured.files_changed), tests_run=measured.tests_run,
            tests_passed=measured.tests_passed, tests_failed=measured.tests_failed, warnings=list(measured.warnings),
            commit=commit, error=measured.error, reported_by_agent=measured.agent_reported_status is not None,
        )
        agent = self.agents.get(measured.agent_id) if measured.agent_id else None
        if agent is None:
            self._record_failure(task.id, None, result, measured.error or "developer agent did not run")
        else:
            self._record_result(task.id, agent, result)
        if limit:
            attempt_log.append({"step": "final", "status": "COMPLETED" if result.status == "completed" else "FAILED",
                                "repairs": repairs, "commit": commit})

            def log(t: Task) -> None:  # small structured record of every developer/QA step
                if t.result is not None:
                    t.result["attempt_log"] = attempt_log
                if t.history:
                    t.history[-1]["attempt_log"] = attempt_log

            self.tasks.mutate(task.id, log)

    def _diagnose(self, task: Task, spec: TaskSpec, measured, worktree):
        """Read-only QA diagnosis of a measured failure (existing QADiagnoser; the model gets no tools)."""
        from ..agents.qa import QADiagnoser, QADiagnosis

        qa_profile = next((p for p in self.profiles.all() if p.role.lower() == "qa"), None)
        if qa_profile is None:
            return QADiagnosis(status="inconclusive", summary="No QA profile is configured.", task_id=task.id), None
        qa_profile = _read_only(qa_profile)
        agent = self.runner.create_agent(qa_profile, task_id=task.id, working_directory=worktree.path,
                                         provider=self.config.provider, model=self.config.model,
                                         attempt=task.attempts)
        agent.transition(AgentStatus.RUNNING)
        self.agents.save(agent)
        try:
            diagnosis = QADiagnoser(self.qa_provider_factory(agent)).diagnose(spec, measured)
        except Exception as e:  # provider construction failed; QADiagnoser itself never raises
            diagnosis = QADiagnosis(status="inconclusive", summary="The QA model could not be created.",
                                    task_id=task.id, warnings=[f"{type(e).__name__}: {e}"])
        agent.result = diagnosis.to_dict()
        agent.transition(AgentStatus.COMPLETED if diagnosis.status == "diagnosed" else AgentStatus.FAILED)
        self.agents.save(agent)
        self.events.emit("qa.diagnosed", task_id=task.id, agent_id=agent.id, status=diagnosis.status,
                         reason=diagnosis.summary[:200])
        return diagnosis, agent.id

    def _prepare_worktree(self, task: Task, graph: TaskGraph):
        deps = [self.tasks.require(d) for d in task.dependencies]
        dep_branches = [d.branch for d in deps if d.branch]
        base = dep_branches[0] if dep_branches else (graph.base_ref or "HEAD")
        role = (self.profiles.get(task.profile or "Developer").role or "agent")
        existing = {w.name: w for w in self.worktrees.list(managed_only=True)}
        worktree = self.worktrees.create(task.id, role, base_ref=base)
        fresh = worktree.name not in existing
        if fresh:
            for branch in dep_branches[1:]:
                self.worktrees.merge_branch(worktree.name, branch, message=f"Merge {branch} into {worktree.branch}")
        return self.worktrees.get(worktree.name)

    def _assignment(self, task: Task, graph: TaskGraph, worktree) -> str:
        profile = self.profiles.get(task.profile or "Developer")
        lines = [
            f"# Assignment: {task.title}",
            f"Project goal: {graph.goal}",
            f"Task id: {task.id} (attempt {task.attempts} of {task.max_attempts})",
            f"Your working directory: {worktree.path}",
            f"Your branch: {worktree.branch} (the orchestrator commits your changes; do not commit)",
            "",
            "## Task",
            task.description,
        ]
        deps = [self.tasks.require(d) for d in task.dependencies]
        if deps:
            lines += ["", "## Completed dependencies (already merged into your worktree)"]
            for dep in deps:
                r = dep.result or {}
                lines.append(f"- {dep.title} [{dep.profile}] branch {dep.branch}: {r.get('summary', '')}"
                             + (f" Files: {', '.join(r.get('files_changed') or [])}" if r.get("files_changed") else ""))
        if task.history:
            lines += ["", "## Previous attempts failed. Do NOT repeat the same approach"]
            for h in task.history:
                lines.append(f"- Attempt {h.get('attempt')}: {h.get('error') or h.get('status')}."
                             + (f" Summary: {h.get('summary')}" if h.get("summary") else "")
                             + (f" Stop reason: {h.get('stop_reason')}." if h.get("stop_reason") else "")
                             + (f" Warnings: {'; '.join(h.get('warnings') or [])}" if h.get("warnings") else ""))
            try:
                leftover = self.worktrees.changed_files(worktree.name)
            except GitError:
                leftover = []
            if leftover:
                lines.append(f"- Uncommitted work from the previous attempt is still in your worktree: {', '.join(leftover)}")
            lines.append("Diagnose why the previous attempt failed before changing anything, then take a different approach.")
        lines += ["", "## Responsibilities"] + [f"- {r}" for r in profile.responsibilities]
        lines += ["", "When finished, call ReportResult exactly once."]
        return redact_secrets("\n".join(lines))

    def _verify_and_commit(self, task: Task, profile, worktree, agent: Agent, result: AgentResult) -> AgentResult:
        """Replace agent claims with facts from git and the test runner."""
        try:
            actual_files = self.worktrees.changed_files(worktree.name)
        except GitError:
            actual_files = []
        claimed = set(result.files_changed)
        missing = sorted(claimed - set(actual_files))
        if missing:
            result.warnings.append(f"agent reported changes that do not exist in its worktree: {missing}")
        if actual_files and not profile.commits:
            result.warnings.append(f"read-only profile left changes that will not be committed: {actual_files}")
        result.files_changed = actual_files  # authoritative: git, not the agent's claim

        if profile.role == "qa":
            run = run_tests(Path(worktree.path), self.config.test_command)
            if run is not None:
                self.events.emit("tests.run", task_id=task.id, command=run.command, exit_code=run.exit_code,
                                 run=run.run, passed=run.passed, failed=run.failed)
                self.tasks.mutate(task.id, lambda t: setattr(t, "tests", run.to_dict()))
                if (result.tests_run, result.tests_passed, result.tests_failed) != (run.run, run.passed, run.failed):
                    if result.tests_run:
                        result.warnings.append(f"agent-reported test counts differ from measured "
                                               f"({result.tests_run}/{result.tests_passed}/{result.tests_failed})")
                result.tests_run, result.tests_passed, result.tests_failed = run.run, run.passed, run.failed
                if result.status == "completed" and not run.ok:
                    result.status, result.error = "failed", f"{run.failed} test(s) failed (exit {run.exit_code})"

        if result.status == "completed" and profile.commits and actual_files:
            try:
                sha = self.worktrees.commit(worktree.name, f"{task.title}\n\nTask: {task.id}\nAgent: {agent.id} ({profile.name})\n"
                                                           f"Summary: {result.summary[:500]}")
                result.commit = sha
            except GitError as e:
                result.status, result.error = "failed", str(e)
        return result

    def _record_result(self, task_id: str, agent: Agent, result: AgentResult) -> None:
        self.bus.send(AgentMessage(type=MessageType.TASK_RESULT, sender=agent.id, recipient=ORCHESTRATOR,
                                   content=result.summary, task_id=task_id,
                                   payload={"status": result.status, "commit": result.commit,
                                            "files_changed": result.files_changed}), sender_is_agent=True)
        if result.review_verdict:
            self.bus.send(AgentMessage(type=MessageType.REVIEW_RESULT, sender=agent.id, recipient=ORCHESTRATOR,
                                       content=result.review_verdict, task_id=task_id,
                                       payload={"risks": result.risks}), sender_is_agent=True)
        if result.status == "completed":
            def done(t: Task) -> None:
                t.status = TaskStatus.COMPLETED
                t.result = result.to_dict()
                t.files_changed = list(result.files_changed)
                t.error = None
                t.history.append(self._history_entry(t, agent, result))

            task = self.tasks.mutate(task_id, done)
            self.events.emit("task.completed", task_id=task_id, graph_id=task.graph_id, agent_id=agent.id,
                             commit=result.commit, files_changed=result.files_changed)
        elif result.status == "blocked":
            def blocked(t: Task) -> None:
                t.status = TaskStatus.BLOCKED
                t.blocked_reason = f"agent reported blocked: {result.summary[:200]}"
                t.result = result.to_dict()
                t.history.append(self._history_entry(t, agent, result))

            task = self.tasks.mutate(task_id, blocked)
            self.bus.send(AgentMessage(type=MessageType.BLOCKED, sender=agent.id, recipient=ORCHESTRATOR,
                                       content=result.summary, task_id=task_id), sender_is_agent=True)
            self.events.emit("task.blocked", task_id=task_id, graph_id=task.graph_id, reason=task.blocked_reason)
        else:
            self._record_failure(task_id, agent, result, result.error or "agent failed")

    def _history_entry(self, task: Task, agent: Agent | None, result: AgentResult | None, error: str | None = None) -> dict:
        return {
            "attempt": task.attempts,
            "agent": agent.id if agent else None,
            "status": result.status if result else "failed",
            "summary": result.summary if result else None,
            "error": error or (result.error if result else None),
            "stop_reason": result.stop_reason if result else None,
            "warnings": list(result.warnings) if result else [],
            "commit": result.commit if result else None,
            "at": now_iso(),
        }

    def _record_failure(self, task_id: str, agent: Agent | None, result: AgentResult | None, error: str) -> None:
        def fail(t: Task) -> None:
            t.history.append(self._history_entry(t, agent, result, error))
            t.error = error
            if result is not None:
                t.result = result.to_dict()
            # Worktree/branch are preserved for inspection and for the next attempt.
            t.status = TaskStatus.READY if t.attempts < t.max_attempts and not self._cancel.is_set() else TaskStatus.FAILED

        task = self.tasks.mutate(task_id, fail)
        if agent is not None:
            self.bus.send(AgentMessage(type=MessageType.ERROR, sender=agent.id, recipient=ORCHESTRATOR, content=error,
                                       task_id=task_id), sender_is_agent=True)
        if task.status is TaskStatus.READY:
            self.events.emit("task.retry", task_id=task_id, graph_id=task.graph_id, attempt=task.attempts,
                             max_attempts=task.max_attempts, error=error)
        else:
            self.events.emit("task.failed", task_id=task_id, graph_id=task.graph_id, attempts=task.attempts, error=error,
                             worktree=task.worktree)
        self.tasks.refresh(task.graph_id)

    def _block(self, task_id: str, reason: str) -> None:
        def apply(t: Task) -> None:
            t.status = TaskStatus.BLOCKED
            t.blocked_reason = reason
            t.error = reason

        task = self.tasks.mutate(task_id, apply)
        self.events.emit("task.blocked", task_id=task_id, graph_id=task.graph_id, reason=reason)

    # -- completion / review -----------------------------------------------------------------
    def _finalize(self, graph_id: str) -> RunSummary:
        graph = self._require_graph(graph_id)
        work = [t for t in self.tasks.list(graph_id=graph_id) if t.kind != "project"]
        summary = RunSummary(
            graph_id=graph_id, status="",
            completed=[t.id for t in work if t.status is TaskStatus.COMPLETED],
            failed=[t.id for t in work if t.status is TaskStatus.FAILED],
            blocked=[t.id for t in work if t.status in (TaskStatus.BLOCKED, TaskStatus.PENDING, TaskStatus.READY)],
        )
        root_id = graph.root_task_id
        if self._cancel.is_set():
            graph.status = "CANCELLED"
        elif work and len(summary.completed) == len(work):
            last = self._final_task(work)
            if self.config.run_verification and last is not None and last.worktree:
                verification = run_tests(Path(self.worktrees.get(last.worktree).path), self.config.test_command)
                if verification is not None:
                    summary.verification = verification.to_dict()
                    graph.settings["verification"] = summary.verification
                    self.events.emit("tests.run", graph_id=graph_id, task_id=last.id, command=verification.command,
                                     exit_code=verification.exit_code, run=verification.run, passed=verification.passed,
                                     failed=verification.failed, stage="verification")
            verified = summary.verification is None or TestRun(**summary.verification).ok
            graph.status = "REVIEW" if verified else "FAILED"
            if root_id:
                self.tasks.mutate(root_id, lambda t: (setattr(t, "status", TaskStatus.REVIEW if verified else TaskStatus.BLOCKED),
                                                      setattr(t, "blocked_reason", None if verified else "verification tests failed")))
            if verified:
                msg = self.bus.send(AgentMessage(
                    type=MessageType.REVIEW_REQUEST, sender=ORCHESTRATOR, recipient=HUMAN, task_id=root_id,
                    content=f"All {len(work)} tasks completed for '{graph.goal}'. Review, then approve merging into "
                            f"the integration branch with: increscent review {graph_id} --merge",
                    payload={"graph_id": graph_id, "branches": [t.branch for t in work if t.branch],
                             "verification": summary.verification},
                ))
                summary.review_message_id = msg.id
                self.events.emit("review.requested", graph_id=graph_id, message_id=msg.id)
        else:
            graph.status = "FAILED" if summary.failed else "BLOCKED"
            if root_id:
                self.tasks.mutate(root_id, lambda t: (setattr(t, "status", TaskStatus.BLOCKED),
                                                      setattr(t, "blocked_reason", f"{len(summary.failed)} failed, "
                                                                                   f"{len(summary.blocked)} not completed")))
        summary.status = graph.status
        self.tasks.update_graph(graph)
        summary.report_path = str(self.write_report(graph_id))
        self.events.emit("graph.finished", graph_id=graph_id, status=graph.status, completed=len(summary.completed),
                         failed=len(summary.failed), blocked=len(summary.blocked))
        return summary

    @staticmethod
    def _final_task(work: list[Task]) -> Optional[Task]:
        """The sink of the work graph (no other task depends on it): its worktree has everything merged."""
        depended_on = {d for t in work for d in t.dependencies}
        sinks = [t for t in work if t.id not in depended_on and t.worktree]
        return sorted(sinks, key=lambda t: t.created_at)[-1] if sinks else None

    def report_data(self, graph_id: str) -> dict[str, Any]:
        graph = self._require_graph(graph_id)
        return {
            "goal": graph.goal,
            "graph": graph.to_dict(),
            "tasks": [t.to_dict() for t in self.tasks.topological_order(graph_id) if t.kind != "project"],
            "verification": graph.settings.get("verification"),
            "review": {"next_step": (f"Human review required. Approve merge into the integration branch with "
                                     f"`increscent review {graph_id} --merge`. Nothing has been merged or pushed.")
                       if graph.status == "REVIEW" else f"Graph status {graph.status}."},
        }

    def write_report(self, graph_id: str) -> Path:
        path = self.state_dir / "reports" / f"{graph_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_delivery_report(self.report_data(graph_id)), encoding="utf-8")
        atomic_write_json(path.with_suffix(".json"), self.report_data(graph_id))
        return path

    def review(self, graph_id: str) -> dict[str, Any]:
        """Per-task branches, commits and diff stats for human review."""
        graph = self._require_graph(graph_id)
        items = []
        for task in self.tasks.topological_order(graph_id):
            if task.kind == "project":
                continue
            diffstat = ""
            if task.worktree:
                try:
                    # this task's own changes (relative to the base its worktree started from)
                    diffstat = self.worktrees.diff(task.worktree, stat=True)
                except GitError as e:
                    diffstat = f"(unavailable: {e})"
            items.append({"task_id": task.id, "title": task.title, "status": task.status.value, "branch": task.branch,
                          "commit": (task.result or {}).get("commit"), "diffstat": diffstat,
                          "risks": (task.result or {}).get("risks", []),
                          "review_verdict": (task.result or {}).get("review_verdict")})
        return {"graph": graph.to_dict(), "tasks": items}

    def prepare_merge(self, graph_id: str, *, approved: bool) -> dict[str, Any]:
        """Merge completed task branches into increscent/<graph>/integration. Requires explicit human approval."""
        graph = self._require_graph(graph_id)
        if graph.status != "REVIEW":
            raise GitError(f"graph {graph_id} is {graph.status}; only graphs in REVIEW can be merged")
        self.events.emit("approval.requested", kind="merge", graph_id=graph_id, risk="HIGH",
                         capabilities=["GIT_BRANCH", "GIT_COMMIT"], target=f"increscent/{graph_id}/integration")
        if not approved:
            self.events.emit("approval.denied", kind="merge", graph_id=graph_id)
            raise PermissionError("merge requires explicit human approval")
        self.events.emit("approval.approved", kind="merge", graph_id=graph_id)
        integration = self.worktrees.create(graph_id, INTEGRATION_ROLE, base_ref=graph.base_ref or "HEAD")
        merged: list[str] = []
        for task in self.tasks.topological_order(graph_id):
            if task.kind == "project" or task.status is not TaskStatus.COMPLETED or not task.branch:
                continue
            ahead = run_git(["rev-list", "--count", f"{integration.branch}..{task.branch}"], self.project_root).stdout.strip()
            if ahead == "0":
                continue
            try:
                self.worktrees.merge_branch(integration.name, task.branch, message=f"Integrate {task.title} ({task.id})")
            except MergeConflict as e:
                self.events.emit("merge.conflict", graph_id=graph_id, branch=task.branch, files=e.files)
                raise
            merged.append(task.branch)
        sha = run_git(["rev-parse", "HEAD"], integration.path).stdout.strip()
        graph.integration_branch = integration.branch
        graph.status = "INTEGRATED"
        self.tasks.update_graph(graph)
        if graph.root_task_id:
            self.tasks.mutate(graph.root_task_id, lambda t: setattr(t, "status", TaskStatus.COMPLETED))
        self.events.emit("merge.prepared", graph_id=graph_id, integration_branch=integration.branch, sha=sha, merged=merged)
        self.write_report(graph_id)
        return {"integration_branch": integration.branch, "worktree": integration.path, "sha": sha, "merged": merged}

    def _require_graph(self, graph_id: str) -> TaskGraph:
        graph = self.tasks.get_graph(graph_id)
        if graph is None:
            raise KeyError(f"unknown graph {graph_id}")
        return graph


def simulated_orchestrator(project_root: str | Path, **kwargs: Any) -> Orchestrator:
    from .simulation import simulated_provider_factory

    return Orchestrator(project_root, provider_factory=simulated_provider_factory, **kwargs)


__all__ = ["Orchestrator", "OrchestratorConfig", "RunSummary", "simulated_orchestrator", "Callable"]


# -- QA-guided repair (Phase 3.3.2) ------------------------------------------------------------

def _read_only(profile):
    """The QA profile as used for diagnosis: no tools at all, READ capability only."""
    from ..security.permissions import Capability

    every = set(profile.allow) | set(profile.ask) | set(profile.deny)
    return dataclasses.replace(profile, allowed_tools=(), allow=frozenset({Capability.READ}), ask=frozenset(),
                               deny=frozenset(every - {Capability.READ}), commits=False)


def _developer_entry(attempt: int, measured) -> dict[str, Any]:
    return {"step": "developer", "attempt": attempt, "repair": attempt > 1, "agent": measured.agent_id,
            "status": measured.status, "agent_claimed": measured.agent_reported_status,
            "tests": {"run": measured.tests_run, "passed": measured.tests_passed, "failed": measured.tests_failed},
            "files_changed": list(measured.files_changed), "error": measured.error}


def _qa_entry(after_attempt: int, agent_id, diagnosis) -> dict[str, Any]:
    return {"step": "qa", "after_attempt": after_attempt, "agent": agent_id, "status": diagnosis.status,
            "confidence": diagnosis.confidence, "summary": diagnosis.summary,
            "failures": [dataclasses.asdict(f) for f in diagnosis.failures],
            "recommended_fixes": list(diagnosis.recommended_fixes), "warnings": list(diagnosis.warnings)}


def _repair_spec(spec: TaskSpec, measured, diagnosis, repair: int, limit: int) -> TaskSpec:
    """The original task, unchanged, followed by the repair context (failure, QA diagnosis, test output)."""
    from ..agents.developer import REPAIR_MARKER

    failures = [f"- {f.test}: {f.error}\n  likely cause: {f.likely_cause}\n  evidence: {f.evidence}"
                for f in diagnosis.failures]
    output = (measured.test_output or "").strip()[-2000:]
    context = [
        "", "", f"{REPAIR_MARKER} {repair} of {limit}",
        "This is a REPAIR attempt. Your previous implementation of the task above failed measured verification.",
        "The original task and its acceptance criteria are unchanged. Fix the existing implementation in this",
        "same workspace (its files are already here); do not start over and do not work anywhere else.",
        "", "### Previous failure",
        f"Summary: {measured.summary or '-'}", f"Error: {measured.error or '-'}",
        f"Measured tests: run {measured.tests_run}, passed {measured.tests_passed}, failed {measured.tests_failed}"
        + (f" ({measured.test_command})" if measured.test_command else ""),
        "", f"### QA diagnosis ({diagnosis.status}, confidence {diagnosis.confidence})", diagnosis.summary,
        *failures, "Recommended fixes:", *[f"- {x}" for x in diagnosis.recommended_fixes],
        "", "### Relevant test output", "```", output or "(none)", "```",
    ]
    return dataclasses.replace(spec, description=spec.description.rstrip() + "\n".join(context))
