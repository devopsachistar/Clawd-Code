"""Multi-agent CLI commands: team, agents, agent, tasks, task, worktrees, worktree, review."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

EXIT_OK, EXIT_FAILURE, EXIT_CONFIG, EXIT_PERMISSION = 0, 1, 2, 3


def add_team_commands(subparsers: argparse._SubParsersAction, add_common) -> None:
    team = subparsers.add_parser("team", help="Plan and run a multi-agent task graph (stops before merging)")
    team.add_argument("goal", nargs="*", help="What to build")
    team.add_argument("--simulate", action="store_true",
                      help="Use the offline scripted demo provider instead of a real model (safe local demo)")
    team.add_argument("--plan", metavar="FILE", help="JSON plan file instead of the standard plan")
    team.add_argument("--resume", metavar="GRAPH_ID", help="Continue an existing graph (ready/retryable tasks)")
    team.add_argument("--parallel", type=int, default=1, help="Agents to run concurrently (default 1)")
    team.add_argument("--max-attempts", type=int, default=3, help="Attempts per task incl. retries (default 3)")
    team.add_argument("--test-command", help="Command used to verify the integrated result (default: auto-detect)")
    team.add_argument("-y", "--yes", action="store_true",
                      help="Auto-approve LOW/MEDIUM-risk requests (never pushes, deploys, DB writes or secret use)")
    team.add_argument("--cwd", help="Project directory (default: current directory)")
    team.add_argument("--json", action="store_true", help="Print the run summary as JSON")
    add_common(team, suppress=True)

    agents = subparsers.add_parser("agents", help="List agents")
    agents.add_argument("--graph", help="Only agents of this graph")
    agents.add_argument("--cwd")
    agent = subparsers.add_parser("agent", help="Agent details")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    agent_show = agent_sub.add_parser("show", help="Show one agent")
    agent_show.add_argument("agent_id")
    agent_show.add_argument("--cwd")

    tasks = subparsers.add_parser("tasks", help="List tasks")
    tasks.add_argument("--graph", help="Only tasks of this graph")
    tasks.add_argument("--cwd")
    task = subparsers.add_parser("task", help="Task details and control")
    task_sub = task.add_subparsers(dest="task_command", required=True)
    for name, help_text in (("show", "Show one task"), ("pause", "Pause a task"), ("resume", "Resume a paused/blocked task"),
                            ("retry", "Retry a failed task"), ("cancel", "Cancel a task")):
        p = task_sub.add_parser(name, help=help_text)
        p.add_argument("task_id")
        p.add_argument("--cwd")

    worktrees = subparsers.add_parser("worktrees", help="List git worktrees created for agents")
    worktrees.add_argument("--all", action="store_true", help="Include worktrees not created by Increscent")
    worktrees.add_argument("--cwd")
    worktree = subparsers.add_parser("worktree", help="Manage agent worktrees")
    wt_sub = worktree.add_subparsers(dest="worktree_command", required=True)
    wt_remove = wt_sub.add_parser("remove", help="Remove an agent worktree")
    wt_remove.add_argument("name")
    wt_remove.add_argument("--force", action="store_true", help="Discard uncommitted changes")
    wt_remove.add_argument("--delete-branch", action="store_true", help="Also delete its branch")
    wt_remove.add_argument("--cwd")

    review = subparsers.add_parser("review", help="Review a graph's changes; optionally merge into its integration branch")
    review.add_argument("graph_id")
    review.add_argument("--merge", action="store_true", help="Merge completed task branches into increscent/<graph>/integration")
    review.add_argument("--approve", action="store_true",
                        help="Explicit human approval for --merge in non-interactive use (never merges into main)")
    review.add_argument("--cwd")


# ---------------------------------------------------------------------------

def _project(args) -> Path:
    from .project import detect_project

    start = Path(args.cwd).expanduser() if getattr(args, "cwd", None) else None
    return detect_project(start).project_root


def _state(args):
    from .orchestration.state import resolve_state_dir

    return resolve_state_dir(_project(args), create=False)


def _console() -> Console:
    # Piped output defaults to 80 columns; tables of ids/branches need more room.
    return Console(highlight=False, soft_wrap=True, width=None if sys.stdout.isatty() else 170)


def _ask(prompt: str) -> bool:
    """Ask a yes/no question; EOF or anything but yes means no."""
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def make_approval_handler(auto_approve: bool, *, err=None):
    """Explicit human approval on a TTY; otherwise only --yes for auto-approvable requests."""
    lock = threading.Lock()
    stream = err or sys.stderr

    def handler(request) -> bool:
        with lock:
            print("\n[approval required]\n" + request.render(), file=stream, flush=True)
            if auto_approve and request.auto_approvable:
                print(f"[auto-approved] {request.id}", file=stream, flush=True)
                return True
            if _interactive():
                approved = _ask(f"Approve {request.id}? [y/N] ")
                print("[approved]" if approved else "[denied]", file=stream, flush=True)
                return approved
            why = "run interactively to approve" if not request.auto_approvable else "re-run with --yes to approve"
            print(f"[denied] {request.id} (no interactive approver; {why})", file=stream, flush=True)
            return False

    return handler


_PROGRESS_EVENTS = {
    "graph.planned", "task.started", "task.completed", "task.retry", "task.failed", "task.blocked",
    "agent.waiting", "approval.approved", "approval.denied", "worktree.created", "commit.created",
    "tests.run", "review.requested", "graph.finished",
}


def _progress_printer(stream):
    def printer(record: dict[str, Any]) -> None:
        event = record.get("event")
        if event not in _PROGRESS_EVENTS:
            return
        detail = {k: record[k] for k in ("task_id", "graph_id", "agent_id", "attempt", "branch", "sha", "status",
                                         "error", "reason", "run", "passed", "failed", "approval_id") if record.get(k)}
        if "sha" in detail:
            detail["sha"] = str(detail["sha"])[:10]
        print(f"- {event} " + " ".join(f"{k}={v}" for k, v in detail.items()), file=stream, flush=True)

    return printer


def handle_team(args) -> int:
    from .config import ConfigError, create_provider, missing_api_key_message, resolve_provider_config
    from .gitops.worktrees import GitError
    from .orchestration.orchestrator import Orchestrator, OrchestratorConfig
    from .orchestration.planner import load_plan_file, standard_plan
    from .project import detect_project

    console = _console()
    start = Path(args.cwd).expanduser() if args.cwd else None
    project = detect_project(start)
    if project.is_unsafe_root:
        console.print(f"[red]Refusing to run in {project.project_root} (home directory or filesystem root).[/red]")
        return EXIT_CONFIG
    if not project.is_git_repo:
        console.print("[red]`team` needs a git repository with at least one commit "
                      "(agents work in git worktrees). Try: git init && git commit --allow-empty -m init[/red]")
        return EXIT_CONFIG
    goal = " ".join(args.goal).strip()
    if not goal and not args.resume:
        console.print("[red]Give a goal, e.g. increscent team \"Create a simple task management application\"[/red]")
        return EXIT_CONFIG

    if args.simulate:
        from .orchestration.simulation import simulated_provider_factory

        factory, provider_name, model = simulated_provider_factory, "simulated", "simulated"
    else:
        try:
            resolved = resolve_provider_config(cli_provider=getattr(args, "provider", None),
                                               cli_model=getattr(args, "model", None),
                                               cli_base_url=getattr(args, "base_url", None),
                                               project_root=project.project_root)
        except ConfigError as e:
            console.print(f"[red]{e}[/red]")
            return EXIT_CONFIG
        if not resolved.has_api_key:
            console.print(f"[red]{missing_api_key_message(resolved)}[/red]")
            return EXIT_CONFIG

        def factory(agent):
            return create_provider(dataclasses.replace(resolved, model=agent.model or resolved.model))

        provider_name, model = resolved.provider, resolved.model

    config = OrchestratorConfig(provider=provider_name, model=model, max_parallel=max(1, args.parallel),
                                max_attempts=max(1, args.max_attempts), test_command=args.test_command)
    try:
        orch = Orchestrator(project.project_root, provider_factory=factory, config=config,
                            approval_handler=make_approval_handler(args.yes))
    except GitError as e:
        console.print(f"[red]{e}[/red]")
        return EXIT_CONFIG
    for warning in orch.profiles.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    orch.events.subscribe(_progress_printer(sys.stderr))

    try:
        if args.resume:
            graph = orch.tasks.get_graph(args.resume)
            if graph is None:
                console.print(f"[red]Unknown graph {args.resume}[/red]")
                return EXIT_CONFIG
        else:
            plan = load_plan_file(args.plan) if args.plan else standard_plan(goal)
            graph = orch.create_graph(goal, plan)
        print(f"graph: {graph.id}  provider: {provider_name}  model: {model}"
              + ("  (SIMULATION: offline scripted agents)" if args.simulate else ""), file=sys.stderr)
        summary = orch.run(graph.id)
    except (GitError, ValueError, KeyError) as e:
        console.print(f"[red]{e}[/red]")
        return EXIT_CONFIG
    except KeyboardInterrupt:
        orch.cancel()
        console.print("[yellow]Cancelled. Task state and worktrees are preserved; resume with --resume.[/yellow]")
        return EXIT_FAILURE

    if args.json:
        print(json.dumps(dataclasses.asdict(summary), indent=2))
    else:
        _print_tasks(console, orch.tasks.list(graph_id=graph.id))
        console.print(f"\nGraph [bold]{graph.id}[/bold]: [bold]{summary.status}[/bold]")
        if summary.verification:
            v = summary.verification
            console.print(f"Verification: {v['command']} -> run {v['run']}, passed {v['passed']}, failed {v['failed']}")
        console.print(f"Report: {summary.report_path}")
        if summary.status == "REVIEW":
            console.print(f"Nothing was merged. Review with: increscent review {graph.id}   "
                          f"then approve the merge into the integration branch with --merge")
    return EXIT_OK if summary.status == "REVIEW" else EXIT_FAILURE


def _print_tasks(console: Console, tasks) -> None:
    table = Table(show_header=True, header_style="bold")
    for col in ("Task", "Title", "Profile", "Status", "Att.", "Deps", "Branch"):
        table.add_column(col)
    for t in tasks:
        table.add_row(t.id, t.title[:48], t.profile or t.kind, t.status.value, str(t.attempts),
                      ",".join(t.dependencies)[:30] if t.kind != "project" else f"{len(t.dependencies)} tasks",
                      t.branch or "-")
    console.print(table)


def handle_tasks(args) -> int:
    from .taskgraph.store import TaskStore

    store = TaskStore(_state(args))
    tasks = store.list(graph_id=args.graph)
    console = _console()
    if not tasks:
        console.print("No tasks.")
        return EXIT_OK
    _print_tasks(console, tasks)
    return EXIT_OK


def handle_task(args) -> int:
    from .taskgraph.store import TaskGraphError, TaskStore

    store = TaskStore(_state(args))
    console = _console()
    try:
        if args.task_command == "show":
            task = store.require(args.task_id)
            console.print_json(json.dumps(task.to_dict(), default=str))
            return EXIT_OK
        action = {"pause": store.pause, "resume": store.resume, "retry": store.retry, "cancel": store.cancel}[args.task_command]
        task = action(args.task_id)
        console.print(f"{task.id}: {task.status.value}" + (f" ({task.blocked_reason})" if task.blocked_reason else ""))
        return EXIT_OK
    except TaskGraphError as e:
        console.print(f"[red]{e}[/red]")
        return EXIT_CONFIG


def handle_agents(args) -> int:
    from .agents.store import AgentStore
    from .taskgraph.store import TaskStore

    state = _state(args)
    agents = AgentStore(state).list()
    if args.graph:
        task_ids = {t.id for t in TaskStore(state).list(graph_id=args.graph)}
        agents = [a for a in agents if a.task_id in task_ids]
    console = _console()
    if not agents:
        console.print("No agents.")
        return EXIT_OK
    table = Table(show_header=True, header_style="bold")
    for col in ("Agent", "Profile", "Status", "Task", "Attempt", "Model", "Working directory"):
        table.add_column(col)
    for a in agents:
        table.add_row(a.id, a.profile, a.status.value, a.task_id or "-", str(a.attempt), a.model,
                      Path(a.working_directory).name)
    console.print(table)
    return EXIT_OK


def handle_agent(args) -> int:
    from .agents.store import AgentStore

    agent = AgentStore(_state(args)).get(args.agent_id)
    console = _console()
    if agent is None:
        console.print(f"[red]Unknown agent {args.agent_id}[/red]")
        return EXIT_CONFIG
    data = agent.to_dict()
    data["system_prompt"] = data["system_prompt"][:400] + ("..." if len(data["system_prompt"]) > 400 else "")
    console.print_json(json.dumps(data, default=str))
    return EXIT_OK


def handle_worktrees(args) -> int:
    from .gitops.worktrees import GitError, WorktreeManager

    console = _console()
    try:
        manager = WorktreeManager(_project(args))
    except GitError as e:
        console.print(f"[red]{e}[/red]")
        return EXIT_CONFIG
    items = manager.list(managed_only=not args.all)
    if not items:
        console.print("No agent worktrees.")
        return EXIT_OK
    table = Table(show_header=True, header_style="bold")
    for col in ("Name", "Branch", "Task", "Role", "Locked by", "Path"):
        table.add_column(col)
    for w in items:
        table.add_row(w.name, w.branch or "-", w.task_id or "-", w.role or "-", w.locked_by or "-",
                      w.path + ("" if w.exists else " (missing)"))
    console.print(table)
    return EXIT_OK


def handle_worktree(args) -> int:
    from .gitops.worktrees import GitError, WorktreeManager

    console = _console()
    try:
        WorktreeManager(_project(args)).remove(args.name, force=args.force, delete_branch=args.delete_branch)
    except GitError as e:
        console.print(f"[red]{e}[/red]")
        return EXIT_FAILURE
    console.print(f"Removed worktree {args.name}")
    return EXIT_OK


def handle_review(args) -> int:
    from .gitops.worktrees import GitError, MergeConflict
    from .orchestration.orchestrator import Orchestrator

    console = _console()

    def no_provider(agent):  # review/merge never runs agents
        raise RuntimeError("review does not run agents")

    try:
        orch = Orchestrator(_project(args), provider_factory=no_provider)
        review = orch.review(args.graph_id)
    except (GitError, KeyError) as e:
        console.print(f"[red]{e}[/red]")
        return EXIT_CONFIG
    graph = review["graph"]
    console.print(f"Graph [bold]{graph['id']}[/bold] ({graph['status']}): {graph['goal']}")
    for item in review["tasks"]:
        console.print(f"\n[bold]{item['task_id']}[/bold] {item['title']} [{item['status']}] "
                      f"branch {item['branch']} commit {(item['commit'] or '-')[:10]}")
        if item["review_verdict"]:
            console.print(f"  verdict: {item['review_verdict']}")
        for risk in item["risks"]:
            console.print(f"  risk: {risk}")
        if item["diffstat"]:
            console.print("  " + item["diffstat"].replace("\n", "\n  "))
    if not args.merge:
        return EXIT_OK

    approved = args.approve
    if not approved and _interactive():
        approved = _ask(f"\nMerge completed branches into increscent/{args.graph_id}/integration "
                        "(main is never touched)? [y/N] ")
    try:
        outcome = orch.prepare_merge(args.graph_id, approved=approved)
    except PermissionError:
        console.print("[yellow]Merge not approved. Nothing was merged.[/yellow]")
        return EXIT_PERMISSION
    except MergeConflict as e:
        console.print(f"[red]{e}. Integration aborted; nothing on main changed.[/red]")
        return EXIT_FAILURE
    except GitError as e:
        console.print(f"[red]{e}[/red]")
        return EXIT_CONFIG
    console.print(f"\nMerged {len(outcome['merged'])} branch(es) into [bold]{outcome['integration_branch']}[/bold] "
                  f"({outcome['sha'][:10]}) at {outcome['worktree']}. Not pushed; main unchanged.")
    return EXIT_OK


HANDLERS = {
    "team": handle_team,
    "tasks": handle_tasks,
    "task": handle_task,
    "agents": handle_agents,
    "agent": handle_agent,
    "worktrees": handle_worktrees,
    "worktree": handle_worktree,
    "review": handle_review,
}


def dispatch(args) -> Optional[int]:
    handler = HANDLERS.get(getattr(args, "command", None))
    return handler(args) if handler else None
