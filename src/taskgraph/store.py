"""Durable task graph store (``tasks.json`` in the project state directory).

Every mutation is a locked read-modify-write followed by an atomic file
replace, so tasks survive restarts and concurrent agent threads never lose
updates.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..orchestration.state import atomic_write_json, now_iso, read_json
from .models import PAUSED_PREFIX, TERMINAL_TASK_STATUSES, Task, TaskGraph, TaskStatus

SCHEMA_VERSION = 1


class TaskGraphError(ValueError):
    pass


class TaskStore:
    def __init__(self, state_dir: Path, *, events=None) -> None:
        self.path = Path(state_dir) / "tasks.json"
        self._lock = threading.RLock()
        self.events = events

    # -- raw persistence --------------------------------------------------------
    def _load(self) -> dict:
        data = read_json(self.path, {})
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("graphs", {})
        data.setdefault("tasks", {})
        return data

    def _save(self, data: dict) -> None:
        atomic_write_json(self.path, data)

    def _emit(self, event: str, **fields) -> None:
        if self.events is not None:
            self.events.emit(event, **fields)

    # -- graphs ---------------------------------------------------------------------
    def create_graph(self, goal: str, *, base_ref: str | None = None, settings: dict | None = None) -> TaskGraph:
        graph = TaskGraph(goal=goal, base_ref=base_ref, settings=dict(settings or {}))
        with self._lock:
            data = self._load()
            data["graphs"][graph.id] = graph.to_dict()
            self._save(data)
        self._emit("graph.created", graph_id=graph.id, goal=goal, base_ref=base_ref)
        return graph

    def get_graph(self, graph_id: str) -> Optional[TaskGraph]:
        with self._lock:
            raw = self._load()["graphs"].get(graph_id)
        return TaskGraph.from_dict(raw) if raw else None

    def update_graph(self, graph: TaskGraph) -> None:
        graph.updated_at = now_iso()
        with self._lock:
            data = self._load()
            data["graphs"][graph.id] = graph.to_dict()
            self._save(data)

    def list_graphs(self) -> list[TaskGraph]:
        with self._lock:
            graphs = [TaskGraph.from_dict(g) for g in self._load()["graphs"].values()]
        return sorted(graphs, key=lambda g: g.created_at)

    # -- tasks ------------------------------------------------------------------------
    def add_task(self, task: Task) -> Task:
        with self._lock:
            data = self._load()
            if task.graph_id not in data["graphs"]:
                raise TaskGraphError(f"unknown graph {task.graph_id}")
            for dep in task.dependencies:
                if dep not in data["tasks"]:
                    raise TaskGraphError(f"task {task.title!r} depends on unknown task {dep}")
            data["tasks"][task.id] = task.to_dict()
            self._check_acyclic(data["tasks"])
            self._save(data)
        self._emit("task.created", task_id=task.id, graph_id=task.graph_id, title=task.title, profile=task.profile,
                   dependencies=task.dependencies, priority=task.priority)
        return task

    def set_dependencies(self, task_id: str, dependencies: Iterable[str]) -> Task:
        with self._lock:
            data = self._load()
            raw = data["tasks"].get(task_id)
            if raw is None:
                raise TaskGraphError(f"unknown task {task_id}")
            deps = list(dependencies)
            for dep in deps:
                if dep not in data["tasks"]:
                    raise TaskGraphError(f"unknown dependency {dep}")
            candidate = dict(data["tasks"])
            candidate[task_id] = {**raw, "dependencies": deps}
            self._check_acyclic(candidate)  # validate before anything is written
            task = Task.from_dict(candidate[task_id])
            task.updated_at = now_iso()
            data["tasks"][task_id] = task.to_dict()
            self._save(data)
        return task

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            raw = self._load()["tasks"].get(task_id)
        return Task.from_dict(raw) if raw else None

    def require(self, task_id: str) -> Task:
        task = self.get(task_id)
        if task is None:
            raise TaskGraphError(f"unknown task {task_id}")
        return task

    def update(self, task: Task) -> Task:
        task.updated_at = now_iso()
        with self._lock:
            data = self._load()
            data["tasks"][task.id] = task.to_dict()
            self._save(data)
        return task

    def mutate(self, task_id: str, fn: Callable[[Task], None]) -> Task:
        """Atomic read-modify-write of one task."""
        with self._lock:
            data = self._load()
            raw = data["tasks"].get(task_id)
            if raw is None:
                raise TaskGraphError(f"unknown task {task_id}")
            task = Task.from_dict(raw)
            fn(task)
            task.updated_at = now_iso()
            data["tasks"][task_id] = task.to_dict()
            self._save(data)
            return task

    def list(self, *, graph_id: str | None = None, status: TaskStatus | None = None) -> list[Task]:
        with self._lock:
            tasks = [Task.from_dict(t) for t in self._load()["tasks"].values()]
        if graph_id is not None:
            tasks = [t for t in tasks if t.graph_id == graph_id]
        if status is not None:
            tasks = [t for t in tasks if t.status is status]
        return sorted(tasks, key=lambda t: t.created_at)

    # -- dependency resolution ------------------------------------------------------------
    @staticmethod
    def _check_acyclic(tasks: dict[str, dict]) -> None:
        state: dict[str, int] = {}

        def visit(tid: str, path: list[str]) -> None:
            if state.get(tid) == 1:
                raise TaskGraphError("dependency cycle: " + " -> ".join(path + [tid]))
            if state.get(tid) == 2:
                return
            state[tid] = 1
            for dep in tasks.get(tid, {}).get("dependencies", []):
                visit(dep, path + [tid])
            state[tid] = 2

        for tid in tasks:
            visit(tid, [])

    def refresh(self, graph_id: str) -> list[Task]:
        """Promote/demote tasks according to their dependencies. Returns changed tasks."""
        changed: list[Task] = []
        with self._lock:
            data = self._load()
            tasks = {tid: Task.from_dict(raw) for tid, raw in data["tasks"].items() if raw["graph_id"] == graph_id}
            for task in tasks.values():
                if task.kind == "project" or task.is_terminal or task.status in (TaskStatus.RUNNING, TaskStatus.REVIEW):
                    continue
                if task.is_paused:
                    continue
                deps = [tasks.get(d) for d in task.dependencies]
                failed = [d for d in deps if d is not None and d.status in (TaskStatus.FAILED, TaskStatus.CANCELLED)]
                blocked_deps = [d for d in deps if d is not None and d.status is TaskStatus.BLOCKED]
                if failed:
                    new_status, reason = TaskStatus.BLOCKED, f"dependency {failed[0].id} ({failed[0].title}) {failed[0].status.value.lower()}"
                elif all(d is not None and d.status is TaskStatus.COMPLETED for d in deps):
                    new_status, reason = TaskStatus.READY, None
                elif blocked_deps and task.status is TaskStatus.BLOCKED:
                    continue
                else:
                    new_status, reason = TaskStatus.PENDING, None
                if task.status is TaskStatus.BLOCKED and task.blocked_reason and \
                        not task.blocked_reason.startswith("dependency") and new_status is not TaskStatus.BLOCKED:
                    continue  # blocked for its own reason (merge conflict, agent blocked): needs explicit resume
                if (new_status, reason) != (task.status, task.blocked_reason):
                    task.status, task.blocked_reason = new_status, reason
                    task.updated_at = now_iso()
                    data["tasks"][task.id] = task.to_dict()
                    changed.append(task)
            if changed:
                self._save(data)
        for task in changed:
            if task.status is TaskStatus.BLOCKED:
                self._emit("task.blocked", task_id=task.id, graph_id=graph_id, reason=task.blocked_reason)
            elif task.status is TaskStatus.READY:
                self._emit("task.ready", task_id=task.id, graph_id=graph_id)
        return changed

    def ready_tasks(self, graph_id: str) -> list[Task]:
        self.refresh(graph_id)
        ready = self.list(graph_id=graph_id, status=TaskStatus.READY)
        return sorted(ready, key=lambda t: (-t.priority, t.created_at))

    def topological_order(self, graph_id: str) -> list[Task]:
        tasks = {t.id: t for t in self.list(graph_id=graph_id)}
        order: list[Task] = []
        seen: set[str] = set()

        def visit(tid: str) -> None:
            if tid in seen or tid not in tasks:
                return
            seen.add(tid)
            for dep in tasks[tid].dependencies:
                visit(dep)
            order.append(tasks[tid])

        for tid in sorted(tasks, key=lambda t: tasks[t].created_at):
            visit(tid)
        return order

    # -- manual control -----------------------------------------------------------------------
    def pause(self, task_id: str, reason: str = "") -> Task:
        def apply(task: Task) -> None:
            if task.is_terminal or task.status is TaskStatus.RUNNING:
                raise TaskGraphError(f"cannot pause task in status {task.status.value}")
            task.status = TaskStatus.BLOCKED
            task.blocked_reason = f"{PAUSED_PREFIX}: {reason}".rstrip(": ") if reason else PAUSED_PREFIX

        task = self.mutate(task_id, apply)
        self._emit("task.blocked", task_id=task.id, graph_id=task.graph_id, reason=task.blocked_reason)
        return task

    def resume(self, task_id: str) -> Task:
        def apply(task: Task) -> None:
            if task.status is not TaskStatus.BLOCKED:
                raise TaskGraphError(f"task {task.id} is not blocked/paused")
            task.status = TaskStatus.PENDING
            task.blocked_reason = None

        task = self.mutate(task_id, apply)
        self.refresh(task.graph_id)
        self._emit("task.resumed", task_id=task.id, graph_id=task.graph_id)
        return self.require(task_id)

    def cancel(self, task_id: str) -> Task:
        def apply(task: Task) -> None:
            if task.status in TERMINAL_TASK_STATUSES:
                raise TaskGraphError(f"task {task.id} is already {task.status.value}")
            task.status = TaskStatus.CANCELLED

        task = self.mutate(task_id, apply)
        self._emit("task.cancelled", task_id=task.id, graph_id=task.graph_id)
        self.refresh(task.graph_id)
        return task

    def retry(self, task_id: str) -> Task:
        """Give a FAILED task another attempt (history/failure context is kept)."""
        def apply(task: Task) -> None:
            if task.status is not TaskStatus.FAILED:
                raise TaskGraphError(f"only FAILED tasks can be retried (task is {task.status.value})")
            task.status = TaskStatus.PENDING
            task.max_attempts = max(task.max_attempts, task.attempts + 1)
            task.error = None

        task = self.mutate(task_id, apply)
        self.refresh(task.graph_id)
        self._emit("task.retry", task_id=task.id, graph_id=task.graph_id, attempts=task.attempts)
        return self.require(task_id)
