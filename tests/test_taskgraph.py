"""Durable task graph: persistence, dependencies, control operations."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from src.taskgraph.models import Task, TaskStatus
from src.taskgraph.store import TaskGraphError, TaskStore


class TaskGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.store = TaskStore(self.dir)
        self.graph = self.store.create_graph("demo")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def add(self, title, deps=(), **kw):
        return self.store.add_task(Task(title=title, description="", graph_id=self.graph.id, dependencies=list(deps), **kw))

    def complete(self, task):
        self.store.mutate(task.id, lambda t: setattr(t, "status", TaskStatus.COMPLETED))

    def test_tasks_survive_restart_with_all_fields(self):
        t = self.add("Backend", profile="BackendDeveloper", priority=5)
        self.store.mutate(t.id, lambda x: (setattr(x, "worktree", "wt"), setattr(x, "branch", "increscent/x/backend"),
                                           x.files_changed.append("api.py"), setattr(x, "attempts", 2),
                                           setattr(x, "tests", {"run": 3, "passed": 3, "failed": 0}),
                                           setattr(x, "result", {"summary": "ok"}), setattr(x, "error", None)))
        reloaded = TaskStore(self.dir).require(t.id)  # "restart"
        for field in ("id", "title", "description", "status", "priority", "owner", "agent", "dependencies", "worktree",
                      "branch", "files_changed", "tests", "attempts", "created_at", "updated_at", "result", "error"):
            self.assertTrue(hasattr(reloaded, field), field)
        self.assertEqual((reloaded.branch, reloaded.files_changed, reloaded.attempts), ("increscent/x/backend", ["api.py"], 2))
        self.assertEqual(TaskStore(self.dir).get_graph(self.graph.id).goal, "demo")

    def test_dependency_chain_becomes_ready_in_order(self):
        db = self.add("Database")
        be = self.add("Backend", [db.id])
        fe = self.add("Frontend", [be.id])
        it = self.add("Integration tests", [fe.id])
        rv = self.add("Review", [it.id])
        order = []
        for _ in range(5):
            ready = self.store.ready_tasks(self.graph.id)
            self.assertEqual(len(ready), 1)
            order.append(ready[0].title)
            self.complete(ready[0])
        self.assertEqual(order, ["Database", "Backend", "Frontend", "Integration tests", "Review"])
        self.assertEqual(self.store.ready_tasks(self.graph.id), [])
        self.assertEqual([t.title for t in self.store.topological_order(self.graph.id)],
                         ["Database", "Backend", "Frontend", "Integration tests", "Review"])
        self.assertEqual(rv.dependencies, [it.id])

    def test_task_not_ready_until_all_dependencies_complete(self):
        a, b = self.add("A"), self.add("B")
        c = self.add("C", [a.id, b.id])
        self.complete(a)
        self.store.refresh(self.graph.id)
        self.assertEqual(self.store.require(c.id).status, TaskStatus.PENDING)
        self.complete(b)
        self.assertIn(c.id, [t.id for t in self.store.ready_tasks(self.graph.id)])

    def test_failure_blocks_dependents_and_retry_unblocks(self):
        a = self.add("A")
        b = self.add("B", [a.id])
        c = self.add("C", [b.id])
        self.store.mutate(a.id, lambda t: (setattr(t, "status", TaskStatus.FAILED), setattr(t, "attempts", 3)))
        self.store.refresh(self.graph.id)
        self.assertEqual(self.store.require(b.id).status, TaskStatus.BLOCKED)
        self.assertIn(a.id, self.store.require(b.id).blocked_reason)
        self.store.refresh(self.graph.id)
        self.assertEqual(self.store.require(c.id).status, TaskStatus.PENDING)  # transitively waiting
        retried = self.store.retry(a.id)
        self.assertEqual(retried.status, TaskStatus.READY)
        self.assertEqual(retried.max_attempts, 4)
        self.assertEqual(self.store.require(b.id).status, TaskStatus.PENDING)

    def test_priority_orders_ready_tasks(self):
        low = self.add("low", priority=1)
        high = self.add("high", priority=9)
        self.assertEqual([t.id for t in self.store.ready_tasks(self.graph.id)], [high.id, low.id])

    def test_cycles_and_unknown_dependencies_rejected(self):
        a = self.add("A")
        b = self.add("B", [a.id])
        with self.assertRaises(TaskGraphError):
            self.store.set_dependencies(a.id, [b.id])
        self.assertEqual(self.store.require(a.id).dependencies, [])  # nothing persisted
        with self.assertRaises(TaskGraphError):
            self.add("C", ["T-missing"])

    def test_pause_resume_cancel(self):
        a = self.add("A")
        paused = self.store.pause(a.id, "waiting for design decision")
        self.assertTrue(paused.is_paused)
        self.assertEqual(self.store.ready_tasks(self.graph.id), [])
        self.assertEqual(self.store.resume(a.id).status, TaskStatus.READY)
        b = self.add("B", [a.id])
        self.store.cancel(a.id)
        self.store.refresh(self.graph.id)
        self.assertEqual(self.store.require(b.id).status, TaskStatus.BLOCKED)
        with self.assertRaises(TaskGraphError):
            self.store.retry(b.id)  # only FAILED tasks

    def test_concurrent_mutations_do_not_lose_updates(self):
        t = self.add("shared")

        def bump():
            for _ in range(20):
                self.store.mutate(t.id, lambda x: setattr(x, "attempts", x.attempts + 1))

        threads = [threading.Thread(target=bump) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(TaskStore(self.dir).require(t.id).attempts, 80)


if __name__ == "__main__":
    unittest.main()
