"""Delivery / review reports (always passed through the secret boundary)."""

from __future__ import annotations

from typing import Any

from ..security.secrets import redact_secrets, redact_structure


def render_delivery_report(data: dict[str, Any]) -> str:
    """Markdown report for humans. ``data``: goal, graph, tasks, verification, review."""
    data = redact_structure(data)
    lines = [f"# Delivery report: {data.get('goal', '')}", ""]
    graph = data.get("graph") or {}
    if graph:
        lines += [
            f"- Graph: `{graph.get('id')}`  status **{graph.get('status')}**",
            f"- Base commit: `{graph.get('base_ref')}`",
        ]
        if graph.get("integration_branch"):
            lines.append(f"- Integration branch: `{graph['integration_branch']}`")
        lines.append("")
    lines += ["## Tasks", "", "| Task | Profile | Status | Attempts | Branch | Commit |", "|---|---|---|---|---|---|"]
    for t in data.get("tasks", []):
        result = t.get("result") or {}
        lines.append(
            f"| `{t.get('id')}` {t.get('title')} | {t.get('profile') or '-'} | {t.get('status')} | "
            f"{t.get('attempts', 0)} | `{t.get('branch') or '-'}` | `{(result.get('commit') or '-')[:10]}` |"
        )
    lines.append("")
    for t in data.get("tasks", []):
        result = t.get("result") or {}
        if not result and not t.get("error"):
            continue
        lines += [f"### {t.get('title')} ({t.get('id')})", ""]
        if result.get("summary"):
            lines += [str(result["summary"]), ""]
        if result.get("files_changed"):
            lines.append("Files: " + ", ".join(f"`{f}`" for f in result["files_changed"]))
        tests = t.get("tests") or {}
        if tests:
            lines.append(f"Tests (measured by orchestrator): run {tests.get('run', 0)}, passed {tests.get('passed', 0)}, "
                         f"failed {tests.get('failed', 0)}")
        for label in ("risks", "warnings"):
            for item in result.get(label) or []:
                lines.append(f"- {label[:-1]}: {item}")
        if result.get("review_verdict"):
            lines.append(f"Review verdict: **{result['review_verdict']}**")
        if t.get("error"):
            lines.append(f"Error: {t['error']}")
        lines.append("")
    verification = data.get("verification") or {}
    if verification:
        lines += ["## Verification", "",
                  f"`{verification.get('command')}` in `{verification.get('worktree')}`: exit {verification.get('exit_code')}, "
                  f"run {verification.get('run', 0)}, passed {verification.get('passed', 0)}, failed {verification.get('failed', 0)}", ""]
    review = data.get("review") or {}
    if review:
        lines += ["## Next step", "", str(review.get("next_step", "")), ""]
    return redact_secrets("\n".join(lines).rstrip() + "\n")
