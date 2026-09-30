"""Phase 1 safety baseline: operation kinds and command risk classification.

This is deliberately small. It is not the Phase 2 permission-policy engine.
It gives the runtime three things:

* every tool is labelled READ, WRITE or EXECUTE (:func:`operation_kind`)
* shell commands are classified (:func:`classify_command`) as
    - ``blocked``   catastrophic or privilege-escalating; always refused
    - ``dangerous`` destructive or outward-facing; needs explicit approval
    - ``mutating``  changes local state; allowed (workspace sandbox still applies)
    - ``read``      recognised read-only command
* secret-bearing files (``.env`` etc.) are recognised so writes to them and
  git operations that would commit them are not allowed silently

"Needs approval" becomes an interactive prompt in the REPL. In headless
mode (no prompt available) the call is denied unless the user passed ``--yes``.
Blocked commands are refused even with ``--yes``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterable


class OperationKind(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


# Explicit labels where a tool's ToolSpec flags are missing or misleading.
_TOOL_KINDS: dict[str, OperationKind] = {
    "bash": OperationKind.EXECUTE,
    "powershell": OperationKind.EXECUTE,
    "repl": OperationKind.EXECUTE,
    "mcp": OperationKind.EXECUTE,
    "skill": OperationKind.EXECUTE,
    "agent": OperationKind.EXECUTE,
    "remotetrigger": OperationKind.EXECUTE,
    "write": OperationKind.WRITE,
    "edit": OperationKind.WRITE,
    "notebookedit": OperationKind.WRITE,
    "config": OperationKind.WRITE,
    "teamcreate": OperationKind.WRITE,
    "teamdelete": OperationKind.WRITE,
    "exitplanmode": OperationKind.WRITE,
    "enterworktree": OperationKind.WRITE,
    "read": OperationKind.READ,
    "glob": OperationKind.READ,
    "grep": OperationKind.READ,
    "webfetch": OperationKind.READ,
    "websearch": OperationKind.READ,
    "lsp": OperationKind.READ,
    "toolsearch": OperationKind.READ,
}


def operation_kind(spec) -> OperationKind:
    """Classify a tool (by ToolSpec) as READ, WRITE or EXECUTE."""
    name = str(getattr(spec, "name", "")).lower()
    if name in _TOOL_KINDS:
        return _TOOL_KINDS[name]
    if getattr(spec, "is_read_only", False):
        return OperationKind.READ
    return OperationKind.WRITE if getattr(spec, "is_destructive", False) else OperationKind.READ


# ---------------------------------------------------------------------------
# Secret files
# ---------------------------------------------------------------------------

_ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template", ".dist", ".defaults")
_SECRET_FILE_NAMES = {
    ".netrc", ".npmrc", ".pypirc", "credentials", "credentials.json",
    "secrets.json", "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
}
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")


def is_secret_file(path: str | Path) -> bool:
    """True for files that conventionally hold credentials (``.env``, keys, ...)."""
    name = Path(path).name.lower()
    if name == ".env" or name.startswith(".env.") or name.endswith(".env"):
        return not name.endswith(_ENV_TEMPLATE_SUFFIXES)
    if name in _SECRET_FILE_NAMES:
        return True
    return name.endswith(_SECRET_SUFFIXES)


_CREDENTIAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
)


def contains_credential(text: str) -> bool:
    """High-confidence check for real-looking credentials in text."""
    return any(p.search(text or "") for p in _CREDENTIAL_PATTERNS)


# ---------------------------------------------------------------------------
# Command classification
# ---------------------------------------------------------------------------

RISK_ORDER = {"read": 0, "mutating": 1, "dangerous": 2, "blocked": 3}


@dataclass(frozen=True)
class CommandAssessment:
    risk: str  # "read" | "mutating" | "dangerous" | "blocked"
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_blocked(self) -> bool:
        return self.risk == "blocked"

    @property
    def needs_approval(self) -> bool:
        return self.risk == "dangerous"

    def message(self) -> str:
        return "; ".join(self.reasons) if self.reasons else self.risk


_I = re.IGNORECASE
_ROOTISH = r"(?:/|/\*|~|~/|~/\*|\$HOME|\$HOME/\*|\$\{HOME\}|\.|\./|\*|[A-Za-z]:[\\/]?\*?|\\)"

_BLOCKED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:sudo|doas|runas|pkexec)\b", _I), "privilege escalation (sudo/doas/runas)"),
    (re.compile(r"(?:^|[;&|(]\s*)su(?:\s|$)", _I), "privilege escalation (su)"),
    (re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b", _I), "system shutdown/reboot"),
    (re.compile(r"\b(?:Stop-Computer|Restart-Computer)\b", _I), "system shutdown/reboot"),
    (re.compile(r"\bmkfs(?:\.\w+)?\b", _I), "filesystem format"),
    (re.compile(r"\b(?:Format-Volume|Clear-Disk|Initialize-Disk|diskpart|bcdedit)\b", _I), "disk/boot modification"),
    (re.compile(r"(?:^|[;&|\s])format(?:\.com)?\s+[A-Za-z]:", _I), "disk format"),
    (re.compile(r"\bdd\b[^\n]*\bif=", _I), "raw disk copy (dd)"),
    (re.compile(r">\s*/dev/(?:sd|nvme|hd|disk)", _I), "write to raw disk device"),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", _I), "fork bomb"),
    (re.compile(r"--no-preserve-root", _I), "rm --no-preserve-root"),
    (re.compile(r"\brm\s+(?:-[A-Za-z]*\s+)*-[A-Za-z]*(?:r[A-Za-z]*f|f[A-Za-z]*r)[A-Za-z]*\s+(?:-[A-Za-z-]+\s+)*" + _ROOTISH + r"(?:\s|$|;|&|\|)", _I),
     "recursive force delete of root/home/current directory"),
    (re.compile(r"\bRemove-Item\b[^\n;|]*-Recurse[^\n;|]*\s(?:[A-Za-z]:\\?|\\|~|\$env:USERPROFILE|\$HOME|\.)(?:\s|$|;)", _I),
     "recursive delete of drive root/home/current directory"),
    (re.compile(r"\b(?:rd|rmdir)\s+/s\b[^\n]*\s[A-Za-z]:\\?(?:\s|$)", _I), "recursive delete of a drive root"),
    (re.compile(r"\bdel\s+/[a-z/\s]*s[^\n]*\s[A-Za-z]:\\\*?(?:\s|$)", _I), "recursive delete of a drive root"),
    (re.compile(r"\bchmod\s+-R\s+\S+\s+/(?:\s|$)", _I), "recursive permission change of /"),
    (re.compile(r"\breg(?:\.exe)?\s+delete\s+HKLM", _I), "system registry deletion"),
)

_DANGEROUS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bgit\s+push\b", _I), "git push (publishes to a remote)"),
    (re.compile(r"\bgit\s+reset\s+[^\n;&|]*--hard\b", _I), "git reset --hard (discards changes)"),
    (re.compile(r"\bgit\s+clean\s+[^\n;&|]*-[A-Za-z]*f", _I), "git clean -f (deletes untracked files)"),
    (re.compile(r"\bgit\s+(?:checkout|restore)\s+[^\n;&|]*(?:--\s+\.|\s\.)(?:\s|$)", _I), "discard working-tree changes"),
    (re.compile(r"\bgit\s+commit\b", _I), "git commit"),
    (re.compile(r"\bgit\s+rebase\b", _I), "git rebase (rewrites history)"),
    (re.compile(r"\bgit\s+branch\s+[^\n;&|]*-D\b"), "force-delete a git branch"),
    (re.compile(r"\bgit\s+(?:filter-branch|filter-repo|update-ref\s+-d|reflog\s+expire|gc\s+--prune)", _I), "git history rewrite"),
    (re.compile(r"\brm\s+(?:-[A-Za-z]*\s+)*-[A-Za-z]*[rR]", _I), "recursive delete (rm -r)"),
    (re.compile(r"\bRemove-Item\b[^\n;|]*-Recurse", _I), "recursive delete (Remove-Item -Recurse)"),
    (re.compile(r"\b(?:rd|rmdir)\s+/s\b", _I), "recursive delete (rmdir /s)"),
    (re.compile(r"\bdel\s+/[a-z/\s]*s\b", _I), "recursive delete (del /s)"),
    (re.compile(r"\b(?:curl|wget|iwr|Invoke-WebRequest|irm|Invoke-RestMethod)\b[^\n]*\|\s*(?:sudo\s+)?(?:ba|z|k)?sh\b", _I), "pipe a download into a shell"),
    (re.compile(r"\b(?:iwr|irm|Invoke-WebRequest|Invoke-RestMethod)\b[^\n]*\|\s*(?:iex|Invoke-Expression)\b", _I), "pipe a download into Invoke-Expression"),
    (re.compile(r"\b(?:chmod|chown)\s+-R\b", _I), "recursive permission/ownership change"),
    (re.compile(r"\b(?:npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b|\bgem\s+push\b", _I), "publish a package"),
    (re.compile(r"\bterraform\s+(?:apply|destroy)\b|\bkubectl\s+(?:delete|apply)\b|\bhelm\s+(?:uninstall|delete)\b", _I), "infrastructure change"),
    (re.compile(r"\bdocker\s+(?:system|volume|image|container)\s+prune\b", _I), "docker prune"),
    (re.compile(r"\b(?:DROP\s+(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s+TABLE)\b", _I), "destructive SQL"),
    (re.compile(r"\bgh\s+(?:pr\s+(?:create|merge)|release\s+create|repo\s+delete)\b", _I), "GitHub write operation"),
    (re.compile(r"\b(?:scp|rsync)\b[^\n]*\s\S+@\S+:", _I), "copy files to a remote host"),
)

_READ_ONLY_COMMANDS = {
    "ls", "dir", "pwd", "cat", "type", "head", "tail", "less", "more", "wc", "echo", "printf",
    "grep", "rg", "find", "which", "where", "whoami", "date", "env", "printenv", "tree", "file",
    "stat", "du", "df", "uname", "hostname", "diff", "sort", "uniq", "cut", "Get-ChildItem",
    "Get-Content", "Get-Location", "Select-String", "Test-Path", "Get-Item", "Get-Command",
}
_READ_ONLY_GIT = {"status", "log", "diff", "show", "branch", "remote", "rev-parse", "ls-files", "blame", "describe", "tag", "config"}
_SEGMENT_SPLIT = re.compile(r"(?:\|\||&&|;|\||\n)")
_GIT_STAGE_OR_COMMIT = re.compile(r"\bgit\s+(?:add|commit|stash)\b", _I)
_GIT_ADD_ARGS = re.compile(r"\bgit\s+add\b([^\n;&|]*)", _I)


def _git_add_targets_secret(text: str) -> bool:
    for match in _GIT_ADD_ARGS.finditer(text):
        for arg in match.group(1).split():
            arg = arg.strip("\"'")
            if arg and not arg.startswith("-") and is_secret_file(arg):
                return True
    return False


def _segment_is_read_only(segment: str) -> bool:
    words = segment.strip().split()
    if not words:
        return True
    head = words[0].strip("\"'")
    if head.lower() == "git" and len(words) > 1:
        return words[1].lower() in _READ_ONLY_GIT
    return head in _READ_ONLY_COMMANDS or head.lower() in {c.lower() for c in _READ_ONLY_COMMANDS}


def _env_files_at_risk(cwd: Path | None) -> list[str]:
    """Secret files git would stage: tracked-and-modified or untracked-not-ignored."""
    if cwd is None:
        return []
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=str(cwd), capture_output=True, text=True, timeout=10,
            stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace",
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    risky: list[str] = []
    for line in proc.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path and is_secret_file(path):
            risky.append(path)
    return risky


def classify_command(command: str, *, cwd: Path | None = None) -> CommandAssessment:
    """Classify a shell command's risk. ``cwd`` enables git secret-file checks."""
    text = command or ""
    blocked = [reason for pat, reason in _BLOCKED if pat.search(text)]
    if _git_add_targets_secret(text):
        blocked.append("staging an environment/secret file (.env) with git")
    if _GIT_STAGE_OR_COMMIT.search(text):
        risky = _env_files_at_risk(cwd)
        if risky:
            blocked.append(
                "git add/commit while secret files are uncommitted and not ignored: "
                + ", ".join(risky[:5])
                + " (add them to .gitignore first)"
            )
    if blocked:
        return CommandAssessment("blocked", tuple(dict.fromkeys(blocked)))

    dangerous = [reason for pat, reason in _DANGEROUS if pat.search(text)]
    if dangerous:
        return CommandAssessment("dangerous", tuple(dict.fromkeys(dangerous)))

    segments = [s for s in _SEGMENT_SPLIT.split(text) if s.strip()]
    if segments and all(_segment_is_read_only(s) for s in segments) and ">" not in text:
        return CommandAssessment("read")
    return CommandAssessment("mutating")


def max_risk(assessments: Iterable[CommandAssessment]) -> str:
    return max((a.risk for a in assessments), key=lambda r: RISK_ORDER[r], default="read")
