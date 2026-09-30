"""Capability-based permission engine.

Every tool invocation is evaluated here by ``ToolRegistry.dispatch``, never
only by prompt instructions:

1. :func:`assess_tool_call` works out *what the call would do*. The result
   is a set of :class:`Capability` values, a :class:`RiskLevel`, and whether
   it is hard-blocked.
2. :class:`PermissionEngine` combines that assessment with the caller's
   :class:`PermissionPolicy` (the interactive session policy, or an agent
   profile's policy), explicit rules, and the tool's own ``check_permissions``.
   The outcome is a :class:`Decision`: ALLOW, DENY or ASK.
3. ASK produces a structured :class:`ApprovalRequest`. Only an explicit
   approval handler can grant it. Blanket auto-approval (``--yes``) is limited
   to requests that are ``auto_approvable``: risk at most MEDIUM, and never
   production deployment, database writes or secret use.

Risk mapping (highest applicable level wins)::

    LOW       READ, GIT_BRANCH (create/switch)
    MEDIUM    WRITE, EXECUTE, NETWORK, GIT_COMMIT, DATABASE_READ,
              recursive delete inside the workspace
    HIGH      GIT_PUSH, DATABASE_WRITE, DEPLOY_STAGING, SECRET_USE,
              history rewrites (reset --hard, rebase), pipe-to-shell,
              force-deleting branches
    CRITICAL  DEPLOY_PRODUCTION, destructive SQL / migration resets,
              anything the Phase 1 safety layer hard-blocks
"""

from __future__ import annotations

import fnmatch
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


class Capability(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    EXECUTE = "EXECUTE"
    NETWORK = "NETWORK"
    GIT_BRANCH = "GIT_BRANCH"
    GIT_COMMIT = "GIT_COMMIT"
    GIT_PUSH = "GIT_PUSH"
    DATABASE_READ = "DATABASE_READ"
    DATABASE_WRITE = "DATABASE_WRITE"
    DEPLOY_STAGING = "DEPLOY_STAGING"
    DEPLOY_PRODUCTION = "DEPLOY_PRODUCTION"
    SECRET_USE = "SECRET_USE"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    ASK = "ASK"


class RiskLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


CAPABILITY_RISK: dict[Capability, RiskLevel] = {
    Capability.READ: RiskLevel.LOW,
    Capability.GIT_BRANCH: RiskLevel.LOW,
    Capability.WRITE: RiskLevel.MEDIUM,
    Capability.EXECUTE: RiskLevel.MEDIUM,
    Capability.NETWORK: RiskLevel.MEDIUM,
    Capability.GIT_COMMIT: RiskLevel.MEDIUM,
    Capability.DATABASE_READ: RiskLevel.MEDIUM,
    Capability.GIT_PUSH: RiskLevel.HIGH,
    Capability.DATABASE_WRITE: RiskLevel.HIGH,
    Capability.DEPLOY_STAGING: RiskLevel.HIGH,
    Capability.SECRET_USE: RiskLevel.HIGH,
    Capability.DEPLOY_PRODUCTION: RiskLevel.CRITICAL,
}

# Capabilities that blanket auto-approval (--yes) can never grant.
NEVER_AUTO_APPROVE: frozenset[Capability] = frozenset({
    Capability.DEPLOY_PRODUCTION,
    Capability.DEPLOY_STAGING,
    Capability.DATABASE_WRITE,
    Capability.SECRET_USE,
    Capability.GIT_PUSH,
})
MAX_AUTO_APPROVE_RISK = RiskLevel.MEDIUM


def parse_capabilities(values: Iterable[Any] | None) -> set[Capability]:
    out: set[Capability] = set()
    for value in values or []:
        name = str(value).strip().upper()
        if not name:
            continue
        try:
            out.add(Capability(name))
        except ValueError as e:
            raise ValueError(f"Unknown capability: {value!r}") from e
    return out


# ---------------------------------------------------------------------------
# Assessment: what would this call do?
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolAssessment:
    capabilities: frozenset[Capability]
    risk: RiskLevel
    reasons: tuple[str, ...] = ()
    blocked: bool = False
    needs_approval: bool = False  # destructive pattern: ASK even if capabilities are allowed
    summary: str = ""


_I = re.IGNORECASE
# (pattern, capability, risk, reason)
_COMMAND_RULES: tuple[tuple[re.Pattern[str], Capability, RiskLevel, str], ...] = (
    # --- production deployment / publishing -------------------------------
    (re.compile(r"\bvercel\b[^\n]*--prod\b", _I), Capability.DEPLOY_PRODUCTION, RiskLevel.CRITICAL, "Vercel production deploy"),
    (re.compile(r"\bnetlify\s+deploy\b[^\n]*--prod\b", _I), Capability.DEPLOY_PRODUCTION, RiskLevel.CRITICAL, "Netlify production deploy"),
    (re.compile(r"\baz\s+(?:webapp|functionapp|staticwebapp|containerapp|deployment)\b[^\n]*\b(?:deploy|deployment|up|create|update)\b", _I),
     Capability.DEPLOY_PRODUCTION, RiskLevel.CRITICAL, "Azure deployment"),
    (re.compile(r"\b(?:gcloud\s+(?:app|run|functions)\s+deploy|firebase\s+deploy|eb\s+deploy|fly\s+deploy|cdk\s+deploy|"
                r"serverless\s+deploy|sls\s+deploy|pulumi\s+(?:up|destroy)|terraform\s+(?:apply|destroy)|"
                r"kubectl\s+(?:apply|delete|rollout|scale|replace|patch)|helm\s+(?:install|upgrade|uninstall|delete|rollback))\b", _I),
     Capability.DEPLOY_PRODUCTION, RiskLevel.CRITICAL, "infrastructure/production change"),
    (re.compile(r"\bgit\s+push\s+(?:\S+\s+)*heroku\b", _I), Capability.DEPLOY_PRODUCTION, RiskLevel.CRITICAL, "Heroku deploy"),
    (re.compile(r"\b(?:npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bcargo\s+publish\b|\bgem\s+push\b|\bdocker\s+push\b", _I),
     Capability.DEPLOY_PRODUCTION, RiskLevel.CRITICAL, "publish to a public registry"),
    (re.compile(r"\bdeploy\b[^\n]*\b(?:staging|stage|preview|dev|development|test)\b|\b(?:staging|preview)\b[^\n]*\bdeploy\b", _I),
     Capability.DEPLOY_STAGING, RiskLevel.HIGH, "staging deployment"),
    (re.compile(r"\bvercel\b(?![^\n]*--prod)|\bnetlify\s+deploy\b(?![^\n]*--prod)", _I),
     Capability.DEPLOY_STAGING, RiskLevel.HIGH, "preview deployment"),
    (re.compile(r"(?:\b(?:npm|pnpm|yarn)\s+(?:run\s+)?|\bmake\s+|\./)deploy\b|\bdeploy\.(?:sh|ps1|py)\b", _I),
     Capability.DEPLOY_PRODUCTION, RiskLevel.CRITICAL, "deployment script"),
    # --- git ---------------------------------------------------------------
    (re.compile(r"\bgit\s+push\b", _I), Capability.GIT_PUSH, RiskLevel.HIGH, "git push"),
    (re.compile(r"\bgh\s+(?:pr\s+(?:create|merge)|release\s+create)\b", _I), Capability.GIT_PUSH, RiskLevel.HIGH, "GitHub write"),
    (re.compile(r"\bgh\s+repo\s+delete\b", _I), Capability.GIT_PUSH, RiskLevel.CRITICAL, "delete a GitHub repository"),
    (re.compile(r"\bgit\s+(?:filter-branch|filter-repo)\b|\bgit\s+update-ref\s+-d\b|\bgit\s+reflog\s+expire\b", _I),
     Capability.GIT_COMMIT, RiskLevel.CRITICAL, "git history rewrite"),
    (re.compile(r"\bgit\s+reset\s+[^\n;&|]*--hard\b|\bgit\s+rebase\b", _I), Capability.GIT_COMMIT, RiskLevel.HIGH, "rewrite/discard git history"),
    (re.compile(r"\bgit\s+(?:commit|merge|cherry-pick|revert|am|tag)\b", _I), Capability.GIT_COMMIT, RiskLevel.MEDIUM, "git commit"),
    (re.compile(r"\bgit\s+branch\s+[^\n;&|]*-D\b"), Capability.GIT_BRANCH, RiskLevel.HIGH, "force-delete a branch"),
    (re.compile(r"\bgit\s+(?:checkout\s+-b|switch\s+-c|branch\s+\w|worktree\s+(?:add|remove))", _I), Capability.GIT_BRANCH, RiskLevel.LOW, "git branch"),
    (re.compile(r"\bgit\s+(?:clone|fetch|pull|ls-remote|submodule\s+update)\b", _I), Capability.NETWORK, RiskLevel.MEDIUM, "git network access"),
    # --- databases ---------------------------------------------------------
    (re.compile(r"\b(?:DROP\s+(?:TABLE|DATABASE|SCHEMA)|TRUNCATE\s+(?:TABLE\s+)?\w)|\bdropdb\b|\bFLUSHALL\b|\bFLUSHDB\b|"
                r"\bprisma\s+migrate\s+reset\b|\bdb:(?:drop|reset)\b|\bmigrate\s+(?:reset|fresh)\b", _I),
     Capability.DATABASE_WRITE, RiskLevel.CRITICAL, "destructive database operation"),
    (re.compile(r"\b(?:INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM|ALTER\s+TABLE|CREATE\s+(?:TABLE|INDEX|DATABASE))\b|"
                r"\b(?:prisma\s+(?:migrate\s+deploy|db\s+push)|alembic\s+(?:upgrade|downgrade)|db:migrate|manage\.py\s+migrate|"
                r"knex\s+migrate|sequelize\s+db:migrate|flyway\s+migrate|liquibase\s+update)\b", _I),
     Capability.DATABASE_WRITE, RiskLevel.HIGH, "database write/migration"),
    (re.compile(r"\b(?:psql|mysql|mariadb|sqlcmd|sqlite3|mongosh|mongo|redis-cli|cqlsh)\b|\bSELECT\b[^\n]*\bFROM\b", _I),
     Capability.DATABASE_READ, RiskLevel.MEDIUM, "database access"),
    # --- secrets -----------------------------------------------------------
    (re.compile(r"(?:(?:^|[\s;&|(])(?:cat|type|more|less|head|tail|bat|Get-Content|gc|source|\.)\s[^\n;&|]*?"
                r"(?:(?<![\w.])\.env(?:\.[\w.-]+)?(?![\w])|\.pem\b|\.key\b|id_rsa|id_ed25519|credentials|secrets?\.(?:json|ya?ml))|"
                r"^\s*(?:printenv|env|set|Get-ChildItem\s+env:|gci\s+env:|dir\s+env:)\s*$|"
                r"\b(?:printenv|echo)\s+[^\n;&|]*\$\{?\w*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)\w*|"
                r"\$env:\w*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)\w*|"
                r"\b(?:gh\s+auth\s+token|aws\s+configure\s+get|az\s+account\s+get-access-token|gcloud\s+auth\s+print-access-token|"
                r"security\s+find-generic-password|kubectl\s+get\s+secrets?))", _I | re.MULTILINE),
     Capability.SECRET_USE, RiskLevel.HIGH, "reads credentials/secret files"),
    # --- network -----------------------------------------------------------
    (re.compile(r"\b(?:curl|wget|iwr|irm|Invoke-WebRequest|Invoke-RestMethod|ssh|scp|sftp|ftp|telnet|nc|ncat)\b", _I),
     Capability.NETWORK, RiskLevel.MEDIUM, "network access"),
    (re.compile(r"\b(?:pip|pip3|uv\s+pip|poetry|pipenv)\s+(?:install|add)\b|\b(?:npm|pnpm)\s+(?:install|i|ci|add)\b|"
                r"\byarn\s+(?:add|install)?\b|\b(?:apt|apt-get|brew|choco|winget|scoop)\s+install\b|"
                r"\bdocker\s+(?:pull|build)\b|\bgo\s+(?:get|mod\s+download)\b|\bcargo\s+(?:install|fetch)\b", _I),
     Capability.NETWORK, RiskLevel.MEDIUM, "install/download packages"),
    # --- filesystem writes ---------------------------------------------------
    (re.compile(r"(?:^|[^<>2&])>{1,2}\s*[^\s&]|\b(?:rm|del|erase|rmdir|rd|mv|move|cp|copy|mkdir|md|touch|tee|chmod|chown|"
                r"New-Item|Remove-Item|Move-Item|Copy-Item|Set-Content|Add-Content|Out-File)\b", _I),
     Capability.WRITE, RiskLevel.MEDIUM, "modifies files"),
)
# Destructive-but-local patterns flagged by the Phase 1 classifier map to these risks.
_DANGEROUS_RISK_OVERRIDES: tuple[tuple[re.Pattern[str], RiskLevel], ...] = (
    (re.compile(r"\|\s*(?:sudo\s+)?(?:ba|z|k)?sh\b|\|\s*(?:iex|Invoke-Expression)\b", _I), RiskLevel.HIGH),
)

# Tools whose effect is fully described by a static capability set.
_TOOL_CAPABILITIES: dict[str, frozenset[Capability]] = {
    "read": frozenset({Capability.READ}),
    "glob": frozenset({Capability.READ}),
    "grep": frozenset({Capability.READ}),
    "lsp": frozenset({Capability.READ}),
    "toolsearch": frozenset({Capability.READ}),
    "brief": frozenset({Capability.READ}),
    "sleep": frozenset({Capability.READ}),
    "askuserquestion": frozenset({Capability.READ}),
    "sendusermessage": frozenset({Capability.READ}),
    "structuredoutput": frozenset({Capability.READ}),
    "todowrite": frozenset({Capability.READ}),
    "taskcreate": frozenset({Capability.READ}),
    "taskget": frozenset({Capability.READ}),
    "tasklist": frozenset({Capability.READ}),
    "taskupdate": frozenset({Capability.READ}),
    "taskoutput": frozenset({Capability.READ}),
    "taskstop": frozenset({Capability.READ}),
    "croncreate": frozenset({Capability.READ}),
    "cronlist": frozenset({Capability.READ}),
    "crondelete": frozenset({Capability.READ}),
    "enterplanmode": frozenset({Capability.READ}),
    "sendmessage": frozenset({Capability.READ}),
    "reportresult": frozenset({Capability.READ}),
    "messageorchestrator": frozenset({Capability.READ}),
    "testingpermission": frozenset({Capability.READ}),
    "exitworktree": frozenset({Capability.READ}),
    "exitplanmode": frozenset({Capability.WRITE}),
    "write": frozenset({Capability.WRITE}),
    "edit": frozenset({Capability.WRITE}),
    "notebookedit": frozenset({Capability.WRITE}),
    "teamcreate": frozenset({Capability.WRITE}),
    "teamdelete": frozenset({Capability.WRITE}),
    "listmcpresourcestool": frozenset({Capability.NETWORK}),
    "readmcpresourcetool": frozenset({Capability.NETWORK}),
    "webfetch": frozenset({Capability.READ, Capability.NETWORK}),
    "websearch": frozenset({Capability.READ, Capability.NETWORK}),
    "mcp": frozenset({Capability.EXECUTE, Capability.NETWORK}),
    "remotetrigger": frozenset({Capability.EXECUTE, Capability.NETWORK}),
    "repl": frozenset({Capability.EXECUTE}),
    "agent": frozenset({Capability.EXECUTE}),
    "enterworktree": frozenset({Capability.GIT_BRANCH}),
}
_SHELL_TOOLS = {"bash", "powershell"}


def analyze_command(command: str) -> tuple[set[Capability], RiskLevel, list[str]]:
    """Capabilities a shell command needs (EXECUTE is always included)."""
    caps: set[Capability] = {Capability.EXECUTE}
    risk = RiskLevel.MEDIUM
    reasons: list[str] = []
    for pattern, capability, cap_risk, reason in _COMMAND_RULES:
        if pattern.search(command or ""):
            caps.add(capability)
            risk = max(risk, cap_risk)
            reasons.append(reason)
    for pattern, override in _DANGEROUS_RISK_OVERRIDES:
        if pattern.search(command or ""):
            risk = max(risk, override)
    return caps, risk, list(dict.fromkeys(reasons))


def _summarize(tool_name: str, tool_input: Mapping[str, Any]) -> str:
    for key in ("command", "file_path", "path", "url", "setting", "pattern", "query", "skill", "name"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return f"{tool_name}: {value[:300]}"
    return tool_name


def assess_tool_call(tool_name: str, tool_input: Mapping[str, Any], spec: Any = None, context: Any = None) -> ToolAssessment:
    """Determine capabilities and risk of one tool invocation."""
    from ..tool_system.safety import classify_command, contains_credential, is_secret_file

    name = tool_name.lower()
    reasons: list[str] = []
    blocked = False
    needs_approval = False
    summary = _summarize(tool_name, tool_input)

    if name in _SHELL_TOOLS:
        command = str(tool_input.get("command") or "")
        caps, risk, reasons = analyze_command(command)
        cwd = getattr(context, "cwd", None) or getattr(context, "workspace_root", None)
        phase1 = classify_command(command, cwd=cwd)
        if phase1.is_blocked:
            blocked, risk = True, RiskLevel.CRITICAL
            reasons = list(phase1.reasons) + reasons
        elif phase1.needs_approval:
            needs_approval = True
            reasons = reasons + [r for r in phase1.reasons if r not in reasons]
        if tool_input.get("secret_env"):
            caps.add(Capability.SECRET_USE)
            risk = max(risk, RiskLevel.HIGH)
            reasons.append("injects secrets: " + ", ".join(map(str, tool_input.get("secret_env") or [])))
        return ToolAssessment(frozenset(caps), risk, tuple(reasons), blocked, needs_approval, summary)

    if name in _TOOL_CAPABILITIES:
        caps = set(_TOOL_CAPABILITIES[name])
    elif name == "config":
        caps = {Capability.WRITE} if "value" in tool_input else {Capability.READ}
    elif name == "skill":
        caps = {Capability.EXECUTE} if "name" in tool_input else {Capability.READ}
    else:
        # Unknown / user-loaded tools: be conservative.
        caps = {Capability.EXECUTE}
        if spec is not None and getattr(spec, "is_read_only", False):
            caps = {Capability.READ}

    if name in {"write", "edit", "notebookedit"}:
        path = tool_input.get("file_path") or tool_input.get("notebook_path")
        text = tool_input.get("content") if name == "write" else tool_input.get("new_string")
        if isinstance(path, str) and is_secret_file(path):
            caps.add(Capability.SECRET_USE)
            reasons.append(f"writes secret file {Path(path).name}")
        if isinstance(text, str) and contains_credential(text):
            caps.add(Capability.SECRET_USE)
            reasons.append("content contains a credential")
    if name == "config" and "value" in tool_input:
        setting = str(tool_input.get("setting") or "")
        if re.search(r"(api[_-]?key|secret|token|password|base_?url|endpoint)", setting, _I):
            caps.add(Capability.SECRET_USE)
            reasons.append(f"changes credential/endpoint setting {setting}")

    risk = max((CAPABILITY_RISK[c] for c in caps), default=RiskLevel.LOW)
    return ToolAssessment(frozenset(caps), risk, tuple(reasons), blocked, needs_approval, summary)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

_RULE_RE = re.compile(r"^\s*([A-Za-z0-9_*?]+)\s*(?:\((.*)\))?\s*$")


@dataclass(frozen=True)
class PermissionRule:
    """``Tool`` or ``Tool(argument-glob)``, e.g. ``Bash(git push*)``, ``Write(src/**)``."""

    tool: str
    pattern: Optional[str]
    decision: Decision

    @classmethod
    def parse(cls, text: str, decision: Decision | str) -> "PermissionRule":
        m = _RULE_RE.match(text or "")
        if not m:
            raise ValueError(f"Invalid permission rule: {text!r}")
        # Enum members: use .value (str() of a str-Enum is "Decision.DENY" on Python 3.11+).
        value = decision.value if isinstance(decision, Decision) else str(decision).upper()
        return cls(tool=m.group(1), pattern=(m.group(2).strip() if m.group(2) is not None else None),
                   decision=Decision(value))

    def matches(self, tool_name: str, tool_input: Mapping[str, Any]) -> bool:
        if not fnmatch.fnmatch(tool_name.lower(), self.tool.lower()):
            return False
        if self.pattern is None:
            return True
        for key in ("command", "file_path", "path", "url", "setting", "pattern"):
            value = tool_input.get(key)
            if isinstance(value, str):
                candidate = value.replace("\\", "/") if key != "command" else value.strip()
                if fnmatch.fnmatch(candidate, self.pattern) or fnmatch.fnmatch(candidate.lower(), self.pattern.lower()):
                    return True
        return False

    def __str__(self) -> str:
        return f"{self.tool}({self.pattern})" if self.pattern is not None else self.tool


@dataclass(frozen=True)
class PermissionPolicy:
    """Capability decisions plus explicit rules for one principal (session or agent)."""

    name: str
    capabilities: Mapping[Capability, Decision]
    default: Decision = Decision.DENY
    rules: tuple[PermissionRule, ...] = ()

    def decision_for(self, capability: Capability) -> Decision:
        return self.capabilities.get(capability, self.default)

    def with_rules(self, rules: Iterable[PermissionRule]) -> "PermissionPolicy":
        return PermissionPolicy(self.name, dict(self.capabilities), self.default, tuple(self.rules) + tuple(rules))

    @classmethod
    def session_default(cls) -> "PermissionPolicy":
        """Interactive / headless user session. Preserves the Phase 1 behaviour."""
        allow = {Capability.READ, Capability.WRITE, Capability.EXECUTE, Capability.NETWORK, Capability.GIT_BRANCH}
        caps = {c: (Decision.ALLOW if c in allow else Decision.ASK) for c in Capability}
        return cls(name="session", capabilities=caps, default=Decision.ASK)

    @classmethod
    def from_lists(cls, name: str, *, allow: Iterable[Any] = (), ask: Iterable[Any] = (), deny: Iterable[Any] = (),
                   rules: Iterable[PermissionRule] = ()) -> "PermissionPolicy":
        caps: dict[Capability, Decision] = {}
        for c in parse_capabilities(allow):
            caps[c] = Decision.ALLOW
        for c in parse_capabilities(ask):
            caps[c] = Decision.ASK
        for c in parse_capabilities(deny):
            caps[c] = Decision.DENY
        return cls(name=name, capabilities=caps, default=Decision.DENY, rules=tuple(rules))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PermissionPolicy":
        caps = {Capability(k): Decision(v) for k, v in (data.get("capabilities") or {}).items()}
        rules = tuple(PermissionRule.parse(r["rule"], r["decision"]) for r in (data.get("rules") or []))
        return cls(name=str(data.get("name", "policy")), capabilities=caps,
                   default=Decision(data.get("default", Decision.DENY.value)), rules=rules)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capabilities": {c.value: d.value for c, d in sorted(self.capabilities.items(), key=lambda kv: kv[0].value)},
            "default": self.default.value,
            "rules": [{"rule": str(r), "decision": r.decision.value} for r in self.rules],
        }


# ---------------------------------------------------------------------------
# Decisions and approval requests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ApprovalRequest:
    """A structured request for explicit human approval."""

    tool: str
    summary: str
    capabilities: tuple[str, ...]
    risk: str
    reasons: tuple[str, ...]
    policy: str
    auto_approvable: bool
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    id: str = field(default_factory=lambda: f"AP-{uuid.uuid4().hex[:8]}")
    decision: str = Decision.ASK.value
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def render(self) -> str:
        lines = [
            f"Tool: {self.summary}",
            f"Capability: {', '.join(self.capabilities)}",
            f"Risk: {self.risk}",
            f"Decision: {self.decision}",
        ]
        if self.reasons:
            lines.append(f"Why: {'; '.join(self.reasons)}")
        if self.agent_id:
            lines.append(f"Agent: {self.agent_id}" + (f" (task {self.task_id})" if self.task_id else ""))
        if not self.auto_approvable:
            lines.append("This request requires explicit approval (cannot be auto-approved).")
        return "\n".join(lines)


@dataclass(frozen=True)
class PermissionDecision:
    decision: Decision
    assessment: ToolAssessment
    reasons: tuple[str, ...]
    auto_approvable: bool
    policy: str

    @property
    def message(self) -> str:
        return "; ".join(self.reasons) if self.reasons else self.decision.value

    def approval_request(self, tool_name: str, *, agent_id: str | None = None, task_id: str | None = None) -> ApprovalRequest:
        return ApprovalRequest(
            tool=tool_name,
            summary=self.assessment.summary or tool_name,
            capabilities=tuple(sorted(c.value for c in self.assessment.capabilities)),
            risk=self.assessment.risk.name,
            reasons=self.reasons,
            policy=self.policy,
            auto_approvable=self.auto_approvable,
            agent_id=agent_id,
            task_id=task_id,
        )


def is_auto_approvable(assessment: ToolAssessment) -> bool:
    return (
        not assessment.blocked
        and assessment.risk <= MAX_AUTO_APPROVE_RISK
        and not (assessment.capabilities & NEVER_AUTO_APPROVE)
    )


class PermissionEngine:
    """Evaluate tool invocations against a policy."""

    def evaluate(
        self,
        tool_name: str,
        tool_input: Mapping[str, Any],
        policy: PermissionPolicy,
        *,
        spec: Any = None,
        context: Any = None,
        tool_check: Any = None,
    ) -> PermissionDecision:
        assessment = assess_tool_call(tool_name, tool_input, spec, context)
        auto = is_auto_approvable(assessment)

        def result(decision: Decision, reasons: Iterable[str]) -> PermissionDecision:
            return PermissionDecision(decision, assessment, tuple(dict.fromkeys(r for r in reasons if r)), auto, policy.name)

        if assessment.blocked:
            return result(Decision.DENY, ["blocked: " + "; ".join(assessment.reasons)])

        tool_behavior = getattr(getattr(tool_check, "behavior", None), "value", None)
        tool_message = getattr(tool_check, "message", None)
        if tool_behavior == "deny":
            return result(Decision.DENY, [tool_message or "denied by tool"])

        matched = [r for r in policy.rules if r.matches(tool_name, tool_input)]
        for rule in matched:
            if rule.decision is Decision.DENY:
                return result(Decision.DENY, [f"denied by rule {rule}"])

        cap_decisions = {c: policy.decision_for(c) for c in assessment.capabilities}
        denied = sorted(c.value for c, d in cap_decisions.items() if d is Decision.DENY)
        if denied and not any(r.decision is Decision.ALLOW for r in matched):
            return result(Decision.DENY, [f"policy '{policy.name}' does not grant {', '.join(denied)}"])

        ask_reasons: list[str] = []
        if any(r.decision is Decision.ASK for r in matched):
            ask_reasons.append("rule requires approval")
        asked = sorted(c.value for c, d in cap_decisions.items() if d is Decision.ASK)
        if asked:
            ask_reasons.append(f"{', '.join(asked)} requires approval")
        if assessment.needs_approval:
            ask_reasons.append("destructive operation: " + "; ".join(assessment.reasons))
        if tool_behavior == "ask":
            ask_reasons.append(tool_message or "tool requires approval")

        explicit_allow = any(r.decision is Decision.ALLOW for r in matched)
        # An explicit user allow rule may waive ordinary asks, but never
        # critical-risk operations or secret use.
        if ask_reasons and explicit_allow and assessment.risk < RiskLevel.CRITICAL \
                and Capability.SECRET_USE not in assessment.capabilities and tool_behavior != "ask":
            return result(Decision.ALLOW, [f"allowed by rule {next(r for r in matched if r.decision is Decision.ALLOW)}"])
        if ask_reasons:
            return result(Decision.ASK, ask_reasons + list(assessment.reasons))
        return result(Decision.ALLOW, [])
