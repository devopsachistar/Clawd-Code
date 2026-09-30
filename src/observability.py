"""Structured, secret-safe logging for the agent runtime.

Every runtime component emits events through :func:`log_event`. Events go to
the standard ``logging`` logger named :data:`LOGGER_NAME`, which has no output
by default. Frontends opt in with :func:`configure_logging`, for example the CLI's
``--log-file`` / ``--verbose`` flags or the ``CLAWD_LOG_FILE`` environment
variable.

Every field is passed through :func:`redact` before it is logged, so API keys,
tokens, passwords and similar credentials never reach a log sink.

Event names (stable, dotted):
    agent.started, agent.completed, agent.max_turns
    provider.request, provider.response, provider.error
    tool.started, tool.completed, tool.failed, permission.denied
    session.saved, session.loaded, run.started, run.completed
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

LOGGER_NAME = "clawd"
REDACTED = "[REDACTED]"
MAX_STRING_CHARS = 2_000

_logger = logging.getLogger(LOGGER_NAME)
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())

_SENSITIVE_KEY_RE = re.compile(
    r"(api[_-]?key|apikey|secret|password|passwd|pwd|token|authorization|auth[_-]?header|"
    r"credential|cookie|private[_-]?key|client[_-]?secret|session[_-]?key)",
    re.IGNORECASE,
)
# Keys that match the pattern above but are not secrets.
_SAFE_KEYS = {"input_tokens", "output_tokens", "total_tokens", "max_tokens", "tokens", "token_count", "tokens_saved"}

_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
)
_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|SECRET|PASSWORD|PASSWD|TOKEN)[A-Z0-9_]*)\s*([=:])\s*(['\"]?)([^\s'\"]+)\3"
)


def redact_text(text: str) -> str:
    """Mask credential-looking substrings inside free text."""
    if not text:
        return text
    from .security.secrets import redact_secrets

    out = text
    for pattern in _SECRET_VALUE_PATTERNS:
        out = pattern.sub(REDACTED, out)
    out = _ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", out)
    # Secret boundary: known secret values and remaining credential patterns.
    return redact_secrets(out)


def is_sensitive_key(key: str) -> bool:
    return key.lower() not in _SAFE_KEYS and bool(_SENSITIVE_KEY_RE.search(key))


def redact(value: Any, *, max_chars: int = MAX_STRING_CHARS, _depth: int = 0) -> Any:
    """Return a JSON-friendly copy of ``value`` with secrets masked."""
    if _depth > 6:
        return "..."
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            out[key] = REDACTED if is_sensitive_key(key) and v not in (None, "") else redact(v, max_chars=max_chars, _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        red = [redact(v, max_chars=max_chars, _depth=_depth + 1) for v in items[:50]]
        if len(items) > 50:
            red.append(f"... (+{len(items) - 50} more)")
        return red
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    text = redact_text(str(value))
    if len(text) > max_chars:
        text = text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"
    return text


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a structured event (fields are redacted before logging)."""
    if not _logger.isEnabledFor(level):
        return
    _logger.log(level, event, extra={"clawd_event": event, "clawd_fields": redact(fields)})


class JsonLinesFormatter(logging.Formatter):
    """Format records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "event": getattr(record, "clawd_event", record.getMessage()),
        }
        fields = getattr(record, "clawd_fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        elif record.args or record.msg:
            payload["message"] = redact_text(record.getMessage())
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


class _HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "clawd_fields", None) or {}
        details = " ".join(f"{k}={json.dumps(v, ensure_ascii=False, default=str)}" for k, v in fields.items())
        return f"[{record.levelname.lower()}] {getattr(record, 'clawd_event', record.getMessage())} {details}".rstrip()


_configured_handlers: list[logging.Handler] = []


def configure_logging(*, log_file: str | Path | None = None, verbose: bool = False, level: int = logging.INFO) -> None:
    """Attach log sinks. Safe to call more than once (replaces earlier sinks)."""
    for handler in _configured_handlers:
        _logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    _configured_handlers.clear()

    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(JsonLinesFormatter())
        _configured_handlers.append(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(_HumanFormatter())
        _configured_handlers.append(sh)

    for handler in _configured_handlers:
        _logger.addHandler(handler)
    _logger.setLevel(level if _configured_handlers else logging.WARNING)
    _logger.propagate = False


def get_logger() -> logging.Logger:
    return _logger
