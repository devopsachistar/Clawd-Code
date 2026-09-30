"""Secret boundary: keep credential values out of model context, logs,
sessions and reports.

Layers (all applied by :func:`redact_secrets`):

1. **Known values.** Values loaded from project ``.env`` files, sensitive
   environment variables and configured API keys are registered in the
   process-wide :class:`SecretRegistry`. Any occurrence is replaced by
   ``<SECRET>``, wherever it appears (tool output, command output, logs).
2. **High-confidence patterns.** Provider/API tokens, JWTs, PEM private
   keys, and passwords embedded in URLs (``postgres://user:pass@host``).
3. **Sensitive assignments.** ``DATABASE_URL=...``, ``STRIPE_SECRET_KEY=...``
   and ``"api_key": "..."``: the value is masked, the key name is kept, so
   the model still sees ``STRIPE_SECRET_KEY=<SECRET>``.

Files that exist to hold secrets (``.env``, ``*.pem``, ``credentials`` …)
get :func:`redact_secret_file`: every value is masked.

The model uses a secret without seeing it through the shell tools'
``secret_env`` parameter. The value is injected into the subprocess
environment, the call needs SECRET_USE approval, and output is redacted.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

SECRET_PLACEHOLDER = "<SECRET>"
MIN_SECRET_LENGTH = 8

_SENSITIVE_NAME = re.compile(
    r"(KEY|SECRET|TOKEN|PASSWORD|PASSWD|PWD|PASSPHRASE|CREDENTIAL|PRIVATE|AUTH|DATABASE_URL|DB_URL|DSN|"
    r"CONNECTION_STRING|CONN_STR|SESSION_SECRET|SESSION_KEY|COOKIE_SECRET|SALT|SIGNING|WEBHOOK)",
    re.IGNORECASE,
)
_NOT_SENSITIVE_NAMES = {"KEYBOARD", "PUBLIC_KEY_ID", "TOKENIZERS_PARALLELISM", "SSH_AUTH_SOCK", "AUTHOR", "AUTHORS",
                        "OLDPWD", "PWD"}
# Identifiers/metadata about a credential are not the credential itself.
_NOT_SENSITIVE_SUFFIX = re.compile(r"_(ID|IDS|NAME|PATH|FILE|DIR|TYPE|ENABLED|REGION|VERSION|HOST|PORT|USER|USERNAME|"
                                   r"URL_PATH|MODE|TIMEOUT|TTL|EXPIRY|EXPIRES|LENGTH|COUNT|PREFIX)$", re.IGNORECASE)

_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{10,}"),          # Stripe
    re.compile(r"\bwhsec_[A-Za-z0-9]{10,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bnpm_[A-Za-z0-9]{30,}"),
    re.compile(r"\bSG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}"),   # SendGrid
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-~+/]{12,}=*"),
    re.compile(r"(?i)(?<=AccountKey=)[A-Za-z0-9+/=]{20,}"),         # Azure storage connection strings
    re.compile(r"(?i)(?<=SharedAccessKey=)[A-Za-z0-9+/=]{20,}"),
)
# scheme://user:password@host  -> keep user and host
_URL_CREDENTIALS = re.compile(r"(?P<prefix>\b[a-z][a-z0-9+.\-]*://[^\s:/@]*:)(?P<secret>[^\s@/]{3,})(?P<suffix>@)", re.IGNORECASE)
# KEY=value / KEY: value / "key": "value" where KEY looks sensitive
_ASSIGNMENT = re.compile(
    r"""(?P<prefix>(?:^|(?<=[\s{,;(]))(?:export\s+|set\s+|\$env:)?["']?(?P<name>[A-Za-z_][A-Za-z0-9_.\-]*)["']?\s*(?::|=|:=)\s*)"""
    r"""(?P<quote>["']?)(?P<value>[^\s"',;}#]+)(?P=quote)""",
    re.MULTILINE,
)
_REFERENCE_VALUE = re.compile(
    r"^(?:\$|%|<|\{\{|process\.env|os\.environ|os\.getenv|env\(|getenv|None$|null$|undefined$|true$|false$|"
    r"\*+$|x+$|changeme$|your[_-]|example|placeholder|dummy|redacted|\[REDACTED\]|\.\.\.|<SECRET>)",
    re.IGNORECASE,
)


def is_sensitive_name(name: str) -> bool:
    upper = name.upper()
    if upper in _NOT_SENSITIVE_NAMES or _NOT_SENSITIVE_SUFFIX.search(upper):
        return False
    return bool(_SENSITIVE_NAME.search(upper))


def looks_like_credential(value: str) -> bool:
    """Heuristic for literal values in ordinary code (mixed, long, no spaces)."""
    if len(value) < 12 or any(ch.isspace() for ch in value):
        return False
    classes = sum([
        any(c.islower() for c in value),
        any(c.isupper() for c in value),
        any(c.isdigit() for c in value),
        any(not c.isalnum() for c in value),
    ])
    return classes >= 3 or (classes >= 2 and len(value) >= 24)


class SecretRegistry:
    """Process-wide set of known secret values (never persisted)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, str] = {}  # value -> source name
        self._loaded_roots: set[str] = set()
        self._pattern: re.Pattern[str] | None = None

    def register(self, value: str | None, name: str = "secret") -> bool:
        if not isinstance(value, str):
            return False
        value = value.strip()
        if len(value) < MIN_SECRET_LENGTH or value == SECRET_PLACEHOLDER or _REFERENCE_VALUE.match(value):
            return False
        with self._lock:
            if value in self._values:
                return False
            self._values[value] = name
            self._pattern = None
        return True

    def names(self) -> list[str]:
        with self._lock:
            return sorted(set(self._values.values()))

    def __len__(self) -> int:
        return len(self._values)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._loaded_roots.clear()
            self._pattern = None

    def mask(self, text: str) -> str:
        with self._lock:
            if not self._values:
                return text
            if self._pattern is None:
                ordered = sorted(self._values, key=len, reverse=True)
                self._pattern = re.compile("|".join(re.escape(v) for v in ordered))
            pattern = self._pattern
        return pattern.sub(SECRET_PLACEHOLDER, text)

    def mark_root_loaded(self, root: Path) -> bool:
        key = str(root)
        with self._lock:
            if key in self._loaded_roots:
                return False
            self._loaded_roots.add(key)
            return True


_REGISTRY = SecretRegistry()


def get_secret_registry() -> SecretRegistry:
    return _REGISTRY


# ---------------------------------------------------------------------------
# Loading secrets
# ---------------------------------------------------------------------------

_ENV_LINE = re.compile(r"^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_.\-]*)\s*=\s*(?P<value>.*?)\s*$")


def parse_env_text(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            continue
        value = m.group("value")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[m.group("key")] = value
    return values


def load_env_file(path: str | Path) -> dict[str, str]:
    try:
        return parse_env_text(Path(path).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return {}


_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", ".next", ".tox"}


def find_secret_env_files(root: Path, max_depth: int = 3) -> list[Path]:
    from ..tool_system.safety import is_secret_file

    found: list[Path] = []
    root = Path(root)
    if not root.is_dir():
        return found

    def walk(directory: Path, depth: int) -> None:
        try:
            entries = list(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_dir():
                if depth < max_depth and entry.name not in _SKIP_DIRS:
                    walk(entry, depth + 1)
            elif is_secret_file(entry.name) and (entry.name.startswith(".env") or entry.name.endswith(".env")):
                found.append(entry)

    walk(root, 0)
    return sorted(found)


def register_env_values(values: Mapping[str, str], source: str, registry: SecretRegistry | None = None) -> int:
    reg = registry or _REGISTRY
    count = 0
    for key, value in values.items():
        if is_sensitive_name(key) or looks_like_credential(value or ""):
            count += reg.register(value, f"{source}:{key}")
    return count


def register_project_secrets(project_root: str | Path, registry: SecretRegistry | None = None) -> int:
    """Register secret values from a project's .env files (once per root)."""
    reg = registry or _REGISTRY
    root = Path(project_root).resolve()
    if not reg.mark_root_loaded(root):
        return 0
    count = 0
    for env_file in find_secret_env_files(root):
        count += register_env_values(load_env_file(env_file), env_file.name, reg)
    return count


def register_environment_secrets(environ: Mapping[str, str] | None = None, registry: SecretRegistry | None = None) -> int:
    source = os.environ if environ is None else environ
    reg = registry or _REGISTRY
    return sum(
        reg.register(value, f"env:{key}")
        for key, value in source.items()
        if is_sensitive_name(key)
    )


def lookup_secret(name: str, project_root: str | Path | None = None) -> str | None:
    """Resolve a secret by name for controlled use (never returned to the model)."""
    if project_root is not None:
        for env_file in find_secret_env_files(Path(project_root)):
            values = load_env_file(env_file)
            if name in values and values[name] != "":
                _REGISTRY.register(values[name], f"{env_file.name}:{name}")
                return values[name]
    value = os.environ.get(name)
    if value:
        _REGISTRY.register(value, f"env:{name}")
    return value


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

_ENV_STYLE_PREFIX = re.compile(r"(?:export\s+|set\s+|\$env:)?[A-Z_][A-Z0-9_]*=")


def _mask_assignments(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        name, value = m.group("name"), m.group("value")
        if not is_sensitive_name(name) or _REFERENCE_VALUE.match(value):
            return m.group(0)
        quoted = bool(m.group("quote"))
        env_style = not quoted and bool(_ENV_STYLE_PREFIX.fullmatch(m.group("prefix").strip()))
        # Env/shell style lines (API_KEY=value) are always masked. Code
        # (spaces, quotes, expressions) only when the literal looks like a
        # real credential, so ordinary code stays editable.
        if env_style or looks_like_credential(value) or ("://" in value and "@" in value):
            q = m.group("quote")
            return f"{m.group('prefix')}{q}{SECRET_PLACEHOLDER}{q}"
        return m.group(0)

    return _ASSIGNMENT.sub(repl, text)


def redact_secrets(text: Any) -> Any:
    """Mask secrets in a string (non-strings are returned unchanged)."""
    if not isinstance(text, str) or not text:
        return text
    out = _REGISTRY.mask(text)
    for pattern in _TOKEN_PATTERNS:
        out = pattern.sub(SECRET_PLACEHOLDER, out)
    out = _URL_CREDENTIALS.sub(lambda m: f"{m.group('prefix')}{SECRET_PLACEHOLDER}{m.group('suffix')}", out)
    out = _mask_assignments(out)
    return out


def redact_secret_file(text: str) -> str:
    """Mask every value in a secret file while keeping its structure visible."""
    masked_lines: list[str] = []
    in_pem = False
    for line in text.splitlines():
        if "PRIVATE KEY-----" in line and "BEGIN" in line:
            in_pem = True
            masked_lines.append(SECRET_PLACEHOLDER)
            continue
        if in_pem:
            if "PRIVATE KEY-----" in line and "END" in line:
                in_pem = False
            continue
        m = _ENV_LINE.match(line)
        if m and not line.lstrip().startswith("#"):
            prefix = line[: m.start("value")]
            masked_lines.append(f"{prefix}{SECRET_PLACEHOLDER}" if m.group("value") else line)
            continue
        kv = re.match(r"^(\s*[\"']?[\w.\-]+[\"']?\s*:\s*)(.+?)(,?)\s*$", line)
        if kv and not line.strip().startswith(("#", "//", "{", "}", "[", "]")):
            masked_lines.append(f"{kv.group(1)}\"{SECRET_PLACEHOLDER}\"{kv.group(3)}")
            continue
        masked_lines.append(redact_secrets(line))
    return "\n".join(masked_lines)


def redact_structure(value: Any, _depth: int = 0) -> Any:
    """Recursively redact every string inside dicts/lists/tuples."""
    if _depth > 12:
        return value
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, Mapping):
        return {k: redact_structure(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_structure(v, _depth + 1) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_structure(v, _depth + 1) for v in value)
    return value


def contains_known_secret(text: str, values: Iterable[str]) -> bool:
    """Test helper: does ``text`` contain any of ``values`` verbatim?"""
    return any(v and v in text for v in values)
